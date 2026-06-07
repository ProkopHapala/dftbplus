"""
SK file utilities: parsing, resampling, channel handling, masking, and generic fitting loops.
"""

import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable, Any
from scipy.interpolate import CubicSpline


# =============================================================================
# Channel definitions
# =============================================================================

CHANNELS = [(0, "ss", 0), (1, "sp", 1), (2, "ppσ", 2), (3, "ppπ", 2)]
KINDS = ["H", "S"]

OLD_CHANNEL_NAMES = [
    "ss", "sp", "pp_sigma", "pp_pi", "sd", "pd_sigma", "pd_pi",
    "dd_sigma", "dd_pi", "dd_delta",
]

EXT_CHANNEL_NAMES = [
    "ss", "sp", "pp_sigma", "pp_pi", "sd", "pd_sigma", "pd_pi",
    "dd_sigma", "dd_pi", "dd_delta",
    "p2d", "d2p", "d3d", "s2d", "d2s", "p2p", "s2s", "p2s", "s2p", "d2d",
]


def get_channel_names(extended: bool) -> List[str]:
    return EXT_CHANNEL_NAMES if extended else OLD_CHANNEL_NAMES


# =============================================================================
# SK file parser (from sk_parser.py)
# =============================================================================

@dataclass
class SkTable:
    """Parsed SK table for one element pair."""
    sp1: str
    sp2: str
    dr: float
    n_grid_raw: int
    n_shell: int
    extended: bool
    h_values: List[List[float]]
    s_values: List[List[float]]

    @property
    def n_grid(self) -> int:
        return len(self.h_values)

    @property
    def n_integ(self) -> int:
        return len(self.h_values[0]) if self.h_values else 0

    @property
    def r_grid(self) -> np.ndarray:
        return np.arange(1, self.n_grid + 1) * self.dr

    @property
    def r_max_grid(self) -> float:
        return self.n_grid * self.dr

    @property
    def r_cutoff(self) -> float:
        DIST_FUDGE = 1.0
        return (self.n_grid - 1) * self.dr + DIST_FUDGE

    def get_channel(self, idx: int, kind: str = "H") -> Tuple[np.ndarray, np.ndarray]:
        vals = self.h_values if kind.upper() == "H" else self.s_values
        if not vals:
            return self.r_grid.copy(), np.array([])
        if not self.extended:
            old_map = {0: 9, 1: 8, 2: 5, 3: 6}
            phys_idx = old_map.get(idx, idx)
        else:
            phys_idx = idx
        arr = np.array([row[phys_idx] for row in vals])
        return self.r_grid.copy(), arr


def _parse_f64(tok: str) -> float:
    return float(tok.replace("D", "E").replace("d", "e"))


def _expand_repeat(tok: str) -> Optional[List[float]]:
    m = re.match(r"(\d+)\*([+-]?[\d.eE+-]+)", tok)
    if m:
        n = int(m.group(1))
        v = _parse_f64(m.group(2))
        return [v] * n
    return None


def _parse_numbers_loose(line: str) -> List[float]:
    line = line.replace(",", " ")
    out = []
    for tok in line.split():
        expanded = _expand_repeat(tok)
        if expanded is not None:
            out.extend(expanded)
            continue
        try:
            out.append(_parse_f64(tok))
        except ValueError:
            pass
    return out


def parse_skf(path: str | Path) -> SkTable:
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()
    name = path.stem
    sp1, sp2 = name.split("-")

    it = iter(lines)
    first = next(it, "").strip()
    extended = first.startswith("@")

    grid_line = next(it, "").strip() if extended else first
    grid_tokens = grid_line.replace(",", " ").split()
    if len(grid_tokens) < 2:
        raise ValueError(f"Bad grid line in {path}: {grid_line!r}")

    dr = _parse_f64(grid_tokens[0])
    n_grid_raw = int(grid_tokens[1])
    n_shell = int(grid_tokens[2]) if len(grid_tokens) >= 3 else 1
    n_grid = max(0, n_grid_raw - 1)

    if sp1 == sp2:
        next(it, None)
        next(it, None)
    else:
        next(it, None)

    h_vals = []
    s_vals = []
    for _ in range(n_grid):
        line = next(it, "").strip()
        nums = _parse_numbers_loose(line)
        if extended:
            nums = nums + [0.0] * (40 - len(nums))
            h_vals.append(nums[:20])
            s_vals.append(nums[20:40])
        else:
            nums = nums + [0.0] * (20 - len(nums))
            h_vals.append(nums[:10])
            s_vals.append(nums[10:20])

    return SkTable(
        sp1=sp1, sp2=sp2, dr=dr, n_grid_raw=n_grid_raw,
        n_shell=n_shell, extended=extended,
        h_values=h_vals, s_values=s_vals,
    )


def load_sk_folder(folder: str | Path) -> List[SkTable]:
    folder = Path(folder)
    tables = []
    for path in sorted(folder.glob("*.skf")):
        try:
            tables.append(parse_skf(path))
        except Exception as e:
            print(f"Warning: failed to parse {path.name}: {e}")
    return tables


# =============================================================================
# Resampling (from resample.py)
# =============================================================================

def trim_constant_prefix(r_raw: np.ndarray, f_raw: np.ndarray, tol: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    if len(f_raw) < 2:
        return r_raw, f_raw
    f0 = f_raw[0]
    i_first = 0
    for i in range(1, len(f_raw)):
        if abs(f_raw[i] - f0) > tol:
            i_first = i
            break
    if i_first == 0:
        return r_raw, f_raw
    return r_raw[i_first:], f_raw[i_first:]


def resample_curve(r_raw: np.ndarray, f_raw: np.ndarray, Rc: float, n_u: int = 1000) -> np.ndarray:
    r_raw, f_raw = trim_constant_prefix(r_raw, f_raw)
    u_grid = np.linspace(0.0, 1.0, n_u)
    r_grid = u_grid * Rc
    cs = CubicSpline(r_raw, f_raw, bc_type="natural")
    r_first = r_raw[0]
    r_last = r_raw[-1]
    f_resampled = np.full(n_u, np.nan)
    for i, r in enumerate(r_grid):
        if r < r_first:
            continue
        elif r <= r_last:
            f_resampled[i] = cs(r)
        elif r < Rc:
            frac = (r - r_last) / (Rc - r_last)
            f_resampled[i] = f_raw[-1] * (1.0 - frac)
        else:
            f_resampled[i] = 0.0
    return f_resampled


def collect_all_curves(
    tables: List[SkTable],
    n_u: int = 1000,
    channels: Optional[List[int]] = None,
    kind: str = "both",
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], List[Dict]]:
    u_grid = np.linspace(0.0, 1.0, n_u)
    h_curves = []
    s_curves = []
    metadata = []

    for tab in tables:
        names = get_channel_names(tab.extended)
        ch_list = channels if channels is not None else range(tab.n_integ)
        for ch_idx in ch_list:
            if ch_idx >= tab.n_integ:
                continue
            ch_name = names[ch_idx] if ch_idx < len(names) else f"ch{ch_idx}"
            Rc = tab.r_cutoff
            if kind in ("both", "H"):
                r_raw, f_raw = tab.get_channel(ch_idx, "H")
                f_u = resample_curve(r_raw, f_raw, Rc, n_u)
                valid = f_u[np.isfinite(f_u)]
                if len(valid) == 0 or np.all(np.abs(valid) < 1e-12):
                    continue
                h_curves.append(f_u)
                metadata.append({
                    "sp1": tab.sp1, "sp2": tab.sp2,
                    "channel": ch_name, "channel_idx": ch_idx,
                    "kind": "H", "Rc": Rc, "dr": tab.dr,
                    "n_grid": tab.n_grid, "extended": tab.extended,
                })
            if kind in ("both", "S"):
                r_raw, f_raw = tab.get_channel(ch_idx, "S")
                f_u = resample_curve(r_raw, f_raw, Rc, n_u)
                valid = f_u[np.isfinite(f_u)]
                if len(valid) == 0 or np.all(np.abs(valid) < 1e-12):
                    continue
                s_curves.append(f_u)
                metadata.append({
                    "sp1": tab.sp1, "sp2": tab.sp2,
                    "channel": ch_name, "channel_idx": ch_idx,
                    "kind": "S", "Rc": Rc, "dr": tab.dr,
                    "n_grid": tab.n_grid, "extended": tab.extended,
                })

    H_mat = np.column_stack(h_curves) if h_curves else None
    S_mat = np.column_stack(s_curves) if s_curves else None
    return u_grid, H_mat, S_mat, metadata


# =============================================================================
# Common masking and preparation utilities
# =============================================================================

def mask_and_clip(u: np.ndarray, f: np.ndarray, chi: np.ndarray, threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply u > threshold mask to all arrays."""
    mask = u > threshold
    return u[mask], f[mask], chi[mask]


def prepare_channel(
    tab: SkTable,
    ch_idx: int,
    ch_name: str,
    kind: str,
    u: np.ndarray,
    chi: np.ndarray,
    threshold: float = 0.1,
    min_points: int = 10,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Get a single channel, trim prefix, resample, mask.
    Returns (u_valid, f_valid, chi_valid) or None if invalid.
    """
    try:
        r, f = tab.get_channel(ch_idx, kind)
        r_t, f_t = trim_constant_prefix(r, f)
        Rc = tab.r_cutoff
        f_u = np.interp(u * Rc, r_t, f_t, left=np.nan, right=0.0)
        valid = np.isfinite(f_u)
        if not np.any(valid):
            return None
        f_u = np.where(valid, f_u, 0.0)
        u_valid = u[valid]
        f_valid = f_u[valid]
        chi_valid = chi[valid]
        u_valid, f_valid, chi_valid = mask_and_clip(u_valid, f_valid, chi_valid, threshold)
        if len(f_valid) < min_points:
            return None
        return u_valid, f_valid, chi_valid
    except Exception:
        return None


def fit_all_channels(
    tables: List[SkTable],
    u: np.ndarray,
    chi: np.ndarray,
    fit_fn: Callable,
    **fit_kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Generic fitting loop over all channels in all tables.

    Parameters
    ----------
    tables : list of SkTable
    u, chi : np.ndarray
    fit_fn : callable
        Signature: fit_fn(u_valid, f_valid, chi_valid, **fit_kwargs) -> result

    Returns
    -------
    list of dicts with keys: pair, channel, kind, result
    """
    results = []
    for tab in tables:
        pair = f"{tab.sp1}-{tab.sp2}"
        for ch_idx, ch_name, _m in CHANNELS:
            for kind in KINDS:
                prepared = prepare_channel(tab, ch_idx, ch_name, kind, u, chi)
                if prepared is None:
                    continue
                u_valid, f_valid, chi_valid = prepared
                try:
                    result = fit_fn(u_valid, f_valid, chi_valid, **fit_kwargs)
                    results.append({
                        "pair": pair,
                        "channel": ch_name,
                        "kind": kind,
                        "result": result,
                        "u": u_valid,
                        "f": f_valid,
                        "chi": chi_valid,
                    })
                except Exception:
                    pass
    return results


# =============================================================================
# Sweep and evaluation helpers
# =============================================================================

def _make_grid(n: int = 500):
    """Create u-grid and chi envelope."""
    from fitting import envelope
    u = np.linspace(0, 1, n)
    chi = envelope(u, power=2, variant="linear")
    return u, chi


def _collect_curves(tables, u, chi):
    """Yield (u_valid, f_valid, chi_valid, meta) for all valid channels."""
    for tab in tables:
        for ch_idx, ch_name, _m in CHANNELS:
            for kind in KINDS:
                prepared = prepare_channel(tab, ch_idx, ch_name, kind, u, chi)
                if prepared is None:
                    continue
                meta = {"pair": f"{tab.sp1}-{tab.sp2}", "ch_name": ch_name, "kind": kind}
                yield (*prepared, meta)


def _sweep(tables, u, chi, components_list, mode, method, **fit_kwargs):
    """
    Sweep over multiple basis configs, return {label: [rmse, ...]}.

    components_list : list of (label, [BasisComponent, ...])
    """
    from fitting import fit_basis
    results = {label: [] for label, _ in components_list}
    for u_v, f_v, chi_v, _meta in _collect_curves(tables, u, chi):
        for label, comps in components_list:
            try:
                _c, _fit, rmse = fit_basis(u_v, f_v, chi_v, comps, mode, method, **fit_kwargs)
                results[label].append(rmse)
            except Exception:
                pass
    return results


def _print_rmse_table(name_to_rmses, title="", ncoefs=None):
    names = list(name_to_rmses.keys())
    print(f"\n{'='*80}")
    if title:
        print(title)
        print(f"{'='*80}")
    hdr = f"{'Config':<22} {'N':>5}"
    if ncoefs:
        hdr += f" {'Ncoef':>6}"
    hdr += f" {'Mean':>12} {'Median':>12} {'Min':>12} {'Max':>12}"
    print(hdr)
    print("-" * 80)
    for name in names:
        rms = np.array(name_to_rmses[name])
        if len(rms) == 0:
            continue
        line = f"{name:<22} {len(rms):>5}"
        if ncoefs and name in ncoefs:
            line += f" {ncoefs[name]:>6}"
        elif ncoefs:
            line += f" {'':>6}"
        line += f" {np.mean(rms):>12.4e} {np.median(rms):>12.4e} {np.min(rms):>12.4e} {np.max(rms):>12.4e}"
        print(line)
    print("-" * 80)
