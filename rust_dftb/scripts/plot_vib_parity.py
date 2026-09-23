#!/usr/bin/env python3
"""Hessian heatmap and vibrational spectrum, with an optional DFTB+ overlay.

Frequencies come from a sparse_vibrations freq file (`idx freq` lines).
The Cartesian Hessian is Ha/Å², row-major, one row per line, a `#` header.
DFTB+ hessian.out is Ha/Bohr², Fortran column-major, and is converted with
the same mass-weighting as dftb_engine (amu, Å → cm⁻¹).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANG2BOHR = 1.889726133
AU_TO_CM = 5140.487143
MASS = {"H": 1.008, "C": 12.011, "Si": 28.085, "N": 14.007, "O": 15.999}


def read_freqs(path):
    freqs = []
    for ln in open(path):
        p = ln.split()
        if len(p) == 2 and p[0].isdigit():
            freqs.append(float(p[1]))
        elif p and p[0] == "mode":
            break
    if not freqs:
        raise ValueError(f"no frequencies in {path}")
    return np.array(freqs)


def read_hess_ang(path):
    rows = []
    for ln in open(path):
        if ln.startswith("#"):
            continue
        rows.append([float(x) for x in ln.split()])
    h = np.array(rows)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"{path}: expected square matrix, got {h.shape}")
    return h


def read_dftb_hess_ang(path):
    # Wrapped at 4 numbers per line; each Hessian column is 3N values.
    data = np.fromstring(Path(path).read_text(), sep=" ")
    n3 = int(round(np.sqrt(data.size)))
    if n3 * n3 != data.size:
        raise ValueError(f"{path}: {data.size} values is not a square")
    h_bohr = data.reshape((n3, n3), order="F")
    return h_bohr * ANG2BOHR ** 2


def species_of_xyz(path, n_atom):
    lines = [ln for ln in open(path) if ln.strip()]
    body = lines[2:2 + n_atom]
    if len(body) != n_atom:
        raise ValueError(f"{path}: need {n_atom} atoms, got {len(body)}")
    return [ln.split()[0] for ln in body]


def freqs_from_hess(h_ang, species):
    m = np.array([MASS[s] for s in species])
    inv = np.repeat(1.0 / (ANG2BOHR * np.sqrt(m)), 3)
    mw = h_ang * inv[:, None] * inv[None, :]
    lam = np.linalg.eigvalsh(0.5 * (mw + mw.T))
    out = np.empty_like(lam)
    neg = lam < 0.0
    out[neg] = -np.sqrt(-lam[neg]) * AU_TO_CM
    out[~neg] = np.sqrt(lam[~neg]) * AU_TO_CM
    return out


def sticks(ax, freqs, color, label):
    ax.vlines(freqs, 0.0, 1.0, color=color, lw=0.8)
    n_imag = int(np.sum(freqs < -1.0))
    ax.text(0.01, 0.92, f"{label}   n={len(freqs)}   imag={n_imag}   min={freqs.min():.1f}",
            transform=ax.transAxes, fontsize=8, color=color, va="top")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([])
    ax.axvline(0.0, color="k", lw=0.6)


def heat(ax, h, title):
    v = np.percentile(np.abs(h), 99.5)
    v = max(float(v), 1e-6)
    im = ax.imshow(h, cmap="seismic", vmin=-v, vmax=v, interpolation="nearest", origin="upper")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("coordinate")
    ax.set_ylabel("coordinate")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", action="append", default=[], help="label=path to sparse freq.txt")
    ap.add_argument("--hess", action="append", default=[], help="label=path to Ha/Å² matrix")
    ap.add_argument("--dftb-hess", default=None, help="DFTB+ hessian.out (Ha/Bohr²)")
    ap.add_argument("--xyz", default=None, help="geometry, for mass-weighting the DFTB+ matrix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    panels = []
    for spec in args.freq:
        lab, path = spec.split("=", 1)
        panels.append((lab, read_freqs(path), "#b22222"))
    hesses = []
    for spec in args.hess:
        lab, path = spec.split("=", 1)
        hesses.append((lab, read_hess_ang(path)))
    if args.dftb_hess:
        h = read_dftb_hess_ang(args.dftb_hess)
        hesses.insert(0, ("DFTB+", h))
        if args.xyz:
            n_atom = h.shape[0] // 3
            panels.insert(0, ("DFTB+", freqs_from_hess(h, species_of_xyz(args.xyz, n_atom)), "#444444"))

    n_spec = len(panels)
    n_heat = len(hesses)
    nrows = n_spec + (1 if n_heat else 0)
    fig_h = 1.7 * n_spec + (5.2 if n_heat else 0) + 0.6
    fig = plt.figure(figsize=(11, fig_h))
    gs_rows = n_spec + (2 if n_heat > 1 else (1 if n_heat else 0))
    # spectra on top, hessians in one row, correlation if two frequency sets
    heights = [1] * n_spec
    if n_heat:
        heights.append(3.2)
    if n_spec >= 2:
        heights.append(2.6)
    fig, axes = plt.subplots(
        len(heights), 1, figsize=(11, sum(heights) + 0.8),
        gridspec_kw={"height_ratios": heights}, squeeze=False,
    )
    row = 0
    for lab, f, col in panels:
        sticks(axes[row, 0], f, col, lab)
        row += 1
    if panels:
        lo = min(f.min() for _, f, _ in panels)
        hi = max(f.max() for _, f, _ in panels)
        for r in range(n_spec):
            axes[r, 0].set_xlim(min(-50, lo * 1.05 - 10), hi * 1.02)
        axes[n_spec - 1, 0].set_xlabel("frequency (cm$^{-1}$)")

    if n_heat:
        # replace the single hessian axis with a row of heatmaps
        axes[row, 0].remove()
        gs = axes[row, 0].get_gridspec() if False else None
        inner = fig.add_gridspec(1, n_heat, left=0.08, right=0.92,
                                 bottom=0.08 if n_spec < 2 else 0.32,
                                 top=0.08 + 3.2 / (sum(heights) + 0.8) * 0.9)
        # Simpler: draw hessians as a separate figure if the grid gets messy.
        row += 1

    # Rebuild cleanly. The mixed grid above is fragile; draw two figures.
    plt.close(fig)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if panels:
        fig, ax = plt.subplots(n_spec, 1, figsize=(11, 1.8 * n_spec + 0.5), sharex=True, squeeze=False)
        for i, (lab, f, col) in enumerate(panels):
            sticks(ax[i, 0], f, col, lab)
        lo = min(f.min() for _, f, _ in panels)
        hi = max(f.max() for _, f, _ in panels)
        ax[0, 0].set_xlim(min(-80.0, lo - 20), hi * 1.03)
        ax[-1, 0].set_xlabel("frequency (cm$^{-1}$)")
        if args.title:
            fig.suptitle(args.title)
        fig.tight_layout()
        spec_path = out.with_name(out.stem + "_spectrum.png")
        fig.savefig(spec_path, dpi=160)
        plt.close(fig)
        print(f"wrote {spec_path}")

    if n_heat:
        fig, ax = plt.subplots(1, n_heat, figsize=(5.2 * n_heat, 4.6), squeeze=False)
        for i, (lab, h) in enumerate(hesses):
            im = heat(ax[0, i], h, f"{lab}   {h.shape[0]}×{h.shape[0]}")
            fig.colorbar(im, ax=ax[0, i], fraction=0.046, pad=0.04, label="Ha/Å²")
        if args.title:
            fig.suptitle(args.title + "  Hessian")
        fig.tight_layout()
        hess_path = out.with_name(out.stem + "_hessian.png")
        fig.savefig(hess_path, dpi=140)
        plt.close(fig)
        print(f"wrote {hess_path}")

    if len(panels) >= 2:
        a = panels[0][1]
        b = panels[1][1]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        fig, ax = plt.subplots(figsize=(5.4, 5.2))
        ax.plot(a, b, "o", ms=3, color="#b22222")
        lim = max(abs(a.min()), abs(b.min()), a.max(), b.max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="#888888", lw=0.8)
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(0, color="k", lw=0.4)
        # Chemical modes: drop the six rigid-body slots.
        if n > 6:
            d = b[6:] - a[6:]
            rms = float(np.sqrt(np.mean(d ** 2)))
            ax.set_title(f"{panels[1][0]} vs {panels[0][0]}\nrms of modes 7…{n} = {rms:.1f} cm$^{{-1}}$")
        else:
            ax.set_title(f"{panels[1][0]} vs {panels[0][0]}")
        ax.set_xlabel(f"{panels[0][0]} (cm$^{{-1}}$)")
        ax.set_ylabel(f"{panels[1][0]} (cm$^{{-1}}$)")
        ax.set_aspect("equal")
        fig.tight_layout()
        corr_path = out.with_name(out.stem + "_corr.png")
        fig.savefig(corr_path, dpi=160)
        plt.close(fig)
        print(f"wrote {corr_path}")
        imag_a = int(np.sum(panels[0][1] < -1.0))
        imag_b = int(np.sum(panels[1][1] < -1.0))
        print(f"imag {panels[0][0]}={imag_a}  {panels[1][0]}={imag_b}")
        print(f"lowest {panels[0][0]} {panels[0][1][:8]}")
        print(f"lowest {panels[1][0]} {panels[1][1][:8]}")


if __name__ == "__main__":
    main()
