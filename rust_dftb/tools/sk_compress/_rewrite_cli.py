import textwrap

content = '''
"""
Unified CLI for SK compression analysis.
Replaces all individual analyze/test/plot scripts.
Refactored: common patterns extracted to helpers.
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional, Any

pkg_dir = Path(__file__).resolve().parent
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))

from sk_utils import load_sk_folder, CHANNELS, KINDS, prepare_channel
from fitting import (
    envelope, build_combined_basis, fit_combined_basis, fit_tikhonov,
    analyze_conditioning, test_normalization, test_orthogonalization,
    analyze_correlation_structure, design_orthogonal_fine_tuning, test_orthogonal_basis,
    compute_orthogonal_component, build_dyadic_basis_vander,
)
from plotting import (
    setup_output_dir, plot_basis_functions, plot_correlation_matrix,
    plot_orthogonal_component, plot_conditioning, plot_tikhonov_results,
    plot_coefficient_stats, plot_rmse_comparison, plot_mio_cluster,
)

DEFAULT_SK_PATH = "/home/prokophapala/SIMULATIONS/dftbplus/slakos/mio/mio-1-1"


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def _load_data(args):
    """Load SK tables from args.sk_path or default."""
    sk_path = args.sk_path or DEFAULT_SK_PATH
    return load_sk_folder(sk_path)


def _make_grid(n=500):
    """Create u-grid and chi envelope."""
    u = np.linspace(0, 1, n)
    chi = envelope(u, power=2, variant="linear")
    return u, chi


def _fit_lstsq(W, f):
    """Linear least-squares fit and RMSE."""
    c, _, _, _ = np.linalg.lstsq(W, f, rcond=None)
    f_fit = W @ c
    rmse = np.sqrt(np.mean((f - f_fit) ** 2))
    return c, f_fit, rmse


def _collect_curves(tables, u, chi):
    """Collect all valid curves as list of (u_valid, f_valid, chi_valid, meta)."""
    curves = []
    for tab in tables:
        for ch_idx, ch_name, _m in CHANNELS:
            for kind in KINDS:
                prepared = prepare_channel(tab, ch_idx, ch_name, kind, u, chi)
                if prepared is None:
                    continue
                meta = {"pair": f"{tab.sp1}-{tab.sp2}", "ch_name": ch_name, "kind": kind}
                curves.append((*prepared, meta))
    return curves


def _sweep_all(tables, u, chi, fit_fn):
    """
    Apply fit_fn(u_valid, f_valid, chi_valid, meta) to all curves.
    Returns list of (meta, result).
    """
    out = []
    for u_v, f_v, chi_v, meta in _collect_curves(tables, u, chi):
        try:
            res = fit_fn(u_v, f_v, chi_v, meta)
            if res is not None:
                out.append((meta, res))
        except Exception:
            pass
    return out


def _print_rmse_table(name_to_rmses, title="", ncoefs=None):
    """Print a formatted RMSE summary table."""
    names = list(name_to_rmses.keys())
    print(f"\\n{'='*80}")
    if title:
        print(title)
        print(f"{'='*80}")
    hdr = f"{'Config':<20} {'N':>5}"
    if ncoefs:
        hdr += f" {'Ncoef':>6}"
    hdr += f" {'Mean':>12} {'Median':>12} {'Min':>12} {'Max':>12}"
    print(hdr)
    print("-" * 80)
    for name in names:
        rms = np.array(name_to_rmses[name])
        if len(rms) == 0:
            continue
        line = f"{name:<20} {len(rms):>5}"
        if ncoefs and name in ncoefs:
            line += f" {ncoefs[name]:>6}"
        elif ncoefs:
            line += f" {'':>6}"
        line += f" {np.mean(rms):>12.4e} {np.median(rms):>12.4e} {np.min(rms):>12.4e} {np.max(rms):>12.4e}"
        print(line)
    print("-" * 80)


def _plot_rmse_bar(name_to_rmses, out_path, title="", baseline=None):
    """Generic bar chart: mean/median RMSE per config."""
    names = list(name_to_rmses.keys())
    means = [np.mean(np.array(name_to_rmses[n])) for n in names]
    medians = [np.median(np.array(name_to_rmses[n])) for n in names]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, means, w, label="Mean", color="steelblue", edgecolor="black")
    ax.bar(x + w / 2, medians, w, label="Median", color="coral", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_scatter(xvals, yvals_dict, out_path, title="", xlabel="", ylabel="", log=True):
    """Generic scatter plot with multiple y-series and y=x line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(yvals_dict)))
    for (label, y), color in zip(yvals_dict.items(), colors):
        ax.scatter(xvals, y, alpha=0.4, s=20, color=color, label=label)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    vmin = min(v.min() for v in yvals_dict.values())
    vmax = max(v.max() for v in yvals_dict.values())
    lim = [min(xvals.min(), vmin), max(xvals.max(), vmax)]
    ax.plot(lim, lim, "k--", lw=1, label="y=x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_fit_overlay(u_v, f_v, f_fit, out_path, title=""):
    """Plot original, fit, and error on twin axis."""
    err = f_v - f_fit
    rmse = np.sqrt(np.mean(err ** 2))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(u_v, f_v, "b-", lw=1.5, alpha=0.7, label="Original")
    ax.plot(u_v, f_fit, "r--", lw=1.5, alpha=0.8, label="Fit")
    ax2 = ax.twinx()
    ax2.plot(u_v, err, "g-", lw=1, alpha=0.6, label=f"Error (RMSE={rmse:.2e})")
    ax2.axhline(0, color="k", ls="--", lw=0.5)
    ax2.set_ylabel("Error", color="g")
    ax2.tick_params(axis="y", labelcolor="g")
    ax.set_title(f"{title}  (RMSE={rmse:.2e})")
    ax.set_xlabel("u = r/Rc")
    ax.set_ylabel("f(u)", color="b")
    ax.tick_params(axis="y", labelcolor="b")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# =============================================================================
# COMMANDS
# =============================================================================

def cmd_basis(args):
    """Plot basis functions: original, orthogonalized, and alternative bases."""
    out_dir = setup_output_dir(pkg_dir, "basis_plots")
    u, chi = _make_grid()
    fname = plot_basis_functions(u, chi, n_dyadic=4, degree_leg=3, start_n=2, out_dir=out_dir)
    print(f"Saved: {fname}")
    from plotting import plot_legendre_contribution, plot_basis_functions_ortho, plot_basis_functions_alternative
    fname2 = plot_legendre_contribution(u, chi, n_dyadic=4, degree_leg=3, start_n=2, out_dir=out_dir)
    print(f"Saved: {fname2}")
    fname3 = plot_basis_functions_ortho(u, chi, n_dyadic=4, degree_leg=3, start_n=2, out_dir=out_dir)
    print(f"Saved: {fname3}")
    fname4 = plot_basis_functions_alternative(u, chi, n_dyadic=4, degree=3, start_n=2, out_dir=out_dir)
    print(f"Saved: {fname4}")


def cmd_conditioning(args):
    """Analyze conditioning of combined basis."""
    out_dir = setup_output_dir(pkg_dir, "conditioning_analysis")
    u, chi = _make_grid()
    W = build_combined_basis(u, 4, 3, chi, start_n=2)
    mask = u > 0.1
    W_valid = W[mask, :]
    cond, S, corr = analyze_conditioning(W_valid)
    cond_norm, norms = test_normalization(W_valid)
    cond_Q, cond_R, R = test_orthogonalization(W_valid)
    print(f"Condition number: {cond:.2e}")
    print(f"Normalized cond:  {cond_norm:.2e}")
    print(f"QR cond (Q):      {cond_Q:.2e}")
    print(f"QR cond (R):      {cond_R:.2e}")
    fname = plot_conditioning(cond, S, corr, norms, R, out_dir=out_dir)
    print(f"Saved: {fname}")


def cmd_correlation(args):
    """Analyze correlation pattern."""
    out_dir = setup_output_dir(pkg_dir, "correlation_analysis")
    u, chi = _make_grid()
    W = build_combined_basis(u, 4, 3, chi, start_n=2)
    corr, blocks = analyze_correlation_structure(W[u > 0.1], 4, 3)
    print("\\nBlock correlations (dyadic groups):")
    for (n1, n2), block in blocks.items():
        print(f"  Block ({n1},{n2}): mean |corr| = {np.mean(np.abs(block)):.3f}, max = {np.max(np.abs(block)):.3f}")
    cond_orth, corr_orth, phi = test_orthogonal_basis(u, chi, 4, 3, 2)
    print(f"\\nCondition with orthogonal phi: {cond_orth:.2e}")
    fname = plot_correlation_matrix(corr, "Original correlation", out_dir / "corr_original.png")
    fname2 = plot_correlation_matrix(corr_orth, f"Orthogonal phi (cond={cond_orth:.2e})", out_dir / "corr_orthogonal.png")
    print(f"Saved: {fname}, {fname2}")


def cmd_orthogonal(args):
    """Plot orthogonal component s_ij."""
    out_dir = setup_output_dir(pkg_dir, "orthogonal_component")
    u, chi = _make_grid()
    fname = plot_orthogonal_component(u, chi, 4, 3, 2, out_dir=out_dir)
    print(f"Saved: {fname}")


def cmd_tikhonov(args):
    """Test Tikhonov regularization on mio dataset."""
    tables = _load_data(args)
    out_dir = setup_output_dir(pkg_dir, "tikhonov_test")
    u, chi = _make_grid()
    lambda_values = [0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    results = {lam: [] for lam in lambda_values}
    for tab in tables:
        for ch_idx, ch_name, _m in CHANNELS:
            for kind in KINDS:
                prepared = prepare_channel(tab, ch_idx, ch_name, kind, u, chi)
                if prepared is None:
                    continue
                u_valid, f_valid, chi_valid = prepared
                W = build_combined_basis(u_valid, 4, 3, chi_valid, start_n=2)
                for lam in lambda_values:
                    if lam == 0:
                        c, _ = fit_combined_basis(u_valid, f_valid, 4, 3, chi_valid, start_n=2)
                    else:
                        c = fit_tikhonov(W, f_valid, lam)
                    f_fit = W @ c
                    rmse = np.sqrt(np.mean((f_valid - f_fit) ** 2))
                    results[lam].append({"rmse": rmse, "max_coeff": np.max(np.abs(c)), "coeff_norm": np.linalg.norm(c)})
    stats = {}
    for lam in lambda_values:
        rmses = [r["rmse"] for r in results[lam]]
        max_coeffs = [r["max_coeff"] for r in results[lam]]
        stats[lam] = {
            "mean_rmse": np.mean(rmses), "median_rmse": np.median(rmses),
            "mean_max_coeff": np.mean(max_coeffs), "median_max_coeff": np.median(max_coeffs),
        }
        print(f"λ={lam:.0e}: mean RMSE={stats[lam]['mean_rmse']:.4e}, mean max|c|={stats[lam]['mean_max_coeff']:.4e}")
    fname = plot_tikhonov_results(lambda_values, stats, out_dir=out_dir)
    print(f"Saved: {fname}")


def cmd_fewer_dyadic(args):
    """Test fewer dyadic terms with more Legendre."""
    tables = _load_data(args)
    out_dir = setup_output_dir(pkg_dir, "fewer_dyadic")
    u, chi = _make_grid()
    configs = [
        (4, 3, "d4_L3 (16 basis, 8 store)"),
        (3, 4, "d3_L4 (12 basis, 8 store)"),
        (2, 5, "d2_L5 (10 basis, 8 store)"),
        (3, 5, "d3_L5 (15 basis, 10 store)"),
        (2, 6, "d2_L6 (12 basis, 9 store)"),
    ]
    results = {}
    for n_d, n_l, label in configs:
        rmses = []
        for u_v, f_v, chi_v, _meta in _collect_curves(tables, u, chi):
            c, _ = fit_combined_basis(u_v, f_v, n_d, n_l, chi_v, start_n=2)
            W = build_combined_basis(u_v, n_d, n_l, chi_v, start_n=2)
            rmse = np.sqrt(np.mean((f_v - W @ c) ** 2))
            rmses.append(rmse)
        results[label] = np.array(rmses)
        print(f"  {label}: n={len(rmses)}  mean={np.mean(rmses):.4e}  median={np.median(rmses):.4e}")
    print("\\nConditioning:")
    for n_d, n_l, label in configs:
        W = build_combined_basis(u, n_d, n_l, chi, start_n=2)
        cond = np.linalg.cond(W[u > 0.1])
        print(f"  {label}: cond={cond:.2e}")
    fname = plot_rmse_comparison(configs, results, out_dir=out_dir)
    print(f"Saved: {fname}")
'''

with open('/home/prokophapala/git/dftbplus/rust_dftb/tools/sk_compress/test_cli.py', 'w') as f:
    f.write(content.strip() + '\n')
print("Part 1 written")
