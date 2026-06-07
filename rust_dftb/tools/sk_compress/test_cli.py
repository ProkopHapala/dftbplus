"""
Lightweight CLI for SK compression analysis.
All reusable logic lives in fitting.py and sk_utils.py.
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

pkg_dir = Path(__file__).resolve().parent
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))

from sk_utils import load_sk_folder, CHANNELS, KINDS, prepare_channel
from sk_utils import _make_grid, _collect_curves, _sweep, _print_rmse_table
from fitting import (
    envelope,
    fit_basis,
    parse_basis_spec,
    make_sweep_configs,
    count_coefficients,
    BasisComponent,
)
from plotting import (
    setup_output_dir,
    plot_rmse_bar_comparison,
    plot_fit_with_error,
)

DEFAULT_SK_PATH = "/home/prokophapala/SIMULATIONS/dftbplus/slakos/mio/mio-1-1"


def _load_data(args):
    sk_path = getattr(args, "sk_path", None) or DEFAULT_SK_PATH
    return load_sk_folder(sk_path)


# =============================================================================
# CLI COMMANDS
# =============================================================================

def cmd_sweep(args):
    """Sweep over basis configurations and compare RMSE."""
    tables = _load_data(args)
    out_dir = setup_output_dir(pkg_dir, "sweep")
    u, chi = _make_grid()

    configs = make_sweep_configs(args.basis)
    mode = args.mode
    method = args.method

    print(f"Sweep mode={mode}, method={method}")
    print(f"Configs: {[label for label, _ in configs]}")

    results = _sweep(tables, u, chi, configs, mode, method,
                     lambda_reg=args.lambda_reg, max_iters=args.max_iters)

    ncoefs = {label: count_coefficients(comps, mode) for label, comps in configs}
    _print_rmse_table(results, title="Basis Sweep Results", ncoefs=ncoefs)

    if len(configs) > 1:
        fname = plot_rmse_bar_comparison(
            results,
            out_dir / "sweep_comparison.png",
            title=f"Basis Sweep: {args.basis}",
        )
        print(f"Saved: {fname}")

    if args.plot_examples:
        _plot_example_fits(tables, u, chi, configs, mode, method, out_dir, args.plot_examples)


def cmd_compare(args):
    """Compare fine-tuning bases (e.g. Chebyshev vs Legendre vs Monomial)."""
    tables = _load_data(args)
    out_dir = setup_output_dir(pkg_dir, "compare")
    u, chi = _make_grid()

    families = args.families.split(",")
    degree = args.degree
    dyadic_n = args.dyadic_n

    configs = []
    for family in families:
        family = family.strip()
        if args.separable:
            spec = f"dyadic:{dyadic_n};{family}:{degree}"
        else:
            spec = f"dyadic:{dyadic_n}+{family}:{degree}"
        comps, mode = parse_basis_spec(spec)
        configs.append((family, comps))

    mode = "separable" if args.separable else "product"
    method = "als" if args.separable else "lstsq"

    results = _sweep(tables, u, chi, configs, mode, method)
    _print_rmse_table(results, title="Fine-Tuning Basis Comparison")

    fname = plot_rmse_bar_comparison(
        results,
        out_dir / "basis_comparison.png",
        title="Fine-Tuning Basis Comparison",
    )
    print(f"Saved: {fname}")


def cmd_fit_single(args):
    """Fit a single representative curve and plot."""
    tables = _load_data(args)
    out_dir = setup_output_dir(pkg_dir, "single_fit")
    u, chi = _make_grid()

    comps, mode = parse_basis_spec(args.basis)
    method = args.method

    for u_v, f_v, chi_v, meta in _collect_curves(tables, u, chi):
        if args.pair and args.pair not in meta["pair"]:
            continue
        if args.channel and meta["ch_name"] != args.channel:
            continue
        if args.kind and meta["kind"] != args.kind:
            continue

        c, f_fit, rmse = fit_basis(u_v, f_v, chi_v, comps, mode, method,
                                   lambda_reg=args.lambda_reg, max_iters=args.max_iters)

        print(f"Fit {meta['pair']} {meta['ch_name']} {meta['kind']}: RMSE={rmse:.4e}")
        if isinstance(c, tuple):
            print(f"  Coeff shapes: {[x.shape for x in c]}")
        else:
            print(f"  Coeff shape: {c.shape}")

        fname = plot_fit_with_error(
            u_v, f_v, f_fit,
            out_dir / "fit.png",
            title=f"{meta['pair']} {meta['ch_name']} {meta['kind']}",
        )
        print(f"Saved: {fname}")
        return

    print("No matching curve found")


def _plot_example_fits(tables, u, chi, configs, mode, method, out_dir, n_examples):
    """Plot example fits for the first n_examples curves."""
    curves = []
    for u_v, f_v, chi_v, meta in _collect_curves(tables, u, chi):
        curves.append((u_v, f_v, chi_v, meta))
        if len(curves) >= n_examples:
            break

    for idx, (u_v, f_v, chi_v, meta) in enumerate(curves):
        fig, axes = plt.subplots(1, len(configs), figsize=(5 * len(configs), 4))
        if len(configs) == 1:
            axes = [axes]
        for ax, (label, comps) in zip(axes, configs):
            try:
                c, f_fit, rmse = fit_basis(u_v, f_v, chi_v, comps, mode, method)
                ax.plot(u_v, f_v, "b-", lw=1.5, alpha=0.7, label="Original")
                ax.plot(u_v, f_fit, "r--", lw=1.5, alpha=0.8, label="Fit")
                ax.set_title(f"{label}\nRMSE={rmse:.2e}")
                ax.legend()
                ax.grid(True, alpha=0.3)
            except Exception:
                ax.set_title(f"{label}\nFAILED")
        fig.suptitle(f"{meta['pair']} {meta['ch_name']} {meta['kind']}", fontsize=14)
        fig.tight_layout()
        fname = out_dir / f"example_fit_{idx}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved: {fname}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified SK compression analysis CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Sweep over basis configurations")
    p_sweep.add_argument("--basis", required=True, help="Basis sweep spec (e.g. 'dyadic:2..6+legendre:1..5' or 'chebyshev:3[v_power:4,6,8]')")
    p_sweep.add_argument("--mode", default="product", choices=["concat", "product", "separable"])
    p_sweep.add_argument("--method", default="lstsq", choices=["lstsq", "tikhonov", "als"])
    p_sweep.add_argument("--lambda-reg", type=float, default=1e-12)
    p_sweep.add_argument("--max-iters", type=int, default=50)
    p_sweep.add_argument("--plot-examples", type=int, default=0, help="Number of example fit plots")
    p_sweep.add_argument("--sk-path", default=None)

    # compare
    p_cmp = subparsers.add_parser("compare", help="Compare fine-tuning bases")
    p_cmp.add_argument("--families", default="chebyshev,legendre,monomial,hermite,bspline")
    p_cmp.add_argument("--degree", type=int, default=3)
    p_cmp.add_argument("--dyadic-n", type=int, default=4)
    p_cmp.add_argument("--separable", action="store_true")
    p_cmp.add_argument("--sk-path", default=None)

    # fit-single
    p_single = subparsers.add_parser("fit-single", help="Fit a single curve")
    p_single.add_argument("--basis", required=True)
    p_single.add_argument("--mode", default="product", choices=["concat", "product", "separable"])
    p_single.add_argument("--method", default="lstsq", choices=["lstsq", "tikhonov", "als"])
    p_single.add_argument("--lambda-reg", type=float, default=1e-12)
    p_single.add_argument("--max-iters", type=int, default=50)
    p_single.add_argument("--pair", default=None)
    p_single.add_argument("--channel", default=None)
    p_single.add_argument("--kind", default=None, choices=["H", "S"])
    p_single.add_argument("--sk-path", default=None)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    globals()[f"cmd_{args.command.replace('-', '_')}"](args)


if __name__ == "__main__":
    main()
