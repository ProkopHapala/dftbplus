"""
Plotting utilities for SK compression analysis.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from matplotlib.lines import Line2D


def setup_output_dir(pkg_dir: Path, name: str) -> Path:
    """Create output_refined/<name> directory."""
    out_dir = pkg_dir / "output_refined" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_basis_functions(u, chi, n_dyadic=4, degree_leg=3, start_n=2, out_dir=None):
    """3-panel plot: Legendre, dyadic, combined basis."""
    from fitting import shifted_legendre_vander, build_dyadic_basis_vander, build_combined_basis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    u_safe = np.clip(u, 1e-8, 1.0)
    v = 1.0 - u_safe

    ax = axes[0]
    legendre = shifted_legendre_vander(u, degree_leg)
    for k in range(degree_leg + 1):
        ax.plot(u, legendre[:, k], lw=2, label=f"P_{k}(2u-1)")
    ax.set_title('Shifted Legendre P_k(2u-1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    dyadic = build_dyadic_basis_vander(v, n_dyadic, start_n=start_n)
    for n in range(n_dyadic):
        ax.plot(u, dyadic[:, n], lw=2, label=f"p_{{{n+start_n}}}")
    ax.set_title('Dyadic basis p_n = (1-u)^(2^n)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    W = build_combined_basis(u, n_dyadic, degree_leg, chi, start_n=start_n)
    n_total = n_dyadic * (degree_leg + 1)
    colors = plt.cm.tab20(np.linspace(0, 1, n_total))
    for idx in range(n_total):
        n = idx // (degree_leg + 1) + start_n
        k = idx % (degree_leg + 1)
        ax.plot(u, W[:, idx], color=colors[idx], lw=1, label=f"W_{{{n},{k}}}")
    ax.set_title(f'Combined χ·p_n·P_k ({n_total} funcs)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = (out_dir or Path('.')) / "basis_functions.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def plot_basis_functions_ortho(u, chi, n_dyadic=4, degree_leg=3, start_n=2, out_dir=None):
    """Plot original, forward-ortho, and backward-ortho dyadic basis + combined."""
    from fitting import (shifted_legendre_vander, build_dyadic_basis_vander,
                         build_combined_basis, orthogonalize_dyadic_basis)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    u_safe = np.clip(u, 1e-8, 1.0)
    v = 1.0 - u_safe
    dyadic_raw = build_dyadic_basis_vander(v, n_dyadic, start_n=start_n)
    legendre = shifted_legendre_vander(u, degree_leg)

    # Row 0: dyadic part
    for n in range(n_dyadic):
        axes[0, 0].plot(u, dyadic_raw[:, n], lw=2, label=f"p_{{{n+start_n}}}")
    axes[0, 0].set_title('Original dyadic')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    dyadic_fwd, _ = orthogonalize_dyadic_basis(dyadic_raw, direction="forward", normalize=True)
    for n in range(n_dyadic):
        axes[0, 1].plot(u, dyadic_fwd[:, n], lw=2, label=f"q_{{{n+start_n}}}^{{fwd}}")
    axes[0, 1].set_title('Forward Gram-Schmidt (preserves smoothest)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    dyadic_bwd, _ = orthogonalize_dyadic_basis(dyadic_raw, direction="backward", normalize=True)
    for n in range(n_dyadic):
        axes[0, 2].plot(u, dyadic_bwd[:, n], lw=2, label=f"q_{{{n+start_n}}}^{{bwd}}")
    axes[0, 2].set_title('Backward Gram-Schmidt (preserves sharpest)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Row 1: combined basis with each dyadic variant
    for col, (dyadic_part, label) in enumerate([(dyadic_raw, "orig"),
                                                   (dyadic_fwd, "fwd"),
                                                   (dyadic_bwd, "bwd")]):
        n_total = n_dyadic * (degree_leg + 1)
        W = np.zeros((len(u), n_total))
        for n in range(n_dyadic):
            for k in range(degree_leg + 1):
                W[:, n * (degree_leg + 1) + k] = chi * dyadic_part[:, n] * legendre[:, k]
        colors = plt.cm.tab20(np.linspace(0, 1, n_total))
        for idx in range(min(n_total, 16)):
            n = idx // (degree_leg + 1) + start_n
            k = idx % (degree_leg + 1)
            axes[1, col].plot(u, W[:, idx], color=colors[idx], lw=1, label=f"W_{{{n},{k}}}")
        axes[1, col].set_title(f'Combined χ·q_n·P_k ({label})')
        axes[1, col].legend(fontsize=7, ncol=2)
        axes[1, col].grid(True, alpha=0.3)

    fig.suptitle('Dyadic Basis: Original vs Forward vs Backward Gram-Schmidt', fontsize=14)
    fig.tight_layout()
    fname = (out_dir or Path('.')) / "basis_functions_orthogonalized.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def plot_basis_functions_alternative(u, chi, n_dyadic=4, degree=3, start_n=2, out_dir=None):
    """Plot all fine-tuning bases: Legendre, Chebyshev, Hermite, B-spline, Monomial."""
    from fitting import (shifted_legendre_vander, build_dyadic_basis_vander,
                         build_chebyshev_basis_vander, build_hermite_basis_vander,
                         build_bspline_basis_vander, build_monomial_basis_vander)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    u_safe = np.clip(u, 1e-8, 1.0)
    v = 1.0 - u_safe
    dyadic = build_dyadic_basis_vander(v, n_dyadic, start_n=start_n)

    bases = [
        ("Legendre P_k(2u-1)", shifted_legendre_vander(u, degree), axes[0, 0]),
        ("Chebyshev T_k(2u-1)", build_chebyshev_basis_vander(u, degree), axes[0, 1]),
        ("Hermite-like exp(-4(u-0.5)^2)(u-0.5)^k", build_hermite_basis_vander(u, degree, alpha=4.0), axes[0, 2]),
        ("Monomial u^k", build_monomial_basis_vander(u_safe, degree), axes[1, 0]),
        ("B-spline cubic", build_bspline_basis_vander(u_safe, degree), axes[1, 1]),
    ]

    for title, fine_basis, ax in bases:
        if fine_basis is None:
            continue
        colors = plt.cm.viridis(np.linspace(0, 0.8, degree + 1))
        for k in range(degree + 1):
            ax.plot(u, fine_basis[:, k], color=colors[k], lw=2, label=f"φ_{k}")
        ax.set_title(title)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', lw=0.5)

    # Panel: Custom linear basis functions
    ax = axes[1, 2]
    from fitting import build_custom_basis_vander
    W_custom = build_custom_basis_vander(u, Rc=4.0)
    colors = plt.cm.viridis(np.linspace(0, 0.8, 6))
    labels = ['(1-u)^4', '(1-u)^4·u', '(1-u)^4·u^2', '(1-u)^4·u^4', '(1-u)^4·r', '(1-u)^4·r^2']
    for k in range(6):
        ax.plot(u, W_custom[:, k], color=colors[k], lw=2, label=labels[k])
    ax.set_title('Custom linear: (1-u)^4·{1,u,u2,u4,r,r2}')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5)

    fig.suptitle('Alternative Fine-Tuning Bases (all combined with dyadic)', fontsize=14)
    fig.tight_layout()
    fname = (out_dir or Path('.')) / "basis_functions_alternative.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def plot_correlation_matrix(corr, title="Correlation", out_path=None):
    """Plot correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path = out_path or Path("correlation.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_orthogonal_component(u, chi, n_dyadic=4, degree_leg=3, start_n=2, out_dir=None):
    """Plot s_ij = sqrt(1-c_ij^2) with block structure."""
    from fitting import build_dyadic_basis_vander, build_combined_basis, compute_orthogonal_component
    n_legendre = degree_leg + 1
    mask = u > 0.1

    dyadic = build_dyadic_basis_vander(np.clip(1 - u, 1e-8, 1), n_dyadic, start_n)
    corr_d = np.corrcoef(dyadic[mask, :].T)
    s2_d = compute_orthogonal_component(corr_d)
    log_s_d = np.log10(np.sqrt(np.maximum(s2_d, 1e-8)))

    W = build_combined_basis(u, n_dyadic, degree_leg, chi, start_n=start_n)
    corr_p = np.corrcoef(W[mask, :].T)
    s2_p = compute_orthogonal_component(corr_p)
    log_s_p = np.log10(np.sqrt(np.maximum(s2_p, 1e-8)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes[0, 0].imshow(log_s_d, aspect='auto', cmap='viridis', vmin=-4, vmax=0)
    axes[0, 0].set_title('Dyadic basis: log10(s_ij)')
    axes[0, 1].imshow(log_s_p, aspect='auto', cmap='viridis', vmin=-4, vmax=0)
    axes[0, 1].set_title('Product basis: log10(s_ij)')
    for i in range(1, n_dyadic):
        axes[0, 1].axhline(i*n_legendre-0.5, color='white', lw=1)
        axes[0, 1].axvline(i*n_legendre-0.5, color='white', lw=1)

    idx = slice(2*n_legendre, 3*n_legendre)
    axes[1, 0].imshow(log_s_p[idx, idx], aspect='auto', cmap='viridis', vmin=-4, vmax=0)
    axes[1, 0].set_title('Within block D4')
    axes[1, 1].imshow(log_s_p[n_legendre:2*n_legendre, 2*n_legendre:3*n_legendre], aspect='auto', cmap='viridis', vmin=-4, vmax=0)
    axes[1, 1].set_title('Cross block D3 vs D4')

    fig.tight_layout()
    fname = (out_dir or Path('.')) / "orthogonal_component.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def plot_conditioning(cond, S, corr, norms, R, out_dir=None):
    """Plot singular values, correlation, norms, R matrix."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].semilogy(range(1, len(S)+1), S, 'o-', color='steelblue')
    axes[0, 0].set_title('Singular values')
    axes[0, 0].grid(True, alpha=0.3)
    im = axes[0, 1].imshow(corr, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 1].set_title('Correlation matrix')
    plt.colorbar(im, ax=axes[0, 1])
    axes[1, 0].bar(range(len(norms)), norms, color='steelblue')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Column norms')
    im2 = axes[1, 1].imshow(np.abs(R), aspect='auto', cmap='viridis')
    axes[1, 1].set_title('|R| from QR')
    plt.colorbar(im2, ax=axes[1, 1])
    fig.suptitle(f'Conditioning (cond={cond:.2e})', fontsize=14)
    fig.tight_layout()
    fname = (out_dir or Path('.')) / "conditioning.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def plot_tikhonov_results(lambda_values, stats, out_dir=None):
    """Plot RMSE and coeff magnitude vs lambda."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    lams = lambda_values[1:]
    axes[0].semilogx(lams, [stats[lam]['mean_rmse'] for lam in lams], 'o-', label='Mean RMSE')
    axes[0].axhline(stats[0]['mean_rmse'], color='r', ls=':', label='No reg')
    axes[0].set_xlabel('λ')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Accuracy vs regularization')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].semilogx(lams, [stats[lam]['mean_max_coeff'] for lam in lams], 'o-', label='Mean max |c|')
    axes[1].axhline(stats[0]['mean_max_coeff'], color='r', ls=':', label='No reg')
    axes[1].set_xlabel('λ')
    axes[1].set_ylabel('Coefficient magnitude')
    axes[1].set_title('Coefficients vs regularization')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fname = (out_dir or Path('.')) / "tikhonov.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def plot_coefficient_stats(C, n_dyadic, degree_leg, out_dir=None):
    """Plot mean|C|, std, max, fraction near-zero as heatmaps."""
    n_leg = degree_leg + 1
    C3d = C.reshape(-1, n_dyadic, n_leg)
    mean_abs = np.mean(np.abs(C3d), axis=0)
    std_abs = np.std(np.abs(C3d), axis=0)
    frac_zero = np.mean(np.abs(C3d) < 1e-6, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title in zip(axes, [mean_abs, std_abs, frac_zero],
                               ['Mean |c_nk|', 'Std |c_nk|', 'Frac |c|<1e-6']):
        im = ax.imshow(np.log10(np.clip(data, 1e-12, None)) if 'Mean' in title else data,
                       aspect='auto', cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('Legendre k')
        ax.set_ylabel('Dyadic n')
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fname = (out_dir or Path('.')) / "coefficient_stats.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def plot_rmse_comparison(configs, results, out_dir=None):
    """Bar chart comparing RMSE across configurations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [label for _, _, label in configs]
    mean_rmses = [np.mean(results[l]) for l in labels]
    x = np.arange(len(labels))
    ax.bar(x, mean_rmses, color='steelblue')
    ax.set_ylabel('Mean RMSE')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fname = (out_dir or Path('.')) / "rmse_comparison.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def plot_mio_cluster(grouped_curves, u, out_dir=None, x_mode='u'):
    """
    Cluster plots per channel type with element group colors.
    Curves colored by element-period group.
    """
    GROUP_COLORS = {
        "H-2": "C0",   # H with C,N,O
        "2-2": "C1",   # C,N,O with C,N,O
        "2-3": "C2",   # C,N,O with S,P
        "H-3": "C3",   # H with S,P
        "3-3": "C4",   # S,P with S,P
    }
    GROUP_LABELS = {
        "H-2": "H–{C,N,O}",
        "2-2": "{C,N,O}–{C,N,O}",
        "2-3": "{C,N,O}–{S,P}",
        "H-3": "H–{S,P}",
        "3-3": "{S,P}–{S,P}",
    }

    def classify_pair(pair):
        s1, s2 = pair.split('-')
        groups = {'H': 'H', 'C': '2', 'N': '2', 'O': '2', 'S': '3', 'P': '3'}
        g1, g2 = groups.get(s1, '?'), groups.get(s2, '?')
        return f"{g1}-{g2}"

    xlabel = "u = r / Rc" if x_mode == 'u' else "r (Å)"

    for ch_name, curve_list in grouped_curves.items():
        if not curve_list:
            continue
        h_curves = [c for c in curve_list if c[2].endswith('_H')]
        s_curves = [c for c in curve_list if c[2].endswith('_S')]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=(x_mode == 'u'), sharey=False)
        for ax, curves, title_suffix in [(axes[0], h_curves, "H"), (axes[1], s_curves, "S")]:
            for f_u, m, label, Rc in curves:
                pair = label.split('_')[0]
                group = classify_pair(pair)
                color = GROUP_COLORS.get(group, "C7")
                if x_mode == 'r':
                    r = u * Rc
                    ax.plot(r, f_u, alpha=0.5, color=color, lw=0.3)
                else:
                    ax.plot(u, f_u, alpha=0.5, color=color, lw=0.3)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Value")
            ax.set_title(f"{title_suffix}  —  {len(curves)} curves")
            ax.grid(True, alpha=0.3)
            legend_elements = [
                Line2D([0], [0], color=GROUP_COLORS[g], lw=2, label=GROUP_LABELS[g])
                for g in ["H-2", "2-2", "2-3", "H-3", "3-3"]
                if any(classify_pair(c[2].split('_')[0]) == g for c in curves)
            ]
            ax.legend(handles=legend_elements, loc="best", fontsize=8)
        fig.suptitle(f"mio  {ch_name}", fontsize=12)
        fig.tight_layout()
        fname = (out_dir or Path('.')) / f"cluster_mio_{ch_name}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved: {fname}")


def plot_legendre_contribution(u, chi, n_dyadic=4, degree_leg=3, start_n=2, out_dir=None):
    """Plot raw products, enveloped products, and Legendre polynomials."""
    from fitting import shifted_legendre_vander, build_dyadic_basis_vander
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    u_safe = np.clip(u, 1e-8, 1.0)
    v = 1.0 - u_safe
    dyadic = build_dyadic_basis_vander(v, n_dyadic, start_n=start_n)
    legendre = shifted_legendre_vander(u, degree_leg)

    ax = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_dyadic * (degree_leg + 1)))
    idx = 0
    for n in range(n_dyadic):
        for k in range(min(3, degree_leg + 1)):
            prod = dyadic[:, n] * legendre[:, k]
            ax.plot(u, prod, color=colors[idx], lw=1.0, label=f"p_{n+start_n}·P_{k}")
            idx += 1
    ax.set_title('Raw products: p_n(u)·P_k(u) (without χ)')
    ax.set_ylabel('Value')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    idx = 0
    for n in range(n_dyadic):
        for k in range(min(3, degree_leg + 1)):
            prod = chi * dyadic[:, n] * legendre[:, k]
            ax.plot(u, prod, color=colors[idx], lw=1.0, label=f"χ·p_{n+start_n}·P_{k}")
            idx += 1
    ax.set_title('Enveloped: χ(u)·p_n(u)·P_k(u)')
    ax.set_ylabel('Value')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    colors_leg = plt.cm.plasma(np.linspace(0, 0.8, degree_leg + 1))
    for k in range(degree_leg + 1):
        ax.plot(u, legendre[:, k], color=colors_leg[k], lw=2.0, label=f"P_{k}(2u-1)")
    ax.set_title('Legendre fine-tuning polynomials P_k(2u-1)')
    ax.set_xlabel('u = r / Rc')
    ax.set_ylabel('P_k(u)')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5)

    ax = axes[1, 1]
    ax.text(0.5, 0.5, 'See coefficient_matrix_statistics.png\nfor per-cell analysis',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('See separate coefficient matrix plot')
    ax.axis('off')

    fig.suptitle(f'Legendre Fine-Tuning Contribution (d={n_dyadic}, L={degree_leg})', fontsize=13)
    fig.tight_layout()
    fname = (out_dir or Path('.')) / "legendre_contribution.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def plot_pareto_summary_and_splits(results_all, out_dir=None):
    """Summary pareto + split comparison plots."""
    out_dir = out_dir or Path('.')
    # Summary: all channels combined
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for kind_idx, kind in enumerate(['H', 'S']):
        ax = axes[kind_idx]
        data = [r for r in results_all if r['kind'] == kind]
        pure = sorted([r for r in data if r['method'] == 'pure_dyadic'], key=lambda x: x['n_store'])
        if pure:
            ax.plot([r['n_store'] for r in pure], [r['rmse'] for r in pure], 'o-', color='blue', label='Pure dyadic')
        combined = [r for r in data if r['method'] == 'combined']
        for nd in sorted(set(r['n_dyadic'] for r in combined)):
            subset = sorted([r for r in combined if r['n_dyadic'] == nd], key=lambda x: x['n_store'])
            ax.plot([r['n_store'] for r in subset], [r['rmse'] for r in subset], 's--', label=f'Combined d={nd}')
        ax.set_xlabel('Storage per function')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{kind} — All channels')
        ax.set_yscale('log')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    fig.suptitle('Pareto Summary: All Channels Combined', fontsize=14)
    fig.tight_layout()
    fname = out_dir / "summary_pareto.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")

    # Split comparison at store=8
    splits = [(6, 1), (5, 2), (4, 3), (3, 4), (2, 5)]
    for ch_name in ['ss', 'sp', 'ppσ', 'ppπ']:
        for kind in ['H', 'S']:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for ax_idx, (metric, ylabel) in enumerate([('rmse', 'RMSE'), ('rel_rmse', 'Relative RMSE')]):
                ax = axes[ax_idx]
                for n_d, n_l in splits:
                    data = [r for r in results_all if r['ch_name'] == ch_name and r['kind'] == kind
                            and r['method'] == 'combined' and r['n_dyadic'] == n_d and r['deg_leg'] == n_l]
                    if data:
                        vals = [r[metric] for r in data]
                        label = f"d{n_d}_L{n_l} (store={n_d + (n_l+1)})"
                        ax.scatter(range(len(vals)), sorted(vals), label=label, s=20, alpha=0.6)
                ax.set_xlabel('Curve index (sorted)')
                ax.set_ylabel(ylabel)
                ax.set_title(f'{ch_name} {kind}')
                ax.set_yscale('log')
                ax.legend(fontsize=8, loc='best')
                ax.grid(True, alpha=0.3)
            fig.suptitle(f'{ch_name} {kind} — Split comparison at store=8', fontsize=13)
            fig.tight_layout()
            fname = out_dir / f"split_{ch_name}_{kind}_store8.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"Saved: {fname}")


def plot_rmse_bar_comparison(results_dict, out_path, title="", baseline=None):
    """Generic bar chart: mean/median RMSE per config."""
    names = list(results_dict.keys())
    means = [np.mean(np.array(results_dict[n])) for n in names]
    medians = [np.median(np.array(results_dict[n])) for n in names]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    w = 0.35
    colors_m = ["green" if n == baseline else "steelblue" for n in names]
    colors_d = ["lightgreen" if n == baseline else "coral" for n in names]
    ax.bar(x - w / 2, means, w, label="Mean", color=colors_m, edgecolor="black")
    ax.bar(x + w / 2, medians, w, label="Median", color=colors_d, edgecolor="black")
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


def plot_scatter_comparison(xvals, yvals_dict, out_path, title="", xlabel="", ylabel="", log=True):
    """Generic scatter plot with multiple y-series and y=x line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(yvals_dict)))
    for (label, y), color in zip(yvals_dict.items(), colors):
        ax.scatter(xvals, y, alpha=0.4, s=20, color=color, label=label)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    all_vals = [xvals] + [v for v in yvals_dict.values()]
    vmin = min(v.min() for v in all_vals)
    vmax = max(v.max() for v in all_vals)
    lim = [vmin, vmax]
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


def plot_fit_with_error(u_v, f_v, f_fit, out_path, title=""):
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
