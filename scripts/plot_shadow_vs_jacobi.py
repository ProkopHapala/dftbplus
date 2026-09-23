#!/usr/bin/env python3
"""Shadow potential against a full Jacobi SCC.

Accuracy is one 0.02 Å step, batch 1. The charge Jacobian in that run was
an exact finite difference, used only to test the Newton formula.
Cost is batch 256: a full SCC against two diagonalizations, which is the
electronic cost once that Jacobian is already known.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "debug" / "dense_multi"

# name, force % level0, force % level1, |ΔE| meV level0, |ΔE| meV level1
# DTH level-1 force printed as 0.00%, i.e. below 0.005.
ACC = [
    ("formic", 65.29, 0.16, 11.327, 0.016),
    ("GC", 15.05, 0.15, 39.626, 0.029),
    ("diazaphen", 24.55, 0.04, 29.537, 0.011),
    ("DTH", 0.57, 0.005, 4.637, 0.009),
]
# name, SCC ms, 2-Jacobi ms, FD-Jacobian estimate ms, speedup of 2-Jacobi
COST = [
    ("formic", 3.94, 1.12, 12.0, 3.51),
    ("GC", 20.48, 5.68, 170.0, 3.60),
    ("diazaphen", 69.35, 16.45, 708.0, 4.22),
    ("DTH", 473.38, 267.90, 22772.0, 1.77),
]

C_SCC = "#08519c"
C_L0 = "#e6550d"
C_L1 = "#6a3d9a"


def grouped(ax, labels, series, ylabel, title, hline=None, hlabel=None):
    x = np.arange(len(labels))
    n = len(series)
    w = 0.8 / n
    for i, (vals, color, label) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, color=color, label=label, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", which="major", alpha=0.3, zorder=0)
    if hline is not None:
        ax.axhline(hline, color="black", lw=1.0, ls="--", label=hlabel, zorder=4)
    ax.legend(fontsize=8, frameon=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = [r[0] for r in ACC]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.0), sharex=True)
    grouped(
        axes[0],
        labels,
        [
            ([r[1] for r in ACC], C_L0, "level 0: one Jacobi at carried q"),
            ([r[2] for r in ACC], C_L1, "level 1: Newton step, then one Jacobi"),
        ],
        "max |ΔF| / max |F_SCC|   (%)",
        "Force against a full Jacobi SCC\nbatch 1, one 0.02 Å step, kT = 0, exact charge Jacobian",
        hline=0.5,
        hlabel="0.5% force",
    )
    axes[0].annotate(
        "DTH level 1 printed 0.00%\n(bar drawn at 0.005)",
        xy=(3, 0.005),
        xytext=(2.15, 2.5),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    grouped(
        axes[1],
        labels,
        [
            ([r[3] for r in ACC], C_L0, "level 0"),
            ([r[4] for r in ACC], C_L1, "level 1"),
        ],
        "|ΔE| against Jacobi SCC   (meV)",
        "Energy against the same Jacobi SCC",
        hline=0.1,
        hlabel="0.1 meV",
    )
    fig.tight_layout()
    acc_path = OUT / "shadow_vs_jacobi_accuracy.png"
    fig.savefig(acc_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    labels = [r[0] for r in COST]
    grouped(
        ax,
        labels,
        [
            ([r[1] for r in COST], C_SCC, "Jacobi SCC"),
            ([r[2] for r in COST], C_L1, "two Jacobi, charge Jacobian already known"),
        ],
        "wall time for 256 copies   (ms)",
        "Two diagonalizations against a full Jacobi SCC\n"
        "batch 256, mean of 3. This is not the bold step, and not a finite-difference Jacobian.",
    )
    ax.set_xlabel("molecule")
    fig.tight_layout()
    cost_path = OUT / "shadow_vs_jacobi_cost.png"
    fig.savefig(cost_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(acc_path)
    print(cost_path)


if __name__ == "__main__":
    main()
