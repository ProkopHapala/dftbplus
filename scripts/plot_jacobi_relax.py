#!/usr/bin/env python3
"""FIRE geometry optimization: Jacobi sweep cap 3 against cap 40.

Reads debug/dense_multi/jacobi_relax.csv.
Energy is plotted relative to the cap-40 endpoint, in meV.
"""
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "debug" / "dense_multi" / "jacobi_relax.csv"
OUT = REPO / "debug" / "dense_multi" / "jacobi_relax.png"

ORDER = ["formic", "GC", "diazaphen", "DTH"]
MEV = 27211.386


def load():
    rows = []
    for line in CSV.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        system, cap, step, e, maxf, rms, failed = line.split(",")
        rows.append(
            {
                "system": system,
                "cap": int(cap),
                "step": int(step),
                "e": float(e),
                "maxf": float(maxf),
            }
        )
    return rows


def main():
    rows = load()
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.2), dpi=140, sharex="col")
    for col, name in enumerate(ORDER):
        ax_e = axes[0, col]
        ax_f = axes[1, col]
        cap40 = [r for r in rows if r["system"] == name and r["cap"] == 40]
        cap3 = [r for r in rows if r["system"] == name and r["cap"] == 3]
        if not cap40:
            ax_e.set_title(name)
            continue
        e_ref = cap40[-1]["e"]
        for series, color, ls, label in (
            (cap40, "#08519c", "-", "cap 40"),
            (cap3, "#e6550d", "--", "cap 3"),
        ):
            if not series:
                continue
            steps = [r["step"] for r in series]
            de = [(r["e"] - e_ref) * MEV for r in series]
            ff = [max(r["maxf"], 1e-5) for r in series]
            ax_e.plot(steps, de, ls, color=color, lw=1.4, label=label)
            ax_f.plot(steps, ff, ls, color=color, lw=1.4, label=label)
        ax_e.axhline(0.0, color="0.5", lw=0.6)
        ax_f.axhline(5e-4, color="0.5", ls=":", lw=0.7)
        ax_f.set_yscale("log")
        ax_e.set_title(name)
        ax_f.set_xlabel("FIRE step")
        ax_e.grid(True, ls=":", alpha=0.5)
        ax_f.grid(True, which="both", ls=":", alpha=0.5)
    axes[0, 0].set_ylabel("E − E(cap 40, end) (meV)")
    axes[1, 0].set_ylabel("max |F| (Ha/Å)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("FIRE geometry optimization. Dotted line on |F| is the 5×10⁻⁴ Ha/Å stop.", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
