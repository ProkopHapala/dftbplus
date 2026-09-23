#!/usr/bin/env python3
"""Accuracy and SCC time versus Jacobi sweep cap.

Reads debug/dense_multi/jacobi_sweeps.csv. One geometry step of 0.02 Å,
batch 1, compared with the same step at cap 40. Duplicate (system, cap)
rows keep the first time and the last accuracy.
"""
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "debug" / "dense_multi" / "jacobi_sweeps.csv"
OUT = REPO / "debug" / "dense_multi" / "jacobi_sweeps_accuracy.png"

COLORS = {
    "formic": "#e6550d",
    "GC": "#08519c",
    "diazaphen": "#238b45",
    "DTH": "#6a3d9a",
}
ORDER = ["formic", "GC", "diazaphen", "DTH"]


def load():
    rows = []
    for line in CSV.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        s, n, cap, ms, iters, failed, de, fp, sweeps, rel, stop = line.split(",")
        rows.append(
            {
                "system": s,
                "cap": int(cap),
                "ms": float(ms),
                "iters": int(iters),
                "failed": int(failed),
                "de": abs(float(de)),
                "fp": float(fp),
            }
        )
    return rows


def series(rows, system):
    by = {}
    for r in rows:
        if r["system"] != system:
            continue
        slot = by.setdefault(r["cap"], {"ms": r["ms"], "de": r["de"], "fp": r["fp"], "iters": r["iters"]})
        slot["de"] = r["de"]
        slot["fp"] = r["fp"]
        slot["iters"] = r["iters"]
    caps = sorted(by)
    return caps, [by[c] for c in caps]


def main():
    rows = load()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), dpi=140)
    ax_e, ax_f, ax_t = axes
    for name in ORDER:
        caps, pts = series(rows, name)
        if not caps:
            continue
        c = COLORS[name]
        de = [max(p["de"], 1e-4) for p in pts]
        fp = [max(p["fp"], 1e-4) for p in pts]
        ms = [p["ms"] for p in pts]
        ax_e.plot(caps, de, "o-", color=c, label=name, ms=5)
        ax_f.plot(caps, fp, "o-", color=c, label=name, ms=5)
        ax_t.plot(caps, ms, "o-", color=c, label=name, ms=5)
        for cap, p in zip(caps, pts):
            if p["iters"] >= 40:
                ax_e.plot(cap, max(p["de"], 1e-4), "x", color=c, ms=9, mew=1.5)
                ax_f.plot(cap, max(p["fp"], 1e-4), "x", color=c, ms=9, mew=1.5)
    ax_e.axhline(0.1, color="0.4", ls="--", lw=0.8)
    ax_f.axhline(0.5, color="0.4", ls="--", lw=0.8)
    ax_e.set_yscale("log")
    ax_f.set_yscale("log")
    ax_e.set_ylabel("|ΔE| vs cap 40 (meV)")
    ax_f.set_ylabel("max |ΔF| / max |F| (%)")
    ax_t.set_ylabel("geometry SCC (ms)")
    for ax in axes:
        ax.set_xlabel("sweep cap")
        ax.set_xticks([1, 2, 3, 4, 6, 8, 12, 20, 40])
        ax.grid(True, which="both", ls=":", alpha=0.5)
    ax_e.legend(frameon=False, fontsize=8)
    ax_e.set_title("energy")
    ax_f.set_title("force")
    ax_t.set_title("time")
    fig.suptitle("Jacobi SCC at a 0.02 Å step, batch 1. Cross = charge RMS not met in 40 iterations. Floor 1e-4.", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
