#!/usr/bin/env python3
"""Throughput, speedup, and SCC iterations vs batch.

Reads debug/dense_multi/throughput.csv. One figure per molecule.
The log axis is labeled with the real batch sizes (8, 16, 32, …).
Speedup is against cpu 8 threads, warm, at the same batch.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CSV = REPO / "debug" / "dense_multi" / "throughput.csv"
OUT = REPO / "debug" / "dense_multi"

# 1-thread and 8-thread must not share a color or a marker.
SERIES = [
    ("cpu 1 thread, warm", "#2ca02c", "^", "--"),
    ("cpu 8 threads, cold", "#7f7f7f", "s", ":"),
    ("cpu 8 threads, warm", "black", "o", "-"),
    ("gpu jacobi cold", "#9ecae1", "D", "-"),
    ("gpu jacobi warm", "#08519c", "o", "-"),
    ("gpu shadow (2 Jacobi)", "#238b45", "h", "-"),
    ("gpu purify cold", "#fdae6b", "s", "-"),
    ("gpu purify warm", "#d94801", "o", "-"),
    ("gpu bold", "#6a3d9a", "P", "-"),
    ("gpu bold x2", "#e7298a", "P", "--"),
    ("gpu bold x3", "#c51b8a", "X", ":"),
    ("gpu bold diis", "#4a148c", "v", "-"),
]
REF = "cpu 8 threads, warm"


def load():
    rows = []
    for line in CSV.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        method = ",".join(parts[3:-5])
        wall, iters, rate, failed, dq = parts[-5:]
        rows.append({
            "system": parts[0], "n": int(parts[1]), "batch": int(parts[2]),
            "method": method, "rate": float(rate), "iters": int(iters),
            "failed": int(failed), "dq": float(dq),
        })
    return rows


def series_points(sub, method):
    """Last row wins. Drop the n>64 Jacobi-cold points whose charges are off."""
    seen = {}
    for r in sub:
        if r["method"] != method or r["failed"] != 0:
            continue
        if method == "gpu jacobi cold" and r["dq"] > 1e-3:
            continue
        b = r["batch"]
        pow2 = b <= 1024 and (b & (b - 1)) == 0
        # GC keeps batch 400: that is where Jacobi peaks. The other three
        # figures are powers of two through 1024.
        if r["system"] == "GC":
            if b > 1024:
                continue
        elif not pow2:
            continue
        seen[r["batch"]] = r
    return [seen[b] for b in sorted(seen)]


def decimal_log_axis(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(round(v))}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", alpha=0.3)


def draw(ax, pts, color, marker, ls, label, ykey):
    if not pts:
        return None
    return ax.plot(
        [p["batch"] for p in pts], [p[ykey] for p in pts],
        color=color, marker=marker, ls=ls, lw=2, ms=7, label=label,
        zorder=5 if marker == "^" else 3,
    )[0]


def main():
    rows = load()
    systems = []
    for r in rows:
        if r["system"] not in systems:
            systems.append(r["system"])
    OUT.mkdir(parents=True, exist_ok=True)
    for name in systems:
        sub = [r for r in rows if r["system"] == name]
        n = max(r["n"] for r in sub)
        ref = {p["batch"]: p["rate"] for p in series_points(sub, REF)}
        fig, axes = plt.subplots(3, 1, figsize=(8.2, 11), sharex=True)
        handles = []
        ticks = set()
        for method, color, marker, ls in SERIES:
            pts = series_points(sub, method)
            if not pts:
                continue
            ticks.update(p["batch"] for p in pts)
            h = draw(axes[0], pts, color, marker, ls, method, "rate")
            handles.append(h)
            both = [p for p in pts if p["batch"] in ref and ref[p["batch"]] > 0]
            if both:
                sp = [{**p, "speedup": p["rate"] / ref[p["batch"]]} for p in both]
                draw(axes[1], sp, color, marker, ls, None, "speedup")
            draw(axes[2], pts, color, marker, ls, None, "iters")
        ticks = sorted(ticks)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("throughput (systems / s)")
        axes[0].set_title(
            f"{name}   n={n}   one 0.02 Å force step, kT=0\n"
            "cpu 8 threads = 8 OS threads, 1 BLAS thread each"
        )
        axes[1].axhline(1.0, color="black", lw=0.8, alpha=0.5)
        axes[1].set_yscale("log")
        axes[1].set_ylabel("speedup vs cpu 8 threads, warm")
        axes[2].set_ylabel("SCC iterations")
        axes[2].set_xlabel("batch (copies of one perturbed molecule)")
        for ax in axes:
            decimal_log_axis(ax, ticks)
        fig.legend(handles, [h.get_label() for h in handles], loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=True)
        fig.tight_layout()
        fname = OUT / f"throughput_{name}.png"
        fig.savefig(fname, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(fname)


if __name__ == "__main__":
    main()
