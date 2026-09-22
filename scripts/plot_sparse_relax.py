#!/usr/bin/env python3
"""Energy and max |F| for a sparse FIRE history.csv."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyBall.plotUtils import plotEF


def load_rows(src: Path):
    rows = []
    cold = []
    with src.open() as f:
        header = f.readline().strip().split(",")
        i_e = header.index("E_Ha")
        i_f = header.index("maxAbsF")
        i_note = header.index("note") if "note" in header else None
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.strip().split(",")
            if p[0] == "step" or not p[i_e]:
                continue
            rows.append((float(p[i_e]), float(p[i_f])))
            if i_note is not None and len(p) > i_note and p[i_note].strip() == "cold":
                cold.append(len(rows) - 1)
    return rows, cold


def main():
    argv = sys.argv[1:]
    if len(argv) >= 2 and not argv[-1].endswith(".csv"):
        dst = Path(argv[-1])
        srcs = [Path(a) for a in argv[:-1]]
    else:
        srcs = [Path(argv[0])] if argv else [Path("debug/sparse_relax/history.csv")]
        dst = srcs[0].with_name("relax_EF.png")
    rows = []
    cold = []
    for src in srcs:
        part, c = load_rows(src)
        cold.extend(i + len(rows) for i in c)
        rows.extend(part)
    if not rows:
        raise SystemExit(f"no rows in {srcs}")
    ef = np.asarray(rows, dtype=float)
    x = np.arange(len(ef), dtype=float)
    plt.figure(figsize=(7.4, 6.2))
    plotEF(x, ef, label=srcs[0].parent.name if srcs[0].parent else "FIRE")
    ax_e, ax_f = plt.gcf().axes
    ax_e.set_ylabel("energy (Ha)")
    ax_e.set_xlabel("FIRE step")
    ax_f.set_ylabel("max |F| (Ha/Å)")
    ax_f.set_xlabel("FIRE step")
    ax_f.set_yscale("log")
    ax_e.set_title(f"{len(rows)} steps")
    if len(ef) > 10:
        tail = ef[-40:, 1]
        floor = float(np.min(tail))
        ax_f.axhline(floor, color="0.35", lw=0.8, ls=":")
        ax_f.text(0.02, floor, f"  late min {floor:.1e}", transform=ax_f.get_yaxis_transform(), va="bottom", fontsize=8)
    for s in cold:
        ax_e.axvline(s, color="0.6", lw=0.6, ls="--")
        ax_f.axvline(s, color="0.6", lw=0.6, ls="--")
    dst.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(dst, dpi=140, bbox_inches="tight")
    print(f"wrote {dst}  steps={len(ef)}  E {ef[0,0]:.6f} → {ef[-1,0]:.6f}  max|F| {ef[-1,1]:.3e}")


if __name__ == "__main__":
    main()
