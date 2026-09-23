#!/usr/bin/env python3
"""12-step velocity Verlet: sweep cap 3 against cap 40.

Reads debug/dense_multi/jacobi_md.csv.
"""
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "debug" / "dense_multi" / "jacobi_md.csv"
OUT = REPO / "debug" / "dense_multi" / "jacobi_md_cap3.png"

COLORS = {
    "formic": "#e6550d",
    "GC": "#08519c",
    "diazaphen": "#238b45",
    "DTH": "#6a3d9a",
}
ORDER = ["formic", "GC", "diazaphen", "DTH"]
MEV = 27211.386


def load():
    rows = []
    for line in CSV.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        system, cap, step, e, ek, etot, maxf, rms, failed, rmsd = line.split(",")
        rows.append(
            {
                "system": system,
                "cap": int(cap),
                "step": int(step),
                "etot": float(etot),
                "rmsd": float(rmsd),
            }
        )
    return rows


def main():
    rows = load()
    fig, (ax_r, ax_e) = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=140)
    for name in ORDER:
        cap40 = {r["step"]: r["etot"] for r in rows if r["system"] == name and r["cap"] == 40}
        cap3 = [r for r in rows if r["system"] == name and r["cap"] == 3]
        if not cap3 or not cap40:
            continue
        steps = [r["step"] for r in cap3]
        rmsd = [max(r["rmsd"], 1e-8) for r in cap3]
        de = [abs(r["etot"] - cap40[r["step"]]) * MEV for r in cap3]
        de = [max(x, 1e-3) for x in de]
        c = COLORS[name]
        ax_r.plot(steps, rmsd, "o-", color=c, ms=4, label=name)
        ax_e.plot(steps, de, "o-", color=c, ms=4, label=name)
    ax_r.set_yscale("log")
    ax_e.set_yscale("log")
    ax_r.set_ylabel("RMSD vs cap 40 (Å)")
    ax_e.set_ylabel("|ΔEtot| vs cap 40 (meV)")
    for ax in (ax_r, ax_e):
        ax.set_xlabel("Verlet step")
        ax.grid(True, which="both", ls=":", alpha=0.5)
    ax_r.legend(frameon=False, fontsize=8)
    ax_r.set_title("geometry")
    ax_e.set_title("total energy")
    fig.suptitle("BOMD, 12 steps, dt=0.5, mass=1, sweep cap 3. Floor 1e-8 Å / 0.001 meV.", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
