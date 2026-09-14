#!/usr/bin/env python3
"""Stick (line) vibrational spectrum for `sparse_vibrations` output.

    python3 rust_dftb/scripts/plot_vib_spectrum.py \
        debug/sparse_vib_cube_si65_freq.txt \
        --ref /home/prokop/SIMULATIONS/SiNCs/DFTB/L1/cube_Si_matsci-0-3/vibrations.tag \
        --out debug/sparse_vib_cube_si65_spectrum.png

Optional second measured file stacks a third panel. Pure matplotlib,
no intensity data -> all sticks unit height.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AU2CM = 219474.63  # Hartree^-1/2 a0^-1 amu^-1/2 -> cm^-1 (DFTB+ vibrations.tag convention)


def read_sparse_freqs(path):
    """Lines `idx freq` before the `mode k` blocks; returns (freqs, header)."""
    freqs, header = [], ""
    for ln in open(path):
        p = ln.split()
        if ln.startswith("#"):
            header = ln.strip()
        elif len(p) == 2 and p[0].isdigit():
            freqs.append(float(p[1]))
        elif p and p[0] == "mode":
            break
    if not freqs:
        raise ValueError(f"no `idx freq` lines in {path}")
    return np.array(freqs), header


def read_dftb_tag(path):
    """`frequencies :real:1:N` block of a DFTB+ vibrations.tag (atomic units)."""
    lines = open(path).read().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("frequencies"))
    out = []
    for ln in lines[i + 1:]:
        if ":" in ln:
            break
        out += [float(x) * AU2CM for x in ln.split()]
    if not out:
        raise ValueError(f"no frequencies block in {path}")
    return np.array(out)


def sticks(ax, freqs, label, color):
    for f in freqs:
        ax.vlines(f, 0.0, 1.0, color=color, lw=0.9)
    n_imag = int(np.sum(freqs < -1e-9))
    ax.text(0.99, 0.95, f"{label}  ({len(freqs)} modes, {n_imag} imag)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=color)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("freqs", nargs="+", help="sparse_vibrations freq.txt file(s)")
    ap.add_argument("--ref", help="DFTB+ vibrations.tag (atomic units)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    panels = []  # (freqs, label, color)
    if args.ref:
        panels.append((read_dftb_tag(args.ref), "DFTB+ reference", "#888888"))
    for p in args.freqs:
        f, hdr = read_sparse_freqs(p)
        name = p.split("/")[-1].replace("sparse_vib_", "").replace("_freq.txt", "")
        panels.append((f, f"sparse ({name})", "#d62728"))

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.2 * len(panels) + 0.8),
                             sharex=True, squeeze=False)
    for ax, (f, lab, col) in zip(axes[:, 0], panels):
        sticks(ax, f, lab, col)
    xhi = max(f.max() for f, _, _ in panels) * 1.03
    axes[0, 0].set_xlim(min(-30, min(f.min() for f, _, _ in panels) * 1.2), xhi)
    axes[-1, 0].set_xlabel("frequency (cm$^{-1}$)")
    for x, lab in [(100, "Si–Si frame"), (700, "Si–H bend"), (2200, "Si–H stretch")]:
        axes[-1, 0].annotate(lab, (x, 0), textcoords="offset points", xytext=(0, -22),
                             ha="center", fontsize=8, color="#555555")
        axes[-1, 0].vlines(x, -0.06, 0.0, color="#555555", lw=0.8)
    fig.suptitle("Vibrational stick spectrum", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    out = args.out or args.freqs[0].replace("_freq.txt", "_spectrum.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}  ({sum(len(f) for f, _, _ in panels)} sticks, {len(panels)} panels)")


if __name__ == "__main__":
    main()
