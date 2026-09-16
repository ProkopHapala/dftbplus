#!/usr/bin/env python3
"""TC2 convergence + FF32-POLISH endgame plot — log-scale R_I vs iter
and vs cumulative wall time.

Reads the per-iter history CSVs written by RUST_DFTB_TC2_HIST (columns:
iter,branch,tr,dev_rel,guard,r_i,best_r_i,snapshotted,t_ms;
'# call'/'# end' markers delimit purify calls; branch=8 marks an
in-loop float-float McWeeny step). Plots the LAST call in each file.

--fflog <run.log> parses 'FFSTEP,<step>,<ri_f64>,<t_ms>' lines printed
by bench_ff_endgame.rhai (explicit sparse_mcw_ff + sparse_ri_f64 after
purify) and appends them as an f64-verified series — starting at the
last purify iter (x) / purify wall time (t) of the matching run.

Usage:
    python3 rust_dftb/scripts/plot_tc2_convergence.py \
        --out debug/tc2_conv_r10.png \
        --fflog label=debug/bench_ff_endgame.log ... \
        acc4=/tmp/tc2_acc4.csv ...
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_f64_blocks(path):
    """Return {variant: (steps, r_i, r_h, dev)} from the MCW-F64 CSV."""
    blocks, cur, name = {}, [], None
    for line in open(path):
        line = line.strip()
        if line.startswith("# call"):
            cur = []
            name = line.split("variant=")[1].split()[0] if "variant=" in line else "?"
        elif line.startswith("# end"):
            if cur:
                blocks[name] = cur
            cur = []
        elif line and not line.startswith("#"):
            f = line.split(",")
            cur.append((int(f[0]), float(f[1]), float(f[2]), float(f[3])))
    if cur:
        blocks[name] = cur
    return blocks


def load_last_call(path):
    """Return (iters, r_i, dev_rel, branch, t_ms) of the last '# call' block."""
    blocks, cur = [], []
    for line in open(path):
        line = line.strip()
        if line.startswith("# call"):
            cur = []
        elif line.startswith("# end"):
            if cur:
                blocks.append(cur)
            cur = []
        elif line and not line.startswith("#"):
            f = line.split(",")
            t_ms = float(f[8]) if len(f) > 8 else float("nan")
            cur.append((int(f[0]), int(f[1]), float(f[3]), float(f[5]), t_ms))
    if cur:
        blocks.append(cur)
    if not blocks:
        raise SystemExit(f"no history blocks in {path}")
    b = blocks[-1]
    it = [r[0] for r in b]
    ri = [r[3] for r in b if r[3] == r[3]]  # drop NaN (non-check iters)
    dev = [r[2] for r in b]
    br = [r[1] for r in b]
    tms = [r[4] for r in b]
    return it, ri, dev, br, tms, b


def load_ffsteps(path):
    """Parse 'FFSTEP,<step>,<ri_f64>,<t_ms>' lines from a run log."""
    rows = []
    for line in open(path):
        line = line.strip()
        if line.startswith("FFSTEP,"):
            f = line.split(",")
            rows.append((int(f[1]), float(f[2]), float(f[3])))
    return rows


def main():
    out = "debug/tc2_conv_r10.png"
    f64_path = None
    title_tag = ""
    fflogs = {}
    args = sys.argv[1:]
    if args and args[0] == "--out":
        out = args[1]
        args = args[2:]
    if args and args[0] == "--f64":
        f64_path = args[1]
        args = args[2:]
    if args and args[0] == "--title":
        title_tag = args[1]
        args = args[2:]
    cost_tag = "f32 iter ~14.5 ms, FF step ~44 ms"
    if args and args[0] == "--cost":
        cost_tag = args[1]
        args = args[2:]
    while args and args[0] == "--fflog":
        spec = args[1]
        lbl, path = spec.split("=", 1)
        fflogs[lbl] = path
        args = args[2:]
    if not args:
        raise SystemExit("usage: plot_tc2_convergence.py [--out png] [--f64 csv] [--fflog label=log] ... label=csv ...")

    nrows = 3 + (1 if f64_path else 0)
    fig, axs = plt.subplots(nrows, 1, figsize=(9, 3.6 * nrows))
    ax1, ax2, ax3 = axs[0], axs[1], axs[2]

    for spec in args:
        label, path = spec.split("=", 1)
        it, ri, dev, br, tms, raw = load_last_call(path)
        it_ri = [r[0] for r in raw if r[3] == r[3]]
        ri_all = [r[3] for r in raw]
        t_ri = [r[4] for r in raw if r[3] == r[3]]
        (line,) = ax1.semilogy(it_ri, ri, ".-", ms=3, lw=0.8, label=label)
        ax3.semilogy(t_ri, ri, ".-", ms=3, lw=0.8, color=line.get_color(), label=label)
        # mark in-loop FF-McWeeny iters (branch=8)
        ff_it = [r[0] for r in raw if r[1] == 8]
        ff_ri = [r[3] for r in raw if r[1] == 8]
        ff_t = [r[4] for r in raw if r[1] == 8]
        if ff_it:
            ax1.semilogy(ff_it, ff_ri, "D", ms=6, mfc="none", mec=line.get_color(), mew=1.5)
            ax3.semilogy(ff_t, ff_ri, "D", ms=6, mfc="none", mec=line.get_color(), mew=1.5)
        ax2.semilogy(it, [max(d, 1e-12) for d in dev], ".-", ms=3, lw=0.8, label=label)
        n = len(raw)
        t_end = tms[-1] if tms[-1] == tms[-1] else float("nan")
        print(f"{label:14s} iters={n:4d}  min R_I={min(ri):.3e}  t_purify={t_end:.0f} ms")
        # f64-verified explicit FF steps: continue past the purify end
        if label in fflogs:
            steps = load_ffsteps(fflogs[label])
            if steps:
                xs = [it[-1] + s for (s, _, _) in steps]
                ys = [r for (_, r, _) in steps]
                tcum, acc = [], tms[-1] if tms[-1] == tms[-1] else 0.0
                for (_, _, t) in steps:
                    acc += t
                    tcum.append(acc)
                ax1.semilogy(xs, ys, "o--", ms=5, lw=1.0, color=line.get_color(),
                             label=f"{label} +FF (f64-verified)")
                ax3.semilogy(tcum, ys, "o--", ms=5, lw=1.0, color=line.get_color(),
                             label=f"{label} +FF (f64-verified)")
                print(f"{label:14s} FF steps: R_I64 " + "  ".join(f"{r:.2e}" for (_, r, _) in steps)
                      + f"  step~{sum(t for (_,_,t) in steps)/len(steps):.1f} ms")

    # f32-storage reference floor — only meaningful when a curve approaches it
    draw_floor = any(
        min([r for (_, r, _) in load_ffsteps(p)] or [1.0]) < 1e-7
        for p in fflogs.values())
    if draw_floor:
        for ax in (ax1, ax3):
            ax.axhline(2.7e-8, color="gray", ls=":", lw=1.0)
            ax.text(0.01, 0.02, "f32 storage floor ~2.7e-8", transform=ax.transAxes,
                    fontsize=7, color="gray", va="bottom")
    ax1.set_ylabel(r"$R_I = \|KSK-K\|_F/\|K\|_F$")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title(f"TC2 cold purify + FF32-POLISH endgame — R10 (330 Si, deg~330){title_tag}  [◇ = in-loop FF step]")
    ax2.set_ylabel(r"$|\mathrm{Tr}(KS)-N_{occ}|/N_{occ}$")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)
    ax3.set_ylabel(r"$R_I$")
    ax3.set_xlabel("cumulative wall time in purify [ms]")
    ax3.set_title(f"accuracy vs cost — {cost_tag}")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(fontsize=8)

    if f64_path:
        ax4 = axs[3]
        blocks = load_f64_blocks(f64_path)
        for name, rows in blocks.items():
            st = [r[0] for r in rows]
            ri = [max(r[1], 1e-17) for r in rows]
            rh = [r[2] for r in rows]
            ax4.semilogy(st, ri, "o-", ms=5, lw=1.2, label=f"{name} $R_I^{{64}}$")
            ax4.semilogy(st, rh, "s--", ms=4, lw=0.8, alpha=0.6, label=f"{name} $R_H$")
            print(f"{name:14s} R_I64: " + "  ".join(f"{r[1]:.2e}" for r in rows)
                  + f"   R_H={rows[-1][2]:.3e}")
        ax4.set_ylabel("residual (f64)")
        ax4.set_title("F64-MCW decisive test — all-f64 vs f32-storage McWeeny")
        ax4.grid(True, which="both", alpha=0.3)
        ax4.legend(fontsize=8)
        ax4.set_xlabel("f64 McWeeny step")

    ax2.set_xlabel("TC2 iteration")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"REVIEW: {out}")


if __name__ == "__main__":
    main()
