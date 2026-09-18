// Frozen-density (clamped-electron) FD Hessian + spectrum on the passivated
// Si sphere R18 (1648 atoms: 1202 Si + 446 H), matsci-0-3 SK set.
//
// Tier-A frozen mode (manifest §4.12.1 / report §15.17): the converged
// (K0, W0, q0) is snapshotted once at the reference geometry (device-
// resident, §15.25), then every ±h column costs ZERO device products —
// GPU pair contraction vs the resident K0/W0 only. Multi-replica batch
// (report §15.27): RUST_DFTB_VIB_BATCH=B runs B displaced geometries per
// launch wave on a gid(1) replica axis — bitwise-identical to sequential.
// CAVEAT: missing charge/orbital response softens Si–H stretches
// (~240 cm-1 on si10h16 vs full SCC) — framework modes are fine; treat the
// top of the spectrum as preview quality. Quantitative mode = VIB_FIXQ
// (cold purify per eval) or the DMM-lite ladder.
//
//   bounded measurement (agent policy §4.12.0, timeout 30):
//     RUST_DFTB_TRUNC_PRODUCTS=1 RUST_DFTB_SCC_RHGATE=1e-2 \
//     RUST_DFTB_VIB_FROZEN=1 RUST_DFTB_VIB_MAXCOL=8 timeout 30 \
//       dftb_engine --script scripts/sparse_vibrations_si_r18_frozen.rhai \
//       --sk-dir $RUST_DFTB_SK_DIR
//   full Hessian (user-launched artifact only): same, no MAXCOL
//     (4944 columns x 2 evals; time extrapolated by the bounded run)
//   relax to the model's own minimum first (needed for a real spectrum):
//     R18_RELAX=1 ... same command  (FIRE -> debug/si_r18_sparse_relaxed_rk12.xyz)
//
// REQUIRED env on this system (1648 atoms, ~37 Å sphere):
//   RUST_DFTB_TRUNC_PRODUCTS=1 — M_TKS = supp(K·S) spans r_k+r_hs≈21 Å →
//     interior rows are DENSE (deg→1648) and exceed the kernel's
//     MAX_LEFT_BLOCKS local-mem row cache. Truncating T to M_K restores
//     the pre-FF32-fix layout; the dropped tail (~7e-6/product) is far
//     below this system's ~5e-4 mask floor. The real fix is a left-row
//     chunking kernel (named in sparse_system.rs as future work).
//   RUST_DFTB_SCC_RHGATE=1e-2 — the R_H stationarity gate (default 5e-4)
//     is unreachable at 1648 atoms: K sits at its mask-truncation floor
//     (R_I(K)~2e-3 → R_H~4e-3). 1e-2 still rejects the wrong-subspace
//     signature (~1.4e-2 measured) the gate exists for.
//   RUST_DFTB_VIB_RHGATE=1e-2 — same for the per-column R_H gate inside
//     scc_fixedq (VIB_FIXQ mode only; frozen mode has no per-column solve).
//
// Tiers measured on this machine (RTX 3090, 1648 atoms, h=0.02):
//   frozen  ~1.6 ms/eval (B=1) → 0.27 ms/eval at VIB_BATCH=16 — full
//           Hessian evals ~2.7 s; total wall ~139 s dominated by the
//           dense 4944^2 host eigensolve (~100 s).  (VIB_FROZEN=1)
//           QUALITATIVE ONLY — si10h16 parity vs scc: rms 157 cm-1,
//           Si-H ~250 cm-1 soft, rigid modes land at +120 cm-1.
//   fixq    ~0.92 s/eval → full Hessian ~2.5 h    (VIB_FIXQ=1)
//           quantitative — si10h16 parity vs scc: rms 5.9 cm-1,
//           Si-H ~8 cm-1 soft, rigid modes intact. RECOMMENDED.
//   full scc ~est. 5-8 s/eval → overnight         (no env)
// Frozen evals are fully GPU-resident (§15.25): central K0/W0/dq live on
// device; per eval only xyzu up + forces down (O(N) transfers).
//
// env knobs: VIB_H (FD step, default 0.02 A), BENCH_ST (scc tol, 1e-5),
//   RUST_DFTB_VIB_BATCH (frozen evals per launch wave, default 1; ~16
//   saturates at R18 — §15.27),
//   R18_XYZ (override input geometry), R18_RELAX=1 (run FIRE + save),
//   R18_FMAX (relax force tol, 1e-4), R18_STEPS (relax max steps, 400),
//   R18_RTRUNC / R18_TAPER / R18_RK / R18_RZ (mask radii; defaults
//   8.0/1.0/12.0/12.0 — for pbc-0-3 use R18_RTRUNC=5.5, its native range).

const xyz_raw = REPO_ROOT + "/debug/nanocrystals/si_sphere_R18.xyz";
const xyz_rel = REPO_ROOT + "/debug/si_r18_sparse_relaxed_rk12.xyz";
let h      = if env("VIB_H")     == "" { 0.02 } else { env("VIB_H").parse_float() };
let tols   = if env("BENCH_ST")  == "" { 1e-5 } else { env("BENCH_ST").parse_float() };
let relax  = env("R18_RELAX") == "1";
let f_tol  = if env("R18_FMAX")  == "" { 1e-4 } else { env("R18_FMAX").parse_float() };
let nsteps = if env("R18_STEPS") == "" { 400  } else { env("R18_STEPS").parse_int() };
let xyz_in = if env("R18_XYZ") != "" { env("R18_XYZ") }
             else if relax { xyz_raw } else { xyz_rel };

print("=== SparseDftb FROZEN vibrations: si_sphere_R18 (1648 atoms, matsci) ===");
print("input: " + xyz_in + "  h=" + ftos(h) + "  scc_tol=" + ftos(tols));
load_xyz("r", xyz_in);
// r_trunc=8 A, taper 1 A: <0.05 mHa measured on cube_si65. r_k=r_z=12 A:
// K mask floor R_I~3.3e-4 on this system (vs 7.7e-4 @ 10 A, TRS4). The wider
// K/Z masks cost nothing per frozen eval (no device products) — only the
// central SCC + relax see them. max_deg=384 covers deg~250 @ r_k=12.
let rtrunc = if env("R18_RTRUNC") == "" { 8.0  } else { env("R18_RTRUNC").parse_float() };
let taper  = if env("R18_TAPER")  == "" { 1.0  } else { env("R18_TAPER").parse_float() };
let rk     = if env("R18_RK")     == "" { 12.0 } else { env("R18_RK").parse_float() };
let rz     = if env("R18_RZ")     == "" { 12.0 } else { env("R18_RZ").parse_float() };
sparse_new("r", SK_DIR, rtrunc, taper, rk, rz, 384.0);
sparse_purifier("r", "trs");          // best measured R_I floor at this size
sparse_tc2_tol("r", 1.0e-3);          // floor-stop: mask floor R_I~5e-4 @r_k=12;
                                      // 1e-6 churns 30 iters on a plateau it can't cross.
                                      // Recovered K measures R_I~2e-3 regardless (P·Z recovery).
sparse_ns_tol("r", 1.0e-4);
print("n_atoms=" + itos(sparse_n_atoms("r")) + " n_orbs(padded)=" + itos(sparse_n_orbs("r")));

if relax {
    let t0 = clock();
    sparse_relax("r", nsteps, f_tol, tols);
    print("relax t=" + ftos(clock() - t0) + " s  max|F|=" + ftos(sparse_max_force("r")));
    save_xyz("r", xyz_rel);
    print("saved relaxed: " + xyz_rel);
}
// Converged central state (the frozen snapshot is taken inside
// sparse_vibrations). Must be the model's own minimum for a spectrum.
sparse_scc("r", 100, tols);
let e = sparse_eval("r", true);
print("center: E=" + ftos(e) + " Ha  max|F|=" + ftos(sparse_max_force("r")) +
      "  Tr(KS)=" + ftos(sparse_tr_ks("r")) + "  scc_iters=" + itos(sparse_scc_iters("r")));

let tv = clock();
let s = sparse_vibrations("r", h, tols, REPO_ROOT + "/debug/sparse_vib_si_r18_freq.txt");
print("VIB: " + s + "  wall=" + ftos(clock() - tv) + " s");
print("=== done ===");
