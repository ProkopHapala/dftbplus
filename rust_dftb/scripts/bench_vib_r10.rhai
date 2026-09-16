// Bounded FD-Hessian column benchmark on the R10 sphere (330 Si).
// Measures per-column cost + TC2-iter counts for the three displaced-
// solve modes; RUST_DFTB_VIB_MAXCOL bounds columns (no eigensolve).
//   env RUST_DFTB_VIB_MAXCOL=2 RUST_DFTB_VIB_FIXQ=1 RUST_DFTB_VIB_DMUPD=1 \
//     dftb_engine --script rust_dftb/scripts/bench_vib_r10.rhai --sk-dir $SK
// Modes: default = full SCC per column (warm q); FIXQ=1 = frozen-charge
// single purify (+DMUPD=1 = δK0 seed + McWeeny); FROZEN=1 = forces only.

const xyz = REPO_ROOT + "/debug/si_r10_sparse_relaxed_rk20.xyz";
let h    = if env("BENCH_H")  == "" { 0.05  } else { env("BENCH_H").parse_float() };
let tols = if env("BENCH_ST") == "" { 1e-5  } else { env("BENCH_ST").parse_float() };

print("=== vib column bench: R10 (330 Si) h=" + ftos(h) + " Å ===");
load_xyz("r", xyz);
sparse_new("r", SK_DIR, 5.45, 0.3, 20.0, 20.0, 1024.0);
print("n_atoms=" + itos(sparse_n_atoms("r")));

// Converge the center first (needs the DIIS loop). sparse_scc_try does
// not panic on non-convergence — the columns still measure fine.
let t0 = clock();
sparse_scc_try("r", 60, tols);
print("center scc t=" + ftos(clock() - t0) + " s iters=" + itos(sparse_scc_iters("r")));

let tv = clock();
let s = sparse_vibrations("r", h, tols, REPO_ROOT + "/debug/bench_vib_r10_freq.txt");
print("VIB: " + s + "  wall=" + ftos(clock() - tv) + " s");
print("=== done ===");
