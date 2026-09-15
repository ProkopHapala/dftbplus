// TC2 convergence study (GPT-5.6 plan): ONE cold purify on a realistic
// converged-regime H_scc of the R10 silicon sphere (330 atoms, deg~330).
// Variants are selected by env at process start (kernel compile defines):
//   RUST_DFTB_SPGEMM_ACC8=1    — 8 f32 accumulators (halved chains)
//   RUST_DFTB_SPGEMM_ACC16=1   — 16 f32 accumulators (quartered chains)
//   RUST_DFTB_TC2_STOP_W=n     — windowed best-R_I floor stop
// Per-iter (R_I, Tr, branch) → CSV via RUST_DFTB_TC2_HIST=<path>.
//   timeout 30 env RUST_DFTB_TC2_HIST=/tmp/tc2_acc4.csv \
//     dftb_engine --script rust_dftb/scripts/tc2_conv_study.rhai \
//     --sk-dir $RUST_DFTB_SK_DIR

const xyz = REPO_ROOT + "/debug/si_r10_sparse_relaxed_rk20.xyz";

print("=== TC2 convergence study: si_sphere_R10 (330 Si, pbc-0-3, deg~330) ===");

load_xyz("r", xyz);
const RK = 40.0;   // r_k/r_z Å — 20 truncates K's true tail (~5e-6); 40 covers the cluster
sparse_new("r", SK_DIR, 5.45, 0.3, RK, RK, 1024.0);
print("n_atoms=" + itos(sparse_n_atoms("r")));

// Few SCC mix iters → realistic (near-converged) H_scc regime.
sparse_scc_try("r", 4, 1.0e-6);

// One controlled COLD purify on that H_scc (K0 rebuilt inside).
// tol=1e-9 is below the f32 floor on purpose: run until the stopping
// identifier cuts — that is exactly what the study measures.
let t0 = clock();
let it = sparse_purify_now("r", 150, 1.0e-9);
print("purify_now iters=" + itos(it) + "  t=" + ftos(clock() - t0) + " s");
// F64-DIAG + DUMMY-DECOMP: true residual of the stored K (f64 dense,
// physical vs dummy lanes) vs the f32-GPU measured R_I.
let t1 = clock();
let r64 = sparse_ri_f64("r");
print("ri_f64 t=" + ftos(clock() - t1) + " s");

// F64-MCW decisive test + A/B storage split (manifest §4.12.2 round 2):
// variant B (all-f64 state): quadratic R_I collapse ⇒ f32 precision is
// the cold-purifier floor. variant A (f32 storage, f64 products):
// stalls ~1e-5 while B collapses ⇒ f32 STORAGE is the limiter.
// Reports R_H alongside R_I per step; CSV → RUST_DFTB_MCW_F64_HIST.
let t2 = clock();
sparse_mcw_f64("r", 4, false);
print("mcw_f64 B t=" + ftos(clock() - t2) + " s");
let t3 = clock();
sparse_mcw_f64("r", 4, true);
print("mcw_f64 A t=" + ftos(clock() - t3) + " s");

// FF32-POLISH (manifest §4.12.2 r2): float-float McWeeny on the stored
// f32 K — expect R_I_f64 ~1e-5 → ~1e-8 after ONE step. Time per step.
sparse_ff_test("r");   // unit check: err(hi) vs err(hi+lo) vs f64 dense
let t4 = clock();
sparse_mcw_ff("r", 1);
print("mcw_ff step1 t=" + ftos(clock() - t4) + " s");
let r64_ff1 = sparse_ri_f64("r");
let t5 = clock();
sparse_mcw_ff("r", 1);
print("mcw_ff step2 t=" + ftos(clock() - t5) + " s");
let r64_ff2 = sparse_ri_f64("r");
print("=== done ===");
