// FF32-POLISH production-policy benchmark (manifest §4.12.2 consolidated):
// Phase A = f32 TC2 hard budget (caller's max_iter, production = 30),
// Phase B = terminal ff McWeeny (≤5 steps, per-step residual early exit).
// NO host-f64 diagnostics in the timed path — ri_f64 is only called
// between timed regions for the plot.
//
// Env (all optional):
//   BENCH_RK=20|40     — r_k/r_z Å; 20 = production mask (tail ~5e-6),
//                        40 = whole-cluster (arithmetic floor only)
//   BENCH_TOL=1e-9     — purify tol; below floor → floor-stop / Phase B
//   BENCH_MAXITER=30   — Phase-A budget (production default)
//   BENCH_FFSTEP=0|1   — also run the per-step / bulk FF cost loop
//   RUST_DFTB_TC2_FF=1            — enable Phase B (hiacc)
//   RUST_DFTB_TC2_FF_SWITCH=1e-3  — FF entry residual gate
//   RUST_DFTB_TC2_FF_STEPS=5      — Phase-B step cap
//   RUST_DFTB_TC2_FF_VTQ=1        — 3-product V=T·Q path
//   RUST_DFTB_FF_LO_GLOBAL=1      — A_lo from global (occupancy A/B)
//   RUST_DFTB_FF_ACC2=1           — 2-way ff accumulators
//   RUST_DFTB_PROF=evt            — marker-event device ms per stage
//   RUST_DFTB_TC2_HIST=<csv>      — per-iter history (branch=8 → FF iter)
//
//   timeout 120 env RUST_DFTB_TC2_FF=1 BENCH_RK=40 \
//     RUST_DFTB_TC2_HIST=debug/tc2_ff_rk40.csv \
//     dftb_engine --script rust_dftb/scripts/bench_ff_endgame.rhai \
//     --sk-dir $RUST_DFTB_SK_DIR

const xyz = REPO_ROOT + "/debug/si_r10_sparse_relaxed_rk20.xyz";

let rk     = if env("BENCH_RK")      == "" { 20.0   } else { env("BENCH_RK").parse_float() };
let tol    = if env("BENCH_TOL")     == "" { 1.0e-9 } else { env("BENCH_TOL").parse_float() };
let maxit  = if env("BENCH_MAXITER") == "" { 30     } else { env("BENCH_MAXITER").parse_int() };
let ffstep = env("BENCH_FFSTEP") == "1";

print("=== purification benchmark: si_sphere_R10 (330 Si, deg~330) ===");
print("BENCH cfg: r_k=" + ftos(rk) + " tol=" + ftos(tol) + " max_iter=" + itos(maxit)
    + " TC2_FF=" + env("RUST_DFTB_TC2_FF")
    + " FF_LO_GLOBAL=" + env("RUST_DFTB_FF_LO_GLOBAL")
    + " FF_ACC2=" + env("RUST_DFTB_FF_ACC2")
    + " FF_VTQ=" + env("RUST_DFTB_TC2_FF_VTQ")
    + " SPGEMM_ACC16=" + env("RUST_DFTB_SPGEMM_ACC16"));

load_xyz("r", xyz);
sparse_new("r", SK_DIR, 5.45, 0.3, rk, rk, 1024.0);
print("n_atoms=" + itos(sparse_n_atoms("r")));

// Few SCC mix iters → realistic (near-converged) H_scc regime.
sparse_scc_try("r", 4, 1.0e-6);

// One controlled COLD purify under the production policy.
let t0 = clock();
let it = sparse_purify_now("r", maxit, tol);
let t_pur = clock() - t0;
print("BENCH purify_now iters=" + itos(it) + "  t=" + ftos(t_pur) + " s  (" + ftos(1000.0 * t_pur / it) + " ms/iter)");

// True residual of the stored K (f64 host diagnostic — untimed region).
let r64 = sparse_ri_f64("r");
print("BENCH ri_f64 after purify: R_I=" + ftos(r64));

if ffstep {
    // Explicit FF polish steps — per-step wall time + f64-verified residual.
    for i in 1..7 {
        let ts = clock();
        sparse_mcw_ff("r", 1);
        sparse_sync("r");          // honest step cost: enqueue + device drain
        let t_step = clock() - ts;
        let r = sparse_ri_f64("r");
        print("FFSTEP," + itos(i) + "," + ftos(r) + "," + ftos(1000.0 * t_step));
    }
    // Steady-state throughput: 10 back-to-back FF steps + one drain.
    let tb = clock();
    sparse_mcw_ff("r", 10);
    sparse_sync("r");
    let t_bulk = clock() - tb;
    let r10 = sparse_ri_f64("r");
    print("BENCH ff_bulk 10 steps t=" + ftos(t_bulk) + " s  (" + ftos(100.0 * t_bulk) + " ms/step)  R_I=" + ftos(r10));
}
print("=== done ===");
