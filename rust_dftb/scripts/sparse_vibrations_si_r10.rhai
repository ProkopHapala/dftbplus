// End-to-end sparse vibrations on the R10 silicon sphere (330 atoms —
// the realistic scale where a sparse solver matters). Pure Si, pbc-0-3.
//   cargo run --release -p rust_dftb --bin dftb_engine -- \
//     --script rust_dftb/scripts/sparse_vibrations_si_r10.rhai \
//     --sk-dir $RUST_DFTB_SK_DIR
//
// r_k=20 -> M_K ~complete (deg~330, ~92% of pairs). Counterintuitively
// FASTER per eval than r_k=12/deg175 on this system: deg175's purify
// never converges (mask floor R_I~6e-4) and churns ~55 iters x ~17 DIIS
// iters ~ 4.5s/eval, while deg330 purifies to 3e-5 and DIIS settles in
// ~3 iters ~ 0.5-1s/eval. Fewest-neighbors wins asymptotically only when
// the narrow mask still has a stable fixed point (report §15.12-2').
// Relaxes from the deg175-relaxed geometry (short re-relax at deg330).

const xyz = REPO_ROOT + "/debug/si_r10_sparse_relaxed.xyz";
const scc_tol_hess = 1.0e-5;
const fd_h = 0.02;

print("=== SparseDftb vibrations: si_sphere_R10 (330 Si, pbc-0-3, deg~330) ===");

load_xyz("r", xyz);
sparse_new("r", SK_DIR, 5.45, 0.3, 20.0, 20.0, 1024.0);
sparse_tc2_tol("r", 1.0e-5);
sparse_ns_tol("r", 1.0e-4);
print("n_atoms=" + itos(sparse_n_atoms("r")));

let t0 = clock();
sparse_relax("r", 60, 1.0e-3, 1.0e-5);
print("relax t=" + ftos(clock() - t0) + " s");
sparse_scc("r", 80, scc_tol_hess);
let e = sparse_eval("r", true);
print("relaxed: E=" + ftos(e) + " Ha  max|F|=" + ftos(sparse_max_force("r")));
save_xyz("r", REPO_ROOT + "/debug/si_r10_sparse_relaxed_rk20.xyz");

let t1 = clock();
let summary = sparse_vibrations("r", fd_h, scc_tol_hess, REPO_ROOT + "/debug/sparse_vib_si_r10_freq.txt");
print("hessian t=" + ftos(clock() - t1) + " s");
print("vibrations: " + summary);
print("=== done ===");
