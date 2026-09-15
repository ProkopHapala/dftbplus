// Per-phase timing of one FD Hessian eval (set_coords / scc / forces)
// on si_sphere_R10. Run at deg175 (r_k=12) and deg330 (r_k=20):
//   RUST_DFTB_PROF=mark dftb_engine --script bench_eval_r10.rhai --sk-dir pbc-0-3
// RK env var switches the mask radius.

const xyz = REPO_ROOT + "/debug/nanocrystals/si_sphere_R10.xyz";
const rk = 20.0;      // 12 = deg~175, 20 = deg~330

load_xyz("r", xyz);
sparse_new("r", SK_DIR, 5.45, 0.3, rk, rk, 1024.0);
sparse_tc2_tol("r", 1.0e-5);
sparse_ns_tol("r", 1.0e-4);

let t = clock();
sparse_scc("r", 80, 1.0e-5);
print("cold scc  t=" + ftos(clock() - t) + " s");
sparse_eval("r", true);
print("relaxed-level E=" + ftos(sparse_eval("r", false)));

let x = get_xyz("r");
let d = 0.02;
// 4 displaced evals on atom 0 x, alternating +/- so the last is -h from start.
for i in 0..4 {
    x[0] += d;
    let t0 = clock();
    sparse_set_coords("r", x);
    let t1 = clock();
    sparse_scc("r", 80, 1.0e-5);
    let t2 = clock();
    sparse_eval("r", true);
    let t3 = clock();
    print("eval " + itos(i) + ":  set_coords=" + ftos(t1 - t0) +
          "  scc=" + ftos(t2 - t1) + " (iters=" + itos(sparse_scc_iters("r")) +
          ", r_I=" + ftos(sparse_tr_ks("r")) + ")" +
          "  forces=" + ftos(t3 - t2) + " s");
    d = -d;
}
print("=== done ===");
