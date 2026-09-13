// MS3 (manifest §15.9): degree-sweep on si_sphere R14 (864 atoms — interior
// degrees saturate like R18 but ~half the cost). r_trunc=8 Å; r_k=r_z sweep
// isolates the K/Z support error. Reference = widest that fits the budget.
// A/Bs purifier k vs p (AL1). Degrees come from the init line (deg_*).
//
//   dftb_engine --script scripts/sparse_degree_sweep_big.rhai --sk-dir <matsci>

const sk_dir = SK_DIR;
const xyz = REPO_ROOT + "/debug/nanocrystals/si_sphere_R14.xyz";
const r_trunc = 8.0;

print("=== MS3 degree sweep si_sphere R14 (864 atoms): r_trunc=8, r_k=r_z sweep ===");

let e0 = 0.0;
let first = true;
for rk in [12.0, 10.0, 8.0, 6.0] {
    for mode in ["k", "p"] {
        let name = "k" + itos((rk * 10.0).to_int()) + mode;
        load_xyz(name, xyz);
        let t_init = clock();
        // max_deg=512: declared budget for this measurement sweep — the
        // production contract wants ~128; here we measure the cost of that.
        sparse_new(name, sk_dir, r_trunc, 1.0, rk, rk, 512.0);
        sparse_tc2_tol(name, 1.0e-5);
        sparse_ns_tol(name, 1.0e-4);
        sparse_purifier(name, mode);
        let t0 = clock();
        let rms = sparse_scc_try(name, 60, 1.0e-5);
        let t_scc = clock() - t0;
        if rms.is_nan() {
            print("r_k=" + ftos(rk) + " purifier=" + mode + "  *** SCC DID NOT CONVERGE in 60 iters (mask floor limit-cycles) ***  t_scc=" + ftos(t_scc * 1000.0) + " ms");
        } else {
            let e = sparse_eval(name, true);
            let f = sparse_max_force(name);
            if first { e0 = e; first = false; }
            print("r_k=" + ftos(rk) + " purifier=" + mode +
                  "  E=" + ftos(e) + " Ha  dE=" + ftos((e - e0) * 1000.0) + " mHa" +
                  "  max|F|=" + ftos(f) +
                  "  rms=" + ftos(rms) + "  iters=" + itos(sparse_scc_iters(name)) +
                  "  t_scc=" + ftos(t_scc * 1000.0) + " ms  (init+masks=" + ftos((t0 - t_init) * 1000.0) + " ms)");
        }
    }
}
print("=== done ===");
