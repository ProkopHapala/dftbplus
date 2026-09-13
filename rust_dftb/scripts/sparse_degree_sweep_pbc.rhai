// L1 (manifest §15.10): the decisive cheap test — SAME R14 degree sweep but
// with pbc-0-3 parameters instead of matsci. pbc Si has stronger wavefunction
// confinement (r0=3.3 vs 4.2 a0) and the Si-Si/Si-H tables END at ~5.5 Å
// (10.4 a0) — half the matsci range. If the P/K block decay follows the
// confinement, low-degree masks could be accurate here.
//
// r_trunc=5.3 Å sits just inside the 5.5 Å pbc table end; taper 0.4.
// Radii sweep {0=full table, 5.0, 4.5, 4.0, 3.5, 3.0}; reference = full.
// Expect interior deg ~ (4/3)pi*r^3*rho ~ {40, 32, 24, 17, 11, 7}.
//
//   dftb_engine --script scripts/sparse_degree_sweep_pbc.rhai --sk-dir <pbc-0-3>

const sk_dir = SK_DIR;
const xyz = REPO_ROOT + "/debug/nanocrystals/si_sphere_R14.xyz";
const r_trunc = 5.3;

print("=== L1 pbc degree sweep si_sphere R14 (864 atoms): r_trunc=5.3, r_k=r_z sweep ===");

let e0 = 0.0;
let first = true;
for rk in [0.0, 5.0, 4.5, 4.0, 3.5, 3.0] {
    for mode in ["k", "p"] {
        let name = "k" + itos((rk * 10.0).to_int()) + mode;
        load_xyz(name, xyz);
        let t_init = clock();
        sparse_new(name, sk_dir, r_trunc, 0.4, rk, rk, 512.0);
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
