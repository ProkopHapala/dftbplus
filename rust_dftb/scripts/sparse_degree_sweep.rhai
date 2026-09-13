// MS3 (manifest §15.9): degree-sweep measurement matrix on cube_Si65.
// r_trunc=8 Å fixed (H/S taper, validated <0.05 mHa); r_k=r_z varied →
// measures wall time + SCC iters + E-parity vs the widest reference as a
// function of K/Z support. ALSO A/Bs the P=KS purifier (AL1) at each radius.
//
//   dftb_engine --script scripts/sparse_degree_sweep.rhai --sk-dir <matsci>
//
// Reference = r_k=0 (full SK radius). Degrees are printed by the init line
// (deg_hs/deg_k/deg_z) — the budget arg is 512 to keep the sweep honest
// (degrees are what they are; we measure the cost of making them small).

const sk_dir = SK_DIR;
const xyz = "/home/prokop/SIMULATIONS/SiNCs/DFTB/L1/cube_Si_matsci-0-3/relaxed.xyz";
const r_trunc = 8.0;

print("=== MS3 degree sweep cube_Si65: r_trunc=8, r_k=r_z sweep ===");

let e0 = 0.0;
let f0 = 0.0;
for rk in [0.0, 12.0, 10.0, 9.0, 8.0, 7.0, 6.0] {
    for mode in ["k", "p"] {
        let name = "k" + itos((rk * 10.0).to_int()) + mode;
        load_xyz(name, xyz);
        let t_init = clock();
        sparse_new(name, sk_dir, r_trunc, 1.0, rk, rk, 512.0);
        sparse_tc2_tol(name, 1.0e-6);
        sparse_ns_tol(name, 1.0e-4);
        sparse_purifier(name, mode);
        let t0 = clock();
        let rms = sparse_scc(name, 80, 1.0e-5);
        let e = sparse_eval(name, true);
        let t_scc = clock() - t0;
        let f = sparse_max_force(name);
        if rk == 0.0 && mode == "k" { e0 = e; f0 = f; }
        print("r_k=" + ftos(rk) + " purifier=" + mode +
              "  E=" + ftos(e) + " Ha  dE=" + ftos((e - e0) * 1000.0) + " mHa" +
              "  max|F|=" + ftos(f) + "  dF=" + ftos((f - f0) * 1000.0) + " mHa/bohr" +
              "  rms=" + ftos(rms) + "  iters=" + itos(sparse_scc_iters(name)) +
              "  t_scc=" + ftos(t_scc * 1000.0) + " ms  (init+masks=" + ftos((t0 - t_init) * 1000.0) + " ms)");
    }
}
print("=== done ===");
