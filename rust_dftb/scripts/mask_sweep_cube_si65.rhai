// Mask-radius sweep isolating the K/Z support-truncation error on cube_Si65:
// r_trunc=8 A fixed (H/S taper), r_k/r_z varied. Reference = r_k=0 (full).
//
//   dftb_engine --script scripts/mask_sweep_cube_si65.rhai --sk-dir <matsci>

const sk_dir = SK_DIR;
const xyz = "/home/prokop/SIMULATIONS/SiNCs/DFTB/L1/cube_Si_matsci-0-3/relaxed.xyz";

print("=== K/Z mask sweep cube_Si65 (r_trunc=8 fixed) ===");
let e0 = 0.0;
for rk in [0.0, 12.0, 10.0, 8.0, 6.0] {
    let name = "k" + itos((rk * 10.0).to_int());
    load_xyz(name, xyz);
    sparse_new(name, sk_dir, 8.0, 1.0, rk, rk);
    sparse_tc2_tol(name, 1.0e-5);
    sparse_ns_tol(name, 1.0e-4);   // truncated Z can't beat the mask floor
    let rms = sparse_scc(name, 80, 1.0e-6);
    let e = sparse_eval(name, true);
    if rk == 0.0 { e0 = e; }
    print("r_k=r_z=" + ftos(rk) + "  E=" + ftos(e) +
          "  dE=" + ftos((e - e0) * 1000.0) + " mHa" +
          "  max|F|=" + ftos(sparse_max_force(name)) + "  iters=" + itos(sparse_scc_iters(name)));
}
print("=== done ===");
