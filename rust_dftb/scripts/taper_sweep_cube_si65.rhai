// H/S truncation-error sweep: E(r_trunc) at the SAME geometry (DFTB+ s+p
// min). Untruncated E = -52.4705762933 Ha (Fortran reference). The taper is
// cosine over [r_trunc-1A, r_trunc]; mask = r_trunc + 1A skin.
//
//   dftb_engine --script scripts/taper_sweep_cube_si65.rhai --sk-dir <matsci-0-3>

const sk_dir = SK_DIR;
const xyz = "/home/prokop/SIMULATIONS/SiNCs/DFTB/L1/cube_Si_matsci-0-3/relaxed.xyz";

print("=== taper sweep cube_Si65 ===");
let e0 = 0.0;
for r_trunc in [0.0, 9.0, 8.0, 7.0, 6.0, 5.0] {
    let name = "t" + itos((r_trunc * 10.0).to_int());
    load_xyz(name, xyz);
    sparse_new(name, sk_dir, r_trunc, 1.0);
    sparse_tc2_tol(name, 1.0e-5);
    let rms = sparse_scc(name, 80, 1.0e-6);
    let e = sparse_eval(name, true);
    if r_trunc == 0.0 { e0 = e; }
    print("r_trunc=" + ftos(r_trunc) + "  E=" + ftos(e) +
          "  dE_vs_untrunc=" + ftos((e - e0) * 1000.0) + " mHa" +
          "  max|F|=" + ftos(sparse_max_force(name)) + "  iters=" + itos(sparse_scc_iters(name)));
}
print("=== done ===");
