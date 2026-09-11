// Parity isolation for cube_Si (65 atoms): sparse charges + energy at the
// DFTB+ minimum vs Fortran detailed.out (net charges Si +0.03..0.05, H -0.029;
// E_el=-55.2130 Ha, E_rep=+1.7609 Ha, E_tot=-53.4520 Ha).
const sk_dir = SK_DIR;
const ref_dir = "/home/prokop/SIMULATIONS/SiNCs/DFTB/L1/cube_Si_matsci-0-3";

print("=== parity: sparse vs Fortran at DFTB+ min ===");
load_xyz("cube", ref_dir + "/relaxed.xyz");
sparse_new("cube", sk_dir);
sparse_tc2_tol("cube", 1.0e-6);
let rms = sparse_scc("cube", 100, 1.0e-5);
let e = sparse_eval("cube", false);
print("E_tot=" + ftos(e) + "  dE=" + ftos(e + 53.4520493137) + "  rms=" + ftos(rms));
print("charges: " + sparse_charges("cube"));
print("=== done ===");
