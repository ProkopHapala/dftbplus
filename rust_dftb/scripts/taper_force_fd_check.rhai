// Taper force-consistency check: analytic sparse force vs central-difference
// energy on the SAME tapered engine (r_trunc=7 A, taper_w=1 A). If the chain
// term (w'·u·V) were missing or wrong, F_fd would deviate beyond SCC noise.
//
//   dftb_engine --script scripts/taper_force_fd_check.rhai --sk-dir <matsci>

const sk_dir = SK_DIR;
const xyz = "/home/prokop/SIMULATIONS/SiNCs/DFTB/L1/cube_Si_matsci-0-3/relaxed.xyz";
const DELTA = 0.001;   // Å

print("=== taper E-Fd check (r_trunc=7 A, taper [6,7] A) ===");
load_xyz("t7", xyz);
sparse_new("t7", sk_dir, 7.0, 1.0);
sparse_tc2_tol("t7", 1.0e-5);
sparse_scc("t7", 80, 1.0e-6);
let n = sparse_n_atoms("t7");
let base = get_xyz("t7");

for a in [0, 10, 30, 64] {
    for c in 0..3 {
        let fa = sparse_force_at("t7", a, c);
        let up = base;
        up[3 * a + c] += DELTA;
        sparse_set_coords("t7", up);
        sparse_scc("t7", 80, 1.0e-6);
        let ep = sparse_eval("t7", false);
        let dn = base;
        dn[3 * a + c] -= DELTA;
        sparse_set_coords("t7", dn);
        sparse_scc("t7", 80, 1.0e-6);
        let em = sparse_eval("t7", false);
        let fd = -(ep - em) / (2.0 * DELTA);
        print("atom " + itos(a) + " c=" + itos(c) +
              "  F_an=" + ftos(fa) + "  F_fd=" + ftos(fd) +
              "  dF=" + ftos((fa - fd) * 1000.0) + " mHa/A");
        // restore base state (warm)
        sparse_set_coords("t7", base);
        sparse_scc("t7", 80, 1.0e-6);
    }
}
print("=== done ===");
