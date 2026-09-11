// End-to-end sparse DFTB vibrational spectrum of a real passivated Si
// nanocrystal: Si10H16 (10 Si + 16 H, data/xyz/Si10_H.xyz, matsci-0-3).
//
//   cargo run --release -p rust_dftb --bin dftb_engine -- \
//     --script rust_dftb/scripts/sparse_vibrations_si10h16.rhai \
//     --sk-dir $RUST_DFTB_SK_DIR
//
// Pipeline: load xyz → sparse_new → sparse_relax (FIRE) → FD Hessian of the
// analytic sparse forces → mass-weighted eigen → frequencies (cm^-1).
// Every SCC at a displaced geometry warm-starts from the previous charges.
//
// NOTE (labbook 2026-09-12): linear mixing diverges on Si10H16 (charge
// sloshing, rms~1.4) — DIIS is default-on (diis_hist=8). The SCC noise
// floor is set by tc2_tol: r_I~2e-5 at the 1e-4 default puts the charge
// floor ~3e-6 > scc_tol → tighten tc2_tol to 1e-7 for Hessian work.

const sk_dir = SK_DIR;
const xyz_dir = REPO_ROOT + "/data/xyz";
const scc_max = 100;
const scc_tol_relax = 1.0e-5;
const scc_tol_hess = 1.0e-6;
const fd_h = 0.02;           // Å — Gate-E plateau step

print("=== SparseDftb end-to-end vibrations: Si10H16 ===");
print("SK_DIR=" + sk_dir);

load_xyz("si10", xyz_dir + "/Si10_H.xyz");
let n_orbs = sparse_new("si10", sk_dir);
sparse_tc2_tol("si10", 1.0e-7);
print("n_atoms=" + itos(sparse_n_atoms("si10")) + " n_orbs(padded)=" + itos(n_orbs));

// Relax tightly before the Hessian — residual force noise (~1e-3) puts the
// 6 rigid modes at ±15 cm-1; f_tol=1e-4 should collapse them toward 0.
let e0 = sparse_relax("si10", 600, 1.0e-4, scc_tol_relax);
let rms = sparse_scc("si10", scc_max, scc_tol_hess);
let e = sparse_eval("si10", true);
print("relaxed: E=" + ftos(e) + " Ha  max|F|=" + ftos(sparse_max_force("si10")) + " Ha/Ang  Tr(KS)=" + ftos(sparse_tr_ks("si10")) + " scc_rms=" + ftos(rms));

// FD Hessian + frequencies.
let summary = sparse_vibrations("si10", fd_h, scc_tol_hess, REPO_ROOT + "/debug/sparse_vib_si10h16_freq.txt");
print("vibrations: " + summary);

print("=== done ===");
