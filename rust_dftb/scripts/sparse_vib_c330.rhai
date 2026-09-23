// Hessian of the relaxed 330-atom carbon particle (c_equiv_R10, 3ob).
// The geometry is that particle's own minimum. Skin is one FIRE step.
//
//   RUST_DFTB_VIB_FROZEN=1 RUST_DFTB_VIB_HSONLY=1 RUST_DFTB_VIB_BATCH=16 \
//   dftb_engine --script scripts/sparse_vib_c330.rhai \
//     --sk-dir /home/prokop/SIMULATIONS/dftbplus/slakos/3ob-3-1
//
// Unset VIB_FROZEN and the column uses the bold FIRE update
// (two commutators at η=8, one McWeeny). VIB_MAXCOL bounds the columns.

let xyz_env = env("RUST_DFTB_XYZ");
let xyz = if xyz_env == "" { REPO_ROOT + "/debug/sparse_relax/mask_c3ob/after.xyz" } else { xyz_env };
let out_env = env("RUST_DFTB_OUT");
let out = if out_env == "" { REPO_ROOT + "/debug/sparse_vib_c330/freq.txt" } else { out_env };

load_xyz("mol", xyz);
let r_auto = sparse_hs_decay("mol", SK_DIR);
let rtxt = env("RUST_DFTB_R_TRUNC");
let r_hs = if rtxt == "" { r_auto } else { parse_float(rtxt) };
// This geometry was relaxed at r_K = 8.76 Å (3ob H/S decay × 12/7).
let ktxt = env("RUST_DFTB_R_K");
let r_k = if ktxt == "" { 8.76 } else { parse_float(ktxt) };
let ztxt = env("RUST_DFTB_R_Z");
let r_z = if ztxt == "" { r_k } else { parse_float(ztxt) };
print("mask  r_trunc=" + ftos(r_hs) + "  r_K=" + ftos(r_k) + "  r_Z=" + ftos(r_z));
sparse_new_mask("mol", SK_DIR, r_hs, 0.3, r_k, r_z, 0.2, 1024.0);
sparse_geom_mode("mol", "bold");
let rms = sparse_scc("mol", 80, 1e-5);
let e = sparse_eval("mol", true);
print("center  E=" + ftos(e) + "  max|F|=" + ftos(sparse_max_force("mol")) + "  rms=" + ftos(rms));
let t0 = clock();
let summary = sparse_vibrations("mol", 0.02, 1e-5, out);
print("hessian t=" + ftos(clock() - t0) + " s");
print(summary);
