// Frozen-density H/S-block Hessian at a geometry given by RUST_DFTB_XYZ.
// Radii are the nanoparticle recipe. RUST_DFTB_FORCE_GEOM_MASK=1 keeps that
// geometric mask on molecules small enough to otherwise get a complete mask.
// r_K defaults to 40 Å so the density kernel is complete on these cages;
// the approximation under test is the Hessian, not a chopped kernel.
//
//   RUST_DFTB_FORCE_GEOM_MASK=1 RUST_DFTB_VIB_FROZEN=1 RUST_DFTB_VIB_HSONLY=1 \
//   RUST_DFTB_XYZ=... RUST_DFTB_OUT=... RUST_DFTB_VIB_HESSOUT=... \
//   dftb_engine --script scripts/sparse_vib_ref.rhai --sk-dir <slakos>

let xyz = env("RUST_DFTB_XYZ");
let out = env("RUST_DFTB_OUT");
if xyz == "" || out == "" {
    print("RUST_DFTB_XYZ and RUST_DFTB_OUT are required");
}

load_xyz("mol", xyz);
let r_auto = sparse_hs_decay("mol", SK_DIR);
let rtxt = env("RUST_DFTB_R_TRUNC");
let r_hs = if rtxt == "" { r_auto } else { parse_float(rtxt) };
let ktxt = env("RUST_DFTB_R_K");
let r_k = if ktxt == "" { 40.0 } else { parse_float(ktxt) };
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
