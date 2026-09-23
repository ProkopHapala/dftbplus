// One sparse FIRE relaxation. The crystal, the kick, and the step cap
// come from the environment, not from a Rust edit.
//
//   RUST_DFTB_XYZ=../debug/nanocrystals/si_sphere_R14.xyz \
//   RUST_DFTB_OUT=../debug/sparse_relax/r14 \
//   RUST_DFTB_WALL_SECS=18 RUST_DFTB_TRUNC_PRODUCTS=1 \
//   RUST_DFTB_SCC_RHGATE=1e-2 RUST_DFTB_PROF=evt RUST_DFTB_PROF_ACCUM=1 \
//   RUST_DFTB_GEOM_CERT=1   # only when you want the per-step Tr/R_H lines \
//   dftb_engine --script scripts/sparse_relax.rhai --sk-dir "$SK_DIR"
//
// A later chunk continues the same trajectory:
//   RUST_DFTB_CONTINUE=1   (loads $OUT/after.xyz, no second kick)

let xyz = env("RUST_DFTB_XYZ");
let out = env("RUST_DFTB_OUT");
if xyz == "" { die("RUST_DFTB_XYZ is empty"); }
if out == "" { die("RUST_DFTB_OUT is empty"); }

let jtxt = env("RUST_DFTB_JITTER");
let jitter = if jtxt == "" { 0.1 } else { parse_float(jtxt) };
let stxt = env("RUST_DFTB_MAX_STEPS");
let steps = if stxt == "" { 400 } else { parse_int(stxt) };
let cont = env("RUST_DFTB_CONTINUE") == "1";
let src = if cont { out + "/after.xyz" } else { xyz };

load_xyz("mol", src);
// H/S radius from this SK set: last grid point with |H| or |S| > 1e-4.
// The density kernel is a different mask. It follows the gap, not the
// table. RUST_DFTB_R_K sets it; unset keeps the matsci-silicon 12 Å.
// RUST_DFTB_R_Z is the overlap inverse, unset = same number as R_K.
let r_auto = sparse_hs_decay("mol", SK_DIR);
let rtxt = env("RUST_DFTB_R_TRUNC");
let r_hs = if rtxt == "" { r_auto } else { parse_float(rtxt) };
let ktxt = env("RUST_DFTB_R_K");
let r_k = if ktxt == "" { 12.0 } else { parse_float(ktxt) };
let ztxt = env("RUST_DFTB_R_Z");
let r_z = if ztxt == "" { r_k } else { parse_float(ztxt) };
print("mask  r_trunc=" + ftos(r_hs) + "  r_K=" + ftos(r_k) + "  r_Z=" + ftos(r_z));
// Skin covers one FIRE step (cap 0.1 Å). Two atoms can close by 2·0.1 Å.
sparse_new_mask("mol", SK_DIR, r_hs, 0.3, r_k, r_z, 0.2, 1024.0);
if !cont && jitter > 0.0 {
    sparse_jitter("mol", jitter, 0x5eed);
}
sparse_geom_mode("mol", "bold");
sparse_fire_dt("mol", 0.1);
// f_tol 1e-5 so the loop is stopped by the wall or by the force plateau, not by 1e-3.
sparse_relax_traj("mol", steps, 1e-5, 1e-5, out + "/traj.xyz", out + "/history.csv");
