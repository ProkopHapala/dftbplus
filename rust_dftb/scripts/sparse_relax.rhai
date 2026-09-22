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
// r_trunc, taper, r_k, r_z, r_skin, max row degree. Same mask as the R10 run.
sparse_new_mask("mol", SK_DIR, 5.45, 0.3, 12.0, 12.0, 3.0, 1024.0);
if !cont && jitter > 0.0 {
    sparse_jitter("mol", jitter, 0x5eed);
}
sparse_geom_mode("mol", "bold");
sparse_fire_dt("mol", 0.1);
// f_tol 1e-5 so the loop is stopped by the wall or by the force plateau, not by 1e-3.
sparse_relax_traj("mol", steps, 1e-5, 1e-5, out + "/traj.xyz", out + "/history.csv");
