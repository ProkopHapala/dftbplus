// One cold SCC and the analytic forces. No geometry step, no kick.
//   RUST_DFTB_XYZ=../debug/sparse_relax/ref/si5h12.xyz \
//   dftb_engine --script scripts/sparse_cold_point.rhai --sk-dir "$SK_DIR"

let xyz = env("RUST_DFTB_XYZ");
if xyz == "" { die("RUST_DFTB_XYZ is empty"); }
load_xyz("mol", xyz);
// Same masks as sparse_relax.rhai. n <= 64 is still a complete mask.
let r_auto = sparse_hs_decay("mol", SK_DIR);
let rtxt = env("RUST_DFTB_R_TRUNC");
let r_hs = if rtxt == "" { r_auto } else { parse_float(rtxt) };
let ktxt = env("RUST_DFTB_R_K");
let r_k = if ktxt == "" { 12.0 } else { parse_float(ktxt) };
let ztxt = env("RUST_DFTB_R_Z");
let r_z = if ztxt == "" { r_k } else { parse_float(ztxt) };
print("mask  r_trunc=" + ftos(r_hs) + "  r_K=" + ftos(r_k) + "  r_Z=" + ftos(r_z));
sparse_new_mask("mol", SK_DIR, r_hs, 0.3, r_k, r_z, 0.2, 1024.0);
let rms = sparse_scc("mol", 80, 1e-5);
let e = sparse_eval("mol", true);
let f = sparse_max_force("mol");
print("cold  E=" + ftos(e) + "  max|F|=" + ftos(f) + "  rms=" + ftos(rms));
