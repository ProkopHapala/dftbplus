// Drive the production SparseDftb engine from a script — not a new cargo test binary.
//
//   cargo run --release -p rust_dftb --bin dftb_engine -- \
//     --script rust_dftb/scripts/test_sparse_dftb_sih4.rhai \
//     --sk-dir $RUST_DFTB_SK_DIR
//
// Si/H needs matsci (or another Si–H SK pack), not mio. One system per sparse_new
// (no replica batch). Distorted SiH4, same construction as tests/sparse_dftb.rs.

const sk_dir = SK_DIR;
const scc_max = 80;
const scc_tol = 1.0e-5;

print("=== SparseDftb SiH4 (reuse SCC, FIRE, MD) ===");
print("SK_DIR=" + sk_dir);
print("REPO_ROOT=" + REPO_ROOT);
print("");

// Distorted tetrahedron, Si–H = 1.48 Å, angle 109.47° (not a local minimum).
print("[1] make_geom + sparse_new");
make_geom("sih4", "Si,H,H,H,H", [
    0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
    1.4800000000000000,  0.0000000000000000,  0.0000000000000000,
   -0.4933036064009911,  1.3953678912429424,  0.0000000000000000,
   -0.4933036064009911, -0.4650946033826280,  1.3155753729133621,
   -0.4933036064009911,  0.4650946033826280, -1.3155753729133621
]);
let n_orbs = sparse_new("sih4", sk_dir);
print("    n_atoms=" + itos(sparse_n_atoms("sih4")) + " n_orbs=" + itos(n_orbs));
print("");

print("[2] SCC");
let rms1 = sparse_scc("sih4", scc_max, scc_tol);
let e1 = sparse_eval("sih4", true);
let fmax1 = sparse_max_force("sih4");
let tr1 = sparse_tr_ks("sih4");
assert_finite(e1, "SiH4 E");
assert_finite(fmax1, "SiH4 max|F|");
assert_finite(tr1, "SiH4 Tr(KS)");
print("    E=" + ftos(e1) + " rms=" + ftos(rms1) + " iters=" + itos(sparse_scc_iters("sih4")) + " Tr(KS)=" + ftos(tr1) + " max|F|=" + ftos(fmax1));
print("    q=" + sparse_charges("sih4"));
if e1 >= 0.0 || e1 <= -20.0 {
    die("SiH4 energy left the physical window: E=" + ftos(e1) + " Ha (expected bound, roughly −3 Ha)");
}
assert_close(tr1, 4.0, 0.05, "SiH4 Tr(KS) vs N_occ=4");
print("");

print("[3] SCC reuse (same geometry, warm charges)");
let rms2 = sparse_scc("sih4", scc_max, scc_tol);
let e2 = sparse_eval("sih4", false);
print("    E=" + ftos(e2) + " rms=" + ftos(rms2) + " iters=" + itos(sparse_scc_iters("sih4")));
assert_close(e1, e2, 1.0e-5, "reuse SCC energy — pipeline is not persistent/warm");
print("");

print("[4] one FIRE step + SCC");
let mf_fire = sparse_fire_step("sih4", 1.0e-2);
let rms3 = sparse_scc("sih4", scc_max, scc_tol);
let e3 = sparse_eval("sih4", true);
assert_finite(e3, "SiH4 E after FIRE");
print("    E=" + ftos(e3) + " max|F|_pre=" + ftos(mf_fire) + " rms=" + ftos(rms3) + " iters=" + itos(sparse_scc_iters("sih4")));
if e3 > 20.0 || e3 < -20.0 {
    die("FIRE step exploded: E=" + ftos(e3));
}
assert_close(e1, e3, 1.0, "FIRE moved energy by more than 1 Ha");
print("");

print("[5] one MD step + SCC");
let mf_md = sparse_md_step("sih4", 0.05);
let rms4 = sparse_scc("sih4", scc_max, scc_tol);
let e4 = sparse_eval("sih4", true);
assert_finite(e4, "SiH4 E after MD");
print("    E=" + ftos(e4) + " max|F|_pre=" + ftos(mf_md) + " rms=" + ftos(rms4) + " iters=" + itos(sparse_scc_iters("sih4")));
if e4 > 20.0 || e4 < -20.0 {
    die("MD step exploded: E=" + ftos(e4));
}

print("");
print("=== SparseDftb SiH4 script finished ===");
