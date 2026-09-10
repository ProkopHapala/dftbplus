// Drive the production GpuDftb engine from a script — not a new cargo test binary.
//
//   cargo run --release -p rust_dftb --bin dftb_engine -- \
//     --script rust_dftb/scripts/test_gpu_dftb_molecules.rhai \
//     --sk-dir $RUST_DFTB_SK_DIR
//
// Homogeneous batch: replicas are copies of one template.
// Different molecules = sequential gpu_new (AT then GC), not mixed in one launch.

const sk_dir = SK_DIR;
const xyz_dir = REPO_ROOT + "/data/xyz";
const scc_max = 100;
const scc_tol = 1.0e-6;
const replica_tol = 1.0e-4; // f32 replica agreement; print actual |dE|

print("=== GpuDftb molecules (H2O replicas, AT, GC, azaindole) ===");
print("SK_DIR=" + sk_dir);
print("REPO_ROOT=" + REPO_ROOT);
print("");

// ─── H2O batch=1 ─────────────────────────────────────────────────
print("[1] H2O batch=1");
make_geom("h2o", "O,H,H", [
    0.0, 0.0, 0.0,
    -0.7580632005, 0.6358101311, 0.0,
    0.7580632005, 0.6358101311, 0.0
]);
let n_h2o = gpu_new("h2o", sk_dir, 1);
print("    N=" + itos(n_h2o));
let rms = gpu_scc("h2o", scc_max, scc_tol);
let e = gpu_eval("h2o", true);
let fmax = gpu_max_force("h2o");
assert_finite(e, "H2O E");
assert_finite(fmax, "H2O max|F|");
print("    E=" + ftos(e) + " rms=" + ftos(rms) + " iters=" + itos(gpu_scc_iters("h2o")) + " max|F|=" + ftos(fmax));
print("");

// ─── H2O batch=4 (identical replicas must agree) ─────────────────
print("[2] H2O batch=4");
make_geom("h2o4", "O,H,H", [
    0.0, 0.0, 0.0,
    -0.7580632005, 0.6358101311, 0.0,
    0.7580632005, 0.6358101311, 0.0
]);
gpu_new("h2o4", sk_dir, 4);
gpu_scc("h2o4", scc_max, scc_tol);
let e0 = gpu_eval("h2o4", true);
assert_finite(e0, "H2O4 E[0]");
let b = gpu_n_batch("h2o4");
print("    batch=" + itos(b) + " E[0]=" + ftos(e0) + " max|F|=" + ftos(gpu_max_force("h2o4")));
let i = 1;
while i < b {
    let ei = gpu_energy_i("h2o4", i);
    print("    E[" + itos(i) + "]=" + ftos(ei));
    assert_close(e0, ei, replica_tol, "H2O replica energy mismatch");
    i += 1;
}
print("");

// ─── AT batch=1 (N>64 tiled Jacobi) ──────────────────────────────
print("[3] adenine-thymine batch=1");
load_xyz("at", xyz_dir + "/adenine-thymine.xyz");
let n_at = gpu_new("at", sk_dir, 1);
print("    atoms loaded, N=" + itos(n_at));
let rms_at = gpu_scc("at", scc_max, scc_tol);
let e_at = gpu_eval("at", true);
assert_finite(e_at, "AT E");
assert_finite(gpu_max_force("at"), "AT max|F|");
print("    E=" + ftos(e_at) + " rms=" + ftos(rms_at) + " iters=" + itos(gpu_scc_iters("at")) + " max|F|=" + ftos(gpu_max_force("at")));
print("");

// ─── AT batch=2 ──────────────────────────────────────────────────
print("[4] adenine-thymine batch=2");
load_xyz("at2", xyz_dir + "/adenine-thymine.xyz");
gpu_new("at2", sk_dir, 2);
gpu_scc("at2", scc_max, scc_tol);
let e_at2 = gpu_eval("at2", true);
assert_finite(e_at2, "AT2 E[0]");
print("    E[0]=" + ftos(e_at2) + " E[1]=" + ftos(gpu_energy_i("at2", 1)) + " max|F|=" + ftos(gpu_max_force("at2")));
assert_close(e_at2, gpu_energy_i("at2", 1), replica_tol, "AT replica energy mismatch");
print("");

// ─── GC batch=1 ──────────────────────────────────────────────────
print("[5] guanine-cytosine batch=1");
load_xyz("gc", xyz_dir + "/guanine-cytosine.xyz");
let n_gc = gpu_new("gc", sk_dir, 1);
print("    N=" + itos(n_gc));
let rms_gc = gpu_scc("gc", scc_max, scc_tol);
let e_gc = gpu_eval("gc", true);
assert_finite(e_gc, "GC E");
assert_finite(gpu_max_force("gc"), "GC max|F|");
print("    E=" + ftos(e_gc) + " rms=" + ftos(rms_gc) + " iters=" + itos(gpu_scc_iters("gc")) + " max|F|=" + ftos(gpu_max_force("gc")));
print("");

// ─── 7-azaindole dimer batch=1 ───────────────────────────────────
print("[6] azaindol_dimer batch=1");
load_xyz("aza", xyz_dir + "/azaindol_dimer.xyz");
let n_aza = gpu_new("aza", sk_dir, 1);
print("    N=" + itos(n_aza));
let rms_aza = gpu_scc("aza", scc_max, scc_tol);
let e_aza = gpu_eval("aza", true);
assert_finite(e_aza, "azaindole E");
assert_finite(gpu_max_force("aza"), "azaindole max|F|");
print("    E=" + ftos(e_aza) + " rms=" + ftos(rms_aza) + " iters=" + itos(gpu_scc_iters("aza")) + " max|F|=" + ftos(gpu_max_force("aza")));
print("");

print("All GpuDftb script cases finished.");
