// Package 2 measurements: frozen-H, mixer A/B, H2O F vs CPU, formic relative ΔE.
// Drive GpuDftb — not a new cargo test binary.
//
//   cargo run --release --bin dftb_engine -- \
//     --script scripts/test_gpu_dftb_measure.rhai \
//     --sk-dir $RUST_DFTB_SK_DIR
//
// mix: 0 = GPU DIIS (production), 1 = GPU simple α=0.3, 2 = host f64 DiisMixer.

const sk_dir = SK_DIR;
const xyz_dir = REPO_ROOT + "/data/xyz";
const scc_max = 100;
const scc_tol = 1.0e-6;

print("=== GpuDftb Package 2 measure (frozen-H, mixer A/B, forces, ΔE scan) ===");
print("SK_DIR=" + sk_dir);
print("REPO_ROOT=" + REPO_ROOT);
print("");

// ─── H2O: mixer 0 / 1 / 2 + frozen-H + CPU forces on mix=0 ─────────
print("[1] H2O mixer A/B + frozen-H + CPU forces");
make_geom("h2o", "O,H,H", [
    0.0, 0.0, 0.0,
    -0.7580632005, 0.6358101311, 0.0,
    0.7580632005, 0.6358101311, 0.0
]);
gpu_new("h2o", sk_dir, 1);

print("  mix=0 GPU DIIS");
gpu_reset_q("h2o");
let rms0 = gpu_scc_mixer("h2o", scc_max, scc_tol, 0);
let e0 = gpu_measure("h2o", true);
assert_finite(e0, "H2O mix0 E");
print("    mix=0 E=" + ftos(e0) + " mixer_rms=" + ftos(rms0) + " iters=" + itos(gpu_scc_iters("h2o")) + " stalled=" + itos(gpu_scc_stalled("h2o")) + " q_rms=" + ftos(gpu_q_rms("h2o")) + " max|F|=" + ftos(gpu_max_force("h2o")));

print("  mix=1 GPU simple α=0.3");
gpu_reset_q("h2o");
let rms1 = gpu_scc_mixer("h2o", scc_max, scc_tol, 1);
let e1 = gpu_eval("h2o", false);
assert_finite(e1, "H2O mix1 E");
print("    mix=1 E=" + ftos(e1) + " mixer_rms=" + ftos(rms1) + " iters=" + itos(gpu_scc_iters("h2o")) + " stalled=" + itos(gpu_scc_stalled("h2o")) + " q_rms=" + ftos(gpu_q_rms("h2o")) + " |E-E0|=" + ftos(e1 - e0));

print("  mix=2 host f64 DIIS (hist=min(10,n_atoms), warmup=1)");
gpu_reset_q("h2o");
let rms2 = gpu_scc_mixer("h2o", scc_max, scc_tol, 2);
let e2 = gpu_measure("h2o", false);
assert_finite(e2, "H2O mix2 E");
print("    mix=2 E=" + ftos(e2) + " mixer_rms=" + ftos(rms2) + " iters=" + itos(gpu_scc_iters("h2o")) + " stalled=" + itos(gpu_scc_stalled("h2o")) + " q_rms=" + ftos(gpu_q_rms("h2o")) + " |E-E0|=" + ftos(e2 - e0));
print("");

// ─── AT: frozen-H + CPU energy/forces (tiled Jacobi N>64) ──────────
print("[2] adenine-thymine mix=0 frozen-H + CPU");
load_xyz("at", xyz_dir + "/adenine-thymine.xyz");
gpu_new("at", sk_dir, 1);
gpu_reset_q("at");
let rms_at = gpu_scc_mixer("at", scc_max, scc_tol, 0);
let e_at = gpu_measure("at", true);
assert_finite(e_at, "AT E");
print("    AT E=" + ftos(e_at) + " mixer_rms=" + ftos(rms_at) + " iters=" + itos(gpu_scc_iters("at")) + " stalled=" + itos(gpu_scc_stalled("at")) + " q_rms=" + ftos(gpu_q_rms("at")) + " max|F|=" + ftos(gpu_max_force("at")));
print("");

// ─── Formic dimer: relative ΔE vs CPU while translating monomer 2 ──
print("[3] formic dimer relative ΔE scan (shift atoms 5-9 along z)");
load_xyz("formic", xyz_dir + "/formic_dimer.xyz");
gpu_new("formic", sk_dir, 1);
let xyz0 = get_xyz("formic");
let n_xyz = 30;
let e_gpu_ref = 0.0;
let e_cpu_ref = 0.0;
let k = 0;
while k < 5 {
    let dz = k * 0.2;
    let xyz = [];
    let i = 0;
    while i < n_xyz {
        xyz.push(xyz0[i]);
        i += 1;
    }
    let a = 5;
    while a < 10 {
        let iz = 3 * a + 2;
        xyz[iz] = xyz[iz] + dz;
        a += 1;
    }
    gpu_set_coords("formic", xyz);
    gpu_reset_q("formic");
    let rms = gpu_scc_mixer("formic", scc_max, scc_tol, 0);
    let e_gpu = gpu_eval("formic", false);
    let e_cpu = gpu_cpu_energy("formic");
    assert_finite(e_gpu, "formic GPU E");
    assert_finite(e_cpu, "formic CPU E");
    if k == 0 {
        e_gpu_ref = e_gpu;
        e_cpu_ref = e_cpu;
    }
    let d_abs = e_gpu - e_cpu;
    let de_gpu = e_gpu - e_gpu_ref;
    let de_cpu = e_cpu - e_cpu_ref;
    print("    k=" + itos(k) + " dz=" + ftos(dz) + " E_gpu=" + ftos(e_gpu) + " E_cpu=" + ftos(e_cpu) + " E_gpu-E_cpu=" + ftos(d_abs) + " ΔE_gpu=" + ftos(de_gpu) + " ΔE_cpu=" + ftos(de_cpu) + " |ΔE_gpu-ΔE_cpu|=" + ftos(de_gpu - de_cpu) + " rms=" + ftos(rms) + " iters=" + itos(gpu_scc_iters("formic")) + " stalled=" + itos(gpu_scc_stalled("formic")));
    k += 1;
}
print("");
print("Package 2 measure script finished.");
