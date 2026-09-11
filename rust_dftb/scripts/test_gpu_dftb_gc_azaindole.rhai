// D3/D4/D5/D9/D10 verification on GC + azaindole dimer (N>64 systems).
// Drive GpuDftb production path; gpu_measure prints full diagnostics + CPU parity.

const sk_dir = SK_DIR;
const xyz_dir = REPO_ROOT + "/data/xyz";
const scc_max = 100;
const scc_tol = 1.0e-6;

print("=== GpuDftb GC + azaindole verify (mix=0, prec default) ===");

load_xyz("gc", xyz_dir + "/guanine-cytosine.xyz");
gpu_new("gc", sk_dir, 1);
gpu_reset_q("gc");
let gc_rms = gpu_scc_mixer("gc", scc_max, scc_tol, 0);
gpu_measure("gc", true);

load_xyz("aza", xyz_dir + "/azaindol_dimer.xyz");
gpu_new("aza", sk_dir, 1);
gpu_reset_q("aza");
let az_rms = gpu_scc_mixer("aza", scc_max, scc_tol, 0);
gpu_measure("aza", true);

print("done");
