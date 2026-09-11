// D6 diagnostic: reuse ON, run k=0/k=1 with full gpu_measure diagnostics.
const sk_dir = SK_DIR;
const xyz_dir = REPO_ROOT + "/data/xyz";

load_xyz("f", xyz_dir + "/formic_dimer.xyz");
let xyz0 = get_xyz("f");
gpu_new("f", sk_dir, 1);
gpu_x_reuse("f", true);

let k = 0;
while k < 2 {
    let dz = k * 0.2;
    let xyz = [];
    let i = 0;
    while i < 30 { xyz.push(xyz0[i]); i += 1; }
    let a = 5;
    while a < 10 { xyz[3 * a + 2] = xyz[3 * a + 2] + dz; a += 1; }
    gpu_set_coords("f", xyz);
    gpu_reset_q("f");
    let rms = gpu_scc_mixer("f", 100, 1.0e-6, 0);
    gpu_measure("f", true);
    k += 1;
}
print("done");
