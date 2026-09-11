// D6 A/B: formic dimer scan with Löwdin-X reuse OFF then ON (two engines).
const sk_dir = SK_DIR;
const xyz_dir = REPO_ROOT + "/data/xyz";

print("=== D6 A/B formic scan ===");

load_xyz("formic", xyz_dir + "/formic_dimer.xyz");
let xyz0 = get_xyz("formic");

let mode = 0;
while mode < 2 {
    let name = if mode == 0 { "f_off" } else { "f_on" };
    load_xyz(name, xyz_dir + "/formic_dimer.xyz");
    gpu_new(name, sk_dir, 1);
    gpu_x_reuse(name, mode == 1);
    let e_gpu_ref = 0.0;
    let e_cpu_ref = 0.0;
    let k = 0;
    while k < 5 {
        let dz = k * 0.2;
        let xyz = [];
        let i = 0;
        while i < 30 { xyz.push(xyz0[i]); i += 1; }
        let a = 5;
        while a < 10 { xyz[3 * a + 2] = xyz[3 * a + 2] + dz; a += 1; }
        gpu_set_coords(name, xyz);
        gpu_reset_q(name);
        let rms = gpu_scc_mixer(name, 100, 1.0e-6, 0);
        let e_gpu = gpu_eval(name, false);
        let e_cpu = gpu_cpu_energy(name);
        if k == 0 { e_gpu_ref = e_gpu; e_cpu_ref = e_cpu; }
        print("  " + name + " k=" + itos(k) + " dz=" + ftos(dz) + " |ΔE_gpu-ΔE_cpu|=" + ftos((e_gpu - e_gpu_ref) - (e_cpu - e_cpu_ref)) + " rms=" + ftos(rms) + " iters=" + itos(gpu_scc_iters(name)) + " status=" + gpu_scc_status(name));
        k += 1;
    }
    mode += 1;
}
print("done");
