// d=1.9 far-transfer point at batch=1, CORRECT species (from file).
// 2026-09-12b finding: CPU build_scc DIIS does NOT converge here either
// (RMS limit-cycle 0.015..0.23 over 300 iters). This is a genuine SCC
// fixed-point instability, not an f32 issue.
const sk_dir = SK_DIR;
load_xyz("gc", "/home/prokop/git/dftbplus/data/xyz/guanine-cytosine.xyz");
let xyz = get_xyz("gc");
let dx = xyz[3*21]   - xyz[3*8];
let dy = xyz[3*21+1] - xyz[3*8+1];
let dz = xyz[3*21+2] - xyz[3*8+2];
let dNN = sqrt(dx*dx + dy*dy + dz*dz);
dx /= dNN; dy /= dNN; dz /= dNN;
let d = 1.9;
let sc = [];
for a in 0..29 {
    if a == 13 {
        sc.push(xyz[3*8]   + d * dx);
        sc.push(xyz[3*8+1] + d * dy);
        sc.push(xyz[3*8+2] + d * dz);
    } else {
        sc.push(xyz[3*a]); sc.push(xyz[3*a+1]); sc.push(xyz[3*a+2]);
    }
}
gpu_new("gc", sk_dir, 1);
gpu_set_coords("gc", sc);
gpu_reset_q("gc");
print("--- mix=0 DIIS, integer occ, 60 iters ---");
gpu_scc("gc", 60, 1.0e-6);
print("--- mix=0 DIIS + Fermi kT=0.002 ---");
gpu_smearing("gc", 0.002);
gpu_reset_q("gc");
gpu_scc("gc", 100, 1.0e-6);
let e2 = gpu_eval("gc", false);
print("GPU E smeared d=1.9: " + ftos(e2));
print("--- cpu_ref (DftbCpu f64 DIIS, incl E_rep) at same geometry ---");
try {
    let ec = gpu_cpu_energy("gc");
    print("CPU E at d=1.9: " + ftos(ec) + "  dE=" + ftos(e2 - ec));
} catch (err) {
    print("cpu_ref FAILED at d=1.9: " + err);
}
print("=== diag done ===");
