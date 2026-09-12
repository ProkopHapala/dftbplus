// A/B: device-FIRE equivalence — same 19-replica GC scan, 30 FIRE steps,
// three modes: (a) freeze H13 (old semantics), (b) distance constraint
// r(N8–H13)=d, (c) no constraint at all. Prints E[0] + constraint residual
// + max|F| per step to locate the divergence.
const sk_dir = SK_DIR;
const N_PTS = 19;
const D0 = 1.0;
const DSTEP = 0.05;
const N_STEPS = 30;

load_xyz("gc", "/home/prokop/git/dftbplus/data/xyz/guanine-cytosine.xyz");
let xyz = get_xyz("gc");
let n_atoms = 29;
let dx = xyz[3*20] - xyz[3*8];  let dy = xyz[3*20+1] - xyz[3*8+1];  let dz = xyz[3*20+2] - xyz[3*8+2];
let dNN = sqrt(dx*dx + dy*dy + dz*dz);
dx /= dNN; dy /= dNN; dz /= dNN;

let all = [];
for p in 0..N_PTS {
    let d = D0 + p * DSTEP;
    for a in 0..n_atoms {
        if a == 13 {
            all.push(xyz[3*8]   + d * dx);
            all.push(xyz[3*8+1] + d * dy);
            all.push(xyz[3*8+2] + d * dz);
        } else {
            all.push(xyz[3*a]); all.push(xyz[3*a+1]); all.push(xyz[3*a+2]);
        }
    }
}

let d_targets = [];
for p in 0..N_PTS { d_targets.push(D0 + p * DSTEP); }

for mode in 0..3 {
    gpu_new("gc", sk_dir, N_PTS);
    gpu_set_coords("gc", all);
    if mode == 0 { gpu_freeze_atoms("gc", [13]); }
    if mode == 1 { gpu_set_constraint("gc", 8, 13, d_targets); }
    gpu_smearing("gc", 0.002);
    gpu_reset_q("gc");
    gpu_scc("gc", 60, 1.0e-5);
    let tag = if mode == 0 { "FREEZE" } else if mode == 1 { "CONSTR" } else { "FREE" };
    print("=== mode " + tag + " ===");
    let t0 = clock();
    for s in 0..N_STEPS {
        let mf = gpu_fire_step("gc", 1.0e-3);
        gpu_scc("gc", 60, 1.0e-5);
        gpu_eval("gc", false);
        print(tag + " step " + (s+1) + "  E[0]=" + gpu_energy_i("gc", 0) + "  E[9]=" + gpu_energy_i("gc", 9) + "  E[18]=" + gpu_energy_i("gc", 18) + "  max|F|=" + mf);
    }
    print(tag + " total " + ftos(clock() - t0) + " s");
    // constraint residual: |x13−x8| vs target per replica
    let c = gpu_get_coords("gc");
    let cerr_max = 0.0;
    let resid = "";
    for p in 0..N_PTS {
        let base = p * n_atoms * 3;
        let rx = c[base+3*13]   - c[base+3*8];
        let ry = c[base+3*13+1] - c[base+3*8+1];
        let rz = c[base+3*13+2] - c[base+3*8+2];
        let r  = sqrt(rx*rx + ry*ry + rz*rz);
        let cerr = abs(r - d_targets[p]);
        if cerr > cerr_max { cerr_max = cerr; }
        resid += " " + (r * 1000).round() / 1000;
    }
    print(tag + " r(N8–H13) per replica:" + resid);
    print(tag + " max|r−d| = " + cerr_max + " Å");
}
print("=== done ===");
