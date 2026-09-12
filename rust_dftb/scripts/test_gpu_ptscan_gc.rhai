// GC proton-transfer relaxed scan — the real workload.
// Replica i = H13 (guanine N1–H···cytosine N3) constrained to d = d0 + i·step
// along the N8→N20 axis; all other atoms relax by FIRE.
// 0-based atom indices: donor N = 8, proton H = 13, acceptor N = 20.
// (Fixed 2026-09-13: the axis endpoint was atom 21 = C — a 1-based indexing
// slip; the true N3 acceptor is atom 20. Constraint is now a DISTANCE
// r(N8–H13)=d — 1 DOF — not a full freeze of H13.)
const sk_dir = SK_DIR;
const N_PTS = 19;
const D0 = 1.0;      // Å — N8–H distance at first point
const DSTEP = 0.05;  // Å

print("=== GC proton-transfer scan (GPU batched FIRE) ===");
load_xyz("gc", "/home/prokop/git/dftbplus/data/xyz/guanine-cytosine.xyz");
let xyz = get_xyz("gc");
let n_atoms = 29;

// donor N8, proton H13, acceptor N20 — unit vector along the H-bond axis
let dx = xyz[3*20]   - xyz[3*8];
let dy = xyz[3*20+1] - xyz[3*8+1];
let dz = xyz[3*20+2] - xyz[3*8+2];
let dNN = sqrt(dx*dx + dy*dy + dz*dz);
dx /= dNN; dy /= dNN; dz /= dNN;
print("N8–N20 = " + ftos(dNN) + " A; scan " + N_PTS + " pts from " + ftos(D0) + " A, step " + ftos(DSTEP));

// build all scan geometries (H13 displaced along the axis, rest = ref)
let all = [];
for p in 0..N_PTS {
    let d = D0 + p * DSTEP;
    for a in 0..n_atoms {
        if a == 13 {
            all.push(xyz[3*8]   + d * dx);
            all.push(xyz[3*8+1] + d * dy);
            all.push(xyz[3*8+2] + d * dz);
        } else {
            all.push(xyz[3*a]);
            all.push(xyz[3*a+1]);
            all.push(xyz[3*a+2]);
        }
    }
}

gpu_new("gc", sk_dir, N_PTS);
gpu_set_coords("gc", all);
// distance constraint r(N8–H13) = d per replica — 1 DOF, H can relax
// off-axis; N8 stays free. (Replaces the full freeze of H13.)
let d_targets = [];
for p in 0..N_PTS { d_targets.push(D0 + p * DSTEP); }
gpu_set_constraint("gc", 8, 13, d_targets);
gpu_smearing("gc", 0.002);   // ~630 K — stabilizes mid-transfer SCC (gap ~0.5 mHa)
gpu_reset_q("gc");

let t0 = clock();
gpu_scc("gc", 60, 1.0e-5);
let mf = gpu_relax("gc", 150, 1.0e-3, 1.0e-5);
let total_s = clock() - t0;

print("--- E(r) profile (Ha) ---");
gpu_eval("gc", true);
let e_min = 1.0e30;
for p in 0..N_PTS { let e = gpu_energy_i("gc", p); if e < e_min { e_min = e; } }
for p in 0..N_PTS {
    let d = D0 + p * DSTEP;
    let e = gpu_energy_i("gc", p);
    print("r_NH=" + ftos(d) + " A  E=" + ftos(e) + "  dE=" + ftos((e - e_min) * 627.509) + " kcal/mol");
}
print("=== done: max|F|=" + ftos(mf) + "  total=" + ftos(total_s) + " s (" + ftos(total_s * 1000.0 / N_PTS) + " ms/pt) ===");
