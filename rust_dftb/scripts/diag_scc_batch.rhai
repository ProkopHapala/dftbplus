// DIAGNOSTIC: is batched SCC broken for non-identical geometries, or do
// strained geometries genuinely fail to converge?
// A) batch=4 IDENTICAL native GC → should all converge like batch=1.
// B) batch=4 with H13 at d=1.0,1.4,1.6,1.9 along N8->N21 → which converge?
const sk_dir = SK_DIR;

print("=== A: batch=4 IDENTICAL GC ===");
load_xyz("gc", "/home/prokop/git/dftbplus/data/xyz/guanine-cytosine.xyz");
let xyz = get_xyz("gc");
gpu_new("gc", sk_dir, 4);
gpu_reset_q("gc");
gpu_scc("gc", 100, 1.0e-6);

print("=== B: batch=4 scan points (d=1.0,1.4,1.6,1.9) ===");
let dx = xyz[3*20]   - xyz[3*8];
let dy = xyz[3*20+1] - xyz[3*8+1];
let dz = xyz[3*20+2] - xyz[3*8+2];
let dNN = sqrt(dx*dx + dy*dy + dz*dz);
dx /= dNN; dy /= dNN; dz /= dNN;
let dvals = [1.0, 1.4, 1.6, 1.9];
let all = [];
for p in 0..4 {
    let d = dvals[p];
    for a in 0..29 {
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
gpu_set_coords("gc", all);
gpu_reset_q("gc");
gpu_scc("gc", 100, 1.0e-6);
print("=== diag done ===");
