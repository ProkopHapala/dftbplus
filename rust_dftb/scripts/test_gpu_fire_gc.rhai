// D12: GPU FIRE relaxation check — GC dimer, Bitzek physics fixed
// (α‖v‖F̂ global norms, per-replica dt/α). Should converge max|F| below
// 1e-3 Ha/Å in a few hundred steps from the crystal geometry.
const sk_dir = SK_DIR;

print("=== GPU FIRE relax: GC ===");
load_xyz("gc", "/home/prokop/git/dftbplus/data/xyz/guanine-cytosine.xyz");
gpu_new("gc", sk_dir, 1);
gpu_reset_q("gc");
gpu_scc("gc", 100, 1.0e-6);
let f0 = gpu_eval("gc", true);
print("start: E=" + ftos(f0) + " max|F|=" + ftos(gpu_max_force("gc")));
let mf = gpu_relax("gc", 200, 1.0e-3, 1.0e-6);
print("relaxed: max|F|=" + ftos(mf));
let e1 = gpu_eval("gc", true);
print("final: E=" + ftos(e1));
print("=== done ===");
