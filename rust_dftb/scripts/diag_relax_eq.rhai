// Device-FIRE sanity: relax the undisplaced GC equilibrium geometry,
// batch=1 — should converge monotonically-ish to max|F| < f_tol.
const sk_dir = SK_DIR;

load_xyz("gc", "/home/prokop/git/dftbplus/data/xyz/guanine-cytosine.xyz");
let xyz = get_xyz("gc");

gpu_new("gc", sk_dir, 1);
gpu_set_coords("gc", xyz);
gpu_smearing("gc", 0.002);
gpu_reset_q("gc");
gpu_scc("gc", 60, 1.0e-5);
gpu_eval("gc", false);
print("initial E=" + gpu_energy_i("gc", 0));
let t0 = clock();
let mf = gpu_relax("gc", 300, 1.0e-3, 1.0e-5);
print("relax done: max|F|=" + mf + "  " + ftos(clock() - t0) + " s");
gpu_eval("gc", false);
print("final E=" + gpu_energy_i("gc", 0));
gpu_measure("gc", true);
print("=== done ===");
