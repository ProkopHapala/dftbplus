// Per-stage timing at batch=19: where does one FIRE step's 0.7 s go?
const sk_dir = SK_DIR;
load_xyz("gc", "/home/prokop/git/dftbplus/data/xyz/guanine-cytosine.xyz");
gpu_new("gc", sk_dir, 19);
gpu_smearing("gc", 0.002);
gpu_reset_q("gc");
gpu_scc("gc", 60, 1.0e-5);

let t = clock();
for i in 0..5 { gpu_scc("gc", 60, 1.0e-5); }
print("scc (warm): " + ftos((clock() - t) / 5.0 * 1000.0) + " ms");

let t = clock();
for i in 0..5 { gpu_eval("gc", true); }
print("eval+forces: " + ftos((clock() - t) / 5.0 * 1000.0) + " ms");

let t = clock();
for i in 0..5 { gpu_fire_step("gc", 1.0e-4); }
print("fire_step (eval+move+set_coords): " + ftos((clock() - t) / 5.0 * 1000.0) + " ms");

let t = clock();
for i in 0..5 { gpu_eval("gc", false); }
print("eval energy only: " + ftos((clock() - t) / 5.0 * 1000.0) + " ms");
print("=== done ===");
