// Time + accuracy of ONE warm SCC iteration vs Jacobi MAX_SWEEPS cap
// (RUST_DFTB_JACOBI_SWEEPS). Quantifies: how many sweeps are actually being
// spent, and how much accuracy we lose by capping them — i.e. the size of
// the warm-start prize.
const sk_dir = SK_DIR;
load_xyz("gc", "/home/prokop/git/dftbplus/data/xyz/guanine-cytosine.xyz");
gpu_new("gc", sk_dir, 19);
gpu_smearing("gc", 0.002);
gpu_reset_q("gc");
let rms = gpu_scc("gc", 60, 1.0e-5);
let it0 = gpu_scc_iters("gc");
let t = clock();
for i in 0..3 { gpu_scc("gc", 60, 1.0e-5); }
let ms = (clock() - t) / 3.0 * 1000.0;
let e = gpu_eval("gc", false);
print("SWEEPCAP_RESULT  cold_iters=" + itos(it0) + "  warm_ms=" + ftos(ms) + "  rms=" + ftos(rms) + "  E=" + ftos(e));
