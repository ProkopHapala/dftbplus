// Is the dense SCC compute-bound or latency/launch-bound?
// Same system (GC, N=86), vary batch. If ms/iter is flat vs batch → we are
// latency-bound (per-launch, per-readback overhead dominates, GPU idle).
// If it scales ~linearly → compute-bound (then Jacobi work is the target).
const sk_dir = SK_DIR;
load_xyz("gc", "/home/prokop/git/dftbplus/data/xyz/guanine-cytosine.xyz");
let xyz = get_xyz("gc");

for nb in [1, 4, 19, 64, 256] {
    let all = [];
    for r in 0..nb { for v in xyz { all.push(v); } }
    let nm = "b" + itos(nb);
    load_xyz(nm, "/home/prokop/git/dftbplus/data/xyz/guanine-cytosine.xyz");
    gpu_new(nm, sk_dir, nb);
    gpu_set_coords(nm, all);
    gpu_smearing(nm, 0.002);
    gpu_reset_q(nm);
    gpu_scc(nm, 60, 1.0e-5);           // warm
    let t = clock();
    let nrep = 3;
    for i in 0..nrep { gpu_scc(nm, 60, 1.0e-5); }
    let ms = (clock() - t) / nrep * 1000.0;
    let it = gpu_scc_iters(nm);
    print("batch=" + itos(nb) + "  " + ftos(ms) + " ms/scc  iters=" + itos(it)
          + "  " + ftos(ms / it) + " ms/iter  " + ftos(ms / it / nb * 1000.0) + " us/iter/system");
}
print("=== scaling done ===");
