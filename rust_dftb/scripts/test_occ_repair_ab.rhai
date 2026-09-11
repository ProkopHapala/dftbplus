// §12 D3 A/B: does occupied-column renormalization inside the SCC loop
// lower the AT charge plateau (rms 1.3e-6 stalled vs tol 1e-6)?
//   mode 1 = finalize repair only (production default)
//   mode 3 = finalize repair + in-SCC renorm

const sk_dir = SK_DIR;
const xyz_dir = REPO_ROOT + "/data/xyz";
const scc_max = 100;
const scc_tol = 1.0e-6;

print("=== occ_repair A/B: AT, GPU DIIS ===");
load_xyz("at", xyz_dir + "/adenine-thymine.xyz");
gpu_new("at", sk_dir, 1);

gpu_occ_repair("at", 1);
gpu_reset_q("at");
let r1 = gpu_scc_mixer("at", scc_max, scc_tol, 0);
let e1 = gpu_measure("at", true);
print("mode=1 (finalize only): rms=" + ftos(r1) + " iters=" + itos(gpu_scc_iters("at")) + " stalled=" + itos(gpu_scc_stalled("at")) + " E=" + ftos(e1));

gpu_occ_repair("at", 3);
gpu_reset_q("at");
let r3 = gpu_scc_mixer("at", scc_max, scc_tol, 0);
let e3 = gpu_measure("at", true);
print("mode=3 (+in-SCC renorm): rms=" + ftos(r3) + " iters=" + itos(gpu_scc_iters("at")) + " stalled=" + itos(gpu_scc_stalled("at")) + " E=" + ftos(e3));

gpu_occ_repair("at", 1);
print("done.");
