// First genuine sparse-topology run: 1648-atom passivated Si sphere
// (FireCore pyBall.nanocrystal_gen sphere cut, R=18 A, diameter ~36 A >>
// r_hs=11.74 A => M_HS geometric, not full). Measures nnz, SCC iters, E.
//
//   dftb_engine --script scripts/sparse_big_nc.rhai --sk-dir <matsci-0-3>

const sk_dir = SK_DIR;
const xyz = REPO_ROOT + "/debug/nanocrystals/si_sphere_R18.xyz";

print("=== SparseDftb large-crystal test: Si sphere R18 (1648 atoms) ===");
load_xyz("big", xyz);
// r_trunc=8 A: matsci measured error <0.05 mHa on cube_si65 (taper [7,8] A).
// M_HS mask = 9 A -> ~165 nbr/atom (vs ~270 at full 11.7 A table range).
// r_k=r_z=10 A: K/Z masks -> ~285 nbr/atom (full-radius 12.7 A -> ~620 rows
// exceeds MAX_LEFT_BLOCKS=512 local-mem cap). K truncation error is a
// separate axis to quantify — this run tests feasibility + convergence.
let n_orbs = sparse_new("big", sk_dir, 8.0, 1.0, 10.0, 10.0);
sparse_tc2_tol("big", 1.0e-6);
sparse_ns_tol("big", 1.0e-4);   // r_z=10 truncation floor ~2e-5 — accept explicitly
print("n_atoms=" + itos(sparse_n_atoms("big")) + " n_orbs(padded)=" + itos(n_orbs));

let rms = sparse_scc("big", 100, 1.0e-5);
let e = sparse_eval("big", true);
print("scc: rms=" + ftos(rms) + " iters=" + itos(sparse_scc_iters("big")));
print("E=" + ftos(e) + " Ha  max|F|=" + ftos(sparse_max_force("big")) + "  Tr(KS)=" + ftos(sparse_tr_ks("big")));
print("=== done ===");
