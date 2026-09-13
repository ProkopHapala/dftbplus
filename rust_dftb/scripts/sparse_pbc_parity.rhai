// pbc parity refinement: previous run used r_trunc=5.3 taper=0.4 which cut
// H/S from 4.9 Å — but pbc integrals may be nonzero to the 5.5 Å table end.
// Retry with r_trunc=5.45, taper_w=0.3 (taper window 5.15–5.45 ≈ table edge).
const sk_dir = SK_DIR;
const xyz = REPO_ROOT + "/debug/nanocrystals/si_sphere_R14.xyz";
load_xyz("pbcp", xyz);
sparse_new("pbcp", sk_dir, 5.45, 0.3, 0.0, 0.0, 512.0);
sparse_tc2_tol("pbcp", 1.0e-5);
sparse_ns_tol("pbcp", 1.0e-4);
let t0 = clock();
let rms = sparse_scc_try("pbcp", 60, 1.0e-5);
let t_scc = clock() - t0;
print("pbc r_trunc=5.45: rms=" + ftos(rms) + " iters=" + itos(sparse_scc_iters("pbcp")));
let e = sparse_eval("pbcp", true);
let f = sparse_max_force("pbcp");
print("E=" + ftos(e) + " Ha  (DFTB+ pbc = -847.0895637, dE=" + ftos((e + 847.0895637) * 1000.0) + " mHa)  max|F|=" + ftos(f) + "  t_scc=" + ftos(t_scc * 1000.0) + " ms");
print("=== done ===");
