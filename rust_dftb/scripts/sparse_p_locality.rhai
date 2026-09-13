// L3 (manifest §15.10): is P already local — does the error live in P or in
// the K=PZ recovery? Purify P at each mask, compare E_P = 2Tr(P·Z·H_scc)
// against the wide reference (r_k=0) BEFORE K recovery. If E_P at deg ~52-108
// matches the reference, the deg-200 floor was K recovery, not P locality —
// and the hot SCC loop could run at low degree.
//
// Run on matsci (the case where truncation failed). mode=p throughout.
//
//   dftb_engine --script scripts/sparse_p_locality.rhai --sk-dir <matsci>

const sk_dir = SK_DIR;
const xyz = REPO_ROOT + "/debug/nanocrystals/si_sphere_R14.xyz";
const r_trunc = 8.0;

print("=== L3 P-locality: E_P=2Tr(P·ZH_scc) vs degree, matsci R14 ===");

let e0 = 0.0;
let first = true;
for rk in [0.0, 10.0, 8.0, 6.0] {
    let name = "p" + itos((rk * 10.0).to_int());
    load_xyz(name, xyz);
    let t_init = clock();
    sparse_new(name, sk_dir, r_trunc, 1.0, rk, rk, 512.0);
    sparse_tc2_tol(name, 1.0e-5);
    sparse_ns_tol(name, 1.0e-4);
    sparse_purifier(name, "p");
    let t0 = clock();
    let rms = sparse_scc_try(name, 60, 1.0e-5);
    let t_scc = clock() - t0;
    if rms.is_nan() {
        print("r_k=" + ftos(rk) + "  *** SCC FAILED ***");
    } else {
        let e_p = sparse_eval_p(name);           // 2·Tr(P·ZH) — no K recovery
        let e_k = sparse_eval(name, false);       // recovered-K energy for contrast
        if first { e0 = e_p; first = false; }
        print("r_k=" + ftos(rk) +
              "  E_P=" + ftos(e_p) + "  dE_P=" + ftos((e_p - e0) * 1000.0) + " mHa" +
              "  E_K=" + ftos(e_k) + "  rms=" + ftos(rms) +
              "  iters=" + itos(sparse_scc_iters(name)) +
              "  t_scc=" + ftos(t_scc * 1000.0) + " ms");
    }
}
print("=== done ===");
