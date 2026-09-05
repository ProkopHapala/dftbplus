// test_graphene_sparse.rhai
//
// Test: generate graphene PAH geometries, run dense DFTB, run sparse BSR4
// purification, and compare the results.
//
// Usage:
//   dftb_engine --script test_graphene_sparse.rhai --sk-dir /path/to/mio-1-1
//
// Convention: sparse K has Tr(KS) = N_occ (no spin factor).
//             dense D = 2·C_occ·C_occ^T has Tr(DS) = 2·N_occ.
//             So we compare 2·K vs D.

const sk_dir = SK_DIR;
const acc = A_CC;
const max_iter = 200;
const tol = 1e-8;
const debug_dir = "/home/prokophapala/git/dftbplus/debug/graphene_sparse";

print("=== Graphene sparse vs dense DFTB parity test ===");
print("");

// ─── Test 1: Benzene (n=0, 6 C atoms) ─────────────────────────────
print("[1] Benzene (PAH n=0, 6 C atoms)");
let n1 = build_pah("benzene", 0, acc);
print("    atoms: " + itos(n1));
save_xyz("benzene", "/tmp/benzene.xyz");

let e1 = run_dftb_nonscc("benzene", sk_dir);
print("    dense energy: " + ftos(e1));

let r1 = run_sparse_purify("benzene", max_iter, tol);
print("    sparse R_I: " + ftos(r1));
save_convergence("benzene", debug_dir + "/benzene_conv.tsv");

let d1 = compare_density("benzene");
print("    ||2K - D||_max: " + ftos(d1));

let q1 = compare_charges("benzene");
print("    max|dq|: " + ftos(q1));
print("");

// ─── Test 2: Coronene (n=1, 24 C atoms) ──────────────────────────
print("[2] Coronene (PAH n=1, 24 C atoms)");
let n2 = build_pah("coronene", 1, acc);
print("    atoms: " + itos(n2));
save_xyz("coronene", "/tmp/coronene.xyz");

let e2 = run_dftb_nonscc("coronene", sk_dir);
print("    dense energy: " + ftos(e2));

let r2 = run_sparse_purify("coronene", max_iter, tol);
print("    sparse R_I: " + ftos(r2));
save_convergence("coronene", debug_dir + "/coronene_conv.tsv");

let d2 = compare_density("coronene");
print("    ||2K - D||_max: " + ftos(d2));

let q2 = compare_charges("coronene");
print("    max|dq|: " + ftos(q2));
print("");

// ─── Test 3: Circumcoronene (n=2, 54 C atoms) ────────────────────
print("[3] Circumcoronene (PAH n=2, 54 C atoms)");
let n3 = build_pah("circumcoronene", 2, acc);
print("    atoms: " + itos(n3));
save_xyz("circumcoronene", "/tmp/circumcoronene.xyz");

let e3 = run_dftb_nonscc("circumcoronene", sk_dir);
print("    dense energy: " + ftos(e3));

let r3 = run_sparse_purify("circumcoronene", max_iter, tol);
print("    sparse R_I: " + ftos(r3));
save_convergence("circumcoronene", debug_dir + "/circumcoronene_conv.tsv");

let d3 = compare_density("circumcoronene");
print("    ||2K - D||_max: " + ftos(d3));

let q3 = compare_charges("circumcoronene");
print("    max|dq|: " + ftos(q3));
print("");

// ─── Summary ──────────────────────────────────────────────────────
print("=== Summary ===");
print("system              atoms  ||2K-D||_max   max|dq|    R_I");
print("benzene             " + itos(n1) + "    " + ftos(d1) + "    " + ftos(q1) + "    " + ftos(r1));
print("coronene            " + itos(n2) + "    " + ftos(d2) + "    " + ftos(q2) + "    " + ftos(r2));
print("circumcoronene      " + itos(n3) + "    " + ftos(d3) + "    " + ftos(q3) + "    " + ftos(r3));
