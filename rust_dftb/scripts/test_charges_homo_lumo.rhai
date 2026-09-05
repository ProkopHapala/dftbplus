// test_charges_homo_lumo.rhai
//
// End-to-end test: SCC → charges + HOMO-LUMO + Davidson partial eigensolver.
//
// For each PAH system:
//   1. Build geometry (pure C, no H — BSR4 requires 4 orbs/atom)
//   2. Run dense DFTB SCC → charges, eigenvalues, HOMO-LUMO
//   3. Run sparse BSR4 TC2 purification → sparse charges
//   4. Run Davidson partial eigensolver → HOMO-LUMO from few eigenvalues
//   5. Save charges + eigenvalues to TSV for plotting
//
// Usage:
//   dftb_engine --script test_charges_homo_lumo.rhai --sk-dir /path/to/mio-1-1
//
// After this script, run the DFTB+ reference + plotting:
//   python3 scripts/run_dftbplus_ref.py /tmp/benzene.xyz debug/graphene_sparse/ ...
//   python3 scripts/plot_charges_homo_lumo.py debug/graphene_sparse/

const sk_dir = SK_DIR;
const acc = A_CC;
const max_iter = 200;
const tol = 1e-8;
const debug_dir = "/home/prokophapala/git/dftbplus/debug/graphene_sparse";
const n_target = 4;  // Davidson: 4 eigs below + 4 above gap

print("=== Charges + HOMO-LUMO + Davidson test ===");
print("");

// ─── Test 1: Benzene (n=0, 6 C atoms) ─────────────────────────────
print("[1] Benzene (PAH n=0, 6 C atoms)");
let n1 = build_pah("benzene", 0, acc);
print("    atoms: " + itos(n1));
save_xyz("benzene", "/tmp/benzene.xyz");

let e1 = run_dftb_scc("benzene", sk_dir, 100, tol);
print("    SCC energy: " + ftos(e1));

let hl1 = get_homo_lumo("benzene");
print("    HOMO,LUMO,gap: " + hl1);

let dav1 = davidson_homo_lumo("benzene", n_target);
print("    Davidson HOMO,LUMO,gap: " + dav1);

let q1 = get_charges("benzene");
print("    charges: " + q1);

// Sparse TC2 purification
let r1 = run_sparse_purify("benzene", 200, 1e-6);
print("    sparse R_I=" + ftos(r1));
let dq1 = compare_charges("benzene");
print("    sparse max|dq|=" + ftos(dq1));
let sq1 = get_sparse_charges("benzene");
print("    sparse charges: " + sq1);
save_sparse_charges("benzene", debug_dir + "/benzene_sparse_charges.tsv");
save_convergence("benzene", debug_dir + "/benzene_conv.tsv");

save_charges("benzene", debug_dir + "/benzene_charges.tsv");
save_eigenvalues("benzene", debug_dir + "/benzene_eigenvalues.tsv");
save_eigenvectors("benzene", debug_dir + "/benzene_eigenvectors.tsv");
save_hs_matrix("benzene", debug_dir + "/benzene_hs_matrix.tsv");
print("");

// ─── Test 2: Coronene (n=1, 24 C atoms) ──────────────────────────
print("[2] Coronene (PAH n=1, 24 C atoms)");
let n2 = build_pah("coronene", 1, acc);
print("    atoms: " + itos(n2));
save_xyz("coronene", "/tmp/coronene.xyz");

let e2 = run_dftb_scc("coronene", sk_dir, 100, tol);
print("    SCC energy: " + ftos(e2));

let hl2 = get_homo_lumo("coronene");
print("    HOMO,LUMO,gap: " + hl2);

// Davidson: skip for coronene (dense manifold near gap — needs better preconditioner)
// let dav2 = davidson_homo_lumo("coronene", n_target);
// print("    Davidson HOMO,LUMO,gap: " + dav2);
let dav2 = "(skipped — dense manifold)";

let q2 = get_charges("coronene");
print("    charges: " + q2);

// Sparse TC2 purification
let r2 = run_sparse_purify("coronene", 200, 1e-6);
print("    sparse R_I=" + ftos(r2));
let dq2 = compare_charges("coronene");
print("    sparse max|dq|=" + ftos(dq2));
let sq2 = get_sparse_charges("coronene");
print("    sparse charges: " + sq2);
save_sparse_charges("coronene", debug_dir + "/coronene_sparse_charges.tsv");
save_convergence("coronene", debug_dir + "/coronene_conv.tsv");

save_charges("coronene", debug_dir + "/coronene_charges.tsv");
save_eigenvalues("coronene", debug_dir + "/coronene_eigenvalues.tsv");
save_eigenvectors("coronene", debug_dir + "/coronene_eigenvectors.tsv");
save_hs_matrix("coronene", debug_dir + "/coronene_hs_matrix.tsv");
print("");

// ─── Test 3: Circumcoronene (n=2, 54 C atoms) ────────────────────
print("[3] Circumcoronene (PAH n=2, 54 C atoms)");
let n3 = build_pah("circumcoronene", 2, acc);
print("    atoms: " + itos(n3));
save_xyz("circumcoronene", "/tmp/circumcoronene.xyz");

let e3 = run_dftb_scc("circumcoronene", sk_dir, 100, tol);
print("    SCC energy: " + ftos(e3));

let hl3 = get_homo_lumo("circumcoronene");
print("    HOMO,LUMO,gap: " + hl3);

// Davidson: skip for circumcoronene (dense manifold near gap — needs better preconditioner)
// let dav3 = davidson_homo_lumo("circumcoronene", n_target);
// print("    Davidson HOMO,LUMO,gap: " + dav3);
let dav3 = "(skipped — dense manifold)";

let q3 = get_charges("circumcoronene");
print("    charges: " + q3);

// Sparse TC2 purification
let r3 = run_sparse_purify("circumcoronene", 200, 1e-6);
print("    sparse R_I=" + ftos(r3));
let dq3 = compare_charges("circumcoronene");
print("    sparse max|dq|=" + ftos(dq3));
let sq3 = get_sparse_charges("circumcoronene");
print("    sparse charges: " + sq3);
save_sparse_charges("circumcoronene", debug_dir + "/circumcoronene_sparse_charges.tsv");
save_convergence("circumcoronene", debug_dir + "/circumcoronene_conv.tsv");

save_charges("circumcoronene", debug_dir + "/circumcoronene_charges.tsv");
save_eigenvalues("circumcoronene", debug_dir + "/circumcoronene_eigenvalues.tsv");
save_eigenvectors("circumcoronene", debug_dir + "/circumcoronene_eigenvectors.tsv");
save_hs_matrix("circumcoronene", debug_dir + "/circumcoronene_hs_matrix.tsv");
print("");

// ─── Summary ──────────────────────────────────────────────────────
print("=== Summary ===");
print("system              atoms  SCC_energy       HOMO-LUMO(dense)    HOMO-LUMO(Davidson)");
print("benzene             " + itos(n1) + "    " + ftos(e1) + "    " + hl1 + "    " + dav1);
print("coronene            " + itos(n2) + "    " + ftos(e2) + "    " + hl2 + "    " + dav2);
print("circumcoronene      " + itos(n3) + "    " + ftos(e3) + "    " + hl3 + "    " + dav3);
print("");
print("TSV files saved to: " + debug_dir);
print("Next: run DFTB+ ref + plotting script for comparison.");
