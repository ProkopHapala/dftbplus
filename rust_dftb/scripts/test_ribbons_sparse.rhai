// test_ribbons_sparse.rhai
//
// Scaling test: H-passivated zigzag carbon ribbons of increasing length.
// Uses non-SCC (H0 only) — no convergence loop, instant.
// Exports H,S matrices + dense reference eigenvalues for the sparse Chebyshev+Ritz solver.
//
// Usage:
//   dftb_engine --script test_ribbons_sparse.rhai --sk-dir /path/to/mio-1-1

const sk_dir = SK_DIR;
const acc = A_CC;
const debug_dir = "/home/prokophapala/git/dftbplus/debug/graphene_sparse/ribbons";
const width = 4;

print("=== Ribbon scaling test (H-passivated zigzag, non-SCC) ===");
print("");

// L=4: 28 atoms, 76 orbitals
print("[1] zigzag_w4_L4 (width=4, length=4)");
let n1 = build_zigzag("zigzag_w4_L4", width, 4, true, acc);
print("    atoms: " + itos(n1));
save_xyz("zigzag_w4_L4", "/tmp/zigzag_w4_L4.xyz");
let e1 = run_dftb_nonscc("zigzag_w4_L4", sk_dir);
print("    band energy: " + ftos(e1));
print("    HOMO,LUMO,gap: " + get_homo_lumo("zigzag_w4_L4"));
save_eigenvalues("zigzag_w4_L4", debug_dir + "/zigzag_w4_L4_eigenvalues.tsv");
save_eigenvectors("zigzag_w4_L4", debug_dir + "/zigzag_w4_L4_eigenvectors.tsv");
save_hs_matrix("zigzag_w4_L4", debug_dir + "/zigzag_w4_L4_hs_matrix.tsv");
print("");

// L=8: 52 atoms, 148 orbitals
print("[2] zigzag_w4_L8 (width=4, length=8)");
let n2 = build_zigzag("zigzag_w4_L8", width, 8, true, acc);
print("    atoms: " + itos(n2));
save_xyz("zigzag_w4_L8", "/tmp/zigzag_w4_L8.xyz");
let e2 = run_dftb_nonscc("zigzag_w4_L8", sk_dir);
print("    band energy: " + ftos(e2));
print("    HOMO,LUMO,gap: " + get_homo_lumo("zigzag_w4_L8"));
save_eigenvalues("zigzag_w4_L8", debug_dir + "/zigzag_w4_L8_eigenvalues.tsv");
save_eigenvectors("zigzag_w4_L8", debug_dir + "/zigzag_w4_L8_eigenvectors.tsv");
save_hs_matrix("zigzag_w4_L8", debug_dir + "/zigzag_w4_L8_hs_matrix.tsv");
print("");

// L=16: 100 atoms, 292 orbitals
print("[3] zigzag_w4_L16 (width=4, length=16)");
let n3 = build_zigzag("zigzag_w4_L16", width, 16, true, acc);
print("    atoms: " + itos(n3));
save_xyz("zigzag_w4_L16", "/tmp/zigzag_w4_L16.xyz");
let e3 = run_dftb_nonscc("zigzag_w4_L16", sk_dir);
print("    band energy: " + ftos(e3));
print("    HOMO,LUMO,gap: " + get_homo_lumo("zigzag_w4_L16"));
save_eigenvalues("zigzag_w4_L16", debug_dir + "/zigzag_w4_L16_eigenvalues.tsv");
save_eigenvectors("zigzag_w4_L16", debug_dir + "/zigzag_w4_L16_eigenvectors.tsv");
save_hs_matrix("zigzag_w4_L16", debug_dir + "/zigzag_w4_L16_hs_matrix.tsv");
print("");

// L=32: 196 atoms, 580 orbitals
print("[4] zigzag_w4_L32 (width=4, length=32)");
let n4 = build_zigzag("zigzag_w4_L32", width, 32, true, acc);
print("    atoms: " + itos(n4));
save_xyz("zigzag_w4_L32", "/tmp/zigzag_w4_L32.xyz");
let e4 = run_dftb_nonscc("zigzag_w4_L32", sk_dir);
print("    band energy: " + ftos(e4));
print("    HOMO,LUMO,gap: " + get_homo_lumo("zigzag_w4_L32"));
save_eigenvalues("zigzag_w4_L32", debug_dir + "/zigzag_w4_L32_eigenvalues.tsv");
save_eigenvectors("zigzag_w4_L32", debug_dir + "/zigzag_w4_L32_eigenvectors.tsv");
save_hs_matrix("zigzag_w4_L32", debug_dir + "/zigzag_w4_L32_hs_matrix.tsv");
print("");

// L=64: 388 atoms, 1156 orbitals
print("[5] zigzag_w4_L64 (width=4, length=64)");
let n5 = build_zigzag("zigzag_w4_L64", width, 64, true, acc);
print("    atoms: " + itos(n5));
save_xyz("zigzag_w4_L64", "/tmp/zigzag_w4_L64.xyz");
let e5 = run_dftb_nonscc("zigzag_w4_L64", sk_dir);
print("    band energy: " + ftos(e5));
print("    HOMO,LUMO,gap: " + get_homo_lumo("zigzag_w4_L64"));
save_eigenvalues("zigzag_w4_L64", debug_dir + "/zigzag_w4_L64_eigenvalues.tsv");
save_eigenvectors("zigzag_w4_L64", debug_dir + "/zigzag_w4_L64_eigenvectors.tsv");
save_hs_matrix("zigzag_w4_L64", debug_dir + "/zigzag_w4_L64_hs_matrix.tsv");
print("");

print("=== Summary ===");
print("TSV files saved to: " + debug_dir);
