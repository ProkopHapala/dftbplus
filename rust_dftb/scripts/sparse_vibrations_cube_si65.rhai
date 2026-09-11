// Gate-G/H prototype on a real nanocrystal with a full DFTB+ reference:
// cube_Si (65 atoms) — DFTB+ L1 matsci-0-3 job, n_imag=0 at its minimum.
//   Reference: /home/prokop/SIMULATIONS/SiNCs/DFTB/L1/cube_Si_matsci-0-3/
//   E_ref = -53.4520493137 Ha (detailed.out), freqs in vibrations.tag
//   (atomic units; x219474.63 -> cm^-1).
//
// n_atom=65 > 64 => GEOMETRIC mask path (first real exercise: M_HS/M_K/M_Z
// from cutoff+skin, not the full mask). Parity plan: scc+energy on the
// DFTB+ minimum, then sparse relax to ITS OWN minimum, then vibrations.
//
//   cargo run --release -p rust_dftb --bin dftb_engine -- \
//     --script rust_dftb/scripts/sparse_vibrations_cube_si65.rhai \
//     --sk-dir $RUST_DFTB_SK_DIR

const sk_dir = SK_DIR;
const ref_dir = "/home/prokop/SIMULATIONS/SiNCs/DFTB/L1/cube_Si_matsci-0-3";
const scc_max = 100;
// f32 TC2 floor ~1e-7·(N/150): at N=152 the 1e-7 tol bounces off the floor
// and trips the divergence guard. tc2=1e-6 → r_I~5e-7 → charge floor ~5e-6,
// so scc_tol=1e-5 is the tightest honest target at this size (labbook).
const scc_tol_hess = 1.0e-5;
const fd_h = 0.02;

print("=== SparseDftb vibrations: cube_Si (65 atoms, matsci) ===");
print("ref E_tot = -53.4520493137 Ha");

load_xyz("cube", ref_dir + "/relaxed.xyz");
let n_orbs = sparse_new("cube", sk_dir);
sparse_tc2_tol("cube", 1.0e-6);
print("n_atoms=" + itos(sparse_n_atoms("cube")) + " n_orbs(padded)=" + itos(n_orbs));

// Energy at the DFTB+ minimum first (direct parity, before our own relax).
let rms0 = sparse_scc("cube", scc_max, scc_tol_hess);
let e_ref_geom = sparse_eval("cube", true);
print("at DFTB+ min: E=" + ftos(e_ref_geom) + " Ha  dE=" + ftos(e_ref_geom + 53.4520493137) + "  max|F|=" + ftos(sparse_max_force("cube")) + "  rms=" + ftos(rms0) + " iters=" + itos(sparse_scc_iters("cube")));

// Relax to the SPARSE model's own minimum (skin: if FIRE moves atoms >0.5 A
// the guard fails — that's the intended loud check).
let e_relax = sparse_relax("cube", 400, 1.0e-4, 1.0e-5);
let rms = sparse_scc("cube", scc_max, scc_tol_hess);
let e = sparse_eval("cube", true);
print("relaxed: E=" + ftos(e) + " Ha  max|F|=" + ftos(sparse_max_force("cube")) + "  rms=" + ftos(rms));
save_xyz("cube", REPO_ROOT + "/debug/cube_si_sparse_relaxed.xyz");

// FD Hessian + frequencies at the sparse minimum.
let summary = sparse_vibrations("cube", fd_h, scc_tol_hess, REPO_ROOT + "/debug/sparse_vib_cube_si65_freq.txt");
print("vibrations: " + summary);

print("=== done ===");
