//! dftb_engine — Rhai-scriptable DFTB + sparse BSR4 engine.
//!
//! This binary loads a Rhai script that orchestrates DFTB calculations.
//! The engine exposes native functions to Rhai for:
//!   - Graphene geometry generation (PAH, flake, ribbon)
//!   - Dense DFTB SCC calculation (CPU, returns H/S/density/charges/energy)
//!   - Sparse BSR4 purification (GPU, Z→K₀→TC2→K)
//!   - Result comparison and diagnostics
//!   - Charge/eigenvalue access: `get_charges`, `get_eigenvalues`, `save_charges`,
//!     `save_eigenvalues`, `save_eigenvectors`, `get_sparse_charges`, `save_sparse_charges`
//!   - Frontier orbitals: `davidson_homo_lumo(name, n_target)` — partial
//!     generalized eigensolver (see `sparse::davidson`)
//!   - Convergence history: `save_convergence(name, path)`
//!   - Persistent GPU DFTB: `gpu_new` / `gpu_scc` / `gpu_eval` (`GpuDftb`)
//!   - Persistent sparse DFTB: `sparse_new` / `sparse_scc` / `sparse_eval` (`SparseDftb`)
//!
//! Usage:
//!   dftb_engine --script test_graphene.rhai
//!   dftb_engine --script test_gpu_dftb_molecules.rhai --sk-dir /path/to/mio-1-1
//!   dftb_engine --script test_sparse_dftb_sih4.rhai --sk-dir /path/to/matsci-0-3

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rhai::{Array, Dynamic, Engine, Scope, INT};
use rust_dftb::geometry::{self, Element, FlakeShape, NanoStructure, A_CC};
use rust_dftb::methods::sparse::gpu_sparse::{
    GpuBsrMatrix, GpuBsrStructure, SparsePurifyWorkspace,
};
use rust_dftb::methods::sparse::{
    build_full_mask, build_geometric_mask, Bsr4Matrix, GeomStep, SparseBsr4Config, SparseBsr4Gpu,
    SparseDftb, SparseDftbConfig, SparsePerfStats,
};
use rust_dftb::qmqm::{GpuDftb, GpuDftbEval};
use rust_dftb::{
    load_sk_for_species, parse_species, parse_xyz, HamiltonianBuilder, SccResult, SystemContext,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

const ANG2BOHR: f64 = 1.889_726_133;

// ─── Shared state (rhai functions can't easily pass large structs, so
//     we use a global registry keyed by string names) ─────────────────

#[derive(Default)]
struct State {
    geometries: HashMap<String, NanoStructure>,
    scc_results: HashMap<String, SccResult>,
    sparse_results: HashMap<String, SparseResult>,
}

#[derive(Clone)]
struct SparseResult {
    k_dense: Vec<f32>,
    n_atom: usize,
    n_occ: f32,
    r_i: f32,
    r_h: f32,
    tr_ks: f32,
    mulliken: Vec<f32>,
    iters: usize,
    /// TC2 convergence history: (iter, R_I, Tr(KS))
    history: Vec<(usize, f32, f32)>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with_state<F, R>(f: F) -> R
where
    F: FnOnce(&mut State) -> R,
{
    let mut guard = STATE.lock().unwrap();
    if guard.is_none() {
        *guard = Some(State::default());
    }
    f(guard.as_mut().unwrap())
}

// ─── Rhai-exposed functions ─────────────────────────────────────────

/// Generate a PAH geometry and store it under `name`.
/// Returns the number of atoms.
fn rhai_build_pah(name: &str, shells: INT, acc: f64) -> INT {
    let st = geometry::build_pah(shells as usize, acc);
    let n = st.natom() as INT;
    with_state(|s| {
        s.geometries.insert(name.to_string(), st);
    });
    n
}

/// Generate a graphene flake and store it.
fn rhai_build_flake(name: &str, radius: f64, shape: &str, passivate: bool, acc: f64) -> INT {
    let sh = if shape == "hex" {
        FlakeShape::Hex
    } else {
        FlakeShape::Circle
    };
    let st = geometry::build_flake(radius, sh, passivate, acc);
    let n = st.natom() as INT;
    with_state(|s| {
        s.geometries.insert(name.to_string(), st);
    });
    n
}

/// Generate a zigzag ribbon and store it.
fn rhai_build_zigzag(name: &str, width: INT, length: INT, passivate: bool, acc: f64) -> INT {
    let st = geometry::build_zigzag_ribbon(width as usize, length as usize, passivate, false, acc);
    let n = st.natom() as INT;
    with_state(|s| {
        s.geometries.insert(name.to_string(), st);
    });
    n
}

/// Save a stored geometry to an XYZ file.
fn rhai_save_xyz(name: &str, path: &str) -> bool {
    with_state(|s| {
        if let Some(st) = s.geometries.get(name) {
            std::fs::write(path, st.to_xyz()).is_ok()
        } else {
            false
        }
    })
}

/// Run dense DFTB SCC on a stored geometry. Stores the result under `name`.
/// Returns the total SCC energy, or NaN on error.
fn rhai_run_dftb_scc(name: &str, sk_dir: &str, max_iter: INT, tol: f64) -> Dynamic {
    let result = with_state(|s| {
        let st = s.geometries.get(name)?.clone();
        let species: Vec<String> = st.elements.iter().map(|e| e.symbol().to_string()).collect();
        let coords: Vec<[f64; 3]> = st.positions.clone();
        Some((species, coords))
    });

    let Some((species, coords)) = result else {
        return Dynamic::from_float(f64::NAN);
    };

    // Dense SCC supports arbitrary species (H, C, N, O, ...).
    // The BSR4 4-orbital restriction only applies to the sparse TC2 path.
    eprintln!("Loading SK tables from {sk_dir} ...");
    let sk = match load_sk_for_species(sk_dir, &species) {
        Ok(sk) => sk,
        Err(e) => {
            eprintln!("ERROR loading SK: {e}");
            return Dynamic::from_float(f64::NAN);
        }
    };

    let builder = HamiltonianBuilder::new(sk);
    eprintln!(
        "[SCC] Starting '{name}': {} atoms, {} orbitals, max_iter={max_iter}, tol={tol}",
        species.len(),
        species.len() * 4
    );
    eprintln!("Running dense DFTB SCC (max_iter={max_iter}, tol={tol}) ...");
    match builder.build_scc(&species, &coords, max_iter as usize, tol) {
        Ok(scc) => {
            let energy = scc.energy;
            let n_iter = scc.n_iter;
            eprintln!("  SCC converged in {n_iter} iters, E = {energy:.10} Ha");
            eprintln!(
                "  charges: {:?}",
                scc.charges
                    .iter()
                    .map(|q| format!("{q:.4}"))
                    .collect::<Vec<_>>()
            );
            with_state(|s| {
                s.scc_results.insert(name.to_string(), scc);
            });
            Dynamic::from_float(energy)
        }
        Err(e) => {
            eprintln!("ERROR SCC: {e}");
            Dynamic::from_float(f64::NAN)
        }
    }
}

/// Run dense DFTB non-SCC (H0 only) on a stored geometry.
fn rhai_run_dftb_nonscc(name: &str, sk_dir: &str) -> Dynamic {
    let result = with_state(|s| {
        let st = s.geometries.get(name)?.clone();
        let species: Vec<String> = st.elements.iter().map(|e| e.symbol().to_string()).collect();
        let coords: Vec<[f64; 3]> = st.positions.clone();
        Some((species, coords))
    });

    let Some((species, coords)) = result else {
        return Dynamic::from_float(f64::NAN);
    };

    eprintln!("Loading SK tables from {sk_dir} ...");
    let sk = match load_sk_for_species(sk_dir, &species) {
        Ok(sk) => sk,
        Err(e) => {
            eprintln!("ERROR loading SK: {e}");
            return Dynamic::from_float(f64::NAN);
        }
    };

    let builder = HamiltonianBuilder::new(sk);
    eprintln!(
        "[non-SCC] '{name}': {} atoms, building H0/S ...",
        species.len()
    );
    match builder.build_non_scc(&species, &coords) {
        Ok(ham) => {
            let n_orbs = ham.h0.nrows();
            eprintln!("  H0: {n_orbs}×{n_orbs}");

            // Build SystemContext for per-atom orbital counts (handles H with 1 orb, C with 4)
            let ctx = match SystemContext::from_sk_data(&builder.sk, &species) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("ERROR SystemContext: {e}");
                    return Dynamic::from_float(f64::NAN);
                }
            };
            let atom_n_orb = ctx.atom_n_orb.clone();
            let atom_orb_off = ctx.atom_orb_off.clone();

            // Solve generalized eigenproblem H0 c = S c ε (dense reference)
            let t0 = std::time::Instant::now();
            eprintln!("  [dense] S eigendecomp ...");
            let s_inv_sqrt = {
                let se = SymmetricEigen::new(ham.s.clone());
                let mut d = DMatrix::<f64>::zeros(n_orbs, n_orbs);
                for i in 0..n_orbs {
                    d[(i, i)] = 1.0 / se.eigenvalues[i].max(1e-12).sqrt();
                }
                &se.eigenvectors * &d * se.eigenvectors.transpose()
            };
            eprintln!("  [dense] H' = S^(-1/2) H S^(-1/2) eigendecomp ...");
            let h_orth = &s_inv_sqrt * &ham.h0 * &s_inv_sqrt;
            let he = SymmetricEigen::new(h_orth);
            eprintln!("  [dense] done in {:.2}s", t0.elapsed().as_secs_f64());
            // Sort ascending
            let mut idx: Vec<usize> = (0..n_orbs).collect();
            idx.sort_by(|&i, &j| he.eigenvalues[i].partial_cmp(&he.eigenvalues[j]).unwrap());

            // Count electrons: C has 4 valence electrons
            let n_electrons: f64 = species
                .iter()
                .map(|s| match s.as_str() {
                    "C" => 4.0,
                    "N" => 5.0,
                    "O" => 6.0,
                    "B" => 3.0,
                    _ => 1.0,
                })
                .sum();
            let n_occ = (n_electrons / 2.0).round() as usize;
            eprintln!("  n_electrons={n_electrons}, n_occ={n_occ}");

            let v_sorted = he.eigenvectors.select_columns(&idx);
            let c = &s_inv_sqrt * &v_sorted;
            let c_occ = c.columns(0, n_occ);
            let density = &c_occ * c_occ.transpose() * 2.0;

            let e_band = (&density * &ham.h0).trace();
            eprintln!("  band energy = {e_band:.10} Ha");

            // Mulliken charges — per-atom orbital count (H=1, C=4)
            let ds = &density * &ham.s;
            let mut charges = Vec::with_capacity(species.len());
            for i in 0..species.len() {
                let n_orb_i = atom_n_orb[i] as usize;
                let off = atom_orb_off[i] as usize;
                let q = (0..n_orb_i).map(|k| ds[(off + k, off + k)]).sum::<f64>();
                charges.push(q);
            }
            eprintln!(
                "  Mulliken charges: {:?}",
                charges
                    .iter()
                    .map(|q| format!("{q:.4}"))
                    .collect::<Vec<_>>()
            );

            let eigs_sorted: DVector<f64> =
                DVector::from_iterator(n_orbs, idx.iter().map(|&i| he.eigenvalues[i]));
            let scc = SccResult {
                h0: ham.h0.clone(),
                h_scc: ham.h0.clone(), // non-SCC: H_scc = H0
                s: ham.s.clone(),
                density,
                eigenvalues: eigs_sorted,
                eigenvectors: c.clone(),
                charges: charges.clone(),
                q0: charges, // placeholder
                energy: e_band,
                n_iter: 0,
            };
            with_state(|s| {
                s.scc_results.insert(name.to_string(), scc);
            });
            Dynamic::from_float(e_band)
        }
        Err(e) => {
            eprintln!("ERROR building H0: {e}");
            Dynamic::from_float(f64::NAN)
        }
    }
}

/// Run sparse BSR4 purification on a stored DFTB result.
/// Uses the H and S from the dense DFTB calculation.
/// Stores the sparse result under `name`.
///
/// **Full mask** — every block (i,j) exists. Use for small-system validation
/// against the dense reference. For large systems use
/// `rhai_run_sparse_purify_geom` with a geometric mask instead.
fn rhai_run_sparse_purify(name: &str, max_iter: INT, tol: f64) -> Dynamic {
    let payload = with_state(|s| {
        let scc = s.scc_results.get(name)?.clone();
        let st = s.geometries.get(name)?.clone();
        Some((scc, st))
    });

    let Some((scc, st)) = payload else {
        eprintln!(
            "ERROR: no DFTB result found for '{name}'. Run run_dftb_scc or run_dftb_nonscc first."
        );
        return Dynamic::from_float(f64::NAN);
    };

    let n_atom = st.natom();
    let n_orbs = n_atom * 4; // BSR4: 4 orbitals per atom
    if max_iter <= 0 || !tol.is_finite() || tol <= 0.0 {
        panic!("sparse purification requires max_iter > 0 and finite tol > 0: max_iter={max_iter}, tol={tol}");
    }
    if scc.h0.nrows() != n_orbs {
        eprintln!(
            "ERROR: H0 size {} != expected {} (n_atom×4). BSR4 requires sp basis (4 orbs/atom).",
            scc.h0.nrows(),
            n_orbs
        );
        return Dynamic::from_float(f64::NAN);
    }

    eprintln!("Sparse BSR4 purification on '{name}' ({n_atom} atoms, {n_orbs} orbitals) ...");

    // Convert dense H, S to BSR4 format with full mask (validation path only;
    // production uses run_sparse_purify_geom with geometric mask)
    let mask = build_full_mask(n_atom);
    let h_bsr = dense_to_bsr4(&scc.h_scc, n_atom, &mask);
    let s_bsr = dense_to_bsr4(&scc.s, n_atom, &mask);

    // Count occupied orbitals
    let n_electrons: f64 = st.elements.iter().map(|e| e.valence_electrons()).sum();
    let n_occ = (n_electrons / 2.0) as f32;
    eprintln!("  n_occ = {n_occ}");

    // Init GPU
    let config = SparseBsr4Config::default();
    let gpu = match SparseBsr4Gpu::new(config) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("ERROR GPU init: {e}");
            return Dynamic::from_float(f64::NAN);
        }
    };

    // The legacy host-roundtrip methods remain available on SparseBsr4Gpu as
    // the reference path; production uses the resident APIs below.
    // 1. Z ≈ S⁻¹ (resident Newton–Schulz)
    eprintln!("  Newton-Schulz Z ≈ S⁻¹ ...");
    let k_struct = match GpuBsrStructure::new(&gpu, n_atom, &mask) {
        Ok(s) => Arc::new(s),
        Err(e) => panic!("ERROR building resident K structure for {n_atom} atoms: {e}"),
    };
    let t_struct = match GpuBsrStructure::new(&gpu, n_atom, &mask) {
        Ok(s) => Arc::new(s),
        Err(e) => panic!("ERROR building resident T structure for {n_atom} atoms: {e}"),
    };
    let s_struct = match GpuBsrStructure::new(
        &gpu,
        n_atom,
        &(s_bsr.row_ptr.clone(), s_bsr.col_idx.clone()),
    ) {
        Ok(s) => Arc::new(s),
        Err(e) => panic!("ERROR building resident S structure for {n_atom} atoms: {e}"),
    };
    let s_dev = match gpu.buf_f32(&s_bsr.values) {
        Ok(values) => GpuBsrMatrix {
            struct_: s_struct,
            values,
        },
        Err(e) => panic!("ERROR uploading resident S values for {n_atom} atoms: {e}"),
    };
    let _ = (&k_struct, &t_struct, &s_dev);
    // Host NS is the physics path until device NS is reverified (second review §3.6).
    let (z, r_z, z_iters) = match gpu.newton_schulz_inverse(&s_bsr, &mask, &mask, 50, 1e-5, 5) {
        Ok(r) => r,
        Err(e) => panic!("ERROR Newton-Schulz (host): {e}"),
    };
    // let (z, r_z, z_iters) = match gpu.newton_schulz_inverse_dev(&s_dev, &k_struct, &t_struct, 30, 1e-4, 3) {
    //     Ok(r) => r,
    //     Err(e) => panic!("ERROR Newton-Schulz: {e}"),
    // };
    eprintln!("    Z: {z_iters} iters, R_Z = {r_z:e}");
    if !r_z.is_finite() || r_z > 1e-4 {
        panic!(
            "Newton-Schulz failed to converge: R_Z={r_z:e}, iterations={z_iters}, tolerance=1e-4"
        );
    }

    // 2. Spectral bounds
    let (emin, emax) = match gpu.spectral_bounds(&h_bsr, &z, &mask, 0.1) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ERROR spectral bounds: {e}");
            return Dynamic::from_float(f64::NAN);
        }
    };
    eprintln!("    spectral bounds: emin={emin:.4} emax={emax:.4}");

    // 3. K₀
    let k0 = match gpu.build_k0(&h_bsr, &s_bsr, &z, &mask, &mask, emin, emax) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("ERROR K0: {e}");
            return Dynamic::from_float(f64::NAN);
        }
    };

    // 4. TC2 purification
    eprintln!("  TC2 purification (max_iter={max_iter}, tol={tol}) ...");
    let n_orb_atom: Vec<u8> = st
        .elements
        .iter()
        .map(|e| if e.symbol() == "H" { 1 } else { 4 })
        .collect();
    let mut workspace =
        match SparsePurifyWorkspace::new(gpu, &k0, &s_bsr, &mask, &mask, &n_orb_atom, n_occ) {
            Ok(ws) => ws,
            Err(e) => panic!("ERROR creating resident TC2 workspace: {e}"),
        };
    let (k_final, r_i, tr_ks, tc2_iters, history) =
        match workspace.tc2_purify_dev(max_iter as usize, tol as f32, 1) {
            Ok(r) => r,
            Err(e) => panic!("ERROR TC2: {e}"),
        };
    eprintln!("    TC2: {tc2_iters} iters, R_I={r_i:e}, Tr(KS)={tr_ks:.6}");
    if !r_i.is_finite() || r_i > tol as f32 || !tr_ks.is_finite() {
        panic!("TC2 failed to converge: R_I={r_i:e}, Tr(KS)={tr_ks:.8}, iterations={tc2_iters}, tolerance={tol:e}");
    }

    // 5. R_H = ||HKS - SKH||
    let r_h = match workspace
        .gpu()
        .hamiltonian_residual(&h_bsr, &k_final, &s_bsr, &mask)
    {
        Ok(r) => r,
        Err(e) => panic!("ERROR R_H: {e}"),
    };
    let r_h_norm = r_h / (n_orbs as f32).sqrt();
    eprintln!("    R_H = {r_h:e} (normalized {r_h_norm:e})");
    if !r_h_norm.is_finite() {
        panic!("Hamiltonian residual is non-finite: R_H={r_h:e}, normalized={r_h_norm:e}");
    }

    // 6. Mulliken charges from sparse K (physical lanes only, R14)
    let (mulliken, q_dum) = match workspace.mulliken_dev() {
        Ok(m) => m,
        Err(e) => panic!("ERROR Mulliken: {e}"),
    };
    let qd_max: f32 = q_dum.iter().map(|x| x.abs()).fold(0.0, f32::max);
    eprintln!(
        "    sparse Mulliken: {:?}  (dummy occ max={qd_max:.3e})",
        mulliken
            .iter()
            .map(|q| format!("{q:.4}"))
            .collect::<Vec<_>>()
    );

    let k_dense = k_final.to_dense();
    let result = SparseResult {
        k_dense,
        n_atom,
        n_occ,
        r_i,
        r_h: r_h_norm,
        tr_ks,
        mulliken,
        iters: tc2_iters,
        history,
    };
    with_state(|s| {
        s.sparse_results.insert(name.to_string(), result);
    });

    Dynamic::from_float(r_i as f64)
}

/// Run sparse BSR4 purification with a **geometric mask** (not full mask).
///
/// Uses `build_geometric_mask(pos, cutoff)` so only blocks within `cutoff`
/// distance are stored — O(N·n_neigh) instead of O(N²). This is the
/// production path for large systems; `run_sparse_purify` (full mask) remains
/// as a dense-reference validation path for small systems.
///
/// `r_max` is the mask cutoff in Å (must exceed the SK interaction range).
fn rhai_run_sparse_purify_geom(name: &str, r_max: f64, max_iter: INT, tol: f64) -> Dynamic {
    let payload = with_state(|s| {
        let scc = s.scc_results.get(name)?.clone();
        let st = s.geometries.get(name)?.clone();
        Some((scc, st))
    });

    let Some((scc, st)) = payload else {
        eprintln!(
            "ERROR: no DFTB result found for '{name}'. Run run_dftb_scc or run_dftb_nonscc first."
        );
        return Dynamic::from_float(f64::NAN);
    };

    let n_atom = st.natom();
    let n_orbs = n_atom * 4;
    if max_iter <= 0 || !tol.is_finite() || tol <= 0.0 {
        panic!("sparse purification requires max_iter > 0 and finite tol > 0: max_iter={max_iter}, tol={tol}");
    }
    if !r_max.is_finite() || r_max <= 0.0 {
        panic!("r_max must be finite and positive, got {r_max}");
    }
    if scc.h0.nrows() != n_orbs {
        eprintln!(
            "ERROR: H0 size {} != expected {} (n_atom×4). BSR4 requires sp basis (4 orbs/atom).",
            scc.h0.nrows(),
            n_orbs
        );
        return Dynamic::from_float(f64::NAN);
    }

    eprintln!("Sparse BSR4 purification (geometric mask, r_max={r_max} Å) on '{name}' ({n_atom} atoms, {n_orbs} orbitals) ...");
    let t_total_start = Instant::now();

    // Geometric mask: only blocks within r_max. O(N²) construction for now
    // (P0e will add cell list); the mask itself is O(N·n_neigh).
    let t0 = Instant::now();
    let mask = build_geometric_mask(&st.positions, r_max);
    let nblock = mask.1.len();
    let fill = nblock as f64 / (n_atom * n_atom) as f64;
    eprintln!(
        "  geometric mask: {nblock} blocks, fill ratio = {fill:.4} ({n_atom}² = {})",
        n_atom * n_atom
    );
    if nblock == 0 {
        panic!("geometric mask is empty — r_max={r_max} too small or positions invalid");
    }
    let t_hs = t0.elapsed();

    let h_bsr = dense_to_bsr4(&scc.h_scc, n_atom, &mask);
    let s_bsr = dense_to_bsr4(&scc.s, n_atom, &mask);

    let n_electrons: f64 = st.elements.iter().map(|e| e.valence_electrons()).sum();
    let n_occ = (n_electrons / 2.0) as f32;
    eprintln!("  n_occ = {n_occ}");

    let config = SparseBsr4Config::default();
    let gpu = match SparseBsr4Gpu::new(config) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("ERROR GPU init: {e}");
            return Dynamic::from_float(f64::NAN);
        }
    };

    eprintln!("  Newton-Schulz Z ≈ S⁻¹ ...");
    let t0 = Instant::now();
    let k_struct = match GpuBsrStructure::new(&gpu, n_atom, &mask) {
        Ok(s) => Arc::new(s),
        Err(e) => panic!("ERROR building resident K structure: {e}"),
    };
    let t_struct = match GpuBsrStructure::new(&gpu, n_atom, &mask) {
        Ok(s) => Arc::new(s),
        Err(e) => panic!("ERROR building resident T structure: {e}"),
    };
    let s_struct = match GpuBsrStructure::new(
        &gpu,
        n_atom,
        &(s_bsr.row_ptr.clone(), s_bsr.col_idx.clone()),
    ) {
        Ok(s) => Arc::new(s),
        Err(e) => panic!("ERROR building resident S structure: {e}"),
    };
    let s_dev = match gpu.buf_f32(&s_bsr.values) {
        Ok(values) => GpuBsrMatrix {
            struct_: s_struct,
            values,
        },
        Err(e) => panic!("ERROR uploading resident S values: {e}"),
    };
    let _ = (&k_struct, &t_struct, &s_dev);
    let (z, r_z, z_iters) = match gpu.newton_schulz_inverse(&s_bsr, &mask, &mask, 50, 1e-5, 5) {
        Ok(r) => r,
        Err(e) => panic!("ERROR Newton-Schulz (host): {e}"),
    };
    // let (z, r_z, z_iters) = match gpu.newton_schulz_inverse_dev(&s_dev, &k_struct, &t_struct, 30, 1e-4, 3) {
    //     Ok(r) => r,
    //     Err(e) => panic!("ERROR Newton-Schulz: {e}"),
    // };
    let t_ns = t0.elapsed();
    eprintln!("    Z: {z_iters} iters, R_Z = {r_z:e}");
    if !r_z.is_finite() || r_z > 1e-4 {
        panic!(
            "Newton-Schulz failed to converge: R_Z={r_z:e}, iterations={z_iters}, tolerance=1e-4"
        );
    }

    let (emin, emax) = match gpu.spectral_bounds(&h_bsr, &z, &mask, 0.1) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ERROR spectral bounds: {e}");
            return Dynamic::from_float(f64::NAN);
        }
    };
    eprintln!("    spectral bounds: emin={emin:.4} emax={emax:.4}");

    let k0 = match gpu.build_k0(&h_bsr, &s_bsr, &z, &mask, &mask, emin, emax) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("ERROR K0: {e}");
            return Dynamic::from_float(f64::NAN);
        }
    };

    eprintln!("  TC2 purification (max_iter={max_iter}, tol={tol}) ...");
    let t0 = Instant::now();
    let n_orb_atom: Vec<u8> = st
        .elements
        .iter()
        .map(|e| if e.symbol() == "H" { 1 } else { 4 })
        .collect();
    let mut workspace =
        match SparsePurifyWorkspace::new(gpu, &k0, &s_bsr, &mask, &mask, &n_orb_atom, n_occ) {
            Ok(ws) => ws,
            Err(e) => panic!("ERROR creating resident TC2 workspace: {e}"),
        };
    let (k_final, r_i, tr_ks, tc2_iters, history) =
        match workspace.tc2_purify_dev(max_iter as usize, tol as f32, 1) {
            Ok(r) => r,
            Err(e) => panic!("ERROR TC2: {e}"),
        };
    let t_tc2 = t0.elapsed();
    eprintln!("    TC2: {tc2_iters} iters, R_I={r_i:e}, Tr(KS)={tr_ks:.6}");
    if !r_i.is_finite() || r_i > tol as f32 || !tr_ks.is_finite() {
        panic!("TC2 failed to converge: R_I={r_i:e}, Tr(KS)={tr_ks:.8}, iterations={tc2_iters}, tolerance={tol:e}");
    }

    let r_h = match workspace
        .gpu()
        .hamiltonian_residual(&h_bsr, &k_final, &s_bsr, &mask)
    {
        Ok(r) => r,
        Err(e) => panic!("ERROR R_H: {e}"),
    };
    let r_h_norm = r_h / (n_orbs as f32).sqrt();
    eprintln!("    R_H = {r_h:e} (normalized {r_h_norm:e})");
    if !r_h_norm.is_finite() {
        panic!("Hamiltonian residual is non-finite: R_H={r_h:e}, normalized={r_h_norm:e}");
    }

    let (mulliken, q_dum) = match workspace.mulliken_dev() {
        Ok(m) => m,
        Err(e) => panic!("ERROR Mulliken: {e}"),
    };
    let qd_max: f32 = q_dum.iter().map(|x| x.abs()).fold(0.0, f32::max);
    eprintln!(
        "    sparse Mulliken: {:?}  (dummy occ max={qd_max:.3e})",
        mulliken
            .iter()
            .map(|q| format!("{q:.4}"))
            .collect::<Vec<_>>()
    );

    let t_total = t_total_start.elapsed();
    let stats = SparsePerfStats {
        n_atom,
        nnz_hs: nblock,
        nnz_k: nblock,
        nnz_z: nblock,
        t_hs: t_hs.as_secs_f64(),
        t_ns: t_ns.as_secs_f64(),
        t_tc2: t_tc2.as_secs_f64(),
        t_total: t_total.as_secs_f64(),
        ..Default::default()
    };
    stats.print_audit();

    let k_dense = k_final.to_dense();
    let result = SparseResult {
        k_dense,
        n_atom,
        n_occ,
        r_i,
        r_h: r_h_norm,
        tr_ks,
        mulliken,
        iters: tc2_iters,
        history,
    };
    with_state(|s| {
        s.sparse_results.insert(name.to_string(), result);
    });

    Dynamic::from_float(r_i as f64)
}
/// Returns max|K - D| where D is the dense density matrix.
fn rhai_compare_density(name: &str) -> Dynamic {
    let payload = with_state(|s| {
        let scc = s.scc_results.get(name)?.clone();
        let sparse = s.sparse_results.get(name)?.clone();
        Some((scc, sparse))
    });

    let Some((scc, sparse)) = payload else {
        eprintln!("ERROR: missing results for '{name}'");
        return Dynamic::from_float(f64::NAN);
    };

    let n_orbs = scc.density.nrows();
    let k_dense_f64: Vec<f64> = sparse.k_dense.iter().map(|&x| x as f64).collect();

    // Check for NaN in sparse result
    let has_nan = k_dense_f64.iter().any(|&x| x.is_nan());
    if has_nan {
        eprintln!("  ||K - D||_max = NaN (sparse K contains NaN — purification diverged)");
        return Dynamic::from_float(f64::NAN);
    }

    // Convention: sparse K is the density kernel with Tr(KS) = N_occ
    // (no spin factor). Dense DFTB D = 2·C_occ·C_occ^T has Tr(DS) = 2·N_occ.
    // So the correct comparison is 2·K vs D (or K vs D/2).
    let spin_factor = 2.0f64;

    let mut max_diff = 0.0f64;
    let mut max_i = 0usize;
    let mut max_j = 0usize;
    let mut sum_sq = 0.0f64;
    for i in 0..n_orbs {
        for j in 0..n_orbs {
            let d = (scc.density[(i, j)] - spin_factor * k_dense_f64[i * n_orbs + j]).abs();
            if d > max_diff {
                max_diff = d;
                max_i = i;
                max_j = j;
            }
            sum_sq += d * d;
        }
    }
    let rms = (sum_sq / (n_orbs * n_orbs) as f64).sqrt();
    eprintln!(
        "  ||2K - D||_max = {max_diff:e}  at ({max_i},{max_j}): D={:.6} 2K={:.6}",
        scc.density[(max_i, max_j)],
        2.0 * k_dense_f64[max_i * n_orbs + max_j]
    );
    eprintln!("  ||2K - D||_rms = {rms:e}");
    // Print diagonal elements for comparison
    eprintln!(
        "  diagonal D[0:4]:  {:?}",
        (0..4)
            .map(|i| format!("{:.6}", scc.density[(i, i)]))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "  diagonal 2K[0:4]: {:?}",
        (0..4)
            .map(|i| format!("{:.6}", 2.0 * k_dense_f64[i * n_orbs + i]))
            .collect::<Vec<_>>()
    );

    Dynamic::from_float(max_diff)
}

/// Compare Mulliken charges between dense and sparse.
fn rhai_compare_charges(name: &str) -> Dynamic {
    let payload = with_state(|s| {
        let scc = s.scc_results.get(name)?.clone();
        let sparse = s.sparse_results.get(name)?.clone();
        Some((scc, sparse))
    });

    let Some((scc, sparse)) = payload else {
        eprintln!("ERROR: missing results for '{name}'");
        return Dynamic::from_float(f64::NAN);
    };

    let has_nan = sparse.mulliken.iter().any(|&x| x.is_nan());
    if has_nan {
        eprintln!("  max|dq| = NaN (sparse charges contain NaN — purification diverged)");
        return Dynamic::from_float(f64::NAN);
    }

    let mut max_diff = 0.0f64;
    eprintln!("  atom  dense_q   sparse_q   diff");
    for i in 0..scc.charges.len() {
        let dq = (scc.charges[i] - sparse.mulliken[i] as f64).abs();
        if dq > max_diff {
            max_diff = dq;
        }
        eprintln!(
            "  [{i:3}]  {:8.4}  {:8.4}  {:8.4}",
            scc.charges[i], sparse.mulliken[i], dq
        );
    }
    eprintln!("  max|dq| = {max_diff:e}");
    Dynamic::from_float(max_diff)
}

/// Get the SCC energy from a stored DFTB result.
fn rhai_get_energy(name: &str) -> f64 {
    with_state(|s| {
        s.scc_results
            .get(name)
            .map(|r| r.energy)
            .unwrap_or(f64::NAN)
    })
}

/// Get the number of atoms in a stored geometry.
fn rhai_get_n_atoms(name: &str) -> INT {
    with_state(|s| {
        s.geometries
            .get(name)
            .map(|st| st.natom() as INT)
            .unwrap_or(0)
    })
}

/// Convert a float to string for printing.
/// clock() -> seconds since first call — for timing scan/bench scripts.
fn rhai_clock() -> f64 {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    T0.get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
}

fn rhai_ftos(x: f64) -> String {
    format!("{x}")
}

/// Convert an int to string for printing.
fn rhai_itos(x: INT) -> String {
    format!("{x}")
}

/// Save TC2 convergence history to a TSV file for plotting.
fn rhai_save_convergence(name: &str, path: &str) -> bool {
    let hist = with_state(|s| s.sparse_results.get(name).map(|r| r.history.clone()));
    let Some(hist) = hist else {
        eprintln!("ERROR: no sparse result for '{name}'");
        return false;
    };
    let mut txt = String::from("iter\tR_I\tTr_KS\n");
    for (iter, r_i, tr) in &hist {
        txt.push_str(&format!("{iter}\t{r_i:e}\t{tr:.6}\n"));
    }
    if let Err(e) = std::fs::write(path, txt) {
        eprintln!("ERROR writing {path}: {e}");
        return false;
    }
    eprintln!("  saved convergence: {path} ({} iters)", hist.len());
    true
}

/// Get dense SCC Mulliken charges as a comma-separated string.
fn rhai_get_charges(name: &str) -> String {
    with_state(|s| {
        s.scc_results
            .get(name)
            .map(|r| {
                r.charges
                    .iter()
                    .map(|q| format!("{q:.6}"))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default()
    })
}

/// Get sparse Mulliken charges as a comma-separated string.
fn rhai_get_sparse_charges(name: &str) -> String {
    with_state(|s| {
        s.sparse_results
            .get(name)
            .map(|r| {
                r.mulliken
                    .iter()
                    .map(|q| format!("{q:.6}"))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default()
    })
}

/// Get all eigenvalues from the dense SCC result as a comma-separated string.
fn rhai_get_eigenvalues(name: &str) -> String {
    with_state(|s| {
        s.scc_results
            .get(name)
            .map(|r| {
                r.eigenvalues
                    .iter()
                    .map(|e| format!("{e:.6}"))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default()
    })
}

/// Get HOMO, LUMO, and gap as "homo,lumo,gap".
/// Uses the dense SCC eigenvalues. n_occ is inferred from electron count.
fn rhai_get_homo_lumo(name: &str) -> String {
    let payload = with_state(|s| {
        s.scc_results.get(name).map(|r| {
            let n_orbs = r.eigenvalues.len();
            let n_atoms = r.charges.len();
            // n_occ from electron count: each atom contributes valence e⁻
            // For pure-carbon systems (4 valence e⁻, 4 orbitals): n_occ = n_atoms
            // But we stored q0, so n_electrons = sum(q0)
            let n_electrons: f64 = r.q0.iter().sum();
            let n_occ = (n_electrons / 2.0).round() as usize;
            (
                r.eigenvalues[n_occ.saturating_sub(1).min(n_orbs - 1)],
                r.eigenvalues[n_occ.min(n_orbs - 1)],
                n_occ,
            )
        })
    });
    let Some((homo, lumo, n_occ)) = payload else {
        return String::new();
    };
    format!("{homo:.6},{lumo:.6},{:.6}", lumo - homo)
}

/// Save dense SCC charges + geometry to a TSV for plotting.
/// Columns: atom_idx, element, x, y, z, charge
fn rhai_save_charges(name: &str, path: &str) -> bool {
    let payload = with_state(|s| {
        let scc = s.scc_results.get(name)?;
        let st = s.geometries.get(name)?;
        Some((
            scc.charges.clone(),
            st.elements.clone(),
            st.positions.clone(),
        ))
    });
    let Some((charges, elements, positions)) = payload else {
        eprintln!("ERROR: no SCC result or geometry for '{name}'");
        return false;
    };
    let mut txt = String::from("atom_idx\telement\tx\ty\tz\tcharge\n");
    for (i, ((el, pos), q)) in elements
        .iter()
        .zip(positions.iter())
        .zip(charges.iter())
        .enumerate()
    {
        txt.push_str(&format!(
            "{i}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",
            el.symbol(),
            pos[0],
            pos[1],
            pos[2],
            q
        ));
    }
    if let Err(e) = std::fs::write(path, txt) {
        eprintln!("ERROR writing {path}: {e}");
        return false;
    }
    eprintln!("  saved charges: {path} ({} atoms)", charges.len());
    true
}

/// Save sparse Mulliken charges + geometry to a TSV for plotting.
fn rhai_save_sparse_charges(name: &str, path: &str) -> bool {
    let payload = with_state(|s| {
        let sparse = s.sparse_results.get(name)?;
        let st = s.geometries.get(name)?;
        Some((
            sparse.mulliken.clone(),
            st.elements.clone(),
            st.positions.clone(),
        ))
    });
    let Some((charges, elements, positions)) = payload else {
        eprintln!("ERROR: no sparse result or geometry for '{name}'");
        return false;
    };
    let mut txt = String::from("atom_idx\telement\tx\ty\tz\tcharge\n");
    for (i, ((el, pos), q)) in elements
        .iter()
        .zip(positions.iter())
        .zip(charges.iter())
        .enumerate()
    {
        txt.push_str(&format!(
            "{i}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",
            el.symbol(),
            pos[0],
            pos[1],
            pos[2],
            q
        ));
    }
    if let Err(e) = std::fs::write(path, txt) {
        eprintln!("ERROR writing {path}: {e}");
        return false;
    }
    eprintln!("  saved sparse charges: {path} ({} atoms)", charges.len());
    true
}

/// Save eigenvalues to a TSV for plotting.
/// Columns: idx, eigenvalue, occupied (1 if idx < n_occ, else 0)
fn rhai_save_eigenvalues(name: &str, path: &str) -> bool {
    let payload = with_state(|s| {
        s.scc_results.get(name).map(|r| {
            let n_electrons: f64 = r.q0.iter().sum();
            let n_occ = (n_electrons / 2.0).round() as usize;
            (r.eigenvalues.clone(), n_occ)
        })
    });
    let Some((eigs, n_occ)) = payload else {
        eprintln!("ERROR: no SCC result for '{name}'");
        return false;
    };
    let mut txt = String::from("idx\teigenvalue\toccupied\n");
    for (i, e) in eigs.iter().enumerate() {
        let occ = if i < n_occ { 1 } else { 0 };
        txt.push_str(&format!("{i}\t{e:.10}\t{occ}\n"));
    }
    if let Err(e) = std::fs::write(path, txt) {
        eprintln!("ERROR writing {path}: {e}");
        return false;
    }
    eprintln!(
        "  saved eigenvalues: {path} ({} eigs, n_occ={n_occ})",
        eigs.len()
    );
    true
}

/// Save eigenvectors + geometry + eigenvalues to a TSV for wavefunction plotting.
/// Format:
///   # natoms norb n_occ
///   # atom_idx element x y z
///   ... (natoms lines)
///   # eigenvector matrix (norb × norb), columns are MOs
///   mo_idx  orb_idx  coeff
///   ... (norb*norb lines)
///   # eigenvalues
///   idx  eigenvalue  occupied
fn rhai_save_eigenvectors(name: &str, path: &str) -> bool {
    let payload = with_state(|s| {
        let scc = s.scc_results.get(name)?;
        let st = s.geometries.get(name)?;
        let species: Vec<String> = st.elements.iter().map(|e| e.symbol().to_string()).collect();
        let coords: Vec<[f64; 3]> = st.positions.clone();
        let n_electrons: f64 = scc.q0.iter().sum();
        let n_occ = (n_electrons / 2.0).round() as usize;
        Some((
            scc.eigenvectors.clone(),
            scc.eigenvalues.clone(),
            species,
            coords,
            n_occ,
        ))
    });
    let Some((evecs, eigs, species, coords, n_occ)) = payload else {
        eprintln!("ERROR: no SCC result for '{name}'");
        return false;
    };
    let natoms = species.len();
    let norb = evecs.nrows();
    let mut txt = String::new();
    txt.push_str(&format!("# natoms={natoms} norb={norb} n_occ={n_occ}\n"));
    txt.push_str("# atom_idx\telement\tx\ty\tz\n");
    for (i, (sp, c)) in species.iter().zip(coords.iter()).enumerate() {
        txt.push_str(&format!(
            "{i}\t{sp}\t{:.10}\t{:.10}\t{:.10}\n",
            c[0], c[1], c[2]
        ));
    }
    txt.push_str("# eigenvector matrix (norb x norb), columns are MOs\n");
    txt.push_str("mo_idx\torb_idx\tcoeff\n");
    for mo in 0..evecs.ncols() {
        for orb in 0..evecs.nrows() {
            txt.push_str(&format!("{mo}\t{orb}\t{:.12e}\n", evecs[(orb, mo)]));
        }
    }
    txt.push_str("# eigenvalues\n");
    txt.push_str("idx\teigenvalue\toccupied\n");
    for (i, e) in eigs.iter().enumerate() {
        let occ = if i < n_occ { 1 } else { 0 };
        txt.push_str(&format!("{i}\t{e:.10}\t{occ}\n"));
    }
    if let Err(e) = std::fs::write(path, txt) {
        eprintln!("ERROR writing {path}: {e}");
        return false;
    }
    eprintln!("  saved eigenvectors: {path} ({natoms} atoms, {norb} orbitals, n_occ={n_occ})");
    true
}

/// Save H_scc and S matrices + geometry to a TSV for sparse iterative eigensolving
/// (Chebyshev filter + Ritz from NumericalMathPlayground).
/// Format:
///   # natoms norb n_occ
///   # atom_idx element x y z
///   ... (natoms lines)
///   # H_scc matrix (norb × norb), row-major
///   i j H_ij
///   ... (norb*norb lines)
///   # S matrix (norb × norb), row-major
///   i j S_ij
///   ... (norb*norb lines)
///   # eigenvalues (from dense diagonalization, for reference)
///   idx eigenvalue occupied
fn rhai_save_hs_matrix(name: &str, path: &str) -> bool {
    let payload = with_state(|s| {
        let scc = s.scc_results.get(name)?;
        let st = s.geometries.get(name)?;
        let species: Vec<String> = st.elements.iter().map(|e| e.symbol().to_string()).collect();
        let coords: Vec<[f64; 3]> = st.positions.clone();
        let n_electrons: f64 = scc.q0.iter().sum();
        let n_occ = (n_electrons / 2.0).round() as usize;
        Some((
            scc.h_scc.clone(),
            scc.s.clone(),
            scc.eigenvalues.clone(),
            species,
            coords,
            n_occ,
        ))
    });
    let Some((h, s_mat, eigs, species, coords, n_occ)) = payload else {
        eprintln!("ERROR: no SCC result for '{name}'");
        return false;
    };
    let natoms = species.len();
    let norb = h.nrows();
    let mut txt = String::new();
    txt.push_str(&format!("# natoms={natoms} norb={norb} n_occ={n_occ}\n"));
    txt.push_str("# atom_idx\telement\tx\ty\tz\n");
    for (i, (sp, c)) in species.iter().zip(coords.iter()).enumerate() {
        txt.push_str(&format!(
            "{i}\t{sp}\t{:.10}\t{:.10}\t{:.10}\n",
            c[0], c[1], c[2]
        ));
    }
    txt.push_str("# H_scc matrix (norb x norb), row-major\n");
    txt.push_str("i\tj\tH_ij\n");
    for i in 0..h.nrows() {
        for j in 0..h.ncols() {
            txt.push_str(&format!("{i}\t{j}\t{:.12e}\n", h[(i, j)]));
        }
    }
    txt.push_str("# S matrix (norb x norb), row-major\n");
    txt.push_str("i\tj\tS_ij\n");
    for i in 0..s_mat.nrows() {
        for j in 0..s_mat.ncols() {
            txt.push_str(&format!("{i}\t{j}\t{:.12e}\n", s_mat[(i, j)]));
        }
    }
    txt.push_str("# eigenvalues (dense reference)\n");
    txt.push_str("idx\teigenvalue\toccupied\n");
    for (i, e) in eigs.iter().enumerate() {
        let occ = if i < n_occ { 1 } else { 0 };
        txt.push_str(&format!("{i}\t{e:.10}\t{occ}\n"));
    }
    if let Err(e) = std::fs::write(path, txt) {
        eprintln!("ERROR writing {path}: {e}");
        return false;
    }
    eprintln!("  saved H,S matrices: {path} ({natoms} atoms, {norb} orbitals, n_occ={n_occ})");
    true
}

/// Run Davidson partial eigensolver on the stored SCC Hamiltonian to find
/// a few eigenvalues around the HOMO-LUMO gap. Returns "homo,lumo,gap".
/// n_target = number of eigenvalues to compute on each side of the gap.
fn rhai_davidson_homo_lumo(name: &str, n_target: INT) -> String {
    let payload = with_state(|s| {
        s.scc_results
            .get(name)
            .map(|r| (r.h_scc.clone(), r.s.clone(), r.q0.clone()))
    });
    let Some((h, s_mat, q0)) = payload else {
        eprintln!("ERROR: no SCC result for '{name}'");
        return String::new();
    };
    let n_electrons: f64 = q0.iter().sum();
    let n_occ = (n_electrons / 2.0).round() as usize;
    let n_target = n_target.max(2) as usize;
    eprintln!(
        "  Davidson: n_orbs={}, n_occ={}, n_target={n_target}",
        h.nrows(),
        n_occ
    );
    match rust_dftb::methods::sparse::davidson_homo_lumo(&h, &s_mat, n_occ, n_target, 100, 1e-8) {
        Ok((eigs, _vecs)) => {
            // eigs is sorted ascending; find HOMO (last occ) and LUMO (first virt)
            // Davidson targets n_target below and n_target above the gap
            let homo = eigs[n_target - 1];
            let lumo = eigs[n_target];
            eprintln!(
                "    Davidson HOMO={homo:.10}, LUMO={lumo:.10}, gap={:.10}",
                lumo - homo
            );
            eprintln!(
                "    all {} eigs: {:?}",
                eigs.len(),
                eigs.iter().map(|e| format!("{e:.6}")).collect::<Vec<_>>()
            );
            format!("{homo:.10},{lumo:.10},{:.10}", lumo - homo)
        }
        Err(e) => {
            eprintln!("ERROR Davidson: {e}");
            String::new()
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────

/// Convert a dense nalgebra DMatrix to BSR4 format.
/// Assumes 4 orbitals per atom, row-major block layout.
fn dense_to_bsr4(m: &DMatrix<f64>, n_atom: usize, mask: &(Vec<u32>, Vec<u32>)) -> Bsr4Matrix {
    let nblock = mask.1.len();
    let mut values = vec![0.0f32; nblock * 16];
    for i in 0..n_atom {
        let (a, b) = (mask.0[i] as usize, mask.0[i + 1] as usize);
        for blk in a..b {
            let j = mask.1[blk] as usize;
            // Block (i,j): values[16*blk + 4*r + c] = M[4*i+r, 4*j+c]
            for r in 0..4 {
                for c in 0..4 {
                    let val = m[(4 * i + r, 4 * j + c)] as f32;
                    values[16 * blk + 4 * r + c] = val;
                }
            }
        }
    }
    Bsr4Matrix::from_parts(n_atom, mask.0.clone(), mask.1.clone(), values)
        .expect("BSR4 from_parts failed")
}

// ─── GpuDftb (persistent GPU engine; one object per named template) ──

struct GpuHandle {
    eng: GpuDftb,
    last_e: Vec<f64>,
    last_f: Option<Vec<f32>>,
    last_rms: f32,
    last_q_rms: f64,
    last_q_max: f64,
    last_iters: i64,
    last_stalled: bool,
    /// §12 D11: per-system SCC status words ("converged"/"plateau"/"failed").
    last_status: Vec<String>,
}

thread_local! {
    static GPU: RefCell<HashMap<String, GpuHandle>> = RefCell::new(HashMap::new());
}

fn with_gpu<F, R>(f: F) -> R
where
    F: FnOnce(&mut HashMap<String, GpuHandle>) -> R,
{
    GPU.with(|g| f(&mut g.borrow_mut()))
}

// ─── SparseDftb (persistent sparse engine; one object per named system) ──

struct SparseHandle {
    eng: SparseDftb,
    last_rms: f64,
    last_iters: i64,
    last_fmax: f64,
    have_forces: bool,
}

thread_local! {
    static SPARSE: RefCell<HashMap<String, SparseHandle>> = RefCell::new(HashMap::new());
}

fn with_sparse<F, R>(f: F) -> R
where
    F: FnOnce(&mut HashMap<String, SparseHandle>) -> R,
{
    SPARSE.with(|g| f(&mut g.borrow_mut()))
}

// ─── GpuPbc (persistent periodic dense engine; n_rep replicas share one cell) ──

struct PbcHandle {
    eng: rust_dftb::qmqm::gpu_pbc::GpuPbc,
    lat: [[f64; 3]; 3],
    /// per-replica coordinates [n_rep*n_atoms] — needed for the host-side
    /// repulsive energy (compute_energy is band+SCC only).
    coords: Vec<[f64; 3]>,
    repulsive: Vec<Option<rust_dftb::methods::dftb::forces::RepulsiveSpline>>,
    species_names: Vec<String>,
    species_code: Vec<u8>,
    last_e: Vec<f64>,
    last_rms: f32,
    last_iters: i64,
}

thread_local! {
    static PBC: RefCell<HashMap<String, PbcHandle>> = RefCell::new(HashMap::new());
}

fn with_pbc<F, R>(f: F) -> R
where
    F: FnOnce(&mut HashMap<String, PbcHandle>) -> R,
{
    PBC.with(|g| f(&mut g.borrow_mut()))
}

fn geom_species_coords(name: &str, ctx: &str) -> (Vec<String>, Vec<[f64; 3]>) {
    with_state(|s| {
        let st = s
            .geometries
            .get(name)
            .unwrap_or_else(|| panic!("{ctx} '{name}': no geometry — load_xyz/make_geom first"));
        let species: Vec<String> = st.elements.iter().map(|e| e.symbol().to_string()).collect();
        (species, st.positions.clone())
    })
}

fn sync_geom_coords(name: &str, coords: &[[f64; 3]]) {
    with_state(|s| {
        let st = s
            .geometries
            .get_mut(name)
            .unwrap_or_else(|| panic!("sync_geom '{name}': no geometry"));
        if st.positions.len() != coords.len() {
            panic!(
                "sync_geom '{name}': n_atom {} != coords {}",
                st.positions.len(),
                coords.len()
            );
        }
        st.positions.copy_from_slice(coords);
    });
}

/// Compile/allocate SparseDftb once for a stored geometry. Not a replica batch (one system).
/// 4-arg form: sparse_new(name, sk_dir, r_trunc_ang, taper_w_ang) — r_trunc <= 0
/// = untruncated (full SK table range); otherwise H/S pairs are cosine-tapered
/// to zero over [r_trunc − taper_w, r_trunc] and the mask shrinks to
/// r_trunc + skin.
fn rhai_sparse_new(name: &str, sk_dir: &str) -> INT {
    rhai_sparse_new_full(name, sk_dir, 0.0, 0.0, 0.0, 0.0)
}

fn rhai_sparse_new_cut(name: &str, sk_dir: &str, r_trunc_ang: f64, taper_w_ang: f64) -> INT {
    rhai_sparse_new_full(name, sk_dir, r_trunc_ang, taper_w_ang, 0.0, 0.0)
}

/// sparse_new(name, sk_dir, r_trunc, taper_w, r_k, r_z) — r_k/r_z <= 0 = default
/// (full SK radius + skin). K/Z decay is set by the gap, not the H/S range.
fn rhai_sparse_new_full(
    name: &str,
    sk_dir: &str,
    r_trunc_ang: f64,
    taper_w_ang: f64,
    r_k_ang: f64,
    r_z_ang: f64,
) -> INT {
    rhai_sparse_new_budget(
        name,
        sk_dir,
        r_trunc_ang,
        taper_w_ang,
        r_k_ang,
        r_z_ang,
        0.0,
    )
}

/// sparse_new(name, sk_dir, r_trunc, taper_w, r_k, r_z, max_deg) — §15.9
/// SC1: max_deg > 0 overrides ALL degree budgets (hs/k/z) — the explicit
/// sparsity-budget declaration for runs whose masks are wider than the
/// production ceilings (512/128/256). Pick the smallest value that fits.
fn rhai_sparse_new_budget(
    name: &str,
    sk_dir: &str,
    r_trunc_ang: f64,
    taper_w_ang: f64,
    r_k_ang: f64,
    r_z_ang: f64,
    max_deg: f64,
) -> INT {
    rhai_sparse_new_mask(
        name,
        sk_dir,
        r_trunc_ang,
        taper_w_ang,
        r_k_ang,
        r_z_ang,
        0.0,
        max_deg,
    )
}

/// `sparse_new_mask(name, sk, r_trunc, taper, r_k, r_z, r_skin, max_deg)`.
/// `r_skin <= 0` keeps the config default. `max_deg <= 0` keeps the budgets.
fn rhai_sparse_new_mask(
    name: &str,
    sk_dir: &str,
    r_trunc_ang: f64,
    taper_w_ang: f64,
    r_k_ang: f64,
    r_z_ang: f64,
    r_skin_ang: f64,
    max_deg: f64,
) -> INT {
    let (species, coords) = geom_species_coords(name, "sparse_new");
    let n_atom = species.len();
    eprintln!("[sparse] sparse_new '{name}' n_atom={n_atom} sk={sk_dir} r_trunc={r_trunc_ang} taper_w={taper_w_ang} r_k={r_k_ang} r_z={r_z_ang} max_deg={max_deg}");
    let sk = load_sk_for_species(sk_dir, &species)
        .unwrap_or_else(|e| panic!("sparse_new '{name}' load SK from {sk_dir}: {e}"));
    let mut cfg = SparseDftbConfig::default();
    if r_trunc_ang > 0.0 {
        cfg.r_trunc_ang = Some(r_trunc_ang);
    }
    if taper_w_ang > 0.0 {
        cfg.taper_w_ang = taper_w_ang;
    }
    if r_k_ang > 0.0 {
        cfg.r_k_ang = Some(r_k_ang);
    }
    if r_z_ang > 0.0 {
        cfg.r_z_ang = Some(r_z_ang);
    }
    if r_skin_ang > 0.0 {
        cfg.r_skin_ang = r_skin_ang;
    }
    if max_deg > 0.0 {
        let d = max_deg as u32;
        cfg.max_deg_hs = Some(d);
        cfg.max_deg_k = Some(d);
        cfg.max_deg_z = Some(d);
    }
    let eng = SparseDftb::with_config(sk, sk_dir, species, coords, cfg)
        .unwrap_or_else(|e| panic!("sparse_new '{name}' SparseDftb::with_config: {e}"));
    let n_orbs = eng.n_orbs() as INT;
    eprintln!("[sparse] sparse_new '{name}' n_orbs={n_orbs}");
    with_sparse(|g| {
        g.insert(
            name.to_string(),
            SparseHandle {
                eng,
                last_rms: f64::NAN,
                last_iters: 0,
                last_fmax: f64::NAN,
                have_forces: false,
            },
        );
    });
    n_orbs
}

fn rhai_sparse_scc(name: &str, max_iter: INT, tol: f64) -> f64 {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_scc '{name}': no engine — sparse_new first"));
        let scc = h
            .eng
            .scc(max_iter as usize, tol)
            .unwrap_or_else(|e| panic!("sparse_scc '{name}': {e}"));
        h.last_rms = scc.rms;
        h.last_iters = scc.n_iters as i64;
        h.have_forces = false;
        eprintln!(
            "[sparse] sparse_scc '{name}' rms={:.3e} iters={} Tr(KS)={:.6} R_I={:.3e} r_scc={:.3e}",
            scc.rms, scc.n_iters, scc.tr_ks, scc.r_i, scc.r_scc
        );
        scc.rms
    })
}

/// sparse_scc_try(name, max_iter, tol) — measurement variant: returns the
/// final rms, or NaN if scc fails. "Did not converge" IS data in a sweep —
/// the failure is still printed loudly to stderr; only the panic becomes
/// a recorded value. Production scripts should use `sparse_scc`.
fn rhai_sparse_scc_try(name: &str, max_iter: INT, tol: f64) -> f64 {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_scc_try '{name}': no engine — sparse_new first"));
        match h.eng.scc(max_iter as usize, tol) {
            Ok(scc) => {
                h.last_rms = scc.rms;
                h.last_iters = scc.n_iters as i64;
                h.have_forces = false;
                eprintln!(
                    "[sparse] sparse_scc_try '{name}' rms={:.3e} iters={} Tr(KS)={:.6} R_I={:.3e}",
                    scc.rms, scc.n_iters, scc.tr_ks, scc.r_i
                );
                scc.rms
            }
            Err(e) => {
                eprintln!("[sparse] sparse_scc_try '{name}' FAILED (recorded as NaN): {e}");
                f64::NAN
            }
        }
    })
}

/// sparse_purify_now(name, max_iter, tol) — TC2 convergence study: rebuild
/// K0 from the CURRENT H_scc (identical cold seed per call) and run one
/// tc2_purify. Pair with RUST_DFTB_TC2_HIST=<csv> to record per-iter
/// (R_I, Tr, branch) for variant comparison. Returns iters (negative = failed).
fn rhai_sparse_purify_now(name: &str, max_iter: INT, tol: f64) -> INT {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_purify_now '{name}': no engine — sparse_new first"));
        match h.eng.purify_cold(max_iter as usize, tol as f32) {
            Ok((status, r_i, tr, iters)) => {
                eprintln!("[sparse] purify_now '{name}' status={status:?} iters={iters} R_I={r_i:.3e} Tr(KS)={tr:.6}");
                iters as INT
            }
            Err(e) => {
                eprintln!("[sparse] purify_now '{name}' FAILED: {e}");
                -1
            }
        }
    })
}

/// sparse_ri_f64(name) — F64-DIAG/DUMMY-DECOMP: dense f64
/// ‖KSK−K‖/‖K‖ of the current device K, split into physical vs
/// dummy-lane parts. Prints all four numbers, returns r_i_f64.
fn rhai_sparse_ri_f64(name: &str) -> f64 {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_ri_f64 '{name}': no engine"));
        let (r, rp, rd, tr) = h
            .eng
            .ri_f64_diag()
            .unwrap_or_else(|e| panic!("sparse_ri_f64 '{name}': {e}"));
        eprintln!(
            "[sparse] ri_f64 '{name}': R_I={r:.3e}  phys={rp:.3e}  dummy={rd:.3e}  Tr(KS)={tr:.6}"
        );
        r
    })
}

/// sparse_mcw_f64(name, n_steps, store_f32) — F64-MCW decisive test
/// (manifest §4.12.2 round 2): all-f64 McWeeny on the stored device K,
/// reporting (R_I⁶⁴, R_H⁶⁴, Tr) per step. store_f32=0 → variant B
/// (all-f64); store_f32=1 → variant A (f32 storage). CSV to
/// RUST_DFTB_MCW_F64_HIST when set.
fn rhai_sparse_mcw_f64(name: &str, n_steps: INT, store_f32: bool) {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_mcw_f64 '{name}': no engine"));
        h.eng
            .mcw_f64_diag(n_steps as usize, store_f32)
            .unwrap_or_else(|e| panic!("sparse_mcw_f64 '{name}': {e}"));
    })
}

/// sparse_mcw_ff(name, n) — FF32-POLISH: n float-float McWeeny steps on
/// the current device K (all-f32 FMA, ~46-bit intermediates). Pair with
/// sparse_ri_f64 to verify the true residual drop.
fn rhai_sparse_mcw_ff(name: &str, n: INT) {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_mcw_ff '{name}': no engine"));
        h.eng
            .mcw_ff(n as usize)
            .unwrap_or_else(|e| panic!("sparse_mcw_ff '{name}': {e}"));
    })
}

/// sparse_sync(name) — block until the sparse engine's in-order queue
/// drains. For honest wall-clock timing of enqueue-only paths
/// (sparse_mcw_ff et al.): time (work; sparse_sync) not work alone.
fn rhai_sparse_sync(name: &str) {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_sync '{name}': no engine"));
        h.eng
            .gpu()
            .finish()
            .unwrap_or_else(|e| panic!("sparse_sync '{name}': {e}"));
    })
}

/// sparse_ff_test(name) — FF32-POLISH unit check: ONE ff product
/// T_ff = K·S vs host-f64 dense reference; prints rel.err of the hi
/// part alone vs (hi+lo). hi≈f32-level, ff should be ~1e-13.
fn rhai_sparse_ff_test(name: &str) {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_ff_test '{name}': no engine"));
        let (e1h, e1f, e2h, e2f) = h
            .eng
            .ws_ff_test_ks()
            .unwrap_or_else(|e| panic!("sparse_ff_test '{name}': {e}"));
        eprintln!("[ff_test] '{name}': KS err(hi)={e1h:.3e} err(ff)={e1f:.3e} | (KS)K err(hi)={e2h:.3e} err(ff)={e2f:.3e}");
    })
}

/// Energy of the last SCC. `want_forces=true` also builds analytic F (CPU contract of D,W).
fn rhai_sparse_eval(name: &str, want_forces: bool) -> f64 {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_eval '{name}': no engine — sparse_new first"));
        let e = h
            .eng
            .energy()
            .unwrap_or_else(|e| panic!("sparse_eval '{name}': {e} — call sparse_scc first"));
        if !e.is_finite() {
            panic!("sparse_eval '{name}': E={e} non-finite");
        }
        if want_forces {
            let f = h
                .eng
                .forces()
                .unwrap_or_else(|err| panic!("sparse_eval '{name}' forces: {err}"));
            let mut max_f = 0.0f64;
            for fi in &f.forces {
                for &c in fi {
                    if !c.is_finite() {
                        panic!("sparse_eval '{name}': non-finite force {c}");
                    }
                    max_f = max_f.max(c.abs());
                }
            }
            h.last_fmax = max_f;
            h.have_forces = true;
            eprintln!("[sparse] sparse_eval '{name}' E={e:.8} Ha  max|F|={max_f:.4e}");
        } else {
            eprintln!("[sparse] sparse_eval '{name}' E={e:.8} Ha  (no forces)");
        }
        e
    })
}

fn rhai_sparse_max_force(name: &str) -> f64 {
    with_sparse(|g| {
        let h = g
            .get(name)
            .unwrap_or_else(|| panic!("sparse_max_force '{name}': no engine"));
        if !h.have_forces {
            panic!("sparse_max_force '{name}': last sparse_eval had want_forces=false");
        }
        h.last_fmax
    })
}

/// sparse_force_at(name, atom_i, comp) -> force component (Ha/Å). Runs the
/// analytic sparse force path (requires a prior sparse_scc at this geometry).
fn rhai_sparse_force_at(name: &str, i: INT, c: INT) -> f64 {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_force_at '{name}': no engine"));
        let f = h
            .eng
            .forces()
            .unwrap_or_else(|e| panic!("sparse_force_at '{name}': {e}"));
        let (i, c) = (i as usize, c as usize);
        if i >= h.eng.n_atom() || c > 2 {
            panic!(
                "sparse_force_at '{name}': bad index i={i} c={c} (n_atom={})",
                h.eng.n_atom()
            );
        }
        f.forces[i][c]
    })
}

fn rhai_sparse_fire_step(name: &str, f_tol: f64) -> f64 {
    let max_f = with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_fire_step '{name}': no engine — sparse_scc first"));
        let mf = h
            .eng
            .fire_step(f_tol)
            .unwrap_or_else(|e| panic!("sparse_fire_step '{name}': {e}"));
        h.have_forces = true;
        h.last_fmax = mf;
        sync_geom_coords(name, h.eng.coords());
        eprintln!("[sparse] sparse_fire_step '{name}' max|F|={mf:.4e}");
        mf
    });
    max_f
}

fn rhai_sparse_md_step(name: &str, dt: f64) -> f64 {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_md_step '{name}': no engine"));
        let mf = h
            .eng
            .md_step(dt)
            .unwrap_or_else(|e| panic!("sparse_md_step '{name}': {e}"));
        h.have_forces = true;
        h.last_fmax = mf;
        sync_geom_coords(name, h.eng.coords());
        eprintln!("[sparse] sparse_md_step '{name}' dt={dt} max|F|={mf:.4e}");
        mf
    })
}

fn rhai_sparse_relax(name: &str, max_steps: INT, f_tol: f64, scc_tol: f64) -> f64 {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_relax '{name}': no engine"));
        let (n, max_f, rms0) = h
            .eng
            .relax(max_steps as usize, f_tol, scc_tol)
            .unwrap_or_else(|e| panic!("sparse_relax '{name}': {e}"));
        h.last_iters = n as i64;
        h.last_rms = rms0;
        h.last_fmax = max_f;
        h.have_forces = true;
        sync_geom_coords(name, h.eng.coords());
        eprintln!("[sparse] sparse_relax '{name}' steps={n} max|F|={max_f:.4e} rms0={rms0:.3e}");
        max_f
    })
}

fn rhai_sparse_geom_mode(name: &str, mode: &str) {
    let step = match mode {
        "bold" | "b3" => GeomStep::BoldXtrDmm,
        "legacy" => GeomStep::LegacyDmm,
        other => panic!("sparse_geom_mode '{name}': unknown mode '{other}' (bold|legacy)"),
    };
    with_sparse(|g| {
        g.get_mut(name)
            .unwrap_or_else(|| panic!("sparse_geom_mode '{name}': no engine"))
            .eng
            .set_geom_step(step);
    });
    eprintln!("[sparse] geom_mode '{name}' = {mode}");
}

fn rhai_sparse_fire_dt(name: &str, dt: f64) {
    if !dt.is_finite() || dt <= 0.0 {
        panic!("sparse_fire_dt '{name}': dt={dt}");
    }
    with_sparse(|g| {
        g.get_mut(name)
            .unwrap_or_else(|| panic!("sparse_fire_dt '{name}': no engine"))
            .eng
            .set_fire_dt(dt);
    });
    eprintln!("[sparse] fire_dt '{name}' = {dt}");
}

/// Move every atom `amp` Å along a deterministic direction.
fn rhai_sparse_jitter(name: &str, amp: f64, seed: INT) -> INT {
    if !(amp > 0.0) || !amp.is_finite() {
        panic!("sparse_jitter '{name}': amp={amp}");
    }
    let mut s = seed as u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as f64) / (u64::MAX as f64)
    };
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_jitter '{name}': no engine"));
        let jittered: Vec<[f64; 3]> = h
            .eng
            .coords()
            .iter()
            .map(|c| {
                let u = rnd() * 2.0 - 1.0;
                let v = rnd() * 2.0 - 1.0;
                let w = rnd() * 2.0 - 1.0;
                let nrm = (u * u + v * v + w * w).sqrt().max(1e-15);
                [c[0] + amp * u / nrm, c[1] + amp * v / nrm, c[2] + amp * w / nrm]
            })
            .collect();
        h.eng
            .set_coords(&jittered)
            .unwrap_or_else(|e| panic!("sparse_jitter '{name}': {e}"));
        sync_geom_coords(name, h.eng.coords());
        eprintln!("[sparse] jitter '{name}' {amp} Å/atom  seed={seed}");
        h.eng.n_atom() as INT
    })
}

/// FIRE until `f_tol`, `max_steps`, `RUST_DFTB_WALL_SECS` (default 40), or 40
/// steps with no new low in max|F|. Appends one XYZ frame and one CSV row per step.
fn rhai_sparse_relax_traj(
    name: &str,
    max_steps: INT,
    f_tol: f64,
    scc_tol: f64,
    traj_path: &str,
    hist_path: &str,
) -> INT {
    let wall_s: f64 = std::env::var("RUST_DFTB_WALL_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40.0);
    let (species, _) = geom_species_coords(name, "sparse_relax_traj");
    if let Some(parent) = std::path::Path::new(traj_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let frame0 = std::fs::read_to_string(traj_path)
        .map(|t| {
            t.lines()
                .filter(|l| {
                    l.starts_with("frame=") || l.starts_with("step=") || l.starts_with("floor=")
                })
                .count()
        })
        .unwrap_or(0);
    let hist_new = !std::path::Path::new(hist_path).exists();
    let mut hist = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(hist_path)
            .unwrap_or_else(|e| panic!("history {hist_path}: {e}")),
    );
    let mut traj = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(traj_path)
            .unwrap_or_else(|e| panic!("traj {traj_path}: {e}")),
    );
    use std::io::Write;
    if hist_new {
        writeln!(hist, "step,E_Ha,maxAbsF,ms,rms,TrKS,R_H,note").unwrap();
    }
    let t_wall = std::time::Instant::now();
    let mut best_f = f64::MAX;
    let mut since_best = 0usize;
    let mut n_done: i64 = frame0 as i64;
    let mut stopped = String::from("max_steps");
    for step in 0..=max_steps as usize {
        if t_wall.elapsed().as_secs_f64() > wall_s {
            stopped = format!("wall {wall_s} s before step {step}");
            eprintln!("[sparse] relax_traj '{name}' {stopped}");
            break;
        }
        let t = std::time::Instant::now();
        let note = if step == 0 { "cold" } else { "" };
        if step > 0 {
            let mf = with_sparse(|g| {
                let h = g
                    .get_mut(name)
                    .unwrap_or_else(|| panic!("sparse_relax_traj '{name}': no engine"));
                let mf = h
                    .eng
                    .fire_step(f_tol)
                    .unwrap_or_else(|e| panic!("sparse_relax_traj '{name}' FIRE {step}: {e}"));
                sync_geom_coords(name, h.eng.coords());
                mf
            });
            if mf < f_tol {
                stopped = format!("max|F|={mf:.3e} < {f_tol:.1e} before move {step}");
                eprintln!("[sparse] relax_traj '{name}' {stopped}");
                break;
            }
        }
        let scc = with_sparse(|g| {
            g.get_mut(name)
                .unwrap_or_else(|| panic!("sparse_relax_traj '{name}': no engine"))
                .eng
                .scc(80, scc_tol)
                .unwrap_or_else(|e| panic!("sparse_relax_traj '{name}' SCC {step}: {e}"))
        });
        let e_tot = with_sparse(|g| {
            g.get_mut(name)
                .unwrap()
                .eng
                .energy()
                .unwrap_or_else(|e| panic!("sparse_relax_traj '{name}' energy: {e}"))
        });
        let max_f = with_sparse(|g| {
            let h = g.get_mut(name).unwrap();
            let f = h
                .eng
                .forces()
                .unwrap_or_else(|e| panic!("sparse_relax_traj '{name}' forces: {e}"));
            let mut max_f = 0.0f64;
            for fi in &f.forces {
                for c in fi {
                    max_f = max_f.max(c.abs());
                }
            }
            h.last_fmax = max_f;
            h.have_forces = true;
            h.last_rms = scc.rms;
            max_f
        });
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let frame = frame0 + step;
        let coords = with_sparse(|g| g.get(name).unwrap().eng.coords().to_vec());
        writeln!(traj, "{}", species.len()).unwrap();
        writeln!(
            traj,
            "frame={frame} E={e_tot:.8} max|F|={max_f:.6e} {note}"
        )
        .unwrap();
        for (sp, c) in species.iter().zip(coords.iter()) {
            writeln!(traj, "{sp:2} {:14.8} {:14.8} {:14.8}", c[0], c[1], c[2]).unwrap();
        }
        traj.flush().unwrap();
        writeln!(
            hist,
            "{frame},{e_tot:.8},{max_f:.6e},{ms:.1},{:.6e},{:.6},{:.6e},{note}",
            scc.rms, scc.tr_ks, scc.r_h
        )
        .unwrap();
        hist.flush().unwrap();
        if max_f < best_f * 0.98 {
            best_f = max_f;
            since_best = 0;
        } else if step > 0 {
            since_best += 1;
        }
        eprintln!(
            "[sparse] step {frame}: E={e_tot:.6} max|F|={max_f:.4e} best={best_f:.4e} stale={since_best} Tr={:.5} {ms:.0} ms {note}",
            scc.tr_ks
        );
        n_done = frame as i64 + 1;
        if step > 0 && max_f < f_tol {
            stopped = format!("max|F|={max_f:.3e} < {f_tol:.1e}");
            break;
        }
        if step > 20 && since_best >= 40 {
            stopped = format!("plateau best max|F|={best_f:.3e}");
            break;
        }
    }
    let after = std::path::Path::new(traj_path).with_file_name("after.xyz");
    let coords = with_sparse(|g| g.get(name).unwrap().eng.coords().to_vec());
    let mut w = std::io::BufWriter::new(
        std::fs::File::create(&after).unwrap_or_else(|e| panic!("after.xyz: {e}")),
    );
    writeln!(w, "{}", species.len()).unwrap();
    writeln!(w, "{stopped}").unwrap();
    for (sp, c) in species.iter().zip(coords.iter()) {
        writeln!(w, "{sp:2} {:14.8} {:14.8} {:14.8}", c[0], c[1], c[2]).unwrap();
    }
    eprintln!("[sparse] relax_traj '{name}' {stopped}  next_frame={n_done}  {traj_path}");
    n_done
}

fn rhai_sparse_set_coords(name: &str, xyz: Array) -> INT {
    let n_atom = with_sparse(|g| {
        g.get(name)
            .unwrap_or_else(|| panic!("sparse_set_coords '{name}': no engine"))
            .eng
            .n_atom()
    });
    if xyz.len() != n_atom * 3 {
        panic!(
            "sparse_set_coords '{name}': xyz len {} != 3*n_atom {}",
            xyz.len(),
            n_atom * 3
        );
    }
    let mut coords = Vec::with_capacity(n_atom);
    for i in 0..n_atom {
        coords.push([
            dyn_f64(&xyz[3 * i], &format!("sparse_set_coords '{name}' x[{i}]")),
            dyn_f64(
                &xyz[3 * i + 1],
                &format!("sparse_set_coords '{name}' y[{i}]"),
            ),
            dyn_f64(
                &xyz[3 * i + 2],
                &format!("sparse_set_coords '{name}' z[{i}]"),
            ),
        ]);
    }
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_set_coords '{name}': no engine"));
        h.eng
            .set_coords(&coords)
            .unwrap_or_else(|e| panic!("sparse_set_coords '{name}': {e}"));
        h.have_forces = false;
    });
    sync_geom_coords(name, &coords);
    eprintln!("[sparse] sparse_set_coords '{name}' n_atom={n_atom}");
    n_atom as INT
}

fn rhai_sparse_n_atoms(name: &str) -> INT {
    with_sparse(|g| {
        g.get(name)
            .unwrap_or_else(|| panic!("sparse_n_atoms '{name}': no engine"))
            .eng
            .n_atom() as INT
    })
}

fn rhai_sparse_n_orbs(name: &str) -> INT {
    with_sparse(|g| {
        g.get(name)
            .unwrap_or_else(|| panic!("sparse_n_orbs '{name}': no engine"))
            .eng
            .n_orbs() as INT
    })
}

fn rhai_sparse_scc_iters(name: &str) -> INT {
    with_sparse(|g| {
        g.get(name)
            .unwrap_or_else(|| panic!("sparse_scc_iters '{name}': no engine"))
            .last_iters
    })
}

fn rhai_sparse_tr_ks(name: &str) -> f64 {
    with_sparse(|g| {
        g.get(name)
            .unwrap_or_else(|| panic!("sparse_tr_ks '{name}': no engine"))
            .eng
            .last_energy()
            .tr_ks as f64
    })
}

/// Relative TC2 tolerance ‖KSK−K‖/‖K‖ for subsequent scc calls.
fn rhai_sparse_tc2_tol(name: &str, tol: f64) {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_tc2_tol '{name}': no engine"));
        h.eng.set_tc2_tol(tol as f32);
        eprintln!("[sparse] sparse_tc2_tol '{name}' tol={tol:e}");
    });
}

fn rhai_sparse_ns_tol(name: &str, tol: f64) {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_ns_tol '{name}': no engine"));
        h.eng.set_ns_tol(tol as f32);
        eprintln!("[sparse] sparse_ns_tol '{name}' tol={tol:e}");
    });
}

/// sparse_purifier(name, "k"|"p"|"trs") — AL1 A/B: select K-TC2 vs P=KS
/// vs TRS4 purifier for subsequent scc calls.
fn rhai_sparse_purifier(name: &str, mode: &str) {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_purifier '{name}': no engine"));
        h.eng.set_purifier(mode);
        eprintln!("[sparse] sparse_purifier '{name}' mode={mode}");
    });
}

/// sparse_eval_p(name) — L3 diagnostic: 2·Tr(P·Z·H_scc), the band energy
/// from P without K=PZ recovery. Panics if the last purifier wasn't P-based.
fn rhai_sparse_eval_p(name: &str) -> f64 {
    with_sparse(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_eval_p '{name}': no engine"));
        let e = h
            .eng
            .band_energy_p()
            .unwrap_or_else(|err| panic!("sparse_eval_p '{name}': {err}"));
        eprintln!("[sparse] sparse_eval_p '{name}' E_band(P)={e:.8}");
        e
    })
}

fn rhai_sparse_charges(name: &str) -> String {
    with_sparse(|g| {
        let h = g
            .get(name)
            .unwrap_or_else(|| panic!("sparse_charges '{name}': no engine"));
        h.eng
            .last_energy()
            .q
            .iter()
            .map(|q| format!("{q:.8}"))
            .collect::<Vec<_>>()
            .join(",")
    })
}

/// Vibrational frequencies from a central-difference Hessian of the sparse
/// analytic forces (manifest Gate G/H machinery — user-level entry point).
///
/// `h` is the displacement step in Å (must be ≪ skin/2 — the Verlet skin
/// guard in `set_coords` enforces this; h = 0.02 Å is the Gate-E plateau).
/// Each column is warm-started: SCC reuses the previous converged charges.
/// The engine is restored to the input geometry and re-converged at the end.
///
/// H[iα,jβ] = −(F_jβ(+h) − F_jβ(−h)) / 2h, symmetrized, mass-weighted with
/// standard atomic weights, diagonalized by dense f64 Jacobi-free symmetric
/// eigensolver (3N ≤ few thousand — dense eig is fine at this scale).
/// ω[cm⁻¹] = sign(λ)·√|λ|·5140.487143 (E_h/Bohr²·amu → cm⁻¹).
///
/// Writes `path` (plain text: one line per mode, "freq_cm1" then mode
/// vector block). Returns a one-line summary (n_imag, lowest/highest freq).
fn rhai_sparse_vibrations(name: &str, h: f64, scc_tol: f64, path: &str) -> String {
    // H is in E_h/Å² (forces are E_h/Å, h in Å). The cm⁻¹ convention
    // ν = √λ·5140.487 assumes λ in E_h/(Bohr²·amu): convert with
    // 1/(ANG2BOHR·√m_amu) per index (m in amu, NOT electron masses).
    const AU_TO_CM: f64 = 5140.487_143;
    let (species, _coords) = geom_species_coords(name, "sparse_vibrations");
    let masses: Vec<f64> = species
        .iter()
        .map(|s| {
            Element::from_symbol(s)
                .unwrap_or_else(|| panic!("sparse_vibrations '{name}': unknown element {s}"))
                .mass()
        })
        .collect();

    with_sparse(|g| {
        let hnd = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("sparse_vibrations '{name}': no engine — sparse_new first"));
        let eng = &mut hnd.eng;
        let n_atom = eng.n_atom();
        let n3 = 3 * n_atom;
        let x0 = eng.coords().to_vec();
        if !(h > 0.0) || h > 0.5 {
            panic!("sparse_vibrations '{name}': h={h} Å must be in (0, 0.5] — skin/2 guard needs h < skin/2");
        }

        let mut hess = vec![0.0f64; n3 * n3];
        let mut work = x0.clone();
        let env_on = |name: &str| {
            std::env::var(name)
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        };
        let frozen = env_on("RUST_DFTB_VIB_FROZEN");
        let fixq = env_on("RUST_DFTB_VIB_FIXQ") && !frozen;
        let lite = env_on("RUST_DFTB_VIB_LITE");
        // F1 batched frozen columns (manifest §F.1) + F5a batched
        // fixed-iteration DMM-lite fixq columns: RUST_DFTB_VIB_BATCH
        // evals in flight per launch — both tiers are uniform
        // fixed-cost jobs, no SCC machinery. GPU-pair path only.
        // fixq WITHOUT lite stays scalar: purify convergence varies per
        // replica and needs the active-mask scheduler (manifest §F.1 F5b).
        let dmm_batch = fixq && lite;
        let vib_batch = if frozen || dmm_batch {
            std::env::var("RUST_DFTB_VIB_BATCH")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1)
                .max(1)
        } else {
            if fixq && std::env::var("RUST_DFTB_VIB_BATCH").is_ok() {
                eprintln!("[sparse] vibrations '{name}': RUST_DFTB_VIB_BATCH ignored for non-lite fixq — variable purify convergence needs the scheduler; set RUST_DFTB_VIB_LITE=1 for the fixed-cost DMM tier");
            }
            1
        };
        // F5a recipe knobs — same envs as the scalar lite path.
        let dmm_ns = std::env::var("RUST_DFTB_VIB_NSMAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let dmm_steps = std::env::var("RUST_DFTB_VIB_DMM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(6);
        let dmm_eta = std::env::var("RUST_DFTB_VIB_DMM_ETA")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(8.0);
        let batched = vib_batch > 1 && (frozen || dmm_batch) && !eng.cpu_pair();
        // Central electronic state (manifest §4.12 / Phase G1): every ±h
        // column restores it — identical solver history, no FD asymmetry
        // from a chained previous column.
        eng.snapshot_electronic_state()
            .unwrap_or_else(|e| panic!("vibrations '{name}' snapshot: {e}"));
        // Policy (manifest §4.12.1): never measure a full Hessian —
        // RUST_DFTB_VIB_MAXCOL bounds the column count; per-column phase
        // timings are always printed so a handful of columns suffices to
        // extrapolate. A bounded run exits before the eigensolve.
        let maxcol = std::env::var("RUST_DFTB_VIB_MAXCOL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        eprintln!("[sparse] vibrations '{name}': {n3} columns, h={h} Å, scc_tol={scc_tol:e} frozen={frozen} fixq={fixq} lite={lite} batch={vib_batch} dmm={{ns:{dmm_ns} steps:{dmm_steps} eta:{dmm_eta}}} maxcol={maxcol:?}");
        let t_vib0 = std::time::Instant::now();
        if batched {
            // F1 batched frozen columns (manifest §F.1): evals in column
            // order (+h,−h adjacent), chunked by vib_batch. JobId = eval
            // index — results scatter to columns, never by slot.
            let mut evals: Vec<(usize, usize, f64)> = Vec::new();
            'collect: for i in 0..n_atom {
                for a in 0..3 {
                    let col = 3 * i + a;
                    if let Some(mc) = maxcol {
                        if col >= mc {
                            break 'collect;
                        }
                    }
                    evals.push((i, a, 1.0));
                    evals.push((i, a, -1.0));
                }
            }
            let fmax = |v: &Vec<[f64; 3]>| {
                v.iter()
                    .flat_map(|a| a.iter())
                    .fold(0.0f64, |m, x| m.max(x.abs()))
            };
            // Per-column (f_plus, f_minus) pending slots — at most one
            // column straddles a chunk boundary.
            let mut pend: std::collections::HashMap<usize, [Option<Vec<[f64; 3]>>; 2]> =
                std::collections::HashMap::new();
            for chunk in evals.chunks(vib_batch) {
                let t0 = std::time::Instant::now();
                let fs = if frozen {
                    eng.forces_frozen_batch(&x0, chunk, h)
                        .unwrap_or_else(|e| panic!("vibrations '{name}' batched frozen evals: {e}"))
                } else {
                    eng.forces_dmm_batch(&x0, chunk, h, dmm_ns, dmm_steps, dmm_eta)
                        .unwrap_or_else(|e| panic!("vibrations '{name}' batched dmm evals: {e}"))
                };
                let ms_ev = t0.elapsed().as_secs_f64() * 1e3 / chunk.len() as f64;
                for (&(i, a, sign), f) in chunk.iter().zip(fs.into_iter()) {
                    let col = 3 * i + a;
                    let si = if sign > 0.0 { 0 } else { 1 };
                    let e = pend.entry(col).or_insert_with(|| [None, None]);
                    e[si] = Some(f.forces);
                    if e[0].is_some() && e[1].is_some() {
                        let [fp, fm] = pend.remove(&col).unwrap();
                        let fp = fp.unwrap();
                        let fm = fm.unwrap();
                        let mut fd_max = 0.0f64;
                        for j in 0..n_atom {
                            for b in 0..3 {
                                fd_max = fd_max.max((fp[j][b] - fm[j][b]).abs());
                            }
                        }
                        eprintln!("[sparse] vib col {col}/{n3} (B={vib_batch}): {ms_ev:.2}ms/eval | max|F+|={:.4e} max|F-|={:.4e} max|ΔF|={:.4e}",
                            fmax(&fp), fmax(&fm), fd_max);
                        if let Ok(dir) = std::env::var("RUST_DFTB_VIB_DUMPCOL") {
                            let mut txt = String::new();
                            for j in 0..n_atom {
                                for b in 0..3 {
                                    txt.push_str(&format!("{:.10e}\n", fp[j][b] - fm[j][b]));
                                }
                            }
                            std::fs::write(format!("{dir}/df_{col}.txt"), txt)
                                .unwrap_or_else(|e| panic!("vib dumpcol {dir}/df_{col}.txt: {e}"));
                        }
                        for j in 0..n_atom {
                            for b in 0..3 {
                                hess[(3 * j + b) * n3 + col] = -(fp[j][b] - fm[j][b]) / (2.0 * h);
                            }
                        }
                    }
                }
            }
        } else {
            'cols: for i in 0..n_atom {
                for a in 0..3 {
                    let col = 3 * i + a;
                    if let Some(mc) = maxcol {
                        if col >= mc {
                            break 'cols;
                        }
                    }
                    let mut f_plus = None;
                    let mut f_minus = None;
                    let mut ms = [0.0f64; 6]; // [+h: rst,set,f] [-h: rst,set,f]
                    for (si, sign) in [1.0f64, -1.0].iter().enumerate() {
                        let sign = *sign;
                        let t0 = std::time::Instant::now();
                        eng.restore_central_state().unwrap_or_else(|e| {
                            panic!("vibrations '{name}' col {col} restore: {e}")
                        });
                        work[i][a] = x0[i][a] + sign * h;
                        eng.set_coords(&work).unwrap_or_else(|e| {
                            panic!("vibrations '{name}' col {col} {sign:+}h set_coords: {e}")
                        });
                        let t1 = std::time::Instant::now();
                        let f = if frozen {
                            eng.forces_frozen()
                            .unwrap_or_else(|e| {
                                panic!("vibrations '{name}' col {col} {sign:+}h forces_frozen: {e}")
                            })
                            .forces
                        } else if fixq {
                            eng.scc_fixedq().unwrap_or_else(|e| {
                                panic!("vibrations '{name}' col {col} {sign:+}h scc_fixedq: {e}")
                            });
                            eng.forces()
                                .unwrap_or_else(|e| {
                                    panic!("vibrations '{name}' col {col} {sign:+}h forces: {e}")
                                })
                                .forces
                        } else {
                            eng.scc(80, scc_tol).unwrap_or_else(|e| {
                                panic!("vibrations '{name}' col {col} {sign:+}h scc: {e}")
                            });
                            eng.forces()
                                .unwrap_or_else(|e| {
                                    panic!("vibrations '{name}' col {col} {sign:+}h forces: {e}")
                                })
                                .forces
                        };
                        let t2 = std::time::Instant::now();
                        ms[3 * si] = (t1 - t0).as_secs_f64() * 1e3;
                        ms[3 * si + 1] = 0.0;
                        ms[3 * si + 2] = (t2 - t1).as_secs_f64() * 1e3;
                        if sign > 0.0 {
                            f_plus = Some(f);
                        } else {
                            f_minus = Some(f);
                        }
                    }
                    // Per-column force magnitudes — the cross-mode FD check
                    // (warm-seed acceptance changes forces, not just timing).
                    let fmax = |v: &Vec<[f64; 3]>| {
                        v.iter()
                            .flat_map(|a| a.iter())
                            .fold(0.0f64, |m, x| m.max(x.abs()))
                    };
                    let fp = f_plus.as_ref().unwrap();
                    let fm = f_minus.as_ref().unwrap();
                    let mut fd_max = 0.0f64;
                    for j in 0..n_atom {
                        for b in 0..3 {
                            fd_max = fd_max.max((fp[j][b] - fm[j][b]).abs());
                        }
                    }
                    eprintln!("[sparse] vib col {col}/{n3}: +h rst+geom={:.1}ms solve+f={:.1}ms | -h rst+geom={:.1}ms solve+f={:.1}ms | max|F+|={:.4e} max|F-|={:.4e} max|ΔF|={:.4e}",
                    ms[0], ms[2], ms[3], ms[5], fmax(fp), fmax(fm), fd_max);
                    // Optional per-column ΔF dump for offline warm-vs-cold
                    // validation (RUST_DFTB_VIB_DUMPCOL=<dir>): one text file
                    // per column, n_atom lines of "fx fy fz" (F+−F−).
                    if let Ok(dir) = std::env::var("RUST_DFTB_VIB_DUMPCOL") {
                        let mut txt = String::new();
                        for j in 0..n_atom {
                            for b in 0..3 {
                                txt.push_str(&format!("{:.10e}\n", fp[j][b] - fm[j][b]));
                            }
                        }
                        std::fs::write(format!("{dir}/df_{col}.txt"), txt)
                            .unwrap_or_else(|e| panic!("vib dumpcol {dir}/df_{col}.txt: {e}"));
                    }
                    work[i][a] = x0[i][a];
                    let f_plus = f_plus.unwrap();
                    let f_minus = f_minus.unwrap();
                    for j in 0..n_atom {
                        for b in 0..3 {
                            hess[(3 * j + b) * n3 + col] =
                                -(f_plus[j][b] - f_minus[j][b]) / (2.0 * h);
                        }
                    }
                    if col % 6 == 0 {
                        eprintln!("[sparse] vibrations '{name}': col {col}/{n3}");
                    }
                }
            }
        }
        if let Some(mc) = maxcol {
            let mc = mc.min(n3);
            return format!("vibrations '{name}': PARTIAL {mc} columns in {:.1}s — measurement run, no eigensolve",
                t_vib0.elapsed().as_secs_f64());
        }
        // Restore input geometry + reconverge so the engine's state is
        // consistent — skipped when batched (the batched evals never
        // mutated engine state; it is still the central snapshot).
        if !batched {
            eng.set_coords(&x0)
                .unwrap_or_else(|e| panic!("vibrations '{name}' restore set_coords: {e}"));
            eng.scc(80, scc_tol)
                .unwrap_or_else(|e| panic!("vibrations '{name}' restore scc: {e}"));
        }
        sync_geom_coords(name, &x0);

        // Symmetrize (FD noise + truncation asymmetry — report, don't hide).
        let mut max_asym = 0.0f64;
        for r in 0..n3 {
            for c in (r + 1)..n3 {
                let a = hess[r * n3 + c];
                let b = hess[c * n3 + r];
                max_asym = max_asym.max((a - b).abs());
                let m = 0.5 * (a + b);
                hess[r * n3 + c] = m;
                hess[c * n3 + r] = m;
            }
        }
        eprintln!("[sparse] vibrations '{name}': Hessian max asymmetry = {max_asym:.3e}");

        // Mass-weight: M_ij = H_ij[E_h/Bohr²] / sqrt(m_i m_j), masses in amu.
        let inv: Vec<f64> = (0..n_atom)
            .flat_map(|i| std::iter::repeat(1.0 / (ANG2BOHR * masses[i].sqrt())).take(3))
            .collect();
        let mut mw = vec![0.0f64; n3 * n3];
        for r in 0..n3 {
            for c in 0..n3 {
                mw[r * n3 + c] = hess[r * n3 + c] * inv[r] * inv[c];
            }
        }
        let dm = nalgebra::DMatrix::from_row_slice(n3, n3, &mw);
        let t_eig = std::time::Instant::now();
        let (evals, evecs) = rust_dftb::core::eigh::symmetric_eigh(dm).unwrap_or_else(|e| {
            panic!("vibrations '{name}' symmetric_eigh n3={n3}: {e}")
        });
        eprintln!(
            "[sparse] vibrations '{name}': eigh n3={n3} in {:.3}s (dsyevd)",
            t_eig.elapsed().as_secs_f64()
        );
        let mut vals: Vec<f64> = evals.iter().copied().collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut freqs: Vec<f64> = Vec::with_capacity(n3);
        let mut n_imag = 0usize;
        for &lam in &vals {
            if lam < 0.0 {
                n_imag += 1;
                freqs.push(-(-lam).sqrt() * AU_TO_CM);
            } else {
                freqs.push(lam.sqrt() * AU_TO_CM);
            }
        }

        // Write: freqs then mass-weighted eigenmodes (columns of eigvecs).
        let mut out = String::new();
        out.push_str(&format!("# sparse vibrations '{name}' n_atom={n_atom} h={h} scc_tol={scc_tol:e} max_asym={max_asym:.3e}\n"));
        for (k, &f) in freqs.iter().enumerate() {
            out.push_str(&format!("{k} {f:.4}\n"));
        }
        let mut order: Vec<usize> = (0..n3).collect();
        order.sort_by(|&a, &b| evals[a].partial_cmp(&evals[b]).unwrap());
        for (k, &col) in order.iter().enumerate() {
            out.push_str(&format!("mode {k} freq {:.4}\n", freqs[k]));
            // Mass-weighted eigenvector → real-space displacement, unit-normed.
            let mut un = 0.0f64;
            let mut u = vec![0.0f64; n3];
            for c in 0..n3 {
                u[c] = evecs[(c, col)] * inv[c] * ANG2BOHR;
                un += u[c] * u[c];
            }
            let un = un.sqrt().max(1e-30);
            for i in 0..n_atom {
                out.push_str(&format!(
                    "  {:.8} {:.8} {:.8}\n",
                    u[3 * i] / un,
                    u[3 * i + 1] / un,
                    u[3 * i + 2] / un
                ));
            }
        }
        std::fs::write(path, &out)
            .unwrap_or_else(|e| panic!("sparse_vibrations '{name}' write {path}: {e}"));

        let lo = freqs.first().copied().unwrap_or(f64::NAN);
        let hi = freqs.last().copied().unwrap_or(f64::NAN);
        let summary = format!(
            "n3={n3} n_imag={n_imag} freq_min={lo:.2} freq_max={hi:.2} cm-1 (written {path})"
        );
        eprintln!("[sparse] vibrations '{name}': {summary}");
        summary
    })
}

fn dyn_f64(d: &Dynamic, ctx: &str) -> f64 {
    if let Ok(x) = d.as_float() {
        return x;
    }
    if let Ok(x) = d.as_int() {
        return x as f64;
    }
    panic!("{ctx}: expected number, got {d}");
}

fn nano_from_species_coords(species: &[String], coords: &[[f64; 3]], ctx: &str) -> NanoStructure {
    if species.len() != coords.len() {
        panic!(
            "{ctx}: species {} != coords {}",
            species.len(),
            coords.len()
        );
    }
    let mut st = NanoStructure::new();
    for (sp, xyz) in species.iter().zip(coords.iter()) {
        let el = Element::from_symbol(sp).unwrap_or_else(|| panic!("{ctx}: unknown element {sp}"));
        st.elements.push(el);
        st.positions.push(*xyz);
    }
    st
}

/// Load an XYZ file into the geometry registry.
fn rhai_load_xyz(name: &str, path: &str) -> INT {
    let mol = parse_xyz(path).unwrap_or_else(|e| panic!("load_xyz '{name}' path={path}: {e}"));
    let n = mol.species.len();
    if n == 0 {
        panic!("load_xyz '{name}': empty XYZ {path}");
    }
    let st = nano_from_species_coords(&mol.species, &mol.coords, &format!("load_xyz '{name}'"));
    with_state(|s| {
        s.geometries.insert(name.to_string(), st);
    });
    eprintln!("[gpu] load_xyz '{name}' {n} atoms from {path}");
    n as INT
}

/// Build a geometry from species CSV + flat xyz array (Å).
fn rhai_make_geom(name: &str, species_csv: &str, xyz: Array) -> INT {
    let species = parse_species(species_csv);
    if species.is_empty() {
        panic!("make_geom '{name}': empty species");
    }
    if xyz.len() != species.len() * 3 {
        panic!(
            "make_geom '{name}': xyz len {} != 3*n_atoms {}",
            xyz.len(),
            species.len() * 3
        );
    }
    let mut coords = Vec::with_capacity(species.len());
    for i in 0..species.len() {
        coords.push([
            dyn_f64(&xyz[3 * i], &format!("make_geom '{name}' x[{i}]")),
            dyn_f64(&xyz[3 * i + 1], &format!("make_geom '{name}' y[{i}]")),
            dyn_f64(&xyz[3 * i + 2], &format!("make_geom '{name}' z[{i}]")),
        ]);
    }
    let n = species.len();
    let st = nano_from_species_coords(&species, &coords, &format!("make_geom '{name}'"));
    with_state(|s| {
        s.geometries.insert(name.to_string(), st);
    });
    eprintln!("[gpu] make_geom '{name}' {n} atoms");
    n as INT
}

/// Create / replace a GpuDftb for a stored geometry. `batch` copies of the same molecule.
fn rhai_gpu_new(name: &str, sk_dir: &str, batch: INT) -> INT {
    let batch = batch as usize;
    if batch == 0 {
        panic!("gpu_new '{name}': batch=0");
    }
    let (species, xyz0) = with_state(|s| {
        let st = s
            .geometries
            .get(name)
            .unwrap_or_else(|| panic!("gpu_new '{name}': no geometry — load_xyz/make_geom first"));
        let species: Vec<String> = st.elements.iter().map(|e| e.symbol().to_string()).collect();
        (species, st.positions.clone())
    });
    let n_atoms = species.len();
    let mut coords = Vec::with_capacity(batch * n_atoms);
    for _ in 0..batch {
        coords.extend_from_slice(&xyz0);
    }
    eprintln!("[gpu] gpu_new '{name}' n_atoms={n_atoms} batch={batch} sk={sk_dir}");
    let sk = load_sk_for_species(sk_dir, &species)
        .unwrap_or_else(|e| panic!("gpu_new '{name}' load SK from {sk_dir}: {e}"));
    let eng = GpuDftb::new(sk, sk_dir, species, coords, batch)
        .unwrap_or_else(|e| panic!("gpu_new '{name}' GpuDftb::new: {e}"));
    let n_orbs = eng.n() as INT;
    eprintln!(
        "[gpu] gpu_new '{name}' N={n_orbs} device={}",
        eng.rt.caps().name
    );
    with_gpu(|g| {
        g.insert(
            name.to_string(),
            GpuHandle {
                eng,
                last_e: Vec::new(),
                last_f: None,
                last_rms: f32::NAN,
                last_q_rms: f64::NAN,
                last_q_max: f64::NAN,
                last_iters: 0,
                last_stalled: false,
                last_status: Vec::new(),
            },
        );
    });
    n_orbs
}

fn rhai_gpu_scc(name: &str, max_iter: INT, tol: f64) -> f64 {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_scc '{name}': no engine — gpu_new first"));
        let scc = h
            .eng
            .scc_with_retry(max_iter as usize, tol as f32) // W11: explicit warm-start retry
            .unwrap_or_else(|e| panic!("gpu_scc '{name}': {e}"));
        h.last_rms = scc.rms;
        h.last_iters = scc.n_iters as i64;
        h.last_stalled = scc.stalled;
        h.last_status = scc
            .statuses
            .iter()
            .map(|s| format!("{s:?}").to_lowercase())
            .collect();
        eprintln!(
            "[gpu] gpu_scc '{name}' rms={:.3e} iters={} stalled={}",
            scc.rms, scc.n_iters, scc.stalled
        );
        scc.rms as f64
    })
}

/// One finalize. `want_forces=true` also builds W and F on GPU. Returns replica-0 energy (Ha).
fn rhai_gpu_eval(name: &str, want_forces: bool) -> f64 {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_eval '{name}': no engine — gpu_new first"));
        let ev = h
            .eng
            .eval(want_forces)
            .unwrap_or_else(|e| panic!("gpu_eval '{name}' want_forces={want_forces}: {e}"));
        for (i, &e) in ev.energy.iter().enumerate() {
            if !e.is_finite() {
                panic!("gpu_eval '{name}': E[{i}]={e} non-finite");
            }
        }
        if let Some(ref f) = ev.forces {
            for (i, &x) in f.iter().enumerate() {
                if !x.is_finite() {
                    panic!("gpu_eval '{name}': F[{i}]={x} non-finite");
                }
            }
        }
        h.last_e = ev.energy;
        h.last_f = ev.forces;
        h.last_q_rms = ev.q_rms;
        h.last_q_max = ev.q_max;
        let e0 = h.last_e[0];
        let mut max_f = 0.0f32;
        if let Some(ref f) = h.last_f {
            for &x in f {
                max_f = max_f.max(x.abs());
            }
        }
        eprintln!("[gpu] gpu_eval '{name}' want_forces={want_forces} n_batch={} q_rms={:.3e} q_max={:.3e} max|F|={:.4e}",
            h.last_e.len(), h.last_q_rms, h.last_q_max, if h.last_f.is_some() { max_f } else { f32::NAN });
        for (i, &e) in h.last_e.iter().enumerate() {
            eprintln!("[gpu]   E[{i}]={e:.12} Ha");
        }
        e0
    })
}

fn rhai_gpu_energy_i(name: &str, i: INT) -> f64 {
    with_gpu(|g| {
        let h = g
            .get(name)
            .unwrap_or_else(|| panic!("gpu_energy_i '{name}': no engine"));
        let i = i as usize;
        if h.last_e.is_empty() {
            panic!("gpu_energy_i '{name}': call gpu_eval first");
        }
        if i >= h.last_e.len() {
            panic!("gpu_energy_i '{name}': i={i} >= batch {}", h.last_e.len());
        }
        h.last_e[i]
    })
}

// ─── GpuPbc rhai bindings: periodic batched engine (replicas share one cell) ──

/// pbc_new(name, sk_dir, lat9, kpts_flat, kw, batch) -> n_orbs
/// lat9 = 3 lattice vectors flattened row-major [Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz] (Å);
/// kpts_flat = [kx,ky,kz, ...] fractional; kw = weights (Σ=1).
/// `batch` replicas of the loaded geometry share the cell — per-replica
/// coordinates are set by pbc_set_coords.
fn rhai_pbc_new(name: &str, sk_dir: &str, lat: Array, kpts: Array, kw: Array, batch: INT) -> INT {
    use rust_dftb::methods::dftb::forces::parse_all_repulsive;
    let batch = batch as usize;
    if batch == 0 {
        panic!("pbc_new '{name}': batch=0");
    }
    let (species, xyz0) = geom_species_coords(name, "pbc_new");
    let n_atoms = species.len();
    let mut coords = Vec::with_capacity(batch * n_atoms);
    for _ in 0..batch {
        coords.extend_from_slice(&xyz0);
    }
    if lat.len() != 9 {
        panic!("pbc_new '{name}': lat needs 9 numbers, got {}", lat.len());
    }
    let mut latm = [[0.0f64; 3]; 3];
    for i in 0..9 {
        latm[i / 3][i % 3] = dyn_f64(&lat[i], "pbc_new lat");
    }
    let nk = kw.len();
    if nk == 0 || kpts.len() != 3 * nk {
        panic!("pbc_new '{name}': nk={nk} vs kpts len {}", kpts.len());
    }
    let mut k_frac = Vec::with_capacity(nk);
    for k in 0..nk {
        k_frac.push([
            dyn_f64(&kpts[3 * k], "pbc_new kx"),
            dyn_f64(&kpts[3 * k + 1], "pbc_new ky"),
            dyn_f64(&kpts[3 * k + 2], "pbc_new kz"),
        ]);
    }
    let kwv: Vec<f32> = kw.iter().map(|d| dyn_f64(d, "pbc_new kw") as f32).collect();
    let wsum: f32 = kwv.iter().sum();
    if (wsum - 1.0).abs() > 1e-3 {
        panic!("pbc_new '{name}': k weights sum {wsum} != 1");
    }
    eprintln!(
        "[pbc] pbc_new '{name}' n_atoms={n_atoms} batch={batch} nk={nk} lat={latm:?} sk={sk_dir}"
    );
    let sk = load_sk_for_species(sk_dir, &species)
        .unwrap_or_else(|e| panic!("pbc_new '{name}' load SK from {sk_dir}: {e}"));
    let eng = rust_dftb::qmqm::gpu_pbc::GpuPbc::new(
        sk,
        species.clone(),
        coords.clone(),
        latm,
        &k_frac,
        &kwv,
        None,
    )
    .unwrap_or_else(|e| panic!("pbc_new '{name}' GpuPbc::new: {e}"));
    let n_orbs = eng.dims().0 as INT;
    eprintln!(
        "[pbc] pbc_new '{name}' N={n_orbs} device={}",
        eng.rt.caps().name
    );
    // repulsive splines + species codes for host-side E_rep in pbc_eval
    let mut names: Vec<String> = Vec::new();
    for s in &species {
        if !names.iter().any(|n| n == s) {
            names.push(s.clone());
        }
    }
    let repulsive = parse_all_repulsive(sk_dir, &names, names.len())
        .unwrap_or_else(|e| panic!("pbc_new '{name}' repulsive tables: {e}"));
    let species_code: Vec<u8> = species
        .iter()
        .map(|s| names.iter().position(|n| n == s).unwrap() as u8)
        .collect();
    with_pbc(|g| {
        g.insert(
            name.to_string(),
            PbcHandle {
                eng,
                lat: latm,
                coords,
                repulsive,
                species_names: names,
                species_code,
                last_e: Vec::new(),
                last_rms: f32::NAN,
                last_iters: 0,
            },
        );
    });
    n_orbs
}

/// pbc_set_coords(name, xyz_flat) -> n_atoms — upload per-replica
/// geometries (n_rep*n_atoms*3), rebuild H0(k)/S(k)/γ + Löwdin prep.
fn rhai_pbc_set_coords(name: &str, xyz: Array) -> INT {
    let (batch, n_atoms) = with_pbc(|g| {
        let h = g
            .get(name)
            .unwrap_or_else(|| panic!("pbc_set_coords '{name}': no engine — pbc_new first"));
        let (_, n_atoms, n_rep, _) = h.eng.dims();
        (n_rep, n_atoms)
    });
    let need = 3 * batch * n_atoms;
    if xyz.len() != need {
        panic!(
            "pbc_set_coords '{name}': xyz len {} != 3*batch*n_atoms = 3*{batch}*{n_atoms} = {need}",
            xyz.len()
        );
    }
    let mut coords: Vec<[f64; 3]> = Vec::with_capacity(batch * n_atoms);
    for i in 0..batch * n_atoms {
        coords.push([
            dyn_f64(&xyz[3 * i], "pbc_set_coords x"),
            dyn_f64(&xyz[3 * i + 1], "pbc_set_coords y"),
            dyn_f64(&xyz[3 * i + 2], "pbc_set_coords z"),
        ]);
    }
    with_pbc(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("pbc_set_coords '{name}': no engine"));
        h.eng
            .set_geometry(&coords)
            .unwrap_or_else(|e| panic!("pbc_set_coords '{name}': {e}"));
        h.coords = coords;
        h.last_e.clear();
    });
    n_atoms as INT
}

/// pbc_scc(name, max_iter, tol) -> rms — one batched DIIS-mixed SCC
/// (alpha=0.3, the tested value). Resets to neutral charges each call.
fn rhai_pbc_scc(name: &str, max_iter: INT, tol: f64) -> f64 {
    with_pbc(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("pbc_scc '{name}': no engine — pbc_new first"));
        let (ok, hist) = h
            .eng
            .scc(0.3, tol as f32, max_iter as usize)
            .unwrap_or_else(|e| panic!("pbc_scc '{name}': {e}"));
        h.last_rms = hist.last().copied().unwrap_or(f32::NAN);
        h.last_iters = hist.len() as i64;
        let n_bad = ok.iter().filter(|&&b| !b).count();
        eprintln!(
            "[pbc] pbc_scc '{name}' rms={:.3e} iters={} uncertified={}",
            h.last_rms, h.last_iters, n_bad
        );
        h.last_rms as f64
    })
}

/// pbc_smearing(name, kT_Ha) — Fermi smearing for occupations.
fn rhai_pbc_smearing(name: &str, kT: f64) {
    with_pbc(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("pbc_smearing '{name}': no engine"));
        h.eng.plan.kT = kT as f32;
        eprintln!("[pbc] pbc_smearing '{name}' kT={kT}");
    })
}

/// pbc_eval(name) -> E[0] — finalize + energy per replica
/// (band + SCC + repulsive). Stores into last_e for pbc_energy_i.
fn rhai_pbc_eval(name: &str) -> f64 {
    with_pbc(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("pbc_eval '{name}': no engine — pbc_new first"));
        let n_occ = h.eng.n_occ();
        let e_el = h
            .eng
            .plan
            .compute_energy(&mut h.eng.rt, n_occ)
            .unwrap_or_else(|e| panic!("pbc_eval '{name}': {e}"));
        let (_, n_atoms, n_rep, _) = h.eng.dims();
        let mut e_tot = vec![0.0f64; n_rep];
        for r in 0..n_rep {
            let rc = &h.coords[r * n_atoms..(r + 1) * n_atoms];
            let e_rep = rust_dftb::methods::dftb::forces::repulsive_energy_pbc(
                rc,
                &h.species_code,
                &h.species_names,
                &h.repulsive,
                h.species_names.len(),
                &h.lat,
            )
            .unwrap_or_else(|e| panic!("pbc_eval '{name}' E_rep[{r}]: {e}"));
            e_tot[r] = e_el[r] + e_rep;
            if !e_tot[r].is_finite() {
                panic!(
                    "pbc_eval '{name}': E[{r}]={} non-finite (el={} rep={e_rep})",
                    e_tot[r], e_el[r]
                );
            }
        }
        h.last_e = e_tot;
        eprintln!(
            "[pbc] pbc_eval '{name}' n_rep={n_rep} E[0]={:.12} Ha",
            h.last_e[0]
        );
        h.last_e[0]
    })
}

fn rhai_pbc_energy_i(name: &str, i: INT) -> f64 {
    with_pbc(|g| {
        let h = g
            .get(name)
            .unwrap_or_else(|| panic!("pbc_energy_i '{name}': no engine"));
        let i = i as usize;
        if h.last_e.is_empty() {
            panic!("pbc_energy_i '{name}': call pbc_eval first");
        }
        if i >= h.last_e.len() {
            panic!("pbc_energy_i '{name}': i={i} >= batch {}", h.last_e.len());
        }
        h.last_e[i]
    })
}

/// gpu_fire_step(name, f_tol) -> max|F| — one FIRE step on all replicas;
/// the engine's SCC state is consumed internally (call after gpu_scc or
/// standalone — it runs its own eval).
fn rhai_gpu_fire_step(name: &str, f_tol: f64) -> f64 {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_fire_step '{name}': no engine — gpu_new first"));
        h.eng
            .fire_step(f_tol)
            .unwrap_or_else(|e| panic!("gpu_fire_step '{name}': {e}"))
    })
}

/// gpu_relax(name, max_steps, f_tol, scc_tol) -> max|F| — FIRE loop until
/// converged or max_steps; prints unbuffered progress.
fn rhai_gpu_relax(name: &str, max_steps: INT, f_tol: f64, scc_tol: f64) -> f64 {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_relax '{name}': no engine"));
        let (n, max_f, rms, conv) = h
            .eng
            .relax(max_steps as usize, f_tol, scc_tol as f32)
            .unwrap_or_else(|e| panic!("gpu_relax '{name}': {e}"));
        eprintln!(
            "[gpu] gpu_relax '{name}' steps={n} max|F|={max_f:.4e} rms={rms:.3e} converged={conv}"
        );
        max_f
    })
}

/// gpu_bench(name, n_runs, max_iter, rms_tol) -> avg ms per SCC call.
/// Times the production GpuDftb path end-to-end (reset_q + scc per run,
/// wall clock — includes queue sync). Replaces the legacy
/// tests/gpu_scc_bench.rs GpuDriver benchmark.
fn rhai_gpu_bench(name: &str, n_runs: INT, max_iter: INT, rms_tol: f64) -> f64 {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_bench '{name}': no engine — gpu_new first"));
        let mut total_ms = 0.0;
        for r in 0..n_runs {
            h.eng
                .reset_q0()
                .unwrap_or_else(|e| panic!("gpu_bench '{name}' reset_q0: {e}"));
            let t0 = std::time::Instant::now();
            let scc = h
                .eng
                .scc(max_iter as usize, rms_tol as f32)
                .unwrap_or_else(|e| panic!("gpu_bench '{name}' scc run {r}: {e}"));
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            total_ms += dt;
            eprintln!(
                "[gpu_bench] {name} run {r}: {dt:.2} ms ({} iters, rms={:.3e})",
                scc.n_iters, scc.rms
            );
        }
        let avg = total_ms / n_runs as f64;
        eprintln!("[gpu_bench] {name} avg={avg:.3} ms/scc over {n_runs} runs");
        avg
    })
}

/// gpu_smearing(name, kT_ha) — Fermi smearing in Hartree (0 = integer occ).
/// Stabilizes SCC at near-degenerate HOMO/LUMO (mid proton transfer):
/// fractional occupation removes the O(1) Δq oscillation.
fn rhai_gpu_smearing(name: &str, kT: f64) {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_smearing '{name}': no engine"));
        h.eng.set_smearing(kT as f32);
    });
}

/// gpu_freeze_atoms(name, [i0,i1,...]) — pin template atoms for ALL
/// replicas (constrained scans: the transferred proton etc.). Forces,
/// velocities, and the per-replica convergence test ignore them.
fn rhai_gpu_freeze_atoms(name: &str, idx: Array) {
    let ids: Vec<usize> = idx
        .iter()
        .map(|d| dyn_f64(d, &format!("gpu_freeze_atoms '{name}' idx")) as usize)
        .collect();
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_freeze_atoms '{name}': no engine"));
        h.eng
            .set_frozen_atoms(&ids)
            .unwrap_or_else(|e| panic!("gpu_freeze_atoms '{name}': {e}"));
    });
    eprintln!("[gpu] gpu_freeze_atoms '{name}' frozen={ids:?}");
}

/// gpu_set_constraint(name, i, j, [d0,d1,...]) — per-replica distance
/// constraint |x_j − x_i| = d[b] Å (the relaxed-scan coordinate; e.g. GC
/// proton transfer: i=8 donor N1, j=13 transferring H, d = N–H distance).
/// Applied inside fire_apply_batched as a closed-form equal-mass
/// projection; a frozen endpoint takes weight 0.
fn rhai_gpu_set_constraint(name: &str, i: INT, j: INT, targets: Array) {
    let d: Vec<f64> = targets
        .iter()
        .map(|x| dyn_f64(x, &format!("gpu_set_constraint '{name}' d")))
        .collect();
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_set_constraint '{name}': no engine"));
        h.eng
            .set_constraint(i as usize, j as usize, &d)
            .unwrap_or_else(|e| panic!("gpu_set_constraint '{name}': {e}"));
    });
    eprintln!("[gpu] gpu_set_constraint '{name}' |x_{j}−x_{i}|=d per replica");
}

/// gpu_clear_constraint(name) — remove the distance constraint.
fn rhai_gpu_clear_constraint(name: &str) {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_clear_constraint '{name}': no engine"));
        h.eng
            .clear_constraint()
            .unwrap_or_else(|e| panic!("gpu_clear_constraint '{name}': {e}"));
    });
}

/// gpu_cdft(name, frag, targets) — attach fragment Mulliken-charge
/// constraints (Dense_Multi_CDFT). `frag` = per-template-atom fragment
/// id (−1 = unconstrained); `targets` = flat [batch*nfrag] excess-charge
/// targets in e, or [nfrag] broadcast to all replicas. Per-replica
/// targets turn one batch into a diabatic-state ladder (PCET scans).
/// Returns nfrag.
fn rhai_gpu_cdft(name: &str, frag: Array, targets: Array) -> INT {
    let fm: Vec<i32> = frag
        .iter()
        .map(|d| dyn_f64(d, &format!("gpu_cdft '{name}' frag")) as i32)
        .collect();
    let mut tg: Vec<f64> = targets
        .iter()
        .map(|d| dyn_f64(d, &format!("gpu_cdft '{name}' targets")))
        .collect();
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_cdft '{name}': no engine"));
        let batch = h.eng.batch();
        if fm.len() != h.eng.n_atoms() {
            panic!(
                "gpu_cdft '{name}': frag.len()={} != n_atoms={}",
                fm.len(),
                h.eng.n_atoms()
            );
        }
        let nfrag = fm.iter().copied().max().unwrap_or(-1) + 1;
        if nfrag <= 0 {
            panic!("gpu_cdft '{name}': no atom in any fragment");
        }
        let nfrag = nfrag as usize;
        if tg.len() == nfrag {
            tg = tg.repeat(batch);
        } // broadcast
        let got = h
            .eng
            .set_cdft(&fm, &tg)
            .unwrap_or_else(|e| panic!("gpu_cdft '{name}': {e}"));
        eprintln!("[gpu] gpu_cdft '{name}' nfrag={got} batch={batch}");
        got as INT
    })
}

/// gpu_cdft_clear(name) — drop the constraint set; the next scc/eval
/// rebuilds h_scc without the λ shift.
fn rhai_gpu_cdft_clear(name: &str) {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_cdft_clear '{name}': no engine"));
        h.eng.clear_cdft();
    });
}

/// gpu_cdft_scc(name, max_outer, scc_iter, rms_tol, q_tol) — outer-λ
/// constrained solve: SCC at fixed λ, then secant update on
/// Q_F − Q_F^target until |err| ≤ q_tol for all (b,f). Returns
/// max |Q_F − target| at exit (e). Energies after this call:
/// gpu_eval gives E_DFTB + Σλ·Q_F; use gpu_cdft_energies for E_DFTB.
fn rhai_gpu_cdft_scc(name: &str, max_outer: INT, scc_iter: INT, rms_tol: f64, q_tol: f64) -> f64 {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_cdft_scc '{name}': no engine"));
        let r = h
            .eng
            .cdft_scc(max_outer as usize, scc_iter as usize, rms_tol as f32, q_tol)
            .unwrap_or_else(|e| panic!("gpu_cdft_scc '{name}': {e}"));
        eprintln!(
            "[gpu] gpu_cdft_scc '{name}' outer={} q_err_max={:.3e} conv={}/{}",
            r.outer_iters,
            r.q_err_max,
            r.converged.iter().filter(|&&x| x).count(),
            r.converged.len()
        );
        r.q_err_max
    })
}

/// gpu_cdft_qf(name, b, f) — fragment excess charge Q_F of replica b.
fn rhai_gpu_cdft_qf(name: &str, b: INT, f: INT) -> f64 {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_cdft_qf '{name}': no engine"));
        let qf = h
            .eng
            .cdft_qfrag()
            .unwrap_or_else(|e| panic!("gpu_cdft_qf '{name}': {e}"));
        let nfrag = qf.len() / h.eng.batch();
        let i = b as usize * nfrag + f as usize;
        if i >= qf.len() {
            panic!("gpu_cdft_qf '{name}': (b={b},f={f}) out of range nfrag={nfrag}");
        }
        qf[i]
    })
}

/// gpu_cdft_lam(name, b, f) — current Lagrange multiplier λ_F (Ha).
fn rhai_gpu_cdft_lam(name: &str, b: INT, f: INT) -> f64 {
    with_gpu(|g| {
        let h = g
            .get(name)
            .unwrap_or_else(|| panic!("gpu_cdft_lam '{name}': no engine"));
        let c = h
            .eng
            .plan
            .cdft
            .as_ref()
            .unwrap_or_else(|| panic!("gpu_cdft_lam '{name}': no constraint set"));
        let i = b as usize * c.nfrag + f as usize;
        if i >= c.lam.len() {
            panic!("gpu_cdft_lam '{name}': (b={b},f={f}) out of range");
        }
        c.lam[i]
    })
}

/// gpu_cdft_set_lam(name, b, f, lam) — set λ manually (Ha) for a
/// response scan: set λ → gpu_scc → gpu_cdft_qf reads Q_F(λ).
fn rhai_gpu_cdft_set_lam(name: &str, b: INT, f: INT, lam: f64) {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_cdft_set_lam '{name}': no engine"));
        h.eng
            .cdft_set_lam(b as usize, f as usize, lam)
            .unwrap_or_else(|e| panic!("gpu_cdft_set_lam '{name}': {e}"));
    });
}

/// gpu_cdft_energies(name) — eval + subtract Σλ·Q_F → E_DFTB of the
/// constrained states; stores into last_e for gpu_energy_i. Returns
/// replica-0 energy.
fn rhai_gpu_cdft_energies(name: &str) -> f64 {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_cdft_energies '{name}': no engine"));
        let e = h
            .eng
            .cdft_energies()
            .unwrap_or_else(|e| panic!("gpu_cdft_energies '{name}': {e}"));
        h.last_e = e.clone();
        e[0]
    })
}

fn rhai_gpu_max_force(name: &str) -> f64 {
    with_gpu(|g| {
        let h = g
            .get(name)
            .unwrap_or_else(|| panic!("gpu_max_force '{name}': no engine"));
        let f = h.last_f.as_ref().unwrap_or_else(|| {
            panic!("gpu_max_force '{name}': last gpu_eval had want_forces=false")
        });
        let mut m = 0.0f32;
        for &x in f {
            m = m.max(x.abs());
        }
        m as f64
    })
}

fn rhai_gpu_n_batch(name: &str) -> INT {
    with_gpu(|g| {
        let h = g
            .get(name)
            .unwrap_or_else(|| panic!("gpu_n_batch '{name}': no engine"));
        h.eng.batch() as INT
    })
}

fn rhai_gpu_n_orbs(name: &str) -> INT {
    with_gpu(|g| {
        let h = g
            .get(name)
            .unwrap_or_else(|| panic!("gpu_n_orbs '{name}': no engine"));
        h.eng.n() as INT
    })
}

fn rhai_gpu_scc_iters(name: &str) -> INT {
    with_gpu(|g| {
        let h = g
            .get(name)
            .unwrap_or_else(|| panic!("gpu_scc_iters '{name}': no engine"));
        h.last_iters
    })
}

fn rhai_gpu_scc_stalled(name: &str) -> INT {
    with_gpu(|g| {
        let h = g
            .get(name)
            .unwrap_or_else(|| panic!("gpu_scc_stalled '{name}': no engine"));
        if h.last_stalled {
            1
        } else {
            0
        }
    })
}

/// §12 D11: per-system convergence status of the last gpu_scc_mixer call.
/// Returns "converged" | "plateau" | "failed" (batch=1) or joined list.
fn rhai_gpu_scc_status(name: &str) -> String {
    with_gpu(|g| {
        let h = g
            .get(name)
            .unwrap_or_else(|| panic!("gpu_scc_status '{name}': no engine"));
        if h.last_status.is_empty() {
            panic!("gpu_scc_status '{name}': no scc_mixer call yet");
        }
        h.last_status.join(",")
    })
}

fn rhai_gpu_q_rms(name: &str) -> f64 {
    with_gpu(|g| {
        let h = g.get(name).unwrap_or_else(|| {
            panic!("gpu_q_rms '{name}': no engine — gpu_eval/gpu_measure first")
        });
        if !h.last_q_rms.is_finite() {
            panic!("gpu_q_rms '{name}': no finalize yet (gpu_eval/gpu_measure)");
        }
        h.last_q_rms
    })
}

fn rhai_gpu_n_atoms(name: &str) -> INT {
    with_gpu(|g| {
        let h = g
            .get(name)
            .unwrap_or_else(|| panic!("gpu_n_atoms '{name}': no engine"));
        h.eng.n_atoms() as INT
    })
}

fn store_gpu_eval(h: &mut GpuHandle, ev: GpuDftbEval, ctx: &str) -> f64 {
    for (i, &e) in ev.energy.iter().enumerate() {
        if !e.is_finite() {
            panic!("{ctx}: E[{i}]={e} non-finite");
        }
    }
    if let Some(ref f) = ev.forces {
        for (i, &x) in f.iter().enumerate() {
            if !x.is_finite() {
                panic!("{ctx}: F[{i}]={x} non-finite");
            }
        }
    }
    h.last_e = ev.energy;
    h.last_f = ev.forces;
    h.last_q_rms = ev.q_rms;
    h.last_q_max = ev.q_max;
    h.last_e[0]
}

fn rhai_gpu_reset_q(name: &str) {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_reset_q '{name}': no engine — gpu_new first"));
        h.eng
            .reset_q0()
            .unwrap_or_else(|e| panic!("gpu_reset_q '{name}': {e}"));
        eprintln!("[gpu] gpu_reset_q '{name}'");
    });
}

fn rhai_gpu_scc_mixer(name: &str, max_iter: INT, tol: f64, mix: INT) -> f64 {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_scc_mixer '{name}': no engine — gpu_new first"));
        let scc = h
            .eng
            .scc_mix(max_iter as usize, tol as f32, mix as i32)
            .unwrap_or_else(|e| panic!("gpu_scc_mixer '{name}' mix={mix}: {e}"));
        h.last_rms = scc.rms;
        h.last_iters = scc.n_iters as i64;
        h.last_stalled = scc.stalled;
        h.last_status = scc
            .statuses
            .iter()
            .map(|s| format!("{s:?}").to_lowercase())
            .collect();
        eprintln!(
            "[gpu] gpu_scc_mixer '{name}' mix={mix} rms={:.3e} iters={} stalled={} status={}",
            scc.rms,
            scc.n_iters,
            scc.stalled,
            h.last_status.join(",")
        );
        scc.rms as f64
    })
}

/// §12 D2 A/B: set JACOBI_PREC for engines created AFTER this call.
/// 0 = pure FP32-FMA, 1 = FP64 rotation construction only, 2 = broad FP64.
fn rhai_gpu_jacobi_prec(mode: INT) {
    rust_dftb::qmqm::gpu_scc_plan::set_jacobi_prec(mode as u32);
    eprintln!("[gpu] gpu_jacobi_prec mode={mode} (0=f32-fma 1=f64-rot 2=f64-all) — affects subsequent gpu_new");
}

/// §12 D3/D4 A/B: mode bit0 = finalize repair (C′ renorm + ρ weights),
/// bit1 = in-SCC occupied-column renormalization. Default production = 1.
fn rhai_gpu_occ_repair(name: &str, mode: INT) {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_occ_repair '{name}': no engine — gpu_new first"));
        h.eng.plan.occ_repair = (mode & 1) != 0;
        h.eng.plan.occ_repair_scc = (mode & 2) != 0;
        eprintln!(
            "[gpu] gpu_occ_repair '{name}' mode={mode} (finalize={} scc={})",
            h.eng.plan.occ_repair, h.eng.plan.occ_repair_scc
        );
    });
}

/// §12 D6 A/B: enable/disable Löwdin-X reuse across geometry changes.
fn rhai_gpu_x_reuse(name: &str, on: bool) {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_x_reuse '{name}': no engine — gpu_new first"));
        h.eng.plan.x_reuse = on;
        eprintln!("[gpu] gpu_x_reuse '{name}' on={on}");
    });
}

fn rhai_gpu_measure(name: &str, want_cpu: bool) -> f64 {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_measure '{name}': no engine — gpu_new first"));
        let ev = h
            .eng
            .measure(want_cpu)
            .unwrap_or_else(|e| panic!("gpu_measure '{name}' want_cpu={want_cpu}: {e}"));
        let e0 = store_gpu_eval(h, ev, &format!("gpu_measure '{name}'"));
        eprintln!("[gpu] gpu_measure '{name}' want_cpu={want_cpu} E[0]={e0:.12} q_rms={:.3e} q_max={:.3e}", h.last_q_rms, h.last_q_max);
        e0
    })
}

fn rhai_gpu_cpu_energy(name: &str) -> f64 {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_cpu_energy '{name}': no engine — gpu_new first"));
        let (e, _, _) = h
            .eng
            .cpu_ref()
            .unwrap_or_else(|e| panic!("gpu_cpu_energy '{name}': {e}"));
        if !e.is_finite() {
            panic!("gpu_cpu_energy '{name}': E={e} non-finite");
        }
        eprintln!("[gpu] gpu_cpu_energy '{name}' E={e:.12} Ha");
        e
    })
}

fn rhai_gpu_set_coords(name: &str, xyz: Array) -> INT {
    let (n_atoms, batch) = with_gpu(|g| {
        let h = g
            .get(name)
            .unwrap_or_else(|| panic!("gpu_set_coords '{name}': no engine — gpu_new first"));
        (h.eng.n_atoms(), h.eng.batch())
    });
    let need = batch * n_atoms * 3;
    if xyz.len() != need {
        panic!(
            "gpu_set_coords '{name}': xyz len {} != 3*batch*n_atoms 3*{batch}*{n_atoms}={need}",
            xyz.len()
        );
    }
    let ntot = batch * n_atoms;
    let mut coords = Vec::with_capacity(ntot);
    for i in 0..ntot {
        coords.push([
            dyn_f64(&xyz[3 * i], &format!("gpu_set_coords '{name}' x[{i}]")),
            dyn_f64(&xyz[3 * i + 1], &format!("gpu_set_coords '{name}' y[{i}]")),
            dyn_f64(&xyz[3 * i + 2], &format!("gpu_set_coords '{name}' z[{i}]")),
        ]);
    }
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_set_coords '{name}': no engine"));
        h.eng
            .set_coords(&coords)
            .unwrap_or_else(|e| panic!("gpu_set_coords '{name}': {e}"));
    });
    if batch == 1 {
        sync_geom_coords(name, &coords);
    }
    eprintln!("[gpu] gpu_set_coords '{name}' n_atoms={n_atoms} batch={batch}");
    n_atoms as INT
}

/// gpu_get_coords(name) -> [x,y,z,...] flat Å, all replicas —
/// syncs device→host first (device-FIRE moves coords on the GPU only).
fn rhai_gpu_get_coords(name: &str) -> Array {
    with_gpu(|g| {
        let h = g
            .get_mut(name)
            .unwrap_or_else(|| panic!("gpu_get_coords '{name}': no engine"));
        h.eng
            .sync_coords_to_host()
            .unwrap_or_else(|e| panic!("gpu_get_coords '{name}': {e}"));
        let mut a = Array::new();
        for p in h.eng.coords() {
            a.push(Dynamic::from_float(p[0]));
            a.push(Dynamic::from_float(p[1]));
            a.push(Dynamic::from_float(p[2]));
        }
        a
    })
}

fn rhai_get_xyz(name: &str) -> Array {
    with_state(|s| {
        let st = s
            .geometries
            .get(name)
            .unwrap_or_else(|| panic!("get_xyz '{name}': no geometry — load_xyz/make_geom first"));
        let mut a = Array::new();
        for p in &st.positions {
            a.push(Dynamic::from_float(p[0]));
            a.push(Dynamic::from_float(p[1]));
            a.push(Dynamic::from_float(p[2]));
        }
        a
    })
}

fn rhai_get_species(name: &str) -> Array {
    with_state(|s| {
        let st = s.geometries.get(name).unwrap_or_else(|| {
            panic!("get_species '{name}': no geometry — load_xyz/make_geom first")
        });
        st.elements
            .iter()
            .map(|e| Dynamic::from(e.symbol().to_string()))
            .collect()
    })
}

fn rhai_assert_finite(x: f64, msg: &str) {
    if !x.is_finite() {
        panic!("assert_finite failed: {msg}: {x}");
    }
}

fn rhai_assert_close(a: f64, b: f64, tol: f64, msg: &str) {
    let d = (a - b).abs();
    if !a.is_finite() || !b.is_finite() || d > tol {
        panic!("assert_close failed: {msg}: a={a} b={b} |d|={d} tol={tol}");
    }
}

fn rhai_die(msg: &str) {
    panic!("{msg}");
}

fn find_repo_root() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|e| panic!("cwd: {e}"));
    let mut cands = vec![cwd.clone(), cwd.join(".."), cwd.join("../..")];
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        cands.push(PathBuf::from(manifest).join(".."));
    }
    for cand in cands {
        let xyz = cand.join("data/xyz/adenine-thymine.xyz");
        if xyz.is_file() {
            return cand
                .canonicalize()
                .unwrap_or_else(|e| panic!("canonicalize {}: {e}", cand.display()));
        }
    }
    panic!(
        "cannot find data/xyz/adenine-thymine.xyz from cwd={}",
        cwd.display()
    );
}

// ─── Main: set up Rhai engine and run script ────────────────────────

// ─── Main: set up Rhai engine and run script ────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut script_path = String::new();
    let mut sk_dir = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--script" | "-s" => {
                i += 1;
                script_path = args[i].clone();
            }
            "--sk-dir" => {
                i += 1;
                sk_dir = args[i].clone();
            }
            "--help" | "-h" => {
                eprintln!("Usage: dftb_engine --script <script.rhai> [--sk-dir <path>]");
                eprintln!();
                eprintln!("Rhai functions available:");
                eprintln!("  build_pah(name, shells, acc) -> n_atoms");
                eprintln!("  build_flake(name, radius, shape, passivate, acc) -> n_atoms");
                eprintln!("  build_zigzag(name, width, length, passivate, acc) -> n_atoms");
                eprintln!("  load_xyz(name, path) -> n_atoms");
                eprintln!("  make_geom(name, species_csv, xyz_flat) -> n_atoms");
                eprintln!(
                    "  gpu_new(name, sk_dir, batch) -> n_orbs   (GpuDftb, homogeneous replicas)"
                );
                eprintln!("  gpu_scc(name, max_iter, tol) -> rms");
                eprintln!("  gpu_scc_mixer(name, max_iter, tol, mix) -> rms   mix 0=GPU DIIS, 1=GPU simple, 2=host f64 DIIS");
                eprintln!("  gpu_reset_q(name)                       reload q0 + reset DIIS");
                eprintln!("  gpu_set_coords(name, xyz_flat) -> n_atoms");
                eprintln!(
                    "  gpu_freeze_atoms(name, [i..])           pin template atoms in all replicas"
                );
                eprintln!("  gpu_set_constraint(name, i, j, [d..])   per-replica |x_j−x_i|=d Å scan coordinate");
                eprintln!("  gpu_clear_constraint(name)");
                eprintln!(
                    "  gpu_eval(name, want_forces) -> E[0]      (one finalize; forces optional)"
                );
                eprintln!("  gpu_measure(name, want_cpu) -> E[0]     frozen-H + energy identities; CPU if true");
                eprintln!("  gpu_cpu_energy(name) -> E               independent CPU f64 SCC+rep at replica 0");
                eprintln!("  gpu_energy_i(name, i) -> E[i]");
                eprintln!("  gpu_max_force(name) -> max|F|           (after gpu_eval(..., true))");
                eprintln!("  get_xyz(name) -> xyz_flat               geometry table, Å");
                eprintln!(
                    "  gpu_get_coords(name) -> xyz_flat      device positions, all replicas, Å"
                );
                eprintln!("  gpu_n_batch / gpu_n_orbs / gpu_n_atoms / gpu_scc_iters / gpu_scc_stalled / gpu_q_rms");
                eprintln!("  sparse_new(name, sk_dir) -> n_orbs      (SparseDftb, one system)");
                eprintln!("  sparse_scc(name, max_iter, tol) -> rms");
                eprintln!(
                    "  sparse_eval(name, want_forces) -> E     (after sparse_scc; forces optional)"
                );
                eprintln!("  sparse_max_force / sparse_n_atoms / sparse_n_orbs / sparse_scc_iters / sparse_tr_ks");
                eprintln!("  sparse_charges(name) -> csv");
                eprintln!("  sparse_set_coords(name, xyz_flat) -> n_atoms");
                eprintln!("  sparse_fire_step(name, f_tol) -> max|F|");
                eprintln!("  sparse_md_step(name, dt) -> max|F|");
                eprintln!("  sparse_relax(name, max_steps, f_tol, scc_tol) -> max|F|");
                eprintln!("  assert_finite(x, msg) / assert_close(a, b, tol, msg) / die(msg)");
                eprintln!("  run_dftb_scc(name, sk_dir, max_iter, tol) -> energy");
                eprintln!("  run_dftb_nonscc(name, sk_dir) -> energy");
                eprintln!("  run_sparse_purify(name, max_iter, tol) -> r_i");
                eprintln!(
                    "  run_sparse_purify_geom(name, r_max, max_iter, tol) -> r_i  (geometric mask)"
                );
                eprintln!("  compare_density(name) -> max_diff");
                eprintln!("  compare_charges(name) -> max_diff");
                eprintln!("  get_energy(name) -> energy");
                eprintln!("  get_n_atoms(name) -> n_atoms");
                eprintln!("  get_charges(name) -> csv_string");
                eprintln!("  get_sparse_charges(name) -> csv_string");
                eprintln!("  get_eigenvalues(name) -> csv_string");
                eprintln!("  get_homo_lumo(name) -> 'homo,lumo,gap'");
                eprintln!("  save_charges(name, path) -> bool");
                eprintln!("  save_sparse_charges(name, path) -> bool");
                eprintln!("  save_eigenvalues(name, path) -> bool");
                eprintln!("  save_eigenvectors(name, path) -> bool");
                eprintln!("  save_hs_matrix(name, path) -> bool");
                eprintln!("  davidson_homo_lumo(name, n_target) -> 'homo,lumo,gap'");
                eprintln!("  save_convergence(name, path) -> bool");
                eprintln!("  ftos(x) / itos(x) -> string");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    if script_path.is_empty() {
        eprintln!("ERROR: --script <path> is required. Use --help for usage.");
        std::process::exit(2);
    }

    // Default SK dir if not provided: env var first, then probe known
    // locations; fail loud listing what was tried (the old hard-coded
    // /home/prokophapala path is gone).
    if sk_dir.is_empty() {
        if let Ok(d) = std::env::var("RUST_DFTB_SK_DIR") {
            sk_dir = d;
        } else {
            let home = std::env::var("HOME").unwrap_or_default();
            let candidates = [
                "../external/slakos/origin/mio-1-1".to_string(), // repo-relative (run from rust_dftb/)
                "external/slakos/origin/mio-1-1".to_string(),    // repo root as CWD
                format!("{home}/SIMULATIONS/dftbplus/slakos/mio-1-1"), // user SK install
                format!("{home}/git_SW/dftbplus/external/slakos/origin/mio-1-1"),
            ];
            sk_dir = candidates.iter()
                .find(|p| std::path::Path::new(p).join("H-H.skf").is_file())
                .cloned()
                .unwrap_or_else(|| panic!("no SK dir found — pass --sk-dir <path> or set RUST_DFTB_SK_DIR (tried: {})", candidates.join(", ")));
            eprintln!("[dftb_engine] SK dir: {sk_dir}");
        }
    }

    // Set up Rhai engine
    let mut engine = Engine::new();
    engine.set_max_expr_depths(64, 64);

    // Register functions
    engine.register_fn("build_pah", rhai_build_pah);
    engine.register_fn("build_flake", rhai_build_flake);
    engine.register_fn("build_zigzag", rhai_build_zigzag);
    engine.register_fn("save_xyz", rhai_save_xyz);
    engine.register_fn("load_xyz", rhai_load_xyz);
    engine.register_fn("make_geom", rhai_make_geom);
    engine.register_fn("gpu_new", rhai_gpu_new);
    engine.register_fn("gpu_scc", rhai_gpu_scc);
    engine.register_fn("gpu_scc_mixer", rhai_gpu_scc_mixer);
    engine.register_fn("gpu_occ_repair", rhai_gpu_occ_repair);
    engine.register_fn("gpu_jacobi_prec", rhai_gpu_jacobi_prec);
    engine.register_fn("gpu_reset_q", rhai_gpu_reset_q);
    engine.register_fn("gpu_set_coords", rhai_gpu_set_coords);
    engine.register_fn("gpu_eval", rhai_gpu_eval);
    engine.register_fn("gpu_measure", rhai_gpu_measure);
    engine.register_fn("gpu_cpu_energy", rhai_gpu_cpu_energy);
    engine.register_fn("gpu_energy_i", rhai_gpu_energy_i);
    engine.register_fn("pbc_new", rhai_pbc_new);
    engine.register_fn("pbc_set_coords", rhai_pbc_set_coords);
    engine.register_fn("pbc_scc", rhai_pbc_scc);
    engine.register_fn("pbc_smearing", rhai_pbc_smearing);
    engine.register_fn("pbc_eval", rhai_pbc_eval);
    engine.register_fn("pbc_energy_i", rhai_pbc_energy_i);
    engine.register_fn("gpu_max_force", rhai_gpu_max_force);
    engine.register_fn("gpu_bench", rhai_gpu_bench);
    engine.register_fn("gpu_fire_step", rhai_gpu_fire_step);
    engine.register_fn("gpu_relax", rhai_gpu_relax);
    engine.register_fn("gpu_freeze_atoms", rhai_gpu_freeze_atoms);
    engine.register_fn("gpu_set_constraint", rhai_gpu_set_constraint);
    engine.register_fn("gpu_clear_constraint", rhai_gpu_clear_constraint);
    engine.register_fn("gpu_cdft", rhai_gpu_cdft);
    engine.register_fn("gpu_cdft_clear", rhai_gpu_cdft_clear);
    engine.register_fn("gpu_cdft_scc", rhai_gpu_cdft_scc);
    engine.register_fn("gpu_cdft_qf", rhai_gpu_cdft_qf);
    engine.register_fn("gpu_cdft_lam", rhai_gpu_cdft_lam);
    engine.register_fn("gpu_cdft_set_lam", rhai_gpu_cdft_set_lam);
    engine.register_fn("gpu_cdft_energies", rhai_gpu_cdft_energies);
    engine.register_fn("gpu_smearing", rhai_gpu_smearing);
    engine.register_fn("gpu_n_batch", rhai_gpu_n_batch);
    engine.register_fn("gpu_n_orbs", rhai_gpu_n_orbs);
    engine.register_fn("gpu_n_atoms", rhai_gpu_n_atoms);
    engine.register_fn("gpu_scc_iters", rhai_gpu_scc_iters);
    engine.register_fn("gpu_scc_stalled", rhai_gpu_scc_stalled);
    engine.register_fn("gpu_scc_status", rhai_gpu_scc_status);
    engine.register_fn("gpu_x_reuse", rhai_gpu_x_reuse);
    engine.register_fn("gpu_q_rms", rhai_gpu_q_rms);
    engine.register_fn("get_xyz", rhai_get_xyz);
    engine.register_fn("gpu_get_coords", rhai_gpu_get_coords);
    engine.register_fn("get_species", rhai_get_species);
    engine.register_fn("sparse_new", rhai_sparse_new);
    engine.register_fn("sparse_new", rhai_sparse_new_cut);
    engine.register_fn("sparse_new", rhai_sparse_new_full);
    engine.register_fn("sparse_new", rhai_sparse_new_budget);
    engine.register_fn("sparse_scc", rhai_sparse_scc);
    engine.register_fn("sparse_scc_try", rhai_sparse_scc_try);
    engine.register_fn("sparse_purify_now", rhai_sparse_purify_now);
    engine.register_fn("sparse_ri_f64", rhai_sparse_ri_f64);
    engine.register_fn("sparse_mcw_f64", rhai_sparse_mcw_f64);
    engine.register_fn("sparse_mcw_ff", rhai_sparse_mcw_ff);
    engine.register_fn("sparse_sync", rhai_sparse_sync);
    engine.register_fn("sparse_ff_test", rhai_sparse_ff_test);
    engine.register_fn("sparse_eval", rhai_sparse_eval);
    engine.register_fn("sparse_max_force", rhai_sparse_max_force);
    engine.register_fn("sparse_force_at", rhai_sparse_force_at);
    engine.register_fn("sparse_fire_step", rhai_sparse_fire_step);
    engine.register_fn("sparse_md_step", rhai_sparse_md_step);
    engine.register_fn("sparse_relax", rhai_sparse_relax);
    engine.register_fn("sparse_new_mask", rhai_sparse_new_mask);
    engine.register_fn("sparse_geom_mode", rhai_sparse_geom_mode);
    engine.register_fn("sparse_fire_dt", rhai_sparse_fire_dt);
    engine.register_fn("sparse_jitter", rhai_sparse_jitter);
    engine.register_fn("sparse_relax_traj", rhai_sparse_relax_traj);
    engine.register_fn("sparse_set_coords", rhai_sparse_set_coords);
    engine.register_fn("sparse_n_atoms", rhai_sparse_n_atoms);
    engine.register_fn("sparse_n_orbs", rhai_sparse_n_orbs);
    engine.register_fn("sparse_scc_iters", rhai_sparse_scc_iters);
    engine.register_fn("sparse_tr_ks", rhai_sparse_tr_ks);
    engine.register_fn("sparse_charges", rhai_sparse_charges);
    engine.register_fn("sparse_vibrations", rhai_sparse_vibrations);
    engine.register_fn("sparse_tc2_tol", rhai_sparse_tc2_tol);
    engine.register_fn("sparse_ns_tol", rhai_sparse_ns_tol);
    engine.register_fn("sparse_purifier", rhai_sparse_purifier);
    engine.register_fn("sparse_eval_p", rhai_sparse_eval_p);
    engine.register_fn("assert_finite", rhai_assert_finite);
    engine.register_fn("assert_close", rhai_assert_close);
    engine.register_fn("die", rhai_die);
    engine.register_fn("run_dftb_scc", rhai_run_dftb_scc);
    engine.register_fn("run_dftb_nonscc", rhai_run_dftb_nonscc);
    engine.register_fn("run_sparse_purify", rhai_run_sparse_purify);
    engine.register_fn("run_sparse_purify_geom", rhai_run_sparse_purify_geom);
    engine.register_fn("compare_density", rhai_compare_density);
    engine.register_fn("compare_charges", rhai_compare_charges);
    engine.register_fn("get_energy", rhai_get_energy);
    engine.register_fn("get_n_atoms", rhai_get_n_atoms);
    engine.register_fn("ftos", rhai_ftos);
    engine.register_fn("clock", rhai_clock);
    engine.register_fn("env", |name: &str| std::env::var(name).unwrap_or_default());
    engine.register_fn("itos", rhai_itos);
    engine.register_fn("save_convergence", rhai_save_convergence);
    engine.register_fn("get_charges", rhai_get_charges);
    engine.register_fn("get_sparse_charges", rhai_get_sparse_charges);
    engine.register_fn("get_eigenvalues", rhai_get_eigenvalues);
    engine.register_fn("get_homo_lumo", rhai_get_homo_lumo);
    engine.register_fn("save_charges", rhai_save_charges);
    engine.register_fn("save_sparse_charges", rhai_save_sparse_charges);
    engine.register_fn("save_eigenvalues", rhai_save_eigenvalues);
    engine.register_fn("save_eigenvectors", rhai_save_eigenvectors);
    engine.register_fn("save_hs_matrix", rhai_save_hs_matrix);
    engine.register_fn("davidson_homo_lumo", rhai_davidson_homo_lumo);

    // Register constants
    let mut scope = Scope::new();
    scope.push_constant("SK_DIR", sk_dir);
    scope.push_constant("A_CC", A_CC);
    scope.push_constant("DEFAULT_TOL", 1e-10f64);
    scope.push_constant("DEFAULT_MAX_ITER", 1000i64);
    scope.push_constant("REPO_ROOT", find_repo_root().to_string_lossy().into_owned());

    // Read and run the script
    let script = match std::fs::read_to_string(&script_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR reading script {script_path}: {e}");
            std::process::exit(1);
        }
    };

    eprintln!("Running script: {script_path}");
    eprintln!(
        "SK dir: {}",
        scope.get_value::<String>("SK_DIR").unwrap_or_default()
    );
    eprintln!();

    if let Err(e) = engine.run_with_scope(&mut scope, &script) {
        eprintln!("ERROR: rhai script failed: {e}");
        std::process::exit(1);
    }

    // RUST_DFTB_PROF=1: dump each GPU engine's accumulated stage table.
    if std::env::var("RUST_DFTB_PROF")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        with_gpu(|g| {
            for (name, h) in g.iter() {
                h.eng.prof_report(&format!("'{name}' (script total)"));
            }
        });
        with_sparse(|g| {
            for (name, h) in g.iter() {
                h.eng
                    .prof_report(&format!("'{name}' (sparse script total)"));
            }
        });
    }

    eprintln!("\nScript completed successfully.");
}
