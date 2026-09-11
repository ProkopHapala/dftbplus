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

use rust_dftb::geometry::{self, A_CC, Element, FlakeShape, NanoStructure};
use rust_dftb::methods::sparse::{
    Bsr4Matrix, SparseBsr4Config, SparseBsr4Gpu, SparseDftb, SparseDftbConfig, SparsePerfStats,
    build_full_mask, build_geometric_mask,
};
use rust_dftb::methods::sparse::gpu_sparse::{GpuBsrMatrix, GpuBsrStructure, SparsePurifyWorkspace};
use rust_dftb::{
    HamiltonianBuilder, SccResult, SystemContext, load_sk_for_species, parse_xyz, parse_species,
};
use rust_dftb::qmqm::{GpuDftb, GpuDftbEval};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rhai::{Array, Dynamic, Engine, Scope, INT};
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
where F: FnOnce(&mut State) -> R {
    let mut guard = STATE.lock().unwrap();
    if guard.is_none() { *guard = Some(State::default()); }
    f(guard.as_mut().unwrap())
}

// ─── Rhai-exposed functions ─────────────────────────────────────────

/// Generate a PAH geometry and store it under `name`.
/// Returns the number of atoms.
fn rhai_build_pah(name: &str, shells: INT, acc: f64) -> INT {
    let st = geometry::build_pah(shells as usize, acc);
    let n = st.natom() as INT;
    with_state(|s| { s.geometries.insert(name.to_string(), st); });
    n
}

/// Generate a graphene flake and store it.
fn rhai_build_flake(name: &str, radius: f64, shape: &str, passivate: bool, acc: f64) -> INT {
    let sh = if shape == "hex" { FlakeShape::Hex } else { FlakeShape::Circle };
    let st = geometry::build_flake(radius, sh, passivate, acc);
    let n = st.natom() as INT;
    with_state(|s| { s.geometries.insert(name.to_string(), st); });
    n
}

/// Generate a zigzag ribbon and store it.
fn rhai_build_zigzag(name: &str, width: INT, length: INT, passivate: bool, acc: f64) -> INT {
    let st = geometry::build_zigzag_ribbon(width as usize, length as usize, passivate, false, acc);
    let n = st.natom() as INT;
    with_state(|s| { s.geometries.insert(name.to_string(), st); });
    n
}

/// Save a stored geometry to an XYZ file.
fn rhai_save_xyz(name: &str, path: &str) -> bool {
    with_state(|s| {
        if let Some(st) = s.geometries.get(name) {
            std::fs::write(path, st.to_xyz()).is_ok()
        } else { false }
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
        Err(e) => { eprintln!("ERROR loading SK: {e}"); return Dynamic::from_float(f64::NAN); }
    };

    let builder = HamiltonianBuilder::new(sk);
    eprintln!("[SCC] Starting '{name}': {} atoms, {} orbitals, max_iter={max_iter}, tol={tol}", species.len(), species.len() * 4);
    eprintln!("Running dense DFTB SCC (max_iter={max_iter}, tol={tol}) ...");
    match builder.build_scc(&species, &coords, max_iter as usize, tol) {
        Ok(scc) => {
            let energy = scc.energy;
            let n_iter = scc.n_iter;
            eprintln!("  SCC converged in {n_iter} iters, E = {energy:.10} Ha");
            eprintln!("  charges: {:?}", scc.charges.iter()
                .map(|q| format!("{q:.4}")).collect::<Vec<_>>());
            with_state(|s| { s.scc_results.insert(name.to_string(), scc); });
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
        Err(e) => { eprintln!("ERROR loading SK: {e}"); return Dynamic::from_float(f64::NAN); }
    };

    let builder = HamiltonianBuilder::new(sk);
    eprintln!("[non-SCC] '{name}': {} atoms, building H0/S ...", species.len());
    match builder.build_non_scc(&species, &coords) {
        Ok(ham) => {
            let n_orbs = ham.h0.nrows();
            eprintln!("  H0: {n_orbs}×{n_orbs}");

            // Build SystemContext for per-atom orbital counts (handles H with 1 orb, C with 4)
            let ctx = match SystemContext::from_sk_data(&builder.sk, &species) {
                Ok(c) => c,
                Err(e) => { eprintln!("ERROR SystemContext: {e}"); return Dynamic::from_float(f64::NAN); }
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
            let n_electrons: f64 = species.iter().map(|s| {
                match s.as_str() { "C" => 4.0, "N" => 5.0, "O" => 6.0, "B" => 3.0, _ => 1.0 }
            }).sum();
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
            eprintln!("  Mulliken charges: {:?}", charges.iter()
                .map(|q| format!("{q:.4}")).collect::<Vec<_>>());

            let eigs_sorted: DVector<f64> = DVector::from_iterator(n_orbs, idx.iter().map(|&i| he.eigenvalues[i]));
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
            with_state(|s| { s.scc_results.insert(name.to_string(), scc); });
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
        eprintln!("ERROR: no DFTB result found for '{name}'. Run run_dftb_scc or run_dftb_nonscc first.");
        return Dynamic::from_float(f64::NAN);
    };

    let n_atom = st.natom();
    let n_orbs = n_atom * 4; // BSR4: 4 orbitals per atom
    if max_iter <= 0 || !tol.is_finite() || tol <= 0.0 {
        panic!("sparse purification requires max_iter > 0 and finite tol > 0: max_iter={max_iter}, tol={tol}");
    }
    if scc.h0.nrows() != n_orbs {
        eprintln!("ERROR: H0 size {} != expected {} (n_atom×4). BSR4 requires sp basis (4 orbs/atom).",
            scc.h0.nrows(), n_orbs);
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
        Err(e) => { eprintln!("ERROR GPU init: {e}"); return Dynamic::from_float(f64::NAN); }
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
    let s_struct = match GpuBsrStructure::new(&gpu, n_atom, &(s_bsr.row_ptr.clone(), s_bsr.col_idx.clone())) {
        Ok(s) => Arc::new(s),
        Err(e) => panic!("ERROR building resident S structure for {n_atom} atoms: {e}"),
    };
    let s_dev = match gpu.buf_f32(&s_bsr.values) {
        Ok(values) => GpuBsrMatrix { struct_: s_struct, values },
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
        panic!("Newton-Schulz failed to converge: R_Z={r_z:e}, iterations={z_iters}, tolerance=1e-4");
    }

    // 2. Spectral bounds
    let (emin, emax) = match gpu.spectral_bounds(&h_bsr, &z, &mask, 0.1) {
        Ok(r) => r,
        Err(e) => { eprintln!("ERROR spectral bounds: {e}"); return Dynamic::from_float(f64::NAN); }
    };
    eprintln!("    spectral bounds: emin={emin:.4} emax={emax:.4}");

    // 3. K₀
    let k0 = match gpu.build_k0(&h_bsr, &s_bsr, &z, &mask, &mask, emin, emax) {
        Ok(k) => k,
        Err(e) => { eprintln!("ERROR K0: {e}"); return Dynamic::from_float(f64::NAN); }
    };

    // 4. TC2 purification
    eprintln!("  TC2 purification (max_iter={max_iter}, tol={tol}) ...");
    let n_orb_atom: Vec<u8> = st.elements.iter().map(|e| if e.symbol() == "H" { 1 } else { 4 }).collect();
    let mut workspace = match SparsePurifyWorkspace::new(gpu, &k0, &s_bsr, &mask, &mask, &n_orb_atom, n_occ) {
        Ok(ws) => ws,
        Err(e) => panic!("ERROR creating resident TC2 workspace: {e}"),
    };
    let (k_final, r_i, tr_ks, tc2_iters, history) = match workspace.tc2_purify_dev(max_iter as usize, tol as f32, 1) {
        Ok(r) => r,
        Err(e) => panic!("ERROR TC2: {e}"),
    };
    eprintln!("    TC2: {tc2_iters} iters, R_I={r_i:e}, Tr(KS)={tr_ks:.6}");
    if !r_i.is_finite() || r_i > tol as f32 || !tr_ks.is_finite() {
        panic!("TC2 failed to converge: R_I={r_i:e}, Tr(KS)={tr_ks:.8}, iterations={tc2_iters}, tolerance={tol:e}");
    }

    // 5. R_H = ||HKS - SKH||
    let r_h = match workspace.gpu().hamiltonian_residual(&h_bsr, &k_final, &s_bsr, &mask) {
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
    eprintln!("    sparse Mulliken: {:?}  (dummy occ max={qd_max:.3e})", mulliken.iter()
        .map(|q| format!("{q:.4}")).collect::<Vec<_>>());

    let k_dense = k_final.to_dense();
    let result = SparseResult {
        k_dense, n_atom, n_occ, r_i, r_h: r_h_norm, tr_ks,
        mulliken, iters: tc2_iters, history,
    };
    with_state(|s| { s.sparse_results.insert(name.to_string(), result); });

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
        eprintln!("ERROR: no DFTB result found for '{name}'. Run run_dftb_scc or run_dftb_nonscc first.");
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
        eprintln!("ERROR: H0 size {} != expected {} (n_atom×4). BSR4 requires sp basis (4 orbs/atom).",
            scc.h0.nrows(), n_orbs);
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
    eprintln!("  geometric mask: {nblock} blocks, fill ratio = {fill:.4} ({n_atom}² = {})", n_atom * n_atom);
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
        Err(e) => { eprintln!("ERROR GPU init: {e}"); return Dynamic::from_float(f64::NAN); }
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
    let s_struct = match GpuBsrStructure::new(&gpu, n_atom, &(s_bsr.row_ptr.clone(), s_bsr.col_idx.clone())) {
        Ok(s) => Arc::new(s),
        Err(e) => panic!("ERROR building resident S structure: {e}"),
    };
    let s_dev = match gpu.buf_f32(&s_bsr.values) {
        Ok(values) => GpuBsrMatrix { struct_: s_struct, values },
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
        panic!("Newton-Schulz failed to converge: R_Z={r_z:e}, iterations={z_iters}, tolerance=1e-4");
    }

    let (emin, emax) = match gpu.spectral_bounds(&h_bsr, &z, &mask, 0.1) {
        Ok(r) => r,
        Err(e) => { eprintln!("ERROR spectral bounds: {e}"); return Dynamic::from_float(f64::NAN); }
    };
    eprintln!("    spectral bounds: emin={emin:.4} emax={emax:.4}");

    let k0 = match gpu.build_k0(&h_bsr, &s_bsr, &z, &mask, &mask, emin, emax) {
        Ok(k) => k,
        Err(e) => { eprintln!("ERROR K0: {e}"); return Dynamic::from_float(f64::NAN); }
    };

    eprintln!("  TC2 purification (max_iter={max_iter}, tol={tol}) ...");
    let t0 = Instant::now();
    let n_orb_atom: Vec<u8> = st.elements.iter().map(|e| if e.symbol() == "H" { 1 } else { 4 }).collect();
    let mut workspace = match SparsePurifyWorkspace::new(gpu, &k0, &s_bsr, &mask, &mask, &n_orb_atom, n_occ) {
        Ok(ws) => ws,
        Err(e) => panic!("ERROR creating resident TC2 workspace: {e}"),
    };
    let (k_final, r_i, tr_ks, tc2_iters, history) = match workspace.tc2_purify_dev(max_iter as usize, tol as f32, 1) {
        Ok(r) => r,
        Err(e) => panic!("ERROR TC2: {e}"),
    };
    let t_tc2 = t0.elapsed();
    eprintln!("    TC2: {tc2_iters} iters, R_I={r_i:e}, Tr(KS)={tr_ks:.6}");
    if !r_i.is_finite() || r_i > tol as f32 || !tr_ks.is_finite() {
        panic!("TC2 failed to converge: R_I={r_i:e}, Tr(KS)={tr_ks:.8}, iterations={tc2_iters}, tolerance={tol:e}");
    }

    let r_h = match workspace.gpu().hamiltonian_residual(&h_bsr, &k_final, &s_bsr, &mask) {
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
    eprintln!("    sparse Mulliken: {:?}  (dummy occ max={qd_max:.3e})", mulliken.iter()
        .map(|q| format!("{q:.4}")).collect::<Vec<_>>());

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
        k_dense, n_atom, n_occ, r_i, r_h: r_h_norm, tr_ks,
        mulliken, iters: tc2_iters, history,
    };
    with_state(|s| { s.sparse_results.insert(name.to_string(), result); });

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
            if d > max_diff { max_diff = d; max_i = i; max_j = j; }
            sum_sq += d * d;
        }
    }
    let rms = (sum_sq / (n_orbs * n_orbs) as f64).sqrt();
    eprintln!("  ||2K - D||_max = {max_diff:e}  at ({max_i},{max_j}): D={:.6} 2K={:.6}",
        scc.density[(max_i, max_j)], 2.0 * k_dense_f64[max_i * n_orbs + max_j]);
    eprintln!("  ||2K - D||_rms = {rms:e}");
    // Print diagonal elements for comparison
    eprintln!("  diagonal D[0:4]:  {:?}", (0..4).map(|i| format!("{:.6}", scc.density[(i,i)])).collect::<Vec<_>>());
    eprintln!("  diagonal 2K[0:4]: {:?}", (0..4).map(|i| format!("{:.6}", 2.0 * k_dense_f64[i*n_orbs+i])).collect::<Vec<_>>());

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
        if dq > max_diff { max_diff = dq; }
        eprintln!("  [{i:3}]  {:8.4}  {:8.4}  {:8.4}",
            scc.charges[i], sparse.mulliken[i], dq);
    }
    eprintln!("  max|dq| = {max_diff:e}");
    Dynamic::from_float(max_diff)
}

/// Get the SCC energy from a stored DFTB result.
fn rhai_get_energy(name: &str) -> f64 {
    with_state(|s| s.scc_results.get(name).map(|r| r.energy).unwrap_or(f64::NAN))
}

/// Get the number of atoms in a stored geometry.
fn rhai_get_n_atoms(name: &str) -> INT {
    with_state(|s| s.geometries.get(name).map(|st| st.natom() as INT).unwrap_or(0))
}

/// Convert a float to string for printing.
/// clock() -> seconds since first call — for timing scan/bench scripts.
fn rhai_clock() -> f64 {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_secs_f64()
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
        s.scc_results.get(name)
            .map(|r| r.charges.iter()
                .map(|q| format!("{q:.6}"))
                .collect::<Vec<_>>()
                .join(","))
            .unwrap_or_default()
    })
}

/// Get sparse Mulliken charges as a comma-separated string.
fn rhai_get_sparse_charges(name: &str) -> String {
    with_state(|s| {
        s.sparse_results.get(name)
            .map(|r| r.mulliken.iter()
                .map(|q| format!("{q:.6}"))
                .collect::<Vec<_>>()
                .join(","))
            .unwrap_or_default()
    })
}

/// Get all eigenvalues from the dense SCC result as a comma-separated string.
fn rhai_get_eigenvalues(name: &str) -> String {
    with_state(|s| {
        s.scc_results.get(name)
            .map(|r| r.eigenvalues.iter()
                .map(|e| format!("{e:.6}"))
                .collect::<Vec<_>>()
                .join(","))
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
            (r.eigenvalues[n_occ.saturating_sub(1).min(n_orbs-1)],
             r.eigenvalues[n_occ.min(n_orbs-1)],
             n_occ)
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
        Some((scc.charges.clone(), st.elements.clone(), st.positions.clone()))
    });
    let Some((charges, elements, positions)) = payload else {
        eprintln!("ERROR: no SCC result or geometry for '{name}'");
        return false;
    };
    let mut txt = String::from("atom_idx\telement\tx\ty\tz\tcharge\n");
    for (i, ((el, pos), q)) in elements.iter().zip(positions.iter()).zip(charges.iter()).enumerate() {
        txt.push_str(&format!("{i}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",
            el.symbol(), pos[0], pos[1], pos[2], q));
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
        Some((sparse.mulliken.clone(), st.elements.clone(), st.positions.clone()))
    });
    let Some((charges, elements, positions)) = payload else {
        eprintln!("ERROR: no sparse result or geometry for '{name}'");
        return false;
    };
    let mut txt = String::from("atom_idx\telement\tx\ty\tz\tcharge\n");
    for (i, ((el, pos), q)) in elements.iter().zip(positions.iter()).zip(charges.iter()).enumerate() {
        txt.push_str(&format!("{i}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",
            el.symbol(), pos[0], pos[1], pos[2], q));
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
    eprintln!("  saved eigenvalues: {path} ({} eigs, n_occ={n_occ})", eigs.len());
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
        Some((scc.eigenvectors.clone(), scc.eigenvalues.clone(), species, coords, n_occ))
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
        txt.push_str(&format!("{i}\t{sp}\t{:.10}\t{:.10}\t{:.10}\n", c[0], c[1], c[2]));
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
        Some((scc.h_scc.clone(), scc.s.clone(), scc.eigenvalues.clone(), species, coords, n_occ))
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
        txt.push_str(&format!("{i}\t{sp}\t{:.10}\t{:.10}\t{:.10}\n", c[0], c[1], c[2]));
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
        s.scc_results.get(name).map(|r| {
            (r.h_scc.clone(), r.s.clone(), r.q0.clone())
        })
    });
    let Some((h, s_mat, q0)) = payload else {
        eprintln!("ERROR: no SCC result for '{name}'");
        return String::new();
    };
    let n_electrons: f64 = q0.iter().sum();
    let n_occ = (n_electrons / 2.0).round() as usize;
    let n_target = n_target.max(2) as usize;
    eprintln!("  Davidson: n_orbs={}, n_occ={}, n_target={n_target}", h.nrows(), n_occ);
    match rust_dftb::methods::sparse::davidson_homo_lumo(&h, &s_mat, n_occ, n_target, 100, 1e-8) {
        Ok((eigs, _vecs)) => {
            // eigs is sorted ascending; find HOMO (last occ) and LUMO (first virt)
            // Davidson targets n_target below and n_target above the gap
            let homo = eigs[n_target - 1];
            let lumo = eigs[n_target];
            eprintln!("    Davidson HOMO={homo:.10}, LUMO={lumo:.10}, gap={:.10}", lumo - homo);
            eprintln!("    all {} eigs: {:?}", eigs.len(),
                eigs.iter().map(|e| format!("{e:.6}")).collect::<Vec<_>>());
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
fn dense_to_bsr4(
    m: &DMatrix<f64>,
    n_atom: usize,
    mask: &(Vec<u32>, Vec<u32>),
) -> Bsr4Matrix {
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
where F: FnOnce(&mut HashMap<String, GpuHandle>) -> R {
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
where F: FnOnce(&mut HashMap<String, SparseHandle>) -> R {
    SPARSE.with(|g| f(&mut g.borrow_mut()))
}

fn geom_species_coords(name: &str, ctx: &str) -> (Vec<String>, Vec<[f64; 3]>) {
    with_state(|s| {
        let st = s.geometries.get(name).unwrap_or_else(|| panic!("{ctx} '{name}': no geometry — load_xyz/make_geom first"));
        let species: Vec<String> = st.elements.iter().map(|e| e.symbol().to_string()).collect();
        (species, st.positions.clone())
    })
}

fn sync_geom_coords(name: &str, coords: &[[f64; 3]]) {
    with_state(|s| {
        let st = s.geometries.get_mut(name).unwrap_or_else(|| panic!("sync_geom '{name}': no geometry"));
        if st.positions.len() != coords.len() {
            panic!("sync_geom '{name}': n_atom {} != coords {}", st.positions.len(), coords.len());
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
fn rhai_sparse_new_full(name: &str, sk_dir: &str, r_trunc_ang: f64, taper_w_ang: f64, r_k_ang: f64, r_z_ang: f64) -> INT {
    let (species, coords) = geom_species_coords(name, "sparse_new");
    let n_atom = species.len();
    eprintln!("[sparse] sparse_new '{name}' n_atom={n_atom} sk={sk_dir} r_trunc={r_trunc_ang} taper_w={taper_w_ang} r_k={r_k_ang} r_z={r_z_ang}");
    let sk = load_sk_for_species(sk_dir, &species).unwrap_or_else(|e| panic!("sparse_new '{name}' load SK from {sk_dir}: {e}"));
    let mut cfg = SparseDftbConfig::default();
    if r_trunc_ang > 0.0 { cfg.r_trunc_ang = Some(r_trunc_ang); }
    if taper_w_ang > 0.0 { cfg.taper_w_ang = taper_w_ang; }
    if r_k_ang > 0.0 { cfg.r_k_ang = Some(r_k_ang); }
    if r_z_ang > 0.0 { cfg.r_z_ang = Some(r_z_ang); }
    let eng = SparseDftb::with_config(sk, sk_dir, species, coords, cfg)
        .unwrap_or_else(|e| panic!("sparse_new '{name}' SparseDftb::with_config: {e}"));
    let n_orbs = eng.n_orbs() as INT;
    eprintln!("[sparse] sparse_new '{name}' n_orbs={n_orbs}");
    with_sparse(|g| {
        g.insert(name.to_string(), SparseHandle {
            eng, last_rms: f64::NAN, last_iters: 0, last_fmax: f64::NAN, have_forces: false,
        });
    });
    n_orbs
}

fn rhai_sparse_scc(name: &str, max_iter: INT, tol: f64) -> f64 {
    with_sparse(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("sparse_scc '{name}': no engine — sparse_new first"));
        let scc = h.eng.scc(max_iter as usize, tol)
            .unwrap_or_else(|e| panic!("sparse_scc '{name}': {e}"));
        h.last_rms = scc.rms;
        h.last_iters = scc.n_iters as i64;
        h.have_forces = false;
        eprintln!("[sparse] sparse_scc '{name}' rms={:.3e} iters={} Tr(KS)={:.6} R_I={:.3e} r_scc={:.3e}",
            scc.rms, scc.n_iters, scc.tr_ks, scc.r_i, scc.r_scc);
        scc.rms
    })
}

/// Energy of the last SCC. `want_forces=true` also builds analytic F (CPU contract of D,W).
fn rhai_sparse_eval(name: &str, want_forces: bool) -> f64 {
    with_sparse(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("sparse_eval '{name}': no engine — sparse_new first"));
        let e = h.eng.energy().unwrap_or_else(|e| panic!("sparse_eval '{name}': {e} — call sparse_scc first"));
        if !e.is_finite() { panic!("sparse_eval '{name}': E={e} non-finite"); }
        if want_forces {
            let f = h.eng.forces().unwrap_or_else(|err| panic!("sparse_eval '{name}' forces: {err}"));
            let mut max_f = 0.0f64;
            for fi in &f.forces {
                for &c in fi {
                    if !c.is_finite() { panic!("sparse_eval '{name}': non-finite force {c}"); }
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
        let h = g.get(name).unwrap_or_else(|| panic!("sparse_max_force '{name}': no engine"));
        if !h.have_forces { panic!("sparse_max_force '{name}': last sparse_eval had want_forces=false"); }
        h.last_fmax
    })
}

/// sparse_force_at(name, atom_i, comp) -> force component (Ha/Å). Runs the
/// analytic sparse force path (requires a prior sparse_scc at this geometry).
fn rhai_sparse_force_at(name: &str, i: INT, c: INT) -> f64 {
    with_sparse(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("sparse_force_at '{name}': no engine"));
        let f = h.eng.forces().unwrap_or_else(|e| panic!("sparse_force_at '{name}': {e}"));
        let (i, c) = (i as usize, c as usize);
        if i >= h.eng.n_atom() || c > 2 {
            panic!("sparse_force_at '{name}': bad index i={i} c={c} (n_atom={})", h.eng.n_atom());
        }
        f.forces[i][c]
    })
}

fn rhai_sparse_fire_step(name: &str, f_tol: f64) -> f64 {
    let max_f = with_sparse(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("sparse_fire_step '{name}': no engine — sparse_scc first"));
        let mf = h.eng.fire_step(f_tol).unwrap_or_else(|e| panic!("sparse_fire_step '{name}': {e}"));
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
        let h = g.get_mut(name).unwrap_or_else(|| panic!("sparse_md_step '{name}': no engine"));
        let mf = h.eng.md_step(dt).unwrap_or_else(|e| panic!("sparse_md_step '{name}': {e}"));
        h.have_forces = true;
        h.last_fmax = mf;
        sync_geom_coords(name, h.eng.coords());
        eprintln!("[sparse] sparse_md_step '{name}' dt={dt} max|F|={mf:.4e}");
        mf
    })
}

fn rhai_sparse_relax(name: &str, max_steps: INT, f_tol: f64, scc_tol: f64) -> f64 {
    with_sparse(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("sparse_relax '{name}': no engine"));
        let (n, max_f, rms0) = h.eng.relax(max_steps as usize, f_tol, scc_tol)
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

fn rhai_sparse_set_coords(name: &str, xyz: Array) -> INT {
    let n_atom = with_sparse(|g| {
        g.get(name).unwrap_or_else(|| panic!("sparse_set_coords '{name}': no engine")).eng.n_atom()
    });
    if xyz.len() != n_atom * 3 {
        panic!("sparse_set_coords '{name}': xyz len {} != 3*n_atom {}", xyz.len(), n_atom * 3);
    }
    let mut coords = Vec::with_capacity(n_atom);
    for i in 0..n_atom {
        coords.push([
            dyn_f64(&xyz[3 * i], &format!("sparse_set_coords '{name}' x[{i}]")),
            dyn_f64(&xyz[3 * i + 1], &format!("sparse_set_coords '{name}' y[{i}]")),
            dyn_f64(&xyz[3 * i + 2], &format!("sparse_set_coords '{name}' z[{i}]")),
        ]);
    }
    with_sparse(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("sparse_set_coords '{name}': no engine"));
        h.eng.set_coords(&coords).unwrap_or_else(|e| panic!("sparse_set_coords '{name}': {e}"));
        h.have_forces = false;
    });
    sync_geom_coords(name, &coords);
    eprintln!("[sparse] sparse_set_coords '{name}' n_atom={n_atom}");
    n_atom as INT
}

fn rhai_sparse_n_atoms(name: &str) -> INT {
    with_sparse(|g| g.get(name).unwrap_or_else(|| panic!("sparse_n_atoms '{name}': no engine")).eng.n_atom() as INT)
}

fn rhai_sparse_n_orbs(name: &str) -> INT {
    with_sparse(|g| g.get(name).unwrap_or_else(|| panic!("sparse_n_orbs '{name}': no engine")).eng.n_orbs() as INT)
}

fn rhai_sparse_scc_iters(name: &str) -> INT {
    with_sparse(|g| g.get(name).unwrap_or_else(|| panic!("sparse_scc_iters '{name}': no engine")).last_iters)
}

fn rhai_sparse_tr_ks(name: &str) -> f64 {
    with_sparse(|g| g.get(name).unwrap_or_else(|| panic!("sparse_tr_ks '{name}': no engine")).eng.last_energy().tr_ks as f64)
}

/// Relative TC2 tolerance ‖KSK−K‖/‖K‖ for subsequent scc calls.
fn rhai_sparse_tc2_tol(name: &str, tol: f64) {
    with_sparse(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("sparse_tc2_tol '{name}': no engine"));
        h.eng.set_tc2_tol(tol as f32);
        eprintln!("[sparse] sparse_tc2_tol '{name}' tol={tol:e}");
    });
}

fn rhai_sparse_ns_tol(name: &str, tol: f64) {
    with_sparse(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("sparse_ns_tol '{name}': no engine"));
        h.eng.set_ns_tol(tol as f32);
        eprintln!("[sparse] sparse_ns_tol '{name}' tol={tol:e}");
    });
}

fn rhai_sparse_charges(name: &str) -> String {
    with_sparse(|g| {
        let h = g.get(name).unwrap_or_else(|| panic!("sparse_charges '{name}': no engine"));
        h.eng.last_energy().q.iter().map(|q| format!("{q:.8}")).collect::<Vec<_>>().join(",")
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
    let masses: Vec<f64> = species.iter()
        .map(|s| Element::from_symbol(s).unwrap_or_else(|| panic!("sparse_vibrations '{name}': unknown element {s}")).mass())
        .collect();

    with_sparse(|g| {
        let hnd = g.get_mut(name).unwrap_or_else(|| panic!("sparse_vibrations '{name}': no engine — sparse_new first"));
        let eng = &mut hnd.eng;
        let n_atom = eng.n_atom();
        let n3 = 3 * n_atom;
        let x0 = eng.coords().to_vec();
        if !(h > 0.0) || h > 0.5 {
            panic!("sparse_vibrations '{name}': h={h} Å must be in (0, 0.5] — skin/2 guard needs h < skin/2");
        }

        let mut hess = vec![0.0f64; n3 * n3];
        let mut work = x0.clone();
        eprintln!("[sparse] vibrations '{name}': {n3} columns, h={h} Å, scc_tol={scc_tol:e}");
        for i in 0..n_atom {
            for a in 0..3 {
                let col = 3 * i + a;
                let (f_plus, f_minus);
                work[i][a] = x0[i][a] + h;
                eng.set_coords(&work).unwrap_or_else(|e| panic!("vibrations '{name}' col {col} +h set_coords: {e}"));
                eng.scc(80, scc_tol).unwrap_or_else(|e| panic!("vibrations '{name}' col {col} +h scc: {e}"));
                f_plus = eng.forces().unwrap_or_else(|e| panic!("vibrations '{name}' col {col} +h forces: {e}")).forces;
                work[i][a] = x0[i][a] - h;
                eng.set_coords(&work).unwrap_or_else(|e| panic!("vibrations '{name}' col {col} -h set_coords: {e}"));
                eng.scc(80, scc_tol).unwrap_or_else(|e| panic!("vibrations '{name}' col {col} -h scc: {e}"));
                f_minus = eng.forces().unwrap_or_else(|e| panic!("vibrations '{name}' col {col} -h forces: {e}")).forces;
                work[i][a] = x0[i][a];
                for j in 0..n_atom {
                    for b in 0..3 {
                        hess[(3 * j + b) * n3 + col] = -(f_plus[j][b] - f_minus[j][b]) / (2.0 * h);
                    }
                }
                if col % 6 == 0 { eprintln!("[sparse] vibrations '{name}': col {col}/{n3}"); }
            }
        }
        // Restore input geometry + reconverge so the engine's state is consistent.
        eng.set_coords(&x0).unwrap_or_else(|e| panic!("vibrations '{name}' restore set_coords: {e}"));
        eng.scc(80, scc_tol).unwrap_or_else(|e| panic!("vibrations '{name}' restore scc: {e}"));
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
        let eig = nalgebra::SymmetricEigen::new(dm);
        let mut vals: Vec<f64> = eig.eigenvalues.iter().copied().collect();
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
        order.sort_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap());
        for (k, &col) in order.iter().enumerate() {
            out.push_str(&format!("mode {k} freq {:.4}\n", freqs[k]));
            // Mass-weighted eigenvector → real-space displacement, unit-normed.
            let mut un = 0.0f64;
            let mut u = vec![0.0f64; n3];
            for c in 0..n3 {
                u[c] = eig.eigenvectors[(c, col)] * inv[c] * ANG2BOHR;
                un += u[c] * u[c];
            }
            let un = un.sqrt().max(1e-30);
            for i in 0..n_atom {
                out.push_str(&format!("  {:.8} {:.8} {:.8}\n", u[3 * i] / un, u[3 * i + 1] / un, u[3 * i + 2] / un));
            }
        }
        std::fs::write(path, &out).unwrap_or_else(|e| panic!("sparse_vibrations '{name}' write {path}: {e}"));

        let lo = freqs.first().copied().unwrap_or(f64::NAN);
        let hi = freqs.last().copied().unwrap_or(f64::NAN);
        let summary = format!("n3={n3} n_imag={n_imag} freq_min={lo:.2} freq_max={hi:.2} cm-1 (written {path})");
        eprintln!("[sparse] vibrations '{name}': {summary}");
        summary
    })
}

fn dyn_f64(d: &Dynamic, ctx: &str) -> f64 {
    if let Ok(x) = d.as_float() { return x; }
    if let Ok(x) = d.as_int() { return x as f64; }
    panic!("{ctx}: expected number, got {d}");
}

fn nano_from_species_coords(species: &[String], coords: &[[f64; 3]], ctx: &str) -> NanoStructure {
    if species.len() != coords.len() {
        panic!("{ctx}: species {} != coords {}", species.len(), coords.len());
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
    if n == 0 { panic!("load_xyz '{name}': empty XYZ {path}"); }
    let st = nano_from_species_coords(&mol.species, &mol.coords, &format!("load_xyz '{name}'"));
    with_state(|s| { s.geometries.insert(name.to_string(), st); });
    eprintln!("[gpu] load_xyz '{name}' {n} atoms from {path}");
    n as INT
}

/// Build a geometry from species CSV + flat xyz array (Å).
fn rhai_make_geom(name: &str, species_csv: &str, xyz: Array) -> INT {
    let species = parse_species(species_csv);
    if species.is_empty() { panic!("make_geom '{name}': empty species"); }
    if xyz.len() != species.len() * 3 {
        panic!("make_geom '{name}': xyz len {} != 3*n_atoms {}", xyz.len(), species.len() * 3);
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
    with_state(|s| { s.geometries.insert(name.to_string(), st); });
    eprintln!("[gpu] make_geom '{name}' {n} atoms");
    n as INT
}

/// Create / replace a GpuDftb for a stored geometry. `batch` copies of the same molecule.
fn rhai_gpu_new(name: &str, sk_dir: &str, batch: INT) -> INT {
    let batch = batch as usize;
    if batch == 0 { panic!("gpu_new '{name}': batch=0"); }
    let (species, xyz0) = with_state(|s| {
        let st = s.geometries.get(name).unwrap_or_else(|| panic!("gpu_new '{name}': no geometry — load_xyz/make_geom first"));
        let species: Vec<String> = st.elements.iter().map(|e| e.symbol().to_string()).collect();
        (species, st.positions.clone())
    });
    let n_atoms = species.len();
    let mut coords = Vec::with_capacity(batch * n_atoms);
    for _ in 0..batch { coords.extend_from_slice(&xyz0); }
    eprintln!("[gpu] gpu_new '{name}' n_atoms={n_atoms} batch={batch} sk={sk_dir}");
    let sk = load_sk_for_species(sk_dir, &species).unwrap_or_else(|e| panic!("gpu_new '{name}' load SK from {sk_dir}: {e}"));
    let eng = GpuDftb::new(sk, sk_dir, species, coords, batch)
        .unwrap_or_else(|e| panic!("gpu_new '{name}' GpuDftb::new: {e}"));
    let n_orbs = eng.n() as INT;
    eprintln!("[gpu] gpu_new '{name}' N={n_orbs} device={}", eng.rt.caps().name);
    with_gpu(|g| {
        g.insert(name.to_string(), GpuHandle {
            eng, last_e: Vec::new(), last_f: None, last_rms: f32::NAN, last_q_rms: f64::NAN, last_q_max: f64::NAN, last_iters: 0, last_stalled: false, last_status: Vec::new(),
        });
    });
    n_orbs
}

fn rhai_gpu_scc(name: &str, max_iter: INT, tol: f64) -> f64 {
    with_gpu(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("gpu_scc '{name}': no engine — gpu_new first"));
        let scc = h.eng.scc(max_iter as usize, tol as f32)
            .unwrap_or_else(|e| panic!("gpu_scc '{name}': {e}"));
        h.last_rms = scc.rms;
        h.last_iters = scc.n_iters as i64;
        h.last_stalled = scc.stalled;
        h.last_status = scc.statuses.iter().map(|s| format!("{s:?}").to_lowercase()).collect();
        eprintln!("[gpu] gpu_scc '{name}' rms={:.3e} iters={} stalled={}", scc.rms, scc.n_iters, scc.stalled);
        scc.rms as f64
    })
}

/// One finalize. `want_forces=true` also builds W and F on GPU. Returns replica-0 energy (Ha).
fn rhai_gpu_eval(name: &str, want_forces: bool) -> f64 {
    with_gpu(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("gpu_eval '{name}': no engine — gpu_new first"));
        let ev = h.eng.eval(want_forces).unwrap_or_else(|e| panic!("gpu_eval '{name}' want_forces={want_forces}: {e}"));
        for (i, &e) in ev.energy.iter().enumerate() {
            if !e.is_finite() { panic!("gpu_eval '{name}': E[{i}]={e} non-finite"); }
        }
        if let Some(ref f) = ev.forces {
            for (i, &x) in f.iter().enumerate() {
                if !x.is_finite() { panic!("gpu_eval '{name}': F[{i}]={x} non-finite"); }
            }
        }
        h.last_e = ev.energy;
        h.last_f = ev.forces;
        h.last_q_rms = ev.q_rms;
        h.last_q_max = ev.q_max;
        let e0 = h.last_e[0];
        let mut max_f = 0.0f32;
        if let Some(ref f) = h.last_f { for &x in f { max_f = max_f.max(x.abs()); } }
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
        let h = g.get(name).unwrap_or_else(|| panic!("gpu_energy_i '{name}': no engine"));
        let i = i as usize;
        if h.last_e.is_empty() { panic!("gpu_energy_i '{name}': call gpu_eval first"); }
        if i >= h.last_e.len() { panic!("gpu_energy_i '{name}': i={i} >= batch {}", h.last_e.len()); }
        h.last_e[i]
    })
}

/// gpu_fire_step(name, f_tol) -> max|F| — one FIRE step on all replicas;
/// the engine's SCC state is consumed internally (call after gpu_scc or
/// standalone — it runs its own eval).
fn rhai_gpu_fire_step(name: &str, f_tol: f64) -> f64 {
    with_gpu(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("gpu_fire_step '{name}': no engine — gpu_new first"));
        h.eng.fire_step(f_tol).unwrap_or_else(|e| panic!("gpu_fire_step '{name}': {e}"))
    })
}

/// gpu_relax(name, max_steps, f_tol, scc_tol) -> max|F| — FIRE loop until
/// converged or max_steps; prints unbuffered progress.
fn rhai_gpu_relax(name: &str, max_steps: INT, f_tol: f64, scc_tol: f64) -> f64 {
    with_gpu(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("gpu_relax '{name}': no engine"));
        let (n, max_f, rms0) = h.eng.relax(max_steps as usize, f_tol, scc_tol as f32)
            .unwrap_or_else(|e| panic!("gpu_relax '{name}': {e}"));
        eprintln!("[gpu] gpu_relax '{name}' steps={n} max|F|={max_f:.4e} rms0={rms0:.3e}");
        max_f
    })
}

/// gpu_bench(name, n_runs, max_iter, rms_tol) -> avg ms per SCC call.
/// Times the production GpuDftb path end-to-end (reset_q + scc per run,
/// wall clock — includes queue sync). Replaces the legacy
/// tests/gpu_scc_bench.rs GpuDriver benchmark.
fn rhai_gpu_bench(name: &str, n_runs: INT, max_iter: INT, rms_tol: f64) -> f64 {
    with_gpu(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("gpu_bench '{name}': no engine — gpu_new first"));
        let mut total_ms = 0.0;
        for r in 0..n_runs {
            h.eng.reset_q0().unwrap_or_else(|e| panic!("gpu_bench '{name}' reset_q0: {e}"));
            let t0 = std::time::Instant::now();
            let scc = h.eng.scc(max_iter as usize, rms_tol as f32)
                .unwrap_or_else(|e| panic!("gpu_bench '{name}' scc run {r}: {e}"));
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            total_ms += dt;
            eprintln!("[gpu_bench] {name} run {r}: {dt:.2} ms ({} iters, rms={:.3e})", scc.n_iters, scc.rms);
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
        let h = g.get_mut(name).unwrap_or_else(|| panic!("gpu_smearing '{name}': no engine"));
        h.eng.set_smearing(kT as f32);
    });
}

/// gpu_freeze_atoms(name, [i0,i1,...]) — pin template atoms for ALL
/// replicas (constrained scans: the transferred proton etc.). Forces,
/// velocities, and the per-replica convergence test ignore them.
fn rhai_gpu_freeze_atoms(name: &str, idx: Array) {
    let ids: Vec<usize> = idx.iter()
        .map(|d| dyn_f64(d, &format!("gpu_freeze_atoms '{name}' idx")) as usize)
        .collect();
    with_gpu(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("gpu_freeze_atoms '{name}': no engine"));
        h.eng.set_frozen_atoms(&ids).unwrap_or_else(|e| panic!("gpu_freeze_atoms '{name}': {e}"));
    });
    eprintln!("[gpu] gpu_freeze_atoms '{name}' frozen={ids:?}");
}

fn rhai_gpu_max_force(name: &str) -> f64 {
    with_gpu(|g| {
        let h = g.get(name).unwrap_or_else(|| panic!("gpu_max_force '{name}': no engine"));
        let f = h.last_f.as_ref().unwrap_or_else(|| panic!("gpu_max_force '{name}': last gpu_eval had want_forces=false"));
        let mut m = 0.0f32;
        for &x in f { m = m.max(x.abs()); }
        m as f64
    })
}

fn rhai_gpu_n_batch(name: &str) -> INT {
    with_gpu(|g| {
        let h = g.get(name).unwrap_or_else(|| panic!("gpu_n_batch '{name}': no engine"));
        h.eng.batch() as INT
    })
}

fn rhai_gpu_n_orbs(name: &str) -> INT {
    with_gpu(|g| {
        let h = g.get(name).unwrap_or_else(|| panic!("gpu_n_orbs '{name}': no engine"));
        h.eng.n() as INT
    })
}

fn rhai_gpu_scc_iters(name: &str) -> INT {
    with_gpu(|g| {
        let h = g.get(name).unwrap_or_else(|| panic!("gpu_scc_iters '{name}': no engine"));
        h.last_iters
    })
}

fn rhai_gpu_scc_stalled(name: &str) -> INT {
    with_gpu(|g| {
        let h = g.get(name).unwrap_or_else(|| panic!("gpu_scc_stalled '{name}': no engine"));
        if h.last_stalled { 1 } else { 0 }
    })
}

/// §12 D11: per-system convergence status of the last gpu_scc_mixer call.
/// Returns "converged" | "plateau" | "failed" (batch=1) or joined list.
fn rhai_gpu_scc_status(name: &str) -> String {
    with_gpu(|g| {
        let h = g.get(name).unwrap_or_else(|| panic!("gpu_scc_status '{name}': no engine"));
        if h.last_status.is_empty() { panic!("gpu_scc_status '{name}': no scc_mixer call yet"); }
        h.last_status.join(",")
    })
}

fn rhai_gpu_q_rms(name: &str) -> f64 {
    with_gpu(|g| {
        let h = g.get(name).unwrap_or_else(|| panic!("gpu_q_rms '{name}': no engine — gpu_eval/gpu_measure first"));
        if !h.last_q_rms.is_finite() { panic!("gpu_q_rms '{name}': no finalize yet (gpu_eval/gpu_measure)"); }
        h.last_q_rms
    })
}

fn rhai_gpu_n_atoms(name: &str) -> INT {
    with_gpu(|g| {
        let h = g.get(name).unwrap_or_else(|| panic!("gpu_n_atoms '{name}': no engine"));
        h.eng.n_atoms() as INT
    })
}

fn store_gpu_eval(h: &mut GpuHandle, ev: GpuDftbEval, ctx: &str) -> f64 {
    for (i, &e) in ev.energy.iter().enumerate() {
        if !e.is_finite() { panic!("{ctx}: E[{i}]={e} non-finite"); }
    }
    if let Some(ref f) = ev.forces {
        for (i, &x) in f.iter().enumerate() {
            if !x.is_finite() { panic!("{ctx}: F[{i}]={x} non-finite"); }
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
        let h = g.get_mut(name).unwrap_or_else(|| panic!("gpu_reset_q '{name}': no engine — gpu_new first"));
        h.eng.reset_q0().unwrap_or_else(|e| panic!("gpu_reset_q '{name}': {e}"));
        eprintln!("[gpu] gpu_reset_q '{name}'");
    });
}

fn rhai_gpu_scc_mixer(name: &str, max_iter: INT, tol: f64, mix: INT) -> f64 {
    with_gpu(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("gpu_scc_mixer '{name}': no engine — gpu_new first"));
        let scc = h.eng.scc_mix(max_iter as usize, tol as f32, mix as i32)
            .unwrap_or_else(|e| panic!("gpu_scc_mixer '{name}' mix={mix}: {e}"));
        h.last_rms = scc.rms;
        h.last_iters = scc.n_iters as i64;
        h.last_stalled = scc.stalled;
        h.last_status = scc.statuses.iter().map(|s| format!("{s:?}").to_lowercase()).collect();
        eprintln!("[gpu] gpu_scc_mixer '{name}' mix={mix} rms={:.3e} iters={} stalled={} status={}", scc.rms, scc.n_iters, scc.stalled, h.last_status.join(","));
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
        let h = g.get_mut(name).unwrap_or_else(|| panic!("gpu_occ_repair '{name}': no engine — gpu_new first"));
        h.eng.plan.occ_repair = (mode & 1) != 0;
        h.eng.plan.occ_repair_scc = (mode & 2) != 0;
        eprintln!("[gpu] gpu_occ_repair '{name}' mode={mode} (finalize={} scc={})", h.eng.plan.occ_repair, h.eng.plan.occ_repair_scc);
    });
}

/// §12 D6 A/B: enable/disable Löwdin-X reuse across geometry changes.
fn rhai_gpu_x_reuse(name: &str, on: bool) {
    with_gpu(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("gpu_x_reuse '{name}': no engine — gpu_new first"));
        h.eng.plan.x_reuse = on;
        eprintln!("[gpu] gpu_x_reuse '{name}' on={on}");
    });
}

fn rhai_gpu_measure(name: &str, want_cpu: bool) -> f64 {
    with_gpu(|g| {
        let h = g.get_mut(name).unwrap_or_else(|| panic!("gpu_measure '{name}': no engine — gpu_new first"));
        let ev = h.eng.measure(want_cpu).unwrap_or_else(|e| panic!("gpu_measure '{name}' want_cpu={want_cpu}: {e}"));
        let e0 = store_gpu_eval(h, ev, &format!("gpu_measure '{name}'"));
        eprintln!("[gpu] gpu_measure '{name}' want_cpu={want_cpu} E[0]={e0:.12} q_rms={:.3e} q_max={:.3e}", h.last_q_rms, h.last_q_max);
        e0
    })
}

fn rhai_gpu_cpu_energy(name: &str) -> f64 {
    with_gpu(|g| {
        let h = g.get(name).unwrap_or_else(|| panic!("gpu_cpu_energy '{name}': no engine — gpu_new first"));
        let (e, _, _) = h.eng.cpu_ref().unwrap_or_else(|e| panic!("gpu_cpu_energy '{name}': {e}"));
        if !e.is_finite() { panic!("gpu_cpu_energy '{name}': E={e} non-finite"); }
        eprintln!("[gpu] gpu_cpu_energy '{name}' E={e:.12} Ha");
        e
    })
}

fn rhai_gpu_set_coords(name: &str, xyz: Array) -> INT {
    let (n_atoms, batch) = with_gpu(|g| {
        let h = g.get(name).unwrap_or_else(|| panic!("gpu_set_coords '{name}': no engine — gpu_new first"));
        (h.eng.n_atoms(), h.eng.batch())
    });
    let need = batch * n_atoms * 3;
    if xyz.len() != need {
        panic!("gpu_set_coords '{name}': xyz len {} != 3*batch*n_atoms 3*{batch}*{n_atoms}={need}", xyz.len());
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
        let h = g.get_mut(name).unwrap_or_else(|| panic!("gpu_set_coords '{name}': no engine"));
        h.eng.set_coords(&coords).unwrap_or_else(|e| panic!("gpu_set_coords '{name}': {e}"));
    });
    if batch == 1 {
        sync_geom_coords(name, &coords);
    }
    eprintln!("[gpu] gpu_set_coords '{name}' n_atoms={n_atoms} batch={batch}");
    n_atoms as INT
}

fn rhai_get_xyz(name: &str) -> Array {
    with_state(|s| {
        let st = s.geometries.get(name).unwrap_or_else(|| panic!("get_xyz '{name}': no geometry — load_xyz/make_geom first"));
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
        let st = s.geometries.get(name).unwrap_or_else(|| panic!("get_species '{name}': no geometry — load_xyz/make_geom first"));
        st.elements.iter().map(|e| Dynamic::from(e.symbol().to_string())).collect()
    })
}

fn rhai_assert_finite(x: f64, msg: &str) {
    if !x.is_finite() { panic!("assert_finite failed: {msg}: {x}"); }
}

fn rhai_assert_close(a: f64, b: f64, tol: f64, msg: &str) {
    let d = (a - b).abs();
    if !a.is_finite() || !b.is_finite() || d > tol {
        panic!("assert_close failed: {msg}: a={a} b={b} |d|={d} tol={tol}");
    }
}

fn rhai_die(msg: &str) { panic!("{msg}"); }

fn find_repo_root() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|e| panic!("cwd: {e}"));
    let mut cands = vec![cwd.clone(), cwd.join(".."), cwd.join("../..")];
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        cands.push(PathBuf::from(manifest).join(".."));
    }
    for cand in cands {
        let xyz = cand.join("data/xyz/adenine-thymine.xyz");
        if xyz.is_file() {
            return cand.canonicalize().unwrap_or_else(|e| panic!("canonicalize {}: {e}", cand.display()));
        }
    }
    panic!("cannot find data/xyz/adenine-thymine.xyz from cwd={}", cwd.display());
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
            "--script" | "-s" => { i += 1; script_path = args[i].clone(); }
            "--sk-dir" => { i += 1; sk_dir = args[i].clone(); }
            "--help" | "-h" => {
                eprintln!("Usage: dftb_engine --script <script.rhai> [--sk-dir <path>]");
                eprintln!();
                eprintln!("Rhai functions available:");
                eprintln!("  build_pah(name, shells, acc) -> n_atoms");
                eprintln!("  build_flake(name, radius, shape, passivate, acc) -> n_atoms");
                eprintln!("  build_zigzag(name, width, length, passivate, acc) -> n_atoms");
                eprintln!("  load_xyz(name, path) -> n_atoms");
                eprintln!("  make_geom(name, species_csv, xyz_flat) -> n_atoms");
                eprintln!("  gpu_new(name, sk_dir, batch) -> n_orbs   (GpuDftb, homogeneous replicas)");
                eprintln!("  gpu_scc(name, max_iter, tol) -> rms");
                eprintln!("  gpu_scc_mixer(name, max_iter, tol, mix) -> rms   mix 0=GPU DIIS, 1=GPU simple, 2=host f64 DIIS");
                eprintln!("  gpu_reset_q(name)                       reload q0 + reset DIIS");
                eprintln!("  gpu_set_coords(name, xyz_flat) -> n_atoms");
                eprintln!("  gpu_eval(name, want_forces) -> E[0]      (one finalize; forces optional)");
                eprintln!("  gpu_measure(name, want_cpu) -> E[0]     frozen-H + energy identities; CPU if true");
                eprintln!("  gpu_cpu_energy(name) -> E               independent CPU f64 SCC+rep at replica 0");
                eprintln!("  gpu_energy_i(name, i) -> E[i]");
                eprintln!("  gpu_max_force(name) -> max|F|           (after gpu_eval(..., true))");
                eprintln!("  get_xyz(name) -> xyz_flat               geometry table, Å");
                eprintln!("  gpu_n_batch / gpu_n_orbs / gpu_n_atoms / gpu_scc_iters / gpu_scc_stalled / gpu_q_rms");
                eprintln!("  sparse_new(name, sk_dir) -> n_orbs      (SparseDftb, one system)");
                eprintln!("  sparse_scc(name, max_iter, tol) -> rms");
                eprintln!("  sparse_eval(name, want_forces) -> E     (after sparse_scc; forces optional)");
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
                eprintln!("  run_sparse_purify_geom(name, r_max, max_iter, tol) -> r_i  (geometric mask)");
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

    // Default SK dir if not provided
    if sk_dir.is_empty() {
        sk_dir = std::env::var("RUST_DFTB_SK_DIR").unwrap_or_else(|_| {
            "/home/prokophapala/git_SW/dftbplus/external/slakos/origin/mio-1-1".to_string()
        });
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
    engine.register_fn("gpu_max_force", rhai_gpu_max_force);
    engine.register_fn("gpu_bench", rhai_gpu_bench);
    engine.register_fn("gpu_fire_step", rhai_gpu_fire_step);
    engine.register_fn("gpu_relax", rhai_gpu_relax);
    engine.register_fn("gpu_freeze_atoms", rhai_gpu_freeze_atoms);
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
    engine.register_fn("get_species", rhai_get_species);
    engine.register_fn("sparse_new", rhai_sparse_new);
    engine.register_fn("sparse_new", rhai_sparse_new_cut);
    engine.register_fn("sparse_new", rhai_sparse_new_full);
    engine.register_fn("sparse_scc", rhai_sparse_scc);
    engine.register_fn("sparse_eval", rhai_sparse_eval);
    engine.register_fn("sparse_max_force", rhai_sparse_max_force);
    engine.register_fn("sparse_force_at", rhai_sparse_force_at);
    engine.register_fn("sparse_fire_step", rhai_sparse_fire_step);
    engine.register_fn("sparse_md_step", rhai_sparse_md_step);
    engine.register_fn("sparse_relax", rhai_sparse_relax);
    engine.register_fn("sparse_set_coords", rhai_sparse_set_coords);
    engine.register_fn("sparse_n_atoms", rhai_sparse_n_atoms);
    engine.register_fn("sparse_n_orbs", rhai_sparse_n_orbs);
    engine.register_fn("sparse_scc_iters", rhai_sparse_scc_iters);
    engine.register_fn("sparse_tr_ks", rhai_sparse_tr_ks);
    engine.register_fn("sparse_charges", rhai_sparse_charges);
    engine.register_fn("sparse_vibrations", rhai_sparse_vibrations);
    engine.register_fn("sparse_tc2_tol", rhai_sparse_tc2_tol);
    engine.register_fn("sparse_ns_tol", rhai_sparse_ns_tol);
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
        Err(e) => { eprintln!("ERROR reading script {script_path}: {e}"); std::process::exit(1); }
    };

    eprintln!("Running script: {script_path}");
    eprintln!("SK dir: {}", scope.get_value::<String>("SK_DIR").unwrap_or_default());
    eprintln!();

    if let Err(e) = engine.run_with_scope(&mut scope, &script) {
        eprintln!("ERROR: rhai script failed: {e}");
        std::process::exit(1);
    }

    eprintln!("\nScript completed successfully.");
}
