//! dftb_engine — Rhai-scriptable DFTB + sparse BSR4 engine.
//!
//! This binary loads a Rhai script that orchestrates DFTB calculations.
//! The engine exposes native functions to Rhai for:
//!   - Graphene geometry generation (PAH, flake, ribbon)
//!   - Dense DFTB SCC calculation (CPU, returns H/S/density/charges/energy)
//!   - Sparse BSR4 purification (GPU, Z→K₀→TC2→K)
//!   - Result comparison and diagnostics
//!   - Charge/eigenvalue access: `get_charges`, `get_eigenvalues`, `save_charges`,
//!     `save_eigenvalues`, `get_sparse_charges`, `save_sparse_charges`
//!   - Frontier orbitals: `davidson_homo_lumo(name, n_target)` — partial
//!     generalized eigensolver (see `sparse::davidson`)
//!   - Convergence history: `save_convergence(name, path)`
//!
//! Usage:
//!   dftb_engine --script test_graphene.rhai
//!   dftb_engine --script test_graphene.rhai --sk-dir /path/to/mio-1-1

use rust_dftb::geometry::{self, A_CC, Element, FlakeShape, NanoStructure};
use rust_dftb::methods::sparse::{
    Bsr4Matrix, SparseBsr4Config, SparseBsr4Gpu,
    build_full_mask, build_geometric_mask,
};
use rust_dftb::{
    HamiltonianBuilder, SccResult, load_sk_for_species,
};
use nalgebra::{DMatrix, SymmetricEigen};
use rhai::{Array, Dynamic, Engine, Scope, INT};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

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

    // Check all atoms have 4 orbitals (BSR4 requirement)
    for sp in &species {
        if sp == "H" {
            eprintln!("ERROR: geometry contains H — BSR4 requires all atoms to have 4 orbitals (s,p). Use pure-carbon systems (no passivation) for sparse tests.");
            return Dynamic::from_float(f64::NAN);
        }
    }

    eprintln!("Loading SK tables from {sk_dir} ...");
    let sk = match load_sk_for_species(sk_dir, &species) {
        Ok(sk) => sk,
        Err(e) => { eprintln!("ERROR loading SK: {e}"); return Dynamic::from_float(f64::NAN); }
    };

    let builder = HamiltonianBuilder::new(sk);
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
    eprintln!("Building non-SCC Hamiltonian ...");
    match builder.build_non_scc_sp_only(&species, &coords) {
        Ok(ham) => {
            let n_orbs = ham.h0.nrows();
            eprintln!("  H0: {n_orbs}×{n_orbs}");

            // Solve generalized eigenproblem H0 c = S c ε
            let s_inv_sqrt = {
                let se = SymmetricEigen::new(ham.s.clone());
                let mut d = DMatrix::<f64>::zeros(n_orbs, n_orbs);
                for i in 0..n_orbs {
                    d[(i, i)] = 1.0 / se.eigenvalues[i].max(1e-12).sqrt();
                }
                &se.eigenvectors * &d * se.eigenvectors.transpose()
            };
            let h_orth = &s_inv_sqrt * &ham.h0 * &s_inv_sqrt;
            let he = SymmetricEigen::new(h_orth);
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

            // Mulliken charges
            let ds = &density * &ham.s;
            let mut charges = Vec::with_capacity(species.len());
            let mut off = 0;
            for i in 0..species.len() {
                let n_orb_i = 4; // sp basis
                let q = (0..n_orb_i).map(|k| ds[(off + k, off + k)]).sum::<f64>();
                charges.push(q);
                off += n_orb_i;
            }
            eprintln!("  Mulliken charges: {:?}", charges.iter()
                .map(|q| format!("{q:.4}")).collect::<Vec<_>>());

            let scc = SccResult {
                h0: ham.h0.clone(),
                h_scc: ham.h0.clone(), // non-SCC: H_scc = H0
                s: ham.s.clone(),
                density,
                eigenvalues: DMatrix::<f64>::zeros(0, 0).diagonal().clone(),
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
    if scc.h0.nrows() != n_orbs {
        eprintln!("ERROR: H0 size {} != expected {} (n_atom×4). BSR4 requires sp basis (4 orbs/atom).",
            scc.h0.nrows(), n_orbs);
        return Dynamic::from_float(f64::NAN);
    }

    eprintln!("Sparse BSR4 purification on '{name}' ({n_atom} atoms, {n_orbs} orbitals) ...");

    // Convert dense H, S to BSR4 format with full mask
    let mask = build_full_mask(n_atom);
    let h_bsr = dense_to_bsr4(&scc.h_scc, n_atom, &mask);
    let s_bsr = dense_to_bsr4(&scc.s, n_atom, &mask);

    // Count occupied orbitals
    let n_electrons: f64 = st.elements.iter().map(|e| {
        match e { Element::C => 4.0, Element::N => 5.0, Element::O => 6.0, Element::B => 3.0, _ => 1.0 }
    }).sum();
    let n_occ = (n_electrons / 2.0) as f32;
    eprintln!("  n_occ = {n_occ}");

    // Init GPU
    let config = SparseBsr4Config::default();
    let gpu = match SparseBsr4Gpu::new(config) {
        Ok(g) => g,
        Err(e) => { eprintln!("ERROR GPU init: {e}"); return Dynamic::from_float(f64::NAN); }
    };

    // 1. Z ≈ S⁻¹
    eprintln!("  Newton-Schulz Z ≈ S⁻¹ ...");
    let (z, r_z, z_iters) = match gpu.newton_schulz_inverse(&s_bsr, &mask, &mask, 30, 1e-4, 3) {
        Ok(r) => r,
        Err(e) => { eprintln!("ERROR Newton-Schulz: {e}"); return Dynamic::from_float(f64::NAN); }
    };
    eprintln!("    Z: {z_iters} iters, R_Z = {r_z:e}");

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
    let (k_final, r_i, tr_ks, tc2_iters, history) = match gpu.tc2_purify(&k0, &s_bsr, n_occ, &mask, &mask, max_iter as usize, tol as f32) {
        Ok(r) => r,
        Err(e) => { eprintln!("ERROR TC2: {e}"); return Dynamic::from_float(f64::NAN); }
    };
    eprintln!("    TC2: {tc2_iters} iters, R_I={r_i:e}, Tr(KS)={tr_ks:.6}");

    // 5. R_H = ||HKS - SKH||
    let r_h = match gpu.hamiltonian_residual(&h_bsr, &k_final, &s_bsr, &mask) {
        Ok(r) => r,
        Err(e) => { eprintln!("ERROR R_H: {e}"); return Dynamic::from_float(f64::NAN); }
    };
    let r_h_norm = r_h / (n_orbs as f32).sqrt();
    eprintln!("    R_H = {r_h:e} (normalized {r_h_norm:e})");

    // 6. Mulliken charges from sparse K
    let ks = match gpu.matmul_masked_bsym(&k_final, &s_bsr, &mask) {
        Ok(m) => m,
        Err(e) => { eprintln!("ERROR KS: {e}"); return Dynamic::from_float(f64::NAN); }
    };
    let mulliken = match gpu.mulliken(&ks) {
        Ok(m) => m,
        Err(e) => { eprintln!("ERROR Mulliken: {e}"); return Dynamic::from_float(f64::NAN); }
    };
    eprintln!("    sparse Mulliken: {:?}", mulliken.iter()
        .map(|q| format!("{q:.4}")).collect::<Vec<_>>());

    let k_dense = k_final.to_dense();
    let result = SparseResult {
        k_dense, n_atom, n_occ, r_i, r_h: r_h_norm, tr_ks,
        mulliken, iters: tc2_iters, history,
    };
    with_state(|s| { s.sparse_results.insert(name.to_string(), result); });

    Dynamic::from_float(r_i as f64)
}

/// Compare sparse K with dense density matrix.
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
                eprintln!("  save_xyz(name, path) -> bool");
                eprintln!("  run_dftb_scc(name, sk_dir, max_iter, tol) -> energy");
                eprintln!("  run_dftb_nonscc(name, sk_dir) -> energy");
                eprintln!("  run_sparse_purify(name, max_iter, tol) -> r_i");
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
    engine.register_fn("run_dftb_scc", rhai_run_dftb_scc);
    engine.register_fn("run_dftb_nonscc", rhai_run_dftb_nonscc);
    engine.register_fn("run_sparse_purify", rhai_run_sparse_purify);
    engine.register_fn("compare_density", rhai_compare_density);
    engine.register_fn("compare_charges", rhai_compare_charges);
    engine.register_fn("get_energy", rhai_get_energy);
    engine.register_fn("get_n_atoms", rhai_get_n_atoms);
    engine.register_fn("ftos", rhai_ftos);
    engine.register_fn("itos", rhai_itos);
    engine.register_fn("save_convergence", rhai_save_convergence);
    engine.register_fn("get_charges", rhai_get_charges);
    engine.register_fn("get_sparse_charges", rhai_get_sparse_charges);
    engine.register_fn("get_eigenvalues", rhai_get_eigenvalues);
    engine.register_fn("get_homo_lumo", rhai_get_homo_lumo);
    engine.register_fn("save_charges", rhai_save_charges);
    engine.register_fn("save_sparse_charges", rhai_save_sparse_charges);
    engine.register_fn("save_eigenvalues", rhai_save_eigenvalues);
    engine.register_fn("davidson_homo_lumo", rhai_davidson_homo_lumo);

    // Register constants
    let mut scope = Scope::new();
    scope.push_constant("SK_DIR", sk_dir);
    scope.push_constant("A_CC", A_CC);
    scope.push_constant("DEFAULT_TOL", 1e-10f64);
    scope.push_constant("DEFAULT_MAX_ITER", 1000i64);

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
