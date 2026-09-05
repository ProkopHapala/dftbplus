//! CPU reference calculation for H-bond switching scans.
//!
//! Computes and caches DFTB SCC reference data for proton-transfer scans
//! in hydrogen-bonded dimers. Saves full per-point data (H0, S, H_scc,
//! density, eigenvalues, charges, energy) so the GPU parity test can
//! compare against it without recomputing.
//!
//! Systems:
//!   - Formic acid dimer (28 orbs): double proton transfer, 1D and 2D
//!   - Formic acid + azaindole mixed dimer (56 orbs): proton transfer
//!
//! Usage:
//!   cargo run --example hbond_ref -- --xyz data/xyz/formic_dimer.xyz \
//!       --mode scc --scan 1d --n 21 \
//!       --h1 4 --donor1 3 --acceptor1 7 \
//!       --h2 9 --donor2 8 --acceptor2 2 \
//!       --out formic_dimer_1d_scc.csv --data-dir cache/formic_dimer_1d_scc
//!
//!   cargo run --example hbond_ref -- --xyz data/xyz/formic_dimer.xyz \
//!       --mode scc --scan 2d --n 11 --n2 11 \
//!       --h1 4 --donor1 3 --acceptor1 7 \
//!       --h2 9 --donor2 8 --acceptor2 2 \
//!       --out formic_dimer_2d_scc.csv --data-dir cache/formic_dimer_2d_scc

use rust_dftb::methods::dftb::forces::{compute_scc_forces, parse_repulsive_spline, RepulsiveSpline};
use rust_dftb::{
    load_sk_for_species, parse_xyz, DftbOutput, HamiltonianBuilder, SccResult,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

const ANG2BOHR: f64 = 1.889_726_133;
const MIN_NEIGH_DIST: f64 = 1.0e-2;

// ── Repulsive energy (same as scan.rs) ────────────────────────────────────

fn repulsive_energy(sk_dir: &str, species: &[String], coords: &[[f64; 3]]) -> f64 {
    let n = coords.len();
    let mut cache: HashMap<(String, String), Option<RepulsiveSpline>> = HashMap::new();
    let mut e_rep = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let key = (species[i].clone(), species[j].clone());
            let spline = if let Some(s) = cache.get(&key) {
                s.clone()
            } else {
                let p1 = format!("{}/{}-{}.skf", sk_dir, key.0, key.1);
                let p2 = format!("{}/{}-{}.skf", sk_dir, key.1, key.0);
                let s = if Path::new(&p1).exists() {
                    parse_repulsive_spline(&p1).ok().flatten()
                } else if Path::new(&p2).exists() {
                    parse_repulsive_spline(&p2).ok().flatten()
                } else {
                    None
                };
                cache.insert(key.clone(), s.clone());
                s
            };
            let Some(spline) = spline else { continue };
            let dx = coords[j][0] - coords[i][0];
            let dy = coords[j][1] - coords[i][1];
            let dz = coords[j][2] - coords[i][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < MIN_NEIGH_DIST * MIN_NEIGH_DIST {
                continue;
            }
            let r_bohr = r2.sqrt() * ANG2BOHR;
            let (e, _de) = spline.eval(r_bohr);
            e_rep += e;
        }
    }
    e_rep
}

// ── Geometry interpolation ────────────────────────────────────────────────

/// Interpolate H atom position between its reactant position and the
/// mirrored product position (mirror across the donor-acceptor midpoint).
/// t=0: H at reactant position (bonded to donor).
/// t=1: H at product position (bonded to acceptor, mirrored across midpoint).
/// This keeps the H at a physically reasonable distance from both Os
/// throughout the scan.
fn interp_h_pos(
    h_reactant: &[f64; 3],
    donor: &[f64; 3],
    acceptor: &[f64; 3],
    t: f64,
) -> [f64; 3] {
    // Product = mirror of reactant H across the donor-acceptor midpoint
    let mid = [
        (donor[0] + acceptor[0]) * 0.5,
        (donor[1] + acceptor[1]) * 0.5,
        (donor[2] + acceptor[2]) * 0.5,
    ];
    let h_product = [
        2.0 * mid[0] - h_reactant[0],
        2.0 * mid[1] - h_reactant[1],
        2.0 * mid[2] - h_reactant[2],
    ];
    [
        h_reactant[0] + t * (h_product[0] - h_reactant[0]),
        h_reactant[1] + t * (h_product[1] - h_reactant[1]),
        h_reactant[2] + t * (h_product[2] - h_reactant[2]),
    ]
}

/// Generate geometry for a 1D scan point.
/// Moves h1 and h2 in opposite directions, both at param t.
fn make_geom_1d(
    base_coords: &[[f64; 3]],
    h1: usize, donor1: usize, acceptor1: usize,
    h2: usize, donor2: usize, acceptor2: usize,
    t: f64,
) -> Vec<[f64; 3]> {
    let mut coords = base_coords.to_vec();
    coords[h1] = interp_h_pos(&base_coords[h1], &base_coords[donor1], &base_coords[acceptor1], t);
    coords[h2] = interp_h_pos(&base_coords[h2], &base_coords[donor2], &base_coords[acceptor2], t);
    coords
}

/// Generate geometry for a 2D scan point.
/// h1 moves at param t1, h2 moves at param t2 (independent).
fn make_geom_2d(
    base_coords: &[[f64; 3]],
    h1: usize, donor1: usize, acceptor1: usize,
    h2: usize, donor2: usize, acceptor2: usize,
    t1: f64, t2: f64,
) -> Vec<[f64; 3]> {
    let mut coords = base_coords.to_vec();
    coords[h1] = interp_h_pos(&base_coords[h1], &base_coords[donor1], &base_coords[acceptor1], t1);
    coords[h2] = interp_h_pos(&base_coords[h2], &base_coords[donor2], &base_coords[acceptor2], t2);
    coords
}

// ── Per-point data saving ─────────────────────────────────────────────────

fn save_point_data(
    dir: &Path,
    idx: usize,
    t1: f64,
    t2: f64,
    species: &[String],
    coords: &[[f64; 3]],
    scc: &SccResult,
    e_rep: f64,
    e_total: f64,
) -> std::io::Result<()> {
    let pt_dir = dir.join(format!("pt_{:04}", idx));
    std::fs::create_dir_all(&pt_dir)?;

    // geometry.xyz
    let mut f = File::create(pt_dir.join("geometry.xyz"))?;
    writeln!(f, "{}", species.len())?;
    writeln!(f, "pt={} t1={:.6} t2={:.6}", idx, t1, t2)?;
    for (sp, c) in species.iter().zip(coords.iter()) {
        writeln!(f, "{sp} {:.10} {:.10} {:.10}", c[0], c[1], c[2])?;
    }

    // Matrices
    for (name, mat) in [
        ("h0.dat", &scc.h0),
        ("h_scc.dat", &scc.h_scc),
        ("s.dat", &scc.s),
        ("density.dat", &scc.density),
    ] {
        DftbOutput::write_square(pt_dir.join(name).to_str().unwrap(), mat)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    }

    // eigenvalues.txt
    let mut f = File::create(pt_dir.join("eigenvalues.txt"))?;
    writeln!(f, "# orbital eigenvalues (Hartree)")?;
    for (k, e) in scc.eigenvalues.iter().enumerate() {
        writeln!(f, "{k} {:.16e}", e)?;
    }

    // charges.txt
    let mut f = File::create(pt_dir.join("charges.txt"))?;
    writeln!(f, "# atom Mulliken_charge reference_q0")?;
    for (a, (q, q0)) in scc.charges.iter().zip(scc.q0.iter()).enumerate() {
        writeln!(f, "{a} {:.16e} {:.16e}", q, q0)?;
    }

    // energy.txt
    let mut f = File::create(pt_dir.join("energy.txt"))?;
    writeln!(f, "# t1 t2 electronic_scc_energy repulsive_energy total_energy n_iter")?;
    writeln!(f, "{:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {}",
        t1, t2, scc.energy, e_rep, e_total, scc.n_iter)?;

    Ok(())
}

// ── Geometry optimization (FIRE algorithm) ─────────────────────────────────

/// FIRE (Fast Inertial Relaxation Engine) optimizer.
/// Ref: Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006).
/// Simple, robust, doesn't need line search. Works well for molecular geometry.
struct FireOptimizer {
    dt: f64,           // time step
    dt_max: f64,       // max time step
    vmax: f64,         // max velocity per atom
    n_min: usize,      // steps after power-on before checking
    f_inc: f64,        // dt increment factor
    f_dec: f64,        // dt decrement factor
    alpha_start: f64,   // initial mixing
    alpha: f64,        // current mixing
    f_alpha: f64,      // alpha decay
    velocities: Vec<[f64; 3]>,
    n_pos: usize,      // consecutive steps with P > 0
    n_neg: usize,      // consecutive steps with P < 0
}

impl FireOptimizer {
    fn new(n_atoms: usize, dt: f64) -> Self {
        Self {
            dt, dt_max: dt * 5.0, vmax: 2.0,
            n_min: 10, f_inc: 1.1, f_dec: 0.7,
            alpha_start: 0.1, alpha: 0.1, f_alpha: 0.95,
            velocities: vec![[0.0; 3]; n_atoms],
            n_pos: 0, n_neg: 0,
        }
    }

    /// Perform one FIRE step. forces = -dE/dx (Hartree/Å).
    /// Returns new coordinates and the max force component.
    fn step(&mut self, coords: &[[f64; 3]], forces: &[[f64; 3]]) -> (Vec<[f64; 3]>, f64) {
        let n = coords.len();
        let mut max_f = 0.0f64;
        for i in 0..n {
            for c in 0..3 {
                max_f = max_f.max(forces[i][c].abs());
            }
        }

        // Power P = F · V
        let mut p = 0.0f64;
        for i in 0..n {
            for c in 0..3 {
                p += forces[i][c] * self.velocities[i][c];
            }
        }

        if p > 0.0 {
            self.n_pos += 1;
            self.n_neg = 0;
            if self.n_pos > self.n_min {
                self.dt = (self.dt * self.f_inc).min(self.dt_max);
                self.alpha *= self.f_alpha;
            }
        } else {
            self.n_pos = 0;
            self.n_neg += 1;
            if self.n_neg > 0 {
                self.dt *= self.f_dec;
                self.alpha = self.alpha_start;
            }
            // Freeze: zero velocities
            for i in 0..n { self.velocities[i] = [0.0; 3]; }
        }

        // Mix velocity: V = (1-alpha)*V + alpha*|F|*F_hat
        for i in 0..n {
            let f_norm = (forces[i][0].powi(2) + forces[i][1].powi(2) + forces[i][2].powi(2)).sqrt();
            let f_hat = if f_norm > 1e-12 {
                [forces[i][0]/f_norm, forces[i][1]/f_norm, forces[i][2]/f_norm]
            } else { [0.0; 3] };
            for c in 0..3 {
                self.velocities[i][c] = (1.0 - self.alpha) * self.velocities[i][c] + self.alpha * f_norm * f_hat[c];
            }
        }

        // Velocity Verlet: x += V*dt + 0.5*F*dt^2
        // Cap max displacement per atom to prevent geometry blow-up
        let max_disp = 0.1; // Å — maximum atomic displacement per step
        let mut new_coords = coords.to_vec();
        for i in 0..n {
            for c in 0..3 {
                self.velocities[i][c] += forces[i][c] * self.dt;
                // Cap velocity
                let v = self.velocities[i][c];
                if v.abs() > self.vmax {
                    self.velocities[i][c] = v.signum() * self.vmax;
                }
                new_coords[i][c] = coords[i][c] + self.velocities[i][c] * self.dt;
            }
            // Cap total displacement per atom
            let dx = [new_coords[i][0]-coords[i][0], new_coords[i][1]-coords[i][1], new_coords[i][2]-coords[i][2]];
            let disp = (dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]).sqrt();
            if disp > max_disp {
                let scale = max_disp / disp;
                for c in 0..3 {
                    new_coords[i][c] = coords[i][c] + dx[c] * scale;
                    self.velocities[i][c] *= scale; // also scale velocity back
                }
            }
        }

        (new_coords, max_f)
    }
}

/// Optimize geometry using the persistent DftbCpu solver with warm-started charges.
/// This is the new architecture: static state built once, per-geometry state rebuilt
/// only on geometry change, charges carried forward between steps.
fn optimize_geometry_persistent(
    sk_dir: &str,
    species: &[String],
    coords: &[[f64; 3]],
    frozen: &std::collections::HashSet<usize>,
    restraints: &[(usize, [f64; 3], f64)],
    max_opt_iter: usize,
    f_tol: f64,
    scc_max_iter: usize,
    scc_tol: f64,
    traj_path: Option<&Path>,
    hist_path: Option<&Path>,
    mixer: &str,
) -> Result<(Vec<[f64; 3]>, f64, usize), String> {
    use rust_dftb::methods::dftb::dftb_cpu::{DftbCpu, CpuSccResult};
    use rust_dftb::methods::dftb::forces::parse_all_repulsive;

    let n = coords.len();
    let mut opt = FireOptimizer::new(n, 1.0);
    let mut current = coords.to_vec();

    // Load SK and build persistent solver state ONCE
    let sk = load_sk_for_species(sk_dir, species)
        .map_err(|e| format!("Failed to load SK: {e}"))?;

    // Parse repulsive splines ONCE (no Strings/HashMaps in hot path)
    let unique_species: Vec<String> = {
        let mut seen = std::collections::HashSet::new();
        let mut uniq = Vec::new();
        for s in species {
            if seen.insert(s.clone()) { uniq.push(s.clone()); }
        }
        uniq
    };
    let repulsive = parse_all_repulsive(sk_dir, &unique_species, unique_species.len())
        .map_err(|e| format!("parse_all_repulsive failed: {e}"))?;

    let mut solver = DftbCpu::new(sk, species.to_vec())
        .map_err(|e| format!("DftbCpu::new failed: {e}"))?;

    if mixer == "broyden" {
        solver.use_broyden(0.5);
        eprintln!("  [opt] Mixer: Broyden (alpha=0.5)");
    } else {
        eprintln!("  [opt] Mixer: DIIS (history=10, alpha=0.5)");
    }

    eprintln!("  [opt] FIRE: dt0={}, dt_max={}, vmax={}, max_disp=0.1Å, max_iter={}, f_tol={:.2e}", 1.0, 5.0, 2.0, max_opt_iter, f_tol);
    eprintln!("  [opt] SCC: max_iter={}, tol={:.2e} (warm-started, persistent state)", scc_max_iter, scc_tol);
    eprintln!("  [opt] frozen atoms: {:?}", frozen.iter().collect::<Vec<_>>());

    // First geometry: start from neutral charges
    solver.update_geometry(&current).map_err(|e| format!("update_geometry failed: {e}"))?;
    solver.reset_charges();

    let mut traj_file = if let Some(p) = traj_path {
        if let Some(parent) = p.parent() { std::fs::create_dir_all(parent).unwrap(); }
        Some(File::create(p).unwrap_or_else(|e| panic!("Cannot create trajectory {}: {e}", p.display())))
    } else { None };
    let mut hist_file = if let Some(p) = hist_path {
        if let Some(parent) = p.parent() { std::fs::create_dir_all(parent).unwrap(); }
        let mut f = File::create(p).unwrap_or_else(|e| panic!("Cannot create history {}: {e}", p.display()));
        writeln!(f, "iter,E_elec,max_F,rms_F,scc_iter,dt").unwrap();
        Some(f)
    } else { None };

    let mut iter = 0;
    let mut max_f = f64::INFINITY;
    let mut last_energy = 0.0f64;
    let t0_total = std::time::Instant::now();

    while iter < max_opt_iter && max_f > f_tol {
        let t0 = std::time::Instant::now();

        // SCC with warm-started charges (charges from previous geometry)
        solver.solve_scc(scc_max_iter, scc_tol)
            .map_err(|e| format!("SCC failed at opt iter {iter}: {e}"))?;
        let scc = solver.build_result();
        let t_scc = t0.elapsed();
        last_energy = scc.energy;

        // Forces — use cached state from DftbCpu (no re-diagonalization, no rebuilds)
        let t0 = std::time::Instant::now();
        let forces = solver.compute_forces(&scc, &repulsive)
            .map_err(|e| format!("forces failed at opt iter {iter}: {e}"))?;
        let t_force = t0.elapsed();

        // Zero forces on frozen atoms
        let mut f = forces.forces.clone();
        for &i in frozen { f[i] = [0.0; 3]; }

        // Add harmonic restraint forces
        for (i, target, k) in restraints {
            for c in 0..3 { f[*i][c] -= k * (current[*i][c] - target[c]); }
        }

        // RMS force
        let mut rms_f = 0.0f64;
        for fi in &f { for c in 0..3 { rms_f += fi[c]*fi[c]; } }
        rms_f = (rms_f / (n as f64 * 3.0)).sqrt();

        let (new_coords, mf) = opt.step(&current, &f);
        current = new_coords;
        max_f = mf;

        eprintln!("  [opt] iter {:>3}  E={:.8e}  max|F|={:.4e}  rms|F|={:.4e}  scc={}  dt={:.3}  t={:.2}s  [scc={:.1}ms f={:.1}ms]",
            iter, last_energy, max_f, rms_f, scc.n_iter, opt.dt,
            t_scc.as_secs_f64() + t_force.as_secs_f64(),
            t_scc.as_secs_f64() * 1e3, t_force.as_secs_f64() * 1e3);

        if let Some(tf) = &mut traj_file {
            writeln!(tf, "{}", n).unwrap();
            writeln!(tf, "iter={} E={:.10e} max_F={:.6e} rms_F={:.6e}", iter, last_energy, max_f, rms_f).unwrap();
            for (sp, c) in species.iter().zip(current.iter()) {
                writeln!(tf, "{sp} {:.10} {:.10} {:.10}", c[0], c[1], c[2]).unwrap();
            }
        }
        if let Some(hf) = &mut hist_file {
            writeln!(hf, "{},{:.16e},{:.10e},{:.10e},{},{:.6e}", iter, last_energy, max_f, rms_f, scc.n_iter, opt.dt).unwrap();
        }

        // Update geometry for next step (charges are kept = warm start)
        solver.update_geometry(&current).map_err(|e| format!("update_geometry failed at iter {iter}: {e}"))?;

        iter += 1;
    }

    // Final energy with converged charges
    solver.solve_scc(scc_max_iter, scc_tol)
        .map_err(|e| format!("final SCC failed: {e}"))?;
    let scc = solver.build_result();
    last_energy = scc.energy;

    let elapsed = t0_total.elapsed();
    eprintln!("  [opt] DONE: {} iters, {:.2}s, final E={:.10e}", iter, elapsed.as_secs_f64(), last_energy);

    Ok((current, last_energy, iter))
}

/// Optimize geometry with DFTB SCC forces using FIRE.
/// `frozen` = set of atom indices to keep fixed (not moved).
/// `restraints` = harmonic restraints pulling atoms toward target positions.
/// `traj_path` = optional path to save trajectory XYZ (every step).
/// `hist_path` = optional path to save energy/force history CSV.
/// Returns optimized coords and final energy.
fn optimize_geometry(
    builder: &HamiltonianBuilder,
    species: &[String],
    coords: &[[f64; 3]],
    frozen: &std::collections::HashSet<usize>,
    restraints: &[(usize, [f64; 3], f64)],
    max_opt_iter: usize,
    f_tol: f64,
    scc_max_iter: usize,
    scc_tol: f64,
    traj_path: Option<&Path>,
    hist_path: Option<&Path>,
) -> Result<(Vec<[f64; 3]>, f64, usize), String> {
    let n = coords.len();
    let dt0 = 1.0; // FIRE timestep
    let mut opt = FireOptimizer::new(n, dt0);
    let mut current = coords.to_vec();

    eprintln!("  [opt] FIRE: dt0={}, dt_max={}, vmax={}, max_disp=0.1Å, max_iter={}, f_tol={:.2e}", dt0, dt0*5.0, 2.0, max_opt_iter, f_tol);
    eprintln!("  [opt] SCC: max_iter={}, tol={:.2e}", scc_max_iter, scc_tol);
    eprintln!("  [opt] frozen atoms: {:?}", frozen.iter().collect::<Vec<_>>());
    if !restraints.is_empty() {
        eprintln!("  [opt] harmonic restraints: {} atom(s)", restraints.len());
        for (i, target, k) in restraints {
            eprintln!("  [opt]   atom {} → ({:.3},{:.3},{:.3}) k={:.1}", i, target[0], target[1], target[2], k);
        }
    }

    // Open trajectory and history files
    let mut traj_file = if let Some(p) = traj_path {
        if let Some(parent) = p.parent() { std::fs::create_dir_all(parent).unwrap(); }
        Some(File::create(p).unwrap_or_else(|e| panic!("Cannot create trajectory {}: {e}", p.display())))
    } else { None };
    let mut hist_file = if let Some(p) = hist_path {
        if let Some(parent) = p.parent() { std::fs::create_dir_all(parent).unwrap(); }
        let mut f = File::create(p).unwrap_or_else(|e| panic!("Cannot create history {}: {e}", p.display()));
        writeln!(f, "iter,E_elec,max_F,rms_F,scc_iter,dt").unwrap();
        Some(f)
    } else { None };

    let mut iter = 0;
    let mut max_f = f64::INFINITY;
    let mut last_energy = 0.0f64;
    let t0_total = std::time::Instant::now();

    while iter < max_opt_iter && max_f > f_tol {
        let t0 = std::time::Instant::now();
        let scc = builder.build_scc(species, &current, scc_max_iter, scc_tol)
            .map_err(|e| format!("SCC failed at opt iter {iter}: {e}"))?;
        let t_scc = t0.elapsed();
        last_energy = scc.energy;

        let t0 = std::time::Instant::now();
        let forces = compute_scc_forces(builder, species, &current, &scc)
            .map_err(|e| format!("forces failed at opt iter {iter}: {e}"))?;
        let t_force = t0.elapsed();

        // Zero forces on frozen atoms
        let mut f = forces.forces.clone();
        for &i in frozen { f[i] = [0.0; 3]; }

        // Add harmonic restraint forces
        for (i, target, k) in restraints {
            for c in 0..3 { f[*i][c] -= k * (current[*i][c] - target[c]); }
        }

        // RMS force
        let mut rms_f = 0.0f64;
        for fi in &f { for c in 0..3 { rms_f += fi[c]*fi[c]; } }
        rms_f = (rms_f / (n as f64 * 3.0)).sqrt();

        let (new_coords, mf) = opt.step(&current, &f);
        current = new_coords;
        max_f = mf;

        eprintln!("  [opt] iter {:>3}  E={:.8e}  max|F|={:.4e}  rms|F|={:.4e}  scc={}  dt={:.3}  t={:.2}s",
            iter, last_energy, max_f, rms_f, scc.n_iter, opt.dt, t_scc.as_secs_f64() + t_force.as_secs_f64());

        // Save trajectory frame
        if let Some(tf) = &mut traj_file {
            writeln!(tf, "{}", n).unwrap();
            writeln!(tf, "iter={} E={:.10e} max_F={:.6e} rms_F={:.6e}", iter, last_energy, max_f, rms_f).unwrap();
            for (sp, c) in species.iter().zip(current.iter()) {
                writeln!(tf, "{sp} {:.10} {:.10} {:.10}", c[0], c[1], c[2]).unwrap();
            }
        }
        // Save history
        if let Some(hf) = &mut hist_file {
            writeln!(hf, "{},{:.16e},{:.10e},{:.10e},{},{:.6e}", iter, last_energy, max_f, rms_f, scc.n_iter, opt.dt).unwrap();
        }
        iter += 1;
    }

    // Final energy
    let scc = builder.build_scc(species, &current, scc_max_iter, scc_tol)
        .map_err(|e| format!("final SCC failed: {e}"))?;

    let elapsed = t0_total.elapsed();
    if max_f <= f_tol {
        eprintln!("  [opt] CONVERGED in {} iterations ({:.1}s), max|F|={:.6e} <= tol={:.2e}", iter, elapsed.as_secs_f64(), max_f, f_tol);
    } else {
        eprintln!("  [opt] REACHED max_iter={} ({:.1}s), max|F|={:.6e}", max_opt_iter, elapsed.as_secs_f64(), max_f);
    }
    eprintln!("  [opt] final energy: {:.10e} Hartree", scc.energy);

    Ok((current, scc.energy, iter))
}

/// Save geometry to XYZ file.
fn save_xyz(path: &Path, species: &[String], coords: &[[f64; 3]], comment: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut f = File::create(path).unwrap();
    writeln!(f, "{}", species.len()).unwrap();
    writeln!(f, "{comment}").unwrap();
    for (sp, c) in species.iter().zip(coords.iter()) {
        writeln!(f, "{sp} {:.10} {:.10} {:.10}", c[0], c[1], c[2]).unwrap();
    }
}

// ── CLI ───────────────────────────────────────────────────────────────────

struct Args {
    xyz: String,
    mode: String,       // "scc", "optimize", "switch"
    scan: String,       // "1d" or "2d"
    n: usize,           // points along t1
    n2: usize,          // points along t2 (for 2d)
    h1: usize, donor1: usize, acceptor1: usize,
    h2: usize, donor2: usize, acceptor2: usize,
    out: String,
    data_dir: String,
    max_iter: usize,
    tol: f64,
    // Optimization args
    opt_max_iter: usize,    // max FIRE iterations
    opt_f_tol: f64,         // force convergence threshold (Hartree/Å)
    opt_frozen: String,     // comma-separated atom indices to freeze during opt
    // Switch args
    switch_from: String,    // optimized reactant XYZ to read for switch mode
    restrain_k: f64,        // harmonic restraint spring constant for h1 (0=none)
    mixer: String,          // "diis" or "broyden"
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| -> String {
        let idx = args.iter().position(|a| a == k)
            .unwrap_or_else(|| panic!("missing --{k}"));
        args[idx + 1].clone()
    };
    let get_opt = |k: &str| -> Option<String> {
        args.iter().position(|a| a == k).map(|idx| args[idx + 1].clone())
    };

    Args {
        xyz: get("--xyz"),
        mode: get("--mode"),
        scan: get_opt("--scan").unwrap_or_else(|| "1d".to_string()),
        n: get_opt("--n").map(|v| v.parse().unwrap()).unwrap_or(21),
        n2: get_opt("--n2").map(|v| v.parse().unwrap()).unwrap_or(1),
        h1: get_opt("--h1").map(|v| v.parse().unwrap()).unwrap_or(0),
        donor1: get_opt("--donor1").map(|v| v.parse().unwrap()).unwrap_or(0),
        acceptor1: get_opt("--acceptor1").map(|v| v.parse().unwrap()).unwrap_or(0),
        h2: get_opt("--h2").map(|v| v.parse().unwrap()).unwrap_or(0),
        donor2: get_opt("--donor2").map(|v| v.parse().unwrap()).unwrap_or(0),
        acceptor2: get_opt("--acceptor2").map(|v| v.parse().unwrap()).unwrap_or(0),
        out: get("--out"),
        data_dir: get("--data-dir"),
        max_iter: get_opt("--max-iter").map(|v| v.parse().unwrap()).unwrap_or(500),
        tol: get_opt("--tol").map(|v| v.parse().unwrap()).unwrap_or(1e-10),
        opt_max_iter: get_opt("--opt-max-iter").map(|v| v.parse().unwrap()).unwrap_or(500),
        opt_f_tol: get_opt("--opt-f-tol").map(|v| v.parse().unwrap()).unwrap_or(1e-4),
        opt_frozen: get_opt("--opt-frozen").unwrap_or_default(),
        switch_from: get_opt("--switch-from").unwrap_or_default(),
        restrain_k: get_opt("--restrain-k").map(|v| v.parse().unwrap()).unwrap_or(0.0),
        mixer: get_opt("--mixer").unwrap_or_else(|| "diis".to_string()),
    }
}

// ── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    let sk_dir = std::env::var("RUST_DFTB_SK_DIR").unwrap_or_else(|_| {
        "/home/prokophapala/SIMULATIONS/dftbplus/slakos/mio/mio-1-1".to_string()
    });

    // Parse geometry
    let mol = parse_xyz(&args.xyz)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", args.xyz));
    let species: Vec<String> = mol.species;
    let base_coords: Vec<[f64; 3]> = mol.coords;

    println!("System: {} atoms, {} species", species.len(), species.len());
    println!("Mode: {}, Scan: {}", args.mode, args.scan);
    println!("H1: atom {} moves donor({}) → acceptor({})", args.h1, args.donor1, args.acceptor1);
    println!("H2: atom {} moves donor({}) → acceptor({})", args.h2, args.donor2, args.acceptor2);

    // Load SK data
    let sk = load_sk_for_species(&sk_dir, &species)
        .unwrap_or_else(|e| panic!("Failed to load SK: {e}"));
    let builder = HamiltonianBuilder::new(sk);

    // ── Optimize mode: relax geometry with DFTB forces ──
    if args.mode == "optimize" {
        let frozen: std::collections::HashSet<usize> = if args.opt_frozen.is_empty() {
            std::collections::HashSet::new()
        } else {
            args.opt_frozen.split(',').map(|s| s.trim().parse::<usize>().unwrap()).collect()
        };

        println!("\n=== Geometry optimization (FIRE, persistent DftbCpu) ===");
        let (opt_coords, opt_energy, n_iter) = optimize_geometry_persistent(
            &sk_dir, &species, &base_coords, &frozen, &[],
            args.opt_max_iter, args.opt_f_tol, args.max_iter, args.tol,
            Some(&PathBuf::from("../debug/hbond_switching/reactant_traj.xyz")),
            Some(&PathBuf::from("../debug/hbond_switching/reactant_hist.csv")),
            &args.mixer,
        ).unwrap_or_else(|e| panic!("Optimization failed: {e}"));

        let out_xyz = PathBuf::from(&args.out);
        save_xyz(&out_xyz, &species, &opt_coords,
            &format!("optimized: E={:.10e} Hartree, {} FIRE iters", opt_energy, n_iter));
        println!("\nSaved optimized geometry to {}", out_xyz.display());
        println!("Final energy: {:.10e} Hartree", opt_energy);
        return;
    }

    // ── Switch mode: take optimized reactant, switch H atoms, optimize product ──
    if args.mode == "switch" {
        // Load the optimized reactant geometry
        let reactant_path = if args.switch_from.is_empty() {
            args.xyz.clone()
        } else {
            args.switch_from.clone()
        };
        let mol_r = parse_xyz(&reactant_path)
            .unwrap_or_else(|e| panic!("Failed to parse reactant {}: {e}", reactant_path));
        let r_coords: Vec<[f64; 3]> = mol_r.coords;

        println!("\n=== Proton switch (reactant → product) ===");
        println!("Reactant geometry: {}", reactant_path);

        // Switch H1: move from donor1 to acceptor1 (t=1.0)
        // Switch H2: move from donor2 to acceptor2 (t=1.0)
        let switched = make_geom_1d(
            &r_coords, args.h1, args.donor1, args.acceptor1,
            args.h2, args.donor2, args.acceptor2, 1.0,
        );

        // Save the switched (unoptimized) product guess
        let switch_xyz = PathBuf::from(&args.data_dir).join("switched_guess.xyz");
        save_xyz(&switch_xyz, &species, &switched, "switched H atoms (before optimization)");

        // Optimize the product
        let frozen: std::collections::HashSet<usize> = if args.opt_frozen.is_empty() {
            std::collections::HashSet::new()
        } else {
            args.opt_frozen.split(',').map(|s| s.trim().parse::<usize>().unwrap()).collect()
        };

        // Harmonic restraint on h1 to keep it near acceptor1 (prevents it from jumping back)
        // The restraint pulls h1 toward its switched position with spring constant k
        let restrain_k = args.restrain_k;
        let restraints: Vec<(usize, [f64; 3], f64)> = if restrain_k > 0.0 {
            vec![(args.h1, switched[args.h1], restrain_k)]
        } else {
            vec![]
        };

        println!("\n=== Product optimization (FIRE) ===");
        let (prod_coords, prod_energy, n_iter) = optimize_geometry(
            &builder, &species, &switched, &frozen, &restraints,
            args.opt_max_iter, args.opt_f_tol, args.max_iter, args.tol,
            Some(&PathBuf::from("../debug/hbond_switching/product_traj.xyz")),
            Some(&PathBuf::from("../debug/hbond_switching/product_hist.csv")),
        ).unwrap_or_else(|e| panic!("Product optimization failed: {e}"));

        let out_xyz = PathBuf::from(&args.out);
        save_xyz(&out_xyz, &species, &prod_coords,
            &format!("product optimized: E={:.10e} Hartree, {} FIRE iters", prod_energy, n_iter));

        // Also compute reactant energy for comparison
        let scc_r = builder.build_scc(&species, &r_coords, args.max_iter, args.tol)
            .unwrap_or_else(|e| panic!("Reactant SCC failed: {e}"));
        let e_rep_r = repulsive_energy(&sk_dir, &species, &r_coords);
        let e_total_r = scc_r.energy + e_rep_r;

        let e_rep_p = repulsive_energy(&sk_dir, &species, &prod_coords);
        let e_total_p = prod_energy + e_rep_p;

        println!("\n=== Results ===");
        println!("Reactant:  E_elec={:.10e}  E_rep={:.10e}  E_total={:.10e}", scc_r.energy, e_rep_r, e_total_r);
        println!("Product:   E_elec={:.10e}  E_rep={:.10e}  E_total={:.10e}", prod_energy, e_rep_p, e_total_p);
        println!("ΔE = {:.6e} Hartree = {:.2} kcal/mol = {:.4} eV",
            e_total_p - e_total_r,
            (e_total_p - e_total_r) * 627.509,
            (e_total_p - e_total_r) * 27.2114);
        println!("\nSaved product geometry to {}", out_xyz.display());
        return;
    }

    // ── Scan mode (default) ──
    // Create output dirs
    let data_dir = PathBuf::from(&args.data_dir);
    std::fs::create_dir_all(&data_dir).unwrap();

    // Open CSV
    let mut csv = File::create(&args.out).unwrap();
    if args.scan == "1d" {
        writeln!(csv, "idx,t,e_elec,e_rep,e_total,n_iter").unwrap();
    } else {
        writeln!(csv, "idx,t1,t2,e_elec,e_rep,e_total,n_iter").unwrap();
    }

    // Generate scan points
    let (n_total, is_2d) = if args.scan == "1d" {
        (args.n, false)
    } else {
        (args.n * args.n2, true)
    };

    let t0 = std::time::Instant::now();
    for idx in 0..n_total {
        let (t1, t2) = if is_2d {
            let i1 = idx / args.n2;
            let i2 = idx % args.n2;
            (i1 as f64 / (args.n - 1).max(1) as f64,
             i2 as f64 / (args.n2 - 1).max(1) as f64)
        } else {
            (idx as f64 / (args.n - 1).max(1) as f64, 0.0)
        };

        let coords = if is_2d {
            make_geom_2d(&base_coords, args.h1, args.donor1, args.acceptor1,
                         args.h2, args.donor2, args.acceptor2, t1, t2)
        } else {
            make_geom_1d(&base_coords, args.h1, args.donor1, args.acceptor1,
                         args.h2, args.donor2, args.acceptor2, t1)
        };

        // Compute SCC (always — non-SCC data is a subset: H0 and S are
        // saved in the SccResult, so non-SCC parity can be checked from
        // the SCC cache without a separate run).
        let scc = match builder.build_scc(&species, &coords, args.max_iter, args.tol) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  WARNING: SCC failed at pt {} (t1={:.3} t2={:.3}): {} — skipping",
                    idx, t1, t2, e);
                // Write a NaN line to CSV so the grid stays regular
                if is_2d {
                    writeln!(csv, "{},{:.6},{:.6},nan,nan,nan,0", idx, t1, t2).unwrap();
                } else {
                    writeln!(csv, "{},{:.6},nan,nan,nan,0", idx, t1).unwrap();
                }
                csv.flush().unwrap();
                continue;
            }
        };

        let e_rep = repulsive_energy(&sk_dir, &species, &coords);
        let e_total = scc.energy + e_rep;

        // Save per-point data
        save_point_data(&data_dir, idx, t1, t2, &species, &coords, &scc, e_rep, e_total)
            .unwrap_or_else(|e| panic!("Failed to save pt {}: {e}", idx));

        // Write CSV line
        if is_2d {
            writeln!(csv, "{},{:.6},{:.6},{:.16e},{:.16e},{:.16e},{}",
                idx, t1, t2, scc.energy, e_rep, e_total, scc.n_iter).unwrap();
        } else {
            writeln!(csv, "{},{:.6},{:.16e},{:.16e},{:.16e},{}",
                idx, t1, scc.energy, e_rep, e_total, scc.n_iter).unwrap();
        }

        if idx % 5 == 0 || idx == n_total - 1 {
            println!("  pt {}/{}: t1={:.3} t2={:.3} E_total={:.10e} n_iter={}",
                idx + 1, n_total, t1, t2, e_total, scc.n_iter);
        }
    }

    let elapsed = t0.elapsed();
    println!("\nDone: {} points in {:.1}s ({:.1}s/point)",
        n_total, elapsed.as_secs_f64(), elapsed.as_secs_f64() / n_total as f64);
    println!("CSV: {}", args.out);
    println!("Data: {}", args.data_dir);
}
