//! Nudged elastic band (NEB) driver (CPU SCC backend).
//!
//! Interpolates a band of images between two endpoint geometries and optimizes
//! it with the nudged elastic band method using real DFTB SCC forces
//! (`compute_scc_forces`). The CPU backend call is isolated in
//! `eval_scc_with_forces` so the coordinator can later swap to a GPU batched
//! SCC + force kernel.
//!
//! Usage:
//!   cargo run --example neb -- --start reactant.xyz --end product.xyz \
//!       --images 20 --k 0.1 --maxiter 100 [--step 0.1] [--tol 1e-3] \
//!       [--out neb_band.csv] [--data-dir neb_data]
//!
//! Output: converged band (image index, position along path, energy) to CSV,
//! plus per-image geometry + SCC data under `<data-dir>/img_XX/`.

use rust_dftb::methods::dftb::forces::compute_scc_forces;
use rust_dftb::{
    load_sk_for_species, parse_xyz, DftbOutput, HamiltonianBuilder, SccResult,
};
use std::fs::File;
use std::io::Write;
use std::path::Path;

const DEFAULT_SK_DIR: &str =
    "/home/prokophapala/git_SW/dftbplus/external/slakos/origin/mio-1-1";

/// Per-image SCC + force evaluation result.
struct ImageEval {
    energy: f64,
    forces: Vec<[f64; 3]>, // Hartree/Å (negative gradient = force)
    scc: SccResult,
}

/// ── CPU SCC + forces backend call site (isolated for later GPU swap) ──────
///
/// Evaluate SCC energy and forces for one image. The coordinator can replace
/// the body with a batched GPU SCC + force kernel call.
fn eval_scc_with_forces(
    builder: &HamiltonianBuilder,
    species: &[String],
    coords: &[[f64; 3]],
    max_iter: usize,
    tol: f64,
) -> Result<ImageEval, String> {
    let scc = builder
        .build_scc(species, coords, max_iter, tol)
        .map_err(|e| format!("SCC failed: {e}"))?;
    let f = compute_scc_forces(builder, species, coords, &scc)
        .map_err(|e| format!("forces failed: {e}"))?;
    Ok(ImageEval {
        energy: scc.energy,
        forces: f.forces,
        scc,
    })
}

// ── Geometry interpolation ────────────────────────────────────────────────

/// Linear interpolation between two equal-length coordinate sets.
/// `t=0` → start, `t=1` → end.
fn interpolate(start: &[[f64; 3]], end: &[[f64; 3]], t: f64) -> Vec<[f64; 3]> {
    assert_eq!(start.len(), end.len());
    start
        .iter()
        .zip(end.iter())
        .map(|(a, b)| {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ]
        })
        .collect()
}

fn rmsd(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    assert_eq!(a.len(), b.len());
    let s: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        })
        .sum();
    (s / a.len() as f64).sqrt()
}

// ── NEB forces ────────────────────────────────────────────────────────────

/// Unit tangent at image `i` from finite-difference of image positions.
/// Endpoints use one-sided differences; interior images use central.
fn tangent(images: &[Vec<[f64; 3]>], i: usize) -> Vec<[f64; 3]> {
    let n = images.len();
    let n_at = images[0].len();
    let mut tau = vec![[0.0_f64; 3]; n_at];
    if n == 1 {
        return tau;
    }
    let (t1, t2) = if i == 0 {
        (images[1].clone(), images[0].clone())
    } else if i == n - 1 {
        (images[i].clone(), images[i - 1].clone())
    } else {
        (images[i + 1].clone(), images[i - 1].clone())
    };
    for a in 0..n_at {
        tau[a] = [
            t1[a][0] - t2[a][0],
            t1[a][1] - t2[a][1],
            t1[a][2] - t2[a][2],
        ];
    }
    // Normalize.
    let norm: f64 = tau
        .iter()
        .map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        .sum::<f64>()
        .sqrt();
    if norm > 1e-12 {
        for a in 0..n_at {
            tau[a][0] /= norm;
            tau[a][1] /= norm;
            tau[a][2] /= norm;
        }
    }
    tau
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute the NEB total force on each image (nudged elastic band).
/// Returns the force array per image. Endpoints are fixed (zero force).
fn neb_forces(
    images: &[Vec<[f64; 3]>],
    real_forces: &[Vec<[f64; 3]>],
    k_spring: f64,
) -> Vec<Vec<[f64; 3]>> {
    let n = images.len();
    let n_at = images[0].len();
    let mut out = vec![vec![[0.0_f64; 3]; n_at]; n];

    for i in 1..n.saturating_sub(1) {
        let tau = tangent(images, i);
        // True force perpendicular component: F_perp = F - (F·τ)τ
        let f_real = &real_forces[i];
        // Spring force along tangent: F_spring = k·(R_{i+1} - 2·R_i + R_{i-1})
        let mut f_spring = vec![[0.0_f64; 3]; n_at];
        for a in 0..n_at {
            f_spring[a] = [
                k_spring * (images[i + 1][a][0] - 2.0 * images[i][a][0] + images[i - 1][a][0]),
                k_spring * (images[i + 1][a][1] - 2.0 * images[i][a][1] + images[i - 1][a][1]),
                k_spring * (images[i + 1][a][2] - 2.0 * images[i][a][2] + images[i - 1][a][2]),
            ];
        }
        for a in 0..n_at {
            let fpar_real = dot(&f_real[a], &tau[a]);
            let f_perp = [
                f_real[a][0] - fpar_real * tau[a][0],
                f_real[a][1] - fpar_real * tau[a][1],
                f_real[a][2] - fpar_real * tau[a][2],
            ];
            let fpar_spring = dot(&f_spring[a], &tau[a]);
            let f_parallel = [
                fpar_spring * tau[a][0],
                fpar_spring * tau[a][1],
                fpar_spring * tau[a][2],
            ];
            out[i][a] = [
                f_perp[0] + f_parallel[0],
                f_perp[1] + f_parallel[1],
                f_perp[2] + f_parallel[2],
            ];
        }
    }
    out
}

fn max_force_norm(forces: &[Vec<[f64; 3]>]) -> f64 {
    forces
        .iter()
        .flat_map(|img| img.iter())
        .map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt())
        .fold(0.0_f64, f64::max)
}

// ── Per-image data saving ─────────────────────────────────────────────────

fn save_image_data(dir: &Path, idx: usize, species: &[String], coords: &[[f64; 3]], ev: &ImageEval) -> std::io::Result<()> {
    let img_dir = dir.join(format!("img_{idx:02}"));
    std::fs::create_dir_all(&img_dir)?;

    let mut f = File::create(img_dir.join("geometry.xyz"))?;
    writeln!(f, "{}", species.len())?;
    writeln!(f, "NEB image {idx} energy={:.10}", ev.energy)?;
    for (sp, c) in species.iter().zip(coords.iter()) {
        writeln!(f, "{sp} {:.10} {:.10} {:.10}", c[0], c[1], c[2])?;
    }

    DftbOutput::write_square(img_dir.join("h0.dat").to_str().unwrap(), &ev.scc.h0)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    DftbOutput::write_square(img_dir.join("h_scc.dat").to_str().unwrap(), &ev.scc.h_scc)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    DftbOutput::write_square(img_dir.join("s.dat").to_str().unwrap(), &ev.scc.s)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    DftbOutput::write_square(img_dir.join("density.dat").to_str().unwrap(), &ev.scc.density)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    let mut f = File::create(img_dir.join("eigenvalues.txt"))?;
    writeln!(f, "# orbital eigenvalues (Hartree)")?;
    for (k, e) in ev.scc.eigenvalues.iter().enumerate() {
        writeln!(f, "{k} {:.16e}", e)?;
    }
    let mut f = File::create(img_dir.join("charges.txt"))?;
    writeln!(f, "# atom Mulliken_charge reference_q0")?;
    for (a, (q, q0)) in ev.scc.charges.iter().zip(ev.scc.q0.iter()).enumerate() {
        writeln!(f, "{a} {:.16e} {:.16e}", q, q0)?;
    }
    let mut f = File::create(img_dir.join("energy.txt"))?;
    writeln!(f, "# total_scc_energy(Hartree) n_iter")?;
    writeln!(f, "{:.16e} {}", ev.scc.energy, ev.scc.n_iter)?;

    let mut f = File::create(img_dir.join("forces.txt"))?;
    writeln!(f, "# atom fx fy fz (Hartree/Ang)")?;
    for (a, fr) in ev.forces.iter().enumerate() {
        writeln!(f, "{a} {:.16e} {:.16e} {:.16e}", fr[0], fr[1], fr[2])?;
    }
    Ok(())
}

// ── CLI parsing ───────────────────────────────────────────────────────────

struct NebArgs {
    start: String,
    end: String,
    images: usize,
    k: f64,
    maxiter: usize,
    step: f64,
    tol: f64,
    out: String,
    data_dir: String,
    scc_max_iter: usize,
    scc_tol: f64,
}

fn parse_f64(v: &str, flag: &str) -> f64 {
    v.parse::<f64>().unwrap_or_else(|_| panic!("invalid float for {flag}: {v}"))
}
fn parse_usize(v: &str, flag: &str) -> usize {
    v.parse::<usize>().unwrap_or_else(|_| panic!("invalid integer for {flag}: {v}"))
}

fn parse_args() -> NebArgs {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut start = None;
    let mut end = None;
    let mut images = 20usize;
    let mut k = 0.1_f64;
    let mut maxiter = 100usize;
    let mut step = 0.1_f64;
    let mut tol = 1e-3_f64;
    let mut out = "neb_band.csv".to_string();
    let mut data_dir = "neb_data".to_string();
    let mut scc_max_iter = 1000usize;
    let mut scc_tol = 1e-10_f64;

    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--start" => { start = Some(args[i + 1].clone()); i += 2; }
            "--end" => { end = Some(args[i + 1].clone()); i += 2; }
            "--images" => { images = parse_usize(&args[i + 1], "--images"); i += 2; }
            "--k" => { k = parse_f64(&args[i + 1], "--k"); i += 2; }
            "--maxiter" => { maxiter = parse_usize(&args[i + 1], "--maxiter"); i += 2; }
            "--step" => { step = parse_f64(&args[i + 1], "--step"); i += 2; }
            "--tol" => { tol = parse_f64(&args[i + 1], "--tol"); i += 2; }
            "--out" => { out = args[i + 1].clone(); i += 2; }
            "--data-dir" => { data_dir = args[i + 1].clone(); i += 2; }
            "--scc-max-iter" => { scc_max_iter = parse_usize(&args[i + 1], "--scc-max-iter"); i += 2; }
            "--scc-tol" => { scc_tol = parse_f64(&args[i + 1], "--scc-tol"); i += 2; }
            other => { eprintln!("warning: ignoring unknown arg {other}"); i += 1; }
        }
    }
    NebArgs {
        start: start.expect("--start <xyz> is required"),
        end: end.expect("--end <xyz> is required"),
        images,
        k,
        maxiter,
        step,
        tol,
        out,
        data_dir,
        scc_max_iter,
        scc_tol,
    }
}

// ── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    let sk_dir = std::env::var("RUST_DFTB_SK_DIR").unwrap_or_else(|_| {
        eprintln!("[neb] RUST_DFTB_SK_DIR not set, using default {DEFAULT_SK_DIR}");
        DEFAULT_SK_DIR.to_string()
    });

    let mol_start = parse_xyz(&args.start).unwrap_or_else(|e| panic!("parse start {}: {e}", args.start));
    let mol_end = parse_xyz(&args.end).unwrap_or_else(|e| panic!("parse end {}: {e}", args.end));
    assert_eq!(
        mol_start.species, mol_end.species,
        "endpoint species must match"
    );
    assert_eq!(
        mol_start.coords.len(),
        mol_end.coords.len(),
        "endpoint atom counts must match"
    );
    let species = &mol_start.species;
    eprintln!(
        "[neb] {} atoms, {} images, k={}, maxiter={}, step={}, tol={}",
        species.len(),
        args.images,
        args.k,
        args.maxiter,
        args.step,
        args.tol
    );
    eprintln!("[neb] endpoint RMSD = {:.6} Å", rmsd(&mol_start.coords, &mol_end.coords));

    let sk = load_sk_for_species(&sk_dir, species)
        .unwrap_or_else(|e| panic!("failed to load SK from {sk_dir}: {e}"));
    let builder = HamiltonianBuilder::new(sk);

    let data_dir = Path::new(&args.data_dir);
    std::fs::create_dir_all(data_dir)
        .unwrap_or_else(|e| panic!("cannot create data dir {}: {e}", args.data_dir));

    // Initial band: linear interpolation (endpoints fixed).
    let mut images: Vec<Vec<[f64; 3]>> = Vec::with_capacity(args.images);
    for i in 0..args.images {
        let t = if args.images > 1 {
            i as f64 / (args.images - 1) as f64
        } else {
            0.0
        };
        images.push(interpolate(&mol_start.coords, &mol_end.coords, t));
    }

    let mut iter = 0;
    let mut max_f = f64::INFINITY;
    while iter < args.maxiter && max_f > args.tol {
        // Evaluate SCC + forces for every image.
        let mut energies = Vec::with_capacity(args.images);
        let mut real_forces = Vec::with_capacity(args.images);
        for (_i, geom) in images.iter().enumerate() {
            let ev = eval_scc_with_forces(
                &builder,
                species,
                geom,
                args.scc_max_iter,
                args.scc_tol,
            )
            .unwrap_or_else(|e| panic!("image eval failed at iter {iter}: {e}"));
            energies.push(ev.energy);
            real_forces.push(ev.forces);
        }

        let neb_f = neb_forces(&images, &real_forces, args.k);
        max_f = max_force_norm(&neb_f);

        // Simple gradient-descent update on interior images.
        for i in 1..args.images.saturating_sub(1) {
            for a in 0..images[i].len() {
                images[i][a][0] += args.step * neb_f[i][a][0];
                images[i][a][1] += args.step * neb_f[i][a][1];
                images[i][a][2] += args.step * neb_f[i][a][2];
            }
        }

        let e_min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let e_max = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "[neb] iter {iter:>3}  max|F|={max_f:.6e}  E_range=[{e_min:.8}, {e_max:.8}]"
        );
        iter += 1;
    }

    if max_f <= args.tol {
        eprintln!("[neb] converged in {iter} iterations (max|F|={max_f:.6e} <= tol={})", args.tol);
    } else {
        let maxiter = args.maxiter;
        eprintln!("[neb] reached maxiter={maxiter} (max|F|={max_f:.6e})");
    }

    // Final evaluation + save.
    let mut final_energies = Vec::with_capacity(args.images);
    for (i, geom) in images.iter().enumerate() {
        let ev = eval_scc_with_forces(&builder, species, geom, args.scc_max_iter, args.scc_tol)
            .unwrap_or_else(|e| panic!("final eval image {i} failed: {e}"));
        final_energies.push(ev.energy);
        if let Err(e) = save_image_data(data_dir, i, species, geom, &ev) {
            eprintln!("[neb] WARNING: failed to save final image {i} data: {e}");
        }
    }

    // Write band CSV.
    let mut f = File::create(&args.out)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", args.out));
    writeln!(f, "# NEB band").unwrap();
    writeln!(f, "# images={} k={} iterations={}", args.images, args.k, iter).unwrap();
    writeln!(f, "image,energy_hartree").unwrap();
    for (i, e) in final_energies.iter().enumerate() {
        writeln!(f, "{i},{e:.16e}").unwrap();
    }
    eprintln!("[neb] wrote band to {} ({} images)", args.out, args.images);
    let e_min = final_energies.iter().cloned().fold(f64::INFINITY, f64::min);
    let e_max = final_energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[neb] final energy range: [{e_min:.10}, {e_max:.10}] Hartree, barrier ~{:.10}", e_max - e_min);
}
