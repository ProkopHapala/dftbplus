//! Rigid coordinate scan driver (CPU SCC backend).
//!
//! Sweeps one internal coordinate (bond length or bond angle) over a range,
//! runs a converged SCC DFTB calculation at each point, and writes the energy
//! curve to a CSV file. Full per-replica data (H0, H_scc, S, density,
//! eigenvalues, charges, energy) is saved to disk so the coordinator can later
//! reuse it as input for the GPU batched-SCC parity test.
//!
//! The CPU backend call is isolated in `eval_scc` — swapping to
//! `gpu_solve_scc_batched` later is a one-function replacement.
//!
//! Usage:
//!   cargo run --example scan -- --xyz data/xyz/h2.xyz --bond 0 1 \
//!       --from 0.5 --to 3.0 --n 20 [--out scan.csv] [--data-dir scan_data] \
//!       [--max-iter 1000] [--tol 1e-10]
//!   cargo run --example scan -- --xyz h2o.xyz --angle 0 1 2 \
//!       --from 80 --to 120 --n 20

use rust_dftb::methods::dftb::forces::{parse_repulsive_spline, RepulsiveSpline};
use rust_dftb::{
    load_sk_for_species, parse_xyz, DftbOutput, HamiltonianBuilder, SccResult,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

const DEFAULT_SK_DIR: &str =
    "/home/prokophapala/git_SW/dftbplus/external/slakos/origin/mio-1-1";

const ANG2BOHR: f64 = 1.889_726_133;
const MIN_NEIGH_DIST: f64 = 1.0e-2;

/// Sum of DFTB repulsive pair energies for a geometry.
///
/// `SccResult.energy` contains only the electronic (band structure + SCC
/// double-counting) energy. The total DFTB energy also includes the repulsive
/// pair potential, which is parsed from the `Spline` section of the `.skf`
/// files. This helper adds that contribution so the scan curve has a physical
/// equilibrium minimum.
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

/// A single scan point: index, swept coordinate value, geometry, SCC result.
struct ScanPoint {
    idx: usize,
    coord_value: f64,
    species: Vec<String>,
    coords: Vec<[f64; 3]>,
    scc: SccResult,
}

/// ── CPU SCC backend call site (isolated for later GPU swap) ───────────────
///
/// Run a converged SCC DFTB calculation for one geometry. The coordinator can
/// replace the body of this function with a call to
/// `gpu_solve_scc_batched(...)` without touching the rest of the driver.
fn eval_scc(
    builder: &HamiltonianBuilder,
    species: &[String],
    coords: &[[f64; 3]],
    max_iter: usize,
    tol: f64,
) -> Result<SccResult, String> {
    builder
        .build_scc(species, coords, max_iter, tol)
        .map_err(|e| format!("SCC failed: {e}"))
}

// ── Geometry generation ───────────────────────────────────────────────────

/// Set the bond length between atoms `i` and `j` to `r` Å by moving atom `j`
/// along the i→j direction (atom `i` is fixed).
fn set_bond_length(coords: &mut [[f64; 3]], i: usize, j: usize, r: f64) {
    let dx = coords[j][0] - coords[i][0];
    let dy = coords[j][1] - coords[i][1];
    let dz = coords[j][2] - coords[i][2];
    let cur = (dx * dx + dy * dy + dz * dz).sqrt();
    assert!(cur > 1e-12, "atoms {i} and {j} are coincident");
    let scale = r / cur;
    coords[j][0] = coords[i][0] + dx * scale;
    coords[j][1] = coords[i][1] + dy * scale;
    coords[j][2] = coords[i][2] + dz * scale;
}

/// Set the bond angle i–j–k (vertex at j) to `theta_deg` by rotating atom `k`
/// in the plane defined by i, j, k. Atoms i and j are fixed.
fn set_bond_angle(coords: &mut [[f64; 3]], i: usize, j: usize, k: usize, theta_deg: f64) {
    let ji = [
        coords[i][0] - coords[j][0],
        coords[i][1] - coords[j][1],
        coords[i][2] - coords[j][2],
    ];
    let jk = [
        coords[k][0] - coords[j][0],
        coords[k][1] - coords[j][1],
        coords[k][2] - coords[j][2],
    ];
    let ri = (ji[0] * ji[0] + ji[1] * ji[1] + ji[2] * ji[2]).sqrt();
    let rk = (jk[0] * jk[0] + jk[1] * jk[1] + jk[2] * jk[2]).sqrt();
    assert!(ri > 1e-12 && rk > 1e-12, "degenerate angle atoms");
    // Normal to the i-j-k plane.
    let nx = ji[1] * jk[2] - ji[2] * jk[1];
    let ny = ji[2] * jk[0] - ji[0] * jk[2];
    let nz = ji[0] * jk[1] - ji[1] * jk[0];
    let nlen = (nx * nx + ny * ny + nz * nz).sqrt();
    // If collinear, pick an arbitrary perpendicular axis.
    let n_axis: [f64; 3] = if nlen < 1e-12 {
        // Rotate k around z-axis relative to the ji direction.
        let axis = if ji[0].abs() < 1e-12 && ji[1].abs() < 1e-12 {
            [1.0_f64, 0.0, 0.0]
        } else {
            [-ji[1], ji[0], 0.0]
        };
        let al = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        [axis[0] / al, axis[1] / al, axis[2] / al]
    } else {
        [nx / nlen, ny / nlen, nz / nlen]
    };
    let [nx, ny, nz] = n_axis;
    // Rotate jk by (theta_new - theta_old) around axis (nx,ny,nz) at vertex j.
    let cos_old = (ji[0] * jk[0] + ji[1] * jk[1] + ji[2] * jk[2]) / (ri * rk);
    let old_deg = cos_old.clamp(-1.0, 1.0).acos().to_degrees();
    let delta = (theta_deg - old_deg).to_radians();
    let (cd, sd) = (delta.cos(), delta.sin());
    // Rodrigues rotation of vector jk around unit axis n.
    let dot = nx * jk[0] + ny * jk[1] + nz * jk[2];
    let cross = [
        ny * jk[2] - nz * jk[1],
        nz * jk[0] - nx * jk[2],
        nx * jk[1] - ny * jk[0],
    ];
    let rot = [
        jk[0] * cd + cross[0] * sd + nx * dot * (1.0 - cd),
        jk[1] * cd + cross[1] * sd + ny * dot * (1.0 - cd),
        jk[2] * cd + cross[2] * sd + nz * dot * (1.0 - cd),
    ];
    coords[k][0] = coords[j][0] + rot[0];
    coords[k][1] = coords[j][1] + rot[1];
    coords[k][2] = coords[j][2] + rot[2];
}

// ── Per-replica data saving ───────────────────────────────────────────────

/// Save full per-replica data for one scan point to `<data_dir>/rep_XX/`.
fn save_replica_data(dir: &Path, pt: &ScanPoint, e_rep: f64, e_total: f64) -> std::io::Result<()> {
    let rep_dir = dir.join(format!("rep_{:02}", pt.idx));
    std::fs::create_dir_all(&rep_dir)?;

    // geometry.xyz
    let mut f = File::create(rep_dir.join("geometry.xyz"))?;
    writeln!(f, "{}", pt.species.len())?;
    writeln!(f, "scan point {} coord={:.6}", pt.idx, pt.coord_value)?;
    for (sp, c) in pt.species.iter().zip(pt.coords.iter()) {
        writeln!(f, "{sp} {:.10} {:.10} {:.10}", c[0], c[1], c[2])?;
    }

    // Matrices in DFTB+ square format.
    DftbOutput::write_square(
        rep_dir.join("h0.dat").to_str().unwrap(),
        &pt.scc.h0,
    )
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    DftbOutput::write_square(
        rep_dir.join("h_scc.dat").to_str().unwrap(),
        &pt.scc.h_scc,
    )
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    DftbOutput::write_square(
        rep_dir.join("s.dat").to_str().unwrap(),
        &pt.scc.s,
    )
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    DftbOutput::write_square(
        rep_dir.join("density.dat").to_str().unwrap(),
        &pt.scc.density,
    )
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    // eigenvalues.txt
    let mut f = File::create(rep_dir.join("eigenvalues.txt"))?;
    writeln!(f, "# orbital eigenvalues (Hartree)")?;
    for (k, e) in pt.scc.eigenvalues.iter().enumerate() {
        writeln!(f, "{k} {:.16e}", e)?;
    }

    // charges.txt
    let mut f = File::create(rep_dir.join("charges.txt"))?;
    writeln!(f, "# atom Mulliken_charge reference_q0")?;
    for (a, (q, q0)) in pt.scc.charges.iter().zip(pt.scc.q0.iter()).enumerate() {
        writeln!(f, "{a} {:.16e} {:.16e}", q, q0)?;
    }

    // energy.txt
    let mut f = File::create(rep_dir.join("energy.txt"))?;
    writeln!(f, "# electronic_scc_energy repulsive_energy total_energy n_iter")?;
    writeln!(f, "{:.16e} {:.16e} {:.16e} {}", pt.scc.energy, e_rep, e_total, pt.scc.n_iter)?;

    Ok(())
}

// ── CLI parsing ───────────────────────────────────────────────────────────

struct ScanArgs {
    xyz: String,
    mode: ScanMode,
    out: String,
    data_dir: String,
    max_iter: usize,
    tol: f64,
}

enum ScanMode {
    Bond { i: usize, j: usize, from: f64, to: f64, n: usize },
    Angle { i: usize, j: usize, k: usize, from: f64, to: f64, n: usize },
}

fn parse_usize(v: &str, flag: &str) -> usize {
    v.parse::<usize>()
        .unwrap_or_else(|_| panic!("invalid integer for {flag}: {v}"))
}

fn parse_f64(v: &str, flag: &str) -> f64 {
    v.parse::<f64>()
        .unwrap_or_else(|_| panic!("invalid float for {flag}: {v}"))
}

fn parse_args() -> ScanArgs {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut xyz: Option<String> = None;
    let mut out: Option<String> = None;
    let mut data_dir: Option<String> = None;
    let mut max_iter: Option<usize> = None;
    let mut tol: Option<f64> = None;
    let mut bond: Option<(usize, usize)> = None;
    let mut angle: Option<(usize, usize, usize)> = None;
    let mut from: Option<f64> = None;
    let mut to: Option<f64> = None;
    let mut n: Option<usize> = None;

    let mut k = 0;
    while k < args.len() {
        let a = args[k].as_str();
        match a {
            "--xyz" => { xyz = Some(args[k + 1].clone()); k += 2; }
            "--out" => { out = Some(args[k + 1].clone()); k += 2; }
            "--data-dir" => { data_dir = Some(args[k + 1].clone()); k += 2; }
            "--max-iter" => { max_iter = Some(parse_usize(&args[k + 1], "--max-iter")); k += 2; }
            "--tol" => { tol = Some(parse_f64(&args[k + 1], "--tol")); k += 2; }
            "--n" => { n = Some(parse_usize(&args[k + 1], "--n")); k += 2; }
            "--from" => { from = Some(parse_f64(&args[k + 1], "--from")); k += 2; }
            "--to" => { to = Some(parse_f64(&args[k + 1], "--to")); k += 2; }
            "--bond" => {
                let i = parse_usize(&args[k + 1], "--bond i");
                let j = parse_usize(&args[k + 2], "--bond j");
                bond = Some((i, j));
                k += 3;
            }
            "--angle" => {
                let i = parse_usize(&args[k + 1], "--angle i");
                let j = parse_usize(&args[k + 2], "--angle j");
                let kk = parse_usize(&args[k + 3], "--angle k");
                angle = Some((i, j, kk));
                k += 4;
            }
            other => { eprintln!("warning: ignoring unknown arg {other}"); k += 1; }
        }
    }

    let xyz = xyz.expect("--xyz <path> is required");
    let n = n.unwrap_or(20);
    let from = from.expect("--from <value> is required");
    let to = to.expect("--to <value> is required");
    let mode = if let Some((i, j)) = bond {
        ScanMode::Bond { i, j, from, to, n }
    } else if let Some((i, j, kk)) = angle {
        ScanMode::Angle { i, j, k: kk, from, to, n }
    } else {
        panic!("either --bond <i> <j> or --angle <i> <j> <k> is required (with --from/--to/--n)");
    };

    ScanArgs {
        xyz,
        mode,
        out: out.unwrap_or_else(|| "scan.csv".to_string()),
        data_dir: data_dir.unwrap_or_else(|| "scan_data".to_string()),
        max_iter: max_iter.unwrap_or(1000),
        tol: tol.unwrap_or(1e-10),
    }
}

// ── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    let sk_dir = std::env::var("RUST_DFTB_SK_DIR").unwrap_or_else(|_| {
        eprintln!("[scan] RUST_DFTB_SK_DIR not set, using default {DEFAULT_SK_DIR}");
        DEFAULT_SK_DIR.to_string()
    });

    let mol = parse_xyz(&args.xyz).unwrap_or_else(|e| panic!("failed to parse XYZ {}: {e}", args.xyz));
    eprintln!("[scan] loaded {} atoms from {}", mol.species.len(), args.xyz);

    let sk = load_sk_for_species(&sk_dir, &mol.species)
        .unwrap_or_else(|e| panic!("failed to load SK from {sk_dir}: {e}"));
    let builder = HamiltonianBuilder::new(sk);

    let data_dir = Path::new(&args.data_dir);
    std::fs::create_dir_all(data_dir)
        .unwrap_or_else(|e| panic!("cannot create data dir {}: {e}", args.data_dir));

    let (n_points, coord_label, unit) = match &args.mode {
        ScanMode::Bond { n, .. } => (*n, "bond_length", "Angstrom"),
        ScanMode::Angle { n, .. } => (*n, "bond_angle", "degree"),
    };

    eprintln!(
        "[scan] sweeping {coord_label} over {n_points} points; saving per-replica data to {}",
        data_dir.display()
    );

    let mut curve: Vec<(usize, f64, f64, usize)> = Vec::with_capacity(n_points); // idx, coord, energy, n_iter

    for p in 0..n_points {
        let t = if n_points > 1 { p as f64 / (n_points - 1) as f64 } else { 0.0 };
        let mut coords = mol.coords.clone();
        let coord_value = match &args.mode {
            ScanMode::Bond { i, j, from, to, .. } => {
                let r = from + (to - from) * t;
                set_bond_length(&mut coords, *i, *j, r);
                r
            }
            ScanMode::Angle { i, j, k, from, to, .. } => {
                let th = from + (to - from) * t;
                set_bond_angle(&mut coords, *i, *j, *k, th);
                th
            }
        };

        let scc = eval_scc(&builder, &mol.species, &coords, args.max_iter, args.tol)
            .unwrap_or_else(|e| panic!("SCC failed at point {p} (coord={coord_value:.6}): {e}"));

        let e_rep = repulsive_energy(&sk_dir, &mol.species, &coords);
        let e_elec = scc.energy;
        let energy = e_elec + e_rep;
        let n_iter = scc.n_iter;
        eprintln!(
            "[scan] point {p:>3}/{n_points}  {coord_label}={coord_value:.6} {unit}  E_elec={e_elec:.10}  E_rep={e_rep:.10}  E_total={energy:.10}  n_iter={n_iter}"
        );

        let pt = ScanPoint {
            idx: p,
            coord_value,
            species: mol.species.clone(),
            coords,
            scc,
        };
        if let Err(e) = save_replica_data(data_dir, &pt, e_rep, energy) {
            eprintln!("[scan] WARNING: failed to save replica data for point {p}: {e}");
        }

        curve.push((p, coord_value, energy, n_iter));
    }

    // Write energy curve CSV.
    let mut f = File::create(&args.out)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", args.out));
    writeln!(f, "# scan energy curve").unwrap();
    writeln!(f, "# coord_label={coord_label} unit={unit} n_points={n_points}").unwrap();
    writeln!(f, "index,coord,energy_hartree,n_iter").unwrap();
    for (p, cv, e, ni) in &curve {
        writeln!(f, "{p},{cv:.10},{e:.16e},{ni}").unwrap();
    }
    eprintln!("[scan] wrote energy curve to {} ({} points)", args.out, curve.len());

    // Report minimum.
    let (min_p, min_cv, min_e, _) = curve
        .iter()
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();
    eprintln!(
        "[scan] minimum energy {min_e:.10} Hartree at point {min_p} ({coord_label}={min_cv:.6} {unit})"
    );
}
