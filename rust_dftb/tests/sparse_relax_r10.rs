//! Short FIRE relaxation of the H-passivated Si R10 sphere (Si196H134, 330 atoms).
//!
//! Ignored by default: one GPU run, wall-capped inside the loop. Not a regression.
//!
//!   cargo test --test sparse_relax_r10 -- --ignored --nocapture --test-threads=1

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rust_dftb::load_sk_for_species;
use rust_dftb::methods::sparse::harness::require_sih_sk_dir;
use rust_dftb::methods::sparse::{GeomStep, SparseDftb, SparseDftbConfig};
use rust_dftb::parse_xyz;

const MAX_STEPS: usize = 400;
const WALL_SECS: f64 = 42.0;
/// Keep stepping until `|F|` is this small, or it stops setting new lows.
const F_FLOOR: f64 = 1e-5;
/// Steps with no new low in max|F| (2% better) before calling it a plateau.
const PLATEAU_PATIENCE: usize = 40;
const JITTER_ANG: f64 = 0.1;

fn si_cfg() -> SparseDftbConfig {
    SparseDftbConfig {
        r_trunc_ang: Some(5.45),
        taper_w_ang: 0.3,
        r_k_ang: Some(12.0),
        r_z_ang: Some(12.0),
        r_skin_ang: 3.0,
        max_deg_hs: Some(1024),
        max_deg_k: Some(1024),
        max_deg_z: Some(1024),
        dense_diag: Some(false),
        purifier_p: Some(false),
        purifier_trs: Some(false),
        ..Default::default()
    }
}

fn out_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../debug/sparse_relax")
}

fn write_frame(w: &mut BufWriter<File>, species: &[String], coords: &[[f64; 3]], comment: &str) {
    writeln!(w, "{}", species.len()).unwrap();
    writeln!(w, "{comment}").unwrap();
    for (s, c) in species.iter().zip(coords.iter()) {
        writeln!(w, "{s:2} {:14.8} {:14.8} {:14.8}", c[0], c[1], c[2]).unwrap();
    }
}

fn write_xyz(path: &std::path::Path, species: &[String], coords: &[[f64; 3]], comment: &str) {
    let f = File::create(path).unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    let mut w = BufWriter::new(f);
    write_frame(&mut w, species, coords, comment);
}

fn nearest_sih(species: &[String], coords: &[[f64; 3]]) -> (f64, f64, usize) {
    let mut sum = 0.0;
    let mut n = 0usize;
    let mut rmin = f64::MAX;
    for (i, s) in species.iter().enumerate() {
        if s != "H" {
            continue;
        }
        let mut best = f64::MAX;
        for (j, t) in species.iter().enumerate() {
            if t != "Si" {
                continue;
            }
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            best = best.min((dx * dx + dy * dy + dz * dz).sqrt());
        }
        sum += best;
        n += 1;
        rmin = rmin.min(best);
    }
    (sum / n.max(1) as f64, rmin, n)
}

/// Every atom moves `amp` Å along a deterministic random direction.
fn jitter_coords(coords: &[[f64; 3]], amp: f64, mut seed: u64) -> Vec<[f64; 3]> {
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64) / (u64::MAX as f64)
    };
    coords
        .iter()
        .map(|c| {
            let u = rnd() * 2.0 - 1.0;
            let v = rnd() * 2.0 - 1.0;
            let w = rnd() * 2.0 - 1.0;
            let n = (u * u + v * v + w * w).sqrt().max(1e-15);
            [c[0] + amp * u / n, c[1] + amp * v / n, c[2] + amp * w / n]
        })
        .collect()
}

fn max_disp(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(u, v)| {
            let dx = u[0] - v[0];
            let dy = u[1] - v[1];
            let dz = u[2] - v[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(0.0, f64::max)
}

#[test]
#[ignore = "R10 FIRE probe, GPU, wall-capped"]
fn relax_si_sphere_r10() {
    std::env::set_var("RUST_DFTB_SPARSE_ALGEBRA_VERBOSE", "0");
    // R10 at r_k=12 sits on the mask floor: cold SCC reaches r_scc~1e-5 and
    // Tr(KS)=Nocc with R_H~2e-3, above the 5e-4 default and below the
    // ~1e-2 wrong-subspace signature. Documented production override.
    std::env::set_var("RUST_DFTB_SCC_RHGATE", "1e-2");
    let dir = out_dir();
    std::fs::create_dir_all(&dir).unwrap();
    std::env::set_var("RUST_DFTB_PROF", "0");
    std::env::set_var("RUST_DFTB_PROF_ACCUM", "0");
    let sk_dir = require_sih_sk_dir();
    let xyz = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../debug/nanocrystals/si_sphere_R10.xyz");
    let mol = parse_xyz(xyz.to_str().unwrap()).unwrap_or_else(|e| panic!("parse: {e}"));
    let n_h = mol.species.iter().filter(|s| s.as_str() == "H").count();
    let n_si = mol.species.iter().filter(|s| s.as_str() == "Si").count();
    eprintln!("R10 sphere: {n_si} Si + {n_h} H = {} atoms", mol.species.len());
    let sk = load_sk_for_species(&sk_dir, &mol.species).unwrap_or_else(|e| panic!("SK: {e}"));
    let cfg = si_cfg();
    let t_new = Instant::now();
    let mut eng = SparseDftb::with_config(
        sk,
        &sk_dir,
        mol.species.clone(),
        mol.coords.clone(),
        cfg,
    )
    .unwrap_or_else(|e| panic!("SparseDftb::new: {e}"));
    eprintln!("init {:.1} s", t_new.elapsed().as_secs_f64());
    let jittered = jitter_coords(eng.coords(), JITTER_ANG, 0x5eed_u64);
    let jmax = max_disp(eng.coords(), &jittered);
    eng.set_coords(&jittered)
        .unwrap_or_else(|e| panic!("jitter set_coords: {e}"));
    eprintln!("jitter: every atom moved {JITTER_ANG} Å (max check {jmax:.4} Å)");
    eng.set_geom_step(GeomStep::BoldXtrDmm);
    eng.set_fire_dt(0.1);
    write_xyz(
        &dir.join("si_R10_before.xyz"),
        &mol.species,
        eng.coords(),
        &format!("Si sphere R10 jittered {JITTER_ANG} Å/atom, before FIRE"),
    );
    let (sih0, sih0_min, _) = nearest_sih(&mol.species, eng.coords());
    let coords0 = eng.coords().to_vec();

    let hist_path = dir.join("history.csv");
    let mut hist = BufWriter::new(File::create(&hist_path).unwrap());
    writeln!(
        hist,
        "step,E_Ha,maxAbsF,ms,fire_ms,scc_ms,force_ms,rms,TrKS,R_H,scc_iters,note"
    )
    .unwrap();

    let traj_path = dir.join("si_R10_traj.xyz");
    let mut traj = BufWriter::new(File::create(&traj_path).unwrap());
    let wall = Instant::now();
    let mut stopped = String::new();
    let mut sum_fire = 0.0f64;
    let mut sum_scc = 0.0f64;
    let mut sum_force = 0.0f64;
    let mut n_done = 0usize;
    let mut best_f = f64::MAX;
    let mut best_step = 0usize;
    let mut since_best = 0usize;
    for step in 0..=MAX_STEPS {
        if wall.elapsed().as_secs_f64() > WALL_SECS {
            stopped = format!("wall cap {WALL_SECS} s at step {step}");
            eprintln!("{stopped}");
            break;
        }
        let t = Instant::now();
        let t_fire = Instant::now();
        if step > 0 {
            if let Err(e) = eng.fire_step(F_FLOOR) {
                stopped = format!("FIRE step {step}: {e}");
                eprintln!("{stopped}");
                break;
            }
        }
        let fire_ms = t_fire.elapsed().as_secs_f64() * 1e3;
        let t_scc = Instant::now();
        let scc = match eng.scc(80, 1e-5) {
            Ok(s) => s,
            Err(e) => {
                stopped = format!("SCC step {step}: {e}");
                eprintln!("{stopped}");
                break;
            }
        };
        let scc_ms = t_scc.elapsed().as_secs_f64() * 1e3;
        let e = eng.energy().unwrap_or_else(|err| panic!("energy step {step}: {err}"));
        let t_f = Instant::now();
        let f = eng.forces().unwrap_or_else(|err| panic!("forces step {step}: {err}"));
        let force_ms = t_f.elapsed().as_secs_f64() * 1e3;
        let mut max_f = 0.0f64;
        for fi in &f.forces {
            for c in fi {
                max_f = max_f.max(c.abs());
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        sum_fire += fire_ms;
        sum_scc += scc_ms;
        sum_force += force_ms;
        n_done += 1;
        writeln!(
            hist,
            "{step},{e:.8},{max_f:.6e},{ms:.1},{fire_ms:.1},{scc_ms:.1},{force_ms:.1},{:.6e},{:.6},{:.6e},{},",
            scc.rms, scc.tr_ks, scc.r_h, scc.n_iters
        )
        .unwrap();
        hist.flush().unwrap();
        write_frame(
            &mut traj,
            &mol.species,
            eng.coords(),
            &format!("step={step} E={e:.8} max|F|={max_f:.6e}"),
        );
        traj.flush().unwrap();
        if max_f < best_f * 0.98 {
            best_f = max_f;
            best_step = step;
            since_best = 0;
        } else if step > 0 {
            since_best += 1;
        }
        eprintln!(
            "step {step}: E={e:.6} max|F|={max_f:.4e} best={best_f:.4e}@{best_step} stale={since_best} rms={:.3e} Tr={:.5} R_H={:.3e} iters={} {ms:.0} ms",
            scc.rms, scc.tr_ks, scc.r_h, scc.n_iters
        );
        if step == 0 {
            assert!(
                max_f > 2e-2,
                "jittered start is already at a minimum: max|F|={max_f:.3e}"
            );
        }
        if step > 0 && max_f < F_FLOOR {
            stopped = format!("max|F|={max_f:.3e} < {F_FLOOR:.1e} at step {step}");
            eprintln!("{stopped}");
            break;
        }
        if step > 20 && since_best >= PLATEAU_PATIENCE {
            stopped = format!(
                "plateau: best max|F|={best_f:.3e} at step {best_step}, no new low for {since_best} steps (now {max_f:.3e})"
            );
            eprintln!("{stopped}");
            break;
        }
    }
    drop(traj);
    eng.prof_report("R10 FIRE");
    if n_done > 0 {
        eprintln!(
            "time: {n_done} steps  fire {sum_fire:.0} ms  scc {sum_scc:.0} ms  forces {sum_force:.0} ms  wall {:.1} s  best max|F|={best_f:.4e} at step {best_step}",
            wall.elapsed().as_secs_f64()
        );
    }
    if !stopped.is_empty() {
        writeln!(hist, "# {stopped}").unwrap();
        hist.flush().unwrap();
    }
    write_xyz(
        &dir.join("si_R10_after.xyz"),
        &mol.species,
        eng.coords(),
        &format!("Si sphere R10 after FIRE ({stopped})"),
    );
    let (sih1, sih1_min, n_h) = nearest_sih(&mol.species, eng.coords());
    let dmax = max_disp(&coords0, eng.coords());
    eprintln!(
        "geometry: max|dR|={dmax:.4} Å  <Si-H> {sih0:.4} → {sih1:.4} Å  min {sih0_min:.4} → {sih1_min:.4} (n_H={n_h})"
    );
    eprintln!("history {}", hist_path.display());
    eprintln!("trajectory {}", traj_path.display());
    let ok = stopped.is_empty()
        || stopped.starts_with("wall")
        || stopped.starts_with("max|F|")
        || stopped.starts_with("plateau");
    assert!(ok, "{stopped}");
}

/// Cold SCC plus a few warm FIRE steps. Timing only — not a relaxation.
fn probe_warm(tag: &str, xyz_name: &str, n_warm: usize) {
    let sk_dir = require_sih_sk_dir();
    let xyz = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../debug/nanocrystals").join(xyz_name);
    let mol = parse_xyz(xyz.to_str().unwrap()).unwrap_or_else(|e| panic!("{tag} parse: {e}"));
    let n = mol.species.len();
    eprintln!("{tag}: {n} atoms  ({xyz_name})");
    let sk = load_sk_for_species(&sk_dir, &mol.species).unwrap_or_else(|e| panic!("{tag} SK: {e}"));
    let t_new = Instant::now();
    let mut eng = SparseDftb::with_config(
        sk,
        &sk_dir,
        mol.species.clone(),
        mol.coords.clone(),
        si_cfg(),
    )
    .unwrap_or_else(|e| panic!("{tag} SparseDftb::new: {e}"));
    let init_s = t_new.elapsed().as_secs_f64();
    let jittered = jitter_coords(eng.coords(), JITTER_ANG, 0x5eed_u64);
    eng.set_coords(&jittered).unwrap_or_else(|e| panic!("{tag} jitter: {e}"));
    eng.set_geom_step(GeomStep::BoldXtrDmm);
    eng.set_fire_dt(0.1);
    let t0 = Instant::now();
    let cold = eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("{tag} cold SCC: {e}"));
    let cold_ms = t0.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "{tag} init {init_s:.1} s  cold {cold_ms:.0} ms  iters={} Tr={:.4} R_H={:.3e} rms={:.3e}",
        cold.n_iters, cold.tr_ks, cold.r_h, cold.rms
    );
    let mut sum = 0.0f64;
    for step in 1..=n_warm {
        let t = Instant::now();
        eng.fire_step(1e-6).unwrap_or_else(|e| panic!("{tag} FIRE {step}: {e}"));
        let scc = eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("{tag} warm {step}: {e}"));
        let ms = t.elapsed().as_secs_f64() * 1e3;
        sum += ms;
        let e = eng.energy().unwrap_or_else(|err| panic!("{tag} energy: {err}"));
        eprintln!(
            "{tag} warm {step}: {ms:.0} ms  E={e:.4} Tr={:.4} R_H={:.3e} rms={:.3e}",
            scc.tr_ks, scc.r_h, scc.rms
        );
    }
    eprintln!(
        "{tag} SCALE  n={n}  init={init_s:.2}s  cold={cold_ms:.0}ms  warm_mean={:.0}ms",
        sum / n_warm as f64
    );
}

#[test]
#[ignore = "R14/R18 warm-step timing, GPU, short"]
fn scale_si_spheres() {
    std::env::set_var("RUST_DFTB_SPARSE_ALGEBRA_VERBOSE", "0");
    std::env::set_var("RUST_DFTB_SCC_RHGATE", "1e-2");
    std::env::set_var("RUST_DFTB_PROF", "0");
    // Interior rows of K·S at r_k=12 do not fit the local-memory row cache
    // once the crystal is larger than R10. Store that product on M_K.
    std::env::set_var("RUST_DFTB_TRUNC_PRODUCTS", "1");
    let wall = Instant::now();
    probe_warm("R10", "si_sphere_R10.xyz", 4);
    if wall.elapsed().as_secs_f64() < 20.0 {
        probe_warm("R14", "si_sphere_R14.xyz", 4);
    }
    if wall.elapsed().as_secs_f64() < 40.0 {
        probe_warm("R18", "si_sphere_R18.xyz", 2);
    } else {
        eprintln!(
            "skip R18: {:.1} s already used",
            wall.elapsed().as_secs_f64()
        );
    }
    eprintln!("scale wall {:.1} s", wall.elapsed().as_secs_f64());
}

/// Continue from the saved R10 endpoint until max|F| stops setting new lows.
#[test]
#[ignore = "R10 force-floor continuation, GPU, wall-capped"]
fn relax_r10_floor() {
    std::env::set_var("RUST_DFTB_SPARSE_ALGEBRA_VERBOSE", "0");
    std::env::set_var("RUST_DFTB_SCC_RHGATE", "1e-2");
    std::env::set_var("RUST_DFTB_PROF", "0");
    let dir = out_dir();
    let xyz = dir.join("si_R10_after.xyz");
    let mol = parse_xyz(xyz.to_str().unwrap()).unwrap_or_else(|e| panic!("parse after: {e}"));
    let sk_dir = require_sih_sk_dir();
    let sk = load_sk_for_species(&sk_dir, &mol.species).unwrap_or_else(|e| panic!("SK: {e}"));
    let t_new = Instant::now();
    let mut eng = SparseDftb::with_config(
        sk,
        &sk_dir,
        mol.species.clone(),
        mol.coords.clone(),
        si_cfg(),
    )
    .unwrap_or_else(|e| panic!("SparseDftb::new: {e}"));
    eprintln!("floor init {:.1} s  atoms={}", t_new.elapsed().as_secs_f64(), mol.species.len());
    eng.set_geom_step(GeomStep::BoldXtrDmm);
    eng.set_fire_dt(0.1);
    let hist_path = dir.join("history_floor.csv");
    let hist_new = !hist_path.exists();
    let mut hist = BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&hist_path)
            .unwrap(),
    );
    if hist_new {
        writeln!(hist, "step,E_Ha,maxAbsF,ms,rms,TrKS,R_H,note").unwrap();
    }
    let traj_path = dir.join("si_R10_traj.xyz");
    let mut frame0 = 0usize;
    if let Ok(text) = std::fs::read_to_string(&traj_path) {
        frame0 = text.lines().filter(|l| l.starts_with("frame=") || l.starts_with("step=") || l.starts_with("floor=")).count();
    }
    let mut traj = BufWriter::new(
        std::fs::OpenOptions::new().append(true).open(&traj_path).unwrap(),
    );
    let wall = Instant::now();
    let mut best_f = f64::MAX;
    let mut best_step = 0usize;
    let mut since_best = 0usize;
    let mut stopped = String::new();
    const WALL: f64 = 45.0;
    for step in 0..=280 {
        if wall.elapsed().as_secs_f64() > WALL {
            stopped = format!("wall cap {WALL} s at floor-step {step}");
            eprintln!("{stopped}");
            break;
        }
        let t = Instant::now();
        if step > 0 {
            eng.fire_step(F_FLOOR).unwrap_or_else(|e| panic!("FIRE {step}: {e}"));
        }
        let scc = eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("SCC {step}: {e}"));
        let f = eng.forces().unwrap_or_else(|e| panic!("forces {step}: {e}"));
        let mut max_f = 0.0f64;
        for fi in &f.forces {
            for c in fi {
                max_f = max_f.max(c.abs());
            }
        }
        let e = eng.energy().unwrap_or_else(|err| panic!("energy {step}: {err}"));
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let note = if step == 0 { "cold" } else { "" };
        let frame = frame0 + step;
        writeln!(
            hist,
            "{frame},{e:.8},{max_f:.6e},{ms:.1},{:.6e},{:.6},{:.6e},{note}",
            scc.rms, scc.tr_ks, scc.r_h
        )
        .unwrap();
        write_frame(
            &mut traj,
            &mol.species,
            eng.coords(),
            &format!("frame={frame} E={e:.8} max|F|={max_f:.6e} {note}"),
        );
        if max_f < best_f * 0.98 {
            best_f = max_f;
            best_step = step;
            since_best = 0;
        } else if step > 0 {
            since_best += 1;
        }
        eprintln!(
            "floor {step}: E={e:.6} max|F|={max_f:.4e} best={best_f:.4e}@{best_step} stale={since_best} Tr={:.5} {ms:.0} ms {note}",
            scc.tr_ks
        );
        if step > 0 && max_f < F_FLOOR {
            stopped = format!("max|F|={max_f:.3e} < {F_FLOOR:.1e}");
            break;
        }
        if step > 15 && since_best >= PLATEAU_PATIENCE {
            stopped = format!(
                "plateau: best max|F|={best_f:.3e} at floor-step {best_step}, stale {since_best}"
            );
            eprintln!("{stopped}");
            break;
        }
    }
    hist.flush().unwrap();
    write_xyz(
        &dir.join("si_R10_after.xyz"),
        &mol.species,
        eng.coords(),
        &format!("Si sphere R10 after floor search ({stopped})"),
    );
    eprintln!(
        "floor done  best max|F|={best_f:.4e} at {best_step}  wall {:.1} s  {stopped}",
        wall.elapsed().as_secs_f64()
    );
}
