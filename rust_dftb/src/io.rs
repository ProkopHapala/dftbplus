//! Input/output utilities for DFTB+ file formats and test helpers.
//!
//! This module consolidates:
//! - DFTB+ matrix file reading/writing (`hamsqr*.dat`, `oversqr.dat`)
//! - XYZ file parsing
//! - Species/coordinate string parsing (for env-var driven tests)
//! - Comparison helpers (`max_abs_diff`, `max_abs_diff_vec`)
//! - Angular momentum maps for common SK sets
//! - Element name capitalization

use crate::core::error::{DftbError, Result};
use crate::methods::dftb::sk_data::SkData;
use nalgebra::DMatrix;
use std::fs::File;
use std::io::Write;

// ─── DFTB+ matrix format ───────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    DftbSquare,
}

/// Read/write DFTB+ square matrix formats (`hamsqr*.dat`, `oversqr.dat`).
pub struct DftbOutput;

impl DftbOutput {
    /// Read DFTB+ `hamsqr*.dat` / `oversqr.dat` format.
    /// Format: header comments, then "T  nOrb  nKpoint", then kpoint index, then "# MATRIX", then data.
    pub fn read_square(path: &str) -> Result<DMatrix<f64>> {
        let content = std::fs::read_to_string(path)?;

        let mut values = Vec::new();
        let mut dim_line_found = false;
        let mut n_orb = 0;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let tokens: Vec<&str> = trimmed.split_whitespace().collect();

            // Dimension line: "T  14  1" (logical, nOrb, nKpoint)
            if tokens.len() == 3 && !dim_line_found {
                if let (Ok(n), Ok(_k)) = (tokens[1].parse::<usize>(), tokens[2].parse::<usize>()) {
                    n_orb = n;
                    dim_line_found = true;
                    continue;
                }
            }

            // Skip kpoint index lines (single integer)
            if tokens.len() == 1 && tokens[0].parse::<usize>().is_ok() {
                continue;
            }

            for token in tokens {
                if let Ok(val) = token.parse::<f64>() {
                    values.push(val);
                }
            }
        }

        if !dim_line_found {
            return Err(DftbError::Parse("could not find matrix dimension line".into()));
        }

        let expected = n_orb * n_orb;
        if values.len() < expected {
            return Err(DftbError::Parse(format!(
                "not enough data: got {} values, expected {} for {}x{} matrix",
                values.len(), expected, n_orb, n_orb
            )));
        }

        let data: Vec<f64> = values[..expected].to_vec();
        Ok(DMatrix::from_row_slice(n_orb, n_orb, &data))
    }

    pub fn write_square(path: &str, mat: &DMatrix<f64>) -> Result<()> {
        if mat.nrows() != mat.ncols() {
            return Err(DftbError::InvalidInput("matrix must be square".into()));
        }
        let n = mat.nrows();
        let mut f = File::create(path)?;
        writeln!(f, "{} {}", n, n)?;
        for i in 0..n {
            for j in 0..n {
                if j + 1 == n {
                    write!(f, "{:.16e}", mat[(i, j)])?;
                } else {
                    write!(f, "{:.16e} ", mat[(i, j)])?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ─── XYZ file parsing ──────────────────────────────────────────────

/// Parsed XYZ file: species labels + coordinates in Ångström.
pub struct XyzMolecule {
    pub species: Vec<String>,
    pub coords: Vec<[f64; 3]>,
    pub comment: String,
}

/// Parse a standard XYZ file. Returns species (capitalized) and coords in Å.
pub fn parse_xyz(path: &str) -> Result<XyzMolecule> {
    let text = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 3 {
        return Err(DftbError::Parse("XYZ file too short".into()));
    }
    let n: usize = lines[0]
        .trim()
        .parse()
        .map_err(|e| DftbError::Parse(format!("invalid atom count: {e}")))?;
    let comment = lines[1].to_string();

    let mut species = Vec::with_capacity(n);
    let mut coords = Vec::with_capacity(n);
    for line in lines.iter().take(2 + n).skip(2) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(DftbError::Parse(format!("malformed XYZ line: {line}")));
        }
        species.push(capitalize(parts[0]));
        coords.push([
            parts[1].parse().map_err(|e| DftbError::Parse(format!("bad x: {e}")))?,
            parts[2].parse().map_err(|e| DftbError::Parse(format!("bad y: {e}")))?,
            parts[3].parse().map_err(|e| DftbError::Parse(format!("bad z: {e}")))?,
        ]);
    }

    Ok(XyzMolecule { species, coords, comment })
}

// ─── String parsing helpers (env-var driven tests) ─────────────────

/// Parse comma-separated species names, e.g. "H,H,O".
pub fn parse_species(s: &str) -> Vec<String> {
    s.split(',')
        .map(|x| capitalize(x.trim()))
        .filter(|x| !x.is_empty())
        .collect()
}

/// Parse coordinates from a flat string, e.g. "0,0,0,1,0,0" or "0 0 0 1 0 0".
pub fn parse_coords(s: &str) -> Vec<[f64; 3]> {
    let vals: Vec<f64> = s
        .split(|c| {
            c == ',' || c == ' ' || c == ';' || c == '\n' || c == '\t' || c == '[' || c == ']'
                || c == '(' || c == ')'
        })
        .filter(|x| !x.is_empty())
        .filter_map(|x| x.parse::<f64>().ok())
        .collect();
    assert!(vals.len() % 3 == 0, "coords count not divisible by 3");
    vals.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Parse comma-separated floats.
pub fn parse_f64_list(s: &str) -> Vec<f64> {
    s.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|x| !x.is_empty())
        .filter_map(|x| x.parse::<f64>().ok())
        .collect()
}

// ─── Comparison helpers ────────────────────────────────────────────

/// Max absolute difference between two matrices.
pub fn max_abs_diff(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    assert_eq!(a.shape(), b.shape());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Max absolute difference between two slices.
pub fn max_abs_diff_vec(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Compare two matrices element-wise, printing max abs/rel error and asserting tolerance.
pub fn compare_matrices(a: &DMatrix<f64>, b: &DMatrix<f64>, name: &str, tol: f64) {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    let mut max_err = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut max_idx = (0, 0);
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let diff = (a[(i, j)] - b[(i, j)]).abs();
            let ref_val = b[(i, j)].abs().max(1e-10);
            let rel = diff / ref_val;
            if diff > max_err {
                max_err = diff;
                max_rel = rel;
                max_idx = (i, j);
            }
        }
    }
    println!("{name}: max_abs_err = {max_err:.6e}, max_rel_err = {max_rel:.6e} at {max_idx:?}");
    assert!(max_err < tol, "{name} mismatch too large: max_err={max_err:.6e}");
}

/// Compare two vectors element-wise, printing max abs/rel error and asserting tolerance.
pub fn compare_vecs(a: &[f64], b: &[f64], name: &str, tol: f64) {
    assert_eq!(a.len(), b.len());
    let mut max_err = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        let ref_val = b[i].abs().max(1e-10);
        let rel = diff / ref_val;
        if diff > max_err {
            max_err = diff;
            max_rel = rel;
        }
    }
    println!("{name}: max_abs_err = {max_err:.6e}, max_rel_err = {max_rel:.6e}");
    assert!(max_err < tol, "{name} mismatch too large: max_err={max_err:.6e}");
}

// ─── Matrix utilities ──────────────────────────────────────────────

/// Permute p-orbital columns/rows within each atom block.
/// Assumes basis [s, p1, p2, p3] per atom (4 orbitals).
/// `perm_p` maps the desired ordering of the 3 p slots.
pub fn permute_sp_per_atom(mat: &DMatrix<f64>, perm_p: [usize; 3]) -> DMatrix<f64> {
    let n = mat.nrows();
    assert_eq!(n, mat.ncols());
    assert_eq!(n % 4, 0);
    let n_at = n / 4;

    let mut idx = Vec::with_capacity(n);
    for a in 0..n_at {
        let base = 4 * a;
        idx.push(base);
        let p = [base + 1, base + 2, base + 3];
        idx.push(p[perm_p[0]]);
        idx.push(p[perm_p[1]]);
        idx.push(p[perm_p[2]]);
    }

    let mut out = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            out[(i, j)] = mat[(idx[i], idx[j])];
        }
    }
    out
}

// ─── Angular momentum maps ─────────────────────────────────────────

/// Build a default angular-momentum map for common elements in mio-1-1.
/// H → [s], C/N/O/F/S/P → [s,p], transition metals → [s,p,d].
pub fn default_ang_map(
    species: &[String],
) -> std::collections::HashMap<String, Vec<i32>> {
    let mut m = std::collections::HashMap::new();
    for sp in species {
        let ang = match sp.as_str() {
            "H" | "He" => vec![0],
            "C" | "N" | "O" | "F" | "S" | "P" | "Cl" | "Br" | "I" | "B" | "Li" | "Na" | "K" | "Si" => {
                vec![0, 1]
            }
            _ => vec![0, 1, 2], // fallback: s+p+d
        };
        m.insert(sp.clone(), ang);
    }
    m
}

/// Load SK files from a directory and set angular momenta for the given species.
/// Convenience wrapper combining `SkData::load_sk_folder` + `set_species_angular_momenta`.
pub fn load_sk_for_species(sk_dir: &str, species: &[String]) -> Result<SkData> {
    let mut sk = SkData::load_sk_folder(sk_dir, ".skf", "-")?;
    sk.set_species_angular_momenta(default_ang_map(species));
    Ok(sk)
}

// ─── String helpers ────────────────────────────────────────────────

/// Capitalize first letter, lowercase rest: "h" → "H", "c" → "C".
pub fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
        None => String::new(),
    }
}
