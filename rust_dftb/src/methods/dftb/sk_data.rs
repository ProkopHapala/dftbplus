use crate::core::error::{DftbError, Result};
use crate::methods::dftb::interpolation::EqGridTable;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Orbital info for a species, matching Fortran TOrbitals per-species fields.
#[derive(Debug, Clone)]
pub struct SpeciesOrbitals {
    pub ang_shells: Vec<i32>,
    pub n_orb: usize,
    pub n_shell: usize,
}

impl SpeciesOrbitals {
    pub fn from_ang_momenta(ang: &[i32]) -> Self {
        let n_orb = ang.iter().map(|&l| (2 * l + 1) as usize).sum();
        Self {
            ang_shells: ang.to_vec(),
            n_orb,
            n_shell: ang.len(),
        }
    }
    pub fn shell_offsets(&self) -> Vec<usize> {
        let mut off = vec![0usize; self.n_shell];
        for i in 1..self.n_shell {
            off[i] = off[i - 1] + (2 * self.ang_shells[i - 1] + 1) as usize;
        }
        off
    }
}

fn parse_f64(tok: &str) -> Result<f64> {
    let t = tok.replace('D', "E").replace('d', "e");
    t.parse::<f64>()
        .map_err(|e: std::num::ParseFloatError| DftbError::Parse(e.to_string()))
}

fn expand_repeat_token(tok: &str) -> Result<Option<Vec<f64>>> {
    if let Some((n_str, val_str)) = tok.split_once('*') {
        let n: usize = n_str
            .parse()
            .map_err(|e: std::num::ParseIntError| DftbError::Parse(e.to_string()))?;
        let v = parse_f64(val_str)?;
        return Ok(Some(vec![v; n]));
    }
    Ok(None)
}

fn parse_numbers_loose(line: &str) -> Vec<f64> {
    let line = line.replace(',', " ");
    let mut out = Vec::new();
    for tok in line.split_whitespace() {
        if let Ok(Some(vs)) = expand_repeat_token(tok) {
            out.extend(vs);
            continue;
        }
        if let Ok(v) = parse_f64(tok) {
            out.push(v);
        }
    }
    out
}

fn parse_numbers_strict(line: &str) -> Result<Vec<f64>> {
    let line = line.replace(',', " ");
    let mut out = Vec::new();
    for tok in line.split_whitespace() {
        if let Some(vs) = expand_repeat_token(tok)? {
            out.extend(vs);
            continue;
        }
        out.push(parse_f64(tok)?);
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct AtomicParamsSp {
    pub e_s: f64,
    pub e_p: f64,
    pub q0: f64,
    pub u_hubbard: f64,
}

#[derive(Debug, Clone)]
pub struct SkTableSp {
    pub sp1: String,
    pub sp2: String,
    pub h: EqGridTable,
    pub s: EqGridTable,
}

impl SkTableSp {
    pub fn cutoff(&self) -> f64 {
        self.h.r_max()
    }

    pub fn eval_shell_integrals_into(
        &self,
        ang1: i32,
        ang2: i32,
        r: f64,
        out_h: &mut [f64],
        out_s: &mut [f64],
    ) -> Result<usize> {
        let mut h_all = [0.0f64; 20];
        let mut s_all = [0.0f64; 20];
        self.h.eval_into(r, &mut h_all)?;
        self.s.eval_into(r, &mut s_all)?;

        let (l_min, l_max) = if ang1 <= ang2 { (ang1, ang2) } else { (ang2, ang1) };
        let n_mm = (l_min + 1) as usize;

        let is_extended = h_all.iter().skip(10).any(|&x| x != 0.0);
        if is_extended {
            for mm in 0..=l_min {
                let new_col = sk_map(mm, l_max, l_min) as usize;
                out_h[mm as usize] = h_all[new_col - 1];
                out_s[mm as usize] = s_all[new_col - 1];
            }
        } else {
            const NEW_TO_OLD: [usize; 21] = {
                let mut arr = [0usize; 21];
                let iSKInterOld: [usize; 10] = [8, 9, 10, 13, 14, 15, 16, 18, 19, 20];
                let mut i = 0;
                while i < 10 {
                    arr[iSKInterOld[i]] = i;
                    i += 1;
                }
                arr
            };
            for mm in 0..=l_min {
                let new_col = sk_map(mm, l_max, l_min) as usize;
                let old_idx = NEW_TO_OLD[new_col];
                out_h[mm as usize] = h_all[old_idx];
                out_s[mm as usize] = s_all[old_idx];
            }
        }
        Ok(n_mm)
    }
}

#[derive(Debug, Clone, Default)]
pub struct SkData {
    pub onsite: HashMap<String, AtomicParamsSp>,
    pub pairs: HashMap<(String, String), SkTableSp>,
    pub orbital_info: HashMap<String, SpeciesOrbitals>,
}

impl SkData {
    pub fn get_pair(&self, a: &str, b: &str) -> Option<&SkTableSp> {
        if let Some(tab) = self.pairs.get(&(a.to_string(), b.to_string())) {
            return Some(tab);
        }
        self.pairs.get(&(b.to_string(), a.to_string()))
    }

    pub fn onsite(&self, a: &str) -> Result<&AtomicParamsSp> {
        self.onsite
            .get(a)
            .ok_or_else(|| DftbError::InvalidInput(format!("missing onsite params for {a}")))
    }

    pub fn set_species_angular_momenta(&mut self, ang: HashMap<String, Vec<i32>>) {
        self.orbital_info.clear();
        for (sp, shells) in ang {
            self.orbital_info.insert(sp, SpeciesOrbitals::from_ang_momenta(&shells));
        }
    }

    pub fn n_orb_species(&self, sp: &str) -> Result<usize> {
        self.orbital_info
            .get(sp)
            .map(|o| o.n_orb)
            .ok_or_else(|| DftbError::InvalidInput(format!("missing orbital info for {sp}")))
    }

    pub fn ang_shells(&self, sp: &str) -> Result<&[i32]> {
        self.orbital_info
            .get(sp)
            .map(|o| o.ang_shells.as_slice())
            .ok_or_else(|| DftbError::InvalidInput(format!("missing orbital info for {sp}")))
    }

    pub fn eval_shell_integrals(&self, sp1: &str, sp2: &str, ang1: i32, ang2: i32, r: f64)
        -> Result<(Vec<f64>, Vec<f64>)> {
        let (lookup_sp1, lookup_sp2) = if ang1 <= ang2 {
            (sp1, sp2)
        } else {
            (sp2, sp1)
        };
        let tab = self.get_pair(lookup_sp1, lookup_sp2).ok_or_else(|| {
            DftbError::InvalidInput(format!("missing SK table for {lookup_sp1}-{lookup_sp2}"))
        })?;
        let h_all = tab.h.eval(r)?;
        let s_all = tab.s.eval(r)?;
        if h_all.len() != s_all.len() {
            return Err(DftbError::Parse("H and S integral count mismatch".into()));
        }
        let is_extended = h_all.len() == 20;
        let h_shell = if is_extended {
            extract_shell_integrals_new(&h_all, ang1, ang2)
        } else {
            extract_shell_integrals_old(&h_all, ang1, ang2)
        };
        let s_shell = if is_extended {
            extract_shell_integrals_new(&s_all, ang1, ang2)
        } else {
            extract_shell_integrals_old(&s_all, ang1, ang2)
        };
        Ok((h_shell, s_shell))
    }

    pub fn load_sk_folder(prefix: impl AsRef<Path>, suffix: &str, sep: &str) -> Result<Self> {
        let prefix = prefix.as_ref();
        let mut out = SkData::default();

        for entry in fs::read_dir(prefix)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some(suffix.trim_start_matches('.')) {
                continue;
            }
            if let Some((a, b)) = parse_pair_from_filename(&path, sep, suffix) {
                let sk = read_skf_all(&path, &a, &b)?;
                if a == b {
                    out.onsite.insert(a.clone(), sk_read_onsite_sp(&path)?);
                }
                out.pairs.insert((a, b), sk);
            }
        }

        Ok(out)
    }
}

fn parse_pair_from_filename(path: &Path, sep: &str, suffix: &str) -> Option<(String, String)> {
    let name = path.file_name()?.to_string_lossy();
    let base = name.strip_suffix(suffix)?;
    let mut it = base.split(sep);
    let a = it.next()?.to_string();
    let b = it.next()?.to_string();
    Some((a, b))
}

fn sk_read_onsite_sp(path: &Path) -> Result<AtomicParamsSp> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();

    let first = lines.next().ok_or_else(|| DftbError::Parse("empty skf".into()))?;
    let first_tokens = first.split_whitespace().collect::<Vec<_>>();
    if first_tokens.get(0) == Some(&"@") || first.trim_start().starts_with('@') {
    } else {
        lines = text.lines();
    }

    let grid_line = lines
        .next()
        .ok_or_else(|| DftbError::Parse("missing grid line".into()))?;
    let grid_nums: Vec<f64> = parse_numbers_strict(grid_line)?;
    let n_shell = if grid_nums.len() >= 3 {
        grid_nums[2] as usize
    } else {
        1
    };

    let line2 = lines
        .next()
        .ok_or_else(|| DftbError::Parse("missing onsite line".into()))?;
    let nums: Vec<f64> = parse_numbers_loose(line2);

    if nums.len() < 3 {
        return Err(DftbError::Parse("onsite line too short".into()));
    }

    let e_d = nums[0];
    let e_p = nums[1];
    let e_s = nums[2];
    let _ = e_d;

    // Hubbard U is at index 6 in the standard mio SK format onsite line.
    // Format: Ed Ep Es <params> U_hub ... q0 values
    let u_hubbard = if nums.len() > 6 { nums[6] } else { 0.0 };

    let q0 = if nums.len() >= n_shell {
        nums[nums.len() - n_shell..].iter().sum()
    } else {
        0.0
    };

    Ok(AtomicParamsSp { e_s, e_p, q0, u_hubbard })
}

fn read_skf_all(path: &Path, sp1: &str, sp2: &str) -> Result<SkTableSp> {
    let text = fs::read_to_string(path)?;
    let mut it = text.lines();

    let first = it.next().ok_or_else(|| DftbError::Parse("empty skf".into()))?;
    let extended = first.trim_start().starts_with('@');
    let grid_line = if extended {
        it.next().ok_or_else(|| DftbError::Parse("missing grid line".into()))?
    } else {
        first
    };

    let grid_line_clean = grid_line.replace(',', " ");
    let mut grid = grid_line_clean.split_whitespace();
    let dist_tok = grid
        .next()
        .ok_or_else(|| DftbError::Parse("missing dist".into()))?;
    let dist: f64 = parse_f64(dist_tok)?;
    let n_grid_raw: usize = grid
        .next()
        .ok_or_else(|| DftbError::Parse("missing ngrid".into()))?
        .parse()
        .map_err(|e: std::num::ParseIntError| DftbError::Parse(e.to_string()))?;
    let n_grid = n_grid_raw.saturating_sub(1);

    if sp1 == sp2 {
        it.next();
        it.next();
    } else {
        it.next();
    }

    let mut h_vals = Vec::with_capacity(n_grid);
    let mut s_vals = Vec::with_capacity(n_grid);

    for _ in 0..n_grid {
        let line = it.next().ok_or_else(|| DftbError::Parse("unexpected EOF in integrals".into()))?;
        let nums = parse_numbers_loose(line);

        if extended {
            if nums.len() < 40 {
                return Err(DftbError::InvalidSkFormat("extended line needs 40 numbers".into()));
            }
            h_vals.push(nums[0..20].to_vec());
            s_vals.push(nums[20..40].to_vec());
        } else {
            if nums.len() < 20 {
                return Err(DftbError::InvalidSkFormat("old line needs 20 numbers".into()));
            }
            h_vals.push(nums[0..10].to_vec());
            s_vals.push(nums[10..20].to_vec());
        }
    }

    Ok(SkTableSp {
        sp1: sp1.to_string(),
        sp2: sp2.to_string(),
        h: EqGridTable { dr: dist, values: h_vals },
        s: EqGridTable { dr: dist, values: s_vals },
    })
}

fn extract_shell_integrals_old(arr10: &[f64], ang1: i32, ang2: i32) -> Vec<f64> {
    let iSKInterOld: [usize; 10] = [8, 9, 10, 13, 14, 15, 16, 18, 19, 20];
    let mut new_to_old = [0usize; 21];
    for (old_idx, &new_col) in iSKInterOld.iter().enumerate() {
        new_to_old[new_col] = old_idx;
    }

    let (l_min, l_max) = if ang1 <= ang2 { (ang1, ang2) } else { (ang2, ang1) };
    let n_mm = (l_min + 1) as usize;
    let mut out = Vec::with_capacity(n_mm);

    for mm in 0..=l_min {
        let new_col = sk_map(mm, l_max, l_min);
        let old_idx = new_to_old[new_col as usize];
        out.push(arr10[old_idx]);
    }
    out
}

fn extract_shell_integrals_new(arr20: &[f64], ang1: i32, ang2: i32) -> Vec<f64> {
    let (l_min, l_max) = if ang1 <= ang2 { (ang1, ang2) } else { (ang2, ang1) };
    let n_mm = (l_min + 1) as usize;
    let mut out = Vec::with_capacity(n_mm);
    for mm in 0..=l_min {
        let new_col = sk_map(mm, l_max, l_min) as usize;
        out.push(arr20[new_col - 1]);
    }
    out
}

pub fn sk_map(mm: i32, l_max: i32, l_min: i32) -> i32 {
    match (mm, l_max, l_min) {
        (0, 0, 0) => 20,
        (0, 1, 0) => 19,
        (0, 1, 1) => 15,
        (1, 1, 1) => 16,
        _ => panic!("sk_map: unsupported ({}, {}, {})", mm, l_max, l_min),
    }
}
