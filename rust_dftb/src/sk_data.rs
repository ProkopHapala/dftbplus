use crate::error::{DftbError, Result};
use crate::interpolation::EqGridTable;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// Indices into the OLD 10-column SK table (0-based) for sp interactions.
// skMap(mm, lMax, lMin) in parser.F90 is Fortran column-major (mm varies fastest):
//   ss-σ: skMap(0, lMax=0, lMin=0)=20 → iSKInterOld pos 10 → old index 9
//   sp-σ: skMap(0, lMax=1, lMin=0)=19 → iSKInterOld pos  9 → old index 8
//   pp-σ: skMap(0, lMax=1, lMin=1)=15 → iSKInterOld pos  6 → old index 5
//   pp-π: skMap(1, lMax=1, lMin=1)=16 → iSKInterOld pos  7 → old index 6
// iSKInterOld = [8,9,10,13,14,15,16,18,19,20] (1-based new indices)
const OLD_SS_SIGMA: usize = 9; // old col 10 (0-based)
const OLD_SP_SIGMA: usize = 8; // old col 9
const OLD_PP_SIGMA: usize = 5; // old col 6
const OLD_PP_PI:    usize = 6; // old col 7

fn parse_f64(tok: &str) -> Result<f64> {
    let t = tok.replace('D', "E").replace('d', "e");
    t.parse::<f64>()
        .map_err(|e: std::num::ParseFloatError| DftbError::Parse(e.to_string()))
}

fn expand_repeat_token(tok: &str) -> Result<Option<Vec<f64>>> {
    // DFTB+ SK files often contain Fortran-like repeat syntax, e.g. "20*0.0" or "9*0.0".
    // We expand that here.
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
    // Extract as many numbers as possible from a line, expanding repeat tokens and skipping
    // any non-numeric tokens (e.g. "T"). Commas are treated as whitespace (mio files).
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
    // Parse a line as numeric-only tokens (with repeat expansion). Any non-numeric token is error.
    // Commas treated as whitespace (mio files).
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
}

#[derive(Debug, Clone)]
pub struct SkTableSp {
    pub sp1: String,
    pub sp2: String,
    pub h: EqGridTable, // [n_grid][4]
    pub s: EqGridTable, // [n_grid][4]
}

impl SkTableSp {
    pub fn cutoff(&self) -> f64 {
        self.h.r_max()
    }
}

#[derive(Debug, Clone, Default)]
pub struct SkData {
    pub onsite: HashMap<String, AtomicParamsSp>,
    pub pairs: HashMap<(String, String), SkTableSp>,
}

impl SkData {
    pub fn get_pair(&self, a: &str, b: &str) -> Option<&SkTableSp> {
        self.pairs
            .get(&(a.to_string(), b.to_string()))
            .or_else(|| self.pairs.get(&(b.to_string(), a.to_string())))
    }

    pub fn onsite(&self, a: &str) -> Result<AtomicParamsSp> {
        self.onsite
            .get(a)
            .cloned()
            .ok_or_else(|| DftbError::InvalidInput(format!("missing onsite params for {a}")))
    }

    /// Load all `.skf` files from a folder with DFTB+ Type2FileNames naming convention:
    /// `Prefix/El1-El2.suffix`. You must have homonuclear files to get onsite energies.
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
                let sk = read_skf_sp(&path, &a, &b)?;
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
    // Read 2nd data line for homonuclear SK file:
    // (skSelf(ii), ii=nShell,1,-1), rDummy, (skHubbU...), (skOcc...)
    // nShell=3 (s,p,d) for non-extended.
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();

    // skip optional '@' header
    let first = lines.next().ok_or_else(|| DftbError::Parse("empty skf".into()))?;
    let first_tokens = first.split_whitespace().collect::<Vec<_>>();
    if first_tokens.get(0) == Some(&"@") || first.trim_start().starts_with('@') {
        // extended file: still OK, but we only support sp onsite
        // read next line as grid header
    } else {
        // first line is grid header, so "unread" it by re-parsing with an iterator over all lines
        lines = text.lines();
    }

    // read grid line
    let grid_line = lines
        .next()
        .ok_or_else(|| DftbError::Parse("missing grid line".into()))?;
    let _grid: Vec<f64> = parse_numbers_strict(grid_line)?;

    let line2 = lines
        .next()
        .ok_or_else(|| DftbError::Parse("missing onsite line".into()))?;
    // Some parameter sets include non-numeric tokens on this line (e.g. "T").
    // We only need Es/Ep, so parse numbers loosely.
    let nums: Vec<f64> = parse_numbers_loose(line2);

    if nums.len() < 3 {
        return Err(DftbError::Parse("onsite line too short".into()));
    }

    // It contains Ed Ep Es ... but readFromFile assigns in reverse into skSelf(ii).
    // The first 3 numbers correspond to (Ed, Ep, Es) for non-extended.
    let e_d = nums[0];
    let e_p = nums[1];
    let e_s = nums[2];
    let _ = e_d;

    Ok(AtomicParamsSp { e_s, e_p })
}

fn read_skf_sp(path: &Path, sp1: &str, sp2: &str) -> Result<SkTableSp> {
    // Minimal parser for the integrals part of old-format .skf.
    // We ignore spline repulsive etc.
    let text = fs::read_to_string(path)?;
    let mut it = text.lines();

    // detect optional '@'
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

    // oldskdata: skData%nGrid = nGrid - 1
    let n_grid = n_grid_raw.saturating_sub(1);

    // skip 2nd/3rd lines (atomic params / poly repulsive) if homonuclear
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
        let nums: Vec<f64> = parse_numbers_strict(line)?;

        if extended {
            // extended has 20 ham + 20 over
            if nums.len() < 40 {
                return Err(DftbError::InvalidSkFormat("extended line needs 40 numbers".into()));
            }
            let ham = &nums[0..20];
            let ovl = &nums[20..40];
            h_vals.push(pick_sp(ham));
            s_vals.push(pick_sp(ovl));
        } else {
            // old has 10 ham + 10 over, corresponding to indices iSKInterOld.
            if nums.len() < 20 {
                return Err(DftbError::InvalidSkFormat("old line needs 20 numbers".into()));
            }
            let ham10 = &nums[0..10];
            let ovl10 = &nums[10..20];
            // Pick sp integrals using correct old-format column positions
            let h_sp = vec![ham10[OLD_SS_SIGMA], ham10[OLD_SP_SIGMA], ham10[OLD_PP_SIGMA], ham10[OLD_PP_PI]];
            let s_sp = vec![ovl10[OLD_SS_SIGMA], ovl10[OLD_SP_SIGMA], ovl10[OLD_PP_SIGMA], ovl10[OLD_PP_PI]];
            h_vals.push(h_sp);
            s_vals.push(s_sp);
        }
    }

    Ok(SkTableSp {
        sp1: sp1.to_string(),
        sp2: sp2.to_string(),
        h: EqGridTable { dr: dist, values: h_vals },
        s: EqGridTable { dr: dist, values: s_vals },
    })
}

fn pick_sp(arr20: &[f64]) -> Vec<f64> {
    // arr20 is 20 long (0-based), using skMap(mm,lMax,lMin) column-major:
    // ss-σ: new col 20 → idx 19
    // sp-σ: new col 19 → idx 18
    // pp-σ: new col 15 → idx 14
    // pp-π: new col 16 → idx 15
    vec![
        arr20[19], // ss-σ
        arr20[18], // sp-σ
        arr20[14], // pp-σ
        arr20[15], // pp-π
    ]
}
