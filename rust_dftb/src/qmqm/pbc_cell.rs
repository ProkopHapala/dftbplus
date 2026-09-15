//! Host-side periodic-cell math for the PBC path: lattice vectors,
//! reciprocal lattice, Ewald α/cutoff autotuning, real/reciprocal
//! lattice-point lists, and image-pair (CSR) enumeration.
//!
//! Direct port of the Fortran machinery (see
//! doc/prokop/tasts/HBond_Relaxed_Scan_GPU/Dense_Multi_PBC.ewald_notes.md):
//!   - `getOptimalAlphaEwald` / `getMaxREwald` / `getMaxGEwald`
//!     (src/dftbp/dftb/coulomb.F90)
//!   - `getLatticePoints` / `getCellTranslations`
//!     (src/dftbp/dftb/periodic.F90)
//!   - `invRPeriodicSerial` (the symmetric invRMat recipe)
//!
//! Conventions: atomic units (Bohr, Hartree). `lat` rows are the lattice
//! vectors a1,a2,a3; `rec` rows are the reciprocal vectors INCLUDING 2π
//! (b_i = 2π·(a_j × a_k)/V), matching DFTB+ `recVecs`. Cartesian G and R
//! vectors are materialized once — kernels take Cartesian lists.

use crate::core::error::{DftbError, Result};

macro_rules! bail {
    ($($t:tt)*) => { return Err(DftbError::InvalidInput(format!($($t)*))) };
}

/// Distance below which two positions count as coincident
/// (Fortran `tolSameDist`).
pub const TOL_SAME_DIST: f64 = 1.0e-5;
/// Default Ewald tolerance (Fortran `EwaldTolerance` default).
pub const TOL_EWALD_DEFAULT: f64 = 1.0e-9;
/// Bisection iteration cap (Fortran `nSearchIter`).
const N_SEARCH_ITER: usize = 30;

// ------------------------------------------------------------------
// Cell
// ------------------------------------------------------------------

/// Periodic cell: lattice + derived reciprocal vectors and volume.
#[derive(Debug, Clone)]
pub struct PbcCell {
    /// Lattice vectors a1,a2,a3 as rows (Bohr).
    pub lat: [[f64; 3]; 3],
    /// Reciprocal vectors b1,b2,b3 as rows, including 2π.
    pub rec: [[f64; 3]; 3],
    /// Cell volume (Bohr³).
    pub vol: f64,
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

impl PbcCell {
    /// Build from the three lattice vectors (rows, Bohr).
    pub fn new(lat: [[f64; 3]; 3]) -> Result<Self> {
        let vol = dot(lat[0], cross(lat[1], lat[2])).abs();
        if vol < 1.0e-12 {
            bail!("PbcCell: degenerate lattice (volume ~ 0): {lat:?}");
        }
        let mut rec = [[0.0; 3]; 3];
        rec[0] = cross(lat[1], lat[2]);
        rec[1] = cross(lat[2], lat[0]);
        rec[2] = cross(lat[0], lat[1]);
        for r in rec.iter_mut() {
            for x in r.iter_mut() {
                *x *= 2.0 * std::f64::consts::PI / vol;
            }
        }
        Ok(Self { lat, rec, vol })
    }

    /// Cartesian position of the cell shift (n1,n2,n3).
    #[inline]
    pub fn rvec(&self, n: [i32; 3]) -> [f64; 3] {
        let mut r = [0.0; 3];
        for d in 0..3 {
            r[d] = n[0] as f64 * self.lat[0][d]
                + n[1] as f64 * self.lat[1][d]
                + n[2] as f64 * self.lat[2][d];
        }
        r
    }
    /// Cartesian reciprocal vector of the integer G-index (n1,n2,n3)
    /// (includes 2π).
    #[inline]
    pub fn gvec(&self, n: [i32; 3]) -> [f64; 3] {
        let mut g = [0.0; 3];
        for d in 0..3 {
            g[d] = n[0] as f64 * self.rec[0][d]
                + n[1] as f64 * self.rec[1][d]
                + n[2] as f64 * self.rec[2][d];
        }
        g
    }

    /// Integer box bounds for enumerating all cells/points within
    /// `dist` of the origin. |n_i| ≤ dist/h_i + 1 where h_i is the
    /// inter-plane spacing: h_i = 2π/|b_i| for real cells,
    /// h_i = 2π/|a_i| for reciprocal points (Fortran uses the same
    /// generous bound then filters by true distance).
    fn ibound(&self, dist: f64, reciprocal: bool) -> [i32; 3] {
        let mut out = [0i32; 3];
        for i in 0..3 {
            // |n_i|·h_i ≤ dist  →  n_i ≤ dist·|basis_i|/(2π)
            let basis_len = if reciprocal { norm(self.lat[i]) } else { norm(self.rec[i]) };
            out[i] = (dist * basis_len / (2.0 * std::f64::consts::PI)).ceil() as i32 + 1;
        }
        out
    }

    /// All real-space cell translations with |R| ≤ `cutoff` (Bohr),
    /// origin first (Fortran `getCellTranslations`/`getLatticePoints`
    /// convention). Returns integer shifts; Cartesian via `rvec`.
    pub fn cell_translations(&self, cutoff: f64) -> Vec<[i32; 3]> {
        let b = self.ibound(cutoff, false);
        let mut out = vec![[0, 0, 0]];
        let c2 = (cutoff + TOL_SAME_DIST) * (cutoff + TOL_SAME_DIST);
        for n1 in -b[0]..=b[0] {
            for n2 in -b[1]..=b[1] {
                for n3 in -b[2]..=b[2] {
                    if n1 == 0 && n2 == 0 && n3 == 0 {
                        continue;
                    }
                    let n = [n1, n2, n3];
                    let r = self.rvec(n);
                    if dot(r, r) <= c2 {
                        out.push(n);
                    }
                }
            }
        }
        out
    }

    /// Half-space reciprocal lattice points with |G| ≤ `gmax` (Bohr⁻¹),
    /// origin excluded, inversion-reduced (only one of ±G kept —
    /// Fortran `reduceByInversion`). Cartesian vectors returned.
    pub fn g_lattice_points(&self, gmax: f64) -> Vec<[f64; 3]> {
        let b = self.ibound(gmax, true);
        let g2max = gmax * gmax;
        let mut out = Vec::new();
        for n1 in -b[0]..=b[0] {
            for n2 in -b[1]..=b[1] {
                for n3 in -b[2]..=b[2] {
                    // half-space: keep n1>0, or n1==0&&n2>0, or 0,0&&n3>0
                    if !(n1 > 0 || (n1 == 0 && n2 > 0) || (n1 == 0 && n2 == 0 && n3 > 0)) {
                        continue;
                    }
                    let g = self.gvec([n1, n2, n3]);
                    if dot(g, g) <= g2max {
                        out.push(g);
                    }
                }
            }
        }
        out
    }
}

// ------------------------------------------------------------------
// Ewald autotuning (port of coulomb.F90 bisections)
// ------------------------------------------------------------------

/// erfc(x) for x ≥ 0 — local implementation (no libm dep).
/// x < 3.5: erf Taylor series (converges fast, cancellation error stays
/// < ~1e-13 of the result). x ≥ 3.5: the standard continued fraction
/// erfc(x) = e^{−x²}/√π · 1/(x + 0.5/(x + 1.0/(x + 1.5/(x + …))))
/// which converges rapidly only for large x (at x≲1 it needs hundreds
/// of terms — do NOT use it there).
/// Accuracy ~1e-13 — well inside the 1e-9 Ewald tolerance.
pub fn erfc_host(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_host(-x);
    }
    if x < 3.5 {
        let x2 = x * x;
        let mut s = x;
        let mut term = x;
        for k in 1..120 {
            term *= -x2 / k as f64;
            let c = term / (2 * k + 1) as f64;
            s += c;
            if c.abs() < 1e-19 {
                break;
            }
        }
        1.0 - 2.0 / std::f64::consts::PI.sqrt() * s
    } else {
        let mut f = 0.0f64;
        for k in (1..=200).rev() {
            f = (k as f64 * 0.5) / (x + f);
        }
        f = 1.0 / (x + f);
        (-x * x).exp() / std::f64::consts::PI.sqrt() * f
    }
}

/// rTerm(r,α) = erfc(αr)/r — real-part term magnitude.
#[inline]
fn r_term(r: f64, alpha: f64) -> f64 {
    erfc_host(alpha * r) / r
}
/// gTerm(g,α,V) = 4π·e^{−g²/4α²}/(V·g²) — reciprocal term magnitude.
#[inline]
fn g_term(g: f64, alpha: f64, vol: f64) -> f64 {
    4.0 * std::f64::consts::PI * (-0.25 * g * g / (alpha * alpha)).exp() / (vol * g * g)
}
/// diffRecReal(α) — balances real/reciprocal decay at multiples of the
/// shortest lattice periods (Fortran: 4·minG vs 5·minG, 2·minR vs 3·minR).
#[inline]
fn diff_rec_real(alpha: f64, min_g: f64, min_r: f64, vol: f64) -> f64 {
    (g_term(4.0 * min_g, alpha, vol) - g_term(5.0 * min_g, alpha, vol))
        - (r_term(2.0 * min_r, alpha) - r_term(3.0 * min_r, alpha))
}

/// Optimal Ewald α — port of `getOptimalAlphaEwald`: double-bracket then
/// bisect `diffRecReal(α) = 0` within `tol`.
pub fn optimal_alpha(cell: &PbcCell, tol: f64) -> Result<f64> {
    if tol <= 0.0 {
        bail!("optimal_alpha: tol <= 0");
    }
    let min_g = cell.rec.iter().map(|r| norm(*r)).fold(f64::MAX, f64::min);
    let min_r = cell.lat.iter().map(|r| norm(*r)).fold(f64::MAX, f64::min);
    let mut alpha = 1.0e-8;
    let mut diff = diff_rec_real(alpha, min_g, min_r, cell.vol);
    let mut guard = 0;
    while diff < -tol && guard < 200 {
        alpha *= 2.0;
        diff = diff_rec_real(alpha, min_g, min_r, cell.vol);
        guard += 1;
    }
    if guard >= 200 {
        bail!("optimal_alpha: failed to bracket (diff<0 forever)");
    }
    if alpha <= 1.0e-8 {
        bail!("optimal_alpha: diff >= 0 at alpha=1e-8 — cell too extreme");
    }
    let mut left = 0.5 * alpha;
    while diff < tol && guard < 400 {
        alpha *= 2.0;
        diff = diff_rec_real(alpha, min_g, min_r, cell.vol);
        guard += 1;
    }
    if diff < tol {
        bail!("optimal_alpha: failed to bracket upper side");
    }
    let mut right = alpha;
    alpha = 0.5 * (left + right);
    let mut iter = 0;
    diff = diff_rec_real(alpha, min_g, min_r, cell.vol);
    while diff.abs() > tol && iter < N_SEARCH_ITER {
        if diff < 0.0 {
            left = alpha;
        } else {
            right = alpha;
        }
        alpha = 0.5 * (left + right);
        diff = diff_rec_real(alpha, min_g, min_r, cell.vol);
        iter += 1;
    }
    if diff.abs() > tol {
        bail!("optimal_alpha: bisection did not converge in {N_SEARCH_ITER} iters");
    }
    Ok(alpha)
}

/// Real-space cutoff: bisection `rTerm(r,α) = tol` (`getMaxREwald`).
pub fn max_r_ewald(alpha: f64, tol: f64) -> Result<f64> {
    let mut x = 1.0e-8;
    let mut y = r_term(x, alpha);
    let mut guard = 0;
    while y > tol && guard < 200 {
        x *= 2.0;
        y = r_term(x, alpha);
        guard += 1;
    }
    if y > tol {
        bail!("max_r_ewald: rTerm never drops below tol (alpha={alpha})");
    }
    let mut xleft = 0.5 * x;
    let mut xright = x;
    let mut yleft = r_term(xleft, alpha);
    let mut yright = y;
    for _ in 0..N_SEARCH_ITER {
        if yleft - yright <= tol {
            break;
        }
        let xm = 0.5 * (xleft + xright);
        let ym = r_term(xm, alpha);
        if ym >= tol {
            xleft = xm;
            yleft = ym;
        } else {
            xright = xm;
            yright = ym;
        }
    }
    Ok(xleft)
}

/// Reciprocal cutoff: bisection `gTerm(g,α,V) = tol` (`getMaxGEwald`).
pub fn max_g_ewald(alpha: f64, vol: f64, tol: f64) -> Result<f64> {
    let mut x = 1.0e-8;
    let mut y = g_term(x, alpha, vol);
    let mut guard = 0;
    while y > tol && guard < 200 {
        x *= 2.0;
        y = g_term(x, alpha, vol);
        guard += 1;
    }
    if y > tol {
        bail!("max_g_ewald: gTerm never drops below tol (alpha={alpha})");
    }
    let mut xleft = 0.5 * x;
    let mut xright = x;
    let mut yleft = g_term(xleft, alpha, vol);
    let mut yright = y;
    for _ in 0..N_SEARCH_ITER {
        if yleft - yright <= tol {
            break;
        }
        let xm = 0.5 * (xleft + xright);
        let ym = g_term(xm, alpha, vol);
        if ym >= tol {
            xleft = xm;
            yleft = ym;
        } else {
            xright = xm;
            yright = ym;
        }
    }
    Ok(xleft)
}

// ------------------------------------------------------------------
// invRMat — host f64 reference (test oracle + fallback)
// ------------------------------------------------------------------

/// Periodic 1/R matrix, full symmetric n×n — reference implementation
/// of `invRPeriodicSerial`. The real part sums images within `max_r` of
/// the PAIR distance |r_i − r_j − R| (the neighbor-list convention —
/// not an origin-centered ball, which misses images on the far side of
/// the cell); `rlat` must therefore cover |R| ≤ max_r + max pair
/// displacement. Reciprocal part over the half-space `gpoints`; then
/// −π/(Vα²) on every element and −2α/√π on the diagonal.
pub fn ewald_invmat_host(
    coords: &[[f64; 3]],
    cell: &PbcCell,
    alpha: f64,
    max_r: f64,
    gpoints: &[[f64; 3]],
    rlat: &[[f64; 3]],
) -> Vec<f64> {
    let n = coords.len();
    let mut m = vec![0.0f64; n * n];
    let c_self = -2.0 * alpha / std::f64::consts::PI.sqrt();
    let c_const = -std::f64::consts::PI / (cell.vol * alpha * alpha);
    let r2max = max_r * max_r;
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for r in rlat {
                let dx = coords[i][0] - coords[j][0] - r[0];
                let dy = coords[i][1] - coords[j][1] - r[1];
                let dz = coords[i][2] - coords[j][2] - r[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 < TOL_SAME_DIST * TOL_SAME_DIST || r2 > r2max {
                    continue;
                }
                let rr = r2.sqrt();
                s += erfc_host(alpha * rr) / rr;
            }
            // reciprocal part over the half-space G list (factor 2)
            let mut rec = 0.0;
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            for g in gpoints {
                let g2 = dot(*g, *g);
                let gr = g[0] * dx + g[1] * dy + g[2] * dz;
                rec += (-0.25 * g2 / (alpha * alpha)).exp() / g2 * gr.cos();
            }
            let mut v = s + 2.0 * rec * 4.0 * std::f64::consts::PI / cell.vol + c_const;
            if i == j {
                v += c_self;
            }
            m[i * n + j] = v;
            m[j * n + i] = v;
        }
    }
    m
}

// ------------------------------------------------------------------
// Image-pair enumeration (CSR)
// ------------------------------------------------------------------

/// CSR image-pair list over unordered central-cell pairs (i ≤ j):
/// for each pair, the R-shifts such that |r_j + R − r_i| < cutoff.
/// For i == j the R = 0 slot is EXCLUDED (self-interaction handled by
/// the caller — onsite blocks / Ewald self-term).
///
/// Convention: slot R means "image of j at cell R near i", displacement
/// d = r_j + R − r_i (direction i → j+R, matches the SK block's l,m,n).
#[derive(Debug, Clone)]
pub struct ImagePairList {
    /// Pair table: (atom_i, atom_j), j ≥ i.
    pub pair_ij: Vec<(u16, u16)>,
    /// CSR row pointer into `rvecs` per pair.
    pub r_off: Vec<i32>,
    /// Cartesian R vectors (Bohr) per slot.
    pub rvecs: Vec<[f32; 3]>,
    /// Distances per slot (Bohr) — host convenience.
    pub dists: Vec<f32>,
}

/// Enumerate image pairs: for each unordered (i ≤ j) all cells R with
/// |r_j + R − r_i| < cutoff. Cells come from
/// `cell.cell_translations(cutoff + max extent)`. `same_cell_self`
/// controls whether the (i, i, R=0) slot is emitted.
/// `emit_empty`: if true, EVERY i ≤ j pair is emitted even with zero
/// slots — required for Ewald/γ lists (reciprocal + background + onsite
/// contributions are nonzero even when no real-space image is in range).
pub fn enumerate_image_pairs(
    coords: &[[f64; 3]],
    cell: &PbcCell,
    cutoff: f64,
    same_cell_self: bool,
    emit_empty: bool,
) -> ImagePairList {
    let n = coords.len();
    // Max intra-cell displacement bound: any pair is within
    // |r_i − r_j| ≤ extent; cells to scan = those within cutoff + extent.
    let mut extent = 0.0f64;
    for a in coords {
        for b in coords {
            extent = extent.max((a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs());
        }
    }
    let cells = cell.cell_translations(cutoff + extent + 1.0);
    let rcarts: Vec<[f64; 3]> = cells.iter().map(|&n| cell.rvec(n)).collect();
    let c2 = cutoff * cutoff;
    let mut pair_ij = Vec::new();
    let mut r_off = vec![0i32];
    let mut rvecs = Vec::new();
    let mut dists = Vec::new();
    for i in 0..n {
        for j in i..n {
            let mut cnt = 0i32;
            for rv in &rcarts {
                if i == j && rv[0].abs() < 1e-12 && rv[1].abs() < 1e-12 && rv[2].abs() < 1e-12 {
                    if same_cell_self {
                        // include the R=0 self slot explicitly
                        rvecs.push([0.0, 0.0, 0.0]);
                        dists.push(0.0);
                        cnt += 1;
                    }
                    continue;
                }
                let dx = coords[j][0] + rv[0] - coords[i][0];
                let dy = coords[j][1] + rv[1] - coords[i][1];
                let dz = coords[j][2] + rv[2] - coords[i][2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 <= c2 {
                    rvecs.push([rv[0] as f32, rv[1] as f32, rv[2] as f32]);
                    dists.push(d2.sqrt() as f32);
                    cnt += 1;
                }
            }
            if cnt > 0 || emit_empty {
                pair_ij.push((i as u16, j as u16));
                r_off.push(r_off.last().unwrap() + cnt);
            }
        }
    }
    ImagePairList { pair_ij, r_off, rvecs, dists }
}

// ------------------------------------------------------------------
// SK image-pair enumeration → slot + fold tables (GPU layouts)
// ------------------------------------------------------------------

/// One (oriented-pair, image-cell) SK eval slot — mirrors CL `ImgSlot`.
/// Eval displacement: d = r_oj + R − r_oi  (image is always the oj side).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuImgSlot {
    pub oi: u16,
    pub oj: u16,
    pub cell: i32,
}
unsafe impl ocl::OclPrm for GpuImgSlot {}

/// One fold out-pair — mirrors CL `FoldPair`. Block
/// H(k)[row_orb.., col_orb..] = Σ_slots e^{ikR}·blk; `herm` also writes
/// the conj-transpose at [col,row]; `diag` adds onsite(H) + I(S).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuFoldPair {
    pub row_orb: u16,
    pub col_orb: u16,
    pub nrow: u8,
    pub ncol: u8,
    pub diag: u8,
    pub herm: u8,
    pub slot_off: i32,
    pub slot_cnt: i32,
    pub pad: i32,
}
unsafe impl ocl::OclPrm for GpuFoldPair {}

/// One bucket of the image-pair list: uniform block geometry
/// (norb_oj × norb_oi) and one SK table — one `assemble_pairs_img` and
/// one `kpoint_phase_sum_batched` launch each.
#[derive(Debug)]
pub struct SkImgBucket {
    pub block_type: u8,
    pub norb_oi: usize,
    pub norb_oj: usize,
    pub slots: Vec<GpuImgSlot>,
    pub outs: Vec<GpuFoldPair>,
    /// Index into the caller's SK-table array for this species pair.
    pub sk_table_idx: usize,
}

/// Enumerate SK image pairs with the fold structure.
///
/// Iterates unordered central-atom pairs (a ≤ b):
///   - a == b: diagonal out-pair (always emitted — onsite + I must be
///     written even with no slots), all R ≠ 0 within cutoff, `diag=1`.
///   - a < b: oriented eval pair (oi, oj) from `orient` (s-atom first
///     for mixed s-p, (a,b) for same-type); slots = cells R with
///     |r_oj + R − r_oi| < cutoff(oi,oj) + margin; out-pair
///     (row=oj, col=oi) with `herm=1` covering the conj side.
///
/// `cells`/`rcarts` = the shared integer/Cartesian cell table (built
/// once by the caller at the largest cutoff + extent) — slot.cell
/// indexes it directly. Callbacks take ATOM indices:
/// `orient(a,b)→(oi,oj)`, `cutoff(oi,oj)→Bohr`, `sk_table(oi,oj)→idx`,
/// `bucket_id(oi,oj)→bucket` (0..n_buckets). Bucket `n_buckets-1` is
/// reserved for diagonal (a==a) out-pairs? No — diagonal pairs go into
/// the same-species bucket (block geometry matches (norb_a × norb_a)).
pub fn enumerate_sk_pairs(
    coords: &[[f64; 3]],
    cell: &PbcCell,
    cells: &[[i32; 3]],
    rcarts: &[[f64; 3]],
    atom_n_orb: &[u8],
    atom_orb_off: &[u16],
    margin: f64,
    orient: &dyn Fn(usize, usize) -> (usize, usize),
    cutoff: &dyn Fn(usize, usize) -> f64,
    sk_table: &dyn Fn(usize, usize) -> usize,
    bucket_id: &dyn Fn(usize, usize) -> usize,
    n_buckets: usize,
) -> Vec<SkImgBucket> {
    let n = coords.len();
    let mut out: Vec<SkImgBucket> = (0..n_buckets)
        .map(|_| SkImgBucket {
            block_type: 0, norb_oi: 0, norb_oj: 0,
            slots: Vec::new(), outs: Vec::new(), sk_table_idx: usize::MAX,
        })
        .collect();
    let _ = cell;
    for a in 0..n {
        for b in a..n {
            let (oi, oj) = if a == b { (a, b) } else { orient(a, b) };
            let cut = cutoff(oi, oj) + margin;
            let c2 = cut * cut;
            let bi = bucket_id(oi, oj);
            let bk = &mut out[bi];
            let slot_off = bk.slots.len() as i32;
            let first = bk.outs.is_empty(); // bucket metadata set on 1st out-pair
            let mut cnt = 0i32;
            for (ci, rv) in rcarts.iter().enumerate() {
                if a == b && cells[ci] == [0, 0, 0] {
                    continue;
                }
                let dx = coords[oj][0] + rv[0] - coords[oi][0];
                let dy = coords[oj][1] + rv[1] - coords[oi][1];
                let dz = coords[oj][2] + rv[2] - coords[oi][2];
                if dx * dx + dy * dy + dz * dz <= c2 {
                    bk.slots.push(GpuImgSlot { oi: oi as u16, oj: oj as u16, cell: ci as i32 });
                    cnt += 1;
                }
            }
            if cnt > 0 || a == b {
                let bt = match (atom_n_orb[oi], atom_n_orb[oj]) {
                    (1, 1) => 0u8,
                    (1, 4) | (4, 1) => 1u8,
                    (4, 4) => 2u8,
                    _ => panic!(
                        "enumerate_sk_pairs: unsupported orbital block {}x{} (atoms {oi},{oj})",
                        atom_n_orb[oi], atom_n_orb[oj]
                    ),
                };
                if first {
                    bk.block_type = bt;
                    bk.norb_oi = atom_n_orb[oi] as usize;
                    bk.norb_oj = atom_n_orb[oj] as usize;
                    bk.sk_table_idx = sk_table(oi, oj);
                }
                bk.outs.push(GpuFoldPair {
                    row_orb: atom_orb_off[oj],
                    col_orb: atom_orb_off[oi],
                    nrow: atom_n_orb[oj],
                    ncol: atom_n_orb[oi],
                    diag: (a == b) as u8,
                    herm: (a != b) as u8,
                    slot_off,
                    slot_cnt: cnt,
                    pad: 0,
                });
            }
        }
    }
    out.retain(|b| !b.slots.is_empty() || !b.outs.is_empty());
    out
}
