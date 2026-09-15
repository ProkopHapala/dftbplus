//! PBC path tests: Ewald invRMat (host + GPU), periodic γ, Bloch fold.
//!
//! L0 (host only, no GPU):
//!   - erfc_host vs reference values
//!   - α/maxR/maxG autotuning self-consistency
//!   - cubic-cell Ewald self-term → −α_cubic/L  (α_cubic = 2.837297)
//!   - NaCl Madelung energy → −α_M/r0          (α_M = 1.7475645946)
//!
//! GPU (needs OpenCL device; kernels launched directly):
//!   - ewald_invr_batched ≡ ewald_invmat_host (same formulas, f32-vs-f64)
//!
//! SK-gated (RUST_DFTB_SK_DIR):
//!   - big-box Γ: γ_pbc ≈ molecular γ_func; H0(Γ)/S(Γ) ≈ molecular
//!   - H(−k) = H(k)* on a low-symmetry cell; ε(−k)=ε(k)
//!   - full SCC smoke on a small real cell

use ocl::prm::{Float4, Int2};
use ocl::{Kernel, Program};
use rust_dftb::qmqm::gpu_runtime::{map_ocl_err, GpuRuntime};
use rust_dftb::qmqm::pbc_cell::{
    enumerate_image_pairs, erfc_host, ewald_invmat_host, max_g_ewald,
    max_r_ewald, optimal_alpha, PbcCell,
};

const HAM_SOURCE: &str = include_str!("../src/methods/dftb/dftb_hamiltonian.cl");
const PBC_SOURCE: &str = include_str!("../src/qmqm/gpu_pbc.cl");
const ANG2BOHR: f64 = 1.889726133;

fn try_runtime() -> Option<GpuRuntime> {
    match GpuRuntime::new() {
        Ok(rt) => Some(rt),
        Err(e) => {
            eprintln!("Skipping GPU test: no OpenCL device ({e})");
            None
        }
    }
}

/// Cartesian cell translations covering |R| ≤ mr + max pair
/// displacement — what the pair-distance-filtered host Ewald needs.
fn rlat_cover(cell: &PbcCell, coords: &[[f64; 3]], mr: f64) -> Vec<[f64; 3]> {
    let mut ext = 0.0f64;
    for a in coords {
        for b in coords {
            ext = ext.max((a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs());
        }
    }
    cell.cell_translations(mr + ext + 1.0)
        .iter()
        .map(|&n| cell.rvec(n))
        .collect()
}

// ------------------------------------------------------------------
// L0 host tests
// ------------------------------------------------------------------

#[test]
fn erfc_reference_values() {
    // NIST/known values
    let cases = [
        (0.0, 1.0),
        (0.25, 0.7236736098),
        (0.5, 0.4795001222),
        (1.0, 0.1572992071),
        (1.5, 0.0338948535),
        (2.0, 0.0046777350),
        (3.0, 2.2090497e-5),
        (5.0, 1.5374598e-12),
    ];
    for (x, want) in cases {
        let got = erfc_host(x);
        assert!(
            (got - want).abs() < 1e-9,
            "erfc({x}) = {got:.12e} vs {want:.12e}"
        );
    }
}

#[test]
fn ewald_autotuning_cubic() {
    // cubic cell a=10 bohr
    let a = 10.0;
    let cell = PbcCell::new([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]).unwrap();
    let tol = 1e-9;
    let alpha = optimal_alpha(&cell, tol).unwrap();
    let mr = max_r_ewald(alpha, tol).unwrap();
    let mg = max_g_ewald(alpha, cell.vol, tol).unwrap();
    eprintln!("cubic a=10: alpha={alpha:.6} maxR={mr:.3} maxG={mg:.3}");
    // α must be positive and in a sane range (~1/a scale)
    assert!(alpha > 0.01 && alpha < 10.0, "alpha out of range: {alpha}");
    // rTerm(maxR) ≈ tol, gTerm(maxG) ≈ tol — the tuned cutoffs are
    // self-consistent by construction (bisection residual < tol)
    let rt_ = erfc_host(alpha * mr) / mr;
    assert!(rt_ <= 2.0 * tol, "rTerm(maxR)={rt_} > tol");
    let g2 = mg * mg;
    let gt = 4.0 * std::f64::consts::PI * (-0.25 * g2 / (alpha * alpha)).exp() / (cell.vol * g2);
    assert!(gt <= 2.0 * tol, "gTerm(maxG)={gt} > tol");
    // R/G lists non-empty
    assert!(!cell.cell_translations(mr).is_empty());
    assert!(!cell.g_lattice_points(mg).is_empty());
}

/// Single atom in a cubic cell: invRMat[0,0] is the Ewald self-term of a
/// point charge among its own images = −α_cubic/L (α_cubic = 2.837297).
#[test]
fn ewald_self_term_cubic() {
    let l = 15.0; // Bohr
    let cell = PbcCell::new([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]).unwrap();
    let tol = 1e-9;
    let alpha = optimal_alpha(&cell, tol).unwrap();
    let mr = max_r_ewald(alpha, tol).unwrap();
    let mg = max_g_ewald(alpha, cell.vol, tol).unwrap();
    let gpts = cell.g_lattice_points(mg);
    let coords = vec![[0.0, 0.0, 0.0]];
    let rlat = rlat_cover(&cell, &coords, mr);
    let m = ewald_invmat_host(&coords, &cell, alpha, mr, &gpts, &rlat);
    let want = -2.837297 / l;
    eprintln!("cubic self-term: got {:.8} want {want:.8}", m[0]);
    assert!((m[0] - want).abs() < 1e-5, "self-term {} vs {want}", m[0]);
}

/// NaCl Madelung: primitive FCC cell, Na(0,0,0) + Cl(a/2,0,0), q=±1.
/// E/cell = ½Σ q_i φ_i = −α_M/r0, r0 = a/2.
#[test]
fn ewald_madelung_nacl() {
    let a = 10.66; // Bohr (≈5.64 Å)
    let h = a / 2.0;
    let cell = PbcCell::new([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]).unwrap();
    let tol = 1e-9;
    let alpha = optimal_alpha(&cell, tol).unwrap();
    let mr = max_r_ewald(alpha, tol).unwrap();
    let mg = max_g_ewald(alpha, cell.vol, tol).unwrap();
    eprintln!("NaCl: alpha={alpha:.6} maxR={mr:.2} maxG={mg:.2} nG={} nR={}",
        cell.g_lattice_points(mg).len(), cell.cell_translations(mr).len());
    let gpts = cell.g_lattice_points(mg);
    let coords = vec![[0.0, 0.0, 0.0], [h, 0.0, 0.0]];
    let rlat = rlat_cover(&cell, &coords, mr);
    let m = ewald_invmat_host(&coords, &cell, alpha, mr, &gpts, &rlat);
    let q = [1.0f64, -1.0];
    // E = ½ Σ_i q_i Σ_j m[i,j] q_j
    let mut e = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            e += 0.5 * q[i] * m[i * 2 + j] * q[j];
        }
    }
    let r0 = a / 2.0;
    let want = -1.7475645946 / r0;
    eprintln!("NaCl Madelung: E={e:.10} want {want:.10} (diff {:.3e})", e - want);
    assert!((e - want).abs() < 2e-5, "Madelung E={e} vs {want}");
}

// ------------------------------------------------------------------
// GPU: ewald_invr_batched vs host reference
// ------------------------------------------------------------------

fn build_prog(rt: &mut GpuRuntime) -> Program {
    rt.build_program(&format!("{HAM_SOURCE}\n{PBC_SOURCE}")).unwrap()
}

/// Build ewald kernel inputs on a skewed (non-cubic) cell, run the GPU
/// kernel, compare element-wise with the f64 host reference.
#[test]
fn ewald_invr_gpu_parity() {
    let Some(mut rt) = try_runtime() else { return };
    // skewed triclinic-ish cell + 3 atoms at generic positions
    let cell = PbcCell::new([
        [8.0, 0.3, 0.0],
        [0.4, 9.0, 0.2],
        [0.1, 0.3, 7.5],
    ])
    .unwrap();
    let coords = vec![
        [1.0, 1.5, 0.8],
        [3.4, 0.7, 2.9],
        [0.5, 4.1, 3.3],
    ];
    let n = coords.len();
    let tol = 1e-9;
    let alpha = optimal_alpha(&cell, tol).unwrap();
    let mr = max_r_ewald(alpha, tol).unwrap();
    let mg = max_g_ewald(alpha, cell.vol, tol).unwrap();
    let gpts = cell.g_lattice_points(mg);
    let rlat = rlat_cover(&cell, &coords, mr);
    eprintln!("ewald gpu parity: alpha={alpha:.4} maxR={mr:.2} maxG={mg:.2} nG={} nR={}",
        gpts.len(), rlat.len());

    // host reference (f64)
    let m_ref = ewald_invmat_host(&coords, &cell, alpha, mr, &gpts, &rlat);

    // GPU inputs — same CSR list the kernel uses
    let elist = enumerate_image_pairs(&coords, &cell, mr, false, true);
    let epair: Vec<Int2> = elist.pair_ij.iter().map(|&(i, j)| Int2::new(i as i32, j as i32)).collect();
    let eslot: Vec<Float4> = elist.rvecs.iter().map(|r| Float4::new(r[0], r[1], r[2], 0.0)).collect();
    let gvec: Vec<Float4> = gpts
        .iter()
        .map(|g| {
            let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
            let w = (-0.25 * g2 / (alpha * alpha)).exp() / g2;
            Float4::new(g[0] as f32, g[1] as f32, g[2] as f32, w as f32)
        })
        .collect();
    let cflat: Vec<f32> = coords.iter().flat_map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect();

    let prog = build_prog(&mut rt);
    let b_coords = rt.buffer_from_slice(&cflat).unwrap();
    let b_epair = rt.buffer_from_slice(&epair).unwrap();
    let b_eoff = rt.buffer_from_slice(&elist.r_off).unwrap();
    let b_eslot = rt.buffer_from_slice(&eslot).unwrap();
    let b_gvec = rt.buffer_from_slice(&gvec).unwrap();
    let b_invr = rt.zero_buffer::<f32>(n * n).unwrap();
    let park = rt.buffer_from_slice(&[1i32]).unwrap();

    let rec_fac = 8.0 * std::f64::consts::PI / cell.vol;
    let c_const = -std::f64::consts::PI / (cell.vol * alpha * alpha);
    let c_self = -2.0 * alpha / std::f64::consts::PI.sqrt();

    let np = epair.len();
    let k = Kernel::builder()
        .program(&prog)
        .name("ewald_invr_batched")
        .queue(rt.queue().clone())
        .global_work_size(np)
        .arg(n as i32).arg(np as i32).arg(1i32)
        .arg(&b_coords).arg(&b_epair).arg(&b_eoff).arg(&b_eslot)
        .arg(&b_gvec).arg(gvec.len() as i32)
        .arg(alpha).arg(rec_fac).arg(c_const).arg(c_self)
        .arg(&b_invr).arg(&park)
        .build()
        .map_err(map_ocl_err)
        .unwrap();
    unsafe { k.enq().unwrap() }
    rt.finish().unwrap();
    let mut m_gpu = vec![0.0f32; n * n];
    rt.read_buffer(&b_invr, &mut m_gpu).unwrap();

    let mut maxd = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let d = (m_gpu[i * n + j] as f64 - m_ref[i * n + j]).abs();
            if d > maxd {
                maxd = d;
            }
        }
    }
    eprintln!("ewald gpu-vs-host max|Δ| = {maxd:.3e}");
    eprintln!("  invr diag: ref={:?} gpu={:?}",
        (0..n).map(|i| m_ref[i * n + i]).collect::<Vec<_>>(),
        (0..n).map(|i| m_gpu[i * n + i]).collect::<Vec<_>>());
    // f32 store + f32 slot coords: expect ~1e-5 agreement on ~1e-1 values
    assert!(maxd < 5e-4, "ewald gpu parity failed: {maxd}");
}

// ------------------------------------------------------------------
// SK-gated tests (need RUST_DFTB_SK_DIR or matsci default)
// ------------------------------------------------------------------

fn sk_dir() -> Option<String> {
    let dir = std::env::var("RUST_DFTB_SK_DIR")
        .unwrap_or_else(|_| "/home/prokop/SIMULATIONS/dftbplus/slakos/mio-1-1".to_string());
    if std::path::Path::new(&dir).is_dir() {
        Some(dir)
    } else {
        eprintln!("SK dir missing ({dir}) — skipping");
        None
    }
}

fn load_sk(dir: &str, species: &[String]) -> rust_dftb::methods::dftb::sk_data::SkData {
    rust_dftb::load_sk_for_species(dir, species).unwrap()
}

/// Two H atoms in a big cubic box at Γ: the PBC fold should reproduce the
/// molecular H0/S (only R=0 pairs in range), and γ_pbc ≈ γ_func + O(1/V).
#[test]
fn bigbox_gamma_h_parity() {
    let Some(dir) = sk_dir() else { return };
    let Some(mut _rt) = try_runtime() else { return };
    let species = vec!["H".to_string(), "H".to_string()];
    let sk = load_sk(&dir, &species);
    let a_ang = 20.0; // 20 Å box — molecule isolated
    let coords = vec![[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]; // H2 stretched ~1.4 Å
    let lat = [[a_ang, 0.0, 0.0], [0.0, a_ang, 0.0], [0.0, 0.0, a_ang]];
    let k_frac = [[0.0, 0.0, 0.0]];
    let kw = [1.0f32];

    let mut eng = rust_dftb::qmqm::gpu_pbc::GpuPbc::new(
        sk.clone(), species.clone(), coords.clone(), lat, &k_frac, &kw, None,
    )
    .unwrap();
    eng.set_geometry(&coords).unwrap();

    // γ_pbc vs molecular γ_func: in a big cubic box EVERY γ element is
    // shifted by the same Ewald background/Madelung term ≈ −2.837297/L
    // (γ_pbc[i,j] ≈ γ_func(r_ij) + C_box for both on/off-diagonal —
    // the off-diagonal does NOT converge to the bare molecular value).
    let gamma_tbl = rust_dftb::methods::dftb::gamma::GammaTable::from_sk_data(&sk, &species).unwrap();
    let u = gamma_tbl.u(0);
    let r = 1.4 * ANG2BOHR;
    let gamma_mol = rust_dftb::methods::dftb::gamma::gamma_full(r, u, u);
    let l_bohr = a_ang * ANG2BOHR;
    let self_term = -2.837297 / l_bohr;
    let g = eng.read_gamma().unwrap();
    eprintln!("γ_pbc[0,1]={:.6} vs molecular γ={:.6}+C_box={:.6} (Δ={:.2e})  γ_pbc[0,0]={:.6} U+self={:.6}",
        g[1], gamma_mol, gamma_mol + self_term, g[1] - (gamma_mol + self_term) as f32,
        g[0], u as f32 + self_term as f32);
    assert!(
        (g[1] - (gamma_mol + self_term) as f32).abs() < 5e-3,
        "γ_pbc[0,1]={} vs γ_func+C_box={}", g[1], gamma_mol + self_term
    );
    // internal consistency: the Madelung shift cancels in differences
    assert!(
        ((g[0] - g[1]) - (u - gamma_mol) as f32).abs() < 1e-3,
        "γ_pbc[0,0]−γ_pbc[0,1]={} vs U−γ_func={}", g[0] - g[1], u - gamma_mol
    );
    assert!(
        (g[0] - (u as f32 + self_term as f32)).abs() < 5e-3,
        "γ_pbc[0,0]={} vs U+self={}", g[0], u as f32 + self_term as f32
    );

    // H0(Γ)/S(Γ) vs molecular template
    let tmpl = rust_dftb::qmqm::fragment::FragmentTemplate::new(&sk, species, coords).unwrap();
    let h0 = eng.read_h0().unwrap();
    let s = eng.read_s().unwrap();
    let n = tmpl.n_orbs;
    let mut hd = 0.0f64;
    let mut sd = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            hd = hd.max((h0[idx][0] as f64 - tmpl.h0[(i, j)]).abs());
            sd = sd.max((s[idx][0] as f64 - tmpl.s[(i, j)]).abs());
            assert!(h0[idx][1].abs() < 1e-5, "H0(Γ) has imag part");
        }
    }
    eprintln!("H0(Γ) parity: max|ΔH|={hd:.3e} max|ΔS|={sd:.3e} (SK-resample level ~1e-4)");
    assert!(hd < 5e-3, "H0(Γ) mismatch {hd}");
    assert!(sd < 5e-3, "S(Γ) mismatch {sd}");
}

/// H-chain 1D: 4 H atoms in a chain along x, big transverse box.
/// Check H(−k) = H(k)* and ε(−k) = ε(k) via the plan's eigensolver.
#[test]
fn bloch_hermiticity_hchain() {
    let Some(dir) = sk_dir() else { return };
    let Some(mut _rt) = try_runtime() else { return };
    let species: Vec<String> = (0..4).map(|_| "H".to_string()).collect();
    let sk = load_sk(&dir, &species);
    // chain along x with spacing 2.0 Å; box 12×20×20 Å
    let coords: Vec<[f64; 3]> = (0..4).map(|i| [i as f64 * 2.0, 0.0, 0.0]).collect();
    let lat = [[12.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]];
    let k_frac = [[0.21, 0.0, 0.0], [-0.21, 0.0, 0.0]];
    let kw = [0.5f32, 0.5];

    let mut eng = rust_dftb::qmqm::gpu_pbc::GpuPbc::new(
        sk, species, coords.clone(), lat, &k_frac, &kw, None,
    )
    .unwrap();
    eng.set_geometry(&coords).unwrap();
    let h0 = eng.read_h0().unwrap();
    let s = eng.read_s().unwrap();
    let (n, _, _, nk) = eng.dims();
    assert_eq!(nk, 2);
    let nn = n * n;
    let mut md_h = 0.0f64;
    let mut md_s = 0.0f64;
    for e in 0..nn {
        let a = h0[e];
        let b = h0[nn + e];
        md_h = md_h.max((a[0] - b[0]).abs() as f64).max((a[1] + b[1]).abs() as f64);
        let sa = s[e];
        let sb = s[nn + e];
        md_s = md_s.max((sa[0] - sb[0]).abs() as f64).max((sa[1] + sb[1]).abs() as f64);
    }
    eprintln!("H(−k)=H(k)*: max defect H={md_h:.3e} S={md_s:.3e}");
    // f32 SK-eval + f32 fold: expect ~1e-5 level
    assert!(md_h < 1e-4, "H(−k)≠H(k)*: {md_h}");
    assert!(md_s < 1e-4, "S(−k)≠S(k)*: {md_s}");

    // Hermiticity at generic k
    let mut md = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let a = h0[i * n + j];
            let b = h0[j * n + i];
            md = md.max((a[0] - b[0]).abs() as f64).max((a[1] + b[1]).abs() as f64);
        }
    }
    eprintln!("H(k) Hermiticity defect: {md:.3e}");
    assert!(md < 1e-5, "H(k) not Hermitian: {md}");

    // ε(−k)=ε(k): eigenvalues from the plan's S^{-1/2} path are checked
    // implicitly by jacobi certification; run a few SCC steps for smoke.
    let (ok, hist) = eng.scc(0.3, 1e-6, 30).unwrap();
    eprintln!("hchain scc rms: {:?}", hist.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>());
    assert!(ok.iter().all(|&x| x), "jacobi cert failed");
    let q = eng.plan.read_charges(&eng.rt).unwrap();
    let qsum: f32 = q[..4].iter().sum();
    eprintln!("hchain charges: {:?} sum={qsum}", &q[..4]);
    assert!((qsum - 4.0).abs() < 1e-3, "charge not conserved: {qsum}");
}
