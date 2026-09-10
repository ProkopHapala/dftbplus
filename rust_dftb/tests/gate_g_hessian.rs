//! Gate G: same-geometry Hessian parity (manifest §5 / review G1.5 / tasks D4).
//!
//! Frozen coordinates = sparse Gate F minimum (not each method's own min — that is Gate H).
//! Hessian is a **3-point FD of analytic forces**, not FD of energy:
//!   H[:, a] = −(F(R + h e_a) − F(R − h e_a)) / (2h)
//! Columns are independent. Do **not** write H[a, j] = H[j, a] and then claim
//! asymmetry is zero (Gate E tautology). η_asym = ||H − Hᵀ||_F / ||H||_F from H_raw.
//!
//! +h and −h each warm-start from the **same center q**, never −h from +h.
//! h = 0.01 Å (manifest also lists 0.02/0.05/0.10; h_ref is not in a sweep).
//!
//! Dense reference: same valence q0 and `compute_forces_from_dw` as G3.3.
//! Does not touch `qmqm/gpu_forces.cl`. No FIRE.

use nalgebra::{DMatrix, SymmetricEigen};
use rust_dftb::methods::dftb::forces::{compute_forces_from_dw, Forces};
use rust_dftb::methods::dftb::gamma::GammaTable;
use rust_dftb::methods::sparse::harness::{require_sih_sk_dir, require_sparse_gpu};
use rust_dftb::methods::sparse::SparseDftb;
use rust_dftb::qmqm::{Fragment, FragmentNeighborList, FragmentTemplate, MultiSystemSolver, SimpleMixer};
use rust_dftb::{load_sk_for_species, SkData};
use std::io::Write;

/// Sparse Gate F FIRE minimum (NVIDIA, 2026-09-10, **SparseDftb**). Si–H mean 1.477 Å, |F|≈9e-4.
fn gate_f_sih4_coords() -> Vec<[f64; 3]> {
    vec![
        [-0.000259,  0.302610, -0.000330],
        [ 1.463846,  0.107836, -0.013743],
        [-0.313014,  1.710702,  0.319181],
        [-0.601402, -0.575463,  1.023930],
        [-0.549074, -0.037178, -1.329038],
    ]
}

fn sih_bonds(coords: &[[f64; 3]]) -> Vec<f64> {
    let si = coords[0];
    (1..coords.len()).map(|i| {
        let d = [coords[i][0] - si[0], coords[i][1] - si[1], coords[i][2] - si[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }).collect()
}

fn f_norm(forces: &[[f64; 3]]) -> f64 {
    forces.iter().map(|f| f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sum::<f64>().sqrt()
}

fn flatten_f(f: &[[f64; 3]]) -> Vec<f64> {
    f.iter().flat_map(|a| a.iter().copied()).collect()
}

/// cm⁻¹ per √(Ha/Å²/amu). √(E_h / (Å² amu)) / (2πc).
const HA_ANG_AMU_TO_CM: f64 = 2720.211;

struct DenseSccRef {
    q: Vec<f64>,
    density: DMatrix<f64>,
    edm: DMatrix<f64>,
    n_iter: usize,
}

fn dense_scc_valence(
    sk: &SkData,
    species: &[String],
    coords: &[[f64; 3]],
    q0: &[f64],
    q_init: Option<&[f64]>,
    mix: f64,
    tol: f64,
    max_iter: usize,
) -> DenseSccRef {
    let mut tmpl = FragmentTemplate::new(sk, species.to_vec(), coords.to_vec())
        .unwrap_or_else(|e| panic!("Gate G dense template at coords {coords:?}: {e}"));
    tmpl.q0 = q0.to_vec();
    let mut frag = Fragment::from_template(tmpl, coords.to_vec());
    if let Some(q) = q_init {
        assert_eq!(q.len(), frag.charges.len(), "Gate G dense q_init len {} != n_atom {}", q.len(), frag.charges.len());
        frag.charges.copy_from_slice(q);
    }
    let gamma = GammaTable::from_sk_data(sk, species).unwrap_or_else(|e| panic!("Gate G dense gamma: {e}"));
    let n = coords.len() as f64;
    let centroid = coords.iter().fold([0.0; 3], |acc, c| [acc[0] + c[0], acc[1] + c[1], acc[2] + c[2]]);
    let frag_neighbors = FragmentNeighborList::build(&vec![[centroid[0] / n, centroid[1] / n, centroid[2] / n]], 10.0);
    let mixer = SimpleMixer::new(mix);
    let mut solver = MultiSystemSolver::new(vec![frag], frag_neighbors, gamma, mixer);
    if let Some(q) = q_init {
        solver.charges.copy_from_slice(q);
        solver.scatter_charges();
    }
    solver.solve_scc(max_iter, tol).unwrap_or_else(|e| panic!(
        "Gate G dense valence SCC failed: {e}  |r| mean={:.4} Å",
        sih_bonds(coords).iter().sum::<f64>() / 4.0
    ));
    let frag = &solver.fragments[0];
    let n_occ = (frag.n_electrons / 2.0).round() as usize;
    let c_occ = frag.eigenvectors.columns(0, n_occ);
    let density = &c_occ * c_occ.transpose() * 2.0;
    let mut edm = DMatrix::<f64>::zeros(frag.template.n_orbs, frag.template.n_orbs);
    for k in 0..n_occ {
        let c = frag.eigenvectors.column(k);
        let eps = frag.eigenvalues[k];
        edm += (2.0 * eps) * (&c * c.transpose());
    }
    DenseSccRef { q: frag.charges.clone(), density, edm, n_iter: solver.n_scc_iter }
}

fn dense_forces(
    sk: &SkData,
    sk_dir: &str,
    species: &[String],
    coords: &[[f64; 3]],
    q0: &[f64],
    q_init: Option<&[f64]>,
) -> (DenseSccRef, Forces) {
    let scc = dense_scc_valence(sk, species, coords, q0, q_init, 0.5, 1e-5, 80);
    let f = compute_forces_from_dw(sk, species, coords, &scc.density, &scc.edm, &scc.q, q0, sk_dir)
        .unwrap_or_else(|e| panic!("Gate G dense force failed: {e}"));
    (scc, f)
}

/// Unsymmetrized force Hessian. Column `a` from ±h on dof `a` only. Never copies the triangle.
fn force_hessian<E>(coords: &[[f64; 3]], h: f64, mut eval: E) -> DMatrix<f64>
where
    E: FnMut(&[[f64; 3]], usize) -> Vec<[f64; 3]>,
{
    let n_atom = coords.len();
    let n_dof = 3 * n_atom;
    let mut hess = DMatrix::<f64>::zeros(n_dof, n_dof);
    for a in 0..n_dof {
        let atom = a / 3;
        let xyz = a % 3;
        let mut rp = coords.to_vec();
        let mut rm = coords.to_vec();
        rp[atom][xyz] += h;
        rm[atom][xyz] -= h;
        let fp = eval(&rp, a);
        let fm = eval(&rm, a);
        assert_eq!(fp.len(), n_atom);
        assert_eq!(fm.len(), n_atom);
        let fp_f = flatten_f(&fp);
        let fm_f = flatten_f(&fm);
        for j in 0..n_dof {
            let hij = -(fp_f[j] - fm_f[j]) / (2.0 * h);
            assert!(hij.is_finite(), "Gate G: non-finite H[{j},{a}]={hij}  atom={atom} xyz={xyz} h={h}");
            hess[(j, a)] = hij;
        }
        let col_rms = (0..n_dof).map(|j| hess[(j, a)] * hess[(j, a)]).sum::<f64>().sqrt() / (n_dof as f64).sqrt();
        eprintln!(
            "    col {a:2} atom {atom} xyz {xyz}  |F+|={:.3e}  |F-|={:.3e}  H_col_rms={col_rms:.4e}",
            f_norm(&fp), f_norm(&fm)
        );
    }
    hess
}

fn eta_asym(h: &DMatrix<f64>) -> f64 {
    let d = h - h.transpose();
    d.norm() / h.norm().max(1e-30)
}

fn h_sym(h: &DMatrix<f64>) -> DMatrix<f64> {
    0.5 * (h + h.transpose())
}

fn eigs_of(h: &DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
    let se = SymmetricEigen::new(h.clone());
    let mut idx: Vec<usize> = (0..se.eigenvalues.len()).collect();
    idx.sort_by(|&a, &b| se.eigenvalues[a].partial_cmp(&se.eigenvalues[b]).unwrap());
    let evals: Vec<f64> = idx.iter().map(|&i| se.eigenvalues[i]).collect();
    let mut vecs = DMatrix::<f64>::zeros(h.nrows(), h.ncols());
    for (c, &i) in idx.iter().enumerate() {
        vecs.set_column(c, &se.eigenvectors.column(i));
    }
    (evals, vecs)
}

fn mac(v: &DMatrix<f64>, w: &DMatrix<f64>, i: usize, j: usize) -> f64 {
    let a = v.column(i);
    let b = w.column(j);
    let num = a.dot(&b).powi(2);
    let den = a.dot(&a) * b.dot(&b);
    if den < 1e-30 { 0.0 } else { num / den }
}

fn subspace_overlap(v: &DMatrix<f64>, w: &DMatrix<f64>, i0: usize, i1: usize) -> f64 {
    let k = i1 - i0;
    if k == 0 { return 0.0; }
    let mut s = 0.0f64;
    for i in i0..i1 {
        for j in i0..i1 {
            s += v.column(i).dot(&w.column(j)).powi(2);
        }
    }
    s / k as f64
}

fn lambda_to_cm(l: f64) -> f64 {
    let mag = HA_ANG_AMU_TO_CM * l.abs().sqrt();
    if l < 0.0 { -mag } else { mag }
}

fn mass_weighted(h: &DMatrix<f64>, masses: &[f64]) -> DMatrix<f64> {
    let n = h.nrows();
    let mut mw = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            mw[(i, j)] = h[(i, j)] / (masses[i / 3] * masses[j / 3]).sqrt();
        }
    }
    mw
}

fn write_hess_csv(path: &std::path::Path, h: &DMatrix<f64>, label: &str) {
    let mut f = std::fs::File::create(path).unwrap_or_else(|e| panic!("Gate G: cannot write {}: {e}", path.display()));
    writeln!(f, "# Gate G {label} unsymmetrized force Hessian (Ha/Å²), row-major, n={}", h.nrows()).unwrap();
    for i in 0..h.nrows() {
        let row: Vec<String> = (0..h.ncols()).map(|j| format!("{:.10e}", h[(i, j)])).collect();
        writeln!(f, "{}", row.join(",")).unwrap();
    }
}

#[test]
fn test_gate_g_sih4_force_hessian_parity() {
    std::env::set_var("RUST_DFTB_SPARSE_ALGEBRA_VERBOSE", "0");
    let Some(_gpu) = require_sparse_gpu() else { return };
    let sk_dir = require_sih_sk_dir();
    let species = vec!["Si".into(), "H".into(), "H".into(), "H".into(), "H".into()];
    let coords = gate_f_sih4_coords();
    let q0 = vec![4.0, 1.0, 1.0, 1.0, 1.0];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let masses = [28.0855, 1.00784, 1.00784, 1.00784, 1.00784];
    let h = 0.01f64;
    let n_atom = species.len();
    let n_dof = 3 * n_atom;

    let bonds = sih_bonds(&coords);
    eprintln!("=== Gate G: unsymmetrized force Hessian at sparse Gate F geometry [SparseDftb] ===");
    eprintln!("  h = {h} Å  (FD of analytic F; not FD of E; H_raw not filled symmetrically)");
    eprintln!("  Si–H = {:.4} {:.4} {:.4} {:.4} Å", bonds[0], bonds[1], bonds[2], bonds[3]);
    for (k, &r) in bonds.iter().enumerate() {
        assert!((1.40..=1.55).contains(&r), "Gate G: frozen Si–H{} = {r:.4} Å is not the Gate F window", k + 1);
    }

    let mut eng = SparseDftb::new(sk.clone(), &sk_dir, species.clone(), coords.clone())
        .unwrap_or_else(|e| panic!("Gate G SparseDftb::new: {e}"));
    eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("Gate G sparse center SCC: {e}"));
    let e0 = eng.last_energy().clone();
    let f0 = eng.forces().unwrap_or_else(|e| panic!("Gate G sparse center forces: {e}"));
    let q_center_sp = e0.q.clone();
    let fn0_sp = f_norm(&f0.forces);
    eprintln!("  sparse center: E_tot={:.10}  E_el={:.10}  E_rep={:.10}  |F|={fn0_sp:.3e}  n_scc={}  Tr(KS)={:.6}  r_scc={:.3e}  R_H={:.3e}",
        e0.e_tot, e0.e_el, e0.e_rep, e0.n_scc, e0.tr_ks, e0.r_scc, e0.r_h);
    assert!(e0.e_rep.abs() > 1e-4, "Gate G: E_rep={:.3e} ~0 — Spline missing", e0.e_rep);
    assert!(fn0_sp < 2e-3, "Gate G: frozen geometry is not near a sparse stationary point: |F|={fn0_sp:.3e}");

    let (d0, fd0) = dense_forces(&sk, &sk_dir, &species, &coords, &q0, None);
    let q_center_dn = d0.q.clone();
    let fn0_dn = f_norm(&fd0.forces);
    eprintln!("  dense  center: |F|={fn0_dn:.3e}  n_scc={}  (same coords; Gate H compares own minima)", d0.n_iter);
    let mut max_df0 = 0.0f64;
    for i in 0..n_atom {
        for c in 0..3 {
            max_df0 = max_df0.max((f0.forces[i][c] - fd0.forces[i][c]).abs());
        }
    }
    eprintln!("  center max|F_sp − F_dn|={max_df0:.3e}");

    eprintln!("  --- sparse Hessian (30 analytic-F evals, SparseDftb, warm-start from center q) ---");
    let h_sp = force_hessian(&coords, h, |xyz, a| {
        eng.set_q(&q_center_sp).unwrap_or_else(|e| panic!("Gate G set_q dof {a}: {e}"));
        eng.set_coords(xyz).unwrap_or_else(|e| panic!("Gate G set_coords dof {a}: {e}"));
        eng.scc(80, 1e-5).unwrap_or_else(|err| panic!("Gate G sparse SCC at dof {a}: {err}"));
        eng.forces().unwrap_or_else(|err| panic!("Gate G sparse F at dof {a}: {err}")).forces
    });

    eprintln!("  --- dense Hessian (30 analytic-F evals, warm-start from center q) ---");
    let h_dn = force_hessian(&coords, h, |xyz, _a| {
        let (_scc, f) = dense_forces(&sk, &sk_dir, &species, xyz, &q0, Some(&q_center_dn));
        f.forces
    });

    let eta_sp = eta_asym(&h_sp);
    let eta_dn = eta_asym(&h_dn);
    let mut max_abs_sp = 0.0f64;
    let mut max_abs_dn = 0.0f64;
    let mut max_dh = 0.0f64;
    let mut worst = (0usize, 0usize);
    let mut fro_dh = 0.0f64;
    for i in 0..n_dof {
        for j in 0..n_dof {
            let a = h_sp[(i, j)].abs();
            let b = h_dn[(i, j)].abs();
            max_abs_sp = max_abs_sp.max(a);
            max_abs_dn = max_abs_dn.max(b);
            let d = (h_sp[(i, j)] - h_dn[(i, j)]).abs();
            fro_dh += d * d;
            if d > max_dh {
                max_dh = d;
                worst = (i, j);
            }
        }
    }
    let fro_dh = fro_dh.sqrt();
    let rel_f = fro_dh / h_dn.norm().max(1e-30);
    let rel_max = max_dh / max_abs_dn.max(1e-30);
    eprintln!("  η_asym sparse={eta_sp:.4e}  dense={eta_dn:.4e}  (from H_raw; not after filling H[j,i]=H[i,j])");
    eprintln!("  ||H_sp||_F={:.4e}  ||H_dn||_F={:.4e}  max|H_sp|={max_abs_sp:.4e}  max|H_dn|={max_abs_dn:.4e}", h_sp.norm(), h_dn.norm());
    eprintln!("  max|ΔH|={max_dh:.4e}  at ({},{})  rel_max={rel_max:.3e}  ||ΔH||_F/||H_dn||_F={rel_f:.3e}", worst.0, worst.1);
    // Translation null space of H_raw (any geometry). Finite |F| does not excuse ||Ht||.
    let nt = (n_atom as f64).sqrt();
    for axis in 0..3 {
        let mut nsp = 0.0f64;
        let mut ndn = 0.0f64;
        for row in 0..n_dof {
            let mut asp = 0.0f64;
            let mut adn = 0.0f64;
            for i in 0..n_atom {
                asp += h_sp[(row, i * 3 + axis)] / nt;
                adn += h_dn[(row, i * 3 + axis)] / nt;
            }
            nsp += asp * asp;
            ndn += adn * adn;
        }
        eprintln!("  ||H t_{axis}|| sparse={:.4e}  dense={:.4e}  (unit translation; study vs h/SCC)", nsp.sqrt(), ndn.sqrt());
    }

    let out_dir = {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../debug/sparse_review");
        std::fs::create_dir_all(&p).unwrap_or_else(|e| panic!("Gate G: cannot create {p:?}: {e}"));
        p
    };
    write_hess_csv(&out_dir.join("gate_g_h_sparse.csv"), &h_sp, "sparse");
    write_hess_csv(&out_dir.join("gate_g_h_dense.csv"), &h_dn, "dense");
    eprintln!("REVIEW: {}", out_dir.join("gate_g_h_sparse.csv").display());
    eprintln!("REVIEW: {}", out_dir.join("gate_g_h_dense.csv").display());

    let hs_sp = h_sym(&h_sp);
    let hs_dn = h_sym(&h_dn);
    let (ev_sp, vec_sp) = eigs_of(&hs_sp);
    let (ev_dn, vec_dn) = eigs_of(&hs_dn);
    eprintln!("  Cartesian H_sym eigenvalues (Ha/Å²), sorted — diagnostic, not a pass criterion:");
    eprintln!("    idx        λ_sp          λ_dn         Δλ       MAC");
    for i in 0..n_dof {
        let m = mac(&vec_sp, &vec_dn, i, i);
        eprintln!(
            "    {i:3}  {:+12.4e}  {:+12.4e}  {:+10.2e}  {m:.4}",
            ev_sp[i], ev_dn[i], ev_sp[i] - ev_dn[i]
        );
    }

    // Degenerate clusters: consecutive |Δλ|/max(|λ|,1e-4) < 0.05
    let mut i0 = 0usize;
    while i0 < n_dof {
        let mut i1 = i0 + 1;
        while i1 < n_dof {
            let scale = ev_dn[i1 - 1].abs().max(ev_dn[i1].abs()).max(1e-4);
            if (ev_dn[i1] - ev_dn[i1 - 1]).abs() / scale < 0.05 { i1 += 1; } else { break; }
        }
        if i1 - i0 >= 2 {
            let ov = subspace_overlap(&vec_sp, &vec_dn, i0, i1);
            eprintln!("  subspace [{i0}..{i1}) overlap={ov:.4}  (near-degenerate; do not pair by index)");
        }
        i0 = i1;
    }

    let mw_sp = mass_weighted(&hs_sp, &masses);
    let mw_dn = mass_weighted(&hs_dn, &masses);
    let (lm_sp, _) = eigs_of(&mw_sp);
    let (lm_dn, _) = eigs_of(&mw_dn);
    eprintln!("  mass-weighted frequencies (cm⁻¹; imaginary printed negative):");
    eprintln!("    idx     sparse      dense      Δν");
    for i in 0..n_dof {
        let a = lambda_to_cm(lm_sp[i]);
        let b = lambda_to_cm(lm_dn[i]);
        eprintln!("    {i:3}  {a:9.1}  {b:9.1}  {:+8.1}", a - b);
    }
    eprintln!("  six smallest |λ| of H_sym (rigid leakage diagnostic, not 'n_unstable ≤ 6'):");
    let mut abs_sp: Vec<(usize, f64)> = ev_sp.iter().enumerate().map(|(i, l)| (i, l.abs())).collect();
    abs_sp.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for k in 0..6.min(n_dof) {
        let i = abs_sp[k].0;
        eprintln!("    sparse λ[{i}]={:+.4e}   dense λ[{i}]={:+.4e}", ev_sp[i], ev_dn[i]);
    }

    // Honest asserts. Keep red if the physics disagrees. Do not symmetrize to kill η_asym.
    assert!(eta_sp.is_finite() && eta_dn.is_finite());
    assert!(
        eta_sp < 0.05,
        "Gate G: sparse η_asym={eta_sp:.3e} — H_raw is not approximately symmetric (analytic F is not a gradient, or FD/SCC noise). Do not fix by copying H[j,i]=H[i,j]."
    );
    assert!(
        eta_dn < 0.05,
        "Gate G: dense η_asym={eta_dn:.3e} — same check on the f64 reference Hessian"
    );
    // Manifest: <5% for ordinary Hessian / modes at identical geometry.
    assert!(
        rel_f < 0.05 || max_dh < 1e-3,
        "Gate G: sparse vs dense Hessian  ||ΔH||_F/||H||_F={rel_f:.3e}  max|ΔH|={max_dh:.3e} (target 5% rel or 1e-3 abs). worst=({},{})",
        worst.0, worst.1
    );
    eprintln!("  Gate G: H_raw unsymmetrized; η_asym and ||ΔH|| printed. Awaiting USER confirmation; not marked done.");
}
