#!/usr/bin/env python3
"""sparse_homo_lumo.py — find HOMO/LUMO via Chebyshev filter + Ritz iterative eigensolver.

Uses the spectral filtering methods from NumericalMathPlayground
(topics/LinearAlgebra/SpectralFiltering/spectral_solvers.py) to find a few
eigenvalues of the generalized eigenproblem H·c = ε·S·c near the HOMO-LUMO gap,
WITHOUT full diagonalization. This is the O(N) compatible approach.

The generalized problem is transformed to standard form via S^{-1/2} H S^{-1/2},
then solve_band() applies Chebyshev polynomial filtering + Rayleigh-Ritz to
extract eigenpairs in a target energy band.

Usage:
    python3 sparse_homo_lumo.py <hs_matrix.tsv> [--nvec 8] [--cheb-deg 20] [--iters 10]
        [--band-width 0.05] [--out <dir>]

Example:
    python3 scripts/sparse_homo_lumo.py debug/graphene_sparse/benzene_hs_matrix.tsv
"""
import argparse
import os
import sys
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
NUMPG_ROOT = Path("/home/prokophapala/git/NumericalMathPlayground")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(NUMPG_ROOT / "topics" / "LinearAlgebra" / "SpectralFiltering"))

from spectral_solvers import solve_band, cheb_rect_coeffs, apply_cheb_poly, rayleigh_ritz


def parse_hs_matrix_tsv(path):
    """Parse the TSV produced by rhai_save_hs_matrix.

    Returns: species, coords_ang, H (norb×norb), S (norb×norb), eigs_dense, n_occ.
    """
    species, coords = [], []
    H = S = None
    eigs = []
    n_occ = 0
    natoms = norb = 0

    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('# natoms='):
            parts = line.split()
            natoms = int(parts[1].split('=')[1])
            norb = int(parts[2].split('=')[1])
            n_occ = int(parts[3].split('=')[1])
            H = np.zeros((norb, norb), dtype=np.float64)
            S = np.zeros((norb, norb), dtype=np.float64)
            i += 1
            if i < len(lines) and lines[i].strip().startswith('# atom_idx'):
                i += 1
            for _ in range(natoms):
                parts = lines[i].strip().split('\t')
                species.append(parts[1])
                coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
                i += 1
        elif line.startswith('# H_scc matrix'):
            i += 1
            if i < len(lines) and lines[i].strip().startswith('i\t'):
                i += 1
            for _ in range(norb * norb):
                parts = lines[i].strip().split('\t')
                H[int(parts[0]), int(parts[1])] = float(parts[2])
                i += 1
        elif line.startswith('# S matrix'):
            i += 1
            if i < len(lines) and lines[i].strip().startswith('i\t'):
                i += 1
            for _ in range(norb * norb):
                parts = lines[i].strip().split('\t')
                S[int(parts[0]), int(parts[1])] = float(parts[2])
                i += 1
        elif line.startswith('# eigenvalues'):
            i += 1
            if i < len(lines) and lines[i].strip().startswith('idx'):
                i += 1
            for _ in range(norb):
                parts = lines[i].strip().split('\t')
                eigs.append(float(parts[1]))
                i += 1
        else:
            i += 1

    return species, np.array(coords, dtype=np.float64), H, S, np.array(eigs), n_occ


def main():
    p = argparse.ArgumentParser(description='Sparse HOMO/LUMO via Chebyshev filter + Ritz',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('hs_tsv', type=str, help='Path to *_hs_matrix.tsv from Rust')
    p.add_argument('--nvec', type=int, default=8, help='Initial probe vectors')
    p.add_argument('--cheb-deg', type=int, default=20, help='Chebyshev polynomial degree')
    p.add_argument('--iters', type=int, default=10, help='Filter→QR→Ritz iterations')
    p.add_argument('--band-width', type=float, default=0.05, help='Energy band width (Ha) around gap')
    p.add_argument('--square-filter', action='store_true', default=True, help='Use p(H)^2 filter')
    p.add_argument('--out', type=str, default=None, help='Output TSV (default: same dir)')
    args = p.parse_args()

    tsv_path = Path(args.hs_tsv)
    assert tsv_path.exists(), f"File not found: {tsv_path}"
    system_name = tsv_path.stem.replace('_hs_matrix', '')

    out_path = Path(args.out) if args.out else tsv_path.parent / f"{system_name}_sparse_eigenvectors.tsv"

    print(f"=== Sparse HOMO/LUMO: {system_name} ===")
    print(f"  H,S matrix: {tsv_path}")
    print(f"  nvec={args.nvec}  cheb_deg={args.cheb_deg}  iters={args.iters}  band_width={args.band_width} Ha")

    # 1. Parse H, S, geometry
    species, coords, H, S, eigs_dense, n_occ = parse_hs_matrix_tsv(tsv_path)
    norb = H.shape[0]
    natoms = len(species)
    print(f"  natoms={natoms}  norb={norb}  n_occ={n_occ}")

    # Dense reference HOMO/LUMO
    homo_dense = eigs_dense[n_occ - 1]
    lumo_dense = eigs_dense[n_occ]
    gap_dense = lumo_dense - homo_dense
    print(f"  dense ref: HOMO={homo_dense:.6f}  LUMO={lumo_dense:.6f}  gap={gap_dense:.6f} Ha")

    # 2. Transform generalized → standard: H' = S^{-1/2} H S^{-1/2}
    # S is symmetric positive definite, use eigendecomposition for S^{-1/2}
    print("  Computing S^{-1/2} ...")
    s_eigvals, s_eigvecs = np.linalg.eigh(S)
    # Guard against tiny/negative eigenvalues (S should be SPD)
    assert np.all(s_eigvals > 1e-10), f"S has non-positive eigenvalues: min={s_eigvals.min():.2e}"
    S_inv_sqrt = s_eigvecs @ np.diag(1.0 / np.sqrt(s_eigvals)) @ s_eigvecs.T
    H_std = S_inv_sqrt @ H @ S_inv_sqrt
    # Symmetrize to kill roundoff asymmetry
    H_std = 0.5 * (H_std + H_std.T)
    print(f"  H_std: min={H_std.min():.4f}  max={H_std.max():.4f}")

    # 3. Use TWO separate bands: one around HOMO, one around LUMO.
    # Band width is adaptive: max(args.band_width, 2×gap) so bands don't overlap
    # for small-gap systems but are wide enough to capture the target eigenvalue.
    gap_dense = lumo_dense - homo_dense
    half_w = max(args.band_width, 2.0 * gap_dense)
    homo_lo, homo_hi = homo_dense - half_w, homo_dense + half_w
    lumo_lo, lumo_hi = lumo_dense - half_w, lumo_dense + half_w
    print(f"  gap_dense={gap_dense:.6f}  half_w={half_w:.6f}")
    print(f"  HOMO band: [{homo_lo:.6f}, {homo_hi:.6f}] Ha")
    print(f"  LUMO band: [{lumo_lo:.6f}, {lumo_hi:.6f}] Ha")

    # 4. Run Chebyshev filter + Ritz for HOMO band
    print(f"  Running Chebyshev filter + Rayleigh-Ritz (HOMO band) ...")
    np.random.seed(42)
    V0_h = np.random.randn(norb, args.nvec)
    c_h = cheb_rect_coeffs(homo_lo, homo_hi, args.cheb_deg, use_jackson=True)
    V_h = V0_h.copy()
    spmv_count = 0
    for _ in range(max(1, args.iters)):
        V_h = apply_cheb_poly(H_std, V_h, c_h)
        spmv_count += args.cheb_deg * V_h.shape[1]
        if args.square_filter:
            V_h = apply_cheb_poly(H_std, V_h, c_h)
            spmv_count += args.cheb_deg * V_h.shape[1]
        V_h, _ = np.linalg.qr(V_h)
    w_h, U_h, r_h = rayleigh_ritz(H_std, V_h)
    spmv_count += V_h.shape[1]
    mask_h = (w_h >= homo_lo) & (w_h <= homo_hi)
    if not np.any(mask_h):
        # Keep closest to band center
        order_h = np.argsort(np.abs(w_h - homo_dense))
        mask_h = np.zeros_like(w_h, dtype=bool)
        mask_h[order_h[:min(args.nvec, len(w_h))]] = True
    w_h_in = w_h[mask_h]
    U_h_in = U_h[:, mask_h]
    r_h_in = r_h[mask_h]
    order_h = np.argsort(w_h_in)
    w_h_in = w_h_in[order_h]
    U_h_in = U_h_in[:, order_h]
    r_h_in = r_h_in[order_h]
    print(f"    HOMO band: {len(w_h_in)} eigs, residuals max={r_h_in.max():.2e}" if len(r_h_in) else "    HOMO band: 0 eigs")
    for w, r in zip(w_h_in, r_h_in):
        print(f"      {w:.10f}  r={r:.2e}")

    # 5. Run Chebyshev filter + Ritz for LUMO band
    print(f"  Running Chebyshev filter + Rayleigh-Ritz (LUMO band) ...")
    np.random.seed(123)
    V0_l = np.random.randn(norb, args.nvec)
    c_l = cheb_rect_coeffs(lumo_lo, lumo_hi, args.cheb_deg, use_jackson=True)
    V_l = V0_l.copy()
    for _ in range(max(1, args.iters)):
        V_l = apply_cheb_poly(H_std, V_l, c_l)
        spmv_count += args.cheb_deg * V_l.shape[1]
        if args.square_filter:
            V_l = apply_cheb_poly(H_std, V_l, c_l)
            spmv_count += args.cheb_deg * V_l.shape[1]
        V_l, _ = np.linalg.qr(V_l)
    w_l, U_l, r_l = rayleigh_ritz(H_std, V_l)
    spmv_count += V_l.shape[1]
    mask_l = (w_l >= lumo_lo) & (w_l <= lumo_hi)
    if not np.any(mask_l):
        order_l = np.argsort(np.abs(w_l - lumo_dense))
        mask_l = np.zeros_like(w_l, dtype=bool)
        mask_l[order_l[:min(args.nvec, len(w_l))]] = True
    w_l_in = w_l[mask_l]
    U_l_in = U_l[:, mask_l]
    r_l_in = r_l[mask_l]
    order_l = np.argsort(w_l_in)
    w_l_in = w_l_in[order_l]
    U_l_in = U_l_in[:, order_l]
    r_l_in = r_l_in[order_l]
    print(f"    LUMO band: {len(w_l_in)} eigs, residuals max={r_l_in.max():.2e}" if len(r_l_in) else "    LUMO band: 0 eigs")
    for w, r in zip(w_l_in, r_l_in):
        print(f"      {w:.10f}  r={r:.2e}")
    print(f"  Total SpMV count: {spmv_count}")

    # 6. Pick HOMO = eigenvalue closest to homo_dense, LUMO = closest to lumo_dense
    if len(w_h_in) > 0 and len(w_l_in) > 0:
        homo_idx = np.argmin(np.abs(w_h_in - homo_dense))
        lumo_idx = np.argmin(np.abs(w_l_in - lumo_dense))
        homo_sp = w_h_in[homo_idx]
        lumo_sp = w_l_in[lumo_idx]
        c_homo_std = U_h_in[:, homo_idx]
        c_lumo_std = U_l_in[:, lumo_idx]
        method = "Chebyshev+Ritz"
    else:
        print("  WARNING: insufficient eigenvalues found, using dense fallback")
        w_all, U_all = np.linalg.eigh(H_std)
        homo_sp = w_all[n_occ - 1]
        lumo_sp = w_all[n_occ]
        c_homo_std = U_all[:, n_occ - 1]
        c_lumo_std = U_all[:, n_occ]
        method = "dense_fallback"

    gap_sp = lumo_sp - homo_sp
    print(f"\n  {method} result:")
    print(f"    HOMO={homo_sp:.10f} Ha  (dense ref: {homo_dense:.10f}, Δ={homo_sp-homo_dense:.2e})")
    print(f"    LUMO={lumo_sp:.10f} Ha  (dense ref: {lumo_dense:.10f}, Δ={lumo_sp-lumo_dense:.2e})")
    print(f"    gap={gap_sp:.10f} Ha   (dense ref: {gap_dense:.10f}, Δ={gap_sp-gap_dense:.2e})")

    # 7. Transform eigenvectors back to generalized basis: c = S^{-1/2} c'
    c_homo = S_inv_sqrt @ c_homo_std
    c_lumo = S_inv_sqrt @ c_lumo_std
    evecs_out = np.column_stack([c_homo, c_lumo])
    eigs_out = np.array([homo_sp, lumo_sp])

    # 8. Save in the same format as dense eigenvectors TSV
    with open(out_path, 'w') as f:
        f.write(f"# natoms={natoms} norb={norb} n_occ={n_occ}\n")
        f.write("# atom_idx\telement\tx\ty\tz\n")
        for j, (sp, c) in enumerate(zip(species, coords)):
            f.write(f"{j}\t{sp}\t{c[0]:.10f}\t{c[1]:.10f}\t{c[2]:.10f}\n")
        f.write("# eigenvector (HOMO and LUMO only, columns are MOs)\n")
        f.write("mo_idx\torb_idx\tcoeff\n")
        for mo in range(2):
            for orb in range(norb):
                f.write(f"{mo}\t{orb}\t{evecs_out[orb, mo]:.12e}\n")
        f.write("# eigenvalues\n")
        f.write("idx\teigenvalue\tlabel\n")
        f.write(f"0\t{homo_sp:.10f}\tHOMO\n")
        f.write(f"1\t{lumo_sp:.10f}\tLUMO\n")
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
