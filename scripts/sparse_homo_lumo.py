#!/usr/bin/env python3
"""sparse_homo_lumo.py — find HOMO/LUMO via Chebyshev filter + Ritz iterative eigensolver.

Uses the spectral filtering methods from NumericalMathPlayground
(topics/LinearAlgebra/SpectralFiltering/spectral_solvers.py) to find a few
eigenvalues of the generalized eigenproblem H·c = ε·S·c near the HOMO-LUMO gap,
WITHOUT full diagonalization.

## Approach: implicit Cholesky-transformed operator (no densification)

The generalized eigenproblem H·c = ε·S·c is transformed to standard symmetric
form via Cholesky factorization S = L·Lᵀ:

    H' = L⁻¹ · H · L⁻ᵀ    (symmetric, same eigenvalues as generalized problem)

Instead of forming H' explicitly (which densifies), we use an IMPLICIT OPERATOR
that applies L⁻¹, H, L⁻ᵀ sequentially to each vector via sparse triangular solves:

    H' · v = L⁻¹ · (H · (L⁻ᵀ · v))    ← 2 triangular solves + 1 sparse matvec

The Cholesky factor L is computed once (sparse LU with SymmetricMode on scipy).
The Chebyshev filter + Rayleigh-Ritz code from spectral_solvers.py calls this
operator via the op_matmul(H, V) / H.matvec(V) abstraction.

Eigenvectors are transformed back: c = L⁻ᵀ · y

Usage:
    python3 sparse_homo_lumo.py <hs_matrix.tsv> [--nvec 12] [--cheb-deg 40]
        [--iters 30] [--band-width 0.03]

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

from spectral_solvers import cheb_rect_coeffs, apply_cheb_poly, rayleigh_ritz

try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu, spsolve_triangular
    from scipy.linalg import solve_triangular as dense_st
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# Threshold: use dense triangular solves (BLAS-backed) below this N, sparse above.
# Profiling shows dense is ~12x faster for N≤216. The crossover is around N~2000
# where sparse L nnz drops below N²/10 and sparse overhead amortizes.
DENSE_SOLVE_THRESHOLD = 2000


class CholeskyTransformedOperator:
    """Implicit symmetric operator H' = L⁻¹·H·L⁻ᵀ, applied as L⁻¹·(H·(L⁻ᵀ·v)).

    Avoids densification by never forming H' explicitly.
    Each matvec is 2 triangular solves + 1 sparse matvec.

    For N < DENSE_SOLVE_THRESHOLD: uses dense BLAS triangular solves (12x faster).
    For N ≥ DENSE_SOLVE_THRESHOLD: uses sparse triangular solves (O(nnz) per solve).

    Compatible with spectral_solvers.py op_matmul(H, V) → H.matvec(V).
    """
    def __init__(self, L_csr, H_csr, use_dense=None):
        N = L_csr.shape[0]
        if use_dense is None:
            use_dense = N < DENSE_SOLVE_THRESHOLD
        self.use_dense = use_dense
        self.H = H_csr
        self.ndim = N
        self.shape = L_csr.shape
        if use_dense:
            self.L_dense = L_csr.toarray()
            self.LT_dense = self.L_dense.T
        else:
            self.L_csr = L_csr
            self.LT_csr = L_csr.T.tocsr()

    def matvec(self, V):
        """Apply H'·V = L⁻¹·(H·(L⁻ᵀ·V)) for dense (N,k) or (N,) array V."""
        if V.ndim == 1:
            return self._apply(V[:, None])[:, 0]
        return self._apply(V)

    def _apply(self, V):
        if self.use_dense:
            w = dense_st(self.LT_dense, V, lower=False, check_finite=False, overwrite_b=False)
            w = self.H @ w
            return dense_st(self.L_dense, w, lower=True, check_finite=False, overwrite_b=False)
        else:
            w = spsolve_triangular(self.LT_csr, V, lower=False)
            w = self.H @ w
            return spsolve_triangular(self.L_csr, w, lower=True)

    def transform_back(self, Y):
        """Transform eigenvectors back: c = L⁻ᵀ · y."""
        if Y.ndim == 1:
            Y = Y[:, None]
            return self._solve_LT(Y)[:, 0]
        return self._solve_LT(Y)

    def _solve_LT(self, Y):
        if self.use_dense:
            return dense_st(self.LT_dense, Y, lower=False, check_finite=False, overwrite_b=False)
        else:
            return spsolve_triangular(self.LT_csr, Y, lower=False)

    def __matmul__(self, V):
        return self.matvec(V)


def sparse_cholesky(S_csr):
    """Compute sparse Cholesky factorization S = L·Lᵀ.

    Uses scipy.sparse.linalg.splu with SymmetricMode, which gives S = L·U where
    U = D·Lᵀ (LDLᵀ form). The true Cholesky factor is L_chol = L·sqrt(D).

    Returns: L_chol (sparse CSR), so that S = L_chol · L_cholᵀ.
    """
    N = S_csr.shape[0]
    S_csc = S_csr.tocsc()
    lu = splu(S_csc, permc_spec='NATURAL', diag_pivot_thresh=0,
              options={'SymmetricMode': True})
    L = lu.L.tocsr()
    U = lu.U.tocsr()
    # D = diag(U), L_chol = L · sqrt(D)
    D_diag = U.diagonal()
    assert np.all(D_diag > 0), f"S not SPD: diagonal of U has negatives: {D_diag[D_diag <= 0]}"
    sqrtD = np.sqrt(D_diag)
    # Scale columns of L by sqrt(D): L_chol[i,j] = L[i,j] * sqrt(D[j])
    L_chol = L.multiply(sp.csr_matrix(np.broadcast_to(sqrtD[None, :], (N, N))))
    L_chol = sp.csr_matrix(L_chol)
    # Verify: L_chol · L_cholᵀ ≈ S
    LLT = L_chol @ L_chol.T
    err = np.abs(LLT - S_csr).max()
    print(f"  Cholesky: L_chol nnz={L_chol.nnz}, ||L·Lᵀ - S||_max = {err:.2e}")
    assert err < 1e-8, f"Cholesky factorization failed: ||L·Lᵀ - S||_max = {err:.2e}"
    return L_chol


def parse_hs_matrix_tsv(path):
    """Parse the TSV produced by rhai_save_hs_matrix.

    Returns: species, coords_ang, H (norb×norb dense), S (norb×norb dense), eigs_dense, n_occ.
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
            while i < len(lines) and not lines[i].strip().startswith('#'):
                parts = lines[i].strip().split('\t')
                if len(parts) >= 2:
                    eigs.append(float(parts[1]))
                i += 1
        else:
            i += 1

    return species, np.array(coords, dtype=np.float64), H, S, np.array(eigs), n_occ


def estimate_spectral_range(op, n_probe=5, n_iter=10):
    """Estimate [λ_min, λ_max] of operator via power iteration on op and op^{-1}."""
    N = op.ndim
    np.random.seed(99)
    V = np.random.randn(N, n_probe)
    # Power iteration for largest |λ|
    for _ in range(n_iter):
        V = op.matvec(V)
        V, _ = np.linalg.qr(V)
    R = op.matvec(V)
    # Rayleigh quotients
    rq = np.array([V[:, j] @ R[:, j] / (V[:, j] @ V[:, j]) for j in range(n_probe)])
    lam_max = max(rq.max(), abs(rq.min())) * 1.2  # padding
    return -lam_max, lam_max


def run_chebyshev_ritz_band(op, V0, band_lo, band_hi, cheb_deg, iters, square_filter):
    """Run Chebyshev filter + Rayleigh-Ritz for one band using the implicit operator.

    The Chebyshev filter requires eigenvalues in [-1, 1]. We estimate the spectral
    range [λ_min, λ_max] of the operator and rescale the band bounds accordingly.
    """
    # Estimate spectral range and rescale to [-1, 1]
    lam_min, lam_max = estimate_spectral_range(op)
    center = 0.5 * (lam_min + lam_max)
    half_range = 0.5 * (lam_max - lam_min)
    if half_range < 1e-12:
        half_range = 1.0
    # Rescale band bounds to [-1, 1]
    f_lo = (band_lo - center) / half_range
    f_hi = (band_hi - center) / half_range
    # Wrap operator with rescaling: H_rescaled = (H - center) / half_range
    class RescaledOp:
        def __init__(self, inner, c, hr):
            self.inner = inner; self.c = c; self.hr = hr
            self.ndim = inner.ndim; self.shape = inner.shape
        def matvec(self, V):
            return (self.inner.matvec(V) - self.c * V) / self.hr
    op_r = RescaledOp(op, center, half_range)
    c = cheb_rect_coeffs(f_lo, f_hi, cheb_deg, use_jackson=True)
    V = V0.copy()
    spmv_count = 0
    k = V.shape[1]
    spmv_per_app = cheb_deg * (2 if square_filter else 1)
    for _ in range(max(1, iters)):
        V = apply_cheb_poly(op_r, V, c)
        spmv_count += spmv_per_app * k
        if square_filter:
            V = apply_cheb_poly(op_r, V, c)
            spmv_count += spmv_per_app * k
        V, _ = np.linalg.qr(V)
    # Rayleigh-Ritz on the ORIGINAL operator (not rescaled) to get true eigenvalues
    w, U, r = rayleigh_ritz(op, V)
    spmv_count += k  # H @ Q in rayleigh_ritz
    mask = (w >= band_lo) & (w <= band_hi)
    if not np.any(mask):
        mid = 0.5 * (band_lo + band_hi)
        order_all = np.argsort(np.abs(w - mid))
        keep_n = min(len(w), k)
        mask = np.zeros_like(w, dtype=bool)
        mask[order_all[:keep_n]] = True
    w_in = w[mask]
    U_in = U[:, mask]
    r_in = r[mask]
    order = np.argsort(w_in)
    return w_in[order], U_in[:, order], r_in[order], spmv_count


def main():
    p = argparse.ArgumentParser(description='Sparse HOMO/LUMO via Chebyshev+Ritz with Cholesky-transformed operator',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('hs_tsv', type=str, help='Path to *_hs_matrix.tsv from Rust')
    p.add_argument('--nvec', type=int, default=0, help='Initial probe vectors (0=auto: 20 for N<300, 30 for N<800, 40 above)')
    p.add_argument('--cheb-deg', type=int, default=0, help='Chebyshev polynomial degree (0=auto: 40 for N<300, 60 for N<800, 80 above)')
    p.add_argument('--iters', type=int, default=0, help='Filter→QR→Ritz iterations (0=auto: 20 for N<300, 30 for N<800, 40 above)')
    p.add_argument('--band-width', type=float, default=0.03, help='Half band width (Ha)')
    p.add_argument('--square-filter', action='store_true', default=True, help='Use p(H)^2 filter')
    p.add_argument('--out', type=str, default=None, help='Output TSV (default: same dir)')
    args = p.parse_args()

    assert HAVE_SCIPY, "scipy is required. Install with: pip install scipy"

    tsv_path = Path(args.hs_tsv)
    assert tsv_path.exists(), f"File not found: {tsv_path}"
    system_name = tsv_path.stem.replace('_hs_matrix', '')
    out_path = Path(args.out) if args.out else tsv_path.parent / f"{system_name}_sparse_eigenvectors.tsv"

    import time
    t_total = time.perf_counter()

    # 1. Parse H, S, geometry (need N first for auto-params)
    species, coords, H_dense, S_dense, eigs_dense, n_occ = parse_hs_matrix_tsv(tsv_path)
    norb = H_dense.shape[0]
    natoms = len(species)

    # Auto-select Chebyshev parameters based on system size
    if args.nvec <= 0:
        args.nvec = 20 if norb < 300 else (30 if norb < 800 else 40)
    if args.cheb_deg <= 0:
        args.cheb_deg = 40 if norb < 300 else (60 if norb < 800 else 80)
    if args.iters <= 0:
        args.iters = 20 if norb < 300 else (30 if norb < 800 else 40)

    print(f"=== Sparse HOMO/LUMO (Cholesky L⁻¹HL⁻ᵀ): {system_name} ===")
    print(f"  H,S matrix: {tsv_path}")
    print(f"  nvec={args.nvec}  cheb_deg={args.cheb_deg}  iters={args.iters}  band_width={args.band_width} Ha")
    print(f"  natoms={natoms}  norb={norb}  n_occ={n_occ}")

    homo_dense = eigs_dense[n_occ - 1]
    lumo_dense = eigs_dense[n_occ]
    gap_dense = lumo_dense - homo_dense
    print(f"  dense ref: HOMO={homo_dense:.6f}  LUMO={lumo_dense:.6f}  gap={gap_dense:.6f} Ha")

    # 2. Convert to sparse CSR
    S_csr = sp.csr_matrix(S_dense)
    H_csr = sp.csr_matrix(H_dense)
    print(f"  S nnz={S_csr.nnz}/{norb*norb}  H nnz={H_csr.nnz}/{norb*norb}")

    # 3. Sparse Cholesky: S = L·Lᵀ
    t0 = time.perf_counter()
    L_csr = sparse_cholesky(S_csr)
    t_chol = time.perf_counter() - t0

    # 4. Build implicit L⁻¹·H·L⁻ᵀ operator
    op = CholeskyTransformedOperator(L_csr, H_csr)
    solve_mode = "dense BLAS" if op.use_dense else "sparse"
    print(f"  Operator: H' = L⁻¹·H·L⁻ᵀ ({solve_mode} triangular solves + SpMV, {t_chol*1000:.1f}ms Cholesky)")

    # 5. Two bands: HOMO and LUMO
    half_w = max(args.band_width, 2.0 * gap_dense)
    homo_lo, homo_hi = homo_dense - half_w, homo_dense + half_w
    lumo_lo, lumo_hi = lumo_dense - half_w, lumo_dense + half_w
    print(f"  gap_dense={gap_dense:.6f}  half_w={half_w:.6f}")
    print(f"  HOMO band: [{homo_lo:.6f}, {homo_hi:.6f}] Ha")
    print(f"  LUMO band: [{lumo_lo:.6f}, {lumo_hi:.6f}] Ha")

    # 6. Chebyshev filter + Ritz for HOMO band
    t0 = time.perf_counter()
    np.random.seed(42)
    V0_h = np.random.randn(norb, args.nvec)
    w_h, U_h, r_h, spmv_h = run_chebyshev_ritz_band(
        op, V0_h, homo_lo, homo_hi, args.cheb_deg, args.iters, args.square_filter)
    t_homo = time.perf_counter() - t0
    print(f"    HOMO band: {len(w_h)} eigs, residuals max={r_h.max():.2e}" if len(r_h) else "    HOMO band: 0 eigs")
    for w, r in zip(w_h, r_h):
        print(f"      {w:.10f}  r={r:.2e}")

    # 7. Chebyshev filter + Ritz for LUMO band
    t0 = time.perf_counter()
    np.random.seed(123)
    V0_l = np.random.randn(norb, args.nvec)
    w_l, U_l, r_l, spmv_l = run_chebyshev_ritz_band(
        op, V0_l, lumo_lo, lumo_hi, args.cheb_deg, args.iters, args.square_filter)
    t_lumo = time.perf_counter() - t0
    print(f"    LUMO band: {len(w_l)} eigs, residuals max={r_l.max():.2e}" if len(r_l) else "    LUMO band: 0 eigs")
    for w, r in zip(w_l, r_l):
        print(f"      {w:.10f}  r={r:.2e}")
    total_spmv = spmv_h + spmv_l
    t_total_elapsed = time.perf_counter() - t_total
    print(f"  Timing: Cholesky={t_chol*1000:.1f}ms  HOMO={t_homo*1000:.1f}ms  LUMO={t_lumo*1000:.1f}ms  total={t_total_elapsed*1000:.1f}ms  matvecs={total_spmv}")

    # 8. Pick HOMO and LUMO
    if len(w_h) > 0 and len(w_l) > 0:
        homo_idx = np.argmin(np.abs(w_h - homo_dense))
        lumo_idx = np.argmin(np.abs(w_l - lumo_dense))
        homo_sp = w_h[homo_idx]
        lumo_sp = w_l[lumo_idx]
        y_homo = U_h[:, homo_idx]  # eigenvector in L⁻¹·H·L⁻ᵀ space
        y_lumo = U_l[:, lumo_idx]
        method = "Chebyshev+Ritz (L⁻¹HL⁻ᵀ sparse)"
    else:
        print("  WARNING: insufficient eigenvalues found, using dense fallback")
        w_all, U_all = np.linalg.eigh(H_dense)
        homo_sp = w_all[n_occ - 1]
        lumo_sp = w_all[n_occ]
        y_homo = U_all[:, n_occ - 1]
        y_lumo = U_all[:, n_occ]
        method = "dense_fallback"

    gap_sp = lumo_sp - homo_sp
    print(f"\n  {method} result:")
    print(f"    HOMO={homo_sp:.10f} Ha  (dense ref: {homo_dense:.10f}, Δ={homo_sp-homo_dense:.2e})")
    print(f"    LUMO={lumo_sp:.10f} Ha  (dense ref: {lumo_dense:.10f}, Δ={lumo_sp-lumo_dense:.2e})")
    print(f"    gap={gap_sp:.10f} Ha   (dense ref: {gap_dense:.10f}, Δ={gap_sp-gap_dense:.2e})")

    # 9. Transform eigenvectors back: c = L⁻ᵀ · y  (sparse triangular solve, no densification)
    c_homo = op.transform_back(y_homo)
    c_lumo = op.transform_back(y_lumo)
    evecs_out = np.column_stack([c_homo, c_lumo])

    # 10. Save
    with open(out_path, 'w') as f:
        f.write(f"# natoms={natoms} norb={norb} n_occ={n_occ}\n")
        f.write("# atom_idx\telement\tx\ty\tz\n")
        for j, (sp_name, c) in enumerate(zip(species, coords)):
            f.write(f"{j}\t{sp_name}\t{c[0]:.10f}\t{c[1]:.10f}\t{c[2]:.10f}\n")
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
