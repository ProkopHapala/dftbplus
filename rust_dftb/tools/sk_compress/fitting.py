"""
Core math: basis construction, fitting, regularization, conditioning, correlation.
"""

import numpy as np
from numpy.polynomial.legendre import legvander
from numpy.polynomial.chebyshev import chebfit, chebval
from typing import Tuple, List, Optional, Dict, Any, Union, Callable


# =============================================================================
# Envelope and utilities
# =============================================================================

def envelope(u: np.ndarray, power: int = 2, variant: str = "linear") -> np.ndarray:
    """Smooth cutoff envelope: linear -> (1-u)^power, quadratic -> (1-u^2)^power."""
    if variant == "linear":
        return np.clip(1.0 - u, 0.0, None) ** power
    elif variant == "quadratic":
        return np.clip(1.0 - u ** 2, 0.0, None) ** power
    else:
        raise ValueError(f"Unknown envelope variant: {variant}")


def normalize_columns(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """L2-normalize each column; return (A_normed, norms)."""
    norms = np.sqrt(np.sum(A ** 2, axis=0))
    norms[norms == 0] = 1.0
    return A / norms, norms


# =============================================================================
# Shifted Legendre
# =============================================================================

def shifted_legendre_vander(u: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate shifted Legendre P_k(2u-1) for k=0..degree. Returns (n_points, degree+1)."""
    x = 2.0 * u - 1.0
    return legvander(x, degree)


# =============================================================================
# Dyadic basis
# =============================================================================

def build_dyadic_basis_vander(v: np.ndarray, n_terms: int, start_n: int = 0) -> np.ndarray:
    """
    Build dyadic basis p_n = v^(2^n) for n=start_n..start_n+n_terms-1.
    Generated recursively. Returns (n_points, n_terms).
    """
    V = np.zeros((len(v), n_terms))
    p = v.copy()
    for _ in range(start_n):
        p = p * p
    for n in range(n_terms):
        V[:, n] = p
        p = p * p
    return V


def build_dyadic_product_basis(
    u: np.ndarray, m: int, n_terms: int, chi: np.ndarray,
    clip_u: float = 1e-8, start_n: int = 0,
) -> np.ndarray:
    """psi_n(u) = chi(u) * u^m * (1-u)^(2^n). Returns (n_points, n_terms)."""
    u_safe = np.clip(u, clip_u, 1.0)
    prefactor = u_safe ** m
    v = 1.0 - u_safe
    dyadic = build_dyadic_basis_vander(v, n_terms, start_n=start_n)
    return chi[:, None] * prefactor[:, None] * dyadic


def build_pure_dyadic_basis(
    u: np.ndarray, n_terms: int, chi: np.ndarray,
    clip_u: float = 1e-8, start_n: int = 0,
) -> np.ndarray:
    """psi_n(u) = chi(u) * (1-u)^(2^n). No u^m prefactor. Returns (n_points, n_terms)."""
    u_safe = np.clip(u, clip_u, 1.0)
    v = 1.0 - u_safe
    dyadic = build_dyadic_basis_vander(v, n_terms, start_n=start_n)
    return chi[:, None] * dyadic


# =============================================================================
# Product basis (old pipeline)
# =============================================================================

def build_product_basis(
    u: np.ndarray, m: int, degree: int, chi: np.ndarray, clip_u: float = 1e-8,
) -> np.ndarray:
    """psi_k(u) = chi(u) * u^m * P_k(2u-1). Returns (n_points, degree+1)."""
    u_safe = np.clip(u, clip_u, 1.0)
    prefactor = u_safe ** m
    legendre = shifted_legendre_vander(u, degree)
    return chi[:, None] * prefactor[:, None] * legendre


# =============================================================================
# Custom linear basis: χ(u)·(1-u)^4 · {1, u, u^2, u^4, r, r^2}
# =============================================================================

def build_custom_basis_vander(
    u: np.ndarray, Rc: float, clip_u: float = 1e-8,
) -> np.ndarray:
    """
    7-function basis inspired by dyadic structure:
    χ(v)·{1, v², v⁶, v¹⁴, r, r², r³} where v = (1-r/Rc), r = u·Rc, χ(v) = v².
    The v-powers give {v², v⁴, v⁸, v¹⁶} — dyadic-like multi-scale resolution.
    Returns (n_points, 7).
    """
    u_safe = np.clip(u, clip_u, 1.0)
    v = 1.0 - u_safe
    r = u_safe * Rc
    chi = v ** 2
    return np.column_stack([
        chi * 1.0,      # v²  →  dyadic-like
        chi * v**2,     # v⁴
        chi * v**6,     # v⁸
        chi * v**14,    # v¹⁶
        chi * r,        # r term
        chi * r**2,     # r² term
        chi * r**3,     # r³ term
    ])


def build_chebyshev_chi_basis(
    u: np.ndarray, chi_power: float, degree: int, clip_u: float = 1e-8,
) -> np.ndarray:
    """
    Basis: v^chi_power · {T_0(v), T_1(v), ..., T_degree(v)} where v = 1-u.
    Uses shifted Chebyshev T_k(2v-1) on [0,1]. Returns (n_points, degree+1).
    """
    u_safe = np.clip(u, clip_u, 1.0)
    v = 1.0 - u_safe
    chi = v ** chi_power
    cheb = build_chebyshev_basis_vander(v, degree)  # T_k(2v-1)
    return chi[:, None] * cheb


def build_dual_envelope_cheb_basis(
    u: np.ndarray, p1: float, p2: float, degree: int, clip_u: float = 1e-8,
) -> np.ndarray:
    """
    Dual-envelope Chebyshev basis: v^p1·T_0..degree(v)  and  v^p2·T_0..degree(v).
    Total coefficients: 2*(degree+1).  Returns (n_points, 2*(degree+1)).
    """
    u_safe = np.clip(u, clip_u, 1.0)
    v = 1.0 - u_safe
    cheb = build_chebyshev_basis_vander(v, degree)  # T_k(2v-1)
    W1 = (v ** p1)[:, None] * cheb
    W2 = (v ** p2)[:, None] * cheb
    return np.hstack([W1, W2])


def fit_custom_basis(
    u: np.ndarray, f: np.ndarray, Rc: float,
    clip_u: float = 1e-8, lambda_reg: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """Fit in custom 7-function basis with dyadic-inspired v-powers."""
    W = build_custom_basis_vander(u, Rc, clip_u=clip_u)
    if lambda_reg > 0:
        WTW = W.T @ W
        n = W.shape[1]
        WTW_reg = WTW + lambda_reg * np.eye(n)
        c = np.linalg.solve(WTW_reg, W.T @ f)
        res_norm = np.sqrt(np.sum((f - W @ c) ** 2))
    else:
        c, residuals, rank, s = np.linalg.lstsq(W, f, rcond=None)
        res_norm = np.sqrt(np.sum(residuals)) if len(residuals) > 0 else 0.0
    return c, res_norm


# =============================================================================
# Dyadic basis orthogonalization (Gram-Schmidt)
# =============================================================================

def orthogonalize_dyadic_basis(
    dyadic: np.ndarray, method: str = "gram_schmidt", normalize: bool = True,
    direction: str = "forward",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orthogonalize dyadic basis columns via Gram-Schmidt or QR.

    Parameters
    ----------
    dyadic : (n_points, n_terms)
        Raw dyadic basis matrix (e.g. v^(2^n)).
    method : "gram_schmidt" or "qr"
    normalize : bool
        If True, return orthonormal columns.
    direction : "forward" or "backward"
        Forward: start from column 0 (smoothest), orthogonalize 1,2,3... against previous.
        Backward: start from last column (sharpest), orthogonalize n-2,... against previous.

    Returns
    -------
    dyadic_ortho : (n_points, n_terms)
        Orthogonalized/orthonormalized columns.
    transform : (n_terms, n_terms)
        Lower-triangular matrix such that dyadic_ortho = dyadic @ transform.T
    """
    n_points, n_terms = dyadic.shape
    if method == "qr":
        Q, R = np.linalg.qr(dyadic)
        if normalize:
            dyadic_ortho = Q
            transform = np.linalg.inv(R).T
        else:
            norms = np.linalg.norm(dyadic, axis=0)
            dyadic_ortho = Q * norms[None, :]
            transform = np.linalg.inv(R).T * norms[None, :]
    else:
        indices = list(range(n_terms)) if direction == "forward" else list(range(n_terms - 1, -1, -1))
        dyadic_ortho = dyadic.copy()
        transform = np.eye(n_terms)
        for idx_j, j in enumerate(indices):
            for idx_i in range(idx_j):
                i = indices[idx_i]
                proj = np.dot(dyadic_ortho[:, i], dyadic[:, j]) / (np.dot(dyadic_ortho[:, i], dyadic_ortho[:, i]) + 1e-30)
                dyadic_ortho[:, j] -= proj * dyadic_ortho[:, i]
                transform[:, j] -= proj * transform[:, i]
            norm = np.linalg.norm(dyadic_ortho[:, j])
            if normalize and norm > 1e-15:
                dyadic_ortho[:, j] /= norm
                transform[:, j] /= norm
    return dyadic_ortho, transform


def build_combined_basis_ortho_dyadic(
    u: np.ndarray, n_dyadic: int, degree_legendre: int, chi: np.ndarray,
    clip_u: float = 1e-8, start_n: int = 2, normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combined basis with Gram-Schmidt orthogonalized dyadic part.
    W_{(n,k)}(u) = chi(u) * q_n(u) * P_k(2u-1)
    where q_n are orthonormal dyadic functions.
    Returns (W, transform) where transform maps original dyadic → orthonormal.
    """
    u_safe = np.clip(u, clip_u, 1.0)
    v = 1.0 - u_safe
    dyadic_raw = build_dyadic_basis_vander(v, n_dyadic, start_n=start_n)
    # Orthogonalize raw dyadic (without envelope, on masked grid if needed)
    dyadic_ortho, transform = orthogonalize_dyadic_basis(dyadic_raw, method="gram_schmidt", normalize=normalize)
    legendre = shifted_legendre_vander(u, degree_legendre)
    n_points = len(u)
    n_combined = n_dyadic * (degree_legendre + 1)
    W = np.zeros((n_points, n_combined))
    for n in range(n_dyadic):
        for k in range(degree_legendre + 1):
            W[:, n * (degree_legendre + 1) + k] = chi * dyadic_ortho[:, n] * legendre[:, k]
    return W, transform


def fit_combined_basis_ortho_dyadic(
    u: np.ndarray, f: np.ndarray, n_dyadic: int, degree_legendre: int,
    chi: np.ndarray, clip_u: float = 1e-8, start_n: int = 2,
    lambda_reg: float = 1e-12, normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit with orthogonalized dyadic basis.
    Returns (coefficients, transform, residual_norm).
    To evaluate: build_combined_basis_ortho_dyadic → W @ c
    """
    W, transform = build_combined_basis_ortho_dyadic(
        u, n_dyadic, degree_legendre, chi, clip_u=clip_u, start_n=start_n, normalize=normalize
    )
    if lambda_reg > 0:
        WTW = W.T @ W
        n = W.shape[1]
        WTW_reg = WTW + lambda_reg * np.eye(n)
        c = np.linalg.solve(WTW_reg, W.T @ f)
        res_norm = np.sqrt(np.sum((f - W @ c) ** 2))
    else:
        c, residuals, rank, s = np.linalg.lstsq(W, f, rcond=None)
        res_norm = np.sqrt(np.sum(residuals)) if len(residuals) > 0 else 0.0
    return c, transform, res_norm


# =============================================================================
# Column-normalized combined basis (poor man's SVD)
# =============================================================================

def build_combined_basis_normalized(
    u: np.ndarray, n_dyadic: int, degree_legendre: int, chi: np.ndarray,
    clip_u: float = 1e-8, start_n: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combined dyadic×Legendre basis with each column normalized to unit L2 norm.
    W_norm[:, j] = W[:, j] / ||W[:, j]||_2.
    Returns (W_norm, norms) where norms[j] = ||W[:, j]||_2.
    """
    W = build_combined_basis(u, n_dyadic, degree_legendre, chi, clip_u=clip_u, start_n=start_n)
    norms = np.linalg.norm(W, axis=0)
    norms_safe = np.where(norms > 1e-15, norms, 1.0)
    W_norm = W / norms_safe[None, :]
    return W_norm, norms


def fit_combined_basis_normalized(
    u: np.ndarray, f: np.ndarray, n_dyadic: int, degree_legendre: int,
    chi: np.ndarray, clip_u: float = 1e-8, start_n: int = 2,
    lambda_reg: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit in column-normalized combined basis.
    Returns (coefficients_norm, norms, residual_norm).
    True coefficients = c_norm / norms.
    """
    W_norm, norms = build_combined_basis_normalized(
        u, n_dyadic, degree_legendre, chi, clip_u=clip_u, start_n=start_n
    )
    if lambda_reg > 0:
        WTW = W_norm.T @ W_norm
        n = W_norm.shape[1]
        WTW_reg = WTW + lambda_reg * np.eye(n)
        c_norm = np.linalg.solve(WTW_reg, W_norm.T @ f)
        res_norm = np.sqrt(np.sum((f - W_norm @ c_norm) ** 2))
    else:
        c_norm, residuals, rank, s = np.linalg.lstsq(W_norm, f, rcond=None)
        res_norm = np.sqrt(np.sum(residuals)) if len(residuals) > 0 else 0.0
    return c_norm, norms, res_norm


# =============================================================================
# Alternative fine-tuning bases
# =============================================================================

def build_monomial_basis_vander(u: np.ndarray, degree: int) -> np.ndarray:
    """φ_k(u) = u^k for k=0..degree."""
    u_col = np.asarray(u).reshape(-1, 1)
    k = np.arange(degree + 1).reshape(1, -1)
    return u_col ** k


def build_chebyshev_basis_vander(u: np.ndarray, degree: int) -> np.ndarray:
    """T_k(2u-1), shifted Chebyshev of the first kind on [0,1]."""
    from numpy.polynomial.chebyshev import chebvander
    x = 2.0 * u - 1.0
    return chebvander(x, degree)


def build_hermite_basis_vander(u: np.ndarray, degree: int, alpha: float = 4.0) -> np.ndarray:
    """
    Gaussian-windowed centered monomials: φ_k(u) = exp(-α(u-0.5)²) * (u-0.5)^k.
    Hermite-like: localized oscillatory basis centered at u=0.5.
    """
    u_c = u - 0.5
    gauss = np.exp(-alpha * u_c ** 2)
    u_col = u_c.reshape(-1, 1)
    k = np.arange(degree + 1).reshape(1, -1)
    return gauss[:, None] * (u_col ** k)


def build_bspline_basis_vander(u: np.ndarray, degree: int, n_knots: int = None) -> np.ndarray:
    """
    Cubic B-spline basis on [0,1].
    Uses scipy.interpolate.BSpline.basis_element.
    """
    try:
        from scipy.interpolate import BSpline
    except ImportError:
        # Fallback to Legendre if scipy not available
        return shifted_legendre_vander(u, degree)
    
    if n_knots is None:
        n_knots = degree + 2
    
    # Uniform knot sequence with boundary multiplicity
    knots = np.linspace(0, 1, n_knots)
    t = np.concatenate([
        np.zeros(degree),  # left boundary multiplicity
        knots,
        np.ones(degree),    # right boundary multiplicity
    ])
    
    n_basis = len(t) - degree - 1
    B = np.zeros((len(u), n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        spl = BSpline(t, c, degree)
        B[:, i] = spl(u)
    
    # If more B-splines than needed, take first (degree+1)
    if n_basis > degree + 1:
        B = B[:, :degree + 1]
    return B


# =============================================================================
# Combined basis: dyadic × Legendre
# =============================================================================

def build_combined_basis(
    u: np.ndarray, n_dyadic: int, degree_legendre: int, chi: np.ndarray,
    clip_u: float = 1e-8, start_n: int = 2,
) -> np.ndarray:
    """
    W_{(n,k)}(u) = chi(u) * (1-u)^(2^n) * P_k(2u-1).
    Returns design matrix (n_points, n_dyadic * (degree_legendre+1)).
    """
    u_safe = np.clip(u, clip_u, 1.0)
    v = 1.0 - u_safe
    dyadic = build_dyadic_basis_vander(v, n_dyadic, start_n=start_n)
    legendre = shifted_legendre_vander(u, degree_legendre)
    n_points = len(u)
    n_combined = n_dyadic * (degree_legendre + 1)
    W = np.zeros((n_points, n_combined))
    for n in range(n_dyadic):
        for k in range(degree_legendre + 1):
            W[:, n * (degree_legendre + 1) + k] = chi * dyadic[:, n] * legendre[:, k]
    return W


def _build_combined_basis_generic(
    u: np.ndarray, n_dyadic: int, degree: int, chi: np.ndarray,
    fine_vander_fn, clip_u: float = 1e-8, start_n: int = 2,
) -> np.ndarray:
    """Generic combined basis builder with any fine-tuning basis."""
    u_safe = np.clip(u, clip_u, 1.0)
    v = 1.0 - u_safe
    dyadic = build_dyadic_basis_vander(v, n_dyadic, start_n=start_n)
    fine = fine_vander_fn(u_safe, degree)
    n_points = len(u)
    n_combined = n_dyadic * (degree + 1)
    W = np.zeros((n_points, n_combined))
    for n in range(n_dyadic):
        for k in range(degree + 1):
            W[:, n * (degree + 1) + k] = chi * dyadic[:, n] * fine[:, k]
    return W


def build_combined_basis_monomial(
    u: np.ndarray, n_dyadic: int, degree: int, chi: np.ndarray,
    clip_u: float = 1e-8, start_n: int = 2,
) -> np.ndarray:
    """Dyadic × monomial u^k."""
    return _build_combined_basis_generic(u, n_dyadic, degree, chi, build_monomial_basis_vander, clip_u, start_n)


def build_combined_basis_chebyshev(
    u: np.ndarray, n_dyadic: int, degree: int, chi: np.ndarray,
    clip_u: float = 1e-8, start_n: int = 2,
) -> np.ndarray:
    """Dyadic × shifted Chebyshev T_k(2u-1)."""
    return _build_combined_basis_generic(u, n_dyadic, degree, chi, build_chebyshev_basis_vander, clip_u, start_n)


def build_combined_basis_hermite(
    u: np.ndarray, n_dyadic: int, degree: int, chi: np.ndarray,
    clip_u: float = 1e-8, start_n: int = 2, alpha: float = 4.0,
) -> np.ndarray:
    """Dyadic × Gaussian-windowed Hermite-like (u-0.5)^k."""
    def _hermite_fn(u, deg):
        return build_hermite_basis_vander(u, deg, alpha=alpha)
    return _build_combined_basis_generic(u, n_dyadic, degree, chi, _hermite_fn, clip_u, start_n)


def build_combined_basis_bspline(
    u: np.ndarray, n_dyadic: int, degree: int, chi: np.ndarray,
    clip_u: float = 1e-8, start_n: int = 2,
) -> np.ndarray:
    """Dyadic × B-spline basis."""
    def _bspline_fn(u, deg):
        return build_bspline_basis_vander(u, deg)
    return _build_combined_basis_generic(u, n_dyadic, degree, chi, _bspline_fn, clip_u, start_n)


# =============================================================================
# Fitting functions
# =============================================================================

def fit_combined_basis(
    u: np.ndarray, f: np.ndarray, n_dyadic: int, degree_legendre: int,
    chi: np.ndarray, clip_u: float = 1e-8, start_n: int = 2,
    lambda_reg: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """
    Fit f(u) in combined dyadic×Legendre basis via least squares.
    Tikhonov regularization: c = argmin ||Wc-f||^2 + λ||c||^2.
    Returns (coefficients, residual_norm).
    """
    W = build_combined_basis(u, n_dyadic, degree_legendre, chi, clip_u=clip_u, start_n=start_n)
    if lambda_reg > 0:
        WTW = W.T @ W
        n = W.shape[1]
        WTW_reg = WTW + lambda_reg * np.eye(n)
        c = np.linalg.solve(WTW_reg, W.T @ f)
        res_norm = np.sqrt(np.sum((f - W @ c) ** 2))
    else:
        c, residuals, rank, s = np.linalg.lstsq(W, f, rcond=None)
        res_norm = np.sqrt(np.sum(residuals)) if len(residuals) > 0 else 0.0
    return c, res_norm


def fit_combined_basis_monomial(
    u: np.ndarray, f: np.ndarray, n_dyadic: int, degree: int,
    chi: np.ndarray, clip_u: float = 1e-8, start_n: int = 2,
    lambda_reg: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """
    Fit f(u) in combined dyadic×monomial basis via least squares.
    Tikhonov regularization: c = argmin ||Wc-f||^2 + λ||c||^2.
    Returns (coefficients, residual_norm).
    """
    W = build_combined_basis_monomial(u, n_dyadic, degree, chi, clip_u=clip_u, start_n=start_n)
    if lambda_reg > 0:
        WTW = W.T @ W
        n = W.shape[1]
        WTW_reg = WTW + lambda_reg * np.eye(n)
        c = np.linalg.solve(WTW_reg, W.T @ f)
        res_norm = np.sqrt(np.sum((f - W @ c) ** 2))
    else:
        c, residuals, rank, s = np.linalg.lstsq(W, f, rcond=None)
        res_norm = np.sqrt(np.sum(residuals)) if len(residuals) > 0 else 0.0
    return c, res_norm


def fit_generic_basis(
    W: np.ndarray, f: np.ndarray, lambda_reg: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """Generic least-squares fit with Tikhonov regularization."""
    if lambda_reg > 0:
        WTW = W.T @ W
        n = W.shape[1]
        WTW_reg = WTW + lambda_reg * np.eye(n)
        c = np.linalg.solve(WTW_reg, W.T @ f)
        res_norm = np.sqrt(np.sum((f - W @ c) ** 2))
    else:
        c, residuals, rank, s = np.linalg.lstsq(W, f, rcond=None)
        res_norm = np.sqrt(np.sum(residuals)) if len(residuals) > 0 else 0.0
    return c, res_norm


def fit_combined_basis_chebyshev(
    u: np.ndarray, f: np.ndarray, n_dyadic: int, degree: int,
    chi: np.ndarray, clip_u: float = 1e-8, start_n: int = 2,
    lambda_reg: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """Fit in dyadic × Chebyshev basis."""
    W = build_combined_basis_chebyshev(u, n_dyadic, degree, chi, clip_u=clip_u, start_n=start_n)
    return fit_generic_basis(W, f, lambda_reg)


def fit_combined_basis_hermite(
    u: np.ndarray, f: np.ndarray, n_dyadic: int, degree: int,
    chi: np.ndarray, clip_u: float = 1e-8, start_n: int = 2,
    lambda_reg: float = 1e-12, alpha: float = 4.0,
) -> Tuple[np.ndarray, float]:
    """Fit in dyadic × Hermite-like basis."""
    W = build_combined_basis_hermite(u, n_dyadic, degree, chi, clip_u=clip_u, start_n=start_n, alpha=alpha)
    return fit_generic_basis(W, f, lambda_reg)


def fit_combined_basis_bspline(
    u: np.ndarray, f: np.ndarray, n_dyadic: int, degree: int,
    chi: np.ndarray, clip_u: float = 1e-8, start_n: int = 2,
    lambda_reg: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """Fit in dyadic × B-spline basis."""
    W = build_combined_basis_bspline(u, n_dyadic, degree, chi, clip_u=clip_u, start_n=start_n)
    return fit_generic_basis(W, f, lambda_reg)


def fit_separable_svd(
    u: np.ndarray, f: np.ndarray, n_dyadic: int, degree_legendre: int,
    chi: np.ndarray, clip_u: float = 1e-8, start_n: int = 2,
    rcond_svd: float = 1e-10, als_iters: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Stable fitting for separable 4+4 model using truncated SVD + rank-1 projection.

    Steps:
    1. Build full N×16 design matrix with Chebyshev fine-tuning basis.
    2. Fit via truncated SVD pseudoinverse (handles ill-conditioning offline).
    3. Reshape 16 coefficients into 4×4 matrix C.
    4. Take rank-1 SVD of C: C ≈ a @ b.T.
    5. Optional ALS polishing to refine separable fit.

    Returns (a_vec, b_vec, rmse_full, rmse_separable) where:
      a_vec : dyadic coefficients (n_dyadic,)
      b_vec : fine-tuning coefficients (degree_legendre+1,)
      rmse_full : RMSE of full 16-coefficient fit
      rmse_separable : RMSE of separable 4+4 fit
    """
    # Step 1: Build combined basis with Chebyshev
    W = build_combined_basis_chebyshev(u, n_dyadic, degree_legendre, chi, clip_u=clip_u, start_n=start_n)

    # Step 2: Truncated SVD pseudoinverse fit
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    threshold = rcond_svd * s[0]
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)
    c_full = Vt.T @ (s_inv * (U.T @ f))
    f_fit_full = W @ c_full
    rmse_full = np.sqrt(np.mean((f - f_fit_full) ** 2))

    # Step 3: Reshape to 4×4 matrix
    Cmat = c_full.reshape(n_dyadic, degree_legendre + 1)

    # Step 4: Rank-1 SVD of coefficient matrix
    Uc, Sc, Vct = np.linalg.svd(Cmat, full_matrices=False)
    sigma0 = np.sqrt(Sc[0])
    a_vec = Uc[:, 0] * sigma0
    b_vec = Vct[0, :] * sigma0

    # Prepare dyadic and fine arrays for reconstruction
    u_safe = np.clip(u, clip_u, 1.0)
    v = 1.0 - u_safe
    dyadic = build_dyadic_basis_vander(v, n_dyadic, start_n=start_n)
    fine = build_chebyshev_basis_vander(u_safe, degree_legendre)

    # Step 5: Optional ALS polishing
    if als_iters > 0:
        for _ in range(als_iters):
            # Fix b, solve for a: f ≈ chi * sum_j(b_j * fine_j) * sum_i(a_i * dyadic_i)
            s_u = fine @ b_vec  # (n_points,)
            W_a = chi[:, None] * dyadic * s_u[:, None]
            a_vec, _, _, _ = np.linalg.lstsq(W_a, f, rcond=None)
            # Fix a, solve for b: f ≈ chi * sum_i(a_i * dyadic_i) * sum_j(b_j * fine_j)
            s_v = dyadic @ a_vec  # (n_points,)
            W_b = chi[:, None] * fine * s_v[:, None]
            b_vec, _, _, _ = np.linalg.lstsq(W_b, f, rcond=None)

    # Reconstruct and evaluate separable fit
    f_fit_sep = chi * (dyadic @ a_vec) * (fine @ b_vec)
    rmse_separable = np.sqrt(np.mean((f - f_fit_sep) ** 2))

    return a_vec, b_vec, rmse_full, rmse_separable


def fit_separable_als(
    u: np.ndarray, f: np.ndarray,
    dyadic: np.ndarray, fine: np.ndarray, chi: np.ndarray,
    a_init: np.ndarray = None, b_init: np.ndarray = None,
    max_iters: int = 10, tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Direct ALS fit for separable model:
      f(u) ≈ chi(u) * (dyadic @ a) * (fine @ b)

    No SVD detour — iteratively optimize a and b via alternating least squares.
    Returns (a_vec, b_vec, rmse).
    """
    n_dyadic = dyadic.shape[1]
    n_fine = fine.shape[1]
    a = a_init if a_init is not None else np.ones(n_dyadic) / n_dyadic
    b = b_init if b_init is not None else np.ones(n_fine) / n_fine

    prev_rmse = np.inf
    for iteration in range(max_iters):
        # Fix b, solve for a
        s_u = fine @ b  # (n_points,)
        W_a = chi[:, None] * dyadic * s_u[:, None]
        a, _, _, _ = np.linalg.lstsq(W_a, f, rcond=None)

        # Fix a, solve for b
        s_v = dyadic @ a  # (n_points,)
        W_b = chi[:, None] * fine * s_v[:, None]
        b, _, _, _ = np.linalg.lstsq(W_b, f, rcond=None)

        # Evaluate
        f_fit = chi * (dyadic @ a) * (fine @ b)
        rmse = np.sqrt(np.mean((f - f_fit) ** 2))

        if abs(prev_rmse - rmse) < tol * rmse:
            break
        prev_rmse = rmse

    return a, b, rmse


def optimize_separable_basis(
    curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    dyadic_ref: np.ndarray, cheb_ref: np.ndarray, chi_ref: np.ndarray,
    u_ref: np.ndarray,
    n_fine: int = 4, max_degree: int = 8,
    n_global_iters: int = 20, als_iters: int = 10,
    lambda_reg: float = 1e-12, verbose: bool = True,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]], float]:
    """
    Globally optimize separable fine-tuning basis F_j as linear combinations
    of Chebyshev polynomials T_0..T_max_degree.

    Model per curve c:
      f_c(u) ≈ chi(u) * (dyadic @ a_c) * (F @ b_c)
    where F_j(u) = Σ_k C[j,k] * T_k(u) for j=0..n_fine-1, k=0..max_degree.

    Total storage per curve: 4 + n_fine coefficients.
    Global storage: n_fine × (max_degree+1) matrix C.

    Returns (C, [(a_c, b_c) for each curve], total_rmse).
    """
    n_dyadic = dyadic_ref.shape[1]
    n_cheb = max_degree + 1
    n_curves = len(curves)

    # Initialize C: pick first n_fine Chebyshevs (identity on subspace)
    C = np.zeros((n_fine, n_cheb))
    for j in range(min(n_fine, n_cheb)):
        C[j, j] = 1.0

    # Evaluate F on reference grid for each iteration
    F_ref = cheb_ref @ C.T  # (n_points, n_fine)

    def _interp_to_grid(u_v, arr_ref):
        """Interpolate reference array (defined on u_ref) to curve grid u_v."""
        return np.interp(u_v, u_ref, arr_ref)

    # Fit all curves with initial F
    coeffs = []
    total_rmse = 0.0
    for u_v, f_v, chi_v in curves:
        dyadic = _interp_to_grid(u_v, dyadic_ref[:, 0])[:, None]
        for i in range(1, n_dyadic):
            dyadic = np.c_[dyadic, _interp_to_grid(u_v, dyadic_ref[:, i])]
        fine = _interp_to_grid(u_v, F_ref[:, 0])[:, None]
        for j in range(1, n_fine):
            fine = np.c_[fine, _interp_to_grid(u_v, F_ref[:, j])]
        a, b, rmse = fit_separable_als(u_v, f_v, dyadic, fine, chi_v,
                                       max_iters=als_iters)
        coeffs.append((a, b))
        total_rmse += rmse

    if verbose:
        print(f"  Initial: total={total_rmse:.4e}")

    for g_iter in range(n_global_iters):
        # === Step 1: Fix C, update all (a,b) via ALS ===
        F_ref = cheb_ref @ C.T
        coeffs = []
        total_rmse = 0.0
        for u_v, f_v, chi_v in curves:
            dyadic = _interp_to_grid(u_v, dyadic_ref[:, 0])[:, None]
            for i in range(1, n_dyadic):
                dyadic = np.c_[dyadic, _interp_to_grid(u_v, dyadic_ref[:, i])]
            fine = _interp_to_grid(u_v, F_ref[:, 0])[:, None]
            for j in range(1, n_fine):
                fine = np.c_[fine, _interp_to_grid(u_v, F_ref[:, j])]
            a, b, rmse = fit_separable_als(u_v, f_v, dyadic, fine, chi_v,
                                           max_iters=als_iters)
            coeffs.append((a, b))
            total_rmse += rmse

        # === Step 2: Fix all (a,b), update C via least squares ===
        # For each curve c: f_c = sum_{j,k} C[j,k] * [s_c * b_{cj} * T_k]
        # where s_c(u) = chi(u) * (dyadic @ a_c)
        # Stack all curves into one big least-squares problem for C.

        rows_A = 0
        for u_v, f_v, chi_v in curves:
            rows_A += len(u_v)

        A = np.zeros((rows_A, n_fine * n_cheb))
        rhs = np.zeros(rows_A)

        row = 0
        for idx, (u_v, f_v, chi_v) in enumerate(curves):
            a, b = coeffs[idx]
            n_pts = len(u_v)
            # Build dyadic on this grid
            dyadic = _interp_to_grid(u_v, dyadic_ref[:, 0])[:, None]
            for i in range(1, n_dyadic):
                dyadic = np.c_[dyadic, _interp_to_grid(u_v, dyadic_ref[:, i])]
            s_c = chi_v * (dyadic @ a)  # (n_pts,)
            # Build Chebyshev on this grid
            cheb = _interp_to_grid(u_v, cheb_ref[:, 0])[:, None]
            for k in range(1, n_cheb):
                cheb = np.c_[cheb, _interp_to_grid(u_v, cheb_ref[:, k])]

            # Column (j,k): s_c * b_j * T_k
            for j in range(n_fine):
                for k in range(n_cheb):
                    col = j * n_cheb + k
                    A[row:row+n_pts, col] = s_c * b[j] * cheb[:, k]
            rhs[row:row+n_pts] = f_v
            row += n_pts

        # Solve for C
        if lambda_reg > 0:
            ATA = A.T @ A
            ATA_reg = ATA + lambda_reg * np.eye(n_fine * n_cheb)
            c_vec = np.linalg.solve(ATA_reg, A.T @ rhs)
        else:
            c_vec, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
        C = c_vec.reshape(n_fine, n_cheb)

        if verbose:
            print(f"  Iter {g_iter+1}: total={total_rmse:.4e}")

    # Final fit with optimized C
    F_ref = cheb_ref @ C.T
    coeffs = []
    total_rmse = 0.0
    rmses = []
    for u_v, f_v, chi_v in curves:
        dyadic = _interp_to_grid(u_v, dyadic_ref[:, 0])[:, None]
        for i in range(1, n_dyadic):
            dyadic = np.c_[dyadic, _interp_to_grid(u_v, dyadic_ref[:, i])]
        fine = _interp_to_grid(u_v, F_ref[:, 0])[:, None]
        for j in range(1, n_fine):
            fine = np.c_[fine, _interp_to_grid(u_v, F_ref[:, j])]
        a, b, rmse = fit_separable_als(u_v, f_v, dyadic, fine, chi_v,
                                       max_iters=als_iters)
        coeffs.append((a, b))
        total_rmse += rmse
        rmses.append(rmse)

    return C, coeffs, total_rmse


def fit_tikhonov(
    W: np.ndarray, f: np.ndarray, lambda_reg: float = 1e-12,
) -> np.ndarray:
    """Standalone Tikhonov fit: c = (W^T W + λI)^(-1) W^T f."""
    WTW = W.T @ W
    n = W.shape[1]
    WTW_reg = WTW + lambda_reg * np.eye(n)
    return np.linalg.solve(WTW_reg, W.T @ f)


def fit_product_basis(
    u: np.ndarray, f: np.ndarray, m: int, degree: int,
    chi: np.ndarray, clip_u: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """Fit f(u) in product basis via least squares."""
    psi = build_product_basis(u, m, degree, chi, clip_u=clip_u)
    c, residuals, rank, s = np.linalg.lstsq(psi, f, rcond=None)
    res_norm = np.sqrt(np.sum(residuals)) if len(residuals) > 0 else 0.0
    return c, res_norm


def fit_dyadic_basis(
    u: np.ndarray, f: np.ndarray, m: int, n_terms: int,
    chi: np.ndarray, clip_u: float = 1e-8, start_n: int = 0,
) -> Tuple[np.ndarray, float]:
    """Fit f(u) in dyadic product basis via least squares."""
    psi = build_dyadic_product_basis(u, m, n_terms, chi, clip_u=clip_u, start_n=start_n)
    c, residuals, rank, s = np.linalg.lstsq(psi, f, rcond=None)
    res_norm = np.sqrt(np.sum(residuals)) if len(residuals) > 0 else 0.0
    return c, res_norm


def fit_pure_dyadic_basis(
    u: np.ndarray, f: np.ndarray, n_terms: int,
    chi: np.ndarray, clip_u: float = 1e-8, start_n: int = 0,
) -> Tuple[np.ndarray, float]:
    """Fit f(u) in pure dyadic basis (no u^m) via least squares."""
    psi = build_pure_dyadic_basis(u, n_terms, chi, clip_u=clip_u, start_n=start_n)
    c, residuals, rank, s = np.linalg.lstsq(psi, f, rcond=None)
    res_norm = np.sqrt(np.sum(residuals)) if len(residuals) > 0 else 0.0
    return c, res_norm


# =============================================================================
# Coefficient compression
# =============================================================================

def compress_coeffs_svd(
    c: np.ndarray, n_dyadic: int, degree_legendre: int, rank: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compress coefficient matrix C (n_dyadic × (degree+1)) via SVD.
    Returns (u_vec, v_vec, sigma) such that C ≈ sigma * u @ v.T.
    """
    Cmat = c.reshape(n_dyadic, degree_legendre + 1)
    U, S, Vt = np.linalg.svd(Cmat, full_matrices=False)
    u_vec = U[:, :rank] * S[:rank]
    v_vec = Vt[:rank, :].T
    sigma = S[:rank]
    return u_vec, v_vec, sigma


def reconstruct_from_svd(u_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """Reconstruct coefficient matrix from SVD compression."""
    if u_vec.ndim == 1:
        u_vec = u_vec[:, None]
    if v_vec.ndim == 1:
        v_vec = v_vec[:, None]
    return u_vec @ v_vec.T


# =============================================================================
# Conditioning and correlation analysis
# =============================================================================

def analyze_conditioning(
    W: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Analyze condition number of design matrix W.
    Returns (condition_number, singular_values, correlation_matrix).
    """
    cond = np.linalg.cond(W)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    corr = np.corrcoef(W.T)
    return cond, S, corr


def test_normalization(W: np.ndarray) -> Tuple[float, np.ndarray]:
    """Normalize columns to unit L2 norm and return (cond_norm, norms)."""
    norms = np.linalg.norm(W, axis=0)
    norms[norms == 0] = 1.0
    W_norm = W / norms[None, :]
    cond_norm = np.linalg.cond(W_norm)
    return cond_norm, norms


def test_orthogonalization(W: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """QR decomposition. Returns (cond_Q, cond_R, R)."""
    Q, R = np.linalg.qr(W)
    cond_Q = np.linalg.cond(Q)
    cond_R = np.linalg.cond(R)
    return cond_Q, cond_R, R


def analyze_correlation_structure(
    W: np.ndarray, n_dyadic: int, degree_legendre: int,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
    """
    Analyze block correlation structure of combined basis.
    Returns (corr_matrix, blocks_dict).
    """
    corr = np.corrcoef(W.T)
    n_legendre = degree_legendre + 1
    blocks = {}
    for n1 in range(n_dyadic):
        for n2 in range(n_dyadic):
            idx1 = slice(n1 * n_legendre, (n1 + 1) * n_legendre)
            idx2 = slice(n2 * n_legendre, (n2 + 1) * n_legendre)
            blocks[(n1, n2)] = corr[idx1, idx2]
    return corr, blocks


def compute_orthogonal_component(corr: np.ndarray) -> np.ndarray:
    """Compute s^2 = 1 - c^2 (orthogonal component). Mask diagonal with NaN."""
    s2 = 1 - corr ** 2
    np.fill_diagonal(s2, np.nan)
    return s2


def design_orthogonal_fine_tuning(
    u: np.ndarray, chi: np.ndarray, degree_leg: int = 3, threshold: float = 0.1,
) -> np.ndarray:
    """
    Design fine-tuning polynomials using Gram-Schmidt on monomials
    with weight chi. Returns phi (n_valid, degree_leg+1).
    """
    mask = u > threshold
    u_valid = u[mask]
    chi_valid = chi[mask]
    monomials = np.column_stack([u_valid ** k for k in range(degree_leg + 1)])
    phi = monomials.copy()
    for k in range(degree_leg + 1):
        for j in range(k):
            phi_j = phi[:, j] * chi_valid
            phi_k = phi[:, k] * chi_valid
            dot = np.dot(phi_j, phi_k)
            norm_sq = np.dot(phi_j, phi_j)
            if norm_sq > 1e-12:
                phi[:, k] -= (dot / norm_sq) * phi[:, j]
        phi_k = phi[:, k] * chi_valid
        norm = np.sqrt(np.dot(phi_k, phi_k))
        if norm > 1e-12:
            phi[:, k] /= norm
    return phi


def test_orthogonal_basis(
    u: np.ndarray, chi: np.ndarray, n_dyadic: int, degree_leg: int = 3, start_n: int = 2,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Test combined basis with orthogonal fine-tuning polynomials.
    Returns (cond_orth, corr_orth, phi).
    """
    phi = design_orthogonal_fine_tuning(u, chi, degree_leg)
    mask = u > 0.1
    u_valid = u[mask]
    phi_valid = phi
    chi_valid = chi[mask]
    u_safe = np.clip(u, 1e-8, 1.0)
    v = 1.0 - u_safe
    dyadic = build_dyadic_basis_vander(v, n_dyadic, start_n=start_n)
    dyadic_valid = dyadic[mask, :]
    n_legendre = degree_leg + 1
    W_orth = np.zeros((len(u_valid), n_dyadic * n_legendre))
    for n in range(n_dyadic):
        for k in range(n_legendre):
            W_orth[:, n * n_legendre + k] = chi_valid * dyadic_valid[:, n] * phi_valid[:, k]
    cond_orth = np.linalg.cond(W_orth)
    corr_orth = np.corrcoef(W_orth.T)
    return cond_orth, corr_orth, phi


# =============================================================================
# Global basis and evaluation (old pipeline)
# =============================================================================

def build_global_basis(
    u: np.ndarray, U: np.ndarray, degree: int, chi: np.ndarray, m: int, L: int,
) -> np.ndarray:
    """Evaluate global basis Q_l(u) = chi(u) * u^m * sum_k U[k,l] * P_k(2u-1). Returns (n_points, L)."""
    legendre = shifted_legendre_vander(u, degree)
    u_safe = np.clip(u, 1e-8, 1.0)
    prefactor = u_safe ** m
    poly_part = legendre @ U[:, :L]
    return chi[:, None] * prefactor[:, None] * poly_part


def reconstruct(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, mean: np.ndarray, L: int) -> np.ndarray:
    """Reconstruct A from top L components."""
    return U[:, :L] @ np.diag(S[:L]) @ Vt[:L, :] + mean[:, None]


# =============================================================================
# Unified basis specification and composition
# =============================================================================

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BasisComponent:
    """One family of basis functions + optional per-family envelope."""
    family: str
    params: Dict[str, Any] = field(default_factory=dict)
    envelope: Optional[Dict[str, Any]] = None


_BASIS_GENERATORS: Dict[str, Callable] = {
    "constant": lambda u, params: np.ones((len(u), 1)),
    "dyadic": lambda u, params: build_dyadic_basis_vander(
        1.0 - np.clip(u, 1e-8, 1.0),
        params.get("n", 4),
        start_n=params.get("start_n", 2),
    ),
    "chebyshev": lambda u, params: build_chebyshev_basis_vander(
        np.clip(u, 1e-8, 1.0),
        params.get("degree", 3),
    ),
    "legendre": lambda u, params: shifted_legendre_vander(
        u,
        params.get("degree", 3),
    ),
    "monomial": lambda u, params: build_monomial_basis_vander(
        np.clip(u, 1e-8, 1.0),
        params.get("degree", 3),
    ),
    "hermite": lambda u, params: build_hermite_basis_vander(
        u,
        params.get("degree", 3),
        alpha=params.get("alpha", 4.0),
    ),
    "bspline": lambda u, params: build_bspline_basis_vander(
        u,
        params.get("degree", 3),
    ),
    "custom": lambda u, params: build_custom_basis_vander(
        u,
        params.get("Rc", 10.0),
    ),
}


def _apply_envelope(phi: np.ndarray, u: np.ndarray, env_spec: Dict[str, Any]) -> np.ndarray:
    """Apply per-family envelope to basis matrix."""
    typ = env_spec.get("type", "none")
    if typ == "none":
        return phi
    u_safe = np.clip(u, 1e-8, 1.0)
    v = 1.0 - u_safe
    if typ == "v_power":
        w = v ** env_spec.get("power", 2)
    elif typ == "chi":
        w = envelope(u, power=env_spec.get("power", 2), variant=env_spec.get("variant", "linear"))
    elif typ == "u_power":
        w = u_safe ** env_spec.get("power", 1)
    else:
        raise ValueError(f"Unknown envelope type: {typ}")
    return phi * w[:, None]


def generate_family(u: np.ndarray, comp: BasisComponent) -> np.ndarray:
    """Generate one family's basis matrix (envelope already applied if specified)."""
    if comp.family not in _BASIS_GENERATORS:
        raise ValueError(f"Unknown basis family: {comp.family}. Available: {list(_BASIS_GENERATORS.keys())}")
    phi = _BASIS_GENERATORS[comp.family](u, comp.params)
    if comp.envelope is not None:
        phi = _apply_envelope(phi, u, comp.envelope)
    return phi


BasisMatrix = Union[np.ndarray, List[np.ndarray]]


def build_basis_matrix(
    u: np.ndarray,
    chi: Optional[np.ndarray],
    components: List[BasisComponent],
    mode: str = "concat",
) -> BasisMatrix:
    """
    Assemble basis matrix from components.

    mode : "concat" | "product" | "separable"
        - concat   : horizontal stack of all families
        - product  : tensor product of all families, multiplied by chi
        - separable: return list of family matrices for ALS
    """
    families = [generate_family(u, c) for c in components]

    if mode == "concat":
        W = np.hstack(families)
        if chi is not None:
            W = W * chi[:, None]
        return W

    elif mode == "product":
        if len(families) == 0:
            raise ValueError("Need at least one component for product mode")
        W = families[0]
        for phi in families[1:]:
            n1, m1 = W.shape
            n2, m2 = phi.shape
            if n1 != n2:
                raise ValueError("Family matrices must have same number of points")
            W_new = np.zeros((n1, m1 * m2))
            for i in range(m1):
                for j in range(m2):
                    W_new[:, i * m2 + j] = W[:, i] * phi[:, j]
            W = W_new
        if chi is not None:
            W = W * chi[:, None]
        return W

    elif mode == "separable":
        if chi is not None:
            families = [phi * chi[:, None] for phi in families]
        return families

    else:
        raise ValueError(f"Unknown mode: {mode}")


def fit_basis(
    u: np.ndarray,
    f: np.ndarray,
    chi: Optional[np.ndarray],
    components: List[BasisComponent],
    mode: str = "concat",
    method: str = "lstsq",
    **kwargs: Any,
) -> Tuple[Any, np.ndarray, float]:
    """
    Fit f(u) with basis specified by components.

    method : "lstsq" | "tikhonov" | "als"
    Returns (coeffs, f_fit, rmse).
    """
    W = build_basis_matrix(u, chi, components, mode)

    if mode == "separable":
        if method != "als":
            raise ValueError("Separable mode requires ALS method")
        if len(W) != 2:
            raise ValueError("ALS requires exactly 2 components")
        als_kwargs = {k: v for k, v in kwargs.items() if k in ("max_iters", "tol", "a_init", "b_init")}
        a, b, rmse = fit_separable_als(u, f, W[0], W[1], np.ones(len(u)), **als_kwargs)
        f_fit = (W[0] @ a) * (W[1] @ b)
        return (a, b), f_fit, rmse

    if method == "lstsq":
        c, _, _, _ = np.linalg.lstsq(W, f, rcond=None)
    elif method == "tikhonov":
        c = fit_tikhonov(W, f, lambda_reg=kwargs.get("lambda_reg", 1e-12))
    else:
        raise ValueError(f"Unknown method: {method}")

    f_fit = W @ c
    rmse = np.sqrt(np.mean((f - f_fit) ** 2))
    return c, f_fit, rmse


# =============================================================================
# Basis spec string parsing
# =============================================================================

def parse_basis_spec(spec_str: str) -> Tuple[List[BasisComponent], str]:
    """
    Parse a basis specification string.

    Format examples:
      "dyadic:4+legendre:3"                 -> product mode
      "dyadic:4|legendre:3"                 -> concat mode
      "dyadic:4;legendre:3"                 -> separable mode
      "chebyshev:3[v_power:4]"              -> Chebyshev with v^4 envelope
      "constant:1[v_power:4]+chebyshev:3"   -> v^4 * Chebyshev (product)
    """
    spec_str = spec_str.strip()
    if ";" in spec_str:
        mode = "separable"
        parts = [p.strip() for p in spec_str.split(";")]
    elif "|" in spec_str:
        mode = "concat"
        parts = [p.strip() for p in spec_str.split("|")]
    else:
        mode = "product"
        parts = [p.strip() for p in spec_str.split("+")]

    components = [_parse_component(p) for p in parts]
    return components, mode


def _parse_component(part: str) -> BasisComponent:
    """Parse 'chebyshev:3[v_power:4]'."""
    envelope = None
    if "[" in part and "]" in part:
        idx_start = part.index("[")
        idx_end = part.index("]")
        env_str = part[idx_start + 1:idx_end]
        part = part[:idx_start] + part[idx_end + 1:]
        envelope = _parse_envelope(env_str)
    if ":" not in part:
        raise ValueError(f"Component spec must be 'family:param': {part}")
    family, param_str = part.split(":", 1)
    params = _parse_params(param_str)
    return BasisComponent(family=family.strip(), params=params, envelope=envelope)


def _parse_envelope(env_str: str) -> Dict[str, Any]:
    """Parse 'v_power:4' or 'chi:power=2,variant=linear'."""
    if ":" not in env_str:
        raise ValueError(f"Envelope spec must be 'type:arg': {env_str}")
    typ, arg = env_str.split(":", 1)
    typ = typ.strip()
    arg = arg.strip()
    if typ == "v_power":
        return {"type": "v_power", "power": float(arg)}
    elif typ == "u_power":
        return {"type": "u_power", "power": float(arg)}
    elif typ == "chi":
        kwargs = {}
        for kv in arg.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                kwargs[k.strip()] = float(v) if "." in v else int(v)
        return {"type": "chi", **kwargs}
    else:
        raise ValueError(f"Unknown envelope type: {typ}")


def _parse_params(param_str: str) -> Dict[str, Any]:
    """Parse '4' or 'degree=3,alpha=4.0'."""
    params = {}
    if "=" not in param_str:
        val = int(param_str) if "." not in param_str else float(param_str)
        params["n"] = val
        params["degree"] = val
        return params
    for kv in param_str.split(","):
        kv = kv.strip()
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            params[k] = int(v) if "." not in v else float(v)
        except ValueError:
            params[k] = v
    return params


# =============================================================================
# Sweep config generation
# =============================================================================

def make_sweep_configs(sweep_spec: str) -> List[Tuple[str, List[BasisComponent]]]:
    """
    Generate list of (label, components) from a sweep spec.

    Examples:
      "dyadic:2..6+legendre:1..5"     -> product sweep over combos
      "chebyshev:3[v_power:4,6,8]"    -> single family, multiple envelope params
    """
    configs = []
    if "|" in sweep_spec:
        sep = "|"
    elif ";" in sweep_spec:
        sep = ";"
    else:
        sep = "+"

    has_range = any(".." in p for p in sweep_spec.replace("[", "").replace("]", "").split(sep))

    if has_range:
        raw_parts = [p.strip() for p in sweep_spec.split(sep)]
        dims = []
        for dim_spec in raw_parts:
            if ".." in dim_spec:
                family, range_str = dim_spec.split(":", 1)
                family = family.strip()
                if "[" in range_str:
                    base_range, bracket = range_str.split("[", 1)
                    start, end = base_range.split("..", 1)
                    start, end = int(start), int(end)
                    dims.append((family, list(range(start, end + 1)), f"[{bracket}"))
                else:
                    start, end = range_str.split("..", 1)
                    start, end = int(start), int(end)
                    dims.append((family, list(range(start, end + 1)), None))
            else:
                comp, _ = parse_basis_spec(dim_spec)
                dims.append(("fixed", [comp], None))

        from itertools import product
        ranges = [d[1] for d in dims]
        for combo in product(*ranges):
            comps = []
            label_parts = []
            for (family, _, bracket), val in zip(dims, combo):
                if family == "fixed":
                    comps.extend(val)
                    label_parts.append(_comp_label(val[0]))
                else:
                    params = {"n": val, "degree": val}
                    if bracket is not None:
                        env = _parse_envelope(bracket[1:-1])
                        comp = BasisComponent(family, params, envelope=env)
                    else:
                        comp = BasisComponent(family, params)
                    comps.append(comp)
                    label_parts.append(f"{family}{val}")
            configs.append(("_".join(label_parts), comps))

    elif "[" in sweep_spec and "," in sweep_spec.split("[")[1].split("]")[0]:
        base, env_part = sweep_spec.split("[", 1)
        env_vals = env_part.rstrip("]").split(",")
        for val in env_vals:
            val = val.strip()
            if ":" in val:
                spec = f"{base}[{val}]"
                label = val.replace(":", "")
            else:
                spec = f"{base}[v_power:{val}]"
                label = f"v{val}"
            comps, _ = parse_basis_spec(spec)
            configs.append((label, comps))

    else:
        comps, _ = parse_basis_spec(sweep_spec)
        configs.append((sweep_spec, comps))

    return configs


def _comp_label(comp: BasisComponent) -> str:
    p = comp.params
    if "n" in p:
        return f"{comp.family}{p['n']}"
    if "degree" in p:
        return f"{comp.family}{p['degree']}"
    return comp.family


def count_coefficients(components: List[BasisComponent], mode: str) -> int:
    sizes = []
    for c in components:
        p = c.params
        if c.family == "dyadic":
            sizes.append(p.get("n", 4))
        elif c.family in ("chebyshev", "legendre", "monomial", "hermite", "bspline"):
            sizes.append(p.get("degree", 3) + 1)
        elif c.family == "constant":
            sizes.append(1)
        elif c.family == "custom":
            sizes.append(7)
        else:
            sizes.append(1)

    if mode == "concat":
        return sum(sizes)
    elif mode == "product":
        prod = 1
        for s in sizes:
            prod *= s
        return prod
    elif mode == "separable":
        return sum(sizes)
    return 0
