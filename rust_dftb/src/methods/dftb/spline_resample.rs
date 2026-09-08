//! B-spline resampling and canonical C² evaluation for SK tables.
//!
//! Given high-resolution SK table data on a uniform grid, this module:
//! 1. Fits a natural cubic spline to the original data (for accurate resampling)
//! 2. Evaluates the spline at N_target equally-spaced points
//! 3. Converts function values to cubic B-spline control points so the GPU
//!    4-point stencil reproduces the original function exactly (to f32 precision)
//!
//! The GPU uses cubic B-spline interpolation with a 4-point stencil:
//!   val = w0*f[i-1] + w1*f[i] + w2*f[i+1] + w3*f[i+2]
//! where w0..w3 are the cubic B-spline basis weights.
//! This requires only 1 float per node (vs 2 for Hermite), and the 4-point
//! stencil shares data between neighbors, so total local memory is lower.
//!
//! This allows representing SK tables with 32-64 points instead of 100-500,
//! reducing GPU local memory usage by 4-8×.
//!
//! # Canonical C² evaluator (P1, manifest v3 §4.2)
//!
//! `bspline3_eval_v_d1_d2` returns `(V, dV/dr, d²V/dr²)` from the same cubic
//! B-spline control points. The derivatives are **analytic** derivatives of
//! exactly the same interpolated function — no numerical finite differences.
//! This is mandatory for Hessian-quality forces: a C¹-only representation
//! (Hermite with independently estimated slopes) has discontinuous V'' at
//! knots, producing artificial Hessian curvature/wiggles.
//!
//! The natural cubic spline boundary (d²=0 at endpoints) is enforced via
//! phantom control points `c_{-1} = 2·c_0 − c_1`, `c_n = 2·c_{n-1} − c_{n-2}`.

// --- Natural cubic spline (used only for accurate resampling on host) ---

pub fn cubic_spline_d2_uniform(y: &[f64], h: f64) -> Vec<f64> {
    let n = y.len();
    if n < 3 {
        return vec![0.0; n];
    }

    let m = n - 2;
    let mut diag = vec![4.0; m];
    let mut lower = vec![1.0; m.saturating_sub(1)];
    let mut upper = vec![1.0; m.saturating_sub(1)];
    let mut rhs = vec![0.0; m];

    for i in 0..m {
        rhs[i] = 6.0 * (y[i + 2] - 2.0 * y[i + 1] + y[i]) / (h * h);
    }

    for i in 1..m {
        let w = lower[i - 1] / diag[i - 1];
        diag[i] -= w * upper[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }

    let mut d2_inner = vec![0.0; m];
    d2_inner[m - 1] = rhs[m - 1] / diag[m - 1];
    for i in (0..m - 1).rev() {
        d2_inner[i] = (rhs[i] - upper[i] * d2_inner[i + 1]) / diag[i];
    }

    let mut d2 = vec![0.0; n];
    d2[1..n - 1].copy_from_slice(&d2_inner);
    d2
}

fn cubic_spline_eval(y: &[f64], d2: &[f64], h: f64, r: f64) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    if r <= 0.0 {
        return y[0];
    }
    let r_max = (n - 1) as f64 * h;
    if r >= r_max {
        return y[n - 1];
    }

    let x = r / h;
    let i = x as usize;
    let i = if i >= n - 1 { n - 2 } else { i };
    let t = x - i as f64;
    let a = 1.0 - t;
    let h2_6 = h * h / 6.0;
    a * y[i] + t * y[i + 1] + ((a * a * a - a) * d2[i] + (t * t * t - t) * d2[i + 1]) * h2_6
}

// --- B-spline resampling ---

/// Solve the tridiagonal system that converts function values to cubic B-spline
/// control points. For uniform cubic B-splines:
///   f_i = (c_{i-1} + 4*c_i + c_{i+1}) / 6
/// with boundary conditions c_0 = f_0, c_{n-1} = f_{n-1} (endpoint interpolation).
///
/// This is essential because the GPU's 4-point B-spline stencil treats stored
/// values as control points, not function values. Without this conversion, the
/// GPU produces a smoothed (approximate) version of the data, causing ~1e-3
/// parity error even with dense grids.
fn function_to_bspline_control_points(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    if n <= 2 {
        return f.to_vec();
    }

    // Interior system: c_{i-1} + 4*c_i + c_{i+1} = 6*f_i, for i=1..n-2
    // Boundary: c_0 = f_0, c_{n-1} = f_{n-1}
    let m = n - 2;
    let mut diag = vec![4.0f64; m];
    let mut upper = vec![1.0f64; m.saturating_sub(1)];
    let mut lower = vec![1.0f64; m.saturating_sub(1)];
    let mut rhs = vec![0.0f64; m];

    for i in 0..m {
        rhs[i] = 6.0 * f[i + 1];
    }
    // Adjust for known boundary values
    rhs[0] -= f[0];
    rhs[m - 1] -= f[n - 1];

    // Thomas algorithm
    for i in 1..m {
        let w = lower[i - 1] / diag[i - 1];
        diag[i] -= w * upper[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }

    let mut c_inner = vec![0.0f64; m];
    c_inner[m - 1] = rhs[m - 1] / diag[m - 1];
    for i in (0..m - 1).rev() {
        c_inner[i] = (rhs[i] - upper[i] * c_inner[i + 1]) / diag[i];
    }

    let mut c = vec![0.0f64; n];
    c[0] = f[0];
    c[n - 1] = f[n - 1];
    c[1..n - 1].copy_from_slice(&c_inner);
    c
}

/// Resample a function from a uniform grid to a target number of points.
///
/// Uses natural cubic spline for accurate interpolation of the original data,
/// evaluates at N_target equally-spaced points, then converts the function
/// values to B-spline control points so the GPU 4-point stencil reproduces
/// the original function exactly (to f32 precision).
///
/// Returns (bspline_control_points, new_dr).
pub fn resample_bspline(
    y_orig: &[f64],
    dr_orig: f64,
    n_target: usize,
) -> (Vec<f32>, f32) {
    let n_orig = y_orig.len();
    if n_orig == 0 {
        return (vec![], 0.0);
    }

    let d2_orig = cubic_spline_d2_uniform(y_orig, dr_orig);

    let r_max = (n_orig - 1) as f64 * dr_orig;
    let dr_new = if n_target > 1 {
        r_max / (n_target - 1) as f64
    } else {
        r_max
    };

    let mut y_new = vec![0.0f64; n_target];
    for i in 0..n_target {
        let r = i as f64 * dr_new;
        y_new[i] = cubic_spline_eval(y_orig, &d2_orig, dr_orig, r);
    }

    // Convert function values to B-spline control points so the GPU 4-point
    // stencil reproduces the original function exactly (to f32 precision).
    let ctrl = function_to_bspline_control_points(&y_new);

    let y_f32: Vec<f32> = ctrl.iter().map(|&v| v as f32).collect();
    (y_f32, dr_new as f32)
}

/// Resample a single SK integral column.
/// Returns (resampled_values, new_dr).
pub fn resample_sk_column(
    values: &[f64],
    dr_orig: f64,
    n_target: usize,
) -> (Vec<f32>, f32) {
    resample_bspline(values, dr_orig, n_target)
}

// ============================================================================
// Canonical C² cubic B-spline evaluator: V, V', V'' from the same controls
// (P1, manifest v3 §4.2)
// ============================================================================

/// Core cubic cardinal B-spline evaluator for one interval.
///
/// Given four control coefficients `c0..c3` surrounding the interval and a
/// fractional position `t ∈ [0,1)`, returns `(V, dV/dr, d²V/dr²)`.
///
/// `inv_dr = 1/Δr` where `Δr` is the uniform grid spacing.
///
/// The basis functions are:
/// ```text
/// B0 = (1-t)³ / 6
/// B1 = (3t³ - 6t² + 4) / 6
/// B2 = (-3t³ + 3t² + 3t + 1) / 6
/// B3 = t³ / 6
/// ```
/// with analytic derivatives `B'_k` and `B''_k` (see code).
///
/// Partition of unity: `Σ B_k = 1`, `Σ B'_k = 0`, `Σ B''_k = 0` — tested in
/// `test_bspline3_partition_of_unity`.
pub fn bspline3_eval_v_d1_d2(
    c0: f64, c1: f64, c2: f64, c3: f64,
    t: f64, inv_dr: f64,
) -> (f64, f64, f64) {
    let t2 = t * t;
    let t3 = t2 * t;
    let u = 1.0 - t;

    // Value basis (partition of unity: Σ B_k = 1)
    let b0 = u * u * u / 6.0;
    let b1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;
    let b2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;
    let b3 = t3 / 6.0;

    // First derivative basis (Σ B'_k = 0)
    let d0 = -0.5 * u * u;
    let d1 = 1.5 * t2 - 2.0 * t;
    let d2 = -1.5 * t2 + t + 0.5;
    let d3 = 0.5 * t2;

    // Second derivative basis (Σ B''_k = 0)
    let dd0 = 1.0 - t;
    let dd1 = 3.0 * t - 2.0;
    let dd2 = -3.0 * t + 1.0;
    let dd3 = t;

    let v = c0 * b0 + c1 * b1 + c2 * b2 + c3 * b3;
    let dv = (c0 * d0 + c1 * d1 + c2 * d2 + c3 * d3) * inv_dr;
    let ddv = (c0 * dd0 + c1 * dd1 + c2 * dd2 + c3 * dd3) * inv_dr * inv_dr;
    (v, dv, ddv)
}

/// Evaluate the canonical C² cubic B-spline at distance `r`, using the full
/// control point array `controls` with uniform spacing `dr`.
///
/// Handles interval lookup and natural-spline boundary phantom control points
/// (`c_{-1} = 2·c_0 − c_1`, `c_n = 2·c_{n-1} − c_{n-2}` for d²=0 at endpoints).
///
/// Returns `(V, dV/dr, d²V/dr²)`. For `r` outside `[0, r_max]` the value and
/// derivatives are zero (matching SK cutoff behaviour).
pub fn bspline3_eval_at(controls: &[f64], dr: f64, r: f64) -> (f64, f64, f64) {
    let n = controls.len();
    if n == 0 || dr <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let r_max = (n - 1) as f64 * dr;
    if r < 0.0 || r > r_max {
        return (0.0, 0.0, 0.0);
    }
    let inv_dr = 1.0 / dr;
    let x = r / dr;
    let i = if x >= (n - 1) as f64 { n - 2 } else { x as usize };
    let t = x - i as f64;

    // Four controls: c[i-1], c[i], c[i+1], c[i+2].
    // Phantom points at boundaries enforce natural spline (d²=0 at endpoints).
    let c0 = if i == 0 { 2.0 * controls[0] - controls[1] } else { controls[i - 1] };
    let c1 = controls[i];
    let c2 = controls[i + 1];
    let c3 = if i + 2 >= n { 2.0 * controls[n - 1] - controls[n - 2] } else { controls[i + 2] };

    bspline3_eval_v_d1_d2(c0, c1, c2, c3, t, inv_dr)
}

/// Same as `bspline3_eval_at` but takes f32 control points (the GPU-side
/// format). Computes in f64 for accuracy, returns f64.
pub fn bspline3_eval_at_f32(controls: &[f32], dr: f32, r: f64) -> (f64, f64, f64) {
    let ctrl_f64: Vec<f64> = controls.iter().map(|&v| v as f64).collect();
    bspline3_eval_at(&ctrl_f64, dr as f64, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_bspline_sin() {
        // Original: 200 points of sin(x), resample to 32
        let h_orig = 0.05;
        let y_orig: Vec<f64> = (0..200).map(|i| (i as f64 * h_orig).sin()).collect();
        let (y_new, dr_new) = resample_bspline(&y_orig, h_orig, 32);

        assert_eq!(y_new.len(), 32);

        // Check that resampled values are close to original at matching points
        let r_max = 199.0 * h_orig;
        let dr = r_max / 31.0;
        for i in 0..32 {
            let r = i as f64 * dr;
            let exact = r.sin();
            let val = y_new[i] as f64;
            assert!(
                (val - exact).abs() < 0.01,
                "point {}: {} vs {}",
                i,
                val,
                exact
            );
        }
    }

    #[test]
    fn test_resample_preserves_endpoints() {
        let h = 0.1;
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * h).exp()).collect();
        let (y_new, _) = resample_bspline(&y, h, 32);

        assert!((y_new[0] - y[0] as f32).abs() < 1e-3);
        assert!((y_new[31] - y[99] as f32).abs() < 1e-3);
    }

    #[test]
    fn test_resample_exp_decay() {
        // Typical SK table shape: exponential decay
        let h = 0.01;
        let y: Vec<f64> = (0..500).map(|i| (-i as f64 * h * 2.0).exp()).collect();
        let (y_new, dr_new) = resample_bspline(&y, h, 64);

        assert_eq!(y_new.len(), 64);
        assert!((y_new[0] - 1.0).abs() < 1e-4);
        assert!(y_new[63].abs() < 1e-4);
    }

    // ==================================================================
    // P1 tests: partition of unity, knot continuity, V/V'/V'' accuracy
    // (manifest v3 §4.2 Gate A)
    // ==================================================================

    /// Partition of unity: Σ B_k(t) = 1, Σ B'_k(t) = 0, Σ B''_k(t) = 0
    /// for many random t. This catches transcription bugs like the v2
    /// `B2 = (-3t³ + 3t² + 3 + 1)/6` (missing `*t`).
    #[test]
    fn test_bspline3_partition_of_unity() {
        let ctrl = [1.0_f64, 1.0, 1.0, 1.0]; // all-ones controls → V should be 1
        let inv_dr = 1.0_f64;
        let n_tests = 1000;
        for k in 0..n_tests {
            let t = (k as f64 + 0.5) / n_tests as f64; // (0, 1)
            let (v, dv, ddv) = bspline3_eval_v_d1_d2(
                ctrl[0], ctrl[1], ctrl[2], ctrl[3], t, inv_dr,
            );
            assert!((v - 1.0).abs() < 1e-12, "ΣB_k != 1 at t={t}: got V={v}");
            assert!(dv.abs() < 1e-12, "ΣB'_k != 0 at t={t}: got dV={dv}");
            assert!(ddv.abs() < 1e-12, "ΣB''_k != 0 at t={t}: got d²V={ddv}");
        }
        // Also test at t=0 and t→1⁻
        let (v0, dv0, ddv0) = bspline3_eval_v_d1_d2(1.0, 1.0, 1.0, 1.0, 0.0, 1.0);
        assert!((v0 - 1.0).abs() < 1e-12, "ΣB_k(0) != 1: {v0}");
        assert!(dv0.abs() < 1e-12, "ΣB'_k(0) != 0: {dv0}");
        assert!(ddv0.abs() < 1e-12, "ΣB''_k(0) != 0: {ddv0}");
        let (v1, dv1, ddv1) = bspline3_eval_v_d1_d2(1.0, 1.0, 1.0, 1.0, 0.999, 1.0);
        assert!((v1 - 1.0).abs() < 1e-10, "ΣB_k(1⁻) != 1: {v1}");
        assert!(dv1.abs() < 1e-10, "ΣB'_k(1⁻) != 0: {dv1}");
        assert!(ddv1.abs() < 1e-10, "ΣB''_k(1⁻) != 0: {ddv1}");
    }

    /// C² continuity at knots: V, V', V'' from the left limit of interval i
    /// must match the right limit of interval i+1 (= left limit at t=0).
    ///
    /// The cardinal cubic B-spline is exactly C² at knots, so we compare
    /// t=1.0 (end of interval i) against t=0.0 (start of interval i+1).
    #[test]
    fn test_bspline3_knot_continuity() {
        // Arbitrary control points with some variation
        let ctrl = [0.1, 0.5, -0.3, 0.8, 0.2, -0.1, 0.6, 0.0];
        let dr = 0.1_f64;
        let inv_dr = 1.0 / dr;
        let n = ctrl.len();
        // Check at each interior knot (between intervals i and i+1)
        for i in 1..(n - 3) {
            // Left limit of interval i: t = 1.0 (end of interval)
            let (v_left, dv_left, ddv_left) = bspline3_eval_v_d1_d2(
                ctrl[i - 1], ctrl[i], ctrl[i + 1], ctrl[i + 2],
                1.0, inv_dr,
            );
            // Right limit of interval i+1: t = 0.0 (start of interval)
            let (v_right, dv_right, ddv_right) = bspline3_eval_v_d1_d2(
                ctrl[i], ctrl[i + 1], ctrl[i + 2], ctrl[i + 3],
                0.0, inv_dr,
            );
            // V, V', V'' should be continuous to machine precision for a
            // cardinal cubic B-spline (it is exactly C² at knots).
            let tol = 1e-12;
            assert!((v_left - v_right).abs() < tol,
                "V discontinuous at knot {i}: left={v_left}, right={v_right}");
            assert!((dv_left - dv_right).abs() < tol,
                "V' discontinuous at knot {i}: left={dv_left}, right={dv_right}");
            assert!((ddv_left - ddv_right).abs() < tol,
                "V'' discontinuous at knot {i}: left={ddv_left}, right={ddv_right}");
        }
    }

    /// Compare f64 B-spline V, V', V'' against the natural cubic spline
    /// through the SAME grid points. A uniform cardinal cubic B-spline with
    /// natural boundary phantom points IS the natural cubic spline through
    /// the same points — they are the same C² function in a different basis.
    /// So V, V', V'' should match to ~machine precision.
    ///
    /// This is the Gate A "f64 canonical spline vs reference interpolation"
    /// test: the B-spline representation must reproduce the same C² function
    /// that the natural cubic spline defines.
    #[test]
    fn test_bspline3_vs_natural_cubic_spline_same_grid() {
        // Typical SK-like data: smooth exponential decay
        let h_orig = 0.01;
        let y_orig: Vec<f64> = (0..500).map(|i| (-i as f64 * h_orig * 2.0).exp()).collect();

        // Resample to 64 nodes
        let n_target = 64;
        let r_max = (y_orig.len() - 1) as f64 * h_orig;
        let dr_new = r_max / (n_target - 1) as f64;

        // Get the 64 function values (sampled from the high-res natural spline)
        let d2_orig = cubic_spline_d2_uniform(&y_orig, h_orig);
        let mut y_new = vec![0.0f64; n_target];
        for i in 0..n_target {
            let r = i as f64 * dr_new;
            y_new[i] = cubic_spline_eval(&y_orig, &d2_orig, h_orig, r);
        }

        // Fit a natural cubic spline through the SAME 64 points
        let d2_new = cubic_spline_d2_uniform(&y_new, dr_new);

        // Get B-spline control points (in f64, before the f32 cast)
        let ctrl = function_to_bspline_control_points(&y_new);

        // Compare V, V', V'' at many points — should match to ~1e-12
        let n_check = 500;
        let mut max_err_v = 0.0_f64;
        let mut max_err_dv = 0.0_f64;
        let mut max_err_ddv = 0.0_f64;
        for k in 1..(n_check - 1) {
            let r = k as f64 * r_max / n_check as f64;
            // Natural cubic spline reference through 64 points
            let v_ref = cubic_spline_eval(&y_new, &d2_new, dr_new, r);
            // V' from natural cubic spline
            let x = r / dr_new;
            let i = if x >= (n_target - 1) as f64 { n_target - 2 } else { x as usize };
            let t = x - i as f64;
            let a = 1.0 - t;
            let dv_ref = (-y_new[i] + y_new[i + 1]
                + ((-3.0 * a * a + 1.0) * d2_new[i] + (3.0 * t * t - 1.0) * d2_new[i + 1])
                    * dr_new * dr_new / 6.0) / dr_new;
            // V'' from natural cubic spline
            let ddv_ref = a * d2_new[i] + t * d2_new[i + 1];

            // B-spline evaluation
            let (v_bs, dv_bs, ddv_bs) = bspline3_eval_at(&ctrl, dr_new, r);

            max_err_v = max_err_v.max((v_bs - v_ref).abs());
            max_err_dv = max_err_dv.max((dv_bs - dv_ref).abs());
            max_err_ddv = max_err_ddv.max((ddv_bs - ddv_ref).abs());
        }
        eprintln!("B-spline vs natural cubic spline (same 64-point grid):");
        eprintln!("  max|ΔV|   = {max_err_v:.3e}");
        eprintln!("  max|ΔV'|  = {max_err_dv:.3e}");
        eprintln!("  max|ΔV''| = {max_err_ddv:.3e}");

        // The B-spline IS the natural cubic spline through the same points,
        // so these should match to near-machine precision.
        assert!(max_err_v < 1e-10, "V error too large: {max_err_v:.3e}");
        assert!(max_err_dv < 1e-9, "V' error too large: {max_err_dv:.3e}");
        assert!(max_err_ddv < 1e-9, "V'' error too large: {max_err_ddv:.3e}");
    }

    /// Verify that V'' is continuous across knots when using the full
    /// `bspline3_eval_at` (with phantom boundary points).
    #[test]
    fn test_bspline3_eval_at_continuity() {
        let h_orig = 0.01;
        let y_orig: Vec<f64> = (0..200).map(|i| (-i as f64 * h_orig * 1.5).exp()).collect();
        let (ctrl_f32, dr_f32) = resample_bspline(&y_orig, h_orig, 64);
        let ctrl: Vec<f64> = ctrl_f32.iter().map(|&v| v as f64).collect();
        let dr = dr_f32 as f64;

        // Check V'' continuity at each interior knot
        let n_knots = ctrl.len() - 2;
        for i in 1..=n_knots {
            let r_knot = i as f64 * dr;
            let eps = dr * 1e-6;
            let (_, _, ddv_left) = bspline3_eval_at(&ctrl, dr, r_knot - eps);
            let (_, _, ddv_right) = bspline3_eval_at(&ctrl, dr, r_knot + eps);
            let jump = (ddv_left - ddv_right).abs();
            assert!(jump < 1e-6 * (ddv_left.abs().max(ddv_right.abs()).max(1e-30)),
                "V'' jump at knot {i} (r={r_knot:.4}): left={ddv_left:.6e}, right={ddv_right:.6e}, jump={jump:.6e}");
        }
    }
}
