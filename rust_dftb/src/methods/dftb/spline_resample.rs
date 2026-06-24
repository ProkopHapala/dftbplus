//! B-spline resampling for SK tables.
//!
//! Given high-resolution SK table data on a uniform grid, this module:
//! 1. Fits a natural cubic spline to the original data (for accurate resampling)
//! 2. Evaluates the spline at N_target equally-spaced points
//! 3. Returns just the values (1 float per node — no derivatives needed)
//!
//! The GPU uses cubic B-spline interpolation with a 4-point stencil:
//!   val = w0*f[i-1] + w1*f[i] + w2*f[i+1] + w3*f[i+2]
//! where w0..w3 are the cubic B-spline basis weights.
//! This requires only 1 float per node (vs 2 for Hermite), and the 4-point
//! stencil shares data between neighbors, so total local memory is lower.
//!
//! This allows representing SK tables with 32-64 points instead of 100-500,
//! reducing GPU local memory usage by 4-8×.

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

/// Resample a function from a uniform grid to a target number of points.
///
/// Uses natural cubic spline for accurate interpolation of the original data,
/// then evaluates at N_target equally-spaced points. The resulting values
/// are used directly as B-spline control points on the GPU (4-point stencil).
///
/// Returns (resampled_values, new_dr).
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

    let y_f32: Vec<f32> = y_new.iter().map(|&v| v as f32).collect();
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
}
