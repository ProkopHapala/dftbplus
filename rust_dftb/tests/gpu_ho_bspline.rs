//! Diagnostic: check B-spline control point conversion for H-O ss column.

use rust_dftb::methods::dftb::spline_resample::resample_bspline;
use rust_dftb::{load_sk_for_species, SkData};

fn cubic_weights(t: f32) -> (f32, f32, f32, f32) {
    let omt = 1.0 - t;
    (
        omt * omt * omt * 0.16666667,
        (3.0 * t * t * t - 6.0 * t * t + 4.0) * 0.16666667,
        (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) * 0.16666667,
        t * t * t * 0.16666667,
    )
}

fn interp_bspline_4pt(tab: &[f32], base: usize, w: (f32, f32, f32, f32)) -> f32 {
    tab[base] * w.0 + tab[base + 1] * w.1 + tab[base + 2] * w.2 + tab[base + 3] * w.3
}

#[test]
fn test_bspline_ho_ss() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    let species = vec!["H".to_string(), "O".to_string()];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let tab = sk.get_pair("H", "O").unwrap();
    let n_grid = tab.h.n_grid();
    let dr = tab.h.dr;
    eprintln!("H-O table: n_grid={n_grid}, dr={dr}");

    // Extract ss column (0,0) using the same logic as gpu_prep
    let mut ss_col = vec![0.0f64; n_grid];
    let mut sp_col = vec![0.0f64; n_grid];
    for k in 0..n_grid {
        let h_all = &tab.h.values[k];
        // Old format: 10 values, ss = arr[9], sp = arr[8]
        // Using same extraction as extract_shell_old_or_new
        let is_extended = h_all.len() == 20;
        if is_extended {
            // new format: ss = h_all[19], sp = h_all[18]
            ss_col[k] = h_all[19];
            sp_col[k] = h_all[18];
        } else {
            // old format: ss = h_all[9], sp = h_all[8]
            ss_col[k] = h_all[9];
            sp_col[k] = h_all[8];
        }
    }

    // Print values around r=1.89 Bohr (k=93, since r=(k+1)*dr)
    eprintln!("Original ss values around r=1.89 Bohr:");
    for k in 88..98 {
        let r = (k + 1) as f64 * dr;
        eprintln!("  k={k} r={r:.4} ss={:.6} sp={:.6}", ss_col[k], sp_col[k]);
    }

    // Convert to B-spline control points (same as gpu_prep)
    let (ss_ctrl, dr_new) = resample_bspline(&ss_col, dr, n_grid);
    let (sp_ctrl, _) = resample_bspline(&sp_col, dr, n_grid);
    eprintln!("After resample_bspline: dr_new={dr_new}, n_ctrl={}", ss_ctrl.len());

    // Print control points around the same region
    eprintln!("B-spline control points (ss) around k=90-95:");
    for k in 88..98 {
        if k < ss_ctrl.len() {
            eprintln!("  k={k} ctrl_ss={:.6} ctrl_sp={:.6}", ss_ctrl[k], sp_ctrl[k]);
        }
    }

    // Build the GPU table with prepended zero (same as gpu_prep)
    let n_sk_cols = 2usize;
    let n_gpu = n_grid + 1;
    let mut sk_h = vec![0.0f32; n_gpu * n_sk_cols];
    for col in 0..n_sk_cols {
        sk_h[col] = 0.0; // dummy at r=0
        let ctrl = if col == 0 { &ss_ctrl } else { &sp_ctrl };
        for k in 0..n_grid {
            sk_h[(k + 1) * n_sk_cols + col] = ctrl[k];
        }
    }

    // Interpolate at r=1.8897 Bohr (1.0 Å)
    let r_eval = 1.0f32 * 1.889726133; // 1.0 Å in Bohr
    let u = r_eval / dr_new;
    let i = u as i32;
    let i_clamped = i.clamp(1, (n_gpu - 3) as i32);
    let t = u - i_clamped as f32;
    let base = (i_clamped - 1) as usize;
    let w = cubic_weights(t);
    eprintln!("Interpolation: r={r_eval:.6}, u={u:.3}, i={i}, i_clamped={i_clamped}, t={t:.4}, base={base}");

    // Read float2 values (ss, sp) from the table
    let ss_val = interp_bspline_4pt(&sk_h[0..], base * n_sk_cols, w); // wrong — need stride
    // Actually the table is interleaved: [ss_0, sp_0, ss_1, sp_1, ...]
    // For interp_sk_2, it casts to float2 and reads tab2[base], tab2[base+1], etc.
    // tab2[k] = (sk_h[2*k], sk_h[2*k+1])
    // So we need to read with stride 2:
    let mut ss_interp = 0.0f32;
    let mut sp_interp = 0.0f32;
    for j in 0..4 {
        let idx = (base + j) * n_sk_cols;
        ss_interp += sk_h[idx] * [w.0, w.1, w.2, w.3][j];
        sp_interp += sk_h[idx + 1] * [w.0, w.1, w.2, w.3][j];
    }
    eprintln!("GPU interpolation result: ss={ss_interp:.6}, sp={sp_interp:.6}");
    eprintln!("Expected (CPU):           ss≈-0.467,  sp≈+0.321");

    // Check: does the 4-point stencil reproduce the original values at knot points?
    eprintln!("Verification at knot points (should match original):");
    for k in [90, 91, 92, 93, 94] {
        let r = (k + 1) as f64 * dr;
        // GPU table index for this knot: tab[k+1] = ctrl[k]
        // At the knot point (t=0), the stencil reads tab[k], tab[k+1], tab[k+2], tab[k+3]
        // with weights (1/6, 4/6, 1/6, 0)
        let gpu_k = k + 1; // +1 for prepended zero
        let base_k = gpu_k - 1; // base = i - 1
        let w0 = (1.0f32 / 6.0, 4.0 / 6.0, 1.0 / 6.0, 0.0);
        let mut ss_knot = 0.0f32;
        let mut sp_knot = 0.0f32;
        for j in 0..4 {
            let idx = (base_k + j) * n_sk_cols;
            ss_knot += sk_h[idx] * [w0.0, w0.1, w0.2, w0.3][j];
            sp_knot += sk_h[idx + 1] * [w0.0, w0.1, w0.2, w0.3][j];
        }
        eprintln!("  k={k} r={r:.4}: orig_ss={:.6}, knot_ss={ss_knot:.6}, orig_sp={:.6}, knot_sp={sp_knot:.6}",
            ss_col[k], sp_col[k]);
    }
}
