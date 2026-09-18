// ============================================================================
// Sparse DFTB — GPU pair-physics kernels (R17): H0/S assembly, analytic pair
// derivatives + K/W force contraction, repulsive energy/forces, per-atom
// force gather.
//
// f32 throughout. One work-item per pair (or per atom) — gather-only outputs,
// NO global atomics. Per-pair force buffers (write-owned) + a per-atom gather
// pass reproduce exact Newton symmetry of the CPU reference (forces.rs:
// non_scc_forces_bsr / scc_shift_forces_bsr / repulsive_force_cached), to f32.
//
// SK evaluation is the canonical C² cubic B-spline (interpolation.rs
// EqGridTable controls, phantom endpoints 2c0-c1 / 2c_{n-1}-c_{n-2}), rotated
// to Cartesian by the s/p Slater-Koster rules of rotation.rs
// (shell_pair / shell_pair_with_derivs). Derivatives are analytic, per Bohr,
// w.r.t. atom j — the taper chain rule converts per Å like the CPU path.
//
// Padded lanes: off-diagonal loops cover physical orbitals only
// (a < n_orb[j], b < n_orb[i]); diagonal writes E_DUMMY on dummy lanes.
// Masks/cutoffs: pairs at r_ang >= r0+w are skipped (buffer pre-zeroed);
// pairs at r_bohr >= min(cut_fwd, cut_rev) return early (eval is ~0 anyway).
//
// Pair record: int4 {i, j, b_ij, b_ji} — block indices in M_HS (per-atom
// orbital block + off-diagonals). b_ij_k lives in a separate i32 array
// (M_K reorders).
// ============================================================================

#define HSK_A2B  1.889726133f
#define HSK_E_DUMMY 2.0f
#define HSK_MIN_NEIGH2 (1.0e-2f * 1.0e-2f)

// ---- C² cubic B-spline eval (EqGridTable::eval_bspline_*) -------------------
// Returns (V(r), dV/dr) per Bohr; dV unused by hs_assemble.
inline float2 sk_eval(__global const float* ctrl, int n_ctrl, float dr, float r)
{
    if (r < 0.0f || r >= (float)n_ctrl * dr) return (float2)(0.0f, 0.0f);
    float inv_dr = 1.0f / dr;
    float x = (r - dr) * inv_dr;
    int i = (int)x;
    if (i < 0) i = 0;
    if (i > n_ctrl - 2) i = n_ctrl - 2;
    float t = x - (float)i;
    float c0 = (i == 0) ? 2.0f * ctrl[0] - ctrl[1] : ctrl[i - 1];
    float c1 = ctrl[i];
    float c2 = ctrl[i + 1];
    float c3 = (i + 2 >= n_ctrl) ? 2.0f * ctrl[n_ctrl - 1] - ctrl[n_ctrl - 2] : ctrl[i + 2];
    float t2 = t * t, t3 = t2 * t, u = 1.0f - t;
    float v = c0 * (u * u * u / 6.0f)
            + c1 * ((3.0f * t3 - 6.0f * t2 + 4.0f) / 6.0f)
            + c2 * ((-3.0f * t3 + 3.0f * t2 + 3.0f * t + 1.0f) / 6.0f)
            + c3 * (t3 / 6.0f);
    float dv = (c0 * (-0.5f * u * u) + c1 * (1.5f * t2 - 2.0f * t)
              + c2 * (-1.5f * t2 + t + 0.5f) + c3 * (0.5f * t2)) * inv_dr;
    return (float2)(v, dv);
}

// ---- s/p Slater-Koster rotation — port of Rotation::shell_pair_with_derivs --
// sk[0] = sigma, sk[1] = pi; dsk = dV/dr per Bohr; l,m,n = unit direction
// (dx,dy,dz)/r; inv_r in Bohr^-1. Out: 9 values + 27 Cartesian derivs
// (per Bohr, w.r.t. atom j), row-major in the p-ordering [py,pz,px].
inline void shell_pair_d(int l1, int l2, const float* sk, const float* dsk,
                         float l, float m, float n, float inv_r,
                         float* val, float* dx, float* dy, float* dz)
{
    if (l1 == 0 && l2 == 0) {
        val[0] = sk[0];
        dx[0] = dsk[0] * l; dy[0] = dsk[0] * m; dz[0] = dsk[0] * n;
    } else if (l1 == 0 || l2 == 0) {
        // sp pair: sub-block is a 3-vector — p axis ordering [m,n,l] = [y,z,x].
        float v = sk[0], vp = dsk[0];
        val[0] = m * v; dx[0] = (-m * l * inv_r) * v + m * vp * l;
                        dy[0] = ((1.0f - m * m) * inv_r) * v + m * vp * m;
                        dz[0] = (-m * n * inv_r) * v + m * vp * n;
        val[1] = n * v; dx[1] = (-n * l * inv_r) * v + n * vp * l;
                        dy[1] = (-n * m * inv_r) * v + n * vp * m;
                        dz[1] = ((1.0f - n * n) * inv_r) * v + n * vp * n;
        val[2] = l * v; dx[2] = ((1.0f - l * l) * inv_r) * v + l * vp * l;
                        dy[2] = (-l * m * inv_r) * v + l * vp * m;
                        dz[2] = (-l * n * inv_r) * v + l * vp * n;
    } else {
        // pp pair: 3x3 in the [py,pz,px] basis.
        float vs = sk[0], vp = sk[1], dvs = dsk[0], dvp = dsk[1];
        float dv = vs - vp, dvp_t = dvs - dvp;
        float ui[3] = {m, n, l};      // u_{p(i)}: py=m, pz=n, px=l
        float ua[3] = {l, m, n};      // u_a for Cartesian a = x,y,z
        for (int ii = 0; ii < 3; ii++) {
            for (int jj = 0; jj < 3; jj++) {
                int idx = ii * 3 + jj;
                float dij = (ii == jj) ? 1.0f : 0.0f;
                val[idx] = vp * dij + dv * ui[ii] * ui[jj];
                for (int a = 0; a < 3; a++) {
                    float dia = (a == (ii + 1) % 3) ? 1.0f : 0.0f;
                    float dja = (a == (jj + 1) % 3) ? 1.0f : 0.0f;
                    float dval = dvp * ua[a] * dij
                               + dvp_t * ua[a] * ui[ii] * ui[jj]
                               + dv * inv_r * ((dia - ui[ii] * ua[a]) * ui[jj]
                                              + ui[ii] * (dja - ui[jj] * ua[a]));
                    if (a == 0) dx[idx] = dval; else if (a == 1) dy[idx] = dval; else dz[idx] = dval;
                }
            }
        }
    }
}

// Value-only rotation for assembly (same math, deriv arrays discarded).
inline void shell_pair_v(int l1, int l2, const float* sk,
                         float l, float m, float n, float* val)
{
    if (l1 == 0 && l2 == 0) {
        val[0] = sk[0];
    } else if (l1 == 0 || l2 == 0) {
        val[0] = m * sk[0]; val[1] = n * sk[0]; val[2] = l * sk[0];
    } else {
        float vs = sk[0], vp = sk[1];
        float dv = vs - vp;
        float ui[3] = {m, n, l};
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                val[ii * 3 + jj] = vp * (float)(ii == jj) + dv * ui[ii] * ui[jj];
    }
}

// ---- per-pair geometry + taper ---------------------------------------------
// Returns w; writes dx,dy,dz (Ang), r_ang, l,m,n, r_b (Bohr), wp_b (taper
// deriv per Bohr). w == 0 means "skip pair".
inline float pair_geom(const float4 ri, const float4 rj, float4 taper,
                       float* dx, float* dy, float* dz,
                       float* r_ang, float* r_b, float* l, float* m, float* n,
                       float* wp_b)
{
    *dx = rj.x - ri.x; *dy = rj.y - ri.y; *dz = rj.z - ri.z;
    float r2 = (*dx) * (*dx) + (*dy) * (*dy) + (*dz) * (*dz);
    if (r2 < 1.0e-20f) return 0.0f;    // coincident atoms — leave zeros
    *r_ang = sqrt(r2);
    *r_b = *r_ang * HSK_A2B;
    float inv_r_a = 1.0f / *r_ang;
    *l = (*dx) * inv_r_a; *m = (*dy) * inv_r_a; *n = (*dz) * inv_r_a;
    float w = 1.0f, wp = 0.0f;
    if (taper.z > 0.5f) {              // taper enabled
        if (*r_ang <= taper.x) {
            // w = 1
        } else if (*r_ang >= taper.x + taper.y) {
            return 0.0f;
        } else {
            float x = (float)M_PI_F * (*r_ang - taper.x) / taper.y;
            w = 0.5f * (1.0f + cos(x));
            wp = -0.5f * sin(x) * (float)M_PI_F / taper.y;
        }
    }
    *wp_b = wp / HSK_A2B;
    return w;
}

// ============================================================================
// Diagonal blocks: h[d][d] = onsite (physical) or E_DUMMY (padded), s = I.
// Requires h,s pre-zeroed.
// ============================================================================
__kernel void hs_diag(
    const uint n,
    __global const uint* hs_diag,
    __global const uint* n_orb,
    __global const float4* onsite,
    __global float* h,
    __global float* s,
    const uint sblk)   // F5 replica: h/s BLOCK stride (dim1 = slot)
{
    uint i = get_global_id(0);
    h += (size_t)get_global_id(1) * sblk * 16;
    s += (size_t)get_global_id(1) * sblk * 16;
    if (i >= n) return;
    int ni = (int)n_orb[i];
    uint bd = hs_diag[i];
    float4 eo = onsite[i];
    for (int d = 0; d < 4; d++) {
        float ev = (d == 0) ? eo.x : (d == 1) ? eo.y : (d == 2) ? eo.z : eo.w;
        h[bd * 16 + d * 4 + d] = (d < ni) ? ev : HSK_E_DUMMY;
        s[bd * 16 + d * 4 + d] = 1.0f;
    }
}

// ============================================================================
// Off-diagonal H0/S assembly over M_HS pairs. One work-item per pair; writes
// both orientations (block layouts differ) — write-owned, no races.
// ============================================================================
__kernel void hs_assemble(
    const uint npairs,
    __global const int4* pairs,
    __global const float4* xyzu,
    __global const uint*  species,
    __global const uint*  n_orb,
    const uint nsp,
    __global const int4*  sk_meta,
    __global const float4* sk_parm,
    __global const float* sk_ctrl,
    const uint max_ctrl,
    const float4 taper,
    __global float* h,
    __global float* s,
    const uint n_atom,  // F5 replica: xyzu float4 stride (dim1 = slot)
    const uint sblk)    // F5 replica: h/s BLOCK stride
{
    uint p = get_global_id(0);
    const uint rep = get_global_id(1);
    xyzu += (size_t)rep * n_atom;
    h += (size_t)rep * sblk * 16;
    s += (size_t)rep * sblk * 16;
    if (p >= npairs) return;
    int4 pr = pairs[p];
    int i = pr.x, j = pr.y, bij = pr.z, bji = pr.w;

    float dx, dy, dz, r_ang, r_b, l, m, n, wp_b;
    float w = pair_geom(xyzu[i], xyzu[j], taper,
                        &dx, &dy, &dz, &r_ang, &r_b, &l, &m, &n, &wp_b);
    if (w == 0.0f) return;

    uint si = species[i], sj = species[j];
    int ni = (int)n_orb[i], nj = (int)n_orb[j];
    int4 mf = sk_meta[si * nsp + sj];
    int4 mr = sk_meta[sj * nsp + si];
    float4 pf_ = sk_parm[si * nsp + sj];
    float4 pr_ = sk_parm[sj * nsp + si];
    if (r_b >= min(pf_.y, pr_.y)) return;   // beyond both cutoffs — eval ~0

    __global const float* cf = sk_ctrl + (si * nsp + sj) * 8 * max_ctrl;
    __global const float* cr = sk_ctrl + (sj * nsp + si) * 8 * max_ctrl;

    int i_col = 0;
    for (int s1 = 0; s1 < (ni > 1 ? 2 : 1); s1++) {
        int l1 = s1;
        int n1 = 2 * l1 + 1;
        int i_row = 0;
        for (int s2 = 0; s2 < (nj > 1 ? 2 : 1); s2++) {
            int l2 = s2;
            int n2 = 2 * l2 + 1;
            int fwd = (l1 <= l2);
            int pt = fwd ? (int)(si * nsp + sj) : (int)(sj * nsp + si);
            int n_ct = fwd ? mf.x : mr.x;
            float dr = fwd ? pf_.x : pr_.x;
            __global const float* ctab = fwd ? cf : cr;
            // channel slot: {ss:0, sp:1, pps:2, ppp:3}; h at ch, s at ch+4.
            int ch = (l1 == 0 && l2 == 0) ? 0 : (l1 == l2 ? 2 : 1);
            int n_mm = min(l1, l2) + 1;
            float skh[2], sks[2];
            for (int c = 0; c < n_mm; c++) {
                skh[c] = sk_eval(ctab + (ch + c) * max_ctrl, n_ct, dr, r_b).x;
                sks[c] = sk_eval(ctab + (ch + c + 4) * max_ctrl, n_ct, dr, r_b).x;
            }
            float sh[9], ss_[9];
            shell_pair_v(l1, l2, skh, l, m, n, sh);
            shell_pair_v(l1, l2, sks, l, m, n, ss_);
            float sign = (l1 > l2 && ((l1 + l2) & 1) == 1) ? -1.0f : 1.0f;
            for (int a = 0; a < n2; a++) {
                for (int b = 0; b < n1; b++) {
                    int sidx = fwd ? a * n1 + b : b * n2 + a;
                    int row = i_row + a, col = i_col + b;
                    float vh = sign * sh[sidx] * w;
                    float vs = sign * ss_[sidx] * w;
                    h[bji * 16 + row * 4 + col] = vh;
                    h[bij * 16 + col * 4 + row] = vh;
                    s[bji * 16 + row * 4 + col] = vs;
                    s[bij * 16 + col * 4 + row] = vs;
                }
            }
            i_row += n2;
        }
        i_col += n1;
    }
}

// ============================================================================
// Force contraction: per-pair F_nscc = 2*(D:dH' - W:dS'), F_shift =
// 2*avg_shift*(D:dS').  Work-item per pair -> pf[2p], pf[2p+1] (float4 xyz).
// Signs: contribution to atom i's force; gather applies +i / -j.
// ============================================================================
__kernel void hs_contract(
    const uint npairs,
    __global const int4* pairs,
    __global const int*   pairs_k,
    __global const float4* xyzu,
    __global const uint*  species,
    __global const uint*  n_orb,
    const uint nsp,
    __global const int4*  sk_meta,
    __global const float4* sk_parm,
    __global const float* sk_ctrl,
    const uint max_ctrl,
    const float4 taper,
    __global const float* k_vals,
    __global const float* w_vals,
    __global const float* v_atom,
    __global float4* pf,
    const uint n_atom,
    const uint k_str,  // F5 replica: k_vals ELEMENT stride (0 = shared)
    const uint w_str)  // F5 replica: w_vals ELEMENT stride (0 = shared)
{
    uint p = get_global_id(0);
    if (p >= npairs) return;
    // F1 replica axis: dim1 = eval slot — xyzu/v_atom/pf are per-replica
    // strided; k_vals/w_vals are shared (strides 0) in the frozen path or
    // per-replica (strides = nnz·16) in the DMM path.
    const uint b = get_global_id(1);
    xyzu   += (size_t)b * n_atom;
    v_atom += (size_t)b * n_atom;
    k_vals += (size_t)b * k_str;
    w_vals += (size_t)b * w_str;
    pf     += (size_t)b * 2 * npairs;
    pf[2 * p] = (float4)0.0f;
    pf[2 * p + 1] = (float4)0.0f;
    int4 pr = pairs[p];
    int i = pr.x, j = pr.y, bij = pr.z;
    int bk = pairs_k[p];

    float dx, dy, dz, r_ang, r_b, l, m, n, wp_b;
    float w = pair_geom(xyzu[i], xyzu[j], taper,
                        &dx, &dy, &dz, &r_ang, &r_b, &l, &m, &n, &wp_b);
    if (w == 0.0f) return;

    uint si = species[i], sj = species[j];
    int ni = (int)n_orb[i], nj = (int)n_orb[j];
    int4 mf = sk_meta[si * nsp + sj];
    int4 mr = sk_meta[sj * nsp + si];
    float4 pf_ = sk_parm[si * nsp + sj];
    float4 pr_ = sk_parm[sj * nsp + si];
    if (r_b >= min(pf_.y, pr_.y)) return;

    __global const float* cf = sk_ctrl + (si * nsp + sj) * 8 * max_ctrl;
    __global const float* cr = sk_ctrl + (sj * nsp + si) * 8 * max_ctrl;
    float inv_r_b = 1.0f / r_b;
    float avg = 0.5f * (v_atom[i] + v_atom[j]);
    float cx = 0.0f, cy = 0.0f, cz = 0.0f, sx = 0.0f, sy = 0.0f, sz = 0.0f;

    int i_col = 0;
    for (int s1 = 0; s1 < (ni > 1 ? 2 : 1); s1++) {
        int l1 = s1;
        int n1 = 2 * l1 + 1;
        int i_row = 0;
        for (int s2 = 0; s2 < (nj > 1 ? 2 : 1); s2++) {
            int l2 = s2;
            int n2 = 2 * l2 + 1;
            int fwd = (l1 <= l2);
            int n_ct = fwd ? mf.x : mr.x;
            float dr = fwd ? pf_.x : pr_.x;
            __global const float* ctab = fwd ? cf : cr;
            int ch = (l1 == 0 && l2 == 0) ? 0 : (l1 == l2 ? 2 : 1);
            int n_mm = min(l1, l2) + 1;
            float skh[2], sks[2], dkh[2], dks[2];
            for (int c = 0; c < n_mm; c++) {
                float2 th = sk_eval(ctab + (ch + c) * max_ctrl, n_ct, dr, r_b);
                float2 ts = sk_eval(ctab + (ch + c + 4) * max_ctrl, n_ct, dr, r_b);
                skh[c] = th.x; dkh[c] = th.y;
                sks[c] = ts.x; dks[c] = ts.y;
            }
            float sh[9], ss_[9], dhx[9], dhy[9], dhz[9], dsx[9], dsy[9], dsz[9];
            shell_pair_d(l1, l2, skh, dkh, l, m, n, inv_r_b, sh, dhx, dhy, dhz);
            shell_pair_d(l1, l2, sks, dks, l, m, n, inv_r_b, ss_, dsx, dsy, dsz);
            float sign = (l1 > l2 && ((l1 + l2) & 1) == 1) ? -1.0f : 1.0f;
            for (int a = 0; a < n2; a++) {
                for (int b = 0; b < n1; b++) {
                    int sidx = fwd ? a * n1 + b : b * n2 + a;
                    int row = i_row + a, col = i_col + b;
                    float hv = sign * sh[sidx];
                    float sv = sign * ss_[sidx];
                    float thx = sign * dhx[sidx], thy = sign * dhy[sidx], thz = sign * dhz[sidx];
                    float tsx = sign * dsx[sidx], tsy = sign * dsy[sidx], tsz = sign * dsz[sidx];
                    // taper chain rule: d(wV)/dR = w dV + w' u_a V (per Bohr)
                    thx = w * thx + wp_b * l * hv;
                    thy = w * thy + wp_b * m * hv;
                    thz = w * thz + wp_b * n * hv;
                    tsx = w * tsx + wp_b * l * sv;
                    tsy = w * tsy + wp_b * m * sv;
                    tsz = w * tsz + wp_b * n * sv;
                    float dm = (bk >= 0) ? 2.0f * k_vals[bk * 16 + col * 4 + row] : 0.0f;
                    float edm = w_vals[bij * 16 + col * 4 + row];
                    cx += dm * thx - edm * tsx;
                    cy += dm * thy - edm * tsy;
                    cz += dm * thz - edm * tsz;
                    sx += avg * tsx * dm;
                    sy += avg * tsy * dm;
                    sz += avg * tsz * dm;
                }
            }
            i_row += n2;
        }
        i_col += n1;
    }
    float fac = 2.0f * HSK_A2B;   // per Bohr -> per Ang, dE = -F => 2*A2B
    pf[2 * p]     = (float4)(fac * cx, fac * cy, fac * cz, 0.0f);
    pf[2 * p + 1] = (float4)(fac * sx, fac * sy, fac * sz, 0.0f);
}

// ============================================================================
// Repulsive energy + force per pair (r_ang < cutoff list). Record layout from
// pack_repulsive_gpu: [n_int(as int), cutoff(Bohr), a,b,c (Ha,Bohr units),
// x_start[max_int], sp_coeffs[(max_int-1)*4], sp_last[6]].
// pf_rep[p] = (F_i [Ha/Ang], E_pair [Ha]); pe_rep[p] = E_pair for reduction.
// ============================================================================
__kernel void rep_eval(
    const uint nrep,
    __global const int2* rpairs,
    __global const float4* xyzu,
    __global const uint*  species,
    const uint nsp,
    __global const int*  rep_off,
    const uint rep_max_int,
    __global const float* rep_data,
    __global float4* pf_rep,
    __global float* pe_rep,
    const uint n_atom)
{
    uint p = get_global_id(0);
    if (p >= nrep) return;
    // F1 replica axis: dim1 = eval slot — xyzu/pf_rep/pe_rep are
    // per-replica strided; rep tables are shared (read-only).
    const uint b = get_global_id(1);
    xyzu   += (size_t)b * n_atom;
    pf_rep += (size_t)b * nrep;
    pe_rep += (size_t)b * nrep;
    pf_rep[p] = (float4)0.0f;
    pe_rep[p] = 0.0f;
    int2 rp = rpairs[p];
    int i = rp.x, j = rp.y;
    uint si = species[i], sj = species[j];
    int off = rep_off[si * nsp + sj];
    if (off < 0) off = rep_off[sj * nsp + si];
    if (off < 0) return;                    // no spline — matches CPU `continue`
    __global const float* rec = rep_data + off;
    int n_int = as_int(rec[0]);   // pack_repulsive_gpu stores n_int via f32::from_bits
    float cutoff = rec[1];

    float dx = xyzu[j].x - xyzu[i].x;
    float dy = xyzu[j].y - xyzu[i].y;
    float dz = xyzu[j].z - xyzu[i].z;
    float r2_ang = dx * dx + dy * dy + dz * dz;
    if (r2_ang < HSK_MIN_NEIGH2) return;
    float r_ang = sqrt(r2_ang);
    float r = r_ang * HSK_A2B;
    if (r >= cutoff) return;

    float e, de;
    __global const float* x_start = rec + 5;
    if (r < x_start[0]) {
        // Exponential head: E = exp(-a·R + b) + c (repulsive.rs exp_coeffs).
        float ea = rec[2], eb = rec[3], ec = rec[4];
        float eh = exp(-ea * r + eb);
        e = eh + ec;
        de = -ea * eh;
    } else if (r < x_start[n_int - 1]) {
        // bisect: largest k with x_start[k] <= r (k in [0, n_int-2])
        int lo = 0, hi = n_int - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (x_start[mid] <= r) lo = mid; else hi = mid;
        }
        __global const float* sp = rec + 5 + rep_max_int + lo * 4;
        float t = r - x_start[lo];
        e = sp[0] + t * (sp[1] + t * (sp[2] + t * sp[3]));
        de = sp[1] + t * (2.0f * sp[2] + t * 3.0f * sp[3]);
    } else {
        __global const float* pl = rec + 5 + rep_max_int + (rep_max_int - 1) * 4;
        float t = r - x_start[n_int - 1];
        e = pl[0] + t * (pl[1] + t * (pl[2] + t * (pl[3] + t * (pl[4] + t * pl[5]))));
        de = pl[1] + t * (2.0f * pl[2] + t * (3.0f * pl[3] + t * (4.0f * pl[4] + t * 5.0f * pl[5])));
    }
    float de_ang = de * HSK_A2B;
    float f = de_ang / r_ang;
    pf_rep[p] = (float4)(f * dx, f * dy, f * dz, e);
    pe_rep[p] = e;
}

// ============================================================================
// Per-atom force gather. fp_list entry = (pair<<1)|is_j -> +pf for i, -pf for j.
// rp_list same over rep pairs (xyz only). Outputs five float4/atom buffers:
// non_scc, scc_shift, repulsive, scc_dc (gamma'), total.
// ============================================================================
__kernel void force_gather(
    const uint n,
    __global const uint* fp_ptr,
    __global const int*  fp_list,
    __global const float4* pf,
    __global const uint* rp_ptr,
    __global const int*  rp_list,
    __global const float4* pf_rep,
    __global const float4* gf,
    __global float4* f_nscc,
    __global float4* f_shift,
    __global float4* f_rep,
    __global float4* f_dc,
    __global float4* f_tot,
    const uint npairs,
    const uint nrep)
{
    uint a = get_global_id(0);
    if (a >= n) return;
    // F1 replica axis: dim1 = eval slot — all per-pair/per-atom buffers
    // are per-replica strided; the gather topology (fp/rp) is shared.
    const uint b = get_global_id(1);
    pf     += (size_t)b * 2 * npairs;
    pf_rep += (size_t)b * nrep;
    gf     += (size_t)b * n;
    f_nscc += (size_t)b * n;
    f_shift += (size_t)b * n;
    f_rep  += (size_t)b * n;
    f_dc   += (size_t)b * n;
    f_tot  += (size_t)b * n;
    float4 fn = (float4)0.0f, fs = (float4)0.0f;
    for (uint e = fp_ptr[a]; e < fp_ptr[a + 1]; e++) {
        int ent = fp_list[e];
        uint p = (uint)(ent >> 1);
        float sgn = (ent & 1) ? -1.0f : 1.0f;
        fn += sgn * pf[2 * p];
        fs += sgn * pf[2 * p + 1];
    }
    float4 fr = (float4)0.0f;
    for (uint e = rp_ptr[a]; e < rp_ptr[a + 1]; e++) {
        int ent = rp_list[e];
        uint p = (uint)(ent >> 1);
        float sgn = (ent & 1) ? -1.0f : 1.0f;
        float4 rp = pf_rep[p];
        fr.x += sgn * rp.x; fr.y += sgn * rp.y; fr.z += sgn * rp.z;
    }
    float4 fd = gf[a];
    f_nscc[a] = fn;
    f_shift[a] = fs;
    f_rep[a] = fr;
    f_dc[a] = fd;
    f_tot[a] = fn + fs + fr + fd;
}

// ============================================================================
// Dummy-orbital occupation guard: per-atom sum |2*K_dd| over padded lanes
// (d >= n_orb[i]) of the diagonal K block. Host reduces + compares to the
// CPU guard's 1e-6.
// ============================================================================
__kernel void hs_kdummy(
    const uint n,
    __global const uint* k_diag,
    __global const float* k_vals,
    __global const uint* n_orb,
    __global float* partial)
{
    uint i = get_global_id(0);
    if (i >= n) return;
    int ni = (int)n_orb[i];
    float acc = 0.0f;
    uint bd = k_diag[i];
    for (int d = ni; d < 4; d++) acc += fabs(2.0f * k_vals[bd * 16 + d * 4 + d]);
    partial[i] = acc;
}
