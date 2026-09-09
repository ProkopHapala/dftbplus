//! gpu_forces.cl — GPU analytic force kernel (Phase 4)
//!
//! Computes DFTB non-SCC electronic force contributions from pair blocks:
//!   F_i[a] += 2 * ANG2BOHR * Σ_{μ∈i,ν∈j} (DM[μ,ν]·dH[μ,ν]/dR_a - EDM[μ,ν]·dS[μ,ν]/dR_a)
//!   F_j[a] -= same
//!
//! Derivatives are fully analytic:
//!   - Radial: analytic B-spline derivative (cubic_weights_d1)
//!   - Angular: chain rule through direction cosines
//!
//!   dH/dR_a = dH/dr · u_a + dH/dl · dl/dR_a + dH/dm · dm/dR_a + dH/dn · dn/dR_a
//!
//! where u = (l,m,n) = R_vec/R, and:
//!   dl/dR_x = (1-l²)/R,  dl/dR_y = -l·m/R,  dl/dR_z = -l·n/R
//!   dm/dR_x = -m·l/R,   dm/dR_y = (1-m²)/R, dm/dR_z = -m·n/R
//!   dn/dR_x = -n·l/R,   dn/dR_y = -n·m/R,   dn/dR_z = (1-n²)/R
//!
//! Block types (same as assemble_pairs):
//!   0 = 1×1 (s-s), 1 = 1×4 (s-p), 2 = 4×4 (p-p)

#define ANG2BOHR_F 1.889726133f
#define SK_GRID_MAX 512
#define N_SK_COLS 5
#define GPU_FORCE_DEBUG 0

// ─── Data structures (must match gpu_prep.rs GpuPairEntry / GpuFragment) ─
typedef struct {
    uint   replica;   // which fragment/replica
    ushort atom_i;    // atom index i   (local to fragment)
    ushort atom_j;    // atom index j   (local to fragment)
    ushort orb_i;     // orbital offset of atom i (local)
    ushort orb_j;     // orbital offset of atom j (local)
    float  r;         // distance (Bohr)
    float  l, m, n;   // direction cosines
} PairEntry;

typedef struct {
    int n_atoms;
    int n_orbs;
    int atom_off;
    int H_base;
} Fragment;

// ─── B-spline weights (from dftb_hamiltonian.cl) ───────────────────────
inline float4 cubic_weights(float t) {
    float omt = 1.0f - t;
    return (float4)(
        omt * omt * omt * 0.16666667f,
        (3.0f * t * t * t - 6.0f * t * t + 4.0f) * 0.16666667f,
        (-3.0f * t * t * t + 3.0f * t * t + 3.0f * t + 1.0f) * 0.16666667f,
        t * t * t * 0.16666667f
    );
}

inline float4 cubic_weights_d1(float t) {
    float u = 1.0f - t;
    return (float4)(
        -0.5f * u * u,
         1.5f * t * t - 2.0f * t,
        -1.5f * t * t + t + 0.5f,
         0.5f * t * t
    );
}

// Compute base index, value weights, and derivative weights.
// Returns base = i-1 for 4-point stencil. Outputs w (value) and wd (derivative).
inline int interp_params(float r, float dr, int n_grid, float4* w, float4* wd) {
    float u = r / dr;
    int i = (int)u;
    i = clamp(i, 1, n_grid - 3);
    float t = u - (float)i;
    *w  = cubic_weights(t);
    *wd = cubic_weights_d1(t);
    return i - 1;
}

inline float interp_sk_1(__local const float* tab, int base, float4 w) {
    return tab[base] * w.x + tab[base + 1] * w.y + tab[base + 2] * w.z + tab[base + 3] * w.w;
}

// 2-channel (1x4: ss, sp), interleaved with stride 2. vload2 avoids relying
// on stronger float2 pointer alignment for a float-declared local array.
inline float2 interp_sk_2(__local const float* tab, int base, float4 w) {
    float2 v0 = vload2(0, tab + 2 * base);
    float2 v1 = vload2(0, tab + 2 * (base + 1));
    float2 v2 = vload2(0, tab + 2 * (base + 2));
    float2 v3 = vload2(0, tab + 2 * (base + 3));
    return v0 * w.x + v1 * w.y + v2 * w.z + v3 * w.w;
}

// 5-channel: (ss, sp, pp_sig, pp_pi, ps) — interleaved with stride 5
inline float4 interp_sk_5(__local const float* tab, int base, float4 w, float* ps_out) {
    float4 r = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float ps = 0.0f;
    for (int j = 0; j < 4; j++) {
        int off = (base + j) * 5;
        float wt;
        if (j == 0) wt = w.x;
        else if (j == 1) wt = w.y;
        else if (j == 2) wt = w.z;
        else wt = w.w;
        r.s0 += tab[off    ] * wt;  // ss
        r.s1 += tab[off + 1] * wt;  // sp
        r.s2 += tab[off + 2] * wt;  // pp_sig
        r.s3 += tab[off + 3] * wt;  // pp_pi
        ps   += tab[off + 4] * wt;  // ps
    }
    *ps_out = ps;
    return r;
}

// Derivative versions: compute d(val)/dr using derivative weights wd.
// For interleaved layout, access tab[(base+j)*stride + channel] * wd[j].

inline float interp_sk_1_d(__local const float* tab, int base, float4 wd) {
    return tab[base] * wd.x + tab[base + 1] * wd.y + tab[base + 2] * wd.z + tab[base + 3] * wd.w;
}

inline float2 interp_sk_2_d(__local const float* tab, int base, float4 wd) {
    float2 v0 = vload2(0, tab + 2 * base);
    float2 v1 = vload2(0, tab + 2 * (base + 1));
    float2 v2 = vload2(0, tab + 2 * (base + 2));
    float2 v3 = vload2(0, tab + 2 * (base + 3));
    return v0 * wd.x + v1 * wd.y + v2 * wd.z + v3 * wd.w;
}

inline float4 interp_sk_5_d(__local const float* tab, int base, float4 wd, float* ps_out) {
    float4 r = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float ps = 0.0f;
    for (int j = 0; j < 4; j++) {
        int off = (base + j) * 5;
        float wt;
        if (j == 0) wt = wd.x;
        else if (j == 1) wt = wd.y;
        else if (j == 2) wt = wd.z;
        else wt = wd.w;
        r.s0 += tab[off    ] * wt;
        r.s1 += tab[off + 1] * wt;
        r.s2 += tab[off + 2] * wt;
        r.s3 += tab[off + 3] * wt;
        ps   += tab[off + 4] * wt;
    }
    *ps_out = ps;
    return r;
}

// ─── Direction cosine derivatives ──────────────────────────────────────
inline void dc_derivs(float l, float m, float n, float r,
                       float* dl_dx, float* dl_dy, float* dl_dz,
                       float* dm_dx, float* dm_dy, float* dm_dz,
                       float* dn_dx, float* dn_dy, float* dn_dz) {
    float inv_r = (r > 1e-12f) ? 1.0f / r : 0.0f;
    *dl_dx = (1.0f - l*l) * inv_r;  *dl_dy = -l*m * inv_r;  *dl_dz = -l*n * inv_r;
    *dm_dx = -m*l * inv_r;          *dm_dy = (1.0f - m*m) * inv_r;  *dm_dz = -m*n * inv_r;
    *dn_dx = -n*l * inv_r;          *dn_dy = -n*m * inv_r;          *dn_dz = (1.0f - n*n) * inv_r;
}

// ─── Block rotation with derivatives ───────────────────────────────────

// 1×1 (s-s): H = sk, dH/dR_a = dsk/dr * u_a
inline void block_1x1_with_derivs(
    float sk, float dsk_dr,
    float l, float m, float n, float r,
    float* h, float* dh_dx, float* dh_dy, float* dh_dz
) {
    *h = sk;
    *dh_dx = dsk_dr * l;
    *dh_dy = dsk_dr * m;
    *dh_dz = dsk_dr * n;
}

// 1×4 (s-p): blk = (ss, sp*m, sp*n, sp*l)
inline void block_1x4_with_derivs(
    float2 sk, float2 dsk_dr,
    float l, float m, float n, float r,
    float4* blk, float4* dx, float4* dy, float4* dz
) {
    float ss = sk.x, sp = sk.y;
    float dss = dsk_dr.x, dsp = dsk_dr.y;
    *blk = (float4)(ss, sp*m, sp*n, sp*l);

    float dl_dx, dl_dy, dl_dz, dm_dx, dm_dy, dm_dz, dn_dx, dn_dy, dn_dz;
    dc_derivs(l, m, n, r, &dl_dx, &dl_dy, &dl_dz, &dm_dx, &dm_dy, &dm_dz, &dn_dx, &dn_dy, &dn_dz);

    (*dx).x = dss * l;
    (*dx).y = dsp * m * l + sp * dm_dx;
    (*dx).z = dsp * n * l + sp * dn_dx;
    (*dx).w = dsp * l * l + sp * dl_dx;

    (*dy).x = dss * m;
    (*dy).y = dsp * m * m + sp * dm_dy;
    (*dy).z = dsp * n * m + sp * dn_dy;
    (*dy).w = dsp * l * m + sp * dl_dy;

    (*dz).x = dss * n;
    (*dz).y = dsp * m * n + sp * dm_dz;
    (*dz).z = dsp * n * n + sp * dn_dz;
    (*dz).w = dsp * l * n + sp * dl_dz;
}

// 4×4 (p-p): blk[4] as float4[4], rows = atom j (s,py,pz,px), cols = atom i
// sk = (ss, sp, pp_sig, pp_pi), ps = separate s-p integral
// Orbital ordering: s, py, pz, px → direction cosines (m, n, l) for (py, pz, px)
// Each pp element H[r][c] = uj*ui*diff + pp_pi*δ(r,c) depends on BOTH uj (row)
// and ui (col), so the derivative has two angular terms:
//   dH/dR_a = (uj*ui*ddiff + dpp_pi*δ)*u_a + duj/dR_a*ui*diff + uj*dui/dR_a*diff
inline void block_4x4_with_derivs(
    float4 sk, float4 dsk_dr, float ps, float dps_dr,
    float l, float m, float n, float r,
    float4* blk, float4* dx, float4* dy, float4* dz
) {
    float ss = sk.x, sp = sk.y, pp_sig = sk.z, pp_pi = sk.w;
    float dss = dsk_dr.x, dsp = dsk_dr.y, dpp_sig = dsk_dr.z, dpp_pi = dsk_dr.w;
    float diff = pp_sig - pp_pi;
    float ddiff = dpp_sig - dpp_pi;

    blk[0] = (float4)(ss, -ps*m, -ps*n, -ps*l);
    blk[1] = (float4)(sp*m, m*m*diff + pp_pi, m*n*diff, m*l*diff);
    blk[2] = (float4)(sp*n, n*m*diff, n*n*diff + pp_pi, n*l*diff);
    blk[3] = (float4)(sp*l, l*m*diff, l*n*diff, l*l*diff + pp_pi);

    float dl_dx, dl_dy, dl_dz, dm_dx, dm_dy, dm_dz, dn_dx, dn_dy, dn_dz;
    dc_derivs(l, m, n, r, &dl_dx, &dl_dy, &dl_dz, &dm_dx, &dm_dy, &dm_dz, &dn_dx, &dn_dy, &dn_dz);

    // Row 0 (s_j): (ss, -ps*m, -ps*n, -ps*l) — only i-orbital angular (j=s has none)
    dx[0] = (float4)(dss*l, -dps_dr*m*l - ps*dm_dx, -dps_dr*n*l - ps*dn_dx, -dps_dr*l*l - ps*dl_dx);
    dy[0] = (float4)(dss*m, -dps_dr*m*m - ps*dm_dy, -dps_dr*n*m - ps*dn_dy, -dps_dr*l*m - ps*dl_dy);
    dz[0] = (float4)(dss*n, -dps_dr*m*n - ps*dm_dz, -dps_dr*n*n - ps*dn_dz, -dps_dr*l*n - ps*dl_dz);

    // Row 1 (py_j, uj=m): col 0 (s_i, no angular), cols 1-3 (pp block)
    //   col 1 (py_i, ui=m): diagonal → 2*m*diff*dm/dR_a
    //   col 2 (pz_i, ui=n): dm/dR_a*n*diff + m*dn/dR_a*diff
    //   col 3 (px_i, ui=l): dm/dR_a*l*diff + m*dl/dR_a*diff
    dx[1] = (float4)(dsp*m*l + sp*dm_dx,
                     m*m*ddiff*l + dpp_pi*l + 2.0f*m*diff*dm_dx,
                     m*n*ddiff*l + n*diff*dm_dx + m*diff*dn_dx,
                     m*l*ddiff*l + l*diff*dm_dx + m*diff*dl_dx);
    dy[1] = (float4)(dsp*m*m + sp*dm_dy,
                     m*m*ddiff*m + dpp_pi*m + 2.0f*m*diff*dm_dy,
                     m*n*ddiff*m + n*diff*dm_dy + m*diff*dn_dy,
                     m*l*ddiff*m + l*diff*dm_dy + m*diff*dl_dy);
    dz[1] = (float4)(dsp*m*n + sp*dm_dz,
                     m*m*ddiff*n + dpp_pi*n + 2.0f*m*diff*dm_dz,
                     m*n*ddiff*n + n*diff*dm_dz + m*diff*dn_dz,
                     m*l*ddiff*n + l*diff*dm_dz + m*diff*dl_dz);

    // Row 2 (pz_j, uj=n): col 0 (s_i), cols 1-3 (pp block)
    //   col 1 (py_i, ui=m): dn/dR_a*m*diff + n*dm/dR_a*diff
    //   col 2 (pz_i, ui=n): diagonal → 2*n*diff*dn/dR_a
    //   col 3 (px_i, ui=l): dn/dR_a*l*diff + n*dl/dR_a*diff
    dx[2] = (float4)(dsp*n*l + sp*dn_dx,
                     n*m*ddiff*l + m*diff*dn_dx + n*diff*dm_dx,
                     n*n*ddiff*l + dpp_pi*l + 2.0f*n*diff*dn_dx,
                     n*l*ddiff*l + l*diff*dn_dx + n*diff*dl_dx);
    dy[2] = (float4)(dsp*n*m + sp*dn_dy,
                     n*m*ddiff*m + m*diff*dn_dy + n*diff*dm_dy,
                     n*n*ddiff*m + dpp_pi*m + 2.0f*n*diff*dn_dy,
                     n*l*ddiff*m + l*diff*dn_dy + n*diff*dl_dy);
    dz[2] = (float4)(dsp*n*n + sp*dn_dz,
                     n*m*ddiff*n + m*diff*dn_dz + n*diff*dm_dz,
                     n*n*ddiff*n + dpp_pi*n + 2.0f*n*diff*dn_dz,
                     n*l*ddiff*n + l*diff*dn_dz + n*diff*dl_dz);

    // Row 3 (px_j, uj=l): col 0 (s_i), cols 1-3 (pp block)
    //   col 1 (py_i, ui=m): dl/dR_a*m*diff + l*dm/dR_a*diff
    //   col 2 (pz_i, ui=n): dl/dR_a*n*diff + l*dn/dR_a*diff
    //   col 3 (px_i, ui=l): diagonal → 2*l*diff*dl/dR_a
    dx[3] = (float4)(dsp*l*l + sp*dl_dx,
                     l*m*ddiff*l + m*diff*dl_dx + l*diff*dm_dx,
                     l*n*ddiff*l + n*diff*dl_dx + l*diff*dn_dx,
                     l*l*ddiff*l + dpp_pi*l + 2.0f*l*diff*dl_dx);
    dy[3] = (float4)(dsp*l*m + sp*dl_dy,
                     l*m*ddiff*m + m*diff*dl_dy + l*diff*dm_dy,
                     l*n*ddiff*m + n*diff*dl_dy + l*diff*dn_dy,
                     l*l*ddiff*m + dpp_pi*m + 2.0f*l*diff*dl_dy);
    dz[3] = (float4)(dsp*l*n + sp*dl_dz,
                     l*m*ddiff*n + m*diff*dl_dz + l*diff*dm_dz,
                     l*n*ddiff*n + n*diff*dl_dz + l*diff*dn_dz,
                     l*l*ddiff*n + dpp_pi*n + 2.0f*l*diff*dl_dz);
}

// ─── Float atomic add via CAS loop (OpenCL 1.2 has no float atomics) ───
inline void atomic_add_f32(volatile __global float* addr, float val) {
    union { unsigned int u; float f; } old_val, new_val;
    old_val.u = as_uint(*addr);
    do {
        new_val.f = old_val.f + val;
        unsigned int prev = atomic_cmpxchg(
            (volatile __global unsigned int*)addr, old_val.u, new_val.u);
        if (prev == old_val.u) break;
        old_val.u = prev;
    } while (true);
}

// ─── Main force kernel ─────────────────────────────────────────────────
__kernel void force_pairs(
    __global const PairEntry* pairs,
    __global const Fragment* fragments,
    __global const float* sk_h,
    __global const float* sk_s,
    __global const float* dm,
    __global const float* edm,
    __global float* forces,
    const float dr,
    const int n_grid,
    const int n_pairs,
    const int n_frags,
    const int block_type,
    const int n_sk_cols
) {
    const int tid = get_local_id(0);
    const int wg  = get_local_size(0);
    const int gid = get_global_id(0);

    __local float l_sk_h[SK_GRID_MAX * N_SK_COLS];
    __local float l_sk_s[SK_GRID_MAX * N_SK_COLS];
    int n_sk_elements = n_grid * n_sk_cols;
    for (int i = tid; i < n_sk_elements; i += wg) {
        l_sk_h[i] = sk_h[i];
        l_sk_s[i] = sk_s[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid >= n_pairs) return;

    PairEntry p = pairs[gid];
    Fragment frag = fragments[p.replica];
    int n_orbs = frag.n_orbs;
    int atom_off = frag.atom_off;
    int ga_i = atom_off + p.atom_i;
    int ga_j = atom_off + p.atom_j;
    int dm_base = p.replica * n_orbs * n_orbs;

    // Value and derivative weights
    float4 w, wd;
    int base_idx = interp_params(p.r, dr, n_grid, &w, &wd);
    float inv_dr = 1.0f / dr;

    float l = p.l, m = p.m, n = p.n;
    float r = p.r;
    float3 f = (float3)(0.0f, 0.0f, 0.0f);

    if (block_type == 0) {
        // 1×1 (s-s)
        float sk_h_val = interp_sk_1(l_sk_h, base_idx, w);
        float sk_s_val = interp_sk_1(l_sk_s, base_idx, w);
        float dsk_h = interp_sk_1_d(l_sk_h, base_idx, wd) * inv_dr;
        float dsk_s = interp_sk_1_d(l_sk_s, base_idx, wd) * inv_dr;

        float h_val, dh_dx, dh_dy, dh_dz;
        block_1x1_with_derivs(sk_h_val, dsk_h, l, m, n, r, &h_val, &dh_dx, &dh_dy, &dh_dz);
        float s_val, ds_dx, ds_dy, ds_dz;
        block_1x1_with_derivs(sk_s_val, dsk_s, l, m, n, r, &s_val, &ds_dx, &ds_dy, &ds_dz);

        int dm_idx = dm_base + p.orb_j * n_orbs + p.orb_i;
        float dm_val = dm[dm_idx];
        float edm_val = edm[dm_idx];

        f.x = dm_val * dh_dx - edm_val * ds_dx;
        f.y = dm_val * dh_dy - edm_val * ds_dy;
        f.z = dm_val * dh_dz - edm_val * ds_dz;
    } else if (block_type == 1) {
        // 1×4 (s-p)
#if GPU_FORCE_DEBUG
        if (gid == 0) printf("GPU_FORCE_1X4 start: gid=%d r=%f dr=%f n_grid=%d base=%d orbs=(%u,%u) atoms=(%u,%u)\n", gid, r, dr, n_grid, base_idx, p.orb_i, p.orb_j, p.atom_i, p.atom_j);
#endif
        float2 sk_h_val = interp_sk_2(l_sk_h, base_idx, w);
        float2 sk_s_val = interp_sk_2(l_sk_s, base_idx, w);
        float2 dsk_h = interp_sk_2_d(l_sk_h, base_idx, wd) * inv_dr;
        float2 dsk_s = interp_sk_2_d(l_sk_s, base_idx, wd) * inv_dr;
#if GPU_FORCE_DEBUG
        if (gid == 0) printf("GPU_FORCE_1X4 interp: h=(%f,%f) s=(%f,%f) dh=(%f,%f) ds=(%f,%f)\n", sk_h_val.x, sk_h_val.y, sk_s_val.x, sk_s_val.y, dsk_h.x, dsk_h.y, dsk_s.x, dsk_s.y);
#endif

        float4 blk_h, dh_dx, dh_dy, dh_dz;
        block_1x4_with_derivs(sk_h_val, dsk_h, l, m, n, r, &blk_h, &dh_dx, &dh_dy, &dh_dz);
        float4 blk_s, ds_dx, ds_dy, ds_dz;
        block_1x4_with_derivs(sk_s_val, dsk_s, l, m, n, r, &blk_s, &ds_dx, &ds_dy, &ds_dz);
#if GPU_FORCE_DEBUG
        if (gid == 0) printf("GPU_FORCE_1X4 deriv: dhx=(%f,%f,%f,%f) dsx=(%f,%f,%f,%f)\n", dh_dx.x, dh_dx.y, dh_dx.z, dh_dx.w, ds_dx.x, ds_dx.y, ds_dx.z, ds_dx.w);
#endif

        for (int k = 0; k < 4; k++) {
            int dm_idx = dm_base + (p.orb_j + k) * n_orbs + p.orb_i;
            float dm_val = dm[dm_idx];
            float edm_val = edm[dm_idx];
            f.x += dm_val * dh_dx[k] - edm_val * ds_dx[k];
            f.y += dm_val * dh_dy[k] - edm_val * ds_dy[k];
            f.z += dm_val * dh_dz[k] - edm_val * ds_dz[k];
        }
#if GPU_FORCE_DEBUG
        if (gid == 0) printf("GPU_FORCE_1X4 contracted: f=(%f,%f,%f)\n", f.x, f.y, f.z);
#endif
    } else {
        // 4×4 (p-p)
        float ps_h, ps_s;
        float4 sk_h_val = interp_sk_5(l_sk_h, base_idx, w, &ps_h);
        float4 sk_s_val = interp_sk_5(l_sk_s, base_idx, w, &ps_s);
        float dps_h, dps_s;
        float4 dsk_h = interp_sk_5_d(l_sk_h, base_idx, wd, &dps_h) * inv_dr;
        float4 dsk_s = interp_sk_5_d(l_sk_s, base_idx, wd, &dps_s) * inv_dr;
        dps_h *= inv_dr;
        dps_s *= inv_dr;

        float4 blk_h[4], dh_dx[4], dh_dy[4], dh_dz[4];
        block_4x4_with_derivs(sk_h_val, dsk_h, ps_h, dps_h, l, m, n, r, blk_h, dh_dx, dh_dy, dh_dz);
        float4 blk_s[4], ds_dx[4], ds_dy[4], ds_dz[4];
        block_4x4_with_derivs(sk_s_val, dsk_s, ps_s, dps_s, l, m, n, r, blk_s, ds_dx, ds_dy, ds_dz);

        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                int dm_idx = dm_base + (p.orb_j + row) * n_orbs + (p.orb_i + col);
                float dm_val = dm[dm_idx];
                float edm_val = edm[dm_idx];
                f.x += dm_val * dh_dx[row][col] - edm_val * ds_dx[row][col];
                f.y += dm_val * dh_dy[row][col] - edm_val * ds_dy[row][col];
                f.z += dm_val * dh_dz[row][col] - edm_val * ds_dz[row][col];
            }
        }
    }

    // Convert: factor 2 (lower-triangle) * ANG2BOHR (Bohr→Å)
    float scale = 2.0f * ANG2BOHR_F;
    f *= scale;

    // Atomic add: F_i += f, F_j -= f
    int fi = 3 * ga_i;
    int fj = 3 * ga_j;
    atomic_add_f32(&forces[fi + 0], f.x);
    atomic_add_f32(&forces[fi + 1], f.y);
    atomic_add_f32(&forces[fi + 2], f.z);
    atomic_add_f32(&forces[fj + 0], -f.x);
    atomic_add_f32(&forces[fj + 1], -f.y);
    atomic_add_f32(&forces[fj + 2], -f.z);
#if GPU_FORCE_DEBUG
    if (gid == 0) printf("GPU_FORCE done: gid=%d fi=%d fj=%d f=(%f,%f,%f)\n", gid, fi, fj, f.x, f.y, f.z);
#endif
}

// ─── SCC shift force kernel (R6 component 2) ──────────────────────────
//
// Computes the SCC shift force contribution from pair blocks:
//   F_i[a] += 2 * ANG2BOHR * Σ_{μ∈i,ν∈j} (0.5*(V_i+V_j) * dS[μ,ν]/dR_a * DM[μ,ν])
//   F_j[a] -= same
//
// where V_i is the SCC potential (gamma·Δq) on atom i.
// Only needs dS/dR and DM (not EDM or dH/dR). Reuses the same SK table
// interpolation and block rotation derivative infrastructure.
//
// Arguments: same as force_pairs, but with v_shift replacing edm.
//   v_shift — [n_frags*n_atoms_frag] SCC potential per atom (Hartree)
//   dm      — density matrix (same as force_pairs)
__kernel void force_pairs_scc_shift(
    __global const PairEntry* pairs,
    __global const Fragment* fragments,
    __global const float* sk_s,       // only S table needed
    __global const float* dm,
    __global const float* v_shift,    // [total_atoms] SCC potential per atom
    __global float* forces,
    const float dr,
    const int n_grid,
    const int n_pairs,
    const int n_frags,
    const int block_type,
    const int n_sk_cols
) {
    const int tid = get_local_id(0);
    const int wg  = get_local_size(0);
    const int gid = get_global_id(0);

    __local float l_sk_s[SK_GRID_MAX * N_SK_COLS];
    int n_sk_elements = n_grid * n_sk_cols;
    for (int i = tid; i < n_sk_elements; i += wg) {
        l_sk_s[i] = sk_s[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid >= n_pairs) return;

    PairEntry p = pairs[gid];
    Fragment frag = fragments[p.replica];
    int n_orbs = frag.n_orbs;
    int atom_off = frag.atom_off;
    int ga_i = atom_off + p.atom_i;
    int ga_j = atom_off + p.atom_j;
    int dm_base = p.replica * n_orbs * n_orbs;

    // SCC shift: 0.5*(V_i + V_j) — the average SCC potential on atoms i and j
    float avg_shift = 0.5f * (v_shift[ga_i] + v_shift[ga_j]);

    // Value and derivative weights
    float4 w, wd;
    int base_idx = interp_params(p.r, dr, n_grid, &w, &wd);
    float inv_dr = 1.0f / dr;

    float l = p.l, m = p.m, n = p.n;
    float r = p.r;
    float3 f = (float3)(0.0f, 0.0f, 0.0f);

    if (block_type == 0) {
        // 1×1 (s-s)
        float sk_s_val = interp_sk_1(l_sk_s, base_idx, w);
        float dsk_s = interp_sk_1_d(l_sk_s, base_idx, wd) * inv_dr;

        float s_val, ds_dx, ds_dy, ds_dz;
        block_1x1_with_derivs(sk_s_val, dsk_s, l, m, n, r, &s_val, &ds_dx, &ds_dy, &ds_dz);

        int dm_idx = dm_base + p.orb_j * n_orbs + p.orb_i;
        float dm_val = dm[dm_idx];

        f.x = avg_shift * dm_val * ds_dx;
        f.y = avg_shift * dm_val * ds_dy;
        f.z = avg_shift * dm_val * ds_dz;
    } else if (block_type == 1) {
        // 1×4 (s-p)
        float2 sk_s_val = interp_sk_2(l_sk_s, base_idx, w);
        float2 dsk_s = interp_sk_2_d(l_sk_s, base_idx, wd) * inv_dr;

        float4 blk_s, ds_dx, ds_dy, ds_dz;
        block_1x4_with_derivs(sk_s_val, dsk_s, l, m, n, r, &blk_s, &ds_dx, &ds_dy, &ds_dz);

        for (int k = 0; k < 4; k++) {
            int dm_idx = dm_base + (p.orb_j + k) * n_orbs + p.orb_i;
            float dm_val = dm[dm_idx];
            f.x += avg_shift * dm_val * ds_dx[k];
            f.y += avg_shift * dm_val * ds_dy[k];
            f.z += avg_shift * dm_val * ds_dz[k];
        }
    } else {
        // 4×4 (p-p)
        float ps_s;
        float4 sk_s_val = interp_sk_5(l_sk_s, base_idx, w, &ps_s);
        float dps_s;
        float4 dsk_s = interp_sk_5_d(l_sk_s, base_idx, wd, &dps_s) * inv_dr;
        dps_s *= inv_dr;

        float4 blk_s[4], ds_dx[4], ds_dy[4], ds_dz[4];
        block_4x4_with_derivs(sk_s_val, dsk_s, ps_s, dps_s, l, m, n, r, blk_s, ds_dx, ds_dy, ds_dz);

        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                int dm_idx = dm_base + (p.orb_j + row) * n_orbs + (p.orb_i + col);
                float dm_val = dm[dm_idx];
                f.x += avg_shift * dm_val * ds_dx[row][col];
                f.y += avg_shift * dm_val * ds_dy[row][col];
                f.z += avg_shift * dm_val * ds_dz[row][col];
            }
        }
    }

    // Convert: factor 2 (lower-triangle) * ANG2BOHR (Bohr→Å)
    float scale = 2.0f * ANG2BOHR_F;
    f *= scale;

    // Atomic add: F_i += f, F_j -= f
    int fi = 3 * ga_i;
    int fj = 3 * ga_j;
    atomic_add_f32(&forces[fi + 0], f.x);
    atomic_add_f32(&forces[fi + 1], f.y);
    atomic_add_f32(&forces[fi + 2], f.z);
    atomic_add_f32(&forces[fj + 0], -f.x);
    atomic_add_f32(&forces[fj + 1], -f.y);
    atomic_add_f32(&forces[fj + 2], -f.z);
}

// ─── Gamma derivative (SCC double-counting) force kernel (R6 component 3) ─
//
// Computes the SCC double-counting (Coulomb) force contribution:
//   F_i[a] += -dq_i * dq_j * gamma'(r) / r * (coord_i - coord_j)_a * ANG2BOHR^2
//   F_j[a] -= same
//
// where gamma'(r) is the full gamma derivative (short-range + Coulomb 1/R^2).
// One workgroup per system, threads handle pairs.
//
// Arguments:
//   n_atoms     — atoms per system
//   batch       — number of systems
//   coords      — [batch*n_atoms*3] coordinates (Å)
//   species_idx — [batch*n_atoms] species index
//   delta_q     — [batch*n_atoms] Δq = q_elec - q0
//   u_hub       — [n_species] Hubbard U per species (Hartree)
//   n_species   — number of species
//   forces      — [batch*n_atoms*3] force accumulator (Hartree/Å)

#define TAU_FACTOR_F 3.2f
#define SAME_U_C0_F 0.6875f
#define SAME_U_C1_F 0.1875f
#define SAME_U_C2_F 0.0208333333f
#define MIN_HUB_DIFF_F 1e-4f
#define TOL_SAME_DIST_F 1e-10f

// gamma_full'(R) = -1/R^2 - S'(R)
inline float gamma_prime_full_f32(float r, float u1, float u2) {
    if (r < TOL_SAME_DIST_F) return 0.0f;
    float short_prime;
    if (fabs(u1 - u2) < MIN_HUB_DIFF_F) {
        float tau = TAU_FACTOR_F * 0.5f * (u1 + u2);
        float e = exp(-tau * r);
        float poly = 1.0f/r + SAME_U_C0_F*tau + SAME_U_C1_F*r*tau*tau
                   + SAME_U_C2_F*r*r*tau*tau*tau;
        float poly_prime = -1.0f/(r*r) + SAME_U_C1_F*tau*tau
                        + 2.0f*SAME_U_C2_F*r*tau*tau*tau;
        short_prime = -tau*e*poly + e*poly_prime;
    } else {
        float tau1 = TAU_FACTOR_F * u1;
        float tau2 = TAU_FACTOR_F * u2;
        float dt2_a = tau1*tau1 - tau2*tau2;
        float dt2_b = tau2*tau2 - tau1*tau1;
        float s_a, s_b;
        {
            float dt2 = dt2_a;
            float dt2_sq = dt2*dt2;
            float dt2_cu = dt2_sq*dt2;
            float term_a = 0.5f * pown(tau2,4) * tau1 / dt2_sq;
            float term_b = (pown(tau2,6) - 3.0f*pown(tau2,4)*tau1*tau1) / (r * dt2_cu);
            float e = exp(-tau1 * r);
            s_a = -tau1*e*(term_a - term_b) + e*(term_b/r);
        }
        {
            float dt2 = dt2_b;
            float dt2_sq = dt2*dt2;
            float dt2_cu = dt2_sq*dt2;
            float term_a = 0.5f * pown(tau1,4) * tau2 / dt2_sq;
            float term_b = (pown(tau1,6) - 3.0f*pown(tau1,4)*tau2*tau2) / (r * dt2_cu);
            float e = exp(-tau2 * r);
            s_b = -tau2*e*(term_a - term_b) + e*(term_b/r);
        }
        short_prime = s_a + s_b;
    }
    return -1.0f/(r*r) - short_prime;
}

__kernel void force_gamma_deriv_batched(
    const int n_atoms,
    const int batch,
    __global const float* coords,     // [batch*n_atoms*3] Å
    __global const int* species_idx,   // [batch*n_atoms]
    __global const float* delta_q,    // [batch*n_atoms]
    __global const float* u_hub,      // [n_species]
    const int n_species,
    __global float* forces            // [batch*n_atoms*3] Hartree/Å
) {
    const int sid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch) return;

    __global const float* crd = coords + (size_t)sid * n_atoms * 3;
    __global const int* spc = species_idx + (size_t)sid * n_atoms;
    __global const float* dq = delta_q + (size_t)sid * n_atoms;
    __global float* frc = forces + (size_t)sid * n_atoms * 3;

    const int npairs = n_atoms * (n_atoms - 1) / 2;
    for (int pair_id = lid; pair_id < npairs; pair_id += lsz) {
        int i = 0, j = 0, acc = 0, found = 0;
        for (int ii = 0; ii < n_atoms - 1 && !found; ++ii) {
            int n_j = n_atoms - 1 - ii;
            if (acc + n_j > pair_id) {
                i = ii; j = ii + 1 + (pair_id - acc); found = 1;
            }
            acc += n_j;
        }
        if (!found) continue;

        float dx = crd[i*3+0] - crd[j*3+0];
        float dy = crd[i*3+1] - crd[j*3+1];
        float dz = crd[i*3+2] - crd[j*3+2];
        float r2_ang = dx*dx + dy*dy + dz*dz;
        if (r2_ang < TOL_SAME_DIST_F) continue;
        float r_ang = sqrt(r2_ang);
        float r_bohr = r_ang * ANG2BOHR_F;

        int si = spc[i], sj = spc[j];
        float u_i = u_hub[si];
        float u_j = u_hub[sj];

        float gprime = gamma_prime_full_f32(r_bohr, u_i, u_j);
        float dq_i = dq[i], dq_j = dq[j];

        float coeff = -dq_i * dq_j * gprime / r_bohr * ANG2BOHR_F * ANG2BOHR_F;
        float fx = coeff * dx;
        float fy = coeff * dy;
        float fz = coeff * dz;

        atomic_add_f32(&frc[i*3+0], fx);
        atomic_add_f32(&frc[i*3+1], fy);
        atomic_add_f32(&frc[i*3+2], fz);
        atomic_add_f32(&frc[j*3+0], -fx);
        atomic_add_f32(&frc[j*3+1], -fy);
        atomic_add_f32(&frc[j*3+2], -fz);
    }
}

// ─── Repulsive pair force kernel (R6 component 4) ──────────────────────
//
// Computes the repulsive pair-potential force contribution:
//   F_i[a] += dE_rep/dr * (coord_j - coord_i)_a / r_ang * ANG2BOHR
//   F_j[a] -= same
//
// One workgroup per system. Spline layout matches repulsive_energy_batched.

#ifndef REP_MAX_INTERVALS
#define REP_MAX_INTERVALS 30
#endif

__kernel void force_repulsive_batched(
    const int n_atoms,
    const int batch,
    __global const float* coords,
    __global const int* species_idx,
    __global const int* spline_offsets,
    const int n_species,
    __global const float* spline_data,
    __global float* forces
) {
    const int sid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch) return;

    __global const float* crd = coords + (size_t)sid * n_atoms * 3;
    __global const int* spc = species_idx + (size_t)sid * n_atoms;
    __global float* frc = forces + (size_t)sid * n_atoms * 3;

    const int npairs = n_atoms * (n_atoms - 1) / 2;
    for (int pair_id = lid; pair_id < npairs; pair_id += lsz) {
        int i = 0, j = 0, acc = 0, found = 0;
        for (int ii = 0; ii < n_atoms - 1 && !found; ++ii) {
            int n_j = n_atoms - 1 - ii;
            if (acc + n_j > pair_id) {
                i = ii; j = ii + 1 + (pair_id - acc); found = 1;
            }
            acc += n_j;
        }
        if (!found) continue;

        int si = spc[i], sj = spc[j];
        int off_idx = si * n_species + sj;
        int offset = spline_offsets[off_idx];
        if (offset < 0) continue;

        float dx = crd[j*3+0] - crd[i*3+0];
        float dy = crd[j*3+1] - crd[i*3+1];
        float dz = crd[j*3+2] - crd[i*3+2];
        float r = sqrt(dx*dx + dy*dy + dz*dz);

        __global const float* sd = spline_data + offset;
        int n_int = as_int(sd[0]);
        float cutoff = sd[1];
        if (r >= cutoff || r < 1.0e-6f) continue;

        float exp_a = sd[2], exp_b = sd[3];
        __global const float* x_start = sd + 5;
        __global const float* sp_coeffs = sd + 5 + REP_MAX_INTERVALS;
        __global const float* sp_last = sd + 5 + REP_MAX_INTERVALS + (REP_MAX_INTERVALS - 1) * 4;

        float de_val = 0.0f;

        if (r < x_start[0]) {
            de_val = -exp_a * exp(-exp_a * r + exp_b);
        } else if (r >= x_start[n_int - 1]) {
            float dr = r - x_start[n_int - 1];
            de_val = sp_last[1] + 2.0f*sp_last[2]*dr + 3.0f*sp_last[3]*dr*dr
                  + 4.0f*sp_last[4]*dr*dr*dr + 5.0f*sp_last[5]*dr*dr*dr*dr;
        } else {
            int lo = 0, hi = n_int - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (x_start[mid] <= r) lo = mid; else hi = mid;
            }
            float dr = r - x_start[lo];
            __global const float* c = sp_coeffs + lo * 4;
            de_val = c[1] + 2.0f*c[2]*dr + 3.0f*c[3]*dr*dr;
        }

        if (de_val == 0.0f) continue;
        float de_ang = de_val * ANG2BOHR_F;
        float inv_r = 1.0f / r;
        float fx = de_ang * dx * inv_r;
        float fy = de_ang * dy * inv_r;
        float fz = de_ang * dz * inv_r;

        atomic_add_f32(&frc[i*3+0], fx);
        atomic_add_f32(&frc[i*3+1], fy);
        atomic_add_f32(&frc[i*3+2], fz);
        atomic_add_f32(&frc[j*3+0], -fx);
        atomic_add_f32(&frc[j*3+1], -fy);
        atomic_add_f32(&frc[j*3+2], -fz);
    }
}
