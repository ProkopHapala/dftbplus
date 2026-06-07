//! OpenCL kernels for DFTB Hamiltonian and overlap assembly.
//!
//! Design:
//! - One kernel `onsite_and_va`  : per-replica workgroup, writes H onsite diagonal + computes V_A.
//! - One kernel `assemble_pairs` : launched per species-pair bucket, loads compact SK table into
//!                               __local, interpolates, rotates, adds H1 electrostatics,
//!                               writes symmetric H and S blocks.
//!
//! Precision: f32 throughout (GPU is ~10x slower in f64).
//!
//! Block types handled inside `assemble_pairs` (uniform branch, no warp divergence):
//!   0 : 1x1  (s-s, e.g. H-H)
//!   1 : 1x4  (s-sp, e.g. H-C)
//!   2 : 4x4  (sp-sp, e.g. C-C, C-O, N-O)

// ------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------
#ifndef SK_GRID_MAX
#define SK_GRID_MAX 1024
#endif

#define TAU_FACTOR 3.2f
#define SAME_U_C0  0.6875f
#define SAME_U_C1  0.1875f
#define SAME_U_C2  0.020833333333333333f

// ------------------------------------------------------------------
// PairEntry layout (32 bytes, packed)
// ------------------------------------------------------------------
typedef struct {
    uint   replica;   // which fragment/replica
    ushort atom_i;    // atom index i   (local to fragment)
    ushort atom_j;    // atom index j   (local to fragment)
    ushort orb_i;     // orbital offset of atom i (local)
    ushort orb_j;     // orbital offset of atom j (local)
    float  r;         // distance
    float  l, m, n;   // direction cosines
} PairEntry;

// ------------------------------------------------------------------
// Fragment metadata (16 bytes)
// ------------------------------------------------------------------
typedef struct {
    int n_atoms;      // number of atoms in this fragment
    int n_orbs;       // number of orbitals in this fragment
    int atom_off;     // prefix sum offset into flat per-atom arrays
    int H_base;       // prefix sum offset into flat H/S output
} Fragment;

// ------------------------------------------------------------------
// Gamma function (f32 port of gamma.rs)
// ------------------------------------------------------------------
inline float exp_gamma_same_u(float r, float tau_mean) {
    float x = -tau_mean * r;
    float e = exp(x);
    float r_inv = 1.0f / r;
    return e * (r_inv
                + SAME_U_C0 * tau_mean
                + SAME_U_C1 * r * tau_mean * tau_mean
                + SAME_U_C2 * r * r * tau_mean * tau_mean * tau_mean);
}

inline float gamma_sub_exprn(float r, float tau1, float tau2) {
    float dt2     = tau1 * tau1 - tau2 * tau2;
    float dt2_sq  = dt2 * dt2;
    float dt2_cu  = dt2_sq * dt2;
    float term_a  = 0.5f * tau2 * tau2 * tau2 * tau2 * tau1 / dt2_sq;
    float term_b  = (tau2 * tau2 * tau2 * tau2 * tau2 * tau2
                     - 3.0f * tau2 * tau2 * tau2 * tau2 * tau1 * tau1)
                    / (r * dt2_cu);
    return exp(-tau1 * r) * (term_a - term_b);
}

// CONSIDER: branch divergence in gamma_full (r<1e-10, du<1e-4).
//           Could pre-sort pairs by u-similarity or use select() if target device supports fast predication.
inline float gamma_full(float r, float u1, float u2) {
    if (r < 1.0e-10f) {
        return 0.5f * (u1 + u2);
    }
    float tau1 = TAU_FACTOR * u1;
    float tau2 = TAU_FACTOR * u2;

    float short_range;
    float du = u1 - u2;
    if (du < 0.0f) du = -du;
    if (du < 1.0e-4f) {
        short_range = exp_gamma_same_u(r, 0.5f * (tau1 + tau2));
    } else {
        short_range = gamma_sub_exprn(r, tau1, tau2)
                    + gamma_sub_exprn(r, tau2, tau1);
    }
    return 1.0f / r - short_range;
}

// ------------------------------------------------------------------
// Cubic B-spline basis weights (shared)
// ------------------------------------------------------------------
inline float4 cubic_weights(float t) {
    float omt = 1.0f - t;
    return (float4)(
        omt * omt * omt * 0.16666667f,
        (3.0f * t * t * t - 6.0f * t * t + 4.0f) * 0.16666667f,
        (-3.0f * t * t * t + 3.0f * t * t + 3.0f * t + 1.0f) * 0.16666667f,
        t * t * t * 0.16666667f
    );
}

// Combined: compute base index and weights in one call (avoids redundant r/dr and floor)
inline int cubic_interp_params(float r, float dr, int n_grid, float4* out_w) {
    if (r < 0.0f || r >= (n_grid - 1) * dr) {
        *out_w = (float4)(0.0f);
        return -1;
    }
    float u = r / dr;
    int i = (int)u;
    if (i < 1) i = 1;
    if (i > n_grid - 3) i = n_grid - 3;
    float t = u - i;
    *out_w = cubic_weights(t);
    return i - 1;
}

// ------------------------------------------------------------------
// 1-channel: scalar with dot()
// ------------------------------------------------------------------
inline float interp_sk_1_indexed(__local const float* tab, int base, float4 w) {
    if (base < 0) return 0.0f;
    __local float4* tab4 = (__local float4*)tab;
    return dot(tab4[base], w);
}

// ------------------------------------------------------------------
// 2-channel: float2 cast (1x4: ss, sp)
// ------------------------------------------------------------------
inline float2 interp_sk_2_indexed(__local const float* tab, int base, float4 w) {
    if (base < 0) return (float2)(0.0f);
    __local float2* tab2 = (__local float2*)tab;
    float2 v0 = tab2[base    ];
    float2 v1 = tab2[base + 1];
    float2 v2 = tab2[base + 2];
    float2 v3 = tab2[base + 3];
    return v0 * w.x + v1 * w.y + v2 * w.z + v3 * w.w;
}

// ------------------------------------------------------------------
// 4-channel: float4 cast (4x4: ss, sp, pp_sig, pp_pi)
// ------------------------------------------------------------------

// Indexed version: base and weights pre-computed (avoid re-evaluating index)
inline float4 interp_sk_4_indexed(__local const float* tab, int base, float4 w) {
    if (base < 0) return (float4)(0.0f);
    __local float4* tab4 = (__local float4*)tab;
    float4 v0 = tab4[base    ];
    float4 v1 = tab4[base + 1];
    float4 v2 = tab4[base + 2];
    float4 v3 = tab4[base + 3];
    return v0 * w.x + v1 * w.y + v2 * w.z + v3 * w.w;
}

// ------------------------------------------------------------------
// Rotation helpers
// ------------------------------------------------------------------
// Orbital ordering: s, py, pz, px  (DFTB+ convention)

// 1x4: s on i, sp on j  (4 rows x 1 col)
// sk = (ss, sp)
// blk as float4: (ss, sp_py, sp_pz, sp_px)
inline void rotate_1x4(float l, float m, float n, float2 sk, float4* blk) {
    float4 sp = (float4)(sk.x, m, n, l) * sk.y;
    *blk = sp;
}

// 4x4: sp-sp block (4x4)
// sk = (ss, sp, pp_sig, pp_pi)
// blk as float4[4]: row-major 4x4
inline void rotate_4x4(float l, float m, float n, float4 sk, float4* blk) {
    float4 v = (float4)(m, n, l, 0.0f);  // (py, pz, px, pad)
    float diff = sk.z - sk.w;
    float4 diff4 = (float4)(diff, diff, diff, diff);

    // Row 0: (ss, -sp_py, -sp_pz, -sp_px)
    blk[0] = (float4)(sk.x, -sk.y * m, -sk.y * n, -sk.y * l);

    // TODO: deduplicate pp outer product v.yzw * v.yzw * diff4.yzw + sk.w * v.yzw
    //       (currently computed 3x identically for rows 1..3).  
    //       Saves 6 mul + 3 add per rotation.
    float4 row1 = v * sk.y;
    row1.x = sk.y * m;
    row1.yzw = v.yzw * v.yzw * diff4.yzw + sk.w * v.yzw;
    blk[1] = row1;

    float4 row2 = v * sk.y;
    row2.x = sk.y * n;
    row2.yzw = v.yzw * v.yzw * diff4.yzw + sk.w * v.yzw;
    blk[2] = row2;

    float4 row3 = v * sk.y;
    row3.x = sk.y * l;
    row3.yzw = v.yzw * v.yzw * diff4.yzw + sk.w * v.yzw;
    blk[3] = row3;
}

// ------------------------------------------------------------------
// Write helpers (symmetric write to global H/S)
// ------------------------------------------------------------------
inline void write_symmetric_1x1(
    __global float* M,
    int n_orbs, int base,
    ushort orb_i, ushort orb_j,
    float val
) {
    int ij = base + orb_i * n_orbs + orb_j;
    int ji = base + orb_j * n_orbs + orb_i;
    M[ij] = val;  M[ji] = val;
}

inline void write_symmetric_1x4(
    __global float* M,
    int n_orbs, int base,
    ushort orb_i, ushort orb_j,
    const float4* blk
) {
    float4 v = *blk;
    // Direct block: 4 rows (j) x 1 col (i)
    M[base + (orb_j    ) * n_orbs + orb_i] =  v.x;
    M[base + (orb_j + 1) * n_orbs + orb_i] =  v.y;
    M[base + (orb_j + 2) * n_orbs + orb_i] =  v.z;
    M[base + (orb_j + 3) * n_orbs + orb_i] =  v.w;
    // Transpose: row orb_i, cols orb_j..orb_j+3  (ss symmetric, sp→-ps)
    M[base + orb_i * n_orbs +  orb_j    ] =   v.x;
    M[base + orb_i * n_orbs + (orb_j + 1)] =  -v.y;
    M[base + orb_i * n_orbs + (orb_j + 2)] =  -v.z;
    M[base + orb_i * n_orbs + (orb_j + 3)] =  -v.w;
}

inline void write_symmetric_4x4(
    __global float* M,
    int n_orbs, int base,
    ushort orb_i, ushort orb_j,
    const float4* blk
) {
    // Write 4x4 block at (orb_j, orb_i) and its transpose at (orb_i, orb_j)
    // blk is float4[4] row-major
    __global float4* M4 = (__global float4*)M;
    int row_j_base = base + orb_j * n_orbs;
    int row_i_base = base + orb_i * n_orbs;

    // Direct block: rows orb_j..orb_j+3, cols orb_i..orb_i+3
    M4[row_j_base + orb_i] = blk[0];
    M4[row_j_base + orb_i + 1] = blk[1];
    M4[row_j_base + orb_i + 2] = blk[2];
    M4[row_j_base + orb_i + 3] = blk[3];

    // Transpose block: rows orb_i..orb_i+3, cols orb_j..orb_j+3
    M4[row_i_base + orb_j    ] = (float4)(blk[0].x, blk[1].x, blk[2].x, blk[3].x);
    M4[row_i_base + orb_j + 1] = (float4)(blk[0].y, blk[1].y, blk[2].y, blk[3].y);
    M4[row_i_base + orb_j + 2] = (float4)(blk[0].z, blk[1].z, blk[2].z, blk[3].z);
    M4[row_i_base + orb_j + 3] = (float4)(blk[0].w, blk[1].w, blk[2].w, blk[3].w);
}

// ------------------------------------------------------------------
// Kernel 1: onsite H0 + V_A computation
// ------------------------------------------------------------------
// One workgroup per fragment.
// Writes onsite diagonal elements of H and computes V_A = sum_C q_C * gamma_AC.
// V_A is a pure gather: each thread handles one atom, loops over its
// CSR neighbor list, accumulates q_j * gamma(r_ij). No atomics.
//
// All per-atom arrays are flat (concatenated across fragments).
// Fragment metadata provides local-to-global mapping.
__kernel void onsite_and_va(
    __global const Fragment* fragments,      // [n_frags]
    __global const int*    atom_species,   // [total_atoms]  global species index
    __global const int*    orb_offsets,    // [total_atoms]  local offset within fragment
    __global const float2* onsite_es_ep,   // [n_global_species]  (e_s, e_p)
    __global const float*  charges,        // [total_atoms]
    __global const float*  hubbard_u,      // [n_global_species]
    __global const int*    neigh_offsets,  // [total_atoms+1]  CSR row pointer (flat)
    __global const int*    neigh_j,        // [total_neigh]    local neighbor atom index
    __global const float*  neigh_r,        // [total_neigh]    distance
    __global float*        H_out,          // flat [total_H_elements]
    __global float*        V_out,          // flat [total_atoms]
    const int n_frags
)
{
    const int tid     = get_local_id(0);
    const int wg      = get_local_size(0);
    const int frag_id = get_group_id(0);

    if (frag_id >= n_frags) return;

    Fragment frag = fragments[frag_id];
    int n_atoms = frag.n_atoms;
    int n_orbs  = frag.n_orbs;
    int atom_off = frag.atom_off;
    int base_H = frag.H_base;

    // 1. Write onsite H0 diagonal
    for (int a = tid; a < n_atoms; a += wg) {
        int ga  = atom_off + a;                // global atom index
        int sp  = atom_species[ga];
        int off = orb_offsets[ga];
        float2 e = onsite_es_ep[sp];
        H_out[base_H + off * n_orbs + off] = e.x;
        H_out[base_H + (off + 1) * n_orbs + (off + 1)] = e.y;
        H_out[base_H + (off + 2) * n_orbs + (off + 2)] = e.y;
        H_out[base_H + (off + 3) * n_orbs + (off + 3)] = e.y;
    }

    // 2. Compute V_A per atom (gather, no atomics)
    // CONSIDER: neighbor list order affects cache locality.
    //           Could sort neighbors by atom index or by spatial bins.
    for (int a = tid; a < n_atoms; a += wg) {
        int ga = atom_off + a;
        int sp = atom_species[ga];
        float q = charges[ga];
        float v = q * hubbard_u[sp];

        int start = neigh_offsets[ga];
        int end   = neigh_offsets[ga + 1];
        for (int n = start; n < end; n++) {
            int b_local = neigh_j[n];         // 0..n_atoms-1 within this fragment
            int gb = atom_off + b_local;      // global atom index
            int sb = atom_species[gb];
            float qb = charges[gb];
            float g  = gamma_full(neigh_r[n], hubbard_u[sp], hubbard_u[sb]);
            v += qb * g;
        }
        V_out[ga] = v;
    }
}

// ------------------------------------------------------------------
// Kernel 2: pairwise assembly (H0 + H1 + S)
// ------------------------------------------------------------------
// Launched per species-pair bucket. All threads in workgroup process
// the same block type (uniform branch).
__kernel void assemble_pairs(
    __global const PairEntry* pairs,       // [n_pairs] for this bucket
    __global const Fragment* fragments,    // [n_frags]  per-fragment metadata
    __global const float*   sk_h,          // compact SK table [n_grid * n_sk_cols]
    __global const float*   sk_s,
    __global const float*   V_a,           // [total_atoms]
    __global float*         H_out,         // flat [total_H_elements]
    __global float*         S_out,
    const float dr,
    const int n_grid,
    const int n_pairs,
    const int block_type       // 0=1x1, 1=1x4, 2=4x4  (uniform for whole launch)
)
{
    const int tid = get_local_id(0);
    const int wg  = get_local_size(0);
    const int gid = get_global_id(0);

    // --- CACHE SK TABLE INTO __LOCAL ---
    int n_sk_cols = (block_type == 0) ? 1 : (block_type == 1) ? 2 : 4;
    __local float l_sk_h[SK_GRID_MAX * 4];
    __local float l_sk_s[SK_GRID_MAX * 4];

    for (int i = tid; i < n_grid * n_sk_cols; i += wg) {
        l_sk_h[i] = sk_h[i];
        l_sk_s[i] = sk_s[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid >= n_pairs) return;

    PairEntry p = pairs[gid];

    // --- PER-FRAGMENT LOOKUP ---
    Fragment frag = fragments[p.replica];
    int n_orbs  = frag.n_orbs;
    int atom_off = frag.atom_off;
    int base = frag.H_base;

    float va = V_a[atom_off + p.atom_i];
    float vb = V_a[atom_off + p.atom_j];
    float h1_factor = 0.5f * (va + vb);

    // Compute index and weights once (same for all block types)
    // TODO: size l_sk_h/l_sk_s dynamically to n_sk_cols (avoids 2-8x local memory waste)
    float4 w;
    int    base_idx = cubic_interp_params(p.r, dr, n_grid, &w);

    if (block_type == 0) {
        // Interpolate with pre-computed index
        float sk_h = interp_sk_1_indexed(l_sk_h, base_idx, w);
        float sk_s = interp_sk_1_indexed(l_sk_s, base_idx, w);
        write_symmetric_1x1(H_out, n_orbs, base, p.orb_i, p.orb_j, sk_h + h1_factor * sk_s);
        write_symmetric_1x1(S_out, n_orbs, base, p.orb_i, p.orb_j, sk_s);
    } else if (block_type == 1) {
        // Interpolate with pre-computed index
        float2 sk_h = interp_sk_2_indexed(l_sk_h, base_idx, w);
        float2 sk_s = interp_sk_2_indexed(l_sk_s, base_idx, w);
        // Use single blk: compute S first, then H = H0 + h1_factor * S
        float4 blk;
        rotate_1x4(p.l, p.m, p.n, sk_s, &blk);
        write_symmetric_1x4(S_out, n_orbs, base, p.orb_i, p.orb_j, &blk);
        rotate_1x4(p.l, p.m, p.n, sk_h + h1_factor * sk_s, &blk);
        write_symmetric_1x4(H_out, n_orbs, base, p.orb_i, p.orb_j, &blk);
    } else {
        // Interpolate with pre-computed index
        // Use single blk array: compute S first, then H = H0 + h1_factor * S
        float4 blk[4];
        float4 sk_s = interp_sk_4_indexed(l_sk_s, base_idx, w);
        rotate_4x4(p.l, p.m, p.n, sk_s, blk);
        write_symmetric_4x4(S_out, n_orbs, base, p.orb_i, p.orb_j, blk);
        // Compute H in-place: blk = rotate(sk_h) + h1_factor * rotate(sk_s)
        // Since rotate is linear: rotate(sk_h + h1_factor * sk_s) = rotate(sk_h) + h1_factor * rotate(sk_s)
        float4 sk_h = interp_sk_4_indexed(l_sk_h, base_idx, w);
        rotate_4x4(p.l, p.m, p.n, sk_h + h1_factor * sk_s, blk);
        write_symmetric_4x4(H_out, n_orbs, base, p.orb_i, p.orb_j, blk);
    }
}
