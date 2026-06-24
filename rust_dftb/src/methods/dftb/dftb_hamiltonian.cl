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
#define SK_GRID_MAX 256
#endif

#ifndef N_SK_COLS
#define N_SK_COLS 4
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

// Branch-minimized gamma: fabs() instead of branch, r<1e-10 handled by host.
// The same_u vs different_u branch remains but is unavoidable without pre-sorting.
inline float gamma_full(float r, float u1, float u2) {
    float tau1 = TAU_FACTOR * u1;
    float tau2 = TAU_FACTOR * u2;

    float du = fabs(u1 - u2);
    float short_range;
    if (du < 1.0e-4f) {
        short_range = exp_gamma_same_u(r, 0.5f * (tau1 + tau2));
    } else {
        short_range = gamma_sub_exprn(r, tau1, tau2)
                    + gamma_sub_exprn(r, tau2, tau1);
    }
    return 1.0f / r - short_range;
}

// ------------------------------------------------------------------
// Cubic B-spline interpolation (4-point stencil, 1 float per node)
//
// SK tables are resampled to 32-64 points on the host using natural cubic
// spline fitting, then stored as plain float values. The GPU interpolates
// using the cubic B-spline 4-point stencil:
//   val = w0*f[i-1] + w1*f[i] + w2*f[i+1] + w3*f[i+2]
//
// This requires only 1 float per node (vs 2 for Hermite), and the 4-point
// stencil shares data between neighbors, minimizing local memory usage.
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

// Compute base index and weights. Returns base = i-1 for 4-point stencil.
inline int cubic_interp_params(float r, float dr, int n_grid, float4* out_w) {
    float u = r / dr;
    int i = (int)u;
    i = clamp(i, 1, n_grid - 3);
    float t = u - (float)i;
    *out_w = cubic_weights(t);
    return i - 1;
}

// 1-channel: scalar with dot()
inline float interp_sk_1(__local const float* tab, int base, float4 w) {
    __local float4* tab4 = (__local float4*)tab;
    return dot(tab4[base], w);
}

// 2-channel: float2 cast (1x4: ss, sp)
inline float2 interp_sk_2(__local const float* tab, int base, float4 w) {
    __local float2* tab2 = (__local float2*)tab;
    float2 v0 = tab2[base    ];
    float2 v1 = tab2[base + 1];
    float2 v2 = tab2[base + 2];
    float2 v3 = tab2[base + 3];
    return v0 * w.x + v1 * w.y + v2 * w.z + v3 * w.w;
}

// 4-channel: float4 cast (4x4: ss, sp, pp_sig, pp_pi)
inline float4 interp_sk_4(__local const float* tab, int base, float4 w) {
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

    // Row 0: (ss, -sp_py, -sp_pz, -sp_px)
    blk[0] = (float4)(sk.x, -sk.y * m, -sk.y * n, -sk.y * l);

    // pp outer product: v.yzw * v.yzw * diff + sk.w * v.yzw — computed once, reused for rows 1..3
    float3 pp = v.yzw * v.yzw * diff + sk.w * v.yzw;

    blk[1] = (float4)(sk.y * m, pp.x, pp.y, pp.z);
    blk[2] = (float4)(sk.y * n, pp.x, pp.y, pp.z);
    blk[3] = (float4)(sk.y * l, pp.x, pp.y, pp.z);
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
// Kernel 0: onsite H0 diagonal (separate, coalesced)
// ------------------------------------------------------------------
// Writes only the diagonal elements of H0 for all fragments in a single launch.
// Each thread writes one atom's diagonal (up to 4 values).
// This is coalesced because consecutive threads write to consecutive atoms.
//
// Onsite diagonal is just onsite_es_ep[species] placed at known positions —
// no interpolation, no rotation, no distance dependence. Separating this from
// the V_A kernel avoids the uncoalesced strided writes that dominated before.
__kernel void onsite_diagonal(
    __global const Fragment* fragments,      // [n_frags]
    __global const int*    atom_species,   // [total_atoms]
    __global const int*    orb_offsets,    // [total_atoms]
    __global const float2* onsite_es_ep,   // [n_global_species]  (e_s, e_p)
    __global float*        H_out,          // flat [total_H_elements]
    const int n_frags,
    const int total_atoms
)
{
    const int gid = get_global_id(0);
    if (gid >= total_atoms) return;

    // Find which fragment this atom belongs to.
    // For simplicity, we pass a pre-computed frag_id per atom from host.
    // But since fragments are contiguous, we can binary search or use a map.
    // Simpler: host passes frag_id_per_atom[] array.
    // To avoid extra buffer, we iterate fragments (n_frags is small).
    int frag_id = 0;
    for (int f = 0; f < n_frags; f++) {
        int next_off = fragments[f].atom_off + fragments[f].n_atoms;
        if (gid < next_off) { frag_id = f; break; }
        frag_id = f;
    }

    Fragment frag = fragments[frag_id];
    int n_orbs  = frag.n_orbs;
    int base_H  = frag.H_base;
    int off     = orb_offsets[gid];
    int sp      = atom_species[gid];
    float2 e    = onsite_es_ep[sp];

    H_out[base_H + off * n_orbs + off] = e.x;
    H_out[base_H + (off + 1) * n_orbs + (off + 1)] = e.y;
    H_out[base_H + (off + 2) * n_orbs + (off + 2)] = e.y;
    H_out[base_H + (off + 3) * n_orbs + (off + 3)] = e.y;
}

// ------------------------------------------------------------------
// Kernel 1: V_A computation (gamma electrostatics)
// ------------------------------------------------------------------
// One workgroup per fragment.
// Computes V_A = sum_C q_C * gamma_AC. Pure gather, no atomics.
// Per-fragment data (charges, species) cached in __local for the inner loop.
//
// Onsite H0 diagonal is handled by the separate onsite_diagonal kernel.
__kernel void onsite_and_va(
    __global const Fragment* fragments,      // [n_frags]
    __global const int*    atom_species,   // [total_atoms]  global species index
    __global const float*  charges,        // [total_atoms]
    __global const float*  hubbard_u,      // [n_global_species]
    __global const int*    neigh_offsets,  // [total_atoms+1]  CSR row pointer (flat)
    __global const int*    neigh_j,        // [total_neigh]    local neighbor atom index
    __global const float*  neigh_r,        // [total_neigh]    distance
    __global float*        V_out,          // flat [total_atoms]
    const int n_frags,
    const int n_global_species
)
{
    const int tid     = get_local_id(0);
    const int wg      = get_local_size(0);
    const int frag_id = get_group_id(0);

    if (frag_id >= n_frags) return;

    Fragment frag = fragments[frag_id];
    int n_atoms  = frag.n_atoms;
    int atom_off = frag.atom_off;

    // --- Cache per-fragment data into __local ---
    // Charges and species for this fragment's atoms (accessed in inner loop)
    __local float l_charges[256];
    __local int   l_species[256];
    // Hubbard U for all species (small, accessed per-neighbor)
    __local float l_u[64];

    for (int i = tid; i < n_atoms && i < 256; i += wg) {
        l_charges[i] = charges[atom_off + i];
        l_species[i] = atom_species[atom_off + i];
    }
    for (int i = tid; i < n_global_species && i < 64; i += wg) {
        l_u[i] = hubbard_u[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute V_A per atom (gather, no atomics)
    for (int a = tid; a < n_atoms; a += wg) {
        int sp = l_species[a];
        float q = l_charges[a];
        float v = q * l_u[sp];

        int ga = atom_off + a;
        int start = neigh_offsets[ga];
        int end   = neigh_offsets[ga + 1];
        for (int n = start; n < end; n++) {
            int b_local = neigh_j[n];         // 0..n_atoms-1 within this fragment
            int sb = l_species[b_local];
            float qb = l_charges[b_local];
            float g  = gamma_full(neigh_r[n], l_u[sp], l_u[sb]);
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
    const int n_frags,
    const int block_type       // 0=1x1, 1=1x4, 2=4x4  (uniform for whole launch)
)
{
    const int tid = get_local_id(0);
    const int wg  = get_local_size(0);
    const int gid = get_global_id(0);

    // --- CACHE SK TABLE INTO __local (1 float per node, B-spline stencil) ---
    __local float l_sk_h[SK_GRID_MAX * N_SK_COLS];
    __local float l_sk_s[SK_GRID_MAX * N_SK_COLS];

    int n_sk_elements = n_grid * N_SK_COLS;
    for (int i = tid; i < n_sk_elements; i += wg) {
        l_sk_h[i] = sk_h[i];
        l_sk_s[i] = sk_s[i];
    }

    // --- CACHE FRAGMENT METADATA INTO __local ---
    __local Fragment l_frags[128];
    for (int i = tid; i < n_frags && i < 128; i += wg) {
        l_frags[i] = fragments[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid >= n_pairs) return;

    PairEntry p = pairs[gid];

    // --- PER-FRAGMENT LOOKUP (from __local) ---
    Fragment frag = l_frags[p.replica];
    int n_orbs  = frag.n_orbs;
    int atom_off = frag.atom_off;
    int base = frag.H_base;

    float va = V_a[atom_off + p.atom_i];
    float vb = V_a[atom_off + p.atom_j];
    float h1_factor = 0.5f * (va + vb);

    // Compute B-spline index and weights once (same for all block types)
    float4 w;
    int    base_idx = cubic_interp_params(p.r, dr, n_grid, &w);

    if (block_type == 0) {
        float sk_h = interp_sk_1(l_sk_h, base_idx, w);
        float sk_s = interp_sk_1(l_sk_s, base_idx, w);
        write_symmetric_1x1(H_out, n_orbs, base, p.orb_i, p.orb_j, sk_h + h1_factor * sk_s);
        write_symmetric_1x1(S_out, n_orbs, base, p.orb_i, p.orb_j, sk_s);
    } else if (block_type == 1) {
        float2 sk_h = interp_sk_2(l_sk_h, base_idx, w);
        float2 sk_s = interp_sk_2(l_sk_s, base_idx, w);
        float4 blk;
        rotate_1x4(p.l, p.m, p.n, sk_s, &blk);
        write_symmetric_1x4(S_out, n_orbs, base, p.orb_i, p.orb_j, &blk);
        rotate_1x4(p.l, p.m, p.n, sk_h + h1_factor * sk_s, &blk);
        write_symmetric_1x4(H_out, n_orbs, base, p.orb_i, p.orb_j, &blk);
    } else {
        float4 blk[4];
        float4 sk_s = interp_sk_4(l_sk_s, base_idx, w);
        rotate_4x4(p.l, p.m, p.n, sk_s, blk);
        write_symmetric_4x4(S_out, n_orbs, base, p.orb_i, p.orb_j, blk);
        float4 sk_h = interp_sk_4(l_sk_h, base_idx, w);
        rotate_4x4(p.l, p.m, p.n, sk_h + h1_factor * sk_s, blk);
        write_symmetric_4x4(H_out, n_orbs, base, p.orb_i, p.orb_j, blk);
    }
}
