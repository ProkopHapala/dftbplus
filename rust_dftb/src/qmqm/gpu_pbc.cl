//! OpenCL kernels for the periodic (complex k-point) DFTB+ path.
//!
//! Built as one program concatenated after `dftb_hamiltonian.cl` (uses
//! its cubic_interp_params / interp_sk_* / rotate_* / exp_gamma_same_u /
//! gamma_sub_exprn helpers).
//!
//! Pipeline per geometry (all assembly-phase, never in the SCC loop):
//!   1. `assemble_pairs_img`  — per (replica, image-slot): SK spline eval at
//!      displacement d = r_oj + R − r_oi, writes pure-H0/S blocks to
//!      per-slot buffers (k-independent — reused every iteration).
//!   2. `kpoint_phase_sum_batched` — per (replica, out-pair, k, elem):
//!      acc = Σ_slots e^{ikR}·blk → H(k)[row,col]; optional conj-transpose
//!      `herm` write; `diag` adds onsite+I. Zero-fill H/S first.
//!   3. `ewald_invr_batched` + `gamma_pbc_batched` — periodic γ:
//!      γ_pbc = Ewald(1/R) − Σ_short expGamma  (see ewald_notes.md).
//!
//! Conventions (match Fortran, ewald_notes.md):
//!   - Slot R is the cell shift of the IMAGE atom; eval displacement
//!     d = r_oj + R − r_oi; fold phase = e^{+ik·R}, k and R Cartesian.
//!   - out-pair (row=oj, col=oi) → H(k)[row,col] += e^{ikR}·blk.
//!     herm=1 also writes H(k)[col,row] += e^{−ikR}·blk^T.
//!   - invRMat: Σ_R erfc(αr)/r (real, per-pair CSR list) +
//!     (8π/V)Σ_{G>0} w_g·cos(G·r) (half-space, w_g precomputed host-side)
//!     − π/(Vα²) on all elements − (2α/√π) on the diagonal.
//!   - expGamma = 1/r − γ_func (the SHORT-RANGE difference), onsite −U.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define MIN_HUB_DIFF 3.125e-6f   // Fortran MinHubDiff = 0.3125e-5

// ------------------------------------------------------------------
// Types
// ------------------------------------------------------------------

/// One (oriented-atom-pair, image-cell) SK evaluation slot. 8B.
/// The eval pair is (oi, oj) with the image shift on the oj side:
/// d = coords[oj] + rcell[cell] − coords[oi].
typedef struct {
    ushort oi;      // oriented atom i (SK table order; s first for s-p)
    ushort oj;      // oriented atom j — the image atom
    int    cell;    // index into rcell (Cartesian R of the oj image)
} ImgSlot;

/// One fold output pair: block H(k)[row_orb.., col_orb..] of size
/// nrow×ncol, filled by slots [slot_off, slot_off+slot_cnt). 16B.
typedef struct {
    ushort row_orb;   // orbital offset of row atom (oj side)
    ushort col_orb;   // orbital offset of col atom (oi side)
    uchar  nrow, ncol;
    uchar  diag;      // 1 → diagonal pair: add onsite(H) + I(S)
    uchar  herm;      // 1 → also write conj-transpose at [col,row]
    int    slot_off;  // into this bucket's slot array (rep-invariant)
    int    slot_cnt;
    int    pad;
} FoldPair;

// ------------------------------------------------------------------
// expGamma = 1/r − γ_func (short-range difference), f32
// r<tol → onsite limit −Ū (note the sign: γ_func(0)−1/r → −U)
// ------------------------------------------------------------------
inline float expgamma_diff(float r, float ua, float ub) {
    if (r < 1e-5f) {
        if (fabs(ua - ub) < MIN_HUB_DIFF) return -0.5f * (ua + ub);
        const float ta = TAU_FACTOR * ua, tb = TAU_FACTOR * ub;
        const float tt = ta * tb, s = ta + tb;
        return -0.5f * (tt / s + tt * tt / (s * s * s));
    }
    if (fabs(ua - ub) < MIN_HUB_DIFF) {
        return exp_gamma_same_u(r, 0.5f * TAU_FACTOR * (ua + ub));
    }
    const float ta = TAU_FACTOR * ua, tb = TAU_FACTOR * ub;
    return gamma_sub_exprn(r, ta, tb) + gamma_sub_exprn(r, tb, ta);
}

// ------------------------------------------------------------------
// Ewald 1/R matrix — one work-item per (replica, lower-triangle pair)
// ------------------------------------------------------------------
// epair CSR: pairs (a,c) with a ≥ c; slots = Cartesian R s.t.
// |r_c + R − r_a| < maxREwald (for a==c the R=0 slot is NOT in the list
// and the r≈0 skip below is a second guard).
__kernel void ewald_invr_batched(
    const int n_atoms, const int n_pairs, const int n_rep,
    __global const float*  coords,     // [n_rep·n_at·3] Bohr
    __global const int2*   epair,      // [n_pairs] (a,c)
    __global const int*    eoff,       // [n_pairs+1]
    __global const float4* eslot,      // [nslot] (Rx,Ry,Rz,unused)
    __global const float4* gvec,       // [ng] (gx,gy,gz, w=e^{−g²/4α²}/g²)
    const int ng,
    const double alpha,
    const double rec_fac,              // 8π/V
    const double c_const,              // −π/(Vα²)
    const double c_self,               // −2α/√π
    __global float* invr,              // [n_rep·n_at²]
    __global const int*  park          // [n_rep]
) {
    const int gid = get_global_id(0);
    const int rep = gid / n_pairs;
    const int p = gid - rep * n_pairs;
    if (rep >= n_rep || p >= n_pairs) return;
    if (park[rep] == 0) return;
    const int a = epair[p].x;
    const int c = epair[p].y;
    __global const float* crd = coords + (size_t)rep * n_atoms * 3;
    const double dx = (double)crd[3 * a]     - crd[3 * c];
    const double dy = (double)crd[3 * a + 1] - crd[3 * c + 1];
    const double dz = (double)crd[3 * a + 2] - crd[3 * c + 2];

    // real part: Σ_R erfc(α·|d − R|)/|d − R|
    double s = 0.0;
    for (int t = eoff[p]; t < eoff[p + 1]; t++) {
        const float4 R = eslot[t];
        const double rx = dx - (double)R.x;
        const double ry = dy - (double)R.y;
        const double rz = dz - (double)R.z;
        const double r2 = rx * rx + ry * ry + rz * rz;
        if (r2 < 1e-10) continue;
        const double r = sqrt(r2);
        s += erfc(alpha * r) / r;
    }
    // reciprocal part: Σ_G w_g·cos(G·r)  (w_g, G precomputed on host)
    double rec = 0.0;
    for (int g = 0; g < ng; g++) {
        const float4 gv = gvec[g];
        rec += (double)gv.w * cos((double)gv.x * dx + (double)gv.y * dy + (double)gv.z * dz);
    }
    double v = s + rec_fac * rec + c_const;
    if (a == c) v += c_self;
    __global float* M = invr + (size_t)rep * n_atoms * n_atoms;
    M[a * n_atoms + c] = (float)v;
    M[c * n_atoms + a] = (float)v;
}

// ------------------------------------------------------------------
// Periodic γ: γ[a,c] = invr[a,c] − Σ_slots expGamma(|r_c+R−r_a|, U_a, U_c)
// ------------------------------------------------------------------
// Same CSR shape as ewald; the (a==c, R=0) slot IS present (onsite →
// expgamma(0) = −Ū → γ += Ū). Cutoff = global short-γ cutoff.
__kernel void gamma_pbc_batched(
    const int n_atoms, const int n_pairs, const int n_rep,
    __global const float*  coords,     // [n_rep·n_at·3]
    __global const int2*   gpair,
    __global const int*    goff,
    __global const float4* gslot,
    __global const int*    species,    // [n_rep·n_at] global species idx
    __global const float*  u_hub,      // [nsp]
    __global const float*  invr,       // [n_rep·n_at²] read
    __global float*        G,          // [n_rep·n_at²] write
    __global const int*    park)
{
    const int gid = get_global_id(0);
    const int rep = gid / n_pairs;
    const int p = gid - rep * n_pairs;
    if (rep >= n_rep || p >= n_pairs) return;
    if (park[rep] == 0) return;
    const int a = gpair[p].x;
    const int c = gpair[p].y;
    __global const float* crd = coords + (size_t)rep * n_atoms * 3;
    const float ua = u_hub[species[rep * n_atoms + a]];
    const float uc = u_hub[species[rep * n_atoms + c]];
    const float ax = crd[3 * a], ay = crd[3 * a + 1], az = crd[3 * a + 2];
    const float cx = crd[3 * c], cy = crd[3 * c + 1], cz = crd[3 * c + 2];
    const size_t base = (size_t)rep * n_atoms * n_atoms;
    float g = invr[base + a * n_atoms + c];
    for (int t = goff[p]; t < goff[p + 1]; t++) {
        const float4 R = gslot[t];
        const float rx = cx + R.x - ax;
        const float ry = cy + R.y - ay;
        const float rz = cz + R.z - az;
        const float r = sqrt(rx * rx + ry * ry + rz * rz);
        g -= expgamma_diff(r, ua, uc);
    }
    G[base + a * n_atoms + c] = g;
    G[base + c * n_atoms + a] = g;
}

// ------------------------------------------------------------------
// Image-pair SK assembly — one work-item per (replica, slot)
// ------------------------------------------------------------------
// Writes the pure-H0/S block (norb_oj × norb_oi, row-major) for slot s
// into h_blk/s_blk at [ (rep·n_slots + s)·bsz ]. The k-dependence lives
// entirely in the fold kernel below — this runs once per geometry.
__kernel void assemble_pairs_img(
    __global const ImgSlot* slots,
    __global const float4*  rcell,     // Cartesian cell shifts
    __global const float*   coords,    // [n_rep·n_at·3] Bohr
    const int n_atoms, const int n_rep,
    __global const float*   sk_h,      // compact SK table [n_grid·n_sk_cols]
    __global const float*   sk_s,
    const float dr, const int n_grid,
    const int n_slots,                 // slots per replica (rep-invariant)
    const int block_type,
    const int n_sk_cols,
    __global float* h_blk,
    __global float* s_blk,
    __global const int* park)
{
    const int tid = get_local_id(0);
    const int wg  = get_local_size(0);
    const int gid = get_global_id(0);

    __local float l_sk_h[SK_GRID_MAX * N_SK_COLS];
    __local float l_sk_s[SK_GRID_MAX * N_SK_COLS];
    const int n_sk_elements = n_grid * n_sk_cols;
    for (int i = tid; i < n_sk_elements; i += wg) {
        l_sk_h[i] = sk_h[i];
        l_sk_s[i] = sk_s[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int rep = gid / n_slots;
    const int s   = gid - rep * n_slots;
    if (rep >= n_rep || s >= n_slots) return;
    if (park[rep] == 0) return;

    const ImgSlot sl = slots[s];
    const float4 R = rcell[sl.cell];
    __global const float* crd = coords + (size_t)rep * n_atoms * 3;
    const float dx = crd[3 * sl.oj]     + R.x - crd[3 * sl.oi];
    const float dy = crd[3 * sl.oj + 1] + R.y - crd[3 * sl.oi + 1];
    const float dz = crd[3 * sl.oj + 2] + R.z - crd[3 * sl.oi + 2];
    const float r = sqrt(dx * dx + dy * dy + dz * dz);
    const float inv_r = 1.0f / fmax(r, 1e-12f);
    const float l = dx * inv_r, m = dy * inv_r, n = dz * inv_r;

    const int bsz = (block_type == 0) ? 1 : (block_type == 1 ? 4 : 16);
    __global float* hb = h_blk + (size_t)gid * bsz;
    __global float* sb = s_blk + (size_t)gid * bsz;

    // Out-of-table or degenerate slot → zero block (same guard as
    // assemble_pair_body — never extrapolate the last stencil).
    const float r_tab = ((float)n_grid - 1.0f) * dr;
    if (r >= r_tab || r < 1e-6f) {
        for (int e = 0; e < bsz; e++) { hb[e] = 0.0f; sb[e] = 0.0f; }
        return;
    }

    float4 w;
    const int base_idx = cubic_interp_params(r, dr, n_grid, &w);

    if (block_type == 0) {
        hb[0] = interp_sk_1(l_sk_h, base_idx, w);
        sb[0] = interp_sk_1(l_sk_s, base_idx, w);
    } else if (block_type == 1) {
        const float2 skh = interp_sk_2(l_sk_h, base_idx, w);
        const float2 sks = interp_sk_2(l_sk_s, base_idx, w);
        float4 blk;
        rotate_1x4(l, m, n, sks, &blk);
        sb[0] = blk.x; sb[1] = blk.y; sb[2] = blk.z; sb[3] = blk.w;
        rotate_1x4(l, m, n, skh, &blk);
        hb[0] = blk.x; hb[1] = blk.y; hb[2] = blk.z; hb[3] = blk.w;
    } else {
        float4 blk[4];
        float ps;
        float4 sks = interp_sk_5(l_sk_s, base_idx, w, &ps);
        rotate_4x4(l, m, n, sks, ps, blk);
        for (int a = 0; a < 4; a++) {
            sb[4 * a]     = blk[a].x;
            sb[4 * a + 1] = blk[a].y;
            sb[4 * a + 2] = blk[a].z;
            sb[4 * a + 3] = blk[a].w;
        }
        float4 skh = interp_sk_5(l_sk_h, base_idx, w, &ps);
        rotate_4x4(l, m, n, skh, ps, blk);
        for (int a = 0; a < 4; a++) {
            hb[4 * a]     = blk[a].x;
            hb[4 * a + 1] = blk[a].y;
            hb[4 * a + 2] = blk[a].z;
            hb[4 * a + 3] = blk[a].w;
        }
    }
}

// ------------------------------------------------------------------
// Bloch fold: H(k),S(k) = Σ_slots e^{ikR}·blk   (+ onsite/I on diag)
// ------------------------------------------------------------------
// One work-item per (replica, out-pair, k, element). Gather inputs,
// write owned outputs — no atomics. H/S buffers must be pre-zeroed
// (out-pairs with no slots contribute zero and are never visited).
__kernel void kpoint_phase_sum_batched(
    __global const FoldPair* outs,   // [n_outs] this bucket's out-pairs
    __global const ImgSlot* slots,
    __global const float4*   rcell,
    __global const float4*   kcart,    // [nk] Cartesian k
    const int nk, const int n_orb, const int n_rep, const int n_outs,
    const int n_slots,               // bucket slots per replica
    const int bsz,                   // nrow·ncol (uniform in bucket)
    __global const float*  h_blk,
    __global const float*  s_blk,
    __global const float*  onsite,   // [n_orb] per-orbital onsite (rep-uniform)
    __global float2* H,              // [rep·nk·n_orb²]
    __global float2* S,
    __global const int*  park)
{
    const int gid = get_global_id(0);
    const int elem = gid % bsz;
    int rest = gid / bsz;
    const int k = rest % nk;   rest /= nk;
    const int o = rest % n_outs;
    const int rep = rest / n_outs;
    if (rep >= n_rep || o >= n_outs) return;
    if (park[rep] == 0) return;

    const FoldPair op = outs[o];
    const int row = elem / op.ncol;
    const int col = elem - row * op.ncol;

    const float kx = kcart[k].x, ky = kcart[k].y, kz = kcart[k].z;
    float2 ah = (float2)(0.0f, 0.0f);
    float2 as = (float2)(0.0f, 0.0f);
    const size_t blk0 = (size_t)rep * n_slots * bsz + elem;
    for (int t = op.slot_off; t < op.slot_off + op.slot_cnt; t++) {
        const float4 R = rcell[slots[t].cell];
        const float ph = kx * R.x + ky * R.y + kz * R.z;
        const float cp = cos(ph), sp = sin(ph);
        const float hv = h_blk[blk0 + (size_t)t * bsz];
        const float sv = s_blk[blk0 + (size_t)t * bsz];
        ah.x += cp * hv;  ah.y += sp * hv;
        as.x += cp * sv;  as.y += sp * sv;
    }
    if (op.diag && row == col) {
        ah.x += onsite[op.col_orb + col];
        as.x += 1.0f;
    }
    const size_t mb = ((size_t)rep * nk + k) * n_orb * n_orb;
    const size_t ij = mb + (op.row_orb + row) * n_orb + op.col_orb + col;
    H[ij] = ah;
    S[ij] = as;
    if (op.herm) {
        const size_t ji = mb + (op.col_orb + col) * n_orb + op.row_orb + row;
        H[ji] = (float2)(ah.x, -ah.y);
        S[ji] = (float2)(as.x, -as.y);
    }
}
