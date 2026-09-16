// ==================================================================
// gpu_block_jacobi.cl — one-WG block Jacobi eigensolver (manifest §16.D)
// ==================================================================
//
// `block_jacobi_1wg` — batched symmetric eigensolver for ALL n ≤ 256.
// New architecture, built ALONGSIDE jacobi_cyclic_global_batched (the old
// direct kernel stays the default; this is selected by
// RUST_DFTB_EIGSOLVER=block).
//
// Structure — different from the old kernel in every way that matters:
//   * WG = ceil(n/32)*32 (one thread per row; n=86 → 96, n=246 → 256),
//     not a fixed 512.
//   * A and V stay in GLOBAL memory; local memory holds ONLY the 2B×2B
//     compound pivot P, its eigenvector accumulator U, rotation tables
//     and the reduction buffer — ~8.4 KB at B=16, not ~13.3 KB, and no
//     dead Fermi scratch (occupation is a separate kernel).
//   * Block Jacobi: serial loop over disjoint block PAIRS (bp,bq). Each
//     pair gathers its 2B×2B pivot into local, diagonalizes it with a
//     Brent-Luk cyclic Jacobi in local memory (all in one WG — no
//     inter-WG sync needed, no global atomics), then every thread
//     transforms its own row strip: A[r][piv] <- A[r][piv]·U with the
//     symmetric mirror write A[piv][r], and V[r][piv] <- V[r][piv]·U.
//     Each output element has exactly one owner, written once.
//   * Warm fast path = the probe: one row-wise off-norm pass (one N²
//     global read); if already within tolerance the kernel returns
//     without touching V or sweeping — the common SCC iteration.
//   * n ≤ PB: the whole matrix IS the pivot — a single local solve.
//     Same kernel serves N≤64 systems; smaller n → smaller WG → more
//     resident WGs per CU.
//
// Eigenvalues: on A's diagonal at exit AND written to `eig` (replaces
// the extract_diag launch; feeds fermi_occ_batched/select_occ/readback).
//
// Specialization defines (rendered by render_block_source):
//   B              block size (default 16; pivot is 2B×2B)
//   PB             2B (pivot dimension, always padded to this)
//   PLD            PB+1 (bank-conflict padding)
//   WG             workgroup size = ceil(n/32)*32
//   MAX_SWEEPS     global block-sweep cap
//   INNER_MAX      pivot-solve sweep cap
//   INNER_TOL      pivot off-norm exit (relative to pivot Frobenius)
//   JACOBI_OFF_TOL global off/‖A‖_F exit threshold
//   PAIR_SKIP_REL  relative pivot-element skip (identity rotation below)
// ------------------------------------------------------------------
#ifndef B
#define B 16
#endif
#ifndef PB
#define PB 32
#endif
#ifndef PLD
#define PLD 33
#endif
#ifndef WG
#define WG 96
#endif
#ifndef MAX_SWEEPS
#define MAX_SWEEPS 40
#endif
#ifndef INNER_MAX
#define INNER_MAX 12
#endif
#ifndef INNER_TOL
#define INNER_TOL 1.0e-7f
#endif
#ifndef JACOBI_OFF_TOL
#define JACOBI_OFF_TOL 1.0e-6f
#endif
#ifndef PAIR_SKIP_REL
#define PAIR_SKIP_REL 1.0e-12f
#endif

// Brent-Luk round-robin pair schedule on the padded pivot (PB even,
// PB-1 rotating elements — same formula as jacobi_pair with JN=PB).
inline int2 bj_pair(int round, int ip) {
    const int m = PB - 1;
    if (ip == 0) return (int2)(m, round);
    return (int2)((round + ip) % m, (round + m - ip) % m);
}

// Workgroup sum for ARBITRARY lsz (WG = ceil(n/32)*32 → 96/160/224 are
// not powers of two). The naive `o = lsz>>1` halving requires lsz to be
// a power of two — at lsz=96 it only reduces lanes 0..63 and silently
// drops every lane ≡2 mod 3. Fold the tail [p2,lsz) onto [0,lsz−p2)
// first (p2 = largest power of two ≤ lsz), then halve over p2 entries.
// All work-items must call this (workgroup barriers inside); the result
// is read after the trailing barrier so `red` is reusable by the next call.
inline float bj_wg_sum(__local float* red, float v, const int lid, const int lsz) {
    red[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    int p2 = 1;
    while ((p2 << 1) <= lsz) p2 <<= 1;
    if (lid + p2 < lsz) red[lid] += red[lid + p2];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = p2 >> 1; o > 0; o >>= 1) {
        if (lid < o) red[lid] += red[lid + o];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float s = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return s;
}

// Pivot-local Brent-Luk Jacobi: diagonalize P (PB×PB, PLD-padded,
// pad rows have huge diagonal + zero off-diagonal → identity rotations),
// accumulating eigenvectors in U. All work-items must call this
// (contains workgroup barriers). f32 throughout — rotation-angle error
// ~1e-7 is self-correcting; accuracy is governed by the global
// off-norm exit + the occ Rayleigh/renorm repair downstream.
// Returns the number of inner sweeps executed (for honest diagnostics).
inline int bj_pivot_solve(
    __local float* P, __local float* U,
    __local float* rot_c, __local float* rot_s,
    __local int* rot_p, __local int* rot_q,
    __local float* reduce, __local int* l_nrot,
    const int m,                            // real pivot rows (rest = pads)
    const int lid, const int lsz)
{
    const int jp = PB / 2;                  // 16 pairs per round
    // Reference norm over the REAL pivot only — pad rows carry a 1e30
    // diagonal that would make off_exit ~1e23 and exit after one sweep.
    float fp = 0.0f;
    for (int i = lid; i < m; i += lsz)
        for (int j = 0; j < m; ++j)
            fp += P[i * PLD + j] * P[i * PLD + j];
    const float off_exit = INNER_TOL * sqrt(fmax(bj_wg_sum(reduce, fp, lid, lsz), 1.0e-30f));

    int nsw_in = 0;
    for (int isw = 0; isw < INNER_MAX; ++isw) {
        for (int r = 0; r < PB - 1; ++r) {
            if (lid == 0) *l_nrot = 0;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int ip = lid; ip < jp; ip += lsz) {
                int2 pq = bj_pair(r, ip);
                int p = min(pq.x, pq.y), q = max(pq.x, pq.y);
                rot_p[ip] = p; rot_q[ip] = q;
                float apq = P[p * PLD + q];
                float app = P[p * PLD + p], aqq = P[q * PLD + q];
                if (!(fabs(apq) > PAIR_SKIP_REL * (fabs(app) + fabs(aqq)))) {
                    rot_c[ip] = 1.0f; rot_s[ip] = 0.0f;
                } else {
                    float tau = (aqq - app) / (2.0f * apq);
                    float t = (tau >= 0.0f)
                        ? 1.0f / (tau + sqrt(1.0f + tau * tau))
                        : -1.0f / (-tau + sqrt(1.0f + tau * tau));
                    float c = 1.0f / sqrt(1.0f + t * t);
                    float s = t * c;
                    float err = fma(-s, s, fma(-c, c, 1.0f));   // renorm (c,s)
                    rot_c[ip] = fma(0.5f * c, err, c);
                    rot_s[ip] = fma(0.5f * s, err, s);
                    atomic_inc(l_nrot);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (*l_nrot == 0) continue;
            // Fused update: P 2×2 quads + U column pairs, one barrier.
            for (int it = lid; it < jp * jp + PB * jp; it += lsz) {
                if (it < jp * jp) {
                    int a = it / jp, b = it - a * jp;
                    float sa = rot_s[a], sb = rot_s[b];
                    if (sa == 0.0f && sb == 0.0f) continue;
                    int pa = rot_p[a], qa = rot_q[a];
                    int pb = rot_p[b], qb = rot_q[b];
                    float ca = rot_c[a], cb = rot_c[b];
                    float apr = P[pa * PLD + pb], aps = P[pa * PLD + qb];
                    float aqr = P[qa * PLD + pb], aqs = P[qa * PLD + qb];
                    float tpr = fma(cb, apr, -sb * aps);
                    float tps = fma(sb, apr, cb * aps);
                    float tqr = fma(cb, aqr, -sb * aqs);
                    float tqs = fma(sb, aqr, cb * aqs);
                    P[pa * PLD + pb] = fma(ca, tpr, -sa * tqr);
                    P[pa * PLD + qb] = fma(ca, tps, -sa * tqs);
                    P[qa * PLD + pb] = fma(sa, tpr, ca * tqr);
                    P[qa * PLD + qb] = fma(sa, tps, ca * tqs);
                } else {
                    int t = it - jp * jp;
                    int k = t / jp, a = t - k * jp;
                    float s = rot_s[a];
                    if (s == 0.0f) continue;
                    float c = rot_c[a];
                    int p = rot_p[a], q = rot_q[a];
                    float vkp = U[k * PLD + p], vkq = U[k * PLD + q];
                    U[k * PLD + p] = fma(c, vkp, -s * vkq);
                    U[k * PLD + q] = fma(s, vkp, c * vkq);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        // Inner off-norm → early exit (real pivot rows only).
        float fo = 0.0f;
        for (int i = lid; i < m; i += lsz)
            for (int j = 0; j < m; ++j)
                if (i != j) fo += P[i * PLD + j] * P[i * PLD + j];
        const float off_in = sqrt(bj_wg_sum(reduce, fo, lid, lsz));
        nsw_in = isw + 1;
        if (!isfinite(off_in) || off_in <= off_exit) break;
    }
    return nsw_in;
}

__kernel void block_jacobi_1wg(
    __global float* A,            // [batch][n*n] symmetric, in/out
    __global float* V,            // [batch][n*n] eigenvector accumulator
    const int n,
    const int batch,
    const int init_v,             // 0 → V=I first; 1 → rotate V in place (warm)
    __global const int* active,   // [batch] 0 → replica parked, early-out
    __global float* diag,         // [batch][4] {off, off/‖A‖_F, stop, sweeps}
    __global float* eig           // [batch][n] eigenvalues out
) {
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (gid >= batch || active[gid] == 0) return;

    __local float P[PLD * PB];
    __local float U[PLD * PB];
    __local float rot_c[PB / 2];
    __local float rot_s[PB / 2];
    __local int   rot_p[PB / 2];
    __local int   rot_q[PB / 2];
    __local float reduce[WG];
    __local int   l_nrot;

    __global float* gA = A + (size_t)gid * n * n;
    __global float* gV = V + (size_t)gid * n * n;

    if (init_v == 0 && lid < n) {
        for (int j = 0; j < n; ++j) gV[lid * n + j] = (j == lid) ? 1.0f : 0.0f;
    }

    // ---- Probe: row-wise Frobenius + off-diagonal norms ----
    // One thread per row (WG ≥ n by construction). The common warm
    // iteration exits right here — one N² global read, nothing else.
    float fro = 0.0f, fo = 0.0f;
    if (lid < n) {
        for (int j = 0; j < n; ++j) {
            float v = gA[lid * n + j];
            fro += v * v;
            if (j != lid) fo += v * v;
        }
    }
    const float frob = sqrt(fmax(bj_wg_sum(reduce, fro, lid, lsz), 1.0e-30f));
    float off_cur = sqrt(bj_wg_sum(reduce, fo, lid, lsz));
    const float off_exit = JACOBI_OFF_TOL * frob;

    int stop = -1, nsw = 0;
    if (!isfinite(off_cur)) stop = 4;
    else if (off_cur <= fmax(off_exit, 1.0e-30f)) stop = 0;   // probe exit
    float prev_off = fmax(off_cur, 1.0e-30f);
    int stall = 0;

    if (stop < 0 && n <= PB) {
        // ---- Single-block path: the whole matrix is the pivot ----
        for (int i = lid; i < PB; i += lsz) {
            for (int j = 0; j < PB; ++j) {
                P[i * PLD + j] = (i < n && j < n) ? gA[i * n + j] : ((i == j) ? 1.0e30f : 0.0f);
                U[i * PLD + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nsw = bj_pivot_solve(P, U, rot_c, rot_s, rot_p, rot_q, reduce, &l_nrot, n, lid, lsz);
        for (int i = lid; i < n; i += lsz) {
            // A row i <- P row i; V row i <- V row i · U (init_v: I·U=U, warm: c·U)
            float xv[PB];
            #pragma unroll
            for (int j = 0; j < PB; ++j) xv[j] = (j < n) ? gV[i * n + j] : 0.0f;
            for (int j = 0; j < n; ++j) {
                gA[i * n + j] = P[i * PLD + j];
                float acc = 0.0f;
                #pragma unroll
                for (int k = 0; k < PB; ++k) acc = fma(xv[k], U[k * PLD + j], acc);
                gV[i * n + j] = acc;
            }
        }
        // Report the achieved inner off-norm against ‖A‖_F — and certify
        // honestly: a finite residual ABOVE the global exit tolerance is a
        // non-converged solve (stop=2, inner cap exhausted), not success.
        float fo2 = 0.0f;
        for (int i = lid; i < PB; i += lsz)
            for (int j = 0; j < PB; ++j)
                if (i != j && i < n && j < n) fo2 += P[i * PLD + j] * P[i * PLD + j];
        off_cur = sqrt(bj_wg_sum(reduce, fo2, lid, lsz));
        stop = !isfinite(off_cur) ? 4 : ((off_cur <= fmax(off_exit, 1.0e-30f)) ? 0 : 2);
    } else if (stop < 0) {
        // ---- Block-pair sweeps ----
        const int nb = (n + B - 1) / B;
        for (int sweep = 0; sweep < MAX_SWEEPS && stop < 0; ++sweep) {
            for (int bp = 0; bp < nb; ++bp) {
                for (int bq = bp + 1; bq < nb; ++bq) {
                    const int p0 = bp * B, q0 = bq * B;
                    const int mp = min(B, n - p0), mq = min(B, n - q0);
                    const int m = mp + mq;
                    // pidx(j) = j<mp ? p0+j : q0+(j-mp) — two contiguous ranges.
                    // Gather pivot into P (pad rows/cols m..PB: big diagonal).
                    for (int i = lid; i < PB; i += lsz) {
                        const int ri = (i < mp) ? p0 + i : q0 + (i - mp);
                        for (int j = 0; j < PB; ++j) {
                            const int cj = (j < mp) ? p0 + j : q0 + (j - mp);
                            P[i * PLD + j] = (i < m && j < m) ? gA[ri * n + cj] : ((i == j) ? 1.0e30f : 0.0f);
                            U[i * PLD + j] = (i == j) ? 1.0f : 0.0f;
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    bj_pivot_solve(P, U, rot_c, rot_s, rot_p, rot_q, reduce, &l_nrot, m, lid, lsz);
                    // ---- Writeback: thread r owns row r ----
                    if (lid < n) {
                        const int r = lid;
                        int il = -1;                                  // pivot-local row, -1 = external
                        if (r >= p0 && r < p0 + mp) il = r - p0;
                        else if (r >= q0 && r < q0 + mq) il = mp + (r - q0);
                        float x[PB];
                        #pragma unroll
                        for (int j = 0; j < PB; ++j) {
                            const int cj = (j < mp) ? p0 + j : q0 + (j - mp);
                            x[j] = (j < m) ? gA[r * n + cj] : 0.0f;
                        }
                        if (il >= 0) {
                            #pragma unroll
                            for (int j = 0; j < PB; ++j) {
                                if (j < m) {
                                    const int cj = (j < mp) ? p0 + j : q0 + (j - mp);
                                    gA[r * n + cj] = P[il * PLD + j];
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int j = 0; j < PB; ++j) {
                                if (j < m) {
                                    float acc = 0.0f;
                                    #pragma unroll
                                    for (int i = 0; i < PB; ++i) acc = fma(x[i], U[i * PLD + j], acc);
                                    const int cj = (j < mp) ? p0 + j : q0 + (j - mp);
                                    gA[r * n + cj] = acc;            // row r, pivot col
                                    gA[cj * n + r] = acc;            // symmetric mirror
                                }
                            }
                        }
                        // V: every row (incl. pivot rows) transforms.
                        float xv[PB];
                        #pragma unroll
                        for (int j = 0; j < PB; ++j) {
                            const int cj = (j < mp) ? p0 + j : q0 + (j - mp);
                            xv[j] = (j < m) ? gV[r * n + cj] : 0.0f;
                        }
                        #pragma unroll
                        for (int j = 0; j < PB; ++j) {
                            if (j < m) {
                                float acc = 0.0f;
                                #pragma unroll
                                for (int i = 0; i < PB; ++i) acc = fma(xv[i], U[i * PLD + j], acc);
                                const int cj = (j < mp) ? p0 + j : q0 + (j - mp);
                                gV[r * n + cj] = acc;
                            }
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
                }
            }
            // ---- Sweep-end off-norm + exit tests (same policy as direct) ----
            fo = 0.0f;
            if (lid < n)
                for (int j = 0; j < n; ++j)
                    if (j != lid) { float v = gA[lid * n + j]; fo += v * v; }
            off_cur = sqrt(bj_wg_sum(reduce, fo, lid, lsz));
            nsw = sweep + 1;
            if (!isfinite(off_cur)) { stop = 4; }
            else if (off_cur <= off_exit) { stop = 0; }
            else if (off_cur > 0.9f * prev_off) { if (++stall >= 3) stop = 1; }
            else stall = 0;
            prev_off = off_cur;
        }
        if (stop < 0) stop = 2;   // MAX_SWEEPS hit
    }

    // ---- Eigenvalues + diagnostics ----
    // First-failure latch (T01): the host clears stop to 0 at the start of
    // each solve window; a recorded stop>0 is never overwritten by a later
    // launch, so a mid-iteration failure can no longer be silently erased
    // by a subsequent converged one before check_jacobi() reads it.
    if (lid < n) eig[(size_t)gid * n + lid] = gA[lid * n + lid];
    if (lid == 0 && diag[4 * gid + 2] <= 0.0f) {
        diag[4 * gid]     = off_cur;
        diag[4 * gid + 1] = off_cur / frob;
        diag[4 * gid + 2] = (float)stop;
        diag[4 * gid + 3] = (float)nsw;
    }
}
