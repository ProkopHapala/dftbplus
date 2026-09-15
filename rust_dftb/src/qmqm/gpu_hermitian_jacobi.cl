// ==================================================================
// GPU Complex Hermitian Cyclic Jacobi Eigensolver (PBC / arbitrary k)
// ==================================================================
//
// Complex-valued sibling of `jacobi_cyclic_global_batched`
// (gpu_tiled_jacobi.cl) for the PBC k-point path: H(k) and S(k) are
// Hermitian float2 matrices, one workgroup per (replica, k) system.
//
// Prototype seeded from the GPT-5.6 discussion in
//   doc/prokop/tasts/HBond_Relaxed_Scan_GPU/Dense_Multi_PBC.chat.md
// (unitary 2x2 rotation J = [[c, s·u], [-s·u*, c]] with u = A_pq/|A_pq|),
// then aligned with the production real kernel's optimizations:
//   - fused prologue: V=I + ||A||_F + initial off in ONE pass + ONE
//     packed two-array reduction (warm-start calls live in this prologue)
//   - ~3 barriers per Brent-Luk round (rotations / any-rotation / update)
//   - relative pivot skipping (PAIR_SKIP_REL), capacity guard, per-sweep
//     stop record identical to the real kernel's diag[4] contract
//   - per-sweep Hermiticity restoration A <- (A + A^H)/2 — the f32
//     restoring invariant (AGENTS §3): O(N^2) once per O(N^3) sweep
//   - NO local atomics: rot_live[] flags + lane-0 scan (jpair <= 128)
//   - per-REPLICA gating: flat system index sid = rep*nk + k addresses
//     the replica mask as active[sid / nk] — a replica frozen by the
//     device-side DIIS convergence test stops ALL its k-point workgroups
//     with zero extra launches (nk=1 degenerates to per-system gating).
//
// Differences vs the real kernel that are FORCED by PBC, not optional:
//   - No in-kernel Fermi tail: mu is shared across the k-points of a
//     replica, so the mu solve lives in kpoint_occ_batched
//     (gpu_zmatrix_ops.cl) which reduces over all nk eigenvalue sets.
//   - Eigenvalues are extracted from Re(diag(A)) by
//     zextract_diagonal_batched — the per-sweep Hermitian restore keeps
//     Im(diag) ~ 0.
//
// Layout: A,V are [batch][n*n] float2 row-major; batch counts FLAT
// (replica,k) systems.
// ------------------------------------------------------------------

#ifndef WG
#define WG 512
#endif
#ifndef MAX_CSWEEPS
#define MAX_CSWEEPS 40
#endif
#ifndef JACOBI_OFF_TOL
#define JACOBI_OFF_TOL 1.0e-6f
#endif
#ifndef PAIR_SKIP_REL
#define PAIR_SKIP_REL 3.0e-8f     // |a_pq| < eps*(|a_pp|+|a_qq|) -> skip
#endif
#ifndef HJ_ROT_FP64
#define HJ_ROT_FP64 1            // 1: f64 scalar c,s construction only
#endif

#if HJ_ROT_FP64
typedef double jrot_t;           // rotation-parameter precision (scalar island)
#else
typedef float jrot_t;
#endif

// ------------------------------------------------------------------
// Minimal complex arithmetic on float2 = (re, im)
// ------------------------------------------------------------------
inline float2 cconj_f(float2 a) { return (float2)(a.x, -a.y); }
inline float2 cmul_f(float2 a, float2 b) {
    return (float2)(fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x));
}
inline float2 cscale_f(float s, float2 a) { return (float2)(s * a.x, s * a.y); }
inline float  cabs2_f(float2 a) { return fma(a.x, a.x, a.y * a.y); }

// ------------------------------------------------------------------
// Brent-Luk round-robin schedule — identical to the real kernel.
// For jn elements (jn even): jn-1 rounds, jn/2 disjoint pairs per round.
// ------------------------------------------------------------------
inline int2 herm_jacobi_pair(int round, int ipair, int jn) {
    const int m = jn - 1;
    if (ipair == 0) return (int2)(m, round % m);
    return (int2)((round + ipair) % m, (round + m - ipair) % m);
}

// ------------------------------------------------------------------
// jacobi_hermitian_cyclic_global_batched
//
// Diagonalizes `batch` Hermitian n x n float2 matrices. One workgroup
// per flat (replica,k) system. On exit A holds eigenvalues on the
// diagonal (imag ~0) and V the eigenvectors (columns).
//
//   init_v: 0 -> V=I (cold), 1 -> keep V (warm basis; V <- V·J)
//   active: PER-REPLICA mask [n_rep] — a flat system sid runs iff
//           active[sid / nk] != 0. All k-points of a frozen replica
//           early-out together.
//   diag:   [batch][4] out: {off, off/||A||_F, stop, sweeps}
//           stop: 0 converged . 1 stagnation . 2 MAX_CSWEEPS .
//                 3 n>capacity . 4 non-finite  (same contract as real)
// ------------------------------------------------------------------
__kernel void jacobi_hermitian_cyclic_global_batched(
    __global float2* A,             // [batch][n*n] in/out — Hermitian
    __global float2* V,             // [batch][n*n] in/out — eigenvectors
    const int n,
    const int batch,                // flat (replica,k) count
    const int init_v,
    const int nk,                   // k-points per replica (mask stride)
    __global const int* active,     // [n_rep] replica gate
    __global float* diag            // [batch][4] out
) {
    const int sid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid / nk] == 0) return;

    const int jn = (n & 1) ? n + 1 : n;
    const int jpair = jn / 2;
    const int jround = jn - 1;

    __local jrot_t rot_c[128];      // supports n <= 256
    __local jrot_t rot_s[128];
    __local float2 rot_u[128];      // unit phase u = a_pq/|a_pq|
    __local int    rot_p[128];
    __local int    rot_q[128];
    __local int    rot_live[128];   // replaces the real kernel's atomic_inc
    __local int    any_rotation;    // lane-0 scan result (hoisted: __local is
                                    // function-scope only in strict OpenCL)
    __local float  reduce[WG];
    __local float  reduce2[WG];     // packed second reduction (off-norm)

    // Capacity guard — same contract as the real kernel (stop=3).
    if (jpair > 128) {
        if (lid == 0) {
            diag[4*sid] = INFINITY; diag[4*sid+1] = INFINITY;
            diag[4*sid+2] = 3.0f;   diag[4*sid+3] = 0.0f;
        }
        return;
    }

    __global float2* gA = A + (size_t)sid * n * n;
    __global float2* gV = V + (size_t)sid * n * n;
    const int nn = n * n;

    // ---- fused prologue: V=I + ||A||_F + initial off in ONE pass ----
    // ||A||_F is invariant under unitary similarity -> computed once at
    // entry; the exit test does not get harder on warm-started
    // (near-diagonal) A. Same packing as the real kernel's R8b prologue.
    float fa = 0.0f, fo = 0.0f;
    for (int idx = lid; idx < nn; idx += lsz) {
        const int r = idx / n;
        const int c = idx - r * n;
        const float2 z = gA[idx];
        const float z2 = cabs2_f(z);
        fa += z2;
        if (r != c) fo += z2;
        if (init_v == 0) gV[idx] = (r == c) ? (float2)(1.0f, 0.0f)
                                            : (float2)(0.0f, 0.0f);
    }
    reduce[lid] = fa;
    reduce2[lid] = fo;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) { reduce[lid] += reduce[lid + off]; reduce2[lid] += reduce2[lid + off]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float frob = sqrt(fmax(reduce[0], 1.0e-30f));
    float off_cur = sqrt(fmax(reduce2[0], 0.0f));
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    const float off_exit = JACOBI_OFF_TOL * frob;
    float prev_off = fmax(off_cur, 1.0e-30f);
    int stall = 0;
    int stop = 0;   // 0 converged . 1 stagnation . 2 max sweeps . 4 non-finite
    int nsw  = 0;

    if (off_cur > off_exit) {
    stop = 2;
    for (int sweep = 0; sweep < MAX_CSWEEPS; ++sweep) {
        nsw = sweep + 1;
        for (int r = 0; r < jround; ++r) {
            // ---- Phase 1: one unitary rotation per disjoint pair ----
            // J = [[c, s·u], [-s·u*, c]] with u = a_pq/|a_pq|. The c,s
            // equation is the ordinary real Jacobi one with apq -> |apq|
            // (zeroing A'_pq requires tau = (a_qq - a_pp) / (2|a_pq|)).
            // f64 for the scalar c,s when HJ_ROT_FP64 — off the
            // throughput path, same split as the real kernel's prec>=1.
            for (int ip = lid; ip < jpair; ip += lsz) {
                const int2 pq = herm_jacobi_pair(r, ip, jn);
                const int p = min(pq.x, pq.y);
                const int q = max(pq.x, pq.y);
                rot_p[ip] = p; rot_q[ip] = q;

                const float2 apq = (q < n) ? gA[p * n + q] : (float2)(0.0f, 0.0f);
                // Hermitian diagonal is real (enforced by the sweep-end
                // restoration) — take .x directly.
                const float app = gA[p * n + p].x;
                const float aqq = (q < n) ? gA[q * n + q].x : 1.0f;

#if HJ_ROT_FP64
                const double ar = (double)apq.x, ai = (double)apq.y;
                const double az = sqrt(ar * ar + ai * ai);
                const double scale = fabs((double)app) + fabs((double)aqq);
                // NaN-safe !(a > b) form, same as the real kernel.
                if (!(az > (double)PAIR_SKIP_REL * scale)) {
                    rot_c[ip] = 1.0; rot_s[ip] = 0.0;
                    rot_u[ip] = (float2)(1.0f, 0.0f);
                    rot_live[ip] = 0;
                } else {
                    rot_u[ip] = (float2)((float)(ar / az), (float)(ai / az));
                    const double tau = ((double)aqq - (double)app) / (2.0 * az);
                    const double t = (tau >= 0.0)
                        ? 1.0 / (tau + sqrt(1.0 + tau * tau))
                        : -1.0 / (-tau + sqrt(1.0 + tau * tau));
                    const double c = 1.0 / sqrt(1.0 + t * t);
                    rot_c[ip] = c; rot_s[ip] = t * c;
                    rot_live[ip] = 1;
                }
#else
                const float az = sqrt(cabs2_f(apq));
                const float scale = fabs(app) + fabs(aqq);
                if (!(az > PAIR_SKIP_REL * scale)) {
                    rot_c[ip] = 1.0f; rot_s[ip] = 0.0f;
                    rot_u[ip] = (float2)(1.0f, 0.0f);
                    rot_live[ip] = 0;
                } else {
                    rot_u[ip] = apq / az;
                    const float tau = (aqq - app) / (2.0f * az);
                    const float t = (tau >= 0.0f)
                        ? 1.0f / (tau + sqrt(1.0f + tau * tau))
                        : -1.0f / (-tau + sqrt(1.0f + tau * tau));
                    const float c = rsqrt(1.0f + t * t);
                    rot_c[ip] = c; rot_s[ip] = t * c;
                    rot_live[ip] = 1;
                }
#endif
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // No atomics: lane 0 scans the <=128 live flags.
            if (lid == 0) {
                int any = 0;
                for (int ip = 0; ip < jpair; ++ip) any |= rot_live[ip];
                any_rotation = any;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (!any_rotation) continue;   // whole round below skip

            // ---- Phase 2+3 fused: A <- J^H A J and V <- V J ----
            // Pair a transforms rows (pa,qa), pair b columns (pb,qb);
            // every (a,b) owns a disjoint 2x2 block (gather in, write once).
            const int nblk = jpair * jpair;
            const int nvec = n * jpair;
            for (int it = lid; it < nblk + nvec; it += lsz) {
                if (it < nblk) {
                    const int a = it / jpair, b = it - a * jpair;
                    const float sa = (float)rot_s[a], sb = (float)rot_s[b];
                    if (sa == 0.0f && sb == 0.0f) continue;   // both identity
                    const float ca = (float)rot_c[a], cb = (float)rot_c[b];
                    const float2 ua = rot_u[a], uac = cconj_f(ua);
                    const float2 ub = rot_u[b], ubc = cconj_f(ub);
                    const int pa = rot_p[a], qa = rot_q[a];
                    const int pb = rot_p[b], qb = rot_q[b];

                    const float2 apr = gA[pa * n + pb];
                    const float2 aps = (qb < n) ? gA[pa * n + qb] : (float2)(0.0f, 0.0f);
                    const float2 aqr = (qa < n) ? gA[qa * n + pb] : (float2)(0.0f, 0.0f);
                    const float2 aqs = (qa < n && qb < n) ? gA[qa * n + qb] : (float2)(0.0f, 0.0f);

                    // Right transform [A_rp | A_rq]·J_b =
                    //   [cb·A_rp - sb·ub*·A_rq,  sb·ub·A_rp + cb·A_rq]
                    const float2 tpr = cscale_f(cb, apr) - cscale_f(sb, cmul_f(ubc, aps));
                    const float2 tps = cscale_f(sb, cmul_f(ub, apr)) + cscale_f(cb, aps);
                    const float2 tqr = cscale_f(cb, aqr) - cscale_f(sb, cmul_f(ubc, aqs));
                    const float2 tqs = cscale_f(sb, cmul_f(ub, aqr)) + cscale_f(cb, aqs);

                    // Left transform J_a^H = [[ca, -sa·ua], [sa·ua*, ca]]:
                    const float2 opr = cscale_f(ca, tpr) - cscale_f(sa, cmul_f(ua, tqr));
                    const float2 ops = cscale_f(ca, tps) - cscale_f(sa, cmul_f(ua, tqs));
                    const float2 oqr = cscale_f(sa, cmul_f(uac, tpr)) + cscale_f(ca, tqr);
                    const float2 oqs = cscale_f(sa, cmul_f(uac, tps)) + cscale_f(ca, tqs);

                    gA[pa * n + pb] = opr;
                    if (qb < n) gA[pa * n + qb] = ops;
                    if (qa < n) {
                        gA[qa * n + pb] = oqr;
                        if (qb < n) gA[qa * n + qb] = oqs;
                    }
                } else {
                    // V'_kp = c·V_kp - s·u*·V_kq ;  V'_kq = s·u·V_kp + c·V_kq
                    const int t = it - nblk;
                    const int k = t / jpair, a = t - (t / jpair) * jpair;
                    const float s = (float)rot_s[a];
                    if (s == 0.0f) continue;
                    const float c = (float)rot_c[a];
                    const float2 u = rot_u[a], uc = cconj_f(u);
                    const int p = rot_p[a], q = rot_q[a];
                    const float2 vp = gV[k * n + p];
                    const float2 vq = (q < n) ? gV[k * n + q] : (float2)(0.0f, 0.0f);
                    gV[k * n + p] = cscale_f(c, vp) - cscale_f(s, cmul_f(uc, vq));
                    if (q < n) gV[k * n + q] = cscale_f(s, cmul_f(u, vp)) + cscale_f(c, vq);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }

        // ---- f32 restoring invariant: A <- (A + A^H)/2 once per sweep ----
        // O(N^2) per O(N^3) sweep; prevents roundoff from violating the
        // Hermitian-pivot assumption the next sweep's phase 1 relies on.
        for (int idx = lid; idx < nn; idx += lsz) {
            const int i = idx / n;
            const int j = idx - i * n;
            if (i == j) {
                gA[idx] = (float2)(gA[idx].x, 0.0f);
            } else if (i < j) {
                const float2 z = 0.5f * (gA[i * n + j] + cconj_f(gA[j * n + i]));
                gA[i * n + j] = z;
                gA[j * n + i] = cconj_f(z);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        // ---- Sweep-end off-norm + exit/stagnation tests ----
        fo = 0.0f;
        for (int idx = lid; idx < nn; idx += lsz) {
            const int r = idx / n;
            if (r != idx - r * n) fo += cabs2_f(gA[idx]);
        }
        reduce[lid] = fo;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int off = lsz >> 1; off > 0; off >>= 1) {
            if (lid < off) reduce[lid] += reduce[lid + off];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        off_cur = sqrt(reduce[0]);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (!isfinite(off_cur)) { stop = 4; break; }   // report, don't loop on NaN
        if (off_cur <= off_exit) { stop = 0; break; }
        if (off_cur > 0.9f * prev_off) { if (++stall >= 3) { stop = 1; break; } } else { stall = 0; }
        prev_off = off_cur;
    }
    }

    // ---- stop-reason + achieved-residual diagnostics (real-kernel contract) ----
    if (lid == 0) {
        diag[4*sid]   = off_cur;
        diag[4*sid+1] = off_cur / frob;
        diag[4*sid+2] = (float)stop;
        diag[4*sid+3] = (float)nsw;
    }
    // Eigenvalues on the diagonal — off-diagonal residue is left in place
    // as the diagnostic record (same policy as the real kernel).
}
