// ==================================================================
// GPU Eigensolver: Brent-Luk Parallel Cyclic Jacobi + S^{-1/2}
// ==================================================================
//
// Agent_4 (Wave 2) owned file. Provides:
//
//   jacobi_cyclic_local_batched  — batched symmetric eigensolver (N<=64)
//   build_inv_sqrt_from_eig      — X = U · diag(rsqrt(λ)) · U^T
//
// Architecture: host-orchestrated, device-resident SCC. One workgroup
// per system. Full-local in __local memory. N/2 independent rotations
// per round (Brent-Luk parallel cyclic ordering), one barrier per round.
//
// Matrix layout: row-major, N×N per batch element, contiguous in global
// memory: element (i,j) of batch b is at index b*N*N + i*N + j.
//
// Specialization constants (set by Rust harness via text substitution):
//   JN         — working dimension (N if even, N+1 if odd; always even)
//   JLD        — leading dimension = JN + 1 (avoids power-of-2 bank conflicts)
//   JPAIR      — pairs per round = JN / 2
//   JROUND     — rounds per sweep = JN - 1
//   WG         — workgroup size
//   PPG        — work-items per pair = WG / JPAIR
//   MAX_SWEEPS — maximum Jacobi sweeps before forced exit
//   JACOBI_TOL — convergence tolerance (relative off-diagonal norm)
//   LAMBDA_FLOOR — floor for rsqrt(λ) in S^{-1/2} (1e-7f)
//
// ------------------------------------------------------------------

#ifndef JACOBI_BLOCK_UPDATE
#define JACOBI_BLOCK_UPDATE 1
#endif
#ifndef JACOBI_NORMALIZE_ROTATION
#define JACOBI_NORMALIZE_ROTATION 1
#endif
#ifndef JN
#define JN 8
#endif
#ifndef JLD
#define JLD 9
#endif
#ifndef JPAIR
#define JPAIR 4
#endif
#ifndef JROUND
#define JROUND 7
#endif
#ifndef WG
#define WG 32
#endif
#ifndef PPG
#define PPG 8
#endif
#ifndef MAX_SWEEPS
#define MAX_SWEEPS 20
#endif
#ifndef JACOBI_TOL
#define JACOBI_TOL 1.0e-7f
#endif
#ifndef PAIR_SKIP_TOL
// Pair skip threshold: if |A[p][q]| < PAIR_SKIP_TOL, the rotation is identity.
// This must be SMALLER than JACOBI_TOL so that the global off-diagonal norm
// can actually reach the convergence threshold. If PAIR_SKIP_TOL == JACOBI_TOL,
// each remaining element is below the threshold but their Frobenius norm
// (sqrt(N^2) * threshold) can never satisfy the global relative tolerance,
// causing the kernel to waste all remaining sweeps doing nothing.
//
// With PAIR_SKIP_TOL = 0, every pair gets a rotation as long as |A[pq]| > 0,
// which is the most aggressive setting. With a small nonzero value, truly
// negligible pairs are skipped for efficiency. The default 1e-12 is small
// enough that the global Frobenius-norm convergence can be reached for
// N<=64 (worst case: N^2 * (1e-12)^2 = 4096e-24, sqrt = 6.4e-11, well
// below JACOBI_TOL * off0), but large enough to avoid numerical issues
// with denormalized floating-point values that can produce NaN rotations.
#define PAIR_SKIP_TOL 1.0e-12f
#endif
#ifndef LAMBDA_FLOOR
#define LAMBDA_FLOOR 1.0e-7f
#endif

// ------------------------------------------------------------------
// Round-robin pair schedule (Brent-Luk parallel cyclic ordering).
//
// For JN elements (JN even), there are JROUND = JN-1 rounds. In each
// round, JPAIR = JN/2 disjoint pairs are formed. Element JN-1 stays
// fixed; the remaining JN-1 elements rotate.
//
//   pair 0:        (JN-1, round)
//   pair ipair>0:  ((round+ipair) mod (JN-1), (round+JN-1-ipair) mod (JN-1))
//
// Over JN-1 rounds, every pair of elements is visited exactly once.
// ------------------------------------------------------------------
inline int2 jacobi_pair(int round, int ipair) {
    int m = JN - 1;
    if (ipair == 0) return (int2)(m, round);
    return (int2)((round + ipair) % m, (round + m - ipair) % m);
}

// ------------------------------------------------------------------
// jacobi_cyclic_local_batched
//
// Diagonalizes `batch` symmetric N×N matrices. One workgroup per system.
// A and V are loaded into __local memory (padded to JN×JLD). The Brent-Luk
// parallel cyclic schedule is applied for up to MAX_SWEEPS sweeps. Each
// round, JPAIR independent Givens rotations are computed and applied in
// parallel — one pair per PPG work-items. No atomics, no races.
//
// On exit:
//   A[batch][N*N] — eigenvalues on diagonal, off-diagonal ~0
//   V[batch][N*N] — eigenvectors (columns are eigenvectors)
// ------------------------------------------------------------------
__kernel void jacobi_cyclic_local_batched(
    __global float* A,   // [batch][N*N] in/out
    __global float* V,   // [batch][N*N] out
    const int n,         // original (unpadded) dimension
    const int batch
) {
    const int gid = get_group_id(0);   // system index
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    // Local matrices: JN rows × JLD leading dimension.
    // A holds the working matrix; V accumulates eigenvectors.
    __local float lA[JLD * JN];
    __local float lV[JLD * JN];
    // Scratch: rotation params (c, s, p, q) per pair + reduction buffer.
    __local float rot_c[JPAIR];
    __local float rot_s[JPAIR];
    __local int   rot_p[JPAIR];
    __local int   rot_q[JPAIR];
    __local float reduce[WG];

    __global float* gA = A + (size_t)gid * n * n;
    __global float* gV = V + (size_t)gid * n * n;

    // ---- Load A into local (pad to JN with dummy state) ----
    // For i < n, j < n: load from global. Dummy row/col: huge diagonal, 0 off-diag.
    for (int i = lid; i < JN; i += lsz) {
        for (int j = 0; j < JN; ++j) {
            if (i < n && j < n) {
                lA[i * JLD + j] = gA[i * n + j];
            } else if (i == j) {
                lA[i * JLD + j] = 1.0e30f;   // huge diagonal for dummy state
            } else {
                lA[i * JLD + j] = 0.0f;
            }
        }
    }
    // ---- Load V = identity (JN × JN) ----
    for (int i = lid; i < JN; i += lsz) {
        for (int j = 0; j < JN; ++j) {
            lV[i * JLD + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- Compute initial off-diagonal norm (for relative convergence) ----
    float off_part = 0.0f;
    for (int i = lid; i < JN; i += lsz) {
        for (int j = 0; j < JN; ++j) {
            if (i != j) off_part += lA[i * JLD + j] * lA[i * JLD + j];
        }
    }
    reduce[lid] = off_part;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) reduce[lid] += reduce[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float off0 = sqrt(fmax(reduce[0], JACOBI_TOL * JACOBI_TOL));
    barrier(CLK_LOCAL_MEM_FENCE);

    // ================ Jacobi sweeps ================
    float prev_off = fmax(off0, 1.0e-30f);  // for stagnation detection
    int stall_count = 0;                     // consecutive non-improving sweeps
    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {

        for (int round = 0; round < JROUND; ++round) {

            // ---- Phase 1: compute rotation params for each pair ----
            // One work-item per pair (sub == 0 within the pair's group).
            int ipair = lid / PPG;
            int sub   = lid % PPG;
            if (sub == 0 && ipair < JPAIR) {
                int2 pr = jacobi_pair(round, ipair);
                int p = pr.x;
                int q = pr.y;
                // Ensure p < q for consistent 2x2 update.
                if (p > q) { int tmp = p; p = q; q = tmp; }
                float apq = lA[p * JLD + q];
                if (fabs(apq) < PAIR_SKIP_TOL) {
                    rot_c[ipair] = 1.0f;
                    rot_s[ipair] = 0.0f;
                } else {
                    float app = lA[p * JLD + p];
                    float aqq = lA[q * JLD + q];
                    float tau = (aqq - app) / (2.0f * apq);
                    float t = (tau >= 0.0f)
                        ? 1.0f / (tau + sqrt(1.0f + tau * tau))
                        : -1.0f / (-tau + sqrt(1.0f + tau * tau));
                    float c = 1.0f / sqrt(1.0f + t * t);
                    float s = t * c;
#if JACOBI_NORMALIZE_ROTATION
                    float err = fma(-s, s, fma(-c, c, 1.0f));
                    c = fma(0.5f * c, err, c);
                    s = fma(0.5f * s, err, s);
#endif
                    rot_c[ipair] = c;
                    rot_s[ipair] = s;
                }
                rot_p[ipair] = p;
                rot_q[ipair] = q;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

#if JACOBI_BLOCK_UPDATE
            for (int block = lid; block < JPAIR * JPAIR; block += lsz) {
                int a = block / JPAIR;
                int b = block % JPAIR;
                int p = rot_p[a], q = rot_q[a];
                int r = rot_p[b], s = rot_q[b];
                float ca = rot_c[a], sa = rot_s[a];
                float cb = rot_c[b], sb = rot_s[b];
                float apr = lA[p * JLD + r], aps = lA[p * JLD + s];
                float aqr = lA[q * JLD + r], aqs = lA[q * JLD + s];
                float tpr = cb * apr - sb * aps;
                float tps = sb * apr + cb * aps;
                float tqr = cb * aqr - sb * aqs;
                float tqs = sb * aqr + cb * aqs;
                lA[p * JLD + r] = ca * tpr - sa * tqr;
                lA[p * JLD + s] = ca * tps - sa * tqs;
                lA[q * JLD + r] = sa * tpr + ca * tqr;
                lA[q * JLD + s] = sa * tps + ca * tqs;
            }
#else
            // ---- Phase 2a: column update of A (A <- A · J) ----
            // For each pair (p,q), update columns p and q for ALL rows k.
            //   A[k][p]' = c·A[k][p] - s·A[k][q]
            //   A[k][q]' = s·A[k][p] + c·A[k][q]
            // No symmetric counterpart writes — different pairs touch disjoint
            // columns, so no races. The 2×2 block (k=p, k=q) is handled by
            // the same formula (no special case needed).
            ipair = lid / PPG;
            sub   = lid % PPG;
            if (ipair < JPAIR) {
                float c = rot_c[ipair];
                float s = rot_s[ipair];
                int p = rot_p[ipair];
                int q = rot_q[ipair];
                for (int k = sub; k < JN; k += PPG) {
                    float akp = lA[k * JLD + p];
                    float akq = lA[k * JLD + q];
                    lA[k * JLD + p] = c * akp - s * akq;
                    lA[k * JLD + q] = s * akp + c * akq;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // ---- Phase 2b: row update of A (A <- J^T · A) ----
            // For each pair (p,q), update rows p and q for ALL columns k.
            //   A[p][k]' = c·A[p][k] - s·A[q][k]
            //   A[q][k]' = s·A[p][k] + c·A[q][k]
            // Reads the column-updated values from phase 2a. Different pairs
            // touch disjoint rows, so no races. This completes A' = J^T·A·J.
            ipair = lid / PPG;
            sub   = lid % PPG;
            if (ipair < JPAIR) {
                float c = rot_c[ipair];
                float s = rot_s[ipair];
                int p = rot_p[ipair];
                int q = rot_q[ipair];
                for (int k = sub; k < JN; k += PPG) {
                    float apk = lA[p * JLD + k];
                    float aqk = lA[q * JLD + k];
                    lA[p * JLD + k] = c * apk - s * aqk;
                    lA[q * JLD + k] = s * apk + c * aqk;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
#endif

            // ---- Phase 3: apply rotations to V ----
            ipair = lid / PPG;
            sub   = lid % PPG;
            if (ipair < JPAIR) {
                float c = rot_c[ipair];
                float s = rot_s[ipair];
                int p = rot_p[ipair];
                int q = rot_q[ipair];
                for (int k = sub; k < JN; k += PPG) {
                    float vkp = lV[k * JLD + p];
                    float vkq = lV[k * JLD + q];
                    lV[k * JLD + p] = c * vkp - s * vkq;
                    lV[k * JLD + q] = s * vkp + c * vkq;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // ---- Convergence check: relative off-diagonal norm ----
        off_part = 0.0f;
        for (int i = lid; i < JN; i += lsz) {
            for (int j = 0; j < JN; ++j) {
                if (i != j) off_part += lA[i * JLD + j] * lA[i * JLD + j];
            }
        }
        reduce[lid] = off_part;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int off = lsz >> 1; off > 0; off >>= 1) {
            if (lid < off) reduce[lid] += reduce[lid + off];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float off_cur = sqrt(reduce[0]);
        barrier(CLK_LOCAL_MEM_FENCE);
        // Convergence: relative off-diagonal norm below tolerance.
        if (off_cur / off0 < JACOBI_TOL) break;
        // Stagnation detection: if the off-diagonal norm did not improve by
        // at least 10% for STALL_LIMIT consecutive sweeps, further sweeps
        // are unlikely to help (all remaining pairs are at the PAIR_SKIP_TOL
        // floor). Break to avoid wasting the remaining sweeps. This is the
        // key fix for the "20 sweeps but 12 are useless" problem identified
        // in GPU_Optimization.chat.md §2. Require multiple consecutive
        // non-improving sweeps to avoid premature exit during transient
        // plateaus.
        if (sweep > 0 && off_cur > 0.9f * prev_off) {
            stall_count++;
            if (stall_count >= 3) break;
        } else {
            stall_count = 0;
        }
        prev_off = off_cur;
    }

    // ---- Store results back to global ----
    // A: eigenvalues on diagonal (only n×n block, not the dummy).
    for (int i = lid; i < n * n; i += lsz) {
        int r = i / n;
        int c = i - r * n;
        gA[i] = (r == c) ? lA[r * JLD + c] : 0.0f;
    }
    // V: eigenvectors (n×n block).
    for (int i = lid; i < n * n; i += lsz) {
        int r = i / n;
        int c = i - r * n;
        gV[i] = lV[r * JLD + c];
    }
}

// ------------------------------------------------------------------
// build_inv_sqrt_from_eig
//
// Given eigenvalues (on diagonal of A) and eigenvectors (V) from
// jacobi_cyclic_local_batched, computes:
//
//   X = U · diag(rsqrt(λ)) · U^T
//
// where λ_k = A[k][k] and U = V. Also reports λ_min per system
// (minimum eigenvalue over the real n elements, excluding dummy).
//
// On exit:
//   X_out[batch][N*N] — S^{-1/2}
//   lambda_min_out[batch] — smallest eigenvalue per system
// ------------------------------------------------------------------
__kernel void build_inv_sqrt_from_eig(
    __global const float* A,   // [batch][N*N] eigenvalues on diagonal
    __global const float* V,   // [batch][N*N] eigenvectors
    __global float* X_out,     // [batch][N*N] S^{-1/2}
    __global float* lambda_min_out, // [batch]
    const int n,
    const int batch
) {
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    __local float lV[JLD * JN];
    __local float lam[JN];
    __local float reduce[WG];

    __global const float* gA = A + (size_t)gid * n * n;
    __global const float* gV = V + (size_t)gid * n * n;
    __global float* gX = X_out + (size_t)gid * n * n;

    // ---- Load V and eigenvalues into local ----
    for (int i = lid; i < JN; i += lsz) {
        for (int j = 0; j < JN; ++j) {
            if (i < n && j < n)
                lV[i * JLD + j] = gV[i * n + j];
            else
                lV[i * JLD + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    for (int k = lid; k < JN; k += lsz) {
        lam[k] = (k < n) ? gA[k * n + k] : 1.0e30f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- Find λ_min over real eigenvalues (k < n) ----
    float my_min = 1.0e30f;
    for (int k = lid; k < n; k += lsz) {
        my_min = fmin(my_min, lam[k]);
    }
    reduce[lid] = my_min;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) reduce[lid] = fmin(reduce[lid], reduce[lid + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) lambda_min_out[gid] = reduce[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- Compute X_ij = Σ_k V[i,k] · rsqrt(λ_k) · V[j,k] ----
    // Precompute rsqrt(λ_k) for k < n.
    __local float rlam[JN];
    for (int k = lid; k < n; k += lsz) {
        rlam[k] = rsqrt(fmax(lam[k], LAMBDA_FLOOR));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Each work-item handles a subset of (i,j) output elements.
    for (int idx = lid; idx < n * n; idx += lsz) {
        int i = idx / n;
        int j = idx - i * n;
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            float vik = lV[i * JLD + k];
            float vjk = lV[j * JLD + k];
            sum += vik * rlam[k] * vjk;
        }
        gX[i * n + j] = sum;
    }
}

// ------------------------------------------------------------------
// scale_eigenvectors_batched
//
// Phase 3: N>64 S^{-1/2} reconstruction helper.
// Computes V_scaled[i][k] = V[i][k] · rsqrt(max(λ_k, LAMBDA_FLOOR))
// and reports λ_min per system. Works for any N (global memory only).
// One workgroup per system; each thread handles a subset of (i,k) pairs.
//
// On exit:
//   V_scaled[batch][N*N] — V scaled by rsqrt(λ)
//   lambda_min_out[batch] — smallest eigenvalue per system
// ------------------------------------------------------------------
__kernel void scale_eigenvectors_batched(
    __global const float* A,        // [batch][N*N] eigenvalues on diagonal
    __global const float* V,        // [batch][N*N] eigenvectors
    __global float* V_scaled,       // [batch][N*N] output: V · rsqrt(λ)
    __global float* lambda_min_out, // [batch]
    const int n,
    const int batch
) {
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    __global const float* gA = A + (size_t)gid * n * n;
    __global const float* gV = V + (size_t)gid * n * n;
    __global float* gVs = V_scaled + (size_t)gid * n * n;

    // Find λ_min (thread 0 does it; could parallelize but n is small)
    if (lid == 0) {
        float lmin = 1.0e30f;
        for (int k = 0; k < n; ++k) {
            float lam = gA[k * n + k];
            if (lam < lmin) lmin = lam;
        }
        lambda_min_out[gid] = lmin;
    }

    // V_scaled[i][k] = V[i][k] · rsqrt(max(λ_k, LAMBDA_FLOOR))
    for (int idx = lid; idx < n * n; idx += lsz) {
        int i = idx / n;
        int k = idx - i * n;
        float lam = gA[k * n + k];
        float rlam = rsqrt(fmax(lam, LAMBDA_FLOOR));
        gVs[idx] = gV[idx] * rlam;
    }
}
