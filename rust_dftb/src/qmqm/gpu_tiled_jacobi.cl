// ==================================================================
// GPU Tiled Block Jacobi Eigensolver (N > 64)
// ==================================================================
//
// Phase 2 of the H-Bond manifest: one-WG tiled/block Jacobi for
// dense symmetric eigendecomposition of matrices larger than the
// N≤64 full-local limit.
//
// Architecture (manifest §4.3):
// - One workgroup per system (parallelism across systems)
// - A and V stored in GLOBAL memory (N×N per system, no N² local limit)
// - Local memory: 2B×2B compound pivot + U rotation + strip workspace
// - Block size B=32 → pivot capacity 2B=64 (fits local memory)
// - Inner pivot diagonalized by Brent–Luk parallel cyclic Jacobi
//   (32 independent pairs/round, 8 threads/pair, all 256 threads active)
// - Strip updates: load [A_kp|A_kq], multiply by U, store + symmetric transpose
// - One kernel launch per batched eigensolve; all sync is WG-local barriers
//
// For N=87 (AT/GC nucleobase pairs): 3 blocks of 32,32,23 → 3 pivots/sweep
// No N² local-memory scaling — N can be 96, 128, 200, 500...
//
// ------------------------------------------------------------------

#ifndef B
#define B 32
#endif
#ifndef PB
#define PB 64          // pivot capacity = 2*B
#endif
#ifndef PLD
#define PLD 65          // padded leading dimension = PB+1 (bank conflicts)
#endif
#ifndef WG
#define WG 256
#endif
#ifndef STRIP_R
#define STRIP_R 32      // rows per strip tile
#endif
#ifndef MAX_SWEEPS
#define MAX_SWEEPS 50
#endif
#ifndef JACOBI_TOL
#define JACOBI_TOL 1.0e-9f
#endif
#ifndef PAIR_SKIP_TOL
#define PAIR_SKIP_TOL 1.0e-12f
#endif

// ---- Arithmetic precision modes (manifest §12 D2) ----
// JACOBI_PREC=0: pure FP32-FMA (rotation params, block/strip updates)
// JACOBI_PREC=1: FP64 only for scalar rotation construction c,s — FP32-FMA updates
// JACOBI_PREC=2: broad FP64 everywhere (accuracy reference; ~1.29M double
//                block-updates per pivot — a throughput disaster on a 3090)
#ifndef JACOBI_PREC
#define JACOBI_PREC 2
#endif
#if JACOBI_PREC >= 1
typedef double jrot_t;   // rotation-parameter precision
#else
typedef float jrot_t;
#endif
#if JACOBI_PREC >= 2
typedef double jupd_t;   // block/strip update accumulation precision
#else
typedef float jupd_t;
#endif

// ---- Inner pivot Brent–Luk parallel Jacobi parameters ----
// PB=64 → JPAIR_INNER=32 independent pairs/round, PPG_INNER=8 threads/pair,
// JROUND_INNER=63 rounds/sweep. 32×8=256=WG threads all active.
#ifndef JPAIR_INNER
#define JPAIR_INNER (PB / 2)
#endif
#ifndef PPG_INNER
#define PPG_INNER (WG / JPAIR_INNER)
#endif
#ifndef JROUND_INNER
#define JROUND_INNER (PB - 1)
#endif
#ifndef INNER_SWEEPS
#define INNER_SWEEPS 20
#endif

// ------------------------------------------------------------------
// Round-robin pair schedule (Brent–Luk parallel cyclic ordering).
// For PB elements (PB even), PB-1 rounds. In each round, PB/2 disjoint
// pairs. Element PB-1 stays fixed; the remaining PB-1 elements rotate.
// Over PB-1 rounds, every pair of elements is visited exactly once.
// ------------------------------------------------------------------
inline int2 inner_jacobi_pair(int round, int ipair) {
    int m = PB - 1;
    if (ipair == 0) return (int2)(m, round);
    return (int2)((round + ipair) % m, (round + m - ipair) % m);
}

// ------------------------------------------------------------------
// tiled_jacobi_batched
//
// Diagonalizes `batch` symmetric N×N matrices. One workgroup per system.
// A and V reside in global memory; only the 2B×2B compound pivot and
// strip workspace live in local memory.
//
// On exit:
//   A[batch][N*N] — eigenvalues on diagonal, off-diagonal ~0
//   V[batch][N*N] — eigenvectors (columns are eigenvectors)
// ------------------------------------------------------------------
__kernel void tiled_jacobi_batched(
    __global float* A,   // [batch][N*N] in/out
    __global float* V,    // [batch][N*N] out
    const int n,          // physical dimension
    const int batch
) {
    const int gid = get_group_id(0);   // system index
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    __local float lA[PLD * PB];      // compound pivot (64×65)
    __local float lU[PLD * PB];     // rotation matrix from local Jacobi (64×65)
    __local float strip[STRIP_R * PB]; // strip workspace (32×64)
    __local float reduce[WG];        // reduction buffer
    // Brent–Luk parallel Jacobi rotation params (32 pairs)
    __local jrot_t rot_c[JPAIR_INNER];
    __local jrot_t rot_s[JPAIR_INNER];
    __local int   rot_p[JPAIR_INNER];
    __local int   rot_q[JPAIR_INNER];

    __global float* gA = A + (size_t)gid * n * n;
    __global float* gV = V + (size_t)gid * n * n;

    // ---- Initialize V = I (N×N) ----
    for (int idx = lid; idx < n * n; idx += lsz) {
        int r = idx / n;
        int c = idx - r * n;
        gV[idx] = (r == c) ? 1.0f : 0.0f;
    }
    // V is in global memory — need global fence before first pivot reads gV
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // ---- Compute initial off-diagonal norm (for relative convergence) ----
    float off_part = 0.0f;
    for (int idx = lid; idx < n * n; idx += lsz) {
        int r = idx / n;
        int c = idx - r * n;
        if (r != c) off_part += gA[idx] * gA[idx];
    }
    reduce[lid] = off_part;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) reduce[lid] += reduce[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float off0 = sqrt(fmax(reduce[0], JACOBI_TOL * JACOBI_TOL));
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- Block schedule ----
    int n_blocks = (n + B - 1) / B;
    // Block sizes: block i has size min(B, n - i*B)
    // Number of block pairs: n_blocks*(n_blocks-1)/2

    // ================ Jacobi sweeps ================
    float prev_off = fmax(off0, 1.0e-30f);
    int stall_count = 0;
    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {

        // Cyclic block pair order: (0,1), (0,2), ..., (1,2), (1,3), ...
        for (int bp = 0; bp < n_blocks; ++bp) {
            for (int bq = bp + 1; bq < n_blocks; ++bq) {
                int Bp = min(B, n - bp * B);  // active rows in block p
                int Bq = min(B, n - bq * B);  // active rows in block q
                int m = Bp + Bq;              // active pivot dimension (≤ PB)

                // ---- Step 1: Load compound pivot into lA ----
                // lA = [A_pp  A_pq]
                //      [A_qp  A_qq]
                // Padded to PB×PLD with dummy sentinel states.
                for (int i = lid; i < PB; i += lsz) {
                    for (int j = 0; j < PB; ++j) {
                        float val;
                        if (i < Bp && j < Bp) {
                            // A_pp block
                            val = gA[(bp*B+i)*n + (bp*B+j)];
                        } else if (i < Bp && j >= Bp && j < m) {
                            // A_pq block (column offset by Bp)
                            val = gA[(bp*B+i)*n + (bq*B+(j-Bp))];
                        } else if (i >= Bp && i < m && j < Bp) {
                            // A_qp block (row offset by Bp)
                            val = gA[(bq*B+(i-Bp))*n + (bp*B+j)];
                        } else if (i >= Bp && i < m && j >= Bp && j < m) {
                            // A_qq block
                            val = gA[(bq*B+(i-Bp))*n + (bq*B+(j-Bp))];
                        } else if (i == j) {
                            // Dummy diagonal: huge value
                            val = 1.0e30f;
                        } else {
                            // Dummy off-diagonal: zero
                            val = 0.0f;
                        }
                        lA[i * PLD + j] = val;
                    }
                }
                // Initialize lU = I (PB×PB, padded)
                for (int i = lid; i < PB; i += lsz) {
                    for (int j = 0; j < PB; ++j) {
                        lU[i * PLD + j] = (i == j) ? 1.0f : 0.0f;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                // ---- Step 2: Local Brent–Luk parallel cyclic Jacobi on lA ----
                // Diagonalizes the PB×PB local matrix (including dummies).
                // JPAIR_INNER=32 independent rotations per round, PPG_INNER=8
                // threads per pair → all 256 threads active (vs serial: thread 0
                // computes one rotation, 75% idle).
                // Block update: J^T·A·J applied via JPAIR² 2×2 blocks in parallel.
                for (int jsweep = 0; jsweep < INNER_SWEEPS; ++jsweep) {
                    for (int round = 0; round < JROUND_INNER; ++round) {

                        // Phase 1: compute rotation params for each pair
                        // JACOBI_PREC>=1 keeps the scalar c,s construction in
                        // f64 (cheap: 32 scalars/round); the expensive part was
                        // the millions of block/strip updates, not these.
                        int ipair = lid / PPG_INNER;
                        int sub   = lid % PPG_INNER;
                        if (sub == 0 && ipair < JPAIR_INNER) {
                            int2 pr = inner_jacobi_pair(round, ipair);
                            int p = pr.x;
                            int q = pr.y;
                            if (p > q) { int tmp = p; p = q; q = tmp; }
                            jrot_t apq = (jrot_t)lA[p * PLD + q];
                            if (fabs(apq) < (jrot_t)PAIR_SKIP_TOL) {
                                rot_c[ipair] = (jrot_t)1.0;
                                rot_s[ipair] = (jrot_t)0.0;
                            } else {
                                jrot_t app = (jrot_t)lA[p * PLD + p];
                                jrot_t aqq = (jrot_t)lA[q * PLD + q];
                                jrot_t tau = (aqq - app) / ((jrot_t)2.0 * apq);
                                jrot_t t = (tau >= (jrot_t)0.0)
                                    ? (jrot_t)1.0 / (tau + sqrt((jrot_t)1.0 + tau * tau))
                                    : -(jrot_t)1.0 / (-tau + sqrt((jrot_t)1.0 + tau * tau));
                                jrot_t c = (jrot_t)1.0 / sqrt((jrot_t)1.0 + t * t);
                                jrot_t s = t * c;
                                rot_c[ipair] = c;
                                rot_s[ipair] = s;
                            }
                            rot_p[ipair] = p;
                            rot_q[ipair] = q;
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);

                        // Phase 2: block update of lA (J^T · lA · J)
                        // Process all JPAIR² 2×2 blocks; 1024 blocks / 256 threads = 4 per thread.
                        // JACOBI_PREC<2 → FP32 FMA (D2: the f64 here was ~1.29M
                        // double ops per pivot, a 3090 throughput disaster).
                        for (int blk = lid; blk < JPAIR_INNER * JPAIR_INNER; blk += lsz) {
                            int a = blk / JPAIR_INNER;
                            int b = blk % JPAIR_INNER;
                            int p = rot_p[a], q = rot_q[a];
                            int r = rot_p[b], sc = rot_q[b];
                            jupd_t ca = (jupd_t)rot_c[a], sa = (jupd_t)rot_s[a];
                            jupd_t cb = (jupd_t)rot_c[b], sb = (jupd_t)rot_s[b];
                            jupd_t apr = (jupd_t)lA[p * PLD + r], aps = (jupd_t)lA[p * PLD + sc];
                            jupd_t aqr = (jupd_t)lA[q * PLD + r], aqs = (jupd_t)lA[q * PLD + sc];
                            jupd_t tpr = fma(cb, apr, -sb * aps);
                            jupd_t tps = fma(sb, apr, cb * aps);
                            jupd_t tqr = fma(cb, aqr, -sb * aqs);
                            jupd_t tqs = fma(sb, aqr, cb * aqs);
                            lA[p * PLD + r] = (float)fma(ca, tpr, -sa * tqr);
                            lA[p * PLD + sc] = (float)fma(ca, tps, -sa * tqs);
                            lA[q * PLD + r] = (float)fma(sa, tpr, ca * tqr);
                            lA[q * PLD + sc] = (float)fma(sa, tps, ca * tqs);
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);

                        // Phase 3: apply rotations to lU (eigenvectors)
                        ipair = lid / PPG_INNER;
                        sub   = lid % PPG_INNER;
                        if (ipair < JPAIR_INNER) {
                            jupd_t c = (jupd_t)rot_c[ipair];
                            jupd_t s = (jupd_t)rot_s[ipair];
                            int p = rot_p[ipair];
                            int q = rot_q[ipair];
                            for (int k = sub; k < PB; k += PPG_INNER) {
                                jupd_t vkp = (jupd_t)lU[k * PLD + p];
                                jupd_t vkq = (jupd_t)lU[k * PLD + q];
                                lU[k * PLD + p] = (float)fma(c, vkp, -s * vkq);
                                lU[k * PLD + q] = (float)fma(s, vkp, c * vkq);
                            }
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                }

                // ---- Step 3: Store transformed principal block back to global ----
                // lA now has U^T·P·U = D + R (eigenvalues on diagonal + residual
                // off-diagonal R). Store the ACTUAL lA values, not just the
                // diagonal — zeroing R breaks the invariant A_k = V_k^T A_0 V_k
                // because with finite inner sweeps R is not guaranteed tiny.
                for (int i = lid; i < m; i += lsz) {
                    for (int j = 0; j < m; ++j) {
                        float val = lA[i * PLD + j];  // R1: store actual value, not (i==j)?diag:0
                        // Map back to global (i,j) in the compound block
                        int gi, gj;
                        if (i < Bp && j < Bp) {
                            gi = bp*B+i; gj = bp*B+j;
                        } else if (i < Bp && j >= Bp) {
                            gi = bp*B+i; gj = bq*B+(j-Bp);
                        } else if (i >= Bp && j < Bp) {
                            gi = bq*B+(i-Bp); gj = bp*B+j;
                        } else {
                            gi = bq*B+(i-Bp); gj = bq*B+(j-Bp);
                        }
                        gA[gi * n + gj] = val;
                    }
                }
                // R3: strip updates read gA — need global fence
                barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

                // ---- Step 4: Update off-diagonal A strips ----
                // For each row tile k not in {p,q}: A[k][p] = A[k][p]·U, A[k][q] = A[k][q]·U
                // Symmetry: A[p][k] = A[k][p]^T
                for (int kt = 0; kt < n_blocks; ++kt) {
                    if (kt == bp || kt == bq) continue;
                    int Bk = min(B, n - kt * B);
                    // Load strip: [A_kp | A_kq] into strip[STRIP_R × PB]
                    // Process in tiles of STRIP_R rows
                    for (int koff = 0; koff < Bk; koff += STRIP_R) {
                        int nrows = min(STRIP_R, Bk - koff);
                        // Load strip
                        for (int i = lid; i < nrows * m; i += lsz) {
                            int r = i / m;
                            int c = i - r * m;
                            int gr = kt*B + koff + r;
                            int gc;
                            if (c < Bp) gc = bp*B + c;
                            else gc = bq*B + (c - Bp);
                            strip[r * PB + c] = gA[gr * n + gc];
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);

                        // Y = strip × lU (nrows × m) × (m × m) → (nrows × m)
                        // JACOBI_PREC>=2: f64 accumulation (reference).
                        // JACOBI_PREC<2:  FP32 FMA, 4 independent accumulators —
                        // shorter dependency chain + better summation than one
                        // serial accumulator; cheaper than Kahan or f64 (§12 D2).
                        for (int i = lid; i < nrows * m; i += lsz) {
                            int r = i / m;
                            int c = i - r * m;
#if JACOBI_PREC >= 2
                            double sum = 0.0;
                            for (int k = 0; k < m; ++k) {
                                sum += (double)strip[r * PB + k] * (double)lU[k * PLD + c];
                            }
                            float sumf = (float)sum;
#else
                            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                            int k = 0;
                            for (; k + 4 <= m; k += 4) {
                                s0 = fma(strip[r * PB + k    ], lU[(k    ) * PLD + c], s0);
                                s1 = fma(strip[r * PB + k + 1], lU[(k + 1) * PLD + c], s1);
                                s2 = fma(strip[r * PB + k + 2], lU[(k + 2) * PLD + c], s2);
                                s3 = fma(strip[r * PB + k + 3], lU[(k + 3) * PLD + c], s3);
                            }
                            for (; k < m; ++k) s0 = fma(strip[r * PB + k], lU[k * PLD + c], s0);
                            float sumf = (s0 + s1) + (s2 + s3);
#endif
                            // Store Y → [A_kp | A_kq]
                            int gr = kt*B + koff + r;
                            int gc;
                            if (c < Bp) gc = bp*B + c;
                            else gc = bq*B + (c - Bp);
                            gA[gr * n + gc] = sumf;
                            // Store Y^T → [A_pk | A_qk] (symmetry)
                            gA[gc * n + gr] = sumf;
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                }

                // ---- Step 5: Update V strips ----
                // V[k][p] = V[k][p]·U, V[k][q] = V[k][q]·U
                for (int kt = 0; kt < n_blocks; ++kt) {
                    int Bk = min(B, n - kt * B);
                    for (int koff = 0; koff < Bk; koff += STRIP_R) {
                        int nrows = min(STRIP_R, Bk - koff);
                        // Load strip: [V_kp | V_kq]
                        for (int i = lid; i < nrows * m; i += lsz) {
                            int r = i / m;
                            int c = i - r * m;
                            int gr = kt*B + koff + r;
                            int gc;
                            if (c < Bp) gc = bp*B + c;
                            else gc = bq*B + (c - Bp);
                            strip[r * PB + c] = gV[gr * n + gc];
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);

                        // Y = strip × lU — same JACOBI_PREC split as A strips
                        for (int i = lid; i < nrows * m; i += lsz) {
                            int r = i / m;
                            int c = i - r * m;
#if JACOBI_PREC >= 2
                            double sum = 0.0;
                            for (int k = 0; k < m; ++k) {
                                sum += (double)strip[r * PB + k] * (double)lU[k * PLD + c];
                            }
                            float sumf = (float)sum;
#else
                            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                            int k = 0;
                            for (; k + 4 <= m; k += 4) {
                                s0 = fma(strip[r * PB + k    ], lU[(k    ) * PLD + c], s0);
                                s1 = fma(strip[r * PB + k + 1], lU[(k + 1) * PLD + c], s1);
                                s2 = fma(strip[r * PB + k + 2], lU[(k + 2) * PLD + c], s2);
                                s3 = fma(strip[r * PB + k + 3], lU[(k + 3) * PLD + c], s3);
                            }
                            for (; k < m; ++k) s0 = fma(strip[r * PB + k], lU[k * PLD + c], s0);
                            float sumf = (s0 + s1) + (s2 + s3);
#endif
                            int gr = kt*B + koff + r;
                            int gc;
                            if (c < Bp) gc = bp*B + c;
                            else gc = bq*B + (c - Bp);
                            gV[gr * n + gc] = sumf;
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                }
                // R3: next pivot and convergence check read gA/gV — need global fence
                barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            }
        }

        // ---- Convergence check: physical off-diagonal Frobenius norm ----
        off_part = 0.0f;
        for (int idx = lid; idx < n * n; idx += lsz) {
            int r = idx / n;
            int c = idx - r * n;
            if (r != c) off_part += gA[idx] * gA[idx];
        }
        reduce[lid] = off_part;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int off = lsz >> 1; off > 0; off >>= 1) {
            if (lid < off) reduce[lid] += reduce[lid + off];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float off_cur = sqrt(reduce[0]);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (off_cur / off0 < JACOBI_TOL) break;
        // Stagnation detection: if off-diagonal norm hasn't improved by 10%
        // for 3 consecutive sweeps, further sweeps won't help (f32 floor).
        if (sweep > 0 && off_cur > 0.9f * prev_off) {
            stall_count++;
            if (stall_count >= 3) break;
        } else {
            stall_count = 0;
        }
        prev_off = off_cur;
    }

    // ---- Final: zero out off-diagonal of A (eigenvalues on diagonal) ----
    for (int idx = lid; idx < n * n; idx += lsz) {
        int r = idx / n;
        int c = idx - r * n;
        if (r != c) gA[idx] = 0.0f;
    }
}
