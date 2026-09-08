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
// - Inner pivot diagonalized by row-parallel cyclic Jacobi
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
#define JACOBI_TOL 1.0e-7f
#endif
#ifndef PAIR_SKIP_TOL
#define PAIR_SKIP_TOL 1.0e-12f
#endif

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
    __local float cs[2];             // (c, s) broadcast for row-parallel Jacobi

    __global float* gA = A + (size_t)gid * n * n;
    __global float* gV = V + (size_t)gid * n * n;

    // ---- Initialize V = I (N×N) ----
    for (int idx = lid; idx < n * n; idx += lsz) {
        int r = idx / n;
        int c = idx - r * n;
        gV[idx] = (r == c) ? 1.0f : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

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

                // ---- Step 2: Local row-parallel cyclic Jacobi on lA ----
                // Diagonalizes the PB×PB local matrix (including dummies).
                // Thread 0 computes (c,s); all threads update rows.
                // Fixed 10 sweeps — sufficient for PB≤64 (typically 6-8 converge to 1e-7).
                for (int jsweep = 0; jsweep < 10; ++jsweep) {
                    for (int p = 0; p < PB; ++p) {
                        for (int q = p + 1; q < PB; ++q) {
                            // Thread 0 computes rotation
                            if (lid == 0) {
                                float apq = lA[p * PLD + q];
                                if (fabs(apq) < PAIR_SKIP_TOL) {
                                    cs[0] = 1.0f;
                                    cs[1] = 0.0f;
                                } else {
                                    float app = lA[p * PLD + p];
                                    float aqq = lA[q * PLD + q];
                                    float tau = (aqq - app) / (2.0f * apq);
                                    float t = (tau >= 0.0f)
                                        ? 1.0f / (tau + sqrt(1.0f + tau * tau))
                                        : -1.0f / (-tau + sqrt(1.0f + tau * tau));
                                    float c = 1.0f / sqrt(1.0f + t * t);
                                    float s = t * c;
                                    cs[0] = c;
                                    cs[1] = s;
                                }
                            }
                            barrier(CLK_LOCAL_MEM_FENCE);
                            float c = cs[0];
                            float s = cs[1];

                            // All threads update rows of lA (symmetric)
                            for (int k = lid; k < PB; k += lsz) {
                                if (k != p && k != q) {
                                    float akp = lA[k * PLD + p];
                                    float akq = lA[k * PLD + q];
                                    float npv = c * akp - s * akq;
                                    float nqv = s * akp + c * akq;
                                    lA[k * PLD + p] = npv;
                                    lA[p * PLD + k] = npv;
                                    lA[k * PLD + q] = nqv;
                                    lA[q * PLD + k] = nqv;
                                }
                                // Update lU (eigenvectors)
                                float vkp = lU[k * PLD + p];
                                float vkq = lU[k * PLD + q];
                                lU[k * PLD + p] = c * vkp - s * vkq;
                                lU[k * PLD + q] = s * vkp + c * vkq;
                            }
                            barrier(CLK_LOCAL_MEM_FENCE);

                            // Update 2×2 diagonal block
                            if (lid == 0) {
                                float app = lA[p * PLD + p];
                                float aqq = lA[q * PLD + q];
                                float apq = lA[p * PLD + q];
                                lA[p * PLD + p] = c*c*app - 2.0f*c*s*apq + s*s*aqq;
                                lA[q * PLD + q] = s*s*app + 2.0f*c*s*apq + c*c*aqq;
                                lA[p * PLD + q] = 0.0f;
                                lA[q * PLD + p] = 0.0f;
                            }
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                    }
                }

                // ---- Step 3: Store transformed principal block back to global ----
                // lA now has eigenvalues on diagonal, lU has eigenvectors.
                // Store the diagonalized principal block: D = U^T · P · U
                for (int i = lid; i < m; i += lsz) {
                    for (int j = 0; j < m; ++j) {
                        float val = (i == j) ? lA[i * PLD + j] : 0.0f;
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
                barrier(CLK_LOCAL_MEM_FENCE);

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
                        for (int i = lid; i < nrows * m; i += lsz) {
                            int r = i / m;
                            int c = i - r * m;
                            float sum = 0.0f;
                            for (int k = 0; k < m; ++k) {
                                sum += strip[r * PB + k] * lU[k * PLD + c];
                            }
                            // Store Y → [A_kp | A_kq]
                            int gr = kt*B + koff + r;
                            int gc;
                            if (c < Bp) gc = bp*B + c;
                            else gc = bq*B + (c - Bp);
                            gA[gr * n + gc] = sum;
                            // Store Y^T → [A_pk | A_qk] (symmetry)
                            gA[gc * n + gr] = sum;
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

                        // Y = strip × lU
                        for (int i = lid; i < nrows * m; i += lsz) {
                            int r = i / m;
                            int c = i - r * m;
                            float sum = 0.0f;
                            for (int k = 0; k < m; ++k) {
                                sum += strip[r * PB + k] * lU[k * PLD + c];
                            }
                            int gr = kt*B + koff + r;
                            int gc;
                            if (c < Bp) gc = bp*B + c;
                            else gc = bq*B + (c - Bp);
                            gV[gr * n + gc] = sum;
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
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
        // Stagnation detection
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
