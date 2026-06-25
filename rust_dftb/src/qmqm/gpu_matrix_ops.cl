// ==================================================================
// GPU Matrix Operations for DFTB SCF Cycle
// ==================================================================
//
// This file contains OpenCL kernels for the core linear algebra
// operations needed in the DFTB self-consistent charge (SCC) cycle
// on GPU. All kernels operate on batched dense matrices — multiple
// fragment Hamiltonians/overlaps/density matrices stacked along a
// third dimension, one per fragment.
//
// Matrix layout: row-major, N×N per batch element, contiguous in
// global memory: element (i,j) of batch b is at index b*N*N + i*N + j.
//
// Tunable constants (set by Rust harness via text substitution):
//   TILE_M, TILE_N, TILE_K — GEMM tile dimensions for local memory
//   WG_REDUCE              — workgroup size for reduction kernels
//   JACOBI_MAX_M           — max block size for local Jacobi kernels
//
// ------------------------------------------------------------------
// Kernel inventory:
//
//   batched_gemm                  — tiled batched matrix multiply C = α·A·B + β·C
//   scale_density_guess           — initial density guess from Hamiltonian eigenvalue bounds
//   purify_mcweeny                — McWeeny purification step: D ← 3D² − 2D³
//   purify_tc2_step               — trace-correcting step (all batches, uniform mode)
//   purify_tc2_one_batch          — trace-correcting step (single batch, per-batch mode)
//   trace_reduce                  — per-batch trace reduction (sum of diagonal)
//   idempotency_reduce            — per-batch idempotency error ||D²−D||_F
//   local_jacobi_blocks           — serial-in-workgroup Jacobi eigendecomposition of small blocks
//   local_jacobi_blocks_parallel  — row-parallel Jacobi eigendecomposition of small blocks
//
// ------------------------------------------------------------------
// Algorithm overview:
//
// The SCC cycle for each fragment requires solving the generalized
// eigenvalue problem H·C = S·C·ε. On GPU we avoid sequential
// algorithms (QR, divide-and-conquer) and instead use either:
//
//   (A) Density matrix purification (Palser–Manolopoulos):
//       Start from a scaled Hamiltonian, iterate D ← 3D²−2D³ (McWeeny)
//       or trace-correcting (TC2) until D is idempotent with the
//       correct electron count. Only needs GEMM + elementwise ops.
//       Does NOT yield orbitals — only the density matrix.
//
//   (B) Direct diagonalization via block Jacobi:
//       Partition the N×N matrix into T×T blocks. Each round,
//       diagonalize a 2T×2T compound block [A_pp, A_pq; A_qp, A_qq]
//       in local memory, then apply the resulting rotation to all
//       other block-rows/columns via tiled GEMM. Brent–Luk ordering
//       ensures non-overlapping block pairs per round. Yields both
//       eigenvalues and eigenvectors (orbitals).
//
// The Löwdin transform H' = X^T·H·X (where X = S^{-1/2}) converts
// the generalized problem to a standard one and is performed via
// two batched GEMM calls.
//
// For small blocks (2T×2T, T≤32 → 64×64), the entire matrix fits
// in GPU local memory. The local Jacobi kernels handle this case.
// ==================================================================

#ifndef TILE_M
#define TILE_M 16
#endif

#ifndef TILE_N
#define TILE_N 16
#endif

#ifndef TILE_K
#define TILE_K 32
#endif

#ifndef WG_REDUCE
#define WG_REDUCE 256
#endif

#ifndef JACOBI_MAX_M
#define JACOBI_MAX_M 64
#endif

// ------------------------------------------------------------------
// batched_gemm
//
// Tiled batched matrix multiply: C_b = α·op(A_b)·op(B_b) + β·C_b
// for each batch element b = 0..batch-1.
//
// Each workgroup computes one TILE_M × TILE_N output tile of one
// batch element. Tiles of A and B are cooperatively loaded into
// local memory and reused across the K dimension, maximizing
// arithmetic intensity (~2·TILE_K FLOPs per loaded element).
// This is the workhorse kernel — used for Löwdin transform,
// purification (D², D³), density matrix construction, and block
// Jacobi rotation updates.
// ------------------------------------------------------------------
__kernel void batched_gemm(
    const int n,
    const int batch,
    const int trans_a,
    const int trans_b,
    const float alpha,
    const float beta,
    __global const float* A,
    __global const float* B,
    __global float* C,
    __local float* As,
    __local float* Bs
) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int row = get_group_id(0) * TILE_M + ly;
    const int col = get_group_id(1) * TILE_N + lx;
    const int ib = get_group_id(2);
    if (ib >= batch) return;

    const int stride = n * n;
    __global const float* Ab = A + ib * stride;
    __global const float* Bb = B + ib * stride;
    __global float* Cb = C + ib * stride;

    float sum = 0.0f;
    const int lid = ly * TILE_N + lx;
    const int wg = TILE_M * TILE_N;

    for (int k0 = 0; k0 < n; k0 += TILE_K) {
        for (int t = lid; t < TILE_M * TILE_K; t += wg) {
            int rr = t / TILE_K;
            int kk = t - rr * TILE_K;
            int ar = get_group_id(0) * TILE_M + rr;
            int ac = k0 + kk;
            float v = 0.0f;
            if (ar < n && ac < n) {
                v = trans_a ? Ab[ac * n + ar] : Ab[ar * n + ac];
            }
            As[t] = v;
        }
        for (int t = lid; t < TILE_K * TILE_N; t += wg) {
            int kk = t / TILE_N;
            int cc = t - kk * TILE_N;
            int br = k0 + kk;
            int bc = get_group_id(1) * TILE_N + cc;
            float v = 0.0f;
            if (br < n && bc < n) {
                v = trans_b ? Bb[bc * n + br] : Bb[br * n + bc];
            }
            Bs[t] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (row < n && col < n) {
            for (int kk = 0; kk < TILE_K; ++kk) {
                sum += As[ly * TILE_K + kk] * Bs[kk * TILE_N + lx];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < n && col < n) {
        int idx = row * n + col;
        Cb[idx] = alpha * sum + beta * Cb[idx];
    }
}

// ------------------------------------------------------------------
// scale_density_guess
//
// Computes the initial density matrix guess for Palser–Manolopoulos
// purification. Given the Hamiltonian H and its Gershgorin eigenvalue
// bounds [λ_min, λ_max], the guess is:
//
//   D = I − (H − λ_min·I) / (λ_max − λ_min)
//
// This maps H's spectrum into [0,1] so that D has eigenvalues in
// [0,1], a necessary condition for purification to converge.
// The diagonal is shifted by λ_min; off-diagonal entries are scaled
// directly. One thread per matrix element across all batches.
// ------------------------------------------------------------------
__kernel void scale_density_guess(
    const int n,
    const int batch,
    __global const float* H,
    __global float* D,
    __global const float2* bounds
) {
    const int gid = get_global_id(0);
    const int stride = n * n;
    const int total = batch * stride;
    if (gid >= total) return;
    const int ib = gid / stride;
    const int rem = gid - ib * stride;
    const int row = rem / n;
    const int col = rem - row * n;
    const float2 b = bounds[ib];
    const float inv = 1.0f / fmax(b.y - b.x, 1.0e-20f);
    float h = H[gid];
    if (row == col) h -= b.x;
    float hs = h * inv;
    D[gid] = (row == col ? 1.0f : 0.0f) - hs;
}

// ------------------------------------------------------------------
// purify_mcweeny
//
// McWeeny purification step: D ← 3·D² − 2·D³
//
// Given D² and D³ (computed by the host via two batched_gemm calls),
// this elementwise kernel produces the next iterate. McWeeny
// purification converges quadratically when all eigenvalues of D
// are in [0,1], driving them toward 0 (unoccupied) or 1 (occupied).
// The fixed point is an idempotent density matrix (D² = D) with the
// same trace as the initial guess. One thread per element.
// ------------------------------------------------------------------
__kernel void purify_mcweeny(
    const int n,
    const int batch,
    __global const float* D2,
    __global const float* D3,
    __global float* D
) {
    const int gid = get_global_id(0);
    const int total = batch * n * n;
    if (gid >= total) return;
    D[gid] = 3.0f * D2[gid] - 2.0f * D3[gid];
}

// ------------------------------------------------------------------
// purify_tc2_step
//
// Trace-correcting (TC2) purification step, applied to ALL batches
// with a uniform mode. Given D and D²:
//
//   mode 0 (trace too large):  D ← D²
//   mode 1 (trace too small):  D ← 2D − D²
//
// TC2 guarantees that Tr(D) is preserved exactly at each step,
// unlike McWeeny which only preserves it in the limit. The mode
// is chosen per-batch by the host based on the current trace vs.
// the target electron count. This kernel applies the same mode
// to all batches — use purify_tc2_one_batch for per-batch control.
// ------------------------------------------------------------------
__kernel void purify_tc2_step(
    const int n,
    const int batch,
    const int mode,
    __global const float* D,
    __global const float* D2,
    __global float* Out
) {
    const int gid = get_global_id(0);
    const int total = batch * n * n;
    if (gid >= total) return;
    float d = D[gid];
    float d2 = D2[gid];
    Out[gid] = mode == 0 ? d2 : (2.0f * d - d2);
}

// ------------------------------------------------------------------
// purify_tc2_one_batch
//
// Same TC2 purification as purify_tc2_step, but applied to a single
// batch element only. This allows the host to choose a different
// mode (D² or 2D−D²) for each fragment independently, since each
// fragment may have a different electron count and thus require a
// different correction direction at each iteration.
// ------------------------------------------------------------------
__kernel void purify_tc2_one_batch(
    const int n,
    const int batch_index,
    const int mode,
    __global const float* D,
    __global const float* D2,
    __global float* Out
) {
    const int gid = get_global_id(0);
    const int stride = n * n;
    if (gid >= stride) return;
    const int idx = batch_index * stride + gid;
    float d = D[idx];
    float d2 = D2[idx];
    Out[idx] = mode == 0 ? d2 : (2.0f * d - d2);
}

// ------------------------------------------------------------------
// trace_reduce
//
// Computes Tr(A_b) = Σ_i A_b[i,i] for each batch element b.
// One workgroup per batch element; partial sums are accumulated
// in local memory via tree reduction. The trace is needed by the
// TC2 purification host logic to decide whether to apply D² or
// 2D−D² at each step (to keep Tr(D) equal to the electron count).
// ------------------------------------------------------------------
__kernel void trace_reduce(
    const int n,
    const int batch,
    __global const float* A,
    __global float* traces,
    __local float* scratch
) {
    const int ib = get_group_id(0);
    const int lid = get_local_id(0);
    if (ib >= batch) return;
    float sum = 0.0f;
    __global const float* Ab = A + ib * n * n;
    for (int i = lid; i < n; i += WG_REDUCE) {
        sum += Ab[i * n + i];
    }
    scratch[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = WG_REDUCE >> 1; off > 0; off >>= 1) {
        if (lid < off) scratch[lid] += scratch[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) traces[ib] = scratch[0];
}

// ------------------------------------------------------------------
// idempotency_reduce
//
// Computes the Frobenius norm ||D² − D||_F for each batch element.
// This is the convergence criterion for purification: when D is
// idempotent (D² = D), the density matrix is fully converged.
// One workgroup per batch; tree reduction in local memory.
// ------------------------------------------------------------------
__kernel void idempotency_reduce(
    const int n,
    const int batch,
    __global const float* D,
    __global const float* D2,
    __global float* errs,
    __local float* scratch
) {
    const int ib = get_group_id(0);
    const int lid = get_local_id(0);
    if (ib >= batch) return;
    const int stride = n * n;
    __global const float* Db = D + ib * stride;
    __global const float* D2b = D2 + ib * stride;
    float sum = 0.0f;
    for (int i = lid; i < stride; i += WG_REDUCE) {
        float d = D2b[i] - Db[i];
        sum += d * d;
    }
    scratch[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = WG_REDUCE >> 1; off > 0; off >>= 1) {
        if (lid < off) scratch[lid] += scratch[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) errs[ib] = sqrt(scratch[0]);
}

// ------------------------------------------------------------------
// local_jacobi_blocks
//
// Classical cyclic Jacobi eigendecomposition of small symmetric
// matrices, one matrix per workgroup. The entire m×m matrix and
// eigenvector accumulator are loaded into local memory, then a
// single thread (lid=0) performs cyclic Jacobi sweeps until the
// off-diagonal norm falls below tol or max_sweeps is reached.
//
// Each sweep visits all m(m−1)/2 off-diagonal elements in order,
// applying a Givens rotation that zeros element (p,q). Convergence
// is quadratic: typically 6–12 sweeps suffice.
//
// This is used for the 2T×2T compound block diagonalization in
// block Jacobi, and for direct diagonalization of small fragment
// Hamiltonians (e.g. N≤64). Other threads in the workgroup are
// idle during the sweep but participate in the cooperative load
// and store of data to/from global memory.
// ------------------------------------------------------------------
__kernel void local_jacobi_blocks(
    const int m,
    const int max_sweeps,
    const float tol,
    __global const float* blocks,
    __global float* eigvals,
    __global float* eigvecs,
    __local float* A,
    __local float* V
) {
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int n2 = m * m;
    __global const float* gA = blocks + gid * n2;
    __global float* gV = eigvecs + gid * n2;

    for (int i = lid; i < n2; i += lsz) {
        A[i] = gA[i];
        int r = i / m;
        int c = i - r * m;
        V[i] = (r == c) ? 1.0f : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float off0 = 0.0f;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                if (i != j) off0 += A[i * m + j] * A[i * m + j];
            }
        }
        off0 = sqrt(off0);
        if (off0 < tol) off0 = 1.0f;

        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            for (int p = 0; p < m; ++p) {
                for (int q = p + 1; q < m; ++q) {
                    float apq = A[p * m + q];
                    if (fabs(apq) < tol) continue;
                    float app = A[p * m + p];
                    float aqq = A[q * m + q];
                    float tau = (aqq - app) / (2.0f * apq);
                    float t = (tau >= 0.0f)
                        ? 1.0f / (tau + sqrt(1.0f + tau * tau))
                        : -1.0f / (-tau + sqrt(1.0f + tau * tau));
                    float c = 1.0f / sqrt(1.0f + t * t);
                    float s = t * c;
                    A[p * m + p] = c * c * app - 2.0f * c * s * apq + s * s * aqq;
                    A[q * m + q] = s * s * app + 2.0f * c * s * apq + c * c * aqq;
                    A[p * m + q] = 0.0f;
                    A[q * m + p] = 0.0f;
                    for (int k = 0; k < m; ++k) {
                        if (k != p && k != q) {
                            float akp = A[k * m + p];
                            float akq = A[k * m + q];
                            A[k * m + p] = c * akp - s * akq;
                            A[p * m + k] = A[k * m + p];
                            A[k * m + q] = s * akp + c * akq;
                            A[q * m + k] = A[k * m + q];
                        }
                    }
                    for (int k = 0; k < m; ++k) {
                        float vkp = V[k * m + p];
                        float vkq = V[k * m + q];
                        V[k * m + p] = c * vkp - s * vkq;
                        V[k * m + q] = s * vkp + c * vkq;
                    }
                }
            }
            float off = 0.0f;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    if (i != j) off += A[i * m + j] * A[i * m + j];
                }
            }
            if (sqrt(off) / off0 < tol) break;
        }
        for (int i = 0; i < m; ++i) eigvals[gid * m + i] = A[i * m + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lid; i < n2; i += lsz) gV[i] = V[i];
}

// ------------------------------------------------------------------
// local_jacobi_blocks_parallel
//
// Row-parallel variant of local Jacobi: all threads in the
// workgroup collaborate to apply each Givens rotation. Thread 0
// computes the rotation angle (c, s) and broadcasts it via local
// memory; then each thread updates its assigned row of A and V.
//
// This reduces the per-rotation cost from O(m) serial work to
// O(m / workgroup_size) parallel work, at the cost of barrier
// synchronizations. For m=64 and wg=64, each rotation touches all
// rows in a single step — no inner loop. Better throughput than
// the serial variant when m is large enough to saturate the
// workgroup but still fits in local memory (m ≤ JACOBI_MAX_M).
//
// Based on the block_jacobi_padded pattern from nested_solver.py.
// ------------------------------------------------------------------
__kernel void local_jacobi_blocks_parallel(
    const int m,
    const int max_sweeps,
    const float tol,
    __global const float* blocks,
    __global float* eigvals,
    __global float* eigvecs,
    __local float* A,
    __local float* V,
    __local float* scratch
) {
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int n2 = m * m;
    __global const float* gA = blocks + gid * n2;
    __global float* gV = eigvecs + gid * n2;

    for (int i = lid; i < n2; i += lsz) {
        A[i] = gA[i];
        int r = i / m;
        int c = i - r * m;
        V[i] = (r == c) ? 1.0f : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float off_part = 0.0f;
    for (int i = lid; i < n2; i += lsz) {
        int r = i / m;
        int c = i - r * m;
        if (r != c) off_part += A[i] * A[i];
    }
    scratch[lid] = off_part;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) scratch[lid] += scratch[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float off0 = sqrt(fmax(scratch[0], tol * tol));
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        for (int p = 0; p < m; ++p) {
            for (int q = p + 1; q < m; ++q) {
                if (lid == 0) {
                    float apq = A[p * m + q];
                    if (fabs(apq) < tol) {
                        scratch[0] = 1.0f;
                        scratch[1] = 0.0f;
                    } else {
                        float app = A[p * m + p];
                        float aqq = A[q * m + q];
                        float tau = (aqq - app) / (2.0f * apq);
                        float t = (tau >= 0.0f)
                            ? 1.0f / (tau + sqrt(1.0f + tau * tau))
                            : -1.0f / (-tau + sqrt(1.0f + tau * tau));
                        float c = 1.0f / sqrt(1.0f + t * t);
                        float s = t * c;
                        scratch[0] = c;
                        scratch[1] = s;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                float c = scratch[0];
                float s = scratch[1];

                for (int k = lid; k < m; k += lsz) {
                    if (k != p && k != q) {
                        float akp = A[k * m + p];
                        float akq = A[k * m + q];
                        float npv = c * akp - s * akq;
                        float nqv = s * akp + c * akq;
                        A[k * m + p] = npv;
                        A[p * m + k] = npv;
                        A[k * m + q] = nqv;
                        A[q * m + k] = nqv;
                    }
                    float vkp = V[k * m + p];
                    float vkq = V[k * m + q];
                    V[k * m + p] = c * vkp - s * vkq;
                    V[k * m + q] = s * vkp + c * vkq;
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if (lid == 0) {
                    float app = A[p * m + p];
                    float aqq = A[q * m + q];
                    float apq = A[p * m + q];
                    A[p * m + p] = c * c * app - 2.0f * c * s * apq + s * s * aqq;
                    A[q * m + q] = s * s * app + 2.0f * c * s * apq + c * c * aqq;
                    A[p * m + q] = 0.0f;
                    A[q * m + p] = 0.0f;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }

        off_part = 0.0f;
        for (int i = lid; i < n2; i += lsz) {
            int r = i / m;
            int c = i - r * m;
            if (r != c) off_part += A[i] * A[i];
        }
        scratch[lid] = off_part;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int off = lsz >> 1; off > 0; off >>= 1) {
            if (lid < off) scratch[lid] += scratch[lid + off];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (sqrt(scratch[0]) / off0 < tol) break;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = lid; i < m; i += lsz) eigvals[gid * m + i] = A[i * m + i];
    for (int i = lid; i < n2; i += lsz) gV[i] = V[i];
}
