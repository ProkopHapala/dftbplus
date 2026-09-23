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
//   fermi_occ_batched             — device-side Fermi μ bisection + occ_w (W5)

#pragma OPENCL EXTENSION cl_khr_fp64 : enable   // f64 for bisection/DIIS decisions
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
// Shared tile body — Ab/Bb/Cb already offset to this workgroup's batch.
static void batched_gemm_core(
    const int n,
    const int trans_a,
    const int trans_b,
    const float alpha,
    const float beta,
    __global const float* Ab,
    __global const float* Bb,
    __global float* Cb,
    __local float* As,
    __local float* Bs
) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int row = get_group_id(0) * TILE_M + ly;
    const int col = get_group_id(1) * TILE_N + lx;

    // W2 (manifest §14): 4 independent f32 FMA accumulators — shorter
    // dependency chain AND better accuracy than one serial sum; Kahan here
    // serialized every multiply and didn't move the measured error.
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    const int lid = ly * TILE_N + lx;
    const int wg = TILE_M * TILE_N;

    // Local-tile layouts follow the transpose flags so GLOBAL reads are
    // always coalesced (consecutive lanes → consecutive addresses):
    //   A nn : As[rr*TILE_K + kk]      trans: As[kk*TILE_M + rr]
    //   B nn : Bs[kk*TILE_N + cc]      trans: Bs[cc*LDB + kk], LDB=TILE_K+1
    const int LDB = TILE_K + 1;
    // Compute-side index: element k of this thread's dot is
    //   As[abase + k*astr] · Bs[bbase + k*bstr]
    const int abase = trans_a ? ly : ly * TILE_K;
    const int astr  = trans_a ? TILE_M : 1;
    const int bbase = trans_b ? lx * LDB : lx;
    const int bstr  = trans_b ? 1 : TILE_N;

    for (int k0 = 0; k0 < n; k0 += TILE_K) {
        if (trans_a) {
            // consecutive t → consecutive ar → coalesced stride-1 reads
            for (int t = lid; t < TILE_M * TILE_K; t += wg) {
                int kk = t / TILE_M, rr = t - kk * TILE_M;
                int ar = get_group_id(0) * TILE_M + rr, ac = k0 + kk;
                As[kk * TILE_M + rr] = (ar < n && ac < n) ? Ab[ac * n + ar] : 0.0f;
            }
        } else {
            for (int t = lid; t < TILE_M * TILE_K; t += wg) {
                int rr = t / TILE_K, kk = t - rr * TILE_K;
                int ar = get_group_id(0) * TILE_M + rr, ac = k0 + kk;
                As[rr * TILE_K + kk] = (ar < n && ac < n) ? Ab[ar * n + ac] : 0.0f;
            }
        }
        if (trans_b) {
            for (int t = lid; t < TILE_K * TILE_N; t += wg) {
                int cc = t / TILE_K, kk = t - cc * TILE_K;
                int br = k0 + kk, bc = get_group_id(1) * TILE_N + cc;
                Bs[cc * LDB + kk] = (br < n && bc < n) ? Bb[bc * n + br] : 0.0f;
            }
        } else {
            for (int t = lid; t < TILE_K * TILE_N; t += wg) {
                int kk = t / TILE_N, cc = t - kk * TILE_N;
                int br = k0 + kk, bc = get_group_id(1) * TILE_N + cc;
                Bs[kk * TILE_N + cc] = (br < n && bc < n) ? Bb[br * n + bc] : 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (row < n && col < n) {
            int kk = 0;
            for (; kk + 4 <= TILE_K; kk += 4) {
                s0 = fma(As[abase + kk       * astr], Bs[bbase + kk       * bstr], s0);
                s1 = fma(As[abase + (kk + 1) * astr], Bs[bbase + (kk + 1) * bstr], s1);
                s2 = fma(As[abase + (kk + 2) * astr], Bs[bbase + (kk + 2) * bstr], s2);
                s3 = fma(As[abase + (kk + 3) * astr], Bs[bbase + (kk + 3) * bstr], s3);
            }
            for (; kk < TILE_K; ++kk) {
                s0 = fma(As[abase + kk * astr], Bs[bbase + kk * bstr], s0);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < n && col < n) {
        int idx = row * n + col;
        Cb[idx] = alpha * ((s0 + s1) + (s2 + s3)) + beta * Cb[idx];
    }
}

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
    __local float* Bs,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int ib = work_ids[get_group_id(2)];
    if (ib >= batch) return;
    const int stride = n * n;
    batched_gemm_core(n, trans_a, trans_b, alpha, beta,
        A + ib * stride, B + ib * stride, C + ib * stride, As, Bs);
}

// W4 (manifest §14): SCC-path variant — replicas with active[ib]==0 exit
// before loading any tile, so a converged replica stops paying for GEMM
// work inside an un-interrupted chunk. Identical output to batched_gemm.
__kernel void batched_gemm_active(
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
    __local float* Bs,
    __global const int* active,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int ib = work_ids[get_group_id(2)];
    if (ib >= batch || active[ib] == 0) return;
    const int stride = n * n;
    batched_gemm_core(n, trans_a, trans_b, alpha, beta,
        A + ib * stride, B + ib * stride, C + ib * stride, As, Bs);
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

// ==================================================================
// Agent_6 (Wave 2): Full-local batched GEMM + SCC component kernels.
//
// All kernels below operate on batched row-major data, one workgroup
// per system (batch element). They use the shared GpuRuntime context
// and are independent of the Jacobi eigensolver (Agent_4).
//
// New kernel inventory:
//   matmul_full_local_batched   — C = A·B, both matrices in __local
//   gamma_matvec_batched        — V = G · Δq (atom-resolved electrostatics)
//   h_scc_update_batched        — H = H0 + 0.5·S·(V_i + V_j)
//   mulliken_charges_batched    — q_A = Σ_{μ∈A} (D·S)_μμ
//   residual_and_mix_batched    — rms = ||q_new-q_old||, q_mixed = α·q_new+(1-α)·q_old
// ==================================================================

#ifndef FL_NORB
#define FL_NORB 64
#endif

#ifndef FL_WG
#define FL_WG 256
#endif

// ------------------------------------------------------------------
// matmul_full_local_batched
//
// Full-local batched matrix multiply: C_b = A_b · B_b for each batch
// element b. One workgroup per system; both A and B are loaded once
// into __local memory with a padded leading dimension (N+1) to avoid
// bank conflicts, then each thread computes N²/WG output elements in
// a strided loop.
//
// Specialized at compile time via FL_NORB (max N, sizes the static
// __local arrays) and FL_WG (workgroup size). The runtime `n` arg
// may be ≤ FL_NORB. The host renders a distinct program per (N, WG)
// pair; GpuRuntime's program cache avoids recompilation for repeats.
//
// Local memory: 2 · FL_NORB · (FL_NORB+1) · 4 B. For FL_NORB=64 that
// is 33 KB, within the 48 KB typical local-memory limit.
// ------------------------------------------------------------------
__kernel void matmul_full_local_batched(
    const int n,
    const int batch,
    __global const float* A,
    __global const float* B,
    __global float* C,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch) return;

    const int stride = n * n;
    __global const float* Ab = A + (size_t)sid * stride;
    __global const float* Bb = B + (size_t)sid * stride;
    __global float* Cb       = C + (size_t)sid * stride;

    __local float LA[FL_NORB * (FL_NORB + 1)];
    __local float LB[FL_NORB * (FL_NORB + 1)];
    const int ld = n + 1;  // padded leading dimension

    // Cooperative load of A and B into local memory (padded ld).
    for (int i = lid; i < n * n; i += lsz) {
        int r = i / n;
        int c = i - r * n;
        LA[r * ld + c] = Ab[r * n + c];
        LB[r * ld + c] = Bb[r * n + c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Strided output: each thread computes N²/WG elements.
    // W2: 4 independent f32 FMA accumulators — shorter dependency chain AND
    // better accuracy than one serial sum (no Kahan: it serialized every FMA).
    for (int idx = lid; idx < n * n; idx += lsz) {
        int r = idx / n;
        int c = idx - r * n;
        const __local float* Ar = LA + r * ld;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int k = 0;
        for (; k + 4 <= n; k += 4) {
            s0 = fma(Ar[k],     LB[k       * ld + c], s0);
            s1 = fma(Ar[k + 1], LB[(k + 1) * ld + c], s1);
            s2 = fma(Ar[k + 2], LB[(k + 2) * ld + c], s2);
            s3 = fma(Ar[k + 3], LB[(k + 3) * ld + c], s3);
        }
        for (; k < n; ++k) s0 = fma(Ar[k], LB[k * ld + c], s0);
        Cb[r * n + c] = (s0 + s1) + (s2 + s3);
    }
}

// ------------------------------------------------------------------
// gamma_matvec_batched
//
// Atom-resolved electrostatic potential: V_A = Σ_B G_AB · Δq_B.
// G is a dense [n_atoms × n_atoms] gamma matrix per system, Δq and V
// are per-atom vectors. One workgroup per system; each thread computes
// one V_A (threads with lid >= n_atoms are idle). Naive reduction over
// B — fine since n_atoms ≤ ~100.
//
// Layout: G[b][Na*Na] row-major, dq[b][Na], V[b][Na].
// ------------------------------------------------------------------
__kernel void gamma_matvec_batched(
    const int n_atoms,
    const int batch,
    __global const float* G,
    __global const float* dq,
    __global float* V,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    if (sid >= batch) return;
    __global const float* Gb  = G  + (size_t)sid * n_atoms * n_atoms;
    __global const float* dqb = dq + (size_t)sid * n_atoms;
    __global float* Vb        = V  + (size_t)sid * n_atoms;

    if (lid < n_atoms) {
        float sum = 0.0f;
        for (int b = 0; b < n_atoms; ++b) {
            sum += Gb[lid * n_atoms + b] * dqb[b];
        }
        Vb[lid] = sum;
    }
}

// ------------------------------------------------------------------
// h_scc_update_batched
//
// SCC Hamiltonian update: H[μ,ν] = H0[μ,ν] + 0.5·S[μ,ν]·(V[A_μ] + V[A_ν])
// where A_μ is the atom owning orbital μ (from orb_atom[μ]). V is the
// per-atom potential vector. Elementwise over N² orbitals.
//
// V_atom is cached in __local once (cooperative load), then each thread
// updates strided (μ,ν) elements. One workgroup per system.
//
// Layout: H0,S,H [b][N*N] row-major; V [b][Na]; orb_atom [b][N] (int32).
// ------------------------------------------------------------------
__kernel void h_scc_update_batched(
    const int n,
    const int n_atoms,
    const int batch,
    __global const float* H0,
    __global const float* S,
    __global const float* V,
    __global const int* orb_atom,
    __global float* H,
    __local float* Vloc,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch) return;
    __global const float* H0b = H0 + (size_t)sid * n * n;
    __global const float* Sb  = S  + (size_t)sid * n * n;
    __global const float* Vb  = V  + (size_t)sid * n_atoms;
    __global const int* oa    = orb_atom + (size_t)sid * n;
    __global float* Hb        = H  + (size_t)sid * n * n;

    // Cache per-atom potential in local memory.
    for (int a = lid; a < n_atoms; a += lsz) Vloc[a] = Vb[a];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Elementwise SCC update over N².
    for (int idx = lid; idx < n * n; idx += lsz) {
        int i = idx / n;
        int j = idx - i * n;
        float vi = Vloc[oa[i]];
        float vj = Vloc[oa[j]];
        Hb[idx] = H0b[idx] + 0.5f * Sb[idx] * (vi + vj);
    }
}

// ------------------------------------------------------------------
// mulliken_charges_batched
//
// Per-atom Mulliken population: q_A = Σ_{μ∈A} (D·S)_μμ, where
// (D·S)_μμ = Σ_ν D[μ,ν]·S[ν,μ] = Σ_ν D[μ,ν]·S[μ,ν] (S symmetric).
//
// Two phases, one workgroup per system:
//   1. Each thread computes strided diag_ds[μ] into __local `diag`.
//   2. Each thread (one per atom) sums diag[μ] for μ belonging to its
//      atom (via orb_atom[μ]).
//
// Layout: D,S [b][N*N] row-major; orb_atom [b][N] (int32); q [b][Na].
// ------------------------------------------------------------------
__kernel void mulliken_charges_batched(
    const int n,
    const int n_atoms,
    const int batch,
    __global const float* D,
    __global const float* S,
    __global const int* orb_atom,
    __global float* q,
    __local float* diag,
    __global const int* active,     // [batch] 0 → replica frozen, early-out
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;
    __global const float* Db = D + (size_t)sid * n * n;
    __global const float* Sb = S + (size_t)sid * n * n;
    __global const int* oa   = orb_atom + (size_t)sid * n;
    __global float* qb       = q + (size_t)sid * n_atoms;

    // Phase 1: diagonal of D·S per orbital.
    for (int mu = lid; mu < n; mu += lsz) {
        float s = 0.0f;
        for (int nu = 0; nu < n; ++nu) {
            s += Db[mu * n + nu] * Sb[mu * n + nu];
        }
        diag[mu] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: per-atom reduction.
    if (lid < n_atoms) {
        float sum = 0.0f;
        for (int mu = 0; mu < n; ++mu) {
            if (oa[mu] == lid) sum += diag[mu];
        }
        qb[lid] = sum;
    }
}

// ------------------------------------------------------------------
// residual_and_mix_batched
//
// Simple mixing step + residual norm:
//   q_mixed = α·q_new + (1-α)·q_old
//   rms     = √(Σ (q_new-q_old)² / n_atoms)  — RMS, same contract as CPU/DIIS
//
// One workgroup per system; tree reduction in __local for the rms.
// ------------------------------------------------------------------
__kernel void residual_and_mix_batched(
    const int n_atoms,
    const int batch,
    const float alpha,
    __global const float* q_new,
    __global const float* q_old,
    __global float* q_mixed,
    __global float* rms,
    __global const int* active,   // [batch] 0 → replica frozen, early-out
    __local float* scratch,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;
    __global const float* qn = q_new + (size_t)sid * n_atoms;
    __global const float* qo = q_old + (size_t)sid * n_atoms;
    __global float* qm       = q_mixed + (size_t)sid * n_atoms;

    float partial = 0.0f;
    for (int a = lid; a < n_atoms; a += lsz) {
        float d = qn[a] - qo[a];
        qm[a] = alpha * qn[a] + (1.0f - alpha) * qo[a];
        partial += d * d;
    }
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) scratch[lid] += scratch[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) rms[sid] = sqrt(scratch[0] / (float)n_atoms);
}

// ------------------------------------------------------------------
// commit_q_batched   (commit-model SCC)
//
// q[sid][a] = q_next[sid][a] for active replicas; done/failed replicas
// keep q (the just-solved input) so the device electronic state stays
// consistent with q. One WG/system, threads stride over atoms.
// ------------------------------------------------------------------
__kernel void commit_q_batched(
    const int n_atoms,
    const int batch,
    __global const float* q_next,
    __global float* q,
    __global const int* active,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;
    __global const float* qn = q_next + (size_t)sid * n_atoms;
    __global float* qb = q + (size_t)sid * n_atoms;
    for (int a = lid; a < n_atoms; a += lsz) qb[a] = qn[a];
}

// ------------------------------------------------------------------
// fused_dq_v_hscc_batched   (manifest §12 D10 / R18)
//
// Fuses Δq = q−q0 → V = γ·Δq → H_scc = H0 + ½S·(V_A+V_B) into one
// WG/system launch. Δq/V live in local memory and are also written to
// global (the energy dot ½Δq·V reads them). Replaces 3 launches and the
// intermediate dq/v global round-trip. γ rows stream from global;
// orb_atom is read per element (L2-cached, small).
// ------------------------------------------------------------------
__kernel void fused_dq_v_hscc_batched(
    const int n,
    const int n_atoms,
    const int batch,
    __global const float* q,        // [batch*n_atoms]
    __global const float* q0,       // [batch*n_atoms]
    __global const float* G,        // [batch*n_atoms*n_atoms]
    __global const float* H0,       // [batch*n*n]
    __global const float* S,        // [batch*n*n]
    __global const int* orb_atom,   // [batch*n]
    __global float* dq,             // [batch*n_atoms] out (energy dot)
    __global float* V,              // [batch*n_atoms] out
    __global float* H,              // [batch*n*n] out
    __local float* ldq,             // [n_atoms] local Δq
    __local float* lv,              // [n_atoms] local V
    __global const int* active,     // [batch] 0 → replica frozen, early-out
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;
    __global const float* qb  = q  + (size_t)sid * n_atoms;
    __global const float* q0b = q0 + (size_t)sid * n_atoms;
    __global const float* Gb  = G  + (size_t)sid * n_atoms * n_atoms;
    __global const float* H0b = H0 + (size_t)sid * n * n;
    __global const float* Sb  = S  + (size_t)sid * n * n;
    __global const int* oa    = orb_atom + (size_t)sid * n;
    __global float* dqb       = dq + (size_t)sid * n_atoms;
    __global float* Vb        = V  + (size_t)sid * n_atoms;
    __global float* Hb        = H  + (size_t)sid * n * n;

    for (int a = lid; a < n_atoms; a += lsz) {
        const float d = qb[a] - q0b[a];
        ldq[a] = d;
        dqb[a] = d;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int a = lid; a < n_atoms; a += lsz) {
        __global const float* row = Gb + (size_t)a * n_atoms;
        float sum = 0.0f;
        for (int b = 0; b < n_atoms; ++b) sum = fma(row[b], ldq[b], sum);
        lv[a] = sum;
        Vb[a] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int nn = n * n;
    for (int idx = lid; idx < nn; idx += lsz) {
        const int i = idx / n;
        const int j = idx - i * n;
        Hb[idx] = H0b[idx] + 0.5f * Sb[idx] * (lv[oa[i]] + lv[oa[j]]);
    }
}

// ------------------------------------------------------------------
// transpose_batched   (D6)
//
// at[b][j][i] = a[b][i][j] — maintain Xᵀ so H' = Xᵀ·H·X stays correct
// for ANY orthonormalizer gauge (Newton-reused X is S^{-1/2}·U, not
// symmetric). One workgroup per system.
// ------------------------------------------------------------------
__kernel void transpose_batched(
    const int n,
    const int batch,
    __global const float* a,
    __global float* at,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch) return;
    __global const float* ab = a + (size_t)sid * n * n;
    __global float* atb = at + (size_t)sid * n * n;
    for (int idx = lid; idx < n * n; idx += lsz) {
        const int i = idx / n;
        const int j = idx - i * n;
        atb[j * n + i] = ab[i * n + j];
    }
}

// ------------------------------------------------------------------
// delta_q_batched
//
// dq[a] = q[a] - q0[a]  per atom, per system.
// One workgroup per system, threads stride over atoms.
// ------------------------------------------------------------------
__kernel void delta_q_batched(
    const int n_atoms,
    const int batch,
    __global const float* q,
    __global const float* q0,
    __global float* dq,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch) return;
    __global const float* qb  = q  + (size_t)sid * n_atoms;
    __global const float* q0b = q0 + (size_t)sid * n_atoms;
    __global float* dqb       = dq + (size_t)sid * n_atoms;
    for (int a = lid; a < n_atoms; a += lsz) {
        dqb[a] = qb[a] - q0b[a];
    }
}

// ------------------------------------------------------------------
// build_density_masked_batched
//
// D = 2 * sum_{k: occ[k]!=0} s_k * C[:,k] * C[:,k]^T
// s_k = 1        when use_eig==0  → closed-shell density D
// s_k = eig[k]   when use_eig==1  → energy-weighted density W
// One kernel for both (no extra program). use_eig is uniform across the WG.
// ------------------------------------------------------------------
__kernel void build_density_masked_batched(
    const int n,
    const int batch,
    __global const float* C,
    __global const int* occ_mask,
    __global float* D,
    const int use_eig,
    __global const float* eig,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch) return;
    __global const float* Cb = C + (size_t)sid * n * n;
    __global const int* mb   = occ_mask + (size_t)sid * n;
    __global float* Db       = D + (size_t)sid * n * n;
    __global const float* eb = eig + (size_t)sid * n;
    const int nn = n * n;
    for (int idx = lid; idx < nn; idx += lsz) {
        const int i = idx / n;
        const int j = idx - i * n;
        float s = 0.0f;
        for (int k = 0; k < n; ++k) {
            float sk = use_eig ? eb[k] : 1.0f;
            s += (float)mb[k] * sk * Cb[i * n + k] * Cb[j * n + k];
        }
        Db[idx] = 2.0f * s;
    }
}

// ------------------------------------------------------------------
// build_density_occ_batched   (manifest §12 D10)
//
// D = 2 * Σ_{t<n_occ} s_t · C[:,k_t] · C[:,k_t]^T
//   s_t = 1 → density D (use_eig=0);  s_t = eig[k_t] → W (use_eig=1).
// Occupied-index list instead of an all-N×mask scan (49 vs 87 columns for
// AT), lower triangle only (i ≥ j) mirrored, occupied weights/indices staged
// in local memory, inner loop = 4 independent FP32 FMA accumulators.
// One workgroup per system.
// ------------------------------------------------------------------
__kernel void build_density_occ_batched(
    const int n,
    const int batch,
    const int n_occ,
    __global const float* C,
    __global const int* occ_idx,
    __global float* D,
    const int use_eig,
    __global const float* eig,
    const int use_w,             // 1 → multiply by occ_w[k] (Fermi smearing)
    __global const float* occ_w, // [batch*n] per-orbital weights f_k
    __local float* lw,      // [n_occ] occupied weights
    __local int*   loi,     // [n_occ] occupied column indices
    __global const int* active,     // [batch] 0 → replica frozen, early-out
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;
    __global const float* Cb = C + (size_t)sid * n * n;
    __global float* Db = D + (size_t)sid * n * n;
    __global const float* eb = eig + (size_t)sid * n;
    __global const int* oib = occ_idx + (size_t)sid * n;
    __global const float* owb = occ_w + (size_t)sid * n;

    for (int t = lid; t < n_occ; t += lsz) {
        const int k = use_w ? t : oib[t];   // smearing: all orbitals, weighted
        loi[t] = k;
        lw[t] = (use_eig ? eb[k] : 1.0f) * (use_w ? owb[k] : 1.0f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int nt = n * (n + 1) / 2;
    for (int idx = lid; idx < nt; idx += lsz) {
        // triangle index → (i,j), i≥j; float sqrt estimate + exact adjust
        int i = (int)(0.5f * (sqrt(8.0f * (float)idx + 1.0f) - 1.0f));
        while ((i + 1) * (i + 2) / 2 <= idx) i++;
        while (i * (i + 1) / 2 > idx) i--;
        const int j = idx - i * (i + 1) / 2;
        const int in = i * n;
        const int jn = j * n;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int t = 0;
        for (; t + 4 <= n_occ; t += 4) {
            s0 = fma(lw[t    ] * Cb[in + loi[t    ]], Cb[jn + loi[t    ]], s0);
            s1 = fma(lw[t + 1] * Cb[in + loi[t + 1]], Cb[jn + loi[t + 1]], s1);
            s2 = fma(lw[t + 2] * Cb[in + loi[t + 2]], Cb[jn + loi[t + 2]], s2);
            s3 = fma(lw[t + 3] * Cb[in + loi[t + 3]], Cb[jn + loi[t + 3]], s3);
        }
        for (; t < n_occ; ++t) s0 = fma(lw[t] * Cb[in + loi[t]], Cb[jn + loi[t]], s0);
        const float s = (s0 + s1) + (s2 + s3);
        Db[in + j] = 2.0f * s;
        Db[jn + i] = 2.0f * s;
    }
}

// ------------------------------------------------------------------
// frobenius_trace_batched
//
// tr[sid] = sum_{i,j} A[i,j] * B[i,j]  (Frobenius inner product).
// For symmetric matrices this equals Tr(A·B).
// One workgroup per system, tree reduction in __local.
// ------------------------------------------------------------------
__kernel void frobenius_trace_batched(
    const int n,
    const int batch,
    __global const float* A,
    __global const float* B,
    __global float* tr,
    __local float* scratch,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch) return;
    __global const float* Ab = A + (size_t)sid * n * n;
    __global const float* Bb = B + (size_t)sid * n * n;
    const int nn = n * n;
    float partial = 0.0f;
    for (int idx = lid; idx < nn; idx += lsz) {
        partial += Ab[idx] * Bb[idx];
    }
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) scratch[lid] += scratch[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) tr[sid] = scratch[0];
}

// ------------------------------------------------------------------
// dot_batched
//
// dot[sid] = sum_a x[a] * y[a]  per system.
// One workgroup per system, tree reduction.
// ------------------------------------------------------------------
__kernel void dot_batched(
    const int n,
    const int batch,
    __global const float* x,
    __global const float* y,
    __global float* dot,
    __local float* scratch,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch) return;
    __global const float* xb = x + (size_t)sid * n;
    __global const float* yb = y + (size_t)sid * n;
    float partial = 0.0f;
    for (int a = lid; a < n; a += lsz) {
        partial += xb[a] * yb[a];
    }
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) scratch[lid] += scratch[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) dot[sid] = scratch[0];
}

// ------------------------------------------------------------------
// extract_diagonal_batched
//
// Extracts the diagonal of a batched N×N matrix: diag[sid*N + i] = A[sid*N*N + i*N + i].
// One thread per (system, orbital) pair. No barriers, no local memory.
// This replaces reading the full N²×batch matrix to host just to get N eigenvalues.
// ------------------------------------------------------------------
__kernel void extract_diagonal_batched(
    const int n,
    const int batch,
    __global const float* a,
    __global float* diag,
    __global const int* active,     // [batch] 0 → replica frozen, early-out
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int gid = get_global_id(0);
    const int total = n * batch;
    if (gid >= total) return;
    const int sid = work_ids[gid / n];
    if (active[sid] == 0) return;
    const int i = gid % n;
    diag[(size_t)sid * n + i] = a[(size_t)sid * n * n + i * n + i];
}

// ------------------------------------------------------------------
// select_occupation_batched
//
// GPU-side occupation selection: sorts eigenvalues per system and marks
// the n_occ lowest as occupied. Eliminates the CPU read → sort → upload
// roundtrip from the SCC hot loop.
//
// One workgroup per system. Bitonic sort of (eigenvalue, index) pairs in
// local memory. Pads to the next power of 2 with (+inf, dummy) sentinels.
//
// Arguments:
//   n        — number of orbitals per system
//   n_occ    — number of occupied orbitals
//   batch    — number of systems
//   eig_diag — [batch][N] eigenvalues (from extract_diagonal_batched)
//   occ_mask — [batch][N] output: 1=occupied, 0=virtual
//   occ_idx  — [batch][N] output: occ_idx[t] = sorted occupied column index
//              for t < n_occ (ascending eigenvalue order); D10 density path
//
// Specialization: OCC_MAX_N must be the next power of 2 ≥ max N.
// ------------------------------------------------------------------
#ifndef OCC_MAX_N
#define OCC_MAX_N 128
#endif

__kernel void select_occupation_batched(
    const int n,
    const int n_occ,
    const int batch,
    __global const float* eig_diag,
    __global int* occ_mask,
    __global int* occ_idx,
    __global const int* active,     // [batch] 0 → replica frozen, early-out
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;

    // Local arrays for bitonic sort: (value, original_index) pairs
    __local float lval[OCC_MAX_N];
    __local int   lidx[OCC_MAX_N];

    // Load eigenvalues into local memory, pad with +inf
    for (int i = lid; i < OCC_MAX_N; i += lsz) {
        if (i < n) {
            lval[i] = eig_diag[(size_t)sid * n + i];
            lidx[i] = i;
        } else {
            lval[i] = 1.0e30f;  // +inf sentinel for padding
            lidx[i] = -1;        // dummy index
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Bitonic sort ascending (OCC_MAX_N must be a power of 2)
    for (int k = 2; k <= OCC_MAX_N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = lid; i < OCC_MAX_N; i += lsz) {
                int ij = i ^ j;
                if (ij > i) {
                    // Compare and swap: ascending sort
                    bool ascending = ((i & k) == 0);
                    float vi = lval[i], vj = lval[ij];
                    bool swap = ascending ? (vj < vi) : (vi < vj);
                    if (swap) {
                        lval[i] = vj; lval[ij] = vi;
                        int ti = lidx[i]; lidx[i] = lidx[ij]; lidx[ij] = ti;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // After sort, lval[0..n] are the n smallest eigenvalues in ascending order.
    // Mark the first n_occ original indices as occupied; also write the
    // occupied index list for the D10 density kernel.
    for (int i = lid; i < n; i += lsz) {
        int orig_idx = lidx[i];
        if (orig_idx >= 0) {
            occ_mask[(size_t)sid * n + orig_idx] = (i < n_occ) ? 1 : 0;
            if (i < n_occ) occ_idx[(size_t)sid * n + i] = orig_idx;
        }
    }
}

// ------------------------------------------------------------------
// fermi_occ_batched   (W5)
//
// Device-side Fermi level + occupation weights: per replica solves
//   Σ_k f_k(μ) = n_occ,  f_k = 1/(1+exp((ε_k−μ)/kT))
// by bracketed bisection on eig_diag (unsorted Jacobi output is fine —
// the sum is order-independent), then writes occ_w[k]=f_k and mu[sid].
// Replaces the per-iteration eig readback + host bisection + occ_w upload.
//
// One workgroup per system. The bisection sum/decision runs in f64
// (discrete-decision rule: decisions in f64; f_k itself is stored f32).
// Eigenvalues are staged in __local le[n]; bisection needs ~40 iterations
// of a workgroup reduction — all inside the one launch, zero host sync.
// ------------------------------------------------------------------
__kernel void fermi_occ_batched(
    const int n,
    const int batch,
    const int n_occ,
    const float kT,
    __global const float* eig_diag,
    __global float* occ_w,
    __global float* mu_out,
    __local float* le,          // [n] staged eigenvalues
    __local double* red,        // [lsz] f64 reduction scratch
    __local double* lohi,       // [4] lo, hi, s_mid, mu
    __global const int* active, // [batch] 0 → replica frozen, early-out
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;
    __global const float* eb = eig_diag + (size_t)sid * n;
    __global float* ob = occ_w + (size_t)sid * n;
    const double kt = (double)kT;

    for (int k = lid; k < n; k += lsz) le[k] = eb[k];
    barrier(CLK_LOCAL_MEM_FENCE);

    // bracket: [min−32kT, max+32kT] — f_k≈0/1 beyond, same as the host path
    double lo = 1.0e300, hi = -1.0e300;
    for (int k = lid; k < n; k += lsz) {
        const double e = (double)le[k];
        lo = fmin(lo, e); hi = fmax(hi, e);
    }
    red[lid] = lo;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] = fmin(red[lid], red[lid + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const double lo_b = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    red[lid] = hi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] = fmax(red[lid], red[lid + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        lohi[0] = lo_b - 32.0 * kt;
        lohi[1] = red[0] + 32.0 * kt;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // bisection: s(μ) decreasing in −μ... s(μ)=Σf is INCREASING in μ:
    // s>n_occ → μ too high → hi=mid; else lo=mid.
    for (int it = 0; it < 40; ++it) {
        const double mid = 0.5 * (lohi[0] + lohi[1]);
        double s = 0.0;
        for (int k = lid; k < n; k += lsz) {
            const double x = ((double)le[k] - mid) / kt;
            s += (x > 40.0) ? 0.0 : ((x < -40.0) ? 1.0 : 1.0 / (1.0 + exp(x)));
        }
        red[lid] = s;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int off = lsz >> 1; off > 0; off >>= 1) {
            if (lid < off) red[lid] += red[lid + off];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            if (red[0] > (double)n_occ) { lohi[1] = mid; } else { lohi[0] = mid; }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const double mu = 0.5 * (lohi[0] + lohi[1]);
    if (lid == 0) mu_out[sid] = (float)mu;
    for (int k = lid; k < n; k += lsz) {
        const double x = ((double)le[k] - mu) / kt;
        ob[k] = (float)((x > 40.0) ? 0.0 : ((x < -40.0) ? 1.0 : 1.0 / (1.0 + exp(x))));
    }
}

// ------------------------------------------------------------------
// repulsive_energy_batched
//
// Computes the repulsive pair-potential energy per system:
//   E_rep[sid] = Σ_{i<j} E_rep^{s_i,s_j}(|r_i - r_j|)
//
// One workgroup per system. Each thread handles a subset of atom pairs.
// Local reduction sums the per-thread contributions.
//
// Spline data layout (flat buffer, f32):
//   For each species pair p (at offset spline_offsets[p]):
//     n_intervals  (1 i32)  — number of spline intervals
//     cutoff       (1 f32)  — cutoff distance
//     exp_coeffs   (3 f32)  — (a, b, c) for exponential head
//     x_start      (REP_MAX_INTERVALS f32) — interval start points
//     sp_coeffs    ((REP_MAX_INTERVALS-1)*4 f32) — cubic coefficients per interval
//     sp_last_coeffs (6 f32) — polynomial tail coefficients
//
// Arguments:
//   n_atoms     — atoms per system
//   batch       — number of systems
//   coords      — [batch][n_atoms*3] atom coordinates (Bohr)
//   species_idx — [batch][n_atoms] species index per atom
//   spline_offsets — [n_species*n_species] offset into spline_data (-1 = no spline)
//   n_species   — number of species
//   spline_data — flat buffer with all spline coefficients
//   e_rep       — [batch] output repulsive energy per system
//
// Specialization: REP_MAX_INTERVALS must be set to the max number of
// spline intervals across all species pairs.
// ------------------------------------------------------------------
#ifndef REP_MAX_INTERVALS
#define REP_MAX_INTERVALS 30
#endif

__kernel void repulsive_energy_batched(
    const int n_atoms,
    const int batch,
    __global const float* coords,        // [batch][n_atoms*3]
    __global const int* species_idx,      // [batch][n_atoms]
    __global const int* spline_offsets,  // [n_species*n_species]
    const int n_species,
    __global const float* spline_data,   // flat
    __global float* e_rep,               // [batch]
    __global const int* work_ids         // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch) return;

    __global const float* crd = coords + (size_t)sid * n_atoms * 3;
    __global const int* spc = species_idx + (size_t)sid * n_atoms;

    // Each thread accumulates its partial energy, then reduce.
    float my_e = 0.0f;
    const int npairs = n_atoms * (n_atoms - 1) / 2;

    // Linear mapping: pair_id → (i, j) with i < j
    // pair_id = i*(2*n_atoms - i - 1)/2 + (j - i - 1)
    // We iterate i from 0 to n_atoms-1, j from i+1 to n_atoms-1
    for (int pair_id = lid; pair_id < npairs; pair_id += lsz) {
        // Find (i, j) from pair_id — linear search (n_atoms is small, ≤ ~30)
        int i = 0, j = 0, acc = 0, found = 0;
        for (int ii = 0; ii < n_atoms - 1 && !found; ++ii) {
            int n_j = n_atoms - 1 - ii;
            if (acc + n_j > pair_id) {
                i = ii;
                j = ii + 1 + (pair_id - acc);
                found = 1;
            }
            acc += n_j;
        }
        if (!found) continue;

        int si = spc[i], sj = spc[j];
        int off_idx = si * n_species + sj;
        int offset = spline_offsets[off_idx];
        if (offset < 0) continue;  // no spline for this pair

        // Compute distance
        float dx = crd[j*3+0] - crd[i*3+0];
        float dy = crd[j*3+1] - crd[i*3+1];
        float dz = crd[j*3+2] - crd[i*3+2];
        float r = sqrt(dx*dx + dy*dy + dz*dz);

        // Load spline data
        __global const float* sd = spline_data + offset;
        int n_int = as_int(sd[0]);
        float cutoff = sd[1];
        if (r >= cutoff || r < 1.0e-6f) continue;

        float exp_a = sd[2], exp_b = sd[3], exp_c = sd[4];
        __global const float* x_start = sd + 5;
        __global const float* sp_coeffs = sd + 5 + REP_MAX_INTERVALS;
        __global const float* sp_last = sd + 5 + REP_MAX_INTERVALS + (REP_MAX_INTERVALS - 1) * 4;

        float e_val = 0.0f;

        if (r < x_start[0]) {
            // Exponential head: E = exp(-a*r + b) + c
            e_val = exp(-exp_a * r + exp_b) + exp_c;
        } else if (r >= x_start[n_int - 1]) {
            // Polynomial tail: E = c0 + c1*dr + ... + c5*dr^5
            float dr = r - x_start[n_int - 1];
            float xh = dr;
            e_val = sp_last[0];
            for (int k = 1; k < 6; ++k) {
                e_val += sp_last[k] * xh;
                xh *= dr;
            }
        } else {
            // Bisection: find interval i such that x_start[i] <= r < x_start[i+1]
            int lo = 0, hi = n_int - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (x_start[mid] <= r) lo = mid; else hi = mid;
            }
            float dr = r - x_start[lo];
            // Cubic: E = c0 + c1*dr + c2*dr^2 + c3*dr^3
            __global const float* c = sp_coeffs + lo * 4;
            e_val = c[0] + c[1]*dr + c[2]*dr*dr + c[3]*dr*dr*dr;
        }

        my_e += e_val;
    }

    // Local reduction
    __local float reduce[256];
    reduce[lid] = my_e;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) reduce[lid] += reduce[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) e_rep[sid] = reduce[0];
}

// ------------------------------------------------------------------
// diis_step_batched
//
// GPU-resident DIIS mixing. One workgroup per system.
//
//   1. residual = q_new - q_old; store (Δq_in = q_in−q0, res) in the ring buffer
//   2. rms = √(Σ res² / n_atoms)   — RMS, not L2; matches CPU SCC tol
//   3. n < 2 after this store: α-mix (never 1-vector DIIS; that is undamped q_new)
//   4. n ≥ 2: Gram in f64, scale by max|B_ij|, tiny GE in f64
//   5. On rank failure: drop the OLDEST history vector and retry n−1 once;
//      only then α-mix. Fallbacks are recorded in diis_flag/diis_reason —
//      no printf in production kernels (manifest §12 D9).
//   6. Mix in Δq = q − q0 space: q_next = q0 + Σc_p·(Δq_p + r_p). The
//      Σc_p=1 constraint is then algebraically exact — f32 coefficient
//      rounding can no longer inject the large neutral population q0.
//
// Buffers (all per-system, indexed by sid * stride + ...):
//   q_new       — [batch*n_atoms] current SCC output charges
//   q_old       — [batch*n_atoms] current input charges (overwritten with mixed)
//   q0          — [batch*n_atoms] neutral atom charges (Δq anchor)
//   dq_hist     — [batch*max_hist*n_atoms] ring buffer of Δq_in vectors
//   r_hist      — [batch*max_hist*n_atoms] ring buffer of residual vectors
//   buf_idx     — [batch] ring buffer write position (i32)
//   n_filled    — [batch] number of valid history entries (i32)
//   coeffs      — [batch*max_hist] DIIS coefficients output
//   diis_flag   — [batch] fallback counter (host reads after solve; reset_diis clears)
//   diis_reason — [batch] last fallback reason (1=pivot/scale 2=nonfinite 3=sum(c)!=1)
//   rms         — [batch] residual RMS output
//
// Specialization: DIIS_MAX_HIST must be set (default 10).
// ------------------------------------------------------------------
#ifndef DIIS_MAX_HIST
#define DIIS_MAX_HIST 10
#endif
#define DIIS_NP1 (DIIS_MAX_HIST + 1)

__kernel void diis_step_batched(
    const int n_atoms,
    const int batch,
    const float alpha,           // simple mixing fallback parameter
    __global const float* q_new, // [batch*n_atoms]
    __global const float* q_old, // [batch*n_atoms] — current input (read-only)
    __global const float* q0,    // [batch*n_atoms] neutral reference (Δq anchor)
    __global float* q_next,      // [batch*n_atoms] mixed next iterate — R7: bound to
                                 //   q_gpu so the kernel commits in place (converged/
                                 //   invalid replicas return before the mix → keep q_n)
    __global float* dq_hist,     // [batch*DIIS_MAX_HIST*n_atoms]
    __global float* r_hist,      // [batch*DIIS_MAX_HIST*n_atoms]
    __global int* buf_idx,       // [batch]
    __global int* n_filled,      // [batch]
    __global float* coeffs,      // [batch*DIIS_MAX_HIST]
    __global int* diis_flag,     // [batch] fallback counter
    __global int* diis_reason,   // [batch] last fallback reason
    __global float* rms,         // [batch]
    __global int* active,        // [batch] 0 → replica frozen; W4: also WRITTEN —
                                 // rms < rms_tol or non-finite → active = 0 (device-side
                                 // convergence so the host can run chunked iterations)
    const float rms_tol,         // SCC convergence tolerance (f32 copy of the host tol)
    __local float* scratch,      // workgroup scratch (≥ lsz)
    __local double* lW,          // [DIIS_MAX_HIST*n_atoms] W13: f64 QR working
                                 //   columns — LOCAL since 2026-09-18: the QR
                                 //   below runs on thread 0, and ~10K serial
                                 //   f64 GLOBAL accesses was the #2 kernel cost
                                 //   (0.33 ms/call, 15.6% of SCC dev time)
    __global const int* work_ids // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __local int l_diis_ok;
    // f64 QR solve scratch — only thread 0 touches these; __local keeps the
    // serial Gram–Schmidt at local latency instead of global.
    __local double l_Rq[DIIS_MAX_HIST * DIIS_MAX_HIST];
    __local double l_Rs[DIIS_MAX_HIST * DIIS_MAX_HIST];
    __local double l_diag[DIIS_MAX_HIST];
    __local double l_cx[DIIS_MAX_HIST];
    __local double l_yy[DIIS_MAX_HIST];
    __local double l_nrm2[DIIS_MAX_HIST];
    __local int    l_order[DIIS_MAX_HIST];
    if (sid >= batch || active[sid] == 0) return;

    __global const float* qn = q_new + (size_t)sid * n_atoms;
    __global const float* q0s = q0 + (size_t)sid * n_atoms;
    __global const float* qo = q_old + (size_t)sid * n_atoms;
    __global float* qn2 = q_next + (size_t)sid * n_atoms;
    __global float* qh = dq_hist + (size_t)sid * DIIS_MAX_HIST * n_atoms;
    __global float* rh = r_hist + (size_t)sid * DIIS_MAX_HIST * n_atoms;
    __global float* c = coeffs + (size_t)sid * DIIS_MAX_HIST;

    int idx = buf_idx[sid];
    int nf = n_filled[sid];

    float partial = 0.0f;
    for (int a = lid; a < n_atoms; a += lsz) {
        float res = qn[a] - qo[a];
        rh[idx * n_atoms + a] = res;
        qh[idx * n_atoms + a] = qo[a] - q0s[a];   // store Δq_in, not q_in
        partial += res * res;
    }

    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) scratch[lid] += scratch[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) rms[sid] = sqrt(scratch[0] / (float)n_atoms);

    if (lid == 0) {
        buf_idx[sid] = (idx + 1) % DIIS_MAX_HIST;
        if (nf < DIIS_MAX_HIST) n_filled[sid] = nf + 1;
        // W4: mark convergence on-device. !(r >= tol) also catches NaN/Inf —
        // a nonfinite replica must not keep iterating. The mix below and the
        // commit gate on the same flag, so nothing downstream consumes it.
        if (!(rms[sid] >= rms_tol)) active[sid] = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    if (active[sid] == 0) return;   // converged/invalid this step: skip the mix

    int n = n_filled[sid];

    // First history point: α-mix. 1-vector DIIS is Σc=1 → c_0=1 → undamped q_new.
    if (n < 2) {
        for (int a = lid; a < n_atoms; a += lsz) {
            qn2[a] = alpha * qn[a] + (1.0f - alpha) * qo[a];
        }
        return;
    }

    // Cooperative load of the residual history into local f64.
    for (int i = lid; i < n * n_atoms; i += lsz) lW[i] = (double)rh[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // W13: pivoted f64 QR on the residual-history matrix — replaces the
    // Gram+GE solve. κ(R) instead of κ²(RᵀR); dependent history columns
    // drop INDIVIDUALLY by pivot magnitude (no drop-oldest retry — that
    // was the blunt instrument this replaces). The Σc=1-constrained
    // min-norm solve reduces to two triangular solves at κ(R):
    //   c ∝ R_S⁻¹ R_S⁻ᵀ 1  over the surviving columns.
    //
    // Cooperative structure (2026-09-18): column j's dot/axpy runs on thread j
    // — serial per column, so operation order is IDENTICAL to the old
    // thread-0 loop (bit-identical results) but n columns proceed in parallel
    // instead of serially on one thread. Only the ≤10×10 solve stays on
    // thread 0. This was the #2 kernel cost: ~6K f64 ops serial on one thread
    // at 1/64 f64 rate ≈ 0.25–0.33 ms/call.
    __local double l_nrm_max;
    __local int    l_nq;
    __local int    l_jstar;
    __local int    l_astop;
    __local double l_inv;
    __local int    l_ok;
    __local int    l_reason;

    // Column norms — thread j owns column j.
    if (lid < n) {
        double s = 0.0;
        for (int a = 0; a < n_atoms; a++) {
            double x = lW[lid * n_atoms + a];
            s += x * x;
        }
        l_nrm2[lid] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        double nrm_max = 0.0;
        for (int j = 0; j < n; j++) nrm_max = fmax(nrm_max, l_nrm2[j]);
        l_nrm_max = nrm_max;
        l_nq = 0;
        l_astop = 0;
        l_ok = (isfinite(nrm_max) && nrm_max >= 1.0e-40) ? 1 : 0;
        l_reason = 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Modified Gram–Schmidt with column pivoting.
    for (int k = 0; k < n && l_ok && !l_astop; k++) {
        if (lid == 0) {
            int jstar = -1; double best = 0.0;
            for (int j = 0; j < n; j++) {
                if (l_nrm2[j] > best) { best = l_nrm2[j]; jstar = j; }
            }
            // dependent: surviving norm below rel tol → drop this column and
            // (loop continues) all remaining dependent ones
            if (jstar < 0 || best < 1.0e-24 * l_nrm_max) {
                l_astop = 1;
            } else {
                l_order[l_nq] = jstar;
                l_diag[l_nq] = sqrt(best);
                l_inv = 1.0 / l_diag[l_nq];
                l_jstar = jstar;
                l_nrm2[jstar] = -1.0;
                l_nq++;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (l_astop) break;
        const int jstar = l_jstar;
        // Normalize the pivot column — one element per thread.
        for (int a = lid; a < n_atoms; a += lsz) lW[jstar * n_atoms + a] *= l_inv;
        barrier(CLK_LOCAL_MEM_FENCE);
        // Dot + axpy for each surviving column — thread j owns column j; the
        // per-column serial order matches the old single-thread loop exactly.
        if (lid < n && l_nrm2[lid] >= 0.0) {
            const int j = lid;
            double rij = 0.0;
            for (int a = 0; a < n_atoms; a++) rij += lW[jstar * n_atoms + a] * lW[j * n_atoms + a];
            l_Rq[(l_nq - 1) * DIIS_MAX_HIST + j] = rij;
            for (int a = 0; a < n_atoms; a++) lW[j * n_atoms + a] -= rij * lW[jstar * n_atoms + a];
            l_nrm2[j] -= rij * rij;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        int ok = l_ok;
        int reason = l_reason;
        const int nq = l_nq;
        if (nq < 2) { ok = 0; reason = 1; }   // no useful DIIS span → α-mix

        if (ok) {
            // Assemble Rs = R_Sᵀ (LOWER-triangular in row-major): stored
            // element (k,i), i≤k, equals R_S[i][k] = q_iᵀ r_{order[k]} for
            // i<k and the pivot norm on the diagonal.
            for (int kp = 0; kp < nq; kp++) {
                const int jp = l_order[kp];
                for (int ip = 0; ip < kp; ip++) l_Rs[kp * DIIS_MAX_HIST + ip] = l_Rq[ip * DIIS_MAX_HIST + jp];
                l_Rs[kp * DIIS_MAX_HIST + kp] = l_diag[kp];
            }
            // R_Sᵀ y = 1 — forward substitution straight down the stored
            // lower triangle. Then R_S x = y — back-substitution over the
            // transpose of the stored triangle. c = x/Σx → Σc=1 exact.
            for (int k = 0; k < nq && ok; k++) {
                double s = 1.0;
                for (int i = 0; i < k; i++) s -= l_Rs[k * DIIS_MAX_HIST + i] * l_yy[i];
                const double piv = l_Rs[k * DIIS_MAX_HIST + k];
                if (!isfinite(piv) || fabs(piv) < 1.0e-30) { ok = 0; reason = 2; break; }
                l_yy[k] = s / piv;
            }
            if (ok) {
                double xsum = 0.0;
                for (int k = nq - 1; k >= 0; k--) {
                    double s = l_yy[k];
                    for (int i = k + 1; i < nq; i++) s -= l_Rs[i * DIIS_MAX_HIST + k] * l_cx[i];
                    const double piv = l_Rs[k * DIIS_MAX_HIST + k];
                    if (!isfinite(piv) || fabs(piv) < 1.0e-30 || !isfinite(s)) { ok = 0; reason = 2; break; }
                    l_cx[k] = s / piv;
                    xsum += l_cx[k];
                }
                if (ok && (!isfinite(xsum) || fabs(xsum) < 1.0e-30)) { ok = 0; reason = 2; }
                if (ok) {
                    for (int i = 0; i < n; i++) c[i] = 0.0f;
                    for (int k = 0; k < nq && ok; k++) {
                        const double ck = l_cx[k] / xsum;
                        if (!isfinite(ck) || fabs(ck) > 10.0) { ok = 0; reason = 2; break; }
                        c[l_order[k]] = (float)ck;
                    }
                }
            }
        }
        if (!ok) {
            // Structured status, no printf (D9): counter + last reason.
            diis_flag[sid] += 1;
            diis_reason[sid] = reason;
        }
        l_diis_ok = ok;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (l_diis_ok) {
        // c[] is indexed by physical slot (0 for dropped columns) — the mix
        // just sums all filled history.
        for (int a = lid; a < n_atoms; a += lsz) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += c[i] * (qh[i * n_atoms + a] + rh[i * n_atoms + a]);
            }
            qn2[a] = q0s[a] + sum;   // Δq-space mix: Σc=1 exactly preserves q0
        }
    } else {
        for (int a = lid; a < n_atoms; a += lsz) {
            qn2[a] = alpha * qn[a] + (1.0f - alpha) * qo[a];
        }
    }
}

// ------------------------------------------------------------------
// energy_reduce_batched — W12
//
// One WG per replica; f64 accumulation; ONE readback replaces the six
// separate scalar tails in energy_from_state. Output scalars[b*4]:
//   [0] e_band = 2 Σ_k f_k·ρ_k        (use_w) or 2 Σ_{occ_mask} eig
//   [1] mts    = Σ f·ln f + (1−f)·ln(1−f)   (use_w; else 0)
//   [2] dqv    = Δq·V      [3] q0v = q0·V
// ------------------------------------------------------------------
__kernel void energy_reduce_batched(
    const int n,
    const int n_atoms,
    const int batch,
    __global const float* eig_rho,
    __global const float* eig_diag,
    __global const float* occ_w,
    __global const int*   occ_mask,
    __global const float* dq,
    __global const float* q0,
    __global const float* v,
    const int use_w,
    const int occ_repair,
    __global double* out,           // [batch*4]
    __local double* scratch,        // [lsz]
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    if (sid >= batch) return;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int bn = sid * n;
    const int ba = sid * n_atoms;

    for (int c = 0; c < 4; c++) {
        double acc = 0.0;
        if (c == 0) {
            if (use_w != 0) {
                for (int k = lid; k < n; k += lsz)
                    acc += 2.0 * (double)occ_w[bn + k] * (double)eig_rho[bn + k];
            } else {
                for (int k = lid; k < n; k += lsz)
                    if (occ_mask[bn + k] != 0)
                        acc += 2.0 * (double)(occ_repair ? eig_rho[bn + k] : eig_diag[bn + k]);
            }
        } else if (c == 1) {
            if (use_w != 0) {
                for (int k = lid; k < n; k += lsz) {
                    const double f = (double)occ_w[bn + k];
                    const double g = 1.0 - f;
                    if (f > 1e-300) acc += f * log(f);
                    if (g > 1e-300) acc += g * log(g);
                }
            }
        } else if (c == 2) {
            for (int a = lid; a < n_atoms; a += lsz)
                acc += (double)dq[ba + a] * (double)v[ba + a];
        } else {
            for (int a = lid; a < n_atoms; a += lsz)
                acc += (double)q0[ba + a] * (double)v[ba + a];
        }
        scratch[lid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int off = lsz >> 1; off > 0; off >>= 1) {
            if (lid < off) scratch[lid] += scratch[lid + off];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) out[sid * 4 + c] = scratch[0];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// ------------------------------------------------------------------
// occ_normalize_batched
//
// Manifest §12 D3: renormalize occupied eigenvector columns of C′
// (orthonormal-basis Jacobi output). n_k = ||C'[:,k]||², then
// C'[:,k] ← C'[:,k]/√n_k for occ_mask[k]!=0. O(N·N_occ) repair of the
// Jacobi orthonormality loss; preserves column direction so ε/W stay valid.
// One workgroup per (system, column); unoccupied columns early-exit.
// ------------------------------------------------------------------
__kernel void occ_normalize_batched(
    const int n,
    const int batch,
    __global const int* occ_mask,
    __global float* Cp,
    __local float* scratch,
    __global const int* active,     // [batch] 0 → replica frozen, early-out
    __global const float* occ_w,    // [batch*n] Fermi weights (use_w=1)
    const int use_w,                // 1 → renorm every column with occ_w[k]>0
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int gid = get_group_id(0);
    const int iw = gid / n;
    const int sid = work_ids[iw];
    const int k = gid - iw * n;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;
    if (occ_mask[sid * n + k] == 0 && !(use_w != 0 && occ_w[sid * n + k] > 1.0e-30f)) return;
    __global float* col = Cp + (size_t)sid * n * n + k;
    float acc = 0.0f;
    for (int r = lid; r < n; r += lsz) {
        float x = col[r * n];
        acc = fma(x, x, acc);
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) scratch[lid] += scratch[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = rsqrt(scratch[0]);
    for (int r = lid; r < n; r += lsz) col[r * n] *= inv;
}

// ------------------------------------------------------------------
// snormalize_batched
//
// S-metric column normalization for the warm AO basis B (=C buffer):
//   n_k = c_kᵀ S c_k,  c_k ← c_k / √n_k  for ALL columns.
// The warm-Jacobi path rotates B in place (B←BJ); each rotation is
// orthogonal so BᵀSB≈I is preserved, but f32 rounding drifts the
// column norms — renormalizing arrests the drift before it perturbs
// the projected eigenproblem A=BᵀHB.
// One workgroup per (system, column).
// loc layout: t_s[n] | red[lsz]
// ------------------------------------------------------------------
__kernel void snormalize_batched(
    const int n,
    const int batch,
    __global float* C,
    __global const float* S,
    __local float* loc,
    __global const int* active,     // [batch] 0 → replica frozen, early-out
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int gid = get_group_id(0);
    const int iw = gid / n;
    const int sid = work_ids[iw];
    const int k = gid - iw * n;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;
    __local float* t_s = loc;
    __local float* red = loc + n;
    const size_t base = (size_t)sid * n * n;
    __global float* col = C + base + k;
    __global const float* Sb = S + base;
    for (int r = lid; r < n; r += lsz) {
        const int ro = r * n;
        float ss = 0.0f;
        for (int cc = 0; cc < n; ++cc) ss = fma(Sb[ro + cc], col[cc * n], ss);
        t_s[r] = ss;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float asx = 0.0f;
    for (int r = lid; r < n; r += lsz) asx = fma(col[r * n], t_s[r], asx);
    red[lid] = asx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = rsqrt(red[0]);
    for (int r = lid; r < n; r += lsz) col[r * n] *= inv;
}

// ------------------------------------------------------------------
// cs_normalize_batched   (T03)
//
// Paired S-metric column renorm using the already-materialized SC = S·C:
//   g_k = c_kᵀ S c_k = Σ_μ C[μ,k]·SC[μ,k];  C[:,k] ← C[:,k]/√g_k,
//   SC[:,k] ← SC[:,k]/√g_k — scaling both keeps SC = S·C exactly.
// Replaces snormalize_batched on the direct-population path: O(N²) reads
// (two column streams) instead of streaming the whole S per column.
// A non-finite or non-positive g_k yields a NaN scale → the column goes
// NaN → the replica fails loudly downstream (never clamped silent).
// One workgroup per (system, column); arbitrary-WG-safe reduction.
// ------------------------------------------------------------------
__kernel void cs_normalize_batched(
    const int n,
    const int batch,
    __global float* C,
    __global float* SC,
    __local float* red,
    __global const int* active,     // [batch] 0 → replica frozen, early-out
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int gid = get_group_id(0);
    const int iw = gid / n;
    const int sid = work_ids[iw];
    const int k = gid - iw * n;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;
    const size_t base = (size_t)sid * n * n + k;
    float acc = 0.0f;
    for (int r = lid; r < n; r += lsz) {
        acc = fma(C[base + r * n], SC[base + r * n], acc);
    }
    // Arbitrary-workgroup-size-safe tree reduction (bj_wg_sum scheme):
    // fold the ragged tail onto a power-of-two front first.
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    int p2 = 1;
    while ((p2 << 1) <= lsz) p2 <<= 1;
    if (lid + p2 < lsz) red[lid] += red[lid + p2];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = p2 >> 1; o > 0; o >>= 1) {
        if (lid < o) red[lid] += red[lid + o];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float g = red[0];
    const float inv = (isfinite(g) && g > 0.0f) ? rsqrt(g) : (float)NAN;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int r = lid; r < n; r += lsz) {
        C[base + r * n]  *= inv;
        SC[base + r * n] *= inv;
    }
}

// ------------------------------------------------------------------
// mulliken_cs_batched   (T03)
//
// Direct atomic Mulliken populations from C and SC = S·C:
//   q_A = 2 Σ_k w_k Σ_{μ∈A} C[μ,k]·SC[μ,k]
//     = Σ_{μ∈A} (D·S)_μμ  with D = 2·Σ_k w_k C[:,k]·C[:,k]ᵀ — identical
// algebra to mulliken_charges_batched(D,S) but the O(N²·n_occ) density
// build and its O(N²) D·S contraction leave the SCC loop entirely.
// w_k = occ_w[k] under Fermi smearing (use_w=1), else occ_mask[k] (0/1).
// Same two-phase shape as mulliken_charges_batched: phase 1 computes the
// per-orbital contraction p[μ] = Σ_k w_k·C[μ,k]·SC[μ,k] (strided lanes),
// phase 2 reduces p[μ] over each atom's orbitals.
// One workgroup per system.
// ------------------------------------------------------------------
__kernel void mulliken_cs_batched(
    const int n,
    const int n_atoms,
    const int batch,
    __global const float* C,
    __global const float* SC,
    __global const int* orb_atom,
    __global const int* occ_mask,
    __global const float* occ_w,
    const int use_w,
    __global float* q,
    __local float* diag,
    __global const int* active,     // [batch] 0 → replica frozen, early-out
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;
    __global const float* Cb = C  + (size_t)sid * n * n;
    __global const float* Sb = SC + (size_t)sid * n * n;
    __global const int*   oa = orb_atom + (size_t)sid * n;
    __global const int*   mb = occ_mask + (size_t)sid * n;
    __global const float* wb = occ_w + (size_t)sid * n;
    __global float* qb = q + (size_t)sid * n_atoms;

    // Phase 1: per-orbital weighted contraction.
    for (int mu = lid; mu < n; mu += lsz) {
        __global const float* crow = Cb + mu * n;
        __global const float* srow = Sb + mu * n;
        float s = 0.0f;
        if (use_w != 0) {
            for (int k = 0; k < n; ++k) s = fma(wb[k] * crow[k], srow[k], s);
        } else {
            for (int k = 0; k < n; ++k) s = fma((float)mb[k] * crow[k], srow[k], s);
        }
        diag[mu] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: per-atom population (factor 2 = spin, as in D=2ΣwCCᵀ).
    if (lid < n_atoms) {
        float sum = 0.0f;
        for (int mu = 0; mu < n; ++mu) {
            if (oa[mu] == lid) sum += diag[mu];
        }
        qb[lid] = 2.0f * sum;
    }
}

// ------------------------------------------------------------------
// occ_rayleigh_batched
//
// Manifest §12 D3/D4: generalized Rayleigh quotient of the AO eigenvectors,
// rho[k] = (c_kᵀ H_scc c_k) / (c_kᵀ S c_k), occupied columns only.
// The Jacobi diagonal ε_k drifts from the true Rayleigh quotient of the
// stored vectors (measured max|ε−ρ|~1e-6 at N=87, biased sum ~2.8e-5 Ha);
// ρ is the variationally correct weight for the energy and the EDM W.
// rho[k]=0 for unoccupied columns.
// use_w=1 (Fermi smearing): every column with occ_w[k]>0 gets a quotient —
// fractional-weight orbitals need ρ just as much as occupied ones.
// One workgroup per (system, column).
// loc layout: t_h[n] | t_s[n] | red[2*lsz]
// ------------------------------------------------------------------
__kernel void occ_rayleigh_batched(
    const int n,
    const int batch,
    __global const int* occ_mask,
    __global const float* C,
    __global const float* H,
    __global const float* S,
    __global float* rho,
    __local float* loc,
    __global const int* active,     // [batch] 0 → replica frozen, early-out
    __global const float* occ_w,    // [batch*n] Fermi weights (use_w=1)
    const int use_w,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int gid = get_group_id(0);
    const int iw = gid / n;
    const int sid = work_ids[iw];
    const int k = gid - iw * n;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;
    const int want = occ_mask[sid * n + k] != 0 || (use_w != 0 && occ_w[sid * n + k] > 1.0e-30f);
    if (!want) {
        if (lid == 0) rho[sid * n + k] = 0.0f;
        return;
    }
    __local float* t_h = loc;
    __local float* t_s = loc + n;
    __local float* red = loc + 2 * n;
    const size_t base = (size_t)sid * n * n;
    __global const float* col = C + base + k;
    __global const float* Hb = H + base;
    __global const float* Sb = S + base;
    for (int r = lid; r < n; r += lsz) {
        const int ro = r * n;
        float sh = 0.0f, ss = 0.0f;
        for (int cc = 0; cc < n; ++cc) {
            const float x = col[cc * n];
            sh = fma(Hb[ro + cc], x, sh);
            ss = fma(Sb[ro + cc], x, ss);
        }
        t_h[r] = sh;
        t_s[r] = ss;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float ah = 0.0f, asx = 0.0f;
    for (int r = lid; r < n; r += lsz) {
        const float x = col[r * n];
        ah = fma(x, t_h[r], ah);
        asx = fma(x, t_s[r], asx);
    }
    red[lid] = ah;
    red[lid + lsz] = asx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) {
            red[lid] += red[lid + off];
            red[lid + lsz] += red[lid + lsz + off];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) rho[sid * n + k] = red[0] / red[lsz];
}

// ==================================================================
// §12 D5: Löwdin X repair on GPU — M=XᵀSX via ordinary f32 GEMMs in the
// driver; these three kernels cover the elementwise/reduction steps.
// ==================================================================

// Q = (3I − M)/2 elementwise. One WG per system.
__kernel void lowdin_q_from_m_batched(
    const int n,
    const int batch,
    __global const float* M,
    __global float* Q,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    if (sid >= batch) return;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __global const float* Mb = M + (size_t)sid * n * n;
    __global float* Qb = Q + (size_t)sid * n * n;
    for (int idx = lid; idx < n * n; idx += lsz) {
        const int r = idx / n;
        const int c = idx - r * n;
        Qb[idx] = 0.5f * ((r == c ? 3.0f : 0.0f) - Mb[idx]);
    }
}

// res[sid] = max |M − I| — first-order metric defect of X (Löwdin residual).
// W7: row-sum ‖M−I‖∞ — elementwise max does NOT bound the eigen-residual
// (‖·‖₂ ≤ ‖·‖∞ row-sum, not ≤ max|entry|). The Newton bound
// e1 ≤ ¾e0² + ¼e0³ (per-eigenvalue) is only rigorous under this norm.
__kernel void metric_residual_batched(
    const int n,
    const int batch,
    __global const float* M,
    __global float* res,
    __local float* scratch,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    if (sid >= batch) return;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __global const float* Mb = M + (size_t)sid * n * n;
    float mx = 0.0f;
    for (int r = lid; r < n; r += lsz) {
        float row = 0.0f;
        for (int c = 0; c < n; c++) {
            row += fabs(Mb[r * n + c] - (r == c ? 1.0f : 0.0f));
        }
        mx = fmax(mx, row);
    }
    scratch[lid] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) scratch[lid] = fmax(scratch[lid], scratch[lid + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) res[sid] = scratch[0];
}

// X ← X1 where the repair improved the metric (e1 < e0); NaN-safe skip.
__kernel void lowdin_accept_batched(
    const int n,
    const int batch,
    __global const float* e0,
    __global const float* e1,
    __global const float* X1,
    __global float* X,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int sid = work_ids[get_group_id(0)];
    if (sid >= batch) return;
    if (!(e1[sid] < e0[sid])) return;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const size_t base = (size_t)sid * n * n;
    for (int idx = lid; idx < n * n; idx += lsz) X[base + idx] = X1[base + idx];
}

// ==================================================================
// LNV reference solver — Li–Nunes–Vanderbilt auxiliary-matrix steepest
// descent (canonical baseline for the minimize+retract density-matrix
// family; design doc Alternative_Dense_Multi_Eigensolve.chat.md §15).
//
//   Ω(L) = Tr[f(L)·F],   f(x) = 3x²−2x³,  D = f(L),  F = (H−μI)/Δε
//   ∇_L Ω = 3(LF+FL) − 2(L²F + LFL + FL²)
//         = [L,[L,F]] + 3[(L−L²)F + F(L−L²)]     ← FP32 residual form
//
// The residual form builds the gradient from the small commutator and
// idempotency residuals instead of subtracting large products — no
// catastrophic cancellation in f32. At a clean projector L=P (f(P)=P):
//   G = PF+FP−2PFP = T_P(H) = [P,[P,H]]   — the exact tangent force,
// so the fixed point is exactly [D,H]=0 with no learned constraint
// state. Per iteration: S=L² (sym square) + A=LF, B=SF, C=A·L=LFL
// (3 general GEMMs) + the two elementwise kernels below.
//
// μ is the Fermi level (trace channel): G(μ+δμ) = G − 6(δμ/Δε)·R with
// R = L−L², so δμ = Δε·⟨R,G⟩/(6⟨R,R⟩) cancels the trace-changing
// component of the next step to first order; a small kmu·(nocc−TrD)
// term steers TrD→nocc from a cold start.
// ==================================================================

// 4-way local reduction helper (same tree pattern as the scratch
// reductions above, packed float4 to keep one local array).
static float4 mo_reduce4(__local float4* red, float4 v, int lid, int lsz)
{
    red[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return red[0];
}

// lnv_scal_batched — per-iteration scalar pass:
//   F = (H − μI)/Δε        (written in the same strided sweep)
//   trd[sys] = Tr f(L) = 3·Tr(L²) − 2·Tr(L³) = 3⟨L,L⟩ − 2⟨S,L⟩
// One workgroup per system; early-outs on done[sys].
__kernel void lnv_scal_batched(
    const int n,
    const int batch,
    __global const float* L,
    __global const float* S,
    __global const float* H,
    __global float*       F,
    __global const float* mu,
    __global const float* spans,
    __global float*       trd,
    __global const int*   done,
    __local float4*       scratch)
{
    const int sys = get_group_id(0);
    if (sys >= batch || done[sys]) return;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int nn = n * n;
    const size_t base = (size_t)sys * nn;
    const float mu_s = mu[sys];
    const float isp = 1.0f / spans[sys];
    float ll = 0.0f, sl = 0.0f;
    for (int idx = lid; idx < nn; idx += lsz) {
        const float l = L[base + idx];
        const float s = S[base + idx];
        const int i = idx / n, j = idx - i * n;
        ll = fma(l, l, ll);
        sl = fma(s, l, sl);
        F[base + idx] = (H[base + idx] - (i == j ? mu_s : 0.0f)) * isp;
    }
    const float4 rr = mo_reduce4(scratch, (float4)(ll, sl, 0.0f, 0.0f), lid, lsz);
    if (lid == 0) trd[sys] = 3.0f * rr.x - 2.0f * rr.y;
}

// lnv_finish_batched — gradient assembly + descent step + μ feedback:
//   G = (B + Bᵀ − 2C) + 3·[(A−B) + (A−B)ᵀ]      (stored to Gbuf)
//   L ← L − α_s·G                                α_s = per-system step
//   μ += Δε·⟨R,G⟩/(6⟨R,R⟩) + kmu·Δε·(nocc−trd)
//   α_s ← BB1 = ⟨s,y⟩/⟨y,y⟩   s = −α_s·G_prev, y = G − G_prev
//   diag[2*sys] = ‖G‖_F, diag[2*sys+1] = trd
//   done[sys] = ‖G‖<tol && |trd−nocc|≤tol_tr
// The Barzilai–Borwein scalar step adapts α to the local curvature —
// bare fixed-α SD is rate-limited by the Hessian spread (measured
// ~40%/100 iters at α=0.2; α≥1.0 diverges). α is clamped to
// [0.02, 0.9] — the stability limit measured on this problem is ~0.7.
// Transposed reads hit A/B only (read-only); L is read+written at the
// same element → race-free. Iteration 0 (G_prev=0) yields sy=0 →
// α stays at its init value.
__kernel void lnv_finish_batched(
    const int n,
    const int batch,
    __global float*       L,
    __global const float* S,
    __global const float* A,
    __global const float* B,
    __global const float* C,
    __global float*       Gbuf,
    __global float*       V,
    __global float*       alpha_b,
    __global float*       mu,
    __global const float* spans,
    __global const float* nocc,
    __global const float* trd,
    const float           kmu,
    const float           mom,
    const float           smax,
    const float           tol,
    const float           tol_tr,
    __global float*       diag,
    __global int*         done,
    __local float4*       scratch)
{
    const int sys = get_group_id(0);
    if (sys >= batch || done[sys]) return;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int nn = n * n;
    const size_t base = (size_t)sys * nn;
    const float a_cur = alpha_b[sys];
    __local float l_meff;   // effective momentum (0 on O'Donoghue restart)
    __local float l_sscale; // step-cap scale
    // pass 1: assemble the residual-form gradient, store to Gbuf,
    // accumulate the reductions needed for μ/restart/cap decisions.
    float g2 = 0.0f, rg = 0.0f, r2 = 0.0f, ggp = 0.0f;
    float gp2 = 0.0f, gv = 0.0f, vv = 0.0f;
    for (int idx = lid; idx < nn; idx += lsz) {
        const int i = idx / n, j = idx - i * n;
        const float a  = A[base + idx];
        const float b  = B[base + idx];
        const float c  = C[base + idx];
        const float aT = A[base + j * n + i];
        const float bT = B[base + j * n + i];
        const float g = (b + bT - 2.0f * c) + 3.0f * ((a - b) + (aT - bT));
        const float gp = Gbuf[base + idx];
        const float r  = L[base + idx] - S[base + idx];
        const float vp = V[base + idx];
        Gbuf[base + idx] = g;
        g2  = fma(g, g, g2);
        rg  = fma(r, g, rg);
        r2  = fma(r, r, r2);
        ggp = fma(g, gp, ggp);
        gp2 = fma(gp, gp, gp2);
        gv  = fma(g, vp, gv);
        vv  = fma(vp, vp, vv);
    }
    const float4 rr = mo_reduce4(scratch, (float4)(g2, rg, r2, ggp), lid, lsz);
    const float4 rp = mo_reduce4(scratch, (float4)(gp2, gv, vv, 0.0f), lid, lsz);
    if (lid == 0) {
        const float gnorm = sqrt(rr.x);
        const float sp = spans[sys];
        const float cap = 0.02f * sp;
        // μ channel — failure modes found by CPU-f64 bisection:
        //  (a) R→0 (warm start): ⟨R,G⟩/⟨R,R⟩ is 0/0 → exploded μ to −2e9
        //      in ONE step. Lower gate: ‖R‖² > 1e-4.
        //  (b) far off-manifold: the linear trace model is wrong → the
        //      projection drove a μ↔L exponential resonance (δμ = +10 →
        //      −115 → +2198, g doubling every ~20 iters). Upper gate:
        //      project ONLY while ‖R‖² < 0.5·n (near-manifold); farther
        //      out the capped steering alone restores the trace.
        //      CPU-validated: this window alone gives 8/8 at Δ=0.5;
        //      panic-reset instead loops forever (1345 resets).
        //  Both δμ terms are capped at 0.02·span — a chemical potential
        //  never needs to jump further per step.
        if (rr.z > 1.0e-4f && rr.z < 0.5f * n) {
            mu[sys] += clamp(sp * rr.y / (6.0f * rr.z), -cap, cap);
        }
        mu[sys] += clamp(kmu * sp * (nocc[sys] - trd[sys]), -cap, cap);
        // Momentum is gated to the SAME near-manifold window as the
        // μ-projection: the divergence is a coupled V↔μ resonance that
        // needs both drivers active while far off-manifold. Outside
        // ‖R‖² < 0.5·n take a pure damped SD step — CPU-validated:
        // momentum-gating gives 8/8 at Δ=0.02 AND Δ=0.5 (ungated: sys3
        // exponential blow-up ~it=970). Plus O'Donoghue restart
        // (⟨G,V⟩>0 → uphill → pure SD step, before overshoot).
        const float meff = (rp.y > 0.0f || rr.z > 0.5f * n) ? 0.0f : mom;
        // step cap: ‖v_new‖ ≤ meff·‖V‖ + ‖G‖ (bound, tight when V ∥ −G
        // which is the unstable case). CPU-validated: smax=1.0 gives
        // 8/8 convergence at Δ=0.02 AND Δ=0.5 (uncapped: 0/8 at 0.5).
        const float vbound = meff * sqrt(rp.z) + gnorm;
        l_meff = meff;
        l_sscale = (a_cur * vbound > smax) ? smax / (a_cur * vbound) : 1.0f;
        // BB1 for bare SD only — with momentum the displacement is
        // a_cur·V, not −a_cur·G_prev, so the secant estimate is invalid.
        const float gp2r = rp.x;
        const float y2 = rr.x - 2.0f * rr.w + gp2r;
        const float sy = a_cur * (gp2r - rr.w);
        if (mom <= 0.0f && y2 > 1.0e-30f && sy > 0.0f) {
            alpha_b[sys] = clamp(sy / y2, 0.02f, 0.9f);
        }
        diag[4 * sys]     = gnorm;
        diag[4 * sys + 1] = trd[sys];
        diag[4 * sys + 2] = rr.z;      // ‖R‖² — off-manifold measure
        diag[4 * sys + 3] = mu[sys];
        done[sys] = (gnorm < tol
            && fabs(trd[sys] - nocc[sys]) <= fmax(tol_tr * nocc[sys], 1.0e-4f));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // pass 2: momentum + L update. v = meff·V_prev − G (meff=0 on
    // restart → pure SD), L += α·s·v. All same-element → race-free.
    const float meff = l_meff;
    const float ss = l_sscale * a_cur;
    for (int idx = lid; idx < nn; idx += lsz) {
        const float g = Gbuf[base + idx];
        const float v = fma(meff, V[base + idx], -g);
        V[base + idx] = v;
        L[base + idx] = fma(ss, v, L[base + idx]);
    }
}

// lnv_combine_batched — final density D = 3S − 2·(S·L) = 3L² − 2L³.
// One thread per element across all systems (T = S·L from a GEMM).
__kernel void lnv_combine_batched(
    const int n,
    const int batch,
    __global const float* S,
    __global const float* T,
    __global float*       D)
{
    const int gid = get_global_id(0);
    if (gid >= batch * n * n) return;
    D[gid] = 3.0f * S[gid] - 2.0f * T[gid];
}

// ------------------------------------------------------------------
// One-GEMM SCC update in the orthogonal basis (bold geometry step).
//
// H = H0 + ½(DS + SD) with D_μμ = V_atom(μ), so
//   H' = Xᵀ H0 X + ½(M + Mᵀ),  M = Xᵀ D (S X).
// C = Xᵀ S is cached per geometry. (S X) = Cᵀ, and
//   T_μk = V_μ · C_kμ   is D·Cᵀ, then one GEMM M = Xᵀ·T.
// Charges: Y = K·C, p_μ = 2·Σ_i X_μi Y_iμ (the 2 is the closed-shell
// density convention D = 2 X K Xᵀ). One workgroup per system.
// ------------------------------------------------------------------
__kernel void orth_dv_ct_batched(
    const int n,
    const int n_atoms,
    const int batch,
    __global const float* C,
    __global const float* V,
    __global const int* orb_atom,
    __global float* T,
    __global const int* active,
    __global const int* work_ids
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;
    const int nn = n * n;
    __global const float* Cb = C + (size_t)sid * nn;
    __global const float* Vb = V + (size_t)sid * n_atoms;
    __global const int* oa = orb_atom + (size_t)sid * n;
    __global float* Tb = T + (size_t)sid * nn;
    for (int idx = lid; idx < nn; idx += lsz) {
        const int mu = idx / n;
        const int k = idx - mu * n;
        Tb[idx] = Vb[oa[mu]] * Cb[k * n + mu];
    }
}

__kernel void hp_from_h0_sym_batched(
    const int n,
    const int batch,
    __global const float* H0p,
    __global const float* M,
    __global float* hp,
    __global const int* active,
    __global const int* work_ids
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;
    const int nn = n * n;
    __global const float* Hb = H0p + (size_t)sid * nn;
    __global const float* Mb = M + (size_t)sid * nn;
    __global float* Pb = hp + (size_t)sid * nn;
    for (int idx = lid; idx < nn; idx += lsz) {
        const int i = idx / n;
        const int j = idx - i * n;
        Pb[idx] = Hb[idx] + 0.5f * (Mb[idx] + Mb[j * n + i]);
    }
}

__kernel void mulliken_kc_batched(
    const int n,
    const int n_atoms,
    const int batch,
    __global const float* X,
    __global const float* Y,
    __global const int* orb_atom,
    __global float* q,
    __local float* diag,
    __global const int* active,
    __global const int* work_ids
) {
    const int sid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;
    __global const float* Xb = X + (size_t)sid * n * n;
    __global const float* Yb = Y + (size_t)sid * n * n;
    __global const int* oa = orb_atom + (size_t)sid * n;
    __global float* qb = q + (size_t)sid * n_atoms;
    for (int mu = lid; mu < n; mu += lsz) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s = fma(Xb[mu * n + i], Yb[i * n + mu], s);
        diag[mu] = 2.0f * s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < n_atoms) {
        float sum = 0.0f;
        for (int mu = 0; mu < n; ++mu) {
            if (oa[mu] == lid) sum += diag[mu];
        }
        qb[lid] = sum;
    }
}
