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
    float kahan_c = 0.0f; // f32 Kahan across K (not f64 GEMM — that reverted: worse E, DIIS blowup)
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
                float prod = As[ly * TILE_K + kk] * Bs[kk * TILE_N + lx];
                float y = prod - kahan_c;
                float t = sum + y;
                kahan_c = (t - sum) - y;
                sum = t;
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
    __global float* C
) {
    const int sid = get_group_id(0);
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
    for (int idx = lid; idx < n * n; idx += lsz) {
        int r = idx / n;
        int c = idx - r * n;
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += LA[r * ld + k] * LB[k * ld + c];
        }
        Cb[r * n + c] = sum;
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
    __global float* V
) {
    const int sid = get_group_id(0);
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
    __local float* Vloc
) {
    const int sid = get_group_id(0);
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
    __local float* diag
) {
    const int sid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch) return;
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
    __local float* scratch
) {
    const int sid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch) return;
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
    __global float* dq
) {
    const int sid = get_group_id(0);
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
    __global const float* eig
) {
    const int sid = get_group_id(0);
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
    __local float* scratch
) {
    const int sid = get_group_id(0);
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
    __local float* scratch
) {
    const int sid = get_group_id(0);
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
    __global float* diag
) {
    const int gid = get_global_id(0);
    const int total = n * batch;
    if (gid >= total) return;
    const int sid = gid / n;
    const int i = gid % n;
    diag[gid] = a[(size_t)sid * n * n + i * n + i];
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
    __global int* occ_mask
) {
    const int sid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch) return;

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
    // Mark the first n_occ original indices as occupied.
    for (int i = lid; i < n; i += lsz) {
        int orig_idx = lidx[i];
        if (orig_idx >= 0) {
            occ_mask[(size_t)sid * n + orig_idx] = (i < n_occ) ? 1 : 0;
        }
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
    __global float* e_rep                // [batch]
) {
    const int sid = get_group_id(0);
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
//   1. residual = q_new - q_old; store (q_in, res) in the ring buffer
//   2. rms = √(Σ res² / n_atoms)   — RMS, not L2; matches CPU SCC tol
//   3. n < 2 after this store: α-mix (never 1-vector DIIS; that is undamped q_new)
//   4. n ≥ 2: Gram in f64, scale by max|B_ij|, tiny GE in f64
//   5. small pivot / non-finite / |Σc−1|>tol → α-mix + printf, never silent c_i=0
//
// Buffers (all per-system, indexed by sid * stride + ...):
//   q_new       — [batch*n_atoms] current SCC output charges
//   q_old       — [batch*n_atoms] current input charges (overwritten with mixed)
//   q_hist      — [batch*max_hist*n_atoms] ring buffer of q_in vectors
//   r_hist      — [batch*max_hist*n_atoms] ring buffer of residual vectors
//   buf_idx     — [batch] ring buffer write position (i32)
//   n_filled    — [batch] number of valid history entries (i32)
//   b_mat       — [batch*(max_hist+1)*(max_hist+1)] unused host-visible workspace
//   rhs         — [batch*(max_hist+1)] unused host-visible workspace
//   coeffs      — [batch*max_hist] DIIS coefficients output
//   rms         — [batch] residual RMS output
//
// Specialization: DIIS_MAX_HIST must be set (default 10).
// ------------------------------------------------------------------
#ifndef DIIS_MAX_HIST
#define DIIS_MAX_HIST 10
#endif
#define DIIS_NP1 (DIIS_MAX_HIST + 1)

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void diis_step_batched(
    const int n_atoms,
    const int batch,
    const float alpha,           // simple mixing fallback parameter
    __global const float* q_new, // [batch*n_atoms]
    __global float* q_old,       // [batch*n_atoms] — overwritten with mixed result
    __global float* q_hist,      // [batch*DIIS_MAX_HIST*n_atoms]
    __global float* r_hist,      // [batch*DIIS_MAX_HIST*n_atoms]
    __global int* buf_idx,       // [batch]
    __global int* n_filled,       // [batch]
    __global float* b_mat,        // [batch*DIIS_NP1*DIIS_NP1]
    __global float* rhs,         // [batch*DIIS_NP1]
    __global float* coeffs,      // [batch*DIIS_MAX_HIST]
    __global float* rms,         // [batch]
    __local float* scratch        // workgroup scratch (≥ n_atoms + DIIS_NP1*DIIS_NP1)
) {
    const int sid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __local int l_diis_ok;
    if (sid >= batch) return;

    __global const float* qn = q_new + (size_t)sid * n_atoms;
    __global float* qo = q_old + (size_t)sid * n_atoms;
    __global float* qh = q_hist + (size_t)sid * DIIS_MAX_HIST * n_atoms;
    __global float* rh = r_hist + (size_t)sid * DIIS_MAX_HIST * n_atoms;
    __global float* c = coeffs + (size_t)sid * DIIS_MAX_HIST;
    // b_mat / rhs kept in the signature (bound at GpuSccPlan::new); solve uses private f64.

    int idx = buf_idx[sid];
    int nf = n_filled[sid];

    float partial = 0.0f;
    for (int a = lid; a < n_atoms; a += lsz) {
        float res = qn[a] - qo[a];
        rh[idx * n_atoms + a] = res;
        qh[idx * n_atoms + a] = qo[a];
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
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    int n = n_filled[sid];
    int np1 = n + 1;

    // First history point: α-mix. 1-vector DIIS is Σc=1 → c_0=1 → undamped q_new.
    if (n < 2) {
        for (int a = lid; a < n_atoms; a += lsz) {
            qo[a] = alpha * qn[a] + (1.0f - alpha) * qo[a];
        }
        return;
    }

    if (lid == 0) {
        double Bd[DIIS_NP1 * DIIS_NP1];
        double bd[DIIS_NP1];
        int ok = 1;
        int reason = 0;
        double scale = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dot = 0.0;
                __global float* ri = rh + i * n_atoms;
                __global float* rj = rh + j * n_atoms;
                for (int a = 0; a < n_atoms; a++) {
                    dot += (double)ri[a] * (double)rj[a];
                }
                Bd[i * np1 + j] = dot;
                scale = fmax(scale, fabs(dot));
            }
        }
        if (!isfinite(scale) || scale < 1.0e-30) {
            ok = 0; reason = 1;
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) Bd[i * np1 + j] /= scale;
                Bd[i * np1 + n] = 1.0;
                Bd[n * np1 + i] = 1.0;
            }
            Bd[n * np1 + n] = 0.0;
            for (int i = 0; i < np1; i++) bd[i] = 0.0;
            bd[n] = 1.0;

            for (int k = 0; k < np1 && ok; k++) {
                int max_row = k;
                double max_val = fabs(Bd[k * np1 + k]);
                for (int i = k + 1; i < np1; i++) {
                    double v = fabs(Bd[i * np1 + k]);
                    if (v > max_val) { max_val = v; max_row = i; }
                }
                if (max_row != k) {
                    for (int j = k; j < np1; j++) {
                        double tmp = Bd[k * np1 + j];
                        Bd[k * np1 + j] = Bd[max_row * np1 + j];
                        Bd[max_row * np1 + j] = tmp;
                    }
                    double tmp = bd[k]; bd[k] = bd[max_row]; bd[max_row] = tmp;
                }
                if (!isfinite(max_val) || max_val < 1.0e-12) {
                    ok = 0; reason = 1;
                    break;
                }
                double piv = Bd[k * np1 + k];
                for (int i = k + 1; i < np1; i++) {
                    double factor = Bd[i * np1 + k] / piv;
                    Bd[i * np1 + k] = 0.0;
                    for (int j = k + 1; j < np1; j++) {
                        Bd[i * np1 + j] -= factor * Bd[k * np1 + j];
                    }
                    bd[i] -= factor * bd[k];
                }
            }
            if (ok) {
                for (int i = np1 - 1; i >= 0; i--) {
                    double sum = bd[i];
                    for (int j = i + 1; j < np1; j++) sum -= Bd[i * np1 + j] * bd[j];
                    double piv = Bd[i * np1 + i];
                    if (!isfinite(piv) || fabs(piv) < 1.0e-12 || !isfinite(sum)) {
                        ok = 0; reason = 2;
                        break;
                    }
                    bd[i] = sum / piv;
                }
            }
            if (ok) {
                double csum = 0.0;
                for (int i = 0; i < n; i++) {
                    if (!isfinite(bd[i]) || fabs(bd[i]) > 10.0) { ok = 0; reason = 2; break; }
                    csum += bd[i];
                }
                if (ok && fabs(csum - 1.0) > 1.0e-4) { ok = 0; reason = 3; }
            }
            if (ok) {
                for (int i = 0; i < n; i++) c[i] = (float)bd[i];
            }
        }
        if (!ok) {
            printf("DIIS fallback sid=%d n=%d reason=%d (1=pivot/scale 2=nonfinite 3=sum(c)!=1) — α-mix\n", sid, n, reason);
        }
        l_diis_ok = ok;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (l_diis_ok) {
        for (int a = lid; a < n_atoms; a += lsz) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += c[i] * (qh[i * n_atoms + a] + rh[i * n_atoms + a]);
            }
            qo[a] = sum;
        }
    } else {
        for (int a = lid; a < n_atoms; a += lsz) {
            qo[a] = alpha * qn[a] + (1.0f - alpha) * qo[a];
        }
    }
}
