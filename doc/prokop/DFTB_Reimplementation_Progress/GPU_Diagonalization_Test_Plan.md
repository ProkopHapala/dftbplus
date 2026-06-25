# GPU Diagonalization Test Plan

## Current State Review

### GPU Infrastructure (`gpu_matrix.rs` + `gpu_matrix_ops.cl`)

**Available OpenCL kernels:**
- `batched_gemm` — tiled batched C = α·op(A)·op(B) + β·C (row-major, batch in 3rd NDRange dim)
- `local_jacobi_blocks` — serial cyclic Jacobi eigendecomposition (single thread, matrix in local memory)
- `local_jacobi_blocks_parallel` — row-parallel Jacobi (all threads collaborate on rotations)
- `scale_density_guess`, `purify_mcweeny`, `purify_tc2_*` — density matrix purification
- `trace_reduce`, `idempotency_reduce` — reduction kernels

**Available Rust API (`GpuMatrixContext`):**
- `new(config)` → creates OpenCL context, compiles kernels
- `buffer_from_slice`, `zero_buffer`, `read_buffer` — buffer management
- `batched_gemm(n, batch, trans_a, trans_b, alpha, beta, a, b, c)` — matrix multiply
- `lowdin_transform(n, batch, x, h, scratch, out)` — H' = X^T·H·X (two GEMMs)
- `local_jacobi_blocks(m, n_blocks, blocks, eigvals, eigvecs, max_sweeps, tol)` — diagonalize small symmetric matrices
- `local_jacobi_blocks_parallel(...)` — same but parallel within workgroup
- `brent_luk_rounds(n_blocks)` — schedule for block Jacobi (not yet driven)

**Key constraints:**
- All GPU data is **f32** (single precision)
- Matrix layout: **row-major**, element (i,j) of batch b at index `b*N*N + i*N + j`
- `local_jacobi_blocks` supports matrices up to `jacobi_max_m` (default 64)
- No full block-Jacobi driver yet — only the subproblem diagonalization + Brent-Luk scheduling exist
- No GPU-side S^{-1/2} reconstruction — Jacobi gives eigenvalues/vectors, host must reconstruct

### CPU Diagonalization (`fragment.rs:215-253`)

Uses Cholesky reduction:
1. S = L·L^T (Cholesky, cached)
2. H' = L^{-1}·H·L^{-T} (triangular solves)
3. SymmetricEigen on H' (nalgebra, f64)
4. C = L^{-T}·C' (back-transform)
5. Sort eigenvalues ascending

All in **f64** (double precision).

### Data Flow for Tests

```
CPU (f64, column-major nalgebra)
  → convert to f32, row-major flat Vec<f32>
  → upload to GPU Buffer<f32>
  → GPU kernels
  → read back to host Vec<f32>
  → convert to f64 for comparison
```

---

## Test Plan

### Test file: `rust_dftb/tests/gpu_diagonalization.rs`

### Test 1: GEMM correctness (foundation check)
- Multiply two known small matrices (e.g. 4×4) on GPU
- Compare with CPU matrix multiply
- Validates the fundamental building block before testing higher-level operations
- Tolerance: ~1e-5 relative

### Test 2: Jacobi — diagonalize a small symmetric matrix
- Create a known symmetric matrix (e.g. 4×4, 8×8) with known eigenvalues
- Diagonalize on CPU: `nalgebra::SymmetricEigen`
- Diagonalize on GPU: `local_jacobi_blocks` (and/or `local_jacobi_blocks_parallel`)
- Compare:
  - Eigenvalues (sorted ascending), tolerance ~1e-4
  - Eigenvectors: handle sign ambiguity via |v_cpu · v_gpu| ≈ 1 or compare |v| elementwise
- Test both serial and parallel Jacobi variants

### Test 3: Diagonalize overlap matrix S (intermediate step)
- Build S from a real molecule (H2: 2×2, N2: 8×8, HCOOH: 14×14)
- Diagonalize S on CPU → Λ_S, V_S
- Diagonalize S on GPU → Λ_S_gpu, V_S_gpu
- Compare eigenvalues and eigenvectors
- Compute S^{-1/2} = V·Λ^{-1/2}·V^T on CPU from both results, compare
- This validates the intermediate step toward Löwdin transform

### Test 4: Löwdin transform on GPU
- Compute X = S^{-1/2} on CPU
- Upload H0 and X to GPU as f32 row-major
- Apply `lowdin_transform` on GPU: H' = X^T·H·X
- Compute H' on CPU for reference (X^T·H0·X in f64)
- Compare, tolerance ~1e-4

### Test 5: Full generalized eigenvalue problem HC = SCε
- Build H0 and S from a real molecule (non-SCC, neutral charges)
- **CPU reference**: `Fragment::diagonalize()` → eigenvalues, eigenvectors
- **GPU pipeline**:
  1. Compute X = S^{-1/2} on CPU (from eigendecomposition)
  2. Upload H0, X to GPU as f32 row-major
  3. `lowdin_transform`: H' = X^T·H·X (two GEMMs)
  4. `local_jacobi_blocks`: diagonalize H' → ε, C'
  5. Read back ε, C'
  6. Sort eigenvalues ascending (host-side)
  7. Back-transform: C = X·C' (one GEMM on GPU, or on host)
- Compare eigenvalues (sorted), tolerance ~1e-4
- Compare eigenvectors (sign-ambiguity handled), tolerance ~1e-3

### Test 6: Batched diagonalization (multiple fragments)
- Build H0 and S for multiple molecules (e.g. 3× H2, or H2 + N2)
- Stack into batched buffers: H[batch][N][N], S[batch][N][N]
- Diagonalize all in one GPU launch
- Compare each batch element with CPU reference
- Note: all fragments must have same N (pad if needed, or use same molecule)

---

## Implementation Notes

### Matrix conversion (CPU → GPU)
```rust
fn dmatrix_to_row_major_f32(m: &DMatrix<f64>) -> Vec<f32> {
    let n = m.nrows();
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = m[(i, j)] as f32;
        }
    }
    out
}
```
nalgebra is column-major, but `DMatrix[(i,j)]` indexes by (row, col) regardless of storage order.

### Eigenvector sign handling
Eigenvectors are defined up to ±1. Two approaches:
1. **Dot product**: For each eigenvector pair, compute `dot = v_cpu · v_gpu`. If `dot < 0`, flip `v_gpu`. Then compare elementwise.
2. **Absolute value**: Compare `|v_cpu|` vs `|v_gpu|` elementwise (less strict, doesn't verify alignment).

Use approach 1 (dot product) for correctness.

### Eigenvalue sorting
GPU Jacobi does NOT sort eigenvalues. Must sort on host after readback:
```rust
let mut idx: Vec<usize> = (0..n).collect();
idx.sort_by(|&a, &b| eigvals_gpu[a].partial_cmp(&eigvals_gpu[b]).unwrap());
```

### S^{-1/2} computation (host-side, from eigendecomposition)
```rust
fn compute_s_inv_half(eigvals: &[f64], eigvecs: &DMatrix<f64>) -> DMatrix<f64> {
    let n = eigvals.len();
    let mut d = DMatrix::zeros(n, n);
    for i in 0..n {
        d[(i, i)] = 1.0 / eigvals[i].sqrt().max(1e-12);
    }
    // S^{-1/2} = V · D^{-1/2} · V^T
    eigvecs * d * eigvecs.transpose()
}
```

### Tolerances
- f32 vs f64: expect ~1e-4 to 1e-5 relative error for eigenvalues
- Eigenvectors: ~1e-3 to 1e-4 (sensitive to eigenvalue gaps)
- GEMM: ~1e-5 relative (well-conditioned)

### Test guard
Tests should skip gracefully if:
- No OpenCL device available (catch `GpuMatrixContext::new` error)
- No SK data directory (same pattern as existing tests: `let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; }`)

### Matrix size considerations
- H2: N=2 (trivial, good for initial debug)
- N2: N=8 (still small, good for Jacobi)
- HCOOH: N=14 (realistic small molecule)
- All fit within `jacobi_max_m=64` default
- For N>64, would need block Jacobi (not yet implemented)

### What's NOT tested (future work)
- Block Jacobi for N>64 (no driver yet, only subproblem kernel exists)
- Purification path (density matrix only, no orbitals)
- SCC cycle on GPU (requires H assembly + shifts + gamma on GPU)
- GPU-side S^{-1/2} reconstruction (currently done on host)
- Batched operation with different N per fragment (would need padding)

---

## Implementation Report

### What was implemented

**Test file**: `rust_dftb/tests/gpu_diagonalization.rs` — 8 tests, all passing.

**Helper functions:**
- `dmatrix_to_row_major_f32` — convert nalgebra DMatrix<f64> to row-major Vec<f32> for GPU upload
- `row_major_f32_to_dmatrix` — convert GPU readback back to nalgebra DMatrix<f64>
- `cpu_symeig` — CPU reference eigendecomposition via nalgebra SymmetricEigen, sorted ascending
- `compute_s_inv_half` — compute S^(-1/2) = V · Λ^(-1/2) · V^T from eigendecomposition
- `sort_gpu_eig` — sort GPU eigenvalues ascending and reorder eigenvectors accordingly
- `align_eigenvector_signs` — fix ±1 ambiguity by flipping GPU eigenvectors when dot product with CPU is negative
- `try_gpu_ctx` — create GpuMatrixContext or return None (skip test if no OpenCL device)
- `gpu_diagonalize` — full GPU pipeline: S^(-1/2) → Löwdin transform → Jacobi → back-transform

**Tests implemented:**

| # | Test name | Description | Status |
|---|-----------|-------------|--------|
| 1 | `test_gpu_gemm_correctness` | 4×4 matrix multiply, GPU vs CPU | PASS |
| 2 | `test_gpu_jacobi_synthetic` | 8×8 symmetric matrix, both serial + parallel Jacobi variants | PASS |
| 3 | `test_gpu_jacobi_overlap_h2` | Diagonalize S from H2 (2×2), compute and compare S^(-1/2) | PASS |
| 4 | `test_gpu_jacobi_overlap_n2` | Diagonalize S from N2 (8×8), compute and compare S^(-1/2) | PASS |
| 5 | `test_gpu_lowdin_transform` | H' = X^T·H·X on GPU (N2, two GEMMs) vs CPU | PASS |
| 6 | `test_gpu_full_diagonalization_h2` | Full HC=SCε pipeline: Löwdin → Jacobi → back-transform (H2) | PASS |
| 7 | `test_gpu_full_diagonalization_n2` | Full HC=SCε pipeline (N2, 8 orbitals) | PASS |
| 8 | `test_gpu_batched_jacobi` | 3 scaled copies of 4×4 matrix diagonalized in one GPU launch | PASS |

### Test results

```
running 8 tests
test test_gpu_full_diagonalization_n2 ... ok
test test_gpu_jacobi_overlap_n2 ... ok
test test_gpu_jacobi_overlap_h2 ... ok
test test_gpu_lowdin_transform ... ok
test test_gpu_full_diagonalization_h2 ... ok
test test_gpu_batched_jacobi ... ok
test test_gpu_gemm_correctness ... ok
test test_gpu_jacobi_synthetic ... ok

test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.70s
```

All tests pass with tolerances:
- Eigenvalues: < 1e-4 max abs diff (f32 GPU vs f64 CPU)
- Eigenvectors: < 1e-3 max abs diff (after sign alignment)
- GEMM: < 1e-4 max abs diff
- S^(-1/2): < 1e-3 max abs diff

### What remains to be implemented or tested

1. **HCOOH test (N=14)** — planned in original plan but not implemented yet; would test a larger realistic molecule with mixed species (H, C, O)
2. **Block Jacobi for N>64** — no driver exists; only the 2T×2T subproblem kernel and Brent-Luk scheduling are implemented; needs a host-side driver that orchestrates block rotations via GEMM
3. **Purification path** — McWeeny and TC2 kernels exist but are untested; should compare GPU-purified density matrix D with CPU-computed D from diagonalization
4. **SCC cycle on GPU** — requires GPU Hamiltonian assembly (H0 + SCC shifts), gamma function evaluation, and charge computation on GPU; none of these are wired up yet
5. **GPU-side S^(-1/2) reconstruction** — currently S^(-1/2) is computed on CPU from GPU-returned eigenvalues/vectors; could be done entirely on GPU using batched GEMM (V · Λ^(-1/2) · V^T)
6. **Batched operation with different N per fragment** — current batched test uses same-N fragments; mixed sizes would need padding or separate kernel launches
7. **Parallel Jacobi for larger matrices** — `local_jacobi_blocks_parallel` is tested on 8×8; should test on larger matrices (e.g. 16×16, 32×32) to verify the parallel rotation logic
8. **f64 precision path** — all GPU operations are f32; for production DFTB calculations, f64 may be needed (would require OpenCL double-precision support or emulation)
