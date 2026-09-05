# H-Bond Switching: Formic Dimer & Azaindole Dimer

## Objective

Validate the GPU DFTB solver against CPU reference on real chemical systems
with hydrogen-bond switching. The target reactions are double proton transfer
in H-bonded dimers, mapped as 2D potential energy surfaces (PES) where the
two coordinates are the positions of the two transferring protons.

## Systems

### 1. Formic acid dimer (HCOOH)₂ — small, fits one workgroup

- File: `data/xyz/formic_dimer.xyz`
- 10 atoms: 4 H, 2 C, 4 O
- **28 orbitals** (mio-1-1: H=1s, C/O=2s2p)
- Fits N≤64 budget — one workgroup per system on GPU
- Reaction: double proton transfer along the two O–H···O hydrogen bonds
- Symmetric: both H atoms transfer simultaneously in the 1D case
- The two H-bond axes are approximately along z; H atoms move in z

### 2. Formic acid + 7-azaindole mixed dimer — fits one workgroup

- Formic acid (HCOOH, 5 atoms, 14 orbs) H-bonded to 7-azaindole (15 atoms, 42 orbs)
- **56 orbitals total** — fits N≤64 in one workgroup
- Geometry: needs to be constructed (place formic acid near an N–H or N··· site of azaindole)
- Reaction: proton transfer between formic acid O–H and azaindole N (or vice versa)
- This gives a real H-bond switching system at a size the dense route can handle
- File: `data/xyz/formic_azaindole_dimer.xyz` (to be constructed)

### 3. 7-Azaindole dimer — large, exceeds one workgroup

- File: `data/xyz/azaindol_dimer.xyz`
- 30 atoms: 12 H, 14 C, 4 N
- **84 orbitals** — exceeds N≤64 single-workgroup limit
- Requires either:
  - (a) Multi-workgroup dense Jacobi (split N across WGs), or
  - (b) Sparse BSR4 purification route (no diagonalization)
- Monomer: 15 atoms, 42 orbitals — fits one workgroup
- Reaction: double proton transfer along N–H···N hydrogen bonds

### Orbital budget summary

| System | Atoms | Orbitals | Fits N≤64? | Route |
|---|---|---|---|---|
| Formic acid monomer | 5 | 14 | yes | dense, 1 WG |
| Formic dimer | 10 | 28 | yes | dense, 1 WG |
| Azaindole monomer | 15 | 42 | yes | dense, 1 WG |
| Formic + azaindole mixed dimer | 20 | **56** | **yes** | dense, 1 WG |
| Azaindole dimer | 30 | 84 | **no** | multi-WG dense or sparse BSR4 |

## Scan geometry

### 1D synchronous scan (both H atoms move in reverse, same t)

For formic dimer: the two transferring H atoms move in **opposite
directions** along their respective O–H···O axes, but share the same
progress parameter `t ∈ [0, 1]`:

```
  O···H-O          O-H···O       (H on right → H on left)
  |     |    →     |     |
  O-H···O          O···H-O       (H on left  → H on right)
```

- `t=0`: H(4) bonded to O(3), H(9) bonded to O(8) — reactant
- `t=0.5`: both H atoms at the O–O midpoint — transition state
- `t=1`: H(4) bonded to O(7), H(9) bonded to O(2) — product

The two protons swap sides simultaneously. H(4) goes O(3)→O(7) (left to
right), H(9) goes O(8)→O(2) (right to left). The interpolation is linear
between the donor and acceptor O positions.

### 2D asynchronous scan (two independent H coordinates)

Two coordinates `(t1, t2)` where `t1` controls H(4) position (O(3)→O(7))
and `t2` controls H(9) position (O(8)→O(2)) independently. This reveals:
- Correlation between the two proton transfers
- Whether the reaction is concerted (diagonal path t1=t2) or stepwise
  (L-shaped path through corners)
- The transition state location on the 2D PES

Grid: `t1 × t2 ∈ [0, 1] × [0, 1]`, e.g. 21×21 = 441 points.

The synchronous 1D path is the diagonal `t1 = t2 = t` on the 2D map.

## Computation modes

### Mode A: Non-SCC (fastest, electronic structure only)

- Build H0/S, diagonalize, fill occupancy, compute energy
- No charge self-consistency
- GPU: `GpuDriver::gpu_assemble_batched` → `jacobi_cyclic_local_batched` → density
- This is the simplest GPU pipeline and tests H/S assembly + eigensolve

### Mode B: SCC (self-consistent charges)

- Full SCC iteration: H0/S → diagonalize → charges → gamma → H_scc → repeat
- GPU pipeline (Wave 3 integration, not yet implemented):
  1. `GpuDriver::gpu_assemble_batched` — H0/S
  2. `jacobi_cyclic_local_batched` — eigendecomposition
  3. `mulliken_charges_batched` — charges from density
  4. `gamma_matvec_batched` — V = G·Δq
  5. `h_scc_update_batched` — H = H0 + 0.5·S·(V_i + V_j)
  6. `residual_and_mix_batched` — convergence check + mixing
  7. Repeat 2–6 until converged
- This is the primary target for GPU parity validation

### Mode C: Relaxed scan (most expensive)

- For each `(t1, t2)` point: fix the y-component of the two transferring H
  atoms (constraining only the H-bond coordinate), relax all other DOFs
- Requires forces: `compute_scc_forces` (CPU) or GPU force kernel (not yet
  implemented)
- Optimization: L-BFGS or simple gradient descent with constraint projection
- This is the most chemically meaningful but also the most complex
- **Defer until GPU forces are implemented**

## Task breakdown

### Task 1: CPU reference cache (do first)

**Goal:** Compute and cache all CPU DFTB reference data so we never recompute it.

**What to save per scan point:**
- Geometry (xyz)
- H0 matrix (non-SCC Hamiltonian)
- S matrix (overlap)
- H_scc matrix (SCC-converged Hamiltonian, if SCC)
- C matrix (MO coefficients / eigenvectors)
- eigenvalues
- density matrix D
- Mulliken charges
- total energy (electronic + repulsive)
- number of SCC iterations
- (if relaxed) final relaxed geometry

**Cache format:**
- One directory per scan: `cache/<system>_<mode>_<scan_type>/`
- One subdirectory per point: `point_XXXX/`
- Files: `geom.xyz`, `h0.dat`, `s.dat`, `h_scc.dat`, `c.dat`, `eigvals.txt`, `density.dat`, `charges.txt`, `energy.txt`
- Use `DftbOutput::write_square` format (`n n` header + n rows)

**Implementation:**
- New file: `rust_dftb/examples/cpu_ref_cache.rs`
- CLI: `--xyz <file> --mode {non-scc,scc} --scan {1d,2d} --n <N> [--n2 <N2>] --out <dir>`
- For 1D: `--bond1 <i> <j> <k> <l>` (H1 moves from atom j to atom l along i-k axis, same for H2)
  - Actually simpler: `--h1 <atom_i> <atom_j>` and `--h2 <atom_k> <atom_l>` where H1 is interpolated between positions of atom_i and atom_j
  - Or: specify reactant and product xyz files, interpolate H positions
- For 2D: same but with independent t1, t2
- Skip if cache already exists (check for `energy.txt` in point dir)

**Validation:**
- `tests/hbond_ref.rs` — test that cache exists, files parse, energies are finite, H2 formic dimer scan has a barrier

### Task 2: GPU non-SCC parity (1D formic dimer)

**Goal:** Reproduce CPU non-SCC reference on GPU for 1D formic dimer scan.

**Steps:**
1. Load cached CPU reference for formic dimer 1D non-SCC scan (N points)
2. For each point: build H0/S on GPU via `GpuDriver::gpu_assemble_batched`
3. Diagonalize on GPU via `jacobi_cyclic_local_batched`
4. Build density matrix on GPU (need a density kernel — use Agent_6's `matmul_full_local_batched` or a dedicated occupancy kernel)
5. Compare: H0, S, eigenvalues, eigenvectors, density, energy vs CPU cache
6. Report max discrepancies per quantity

**Tolerances (f32):**
- H0, S: < 1e-5 (already achieved)
- Eigenvalues: < 1e-4
- Eigenvectors: < 1e-3 (degeneracy-robust check)
- Density: < 1e-3
- Energy: < 1e-4 Ha

**Implementation:**
- New file: `rust_dftb/tests/hbond_gpu_non_scc.rs`
- Batch all scan points into one `GpuBatch` (28 orbitals × ~20 points = very manageable)
- This is the first **end-to-end** GPU test on a real molecule

### Task 3: GPU SCC parity (1D formic dimer)

**Goal:** Reproduce CPU SCC reference on GPU for 1D formic dimer scan.

**This requires the Wave 3 SCC loop integration (not yet done).** The task is
blocked until the coordinator integrates Agent_4 + Agent_6 kernels into
`gpu_driver.rs` with a device-resident SCC loop.

**Steps (after Wave 3):**
1. Load cached CPU reference for formic dimer 1D SCC scan
2. Run GPU SCC loop for all scan points (batched)
3. Compare: H_scc, charges, energy, density vs CPU cache
4. Report max discrepancies and SCC iteration counts

**Tolerances (f32):**
- H_scc: < 1e-3 (SCC amplifies f32 rounding through iterations)
- Charges: < 1e-3
- Energy: < 1e-4 Ha
- Density: < 1e-3

### Task 4: GPU 2D scan (formic dimer, non-SCC then SCC)

**Goal:** Full 2D PES on GPU.

**Steps:**
1. CPU cache: 21×21 = 441 points (non-SCC and SCC)
2. GPU batched: all 441 points in one batch (28×441 = 12,348 orbital rows — well within GPU memory)
3. Plot 2D PES (energy as function of t1, t2)
4. Compare GPU vs CPU PES: max energy difference, RMS difference
5. Identify transition state region on the PES

**Visualization:**
- `examples/plot_pes.py` — read CSV, plot 2D contour using matplotlib
- Output: `pes_gpu.png`, `pes_cpu.png`, `pes_diff.png`

### Task 5: Azaindole dimer (large system)

**Goal:** Test the system that exceeds N≤64.

**Blocked on:** either multi-workgroup dense Jacobi or sparse BSR4 purification.

**Option A: Multi-WG dense**
- Split the 84×84 eigensolve across multiple workgroups
- Requires a different Jacobi kernel that uses global memory + cooperation
- More complex but reuses the dense algebra

**Option B: Sparse BSR4 purification**
- Use the `methods/sparse` route (no diagonalization)
- Atom-block CSR with 4×4 blocks
- Generalized McWeeny purification (3KSK - 2KSKSK)
- Masked sparse products — never densify
- This is the architecturally preferred route for large systems

**Recommendation:** Start with Option B (sparse) since the infrastructure
is already being built in `methods/sparse/`. The azaindole dimer (84 orbs,
30 atoms) is a good test case — large enough to need the sparse route, small
enough to debug.

### Task 6: Relaxed scan (deferred)

**Goal:** Constrained relaxation at each scan point.

**Blocked on:** GPU force kernel implementation.

**Steps (after GPU forces):**
1. For each (t1, t2): fix y-positions of the two transferring H atoms
2. Relax all other DOFs using GPU forces
3. Compute energy at the relaxed geometry
4. This gives the "relaxed PES" which is chemically more meaningful than the rigid PES

**Deferred until:**
- GPU force kernel is implemented
- Constraint projection is designed
- L-BFGS or FIRE optimizer is available on GPU

## Execution order

```
Task 1 (CPU cache)          ← do now, no GPU needed
  ↓
Task 2 (GPU non-SCC 1D)     ← do now, uses existing GPU kernels
  ↓
Task 3 (GPU SCC 1D)         ← blocked on Wave 3 SCC integration
  ↓
Task 4 (GPU 2D scan)        ← after Task 3
  ↓
Task 5 (Azaindole dimer)    ← blocked on sparse route or multi-WG
  ↓
Task 6 (Relaxed scan)       ← blocked on GPU forces
```

## File ownership

| File | Owner | Status |
|---|---|---|
| `examples/cpu_ref_cache.rs` | coordinator | new |
| `tests/hbond_ref.rs` | coordinator | new |
| `tests/hbond_gpu_non_scc.rs` | coordinator | new |
| `tests/hbond_gpu_scc.rs` | coordinator | new (after Wave 3) |
| `examples/plot_pes.py` | coordinator | new |
| `data/xyz/formic_dimer.xyz` | existing | ready |
| `data/xyz/azaindol_dimer.xyz` | existing | ready |

## Physical context

### Formic acid dimer double proton transfer

The formic acid dimer is the prototypical system for studying double proton
transfer in hydrogen bonds. It has C₂h symmetry with two equivalent O–H···O
hydrogen bonds. The two H-bond protons transfer in **opposite directions**
(swapping sides):

```
  O···H-O          O-H···O       (top: H moves right→left)
  |     |    →     |     |
  O-H···O          O···H-O       (bottom: H moves left→right)
```

At t=0 the top H is on the right monomer and the bottom H is on the left
monomer. At t=1 they have swapped. The synchronous 1D path (t1=t2) is the
diagonal on the 2D PES.

The 2D PES (with t1, t2 as the two H-bond coordinates) reveals whether the
transfer is concerted (diagonal path t1=t2) or stepwise (L-shaped path
through the corners (0,1) or (1,0), where one H has transferred but the
other hasn't).

For the symmetric formic dimer, the 1D synchronous path goes through a
transition state at t=0.5. The 2D map should show:
- Two equivalent minima at (0,0) and (1,1) — reactant and product
- A saddle point at (0.5, 0.5) — concerted transition state
- Possibly metastable intermediates at (0,1) and (1,0) — zwitterionic
  structures where only one proton has transferred (if the stepwise
  mechanism is accessible)

### 7-Azaindole dimer

7-Azaindole dimer is a larger system with N–H···N hydrogen bonds. It is
important in photochemistry (excited-state proton transfer) and as a model
for DNA base pairs. The double proton transfer mechanism has been debated
(concerted vs stepwise).

## Notes

- The formic dimer xyz has fractional charges in columns 5-6 (Mulliken/ESP
  charges from a prior calculation). These are NOT used by DFTB — DFTB
  computes its own charges via SCC. The xyz parser should ignore extra
  columns.
- The formic dimer in the xyz file is already in the reactant geometry
  (both H-bond protons on one side). The product geometry is obtained by
  mirroring the H positions across the midpoint of each O–O pair.

### Atom indices (0-based) for formic dimer

```
 0 H     3.1688    -0.0000     0.0637   ← C-H proton (does NOT transfer)
 1 C     2.0720    -0.0000    -0.0331
 2 O     1.4616     0.0000     1.0149   ← acceptor for H(9)
 3 O     1.4621     0.0000    -1.1847   ← donor for H(4)
 4 H     0.5000     0.0000    -1.0000   ← TRANSFERRING H #1 (O(3)→O(7))
 5 H    -3.1667    -0.0000    -0.0784   ← C-H proton (does NOT transfer)
 6 C    -2.0711    -0.0000     0.0304
 7 O    -1.4598    -0.0000    -1.0170   ← acceptor for H(4)
 8 O    -1.4625    -0.0000     1.1829   ← donor for H(9)
 9 H    -0.5000     0.0000     1.0000   ← TRANSFERRING H #2 (O(8)→O(2))
```

- **H(4)** transfers from O(3) to O(7) (left→right along z)
- **H(9)** transfers from O(8) to O(2) (right→left along z) — **opposite direction**
- H(0) and H(5) are C-H protons, they do NOT transfer
- For the scan: interpolate H(4) position between O(3) and O(7), and
  H(9) position between O(8) and O(2)
- Reactant (t=0): H(4) near O(3), H(9) near O(8) — protons on "outer" sides
- Product  (t=1): H(4) near O(7), H(9) near O(2) — protons swapped to "inner" sides
- The two protons move in opposite directions but share the same parameter t
  in the 1D synchronous scan

## Status (2025-09-05)

### Reactant optimization — DONE

Ran FIRE geometry optimization on the formic-acid/7-azaindole mixed dimer
(20 atoms, 56 orbitals) using the CPU dense SCC path with LAPACK eigensolver.

- **Example**: `rust_dftb/examples/hbond_ref.rs`
- **Mode**: `--mode optimize` (FIRE optimizer)
- **Steps**: 100, **Time**: 32.8s (after LAPACK fix, was 75s with nalgebra)
- **Energy**: -29.77 → -32.11 Hartree (ΔE = -2.34)
- **max|F|**: 4.73 → 0.049 (did not reach f_tol=5e-3, needs more steps or LBFGS)
- **SCC**: 16-18 iterations per step, tol=1e-7, max-iter=50
- **Trajectory**: `debug/hbond_switching/reactant_traj.xyz` (100 frames)
- **Plot**: `debug/hbond_switching/reactant_opt_trajectory.png`

### Performance bottleneck — IDENTIFIED AND PARTIALLY FIXED

**Root cause**: nalgebra's `SymmetricEigen` (Jacobi algorithm) took 20ms for a
56×56 matrix — 29× slower than LAPACK `dsyevd` (0.7ms). With 16 SCC iterations
per step, this was 320ms of pure eigensolve overhead per FIRE step.

**Fix applied**: Replaced nalgebra Jacobi with LAPACK `dsyevd` via the `lapack`
crate + system OpenBLAS. 2.3× total speedup.

**Remaining bottlenecks** (see `doc/prokop/topical_audit/eigensolver_performance.md`):
1. No SCC warm start — 16 iter → should be 3-4 (4× potential speedup)
2. nalgebra triangular solves — 7.6ms/iter (could use LAPACK `dtrtrs`)
3. Template rebuild every step — 8ms (SystemContext/GammaTable cacheable)
4. Force evaluation — 100ms (not yet profiled)

### Next steps

1. **SCC warm start** — pass previous charges as initial guess → 4× fewer iters
2. **LAPACK triangular solves** — replace nalgebra `solve_lower_triangular`
3. **Profile forces** — 100ms/step is suspicious, likely similar nalgebra overhead
4. **Product geometry** — optimize from a switched-H guess
5. **1D scan** — interpolate H positions between reactant and product, compute PES

