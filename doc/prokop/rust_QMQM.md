# USER

now analyse this structure from point of view of making multi-system solver. I wanted to do something like QM/QM solver which means that I have multiple non-covalently bonded fragments in the system. I do not want to diagonalize the whole system as it it, but rather diagonalize hamiltonina of each system separately, and treat interactions between system as perturbation, mostly just electrostatic field, where the system mutually polarize each other.  To do so we should make architecture which will allow us keep in memory encapsulated objects represeting all quentities related to each sub-system (frament) and manage them efficiently (creat new sub system, solve each sub-system separately). Can  you analuse how to do this efficiently starting from current architecture? 

do not code yet, make throughfull analysis and define key desing decissions I have to make. make sure it will be efficient and scaleable desing to thousands of sub-systems, efficien ency is paramout, read notes about efficiency in @rust_DFTB_optimization.md 

---

# Kimi-2.6

Here is the architectural analysis for a multi-fragment QM/QM solver, built on top of our optimized Rust DFTB core.


## 1. Core Constraint: Never Assemble the Global Hamiltonian

The entire system's Hamiltonian is **never materialized as a single matrix**. For 1000 water molecules (300 atoms, ~2100 orbitals), a global dense H would be 2100×2100 ≈ 35 MB — still small, but for 10,000 waters it is 3.5 GB and diagonalization is O(N³). Instead:

- Each fragment has its own **dense** H and S (small: 21×21 for water).
- The solver orchestrates fragment-by-fragment diagonalization.
- Inter-fragment coupling enters only as a **shift to onsite energies** (SCC-like).


## 2. Fragment Encapsulation

Each fragment is a self-contained unit that owns everything needed for its own diagonalization, but references immutable global data.

```rust
pub struct Fragment {
    // --- Geometry (owned, fragment-specific) ---
    pub coords: Vec<[f64; 3]>,
    pub species: Vec<u8>,          // indices into global species table
    
    // --- Precomputed lookups (owned, built once) ---
    pub ctx: SystemContext,        // integer-indexed: no HashMap in hot loop
    
    // --- Dense matrices (owned, sized to fragment) ---
    pub h0: DMatrix<f64>,           // non-SCC Hamiltonian (constant)
    pub s: DMatrix<f64>,            // overlap (constant)
    pub h_scc: DMatrix<f64>,        // H0 + external SCC shift (updated each iteration)
    
    // --- Electronic structure (updated each SCC step) ---
    pub eigenvalues: DVector<f64>,
    pub eigenvectors: DMatrix<f64>,
    pub density: DMatrix<f64>,
    pub fermi_level: f64,
    
    // --- Properties (updated each SCC step) ---
    pub charges: Vec<f64>,          // per-atom charges
    pub dipole: [f64; 3],
}
```

**Key design decision**: Fragments own their matrices. Do not use a global arena or pool for fragment matrices. The per-fragment memory is tiny (a few KB), and ownership makes the API clean. However, see Section 5 for how to avoid per-SCC-step reallocation.


## 3. Fragment Templating (Critical for Efficiency)

If your system contains many chemically identical fragments (e.g., 5000 water molecules, 1000 identical ligands), you must exploit this.

**Template** = everything that is independent of geometry:
- Species list and ordering
- [SystemContext](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:16:0-36:1) (species indices, pair LUTs, angular shells)
- [SkData](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:208:0-213:1) references
- Matrix dimensions

**Instance** = geometry-specific data:
- Coordinates
- Current density matrix, charges, dipole

```rust
pub struct FragmentTemplate {
    pub species: Vec<u8>,
    pub ctx: SystemContext,         // precomputed once
    pub n_orb: usize,
    pub n_atom: usize,
}

pub struct FragmentInstance {
    pub template_id: usize,         // index into solver.templates
    pub coords: Vec<[f64; 3]>,
    pub charges: Vec<f64>,
    pub dipole: [f64; 3],
    pub eigenvalues: DVector<f64>,
    pub eigenvectors: DMatrix<f64>,
    pub density: DMatrix<f64>,
}
```

**Memory impact**: 5000 water fragments without templating = 5000 copies of [SystemContext](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:16:0-36:1) (each ~few hundred bytes). With templating = 1 copy of [SystemContext](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:16:0-36:1) + 5000 sets of coordinates and matrices. For 5000 × 21 orbitals, the matrices dominate anyway, but templating saves setup time.

## 4. The Multi-System Solver

```rust
pub struct MultiSystemSolver<'a> {
    // Global immutable data
    pub sk: &'a SkData,
    
    // Fragment templates (shared)
    pub templates: Vec<<FragmentTemplate>,
    
    // Fragment instances (owned)
    pub fragments: Vec<<FragmentInstance>,
    
    // --- Inter-fragment coupling ---
    pub frag_centroids: Vec<[f64; 3]>,
    pub frag_neighbors: Vec<Vec<<usize>>,  // spatial neighbor list on centroids
    pub frag_charges: Vec<f64>,            // flat: concatenated per-atom charges
    
    // --- Global SCC state ---
    pub scc_history: Vec<Vec<f64>>,        // Broyden/DIIS charge history
    pub external_potential: Vec<f64>,     // flat: V_ext per atom
    
    // --- Pre-allocated workspace ---
    pub ws: FragmentWorkspace,
}
```

### API:
```rust
impl<'a> MultiSystemSolver<'a> {
    pub fn new(sk: &'a SkData) -> Self;
    
    // Register a template (e.g., "water", "amino_acid")
    pub fn add_template(&mut self, species: &[String]) -> usize;
    
    // Add an instance (geometry only)
    pub fn add_fragment(&mut self, template_id: usize, coords: &[[f64; 3]]);
    
    // Build H0 and S for all fragments (embarrassingly parallel)
    pub fn build_all_non_scc(&mut self) -> Result<<()>;
    
    // Solve all fragments independently (no SCC)
    pub fn solve_all_non_scc(&mut self) -> Result<<()>;
    
    // SCC with inter-fragment polarization
    pub fn solve_scc(&mut self, opts: &SccOptions) -> Result<<()>;
}
```

## 5. Efficiency: Zero Allocations Per SCC Step

Following the optimization principles, **nothing should be allocated inside the SCC loop**. The current [build_non_scc](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:22:4-69:5) allocates [HWorkspace](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:122:0-129:1) and `DMatrix` each call. For 10,000 fragments × 50 SCC steps, this is 500,000 allocations.

**Solution: Pre-sized workspace in the solver**

```rust
pub struct FragmentWorkspace {
    // Max fragment size encountered across all templates
    pub max_n_orb: usize,
    
    // Reusable flat buffers for the largest fragment
    pub block_h: Vec<f64>,
    pub block_s: Vec<f64>,
    
    // Reusable dense matrices (sized to max fragment)
    pub h_buf: DMatrix<f64>,
    pub s_buf: DMatrix<f64>,
    pub h_scc_buf: DMatrix<f64>,
    pub evecs_buf: DMatrix<f64>,
    pub evals_buf: DVector<f64>,
    pub dens_buf: DMatrix<f64>,
}
```

Each fragment uses slices/views into these buffers, or we use a `bumpalo`-style arena if fragments have wildly different sizes. But for simplicity, size the workspace to the **largest fragment** and reuse it for all.

**Even better**: If most fragments are the same size (e.g., all water), the workspace is exactly sized and there is zero waste.

## 6. Inter-Fragment Electrostatics: The Key Physics Design Decision

You have three options, in order of increasing accuracy and cost:

### Option A: Pure Point-Charge (Fastest)
Each fragment is a point charge `Q_frag = Σ_A (q_A - q0_A)`. The external potential on atom A in fragment F is:
```
V_ext(A) = Σ_{G ≠ F} Q_G / R_FG
```
This is O(N_frag²) naively, but O(N_frag × n_neigh) with a fragment neighbor list. No γ_AB needed.

**When to use**: Fragments are far apart (>10 Å), or for a first-pass polarization.

### Option B: Atomistic γ with Cutoff (Standard)
For fragments within a cutoff `R_cut`, compute the full SCC shift:
```
V_ext(A in F) = Σ_{G ≠ F, R_FG < R_cut} Σ_{B in G} γ_AB (q_B - q0_B)
```
For fragments beyond `R_cut`, use Option A or ignore.

**When to use**: Close fragments where charge penetration matters. This is essentially standard DFTB SCC restricted to inter-fragment pairs.

### Option C: Multipole Expansion (Intermediate)
Each fragment contributes point charge + dipole + quadrupole. The potential is:
```
V_ext(r) = Σ [Q/|r-R| + μ·∇(1/|r-R|) + ...]
```

**Recommendation**: Start with **Option B with a cutoff**. It reuses our existing SCC infrastructure (`gamma_AB` table) and is physically correct for close fragments. The cutoff makes it O(N_frag) with spatial hashing.

## 7. SCC Iteration Strategy: Jacobi vs. Gauss-Seidel

### Jacobi (Parallel, Recommended)
```
For each SCC step:
  1. Compute external potential on all fragments using charges from step k
  2. In parallel: diagonalize all fragments independently (H0 + V_ext)
  3. In parallel: compute new charges from new eigenvectors
  4. Mix new charges with old (Broyden/DIIS)
```

**Pros**: Embarrassingly parallel step 2. Easy to implement with Rayon.  
**Cons**: Slower convergence than Gauss-Seidel.

### Gauss-Seidel (Sequential)
Update fragments one by one, using the most recent charges from fragments already updated in the current iteration.

**Pros**: Faster convergence.  
**Cons**: Sequential bottleneck. Not worth it for thousands of fragments.

**Decision**: Use **Jacobi + global Broyden mixing**. The parallel speedup dominates the slightly slower convergence.

## 8. Spatial Data Structure for Fragment Neighbors

To avoid O(N_frag²) inter-fragment coupling:

```rust
pub struct FragmentNeighborList {
    pub cutoff: f64,
    pub cells: HashMap<(i32, i32, i32), Vec<<usize>>,  // spatial hashing
    pub neighbors: Vec<Vec<<usize>>,
}

// Build once at initialization, update if fragments move (MD)
fn build_frag_neighbors(centroids: &[[f64; 3]], cutoff: f64) -> FragmentNeighborList;
```

For static geometries (e.g., a protein with fixed fragments), build once. For MD, rebuild every N steps or use a cell-list update.

## 9. Scaling Analysis

| Property | Single Global System | Multi-Fragment (this design) |
|---|---|---|
| Memory | O(N_orb²) dense | O(N_frag × n_orb_frag²) |
| Diagonalization | O(N_orb³) | O(N_frag × n_orb_frag³) |
| SCC build | O(N_pairs) | O(N_frag × n_neigh_frag × n_atom_frag²) |
| Parallelism | Limited (LAPACK threading) | Embarrassingly parallel at fragment level |

**Example: 10,000 water molecules**
- Global: 2100×2100 dense, diag ~20s (single core), memory 35 MB.
- Multi-fragment: 10,000 × 21×21 dense, diag ~0.1s (parallel), memory ~10 MB.
- Inter-fragment coupling: ~10 neighbors per fragment × 3 atoms = 30K γ evaluations per step. Trivial.

**Example: 1000 protein residues (50 atoms each, ~200 orbitals)**
- Global: 200K orbitals → dense matrix impossible.
- Multi-fragment: 1000 × 200×200 = 40 MB. Diagonalization: 1000 × 8ms = 8s (parallel). Feasible.

## 10. Concrete Design Decisions to Make

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 1 | Fragment matrix format | Dense (per fragment) vs Sparse (per fragment) | **Dense** for now. Sparse only if fragments >100 atoms. |
| 2 | Templating | Yes / No | **Yes**. Essential for identical fragments. |
| 3 | Workspace reuse | Per-fragment owned matrices vs solver-level arena | **Solver-level arena** (`FragmentWorkspace`) to eliminate allocations. |
| 4 | Inter-fragment coupling | Point-charge / Atomistic γ / Multipole | **Atomistic γ with cutoff** (Option B). Reuses existing SCC code. |
| 5 | SCC update order | Jacobi (parallel) / Gauss-Seidel (sequential) | **Jacobi + global Broyden**. Parallelism wins. |
| 6 | Fragment neighbor list | Full O(N²) / Cell list / KD-tree | **Cell list (spatial hash)**. Rebuild on demand for MD. |
| 7 | Charge mixing | Simple mixing / Broyden / DIIS | **Broyden** on the global flattened charge vector. |
| 8 | Geometry updates | Fragments fixed / Rigid body / Fully flexible | Start with **fixed fragments**, add rigid/flexible later. |

## 11. Implementation Roadmap (After Current Optimizations)

1. **Finish Phase 1–3 of current optimization** ([SystemContext](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:16:0-36:1), zero-allocation hot loop, [HWorkspace](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:122:0-129:1)).
2. **Add `FragmentTemplate` + `FragmentInstance`** structs and a `build_fragment_non_scc` method that uses pre-allocated workspace.
3. **Add fragment-level diagonalization** (wrap `nalgebra::SymmetricEigen` or LAPACK).
4. **Add `FragmentNeighborList`** (spatial hashing on centroids).
5. **Add inter-fragment SCC shift** (`gamma_AB` lookup restricted to neighboring fragments).
6. **Add global Broyden mixer** on the flattened charge vector.
7. **Add `MultiSystemSolver::solve_scc()`** that loops: build V_ext → diag all fragments → update charges → Broyden mix → check convergence.
8. **Performance validation**: Verify that 10,000 water molecules diagonalizes in <1s on a workstation.

Do you want me to:
- **(A)** Continue with the current single-system optimization (Phase 1–3), then come back to multi-system?
- **(B)** Start designing the `FragmentTemplate` / `FragmentInstance` structures now?
- **(C)** Dive deeper into the inter-fragment γ-coupling and Broyden mixer design?

---

## 1. Physics of Inter-Fragment Coupling: What We Actually Need

From the Fortran analysis ([scc.F90](cci:7://file:///home/prokophapala/git/dftbplus/src/dftbp/dftb/scc.F90:0:0-0:0), [shortgamma.F90](cci:7://file:///home/prokophapala/git/dftbplus/src/dftbp/dftb/shortgamma.F90:0:0-0:0), [shift.F90](cci:7://file:///home/prokophapala/git/dftbplus/src/dftbp/dftb/shift.F90:0:0-0:0)), the SCC Hamiltonian is:

```fortran
H = H0 + S * 0.5 * (shift(A) + shift(B))   per atom pair block
```

where `shift` on atom A is the electrostatic potential from all other atoms:

```
shift(A) = Σ_B γ_AB(R_AB) * (q_B - q0_B)
```

### For a QM/QM multi-fragment system:

- **Intra-fragment** γ: handled by each fragment's own dense H/S diagonalization. This is standard DFTB SCC confined to the fragment.
- **Inter-fragment** γ: treated as an **external potential** applied to each fragment before diagonalization. There is no orbital overlap between fragments (they are non-bonded), so no off-diagonal H/S blocks.

**Key simplification:** The inter-fragment interaction is purely classical electrostatics via the γ function. No SK rotation, no overlap matrix, no sparse indexing.

## 2. The γ Function: What to Compute

DFTB+ stores `gamma_AB` as:

```
gamma_AB(R) = 1/R - S_AB(R)
```

where `S_AB(R)` is the short-range screening function computed by `expGamma(R, U_A, U_B)` in Fortran. For large R, `S → 0` and `gamma → 1/R`. For R → 0, `gamma → (U_A + U_B)/2`.

### Option A: Direct analytical formula (Recommended)
Compute `gamma(R, U_A, U_B)` on the fly using the closed-form DFTB expression. For organic species (H, C, N, O), there are only ~10 unique species pairs. With a 20 Å cutoff and 1000 fragments × 10 neighbors × 50 atoms, that's ~25M evaluations per SCC step — trivial for a modern CPU.

### Option B: Precomputed grid + interpolation
Build a 1D grid of gamma vs R for each species pair, then interpolate. Adds memory but slightly faster if called billions of times.

**Decision: Option A** for clarity and zero setup. We can always add tabulation later behind a trait.

## 3. External Potential Calculation

The external potential on atom A in fragment F is:

```
V_ext(A) = Σ_{G ≠ F} Σ_{B in G} γ(R_AB, U_A, U_B) * (q_B - q0_B)
```

### Algorithm per SCC step:

```rust
// Input: global charges (flat vector over all atoms)
// Output: V_ext per atom (same flat layout)

fn compute_v_ext(
    fragments: &[Fragment],
    frag_neighbors: &[Vec<<usize>],  // spatial neighbor list
    charges: &[f64],
    q0: &[f64],
    out_v_ext: &mut [f64],
) {
    out_v_ext.fill(0.0);
    for (fi, frag) in fragments.iter().enumerate() {
        for &fj in &frag_neighbors[fi] {
            if fi == fj { continue; }
            let frag_j = &fragments[fj];
            for (ai, atom_i) in frag.atoms.iter().enumerate() {
                let vi = &mut out_v_ext[frag.atom_offset + ai];
                for (aj, atom_j) in frag_j.atoms.iter().enumerate() {
                    let dq = charges[frag_j.atom_offset + aj] - q0[frag_j.atom_offset + aj];
                    let r = distance(atom_i.coord, atom_j.coord);
                    let g = gamma_eval(r, atom_i.species, atom_j.species);
                    *vi += g * dq;
                }
            }
        }
    }
}
```

### Complexity:
- O(N_frag × n_neigh × n_atom²) per step
- For 10,000 water dimers (20K fragments? No, 10K molecules = 10K fragments): 10K × 10 neighbors × 9 atom pairs = 900K γ evaluations. Negligible.

## 4. Fragment Internal Architecture

Each fragment is a **self-contained DFTB system** that receives an external potential vector and returns new charges.

```rust
pub struct Fragment {
    // --- Constant data (set at creation) ---
    pub template_id: usize,
    pub coords: Vec<[f64; 3]>,
    pub species: Vec<u8>,          // indices into global species table
    
    // --- Precomputed (built once from template) ---
    pub ctx: SystemContext,         // integer-indexed lookups
    pub h0: DMatrix<f64>,          // non-SCC Hamiltonian (constant)
    pub s: DMatrix<f64>,           // overlap (constant)
    
    // --- Mutable state (updated each SCC step) ---
    pub v_ext: Vec<f64>,          // external potential per atom (from other fragments)
    pub h_scc: DMatrix<f64>,      // H0 + S·shift (built each step)
    pub eigenvalues: DVector<f64>,
    pub eigenvectors: DMatrix<f64>,
    pub charges: Vec<f64>,         // atom-resolved charges
    pub energy: f64,
}
```

### Fragment lifecycle per SCC step:

```rust
impl Fragment {
    /// Build H_scc = H0 + S * shift, where shift = V_intra + V_ext
    pub fn build_h_scc(&mut self, v_ext: &[f64]) {
        // V_intra is computed from self.charges using intra-fragment γ
        // For atom-resolved SCC, shift(A) = V_intra(A) + V_ext(A)
        // Then: H_scc[(i,j)] = H0[(i,j)] + S[(i,j)] * 0.5 * (shift(A_i) + shift(A_j))
    }
    
    pub fn diagonalize(&mut self) {
        // Generalized eigenvalue problem: H·c = E·S·c
        // Use nalgebra or LAPACK
    }
    
    pub fn compute_charges(&mut self) -> Vec<f64> {
        // Mulliken population analysis from density matrix
        // q_A = Σ_{μ in A} (D·S)_{μμ}
    }
}
```

## 5. Global SCC Loop: Single Broyden, No Nested Iterations

### Rejected approach: Nested loops (intra-SCC inside inter-Jacobi)
```
for each outer Jacobi step:
  for each fragment:
    converge fragment SCC internally (10-20 iterations)
```
**Problem:** 10 outer × 15 inner = 150 diagonalizations per fragment. Too expensive.

### Adopted approach: Single global Broyden on concatenated charges

The entire system's charges form one flat vector. The SCC residual is:

```
F(q) = q_out(q) - q_in
```

where `q_out` is obtained by:
1. Computing V_ext from q_in
2. Building H_scc for each fragment (H0 + V_intra + V_ext)
3. Diagonalizing each fragment
4. Extracting charges

This is a **fixed-point iteration** on the global charge vector. We accelerate it with Broyden/Anderson mixing.

```rust
pub struct MultiSystemSolver {
    pub fragments: Vec<<Fragment>,
    pub frag_neighbors: Vec<Vec<<usize>>,
    pub gamma: GammaTable,
    
    // --- Global state ---
    pub charges: Vec<f64>,           // flattened: all atoms
    pub q0: Vec<f64>,               // reference charges (from SK params)
    
    // --- Mixer ---
    pub mixer: Box<dyn Mixer>,
    
    // --- Workspace ---
    pub v_ext: Vec<f64>,            // flattened external potential
}
```

### SCC loop:

```rust
impl MultiSystemSolver {
    pub fn solve_scc(&mut self, max_iter: usize, tol: f64) -> Result<()> {
        for iter in 0..max_iter {
            // 1. Compute V_ext from current global charges
            self.compute_v_ext();
            
            // 2. Parallel: each fragment builds H_scc, diagonalizes, returns new charges
            let mut q_out = Vec::with_capacity(self.charges.len());
            for frag in &mut self.fragments {
                frag.build_h_scc(&self.v_ext[frag.atom_range()]);
                frag.diagonalize();
                let q_frag = frag.compute_charges();
                q_out.extend_from_slice(&q_frag);
            }
            
            // 3. Global mixer
            let residual = vec_sub(&q_out, &self.charges);
            let rms = rms_norm(&residual);
            if rms < tol {
                return Ok(());
            }
            
            self.mixer.mix(&mut self.charges, &q_out, &residual);
        }
        Err(DftbError::SccNotConverged)
    }
}
```

## 6. Mixer Design: From Simple to DIIS

### Phase 1: Simple Linear Mixer (implement first)
```rust
pub struct SimpleMixer {
    pub alpha: f64,  // 0.1 .. 0.3
}

impl Mixer for SimpleMixer {
    fn mix(&mut self, q_in: &mut [f64], q_out: &[f64], _residual: &[f64]) {
        for i in 0..q_in.len() {
            q_in[i] = (1.0 - self.alpha) * q_in[i] + self.alpha * q_out[i];
        }
    }
}
```
- **Pros**: trivial, zero memory overhead
- **Cons**: slow convergence (30-100 iterations for polar systems)

### Phase 2: Anderson/DIIS Mixer (add next)
```rust
pub struct DiisMixer {
    pub max_history: usize,        // 5-8
    pub q_in_history: VecDeque<Vec<f64>>,
    pub residual_history: VecDeque<Vec<f64>>,
}
```
Anderson mixing minimizes the residual in the subspace spanned by recent iterations:
```
q^(k+1) = q_out^(k) + Σ_i c_i * (q_out^(i) - q_out^(k))
```
where `c_i` are found by solving a small linear system (`max_history × max_history`).

- **Pros**: robust, faster than Broyden for large systems
- **Cons**: stores `max_history` vectors of length N_atoms

For 50K atoms × 8 history × 8 bytes = 3.2 MB. Trivial.

### Phase 3: Broyden (optional, for tough cases)
Standard Broyden second method with limited memory.

**Decision: Implement Simple → DIIS. Skip Broyden unless DIIS fails.**

## 7. Spatial Neighbor List for Fragments

To avoid O(N_frag²) inter-fragment loops:

```rust
pub struct FragmentNeighborList {
    pub cutoff: f64,
    pub neighbors: Vec<Vec<<usize>>,  // per-fragment neighbor indices
}

impl FragmentNeighborList {
    pub fn build(centroids: &[[f64; 3]], cutoff: f64) -> Self {
        // Cell-list (spatial hash) for O(N) construction
        // For static geometries: build once
        // For MD: rebuild every N steps or incrementally update
    }
}
```

**Cutoff choice:** 20-30 Å covers all γ interactions for organic species. Beyond this, `gamma ≈ 1/R` and the interaction is weak.

## 8. Charge Analysis: Atom-Resolved vs Shell-Resolved

| Model | Variables per water | Accuracy | Use Case |
|---|---|---|---|
| Atom-resolved | 3 (q_H, q_H, q_O) | Good for sp | Organic, mio basis |
| Shell-resolved | 5 (q_s, q_py, q_pz, q_px, q_O) | Exact DFTB | Metals, 3ob, precise work |

**Decision: Start atom-resolved.** For H/C/N/O with mio-1-1, atom-resolved SCC is standard. The γ function uses one Hubbard U per species.

If we need shell-resolved later, the architecture supports it: `charges` becomes `charges_shell` and the shift calculation loops over shells.

## 9. Zero-Allocation Guarantee for the SCC Loop

Following the optimization principles, **the entire SCC loop must be allocation-free:**

| Step | Allocation? | Fix |
|---|---|---|
| `compute_v_ext` | `out_v_ext` is preallocated slice | Reuse solver.v_ext |
| `frag.build_h_scc` | `h_scc` matrix | Preallocated in Fragment |
| `frag.diagonalize` | Eigenvectors/eigenvalues | Preallocated in Fragment |
| `frag.compute_charges` | Charge vector | Write into preallocated slice |
| `mixer.mix` | History vectors | Preallocated with max_history |
| `q_out` concatenation | Extend vector | Preallocate to n_atoms_total |

**All vectors are sized once at solver initialization and reused.**

---

## 10. Module Layout (Non-Interfering)

```
rust_dftb/src/
  lib.rs              -- add: pub mod qmqm;
  sk_data.rs          -- untouched
  hamiltonian.rs      -- untouched
  scc.rs              -- untouched (or later replaced)
  qmqm/               -- NEW directory
    mod.rs            -- exports
    fragment.rs       -- Fragment, FragmentTemplate
    gamma.rs          -- GammaTable, gamma_eval()
    mixer.rs          -- Mixer trait, SimpleMixer, DiisMixer
    neighbor.rs       -- FragmentNeighborList
    solver.rs         -- MultiSystemSolver
    shifts.rs         -- Intra-fragment shift builder
    charges.rs        -- Mulliken population analysis
```

The existing code continues to compile and test normally. The `qmqm` module imports from the crate root (`crate::sk_data`, `crate::hamiltonian::SystemContext`, etc.) but does not modify them.

## 11. Summary of Concrete Design Decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Inter-fragment physics | Pure γ electrostatics | No orbital overlap between fragments |
| 2 | γ evaluation | Direct analytical | ~10 species pairs, cutoff makes cost trivial |
| 3 | Charge model | Atom-resolved | Sufficient for organic/mio; simpler |
| 4 | SCC loop | Single global Broyden/DIIS | No nested iterations, one charge vector |
| 5 | Fragment diagonalization | Parallel (Rayon) | Embarrassingly parallel |
| 6 | Mixer | DIIS (Anderson) | Robust, low memory, fast convergence |
| 7 | Spatial structure | Cell-list on fragment centroids | O(N) neighbor finding |
| 8 | Fragment memory | Preallocated dense matrices | Zero allocation per SCC step |
| 9 | Workspace | Solver-level arena | Reuse buffers across all fragments |
| 10 | Intra-fragment SCC | Standard dense DFTB | Reuse existing H0/S + shift code |

---

## 12. Implementation Status (2025-06-06)

### Completed Modules

All core modules have been implemented under `rust_dftb/src/qmqm/`:

| Module | File | Status | Notes |
|--------|------|--------|-------|
| Module root | `mod.rs` | Complete | Exports all submodules and re-exports key types |
| Gamma function | `gamma.rs` | Complete | Full DFTB gamma(R, U1, U2) evaluator with `GammaTable` |
| Charge mixing | `mixer.rs` | Complete | `Mixer` trait, `SimpleMixer`, `DiisMixer` (Anderson) |
| Fragment neighbors | `neighbor.rs` | Complete | Cell-list spatial hashing for O(N) neighbor finding |
| Fragment types | `fragment.rs` | Complete | `FragmentTemplate` and `Fragment` with owned data |
| Intra-fragment shifts | `shifts.rs` | Complete | `compute_intra_shifts` for self-consistent shifts |
| Mulliken charges | `charges.rs` | Complete | Population analysis helpers |
| Global solver | `solver.rs` | Complete | `MultiSystemSolver` with SCC loop skeleton |

### Integration

- `src/lib.rs`: Added `pub mod qmqm;` to expose the new module
- `src/error.rs`: Added `SccNotConverged(String)` error variant

### Compilation and Unit Tests

All modules compile cleanly with only pre-existing warnings in unrelated files (`output.rs` unused imports).

Unit tests pass (10/10):
- `qmqm::gamma::tests::*` — 3 tests for gamma function
- `qmqm::mixer::tests::*` — 2 tests for mixers
- `qmqm::neighbor::tests::*` — 2 tests for cell-list
- `qmqm::shifts::tests::*` — 2 tests for intra shifts
- `qmqm::charges::tests::*` — 1 test for Mulliken

### Known Limitations / TODOs

1. **No integration tests**: The module has unit tests for individual components but no end-to-end test comparing full-system vs fragment results.

2. **Fortran debug export**: Added `exportSccDebug` subroutine to scc.F90 and call in main.F90, but needs testing with actual DFTB+ runs.

3. **Charge initialization**: q0 extraction from SK data is approximate (uses full shell occupations). Should read exact neutral charges from onsite parameters.

### Completed Since Last Update

- **Generalized eigensolver**: Implemented Cholesky-based solver in `Fragment::diagonalize()`. Uses `nalgebra::linalg::Cholesky` and `SymmetricEigen`.
- **Charge computation**: `Fragment::compute_charges()` now performs full Mulliken population analysis with closed-shell density matrix.
- **SCC loop**: `MultiSystemSolver::solve_scc()` is fully wired with diagonalization and charge computation.
- **Fixed-charge testing**: Added `build_h_scc_with_fixed_charges()` and `diagonalize_all()` for one-shot testing without SCC iteration.

### Next Steps for Testing

Create integration tests that:
1. Build full-system Hamiltonian with existing code
2. Build fragment-based solver with same geometry
3. Compare non-SCC H0/S (should match exactly)
4. Compare SCC with fixed charges (inject same charges into both, compare shifts and H_scc)
5. Compare full SCC convergence (should match within tolerance)

### Design Notes from Implementation

- **Owned data**: `FragmentTemplate` stores copies of `SystemContext` integer arrays (`atom_species`, `atom_n_orb`, `atom_orb_off`, `species_ang`) to avoid lifetime constraints. This trades memory for API simplicity.

- **Zero-allocation SCC loop**: All global vectors (`charges`, `v_ext`, `q_out`, `residual`, per-fragment scratch) are pre-allocated in `MultiSystemSolver::new()`.

- **Fragment neighbor list**: Includes self-fragments in the neighbor list (no lower-triangle filtering) to simplify the SCC loop. The solver skips `fi == fj` when computing external potential.

- **Mixer**: DIIS implementation includes a tiny dense Gaussian-elimination solver for the history linear system. This is pure Rust and sufficient for `max_history <= 8`.