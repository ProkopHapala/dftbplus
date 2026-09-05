# Import from Other Repos

Reference repositories we import algorithms, geometry builders, theory, and
project-organization patterns from. The sparse BSR4 purification route in
`rust_dftb/src/methods/sparse/` is the current focus; these repos contain
graphene/nanoribbon geometry generators, linear-scaling electronic-structure
theory and prototypes, and OpenCL sparse-kernel patterns that should inform
and accelerate the next stages (real-system testing, SCC integration, LNV,
forces).

**Cross-repo rules:**
- When porting/mirroring a feature, cite the reference file + function in a
  comment, e.g. `// ported from NumericalMathPlayground/topics/LinearScalingQM/DensityMatrix/OrderN.py:estimate_spectral_bounds`.
- CPU Rust references are authoritative for correctness; GPU (OpenCL) must
  match CPU within tolerance.
- **Do not copy Python into Rust.** Port the algorithm, cite the source, use
  Rust idioms (flat arrays, `&[f32]`, `bytemuck` casts).

---

## 1. SPAMMM — `/home/prokophapala/git/SPAMMM/`

**Role:** Full-featured Python + pyOpenCL scanning-probe microscopy and
manipulation engine. For dftbplus, the relevant jewels are: **graphene
nanoribbon geometry builders** (for real-system sparse tests), **sparse
density-matrix projection kernels** (LCAO_grid.cl), and the **DFTB+ wrapper**
that exports H/S/density matrices.

### Top-level layout
| Dir | Purpose |
|-----|---------|
| `spammm/topology/` | Molecular topology, **graphene ribbon builders**, hex grid |
| `spammm/quantum/DFTB/` | DFTB+ ctypes wrapper, GPU density projection |
| `spammm/quantum/` | Pauli master equation solvers (sparse GPU) |
| `kernels/` | OpenCL `.cl` sources |

### 1.1. Graphene nanoribbon / honeycomb builders — P0 for real-system tests

These are the most directly useful files for generating test geometries for
the sparse BSR4 route.

| What | File | Key functions | Notes |
|------|------|---------------|-------|
| **Zigzag ribbon builder** | `spammm/topology/MoleculeEditorBackend.py` | `honeycomb_ring_nodes(q,r,a_CC)` (L48-67), `build_zigzag_ribbon(...)` (L1776-1829), `_build_strip_ribbon(...)` (L2028-2138), `build_two_ribbon_cell(...)` (L2215-2273), `combine_ribbons(...)` (L2164-2213) | The only live graphene ribbon builder in the repo. Supports width, length, passivation, Lx. **Port to Rust** as a geometry generator for sparse tests. |
| **Hex grid primitives** | `spammm/topology/HexGrid.py` | `class HexGrid`, `ring_nodes(q,r)` (L16-80), `snap_to_ring(x,y)` | Low-level honeycomb coordinate math. Port the ring-node formula. |
| **Heterocycle generator** | `spammm/topology/heterocycle_generator.py` | `build_geometry(system,...)` (L294-413), `parse_row(...)` (L74-118) | Builds graphene-like heterocycles from a sparse grid description. Useful for doped/defective graphene test systems. |
| **GUI wrappers** | `spammm/GUI/SPAMMM_GUI.py` | `create_ribbon_section()` (L643-703), `generate_single_ribbon()` (L705-737) | GUI calls the builders above. Not needed for Rust port, but documents the API. |

**Honeycomb geometry** (`honeycomb_ring_nodes`, L48-67):
The fundamental formula for graphene atom positions on a hexagonal lattice,
parameterized by ring indices (q,r) and C-C bond length a_CC. This is the
starting point for any graphene ribbon/sheet/flake generator.

**What to port to dftbplus:**
- A Rust `graphene` module (e.g. `rust_dftb/src/geometry/graphene.rs`) with:
  - `build_zigzag_ribbon(width, length, a_CC, passivation) -> Vec<[f64;3]>`
  - `build_armchair_ribbon(width, length, a_CC, passivation)`
  - `build_sheet(nx, ny, a_CC) -> Vec<[f64;3]>` (supercell)
  - `build_flake(radius, a_CC) -> Vec<[f64;3]>` (circular flake)
- Output atom positions + element types (C + H for passivation).
- These become inputs to the sparse BSR4 test harness.

**Caveat:** SPAMM has no stand-alone infinite-sheet generator. 2D supercell
generation must be built on top of `honeycomb_ring_nodes`.

### 1.2. Sparse density-matrix GPU projection — reference for K→grid

| What | File | Key kernels | Notes |
|------|------|-------------|-------|
| **Sparse density projection** | `kernels/LCAO_grid.cl` | `project_density_sparse` (L351-444), `project_density_sparse_tiled` (L720+), `compact_tasks` | Projects a sparse DFTB density matrix onto a 3D real-space grid. Not a solver, but shows how to gather K_ij blocks onto grid points. Relevant if we later need density visualization. |
| **Sparse vs dense discussion** | `kernels/LCAO_grid.cl` (L55-76) | Comments comparing sparse and dense projection strategies | Design rationale for sparse density handling. |
| **DFTB+ matrix export** | `spammm/quantum/DFTB/DFTBcore.py` | `get_dm_dense()` (L475-479), `enable_matrix_collection(...)` (L266-274) | Wraps DFTB+ C-API to export H, S, density matrix. **This is how SPAMM gets reference matrices for parity tests.** dftbplus already has its own dense path, but the pattern is useful. |
| **Grid projector** | `spammm/quantum/DFTB/Grid_dftb.py` | `project_density(...)`, `project_density_dense(...)` (L1372-1385), `build_tasks_gpu(...)` | Host-side orchestration of sparse density projection. |

### 1.3. Other sparse GPU patterns

| What | File | Notes |
|------|------|-------|
| **CSR-like force assembly** | `kernels/UFF.cl:assembleForces_UFF` (L980-1024) | Uses `a2f_offsets/counts/indices` (CSR-like) to gather angle/dihedral forces onto atoms. Pattern relevant for sparse force assembly in dftbplus. |
| **Sparse PME solver** | `kernels/PME8.cl:solve_pme8` (L136-260) | Sparse Jacobi iteration on a rate matrix (8 neighbors/state, 256 threads). Shows GPU sparse iterative solve pattern. |
| **Batched dense eigensolver** | `kernels/lingebra.cl:local_jacobi_blocks_parallel` (L67-80), `spammm/utils/Lingebra_ocl.py` | Batched Jacobi rotations for small dense matrices on GPU. Potentially useful for the 4×4 block operations in BSR4. |

### 1.4. Kekulé bond-order optimizer (empirical, not quantum)

| What | File | Notes |
|------|------|-------|
| **KekulePure** | `spammm/topology/KekulePure.py` | `class KekulePure` (L29-103), `pi_bond_orders()` (L222-246), `solve_constrained(...)` (L314-344) | Empirical pi-bond-order optimizer for aromatic systems. **Not** a quantum density matrix. Useful as a comparison/initial-guess for graphene pi-system tests, but not a purification solver. |

### 1.5. Stale doc references
- `doc/molecular_topology_editors.md` and `doc/FireCore_migration_codemap.md`
  reference `KekuleBackend.py`, which no longer exists. The live code is in
  `MoleculeEditorBackend.py`. Do not chase the stale filename.

---

## 2. NumericalMathPlayground — `/home/prokophapala/git/NumericalMathPlayground/`

**Role:** Theoretical derivations and prototype implementations for
linear-scaling electronic-structure methods, bond-order potentials, and
sparse GPU solvers. **This is the most important reference repo for the
sparse BSR4 route** — it contains working Python prototypes of FOE, Green's
function density matrix, CheFSI, OMM, and Chebyshev bond-order probes, plus
design documents discussing McWeeny purification, idempotency, and spectral
bounds.

### Top-level layout
| Dir | Purpose |
|-----|---------|
| `topics/LinearScalingQM/` | **Core reference**: density matrix methods, CheFSI, OMM, BOP, Kekulé fluid |
| `topics/LinearAlgebra/SpectralFiltering/` | OpenCL CSR SpMM, Chebyshev filtering, resolvent solvers |
| `topics/LinarElasticity/` | GPU sparse truss solver (CSR, block-Jacobi) |
| `topics/ChemicalGraphs/` | Kekulé backend, heterocycle generator, honeycomb builders |
| `py/DFTB/` | DFTB+ ctypes wrapper, GPU density projection |
| `py/FFs/` | UFF builder with bond-order assignment |

### 2.1. LinearScalingQM — density matrix methods (P0 reference)

This is the single most valuable subdirectory for the sparse BSR4 work.

| What | File | Key functions | Notes |
|------|------|---------------|-------|
| **O(N) shared utilities** | `topics/LinearScalingQM/DensityMatrix/OrderN.py` | `find_neighbor_pairs()` (L16), `build_matrix_from_pairs()` (L58), `solve_generalized()` (L97), `mulliken_density()` (L114), `jacobi_solve()` (L165), `_estimate_spectral_bounds()` (L140), `_scale_hamiltonian()` (152) | **Directly relevant.** Neighbor-pair finding, sparse assembly, spectral bounds, Mulliken density — all patterns we need in Rust. Port algorithms, cite this file. |
| **Fermi Operator Expansion** | `topics/LinearScalingQM/DensityMatrix/FOE.py` | `chebyshev_coeffs_fermi()` (L12), `foe_stochastic_density()` (L27) | FOE via Chebyshev + stochastic trace. An alternative to purification for metals. **Not yet needed** for the gapped systems the sparse BSR4 route targets, but relevant for future metallic graphene. |
| **Green's function density** | `topics/LinearScalingQM/DensityMatrix/GF.py` | `get_contour_poles()` (L11), `greens_function_probing()` (L30), `greens_function_random()` (L115) | Contour integration + probing. Another alternative to purification. |
| **Idempotency discussion** | `topics/LinearScalingQM/DensityMatrix/DensityMatrix_Idenpotency_NearestNeighbor.chat.md` | L5, L34, L88, L117, L190 | **Read this.** Discusses enforcing PSP=P idempotency in local-basis density matrix methods. Directly relevant to our R_I diagnostic and McWeeny/TC2 purification. |
| **Order-N solver design** | `topics/LinearScalingQM/DensityMatrix/OrderN_Electronic_Structure_Solver_Gemini.chat.md` | L1, L71, L175, L352, L950 | Design chat for the O(N) solver. Covers spectral bounds, truncation, convergence. |
| **DensityMatrix README** | `topics/LinearScalingQM/DensityMatrix/README.md` | L1-42 | Overview of FOE/GF methods and their trade-offs. |

### 2.2. CheFSI — Chebyshev-filtered subspace iteration

| What | File | Key functions/kernels | Notes |
|------|------|-----------------------|-------|
| **CheFSI solver** | `topics/LinearScalingQM/CheFSI/CheFSI.py` | `FrontierSolver` class (L13), `prepare()` (L70), `solve()` (L182), `_spmm()` (L103), `_orthogonalize()` (L145) | GPU-accelerated frontier orbital computation using ELLPACK sparse format. An alternative to purification for computing HOMO/LUMO without full diagonalization. |
| **CheFSI kernels** | `topics/LinearScalingQM/CheFSI/CheFSI.cl` | `spmm_ellpack_chebyshev` (L20), `tall_skinny_gram` (L101), `subspace_rotate` (L245) | OpenCL ELLPACK SpMM + Chebyshev filtering. **The ELLPACK SpMM is a reference for our BSR4 SpGEMM** — different format but similar gather pattern. |
| **CheFSI design chat** | `topics/LinearScalingQM/CheFSI/CheFSI_Frontier_Orbitals.chat.md` | — | Design discussion. |

### 2.3. OMM — Orbital Minimization Method

| What | File | Key functions/kernels | Notes |
|------|------|-----------------------|-------|
| **OMM solver** | `topics/LinearScalingQM/OMM/OMM.py`, `OMM_ocl.py` | — | Direct optimization of localized molecular orbitals with finite support. An alternative to purification that directly produces localized orbitals. |
| **OMM OpenCL kernels** | `topics/LinearScalingQM/OMM/cl/OMM.cl` | `apply_operator` (L19), `project_orbitals` (L71), `assemble_gradient` (L141) | Uses CSR-like `pairRowPtr`/`orbPairs`. Shows GPU sparse gradient assembly for orbital optimization. |
| **OMM design chats** | `topics/LinearScalingQM/OMM/Orbital_Minimization_Methond.chat.md`, `OMM_Kernels.chat.md`, `OMM_VB.chat.md`, `Approximate_Overlap.chat.md` | — | Theory and implementation discussions. `Approximate_Overlap.chat.md` is relevant to our Newton-Schulz Z≈S⁻¹ work. |

### 2.4. Kekule_BOP — Bond-Order Potentials

| What | File | Key functions | Notes |
|------|------|---------------|-------|
| **Chebyshev bond-density probe** | `topics/LinearScalingQM/Kekule_BOP/KekuleOrderN_Gemini1.py` | `get_energy_chebyshev()` (L44), `get_bond_density_CF()` (L137), `get_energy_lanczos()` (L166) | Computes bond densities without diagonalization via Chebyshev/Lanczos probe propagation. **This is the BOP concept** — bond orders as local spectral quantities. |
| **2D BOP convergence test** | `topics/LinearScalingQM/Kekule_BOP/KekuleOrderN_Gemini_BOP_2D.py` | `generate_graphene_flake()` (L13), `get_bond_density_convergence()` (L87) | Tests BOP convergence on graphene flakes. **Use as reference for graphene flake generation + bond-order validation.** |
| **2D BOP v2** | `topics/LinearScalingQM/Kekule_BOP/KekuleOrderN_Gemini_BOP_2D_v2.py` | `generate_graphene_flake(radius)` (L13), `get_bond_density_cheb()` (L91), `get_bond_density_CF()` (L169) | Updated version. |
| **BOP theory** | `topics/LinearScalingQM/Kekule_BOP/KekuleBOP_classic_hybrid.theory.md` | §4.2 "Topological defects in graphene nanoribbons" (L224), §9.2 "Graphene nanoribbons and doped sheets" (L1006) | **Read this.** Theory of Kekulé-BOP hybrid for graphene nanoribbons. Directly relevant to our graphene test systems. |
| **BOP briefing** | `topics/LinearScalingQM/Kekule_BOP/KekuleBOP_hybrid.LLM_briefing.md` | L1-78 | Design notes for BOP avoiding O(N³) and Hückel. |
| **BOP design chat** | `topics/LinearScalingQM/Kekule_BOP/KekuleOrderN.chat.md` | L1, L7, L88, L745, L1219 | Design chat for O(N) BOP solvers. |
| **SP3 hybridization** | `topics/LinearScalingQM/Kekule_BOP/LocalSP3Hybridization_with_dimer_matching.md` | L389, L407, L506, L566, L673 | SP3 reactive FF with local dimer-matching / bond-order. Relevant if we extend beyond sp2 graphene. |

### 2.5. KekuleFluid — honeycomb graph + dynamics

| What | File | Key functions | Notes |
|------|------|---------------|-------|
| **Honeycomb graph builder** | `topics/LinearScalingQM/KekuleFluid/Graph.py` | `HoneycombGraph` class (L59), `build_rect_patch()` (L80), `build_pah()` (L127), `build_flake()` (L264), `get_neighbor_list()` (L448), `get_arrays()` (L467) | **Current honeycomb graph builder** used by the Kekulé fluid solvers. Builds rectangular patches, PAHs, and flakes. **Port this to Rust** for graphene test geometry generation. |
| **Legacy hex grid** | `topics/LinearScalingQM/KekuleFluid/hexgrid.py` | `build_honeycomb_patch()` (L55), `Vortex` (L36), `HoneycombGraph` (L42), `init_vortex_phase()` (L242) | Older numpy honeycomb builder. Superseded by `Graph.py` but contains vortex-phase initialization. |
| **Kekulé fluid solver** | `topics/LinearScalingQM/KekuleFluid/ModelA.py` | `KekuleFluidSolver` class (L64), `evolveZ_RK4()` (L229), `projectBondOrders()` (L281) | Dynamics solver on honeycomb graph. The `projectBondOrders()` pattern is relevant to our K→bond-order extraction. |
| **Kekulé fluid kernels** | `topics/LinearScalingQM/KekuleFluid/kekule_fluid.cl` | `bondsToZ` (L173), `rhsZ` (L333), `projectBondOrdersSub` (L656) | GPU kernels for bond-order dynamics on honeycomb. |
| **Dirac lattice** | `topics/LinearScalingQM/KekuleFluid/DiracLattice.cl` | `rhs_psi` (L192), `rk4_intermediate_half` (L250), `rk4_combine` (L307) | Lattice tight-binding π-electron propagation (Dirac on honeycomb). |
| **Kekulé fluid docs** | `topics/LinearScalingQM/KekuleFluid/KekuleFluid.chat.md`, `KekuleFluid.md` | — | Design and theory documents. |

### 2.6. KekuleQM — GPU Kekulé quantum solver

| What | File | Key functions | Notes |
|------|------|---------------|-------|
| **KekuleQM OCL** | `topics/LinearScalingQM/KekuleQM/KekuleQM_ocl.py` | `KekuleQM_OCL` class (L274), `generate_graphene_flake()` (L54), `from_ascii_art()` (L77), `build_neighbor_lists()` (L169) | GPU Kekulé QM solver with graphene flake generator and ASCII-art topology parser. **The `from_ascii_art()` pattern is useful for quick test geometries.** |
| **KekuleQM kernels** | `topics/LinearScalingQM/KekuleQM/KekuleQM.cl` | `KekuleQM_gatherLocalBonding` (L106), `KekuleQM_gatherCoulombDirect` (L183), `KekuleQM_updateDOFs` (L234) | GPU local bond-order and Coulomb update kernels. |

### 2.7. SpectralFiltering — OpenCL CSR sparse linear algebra

| What | File | Key functions/kernels | Notes |
|------|------|-----------------------|-------|
| **CSR SpMM (OpenCL)** | `topics/LinearAlgebra/SpectralFiltering/spectral_solvers.py` | `_OPENCL_CSR_SPMM` source (L57-108), `OpenCLCSRSpMM` class (L156) | **Reference OpenCL CSR SpMM implementation.** Compare with our BSR4 SpGEMM. Different format (CSR scalar vs BSR 4×4 blocks) but same gather pattern. |
| **Chebyshev filtering** | `topics/LinearAlgebra/SpectralFiltering/spectral_solvers.py` | `chebyshev_filter()` (L945), `solve_band()` (L900), `solve_spectrum()` (L752) | Chebyshev band-pass filters + Rayleigh-Ritz. Alternative to purification for spectral computation. |
| **Resolvent solvers** | `topics/LinearAlgebra/SpectralFiltering/resolvent_solvers.py` | `_OPENCL_CSR_SPMM_FLOAT` (L47), `_OPENCL_CSR_SPMM_COMPLEX` (L69), `OpenCLBatchedSpMM` (L113), `solve_minres()` (L320), `solve_gmres()` (L439) | Batched resolvent/Green's function solvers with MINRES/COCR/BiCGSTAB/GMRES. **Relevant if we need iterative linear solvers** for Z≈S⁻¹ alternatives or LNV. |
| **Spectral filtering tutorial** | `topics/LinearAlgebra/SpectralFiltering/doc/SpectralFiltering_doc_tutorial.md` | L1-80 | Chebyshev filtering + KPM + Jackson damping. |

### 2.8. LinarElasticity — GPU sparse truss solver

| What | File | Key functions/kernels | Notes |
|------|------|-----------------------|-------|
| **Sparse truss OCL** | `topics/LinarElasticity/SparseTruss_ocl.py` | `SparseTrussOCL` class (L91), `edges_to_csr()` (L653), `build_cluster_map()` (L742) | GPU sparse truss/spring-mass solver. The `edges_to_csr()` pattern is relevant to our BSR4 CSR construction. |
| **Block-Jacobi kernels** | `topics/LinarElasticity/kernels_block_jacobi.cl` | `block_jacobi_step` (L72), `compute_residual` (L204), `compute_diagonal_dinv` (L268) | CSR block-Jacobi smoother. Shows GPU CSR iterative solve with block diagonal preconditioning. |

### 2.9. ChemicalGraphs — Kekulé backend + heterocycle generator

| What | File | Key functions | Notes |
|------|------|---------------|-------|
| **Kekulé backend** | `topics/ChemicalGraphs/KekuleBackend.py` | `KekuleBackend` class (L107), `build_zigzag_ribbon()` (L1225), `build_two_ribbon_cell()` (L1663), `honeycomb_ring_nodes()` (L32), `build_lattice_vectors()` (L951) | Persistent molecular editor on hexagonal grid. **Older version** of the SPAMM ribbon builder. Use SPAMM's `MoleculeEditorBackend.py` as the primary reference; this is the ancestor. |
| **Heterocycle generator** | `topics/ChemicalGraphs/heterocycle_generator.py` | — | Heterocycle generator on hexagonal grid. |
| **KekulePure (original)** | `topics/ChemicalGraphs/KekulePure.py` | `KekulePure` class (L27), `solve_constrained()` (L312), `optimize_pi_bonds()` (L463) | Original empirical Kekulé bond-order optimizer. |

### 2.10. DFTB+ wrapper (NMP version)

| What | File | Key functions | Notes |
|------|------|---------------|-------|
| **DFTBcore** | `py/DFTB/DFTBcore.py` | `DFTBcore` class (L277), `get_dm_dense()` (L385), `get_h_dense()` (L386), `get_s_dense()` (L387) | ctypes C-API wrapper around DFTB+. Exports H, S, density matrix. **This is the pattern for getting reference matrices.** dftbplus has its own dense path but the API design is instructive. |
| **DFTB utils** | `py/DFTB/DFTB_utils.py` | `makeDFTBjob_pbc()` (L904), `run_pbc()` (L332), `run_dftb_sp()` (L729) | DFTB+ I/O, SK path management, periodic job setup. |
| **Grid projection** | `py/DFTB/Grid_dftb.py` | `GridProjector` class, `project_density()` | GPU projection of DFTB wavefunctions/density onto 3D grids. |

### 2.11. Forcefields with bond-order assignment

| What | File | Key functions | Notes |
|------|------|---------------|-------|
| **UFF builder** | `py/FFs/UFFbuilder.py` | `UFF_Builder` class (L67), `assign_uff_types()` (L183), `assign_uff_params()` (L246), `assign_uff_types_findrings()` (L734) | UFF topology builder with aromaticity and bond-order assignment. The ring-finding and aromaticity logic is relevant if we need to classify graphene edge types. |
| **UFF params** | `py/FFs/FFparams.py` | `UFFParam` / `AtomTypeParam` classes (L44, L78) | UFF parameter records. `valence = sum of bond orders`. |

---

## 3. Key theory documents to read before next implementation steps

These are the most important documents to review for the sparse BSR4 route:

| Priority | Document | Why |
|----------|----------|-----|
| **P0** | `NMP/topics/LinearScalingQM/DensityMatrix/DensityMatrix_Idenpotency_NearestNeighbor.chat.md` | Discusses PSP=P idempotency in local-basis methods. Directly relevant to our R_I diagnostic and McWeeny/TC2. |
| **P0** | `NMP/topics/LinearScalingQM/DensityMatrix/OrderN_Electronic_Structure_Solver_Gemini.chat.md` | O(N) solver design: spectral bounds, truncation, convergence. |
| **P0** | `NMP/topics/LinearScalingQM/Kekule_BOP/KekuleBOP_classic_hybrid.theory.md` | BOP theory for graphene nanoribbons. §4.2 topological defects, §9.2 doped sheets. |
| **P1** | `NMP/topics/LinearScalingQM/Kekule_BOP/KekuleOrderN.chat.md` | O(N) BOP solver design chat. |
| **P1** | `NMP/topics/LinearScalingQM/CheFSI/CheFSI_Frontier_Orbitals.chat.md` | CheFSI design — alternative to purification for frontier orbitals. |
| **P1** | `NMP/topics/LinearScalingQM/OMM/Approximate_Overlap.chat.md` | Approximate overlap / inverse — relevant to Newton-Schulz Z≈S⁻¹. |
| **P2** | `NMP/topics/LinearAlgebra/SpectralFiltering/doc/SpectralFiltering_doc_tutorial.md` | Chebyshev filtering + KPM + Jackson damping. |
| **P2** | `NMP/topics/LinearScalingQM/KekuleFluid/KekuleFluid.md` | Kekulé fluid theory — bond-order dynamics on honeycomb. |

---

## 4. What to port — concrete action items

### 4.1. Graphene geometry generator (immediate, for real-system tests)

**Source:** SPAMM `MoleculeEditorBackend.py:honeycomb_ring_nodes` + NMP
`KekuleFluid/Graph.py:HoneycombGraph`.

**Target:** `rust_dftb/src/geometry/graphene.rs` (new file).

**Functions to port:**
- `honeycomb_ring_nodes(q, r, a_CC) -> [f64; 3]` — fundamental hex coordinate
- `build_zigzag_ribbon(width, length, a_CC, passivation) -> (pos, types, bonds)`
- `build_armchair_ribbon(width, length, a_CC, passivation)`
- `build_sheet(nx, ny, a_CC) -> (pos, types, bonds)` — supercell
- `build_flake(radius, a_CC) -> (pos, types, bonds)` — circular flake
- `build_two_ribbon_cell(...)` — two-ribbon junction (from SPAMM)

**Output:** atom positions, element types (C/H), bond connectivity. These
feed directly into the sparse BSR4 test harness as real DFTB geometries.

**Caveat:** BSR4 currently assumes 4 orbitals/atom. Hydrogen has 1 orbital.
Either (a) assert "all atoms have 4 orbitals" and use pure-carbon graphene
(no passivation) for now, or (b) implement mixed-block support (padded 4×4
with inactive rows/cols = 0). See `sparse_bsr4_report.md` §5.8.

### 4.2. Spectral bounds — compare implementations

**Source:** NMP `DensityMatrix/OrderN.py:_estimate_spectral_bounds` (L140).

**Current:** `bsr4.rs:gershgorin_bounds` — orbital-level Gershgorin.

**Action:** Compare the NMP approach (likely Lanczos-based) with our Gershgorin
bounds. If Lanczos bounds are tighter and more robust, port the algorithm.
Conservative bounds are critical for TC2 convergence (see
`sparse_bsr4_report.md` §4.1).

### 4.3. CSR SpMM — compare with reference

**Source:** NMP `SpectralFiltering/spectral_solvers.py:_OPENCL_CSR_SPMM` (L57-108).

**Current:** `sparse_bsr4_purification.cl:bsr4_spgemm_masked`.

**Action:** Compare the CSR scalar SpMM with our BSR4 block SpGEMM. The CSR
version is simpler (scalar entries, no 4×4 blocks) but the gather pattern and
workgroup strategy may inform optimizations. The ELLPACK SpMM in
`CheFSI/CheFSI.cl:spmm_ellpack_chebyshev` is also relevant — ELLPACK is
padded-fixed-row like our BSR4 with MAX_LEFT_BLOCKS.

### 4.4. Bond-order extraction from K

**Source:** NMP `KekuleFluid/ModelA.py:projectBondOrders` (L281),
`Kekule_BOP/KekuleOrderN_Gemini1.py:get_bond_density_CF` (L137).

**Current:** Not yet implemented in dftbplus sparse route.

**Action:** When we need bond orders (for validation, visualization, or BOP
forcefields), port the bond-order extraction: BO_ij = 2·Tr(K_ij) or similar.
This is a simple reduction over K blocks but the NMP code shows the pattern.

### 4.5. LNV gradient — reference for finite-difference validation

**Source:** NMP `LinearScalingQM/OMM/cl/OMM.cl:assemble_gradient` (L141).

**Current:** `sparse_bsr4_purification.cl:bsr4_lnv_gradient` (compiled, not
wrapped/tested).

**Action:** When validating the LNV gradient, compare with the OMM gradient
assembly pattern. The OMM approach optimizes localized orbitals directly;
our LNV optimizes the density matrix. Different objects but similar gradient
structure (sparse products + projection).

---

## 5. What is NOT in these repos (gaps)

- **No McWeeny/TC2/SP2/Palser purification implementations.** The NMP docs
  discuss purification but do not implement it. Our `sparse_bsr4_purification.cl`
  is the first working purification implementation in this ecosystem.
- **No BSR (block sparse row) format.** All existing GPU sparse code uses CSR
  (scalar) or ELLPACK (padded scalar). Our BSR4 (4×4 blocks) is a new format.
- **No nonorthogonal density-matrix solvers.** NMP's OMM and CheFSI assume
  orthogonal or pre-orthogonalized bases. Our KSK=K + Newton-Schulz Z≈S⁻¹
  approach for the nonorthogonal case is novel in this context.
- **No graphene sheet (2D periodic) generator.** Only ribbons and flakes.
  A sheet generator must be built from `honeycomb_ring_nodes`.
- **No sparse SCC (self-consistent charge) loop.** SPAMM and NMP use DFTB+
  as a black box for SCC. Our sparse SCC integration is new work.

---

## 6. Summary — relevance matrix

| Topic | SPAMMM | NMP | dftbplus current | Action |
|-------|--------|-----|------------------|--------|
| Graphene ribbon builder | ✅ `MoleculeEditorBackend.py` | ✅ `KekuleFluid/Graph.py` | ❌ | **Port to Rust** |
| Honeycomb geometry | ✅ `HexGrid.py` | ✅ `hexgrid.py` | ❌ | **Port to Rust** |
| Spectral bounds | ❌ | ✅ `OrderN.py` | ✅ `gershgorin_bounds` | Compare |
| CSR SpMM (OpenCL) | ✅ `LCAO_grid.cl` | ✅ `spectral_solvers.py` | ✅ BSR4 SpGEMM | Compare patterns |
| ELLPACK SpMM | ❌ | ✅ `CheFSI.cl` | ❌ | Reference for BSR4 |
| McWeeny/TC2 purification | ❌ | ❌ (discussed only) | ✅ **implemented** | — |
| Newton-Schulz Z≈S⁻¹ | ❌ | ❌ (discussed) | ✅ **implemented** | — |
| Nonorthogonal KSK=K | ❌ | ❌ | ✅ **implemented** | — |
| R_H commutator diagnostic | ❌ | ❌ | ✅ **implemented** | — |
| FOE density matrix | ❌ | ✅ `FOE.py` | ❌ | Future (metals) |
| Green's function density | ❌ | ✅ `GF.py` | ❌ | Future |
| CheFSI frontier orbitals | ❌ | ✅ `CheFSI/` | ❌ | Future alternative |
| OMM localized orbitals | ❌ | ✅ `OMM/` | ❌ | Future alternative |
| BOP Chebyshev probes | ❌ | ✅ `Kekule_BOP/` | ❌ | Future (bond orders) |
| DFTB+ H/S/density export | ✅ `DFTBcore.py` | ✅ `DFTBcore.py` | ✅ (native) | — |
| Sparse density → grid | ✅ `LCAO_grid.cl` | ✅ `Grid_dftb.py` | ❌ | Future (visualization) |
| Kekulé bond-order (empirical) | ✅ `KekulePure.py` | ✅ `KekulePure.py` | ❌ | Not needed (quantum K is better) |
| UFF bond-order assignment | ❌ | ✅ `UFFbuilder.py` | ❌ | Not needed for DFTB |
| Block-Jacobi preconditioner | ❌ | ✅ `kernels_block_jacobi.cl` | ❌ | Future (iterative solvers) |
| Resolvent/MINRES/GMRES | ❌ | ✅ `resolvent_solvers.py` | ❌ | Future (LNV linear solves) |
