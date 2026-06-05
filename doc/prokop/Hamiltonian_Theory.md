https://chat.deepseek.com/share/bf4uxyjoemls8fbm7p
https://www.kimi.com/share/19e97ee3-7182-89c0-8000-00009b3371a3
https://gemini.google.com/share/5ce2838ee954

CODEMAPS: DFTB+ Hamiltonian Assembly Workflow
https://windsurf.com/codemaps/4d4be190-5a2f-4265-89c9-ab4d20b15c07-fe86ab10a43f3d18
https://windsurf.com/codemaps/a06e44a5-9f48-409b-b1f9-6d4c51841bc4-fe86ab10a43f3d18



---

# USER

I'm interested to reimplement core of dftb+ xTB or t light packages in my own rust or opencl code. Can you tell me how difficult it would be what are the essential parts and how many lines of code I need for it?
I was thinking to do it in phases. First implement just basic tight binding Hamiltonian and only then work on self-consistent SCC part. I have to say clearly I basically care just about the part which assemble Hamiltonian from positions of atoms and slater koster tables. We do not need 3-center interactions as long as we don't do SCC part (just neutral atom density, pseudo potentials).

Please write systematically how many parts there will be (how many different interaction types or slater koster tables). How dftb, xTB and tblight is doing 3-center necessary first SCC?


---

# Gemini-3.5-flash

Reimplementing the core of an electronic structure method like DFTB+, xTB, or `tblight` is an ambitious but highly rewarding project, especially if you have a background in GPU acceleration and scientific computing.

Focusing phase one strictly on the **non-self-consistent (non-SCC) tight-binding Hamiltonian assembly** makes the project highly manageable. At this stage, you are essentially building an advanced, physically motivated bond-graph evaluator.

Here is a systematic breakdown of the architecture, the complexity, and how these different packages handle the self-consistent field (SCF/SCC) terms and multi-center interactions.

## 1. Complexity & Code Size (Phase 1 vs. Phase 2)

Implementing this in Rust or OpenCL is an excellent choice for performance. Rust provides the type safety needed for complex indexing, while OpenCL maps perfectly to the embarrassingly parallel task of pair-list evaluation and dense matrix diagonalization.

### Phase 1: Non-SCC Hamiltonian Assembly (No Iterations)

* **Difficulty:** Moderate. It is mostly a geometry and interpolation problem (building a neighbor list, rotating Slater-Koster parameters, and packing a matrix).
* **Estimated Lines of Code (Rust/C++):** **~1,500 – 3,000 lines**.
* *Neighbor List & Geometry:* ~300 lines.
* *Slater-Koster File Parser:* ~400 lines (parsing `.skf` formats can be messy due to historical conventions).
* *SK Transformation Rules (Matrix Rotations):* ~500 lines (pure math, direction cosines).
* *Hamiltonian/Overlap Assembly:* ~400 lines.
* *OpenCL Kernels (Optional for this phase):* ~500 lines.

* **External Dependencies:** You will need a reliable dense linear algebra library (like `ndarray-linalg` or `faer` in Rust) to solve the generalized eigenvalue problem ($H\psi = ES\psi$) to get the band energy.

### Phase 2: Adding the SCC / xTB Electrostatics

* **Difficulty:** High. You introduce self-consistency loops, charge fluctuation descriptors, and long-range Coulomb interactions.
* **Estimated Lines of Code:** **+2,000 – 4,000 additional lines** (plus infrastructure for mixing like DIIS).

## 2. Systematic Breakdown of Phase 1 Components

To assemble the non-SCC Hamiltonian ($H^0$) and Overlap matrix ($S$), your code needs to handle exactly **three distinct interaction types**:

### Part A: On-Site Terms (1-Center)

* **What it is:** The diagonal elements of $H$ and $S$ when looking at the same atom ($i = j$).
* **Data required:** Atomic orbital energies (e.g., $\epsilon_s, \epsilon_p, \epsilon_d$) for each element.
* **Matrix values:** * $S_{\mu\nu} = \delta_{\mu\nu}$ (On-site orbitals are orthonormal).
* $H_{\mu\nu} = \epsilon_\mu \delta_{\mu\nu}$.

### Part B: Two-Center Hamiltonians & Overlaps ($H^0_{ij}$ and $S_{ij}$)

* **What it is:** The core of your requested engine. It models the hopping integrals and overlaps between orbital $\mu$ on atom $i$ and orbital $\nu$ on atom $j$.
* **Data required:** **Slater-Koster (SK) tables**. For every unique *pair of elements* (e.g., C-C, C-H, H-H), you need a table. If you have $N$ element types, you need $\frac{N(N+1)}{2}$ tables.
* **The SK Table Structure:** A 1D grid of points sampled along the interatomic distance $R$. For a given $R$, it provides up to 10 independent integrals depending on the maximum angular momentum ($L_{max}$):
* *s-s:* $\sigma$
* *s-p:* $\sigma$
* *p-p:* $\sigma, \pi$
* *s-d:* $\sigma$
* *p-d:* $\sigma, \pi$
* *d-d:* $\sigma, \pi, \delta$

* **The Transformation:** The SK table gives you these values in a local reference frame where the bond lies along the z-axis. Your code must use the **direction cosines** ($l, m, n$) of the actual bond vector to rotate these local $\sigma, \pi, \delta$ values into the global Cartesian coordinate Hamiltonian matrix. (These are the classic Slater-Koster transformation equations).

### Part C: Repulsive Potential ($E_{rep}$)

* **What it is:** A classical pair-potential that compensates for the omission of explicit core-core repulsion and double-counting of electron-electron interactions.
* **Data required:** Also stored at the end of the Slater-Koster files as a 1D function of distance $R$.
* **Computation:** Spline interpolation of $E_{rep}(R)$ for all pairs within a cutoff distance. This contributes only to the total energy and gradients (forces), not to the $H$ matrix.

## 3. How DFTB, xTB, and `tblight` Handle 3-Center Terms and SCC

You noted that 3-center terms are not needed if we don't do SCC. That is largely true for traditional DFTB, but the modern history of these three packages differs significantly in how they handle density corrections and multi-center effects.

Here is the breakdown of how they cross the bridge from Phase 1 to SCC:

### 1. DFTB+ (Standard DFTB2 / DFTB3)

* **How it handles 3-center terms:** **It avoids them entirely by approximation.** Traditional DFTB expands the total Kohn-Sham energy in fluctuations of the electron density around a reference density ($\delta n = n - n_0$).
* **The SCC Part:** The second-order term (DFTB2) assumes charge fluctuations are localized as spherical shells on atoms. The interaction between these charges is modeled by a modified Coulomb interaction (the Hubbard $U$ or Gamma function $\gamma_{AB}$).
* **The 3-Center Trick:** Because the charge fluctuations are treated as monopoles localized on the atomic centers, **all 3-center integrals reduce to simple 2-center electrostatics**. The Hamiltonian is updated via:

$$H_{\mu\nu} = H^0_{\mu\nu} + \frac{1}{2} S_{\mu\nu} \sum_{K} (\gamma_{IK} + \gamma_{JK})\Delta q_K$$

Where $\Delta q_K$ is the net Mulliken charge fluctuation on atom $K$. This requires zero 3-center quantum chemical integrals.

### 2. GFN-xTB (Grimme's Extended Tight Binding)

* **How it handles 3-center terms:** xTB abandons the strict Slater-Koster table paradigm. Instead of reading pre-computed tables for specific element pairs, xTB calculates overlap integrals **on the fly** using an auxiliary basis set of clean, analytical Slater-type orbitals (STOs).
* **The SCC Part:** xTB includes not just monopoles (charges), but also **atomic dipoles and quadrupoles** in its self-consistent field loop.
* **The 3-Center Trick:** Even with higher-order multipoles, xTB avoids true quantum-chemical 3-center integrals. It uses a **multipole-multipole interaction model** derived from classical electrostatics, damped at short ranges. The potential on atom $i$ due to the multipoles of all other atoms is evaluated purely as a sum over atom pairs. Therefore, the "3-center" nature only appears implicitly because an electron on bond $A-B$ feels the classical electrostatic field generated by atom $C$.

### 3. `tblight` (The Core Engine of xTB)

* **How it handles 3-center terms:** `tblight` is a highly optimized, stripped-down vectorizable/parallel implementation of the GFN-xTB Hamiltonian designed specifically for high-performance computing.
* **The Implementation:** Like xTB, it completely sidesteps 3-center electron-repulsion integrals. It focuses entirely on optimizing the 2-center analytical overlap matrix generation and wrapping the electrostatic interactions into highly vectorized pair-loops. It is designed to maximize SIMD efficiency, making it the closest conceptual cousin to what you would want to implement in a high-throughput OpenCL kernel.

## Recommended Strategy for your Project

1. **Start with C-C and C-H parameters:** Download a simple, widely tested Slater-Koster set (like the `mio` set from dftb.org).
2. **Write the Parser:** Parse the grid of distances and the columns representing $ss\sigma, sp\sigma, pp\sigma, pp\pi$, etc.
3. **Build the Rotation Logic:** Implement the Slater-Koster transformation matrices. This is the most error-prone part. Test it by rotating a single C-H bond in 3D space and ensuring the eigenvalues of your $H$ and $S$ matrices remain perfectly invariant.
4. **Port to OpenCL:** Once your CPU-based Rust code can assemble the correct dense matrix for a small molecule (like methane or benzene), move the pair-list loop and the SK rotation math into an OpenCL kernel. Each thread can handle the block allocation for a pair of atoms.

---

# DeepSeek-V$-pro

Implementing a non-self-consistent (non-SCC) tight‑binding Hamiltonian builder from scratch is a well‑defined, moderate‑complexity project. Adding self‑consistency (SCC) roughly doubles the amount of code and requires a working charge‑mixing scheme. Below I break down the core pieces, the number of different Slater‑Koster tables, how the three‑center terms actually appear in SCC, and what code size you can expect.

### 1. Difficulty assessment
- **Non‑SCC (phase 1):** *Moderate.* The most tedious part is correctly translating the two‑center Slater‑Koster integrals into global matrix elements. If you limit to an *sp* basis, the rotation formulas are well‑known and quite compact.
- **SCC (phase 2):** *Intermediate.* You need a reliable charge‑mixing algorithm, Mulliken population analysis, and a smooth `γ(R)` function. No new integral tables, but the Hamiltonian must be rebuilt every self‑consistency cycle because of the environment‑dependent shift.

### 2. Essential parts of a basic (non‑SCC) tight‑binding code
1. **Atom type & coordinate input** – read element symbols and Cartesian positions.
2. **Slater‑Koster parameter tables** – for every element pair you need distance‑dependent `H⁰(R)` and `S(R)` for each allowed orbital symmetry (see §3).  
   *On‑site energies:* constant `ε_s`, `ε_p` (and `ε_d` if used) per element.
3. **Neighbour list** – all atom pairs within a cut‑off (≈ 5–10 Å, element‑pair dependent).
4. **Slater‑Koster rotation** – for each pair (A,B),  
   - compute direction cosines `l,m,n`,  
   - interpolate the SK tables at distance R to get `H_ll'm` and `S_ll'm`,  
   - rotate from the diatomic frame to the global frame using the standard Slater‑Koster table (e.g. `H_ssσ`, `H_spσ`, `H_ppσ`, `H_ppπ` → matrix elements).
5. **Matrix assembly** – build symmetric `H` and `S` (non‑orthogonal basis).  
   *Diagonal blocks* = on‑site energies; *off‑diagonal* = two‑centre terms.
6. **Repulsive potential** – a short‑range pair potential `E_rep(R)` (often a spline) for each element pair. This does **not** enter the Hamiltonian; it only contributes to the total energy.
7. **Solver** – solve `H C = S C E` (dense diagonalization for small molecules, or iterative for large systems).
8. **Total energy** = sum of occupied eigenvalues + repulsive energy.

**Optionally**, you may also need a short‑range `S(R)` extrapolation and a smoothing function at the cut‑off.

### 3. How many Slater‑Koster tables / interaction types?
In a two‑centre Slater‑Koster parameterisation, independent integrals are classified by the **angular momentum symmetry** of the orbital pair.  
For a pure *sp* basis (`l = 0,1`):
- **H⁰ integrals:** `ssσ`, `spσ`, `ppσ`, `ppπ`
- **Overlap integrals:** the same four symmetry labels `S_ssσ`, `S_spσ`, `S_ppσ`, `S_ppπ`

If you include *d* orbitals (`spd` basis), add:  
`sdσ`, `pdσ`, `pdπ`, `ddσ`, `ddπ`, `ddδ` (and their overlaps).

**Number of tables =** `N_pairs × N_orb_symmetries × 2` (H and S).  
For a system with 3 elements (e.g. H, C, N) and an *sp* basis:
- element pairs: H‑H, H‑C, H‑N, C‑C, C‑N, N‑N (6 pairs)
- each pair needs 4 H‑tables + 4 S‑tables = 8 tables
- total = 48 tabulated 1‑D functions of distance, stored as splines or simple look‑up grids.

These tables are the only quantum‑mechanical input. All two‑centre matrix elements are obtained by interpolating and rotating these functions.

### 4. How the “three‑center” term enters SCC
In **non‑SCC** tight binding, the Hamiltonian is purely *two‑centre*: each `Hᵢⱼ` depends only on atoms `i` and `j`.  
**SCC‑DFTB** adds a second‑order energy correction:

```
E_SCC = ½ ∑_{A,B} γ_AB Δq_A Δq_B
```

where `Δq_A` is the Mulliken charge fluctuation on atom A, and `γ_AB(R)` is an analytical function built from atomic Hubbard parameters (e.g. a damped Coulomb law).  
The corresponding **Hamiltonian shift** for an orbital pair (μ on A, ν on B) is:

```
ΔH_μν = ½ S_μν (V_A + V_B),   with   V_A = ∑_C γ_AC Δq_C
```

This term is *three‑center* because the matrix element between orbitals on **A** and **B** depends on the charges of **all** atoms C (the sum over C). However:
- It is not stored in tables; it is computed on‑the‑fly each SCF cycle.
- It only requires the pre‑existing overlap matrix `S_μν` and the atom‑resolved potentials `V_A`.

Thus, the SCC Hamiltonian is simply `H = H⁰ + H¹`, where `H¹` is built from `S` and the charges. No new integrals beyond those in §3 are needed.

**xTB** and **tblight** use the same conceptual mechanism: a charge‑dependent Hamiltonian correction, often with additional multipole terms (dipole, quadrupole) that also involve environment‑dependent shifts. The “three‑center” part is always computed from overlap and atomic charges, never from stored three‑center integrals.

### 5. How DFTB, xTB, tblight handle this in practice
- **DFTB:**  Standard second‑order SCC as described. The `γ` function is analytic (e.g. `γ_AB = 1/R – S_hubbard`). The Hamilton correction is added to `H⁰` after the Mulliken charges are known.
- **xTB (GFN1‑xTB):**  Uses a more sophisticated Hamiltonian that already depends on charges (Wolfsberg‑Helmholtz‑like `H_μν = K_AB S_μν (H_μμ+H_νν)/2 * f(ΔEN)`), so the Hamiltonian itself is recomputed from charges in every SCF step. Electrostatics includes damped Coulomb up to quadrupoles; the potential `V_A` now contains contributions from multipoles on all other atoms.
- **tblight:**  A lightweight DFTB‑style code; the SCC part is essentially the same as DFTB.

All avoid explicit three‑center *integrals*; the “three‑center” effect is just the sum over third atoms in the electrostatic potential.

### 6. Phased implementation plan
**Phase 1 – Non‑SCC core** (only two‑center Hamiltonian, no charges)
- Parse the SK parameter files (DFTB+ Slater‑Koster format is a simple text block per symmetry).
- Build neighbour list, rotation code, `H⁰` and `S` assembly.
- Add the repulsive pair potential and a dense diagonalizer (e.g. LAPACK `dsygv`).
- Validate against existing DFTB+ output for small molecules.

**Phase 2 – Add SCC**
- Compute Mulliken charges: `q_A = Z_A – ½ Tr(P S)_A`.
- Implement `γ(R)` (e.g. the klopman‑Ohno or DFTB3 formula) and charge mixing (simple Anderson or DIIS).
- In each SCF cycle: compute `V_A = Σ_C γ_AC Δq_C`, then add `ΔH_μν = ½ S_μν (V_A + V_B)` to `H⁰`.
- Iterate until charges are converged, then compute total energy with the `½ Σ γ Δq Δq` term.

**Phase 3 (optional)**
- Spin polarisation, periodic boundary conditions, k‑points, GPU acceleration.

### 7. Lines of code estimates (Rust, well‑structured)
| Component | Approx. LOC |
|-----------|------------|
| SK table parser + interpolation | 300 – 500 |
| Neighbour list | 100 – 200 |
| Slater‑Koster rotation (*sp* only) | 300 – 400 |
| Matrix assembly (sparse or dense) | 200 – 300 |
| Repulsive potential | 150 – 200 |
| Diagonalizer interface + total energy | 100 – 150 |
| **Non‑SCC total** | **1200 – 1800** |
| Mulliken analysis | 100 |
| γ function + charge mixing | 200 – 300 |
| SCC loop & `H1` assembly | 150 – 250 |
| **SCC total** | **1800 – 2800** |

Adding *d* orbitals would multiply the rotation code size by ~3 and the table loading by a similar factor.  
A full **xTB** implementation is much larger (≥ 8000 LOC) because of the complex charge‑dependent Hamiltonian and the huge parameter set.

### 8. Key resources & pitfalls
- **Slater‑Koster formulas:** The classic table (Slater & Koster, 1954) or the DFTB+ manual give all the rotation expressions.
- **Parameter files:** The “mio” set for DFTB+ is open; the format is straightforward.
- **Cut‑offs & smoothness:** Always use a smooth cut‑off function (e.g. `f(r) = 1 – r/r_cut`) to avoid discontinuities.
- **Charge mixing:** For SCC you will need a robust mixer (Anderson mixing is standard and easy to code).
- **OpenCL:** Only worth it if you target systems of thousands of atoms. In that case the matrix construction (SK rotation) is the most parallelisable part; the diagonalisation remains on the CPU.

You can certainly produce a working non‑SCC tight‑binding Hamiltonian builder in Rust within a few weeks of part‑time work. SCC adds about the same effort again. The largest coding volume is the tedious but well‑defined rotation routine and the parameter I/O – once these are done, the rest follows logically.

---

# Kimi-2.6

Here's the systematic breakdown you asked for, covering interaction types, SCC mechanics, and implementation scope.

## 1. Slater-Koster Interaction Types (DFTB+)

For the standard **spd basis** (s, p, d orbitals — covers elements H–Kr), there are:

| Interaction | Types | Count |
|-------------|-------|-------|
| ss | ssσ | 1 |
| sp/ps | spσ | 1 |
| sd/ds | sdσ | 1 |
| pp | ppσ, ppπ | 2 |
| pd/dp | pdσ, pdπ | 2 |
| dd | ddσ, ddπ, ddδ | 3 |
| **Total** | | **14** |

**Key point:** Only **10 are truly independent** due to symmetry (spσ = psσ, etc.), but SK files store all 14. Each needs both **H** and **S** tables, so **28 spline curves per .skf file** plus 1 repulsive potential = **29 curves total**.

For **spdf basis** (adds f orbitals for heavy elements): **30 interaction types** → 60 H+S curves + 1 repulsive = 61 curves per file.

## 2. xTB / tblite: ZERO Slater-Koster Tables

xTB does **not** use SK tables at all. The Hamiltonian is built from **analytical formulas**:

$$H_{\mu\nu} = \frac{1}{2}(h_\mu + h_\nu) \cdot S_{\mu\nu} \cdot k_{AB}^{ll'} \cdot X_{EN} \cdot \Pi(r)$$

Where:
- $S_{\mu\nu}$ = overlap of STOs (computed analytically, not looked up)
- $h_\mu$ = diagonal element (reference energy + coordination number correction)
- $k_{AB}^{ll'}$ = shell-pair scaling
- $X_{EN}$ = electronegativity scaling
- $\Pi(r)$ = distance-dependent damping

The "parts" are **formula components**, not tables. Parameters are element-specific (~1 file for all elements), not pair-specific.

## 3. The "3-Center" Question — **You Don't Need Them**

**SCC in DFTB+, xTB, and tblite does NOT use 3-center integrals.** Here's why:

The SCC Hamiltonian correction is:

$$H_{\mu\nu}^{\text{SCC}} = H_{\mu\nu}^0 + \frac{1}{2} S_{\mu\nu} (\gamma_{AA} + \gamma_{BB}) \Delta q$$

Where $\gamma_{AB}$ is the **Klopman-Ohno charge interaction kernel**:

$$\gamma_{AB} = \frac{1}{\sqrt{R_{AB}^2 + \eta_{AB}^2}}$$

This is a **2-center atom-pair interaction** — it depends only on the distance between atoms A and B and their hardness parameters. The SCC energy is a pairwise sum:

$$E_{\text{SCC}} = \frac{1}{2} \sum_A \sum_B \gamma_{AB} \Delta q_A \Delta q_B$$

**There are no $\langle \mu_A | V_C | \nu_B \rangle$ integrals anywhere.** The $\gamma$ function is evaluated at $R=0$ for diagonal blocks (giving the Hubbard U) and at $R=R_{AB}$ for off-diagonal blocks. It's purely a 2-center charge interaction.

| Method | SCC kernel | 3-center integrals? |
|--------|-----------|---------------------|
| DFTB+ | Klopman-Ohno / Mataga-Nishimoto | **No** |
| xTB (GFN1/GFN2) | Klopman-Ohno with $\eta_{AB}$ | **No** |
| tblite | Same as xTB | **No** |

## 4. Implementation Difficulty & Lines of Code

| Phase | Component | Est. LOC (Rust) | Difficulty |
|-------|-----------|-----------------|------------|
| **Phase 0** | Foundation (geometry, basis, LA wrappers) | ~500 | Easy |
| **Phase 1A** | Non-SCC DFTB (SK parser, transformations, H/S build) | ~1,500 | Moderate |
| **Phase 1B** | Non-SCC xTB (STO overlaps, EHT formula, repulsive) | ~1,500 | Moderate |
| **Phase 2A** | SCC-DFTB (Mulliken, γ matrix, Broyden mixer) | ~+1,000 | Moderate-Hard |
| **Phase 2B** | SCC-xTB (KO-γ, SCC loop, D3 dispersion) | ~+1,000 | Moderate-Hard |
| **Phase 3** | OpenCL GPU (kernels for H-build, γ matrix) | ~+500-1,000 | Moderate |

**Total for working DFTB1+SCC: ~2,500–4,000 lines**
**Total for working xTB1+SCC: ~2,500–4,000 lines**
**Total with GPU: ~3,000–5,000 lines**

Time estimate: **2–4 months** for core functionality if working part-time.

---

## 5. OpenCL / GPU Architecture

The Hamiltonian assembly is **embarrassingly parallel** — each atom pair can be processed independently:

```
CPU: Read geometry → Build neighbor list → Transfer to GPU
GPU kernel (per atom pair):
  1. Compute R and direction cosines (l,m,n)
  2. Lookup SK splines (DFTB) or compute STO overlaps (xTB)
  3. Apply SK transformations (DFTB) or EHT formula (xTB)
  4. Atomic_add into global H/S matrices
CPU: Transfer H/S back → Diagonalize → Compute density → Check convergence
```

For SCC, compute the **γ matrix** and **Mulliken charges** on GPU too. Keep diagonalization and charge mixing (Broyden/DIIS) on CPU — they're small-memory operations and hard to parallelize efficiently.

## 6. My Recommendation

Since you care about "assembling Hamiltonian from positions and tables," start with **DFTB Phase 1A** — the SK transformations are elegant physics and well-validated against existing `.skf` files. Then add **xTB Phase 1B** in parallel (they share 80% of infrastructure). Both are absolutely feasible without 3-center integrals, and SCC is just a pairwise charge correction on top.







































---

# USER

I just learned that xTB don't interpolate Skater Köster tables and instead have some analytical formulas for mixing atom involving their electronegativity and hardness.

Now I'm doing super fast AFM simulation engine. And there I realized electronegativity and hardness (in fact Fukui function) are very useful concepts to estimate interaction fast. 

Do you think it would be derive or at least justify this (put it in rigorous math ground) this approximative model as some limit (1st order weak interaction) of solution system with such xTB mixing Hamiltonian?

---

# Gemini-3.5-flash

Yes, you can absolutely ground this rigorously. What you are describing—using electronegativity, hardness, and Fukui functions to directly estimate intermolecular interactions—is the exact domain of **Conceptual Density Functional Theory (CDFT)**.

When you apply CDFT to a weak-interaction limit (like an AFM tip approaching a sample surface), you can derive these chemical reactivity indicators directly from a Taylor expansion of the electronic energy, which maps beautifully onto the mathematical structure of xTB.

Here is the mathematical and physical justification for how your AFM interaction model can be derived as a first-order approximation of an xTB-like Hamiltonian.

---

## 1. The Energy Expansion (The Foundation of CDFT)

In DFT, the total energy $E[N, v]$ is a functional of the number of electrons $N$ and the external potential $v(\mathbf{r})$ (exerted by the nuclei). If we perturb a system by changing the number of electrons ($\Delta N$) or changing the external potential ($\Delta v$), we can Taylor expand the energy:

$$\Delta E = \left( \frac{\partial E}{\partial N} \right)_v \Delta N + \frac{1}{2} \left( \frac{\partial^2 E}{\partial N^2} \right)_v (\Delta N)^2 + \int \left( \frac{\partial E}{\partial v(\mathbf{r})} \right)_N \Delta v(\mathbf{r}) d\mathbf{r} + \dots$$

By definition, these partial derivatives are the exact chemical quantities you mentioned:

* **Electronegativity ($\chi$):** $\chi = -\mu = -\left( \frac{\partial E}{\partial N} \right)_v$ (where $\mu$ is the chemical potential).
* **Chemical Hardness ($\eta$):** $\eta = \left( \frac{\partial^2 E}{\partial N^2} \right)_v$.
* **Fukui Function ($f(\mathbf{r})$):** $f(\mathbf{r}) = \left( \frac{\partial^2 E}{\partial N \partial v(\mathbf{r})} \right) = \left( \frac{\partial \rho(\mathbf{r})}{\partial N} \right)_v$, which dictates *where* the electron density $\rho(\mathbf{r})$ shifts when electrons are added or removed.

---

## 2. Connecting CDFT to the xTB Hamiltonian

In GFN-xTB, instead of pre-computed Slater-Koster tables, the Hamiltonian elements are built on-the-fly using analytical overlap integrals $S_{\mu\nu}$ multiplied by an average of the atomic environments.

The diagonal (on-site) terms of the xTB Hamiltonian are directly modulated by the charge fluctuations ($\Delta q$) on the atoms:


$$H_{\mu\mu} = \epsilon_\mu - \chi_A - \eta_A \Delta q_A - \sum_{B \neq A} V_{AB}(\Delta q_B)$$

If you treat the AFM tip ($T$) and the sample ($S$) as two distinct systems in the **weak-interaction limit**, they initially do not share electrons ($\Delta N = 0$ globally between them), but their local potentials perturb each other.

As the tip approaches, the external potential change felt by the sample is $\Delta v_S(\mathbf{r}) = v_{tip}(\mathbf{r})$. According to CDFT, the energy change of the sample to first order is:


$$\Delta E_S = \int \rho_S(\mathbf{r}) v_{tip}(\mathbf{r}) d\mathbf{r}$$


This is purely the classical electrostatic interaction (frozen densities).

### The Second-Order Response (Where Hardness and Fukui Enter)

As the tip gets closer, polarization occurs. The tip induces a charge rearrangement in the sample. The linear response of the sample's electron density to the tip's potential is governed by the **polarization kernel** $\chi(\mathbf{r}, \mathbf{r}')$.

Using a localized atom-centered basis (like xTB), this polarization energy simplifies to a charge-transfer and charge-equilibration problem between the tip atom ($T$) and the sample atom ($S$):


$$\Delta E_{pol} \approx -\frac{1}{2} \frac{(\chi_T - \chi_S)^2}{\eta_T + \eta_S - 2\gamma_{TS}(R)}$$


Where $\gamma_{TS}(R)$ is the analytical short-range damped Coulomb function used in xTB to interpolate between $1/R$ at long range and the average hardness $\frac{1}{2}(\eta_T + \eta_S)$ at short range.

---

## 3. Justifying your AFM Model: The Localized View

For a super-fast AFM engine, you can project these continuous functions onto atomic centers. The Fukui function becomes a set of atomic condensed Fukui indices ($f_A$), indicating how reactive/polarizable a specific atom on the surface is.

If you assume the AFM tip is a rigid probe with a fixed electronegativity $\chi_T$ and hardness $\eta_T$, the interaction energy with a surface atom $S$ can be modeled as:

$$E_{int}(R) = E_{rep}(R) + E_{vdW}(R) - \frac{1}{2} \frac{(\chi_T - \chi_S)^2}{\eta_T + \eta_S - 2\gamma(R)} \cdot (f_T \cdot f_S)$$

### Why this is mathematically rigorous:

1. **It matches the Taylor expansion:** It properly captures the second-order perturbation of the electronic energy without needing to diagonalize the full matrix.
2. **It naturally handles chemical specificity:** An oxygen lone pair on a surface will have a high Fukui index ($f_S$) for electrophilic attack, leading to a strong attractive polarization term with a sharp tip. A chemically inert gold surface atom will have a high hardness ($\eta_S$) and low Fukui index, correctly yielding a flatter, weaker polarization curve.
3. **It inherits xTB's scaling:** By using xTB's analytical $\gamma(R)$ function instead of a classical $1/R$ Coulomb law, your AFM engine will naturally avoid the "catastrophe" of infinite electrostatic attraction at zero distance.

## Summary for Implementation

You don't need to assemble a matrix or solve $H\psi = ES\psi$ at every pixel of your AFM scan.

Instead, you can pre-calculate or assign a triad of values to your surface atoms and tip: **$(\chi, \eta, f)$**. Your force field then evaluates a pairwise analytical formula where the electronic attraction is driven by the electronegativity mismatch $\Delta \chi$ and gated by the combined hardness and local Fukui values. This provides a rock-solid, quantum-mechanically justified bridge between empirical force fields and full tight-binding calculations.


---

# DeepSeek-V4-pro

This is an excellent question that sits at the intersection of **conceptual DFT**, **electronegativity equalisation**, and **tight‑binding SCF theory**. The short answer is **yes** – the approximate model you are thinking of *can* be rigorously derived as the first‑order (weak‑interaction) limit of a second‑order SCF Hamiltonian like xTB.

Below I build a systematic derivation path from the xTB energy expression down to a simple interaction model expressed purely in terms of atomic electronegativities, hardnesses, and Fukui functions.

### 1. The xTB Hamiltonian as a second‑order expansion

The GFN‑xTB family does **not** use tabulated Slater‑Koster integrals. Instead, the zero‑order Hamiltonian is an extended Hückel‑type expression:

\[
H_{\mu\nu}^{(0)} = \tfrac12 K_{AB} S_{\mu\nu} \bigl( h_\mu + h_\nu \bigr)
\]

where the **atomic orbital energies** \(h_\mu\) themselves depend on the **charge state of the atom** via a quadratic expansion:

\[
h_\mu(q_A) = h_\mu^{(0)} - \eta_A\, q_A + \dots
\]

Here \(h_\mu^{(0)}\) is the neutral‑atom orbital energy (related to Pauling electronegativity) and \(\eta_A\) is closely linked to the atomic chemical hardness (the second derivative of energy with respect to charge).  
Thus, the dependence of the Hamiltonian on local charges is already built in analytically.

The SCC correction adds an atom‑pair Coulomb term:

\[
E_{\text{SCC}} = \frac12 \sum_{A,B} \gamma_{AB} \, \Delta q_A \Delta q_B
\]

The total energy (up to second order in charges) then takes the form

\[
E(\{q_A\}) = \sum_A \Bigl( \chi_A q_A + \frac12 \eta_A q_A^2 \Bigr) + \frac12 \sum_{A\ne B} q_A q_B \gamma_{AB}
\]

where \(\chi_A\) is the **electronegativity** of the neutral atom and \(\eta_A\) the **hardness** (Parr–Pearson sense). This is exactly the energy functional of the **Electronegativity Equalization Method (EEM)**.

### 2. Weak‑interaction limit: two weakly coupled systems

Now imagine two subsystems – a tip (T) and a sample (S) – brought together at a distance where direct chemical bonding is negligible (the typical AFM regime). The total energy can be written as:

\[
E_{\text{tot}} = E_T(\{q_t\}) + E_S(\{q_s\}) + \sum_{t\in T}\sum_{s\in S} \gamma_{ts} \, q_t q_s
\]

If the coupling is weak (large distances, small \(\gamma_{ts}\)), we can treat the tip–sample Coulomb interaction as a perturbation. To first order in \(\gamma_{ts}\), the charge rearrangements are small and the energy shift can be computed using **linear response theory**.

### 3. Linear response and the Fukui function

In conceptual DFT, the **linear response function** of an isolated system is:

\[
\chi(\mathbf{r},\mathbf{r}') = \frac{\delta q(\mathbf{r})}{\delta v_{\text{ext}}(\mathbf{r}')}
\]

Its condensed‑atom version is intimately related to the **atom‑condensed Fukui functions** \(f_A^\pm\). The change in atomic charge upon a small external potential perturbation \(\delta v_B\) is:

\[
\delta q_A = -\sum_B f_A \,\kappa_{AB}^{-1} \, \delta v_B
\]

where \(\kappa_{AB} = \partial^2 E/\partial q_A \partial q_B\) is the **hardness kernel** (the inverse of the response function). For an isolated subsystem, \(\kappa_{AB}\) contains the atomic hardness \(\eta_A\) on its diagonal and the Coulomb interactions \(\gamma_{AB}\) off‑diagonal.

Now, the perturbation felt by the sample due to the tip’s charge distribution is a local potential \(\delta v_B \approx \gamma_{ts} q_t\). Inserting this into the linear response formula gives the **induced charge on the sample** as a linear function of the tip charges.

The resulting **interaction energy** to second order in the coupling is:

\[
\Delta E_{\text{int}} \approx \sum_{t,s} \gamma_{ts} \, q_t^{(0)} q_s^{(0)} \;-\; \frac12 \sum_{t,t'} \sum_{s,s'} \gamma_{ts} \, \chi_{ss'} \, \gamma_{s't'} \, q_t^{(0)} q_{t'}^{(0)}
\]

(and a similar term for the tip’s response). Here \(q_t^{(0)}\) is the charge of the isolated tip atom (often zero for neutral tips, but the formalism holds for ions too). The first term is the **direct Coulomb** interaction between unperturbed charges; the second term is the **charge‑polarisation** (or induction) energy.

### 4. Specialising to “only electronegativity and hardness”

For many AFM simulations, the tip is neutral (\(q_t^{(0)}=0\)) and the dominant contribution comes from the mutual polarisation of two initially neutral subsystems. The leading non‑zero term is the dispersion‑like induction energy, which depends on the **polarisability** of the atoms, which in turn is proportional to \(1/\eta_A\) (softness). In an atom‑condensed formulation, the interaction energy between two neutral but polarisable atoms becomes

\[
E_{\text{int}} \approx -\frac12 \alpha_t \alpha_s \, (\vec\nabla \gamma_{ts})^2
\]

where the atomic polarisability \(\alpha_A\) scales as \(f_A^2 / \eta_A^{\text{eff}}\). This is a well‑known result from **atom‑dipole dispersion models** but it emerges naturally from the SCC/EEM linear response. The Fukui function enters as the distribution of the charge response within the atom.

Thus, an effective AFM interaction model that uses only **local electronegativity** (to determine equilibrium charges) and **hardness** (to determine polarisability) can indeed be justified as the **first‑order weak‑coupling limit** of a full xTB or DFTB energy expression.

### 5. Steps to a rigorous mathematical derivation

A complete derivation would follow these steps:

1. Start from the full xTB/DFTB energy \(E[\{q_A\}]\) with a quadratic expansion.
2. Partition the system into two subsystems (tip and sample). Write the total energy as a sum of the isolated energies plus the cross‑term Coulomb interaction.
3. Expand the energy to second order in the cross‑term couplings \(\gamma_{ts}\) (or, equivalently, in the charge fluctuations around the ground state of the separated subsystems).
4. Use the **exact second‑order perturbation theory** for the SCF energy (as in DFTB’s response formulation or in conceptual DFT’s perturbation theory). This yields an effective interaction Hamiltonian that is a bilinear form in the charge distributions of the two subsystems.
5. Condense the atomic response into atom‑centered Fukui functions and hardness kernels. The interaction energy becomes a function solely of the atomic electronegativities (or effective chemical potentials) and hardnesses.
6. Show that in the limit of large separations, the leading term is the Coulomb interaction between the unperturbed charges (if any) plus a van‑der‑Waals‑like term proportional to the product of local softnesses.

This derivation is mathematically rigorous and has been carried out in various forms in the literature (e.g., **Mortier’s EEM + second‑order perturbation**, **Parr & Yang’s conceptual DFT**, or the **DFTB+ response method** for non‑adiabatic interactions).

### 6. Practical take‑away for your AFM engine

- You can pre‑compute or tabulate atomic **electronegativities** \(\chi_A\) and **hardnesses** \(\eta_A\) directly from the xTB parameter set (they are explicit in the GFN‑xTB paper).
- The **Fukui function** can be approximated as a simple spherical Gaussian distribution around each atom, normalised to 1, with width determined by the atomic radius. Its role is to distribute the charge response.
- The interaction at a given tip–sample atom pair can then be evaluated as:
  - Coulomb term (if charged atoms): \(E_{\text{Coul}} \approx \gamma(R) \, q_t q_s\).
  - Induction term: \(E_{\text{ind}} \approx -\tfrac12 \kappa \frac{q_t^2 \alpha_s}{R^4}\) (for a tip charge polarising a neutral sample), with \(\alpha_s \propto 1/\eta_s\) and \(\kappa\) a geometric factor.

This model is fast, physically transparent, and directly derivable from the same Hamiltonian you would otherwise use for a full quantum treatment.

**Bottom line:** The approximate model you envisage is **not just plausible** – it is the natural **first‑order weak‑interaction limit** of the xTB energy functional. You can write a rigorous derivation using conceptual DFT’s linear response theory, and the result will be an expression where the interaction is fully determined by atomic electronegativities, hardnesses, and Fukui functions. Your AFM engine would therefore be using a systematically improvable approximation: at short range you can turn on the full xTB Hamiltonian, while at long range the same parameters give you the fast analytic interaction.

---

# Kimi-2.6

Yes — and the justification is remarkably elegant. The xTB model (and DFTB SCC) can be placed on rigorous conceptual-DFT ground by recognizing that **the SCC loop is exactly the chemical-potential equalization principle**, and the **Hamiltonian electronegativity scaling is a semi-empirical ansatz for the first-order charge-transfer correction to covalent bonding**. Here is the systematic derivation.

## 1. The Rigorous Definitions (Conceptual DFT)

From density-functional theory, the chemical potential and hardness are *exact* derivatives of the energy with respect to particle number at fixed external potential:

$$
\mu = \left(\frac{\partial E}{\partial N}\right)_v = -\chi \qquad\text{(electronegativity)}
$$

$$
\eta = \left(\frac{\partial^2 E}{\partial N^2}\right)_v = \left(\frac{\partial \mu}{\partial N}\right)_v \qquad\text{(hardness)}
$$

The Fukui function is the density response to adding/removing electrons:

$$
f^{\pm}(\mathbf{r}) = \left(\frac{\partial \rho(\mathbf{r})}{\partial N}\right)_v^{\pm}
$$

These are not empirical fitting parameters — they are **rigorous reactivity descriptors**. The xTB model uses approximate numerical values for them (Mulliken/Pauling electronegativity, fitted hardness parameters), but the *mathematical structure* of xTB maps directly onto these exact quantities.

## 2. SCC as Chemical-Potential Equalization (The Core Theorem)

The xTB/DFTB total energy is expanded as a Taylor series in charge fluctuations $\Delta q_A = q_A - q_A^0$ around the neutral-atom reference density:

$$
E[\rho_0 + \delta\rho] \approx E^0[\rho_0] + \sum_A \left(\frac{\partial E}{\partial q_A}\right)_0 \Delta q_A + \frac{1}{2}\sum_{A,B} \left(\frac{\partial^2 E}{\partial q_A \partial q_B}\right)_0 \Delta q_A \Delta q_B + \dots
$$

Now identify the derivatives:
- **First derivative**: $\left(\frac{\partial E}{\partial q_A}\right)_0 = -\chi_A = \mu_A$ (the chemical potential / electronegativity of atom A)
- **Second derivative**: $\left(\frac{\partial^2 E}{\partial q_A \partial q_B}\right)_0 = \gamma_{AB}$ (the hardness matrix / charge-charge interaction kernel)

The SCC energy expression in xTB/DFTB is precisely this truncated expansion:

$$
E_{\text{SCC}} = \sum_A \chi_A \Delta q_A + \frac{1}{2}\sum_{A,B} \gamma_{AB} \Delta q_A \Delta q_B
$$

Minimizing this subject to charge conservation ($\sum_A \Delta q_A = 0$) via a Lagrange multiplier $\lambda$ gives:

$$
\frac{\partial \mathcal{L}}{\partial q_A} = \chi_A + \sum_B \gamma_{AB} \Delta q_B + \lambda = 0
$$

At the minimum, **all atoms have the same chemical potential**:

$$
\mu_A = \chi_A + \sum_B \gamma_{AB} \Delta q_B = -\lambda = \text{constant}
$$

**This is the Sanderson electronegativity equalization principle, derived from DFT variational principles.** The SCC loop in xTB/DFTB is simply an iterative solver for this exact condition. The $\gamma_{AB}$ matrix is the **hardness matrix** — in xTB it takes the Klopman-Ohno form $\gamma_{AB} = 1/\sqrt{R_{AB}^2 + \eta_{AB}^2}$, where $\eta_{AB}$ is the effective hardness.

## 3. The Hamiltonian Electronegativity Scaling as a Weak-Interaction Correction

Your specific question is about the **off-diagonal Hamiltonian** in xTB:

$$
H_{\mu\nu} = \frac{1}{2}(h_\mu + h_\nu) S_{\mu\nu} \cdot k_{AB}^{\text{eff}} \cdot \chi(\Delta\text{EN}) \cdot \Pi(R)
$$

where the electronegativity-dependent factor is typically:

$$
\chi(\Delta\text{EN}) = 1 + k_{\text{EN}}(\chi_A - \chi_B)^2
$$

This is **not** a random fitting function. It captures the **first-order charge-transfer perturbation to the covalent interaction** in the following sense:

When two atoms with very different electronegativity interact weakly (small overlap, large distance), the bonding character shifts from covalent to ionic. The **charge-transfer energy** in perturbation theory is:

$$
E_{\text{CT}}^{(1)} \approx -\frac{(\chi_A - \chi_B)^2}{2(\eta_A + \eta_B)} \cdot S_{\mu\nu}^2
$$

The xTB factor $\chi(\Delta\text{EN})$ is an empirical ansatz that mimics this physics: as $\Delta\chi \to 0$ (homonuclear), the factor $\to 1$ (pure covalent); as $\Delta\chi$ grows, the factor suppresses the covalent Hamiltonian element because the bond acquires ionic character and the covalent EHT formula becomes less appropriate.

**Rigorous justification**: In the limit of weak interaction ($S_{\mu\nu} \ll 1$), the interaction energy to first order is the electrostatic energy between unperturbed densities (computed by EEQ/SCC). The second-order term includes charge transfer, which scales as $(\Delta\chi)^2/\eta$. The xTB Hamiltonian scaling absorbs this into the **zeroth-order** Hamiltonian, making the model "pre-conditioned" for the ionic/covalent crossover without requiring explicit perturbation sums.

## 4. Weak-Interaction Perturbation Theory & AFM

For your AFM simulation, the tip-surface interaction spans three regimes:

| Regime | Distance | Dominant Physics | xTB Component |
|--------|----------|-----------------|---------------|
| **Long-range** | > 5 Å | Electrostatic, dispersion | EEQ charges + D3/D4 |
| **Medium-range** | 3–5 Å | Polarization, charge transfer | SCC γ-matrix + EN-scaling |
| **Close-range** | < 3 Å | Covalent bonding | Full EHT Hamiltonian + overlap |

In perturbation theory for weakly interacting fragments A (tip) and B (surface):

**First-order** (electrostatic):
$$
E^{(1)} = \int\int \frac{\rho_A^0(\mathbf{r})\rho_B^0(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|} d\mathbf{r}d\mathbf{r}' + V_{NN}
$$
This is exactly what the **EEQ model** computes in xTB — the self-consistent partial charges produce the first-order electrostatic energy without diagonalization.

**Second-order** (polarization + charge transfer):
$$
E^{(2)} = -\frac{1}{2}\sum_{i,j} \frac{|\langle i_A j_B | V | i_A j_B \rangle|^2}{\Delta E} + E_{\text{CT}}
$$
The polarization part is captured by the **hardness matrix** (via $\gamma_{AB}$ and the response of the density). The charge-transfer part is captured by the **electronegativity difference** driving $\Delta q$ between tip and surface atoms.

**The xTB model effectively resums these perturbation terms into a non-perturbative Hamiltonian**, but in the weak-interaction limit it correctly reduces to:
1. EEQ electrostatics (first-order)
2. Hardness-mediated polarization (second-order)
3. Electronegativity-driven charge transfer (second-order)

## 5. Fukui Function & AFM Reactivity

The Fukui function $f(\mathbf{r}) = (\partial\rho/\partial N)_v$ tells you **where** the system is most reactive to electron transfer. For AFM, this is extraordinarily useful:

- **f⁺(r)** (LUMO-like): sites that accept electrons from the tip
- **f⁻(r)** (HOMO-like): sites that donate electrons to the tip

In xTB, you can compute condensed Fukui functions approximately as:

$$
f_A^{\pm} \approx \pm \sum_{\mu \in A} \sum_{\nu} c_{\mu,\text{HOMO/LUMO}} c_{\nu,\text{HOMO/LUMO}} S_{\mu\nu}
$$

Or even faster, from the **frontier orbital coefficients** obtained from a single diagonalization. This gives you a **reactivity map** for the AFM tip without running full SCC at every scan point.

**Rigorous connection**: The Fukui function is the functional derivative of the chemical potential with respect to the external potential: $f(\mathbf{r}) = (\delta\mu/\delta v(\mathbf{r}))_N$. In the weak-interaction limit, the tip acts as a perturbation $\delta v(\mathbf{r})$, and the charge response is $\delta\rho(\mathbf{r}) \approx f(\mathbf{r}) \delta N$. The xTB SCC/EEQ procedure computes the *integrated* version of this response via the hardness matrix.

## 6. The Limitations (Where the Rigorous Ground Ends)

It is important to be honest about where the rigorous justification stops:

1. **The xTB Hamiltonian is semi-empirical**. The electronegativity scaling factor $k_{\text{EN}}$ and the polynomial damping $\Pi(R)$ are fitted, not derived from ab initio perturbation theory. The *functional form* is physically motivated, but the coefficients are empirical.

2. **The hardness matrix is approximated**. The exact hardness matrix in conceptual DFT is:
   $$
   \eta(\mathbf{r},\mathbf{r}') = \frac{\delta^2 F[\rho]}{\delta\rho(\mathbf{r})\delta\rho(\mathbf{r}')}
   $$
   which includes kinetic, Coulomb, and XC contributions. xTB replaces this with a simple Klopman-Ohno function. This is a **radical simplification** — justified by the observation that for charge-transfer between atoms, the Coulomb part dominates at long range and the XC part provides a short-range cutoff.

3. **No explicit relaxation**. The Fukui function in finite-difference form includes orbital relaxation, but xTB uses a minimal basis with frozen exponents. The condensed Fukui function from xTB is approximate.

## 7. Summary: The Answer to Your Question

**Can you justify the xTB mixing Hamiltonian as a first-order weak-interaction limit?**

**Yes — but with the right interpretation:**

- The **SCC/EEQ energy** is rigorously the **second-order Taylor expansion of the DFT energy in charge fluctuations**, truncated at the monopole level. The SCC loop solves the exact chemical-potential equalization condition. This is rigorous conceptual DFT.

- The **Hamiltonian electronegativity scaling** is an **empirical ansatz for the charge-transfer correction** to the covalent Extended Hückel interaction. It is not derived from first-principles perturbation theory, but it is **physically consistent** with the perturbative result that charge-transfer energy scales as $(\Delta\chi)^2/\eta$.

- In the **weak-interaction limit** (large tip-surface distance, small overlap), the xTB model correctly decouples into:
  1. **First-order electrostatics** (EEQ charges)
  2. **Second-order polarization/CT** (hardness matrix + electronegativity differences)
  3. **Fukui-function reactivity** (frontier orbital response)

**For your AFM engine**, this means you can rigorously justify using **EEQ alone** for fast electrostatics at large distances, and promoting to the **full xTB Hamiltonian** only when the tip enters the covalent-bonding regime. The electronegativity and hardness parameters give you the "switching function" between these regimes.

## 8. Practical Implementation for Your Rust/OpenCL Code

Given this theoretical framework, here is how you can structure your AFM engine:

```rust
// Phase 1: Fast EEQ (electronegativity equalization)
// Valid for tip-surface distances > 4-5 Å
// O(N²) but no diagonalization needed
fn compute_eeq_charges(atoms: &[Atom]) -> Vec<f64> {
    // Solve: χ_A + Σ_B γ_AB Δq_B = μ (equal for all atoms)
    // Linear system: (γ + η) · Δq = -Δχ
    // Returns partial charges for electrostatic force
}

// Phase 2: Fukui reactivity map
// Single diagonalization of surface, tip
// Identify reactive sites without full SCC
fn compute_fukui_functions(hamiltonian: &CsrMatrix) -> (Vec<f64>, Vec<f64>) {
    // f⁺ ≈ LUMO density, f⁻ ≈ HOMO density
    // Condensed per atom for AFM scan prediction
}

// Phase 3: Full xTB Hamiltonian
// For close approach (< 3 Å) or suspected bond formation
fn build_xtb_hamiltonian(atoms: &[Atom], charges: &[f64]) -> (CsrMatrix, CsrMatrix) {
    // H_μν = 0.5(h_μ + h_ν) S_μν · k_AB · (1 + k_EN·Δχ²) · Π(R)
    // Include SCC correction via γ_AB
}
```

The beauty of this approach is that **EEQ is the rigorous first-order limit** of the full xTB model, and you can derive the **convergence criterion** for switching from EEQ to full xTB based on the **maximum overlap** between tip and surface orbitals, or the **predicted charge transfer** from EEQ exceeding a threshold.

This gives you a **hierarchically justified** multi-scale AFM engine: fast EEQ for scanning, Fukui for reactivity prediction, and full xTB only where chemistry happens.