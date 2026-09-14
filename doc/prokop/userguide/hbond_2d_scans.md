# 2-D proton-transfer scans on the GPU — a didactic guide

How to build and run a two-dimensional hydrogen-bond scan with the dense
GPU DFTB engine: hundreds of distinct geometries solved **in parallel in
one kernel launch**, then plotted as a potential-energy surface (PES).

Audience: a student who knows what a hydrogen bond and a proton-transfer
reaction are, and wants to run (and understand) the scans, not read the
solver source. Numerical details are explained as we go.

---

## 1. The physics problem

Many biologically interesting H-bonded dimers have **two** coupled
proton-transfer junctions. Examples in this repository:

| System | Junctions (0-based atom indices) | Chemistry |
|--------|----------------------------------|-----------|
| 7-azaindole dimer | `N9-H20···N23`, `N30-H41···N2` | two N–H···N bonds, tautomerizing |
| diazaphenalene dimer | `N9-H20···N23`, `N30-H41···N2` | two N–H···N bonds |
| pyridone dimer | `N16-H20···O6`, `N5-H23···O17` | two N–H···O bonds (lactam) |
| pyridone isodimer | `O17-H22···N5`, `O6-H23···N16` | two O–H···N bonds (lactim tautomer) |

For each junction we define one coordinate: the donor–proton distance

```
d_i = |r(proton_i) - r(donor_i)|,   i = 1, 2
```

Moving the proton along the donor→acceptor line scans the reaction
coordinate of that junction. Two junctions → a 2-D surface
`E(d1, d2)` on a 20×20 grid = **400 independent replicas**, all solved
by a single batched SCC.

What the surface tells you:

- **minima** = stable tautomers;
- **diagonal path** = synchronous double-proton transfer;
- **edge paths** (one coordinate moves first) = stepwise transfer;
- the comparison answers the classic question *concerted or stepwise?*

---

## 2. The workflow, end to end

Everything is driven by scripts; there is no per-molecule binary.

```
mol2 / xyz input
   │   scripts/make_dimers_2d_scan.py        (convert + auto-detect junctions)
   ▼
endpoint geometries
   │   scripts/relax_dimers_endpoints.rhai   (GPU FIRE relax of each tautomer)
   ▼
relaxed endpoints  ──►  scripts/plot_dimers_relaxed.py   (endpoint figure)
   │
   │   choose a scan scaffold (§5) ──► scripts/make_dimers_movies.py
   ▼
400-frame .xyz movie + DOF figure (scripts/plot_dimers_dofs.py)
   │
   │   scripts/scan2d_<system>.rhai          (the actual simulation)
   ▼
(d1, d2, E) table  ──►  contour map (debug/*_Emap.png)
```

File inventory (all paths relative to repo root):

| Path | Role |
|------|------|
| `data/mol/*.mol2`, `data/xyz/*.xyz` | inputs and generated geometries |
| `rust_dftb/scripts/make_dimers_2d_scan.py` | mol2→xyz, H-bond junction auto-detection |
| `rust_dftb/scripts/make_dimers_endpoints.py` | constructs the transferred tautomer endpoint |
| `rust_dftb/scripts/relax_dimers_endpoints.rhai` | relaxes each endpoint (batch=1 each) |
| `rust_dftb/scripts/make_dimers_movies.py` | writes the 20×20 .xyz movie |
| `rust_dftb/scripts/plot_dimers_dofs.py`, `plot_dimers_relaxed.py` | figures |
| `rust_dftb/scripts/scan2d_{azaindol,pyridone,dzp,dzp_sym,dzp_mid}.rhai` | the scans |
| `rust_dftb/scripts/plot_scan2d.py`, inline plotting blocks | energy maps |
| `debug/*.png` | output figures (never committed) |

Run a scan like any engine job:

```bash
export RUST_DFTB_SK_DIR=/path/to/slakos/mio-1-1
cargo run --release --bin dftb_engine -- --script scripts/scan2d_dzp.rhai
```

---

## 3. Anatomy of a scan script

`scan2d_dzp.rhai` in five steps — read this before writing your own:

```rhai
const NP = 20;                    // grid points per coordinate
const D_LO = 1.0; const D_HI = 2.05;   // Angstrom range of d_i
const J1D = 9;  const J1P = 20; const J1A = 23;   // donor, proton, acceptor
const J2D = 30; const J2P = 41; const J2A = 2;
```

**Step 1 — load the scaffold** (the fixed heavy-atom geometry):

```rhai
load_xyz("dzp", "/path/to/dzp_relaxed_mid.xyz");
let xyz = get_xyz("dzp");              // flat [x,y,z,...] array
```

**Step 2 — build the junction axes.** Each proton is moved along the
donor→acceptor unit vector, not arbitrarily in space:

```rhai
let a1x = xyz[3*J1A] - xyz[3*J1D];  // ... normalize -> unit axis a1
```

**Step 3 — generate the batch.** One flat array, `n_atoms*3` floats per
replica, replica index `b = i1*NP + i2`; only the two mobile protons
differ between replicas:

```rhai
for a in 0..n_atoms {
    if a == J1P      { all.push(xyz[3*J1D] + d1*a1x); /* y,z */ }
    else if a == J2P { all.push(xyz[3*J2D] + d2*a2x); /* y,z */ }
    else             { all.push(xyz[3*a]);  /* y,z */ }
}
```

**Step 4 — one engine, one launch, one SCC:**

```rhai
gpu_new("dzp", SK_DIR, NP*NP);      // compile/allocate once
gpu_set_coords("dzp", all);         // upload all 400 geometries
gpu_smearing("dzp", 0.002);         // Fermi smearing kT in Ha
gpu_reset_q("dzp");                 // neutral start charges
let rms = gpu_scc("dzp", 100, 1e-5);
gpu_eval("dzp", false);             // energies for all replicas
```

**Step 5 — print the map** for plotting:

```rhai
for b in 0..NP*NP { print(ftos(d1)+" "+ftos(d2)+" "+ftos(gpu_energy_i("dzp",b))); }
```

Cold vs warm timing is reported by re-running `gpu_scc` after the first
solve: the same geometries restart from converged charges and need ~8
iterations instead of ~24–64.

---

## 4. Choosing the settings (accuracy/speed tradeoffs)

Measured on an RTX 3090, N=120 (diazaphenalene, 42 atoms), batch=400:

| Knob | Choices | What it costs / buys |
|------|---------|----------------------|
| `batch` | 1 … 1000+ | Per-iteration cost per system drops from ~40 µs (batch 1) to **~1 µs** (batch ≥256). This is the whole point of the batched solver: SCC iterations on 400 systems cost little more than on 1. |
| `gpu_scc` tol | `1e-4 … 1e-6` | `1e-5` converges in ~8–24 iters and energies are stable to ~1e-6 Ha (the f32 eigensolve floor dominates anyway — see §8). Tighter (`1e-7`) costs more iterations and buys little on the f32 path. `1e-8` underflows. |
| `gpu_smearing` kT | `0.001 … 0.005` Ha | Needed for real systems — near degeneracies and the stretched-bond region produce fractional occupations; smearing makes the density a smooth function of charges. `0.002` is the tested default. Too small → SCC stalls; too large → biased energetics. |
| grid `NP` | 20 typical | 400 replicas ≈ a few ms warm. Finer grids are nearly free compute-wise; the limit is your plotting/inspection, not the GPU. |
| `D_LO`/`D_HI` | ~1.0 … rNN−1.0 Å | Cover from a normal bond length to the transferred position. Exceeding `rNN−0.9` puts the proton unphysically close to the acceptor — energies just blow up. |
| SCC max_iter | 100 | A replica that has not converged is reported (`gpu_scc_iters`, `stalled`); never hide it. |

Rule of thumb for the accuracy/speed compromise we converged to:
**f32 bulk arithmetic + f64 only where it is cheap and mathematically
necessary** (scalar energy reductions, DIIS QR pivots, bound checks).
Everything else follows from that.

---

## 5. The scaffold — the subtle part

The scan moves only the two protons; every other atom is frozen at the
**scaffold** geometry. The scaffold is a modeling choice, and it visibly
moves the wells. Three variants we measured (diazaphenalene, same 400
replicas, same protons):

| Scaffold | File | Result |
|----------|------|--------|
| relaxed donor tautomer | `dzp_relaxed.xyz` | donor well at 0, transferred well +11.4 kcal — frozen-scaffold penalty |
| + inversion symmetrized | `dzp_relaxed_sym.xyz` | J1↔J2 asymmetry → 0.003 mHa; same 11 kcal well offset |
| midpoint of the two relaxed tautomers | `dzp_relaxed_mid.xyz` | wells degenerate to **0.56 kcal** |
| **interpolated per replica** | `scan2d_dzp_interp.rhai` | wells degenerate to **0.15 kcal**, both within ~0.5 kcal of the relaxed endpoints |

The last variant blends the heavy-atom scaffold per replica between the
two relaxed endpoints by the mean transfer progress
`s = clamp((d1+d2−2·d_min)/(2·(d_max−d_min)), 0, 1)` (donor axis and
donor positions are interpolated the same way). It approximates the
relaxed surface without paying for 400 relaxations — the cheap stand-in
until the two-constraint relaxed scan exists.

Physics of what you see:

1. **Frozen-scaffold penalty.** Each relaxed tautomer's heavy scaffold
   differs from the other's by ≤0.06 Å near the junctions — worth ~3
   kcal of relaxation energy per well. A fixed-scaffold scan can
   *never* reproduce the relaxed endpoint energies; it shows a
   constrained cut through the full PES.
2. **Mirror symmetry.** If the two junctions are symmetry-equivalent,
   `E(d1,d2)` must equal `E(d2,d1)`. The input geometry had
   r(N···N) = 3.042 vs 3.034 Å → 1.9 mHa (1.2 kcal) asymmetry.
   Enforcing the dimer's inversion symmetry on the scaffold reduced it
   to 0.003 mHa — a good sanity check that the solver is not adding
   spurious asymmetry.
3. **Well degeneracy** is real only to the extent the two tautomers are
   chemically equivalent (here ~0.04 kcal after full relaxation). The
   midpoint scaffold splits the frozen penalty evenly → near-degenerate
   wells; the residual is genuine small asymmetry plus grid sampling
   (~0.3 kcal — the relaxed protons sit slightly off the N···N axis and
   the grid step is 0.055 Å).

Practical advice: for a *test* of solver correctness use the symmetrized
or midpoint scaffold (symmetry violations then come only from the
numerics). For a *production* PES use the relaxed donor scaffold and
accept that the transferred well is offset by its frozen penalty —
or run the relaxed scan (§7).

---

## 6. What the numerics do — a student's tour

### 6.1 The equations being solved

DFTB is a tight-binding model: solve the generalized eigenproblem

```
H c_i = e_i S c_i
```

build the density `P` from occupied states, update atomic charges `q`,
rebuild `H[q]`, and iterate to self-consistency (SCC). All matrices are
small (N ≈ 66–120 orbitals here) but there are hundreds of replicas —
that shapes every design choice.

### 6.2 Orthogonalizing: Löwdin instead of Cholesky

`Hc = eSc` is solved by transforming to an orthonormal basis with
`X = S^{-1/2}`: `XᵀHX` is then an ordinary eigenproblem. Computing
`S^{-1/2}` costs an eigendecomposition of S **unless** you already have
X for a similar geometry. The engine exploits the scan structure:

- within a replica, S changes slowly between SCC iterations and scan
  neighbors → the **Newton iteration**
  `X_{k+1} = ½ X_k (3I − X_kᵀ S X_k)` refines a reused X in a few dense
  GEMMs (the `gemm_th` stages);
- it certifies reuse by measuring `‖XᵀSX − I‖` — if Newton cannot
  repair within 3 steps it falls back to a full Jacobi solve
  (you see `[GpuSccPlan] Löwdin X reuse ...` lines in the log).

Newton on symmetric matrices is the standard trick: it is cubically
convergent in the matrix-norm sense and needs only matrix products —
exactly what GPUs are good at.

### 6.3 The eigensolver: batched Jacobi

Each replica needs `eig(XᵀHX)` every SCC iteration. Two classical
options:

- **Householder QR / LAPACK** — fast per-matrix on a CPU, but it is a
  sequential pipeline of reflectors: terrible at 400-way batching.
- **Cyclic Jacobi** — zero out off-diagonal pairs with plane rotations;
  many rotations are *independent* and can be applied in parallel, and
  the same sweep schedule runs on all 400 replicas simultaneously.

We use Jacobi: each replica is a small matrix living in shared/local
memory of one GPU workgroup; the batch dimension fills the whole
device. With warm starts (eigenvectors from the previous SCC iteration
or scan neighbor are already nearly diagonal) the sweep count is often
zero — the residual is measured first and Jacobi exits early. The
remaining cost is the fixed prologue (norm reductions, diagonal
handling), which is why `scc.jacobi` still dominates device time at
batch 400 (~76%).

### 6.4 Occupations and smearing

Eigenvalues → occupation numbers via a Fermi function

```
f_i = 1 / (1 + exp((e_i − μ)/kT))
```

with the chemical potential `μ` found by bisection on the electron
count. Without smearing, near-degenerate frontier orbitals flip
occupation between SCC iterations and the charge map develops a limit
cycle instead of converging. kT = 0.002 Ha is a small bias (energies
shift ≪ 1 mHa) that buys convergence.

### 6.5 The SCC mixer: DIIS on the GPU

Pulay DIIS extrapolates the next charge vector as a least-squares
combination of the last few iterates' charge+error pairs. The history
lives on the GPU; the tiny (≤8×8) least-squares problem is solved by
**QR in f64** — the one place where higher precision is genuinely
cheap and mathematically justified (the normal equations are nearly
singular by construction). Fallback counters
(`DIIS fallbacks: N total`) are printed, not hidden.

### 6.6 Memory layout and the gather rule

Every per-replica array is laid out **structure-of-arrays, replica-major
where coalescing wants it**: consecutive GPU threads read consecutive
replica data. Two invariants the whole engine obeys (they cost weeks to
learn — don't violate them):

- **Gather, never scatter.** Each thread owns an output location and
  *reads* the inputs it needs. No two threads write the same address →
  no atomics, deterministic results, no serialization.
- **Zero allocation inside solver loops.** Every buffer, plan, and
  kernel is built once at `gpu_new`/`gpu_set_coords` time; SCC and FIRE
  iterations only launch kernels into persistent storage.

### 6.7 Forces and FIRE on the device

Forces use the standard DFTB gradient (Pulay + repulsive +
SCC-response terms) accumulated by the same gather rule — one work-item
owns an atom's force output and gathers pair contributions. Geometry
relaxation is FIRE: velocity-Verlet-like steps with the adaptive
damping (timestep grow/shrink, mixing factor α, velocity zeroing on
`P<0`) evaluated **on the GPU** — the FIRE state never leaves the
device; the host only reads the convergence flag.

### 6.8 Where f32 ends

Measured GPU-vs-CPU differences at convergence: energies ~1e-6 Ha,
max force ~5e-6, max charge ~5e-6, eigensolve residuals ~1e-6. That is
the dense-path f32 floor. Tightening SCC below ~1e-7 does not improve
the energy (the eigensolve/density floor dominates), and `1e-8`
underflows in f32. The map-level consequences we verified: replica
energies are smooth to ~1e-3 Ha across the grid; mirror symmetry is
reproduced to 0.003 mHa when the scaffold has it.

---

## 7. Fixed vs relaxed scans

**Fixed scan (what we ran):** protons move on the frozen scaffold;
everything else frozen. One batched SCC — cheap, and the right tool for
testing and for surfaces where the scaffold genuinely doesn't matter.

**Relaxed scan:** at each grid point, constrain the two donor–proton
distances and relax everything else (outer FIRE, inner SCC). This
recovers the true endpoint energies (both wells land at their relaxed
−53.4199 Ha automatically) and reveals the stepwise/concerted
mechanism with realistic barriers — usually ~10–15 kcal lower than the
frozen cut.

Current status of the API:

- `gpu_relax(name, max_steps, f_tol, scc_tol)` — free relaxation, works
  today (used for the endpoints).
- `gpu_set_constraint(name, i, j, d0_array)` — **one** distance
  constraint; a relaxed 1-D scan is possible today.
- A true relaxed 2-D scan needs **two simultaneous constraints** — that
  extension (constraint buffer + force kernel) is planned work, not
  yet in the CLI.
- Approximation available now: `gpu_freeze_atoms` — freeze the 6
  junction atoms (both donors, protons, acceptors) and relax the rest.
  That is *not* the same as distance constraints (it also freezes the
  donor–acceptor breathing), but it is a usable intermediate.

Recipe for a relaxed scan once the two-constraint API lands:

```rhai
// per replica b: set d1_target[b], d2_target[b]
gpu_set_constraint("dzp", J1D, J1P, d1_targets);   // pair (donor, proton)
gpu_set_constraint("dzp", J2D, J2P, d2_targets);   // second pair (TODO)
gpu_relax("dzp", 300, 1e-3, 1e-5);                 // outer FIRE + inner SCC
```

The host orchestrates the constraint values (one batch, each replica
with its own targets); the GPU does SCC + forces + FIRE per step.

---

## 8. Reproducing the published artifacts

```bash
export RUST_DFTB_SK_DIR=/path/to/slakos/mio-1-1
cd rust_dftb

# 1) convert mol2 -> xyz + detect junctions
python3 scripts/make_dimers_2d_scan.py

# 2) build the transferred endpoint + endpoint figure
python3 scripts/make_dimers_endpoints.py          # writes diazaphenalene_transferred.xyz

# 3) relax all four endpoints (GPU FIRE, ~8 s total)
dftb_engine --script scripts/relax_dimers_endpoints.rhai
python3 scripts/plot_dimers_relaxed.py            # debug/dimers_relaxed.png

# 4) movies + DOF figures
python3 scripts/make_dimers_movies.py             # data/xyz/*_2d_scan_20x20.xyz
python3 scripts/plot_dimers_dofs.py               # debug/*_2d_scan_dofs.png

# 5) the scans (each: cold ~50–950 ms + warm ~2–9 ms for all 400)
dftb_engine --script scripts/scan2d_pyridone.rhai
dftb_engine --script scripts/scan2d_dzp_mid.rhai
```

Timings to expect (RTX 3090): 66-orbital system, batch 400 — cold
~50 ms, warm ~1.7 ms; 120-orbital system — cold ~0.9 s (DIIS fallbacks
on the hardest asymmetric replicas, all recovered), warm ~8 ms.

---

## 9. Interpreting the maps (checklist)

- [ ] Minima where the chemistry says they should be (donor corner,
      transferred corner).
- [ ] Symmetries the molecule actually has: `E(d1,d2)=E(d2,d1)` if the
      junctions are equivalent — our symmetric dzp scaffold: 0.003 mHa.
- [ ] Frozen-scaffold penalty: scan wells sit *above* the relaxed
      endpoints (~3 kcal here). A scan well that looks ~10+ kcal off
      its relaxed endpoint is normal; it is not a solver error.
- [ ] Diagonal vs edge paths → synchronous vs stepwise mechanism.
- [ ] All replicas converged (`gpu_scc_iters`, rms, DIIS fallback count
      is informational, not a failure).
- [ ] No hidden failures: the script prints every energy; grep the
      `(d1,d2,E)` table, do not clip it.

## 10. Reference numbers (2026-09-14)

| Scan | Scaffold | E_min | E_max | key result |
|------|----------|-------|-------|------------|
| azaindole 20×20 | relaxed | −38.0883 | −37.9874 | symmetric to 2e-4 Ha; donor min at (1.05,1.05) |
| pyridone 20×20 | relaxed lactam | −32.3583 | −32.2913 | lactam min; lactim shoulder ~17 kcal (frozen) |
| dzp 20×20 | midpoint | −53.4153 | −53.3362 | two degenerate wells (0.56 kcal); stepwise ~30 vs synchronous ~46 kcal |
| dzp 20×20 | interpolated | −53.4193 | −53.3719 | wells degenerate to 0.15 kcal, ≈ relaxed endpoint energies |

Relaxed endpoints: pyridone lactam −32.35907 / lactim −32.36804
(lactim lower by 5.6 kcal); dzp donor −53.41990 / transferred −53.41984
(degenerate).
