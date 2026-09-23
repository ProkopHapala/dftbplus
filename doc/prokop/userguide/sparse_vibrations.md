# Vibrational frequencies of nanocrystals — tutorial

How to compute a **full vibrational spectrum** of a hydrogen-passivated
nanocrystal (Si–H on silicon, C–H on diamond, or any geometry you have)
with the sparse GPU engine, and how the numbers are produced so you can
judge them yourself. Same program as in [sparse_dftb.md](sparse_dftb.md):
one binary, one Rhai script.

```text
dftb_engine --script my_vib.rhai --sk-dir /path/to/3ob-3-1
```

The radii, the skin, and which Hessian column you want live in the
`.rhai` job. `.rs` / `.cl` are the engine. Change the job, not the solver.

Ready-made jobs: `rust_dftb/scripts/sparse_vib_c330.rhai` (330-atom carbon
particle) and `sparse_vib_ref.rhai` (a geometry you pass in, used below
for adamantane and Si₁₀H₁₆). Older scripts
`sparse_vibrations_si10h16.rhai` and `sparse_vibrations_cube_si65.rhai`
are the pre-2026-09-23 column ladder in §4.

---

## 0. Current recipe (measured 2026-09-23)

Two Hessian columns, same finite-difference step `h = 0.02` Å, same
engine.

| Column | What is held fixed | When to use it |
|--------|--------------------|----------------|
| Frozen density | `D`, `W`, and the charges at the minimum. Only the explicit `dH/dR`, `dS/dR`, γ(R), and the repulsion move. | A full spectrum of a few hundred atoms. On the 330-atom carbon particle this is 0.1 ms/eval, 0.6 s for all 990 columns. |
| FIRE electronic step | Nothing electronic is clamped. Each displaced geometry takes the same update as a relaxation step: two commutators at η = 8, then one McWeeny (`sparse_geom_mode(..., "bold")`). | The column that matches a converged DFTB+ SCC Hessian. On 26-atom cages it is ~0.2 s for the whole matrix. |

`RUST_DFTB_VIB_FROZEN=1` selects the first. Leave it unset, with the bold
geometry mode, for the second. `RUST_DFTB_VIB_HSONLY=1` then zeroes every
3×3 block whose two atoms are not an H/S pair. That drops the long-range
SCC electrostatic second derivative on purpose. Pairs inside the H/S
cutoff are kept, including the diagonal blocks.

### 0.1 Masks and the skin

Three radii, set in the job, not tied to each other:

- **H/S** — `sparse_hs_decay` before any SCC. That is where the table has
  already fallen below 10⁻⁴. 3ob C–C is 5.11 Å. pbc Si–Si is 5.42 Å.
- **Density kernel `r_K`** — follows the gap, not the Hamiltonian. On the
  330-atom carbon particle the relaxation used 8.76 Å. On pbc silicon,
  12 Å leaks charge; 14 Å holds `Tr(KS)` to ~0.02 electrons and 16 Å is
  tighter. Do not set `r_K = r_trunc · 12/7`.
- **`r_Z`** — overlap inverse. The jobs set it equal to `r_K` unless
  `RUST_DFTB_R_Z` says otherwise.

The **skin is one FIRE step**. A step is capped at 0.1 Å, two atoms can
close by twice that, so the jobs pass **0.2 Å**. `set_coords` measures
the step against the previous accepted geometry and does not rebuild the
pair list. A jump larger than `skin/2` aborts. Molecules with ≤ 64 atoms
otherwise get a complete mask and ignore the radii;
`RUST_DFTB_FORCE_GEOM_MASK=1` keeps the geometric H/S mask so a small
cage tests the same cutoff as the particle.

`RUST_DFTB_SCC_RHGATE` defaults to 5×10⁻⁴. A truncated mask floors `R_H`
near 10⁻³; the carbon particle and the cage jobs use `1e-2`. That is the
gate for this mask, not a looser electron count. `|Σq − N_elec| > 0.5`
still aborts.

### 0.2 Which Slater–Koster set

| System | Pack | Why |
|--------|------|-----|
| Carbon / adamantane / the 330-atom particle | `3ob-3-1` | C–C decay 5.11 Å. The particle minimum used here is `debug/sparse_relax/mask_c3ob/after.xyz`. |
| Si–H vibrations | `pbc-0-3` | The set that is close on both the Si phonon and the Si–H stretch. matsci’s Γ optical mode is ~1250 cm⁻¹ and is not elemental silicon. |

### 0.3 Run the 330-atom carbon spectrum

```bash
cd rust_dftb
export CARGO_TARGET_DIR=$HOME/.cargo/shared_target   # if that is your target dir
RUST_DFTB_VIB_FROZEN=1 RUST_DFTB_VIB_HSONLY=1 RUST_DFTB_VIB_BATCH=32 \
RUST_DFTB_SCC_RHGATE=1e-2 \
RUST_DFTB_VIB_HESSOUT=../debug/sparse_vib_c330/hess.txt \
cargo run --bin dftb_engine -- \
  --script scripts/sparse_vib_c330.rhai \
  --sk-dir /path/to/slakos/3ob-3-1
```

The script writes `debug/sparse_vib_c330/freq.txt` (paths inside the
script are from the repository root). At the saved minimum:
E = −385.974 Ha, max |F| = 3.2×10⁻⁵ Ha/Å. **No imaginary frequencies.**
Lowest mode +9.6 cm⁻¹. 134 modes at 2842–2845 cm⁻¹, one per hydrogen.
The H/S mask (5.31 Å = 5.11 + 0.2 skin) kept 11 348 pairs and zeroed
85 874 atom-blocks. The FIRE-step column was not run on this particle:
one such step is ~100 ms, so 990 columns is a few minutes.

From the repository root:

```bash
python3 rust_dftb/scripts/plot_vib_parity.py \
  --title "C330 3ob, frozen density, H/S blocks only" \
  --freq sparse=debug/sparse_vib_c330/freq.txt \
  --hess sparse=debug/sparse_vib_c330/hess.txt \
  --out debug/sparse_vib_c330/c330.png
```

Plots: `debug/sparse_vib_c330/c330_spectrum.png`,
`c330_hessian.png`. `*.png` is gitignored except the folders listed at
the bottom of `.gitignore`; these plot folders are on that list.

### 0.4 Check it against DFTB+ on a cage

Adamantane (C₁₀H₁₆, 3ob) and Si₁₀H₁₆ (pbc-0-3), both at the DFTB+
conjugate-gradient minimum, `SecondDerivatives` with `Delta` = 0.02 Å
in Bohr (0.0378). Our side is the same geometry, so a nonzero force is
the Hamiltonian difference, not a different minimum. DFTB+ energies:
adamantane −23.18101 Ha (our frozen/SCC center −23.18083, max |F| =
8.8×10⁻⁵); Si₁₀H₁₆ −18.23456 Ha (ours −18.23448, max |F| = 1.3×10⁻⁴).

Still in `rust_dftb/` (paths are `../debug`, so the output stays out of the crate):

```bash
RUST_DFTB_FORCE_GEOM_MASK=1 \
RUST_DFTB_VIB_FROZEN=0 RUST_DFTB_VIB_HSONLY=0 \
RUST_DFTB_SCC_RHGATE=1e-2 \
RUST_DFTB_XYZ=../debug/vib_ref/si10h16_pbc/opt.xyz \
RUST_DFTB_OUT=../debug/vib_ref/si10h16_pbc/freq_scc.txt \
RUST_DFTB_VIB_HESSOUT=../debug/vib_ref/si10h16_pbc/hess_scc.txt \
cargo run --bin dftb_engine -- \
  --script scripts/sparse_vib_ref.rhai \
  --sk-dir /path/to/slakos/pbc-0-3
```

`sparse_vib_ref.rhai` puts `r_K = r_Z = 40` Å unless you override it, so
on these cages the kernel is complete and the approximation under test
is the Hessian column, not a chopped density matrix. Set
`RUST_DFTB_VIB_FROZEN=1` and `RUST_DFTB_VIB_HSONLY=1` for the fast column.
Adamantane’s H/S cutoff already contains every pair (0 blocks dropped).
Si₁₀H₁₆ drops 180 of 676 atom-blocks at the pbc H/S radius.

Sorted frequencies, cm⁻¹:

| | DFTB+ SCC | frozen density | FIRE step |
|--|-----------|----------------|-----------|
| adamantane, lowest | −11, −11, −3, then ~0 | −0.6, −0.4, +0.4 | −17, −13, −6 |
| adamantane, C–H | 2906–2992 | 2768–2803 | 2923–3011 |
| Si₁₀H₁₆, lowest | −6, −5, −4, then ~0 | +4.8, +4.8, +4.8 | −1.7, −1.2, −0.9 |
| Si₁₀H₁₆, Si–H | 2101–2158 | 2040–2097 | 2098–2163 |

The FIRE-step column is the one that tracks DFTB+. Root-mean-square of
sorted modes 7…78 is 40 cm⁻¹ on Si₁₀H₁₆ and 42 cm⁻¹ on adamantane, and
the stretch cluster sits on the diagonal
(`debug/vib_ref/si10h16_pbc/scc_corr.png`,
`debug/vib_ref/adamantane_3ob/scc_corr.png`). Frozen density is softer:
C–H about 140–190 cm⁻¹ low, Si–H about 65 cm⁻¹ low. There is no chemical
imaginary mode in any of these spectra. The negatives are a few cm⁻¹ on
the rigid-body slots. DFTB+ itself has three of those, at the same size,
because the optimization stopped at a force of 10⁻⁴ Ha/Bohr. A large
negative (tens of cm⁻¹ and not one of the six soft modes) would be a
saddle. Frozen density on a particle also fails to put all six rigid
modes at zero: C330 has three near +10 cm⁻¹ and the next at +113 cm⁻¹.

From the repository root:

```bash
python3 rust_dftb/scripts/plot_vib_parity.py \
  --title "Si10H16, bold SCC step vs DFTB+" \
  --xyz debug/vib_ref/si10h16_pbc/opt.xyz \
  --dftb-hess debug/vib_ref/si10h16_pbc/hessian.out \
  --freq scc=debug/vib_ref/si10h16_pbc/freq_scc.txt \
  --hess scc=debug/vib_ref/si10h16_pbc/hess_scc.txt \
  --out debug/vib_ref/si10h16_pbc/scc.png
```

`hessian.out` is DFTB+’s matrix in Ha/Bohr², Fortran column order,
wrapped at four numbers per line. The plot script converts it with the
same mass-weighting as the engine (§1.2). Our dumped Hessian
(`RUST_DFTB_VIB_HESSOUT`) is Ha/Å², row-major.

---

## 1. What you are actually computing (the math, gently)

### 1.1 The potential-energy surface and the Hessian

In the Born–Oppenheimer picture the nuclei move on a potential-energy
surface `E(R)` — the electronic energy of the system when the atoms are
frozen at positions `R`. Near a local minimum `R₀` the surface is, to
second order, a multidimensional parabola:

```text
E(R₀ + Δx)  ≈  E₀  +  ½ Δxᵀ H Δx      (gradient is zero at a minimum)
```

The matrix `H` is the **Hessian**: `H_ij = ∂²E / ∂x_i ∂x_j` over all `3N`
Cartesian coordinates. Since forces are `F = −∇E`, the Hessian is also
the **negative Jacobian of the forces**:

```text
H_ij  =  −∂F_j / ∂x_i
```

That second form is what we compute: we do not differentiate the energy
twice, we differentiate the (already analytic) force once.

### 1.2 From Hessian to frequencies

A small displacement `u` oscillates according to `m_i ü_i = −Σ_j H_ij u_j`.
Absorbing the masses gives the **mass-weighted Hessian (dynamical
matrix)**

```text
D_ij = H_ij / √(m_i m_j)
```

whose eigenproblem `D v = λ v` gives the normal modes. With `λ` in
Hartree/Bohr²/amu the angular frequency converts to a spectroscopic
wavenumber via

```text
ν [cm⁻¹] = √λ · 5140.487        (imaginary modes: ν < 0 by convention)
```

so stiff bonds on light atoms (Si–H stretch ≈ 2100–2300 cm⁻¹,
C–H stretch ≈ 2900–3100 cm⁻¹) sit at the top of the spectrum, heavy-atom
framework modes (Si–Si ≈ 60–700 cm⁻¹) at the bottom.

### 1.3 The rigid modes — your built-in correctness check

3 translations + 3 rotations of the whole crystal cost no energy, so a
nonlinear molecule/cluster always has **6 modes at ν = 0**. They are not
input to the calculation — they *emerge*. A converged SCC Hessian (the
FIRE-step column, or DFTB+) puts them at |ν| ≲ 15 cm⁻¹; treat that band
as zero. Frozen density does not: on the 330-atom carbon particle three
modes sit near +10 cm⁻¹ and the next are near +113 cm⁻¹ (§0.4). `n_imag`
counts every negative eigenvalue. A few cm⁻¹ on that soft band is noise.
A large negative that is not one of those six slots is a saddle.

### 1.4 Central differences and why we SCC at every displacement

We compute each Hessian column by displacing coordinate `i` by `±h` and
differencing the forces:

```text
H_:,i  =  −( F(x₀ + h·e_i) − F(x₀ − h·e_i) ) / (2h)      error O(h²)
```

Every displaced geometry is a **new electronic problem**: the charges
must be re-self-consistenced (new `H_scc`, new density matrix `K`,
new forces). That is `3N` columns × 2 displacements = **6N full SCC+
force evaluations** per Hessian — the dominant cost of the whole
procedure, and the reason warm starts matter (§4).

The central difference error is `O(h²)` in `h`, but the *force noise*
`σ_F` enters the column as `σ_F/h`. There is a sweet spot: too large `h`
truncates, too small amplifies SCC noise. For this engine `h = 0.02 Å`
is the calibrated choice (the routine guards `h < skin/2`).

---

## 2. Running it — arbitrary geometry

### 2.1 Minimal script

```rhai
const sk_dir = SK_DIR;
const scc_tol_hess = 1.0e-5;      // see §3.2 — tighten to 1e-6 on small systems
const fd_h = 0.02;                // Å, calibrated plateau

load_xyz("nc", REPO_ROOT + "/data/xyz/my_nanocrystal.xyz");   // or make_geom(...)
let n_orbs = sparse_new("nc", sk_dir, 0.0, 0.0, 0.0, 0.0, 512.0);
sparse_tc2_tol("nc", 1.0e-6);     // below the measured f32 floor it bounces — see §3.3
print("n_atoms=" + itos(sparse_n_atoms("nc")));

// 1. Relax to THIS model's own minimum (frequencies are only meaningful
//    at a stationary point of the same PES).
sparse_relax("nc", 400, 1.0e-4, 1.0e-5);
sparse_scc("nc", 100, scc_tol_hess);
let e = sparse_eval("nc", true);
print("relaxed: E=" + ftos(e) + " Ha  max|F|=" + ftos(sparse_max_force("nc")));
save_xyz("nc", REPO_ROOT + "/debug/my_nc_relaxed.xyz");

// 2. FD Hessian + eigenvalues + modes (writes debug/my_nc_freq.txt).
print("vibrations: " + sparse_vibrations("nc", fd_h, scc_tol_hess,
                                       REPO_ROOT + "/debug/my_nc_freq.txt"));
```

```bash
cargo run --release -p rust_dftb --bin dftb_engine -- \
    --script rust_dftb/scripts/my_vib.rhai --sk-dir "$RUST_DFTB_SK_DIR"
```

Works for **Si–H/Si nanocrystals and C–H/diamond** alike — `matsci-0-3`
contains `Si-Si`, `Si-H`, `C-C`, `C-H`, `C-Si`, `H-H` (check your pack
has *every* species pair present; `sparse_new` fails loudly otherwise).
Nothing in the script is Si-specific — the physics branches only on the
SK tables and masses.

### 2.2 Output format

`my_nc_freq.txt`: header line (`n_atom`, `h`, `scc_tol`, `max_asym`),
then `idx freq_cm-1` for all `3N` modes sorted ascending, then `mode k`
blocks with the unit-normed Cartesian displacement pattern per atom —
enough to animate modes in any viewer. Console prints `n_imag`,
`freq_min`, `freq_max`, and the **Hessian max asymmetry** (numerical
symmetry check — symmetrization is applied but reported, not hidden).

### 2.3 Plot the spectrum

```bash
python3 rust_dftb/scripts/plot_vib_spectrum.py debug/my_nc_freq.txt \
    --out debug/my_nc_spectrum.png                    # stick spectrum

python3 rust_dftb/scripts/plot_vib_spectrum.py debug/my_nc_freq.txt \
    --ref /path/to/DFTB+_job/vibrations.tag           # + reference panel
```

---

## 3. Settings that matter

### 3.1 Masks: `sparse_new(name, sk_dir, r_trunc, taper_w, r_k, r_z, max_deg)`

All radii `≤ 0` = defaults (full SK radius + skin) — the *permissive
wide-mask regime*, fine for first runs and small systems. For >64 atoms
the mask becomes geometric: `r_k` sets the stored density-matrix support
and is the **accuracy knob** (the map itself needs the DM tails —
deg~275 ≈ 1 mHa on R10-class systems, ~full for ~20 µHa). For a first
spectrum just use defaults and `max_deg` comfortably above the expected
degree (512 covers ~300 nbr/atom systems); sweep masks afterwards.

### 3.2 Tolerances

| Setting | Suggested | Why |
|---------|-----------|-----|
| `sparse_tc2_tol` | `1e-6` (large) … `1e-7` (small) | f32 TC2 floor ~`1e-7·(N/150)`; asking below the floor makes iterations bounce, not converge |
| `scc_tol` (relax) | `1e-5` | relaxation does not need tighter |
| `scc_tol` (Hessian) | `1e-5`–`1e-6` | sets force noise `σ_F` → Hessian noise `σ_F/h` |
| `sparse_scc` cap | `80`–`100` | non-convergence fails loudly |
| `fd_h` | `0.02 Å` | calibrated plateau; must be `< skin/2` |

### 3.3 The f32 floor

GPU algebra is f32 by design. There is a residual floor `R_I` below
which TC2 purification cannot descend — tolerance requests below it
waste iterations. The engine reports `R_I` per SCC; set `tc2_tol` one
decade *above* the measured floor, not below it.

---

## 4. Making it fast

The timings in this section are the 2026-09-16 column ladder (frozen
orbital, lite DMM, cold fixq). Stretch errors quoted here (“Si–H ~240
cm⁻¹ too soft”) are that ladder. The 2026-09-23 comparison against a
converged DFTB+ Hessian is §0: frozen density is ~65 cm⁻¹ soft on Si–H
and ~140–190 cm⁻¹ soft on C–H; the FIRE-step column matches the DFTB+
stretches.

The Hessian costs `6N` force evaluations — this is where all the time
goes, so optimize the per-evaluation cost:

1. **Warm starts are automatic and are the single biggest speedup.** Each
   displaced geometry inherits charges `q`, `Z`, and `K` from the previous
   one — typically 3–8 SCC iterations vs ~15+ cold. Do not recreate the
   engine (`sparse_new`) between displacements; `sparse_vibrations`
   reuses the same engine throughout (and restores the original geometry
   afterwards).
2. **Freeze topology once.** `sparse_new` builds the sparse mask +
   SpGEMM plans once; the whole Hessian shares them. `h` must be small
   enough that no new neighbor crosses the mask skin (guard enforces
   `h < skin/2`) — that is *why* `h` should stay small.
3. **Fewer neighbors = faster.** Every SpGEMM's work scales with the
   operand degrees (`K·S·K` ≈ 85% of device time). A narrower `r_k` is
   cheaper per iteration *but* raises the accuracy floor — the measured
   trade: deg175→16 mHa bias, deg275→1.2 mHa, deg330→0.02 mHa on the
   R10-class crystal. Pick the narrowest mask whose bias you can live
   with, and keep it **identical across the whole Hessian**.
4. **Looser SCC during relax, tighter only for the Hessian** (§3.2).
5. **Watch where the time goes** — `RUST_DFTB_PROF=mark` prints a stage
   table (`tc2.ksk`, `tc2.ks`, reads, mixing); `RUST_DFTB_KTIME=1` gives
   true kernel-exec times for the two SpGEMMs (~16% instrumentation
   overhead itself). `RUST_DFTB_TC2_STOP_W=28` enables the
   replay-validated floor detector: ~15–30% fewer iterations on
   plateaued runs, bit-identical energies on tested workloads.
6. **Frozen and lite-DMM columns run B-at-a-time on one GPU**
   (`RUST_DFTB_VIB_BATCH`, see §4.2). Cold fixq/full-SCC columns remain
   serial — variable convergence needs per-replica scheduling
   (deferred, manifest §F5b). `VIB_BATCH>1` without `FROZEN` or `LITE`
   fails loud; there is no implicit scalar fallback.

### 4.1 Measured wall times (RTX 3090, `--release`, warm starts on)

| System | N | mask | evals (6N) | per-eval | relax | Hessian |
|--------|---|------|-----------|----------|-------|---------|
| Si10H16 | 26 | complete | 156 | ~0.5 s | ~30 s | ~1.5 min |
| cube_Si65 | 65 | complete | 390 | ~0.46 s | ~1 min | ~3 min |
| si_sphere_R10 | 330 | deg~175 | 1980 | ~3.5 s | ~5 min | ~1.9 h (extrap.) |
| si_sphere_R10 | 330 | deg~330 | 1980 | ~4.3–7.3 s | ~10 s* | ~2.5–4 h (extrap.) |
| si_sphere_R10 | 330 | deg~330, `RUST_DFTB_VIB_FROZEN` (clamped) | 1980 | **5.5 ms** | ~10 s* | ~15–20 s (extrap.) |
| si_sphere_R10 | 330 | deg~330, `FIXQ+DMUPD+LITE DMM=2 NSMAX=2` | 1980 | **~64 ms** | — | ~2 min (extrap.) |
| si_sphere_R10 | 330 | deg~330, `FIXQ+DMUPD+LITE DMM=4 NSMAX=2` | 1980 | **~105 ms** | — | ~3.5 min (extrap.) |
| si_sphere_R10 | 330 | deg~330, `FIXQ+DMUPD+LITE DMM=4 NSMAX=0` `VIB_BATCH=16` | 1980 | **~53 ms** | — | ~1.7 min (extrap.) |
| si_sphere_R10 | 330 | deg~330, `FIXQ=1` cold | 1980 | ~345 ms | — | ~11 min (extrap.) |
| si_sphere_R18 | 1648 | r_k=12, `FROZEN` B=1, GPU-resident | 9888 | **~1.6 ms** | — | ~16 s evals / 138 s wall† |
| si_sphere_R18 | 1648 | r_k=12, `FROZEN` `VIB_BATCH=16` | 9888 | **~0.27 ms** | — | ~2.7 s evals / 139 s wall† |
| si_sphere_R18 | 1648 | r_k=12, `FROZEN`, CPU ref (`SPARSE_CPU=1`) | 9888 | ~194 ms | — | ~32 min (extrap.) |
| si_sphere_R18 | 1648 | r_k=12, `FIXQ+DMUPD+LITE DMM=4 NSMAX=0` B=1 | 9888 | ~257 ms | — | ~42 min (extrap.) |
| si_sphere_R18 | 1648 | r_k=12, `FIXQ+DMUPD+LITE DMM=4 NSMAX=0` `VIB_BATCH=16` | 9888 | **~208 ms** | — | ~34 min (extrap.) |
| si_sphere_R18 | 1648 | r_k=12, `FIXQ=1` cold | 9888 | ~920 ms | — | ~2.5 h (extrap.) |

\* R10 relax numbers are from the pre-relaxed geometry; a cold relax was
~5 min (75 FIRE steps) at deg175. Warm-tier column errors measured at
h=0.02 Å vs cold fixq: clamped 6.3%, lite-DMM2 3.1%, lite-DMM4 1.0%.
† R18 wall is dominated by the dense 4944×4944 host eigensolve
(~100 s); force evals are ~2% of wall at B=16 (report §15.27).

Two important things this table exposes:

- **A displaced eval is 98%+ SCC.** Measured split at N=330:
  `set_coords` (H0/S + γ + uploads, CPU) ≈ 5 ms, `scc` ≈ 3.4–7.3 s,
  `forces` ≈ 7 ms. Each SCC = 9–17 DIIS iters × a full ~40–55-iter TC2
  purify — the purifier always restarts from K0/P0; the previous
  converged K is not reused (a warm-start variant exists but is
  *refuted*: the truncated-map fixed point repels a warm seed).
  Counterintuitively deg330 is *slower* per eval than deg175 — 12 ms vs
  2.7 ms per K·S·K — and deg175's purifier floor-churns anyway.
- **`RUST_DFTB_VIB_FROZEN=1`** computes the FD Hessian with the **entire
  electronic state clamped** at the central snapshot — `D=D₀`, `W=W₀`,
  `q=q₀` (the "frozen-orbital" / clamped-electron approximation: the
  SCF Lagrangian differentiated with orbitals AND Lagrange multipliers
  frozen). Only explicit geometry dependence runs per eval → **zero
  device products, ~5.5 ms/eval**. Measured Hessian-column error vs
  cold fixq: **~6.3%** (h-independent). On si10h16 frequencies:
  framework modes within ~1–9 cm⁻¹ of full SCC, but Si–H stretches
  ~240 cm⁻¹ (10%) too soft — the charge response stiffens the highest
  modes. Screening/framework tier, not quantitative stretches.
- **`RUST_DFTB_VIB_FIXQ=1`** (recommended production mode) freezes the
  SCC *charges* at the minimum but still solves the density matrix for
  each displaced `H_scc` (one purify, no DIIS loop): si10h16 matches
  full SCC to rms 6.1 cm⁻¹. With **`RUST_DFTB_VIB_TC2TOL=5e-5`** (just
  above the measured purification floor) each displaced eval is ~0.36 s
  on R10 → full Hessian ≈ 12 min.
- **`RUST_DFTB_VIB_MAXCOL=n`** bounds the FD columns and prints
  per-column phase timings — use it to measure/extrapolate instead of
  waiting for a full Hessian (8 columns ≈ 10 s).
- Every ±h column now restores a snapshot of the central converged
  state (`snapshot_electronic_state`) — identical solver history per
  column, no FD asymmetry from chained solves. In GPU mode the snapshot
  is **device-resident** (`GpuCentralState`: `k`/`z`/`k0`/`w0` device
  buffers captured once; restore = device→device copy, no PCIe
  traffic — §15.25).

### 4.2 Multi-replica batch — `RUST_DFTB_VIB_BATCH=B`

Works for **frozen** and **lite-DMM** (`VIB_FIXQ+DMUPD+LITE`) column
modes — the two fixed-cost recipes. Runs `B` displaced geometries
**per kernel-launch wave** on a single GPU: every eval kernel gets a
second NDRange axis `get_global_id(1) = b` = the replica slot.
**Each replica is a complete, independent system eval** — full physics
of one displaced geometry against shared read-only central state —
not a partitioned system.

```bash
RUST_DFTB_VIB_FROZEN=1 RUST_DFTB_VIB_BATCH=16 dftb_engine \
    --script my_vib.rhai --sk-dir $RUST_DFTB_SK_DIR
# or the ~1%-error tier:
RUST_DFTB_VIB_FIXQ=1 RUST_DFTB_VIB_DMUPD=1 RUST_DFTB_VIB_LITE=1 \
RUST_DFTB_VIB_DMM=4 RUST_DFTB_VIB_NSMAX=0 RUST_DFTB_VIB_BATCH=16 \
    dftb_engine --script my_vib.rhai --sk-dir $RUST_DFTB_SK_DIR
```

**Frozen path** — kernels `gamma_matvec`, `gamma_force`, `hs_contract`,
`rep_eval`, `force_gather`:

- **Shared:** pair topology, SK/repulsive tables, `dq₀`, central
  `K₀`/`W₀` — all read-only during the batch.
- **Replicated:** `xyzu`, `v_atom`, pair-force records, force outputs —
  ~7 MB/replica at N=1648.
- **Measured at R18:** B=1 ~1.6 ms/eval → B=8 0.31 → **B=16 0.27
  ms/eval** (~5.9×) → B=32 saturated. Full 4944-column Hessian: ~2.7 s
  in force evals; wall then dominated by the dense host eigensolve
  (~100 s at this size).

**Lite-DMM path** — the replica axis extends through the whole solve
chain (`hs_diag`/`hs_assemble` → γ → `build_Hscc` → [warm-NS] →
`B=Z·H` → `n_dmm`×3 SpGEMMs + updates → `W=2·sym(B·K)` → contract +
gather). Replica strides: **block strides on SpGEMM plans, element
strides on elementwise/contract kernels, 0 = shared operand**.

- **Shared:** all sparse structures + symbolic SpGEMM plans, `dq₀`,
  S operand in `T=K·S`, K₀/Z₀ seed (broadcast once per batch).
- **Replicated:** geometry + ~10 BSR value slabs per replica (h, s,
  hscc, z, k_ev, b_zh, t_ks, t_zs, x, y, w + force buffers) —
  ~310 MB/replica at R18 → B=16 ≈ 5 GB.
- **Measured, lite-DMM4:** R10 107.8 → **~53 ms/eval** at B=16 (~2×);
  R18 257 → **~208 ms/eval** at B=16 (~1.24×). Modest gain is expected:
  the chain is *memory-bound* (15 fat SpGEMMs/eval on deg-378 masks;
  each plan term is a scattered 64 B B-block read, ~2 FLOP/B, already
  saturating the GPU at B=1) — batching removes host-side
  per-eval overhead, not parallel headroom. Saturation at B=8.
- **Guarantees (both paths):** bitwise-identical to the sequential path
  (same kernel math per replica, no atomics, gather-only writes;
  `test_sparse_frozen_batch_parity` + `test_sparse_dmm_batch_parity`,
  max|dF| = 0.0). B=1 is exactly the sequential loop.
- **Why only fixed-cost modes:** frozen has zero iterations and lite
  runs a predetermined step count — both are uniform one-shot jobs, so
  static batching is optimal. Cold fixq has *variable* purify
  iterations per replica; batching it needs either active-mask
  scheduling (manifest §F, deferred) or the proposed fixed-M TC2 recipe
  (F5b, not yet implemented). `VIB_BATCH>1` outside `FROZEN`/`LITE`
  **fails loud** — no silent fallback.
- CPU reference mode (`RUST_DFTB_SPARSE_CPU=1`) rejects `VIB_BATCH > 1`
  with an explicit error — there is no implicit fallback.
- **`RUST_DFTB_VIB_DMUPD=1`** (warm density update, needs `FIXQ=1`):
  reuses the central projector and descends the generalized-commutator
  residual via **DMM steps** `δK = −η(X + Xᵀ − 2Y)`, `X = (Z·H)·K`,
  `Y = (K·S)·X` — 3 SpGEMMs/step, per-step K symmetrization. Recommended
  seedless (`VIB_SEED=0` — the δK0 initializer seed contaminates the
  manifold; measured refuted). **Production tiers** (h=0.02 Å, R10):
  `VIB_LITE=1 VIB_DMM=2 VIB_NSMAX=2 VIB_NSTOL=3e-5` → ~64 ms/eval @ 3.1%
  column error; `VIB_DMM=4` → ~105 ms @ 1.0%. `VIB_LITE` strips ALL
  per-eval residual gates/measurements (calibrated recipes only — use
  the gated `DMUPD` path, ~175 ms, to validate a new recipe first).
  `VIB_LINEAR=1` is a 4-product single-response tier — measured refuted
  (6.7%, dominated by the free clamped mode). Knobs: `VIB_DMM` /
  `VIB_DMM_ETA` (η·Δε, 8) / `VIB_DMM_RET` (retraction interval — now 0
  in lite; retractions *hurt*, H-blind) / `VIB_NSMAX`/`VIB_NSTOL` (warm
  NS cap — 1 Newton update is what unlocks <6% accuracy) /
  `VIB_GATES=0` (skip R_H certification). Report:
  `doc/prokop/reports/2026-09-16_sparse_dmm_warm_density_hessian.md`.

Full bottleneck analysis + recommendations:
`doc/prokop/topical_audit/hessian_eval_bottleneck.md`.

---

## 5. Reading the result honestly

- **Rigid modes ≈ 0** (|ν| ≲ 15–20 cm⁻¹): required sanity check.
- **Degeneracies**: symmetric crystals (cubes, spheres) show
  doublets/triplets — a smooth continuum instead means trouble.
- **Expected bands**: Si–Si framework 60–700, Si–H bend ~600–950,
  Si–H stretch 2100–2300; C–C frame up to ~1300, C–H stretch ~2900–3100
  cm⁻¹.
- **`max_asym`** is the FD+truncation asymmetry — ~1e-3 is normal at
  `scc_tol=1e-5`.
- **Known caveat (2026-09):** vs the DFTB+ reference on cube_Si65 the
  spectrum is qualitatively correct but systematically **stiff by
  ~5–15%** (mean |Δ| = 74 cm⁻¹) — traced to a ~1 Ha energy-parity gap,
  under investigation, not mask/f32 noise. Treat absolute values as
  method-level estimates, not benchmark-grade, until that gap closes.
