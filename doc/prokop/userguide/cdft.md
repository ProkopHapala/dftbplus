# Constrained DFT (CDFT): charge-localized diabatic states

This guide shows how to run **constrained SCC-DFTB** on the GPU engine:
fixing the Mulliken charge of a *fragment* (a group of atoms) to a target
value while the electron density stays self-consistent. This is the tool
for **charge-localized diabatic states** — e.g. "the extra electron sits
on molecule A" vs "on molecule B" — which is what you need for
proton-coupled electron transfer (PCET), Marcus theory, and CDFT-CI.

## 1. The physics in one paragraph

Ordinary SCC-DFTB minimizes the energy over charge transfers. If the true
(adiabatic) ground state has charge delocalized or sitting on the "wrong"
molecule, you cannot prepare the *diabatic* state you want. CDFT adds a
constraint with a Lagrange multiplier:

    minimize  E_DFTB + λ_F · (Q_F − Q_F^target),   Q_F = Σ_{A∈F} Δq_A

`Δq_A` is the Mulliken excess charge on atom A (e⁻ count above neutral).
The solver finds the multiplier λ_F that forces the fragment's total
excess charge to the target. Under the hood the constraint is just an
on-site potential shift `V_A → V_A + λ_F` — exactly the same machinery
the SCC Hamiltonian already uses — so it costs one extra kernel launch
per Hamiltonian rebuild and an outer loop over λ on the host.

> **Closed-shell caveat.** The constraint localizes *charge*, not *spin*.
> A D⁺A⁻ state has two radicals; this closed-shell implementation does
> not resolve the singlet/triplet coupling. For "where does the electron
> sit" surfaces it is the right tool; for open-shell energetics it is a
> first approximation.

## 2. Minimal example

```rhai
// H2O, batch of 2 replicas; constrain the oxygen's excess charge.
make_geom("w", "O,H,H",
  [0.0,0.0,0.0, -0.7580632,0.6358101,0.0, 0.7580632,0.6358101,0.0]);
gpu_new("w", SK_DIR, 2);
gpu_scc("w", 100, 1e-6);            // ordinary ground state first

// frag: one entry per atom — -1 = unconstrained, else fragment id.
// targets: per-replica (b,f) targets, flat [batch*nfrag], or [nfrag]
// broadcast to all replicas. Here replica 0 wants Q_O=+0.10, replica 1 +0.20.
let nf = gpu_cdft("w", [0, -1, -1], [0.10, 0.20]);

// outer-λ solve: SCC at fixed λ, secant update, until |Q_F−target| ≤ 1e-4.
let err = gpu_cdft_scc("w", 30, 60, 1e-6, 1e-4);
print("converged, max charge error = " + err);

print("Q_O replica0 = " + gpu_cdft_qf("w", 0, 0));
print("lambda replica0 = " + gpu_cdft_lam("w", 0, 0) + " Ha");

// E_DFTB of the constrained state (the λ·Q term is removed):
gpu_cdft_energies("w");
print("E_constrained replica0 = " + gpu_energy_i("w", 0));
```

Run it:

```bash
cargo run --release --bin dftb_engine -- \
  --script my_cdft.rhai --sk-dir /path/to/mio-1-1
```

## 3. Rhai API

| function | returns | meaning |
| --- | --- | --- |
| `gpu_cdft(name, frag, targets)` | nfrag | attach constraints; `frag[a]` = fragment id or −1; `targets` flat `[batch*nfrag]` or `[nfrag]` (broadcast) |
| `gpu_cdft_scc(name, max_outer, scc_iter, rms_tol, q_tol)` | max charge err | outer-λ loop: full warm-started SCC at fixed λ, then secant update |
| `gpu_cdft_qf(name, b, f)` | Q_F (e) | fragment excess charge of replica b |
| `gpu_cdft_lam(name, b, f)` | λ (Ha) | converged multiplier |
| `gpu_cdft_set_lam(name, b, f, lam)` | — | set λ manually → `gpu_scc` + `gpu_cdft_qf` maps the Q(λ) response curve |
| `gpu_cdft_energies(name)` | E (Ha), replica 0 | `eval` minus Σλ·Q_F → fills `gpu_energy_i` |
| `gpu_cdft_clear(name)` | — | remove the constraint set |

`gpu_eval` after `cdft_scc` still works and returns the *augmented* value
`E_DFTB + Σλ_F·Q_gross(F)` — the on-site shift enters e_band through the
**gross** Mulliken population (q0+Δq), not just the excess. Always use
`gpu_cdft_energies` (which subtracts `λ_F·(Q_F + Q0_F)`) for the physical
constrained-state energy; the raw eval offset is `~λ·N_e(F)` — tens of
Hartree for a whole monomer, NOT small.

## 4. What it is good for

- **PCET / electron-transfer surfaces.** Along a proton scan, run two
  batches (or replicas with different targets): electron on donor vs on
  acceptor. `E_R(x_H) − E_L(x_H)` is the driving force; its zero crossing
  is the Marcus degeneracy point.
- **Diabatic-state ladders.** Different targets per replica in ONE batch —
  the λ values and the whole solve stay on the GPU; only the scalar
  λ-secant runs on the host.
- **Charged-region embeddings** — constrain a subsystem's charge in a
  larger aggregate.

**Worked example — pyridone-dimer PCET maps:**
`rust_dftb/scripts/scan2d_pyridone_cdft.rhai` reproduces the 20×20
double-proton-transfer grid three times: neutral (reference), Q₁=−1 e
(M1⁺M2⁻), Q₁=+1 e (M1⁻M2⁺) — 400 replicas each, ~5 s per CT map.
Plot with `scripts/plot_scan2d_cdft.py`. Result
(`debug/pyridone_2d_scan_cdft.png`): oxidizing M1 (Q₁=−1) collapses the
double well into a single minimum at (d1≈1.78, d2≈1.02) — the proton on
junction 1 is pulled onto the oxidized monomer's oxygen, exactly the
expected PCET coupling. Spot check at (1.0, 1.28): E(CT⁻) − E(neutral)
= +0.250 Ha, and releasing the constraint warm-started from the CT state
returns the neutral energy to ~1e-6 Ha.

## 5. Settings, convergence machinery, and reachability

- `scc_iter` (inner iterations) 60–100; each outer iteration warm-starts
  from the previous solution. Small systems converge in 4–8 outer iters;
  hard charge-transfer targets can take 20–40.
- `q_tol` = 1e-4 e sane default; use 1e-3 for large |target|.
- Targets are **excess Mulliken charge** (Δq sum): positive = extra
  electrons. λ > 0 pushes electrons OFF the fragment.

The outer solver does four things automatically:

1. **Target continuation** — the effective target is ramped from the
   natural fragment charge to `target` over the first 8 outer iters, so
   the constrained state tracks the basin adiabatically connected to the
   neutral state instead of jumping.
2. **Damped secant** on λ (per replica, host f64).
3. **Basin reset** — a replica whose error stalls for 3 outer iters gets
   its charges reset to q0 and λ→0, to re-approach from the
   neutral-connected basin.
4. **Best-λ restore** — at exit, replicas that never hit `q_tol` get the
   best λ sampled (plus one final SCC), so reported energies are always
   the closest-achievable constrained state.

**Reachability is physics, not a bug.** Q(λ) is smooth within a basin but
the SCC has metastable self-consistent solutions separated by ~1–2 e
(closed shell moves charge in pairs). If the target falls between two
basins, no λ reaches it — the replica converges to the nearest achievable
state instead. Measured on the pyridone 20×20 PT grid (monomer targets
±1 e): **385/400 and 371/400** replicas converged; the rest are points
where the fully-transferred Mulliken state does not exist. Diagnostic:
`gpu_cdft_set_lam` + a λ-sweep prints Q(λ) directly (see
`scripts/` `cdft`-related examples).

- Forces on the constrained surface are consistent at fixed λ (the
  −λ·dQ/dR term comes through the existing dS/dR force machinery), so
  `gpu_relax` / `gpu_fire_step` on a constrained state is meaningful once
  `cdft_scc` has converged — the small `(Q−target)·dλ/dR` term vanishes
  at convergence.

## 6. How it works inside (short version)

1. `gpu_cdft` uploads the atom→fragment map and creates the λ buffer.
2. Every time the SCC loop rebuilds `H_scc` from the current charges, one
   extra kernel (`cdft_hscc_shift_batched`) adds `½λ_F·S·(w_μ+w_ν)`.
3. `cdft_scc` alternates: converge SCC at fixed λ → read Δq → host-side
   f64 fragment sums → secant update λ → repeat.
4. `cdft_energies` subtracts the λ·Q_gross offset the shifted band
   structure introduced.

Spec and internals: `doc/prokop/tasts/HBond_Relaxed_Scan_GPU/Dense_Multi_CDFT.spec.md`,
`doc/prokop/topical_audit/cdft_constraints.md`.
