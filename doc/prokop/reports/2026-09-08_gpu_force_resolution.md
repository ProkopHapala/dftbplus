# GPU force resolution report (2026-09-08)

Status: implemented and agent-verified on NVIDIA GeForce RTX 3090; awaiting user confirmation before marking complete in the roadmap.

## Symptoms

- Real H2O crashed in the 1x4 (s-p) force bucket. `clFinish` reported `CL_INVALID_COMMAND_QUEUE` after the s-s bucket completed.
- After removing the crash, heterogeneous systems produced physically wrong forces: H2O initially had relative GPU/CPU error 1.094 and formic dimer 0.0462.
- The synthetic "sp3" test was not a 1x4 control: two four-orbital atoms exercise only the 4x4 path.

## Root causes and corrections

1. **Unsafe local-memory vector reinterpretation** (`rust_dftb/src/qmqm/gpu_forces.cl`)
   - The 1x4 interpolator cast a `__local float*` array to `__local float2*` and NVIDIA faulted at the first interpolation.
   - Replaced the cast-based reads with alignment-safe `vload2` reads. The exact interleaved stride-2 interpolation math is unchanged.
   - Retained single-work-item diagnostic prints behind `GPU_FORCE_DEBUG=0`.

2. **Species-pair orientation was discarded** (`rust_dftb/src/qmqm/gpu_prep.rs`)
   - 1x4 pairs are deliberately oriented with the s atom as `atom_i`, but bucket construction then used `(min(species_i), max(species_j))` and could select the reverse SK table.
   - Buckets now preserve ordered `(species_i, species_j)` keys and enumerate all ordered species pairs. This also fixes heterogeneous 4x4 forward/reverse channel selection.

3. **One-based SK grid was shifted during downsampling** (`rust_dftb/src/qmqm/gpu_prep.rs`)
   - DFTB SK row `k` is located at `(k+1)*dr`. The old code resampled rows as if row zero were at `r=0`, then prepended a zero afterward. When `dr_new != dr_orig`, this shifted the entire GPU table.
   - The `r=0` dummy is now included before resampling, preserving the physical origin and cutoff. H2O relative force error fell from 2.72e-2 to 1.53e-6.

4. **Missing radial units on the separate p-s derivative** (`rust_dftb/src/qmqm/gpu_forces.cl`)
   - `(ss, sp, pp_sigma, pp_pi)` derivatives were multiplied by `1/dr`, but the separately returned `dps` scalar was not.
   - Applied `dps_h *= inv_dr` and `dps_s *= inv_dr`. This removed the remaining C/O force error and improved the synthetic 4x4 relative error from 1.56e-3 to 1.94e-6.

## Verification

All commands ran synchronously with full output on the RTX 3090.

- `cargo test --test gpu_forces -- --nocapture`: **5 passed, 0 failed**
  - H2: relative error 1.50e-6
  - tilted H2: 5.14e-7
  - synthetic 4x4: 1.94e-6
  - real H2O: 1.53e-6; max absolute error 5.37e-7 Hartree/Angstrom
  - real formic dimer: 8.38e-6; max absolute error 3.84e-6 Hartree/Angstrom
  - Net GPU force was <= 1.86e-8 Hartree/Angstrom for all real cases.
- `cargo test --test gpu_hamiltonian -- --nocapture`: **4 passed, 0 failed** using a complete C/H/N/O SK set.
- `cargo test --test gpu_scc -- --nocapture`: **5 passed, 0 failed**, including H2O, 10-replica H2O, and a 72-orbital H2O cluster.

No unrelated files were edited or cleaned up. The worktree already contained many pre-existing changes; this work touched only the force kernel and shared GPU pair preparation.
