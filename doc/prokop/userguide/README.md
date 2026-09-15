User-facing documentation for the Rust DFTB engine. Write these as if a person will run the program, not as agent notes. Internal status lives in `doc/prokop/tasts/` and `doc/prokop/topical_audit/`.

- [dftb_engine.md](dftb_engine.md) — the CLI: one binary, input scripts, dense GPU DFTB (`gpu_*`).
- [sparse_dftb.md](sparse_dftb.md) — the same CLI for sparse BSR4 DFTB (`sparse_*`).
- [sparse_vibrations.md](sparse_vibrations.md) — vibrational frequencies of passivated nanocrystals: FD Hessian, settings, speed, and the math behind it.
- [hbond_2d_scans.md](hbond_2d_scans.md) — didactic guide to 2-D proton-transfer scans: junction identification, endpoint relaxation, scaffold choice, batched SCC, energy maps, and the GPU numerics behind it.
- [hbond_pbc_scans.md](hbond_pbc_scans.md) — the periodic variant (`pbc_*`/`GpuPbc`): ASCII-art chain builder, boundary-crossing junctions, k-points, Ewald γ, and why PBC gives degenerate endpoints a dimer can't.
- [cdft.md](cdft.md) — constrained DFT on the dense GPU solver (`gpu_cdft_*`): fragment Mulliken-charge constraints for charge-localized diabatic states (PCET/ET surfaces), per-replica targets, λ-secant loop, energy conventions.
