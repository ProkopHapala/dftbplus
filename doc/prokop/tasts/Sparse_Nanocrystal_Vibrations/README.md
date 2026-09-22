Sparse GPU DFTB for Si/H nanocrystals. The clock is the warm
electronic solve: reuse `H,S,D` from the previous geometry on the
next FIRE/L-BFGS step, and on a few finite-difference displacements.
A full Hessian and its diagonalization are not the measurement.

**Start at [`Sparse_Performance.md`](Sparse_Performance.md) §0.**
That section is the standing order. The rest of the file is the
evidence (why a cold purify is too expensive, what the warm density
update costs, why the SpGEMM is bandwidth-bound).

- **Sparse_Performance.md** — what to time, the DMM ladder, the
  SpGEMM ceiling, the path.
- **Warm_Geometry_DM.md** — geometry-step variants that reuse the
  previous density. §7 is the gated SiH₄ trial (warm accept only near
  0.01 Å). §8–§9 are the recipe that is on the clock: two commutator
  steps, one McWeeny, keep the kernel. §9 is the R10 relaxation
  (energy −311 → −314 Ha, `Tr(KS)` on 459, ~180 ms/step) and the
  R10/R14 step-time scaling. No R14 trajectory was written.
- **Sparse_Nanocrystal_Vibrations.report.md** — measurement record.
  §15.16–§15.28 are the ones that count.
- **Sparse_Nanocrystal_Vibrations.chat.md** — design discussion. The
  Tier-1 rebuttal (~line 11600 through ~13870) overrides the earlier
  review checklists.
- **Sparse_Nanocrystal_Vibrations.manifest.md** §F — residency and the
  batch plan. F1 and F5a are measured; F2/F5b are still open and are
  ordered by the performance note, not by their position in the file.
- **Sparse_MultiSystem_Scheduler.chat.md** — slot vocabulary for a
  uniform batch. The ending of that chat is a dense-solver argument.
- **Sparse_Nanocrystal_Vibrations.tasks.md**, **.review.md** — phase
  list and the 2026-09-09 review. Historical.

Dense counterpart, different algorithm:
[`../HBond_Relaxed_Scan_GPU/Dense_Multi_Performance.md`](../HBond_Relaxed_Scan_GPU/Dense_Multi_Performance.md).
What transfers (short fixed recipes, certificates off the clock) is
listed in both mandates.
