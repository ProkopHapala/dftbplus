Dense multi-system GPU DFTB (H-bond scans, batched SCC). The inner
solve is the open problem: Jacobi is latency-bound, and density-matrix
purification is faster only on a short, device-resident schedule.

**Start at [`Dense_Multi_Performance.md`](Dense_Multi_Performance.md).**
It is the standing order (100× a single CPU thread, publishable
chemistry, no defensive iteration caps) and the map of which files
below are still evidence.

- **Dense_Multi_Performance.md** — goal, then the open problem:
  purification takes 24–40 SCC iterations and a warm density does not
  reduce that count. Jacobi ceiling and the GC batch-256 kernel
  breakdown are still in the same file.
- **Bold_Step_Accuracy.md** — the short commutator geometry step is
  fast and stable and does not converge to a Jacobi SCC. What was
  run, the N = 1…8 scan, and the accuracy-versus-speed hypotheses.
- **TrDH_minimization_purification_Notes.md** — measurement record.
  Parts VI–VIII are the ones that count.
- **Alternative_Dense_Multi_Eigensolve.chat.md** — design discussion.
  The later turns (learned-Λ autopsy, then extrapolation) override the
  earlier ones.
- **Alternative_Dense_Multi_Eigensolve.md** — why the solve has to be
  batched GEMMs, and the fused-TC2 timings.
- **Measured_Facts_Jacobi_Sweeps.md** — closed Jacobi measurements.
- **Dense_Multi_GPU_Optimization.tasks.md** — Jacobi task status and
  the purify follow-on (T11).
- **HBond_Relaxed_Scan_GPU.manifest..md**, **.chat.md**, **.report.md**,
  **.labbook.md**, **.review.md** — H-bond pipeline diary. The 2026-09-09
  review’s blockers were fixed later; do not restart from it.
- **Dense_Jacobi_Eigen_Tiling_Opt.md**, **Divide_and_Conquare_Jacobi.chat.md**
  — Jacobi tiling ideas, measured and mostly closed.
- **Slot_Pool_Scheduler.design.md** — replica scheduler, after the
  inner solve is short.
- **Dense_Multi_PBC.*** , **Dense_Multi_CDFT.*** — periodic scans and
  constrained DFT on the same SCC loop.

Sparse counterpart, different algorithm:
[`../Sparse_Nanocrystal_Vibrations/Sparse_Performance.md`](../Sparse_Nanocrystal_Vibrations/Sparse_Performance.md).
What transfers (residency, short recipes, on-device branch) is listed
in the performance mandate.
