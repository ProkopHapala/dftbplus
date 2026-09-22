# Relax solver — implementation notes and open questions

> **Standing order (2026-09-22):** [`Dense_Multi_Performance.md`](Dense_Multi_Performance.md).
> Performance first — ≥100× one CPU thread, publishable chemistry.
> Inside this file, **Parts VI–VIII** are the measurements that count;
> Parts I–II are the superseded Λ / LNV campaign.

Working notes distilled from `Alternative_Dense_Multi_Eigensolve.chat.md` plus what we
learned implementing it. Purpose: a clean spec to discuss with another LLM before
changing the algorithm further.

Status: **experimental, NOT integrated**. Cold start works; warm start is broken.

**Update (Part II below, canonical-force experiments):** the learned-Λ scheme in
§1–9 is superseded by the exact-tangent-force formulation (`G=[D,[D,H]]` +
McWeeny retraction) — which turns out to be *already implemented* as
`dmm_update_batched` + `mcweeny_combine_batched` inside `PurifyScc`
(`gpu_scc_plan.rs`) and in the sparse path (`dmm_descend`). Part II documents
the measurement campaign, a product-order bug found in my own reimplementation,
the accuracy/iteration-count analysis, and the resolution of the
"warm-start slower than cold purification" paradox.

---

## 1. Goal

Replace the tiled multi-Jacobi eigensolver for small dense systems
(n ≈ 86 orbitals, batch ≈ 16–400 replicas, f32, one WG per system) by direct
minimization of

    E[D] = Tr(D·H),   H = H[D] charge-dependent (SCC)

subject to approximate idempotency `D²≈D` and `Tr D = N_occ`.

Motivation: Jacobi is dominated by serialized sweep rounds and sync latency at this
size; our symmetric matrix-square primitive (`square_regtile_kern`) is
disproportionately fast, so a scheme built from 1–2 resident squares per iteration
should win if its iteration count is reasonable.

## 2. Algorithm structure (as designed in the chat)

Two conceptually separate parts:

1. **Energy minimizer** — does NOT need to respect the projector manifold.
   Only the *tangent* component of the force matters physically; the normal
   component is erased by the retraction anyway.
2. **Purification retraction** — restores `D²≈D`, `Tr D≈N_occ` after each step.
   Uses only symmetric squares:
   `q₋(X)=X²`, `q₊(X)=2X−X²`, applied as a complementary pair (TC2).

The learned constraint force Λ exists **only for f32 noise control**: it makes the
raw step approximately tangent, so purification removes an `O(α²)` defect instead
of an `O(α‖H‖)` excursion. Near convergence `F→0`, so no large cancellation cycle
runs at the fp32 floor (~1e-6 in the relevant norms).

Per iteration:

    F₀ = H − Λ
    F  = F₀ − aI − bD        (a,b from 2×2 LS projection of F₀ onto {I,D};
                              needs 4 reductions: TrD, ⟨D,D⟩, TrF₀, ⟨D,F₀⟩)
    X  = D − αF  = (1+αb)D − αF₀ + αaI
    Y  = q±(X)               (branch by predicted trace of X vs N_occ)
    D' = q∓(Y)               (optional second, complementary square)
    Λ += β·(Y−X)/α + β·(D'−Y)/α   — accumulated during the write-back, no extra
                                    storage of X needed

Fixed point: `D'=D` ⟹ `Λ=H` (chat's gauge) ⟹ `F=0`. With the `{I,D}`-projection in
the loop the converged Λ is `H−aI−bD` — differs only by commuting gauge.

Parameters (chat recommendations):
- `α ≈ (0.5–1)/spectral_width(H)`  — fixed, NOT shrunk to 0 near convergence
  (otherwise roundoff/α contaminates Λ).
- `β ≈ 0.3–0.7` — damped, because `D'−X` contains O(α²) truncation + fp32 GEMM noise.
- NPUR adaptive: 1 square far from convergence, 2 near it / when idempotency drifts;
  0 near the fp32 floor. Compile-time specialization (`PURIFY_NPUR`).
- Λ update freeze when `‖D'−X‖` reaches fp32 noise floor (per-system gate).
- Diagnostics per system per step are nearly free: step norm, correction norm,
  trace, idempotency — merge into the existing reductions.

Warm start (chat's recipe — **NOT persisting the learned Λ**):

    Λ₀ = H_prev          (previous geometry / previous SCC Hamiltonian)
    ⇒ F = H_new − H_prev = ΔH   — the small perturbation, one elementwise op.

## 3. Kernel architecture (chat spec, all implemented)

- One WG per system; thread owns RTY×RTX register tile; K-slabs staged through
  ~8–12 kB local memory (`Ls[TK×LN]`), same geometry as tuned GEMM
  (TX=22, TY=11, RTX=4, RTY=8, TK=22, WG=242).
- `D→X` transform fused into slab staging: each `D_ij` read exactly once,
  `X_ij` written to global D in place, staged `Ls` feeds `X²`.
- After `X²` completes (regs), write-back applies `q±` AND deposits
  `Λ += β(y−x)/α` in the same pass; second square re-stages Y in place and
  repeats with complementary branch.
- Per-iteration cost: 1 or 2 symmetric squares, **zero general GEMMs**,
  O(n²) elementwise + one small reduction.
- Workgroup barriers only inside one WG — hence 1 WG/system is mandatory.

## 4. What is implemented (files)

**Primary — the new code:**

- `rust_dftb/src/qmqm/gpu_purify.cl`
  - `relax_step_batched` (L678) — the fused kernel: reductions → a,b → stage X
    + X² → q± write-back + Λ deposit → optional complementary second square.
    `PURIFY_NPUR` compile-time, per-system diagnostics + done flags.
  - `relax_canon_batched` (L632) — experimental one-shot Λ canonicalization
    `Λ += aI+bD` (makes Λ≈H; see §6 — the drift experiment).
  - `comm_gate_batched` (L963) — commutator certificate; needs one general
    product `T=H·D` (NOT a symmetric square) computed beforehand.
  - Reused infrastructure: `tc2_init/extrap/step_batched` (L116–427),
    `spec_span_batched` (L427, gives α), `dmm_update_batched` (L453),
    `mcweeny_combine_batched` (L517), `square_regtile_*` symmetric-square
    machinery + `pur_reduce4` reductions.
- `rust_dftb/src/qmqm/gpu_purify.rs` — `relax_purify_batched` driver, `RelaxDiag`,
  Palser cold init, `α=η/span(H)`, chunked launch loop with done-flag freezing.
- `rust_dftb/tests/gpu_purify.rs` — `test_relax_purify_parity` (cold parity vs
  Jacobi reference + warm perturbation round), `certify_d` (energy, ‖D²−D‖,
  comm, symmetry, trace).

**Reused primitives / comparison targets:**

- `rust_dftb/src/qmqm/gpu_gemm.cl` — `gemm_sq_iter` (L372) / `gemm_sq_iter_tri`
  (L624): the optimized resident symmetric-square kernels the relax kernel is
  modeled on; `gemm_regtile` (L156): general `A·B` — relevant for Q2 (occasional
  `T_D(Λ)` projection) and for `comm_gate`'s `H·D`.
- `rust_dftb/src/qmqm/gpu_matrix_ops.cl` — `batched_gemm` (L186): another general
  product path already used for `H·D`.
- `rust_dftb/src/qmqm/gpu_tiled_jacobi.cl` — `jacobi_resident_batched` (L562),
  `tiled_jacobi_batched` (L875): the Jacobi path to beat.
- `rust_dftb/src/qmqm/gpu_block_jacobi.cl`, `gpu_eigen.cl`
  (`jacobi_cyclic_local_batched` L112) — other Jacobi variants.

**Integration targets (not yet touched):**

- `rust_dftb/src/qmqm/gpu_eigen.rs` — `EigKind` dispatch (where `EigKind::Relax`
  would plug in).
- `rust_dftb/src/qmqm/gpu_scc_plan.rs` — SCC solver plan; where a per-SCC-iteration
  relax step / `Λ₀=H_{k−1}` policy would live.
- `rust_dftb/src/qmqm/gpu_runtime.rs` — program/kernel management.
- `rust_dftb/tests/gpu_scc_bench.rs` — the end-to-end benchmark vs Jacobi.

## 5. Deviations from the chat spec (intentional, must be re-examined)

1. **Trace pinning.** Chat computes `a` by least squares (`Tr F = 0`). We instead
   pin `Tr(X)=N_occ` through the `a` coefficient — a chemical-potential channel.
   Reason: observed wrong-rank stall — a clean rank-42 projector is an exact fixed
   point of both TC2 folds; once reached, deposits ≈ 0 and it cannot recover
   (15/16 converged, sys4 stuck at Tr=42). The pin fixed cold convergence.
   NOTE: the chat kernel has the same latent defect (its `l_expand1` uses
   predicted trace but nothing forces the trace).
2. `PURIFY_NPUR` compile-time instead of runtime `two_step` — as chat §11 suggests.
3. Extra per-system diagnostics + done-freeze — as chat §12 suggests.

## 6. Measured results (16 systems, n=86, N_occ=43, η/span α, β=0.5, NPUR=2)

| run | iters | result |
|---|---|---|
| cold, Λ=0 | 48 | ✓ all 16: ΔE~1e-6, ‖D−Dref‖~1e-5, ‖D²−D‖~1e-6, comm~5e-5, Tr=43 |
| warm ΔH=0.02·R, keep Λ₁ | ~72 | ✗ clean idempotent Tr=43 projector, but comm~0.31, ΔE~1.5e-2 — biased fixed point |
| warm, Λ₀≈H₁ (canon = chat recipe) | 200 (no conv) | ✗ drifts toward eig(ΔH), comm→7.5 |
| warm, Λ=0 | ~116 | converges but comm~0.3, ‖D−Dref‖~0.13 — still biased |
| β=0 (no Λ learning) | — | ✗ cold doesn't converge — Λ is essential |
| Λ decay (λ>0) | — | ✗ residual floor, cold broken |
| β gate (learn late) | — | ✗ cold broken — Λ needed from start |
| absorb aI+bD gauge into Λ per step | 12–15 | ✗ fp32-clean (‖H−Λ‖~3e-4) but lands on excited projectors (ΔE~+43) |

## 7. Analysis of the failure — the actual problem

The deposit `(D'−X)/α` equals the **normal part of F in the current D frame**
(up to O(α²)). The tangent part of F passes through the retraction unchanged
(becomes motion). Therefore:

**The Λ update can only ever add block-diagonal (normal+gauge) content measured in
the frame where it was deposited. The tangent component of Λ is invisible to the
learning channel — it is a hidden, approximately conserved state.**

Consequences:

- The fixed-point set is degenerate: for ANY clean rank-N_occ projector `D*`,
  setting `Λ*=H−aI−bD*` makes `F∈span{I,D*}` and deposits→0 — the map freezes.
  Correctness of `D*` (i.e. `[D*,H]=0`) is NOT enforced by the fixed point; it is
  selected purely by trajectory/basin. Cold works because the trajectory is a
  clean descent from Λ=0. Warm starts can freeze at self-consistent wrong pairs.
- Equivalently: Λ's tangent acts as a phantom force — the solver effectively
  minimizes `Tr((H−Λ_tan)D)`, a different functional.
- Two ways tangent enters Λ: (a) carried in (warm Λ), (b) frame-drag — a deposit
  normal in frame k acquires tangent ~θ·deposit after D rotates by θ.
- Learning freezes when F is ≈tangent (deposits ∝ normal residual → ~0), which is
  exactly the warm `Λ=H₁` case: `F=ΔH` tangent-ish → Λ frozen → literal
  "ΔH-forever" descent → eig(ΔH). This is the drift we measured.

Useful algebra (exact for our update with the {I,D} projection):

    Λ' = (1−β)·Λ + β·(H − aI − bD_n) + β·(D'−D_n)/α

i.e. Λ is a *leaky* relaxation toward `H−gauge` with time constant ~1/|ln(1−β)|,
plus the last-step displacement. At β=1 it degenerates to `Λ=H+(D'−D)/α` —
one-step memory — and `F=−(D'−D)/α` becomes pure momentum (no gradient!). So β
tunes a memory-vs-momentum trade-off; the bias is not removed by any fixed β.

## 7.5 Verdict: algorithm flaw or implementation bug?

Honest assessment: **primarily a structural gap in the proposed algorithm's
warm-start story, not an implementation bug** — with caveats.

Evidence the implementation is correct:

- The cold path works exactly as designed (48 it, ΔE~1e-6, comm~5e-5, all 16
  systems). This exercises every kernel mechanism — staging, both squares,
  deposits, reductions — so the mechanics are validated.
- The warm failure reproduces *analytically*: deposits measure only the normal
  part of F; when `F=ΔH` is ~tangent, deposits→0, Λ freezes, and the dynamics
  descend `T_D(ΔH)` toward `eig(ΔH)`. That mechanism lives in the equations,
  not the code — and it contradicts the chat's claim that "Λ updates prevent
  the ΔH-forever stall." They prevent it only while F has a normal component.

Evidence the *algorithm spec itself* is incomplete (independent of my code):

- **Wrong-rank stall exists in the chat kernel too**: a clean rank-42 projector
  is an exact fixed point of both TC2 folds; the predicted-trace branch ordering
  does not recover rank. My `TrX=N_occ` pin is an addition the spec needed.
- **Degenerate fixed manifold**: `Λ*=H−aI−bD*` self-consistently freezes the map
  at *any* clean projector; the chat never discusses how the *correct* projector
  is selected.

Residual risk that my deviations contribute (untested isolations):

- My `a` = trace-pin vs chat's LS `a`. For clean-trace D they coincide, and cold
  works, but the interaction in warm transients was not isolated.
- Canon gives Λ≈H₁−residual, not literally Λ₀=H₁.
- β=0.5, NPUR=2 fixed, η/span α — none of these were swept.
- To isolate cleanly: run the chat's *literal* kernel (LS-`a`, no pin) in the
  warm `Λ₀=H₁` case and check whether the eig(ΔH) drift persists.

One more nuance: the chat recipe may still be adequate **inside SCC**, where
per-iteration ΔH is small and only a few relax steps run per H update — a
different regime from the standalone large-Δ warm solve tested here (see Q6).

## 8. Open questions for the next LLM discussion

Q1. **Is the degenerate fixed-point manifold the chat's design flaw or our
    misreading?** With Λ free to absorb `H−aI−bD` at any projector, does anything
    in the scheme force `D*→eig(H)`? If not, the learned-Λ freedom must be
    restricted. How?

Q2. **Λ tangent projection.** Standard augmented-Lagrangian practice would project
    the multiplier into the constraint's normal space: `Λ ← Λ − T_D(Λ)` with
    `T_D(Λ)=DΛ+ΛD−2DΛD` — 2 general GEMMs. Acceptable occasionally (per warm start
    / every K iters / triggered by comm certificate), but is there a cheaper
    square-only surrogate? Note `DΛD` is not a symmetric square; do we have/want a
    general `A·B` resident kernel?

Q3. **Warm start `Λ₀=H_prev`.** Chat claims F=ΔH is ideal; we measure Λ freezing
    and drift to eig(ΔH) because deposits≈0 once F is tangent. Is the scheme only
    valid for small ‖ΔH‖, or does it need a kick (e.g. keep a fraction of H in F)
    to keep deposits alive? Quantify the basin: for which ‖Δ‖ does `Λ₀=H_prev`
    actually converge?

Q4. **Trace control.** We added the `TrX=N_occ` pin (chemical-potential channel).
    Is there a cleaner way that doesn't perturb the LS gauge — e.g. separate μ
    update `μ += γ(TrD−N_occ)`, or NC2-style trace-selecting polynomials?
    Does the pin interact badly with Λ (the pin residual is a real force that
    gets deposited)?

Q5. **β and the momentum limit.** β→1 makes F pure momentum. Is there an optimal
    β? Should β be per-system adaptive (e.g. scaled by ‖step‖)?

Q6. **SCC context changes everything?** In the real use case H changes by small
    ΔH each SCC iteration and we re-solve D each time. If per-iteration ΔH is
    small, maybe Λ₀=H_{k−1} bias is negligible *for the SCC loop* even if large-Δ
    warm starts fail? Or should SCC instead do ONE relax step per iteration
    (solver folded into SCC, H and D converging together — then Λ is always
    fresh and D never leaves the manifold)? This might be the actual right
    integration: not "solve D given H" but "joint (D,Λ,H) iteration".

Q7. **Alternative: occupied-subspace C (chat §12).** `C∈R^{n×m}, C^TC=I`,
    `R=HC−C(CᵀHC)` ≈ 2n³ flops (one GEMM-equivalent) — half the flops of
    density-matrix descent, no Λ subtleties, orthonormalize periodically.
    Should we prototype this in parallel? Its SCC warm start is trivial
    (C₀ from previous D).

Q8. **Certification.** Whatever the solver, `comm_gate` (needs one `H·D` general
    product) or energy-parity vs an occasional Jacobi solve can certify the
    fixed point; fail → Jacobi fallback. Design the dispatch: when is the
    certificate run — every solve, on stall, or adaptively?

## 9. Immediate next steps (agreed with user)

1. Decide direction on Q1/Q2 (Λ tangent control) and Q6 (SCC-fused iteration
   vs standalone solver) before more kernel work.
2. Kernel/driver cleanup is blocked on that decision; Jacobi benchmark and
   `EigKind::Relax` wiring come after a correct warm-start policy exists.

---
---

# PART II — Canonical tangent-force experiments (exact gradient, no learned Λ)

Working notes from the follow-up campaign: drop learned Λ entirely, compute the
*exact* tangent force of `E=Tr(DH)` and let purification be only a retraction.
This is the architecture the chat's Λ was trying to approximate.

## II.1 What was implemented this round

### GPU (new code, uncommitted)

- `rust_dftb/src/qmqm/gpu_matrix_ops.cl`
  - `lnv_trace_batched` — elementwise trace/mu bookkeeping.
  - `lnv_finish_batched` — the canonical **LNV residual-form gradient**
    `G = (B+Bᵀ−2C) + 3[(A−B)+(A−B)ᵀ]` with `S=L²`, `A=LF`, `B=SF`, `C=AL`
    (≡ the tangent force `[L,[L,F]]` at idempotency), plus velocity/momentum,
    O'Donoghue restart, μ-projection gating, δμ cap, step cap, per-system
    diagnostics. (≈220 lines.)
  - `gemm_nn_batched` — occasional-use general `A·B` (for `H·D` certificates).
- `rust_dftb/src/qmqm/gpu_matrix.rs` — `lnv_solve_batched` driver:
  1 symmetric square + 3 general GEMMs per iteration, BB scalar step
  (disabled under momentum), chunked host checks, env knobs
  (`RUST_DFTB_LNV_{ETA,KMU,MOM,CHUNK,DEBUG}`).
- `rust_dftb/tests/gpu_purify.rs` — `test_lnv_gradient_parity` (GPU kernel vs
  CPU-f64 residual-form reference at a non-stationary projector AND a generic
  interior L: rel err 4.7e-6 / 6.3e-7 — the kernel is **verified correct**) and
  `test_lnv_solve_parity` (warm Δ=0.02 / Δ=0.5 from exact projectors;
  energy/projector/idempotency/symmetry/comm certificate checks).

### CPU/f64 reference (`/tmp/lnv_check/`, scratch)

Independent reimplementation used to separate algorithm from GPU bugs:
- bare LNV-SD: reproduced the GPU's ~2000-iter convergence exactly → slowness
  was geometric, not a code bug.
- heavy-ball momentum: ~10–20× speedup at Δ=0.02 (3882→188 iters) but NaN
  divergence at Δ=0.5 (μ-projection × momentum resonance).
- **tangent descent + retraction** (the candidate production scheme):

      G = (HD + DH − 2·D·H·D)/span          exact tangent force, 3 products
      D ← D − αG                            (heavy-ball velocity optional)
      D ← 3D² − 2D³                         McWeeny cubic retraction, 2 squares

  No auxiliary variable, no μ channel — trace/idempotency enforced exactly by
  the retraction each iteration.

## II.2 The product-order bug (cautionary)

My first tangent-descent version wrote `HD @ D` (= **H·D²**) where the formula
needs `D @ HD` (= **D·H·D**). At idempotency this degenerates to `DH−HD` — the
*antisymmetric* commutator, not a symmetric gradient. Consequences observed:
steps produce a non-symmetric `D−αG`; `eigvalsh` silently reads one triangle
and reports impossible eigenvalues; every diagnostic (eigenvalue excursions,
"retraction does nothing") was an artifact of a single transposed product.
**Lesson for review: `Y = K·T` with `T = H·K` in `dmm_update_batched` is the
correct `KHK` ordering — check against it.**

## II.3 Measured results (n=86, batch=8, N_occ=43, warm start = exact
projector of unperturbed H; CPU f64)

Convergence criterion for the iteration counts: **physical** —
`‖D−D_ref‖_F < 0.05` (5% projector distance) AND `|E−E_ref| < 1e-3` Ha.

| scheme | Δ=0.02 | Δ=0.5 |
|---|---|---|
| stateless `F=H−aI−bD` (no Λ, TC2 retract) | **no fixed point** — step/corr freeze at 0.45/0.43, `‖D²−D‖≈0.21` | same |
| LNV residual SD, bare α | ~1900–3900 iters | diverges |
| LNV + heavy-ball (α=0.6, β=0.9) + safeguards | ~400–7200 iters (8/8 at tol 1e-6 on GPU) | 7/8, one system NaN |
| **tangent descent + cubic retraction, α=2·(1/span), β=0.9** | **7/8; median ~30 iters** (22–300, sys7 degenerate fails) | **8/8; median ~180 iters** (104–386) |
| same + DIIS every 4 iters | slow systems 187→92 | ~20% faster, no new convergences |

GPU parity (LNV path, tolerance 1e-6): warm Δ=0.02 → **8/8 pass** (~7250 iters,
max comm 9.9e-5); warm Δ=0.5 → 7/8 (sys3 hits the μ↔momentum resonance late,
~it=970; trajectory shows slow exponential spiral in `‖G‖` while `r²` hovers
at the gate boundary).

## II.4 Accuracy — are we dancing on the numerical floor?

**No.** The reported iteration counts use criteria ~2 orders of magnitude
above the f32 floor:

- Measured f32 floors (from the converged LNV runs): `‖D−D_ref‖ ~ 5e-5…3e-3`,
  `comm ~ 1e-4·span ~ 5e-3`, `ΔE ~ 1e-5…1e-4`.
- The `dp<0.05 / ΔE<1e-3` thresholds are loose-but-physical; reaching them is
  limited by *geometry* (gradient descent rate `1−1/κ` per iter on a linear
  functional), not by noise.
- The genuinely hard cases are **physics-limited, not floor-limited**: sys7's
  Fermi gap/span = 2.7e-4 — a near-degenerate occ/virt pair whose mixing
  gradient is `~gap·θ`; below `θ≈0.05` the force is smaller than what loose
  convergence criteria need, so it stalls at a small-but-nonzero projector
  error. Tightening the LNV tolerance to 1e-6 let it fully resolve (proved the
  residual was a real direction, not noise) — at ~2× the iteration cost.

## II.5 Why bare descent is slow — didactic summary

`E=Tr(DH)` is **linear** in D on the Grassmannian of rank-43 projectors. The
curvature along an occ–virt rotation mode `(i,a)` is the orbital energy
difference `ε_a−ε_i`. Dense random test matrices at exactly half filling
guarantee several near-degenerate pairs at the Fermi edge → condition number
`κ = span/gap ~ 300–4000`. Every first-order method pays `~κ` iterations on
that mode; momentum buys ~10×, DIIS a bit more, nothing removes it. **Newton
would fix it — but the Newton solve for `Tr(DH)` is equivalent to
diagonalizing H**, which is literally what Jacobi does pair-by-pair. This is
the deep reason Jacobi is competitive.

## II.6 The μ-channel resonance (LNV-specific, now avoided)

LNV enforces `Tr D = N_occ` softly via a chemical potential
`δμ = span·⟨R,G⟩/(6⟨R,R⟩)` with `R=L−L²`. Two singular regimes:

- `R→0` (warm start at a projector): 0/0 — amplifies noise into huge μ kicks
  (measured μ → −2.4e9 in f64). Fixed by gating on `⟨R,R⟩`.
- `r²` large (far off-manifold): the linear trace model is invalid; μ kicks of
  ~10×span couple with momentum into a slow exponential resonance
  (measured: `‖G‖` spirals 0.06→1e8 over ~500 iters in sys3).

The clean fix is not more gating — it is **eliminating the channel**:
retraction-based trace enforcement (cubic fold) is exact and stateless. This
is what the tangent-descent scheme does.

## II.7 ⚠ The pre-existing code — the same architecture already existed

Reviewing `gpu_purify.cl`/`gpu_scc_plan.rs` after the fact: the exact
tangent-descent + retraction scheme is **already implemented and verified in
production context**:

- `gpu_purify.cl:453 dmm_update_batched` — `K ← sym(K) − s·(T+Tᵀ−2Y)`,
  `T=H′K`, `Y=K·T` (= KH′K — correct ordering), with **trust region**
  `s = min(η/Δε, cap/‖G‖_F)`. Defaults: `η=1.0`, `cap=10` (η=8 diverges at
  SCC-scale ΔH — measured comm 5e-7→0.72→NaN).
- `gpu_purify.cl:517 mcweeny_combine_batched` — `K ← 3K²−2K³` (the cubic
  retraction, pair-owner symmetrization).
- `gpu_scc_plan.rs:1836 purify_solve_enq` — the SCC warm path:
  seed = last converged K in place → **`dmm_steps=16` DMM steps** (2 GEMMs
  each) + McWeeny retract every 2 steps (2 squares) + `tc2_tail=4` TC2 steps →
  `R_H` commutator certificate → `D = XKXᵀ` → Mulliken. **Verified on GC/H2O:
  failed=0, E0 exact.**
- `sparse_system.rs:4075 dmm_descend` — the sparse generalized-overlap
  ancestor (`G = X+Xᵀ−2Y`, `X=(ZH)K`, `Y=(KS)X`, 3 SpGEMMs + McWeeny every
  N steps, η=8/Δε for tiny force-eval perturbations).

My GPU LNV work re-derived this from the chat doc without inventorying it —
the kernel machinery (and the hard-won step-size/trust-region tuning) already
exists.

## II.8 The paradox: warm minimization slower than cold purification

Measured: cold TC2 converges in ~20–40 **single-square** iterations and beats
Jacobi; warm tangent-descent needs ~30–300 iterations × ~5 products. Why is
the "easy" warm problem slower than the "hard" cold one?

**Because they are different problems:**

- **Cold purification never has to rotate the occupied subspace.** The Palser
  init `D₀ = (λ_max·I − H)/span` is an *affine function of H* → it has H's
  eigenvectors **exactly, from iteration 0**. The only remaining work is
  eigenvalue expulsion (push each λ to 0 or 1) — a per-mode *scalar* dynamics
  that purification does superlinearly. TC2 = pure eigenvalue work, ~1
  product/iter.
- **Warm minimization must do exactly the thing purification cannot do:**
  rotate eigenvectors. Polynomials in D preserve the eigenbasis, so the
  rotation can only come from the force `[D,[D,H]]` — a first-order,
  κ-limited, ~5-product/iter descent on a *linear* functional.

So the warm-start premise was framed wrongly for standalone re-solves: for
any fixed H, Palser+TC2 gets the eigenbasis **for free** — warm minimization
can never beat that unless it converges in <~8 iterations. The honest
remaining value of the descent machinery is precisely where it is already
used:

- **SCC inner loop** (`PurifyScc`): H changes by a small charge-dependent ΔH
  each iteration; ~16 DMM steps per iteration *track* the solution — no full
  solve is ever needed, convergence is joint with the charges. Here a full
  cold re-solve per iteration would be waste.
- **Force-response perturbations** (sparse `dmm_descend`, η=8): tiny ΔH,
  few steps.
- Geometry changes reset `seeded=false` → cold Palser re-init anyway (K lives
  in the old X basis) — production already knows warm tracking is only valid
  within a fixed basis.

**Corollary:** the standalone benchmark "re-converge D after a frozen ΔH=0.5"
measures a scenario production never needs. The meaningful comparison is
*per-SCC-iteration cost* of the DMM+TC2 mix vs Jacobi — which is where
`gpu_scc_bench.rs` should answer.

## II.9 Ideas for improvement (for the smarter-LLM discussion)

1. **Trust-region + bigger η** (already in `dmm_update_batched`):
   `s=min(η/Δε, cap/‖G‖)` allows normalized full-length steps far from the
   manifold. My fixed-α experiments topped out at α≈2–3/span before overshoot;
   capped normalized steps may go further safely.
2. **Retract every k-th step, not every step** (sparse recipe: `n_ret≈2`):
   the cubic fold is the most expensive part (2 products); descending 2–3
   steps between retractions roughly halves the cost/iter — needs re-testing
   for stability at α≈2–3.
3. **DIIS on the commutator residual** helped ~30–50% on flat modes in the CPU
   test (187→92 iters). Worth keeping as an option; extrapolate D, then
   retract once.
4. **Newton/PT1 finish for warm starts**: if the previous eigensolve's
   eigenvectors (or D0's) are available, the first-order density response
   `δD_ia = (v_iᵀΔH v_a)/(ε_i−ε_a)` is an O(1)-step solve for small ΔH.
   Crosses the "don't diagonalize" line, but D0's eigenproblem is trivial
   (clustered {0,1} spectrum) — or reuse cached eigvecs from the previous
   solve. This is the principled fix for the flat modes: it *divides by the
   gap* instead of crawling at gap-rate.
5. **Hybrid certificate-driven fallback**: descent for ~K steps → `R_H`
   comm certificate → if not converged, Jacobi finish. The certificate kernel
   (`comm_gate_batched`/`k_comm`, one `H·D` product) already exists.
6. **Occupied-subspace formulation** (Q7, chat §12): `C∈R^{n×m}` minimizes
   the same energy with `R = HC − C(CᵀHC)` at ~2n³ flops/iter — half the cost
   per step, no idempotency channel at all; orthonormalize periodically.
7. **Accept the SCC-fused frame** (Q6): production warm starts live inside a
   converging (D,H) pair; standalone re-solve-to-tolerance is the wrong
   benchmark. Optimize the *per-SCC-iteration* kernel count.

## II.10 Open questions for the LLM discussion

- Q9. Is `dp<0.05 / ΔE<1e-3` the right production convergence criterion, or
  should the certificate be `comm` + energy only (projector distance is
  unobservable in production — what do we gate on)?
- Q10. For the SCC-fused path, is the per-iteration budget (~16 DMM + 4 TC2)
  optimal vs fewer DMM steps with more TC2 tail, or vs the occupied-subspace
  C formulation?
- Q11. Does the flat-mode limit actually matter for real DFTB systems? Our
  benchmark uses *dense random* matrices at exact half-filling — worst case
  by construction. Physical systems have gapped valence/CBO structure; the
  κ-limit may be far kinder. The H2O/GC verification suggests yes in
  practice — quantify the gap distribution on real systems.
- Q12. The degenerate Fermi-edge mode (sys7-type) is unresolvable by any
  first-order method — but it is also *physically irrelevant* below a
  threshold (rotation within a degenerate pair costs `gap·θ²` energy).
  Should convergence certificates deliberately ignore below-gap modes?
- Q13. Jacobi's ~50 sweeps are ~50 n³-equivalents; tangent descent is ~5
  products/iter — break-even is ~10 iters. Only SCC-amortized descent or a
  PT1-type step beats that. Is there any formulation of "solve D(H) once"
  that survives this accounting? (Suspected answer: no — Palser+TC2 owns that
  niche at ~30 products.)
- Q14. Given `dmm_update_batched` already exists, the remaining engineering
  is: expose a standalone batched `warm_dmm_solve` for testing (same kernels,
  host loop like `lnv_solve_batched`), benchmark vs Jacobi inside
  `gpu_scc_bench.rs`, and wire `EigKind` dispatch. Agree?

## II.9b The missing point — what the warm estimate is actually worth

User objection (correct): reusing a nearly-converged D for a nearly-identical
system (atom moved ~0.1 Å) **must** beat cold restart. The failure is not in
the physics of warm starting — it is in **which object we warm-start**.

**The density matrix alone does not contain the exploitable information.**
D₀ encodes only the occupied subspace. The exact first-order correction is

    δP_ia = ⟨v_i|ΔH|v_a⟩ / (ε_i − ε_a)     i∈occ, a∈virt

— i.e. a rotation whose per-mode angle requires **dividing by the energy
gap**. The gap lives in `(V, ε)`, the eigendecomposition — which the
density-matrix representation throws away. Every D-only scheme must therefore
replace division by iteration: descent moves mode `(i,a)` at rate
`gap_ia` → `1−1/κ` per step → the κ-limited 30–300 iters we measured. We
discarded the eigenvectors and then paid κ iterations to rediscover the
denominators. Purification cannot substitute either: polynomials in D
preserve the eigenbasis — they can never rotate.

**Consequences — the honest warm-start hierarchy:**

1. **Warm-started Jacobi on `W = V₀ᵀ·H_new·V₀`** — the standard eigensolver
   warm start, already supported by existing machinery. `W` is nearly
   diagonal (off-diagonals ~‖ΔH‖); classical Jacobi contracts quadratically
   near diagonal → **~3–5 sweeps instead of ~50**. Cost ≈ 2 GEMMs (form W) +
   ~4 sweeps + 2 GEMMs (rotate eigvecs) ≈ **~10 products vs ~30 cold TC2 —
   and it produces the new (V,ε) for the next step.** The clean answer:
   Jacobi cold, warm-Jacobi for updates; purification optional.
2. **PT1/Newton step** — `W = V₀ᵀH_newV₀` (2 GEMMs), `M_ia=W_ia/(ε_i−ε_a)`
   on occ–virt blocks (elementwise, denominators known), `δP=V₀MV₀ᵀ`
   (2 GEMMs), one purify fold + comm certificate ≈ **~6 products**,
   first-order exact; repeat for O(ΔH²) if needed. Requires cached (V,ε).
3. **SCC-fused DMM tracking** (existing `PurifyScc`) — correct for its niche:
   small per-iteration ΔH, no full solve ever needed, convergence joint with
   charges. It is NOT a general warm solve.
4. **Basis transport across geometry steps (missing piece)** — `set_geometry`
   sets `seeded=false` because K lives in the old X basis. But the AO density
   `D_ao = X·K·Xᵀ` is basis-free: `K_new = B·K·Bᵀ` with `B = X_new⁻¹X_old`
   is a 2-GEMM transport preserving the physical density. Required to make
   *any* warm scheme (Jacobi or DMM) usable across geometry changes — the
   eigenvectors transport identically: `V_new ≈ B·V_old` (then re-orthogonal
   or just let Jacobi absorb the O(ΔS) error).

**Revised verdict on §II.8:** the conclusion "warm minimization is only for
SCC tracking" was correct *for D-only methods* — but the right conclusion is
that the D-only formulation is the wrong tool for warm starts entirely.
Cold: Palser+TC2 (~30 products) or Jacobi (~50 sweeps). Warm: keep (V,ε) and
do warm Jacobi (~10 products) or PT1 (~6 products). The chat's Λ/descent
machinery tried to make the D-only representation work; it cannot beat the
eigenbasis representation for rotation because the gap denominators are not
accessible from D alone. **To verify on GPU: run `tiled_jacobi`/`jacobi_resident`
on `W = V₀ᵀH_newV₀` vs on `H_new` and count sweeps — predicted ~5× fewer.**

## II.9c REVISION — under the actual constraint: NO eigensolver, ever

The §II.9b conclusion (warm-start the eigensolver) violates the design intent:
the whole point is a DM-only path that avoids Jacobi — purification is the
cheap Pauli/idempotency enforcer *instead of* diagonalization. Re-analyzed
under that constraint, the failure decomposes differently and the DM-only
warm start is NOT dead:

### Why the measured warm descent was slow — correct decomposition

1. **A stale converged D carries no rotation information.** `D₀=P(H₀)` sits at
   rest on the manifold: it encodes the old subspace but not the direction to
   the new one. All rotation must then be generated by the force — first-order,
   rate `1−1/κ` per step → the 30–300 iters. The seed was the problem, not the
   concept.
2. **Rotation CAN be injected into the seed, matrix-only:**
   - *Trajectory extrapolation* `S = 2K₁−K₂` (XL-BOMD / Niklasson): for a
     small subspace motion `K₂=e^RK₁e^{−R}≈K₁+[R,K₁]`, `S≈K₁−[R,K₁]`. Since
     `[R,K₁]` is purely occ–virt, perturbation theory gives the eigenvectors
     of `S` rotated by exactly `R` to first order — **the extrapolated seed
     carries the motion**. Purification then only snaps eigenvalues
     (~5–10 folds, not 30). This is exactly `tc2_extrap_batched` (L168) —
     which measured *drift* (comm 0.24) — a debugging problem (fold-branch
     misfires on the extrapolated spectrum; the Gershgorin renorm was added
     for range, not for branch selection), not a conceptual dead end.
   - *Basis transport across geometry steps* `K_new = B·K·Bᵀ`,
     `B = X_new⁻¹X_old`: preserves the AO density exactly; residual error is
     only the ΔH-rotation. 2 GEMMs; currently `set_geometry` just discards K.
3. **The mean occ–virt gap is trace-computable** — a quasi-Newton step
   without eigenvectors: `Δ̄ = Tr(HQ)/nvirt − Tr(HP)/nocc =
   (TrH − E)/(n−nocc) − E/nocc` (two traces, ~free). If gaps cluster around
   Δ̄, the step `η ≈ span/Δ̄ ~ 300` approximates Newton. Only η≤4 was ever
   tested (η=8 diverged *without* trust region); `dmm_update_batched`'s cap
   `s=min(η/Δε, cap/‖G‖)` makes the large-η regime safe and unexplored.
4. **Honest residual limit:** if occ–virt gaps are *spread* (dense random
   half-filled benchmark = worst case), no scalar step fixes all modes —
   convergence ~1/δ for relative gap spread δ. For real DFTB spectra the
   coupling may concentrate on a few frontier modes → effective κ much
   smaller. **The benchmark's pessimism is unverified.**

### Questions for the clever LLM (sharp, testable)

- **Q-A (seed carries rotation):** Is the correct warm architecture
  `extrapolate (2K₁−K₂ or higher-order XL-BOMD) → few DMM steps → few TC2
  folds` rather than `descend from stale D to convergence`? The DMM steps
  should only correct extrapolation error, not generate the whole rotation.
  Confirm the first-order claim: eigenvectors of `2K₁−K₂` are rotated by the
  K₁→K₂ subspace motion — and identify why the measured trajectory drifted
  (branch selection on extrapolated trace? renorm destroying the occ/virt
  ordering? need a comm certificate gate before accepting the seed?).
- **Q-B (quasi-Newton step):** Is `η = span/Δ̄` with trace-computed
  `Δ̄=(TrH−E)/(n−nocc) − E/nocc` a defensible mean-gap Newton step? Under
  `s=min(η/Δε, cap/‖G‖)` + McWeeny retraction every step, is it stable, and
  does it converge in ~5–10 iters on clustered-gap spectra? What is the right
  per-mode generalization — a diagonal preconditioner `1/(H_aa−H_ii)` (à la
  Davidson) applied to `G` in the AO basis?
- **Q-C (geometry-step transport):** `K_new=B·K·Bᵀ` with `B=X_new⁻¹X_old`
  preserves the physical density. Is the transported K's error purely the
  ΔH-rotation (making it the right warm seed), and can the gap estimates be
  carried too (Ritz values on transported vectors, or Δ̄ refreshed by the two
  traces)?
- **Q-D (benchmark realism):** What is the actual occ–virt gap distribution
  for real DFTB systems under SCC/geometry perturbations? Do the relevant
  modes cluster (quasi-Newton viable) or spread (κ-limit real)? Decide the
  benchmark on physical spectra, not dense random half-filled matrices.
- **Q-E (accounting):** Break-even vs Jacobi (~50 sweeps ≈ 50 n³-equiv)
  needs <~10 descent+retract iters; vs cold TC2 (~30 products) needs <~6.
  Which candidate reaches that for realistic ‖ΔH‖ — and is the fair metric
  per-solve or per-SCC-iteration (joint convergence already amortizes)?
- **Q-F (is anything fundamentally better?):** The occ–virt Hessian inverse
  is `diag(1/(ε_a−ε_i))`. Is there ANY matrix-product-only application of it
  (rational approximant / Neumann in `I−ad_H/Δ̄`, converging in ~1/δ for
  relative gap spread δ), or is mean-gap quasi-Newton + extrapolation the
  practical ceiling for DM-only?

---
---

# PART III — Corrections from the GPT-5.6 review (chat doc §6405+)

The external review found the benchmark conclusions over-generalized and two
concrete bugs in the extrapolation experiment. Corrections incorporated below.

## III.1 The benchmark was adversarial, not representative — own it

The decisive warm parameter is NOT `Δ` but

    η_warm = ‖Q₀·ΔH·P₀‖ / gap      (first-order rotation amplitude)

With `gap/span ≈ 2.7e-4` (dense random, half-filled — no gap mechanism by
construction), `Δ=0.02` produced an **order-one subspace rotation** — not a
warm start at all. The 30–300-iter measurements are valid *for that regime*
but were over-generalized into "warm minimization is hopeless." Corrected
conclusion:

> Unpreconditioned D-only descent is slow on nearly-gapless random matrices.
> Its performance in the intended regime — gapped systems, genuinely nearby
> previous density — was never measured.

**Benchmark redesign (agreed):**
- (A) real production trajectories: consecutive SCC iterations, geometry
  steps 0.01/0.05/0.1 Å, frozen-deflection sequences — measure `gap/span`,
  `‖P₁−P₀‖`, `‖[H₁,P₀]‖`, `‖Q₀ΔH P₀‖/g`, corrections needed.
- (B) controlled synthetic GAPPED systems: `H=U·diag(ε)·Uᵀ` with an explicit
  gap (`g/span = 0.3 … 0.02`) and perturbations generated as *controlled
  subspace rotations* `U₁=e^A U₀` (occ–virt block only) at prescribed
  `‖P₁−P₀‖ ∈ {1e-3, 1e-2, 5e-2, 0.1}` — exact answer known, warm distance
  controlled.
- (C) keep random half-filled ONLY as adversarial stress test
  (NaN/certificate/robustness), never for performance judgments.

## III.2 Two concrete bugs explain the extrapolation failure

`tc2_extrap_batched` (S=2P₁−P₀) was dismissed after "measured drift" — the
drift was caused by:

1. **Gershgorin renorm destroys the seed.** Gershgorin bounds of a dense
   rank-43 projector are ~[−3.4, 4.4] for an exact {0,1} spectrum — the
   `lmin<−0.02 || lmax>1.02` test ALWAYS triggers, and the affine rescale
   maps eigenvalues 0→~0.44, 1→~0.56. Eigenvectors survive but the seed's
   near-idempotency — its whole value — is destroyed; purification must redo
   the full spectral separation. **Fix: remove the renorm entirely.**
2. **TC2 is the wrong retraction for an extrapolated seed.** The folds have
   derivative 2 at the endpoints (`x²` at 1, `2x−x²` at 0) — whichever branch
   fires repairs one side and amplifies the other side's O(h²) overshoot.
   **McWeeny `R(x)=3x²−2x³` has R′(0)=R′(1)=0**: kills the radial
   extrapolation error quadratically while `DR[tangent]=identity` preserves
   the predicted rotation. **Fix: retract with McWeeny, not TC2.**

## III.3 The gap information IS available — Sylvester/Newton, matrix-only

My §II.9c claim "D lacks the denominators" was overstated: the pair
`(P, H₀)` with `[P,H₀]=0` gives `H_oo=PH₀P`, `H_vv=QH₀Q` via products, and
the Newton/Sylvester operator

    A(X) = [P,[X,H₀]]   ⇒   A(X)_ia = (ε_a−ε_i)·X_ia   on tangent X

is the missing gap-Hessian. Per matvec: `M=H₀X` (1 GEMM), `C=Mᵀ−M` (free),
`N=PC` (1 GEMM), `A=N+Nᵀ` — **2 GEMMs per matvec**, CG-solvable in ~√κ
iterations. RHS: `−[P,[P,ΔH]]`. For realistic gapped κ_eff~3–10 this is a
2–5-matvec solve ≈ 4–10 products total. (DMPT literature: recursive
purification and Sylvester formulations both exist, no eigenvectors needed.)

## III.4 SCC-DIIS projector extrapolation — nearly free

`H(q)=H₀+Bq` is affine in charges; the existing charge-DIIS coefficients
`cᵢ` apply directly: `S = ΣcᵢPᵢ` — O(n²) combine + 1 McWeeny + certificate.
Higher order than `2P₁−P₀`, uses information already computed. **Highest
priority experiment for the SCC regime.**

## III.5 CheFSI — the strongest new production candidate

Carry the occupied subspace `C∈R^{n×m}` (`P=CCᵀ`), not P:

    C' ~ p_k(H_new)·C_old  (degree-k polynomial filter; HC ≈ ½ GEMM)
    orthonormalize(C')     (small QR/CholeskyQR)
    Mulliken directly from C/SC  — P never needed

Degree-4 filter ≈ 1 full-GEMM equivalent + cheap orthonormalization.
Established method (CheFSI) for exactly this: warm subspace refinement
without eigensolves. Extract initial C without diagonalizing:
`Y=PΩ` (fixed random Ω), `C=orth(Y)`.

## III.6 Geometry-step transport — corrected

My `B=X_new⁻¹X_old` preserves the AO coefficient representation, not the
physical orbitals (AO basis functions move). Correct transport includes the
old–new cross overlap `S_cross=⟨φ_new|φ_old⟩`:
`B = X_newᵀ·S_cross·X_old`, `K_guess = B·K_old·Bᵀ`. (Nonorthogonal DMPT
literature exists for perturbation-dependent bases.)

## III.7 Revised architecture

    cold:      H → Palser + TC2 (~30 squares)                       [unchanged]
    warm/SCC:  DIIS-coeff extrapolation ΣcᵢPᵢ or γ-scaled 2P₁−P₂
               → McWeeny retract (NOT TC2, NO Gershgorin renorm)
               → comm certificate → 0–2 DMM corrections             [~5 products]
    warm/1st:  Sylvester-CG Newton response (2 GEMMs/matvec, ~√κ)
    candidate: CheFSI subspace filtering — carry C, skip P entirely

The reframing that resolves the paradox: a converged density is not an
**initial point for another optimization** — it is a point on a **smooth
trajectory of occupied subspaces**. Predict the motion; correct the residual.

## III.8 Honest self-assessment

Fair criticism: benchmark choice was adversarial and conclusions were
over-generalized; the extrapolation bugs (renorm + TC2 retraction) were
cited as evidence against extrapolation without auditing them; the Sylvester
route answered my own Q-F prematurely. Not sabotage — but the test did stack
the deck. Measurements themselves (LNV verification, product-order bug fix,
cold-TC2 eigenbasis analysis, pre-existing DMM inventory) stand and were
independently confirmed.

## III.9 File map update

- Pre-existing production machinery (do not duplicate):
  `gpu_purify.cl` `dmm_update_batched` (L453), `mcweeny_combine_batched`
  (L517), `spec_span_batched` (L427), `tc2_*` (L116–406), `comm_gate_batched`
  (L963); `gpu_scc_plan.rs` `purify_solve_enq` (L1836), `PurifyScc` struct
  (L163–210), DMM knobs (L1396–1410); `sparse_system.rs` `dmm_descend`
  (L4075).
- New this round (uncommitted, GPU): `gpu_matrix_ops.cl` LNV kernels +
  `gemm_nn_batched`; `gpu_matrix.rs::lnv_solve_batched`; tests in
  `tests/gpu_purify.rs`. The LNV path is now a *reference* implementation —
  its value is the verified gradient (parity vs CPU f64) and the μ-channel
  analysis, not production use.

---

# Part IV — GPU implementation of the extrapolation warm start

## IV.1 What was implemented

**`tc2_extrap_batched` rewritten** (`gpu_purify.cl`): the Gershgorin
renorm was *removed* — for a dense projector the bounds are ~[−4,5] even
for an exact {0,1} spectrum, so the old rescale fired on every seed and
mapped λ→0.44/0.56, destroying the near-idempotency that made the seed
valuable (this — plus TC2's endpoint-amplifying folds — was the entire
measured "extrapolation drift"). The kernel is now the pure predictor
    `Out = K1 + γ·(K1 − K2)`
with γ as a kernel argument (γ=1 is XL-BOMD `2K1−K2`).

**`warm_extrap_solve` driver** (`gpu_purify.rs`): the predictor →
retract → certify → correct pipeline, reusing only existing kernels —
    spec_span_batched            (Δε for the DMM trust region)
    tc2_extrap_batched           (S = K1 + γ·(K1−K2))
    gemm_nn_batched ×2           (S², S³ → mcweeny_combine_batched)
    gemm_nn_batched + comm_gate  (certificate T=H·D, rh per system)
    [repeat ≤ max_corr]:  gemm(D·T) → dmm_update → retract → certify
One program build, all buffers allocated once, one host sync per
certificate. Product accounting: retract+cert = 3, each DMM round = 4
(the cert's T=H·D is reused as the DMM force input — no extra product).
`converged=false` is an explicit signal — the caller must fall back to
`purify_tc2_batched`; nothing is silently accepted.

**Adaptive-η trust region** (added after real-molecule testing — Part
VI.2): each DMM round is checkpointed and kept only if max rh decreased;
a rejected round restores D and halves η, an accepted one doubles η
(cap 8). Fixed η is unworkable — synthetic gapped systems want η≈4 while
real DFTB spectra diverge at η=4 and converge at η≈1–2.

## IV.2 Controlled gapped benchmark (`test_warm_extrap_gapped`)

n=48, nocc=16, batch=6. `H = U·diag(ε)·Uᵀ`, ε∈[−1,1] with a real
HOMO–LUMO gap (occ∈[−1,−0.15], virt∈[0.15,1], gap/span≈0.15 — a
realistic molecular regime). Trajectory = fixed-angle rotation θ/step
in a random 2-plane, so `P_k = R(kθ) P R(kθ)ᵀ` and the target
`P* = R(2θ) P R(2θ)ᵀ` is exact. θ spans 0.03–0.20 rad — realistic
SCC/geometry subspace motion.

### Results (adaptive η, 4 accepted corrections, 19 products)

| θ (rad/step) | ΔE | ‖D−Dref‖ | ‖D²−D‖ | comm | rh |
|---|---|---|---|---|---|
| 0.03 | 1.5e-7 | 6.6e-5 | 6.2e-7 | 4.9e-5 | 9.7e-6 |
| 0.05 | 4.1e-8 | 2.6e-4 | 6.4e-7 | 2.1e-4 | 4.1e-5 |
| 0.08 | 1.0e-7 | 3.7e-4 | 7.2e-7 | 3.8e-4 | 7.5e-5 |
| 0.12 | 1.0e-6 | 1.2e-3 | 9.7e-7 | 1.1e-3 | 2.1e-4 |
| 0.16 | 6.8e-6 | 5.1e-3 | 9.4e-7 | 3.1e-3 | 5.9e-4 |
| 0.20 | 7.5e-6 | 4.3e-3 | 9.9e-7 | 3.5e-3 | 6.5e-4 |

**6/6 certified**, `converged=true` at comm_tol=1e-3, **19 products vs
28 for cold TC2** — and the cost is dominated by the worst system
(no per-system freeze in `dmm_update_batched`; θ≤0.08 systems passed
the gate after ~0–2 corrections).

η scan on this benchmark (max_rh over 4 fixed corrections, before the
adaptive scheme): η=1 → 5.4e-3 (×0.86/round, fails gate); η=4 → 7.2e-4
(×0.50/round, passes); η=8 → 1.7e-3 (stalls). Real systems behave
differently — see VI.2.

### The physics, confirmed on GPU

The bare retract already delivers production-relevant accuracy: after
`extrap + 1 McWeeny` (3 products) max rh = 1.1e-2 ↔ dp ≈ 0.03 —
inside the physical gate for θ≤0.2. The DMM corrections then act as a
*polish*, converging ×0.5/round because the seed is already
near-manifold (a completely different regime from raw descent:
descent-from-rest must *generate* the rotation; here the extrapolation
already carried it — measured earlier on CPU: `S=2P1−P2` removes ~90%
of ‖P−P*‖ at θ=0.05).

## IV.3 Status after Part VI measurements

1. SCC integration — pending; measured win is only ~25% on the late-SCC
   tail (see VI.1) — the integration must route early/mid iterations to
   the existing path via the certificate.
2. Per-system freeze — pending (batch-uniform correction rounds waste
   products on converged systems).
3. Basis transport — implemented host-side in the bench
   (`K=L_newᵀ(D/2)L_new`); measured NOT to beat cold TC2 at δ≥0.01 Å
   (VI.2).
4. Real-trajectory benchmark — done: `tests/gpu_warm_bench.rs`
   (H2O/formic/GC), results in Part VI.
5. DIIS-coeff predictor — pending; motivated by the measured γ=1
   overshoot on decelerating SCC sequences (VI.1 finding 2).
6. `gemm_nn_batched` in this driver is the naive row·col GEMM — fine at
   3–19 calls/solve but worth a regtile variant if correction counts grow.

---

# Part V — Real-molecule benchmark plan (step by step)

Goal: replace the synthetic gapped test with real DFTB matrices. Converge a
molecule, capture the actual `H′/K` trajectory, then measure — for a real
SCC iteration and a real 0.1 Å geometry step — how many matrix products and
how much wall time each solver needs from the previous state as a guess.

## V.0 The two scenarios (both are production use cases)

**Scenario A — within-geometry SCC iterations.** Inside one `solve_scc`,
each iteration produces a new `H′_i` (charges change) and its projector
`K_i`. The projector sequence `K_0, K_1, …` is a real trajectory — exactly
what `PurifyScc` warm-solves today. No basis transport needed (fixed
geometry → fixed orthogonal basis). **This is the primary test.**

**Scenario B — geometry step (0.01/0.05/0.1 Å).** Basis changes
(`L_new ≠ L_old`), so `K` cannot be reused directly — but the AO density
`D_ao = 2·C_ao·C_aoᵀ` is basis-free, and
    `K_guess = L_newᵀ · (D_ao/2) · L_new`
is an exact transport into the new orthogonal basis (derivation: `C_ao =
L⁻ᵀY` → `Y = LᵀC_ao` → `K = YYᵀ = Lᵀ(D_ao/2)L`). Extrapolation is also
cleaner in AO space: `D_pred = D_1 + γ(D_1 − D_0)`, then transport once.
Residual error is only the ΔH-induced rotation — the certificate gates it.

## V.1 Step 1 — CPU capture (one small instrumentation)

In `DftbCpu::solve_scc` (`dftb_cpu.rs`), right after `y_full` is available
(~line 636, before back-transform), add an env-gated trace:

    if capture {   // RUST_DFTB_SCC_TRACE
        self.scc_trace.push(SccIter {
            h_prime: self.h_prime.clone(),          // the matrix solved this iter
            k:       &y_occ * y_occ.transpose(),    // exact projector K_i (orth basis)
            charges: self.charges.clone(),
            dq_rms,                                  // SCC residual this iter
        });
    }

`pub scc_trace: Vec<SccIter>` (empty unless flagged) — in-memory, no file
I/O needed for the in-Rust benchmark. (~15 lines, additive only.)
`h_prime` is currently `pub`? — no, private: expose a getter `pub fn
h_prime(&self)` or make the trace the only export (preferred: trace
already carries it).

Also capture the **reference answer at the target state**: after
convergence, `K*` from the final iteration is the exact target for the
last step; for per-iteration targets `K_i` itself is exact (it was
produced by dsyevd).

Optional (only if Python cross-checks are wanted): `RUST_DFTB_DUMP_MATS=dir`
writing row-major f64 `h_i.bin`, `k_i.bin` + a manifest — defer unless
needed (YAGNI).

## V.2 Step 2 — standalone solver benchmark on the real trajectory

New test `tests/warm_bench.rs` (or extend `gpu_purify.rs`), gated on
`RUST_DFTB_SK_DIR` like `gpu_scc_bench` — `#[ignore]`d bench, run with
`--ignored --nocapture`.

1. Load molecule via existing helpers (`xyz_file`, `load_sk_for_species`).
   Start **H2O** (n≈6 orbs — fast smoke), then **formic dimer** (28 orbs)
   and **GC** (~70–80 orbs → needs tiled Jacobi, the interesting size).
2. `DftbCpu::new → update_geometry(base) → set_smearing(0) → solve_scc`
   with trace on. `kT=0` gives clean idempotent projectors for the first
   benchmark (see V.5 caveat on smearing).
3. For each captured iteration `i ≥ 2`, assemble the benchmark triple
   `(H′_i, K_{i-1}, K_{i-2})` — batch replicas = iterations.
4. Methods compared on identical inputs (wall time = `Instant` around
   enqueues + one `read_buffer` sync):

   | method | call | cost metric |
   |---|---|---|
   | CPU dsyevd | (already timed via `RUST_DFTB_TIMING`) | ms/iter |
   | GPU Jacobi | `jacobi_cyclic_local_batched` (n≤64) / tiled (n>64) | sweeps, ms |
   | cold TC2 | `purify_tc2_batched` | iters≈products, ms |
   | warm extrap | `warm_extrap_solve(H′_i, K_{i-1}, K_{i-2}, γ=1)` | products, ms |
   | warm no-extrap | `warm_extrap_solve(γ=0)` = transport+retract only | products, ms |
   | (optional) warm Jacobi | Jacobi on `W = V_{i-1}ᵀ H′_i V_{i-1}` — eigvecs are in the trace | sweeps, ms |

   The γ=0 row is the ablation: isolates how much of the win is the
   extrapolation vs. just starting near the manifold.
5. Certify each result vs the exact `K_i` (CPU f64): `ΔE`, `‖ΔK‖`,
   `comm`, `idem`, `Tr`. Report a table:

       iter_i  ‖ΔK_{i-1}‖  ‖QΔHP‖/gap  | TC2 prod ms | xtr prod ms | dmm+TC2 prod ms | jacobi sweeps ms
       (rows: SCC iterations 2..N; expect ‖ΔK‖ to shrink as SCC converges
       — the interesting question is where extrapolation crosses the
       certificate gate relative to cold TC2)

6. Sanity anchors: `‖K_i−K_{i-1}‖` per iteration (trajectory step size —
   should shrink toward convergence), `η_warm = ‖Q_{i-1}ΔH′P_{i-1}‖/gap`
   (the effective perturbation — the number that decides if warm start
   should win at all).

## V.3 Step 3 — geometry-step benchmark

1. Converge CPU at `G0`. Build displaced geoms at δ = 0.01/0.05/0.10 Å —
   two variants: (a) along an H-bond junction via `scan_geoms`-style
   displacement (physical, couples strongly), (b) uniform random
   Cartesian jitter of magnitude δ (generic optimizer step).
   For the extrapolation row also converge at one intermediate geometry
   to have `D_0, D_1` history.
2. At the target geometry: `update_geometry(G_new)` → capture
   `L_new, H0′_new`; run a *fresh* CPU SCC for the reference `K*_new`
   (and its `H′` trajectory if SCC is included).
3. Warm candidates (all matrix-products only):
   - transport only: `K = L_newᵀ(D_old/2)L_new` → McWeeny → cert → corr
   - extrapolate: `D_p = D_1+γ(D_1−D_0)` (AO space) → transport → retract → cert → corr
4. Baselines: cold TC2, Jacobi, dsyevd on the same `H′_new`.
   Table rows = δ ∈ {0.01, 0.05, 0.10 Å} × {first SCC iter, mid-SCC}.

## V.4 Step 4 — end-to-end (only if standalone wins)

Wire `warm_extrap_solve` into `PurifyScc::purify_solve_enq`: keep a
2-deep K-history ring per replica, call with γ=1 each SCC iteration,
`converged=false` → existing DMM+TC2 recipe (explicit fallback).
Measure `scc()` wall time + iterations on `test_gpu_scc_scan400`
(H2O/formic/GC rows, batch=400) vs current defaults.

## V.5 Caveats to respect

- **Smearing (kT=0.002 production)**: smeared `K = Σ f_k c_k c_kᵀ` is NOT
  idempotent — McWeeny `3K²−2K³` would destroy the occupation profile.
  Phase 1–3 run `kT=0`. For production kT: the comm certificate still
  holds (`[H,D]=0` for any function of H) but the retract must change —
  either purify only the subspace (purify `C` not `D`, i.e. CheFSI) or
  re-apply Fermi weights after purification. Flagged, not solved here.
- **n>64**: `jacobi_cyclic_local_batched` caps at n=64; GC needs the
  tiled path — check which kernel `GpuDftb` actually uses for GC and
  time that.
- **Latency vs throughput**: at batch=1, n=28 a kernel launch (~5–10 µs)
  may dominate — report both per-launch count and wall ms, and run the
  batch=400 row for the production-relevant number.
- **Fairness**: warm methods get `K` history for free in production;
  cold TC2/Jacobi don't. Charge cold TC2 its full Palser+fold cost, and
  count the warm transport GEMMs honestly (transport = 2 products).
- **No long loops**: max_corr ≤ 4, cold TC2 ≤ 60 iters, Jacobi ≤ 60
  sweeps — anything slower is reported as a failure, not iterated on.

## V.6 Deliverable

A printed table (per system × δ × SCC-iter): products, wall ms, ΔE,
‖ΔK‖, comm for {dsyevd, Jacobi, cold TC2, warm transport, warm
extrapolation} — answering: *on a real molecule with a real gap and a
real previous answer, does extrapolation+retract beat cold TC2, and by
how much?* Then the SCC-integration decision follows from the numbers.

---

# Part VI — Real-molecule results (`tests/gpu_warm_bench.rs`)

Instrumentation: `DftbCpu.scc_trace` (env `RUST_DFTB_SCC_TRACE`) captures
`(h_prime, eigvals, eigvecs=Y, rms)` per SCC iteration — Y are the
Cholesky-orthonormal eigenvectors, so `K_i = Y_occ Y_occᵀ` is exactly the
object the GPU purifiers produce. Benchmarks: `test_warm_bench_scc`
(per-SCC-iteration, batch = iterations 2..N) and `test_warm_bench_geom`
(geometry steps with basis transport `K = L_newᵀ(D/2)L_new`).

Also discovered during implementation: the production pipeline ALREADY
has warm-started Jacobi (`PurifyScc.b_warm`, `gpu_scc_plan.rs` L386-390:
keeps previous S-orthonormal `c`, solves `W=cᵀH_sc c`, ~1-2 sweeps). The
bench includes it explicitly as `jacW` — it is the real incumbent.

## VI.1 SCC-iteration trajectory (fixed geometry, real charge swings)

`‖ΔK‖` = ‖K_i−K_{i-1}‖ (trajectory step), `η_w` = ‖QΔHP‖/gap.

**H2O** (n=6, nocc=4, 13 iters) — TC2 16 prod vs xtr 19 prod:
  i=2 (η=0.03):  xtr ΔE=8.7e-2 FAIL · γ0 4e-3 · γ0.5 3.5e-2 · jacW 1.1e-7
  i≥4 (ΔK≈0):    all warm ΔE≈3.5e-8 (f32 floor) — indistinguishable

**formic** (n=28, nocc=18) — TC2 24 prod vs xtr 19 prod:
  i=2 (η=0.40):  xtr 7.7e-3 · γ0 1.2e-3 · γ0.5 3.9e-3 · jacW 1.8e-7
  i=3 (η=0.11):  xtr 1.8e-3 · γ0 8.1e-5 · γ0.5 6.8e-4 · jacW 1.1e-7
  i=4 (η=0.008): xtr 7.9e-5 · γ0 1.9e-7 · jacW 5.6e-7
  i≥9:           all warm ≈2.8e-7 — converged

**GC** (n=86, nocc=49) — TC2 28 prod/0.12ms vs xtr 19 prod/0.115ms,
  cold Jacobi ≈75ms/solve, warm Jacobi ≈19ms/solve (tiled path —
  production resident kernel is faster; standalone upper bound):
  i=2 (η=1.7):   xtr 3.6e-2 FAIL · γ0 8.7e-3 · jacW 8.3e-6
  i=5 (η=0.027): xtr 1.4e-4 · γ0 2.2e-7 · jacW 5.6e-6
  i≥9:           all warm ≈1.6e-6 ≈ TC2 (1.8e-6)

### Read-out

1. **Warm DM wins only the late-SCC tail** (η_w≲0.01): matches TC2
   accuracy at 19 vs 24-28 products (~25-30% fewer). Early/mid SCC
   iterations have η_w=0.1-1.7 — the certificate correctly rejects
   (comms_max 0.14/0.03/0.02 vs tol 1e-3 → `converged=false` → fallback).
2. **γ=1 overshoots a decelerating sequence.** SCC is a fixed-point
   iteration — ‖ΔK‖ shrinks each step, so `2K₁−K₂` predicts a step the
   trajectory doesn't take. Consistently γ=0 > γ=0.5 > γ=1 in mid-SCC
   (formic i=4: 1.9e-7 / 1.9e-5 / 7.9e-5). The principled fix is
   DIIS-coefficient extrapolation `ΣcᵢPᵢ` (Σcᵢ=1 — no overshoot), not a
   fixed γ — pending.
3. **Warm Jacobi is uniformly excellent** (ΔE~1e-6 even at i=2, η=1.7) —
   it solves rather than corrects, so trajectory size doesn't matter.
   Its standalone cost at n=86 (~19ms/solve, tiled kernel) is ~150× the
   DM solve (~0.12ms) — but production's resident kernel at 1-2 sweeps
   is far cheaper; the standalone comparison overstates its cost.
4. Timing at n≤28 (batch~11): everything ~0.5ms — launch-latency bound;
   product counts are the meaningful metric at these sizes.

## VI.2 Geometry steps — REVISED (the earlier version measured the wrong scenario)

First implementation found a real bug class: fixed η=4 (synthetic-tuned)
DIVERGED on real spectra (rh ×1.6/round up); η=1 converged (×0.6/round).
Fixed properly: **accept/reject trust region** — checkpoint D, reject a
round if rh increases (restore, η/2), accept → η×2 (cap 8). Now
self-calibrating per system.

### VI.2a The wrong-scenario bug (important, was the whole "failure")

The first version of this benchmark restarted the new geometry with
**neutral charges** (`reset_charges` → target = `H'(new | q0)`), while
the seed was a *converged-SCC* density. The seed error was therefore
dominated by the SCC-response jump, not by the geometry displacement —
dp≈0.08 nearly *independent of δ* (0.085/0.083/0.080 at δ=0.01/0.05/0.10)
was the tell. Production restarts SCC with the **previous converged
charges** (`set_charges` exists for exactly this); the honest target is
`H'(new | q_old)`. With that fixed, the same code produces:

### VI.2b Corrected results (charges carried over; G0→G½→Gδ trajectory)

Raw seed quality (comm = ‖HK−KH‖_F before any correction) and
`warm_extrap_solve` outcome (γ=0 for transported seeds — the AO
extrapolation is already inside `xtrAO`):

| system | δ | seed | raw comm | prod | corr | ΔE | conv |
|---|---|---|---|---|---|---|---|
| H2O | 0.01 | transp(D_½) | 1.2e-3 | 3 | 0 | 1.9e-6 | ✓ |
| H2O | 0.01 | **xtrAO 2D₁−D₀** | **3.8e-5** | **3** | **0** | 1.6e-7 | ✓ |
| H2O | 0.05 | transp(D_½) | 5.6e-3 | 11 | 2 | 1.1e-6 | ✓ |
| H2O | 0.05 | **xtrAO** | **2.3e-4** | **3** | **0** | 1.6e-7 | ✓ |
| H2O | 0.10 | transp(D_½) | 1.4e-2 | 11 | 2 | 9.1e-7 | ✓ |
| H2O | 0.10 | **xtrAO** | **6.0e-4** | **3** | **0** | 8.5e-7 | ✓ |
| formic | 0.01 | transp(D_½) | 2.0e-3 | 3 | 0 | 4.3e-6 | ✓ |
| formic | 0.01 | **xtrAO** | **3.2e-4** | **3** | **0** | 2.8e-7 | ✓ |
| formic | 0.05 | transp(D_½) | 1.0e-2 | 11 | 2 | 7.4e-6 | ✓ |
| formic | 0.05 | **xtrAO** | **1.8e-3** | **3** | **0** | 2.6e-6 | ✓ |
| formic | 0.10 | transp(D_½) | 1.6e-2 | 15 | 3 | 8.7e-6 | ✓ |
| formic | 0.10 | **xtrAO** | **1.5e-3** | **3** | **0** | 1.6e-6 | ✓ |

Baselines on the same target: cold TC2 = 16 (H2O) / 24 (formic)
products to comm~1e-7; cold Jacobi ~0.1–2.4 ms, comm~1e-7.

**`xtrAO` = 3 products (extrapolate→McWeeny→certify, ZERO DMM
corrections) at every δ tested — a 5–8× product reduction vs cold TC2,
certificate-passing.** The AO-indexed density extrapolates linearly in
δ because same-coefficient orbitals ≈ atom-following orbitals, so
`2D₁−D₀` captures the O(δ) response exactly; residual is O(δ²).

### VI.2c Negative result — faithful orbital transport is WRONG ansatz

`S_cross[ν,μ]=⟨χ_ν^new|χ_μ^old⟩` (cross-geometry SK overlap, verified
`s_cross(G,G)=S` to 1e-16) + projection `d=S_new⁻¹·S_cross·C_old` +
re-orthonormalization (self-transport reproduces K exactly) — produces
a seed **10× worse** than same-coefficients (comm ~0.7 vs 0.001–0.02).
Physical reason: faithful projection pins each orbital to its OLD
absolute position; the ground state instead follows the atoms. For
atom-centered bases **same-AO-coefficients IS the correct transport**;
S_cross transport is the wrong ansatz, not an implementation bug.

## VI.3 Honest verdict (revised)

- **Geometry-step warm start WORKS**: `2D₁−D₀` in AO space + transport
  `L_newᵀ(D/2)L_new` + 1 McWeeny + certificate = **3 products vs 16–24
  cold TC2**, all δ∈{0.01,0.05,0.10} Å, both systems, `converged=true`.
  Prerequisites: carry converged charges (production does) and use the
  last TWO geometry densities (one-point transport alone also passes:
  3–15 products).
- SCC tail: ~25–30% fewer products than cold TC2 (VI.1) — modest but real.
- The two wins share one pipeline: `warm_extrap_solve` already implements
  predict→retract→certify→correct→explicit-fallback; the geometry case
  adds host-side AO extrapolation + `LᵀDL` transport (2 GEMMs, could be
  fused into a kernel later).
- The incumbent `b_warm` Jacobi remains uniformly strong (VI.1 item 3).
  The DM path's edge is 3 pure GEMMs — no sweeps, no eigensolver —
  exactly the architecture goal; at n≤28 wall-clock is launch-bound
  (~0.35 ms vs 0.45–0.65 ms cold TC2) so the advantage must be realized
  at batch/large-n.
- Still open: DIIS-coeff predictor for decelerating SCC (γ=1 overshoots,
  VI.1 item 2); smearing (kT>0) invalidates McWeeny — needs CheFSI-style
  handling; per-system freeze in batched corrections; wire into
  `purify_solve_enq` warm path with the explicit TC2 fallback.

---

# Part VII — End-to-end production comparison: Jacobi vs DM path

## VII.1 The three loops (naming the confusion)

```
Loop 1  GEOMETRY  (FIRE/BFGS)              — moves nuclei; ~10–100 steps
  └ Loop 2  SCC   (charge self-consistency) — H'(q) → ρ → q' → mix → repeat
      └ Loop 3  INNER SOLVE  H' → ρ         — THE CHOICE LIVES HERE
                · Jacobi eigendecomposition (sweeps inside one launch)
                · purification TC2 / DMM / warm-extrapolation (GEMMs)
```

DM-minimization is NOT the SCC loop — it is one implementation of the
inner solve (level 3). The confusion arises because the production
`PurifyScc` path runs a *fixed* DMM schedule per SCC iteration, so the
two levels blur. Conceptually: level 3 produces `ρ(H')`, level 2
produces `q(ρ)` and mixes charges. A fourth "level 0" exists implicitly:
across BOTH loop-1 steps and loop-2 iterations the *previous* solve is
a warm seed — that is what this whole project exploits.

## VII.2 Measured: production scan400 bench (20×20 grid, batch=400,
##      kT=0.002 smearing, SCC tol=1e-6, GPU = this machine's OpenCL)

`RUST_DFTB_EIGSOLVER=auto` (Jacobi: resident kernel n≤128, block n>128,
warm `b_warm` c-reuse for n>64) vs `=purify` (existing PurifyScc:
fixed ~16 DMM + TC2 schedule per SCC iter — NOT the new extrapolation).

| system | n | solver | iters | ms/iter | scc ms | sys/s | failed |
|---|---|---|---|---|---|---|---|
| formic | 28 | Jacobi | 40 | 0.51 | 20.2 | 19 816 | 0 |
| formic | 28 | purify | 64 | 1.02 | 65.0 | 6 155 | 0 |
| GC | 86 | Jacobi | 80 | 2.25 | 180.3 | 2 219 | 0 |
| GC | 86 | purify | 100* | 18.95 | 1 895 | 211 | **236/400** |
| DTH | 246 | Jacobi | 24 | 71.5 | 1 715 | 233 | 0 |
| DTH | 246 | purify | 56 | 915.3 | 51 257 | 7.8 | 0 |

(*hit max_iter.) batch=1 rows for reference — formic: 5.9 vs 29.3 ms;
GC: 12.2 vs 25.4 ms; DTH: 180 vs 361 ms — same story, worse.

**Verdict: warm Jacobi wins decisively today** — purify is 3–30× slower
per iteration, needs 1.5–2.3× more SCC iterations (noisier Mulliken
charges → DIIS fallbacks: 4299 on GC), and FAILS to converge on 59% of
GC replicas under kT=0.002 smearing. At n=246 the purify regtile GEMM
cannot even launch (fell back to row·row = 915 ms/iter).

## VII.3 Where the time actually goes (GC, batch=400, Jacobi path,
##      RUST_DFTB_PROF — host ms over 80 iters × chunks, 276 calls)

| stage | ms/call | share | content |
|---|---|---|---|
| `scc.jacobi` | 1.53 | 65% | inner eigensolve (warm-started resident Jacobi) |
| `scc.hscc` | 0.39 | 17% | Δq→V=γΔq→H_scc assembly |
| `scc.diis` | 0.17 | 7% | charge mixing |
| `scc.sc_gemm`+`gemm_th*` | 0.26 | 11% | Mulliken populations GEMMs |
| everything else | <0.01 | <1% | occ, finishes, checks |

**The inner solve is 65% of an SCC iteration.** A DM inner solve using
the measured GEMM cost (~0.087 ms per n=86 batched GEMM — see
`scc.sc_gemm` row) at the VI.2b-validated cost of 3 products + 2-GEMM
transport ≈ **0.3–0.45 ms/iter** → ~1.2 ms/iter total ≈ **~1.9×
end-to-end SCC speedup** vs Jacobi — IF it converges like the VI.2b
bench (it did there: 0 corrections needed at δ≤0.10 Å).

## VII.4 Why existing `purify` loses — profiled decomposition

`scc.purify` alone = **20.09 ms/call = 90.4%** of the purify SCC
iteration (GC b400, RUST_DFTB_PROF). The reason is not the algorithm —
it is the schedule:

```
per SCC iteration (warm path), unconditionally:
  ortho   Xᵀ·H·X                      2 GEMM
  16× DMM step (T=HK, Y=KT, update)  32 GEMM   ← fixed count, no early exit
  8× McWeeny retract (K², K³)        16 GEMM
  4× TC2 tail step                    4 launches
  certificate T=HK + rh               1 GEMM
  back-transform X·K·Xᵀ               2 GEMM
  ≈ 57 launches × ~0.35 ms/batched_gemm_active (n=86, b400) ≈ 20 ms
```

vs the Jacobi inner solve: **one launch, 1.53 ms** (resident kernel,
1–2 warm sweeps).

The killer facts:
1. **No early exit** — the certificate is computed but the schedule
   already ran; converged replicas still pay all 57 launches.
2. **Per-product launches** — every GEMM is a separate
   `batched_gemm_active` enqueue (~0.35 ms each at b400). The fast
   machinery exists and was measured (see VII.6): resident
   `gemm_sq_iter` does a purification step in **0.047–0.054 ms** —
   ~7× cheaper per product — and `tc2_step_batched` is already fused.
3. **Cold-schedule every iteration** — 16 DMM + retracts + tail is the
   *hard-start* recipe. The VI.2b warm path needs **3 GEMMs total**
   (extrapolate → McWeeny → certify) when the seed is good — which it
   is for real trajectories.
4. **Smearing**: kT=0.002; TC2/McWeeny assume integer occupation →
   noisy Mulliken q → 1.5–2.3× more SCC iters, 4299 DIIS fallbacks on
   GC, 236/400 hard failures.
5. **n>128**: regtile GEMM infeasible at n=246 → row·row fallback
   → 915 ms/iter (DTH).

## VII.6 Earlier evidence that purification IS faster than Jacobi

(The scan400 test above uses the old fixed schedule — the fast path
was measured standalone, in the dense-multi-eigensolve work.)

| measurement | where | number |
|---|---|---|
| resident `gemm_sq_iter` step, n=86 b400 | chat doc §gemm_sq_iter | **0.047–0.054 ms/iter** (9.4–10.8 TF-eq) |
| fused `tc2_step_batched` step | same | 0.141 ms/iter |
| Jacobi solve, n=86 resident/direct | gpu_eigen T08 notes | 11.4 / 21.9 ms per solve |
| standalone cold TC2, n=86 (my bench) | VI.1 | ~0.32 ms/solve |
| standalone cold Jacobi, n=86 | VI.1 | ~75 ms/solve |
| warm Jacobi `W=cᵀHc`, n=86 | VI.1 | ~19 ms/solve |

So purification-as-inner-solve measured **~30–200× cheaper** than the
Jacobi solve at n=86 — the production purify path simply does not use
that machinery (it launches `batched_gemm_active` per product instead
of the resident/fused kernels, and runs the full cold schedule every
iteration).

## VII.5 Practical path to a production win

| step | content | expected |
|---|---|---|
| 1 | Geometry-step warm start (VI.2b): AO extrap `2D₁−D₀` + `LᵀDL` transport + retract + certify → TC2 fallback | inner solve ~3 GEMMs at scan/relax steps |
| 2 | SCC tail: extrapolate `K_{i-1},K_{i-2}` once η_w<~0.01 (certificate-gated) | ~25% fewer products on tail iters |
| 3 | kT=0 first; smearing needs CheFSI or FDM-aware retraction — do NOT run McWeeny on fractional occ | correctness |
| 4 | tiled GEMM for n>128 purify path | enables n=246+ |

The honest summary for the geometry-optimization task: **today, warm
Jacobi is faster end-to-end** (VII.2). The DM path's win exists but is
*unrealized* — it requires shipping the extrapolation+certificate
pipeline (proven at 3 GEMMs/solve on real geometry steps) instead of
the fixed-schedule recipe currently behind `EIGSOLVER=purify`.

---

# Part VIII — Definitive measured comparison + the actual root cause

All numbers MEASURED on the production scan400 bench (20×20 geometry
grid, batch=400, mio-1-1 SK, SCC tol=1e-6, max_iter=100). Nothing in
this section is extrapolated; predictions are marked **[PREDICTED]**.

## VIII.0 Reproducing the old "cold purify beats Jacobi" result

`purify_bench` (`tests/gpu_purify.rs`, random symmetric n=86, nocc=43,
batch=400, max 60 iters, tol=1e-5), this GPU, this session:

| solve | ms/solve | ms/iter |
|---|---|---|
| **cold TC2 (`purify_tc2_batched`, fused kernels)** | **10.70** | 0.178 |
| cold resident Jacobi (recorded ref, this machine) | ~17.1 | — |
| warm/"one" resident Jacobi (recorded ref) | ~9.7 | — |

**So the old result stands: cold purify beats COLD Jacobi ~1.6×**
(design doc recorded 8.46 ms — same ballpark, kernels drifted since).
The catch: production Jacobi is never cold — `b_warm` reuses the
previous eigenvectors and Jacobi rotates them in ~1–3 sweeps →
**1.53 ms/iter inner solve**. The comparison that decides production is
warm-vs-warm, and the design doc already knew always-cold loses
(§7.6.5-C: "~7.9 ms/iter on GC — loses at n=86, wins vs cold Jacobi").

## VIII.1 The measured production table — all systems, batch=400

kT=0.002 (production smearing). `EIGSOLVER=auto` = Jacobi path;
`=purify` = cert-gated warm path (cert + ≤4 DMM/McWeeny rounds,
unconditional fused-TC2 fallback for uncertified replicas);
`=purify WARM=0` = cold Palser+60 TC2 every iteration.

| system | n | Jacobi it | Jacobi ms/it | Jacobi ms | fail | purify it | purify ms/it | purify ms | fail |
|---|---|---|---|---|---|---|---|---|---|
| H2O | 6 | 16 | 0.155 | 2.5 | 0 | 80 | 0.369 | 29.5 | 0 |
| formic | 28 | 40 | 0.520 | 20.8 | 0 | 88 | 0.517 | 45.5 | 0 |
| azaindol | 84 | 40 | 2.045 | 81.8 | 0 | 100* | 3.133 | 313.3 | 16 |
| GC | 86 | 80 | 2.262 | 180.9 | 0 | 100* | 6.344 | 634.4 | 193 |
| AT | 87 | 48 | 1.919 | 92.1 | 0 | 40 | 3.077 | 123.1 | 0 |
| diazaphen | 120 | 40 | 9.099 | 364.0 | 0 | 100* | **5.405** | 540.5 | 4 |

cold-purify-every-iteration rows (GC, formic):

| system | it | ms/it | ms | fail |
|---|---|---|---|---|
| formic | 100* | 0.665 | 66.5 | 4 |
| GC | 100* | 6.655 | 665.5 | 207 |

(*hit max_iter.) Reads:
- **per-iteration crossover exists**: at n=120 purify is already faster
  than Jacobi per-iter (5.4 vs 9.1 ms — GEMM scaling beats Jacobi
  sweeps); at n≤87 it loses. AT converged in FEWER iters under purify
  (40 vs 48); most systems need MORE iters under purify (integer-occ
  density changes the residual surface).
- The failure populations (GC 193, azaindol 16, diazaphen 4) are
  explained in VIII.2 — not a warm-start bug.

## VIII.2 Root cause of the failures — found; it is NOT the
##      warm-start machinery and NOT TC2

Ablation chain, all measured on GC batch=400:

| experiment | result | eliminates |
|---|---|---|
| cold Palser+60 TC2 every iter (WARM=0) | 207 fail | warm-start logic, DMM, certificates |
| same + COLD_STEPS=150 | 207 fail | TC2 step budget |
| same + kT=0 | 207 fail | — |
| **Jacobi at kT=0** | **232 fail** | ← the discriminator |
| cold purify cert readback | `max_rh ≈ 1e-6`, **0 uncertified every chunk** | TC2 non-convergence |

**The purification converges perfectly** — every replica, every chunk,
produces a commuting projector at the f32 floor (rh≈1e-6). The failures
come from what that projector IS: an **integer-occupation** object.
Under kT=0.002, Jacobi builds `D` with fractional Fermi weights `f_bk`;
the purify path builds `K` = projector (0/1 eigenvalues) — fractional
occupation is never applied. For the ~half of the GC scan grid where
the proton has transferred and the HOMO–LUMO gap ≲ kT, that is the
*wrong density*. Proof: at kT=0 Jacobi fails on essentially the same
replica population (232 vs purify's 207) — integer-occupation SCC is
intrinsically unstable on near-degenerate geometries (discontinuous
occ → charge oscillation), and purify-at-any-kT effectively runs
kT=0 physics. At kT=0 the purify path is actually *more* robust than
Jacobi (207 < 232 failures).

The design doc already flagged this: §6.9 "0 K density only … parity
tests must compare against the kT→0 eigensolve contract."

## VIII.3 Why a fixed schedule can never win at n=86 — arithmetic

Measured unit costs (this GPU, n=86, batch=400):

| operation | cost |
|---|---|
| fused `tc2_step_batched` launch | ~0.09–0.14 ms |
| `batched_gemm_active` launch | ~0.35 ms |
| resident `gemm_sq_iter` GEMM | ~0.05–0.09 ms |
| warm Jacobi inner solve (production) | **1.53 ms** |

Cold purify per SCC iter = 60 TC2 launches ≈ 6.6 ms — matches the
measured 6.655 ms/iter. Even a 20-step cold TC2 (2.2 ms) ≥ Jacobi 1.53.
**Any fixed per-iteration cold schedule loses at n≤~90.** The only way
purify wins: a warm schedule of ≲10 fused products that actually
converges — or n>~100 where GEMM scaling wins anyway (diazaphen row).

## VIII.4 Where the warm win must come from — and the real obstacles

Mid-SCC the charge update produces ΔH' big enough that the previous
projector needs a subspace rotation of order θ~1 rad; a trust-region
DMM step rotates only a little per round — 4 rounds at η=1 gives ~×0.1
→ rh~0.03–0.27 ≫ tol → measured uncertified counts of 135–178/400 on
fb-off chunks. Standalone proved fixed-η cannot cover this range
(η=4 converges fast on some systems, diverges on others → needs the
accept/reject trust region that exists in `warm_extrap_solve` but was
never ported into production).

The designs that can actually win:

**(A) co-iteration** — drop the per-iteration certificate *gate*. K
persists on device and tracks H'(q) with a fixed small budget per SCC
iter (1 cert GEMM + 1–2 DMM+McWeeny rounds ≈ 5–11 GEMMs ≈ 0.4–0.8 ms);
SCC convergence requires `dq_rms < tol AND rh < tol` — both already
read at the chunk-end sync. K converges together with q; as SCC
settles, ΔH→0 and the last rounds are trivially certified. No fallback
block exists. **[PREDICTED] ~0.4–0.8 ms/iter inner** vs Jacobi 1.53 ms
— arithmetic measured, schedule not yet.

**(B) fused device-side loop** (chat option B): the same body inside
ONE kernel with per-workgroup exit → **[PREDICTED] ~0.15–0.3 ms/iter**.
Bigger change; (A)'s kernels become its body.

Safety note for (A): a replica whose dq converges while K lags would be
spuriously "converged" — the rh<tol term in the convergence criterion
is the guard. Wrong-subspace lock-in (a commuting projector onto the
wrong invariant subspace passes rh) is bounded by DMM's Tr(HK)-descent
+ accept/reject η (can't leave the correct basin without an overshoot
the reject catches).

## VIII.5 The smearing problem is the real blocker — options

To compete on kT>0 workloads at all, purify must produce `f_β(H−μ)`,
not θ(μ−H):

1. **Early-stopped purification** — intermediate TC2 iterates are
   sigmoid approximations to the step; transition width shrinks ~×2 per
   step. Stopping at the step whose width ≈ kT gives a Fermi-LIKE
   density (~9 steps for kT=0.002, span~1 Ha). Cheap; but the sigmoid
   shape ≠ exact Fermi function → slightly different fixed point than
   Jacobi's. Probably acceptable (smoothness is what stabilizes SCC);
   needs parity check.
2. **Chebyshev expansion of f_β** — exact FD shape; needs
   ~Δε/(5·kT) ~ 100+ terms at kT=0.002 → ~11 ms/solve at n=86 —
   **loses to Jacobi**. Reject for this kT.
3. **Projector + boundary low-rank correction** — K does not separate
   eigenvectors; boundary extraction needs a subspace eigensolve —
   mostly defeats the purpose.
4. **Scope to kT=0** — honest for workloads without smearing; but at
   kT=0 the GC benchmark is intrinsically hard even for Jacobi
   (232 fails) → measure the win on convergent gapped systems
   (H2O/formic/azaindol/AT).

## VIII.6 Status

- Warm purify is correct wherever the physics is well-defined: formic
  0.517 vs Jacobi 0.494 ms/iter (parity at n=28), AT converged in fewer
  iters (40 vs 48), diazaphen already faster per-iter (5.4 vs 9.1 at
  n=120).
- GC-class failures = missing finite-T density, not the warm machinery.
- Speed win needs BOTH: (a) co-iteration/fused warm schedule replacing
  the fixed fallback schedule, (b) a finite-T density for kT>0 — or
  scope to kT=0 and accept that degenerate scans are hard for everyone.

## VIII.7 The fair fight — kT=0, all systems (equal physics)

Since purify produces an integer projector regardless, the correct
comparison is Jacobi at kT=0 vs purify. Measured, batch=400:

| system | n | Jacobi it | ms/it | fail | purify it | ms/it | fail |
|---|---|---|---|---|---|---|---|
| H2O | 6 | 16 | 0.069 | 0 | 80 | 0.358 | 0 |
| formic | 28 | 100* | 0.338 | 4 | 88 | 0.567 | **0** |
| azaindol | 84 | 100* | 2.381 | 32 | 100* | 3.143 | **16** |
| GC | 86 | 100* | 4.893 | 232 | 100* | 6.695 | **193** |
| AT | 87 | 100* | 2.066 | 6 | **40** | 3.159 | **0** |
| diazaphen | 120 | 100* | 7.638 | 25 | 100* | **5.693** | **4** |

Two surprises, both in purify's favor:

1. **Purify is more robust than Jacobi at kT=0 — fewer failures on
   every system** (formic 0v4, azaindol 16v32, AT 0v6, diazaphen 4v25,
   GC 193v232). Likely mechanism: Jacobi's integer occupation picks an
   arbitrary eigendecomposition inside a near-degenerate manifold and
   the choice can flip between SCC iterations → charge oscillation;
   TC2's projector is a smooth function of H′ → consistent across
   iterations → SCC settles. AT is the showcase: purify converged all
   400 replicas in 40 iters; Jacobi hit the cap with 6 failed.
2. **Purify already wins per-iteration at n=120** (5.69 vs 7.64 ms —
   the dumb schedule included) and wins outright on AT (126 vs 207 ms
   total, because it converged and Jacobi did not).

## VIII.8 Cheap smearing options — ranked by cost

To survive kT>0 the purify path needs `f_β(H−μ)` not θ(μ−H):

1. **Early-stopped purification — nearly free.** Intermediate TC2
   iterates are smooth sigmoid approximations to the step; the
   transition width halves each step. Palser maps the spectrum to
   [0,1]; stopping after m steps leaves a sigmoid of width ≈ 2⁻ᵐ·Δε.
   For kT=0.002 Ha / Δε~1 Ha → m≈9 steps. **Finite-T is then CHEAPER
   than full purification** (~9 vs ~56 steps). Caveats: the sigmoid is
   a specific polynomial shape, not the exact Fermi function → the
   fixed point differs slightly from Jacobi's (needs a charges/energy
   parity check vs a f64 Fermi reference); trace must be watched since
   the branch keeps Tr≈nocc — fine.
2. **Smearing annealing in the outer loop** — run SCC at large kT
   (converges, smooth), then continue at target kT warm-started. Costs
   nothing in the inner solve, but helps Jacobi, not purify (purify is
   integer regardless). Useful only as a convergence stabilizer.
3. **Chebyshev/FOE of the true Fermi function** — ~Δε/(5kT) ≈ 100+
   terms at kT=0.002 → ~11 ms/solve at n=86. Loses to Jacobi. Reject.
4. **Two-projector boundary smearing** — `D ≈ w_lo·K_{n} + w_hi·K_{n+1}`
   mimics one fractional level; needs eigenvalue estimates for the
   weights and doubles the purify cost. Not competitive.
5. **Scope to kT=0** — legitimate: at kT=0 purify is already *more*
   robust than Jacobi (VIII.7). The win then lives on convergent
   gapped systems (AT/diazaphen today) plus the schedule work.

## VIII.9 Bottom line

- The old result stands and is reproduced: **cold TC2 = 10.70 ms vs
  cold Jacobi ~17.1 ms** standalone — but production Jacobi is warm
  (1.53 ms/iter inner), so cold-per-iteration never wins at n=86.
- At kT=0 purify is **more robust AND already faster at n≥~100**
  (diazaphen 5.69 vs 7.64 ms/iter; AT converged where Jacobi failed).
- The remaining gap at n≤87 is the fixed warm schedule + unconditional
  fallback block — the co-iteration design (VIII.4-A) is the intended
  fix; smearing via early-stopped purification (VIII.8-1) is the cheap
  candidate for kT>0.

