# Relax solver — implementation notes and open questions

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

## II.11 File map update

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
