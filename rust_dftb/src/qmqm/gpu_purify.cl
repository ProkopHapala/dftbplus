// ============================================================================
// gpu_purify.cl — dense batched density-matrix purification (TC2)
//
// One workgroup per system. Orthogonal-basis TC2 (Niklasson trace-
// correcting purification, second order): iterate D ← D² or 2D − D²
// chosen per system so Tr(D) → Nocc while eigenvalues polarize to {0,1}.
// Same semantics as the sparse metric-TC2 path (gpu_sparse.rs::tc2_step):
//   branch = (Tr(D) > Nocc) ? contract : expand.
//
// The fused step is ONE kernel launch per iteration and performs ZERO
// host synchronization inside the loop:
//   pass 1: T = D·D  (GEMM; T written into Dout) + ‖T−D‖² accumulated
//   pass 2: Dout ← branch ? T : 2·D − T   (in-place; each thread owns
//           its slots) + Tr(Dout) accumulated → traces[sys]
// Branch input is Tr(Din) from traces[sys] (written by the previous
// iteration / init kernel) — no trace recompute, no extra pass.
//
// GEMM interior — PURIFY_GEMM / PURIFY_TILE select the variant
// (identical results):
//   PURIFY_GEMM=2  register-tiled symmetric-square GEMM (ported from
//      gpu_gemm.cl gemm_regtile/sq — measured 6.7 TFLOPS ≈ 19% of peak
//      at n=86/batch=400 vs 0.9 TFLOPS row·row): TX×TY thread grid,
//      each thread owns an RTY×RTX register micro-tile, the WG tile
//      (TY·RTY)×(TX·RTX) must cover the WHOLE n×n output (the As_t
//      staging serves both operands — D symmetric ⇒ B[k][j] = A[j][k]
//      = As_t[k][j]). Requires lsz == TX·TY — enforced host-side.
//   PURIFY_TILE>0  cooperative tiled GEMM — BROKEN (never writes errs,
//      unwritten-diag symptom); kept for reference only.
//   else           row·row: T_ij = row_i·row_j — coalesced 86-fma dots,
//      ZERO local memory / barriers in the GEMM; ~0.9 TFLOPS.
//
// Batched/multi-system only — a single small system does not saturate
// the GPU and is explicitly out of scope.
// ============================================================================

#ifndef PURIFY_WG
#define PURIFY_WG 256
#endif
#ifndef PURIFY_TILE
#define PURIFY_TILE 0
#endif
#ifndef PURIFY_GEMM
#define PURIFY_GEMM 0
#endif
// register-tile shape (only used when PURIFY_GEMM == 2). Defaults are
// the measured n=86 optimum: 22×11 threads × 8×4 regs = 88×88 cover.
#ifndef PURIFY_TX
#define PURIFY_TX 22
#define PURIFY_TY 11
#define PURIFY_RTX 4
#define PURIFY_RTY 8
#define PURIFY_TK 8
#endif
#define PURIFY_WM (PURIFY_TY * PURIFY_RTY)
#define PURIFY_WN (PURIFY_TX * PURIFY_RTX)

// Fold-then-halve workgroup reduce — correct for arbitrary (non-PoT) lsz.
// For odd s the fold MUST use h = ceil(s/2): pairs (i, i+h) for i < s−h
// fold sources h+1..s−1 into dests 0..h−1 while element h carries — every
// contribution counted exactly once, no dest/source overlap. (The earlier
// h = floor(s/2) form silently DROPPED the last element for odd s — e.g.
// lsz=242 lost lids 120,241 ≈ 3.1 of the diagonal trace → wrong TC2
// branch → wrong-rank projector.)
// The trailing barrier before `return` is REQUIRED: without it a fast
// thread can enter the NEXT reduce call and overwrite red[0] before a
// straggler has read the result (observed as random corrupted bounds/
// traces at large batch — race, nondeterministic victim).
static inline float pur_reduce(__local float* red, float v, const int lid, const int lsz) {
    red[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz; s > 1; s = (s + 1) >> 1) {
        const int h = (s + 1) >> 1;
        if (lid < s - h) red[lid] += red[lid + h];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float out = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return out;
}
static inline float pur_reduce_min(__local float* red, float v, const int lid, const int lsz) {
    red[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz; s > 1; s = (s + 1) >> 1) {
        const int h = (s + 1) >> 1;
        if (lid < s - h) red[lid] = fmin(red[lid], red[lid + h]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float out = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return out;
}
static inline float pur_reduce_max(__local float* red, float v, const int lid, const int lsz) {
    red[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz; s > 1; s = (s + 1) >> 1) {
        const int h = (s + 1) >> 1;
        if (lid < s - h) red[lid] = fmax(red[lid], red[lid + h]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float out = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return out;
}

// ------------------------------------------------------------------
// tc2_init_batched — Palser start:  D0 = (λmax·I − H) / (λmax − λmin)
// with Gershgorin bounds (row |·| sums). Eigenvalues land in [0,1];
// exact trace need not equal Nocc — TC2 corrects it (that's the point
// of trace-correcting purification). Writes the true Tr(D0).
// ------------------------------------------------------------------
__kernel void tc2_init_batched(
    __global const float* H,      // [batch][n*n] — H̃ (orthogonal basis)
    __global float*       D,      // [batch][n*n] — out: D0
    __global float*       traces, // [batch]      — out: Tr(D0)
    const int n,
    const int batch)
{
    const int sys = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __global const float* h = H + (size_t)sys * n * n;
    __global float*       d = D + (size_t)sys * n * n;

    __local float red[PURIFY_WG];
    __local float lmax, lmin;

    float rmax = -1.0e30f, rmin = 1.0e30f;
    for (int i = lid; i < n; i += lsz) {
        float s = 0.0f;
        for (int j = 0; j < n; ++j) s += fabs(h[i * n + j]);
        const float dg = h[i * n + i];
        rmax = fmax(rmax, dg + (s - fabs(dg)));
        rmin = fmin(rmin, dg - (s - fabs(dg)));
    }
    lmax = pur_reduce_max(red, rmax, lid, lsz);
    lmin = pur_reduce_min(red, rmin, lid, lsz);
    const float span = fmax(lmax - lmin, 1.0e-30f);

    float tr = 0.0f;
    for (int idx = lid; idx < n * n; idx += lsz) {
        const int i = idx / n;
        const float v = ((i == idx - i * n ? lmax : 0.0f) - h[idx]) / span;
        d[idx] = v;
        if (i == idx - i * n) tr += v;
    }
    tr = pur_reduce(red, tr, lid, lsz);
    if (lid == 0) traces[sys] = tr;
}

// ------------------------------------------------------------------
// tc2_extrap_batched — trajectory-extrapolation warm seed (XL-BOMD /
// Niklasson):  Out = 2·K1 − K2, where K1,K2 are the converged
// projectors of the previous two SCC iterations. A converged projector
// is a HARD fixed point of the TC2 map (K²=K — polynomials never
// rotate the subspace), so re-purifying K1 is a no-op, and the Palser-
// delta shift K1+(D0'−D0) drifts to a wrong-but-idempotent subspace
// (measured: ‖HK−KH‖=0.24 at "convergence"). The difference K1−K2
// carries the ACTUAL occupied-subspace motion — a first-order rotation
// — and vanishes at the SCC fixed point → no residual floor.
// Gershgorin guard: if the extrapolated spectrum escapes [0,1] the seed
// is renormalized (eigenvectors/ordering preserved), else identity.
// ------------------------------------------------------------------
__kernel void tc2_extrap_batched(
    __global float*       Out, // [batch][n*n] — out: warm seed
    __global const float* H1,  // [batch][n*n] — latest converged projector
    __global const float* H2,  // [batch][n*n] — previous converged projector
    const int n,
    const int batch)
{
    const int sys = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __global float*       o  = Out + (size_t)sys * n * n;
    __global const float* h1 = H1  + (size_t)sys * n * n;
    __global const float* h2 = H2  + (size_t)sys * n * n;
    __local float red[PURIFY_WG];

    // Pass 1 — Gershgorin bounds of the extrapolated seed S = 2·K1 − K2.
    // Extrapolation overshoot can push eigenvalues outside [0,1], where
    // the TC2 branch map diverges (observed: Tr → −2e5 → NaN).
    float rmax = -1.0e30f, rmin = 1.0e30f;
    for (int i = lid; i < n; i += lsz) {
        float s = 0.0f;
        for (int j = 0; j < n; ++j)
            s += fabs(2.0f * h1[i * n + j] - h2[i * n + j]);
        const float dg = 2.0f * h1[i * n + i] - h2[i * n + i];
        rmax = fmax(rmax, dg + (s - fabs(dg)));
        rmin = fmin(rmin, dg - (s - fabs(dg)));
    }
    const float lmax = pur_reduce_max(red, rmax, lid, lsz);
    const float lmin = pur_reduce_min(red, rmin, lid, lsz);
    const bool renorm = (lmin < -0.02f) || (lmax > 1.02f);
    const float inv = renorm ? 1.0f / fmax(lmax - lmin, 1.0e-30f) : 1.0f;
    const float off = renorm ? lmin : 0.0f;

    // Pass 2 — Out ← (S − off·I)·inv — affine map preserves eigenvectors
    // and ordering → the tracked occupied subspace survives.
    for (int idx = lid; idx < n * n; idx += lsz) {
        const int i = idx / n;
        const float v = 2.0f * h1[idx] - h2[idx];
        o[idx] = (v - (i == idx - i * n ? off : 0.0f)) * inv;
    }
}

// ------------------------------------------------------------------
// tc2_step_batched — one fused TC2 iteration (see header). Ping-pong
// Din → Dout; host swaps the two args each launch. errs[sys] receives
// ‖Din²−Din‖_F (idempotency of the CURRENT iterate — converge checks
// apply to Din; overshooting past convergence is harmless since the
// update is contractive at the fixed point).
// ------------------------------------------------------------------
__kernel void tc2_step_batched(
    __global const float* Din,
    __global float*       Dout,
    __global const float* nocc,
    __global float*       traces,   // in: Tr(Din) · out: Tr(Dout)
    __global float*       errs,     // out: ‖Din²−Din‖_F
    __global int*         done,     // in/out: per-system converged flag
    const float           tol,      // idempotency tolerance
    const int n,
    const int batch)
{
    const int sys = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __global const float* A = Din  + (size_t)sys * n * n;
    __global       float* C = Dout + (size_t)sys * n * n;

    __local float red[PURIFY_WG];
    __local float l_trd;
    __local int   l_frozen;
    float er = 0.0f;

    // Frozen systems only mirror Din → Dout (ping-pong keeps D valid in
    // both buffers) — NO update: the TC2 fixed point is marginally stable
    // in f32 — dust eigenvalues regrow ×2/step under the wrong branch, so
    // iterating a converged system destroys it while others finish.
    if (done[sys]) {
        for (int idx = lid; idx < n * n; idx += lsz) C[idx] = A[idx];
        return;
    }

#if PURIFY_GEMM == 2
    // ---- register-tiled symmetric-square GEMM: T = D·D, D symmetric ----
    // Ported from gpu_gemm.cl gemm_regtile/GEMM_SQ (measured best
    // 22×11thr × 8×4regs = 88×88 tile, 6.7 TFLOPS at n=86/b400). One
    // workgroup covers the whole output; As_t[kk][m] transposed staging
    // serves BOTH operands: B[kk][j] = A[j][kk] = As_t[kk][j].
    // lsz MUST equal PURIFY_TX*PURIFY_TY (host-enforced).
    __local float As_t[PURIFY_TK * PURIFY_WM];
    const int tx = lid % PURIFY_TX;
    const int ty = lid / PURIFY_TX;
    float acc[PURIFY_RTY][PURIFY_RTX];
    for (int i = 0; i < PURIFY_RTY; ++i)
        for (int j = 0; j < PURIFY_RTX; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < n; k0 += PURIFY_TK) {
        // stage A[0..WM)[k0..k0+TK) transposed — zero-padded beyond n
        for (int e = lid; e < PURIFY_TK * PURIFY_WM; e += lsz) {
            const int m = e / PURIFY_TK, kk = e - m * PURIFY_TK;
            const int gk = k0 + kk;
            As_t[kk * PURIFY_WM + m] = (m < n && gk < n) ? A[m * n + gk] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int kend = min(PURIFY_TK, n - k0);
        for (int kk = 0; kk < kend; ++kk) {
#if PURIFY_RTY == 8
            const float8 av = vload8(0, &As_t[kk * PURIFY_WM + ty * PURIFY_RTY]);
            const float af[PURIFY_RTY] = {av.s0, av.s1, av.s2, av.s3,
                                          av.s4, av.s5, av.s6, av.s7};
#elif PURIFY_RTY == 4
            const float4 av = vload4(0, &As_t[kk * PURIFY_WM + ty * PURIFY_RTY]);
            const float af[PURIFY_RTY] = {av.s0, av.s1, av.s2, av.s3};
#else
            float af[PURIFY_RTY];
            for (int i = 0; i < PURIFY_RTY; ++i)
                af[i] = As_t[kk * PURIFY_WM + ty * PURIFY_RTY + i];
#endif
#if PURIFY_RTX == 8
            const float8 bv = vload8(0, &As_t[kk * PURIFY_WM + tx * PURIFY_RTX]);
            const float bf[PURIFY_RTX] = {bv.s0, bv.s1, bv.s2, bv.s3,
                                          bv.s4, bv.s5, bv.s6, bv.s7};
#elif PURIFY_RTX == 4
            const float4 bv = vload4(0, &As_t[kk * PURIFY_WM + tx * PURIFY_RTX]);
            const float bf[PURIFY_RTX] = {bv.s0, bv.s1, bv.s2, bv.s3};
#else
            float bf[PURIFY_RTX];
            for (int j = 0; j < PURIFY_RTX; ++j)
                bf[j] = As_t[kk * PURIFY_WM + tx * PURIFY_RTX + j];
#endif
            for (int i = 0; i < PURIFY_RTY; ++i)
                for (int j = 0; j < PURIFY_RTX; ++j)
                    acc[i][j] = fma(af[i], bf[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // accumulate ‖T−D‖² only — T stays in acc[] registers across the
    // err reduction; the C write happens in the fused pass below where
    // the branch/freeze decision is already known (saves a full global
    // T write + re-read per iteration).
    for (int i = 0; i < PURIFY_RTY; ++i) {
        const int gi = ty * PURIFY_RTY + i;
        if (gi < n) {
            for (int j = 0; j < PURIFY_RTX; ++j) {
                const int gj = tx * PURIFY_RTX + j;
                if (gj < n) {
                    const float dd = acc[i][j] - A[gi * n + gj];
                    er += dd * dd;
                }
            }
        }
    }
#elif PURIFY_TILE > 0
    // ---- cooperative tiled GEMM: T = D·D via local tiles ----
    __local float tA[PURIFY_TILE * PURIFY_TILE];
    __local float tB[PURIFY_TILE * PURIFY_TILE];
    const int nt = (n + PURIFY_TILE - 1) / PURIFY_TILE;
    for (int bi = 0; bi < nt; ++bi) {
        for (int bj = 0; bj < nt; ++bj) {
            // each thread owns PURIFY_TILE²/lsz output elements of the tile
            for (int e = lid; e < PURIFY_TILE * PURIFY_TILE; e += lsz) {
                const int r = e / PURIFY_TILE, c = e - r * PURIFY_TILE;
                float acc = 0.0f;
                for (int kt = 0; kt < nt; ++kt) {
                    // cooperative tile loads (zero-padded beyond n)
                    for (int q = lid; q < PURIFY_TILE * PURIFY_TILE; q += lsz) {
                        const int qr = q / PURIFY_TILE, qc = q - qr * PURIFY_TILE;
                        const int gi = bi * PURIFY_TILE + qr, gk = kt * PURIFY_TILE + qc;
                        const int gj = bj * PURIFY_TILE + qc;
                        tA[q] = (gi < n && gk < n) ? A[gi * n + gk] : 0.0f;
                        tB[q] = (gk < n && gj < n) ? A[gk * n + gj] : 0.0f;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    for (int t = 0; t < PURIFY_TILE; ++t) acc += tA[r * PURIFY_TILE + t] * tB[t * PURIFY_TILE + c];
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                const int gi = bi * PURIFY_TILE + r, gj = bj * PURIFY_TILE + c;
                if (gi < n && gj < n) {
                    C[gi * n + gj] = acc;
                    const float dd = acc - A[gi * n + gj];
                    er += dd * dd;
                }
            }
        }
    }
#else
    // ---- row·row variant: T_ij = row_i · row_j (D symmetric ⇒ D·D = D·Dᵀ) ----
    for (int idx = lid; idx < n * n; idx += lsz) {
        const int i = idx / n;
        const int j = idx - i * n;
        __global const float* ri = A + i * n;
        __global const float* rj = A + j * n;
        float acc = 0.0f;
        for (int k = 0; k < n; ++k) acc += ri[k] * rj[k];
        C[idx] = acc;
        const float dd = acc - A[idx];
        er += dd * dd;
    }
#endif

    er = sqrt(pur_reduce(red, er, lid, lsz));
    if (lid == 0) {
        errs[sys] = er;
        l_trd = traces[sys];              // Tr(Din) — branch decided on device
        // freeze: Din already converged → mirror it unchanged (device-side,
        // catches convergence mid-chunk before f32 dust regrows past tol)
        const float tol_tr = fmax(2e-5f * nocc[sys], 1e-4f);
        l_frozen = (er < tol && fabs(l_trd - nocc[sys]) <= tol_tr);
        done[sys] = l_frozen;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int contract = (l_trd > nocc[sys]);
    float trn = 0.0f;
#if PURIFY_GEMM == 2
    // fused write — acc[] registers still hold T_ij; apply branch/freeze
    // here (no raw-T global round-trip).
    for (int i = 0; i < PURIFY_RTY; ++i) {
        const int gi = ty * PURIFY_RTY + i;
        if (gi < n) {
            for (int j = 0; j < PURIFY_RTX; ++j) {
                const int gj = tx * PURIFY_RTX + j;
                if (gj < n) {
                    const float v = l_frozen ? A[gi * n + gj]
                        : (contract ? acc[i][j] : (2.0f * A[gi * n + gj] - acc[i][j]));
                    C[gi * n + gj] = v;
                    if (gi == gj) trn += v;
                }
            }
        }
    }
#else
    for (int idx = lid; idx < n * n; idx += lsz) {
        const float t = C[idx];           // T_ij written in pass 1
        const float v = l_frozen ? A[idx] : (contract ? t : (2.0f * A[idx] - t));
        C[idx] = v;
        if (idx / n == idx - (idx / n) * n) trn += v;
    }
#endif
    trn = pur_reduce(red, trn, lid, lsz);
    if (lid == 0) traces[sys] = trn;
}

// ------------------------------------------------------------------
// dmm_update_batched — one DMM commutator-descent step (orthogonal
// basis, S=I — the dense path works on H′ = XᵀH_sccX so the sparse
// dmm_descend simplifies):
//     K ← ½(K + Kᵀ) − s·(T + Tᵀ − 2Y),   T = H′·K,  Y = K·T
// The gradient (I−K)H′K + KH′(I−K) is the occ–virt coupling block of
// the commutator — because the update MULTIPLIES K by H′ it rotates the
// occupied subspace downhill on Tr(KH′), which no polynomial in K can
// do (TC2 preserves eigenvectors). s = η/Δε with Δε the Gershgorin
// spectral span of H′.
// Pair-owner scheme: one thread per upper-triangle cell reads BOTH
// (i,j) and (j,i) before writing either → race-free in-place
// symmetrization (matches sparse's explicit symmetrize each step).
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// spec_span_batched — Gershgorin spectral span of H′ per replica
// (same row-|·|-sum bounds as tc2_init_batched). Needed every warm
// solve: the DMM step scale is s = eta / Δε per replica.
// ------------------------------------------------------------------
__kernel void spec_span_batched(
    __global const float* H,      // [batch][n*n] — H̃ (orthogonal basis)
    __global float*       spans,  // [batch]      — out: λmax − λmin
    const int n,
    const int batch)
{
    const int sys = get_group_id(0);
    if (sys >= batch) return;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __global const float* h = H + (size_t)sys * n * n;
    __local float red[PURIFY_WG];

    float rmax = -1.0e30f, rmin = 1.0e30f;
    for (int i = lid; i < n; i += lsz) {
        float s = 0.0f;
        for (int j = 0; j < n; ++j) s += fabs(h[i * n + j]);
        const float dg = h[i * n + i];
        rmax = fmax(rmax, dg + (s - fabs(dg)));
        rmin = fmin(rmin, dg - (s - fabs(dg)));
    }
    const float lmax = pur_reduce_max(red, rmax, lid, lsz);
    const float lmin = pur_reduce_min(red, rmin, lid, lsz);
    if (lid == 0) spans[sys] = fmax(lmax - lmin, 1.0e-30f);
}

__kernel void dmm_update_batched(
    __global float*       K,     // [batch][n*n] — in/out projector seed
    __global const float* T,     // [batch][n*n] — H′·K
    __global const float* Y,     // [batch][n*n] — K·T
    __global const float* spans, // [batch]      — Δε per replica
    const float eta,             // η_scale (sparse used ≈8)
    const float cap,             // trust region: ‖δK‖_F ≤ cap per step
    const int n,
    const int batch)
{
    const int sys = get_group_id(0);
    if (sys >= batch) return;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __global float*       kg = K + (size_t)sys * n * n;
    __global const float* tg = T + (size_t)sys * n * n;
    __global const float* yg = Y + (size_t)sys * n * n;
    __local float red[PURIFY_WG];

    const int ntri = n * (n + 1) / 2;
    const float two_n1 = 2.0f * n + 1.0f;

    // Pass 1 — ‖G‖_F of the symmetric gradient G = T+Tᵀ−2Y (diag×1,
    // off-diag×2). The sparse η=8/Δε step was tuned on geometry-
    // perturbation ΔH (tiny G); inside SCC the first iterations have
    // O(1) charge swings → O(1) gradients → the same step overshoots
    // to NaN (measured: comm 5e-7 → 0.72). Trust-region: s_eff =
    // min(η/Δε, cap/‖G‖) — normalized descent far from the manifold,
    // the sparse scaling near it.
    float g2 = 0.0f;
    for (int e = lid; e < ntri; e += lsz) {
        const int i = (int)floor(
            (two_n1 - sqrt(fmax(two_n1 * two_n1 - 8.0f * e, 0.0f))) * 0.5f);
        const int j = i + (e - i * (2 * n - i + 1) / 2);
        const float gu = tg[i * n + j] + tg[j * n + i] - 2.0f * yg[i * n + j];
        g2 += (j > i ? 2.0f : 1.0f) * gu * gu;
    }
    const float gn = sqrt(pur_reduce(red, g2, lid, lsz));
    const float s = fmin(eta / spans[sys], cap / fmax(gn, 1.0e-30f));

    // Pass 2 — K ← ½(K+Kᵀ) − s·G. Pair-owner scheme: one thread per
    // upper-triangle cell reads BOTH (i,j) and (j,i) before writing
    // either → race-free in-place symmetrization (matches sparse's
    // explicit symmetrize each step).
    for (int e = lid; e < ntri; e += lsz) {
        const int i = (int)floor(
            (two_n1 - sqrt(fmax(two_n1 * two_n1 - 8.0f * e, 0.0f))) * 0.5f);
        const int j = i + (e - i * (2 * n - i + 1) / 2);
        const float ks = 0.5f * (kg[i * n + j] + kg[j * n + i]);
        const float gu = tg[i * n + j] + tg[j * n + i] - 2.0f * yg[i * n + j];
        const float gl = tg[j * n + i] + tg[i * n + j] - 2.0f * yg[j * n + i];
        kg[i * n + j] = ks - s * gu;
        kg[j * n + i] = ks - s * gl;
    }
}

// ------------------------------------------------------------------
// mcweeny_combine_batched — McWeeny retraction applied after the two
// GEMMs T2 = K·K and B = T2·K have been computed into scratch:
//     K ← 3·T2 − 2·B     (= 3K² − 2K³)
// Pulls the DMM-rotated seed back to the idempotent manifold — the
// retract the sparse recipe applies every ~2 descent steps. Same
// pair-owner symmetrization as dmm_update_batched.
// ------------------------------------------------------------------
__kernel void mcweeny_combine_batched(
    __global float*       K,   // [batch][n*n] — out
    __global const float* A,   // [batch][n*n] — K²
    __global const float* B,   // [batch][n*n] — K²·K
    const int n,
    const int batch)
{
    const int sys = get_group_id(0);
    if (sys >= batch) return;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __global float*       kg = K + (size_t)sys * n * n;
    __global const float* ag = A + (size_t)sys * n * n;
    __global const float* bg = B + (size_t)sys * n * n;

    const int ntri = n * (n + 1) / 2;
    const float two_n1 = 2.0f * n + 1.0f;
    for (int e = lid; e < ntri; e += lsz) {
        const int i = (int)floor(
            (two_n1 - sqrt(fmax(two_n1 * two_n1 - 8.0f * e, 0.0f))) * 0.5f);
        const int j = i + (e - i * (2 * n - i + 1) / 2);
        const float vu = 3.0f * ag[i * n + j] - 2.0f * bg[i * n + j];
        const float vl = 3.0f * ag[j * n + i] - 2.0f * bg[j * n + i];
        const float v = 0.5f * (vu + vl);
        kg[i * n + j] = v;
        kg[j * n + i] = v;
    }
}

// ------------------------------------------------------------------
// comm_gate_batched — the R_H stationarity certificate (same quantity
// as sparse rh_stationarity, orthogonal basis):
//     rh[sys] = ‖T − Tᵀ‖_F / (2‖T‖_F),   T = H′·K (post-update)
// For the true occupied projector comm = 0; an idempotent rank-nocc
// projector on a WRONG subspace passes trace/idempotency but fails this
// — it is the gate that detects the spurious fixed point measured with
// the δK0 seed (comm=0.238). T must be recomputed on the FINAL K.
// ------------------------------------------------------------------
__kernel void comm_gate_batched(
    __global const float* T,   // [batch][n*n] — H′·K final
    __global float*       rh,  // [batch]      — out
    const int n,
    const int batch)
{
    const int sys = get_group_id(0);
    if (sys >= batch) return;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __global const float* tg = T + (size_t)sys * n * n;
    __local float red[PURIFY_WG];

    float na = 0.0f, nd = 0.0f;
    for (int e = lid; e < n * n; e += lsz) {
        const int i = e / n;
        const float t = tg[e];
        const float d = t - tg[(e - i * n) * n + i];   // T[j,i]
        na += t * t;
        nd += d * d;
    }
    na = pur_reduce(red, na, lid, lsz);
    nd = pur_reduce(red, nd, lid, lsz);
    if (lid == 0) rh[sys] = sqrt(nd) / (2.0f * sqrt(na) + 1.0e-30f);
}
