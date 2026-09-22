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
#define PURIFY_TILELOOP 0
#endif
#define PURIFY_WM (PURIFY_TY * PURIFY_RTY)
#define PURIFY_WN (PURIFY_TX * PURIFY_RTX)
// relax_step_batched: retraction count — 1 = loose mode (single
// quadratic map per step), 2 = proper retraction pair q∓∘q±.
#ifndef PURIFY_NPUR
#define PURIFY_NPUR 2
#endif

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
    const int batch,
    __global const int* done)     // [batch] — 1 = skip (warm fallback keeps certified K)
{
    const int sys = get_group_id(0);
    if (done[sys]) return;
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
// Niklasson):  Out = K1 + γ·(K1 − K2), where K1,K2 are the converged
// projectors of the previous two iterations. A converged projector is a
// HARD fixed point of any polynomial map (it never rotates the
// subspace), so re-purifying K1 is a no-op — but the difference K1−K2
// carries the ACTUAL occupied-subspace motion: for a small rotation
// K2 = e^R·K1·e^{-R} ≈ K1+[R,K1], the seed K1−γ[R,K1] has eigenvectors
// rotated by γR to first order. Follow with a McWeeny fold 3S²−2S³
// (R'(0)=R'(1)=0 — kills the O(h²) radial overshoot, DR[tangent]=id
// preserves the predicted rotation). NO Gershgorin renorm here: bounds
// of a dense projector are ~[−4,5] even for an exact {0,1} spectrum, so
// the old rescale always fired and destroyed the seed's near-
// idempotency (mapped λ→0.44/0.56) — the measured "extrapolation
// drift" was this bug + TC2 branch amplification, not the idea.
// ------------------------------------------------------------------
__kernel void tc2_extrap_batched(
    __global float*       Out, // [batch][n*n] — out: warm seed
    __global const float* H1,  // [batch][n*n] — latest converged projector
    __global const float* H2,  // [batch][n*n] — previous converged projector
    const float           gamma, // extrapolation coefficient (1 = 2K1−K2)
    const int n,
    const int batch)
{
    const int sys = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __global float*       o  = Out + (size_t)sys * n * n;
    __global const float* h1 = H1  + (size_t)sys * n * n;
    __global const float* h2 = H2  + (size_t)sys * n * n;
    for (int idx = lid; idx < n * n; idx += lsz)
        o[idx] = h1[idx] + gamma * (h1[idx] - h2[idx]);
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
    const int batch,
    const int             mirror_done) // 1 = copy Din→Dout for frozen systems
                                       // (ping-pong parity); 0 = skip entirely —
                                       // used when the caller guarantees an even
                                       // step count so all results end on the
                                       // same side (production warm fallback).
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
        if (mirror_done) {
            for (int idx = lid; idx < n * n; idx += lsz) C[idx] = A[idx];
        }
        return;
    }

#if PURIFY_GEMM == 2 && PURIFY_TILELOOP
    // n does not fit in one ≤1024-thread full-cover square (n=246 would
    // be 1922 threads at the 4×8 micro-tile, and 961 threads at 8×8
    // spills the register file: CL_OUT_OF_RESOURCES). Same 11×11 × 8×8
    // tile as the warm GEMM: 121 threads, both operands staged, the
    // workgroup walks 88×88 output tiles. T is written to C; the branch
    // below reads it back.
    __local float As_t[PURIFY_TK * PURIFY_WM];
    __local float Bs[PURIFY_TK * PURIFY_WN];
    const int tx = lid % PURIFY_TX;
    const int ty = lid / PURIFY_TX;
    for (int m0 = 0; m0 < n; m0 += PURIFY_WM) {
        for (int n0 = 0; n0 < n; n0 += PURIFY_WN) {
            float acc[PURIFY_RTY][PURIFY_RTX];
            for (int i = 0; i < PURIFY_RTY; ++i)
                for (int j = 0; j < PURIFY_RTX; ++j) acc[i][j] = 0.0f;
            for (int k0 = 0; k0 < n; k0 += PURIFY_TK) {
                for (int e = lid; e < PURIFY_TK * PURIFY_WM; e += lsz) {
                    const int r = e / PURIFY_TK, kk = e - r * PURIFY_TK;
                    const int gr = m0 + r, gk = k0 + kk;
                    As_t[kk * PURIFY_WM + r] = (gr < n && gk < n) ? A[gr * n + gk] : 0.0f;
                }
                for (int e = lid; e < PURIFY_TK * PURIFY_WN; e += lsz) {
                    const int kk = e / PURIFY_WN, c = e - kk * PURIFY_WN;
                    const int gk = k0 + kk, gc = n0 + c;
                    Bs[kk * PURIFY_WN + c] = (gk < n && gc < n) ? A[gk * n + gc] : 0.0f;
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
                    const float8 bv = vload8(0, &Bs[kk * PURIFY_WN + tx * PURIFY_RTX]);
                    const float bf[PURIFY_RTX] = {bv.s0, bv.s1, bv.s2, bv.s3,
                                                  bv.s4, bv.s5, bv.s6, bv.s7};
#elif PURIFY_RTX == 4
                    const float4 bv = vload4(0, &Bs[kk * PURIFY_WN + tx * PURIFY_RTX]);
                    const float bf[PURIFY_RTX] = {bv.s0, bv.s1, bv.s2, bv.s3};
#else
                    float bf[PURIFY_RTX];
                    for (int j = 0; j < PURIFY_RTX; ++j)
                        bf[j] = Bs[kk * PURIFY_WN + tx * PURIFY_RTX + j];
#endif
                    for (int i = 0; i < PURIFY_RTY; ++i)
                        for (int j = 0; j < PURIFY_RTX; ++j)
                            acc[i][j] = fma(af[i], bf[j], acc[i][j]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            for (int i = 0; i < PURIFY_RTY; ++i) {
                const int gi = m0 + ty * PURIFY_RTY + i;
                if (gi < n) {
                    for (int j = 0; j < PURIFY_RTX; ++j) {
                        const int gj = n0 + tx * PURIFY_RTX + j;
                        if (gj < n) {
                            C[gi * n + gj] = acc[i][j];
                            const float dd = acc[i][j] - A[gi * n + gj];
                            er += dd * dd;
                        }
                    }
                }
            }
        }
    }
#elif PURIFY_GEMM == 2
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
#if PURIFY_GEMM == 2 && !PURIFY_TILELOOP
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
    const int batch,
    __global const int* pend)    // [batch] — 1 = still unconverged (0 = frozen)
{
    const int sys = get_group_id(0);
    if (sys >= batch || !pend[sys]) return;
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
    const int batch,
    __global const int* pend) // [batch] — 1 = still unconverged (0 = frozen)
{
    const int sys = get_group_id(0);
    if (sys >= batch || !pend[sys]) return;
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

// float4 workgroup reduce — same fold-then-halve semantics as
// pur_reduce (ceil-half fold, correct for non-PoT lsz, trailing
// barrier before return).
static inline float4 pur_reduce4(__local float4* red, float4 v, const int lid, const int lsz) {
    red[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz; s > 1; s = (s + 1) >> 1) {
        const int h = (s + 1) >> 1;
        if (lid < s - h) red[lid] += red[lid + h];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float4 out = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return out;
}

// ------------------------------------------------------------------
// relax_step_batched — fused energy-descent + constraint retraction:
// purification is NOT the solver here, it is the retraction of a
// density-matrix energy minimizer min_D Tr(D·H) (design: chat doc §17,
// "learned constraint force" scheme). One step:
//
//   F0 = H − Λ                    Λ = learned constraint force (≈N, the
//                                 part of H forbidden by the projector
//                                 constraint; → H at the fixed point)
//   F  = F0 − a·I − b·D           b = LS removal of the D-mode
//                                 (⟨D,F⟩ = 0); a is NOT the LS value —
//                                 it pins Tr(X) = nocc exactly, i.e.
//                                 the I-component is the particle-number
//                                 feedback channel (chemical potential).
//                                 This prevents the wrong-rank clean-
//                                 projector fixed point of the bare
//                                 scheme: a stalled Tr error keeps
//                                 injecting a global shift every step.
//   X  = D − αF = cD·D − α(H−Λ) + αa·I     (cD = 1+αb, α = η/Δε)
//   D' = q∓(q±(X))                1 (PURIFY_NPUR=1) or 2 (NPUR=2)
//                                 symmetric-square retractions
//   Λ += β(D'−X)/α                constraint force learned from what
//                                 the retraction had to remove (fused
//                                 elementwise into the write passes).
//                                 Gauge canonicalization (Λ += aI+bD ⇒
//                                 Λ ≈ H) is done ONCE per warm solve by
//                                 relax_canon_batched — per-step folding
//                                 degenerates the {I,D} solve and lands
//                                 on excited projectors (measured).
//
// Retraction pair (TC2 quadratic maps): q₋(X)=X², q₊(X)=2X−X². The
// composed pair has R'(0)=R'(1)=0 — a proper retraction killing the
// first-order normal error of the raw step. Branch of the FIRST map is
// the TC2 trace steer: Tr(X) ≤ nocc → expand first (q₊), else contract
// (q₋); the second map (NPUR=2) is always the complement.
//
// In-place layout (no ping-pong, no second n² scratch):
//   * X is constructed ON THE FLY while staging each K-slab of the
//     first symmetric square — every real element of D is staged
//     exactly once, so overwriting D←X during staging is race-free and
//     the square needs only the local slab.
//   * write pass applies the quadratic combine to the register acc
//     (no raw-T global round-trip), updates Λ, leaves D = Y.
//   * NPUR=2: WG global-fence barrier, then the SAME slab machinery
//     squares D=Y in place → D = Z.
// Requires lsz = PURIFY_TX·PURIFY_TY (regtile interior — host-enforced).
//
// Diagnostics per launch → diag[4*sys + {0,1,2,3}]:
//   step = ‖X−D‖_F = α·‖F_proj‖    residual force → 0 at the fixed pt
//   corr = retraction work         (NPUR1: ‖Y−X‖; NPUR2: √(‖Y−X‖²+‖Z−Y‖²))
//   tr   = Tr(D')
//   er   = ‖M²−M‖ of the last pre-combination square (idem defect of
//          the intermediate — the post-combination D' is O(·²) better)
// done[sys]: in/out freeze flag — set when step,corr < tol and trace
// ok; a done WG early-returns (in-place D needs no mirror). Unlike the
// TC2 fixed point this map's fixed point is *attracting* (F→0), so
// freeze is an optimization, not a stability requirement.
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// relax_canon_batched — canonicalize the learned constraint force once
// at a warm-solve boundary. At a converged fixed point
//     Λ = H − aI − bD  (gauge freedom: any aI+bD shift is invisible
// to the dynamics). Carried into the next solve, the b·D_old term is a
// FIXED matrix that becomes tangent as D rotates toward the new
// solution — it biases the fixed point (measured: comm ≈ b·‖ΔD‖).
// Folding the {I,D}-components of F0 back into Λ removes that gauge:
//     Λ += a_ls·I + b_ls·D   ⇒   F0' = H−Λ = F (the projected residual)
// so the next solve starts from F0 ≈ ΔH with Λ ≈ H and no stale
// tangent content. One launch per solve, never inside the loop.
// ------------------------------------------------------------------
__kernel void relax_canon_batched(
    __global const float* D,      // [batch][n*n]
    __global const float* H,      // [batch][n*n]
    __global float*       Lambda, // [batch][n*n] — in/out
    const int n,
    const int batch)
{
    const int sys = get_group_id(0);
    if (sys >= batch) return;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int nn = n * n;
    __global const float* Dg = D      + (size_t)sys * nn;
    __global const float* Hg = H      + (size_t)sys * nn;
    __global float*       Lg = Lambda + (size_t)sys * nn;
    __local float4 red4[PURIFY_WG];
    __local float l_a, l_b;

    float s1 = 0.0f, s2 = 0.0f, h0 = 0.0f, h1 = 0.0f;
    for (int idx = lid; idx < nn; idx += lsz) {
        const int i = idx / n;
        const float d = Dg[idx];
        const float f0 = Hg[idx] - Lg[idx];
        s2 = fma(d, d, s2);
        h1 = fma(d, f0, h1);
        if (i == idx - i * n) { s1 += d; h0 += f0; }
    }
    {
        const float4 rr = pur_reduce4(red4, (float4)(s1, s2, h0, h1), lid, lsz);
        if (lid == 0) {
            const float det = (float)n * rr.y - rr.x * rr.x;
            float a = rr.z / (float)n, b = 0.0f;
            if (fabs(det) > 1.0e-20f) {
                a = (rr.z * rr.y - rr.w * rr.x) / det;
                b = ((float)n * rr.w - rr.x * rr.z) / det;
            }
            l_a = a; l_b = b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int idx = lid; idx < nn; idx += lsz) {
        const int i = idx / n;
        Lg[idx] += l_b * Dg[idx] + (i == idx - i * n ? l_a : 0.0f);
    }
}

__kernel void relax_step_batched(
    __global float*       D,      // [batch][n*n] — in/out density (ortho basis)
    __global const float* H,      // [batch][n*n] — H̃ (orthogonal basis)
    __global float*       Lambda, // [batch][n*n] — in/out learned constraint force
    __global const float* nocc,   // [batch]      — Tr target
    __global const float* spans,  // [batch]      — Gershgorin Δε of H
    __global float*       diag,   // [batch*4]    — {step, corr, tr, er}
    __global int*         done,   // [batch]      — in/out freeze flags
    const float           eta,    // α = η/Δε (stability: α·Δε ≲ 2)
    const float           beta,   // Λ learning rate (~0.3–0.7)
    const float           tol,    // freeze threshold on step/corr
    const float           ldecay, // Λ forgetting rate per step (0..1):
                                  // increments β(Y−X)/α deposit block-
                                  // diagonal content in the CURRENT D
                                  // frame — as D rotates, old deposits
                                  // become stale *tangent* content that
                                  // no later increment can remove →
                                  // biased fixed point [D,H−Λ_tan]=0.
                                  // ldecay>0 lets it wash out. Drive it
                                  // ∝ residual: 0 near convergence (the
                                  // true fixed point is unaffected).
    const int n,
    const int batch)
{
    const int sys = get_group_id(0);
    if (sys >= batch) return;
    if (done[sys]) return;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int tx = lid % PURIFY_TX;
    const int ty = lid / PURIFY_TX;
    const int nn = n * n;
    __global float*       Dg = D      + (size_t)sys * nn;
    __global const float* Hg = H      + (size_t)sys * nn;
    __global float*       Lg = Lambda + (size_t)sys * nn;

    __local float  As_t[PURIFY_TK * PURIFY_WM];
    __local float4 red4[PURIFY_WG];
    __local float  l_a, l_b, l_cD, l_alpha;
    __local int    l_expand1;

    // ---- pass 0: D-mode removal + trace pin -----------------------
    // b = LS coefficient killing the D-parallel (scaling) mode of F0
    //     (⟨D,F⟩ = 0). For an exact rank-m projector det = m(n−m) —
    //     well conditioned at half filling.
    // a = NOT the LS coefficient — chosen so the trial lands exactly on
    //     the particle-number plane:
    //         Tr(X) = cD·s1 − α·h0 + α·a·n  :=  nocc
    //     ⇒ a = (nocc − s1 − α·b·s1 + α·h0)/(α·n).
    //     The residual I-component of F is then the integral feedback
    //     on Tr(D) — a clean wrong-rank projector (fixed point of both
    //     quadratic folds with F = 0 under the LS gauge) can no longer
    //     stall: each step keeps injecting a global shift that the
    //     expand fold amplifies near the 0.5 boundary until the rank
    //     is restored.
    float s1 = 0.0f, s2 = 0.0f, h0 = 0.0f, h1 = 0.0f;
    for (int idx = lid; idx < nn; idx += lsz) {
        const int i = idx / n;
        const float d = Dg[idx];
        // Λ forgetting must happen HERE, before the a,b/pin reductions:
        // decaying it later (in the staging loop) would make X use a
        // different force than the coefficients were computed from and
        // the Tr(X)=nocc pin would no longer hold (measured bug).
        // Each element is visited exactly once in this pass → race-free.
        float lam = Lg[idx];
        if (ldecay > 0.0f) { lam *= 1.0f - ldecay; Lg[idx] = lam; }
        const float f0 = Hg[idx] - lam;
        s2 = fma(d, d, s2);
        h1 = fma(d, f0, h1);
        if (i == idx - i * n) { s1 += d; h0 += f0; }
    }
    {
        const float4 rr = pur_reduce4(red4, (float4)(s1, s2, h0, h1), lid, lsz);
        if (lid == 0) {
            const float alpha = eta / spans[sys];
            const float noc = nocc[sys];
            const float det = (float)n * rr.y - rr.x * rr.x;
            const float b = (fabs(det) > 1.0e-20f)
                ? ((float)n * rr.w - rr.x * rr.z) / det
                : 0.0f;      // degenerate D ∝ I: no D-mode to remove
            // pin Tr(X) = nocc (exact, by construction of X)
            const float a = (noc - rr.x - alpha * (b * rr.x - rr.z))
                            / (alpha * (float)n);
            l_a = a;
            l_b = b;
            l_cD = fma(alpha, b, 1.0f);
            l_alpha = alpha;
            // underfilled → expand first (q₊), overfilled → contract
            l_expand1 = (rr.x <= noc);
        }
        // GLOBAL fence: decayed-Λ writes (pass 0, other threads) must be
        // visible to the staging reads of Lg below.
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    const float alpha = l_alpha;

    // ---- square 1: acc = X²; D converted to X during staging -------
    float acc[PURIFY_RTY][PURIFY_RTX];
    for (int i = 0; i < PURIFY_RTY; ++i)
        for (int j = 0; j < PURIFY_RTX; ++j) acc[i][j] = 0.0f;
    float step2 = 0.0f;
    for (int k0 = 0; k0 < n; k0 += PURIFY_TK) {
        // stage X[0..WM)[k0..k0+TK) transposed; D[m][k] → X[m][k] in
        // place — each real element is staged exactly once (race-free).
        for (int e = lid; e < PURIFY_TK * PURIFY_WM; e += lsz) {
            const int m = e / PURIFY_TK, kk = e - m * PURIFY_TK;
            const int gk = k0 + kk;
            float x = 0.0f;
            if (m < n && gk < n) {
                const int idx = m * n + gk;
                const float d = Dg[idx];
                const float f0 = Hg[idx] - Lg[idx];
                // NOTE: the aI+bD gauge must NOT be folded into Λ here
                // — folding it every step degenerates the {I,D} solve
                // (b→0, the D-rescaling term of X vanishes) and the
                // dynamics lands on excited eigenprojectors (measured
                // ΔE ≈ +43 on all 16 test systems). Canonicalization
                // happens ONCE per solve in relax_canon_batched.
                x = fma(l_cD, d, -alpha * f0);
                if (m == gk) x += alpha * l_a;
                const float dx = x - d;
                step2 = fma(dx, dx, step2);
                Dg[idx] = x;
            }
            As_t[kk * PURIFY_WM + m] = x;
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
    // all staging writes D→X must be visible to the whole WG before the
    // write pass reads back its own X elements.
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // ---- write pass 1: Y = q±(X), Λ update, diagnostics ------------
    float er1 = 0.0f, tr1 = 0.0f;
    for (int i = 0; i < PURIFY_RTY; ++i) {
        const int gi = ty * PURIFY_RTY + i;
        if (gi < n) {
            for (int j = 0; j < PURIFY_RTX; ++j) {
                const int gj = tx * PURIFY_RTX + j;
                if (gj < n) {
                    const int idx = gi * n + gj;
                    const float x = Dg[idx];
                    const float t = acc[i][j];
                    const float y = l_expand1 ? fma(2.0f, x, -t) : t;
                    const float dd = t - x;
                    er1 = fma(dd, dd, er1);
                    Lg[idx] += beta * (y - x) / alpha;
                    Dg[idx] = y;
                    if (gi == gj) tr1 += y;
                }
            }
        }
    }
    float4 rr = pur_reduce4(red4, (float4)(step2, er1, tr1, 0.0f), lid, lsz);
    const float stepn = sqrt(rr.x);
    const float corrn1 = sqrt(rr.y);

#if PURIFY_NPUR == 2
    // ---- square 2: acc = Y² on D=Y (re-staged) ---------------------
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int i = 0; i < PURIFY_RTY; ++i)
        for (int j = 0; j < PURIFY_RTX; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < n; k0 += PURIFY_TK) {
        for (int e = lid; e < PURIFY_TK * PURIFY_WM; e += lsz) {
            const int m = e / PURIFY_TK, kk = e - m * PURIFY_TK;
            const int gk = k0 + kk;
            As_t[kk * PURIFY_WM + m] = (m < n && gk < n) ? Dg[m * n + gk] : 0.0f;
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

    // ---- write pass 2: Z = q∓(Y) — the complementary branch --------
    const int expand2 = !l_expand1;
    float er2 = 0.0f, tr2 = 0.0f, co2 = 0.0f;
    for (int i = 0; i < PURIFY_RTY; ++i) {
        const int gi = ty * PURIFY_RTY + i;
        if (gi < n) {
            for (int j = 0; j < PURIFY_RTX; ++j) {
                const int gj = tx * PURIFY_RTX + j;
                if (gj < n) {
                    const int idx = gi * n + gj;
                    const float y = Dg[idx];
                    const float t = acc[i][j];
                    const float z = expand2 ? fma(2.0f, y, -t) : t;
                    const float dd = t - y;
                    const float dz = z - y;
                    er2 = fma(dd, dd, er2);
                    co2 = fma(dz, dz, co2);
                    Lg[idx] += beta * dz / alpha;
                    Dg[idx] = z;
                    if (gi == gj) tr2 += z;
                }
            }
        }
    }
    rr = pur_reduce4(red4, (float4)(co2, er2, tr2, 0.0f), lid, lsz);
    if (lid == 0) {
        const float corrn = sqrt(corrn1 * corrn1 + rr.x);
        const float tr = rr.z;
        diag[4 * sys + 0] = stepn;
        diag[4 * sys + 1] = corrn;
        diag[4 * sys + 2] = tr;
        diag[4 * sys + 3] = sqrt(rr.y);
        const float tol_tr = fmax(2e-5f * nocc[sys], 1e-4f);
        done[sys] = (stepn < tol && corrn < tol && fabs(tr - nocc[sys]) <= tol_tr);
    }
#else
    if (lid == 0) {
        const float tr = rr.z;
        diag[4 * sys + 0] = stepn;
        diag[4 * sys + 1] = corrn1;
        diag[4 * sys + 2] = tr;
        diag[4 * sys + 3] = corrn1;   // NPUR=1: corr ≡ ‖X²−X‖
        const float tol_tr = fmax(2e-5f * nocc[sys], 1e-4f);
        done[sys] = (stepn < tol && corrn1 < tol && fabs(tr - nocc[sys]) <= tol_tr);
    }
#endif
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
    const int batch,
    __global int*         done, // [batch] — out: 1 = certified (rh < tol)
    __global int*         pend, // [batch] — out: cleared on certification —
                                //             the batched_gemm_active freeze mask
    const float           tol)  // rh acceptance threshold
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
    if (lid == 0) {
        const float r = sqrt(nd) / (2.0f * sqrt(na) + 1.0e-30f);
        rh[sys] = r;
        // Unlatched write: a replica whose K regressed (or was rebuilt by
        // the cold fallback since the last cert) must re-certify.
        done[sys] = (r < tol) ? 1 : 0;
        if (r < tol) pend[sys] = 0;
    }
}

// ------------------------------------------------------------------
// gemm_nn_batched — occasional-use GENERAL product T = A·B (the
// regtile square interior only computes A·A). Naive row·col dots, one
// WG per system; launched once per solve to build T = H·D for the
// comm_gate stationarity certificate — never inside the iteration
// loop, so no tile machinery is warranted.
// ------------------------------------------------------------------
__kernel void gemm_nn_batched(
    __global const float* A,   // [batch][n*n]
    __global const float* B,   // [batch][n*n]
    __global float*       T,   // [batch][n*n] — out
    const int n,
    const int batch)
{
    const int sys = get_group_id(0);
    if (sys >= batch) return;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __global const float* ag = A + (size_t)sys * n * n;
    __global const float* bg = B + (size_t)sys * n * n;
    __global float*       tg = T + (size_t)sys * n * n;
    for (int idx = lid; idx < n * n; idx += lsz) {
        const int i = idx / n;
        const int j = idx - i * n;
        float s = 0.0f;
        for (int k = 0; k < n; ++k) s = fma(ag[i * n + k], bg[k * n + j], s);
        tg[idx] = s;
    }
}
