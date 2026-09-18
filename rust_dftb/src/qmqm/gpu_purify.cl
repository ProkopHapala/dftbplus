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
// GEMM interior — PURIFY_TILE selects the variant (identical results):
//   0  row·row: D symmetric ⇒ T = D·Dᵀ ⇒ T_ij = row_i·row_j — coalesced
//      86-fma dots, ZERO local memory and ZERO barriers in the GEMM
//      (D is ~30 KB → L1-resident). Max occupancy (threads-capped).
//  >0  cooperative tiled GEMM, PURIFY_TILE×PURIFY_TILE local tiles
//      (16 → 2 KB, 32 → 8 KB) — for A/B against variant R.
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

// fold-then-halve workgroup reduce — correct for arbitrary (non-PoT) lsz.
// The trailing barrier before `return` is REQUIRED: without it a fast
// thread can enter the NEXT reduce call and overwrite red[0] before a
// straggler has read the result (observed as random corrupted bounds/
// traces at large batch — race, nondeterministic victim).
static inline float pur_reduce(__local float* red, float v, const int lid, const int lsz) {
    red[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz; s > 1; s = (s + 1) >> 1) {
        const int h = s >> 1;
        if (lid < h && lid + h < s) red[lid] += red[lid + h];
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
        const int h = s >> 1;
        if (lid < h && lid + h < s) red[lid] = fmin(red[lid], red[lid + h]);
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
        const int h = s >> 1;
        if (lid < h && lid + h < s) red[lid] = fmax(red[lid], red[lid + h]);
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

#if PURIFY_TILE > 0
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
    for (int idx = lid; idx < n * n; idx += lsz) {
        const float t = C[idx];           // T_ij written in pass 1
        const float v = l_frozen ? A[idx] : (contract ? t : (2.0f * A[idx] - t));
        C[idx] = v;
        if (idx / n == idx - (idx / n) * n) trn += v;
    }
    trn = pur_reduce(red, trn, lid, lsz);
    if (lid == 0) traces[sys] = trn;
}
