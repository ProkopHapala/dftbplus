// gpu_gemm.cl — batched dense GEMM variants for the FOE/purify path.
//
// C[s] = A[s]·B[s] for s = 0..batch-1, row-major n×n, one contiguous
// [batch][n*n] buffer per matrix. No transpose/alpha/beta — the purify
// interior needs exactly C = A·B; generality lives in batched_gemm
// (gpu_matrix_ops.cl), which is the tile-per-WG baseline.
//
// Variants (one system per workgroup unless noted):
//   gemm_1elem    — 1 thread per output element, global row·col dot.
//                   The floor: ~2 global loads per FMA.
//   gemm_regtile  — register-tiled schoolbook: TX×TY thread grid, each
//                   thread owns an RTY×RTX register micro-tile; WG tile
//                   = (TY·RTY)×(TX·RTX); A/B staged through small
//                   __local k-tiles (masked at boundaries). User spec:
//                   8×8 tiles on 64 threads → covers 64×64 per pass.
//                   RTX=RTY=1 degenerates to the classic 1-elem tiled.
//   gemm_fulla    — whole A resident in __local (n²·4 B), B staged in
//                   TK-row tiles; threads stride the output. Tests the
//                   "A is read n times — keep it resident" hypothesis.
//
// Compile-time params via #define (rendered by gpu_gemm.rs):
//   GEMM_TX, GEMM_TY, GEMM_RTX, GEMM_RTY, GEMM_TK, FULLA_TK, FULLA_MAXELEM

#ifndef GEMM_TX
#define GEMM_TX 8
#define GEMM_TY 8
#define GEMM_RTX 8
#define GEMM_RTY 8
#define GEMM_TK 16
#define GEMM_SPLIT_M 1
#define GEMM_SPLIT_N 1
#define GEMM_SQ 0
#define GEMM_TRI 0
#define GEMM_NFIX 0
#define GEMM_ITER_RESIDENT 0
#define GEMM_ITER_RED 0
#define GEMM_KU 1
#define GEMM_VEC 0
#define GEMM_MASK 0
#define FULLA_TK 8
#define FULLA_MAXELEM 116
#endif

#define GEMM_LSZ (GEMM_TX * GEMM_TY)
#define GEMM_LN (GEMM_TY * GEMM_RTY)

// Explicit vector accumulator type for GEMM_VEC / the tri kernel —
// acc[r] is one vector covering the whole RTX-wide register row, so the
// FMA loop issues RTY vector-FMAs per k instead of RTY×RTX scalar ones.
#if GEMM_RTX == 2
typedef float2 vacc_t;
#elif GEMM_RTX == 4
typedef float4 vacc_t;
#elif GEMM_RTX == 8
typedef float8 vacc_t;
#endif

// Workgroup reduce, fold-then-halve, correct for arbitrary (non-PoT)
// lsz (ceil-half fold — copied semantics from gpu_purify.cl pur_reduce:
// floor-half silently drops the tail element for odd lsz).
static inline float gemm_red(__local float* red, float v, const int lid, const int lsz) {
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
static inline float2 gemm_red2(__local float2* red, float2 v, const int lid, const int lsz) {
    red[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz; s > 1; s = (s + 1) >> 1) {
        const int h = (s + 1) >> 1;
        if (lid < s - h) red[lid] += red[lid + h];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float2 out = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return out;
}
// 2-level: 32 strided partials then a serial 32-fold by lid 0 — trades
// ~2·log2(lsz) barriers for 3 barriers + a short serial tail. For the
// lsz=242 regime this cuts ~30 barriers to ~4 per reduce pair.
static inline float2 gemm_red2_2lvl(__local float* red, float2 v, const int lid, const int lsz) {
    red[lid] = v.x;
    red[lid + lsz] = v.y;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 32) {
        float sx = 0.0f, sy = 0.0f;
        for (int t = lid; t < lsz; t += 32) { sx += red[t]; sy += red[t + lsz]; }
        red[lid] = sx;
        red[lid + 32] = sy;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float2 out = 0.0f;
    if (lid == 0) {
        float sx = 0.0f, sy = 0.0f;
        for (int t = 0; t < 32; ++t) { sx += red[t]; sy += red[t + 32]; }
        red[0] = sx;
        red[32] = sy;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    out = (float2)(red[0], red[32]);
    barrier(CLK_LOCAL_MEM_FENCE);
    return out;
}

// ------------------------------------------------------------------
// gemm_1elem — one thread per output element, row·col dot from global.
// ------------------------------------------------------------------
__kernel void gemm_1elem(
    const int n,
    const int batch,
    __global const float* A,
    __global const float* B,
    __global float* C)
{
    const int sys = get_group_id(0);
    if (sys >= batch) return;
    const int lid = get_local_id(0);
    const int wg = get_local_size(0);
    const int stride = n * n;
    __global const float* Ag = A + sys * stride;
    __global const float* Bg = B + sys * stride;
    __global float* Cg = C + sys * stride;
    for (int idx = lid; idx < stride; idx += wg) {
        const int i = idx / n, j = idx - i * n;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int k = 0;
        for (; k + 4 <= n; k += 4) {
            s0 = fma(Ag[i * n + k],     Bg[k * n + j],         s0);
            s1 = fma(Ag[i * n + k + 1], Bg[(k + 1) * n + j], s1);
            s2 = fma(Ag[i * n + k + 2], Bg[(k + 2) * n + j], s2);
            s3 = fma(Ag[i * n + k + 3], Bg[(k + 3) * n + j], s3);
        }
        for (; k < n; ++k) s0 = fma(Ag[i * n + k], Bg[k * n + j], s0);
        Cg[idx] = (s0 + s1) + (s2 + s3);
    }
}

// ------------------------------------------------------------------
// gemm_regtile — register-tiled schoolbook GEMM, one system per WG.
//
// Thread grid GEMM_TX×GEMM_TY; thread (tx,ty) owns the contiguous
// GEMM_RTY×GEMM_RTX micro-tile at (ty·RTY, tx·RTX) inside the
// (TY·RTY)×(TX·RTX) workgroup tile; the WG loops the tile over the
// matrix (masked). k staged in GEMM_TK-deep local tiles:
//   As_t[kk][row]  — A stored TRANSPOSED so each thread's RTY-element
//                    fragment is contiguous (vectorizable), LDA = WM
//   Bs[kk][col]    — B row-major, LDB = WN
// Zero-padding outside the matrix contributes nothing to the sums.
// ------------------------------------------------------------------
#if GEMM_MASK
// SCC warm path: launch index → physical slot, and a replica whose
// mask entry is 0 returns before any barrier (the decision is uniform
// across the workgroup). Same arithmetic as gemm_regtile.
__kernel void gemm_regtile_masked(
    const int n,
    const int batch,
    __global const float* A,
    __global const float* B,
    __global float* C,
    __local float* As_t,
    __local float* Bs,
    __global const int* active,
    __global const int* work_ids)
#else
__kernel void gemm_regtile(
    const int n,
    const int batch,
    __global const float* A,
    __global const float* B,
    __global float* C,
    __local float* As_t,
    __local float* Bs)
#endif
{
    // GEMM_SPLIT_M×GEMM_SPLIT_N workgroups per system: each takes an
    // interleaved subset of the WG output tiles (more resident WGs →
    // higher occupancy at small batch — the n=86/400-system regime is
    // thread-starved: 400 WGs ≪ SM thread slots).
    const int wg_idx = get_group_id(0) % (GEMM_SPLIT_M * GEMM_SPLIT_N);
#if GEMM_MASK
    const int slot = get_group_id(0) / (GEMM_SPLIT_M * GEMM_SPLIT_N);
    const int sys = work_ids[slot];
    if (sys < 0 || sys >= batch || active[sys] == 0) return;
#else
    const int sys = get_group_id(0) / (GEMM_SPLIT_M * GEMM_SPLIT_N);
    if (sys >= batch) return;
#endif
    const int part_m = wg_idx % GEMM_SPLIT_M;
    const int part_n = wg_idx / GEMM_SPLIT_M;
    // 1-D launch: thread grid derived from the flat local id so the WG
    // size is just GEMM_TX·GEMM_TY on a single dimension.
    const int lid = get_local_id(0);
    const int tx = lid % GEMM_TX;
    const int ty = lid / GEMM_TX;
    const int nthreads = GEMM_TX * GEMM_TY;
    const int stride = n * n;
    __global const float* Ag = A + sys * stride;
    __global const float* Bg = B + sys * stride;
    __global float* Cg = C + sys * stride;

    const int WM = GEMM_TY * GEMM_RTY;   // WG tile rows
    const int WN = GEMM_TX * GEMM_RTX;   // WG tile cols
#if GEMM_NFIX
    const int nn = GEMM_NFIX;            // compile-time n → unrolled loops
#else
    const int nn = n;
#endif
#if GEMM_TRI
    // Symmetric square, upper-triangle only: tile (ty,tx) overlaps j≥i
    // iff its max col ≥ its min row. Inactive tiles skip the FMA loop
    // but MUST still join staging + barriers. Real speedup needs the
    // per-thread tile area to halve (~4×4 → 253 tiles at n=88 cover)
    // — masking alone leaves per-thread latency unchanged.
    const int tile_on = (tx * GEMM_RTX + GEMM_RTX - 1) >= (ty * GEMM_RTY);
#endif

    for (int m0 = part_m * WM; m0 < nn; m0 += WM * GEMM_SPLIT_M) {
        for (int n0 = part_n * WN; n0 < nn; n0 += WN * GEMM_SPLIT_N) {
            float acc[GEMM_RTY][GEMM_RTX];
            for (int r = 0; r < GEMM_RTY; ++r)
                for (int c = 0; c < GEMM_RTX; ++c)
                    acc[r][c] = 0.0f;
            for (int k0 = 0; k0 < nn; k0 += GEMM_TK) {
                // stage A[ m0..m0+WM , k0..k0+TK ] transposed → As_t[kk][r]
                for (int t = lid; t < WM * GEMM_TK; t += nthreads) {
                    const int r = t / GEMM_TK, kk = t - r * GEMM_TK;
                    const int gr = m0 + r, gk = k0 + kk;
                    As_t[kk * WM + r] = (gr < nn && gk < nn) ? Ag[gr * n + gk] : 0.0f;
                }
#if !GEMM_SQ
                // stage B[ k0..k0+TK , n0..n0+WN ] → Bs[kk][c]
                for (int t = lid; t < GEMM_TK * WN; t += nthreads) {
                    const int kk = t / WN, c = t - kk * WN;
                    const int gk = k0 + kk, gc = n0 + c;
                    Bs[kk * WN + c] = (gk < nn && gc < nn) ? Bg[gk * n + gc] : 0.0f;
                }
#endif
                barrier(CLK_LOCAL_MEM_FENCE);
#if GEMM_TRI
                if (tile_on)
#endif
#if GEMM_NFIX
#pragma unroll
#endif
                for (int kk = 0; kk < GEMM_TK; ++kk) {
#if GEMM_RTY == 8
                    const float8 av = vload8(0, &As_t[kk * WM + ty * GEMM_RTY]);
                    const float af[GEMM_RTY] = {av.s0, av.s1, av.s2, av.s3,
                                                av.s4, av.s5, av.s6, av.s7};
#elif GEMM_RTY == 4
                    const float4 av = vload4(0, &As_t[kk * WM + ty * GEMM_RTY]);
                    const float af[GEMM_RTY] = {av.s0, av.s1, av.s2, av.s3};
#else
                    float af[GEMM_RTY];
                    for (int r = 0; r < GEMM_RTY; ++r)
                        af[r] = As_t[kk * WM + ty * GEMM_RTY + r];
#endif
#if GEMM_SQ
                    // C = A·A with A SYMMETRIC (purify D²) ⇒
                    // B[k0+kk][n0+c] = A[k0+kk][n0+c] = A[n0+c][k0+kk]
                    // = As_t[kk][n0−m0+c]; the single full-size tile has
                    // m0 = n0 = 0, so the B fragment lives IN As_t — no
                    // second staging loop, no B global traffic at all.
                    __local const float* bptr = &As_t[kk * WM + tx * GEMM_RTX];
#else
                    __local const float* bptr = &Bs[kk * WN + tx * GEMM_RTX];
#endif
#if GEMM_RTX == 8
                    const float8 bv = vload8(0, bptr);
                    const float bf[GEMM_RTX] = {bv.s0, bv.s1, bv.s2, bv.s3,
                                                bv.s4, bv.s5, bv.s6, bv.s7};
#elif GEMM_RTX == 4
                    const float4 bv = vload4(0, bptr);
                    const float bf[GEMM_RTX] = {bv.s0, bv.s1, bv.s2, bv.s3};
#else
                    float bf[GEMM_RTX];
                    for (int c = 0; c < GEMM_RTX; ++c)
                        bf[c] = bptr[c];
#endif
                    for (int r = 0; r < GEMM_RTY; ++r)
                        for (int c = 0; c < GEMM_RTX; ++c)
                            acc[r][c] = fma(af[r], bf[c], acc[r][c]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            const int r_base = m0 + ty * GEMM_RTY;
            const int c_base = n0 + tx * GEMM_RTX;
#if GEMM_TRI
            // upper-triangle elements: write (gr,gc) and its mirror.
            // Fully-lower tiles write nothing (tile_on=false ⇒ acc=0
            // anyway, and every element fails gc>=gr). Output is EXACTLY
            // symmetric — each element written once by a unique owner.
            for (int r = 0; r < GEMM_RTY; ++r) {
                const int gr = r_base + r;
                for (int c = 0; c < GEMM_RTX; ++c) {
                    const int gc = c_base + c;
                    if (gc >= gr && gc < nn) {
                        Cg[gr * n + gc] = acc[r][c];
                        if (gc > gr) Cg[gc * n + gr] = acc[r][c];
                    }
                }
            }
#else
            for (int r = 0; r < GEMM_RTY; ++r) {
                const int gr = r_base + r;
                if (gr < nn) {
                    for (int c = 0; c < GEMM_RTX; ++c) {
                        const int gc = c_base + c;
                        if (gc < nn) Cg[gr * n + gc] = acc[r][c];
                    }
                }
            }
#endif
        }
    }
}

// ------------------------------------------------------------------
// gemm_fulla — whole A resident in __local, B staged in FULLA_TK rows.
// Each thread owns elements (i_e, j_e) = (lid + e·wg) decomposed;
// acc[] registers persist across k-blocks. Tests whether keeping A
// fully resident beats re-staging it per output tile.
// ------------------------------------------------------------------
__kernel void gemm_fulla(
    const int n,
    const int batch,
    __global const float* A,
    __global const float* B,
    __global float* C,
    __local float* As,
    __local float* Bs)
{
    const int sys = get_group_id(0);
    if (sys >= batch) return;
    const int lid = get_local_id(0);
    const int wg = get_local_size(0);
    const int stride = n * n;
    __global const float* Ag = A + sys * stride;
    __global const float* Bg = B + sys * stride;
    __global float* Cg = C + sys * stride;

    for (int t = lid; t < stride; t += wg) As[t] = Ag[t];

    float acc[FULLA_MAXELEM];
    int ne = 0;
    for (int idx = lid; idx < stride && ne < FULLA_MAXELEM; idx += wg, ++ne) acc[ne] = 0.0f;

    for (int k0 = 0; k0 < n; k0 += FULLA_TK) {
        for (int t = lid; t < FULLA_TK * n; t += wg) {
            const int kk = t / n, c = t - kk * n;
            const int gk = k0 + kk;
            Bs[kk * n + c] = (gk < n) ? Bg[gk * n + c] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int e = 0;
        for (int idx = lid; idx < stride && e < FULLA_MAXELEM; idx += wg, ++e) {
            const int i = idx / n, j = idx - i * n;
            const int kmax = min(FULLA_TK, n - k0);
            for (int kk = 0; kk < kmax; ++kk)
                acc[e] = fma(As[i * n + k0 + kk], Bs[kk * n + j], acc[e]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int e = 0;
    for (int idx = lid; idx < stride && e < FULLA_MAXELEM; idx += wg, ++e) Cg[idx] = acc[e];
}

// ------------------------------------------------------------------
// gemm_sq_iter — iterated symmetric square D ← scl·D² for niter rounds
// in ONE launch (one WG per system). Isolates the two costs hidden in
// the fused tc2 step (0.141 ms vs 0.082 ms isolated GEMM):
//   GEMM_ITER_RESIDENT=1 — D lives in __local Ls[LN²] TRANSPOSED
//     (Ls[k*LN+m] = D[m][k]) for the whole loop: no k-tile staging, no
//     global traffic, 2 barriers/iter. LN=TY·RTY must equal TX·RTX ≥ n.
//     Local footprint at LN=88: 30.9 KB (+~2-4 KB reduce) < 48 KB —
//     but ~3 WGs/SM instead of ~5: occupancy trade-off, measured not
//     assumed.
//   =0 — global ping-pong B↔C per iter with the regtile-sq staging
//     interior: isolates launch/set_arg amortization from residency.
// GEMM_ITER_RED — per-iter ‖T−D‖² + Tr(scl·T) reduce → errs/trs[sys]:
//   1 = two separate fold-halve reduces (current tc2_step structure:
//       ~2·2·ceil(log2 lsz) barriers ≈ 32 at lsz=242),
//   2 = one merged float2 fold-halve (~16 barriers),
//   3 = merged + 2-level reduce (~4 barriers + 32-serial tail).
// Requires lsz == TX·TY and (resident) LN == TX·RTX == TY·RTY ≥ n.
// ------------------------------------------------------------------
__kernel void gemm_sq_iter(
    const int n,
    const int batch,
    const int niter,
    const float scl,
    __global const float* A,
    __global float* B,
    __global float* C,
    __global float* errs,
    __global float* trs)
{
    const int sys = get_group_id(0);
    if (sys >= batch) return;
    const int lid = get_local_id(0);
    const int lsz = GEMM_LSZ;
    const int tx = lid % GEMM_TX;
    const int ty = lid / GEMM_TX;
    const int stride = n * n;
    const int gi0 = ty * GEMM_RTY;
    const int gj0 = tx * GEMM_RTX;
    __global const float* Ag = A + sys * stride;
    __global float* Bg = B + sys * stride;
    __global float* Cg = C + sys * stride;
#if GEMM_ITER_RED
  #if GEMM_ITER_RED == 3
    __local float red[2 * GEMM_LSZ];
  #else
    __local float red[GEMM_LSZ];
    __local float2 red2[GEMM_LSZ];
  #endif
#endif

#if GEMM_ITER_RESIDENT
    __local float Ls[GEMM_LN * GEMM_LN];
    // load transposed into padded LN² tile: Ls[k][m] = A[m][k]
    for (int e = lid; e < GEMM_LN * GEMM_LN; e += lsz) {
        const int k = e / GEMM_LN, m = e - k * GEMM_LN;
        Ls[e] = (m < n && k < n) ? Ag[m * n + k] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int it = 0; it < niter; ++it) {
#if GEMM_VEC
        vacc_t acc[GEMM_RTY];            // acc[r][c] subscripts the vector
        for (int r = 0; r < GEMM_RTY; ++r) acc[r] = (vacc_t)0.0f;
#else
        float acc[GEMM_RTY][GEMM_RTX];
        for (int r = 0; r < GEMM_RTY; ++r)
            for (int c = 0; c < GEMM_RTX; ++c) acc[r][c] = 0.0f;
#endif
        // T_ij = Σ_k D[i,k]·D[j,k] = Σ_k Ls[k][i]·Ls[k][j] — padded
        // zeros are exact; uniform kk trip count, no staging/barriers.
#if GEMM_KU > 1
#pragma unroll GEMM_KU
#endif
        for (int kk = 0; kk < GEMM_LN; ++kk) {
#if GEMM_RTY == 8
            const float8 av = vload8(0, &Ls[kk * GEMM_LN + gi0]);
            const float af[GEMM_RTY] = {av.s0, av.s1, av.s2, av.s3,
                                        av.s4, av.s5, av.s6, av.s7};
#elif GEMM_RTY == 4
            const float4 av = vload4(0, &Ls[kk * GEMM_LN + gi0]);
            const float af[GEMM_RTY] = {av.s0, av.s1, av.s2, av.s3};
#else
            float af[GEMM_RTY];
            for (int r = 0; r < GEMM_RTY; ++r) af[r] = Ls[kk * GEMM_LN + gi0 + r];
#endif
#if GEMM_VEC
  #if GEMM_RTX == 8
            const vacc_t bv = vload8(0, &Ls[kk * GEMM_LN + gj0]);
  #elif GEMM_RTX == 4
            const vacc_t bv = vload4(0, &Ls[kk * GEMM_LN + gj0]);
  #elif GEMM_RTX == 2
            const vacc_t bv = vload2(0, &Ls[kk * GEMM_LN + gj0]);
  #endif
            for (int r = 0; r < GEMM_RTY; ++r)
                acc[r] = fma((vacc_t)af[r], bv, acc[r]);
#else
  #if GEMM_RTX == 8
            const float8 bv = vload8(0, &Ls[kk * GEMM_LN + gj0]);
            const float bf[GEMM_RTX] = {bv.s0, bv.s1, bv.s2, bv.s3,
                                        bv.s4, bv.s5, bv.s6, bv.s7};
  #elif GEMM_RTX == 4
            const float4 bv = vload4(0, &Ls[kk * GEMM_LN + gj0]);
            const float bf[GEMM_RTX] = {bv.s0, bv.s1, bv.s2, bv.s3};
  #else
            float bf[GEMM_RTX];
            for (int c = 0; c < GEMM_RTX; ++c) bf[c] = Ls[kk * GEMM_LN + gj0 + c];
  #endif
            for (int r = 0; r < GEMM_RTY; ++r)
                for (int c = 0; c < GEMM_RTX; ++c)
                    acc[r][c] = fma(af[r], bf[c], acc[r][c]);
#endif
        }
#if GEMM_ITER_RED
        // ‖T−D‖² + Tr(scl·T) — Din element (gi,gj) sits at Ls[gj][gi];
        // each thread owns its cells exclusively (read-then-write is safe).
        float er = 0.0f, trd = 0.0f;
        for (int r = 0; r < GEMM_RTY; ++r) {
            const int gi = gi0 + r;
            for (int c = 0; c < GEMM_RTX; ++c) {
                const int gj = gj0 + c;
                const int l = gj * GEMM_LN + gi;
                const float dd = acc[r][c] - Ls[l];
                er += dd * dd;
                if (gi == gj) trd += acc[r][c] * scl;
            }
        }
  #if GEMM_ITER_RED == 1
        er = gemm_red(red, er, lid, lsz);
        trd = gemm_red(red, trd, lid, lsz);
  #elif GEMM_ITER_RED == 2
        {
            const float2 s = gemm_red2(red2, (float2)(er, trd), lid, lsz);
            er = s.x; trd = s.y;
        }
  #else
        {
            const float2 s = gemm_red2_2lvl(red, (float2)(er, trd), lid, lsz);
            er = s.x; trd = s.y;
        }
  #endif
        if (lid == 0) { errs[sys] = sqrt(er); trs[sys] = trd; }
#endif
        barrier(CLK_LOCAL_MEM_FENCE);   // operand reads complete
        // transposed write-back: D'[gi][gj] = acc → Ls[gj][gi]
        for (int r = 0; r < GEMM_RTY; ++r)
            for (int c = 0; c < GEMM_RTX; ++c)
                Ls[(gj0 + c) * GEMM_LN + gi0 + r] = acc[r][c] * scl;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int r = 0; r < GEMM_RTY; ++r) {
        const int gi = gi0 + r;
        if (gi < n) {
            for (int c = 0; c < GEMM_RTX; ++c) {
                const int gj = gj0 + c;
                if (gj < n) Cg[gi * n + gj] = Ls[gj * GEMM_LN + gi];
            }
        }
    }
#else
    // global ping-pong: same regtile-sq interior, src → scl·src² → dst,
    // swapping Bg/Cg each iter; WG-only sync via GLOBAL_MEM_FENCE.
    __local float As_t[GEMM_TK * GEMM_LN];
    __global const float* src = Ag;
    __global float* dst = Bg;
    for (int it = 0; it < niter; ++it) {
        float acc[GEMM_RTY][GEMM_RTX];
        for (int r = 0; r < GEMM_RTY; ++r)
            for (int c = 0; c < GEMM_RTX; ++c) acc[r][c] = 0.0f;
        for (int k0 = 0; k0 < n; k0 += GEMM_TK) {
            for (int t = lid; t < GEMM_LN * GEMM_TK; t += lsz) {
                const int m = t / GEMM_TK, kk = t - m * GEMM_TK;
                const int gk = k0 + kk;
                As_t[kk * GEMM_LN + m] = (m < n && gk < n) ? src[m * n + gk] : 0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int kk = 0; kk < GEMM_TK; ++kk) {
#if GEMM_RTY == 8
                const float8 av = vload8(0, &As_t[kk * GEMM_LN + gi0]);
                const float af[GEMM_RTY] = {av.s0, av.s1, av.s2, av.s3,
                                            av.s4, av.s5, av.s6, av.s7};
#elif GEMM_RTY == 4
                const float4 av = vload4(0, &As_t[kk * GEMM_LN + gi0]);
                const float af[GEMM_RTY] = {av.s0, av.s1, av.s2, av.s3};
#else
                float af[GEMM_RTY];
                for (int r = 0; r < GEMM_RTY; ++r) af[r] = As_t[kk * GEMM_LN + gi0 + r];
#endif
#if GEMM_RTX == 8
                const float8 bv = vload8(0, &As_t[kk * GEMM_LN + gj0]);
                const float bf[GEMM_RTX] = {bv.s0, bv.s1, bv.s2, bv.s3,
                                            bv.s4, bv.s5, bv.s6, bv.s7};
#elif GEMM_RTX == 4
                const float4 bv = vload4(0, &As_t[kk * GEMM_LN + gj0]);
                const float bf[GEMM_RTX] = {bv.s0, bv.s1, bv.s2, bv.s3};
#else
                float bf[GEMM_RTX];
                for (int c = 0; c < GEMM_RTX; ++c) bf[c] = As_t[kk * GEMM_LN + gj0 + c];
#endif
                for (int r = 0; r < GEMM_RTY; ++r)
                    for (int c = 0; c < GEMM_RTX; ++c)
                        acc[r][c] = fma(af[r], bf[c], acc[r][c]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
#if GEMM_ITER_RED
        float er = 0.0f, trd = 0.0f;
        for (int r = 0; r < GEMM_RTY; ++r) {
            const int gi = gi0 + r;
            if (gi < n) {
                for (int c = 0; c < GEMM_RTX; ++c) {
                    const int gj = gj0 + c;
                    if (gj < n) {
                        const float dd = acc[r][c] - src[gi * n + gj];
                        er += dd * dd;
                        if (gi == gj) trd += acc[r][c] * scl;
                    }
                }
            }
        }
  #if GEMM_ITER_RED == 1
        er = gemm_red(red, er, lid, lsz);
        trd = gemm_red(red, trd, lid, lsz);
  #elif GEMM_ITER_RED == 2
        {
            const float2 s = gemm_red2(red2, (float2)(er, trd), lid, lsz);
            er = s.x; trd = s.y;
        }
  #else
        {
            const float2 s = gemm_red2_2lvl(red, (float2)(er, trd), lid, lsz);
            er = s.x; trd = s.y;
        }
  #endif
        if (lid == 0) { errs[sys] = sqrt(er); trs[sys] = trd; }
#endif
        for (int r = 0; r < GEMM_RTY; ++r) {
            const int gi = gi0 + r;
            if (gi < n) {
                for (int c = 0; c < GEMM_RTX; ++c) {
                    const int gj = gj0 + c;
                    if (gj < n) dst[gi * n + gj] = acc[r][c] * scl;
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
        src = dst;
        dst = (dst == Bg) ? Cg : Bg;
    }
    if (src != Cg)
        for (int e = lid; e < stride; e += lsz) Cg[e] = src[e];
#endif
}

// ------------------------------------------------------------------
// gemm_sq_iter_tri — compact SYRK: resident iterated D ← scl·D²
// computing only the UPPER triangle of 8×8 block tiles.
//
// The earlier GEMM_TRI masking failed because it left per-thread FMA
// latency unchanged (idle threads don't shorten the critical path).
// This variant instead SPLITS each upper-triangular 8×8 tile across
// TS = 8/RTX threads (lid → tile t = lid/TS, column part p = lid%TS):
//   RTX=4 → 132 thr, each 8×4 = 32 acc (halves the FMA chain vs r8x8)
//   RTX=2 → 264 thr, each 8×2 = 16 acc
// 66 upper blocks (LN/8 = 11 per row) ⇒ 66·TS threads — lsz must equal
// that (host-enforced). Mirror writes keep the resident matrix full
// symmetric; each output element still has exactly one writer.
// Frobenius uses the symmetric weighting Σdiag d² + 2Σ_{i<j} d².
// Requires RTY=8, GEMM_ITER_RESIDENT semantics (always resident).
// Compiled only for usable RTX/RTY — host validates the same shape.
// ------------------------------------------------------------------
#if (GEMM_RTX == 2 || GEMM_RTX == 4) && GEMM_RTY == 8
__kernel void gemm_sq_iter_tri(
    const int n,
    const int batch,
    const int niter,
    const float scl,
    __global const float* A,
    __global float* B,
    __global float* C,
    __global float* errs,
    __global float* trs)
{
    const int sys = get_group_id(0);
    if (sys >= batch) return;
    const int lid = get_local_id(0);
    const int lsz = GEMM_LSZ;
    const int stride = n * n;
    __global const float* Ag = A + sys * stride;
    __global float* Cg = C + sys * stride;

    // lid → upper-triangular block (bi,bj), bj≥bi, over NB=LN/8 per row,
    // plus column part (RTX-wide slice of the 8×8 tile).
    const int TS = 8 / GEMM_RTX;
    const int NB = GEMM_LN / 8;
    const int t = lid / TS, part = lid - t * TS;
    int bi = 0, rem = t;
    while (rem >= NB - bi) { rem -= NB - bi; ++bi; }   // ≤11 steps
    const int gj0 = (bi + rem) * 8 + part * GEMM_RTX;
    const int gi0 = bi * 8;

    __local float Ls[GEMM_LN * GEMM_LN];
#if GEMM_ITER_RED == 3
    __local float red[2 * GEMM_LSZ];
#elif GEMM_ITER_RED
    __local float red[GEMM_LSZ];
    __local float2 red2[GEMM_LSZ];
#endif

    for (int e = lid; e < GEMM_LN * GEMM_LN; e += lsz) {
        const int k = e / GEMM_LN, m = e - k * GEMM_LN;
        Ls[e] = (m < n && k < n) ? Ag[m * n + k] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int it = 0; it < niter; ++it) {
        vacc_t acc[GEMM_RTY];
        for (int r = 0; r < GEMM_RTY; ++r) acc[r] = (vacc_t)0.0f;
#if GEMM_KU > 1
#pragma unroll GEMM_KU
#endif
        for (int kk = 0; kk < GEMM_LN; ++kk) {
            const float8 av = vload8(0, &Ls[kk * GEMM_LN + gi0]);
            const float af[GEMM_RTY] = {av.s0, av.s1, av.s2, av.s3,
                                        av.s4, av.s5, av.s6, av.s7};
#if GEMM_RTX == 4
            const vacc_t bv = vload4(0, &Ls[kk * GEMM_LN + gj0]);
#elif GEMM_RTX == 2
            const vacc_t bv = vload2(0, &Ls[kk * GEMM_LN + gj0]);
#endif
            for (int r = 0; r < GEMM_RTY; ++r)
                acc[r] = fma((vacc_t)af[r], bv, acc[r]);
        }
#if GEMM_ITER_RED
        // ‖T−D‖² (symmetric: diag×1, off-diag×2) + Tr(scl·T); Din element
        // (gi,gj) sits at own primary cell Ls[gj][gi].
        float er = 0.0f, trd = 0.0f;
        for (int r = 0; r < GEMM_RTY; ++r) {
            const int gi = gi0 + r;
            for (int c = 0; c < GEMM_RTX; ++c) {
                const int gj = gj0 + c;
                if (gj >= gi) {
                    const float dd = acc[r][c] - Ls[gj * GEMM_LN + gi];
                    er += (gj > gi ? 2.0f : 1.0f) * dd * dd;
                    if (gi == gj) trd += acc[r][c] * scl;
                }
            }
        }
  #if GEMM_ITER_RED == 1
        er = gemm_red(red, er, lid, lsz);
        trd = gemm_red(red, trd, lid, lsz);
  #elif GEMM_ITER_RED == 2
        {
            const float2 s = gemm_red2(red2, (float2)(er, trd), lid, lsz);
            er = s.x; trd = s.y;
        }
  #else
        {
            const float2 s = gemm_red2_2lvl(red, (float2)(er, trd), lid, lsz);
            er = s.x; trd = s.y;
        }
  #endif
        if (lid == 0) { errs[sys] = sqrt(er); trs[sys] = trd; }
#endif
        barrier(CLK_LOCAL_MEM_FENCE);   // operand reads complete
        // write primary (gj≥gi) + mirror — each output cell has exactly
        // one writer; diagonal-block sub-diagonal elements are computed
        // but discarded (their values arrive via mirrors).
        for (int r = 0; r < GEMM_RTY; ++r) {
            const int gi = gi0 + r;
            for (int c = 0; c < GEMM_RTX; ++c) {
                const int gj = gj0 + c;
                if (gj >= gi) {
                    const float v = acc[r][c] * scl;
                    Ls[gj * GEMM_LN + gi] = v;
                    if (gj > gi) Ls[gi * GEMM_LN + gj] = v;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // store full symmetric output: C[gi][gj] = D[gi][gj] = Ls[gj][gi]
    for (int r = 0; r < GEMM_RTY; ++r) {
        const int gi = gi0 + r;
        for (int c = 0; c < GEMM_RTX; ++c) {
            const int gj = gj0 + c;
            if (gj >= gi && gi < n && gj < n) {
                const float v = Ls[gj * GEMM_LN + gi];
                Cg[gi * n + gj] = v;
                if (gj > gi) Cg[gj * n + gi] = v;
            }
        }
    }
}
#endif
