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
#define FULLA_TK 8
#define FULLA_MAXELEM 116
#endif

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
__kernel void gemm_regtile(
    const int n,
    const int batch,
    __global const float* A,
    __global const float* B,
    __global float* C,
    __local float* As_t,
    __local float* Bs)
{
    // GEMM_SPLIT_M×GEMM_SPLIT_N workgroups per system: each takes an
    // interleaved subset of the WG output tiles (more resident WGs →
    // higher occupancy at small batch — the n=86/400-system regime is
    // thread-starved: 400 WGs ≪ SM thread slots).
    const int wg_idx = get_group_id(0) % (GEMM_SPLIT_M * GEMM_SPLIT_N);
    const int sys = get_group_id(0) / (GEMM_SPLIT_M * GEMM_SPLIT_N);
    if (sys >= batch) return;
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

    for (int m0 = part_m * WM; m0 < n; m0 += WM * GEMM_SPLIT_M) {
        for (int n0 = part_n * WN; n0 < n; n0 += WN * GEMM_SPLIT_N) {
            float acc[GEMM_RTY][GEMM_RTX];
            for (int r = 0; r < GEMM_RTY; ++r)
                for (int c = 0; c < GEMM_RTX; ++c)
                    acc[r][c] = 0.0f;
            for (int k0 = 0; k0 < n; k0 += GEMM_TK) {
                // stage A[ m0..m0+WM , k0..k0+TK ] transposed → As_t[kk][r]
                for (int t = lid; t < WM * GEMM_TK; t += nthreads) {
                    const int r = t / GEMM_TK, kk = t - r * GEMM_TK;
                    const int gr = m0 + r, gk = k0 + kk;
                    As_t[kk * WM + r] = (gr < n && gk < n) ? Ag[gr * n + gk] : 0.0f;
                }
#if !GEMM_SQ
                // stage B[ k0..k0+TK , n0..n0+WN ] → Bs[kk][c]
                for (int t = lid; t < GEMM_TK * WN; t += nthreads) {
                    const int kk = t / WN, c = t - kk * WN;
                    const int gk = k0 + kk, gc = n0 + c;
                    Bs[kk * WN + c] = (gk < n && gc < n) ? Bg[gk * n + gc] : 0.0f;
                }
#endif
                barrier(CLK_LOCAL_MEM_FENCE);
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
            for (int r = 0; r < GEMM_RTY; ++r) {
                const int gr = r_base + r;
                if (gr < n) {
                    for (int c = 0; c < GEMM_RTX; ++c) {
                        const int gc = c_base + c;
                        if (gc < n) Cg[gr * n + gc] = acc[r][c];
                    }
                }
            }
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
