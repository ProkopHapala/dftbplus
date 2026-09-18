// ==================================================================
// GPU Tiled Block Jacobi Eigensolver (N > 64)
// ==================================================================
//
// Phase 2 of the H-Bond manifest: one-WG tiled/block Jacobi for
// dense symmetric eigendecomposition of matrices larger than the
// N≤64 full-local limit.
//
// Architecture (manifest §4.3):
// - One workgroup per system (parallelism across systems)
// - A and V stored in GLOBAL memory (N×N per system, no N² local limit)
// - Local memory: 2B×2B compound pivot + U rotation + strip workspace
// - Block size B=32 → pivot capacity 2B=64 (fits local memory)
// - Inner pivot diagonalized by Brent–Luk parallel cyclic Jacobi
//   (32 independent pairs/round, 8 threads/pair, all 256 threads active)
// - Strip updates: load [A_kp|A_kq], multiply by U, store + symmetric transpose
// - One kernel launch per batched eigensolve; all sync is WG-local barriers
//
// For N=87 (AT/GC nucleobase pairs): 3 blocks of 32,32,23 → 3 pivots/sweep
// No N² local-memory scaling — N can be 96, 128, 200, 500...
//
// ------------------------------------------------------------------

#ifndef B
#define B 32
#endif
#ifndef PB
#define PB 64          // pivot capacity = 2*B
#endif
#ifndef PLD
#define PLD 65          // padded leading dimension = PB+1 (bank conflicts)
#endif
#ifndef WG
#define WG 256
#endif
#ifndef STRIP_R
#define STRIP_R 32      // rows per strip tile
#endif
#ifndef MAX_SWEEPS
#define MAX_SWEEPS 50
#endif
#ifndef JACOBI_TOL
#define JACOBI_TOL 1.0e-9f
#endif
#ifndef PAIR_SKIP_TOL
#define PAIR_SKIP_TOL 1.0e-12f
#endif

// ---- Arithmetic precision modes (manifest §12 D2) ----
// JACOBI_PREC=0: pure FP32-FMA (rotation params, block/strip updates)
// JACOBI_PREC=1: FP64 only for scalar rotation construction c,s — FP32-FMA updates
// JACOBI_PREC=2: broad FP64 everywhere (accuracy reference; ~1.29M double
//                block-updates per pivot — a throughput disaster on a 3090)
#ifndef JACOBI_PREC
#define JACOBI_PREC 2
#endif
#if JACOBI_PREC >= 1
typedef double jrot_t;   // rotation-parameter precision
typedef double2 jlog2_t; // rotlog entry: bit-exact logged (c,s) for deferred V
#else
typedef float jrot_t;
typedef float2 jlog2_t;  // prec=0: c,s are f32 — double2 log would be pure waste
#endif
#if JACOBI_PREC >= 2
typedef double jupd_t;   // block/strip update accumulation precision
#else
typedef float jupd_t;
#endif

// ---- Inner pivot Brent–Luk parallel Jacobi parameters ----
// PB=64 → JPAIR_INNER=32 independent pairs/round, PPG_INNER=8 threads/pair,
// JROUND_INNER=63 rounds/sweep. 32×8=256=WG threads all active.
#ifndef JPAIR_INNER
#define JPAIR_INNER (PB / 2)
#endif
#ifndef PPG_INNER
#define PPG_INNER (WG / JPAIR_INNER)
#endif
#ifndef JROUND_INNER
#define JROUND_INNER (PB - 1)
#endif
#ifndef INNER_SWEEPS
#define INNER_SWEEPS 20
#endif

// ------------------------------------------------------------------
// Round-robin pair schedule (Brent–Luk parallel cyclic ordering).
// For PB elements (PB even), PB-1 rounds. In each round, PB/2 disjoint
// pairs. Element PB-1 stays fixed; the remaining PB-1 elements rotate.
// Over PB-1 rounds, every pair of elements is visited exactly once.
// ------------------------------------------------------------------
inline int2 inner_jacobi_pair(int round, int ipair) {
    int m = PB - 1;
    if (ipair == 0) return (int2)(m, round);
    return (int2)((round + ipair) % m, (round + m - ipair) % m);
}

// ------------------------------------------------------------------
// jacobi_cyclic_global_batched  (N>64 direct solver — replaces the
// tiled block-Jacobi whose cost was ~16k workgroup barriers/eigensolve)
//
// Direct parallel cyclic Jacobi on the FULL matrix in global memory.
// One WG/system, one kernel launch, ~3 barriers per round.
//
// Schedule: N padded to even JN; Brent–Luk round-robin gives JN/2
// disjoint pairs per round, JN−1 rounds per sweep. Each pair (p,q)
// applies the SAME rotation convention as the old inner pivot:
//   A ← JᵀAJ on the 2×2 blocks (pa,qa)×(pb,qb) for ALL pair-combos (a,b)
//   V ← VJ  on columns (pa,qa)
// The pad index (when n odd) is never stored: pairs touching it are
// identity and out-of-range writes are skipped.
//
// Exit test (f32-aware): off = ‖offdiag(A)‖_F each sweep; stop when
//   off < JACOBI_OFF_TOL · ‖A‖_F
// ‖A‖_F is invariant under orthogonal similarity → computed ONCE at
// entry; the test does not get harder on warm-started (near-diagonal) A.
// Backstop: stagnation counter as before.
//
// Rotations (c,s) still built in f64 when JACOBI_PREC>=1 — one scalar
// pair per (round,pair), off the throughput path. PAIR_SKIP is now
// RELATIVE to the local diagonal scale (the old absolute 1e-12 could
// never trigger in f32).
//
// init_v: 0 → V=I (cold), 1 → keep V (warm AO basis B; A=BᵀHB formed
// by the caller's GEMMs — on exit V holds the new eigenvectors C=BJ).
// ------------------------------------------------------------------
#ifndef JACOBI_OFF_TOL
#define JACOBI_OFF_TOL 1.0e-6f    // off/‖A‖_F exit threshold
#endif
#ifndef PAIR_SKIP_REL
#define PAIR_SKIP_REL 3.0e-8f     // |a_pq| < ε·(|a_pp|+|a_qq|) → skip
#endif
#ifndef MAX_CSWEEPS
#define MAX_CSWEEPS 40
#endif

inline int2 cyclic_jacobi_pair(int round, int ipair, int jn) {
    int m = jn - 1;
    if (ipair == 0) return (int2)(m, round % m);
    return (int2)((round + ipair) % m, (round + m - ipair) % m);
}

// Packed symmetric-A index: the resident kernel stores only the lower
// triangle (r>=c) of A as lA[r*(r+1)/2 + c] — halves the resident
// footprint (n(n+1)/2·4 B ≈ 14.6 KB at n=86 vs 29.9 KB square) so TWO
// workgroups can share an SM's 48 KB local memory. Accesses sort (r,c);
// A is symmetric so A[r][c] == A[c][r] always.
inline int lat(const int r, const int c) {
    const int mx = max(r, c), mn = min(r, c);
    return (mx * (mx + 1) >> 1) + mn;
}

// Workgroup reductions for ARBITRARY lsz (WG≈N sweeps want 96/160/224 —
// the naive `lsz>>1` halving requires a power of two and silently drops
// lanes otherwise). Fold the tail [p2,lsz) onto [0,lsz−p2) first
// (p2 = largest power of two ≤ lsz), then halve over p2 entries.
// All work-items must call (workgroup barriers inside); the result is
// read after the trailing barrier so `r` is reusable by the next call.
inline float jac_sum_f(__local float* r, float v, const int lid, const int lsz) {
    r[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    int p2 = 1;
    while ((p2 << 1) <= lsz) p2 <<= 1;
    if (lid + p2 < lsz) r[lid] += r[lid + p2];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = p2 >> 1; o > 0; o >>= 1) {
        if (lid < o) r[lid] += r[lid + o];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float s = r[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return s;
}

// f64 variant with op select: 0=sum, 1=min, 2=max (Fermi bracket+Newton).
inline double jac_red_d(__local double* r, double v, const int lid, const int lsz, const int op) {
    r[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    int p2 = 1;
    while ((p2 << 1) <= lsz) p2 <<= 1;
    if (lid + p2 < lsz) {
        const double w = r[lid + p2];
        r[lid] = (op == 0) ? (r[lid] + w) : ((op == 1) ? fmin(r[lid], w) : fmax(r[lid], w));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = p2 >> 1; o > 0; o >>= 1) {
        if (lid < o) {
            const double w = r[lid + o];
            r[lid] = (op == 0) ? (r[lid] + w) : ((op == 1) ? fmin(r[lid], w) : fmax(r[lid], w));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const double s = r[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return s;
}

__kernel void jacobi_cyclic_global_batched(
    __global float* A,   // [batch][n*n] in/out — eigenvalues on diag at exit
    __global float* V,   // [batch][n*n] in/out — eigenvectors (cols) at exit
    const int n,
    const int batch,
    const int init_v,
    __global const int* active,     // [batch] 0 → replica frozen, early-out
    __global float* diag,           // [batch][4] out: {off, off/‖A‖_F, stop, sweeps}
                                    // stop: 0 converged · 1 stagnation ·
                                    //       2 MAX_CSWEEPS · 3 n>capacity · 4 non-finite
    // R5 tail: Fermi smearing on the just-solved spectrum — replaces the
    // separate extract_diag + fermi_occ launches on the production path.
    const int fermi_tail,           // 1 → solve Σ f_k(μ)=n_occ and write occ_w+mu
    const int n_occ_fermi,
    const float kT,
    __global float* occ_w,          // [batch*n] out: Fermi weights f_k
    __global float* mu,             // [batch] in/out: μ warm-start in, result out
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int gid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (active[gid] == 0) return;

    const int jn = (n & 1) ? n + 1 : n;   // even pad
    const int jpair = jn / 2;
    const int jround = jn - 1;

    __local jrot_t rot_c[128];    // supports n ≤ 256
    __local jrot_t rot_s[128];
    __local int    rot_p[128];
    __local int    rot_q[128];
    __local volatile int l_nrot;  // live (non-skipped) rotations this round
    __local float  reduce[WG];
    __local float  red2[WG];      // second column of the packed prologue reduce
#ifndef JACOBI_NO_TAIL
    // R5 Fermi tail scratch (used only when fermi_tail=1). JACOBI_NO_TAIL
    // compiles these out — ~9 KB of local per WG on WG512.
    __local float  le[256];       // staged eigenvalues (diag of A)
    __local double dred[WG];      // f64 reductions: sum
    // dred2 removed — f64 reductions are sequential, share `dred` (-4 KB/WG).
    __local double lmu[4];        // {lo, hi, μ, done} — bracket state
#endif

    // R5: defensive capacity guard — rot_*[] hold jpair ≤ 128 → jn ≤ 256.
    // The host rejects n>256 at plan build; if it ever reaches the kernel,
    // report loudly via diag instead of scribbling past local arrays.
    if (jpair > 128) {
        // First-failure latch — same contract as the main diag write below.
        if (lid == 0 && diag[4*gid+2] <= 0.0f) {
            diag[4*gid] = INFINITY; diag[4*gid+1] = INFINITY;
            diag[4*gid+2] = 3.0f;   diag[4*gid+3] = 0.0f;
        }
        return;
    }

    __global float* gA = A + (size_t)gid * n * n;
    __global float* gV = V + (size_t)gid * n * n;
    const int nn = n * n;

    // ---- fused prologue: V=I init + ‖A‖_F + initial off in ONE pass ----
    // R8b: the warm path exits at ~0 sweeps — this prologue IS the kernel
    // cost, so the three nn-passes and two barrier chains are fused into
    // one pass + one packed reduce. ‖A‖_F is invariant under Jacobi
    // similarity → computed once at entry; the exit test does not get
    // harder on warm-started (near-diagonal) A.
    float fa = 0.0f, fo = 0.0f;
    for (int idx = lid; idx < nn; idx += lsz) {
        int r = idx / n;
        float v = gA[idx];
        fa += v * v;
        if (r != idx - r * n) fo += v * v;
        if (init_v == 0) gV[idx] = (r == idx - r * n) ? 1.0f : 0.0f;
    }
    const float frob = sqrt(fmax(jac_sum_f(reduce, fa, lid, lsz), 1.0e-30f));
    float off_cur = sqrt(fmax(jac_sum_f(red2, fo, lid, lsz), 0.0f));
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    const float off_exit = JACOBI_OFF_TOL * frob;
    float prev_off = fmax(off_cur, 1.0e-30f);
    int stall = 0;
    int stop = 0;   // R5 stop reason: 0 converged · 1 stagnation · 2 max sweeps · 4 non-finite
    int nsw  = 0;

    if (off_cur > off_exit) {
    stop = 2;
    for (int sweep = 0; sweep < MAX_CSWEEPS; ++sweep) {
        nsw = sweep + 1;
        for (int r = 0; r < jround; ++r) {
            // ---- Phase 1: rotation params per pair (strided for n>256) ----
            // KNOWN COST (JACOBI_PREC≥1, jrot_t=double): only jpair≈44 of the
            // 256 threads do work here, each running an f64 divide + 2 f64
            // square roots. On consumer GPUs (RTX 3090: FP64 = 1/64 FP32 rate,
            // f64 sqrt/div are multi-instruction software sequences) this is
            // a serialized scalar island — the rest of the workgroup waits at
            // the barrier below. GPT-5.6 instruction 5: once this kernel is
            // fast, retry JACOBI_PREC=0 (f32 c,s): the formula
            // t = sign(τ)/(|τ|+√(1+τ²)) is already cancellation-safe in f32;
            // a ~1e-7 rotation-angle error is self-correcting (later rounds
            // re-zero the off-diagonal; accuracy is set by JACOBI_OFF_TOL and
            // the Rayleigh/renorm repair, not by rotation exactness).
            // A/B: rebuild with JACOBI_PREC=0 and compare δ_eig, ‖HC−SCε‖,
            // AT/GC ΔE vs prec1.
            if (lid == 0) l_nrot = 0;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int ip = lid; ip < jpair; ip += lsz) {
                int2 pq = cyclic_jacobi_pair(r, ip, jn);
                int p = min(pq.x, pq.y);   // p < q
                int q = max(pq.x, pq.y);
                rot_p[ip] = p; rot_q[ip] = q;
                float apq = (q < n) ? gA[p * n + q] : 0.0f;
                float app = gA[p * n + p];
                float aqq = (q < n) ? gA[q * n + q] : 1.0f;
                // R5: !(a > b) form is NaN-safe AND handles the exact-zero
                // pivot (apq=0, app=aqq=0 → 0 > 0 false → skip; the old
                // `0 < 0` fell through to a 0/0 rotation → NaN).
                if (!(fabs(apq) > PAIR_SKIP_REL * (fabs(app) + fabs(aqq)))) {
                    rot_c[ip] = (jrot_t)1.0; rot_s[ip] = (jrot_t)0.0;
                } else {
                    jrot_t aq = (jrot_t)apq;
                    jrot_t tau = ((jrot_t)aqq - (jrot_t)app) / ((jrot_t)2.0 * aq);
                    jrot_t t = (tau >= (jrot_t)0.0)
                        ? (jrot_t)1.0 / (tau + sqrt((jrot_t)1.0 + tau * tau))
                        : -(jrot_t)1.0 / (-tau + sqrt((jrot_t)1.0 + tau * tau));
                    jrot_t c = (jrot_t)1.0 / sqrt((jrot_t)1.0 + t * t);
                    rot_c[ip] = c; rot_s[ip] = t * c;
                    atomic_inc(&l_nrot);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (l_nrot == 0) continue;   // this round's pairs all below skip; other rounds may not be

            // ---- Phase 2+3 fused: A 2×2 blocks + V columns, one barrier ----
            const int nblk = jpair * jpair;
            const int nvt = n * jpair;
            for (int it = lid; it < nblk + nvt; it += lsz) {
                if (it < nblk) {
                    int a = it / jpair, b = it - a * jpair;
                    jupd_t sa = (jupd_t)rot_s[a], sb = (jupd_t)rot_s[b];
                    if (sa == (jupd_t)0.0 && sb == (jupd_t)0.0) continue;  // both identity
                    int pa = rot_p[a], qa = rot_q[a];
                    int pb = rot_p[b], qb = rot_q[b];
                    jupd_t ca = (jupd_t)rot_c[a], cb = (jupd_t)rot_c[b];
                    jupd_t apr = (jupd_t)gA[pa * n + pb];
                    jupd_t aps = (qb < n) ? (jupd_t)gA[pa * n + qb] : (jupd_t)0.0;
                    jupd_t aqr = (qa < n) ? (jupd_t)gA[qa * n + pb] : (jupd_t)0.0;
                    jupd_t aqs = (qa < n && qb < n) ? (jupd_t)gA[qa * n + qb] : (jupd_t)0.0;
                    jupd_t tpr = fma(cb, apr, -sb * aps);
                    jupd_t tps = fma(sb, apr, cb * aps);
                    jupd_t tqr = fma(cb, aqr, -sb * aqs);
                    jupd_t tqs = fma(sb, aqr, cb * aqs);
                    gA[pa * n + pb] = (float)fma(ca, tpr, -sa * tqr);
                    if (qb < n) gA[pa * n + qb] = (float)fma(ca, tps, -sa * tqs);
                    if (qa < n) {
                        gA[qa * n + pb] = (float)fma(sa, tpr, ca * tqr);
                        if (qb < n) gA[qa * n + qb] = (float)fma(sa, tps, ca * tqs);
                    }
                } else {
                    int t = it - nblk;
                    int k = t / jpair, a = t - (t / jpair) * jpair;
                    jupd_t s = (jupd_t)rot_s[a];
                    if (s == (jupd_t)0.0) continue;
                    int p = rot_p[a], q = rot_q[a];
                    jupd_t c = (jupd_t)rot_c[a];
                    jupd_t vkp = (jupd_t)gV[k * n + p];
                    jupd_t vkq = (q < n) ? (jupd_t)gV[k * n + q] : (jupd_t)0.0;
                    gV[k * n + p] = (float)fma(c, vkp, -s * vkq);
                    if (q < n) gV[k * n + q] = (float)fma(s, vkp, c * vkq);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }

        // ---- Sweep-end off-norm + exit/stagnation tests ----
        fo = 0.0f;
        for (int idx = lid; idx < nn; idx += lsz) {
            int r = idx / n;
            if (r != idx - r * n) { float v = gA[idx]; fo += v * v; }
        }
        off_cur = sqrt(jac_sum_f(reduce, fo, lid, lsz));
        if (!isfinite(off_cur)) { stop = 4; break; }        // R5: NaN/Inf → report, don't loop on NaN
        if (off_cur <= off_exit) { stop = 0; break; }
        if (off_cur > 0.9f * prev_off) { if (++stall >= 3) { stop = 1; break; } } else { stall = 0; }
        prev_off = off_cur;
    }
    }

    // ---- R5: stop-reason + achieved-residual diagnostics ----
    // First-failure latch (T01): the host clears stop to 0 at the start of
    // each solve window; a recorded stop>0 is never overwritten by a later
    // launch, so a mid-iteration failure can no longer be silently erased
    // by a subsequent converged one before check_jacobi() reads it.
    if (lid == 0 && diag[4*gid+2] <= 0.0f) {
        diag[4*gid]   = off_cur;
        diag[4*gid+1] = off_cur / frob;
        diag[4*gid+2] = (float)stop;
        diag[4*gid+3] = (float)nsw;
    }

    // ---- Eigenvalues on the diagonal ----
    // R8b: the old success-path pass that zeroed the off-diagonal residue
    // was dead work — nothing reads hp off the diagonal (extract_diag and
    // the Fermi tail read gA[k*n+k] only; the next GEMM overwrites hp
    // wholesale). On failure the residue stays: it is the diagnostic
    // record of WHERE the solver stopped (W1).

    // ---- R5 tail: Fermi μ + occ_w on the just-solved spectrum ----
    // Safeguarded Newton (bisection fallback) solving Σ f_k(μ) = n_occ,
    // warm-started from mu[sid] of the previous solve. f64 for the
    // sum/decision (discrete-decision rule); f_k stored f32.
    // On every CERTIFIABLE exit (stop≤2 — the host certifies 1/2 at the
    // f32 floor rel≤1e-4, so their occ_w must be fresh too — a tail gated
    // on stop==0 alone would certify stale occupations). A non-finite
    // solve (stop=4) can never be certified — its occ_w stays stale and
    // the replica is parked as Failed downstream.
#ifndef JACOBI_NO_TAIL
    if (fermi_tail != 0 && stop <= 2) {
        const double kt = (double)kT;
        const double want = (double)n_occ_fermi;
        for (int k = lid; k < n; k += lsz) le[k] = gA[k * n + k];
        barrier(CLK_LOCAL_MEM_FENCE);

        // bracket [emin−32kT, emax+32kT] — same guaranteed bracket as
        // fermi_occ_batched (s(μ) increasing in μ; s(lo)<n_occ<s(hi))
        double elo = 1.0e300, ehi = -1.0e300;
        for (int k = lid; k < n; k += lsz) {
            const double e = (double)le[k];
            elo = fmin(elo, e); ehi = fmax(ehi, e);
        }
        elo = jac_red_d(dred, elo, lid, lsz, 1) - 32.0 * kt;
        ehi = jac_red_d(dred, ehi, lid, lsz, 2) + 32.0 * kt;

        if (lid == 0) {
            const double mp = (double)mu[gid];
            lmu[0] = elo; lmu[1] = ehi;
            lmu[2] = (mp > elo && mp < ehi) ? mp : 0.5 * (elo + ehi);
            lmu[3] = 0.0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int it = 0; it < 24; ++it) {
            const double m = lmu[2];
            double s = 0.0, ds = 0.0;
            for (int k = lid; k < n; k += lsz) {
                const double x = ((double)le[k] - m) / kt;
                const double f = (x > 40.0) ? 0.0 : ((x < -40.0) ? 1.0 : 1.0 / (1.0 + exp(x)));
                s += f;
                ds += (x > 40.0 || x < -40.0) ? 0.0 : f * (1.0 - f);
            }
            const double st = jac_red_d(dred, s, lid, lsz, 0);
            const double dst = jac_red_d(dred, ds, lid, lsz, 0);
            if (lid == 0) {
                const double g = st - want;
                if (fabs(g) < 1.0e-7 * fmax(want, 1.0)) {
                    lmu[3] = 1.0;
                } else {
                    if (st > want) lmu[1] = fmin(lmu[1], m); else lmu[0] = fmax(lmu[0], m);
                    double mn = m;
                    if (dst > 0.0) mn = m - g * kt / dst;   // ds/dμ = Σf(1−f)/kT
                    if (!(mn > lmu[0] && mn < lmu[1])) mn = 0.5 * (lmu[0] + lmu[1]);
                    lmu[2] = mn;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lmu[3] > 0.5) break;
        }
        const double muf = lmu[2];
        if (lid == 0) mu[gid] = (float)muf;
        for (int k = lid; k < n; k += lsz) {
            const double x = ((double)le[k] - muf) / kt;
            occ_w[(size_t)gid * n + k] = (float)((x > 40.0) ? 0.0 : ((x < -40.0) ? 1.0 : 1.0 / (1.0 + exp(x))));
        }
    }
#else
    // Fail-loud: the host requested the Fermi tail but this program was
    // compiled JACOBI_NO_TAIL — occ_w/mu would silently stay stale.
    // stop=5 marks the solve failed (host check_jacobi reports it).
    if (fermi_tail != 0 && stop <= 2 && lid == 0) diag[4*gid+2] = 5.0f;
#endif
}

// ------------------------------------------------------------------
// jacobi_resident_batched  (T08b: data-residency experiment)
//
// Same Brent–Luk parallel cyclic Jacobi mathematics and diag/tail
// contract as jacobi_cyclic_global_batched, but A lives in __local for
// the WHOLE solve — the streaming kernel re-reads/re-writes all of A and
// V every one of ~n−1 rounds per sweep (~10 MB/sweep/system at n=86 ≈
// 0.8 TB/s logical traffic ≈ DRAM-bound). Two residency modes:
//
//   RESIDENT_V=1  A AND V in local (per-round V update on lV, one
//                 write-back at exit). ~2×n(n+1)×4 B local/WG — n=86 →
//                 ~60 KB + scratch ≈ 1 WG/SM on a 100 KB SM.
//   RESIDENT_V=0  A local, V deferred: rotations are logged to a global
//                 rotlog (jlog2_t (c,s) per (round,pair); (p,q) is
//                 deterministic from the Brent–Luk schedule) and applied
//                 to global V once per SWEEP in an apply epoch — V's
//                 footprint (~30 KB at n=86) stays L1/L2-resident across
//                 the epoch, so V global traffic drops ~n× (once per
//                 sweep instead of once per round). Packed lA ≈ n(n+1)/2×4 B
//                 (~14.6 KB at n=86) + ~9 KB scratch ≈ 24 KB/WG
//                 → 2 WGs/SM on a 48 KB device.
//
// lA/lV are dynamic __local args; the host sizes them to the actual n
// and must check local_mem_size. lA is PACKED SYMMETRIC — only the
// lower triangle lA[r*(r+1)/2+c] (lat() helper), n(n+1)/2·4 B ≈ 14.6 KB
// at n=86 (vs 29.9 KB square); lV stays row-padded square (V is
// orthogonal, not symmetric). Same capacity limits: jpair ≤ 128 (rot
// arrays) → n ≤ 256; A-local practical bound is the device local limit.
// ------------------------------------------------------------------
#ifndef RESIDENT_V
#define RESIDENT_V 0
#endif

__kernel void jacobi_resident_batched(
    __global float* A,   // [batch][n*n] in/out — eigenvalues on diag at exit
    __global float* V,   // [batch][n*n] in/out — eigenvectors (cols) at exit
    const int n,
    const int batch,
    const int init_v,
    __global const int* active,
    __global float* diag,              // [batch][4] — same contract as direct kernel
    // Tail args at the SAME indices as jacobi_cyclic_global_batched (7..11)
    // so the plan's bind_solve_params set_arg calls work on both kernels.
    const int fermi_tail,
    const int n_occ_fermi,
    const float kT,
    __global float* occ_w,
    __global float* mu,
    __global jlog2_t* rotlog,          // [batch][jround*jpair] scratch (RESIDENT_V=0 only; pass any valid buf otherwise)
    __local float* lA,                 // n*(n+1)/2 — packed lower triangle (lat())
    __local float* lV,                 // n*(n+1) when RESIDENT_V, else 1-elem dummy
    __global const int* work_ids       // launch-index → physical slot (identity at full batch)
) {
    const int gid = work_ids[get_group_id(0)];
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (active[gid] == 0) return;

    const int jn = (n & 1) ? n + 1 : n;
    const int jpair = jn / 2;
    const int jround = jn - 1;
    const int lald = n + 1;                    // padded leading dim — lV only (lA is packed-triangular)
    const size_t log_stride = (size_t)jround * jpair;

    __local jrot_t rot_c[128];
    __local jrot_t rot_s[128];
    __local int    rot_p[128];
    __local int    rot_q[128];
    __local volatile int l_nrot;
    __local float  reduce[WG];
    // red2/dred2 removed: the reductions are strictly sequential — one
    // buffer per dtype suffices, and the saved ~6 KB/WG of scratch is
    // what puts TWO resident WGs (packed lA ~15 KB + ~9 KB scratch)
    // inside a 48 KB SM.
#ifndef JACOBI_NO_TAIL
    __local float  le[256];
    __local double dred[WG];
    __local double lmu[4];
#endif

    if (jpair > 128) {
        if (lid == 0 && diag[4*gid+2] <= 0.0f) {
            diag[4*gid] = INFINITY; diag[4*gid+1] = INFINITY;
            diag[4*gid+2] = 3.0f;   diag[4*gid+3] = 0.0f;
        }
        return;
    }

    __global float* gA = A + (size_t)gid * n * n;
    __global float* gV = V + (size_t)gid * n * n;
    const int nn = n * n;

    // ---- fused prologue: A→lA (+ ‖A‖_F + off) and V→lV / gV=I, one pass ----
    float fa = 0.0f, fo = 0.0f;
    for (int idx = lid; idx < nn; idx += lsz) {
        int r = idx / n;
        int c = idx - r * n;
        float v = gA[idx];
        if (r >= c) lA[lat(r, c)] = v;   // packed lower triangle only
        fa += v * v;
        if (r != c) fo += v * v;
#if RESIDENT_V
        lV[r * lald + c] = (init_v == 0) ? ((r == c) ? 1.0f : 0.0f) : gV[idx];
#else
        if (init_v == 0) gV[idx] = (r == c) ? 1.0f : 0.0f;
#endif
    }
    const float frob = sqrt(fmax(jac_sum_f(reduce, fa, lid, lsz), 1.0e-30f));
    float off_cur = sqrt(fmax(jac_sum_f(reduce, fo, lid, lsz), 0.0f));
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    const float off_exit = JACOBI_OFF_TOL * frob;
    float prev_off = fmax(off_cur, 1.0e-30f);
    int stall = 0;
    int stop = 0;
    int nsw  = 0;

    if (off_cur > off_exit) {
    stop = 2;
    for (int sweep = 0; sweep < MAX_CSWEEPS; ++sweep) {
        nsw = sweep + 1;
        for (int r = 0; r < jround; ++r) {
            // ---- Phase 1: rotation params per pair — from lA ----
            if (lid == 0) l_nrot = 0;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int ip = lid; ip < jpair; ip += lsz) {
                int2 pq = cyclic_jacobi_pair(r, ip, jn);
                int p = min(pq.x, pq.y);
                int q = max(pq.x, pq.y);
                rot_p[ip] = p; rot_q[ip] = q;
                float apq = (q < n) ? lA[lat(p, q)] : 0.0f;
                float app = lA[lat(p, p)];
                float aqq = (q < n) ? lA[lat(q, q)] : 1.0f;
                jrot_t c, s;
                if (!(fabs(apq) > PAIR_SKIP_REL * (fabs(app) + fabs(aqq)))) {
                    c = (jrot_t)1.0; s = (jrot_t)0.0;
                } else {
                    jrot_t aq = (jrot_t)apq;
                    jrot_t tau = ((jrot_t)aqq - (jrot_t)app) / ((jrot_t)2.0 * aq);
                    jrot_t t = (tau >= (jrot_t)0.0)
                        ? (jrot_t)1.0 / (tau + sqrt((jrot_t)1.0 + tau * tau))
                        : -(jrot_t)1.0 / (-tau + sqrt((jrot_t)1.0 + tau * tau));
                    c = (jrot_t)1.0 / sqrt((jrot_t)1.0 + t * t);
                    s = t * c;
                    atomic_inc(&l_nrot);
                }
                rot_c[ip] = c; rot_s[ip] = s;
#if !RESIDENT_V
                rotlog[(size_t)gid * log_stride + r * jpair + ip] = (jlog2_t)((jrot_t)c, (jrot_t)s);
#endif
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (l_nrot == 0) continue;

            // ---- Phase 2: A 2×2 blocks — entirely in local ----
            const int nblk = jpair * jpair;
#if RESIDENT_V
            const int nvt = n * jpair;
            for (int it = lid; it < nblk + nvt; it += lsz) {
#else
            for (int it = lid; it < nblk; it += lsz) {
#endif
                if (it < nblk) {
                    int a = it / jpair, b = it - a * jpair;
                    // Packed lA: block (a,b) and its transpose (b,a) map to
                    // the SAME packed slots — only a<=b may run (each slot
                    // gets exactly one writer; new M_ba = (new M_ab)' so the
                    // transpose block would rewrite identical values, and
                    // racing with it would double-rotate). Also halves the
                    // phase-2 work vs the square layout.
                    if (b < a) continue;
                    jupd_t sa = (jupd_t)rot_s[a], sb = (jupd_t)rot_s[b];
                    if (sa == (jupd_t)0.0 && sb == (jupd_t)0.0) continue;
                    int pa = rot_p[a], qa = rot_q[a];
                    int pb = rot_p[b], qb = rot_q[b];
                    jupd_t ca = (jupd_t)rot_c[a], cb = (jupd_t)rot_c[b];
                    jupd_t apr = (jupd_t)lA[lat(pa, pb)];
                    jupd_t aps = (qb < n) ? (jupd_t)lA[lat(pa, qb)] : (jupd_t)0.0;
                    jupd_t aqr = (qa < n) ? (jupd_t)lA[lat(qa, pb)] : (jupd_t)0.0;
                    jupd_t aqs = (qa < n && qb < n) ? (jupd_t)lA[lat(qa, qb)] : (jupd_t)0.0;
                    jupd_t tpr = fma(cb, apr, -sb * aps);
                    jupd_t tps = fma(sb, apr, cb * aps);
                    jupd_t tqr = fma(cb, aqr, -sb * aqs);
                    jupd_t tqs = fma(sb, aqr, cb * aqs);
                    lA[lat(pa, pb)] = (float)fma(ca, tpr, -sa * tqr);
                    if (qb < n) lA[lat(pa, qb)] = (float)fma(ca, tps, -sa * tqs);
                    if (qa < n) {
                        lA[lat(qa, pb)] = (float)fma(sa, tpr, ca * tqr);
                        if (qb < n) lA[lat(qa, qb)] = (float)fma(sa, tps, ca * tqs);
                    }
                }
#if RESIDENT_V
                else {
                    int t = it - nblk;
                    int k = t / jpair, a = t - (t / jpair) * jpair;
                    jupd_t s = (jupd_t)rot_s[a];
                    if (s == (jupd_t)0.0) continue;
                    int p = rot_p[a], q = rot_q[a];
                    jupd_t c = (jupd_t)rot_c[a];
                    jupd_t vkp = (jupd_t)lV[k * lald + p];
                    jupd_t vkq = (q < n) ? (jupd_t)lV[k * lald + q] : (jupd_t)0.0;
                    lV[k * lald + p] = (float)fma(c, vkp, -s * vkq);
                    if (q < n) lV[k * lald + q] = (float)fma(s, vkp, c * vkq);
                }
#endif
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

#if !RESIDENT_V
        // ---- Sweep-end deferred V apply ----
        // All of this sweep's rotations replayed against global V. V's
        // footprint (n²×4B) stays cache-resident across the epoch so this
        // is ~n× less global traffic than the per-round update; the log
        // (jpair jlog2_t/round) streams through L2 trivially. Rounds must
        // apply in order; pairs within a round are disjoint.
        for (int r = 0; r < jround; ++r) {
            for (int ip = lid; ip < jpair; ip += lsz) {
                const jlog2_t cs = rotlog[(size_t)gid * log_stride + r * jpair + ip];
                int2 pq = cyclic_jacobi_pair(r, ip, jn);
                rot_c[ip] = (jrot_t)cs.x; rot_s[ip] = (jrot_t)cs.y;
                rot_p[ip] = min(pq.x, pq.y); rot_q[ip] = max(pq.x, pq.y);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int it = lid; it < n * jpair; it += lsz) {
                int k = it / jpair, a = it - (it / jpair) * jpair;
                jupd_t s = (jupd_t)rot_s[a];
                if (s == (jupd_t)0.0) continue;
                int p = rot_p[a], q = rot_q[a];
                jupd_t c = (jupd_t)rot_c[a];
                jupd_t vkp = (jupd_t)gV[k * n + p];
                jupd_t vkq = (q < n) ? (jupd_t)gV[k * n + q] : (jupd_t)0.0;
                gV[k * n + p] = (float)fma(c, vkp, -s * vkq);
                if (q < n) gV[k * n + q] = (float)fma(s, vkp, c * vkq);
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
#endif

        // ---- Sweep-end off-norm + exit/stagnation tests — from lA ----
        // Packed storage: only r>c elements exist; 2·Σ_{r>c} = Σ_{r≠c}
        // (×2 is exact in fp — same off_cur value as the square loop).
        fo = 0.0f;
        for (int idx = lid; idx < nn; idx += lsz) {
            int r = idx / n;
            int c = idx - r * n;
            if (r > c) { float v = lA[lat(r, c)]; fo += 2.0f * v * v; }
        }
        off_cur = sqrt(jac_sum_f(reduce, fo, lid, lsz));
        if (!isfinite(off_cur)) { stop = 4; break; }
        if (off_cur <= off_exit) { stop = 0; break; }
        if (off_cur > 0.9f * prev_off) { if (++stall >= 3) { stop = 1; break; } } else { stall = 0; }
        prev_off = off_cur;
    }
    }

    // ---- Write-back: transformed A (residue on failure = diagnostic
    // record, same contract as the streaming kernel) and lV → gV ----
    for (int idx = lid; idx < nn; idx += lsz) {
        int r = idx / n;
        int c = idx - r * n;
        gA[idx] = lA[lat(r, c)];        // packed slot serves both triangles
#if RESIDENT_V
        gV[idx] = lV[r * lald + c];
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // ---- stop-reason + achieved-residual diagnostics (same latch) ----
    if (lid == 0 && diag[4*gid+2] <= 0.0f) {
        diag[4*gid]   = off_cur;
        diag[4*gid+1] = off_cur / frob;
        diag[4*gid+2] = (float)stop;
        diag[4*gid+3] = (float)nsw;
    }

    // ---- R5 tail: Fermi μ + occ_w on the just-solved spectrum ----
#ifndef JACOBI_NO_TAIL
    if (fermi_tail != 0 && stop <= 2) {
        const double kt = (double)kT;
        const double want = (double)n_occ_fermi;
        for (int k = lid; k < n; k += lsz) le[k] = lA[lat(k, k)];
        barrier(CLK_LOCAL_MEM_FENCE);

        double elo = 1.0e300, ehi = -1.0e300;
        for (int k = lid; k < n; k += lsz) {
            const double e = (double)le[k];
            elo = fmin(elo, e); ehi = fmax(ehi, e);
        }
        elo = jac_red_d(dred, elo, lid, lsz, 1) - 32.0 * kt;
        ehi = jac_red_d(dred, ehi, lid, lsz, 2) + 32.0 * kt;

        if (lid == 0) {
            const double mp = (double)mu[gid];
            lmu[0] = elo; lmu[1] = ehi;
            lmu[2] = (mp > elo && mp < ehi) ? mp : 0.5 * (elo + ehi);
            lmu[3] = 0.0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int it = 0; it < 24; ++it) {
            const double m = lmu[2];
            double s = 0.0, ds = 0.0;
            for (int k = lid; k < n; k += lsz) {
                const double x = ((double)le[k] - m) / kt;
                const double f = (x > 40.0) ? 0.0 : ((x < -40.0) ? 1.0 : 1.0 / (1.0 + exp(x)));
                s += f;
                ds += (x > 40.0 || x < -40.0) ? 0.0 : f * (1.0 - f);
            }
            const double st = jac_red_d(dred, s, lid, lsz, 0);
            const double dst = jac_red_d(dred, ds, lid, lsz, 0);
            if (lid == 0) {
                const double g = st - want;
                if (fabs(g) < 1.0e-7 * fmax(want, 1.0)) {
                    lmu[3] = 1.0;
                } else {
                    if (st > want) lmu[1] = fmin(lmu[1], m); else lmu[0] = fmax(lmu[0], m);
                    double mn = m;
                    if (dst > 0.0) mn = m - g * kt / dst;
                    if (!(mn > lmu[0] && mn < lmu[1])) mn = 0.5 * (lmu[0] + lmu[1]);
                    lmu[2] = mn;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lmu[3] > 0.5) break;
        }
        const double muf = lmu[2];
        if (lid == 0) mu[gid] = (float)muf;
        for (int k = lid; k < n; k += lsz) {
            const double x = ((double)le[k] - muf) / kt;
            occ_w[(size_t)gid * n + k] = (float)((x > 40.0) ? 0.0 : ((x < -40.0) ? 1.0 : 1.0 / (1.0 + exp(x))));
        }
    }
#else
    if (fermi_tail != 0 && stop <= 2 && lid == 0) diag[4*gid+2] = 5.0f;
#endif
}

// ------------------------------------------------------------------
// tiled_jacobi_batched  (DEPRECATED — kept for A/B measurement only;
// the production N>64 path is jacobi_cyclic_global_batched)
//
// Diagonalizes `batch` symmetric N×N matrices. One workgroup per system.
// A and V reside in global memory; only the 2B×2B compound pivot and
// strip workspace live in local memory.
//
// On exit:
//   A[batch][N*N] — eigenvalues on diagonal, off-diagonal ~0
//   V[batch][N*N] — eigenvectors (columns are eigenvectors)
// ------------------------------------------------------------------
__kernel void tiled_jacobi_batched(
    __global float* A,   // [batch][N*N] in/out
    __global float* V,    // [batch][N*N] out
    const int n,          // physical dimension
    const int batch,
    __global const int* work_ids    // launch-index → physical slot (identity at full batch)
) {
    const int gid = work_ids[get_group_id(0)];   // system index
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    __local float lA[PLD * PB];      // compound pivot (64×65)
    __local float lU[PLD * PB];     // rotation matrix from local Jacobi (64×65)
    __local float strip[STRIP_R * PB]; // strip workspace (32×64)
    __local float reduce[WG];        // reduction buffer
    // Brent–Luk parallel Jacobi rotation params (32 pairs)
    __local jrot_t rot_c[JPAIR_INNER];
    __local jrot_t rot_s[JPAIR_INNER];
    __local int   rot_p[JPAIR_INNER];
    __local int   rot_q[JPAIR_INNER];

    __global float* gA = A + (size_t)gid * n * n;
    __global float* gV = V + (size_t)gid * n * n;

    // ---- Initialize V = I (N×N) ----
    for (int idx = lid; idx < n * n; idx += lsz) {
        int r = idx / n;
        int c = idx - r * n;
        gV[idx] = (r == c) ? 1.0f : 0.0f;
    }
    // V is in global memory — need global fence before first pivot reads gV
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // ---- Compute initial off-diagonal norm (for relative convergence) ----
    float off_part = 0.0f;
    for (int idx = lid; idx < n * n; idx += lsz) {
        int r = idx / n;
        int c = idx - r * n;
        if (r != c) off_part += gA[idx] * gA[idx];
    }
    reduce[lid] = off_part;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) reduce[lid] += reduce[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float off0 = sqrt(fmax(reduce[0], JACOBI_TOL * JACOBI_TOL));
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- Block schedule ----
    int n_blocks = (n + B - 1) / B;
    // Block sizes: block i has size min(B, n - i*B)
    // Number of block pairs: n_blocks*(n_blocks-1)/2

    // ================ Jacobi sweeps ================
    float prev_off = fmax(off0, 1.0e-30f);
    int stall_count = 0;
    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {

        // Cyclic block pair order: (0,1), (0,2), ..., (1,2), (1,3), ...
        for (int bp = 0; bp < n_blocks; ++bp) {
            for (int bq = bp + 1; bq < n_blocks; ++bq) {
                int Bp = min(B, n - bp * B);  // active rows in block p
                int Bq = min(B, n - bq * B);  // active rows in block q
                int m = Bp + Bq;              // active pivot dimension (≤ PB)

                // ---- Step 1: Load compound pivot into lA ----
                // lA = [A_pp  A_pq]
                //      [A_qp  A_qq]
                // Padded to PB×PLD with dummy sentinel states.
                for (int i = lid; i < PB; i += lsz) {
                    for (int j = 0; j < PB; ++j) {
                        float val;
                        if (i < Bp && j < Bp) {
                            // A_pp block
                            val = gA[(bp*B+i)*n + (bp*B+j)];
                        } else if (i < Bp && j >= Bp && j < m) {
                            // A_pq block (column offset by Bp)
                            val = gA[(bp*B+i)*n + (bq*B+(j-Bp))];
                        } else if (i >= Bp && i < m && j < Bp) {
                            // A_qp block (row offset by Bp)
                            val = gA[(bq*B+(i-Bp))*n + (bp*B+j)];
                        } else if (i >= Bp && i < m && j >= Bp && j < m) {
                            // A_qq block
                            val = gA[(bq*B+(i-Bp))*n + (bq*B+(j-Bp))];
                        } else if (i == j) {
                            // Dummy diagonal: huge value
                            val = 1.0e30f;
                        } else {
                            // Dummy off-diagonal: zero
                            val = 0.0f;
                        }
                        lA[i * PLD + j] = val;
                    }
                }
                // Initialize lU = I (PB×PB, padded)
                for (int i = lid; i < PB; i += lsz) {
                    for (int j = 0; j < PB; ++j) {
                        lU[i * PLD + j] = (i == j) ? 1.0f : 0.0f;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                // ---- Step 2: Local Brent–Luk parallel cyclic Jacobi on lA ----
                // Diagonalizes the PB×PB local matrix (including dummies).
                // JPAIR_INNER=32 independent rotations per round, PPG_INNER=8
                // threads per pair → all 256 threads active (vs serial: thread 0
                // computes one rotation, 75% idle).
                // Block update: J^T·A·J applied via JPAIR² 2×2 blocks in parallel.
                for (int jsweep = 0; jsweep < INNER_SWEEPS; ++jsweep) {
                    for (int round = 0; round < JROUND_INNER; ++round) {

                        // Phase 1: compute rotation params for each pair
                        // JACOBI_PREC>=1 keeps the scalar c,s construction in
                        // f64 (cheap: 32 scalars/round); the expensive part was
                        // the millions of block/strip updates, not these.
                        int ipair = lid / PPG_INNER;
                        int sub   = lid % PPG_INNER;
                        if (sub == 0 && ipair < JPAIR_INNER) {
                            int2 pr = inner_jacobi_pair(round, ipair);
                            int p = pr.x;
                            int q = pr.y;
                            if (p > q) { int tmp = p; p = q; q = tmp; }
                            jrot_t apq = (jrot_t)lA[p * PLD + q];
                            if (fabs(apq) < (jrot_t)PAIR_SKIP_TOL) {
                                rot_c[ipair] = (jrot_t)1.0;
                                rot_s[ipair] = (jrot_t)0.0;
                            } else {
                                jrot_t app = (jrot_t)lA[p * PLD + p];
                                jrot_t aqq = (jrot_t)lA[q * PLD + q];
                                jrot_t tau = (aqq - app) / ((jrot_t)2.0 * apq);
                                jrot_t t = (tau >= (jrot_t)0.0)
                                    ? (jrot_t)1.0 / (tau + sqrt((jrot_t)1.0 + tau * tau))
                                    : -(jrot_t)1.0 / (-tau + sqrt((jrot_t)1.0 + tau * tau));
                                jrot_t c = (jrot_t)1.0 / sqrt((jrot_t)1.0 + t * t);
                                jrot_t s = t * c;
                                rot_c[ipair] = c;
                                rot_s[ipair] = s;
                            }
                            rot_p[ipair] = p;
                            rot_q[ipair] = q;
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);

                        // Phase 2: block update of lA (J^T · lA · J)
                        // Process all JPAIR² 2×2 blocks; 1024 blocks / 256 threads = 4 per thread.
                        // JACOBI_PREC<2 → FP32 FMA (D2: the f64 here was ~1.29M
                        // double ops per pivot, a 3090 throughput disaster).
                        for (int blk = lid; blk < JPAIR_INNER * JPAIR_INNER; blk += lsz) {
                            int a = blk / JPAIR_INNER;
                            int b = blk % JPAIR_INNER;
                            int p = rot_p[a], q = rot_q[a];
                            int r = rot_p[b], sc = rot_q[b];
                            jupd_t ca = (jupd_t)rot_c[a], sa = (jupd_t)rot_s[a];
                            jupd_t cb = (jupd_t)rot_c[b], sb = (jupd_t)rot_s[b];
                            jupd_t apr = (jupd_t)lA[p * PLD + r], aps = (jupd_t)lA[p * PLD + sc];
                            jupd_t aqr = (jupd_t)lA[q * PLD + r], aqs = (jupd_t)lA[q * PLD + sc];
                            jupd_t tpr = fma(cb, apr, -sb * aps);
                            jupd_t tps = fma(sb, apr, cb * aps);
                            jupd_t tqr = fma(cb, aqr, -sb * aqs);
                            jupd_t tqs = fma(sb, aqr, cb * aqs);
                            lA[p * PLD + r] = (float)fma(ca, tpr, -sa * tqr);
                            lA[p * PLD + sc] = (float)fma(ca, tps, -sa * tqs);
                            lA[q * PLD + r] = (float)fma(sa, tpr, ca * tqr);
                            lA[q * PLD + sc] = (float)fma(sa, tps, ca * tqs);
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);

                        // Phase 3: apply rotations to lU (eigenvectors)
                        ipair = lid / PPG_INNER;
                        sub   = lid % PPG_INNER;
                        if (ipair < JPAIR_INNER) {
                            jupd_t c = (jupd_t)rot_c[ipair];
                            jupd_t s = (jupd_t)rot_s[ipair];
                            int p = rot_p[ipair];
                            int q = rot_q[ipair];
                            for (int k = sub; k < PB; k += PPG_INNER) {
                                jupd_t vkp = (jupd_t)lU[k * PLD + p];
                                jupd_t vkq = (jupd_t)lU[k * PLD + q];
                                lU[k * PLD + p] = (float)fma(c, vkp, -s * vkq);
                                lU[k * PLD + q] = (float)fma(s, vkp, c * vkq);
                            }
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                }

                // ---- Step 3: Store transformed principal block back to global ----
                // lA now has U^T·P·U = D + R (eigenvalues on diagonal + residual
                // off-diagonal R). Store the ACTUAL lA values, not just the
                // diagonal — zeroing R breaks the invariant A_k = V_k^T A_0 V_k
                // because with finite inner sweeps R is not guaranteed tiny.
                for (int i = lid; i < m; i += lsz) {
                    for (int j = 0; j < m; ++j) {
                        float val = lA[i * PLD + j];  // R1: store actual value, not (i==j)?diag:0
                        // Map back to global (i,j) in the compound block
                        int gi, gj;
                        if (i < Bp && j < Bp) {
                            gi = bp*B+i; gj = bp*B+j;
                        } else if (i < Bp && j >= Bp) {
                            gi = bp*B+i; gj = bq*B+(j-Bp);
                        } else if (i >= Bp && j < Bp) {
                            gi = bq*B+(i-Bp); gj = bp*B+j;
                        } else {
                            gi = bq*B+(i-Bp); gj = bq*B+(j-Bp);
                        }
                        gA[gi * n + gj] = val;
                    }
                }
                // R3: strip updates read gA — need global fence
                barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

                // ---- Step 4: Update off-diagonal A strips ----
                // For each row tile k not in {p,q}: A[k][p] = A[k][p]·U, A[k][q] = A[k][q]·U
                // Symmetry: A[p][k] = A[k][p]^T
                for (int kt = 0; kt < n_blocks; ++kt) {
                    if (kt == bp || kt == bq) continue;
                    int Bk = min(B, n - kt * B);
                    // Load strip: [A_kp | A_kq] into strip[STRIP_R × PB]
                    // Process in tiles of STRIP_R rows
                    for (int koff = 0; koff < Bk; koff += STRIP_R) {
                        int nrows = min(STRIP_R, Bk - koff);
                        // Load strip
                        for (int i = lid; i < nrows * m; i += lsz) {
                            int r = i / m;
                            int c = i - r * m;
                            int gr = kt*B + koff + r;
                            int gc;
                            if (c < Bp) gc = bp*B + c;
                            else gc = bq*B + (c - Bp);
                            strip[r * PB + c] = gA[gr * n + gc];
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);

                        // Y = strip × lU (nrows × m) × (m × m) → (nrows × m)
                        // JACOBI_PREC>=2: f64 accumulation (reference).
                        // JACOBI_PREC<2:  FP32 FMA, 4 independent accumulators —
                        // shorter dependency chain + better summation than one
                        // serial accumulator; cheaper than Kahan or f64 (§12 D2).
                        for (int i = lid; i < nrows * m; i += lsz) {
                            int r = i / m;
                            int c = i - r * m;
#if JACOBI_PREC >= 2
                            double sum = 0.0;
                            for (int k = 0; k < m; ++k) {
                                sum += (double)strip[r * PB + k] * (double)lU[k * PLD + c];
                            }
                            float sumf = (float)sum;
#else
                            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                            int k = 0;
                            for (; k + 4 <= m; k += 4) {
                                s0 = fma(strip[r * PB + k    ], lU[(k    ) * PLD + c], s0);
                                s1 = fma(strip[r * PB + k + 1], lU[(k + 1) * PLD + c], s1);
                                s2 = fma(strip[r * PB + k + 2], lU[(k + 2) * PLD + c], s2);
                                s3 = fma(strip[r * PB + k + 3], lU[(k + 3) * PLD + c], s3);
                            }
                            for (; k < m; ++k) s0 = fma(strip[r * PB + k], lU[k * PLD + c], s0);
                            float sumf = (s0 + s1) + (s2 + s3);
#endif
                            // Store Y → [A_kp | A_kq]
                            int gr = kt*B + koff + r;
                            int gc;
                            if (c < Bp) gc = bp*B + c;
                            else gc = bq*B + (c - Bp);
                            gA[gr * n + gc] = sumf;
                            // Store Y^T → [A_pk | A_qk] (symmetry)
                            gA[gc * n + gr] = sumf;
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                }

                // ---- Step 5: Update V strips ----
                // V[k][p] = V[k][p]·U, V[k][q] = V[k][q]·U
                for (int kt = 0; kt < n_blocks; ++kt) {
                    int Bk = min(B, n - kt * B);
                    for (int koff = 0; koff < Bk; koff += STRIP_R) {
                        int nrows = min(STRIP_R, Bk - koff);
                        // Load strip: [V_kp | V_kq]
                        for (int i = lid; i < nrows * m; i += lsz) {
                            int r = i / m;
                            int c = i - r * m;
                            int gr = kt*B + koff + r;
                            int gc;
                            if (c < Bp) gc = bp*B + c;
                            else gc = bq*B + (c - Bp);
                            strip[r * PB + c] = gV[gr * n + gc];
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);

                        // Y = strip × lU — same JACOBI_PREC split as A strips
                        for (int i = lid; i < nrows * m; i += lsz) {
                            int r = i / m;
                            int c = i - r * m;
#if JACOBI_PREC >= 2
                            double sum = 0.0;
                            for (int k = 0; k < m; ++k) {
                                sum += (double)strip[r * PB + k] * (double)lU[k * PLD + c];
                            }
                            float sumf = (float)sum;
#else
                            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                            int k = 0;
                            for (; k + 4 <= m; k += 4) {
                                s0 = fma(strip[r * PB + k    ], lU[(k    ) * PLD + c], s0);
                                s1 = fma(strip[r * PB + k + 1], lU[(k + 1) * PLD + c], s1);
                                s2 = fma(strip[r * PB + k + 2], lU[(k + 2) * PLD + c], s2);
                                s3 = fma(strip[r * PB + k + 3], lU[(k + 3) * PLD + c], s3);
                            }
                            for (; k < m; ++k) s0 = fma(strip[r * PB + k], lU[k * PLD + c], s0);
                            float sumf = (s0 + s1) + (s2 + s3);
#endif
                            int gr = kt*B + koff + r;
                            int gc;
                            if (c < Bp) gc = bp*B + c;
                            else gc = bq*B + (c - Bp);
                            gV[gr * n + gc] = sumf;
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                }
                // R3: next pivot and convergence check read gA/gV — need global fence
                barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            }
        }

        // ---- Convergence check: physical off-diagonal Frobenius norm ----
        off_part = 0.0f;
        for (int idx = lid; idx < n * n; idx += lsz) {
            int r = idx / n;
            int c = idx - r * n;
            if (r != c) off_part += gA[idx] * gA[idx];
        }
        reduce[lid] = off_part;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int off = lsz >> 1; off > 0; off >>= 1) {
            if (lid < off) reduce[lid] += reduce[lid + off];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float off_cur = sqrt(reduce[0]);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (off_cur / off0 < JACOBI_TOL) break;
        // Stagnation detection: if off-diagonal norm hasn't improved by 10%
        // for 3 consecutive sweeps, further sweeps won't help (f32 floor).
        if (sweep > 0 && off_cur > 0.9f * prev_off) {
            stall_count++;
            if (stall_count >= 3) break;
        } else {
            stall_count = 0;
        }
        prev_off = off_cur;
    }

    // ---- Final: zero out off-diagonal of A (eigenvalues on diagonal) ----
    for (int idx = lid; idx < n * n; idx += lsz) {
        int r = idx / n;
        int c = idx - r * n;
        if (r != c) gA[idx] = 0.0f;
    }
}
