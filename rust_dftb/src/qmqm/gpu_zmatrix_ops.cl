// ==================================================================
// GPU complex (float2) matrix ops for the PBC k-point SCC path
// ==================================================================
//
// Complex-valued siblings of the real `gpu_matrix_ops.cl` kernels.
// Matrices are row-major [sid][n*n] float2 = (re, im); the flat system
// index is sid = rep*nk + kpt (all k-points of a replica contiguous).
//
// The replica mask contract: every flat-system kernel gates on
//   active[sid / nk]
// where `active` is the PER-REPLICA mask [n_rep] — the same buffer the
// (real) DIIS/mixer kernels clear on device-side convergence. One mask,
// zero per-iteration mask-sync launches; a frozen replica stops ALL its
// k-point workgroups. nk=1 degenerates to per-system gating.
//
// Charge-side state (q, dq, V, q_new, rms) stays REAL and per-replica —
// Mulliken charges are real by construction (Re[(D·S)_uu]); the real
// diis_step_batched / residual_and_mix_batched / commit_q_batched /
// dot_batched kernels in gpu_matrix_ops.cl are reused unchanged with
// batch = n_rep.
//
// Kernels:
//   zgemm_batched / zgemm_active_batched — tiled complex GEMM, op in
//       {0=N, 1=T, 2=H(conj-transpose)}; conjugation applied at tile load
//   zdq_v_batched          — per replica: dq = q-q0, V = gamma·dq (real)
//   zhscc_batched          — flat: H_scc = H0 + 1/2·S·(V_A + V_B)
//   zextract_diagonal_batched — flat: Re(diag A) -> eig
//   kpoint_occ_batched     — per replica: mu solves
//       sum_k w_k sum_n f(eps_nk - mu) = n_occ over ALL (band,k);
//       writes occ_w[sid*n+b] = w_k*f, mu[rep], e_band+mts into e_scal
//   zsc_mulliken_batched   — flat: per-orbital 2*Re sum_t w_t C_ut conj((SC)_ut)
//       then per-atom partial charges qk[sid][a]  (k-weight already in occ_w)
//   kpoint_qreduce_batched — per replica: q_new = sum_k qk
//   zscale_eigenvectors_batched — flat: V_scaled = V·rsqrt(max(lam,floor))
//   zsnormalize_batched    — per (sid,col): S-metric column renorm
//   zbuild_density_batched — flat: D = 2 sum_t w_t C C^H (W/forces later)
//   zkpoint_energy_tail_batched — per replica f64 scalar merge into e_scal
// ------------------------------------------------------------------

#ifndef ZTILE_M
#define ZTILE_M 16
#endif
#ifndef ZTILE_N
#define ZTILE_N 16
#endif
#ifndef ZTILE_K
#define ZTILE_K 32
#endif
#ifndef ZLAMBDA_FLOOR
#define ZLAMBDA_FLOOR 1.0e-7f
#endif

// ---- complex helpers (float2 = (re, im)) ----
inline float2 zconj(float2 a) { return (float2)(a.x, -a.y); }
inline float2 zmul(float2 a, float2 b) {
    return (float2)(fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x));
}
inline float  zre_mul_conj(float2 a, float2 b) {   // Re(a * conj(b))
    return fma(a.x, b.x, a.y * b.y);
}

// ------------------------------------------------------------------
// zgemm — tiled batched complex GEMM: C_b = alpha·op(A_b)·op(B_b) + beta·C_b
//
// Same tile geometry as real batched_gemm (ZTILE_M x ZTILE_N output tile,
// ZTILE_K depth, one WG per (tile, system)); op = 0 N / 1 T / 2 H, with
// the conjugate applied during the local-tile load so the inner product
// is a uniform complex multiply-add. alpha/beta are real (all SCC uses
// are alpha=1, beta=0). Local tiles are float2: ~2x the real kernel's
// local footprint (16*32 + 16*33 complex ~= 8.4 KB — fine).
// ------------------------------------------------------------------
static void zgemm_core(
    const int n,
    const int op_a,
    const int op_b,
    const float alpha,
    const float beta,
    __global const float2* Ab,
    __global const float2* Bb,
    __global float2* Cb,
    __local float2* As,
    __local float2* Bs
) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int row = get_group_id(0) * ZTILE_M + ly;
    const int col = get_group_id(1) * ZTILE_N + lx;
    const int lid = ly * ZTILE_N + lx;
    const int wg = ZTILE_M * ZTILE_N;
    const int LDB = ZTILE_K + 1;   // padded Bs (bank-conflict pad, as real)

    // Compute-side indexing identical to the real kernel: element k of
    // this thread's dot is As[abase + k*astr] * Bs[bbase + k*bstr].
    const int abase = (op_a != 0) ? ly : ly * ZTILE_K;
    const int astr  = (op_a != 0) ? ZTILE_M : 1;
    const int bbase = (op_b != 0) ? lx * LDB : lx;
    const int bstr  = (op_b != 0) ? 1 : ZTILE_N;

    // 4 independent complex accumulators — same chain-shortening as the
    // real kernel's s0..s3 (complex acc = re,im pairs).
    float2 s0 = (float2)(0.0f, 0.0f);
    float2 s1 = (float2)(0.0f, 0.0f);
    float2 s2 = (float2)(0.0f, 0.0f);
    float2 s3 = (float2)(0.0f, 0.0f);

    for (int k0 = 0; k0 < n; k0 += ZTILE_K) {
        // op(A) N: As[rr*K + kk];  T/H: As[kk*M + rr] with conj for H.
        if (op_a != 0) {
            for (int t = lid; t < ZTILE_M * ZTILE_K; t += wg) {
                int kk = t / ZTILE_M, rr = t - kk * ZTILE_M;
                int ar = get_group_id(0) * ZTILE_M + rr, ac = k0 + kk;
                float2 v = (ar < n && ac < n) ? Ab[ac * n + ar] : (float2)(0.0f, 0.0f);
                As[kk * ZTILE_M + rr] = (op_a == 2) ? zconj(v) : v;
            }
        } else {
            for (int t = lid; t < ZTILE_M * ZTILE_K; t += wg) {
                int rr = t / ZTILE_K, kk = t - rr * ZTILE_K;
                int ar = get_group_id(0) * ZTILE_M + rr, ac = k0 + kk;
                As[rr * ZTILE_K + kk] = (ar < n && ac < n) ? Ab[ar * n + ac] : (float2)(0.0f, 0.0f);
            }
        }
        if (op_b != 0) {
            for (int t = lid; t < ZTILE_K * ZTILE_N; t += wg) {
                int cc = t / ZTILE_K, kk = t - cc * ZTILE_K;
                int br = k0 + kk, bc = get_group_id(1) * ZTILE_N + cc;
                float2 v = (br < n && bc < n) ? Bb[bc * n + br] : (float2)(0.0f, 0.0f);
                Bs[cc * LDB + kk] = (op_b == 2) ? zconj(v) : v;
            }
        } else {
            for (int t = lid; t < ZTILE_K * ZTILE_N; t += wg) {
                int kk = t / ZTILE_N, cc = t - kk * ZTILE_N;
                int br = k0 + kk, bc = get_group_id(1) * ZTILE_N + cc;
                Bs[kk * ZTILE_N + cc] = (br < n && bc < n) ? Bb[br * n + bc] : (float2)(0.0f, 0.0f);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (row < n && col < n) {
            int kk = 0;
            for (; kk + 4 <= ZTILE_K; kk += 4) {
                s0 += zmul(As[abase + kk       * astr], Bs[bbase + kk       * bstr]);
                s1 += zmul(As[abase + (kk + 1) * astr], Bs[bbase + (kk + 1) * bstr]);
                s2 += zmul(As[abase + (kk + 2) * astr], Bs[bbase + (kk + 2) * bstr]);
                s3 += zmul(As[abase + (kk + 3) * astr], Bs[bbase + (kk + 3) * bstr]);
            }
            for (; kk < ZTILE_K; ++kk)
                s0 += zmul(As[abase + kk * astr], Bs[bbase + kk * bstr]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < n && col < n) {
        const int idx = row * n + col;
        const float2 s = (s0 + s1) + (s2 + s3);
        Cb[idx] = alpha * s + beta * Cb[idx];
    }
}

__kernel void zgemm_batched(
    const int n,
    const int batch,
    const int op_a,
    const int op_b,
    const float alpha,
    const float beta,
    __global const float2* A,
    __global const float2* B,
    __global float2* C,
    __local float2* As,
    __local float2* Bs
) {
    const int ib = get_group_id(2);
    if (ib >= batch) return;
    const int stride = n * n;
    zgemm_core(n, op_a, op_b, alpha, beta,
        A + ib * stride, B + ib * stride, C + ib * stride, As, Bs);
}

// SCC-path variant gated on the per-REPLICA mask: frozen replicas' tiles
// exit before loading anything (same role as real batched_gemm_active).
__kernel void zgemm_active_batched(
    const int n,
    const int batch,
    const int op_a,
    const int op_b,
    const float alpha,
    const float beta,
    __global const float2* A,
    __global const float2* B,
    __global float2* C,
    __local float2* As,
    __local float2* Bs,
    const int nk,                       // k-points per replica
    __global const int* active          // [n_rep] replica mask
) {
    const int ib = get_group_id(2);
    if (ib >= batch || active[ib / nk] == 0) return;
    const int stride = n * n;
    zgemm_core(n, op_a, op_b, alpha, beta,
        A + ib * stride, B + ib * stride, C + ib * stride, As, Bs);
}

// ------------------------------------------------------------------
// zdq_v_batched — per REPLICA, real: dq = q - q0 ; V = gamma·dq
//
// Phases 1-2 of the real fused_dq_v_hscc_batched, split out because the
// H_scc assembly is per (replica,k) while dq/V are per replica — fusing
// them would redo the O(n_atoms^2) gamma matvec nk times.
// ------------------------------------------------------------------
__kernel void zdq_v_batched(
    const int n_atoms,
    const int n_rep,
    __global const float* q,        // [n_rep*n_atoms]
    __global const float* q0,       // [n_rep*n_atoms]
    __global const float* G,        // [n_rep*n_atoms*n_atoms] periodic gamma (real)
    __global float* dq,             // [n_rep*n_atoms] out
    __global float* V,              // [n_rep*n_atoms] out
    __local float* ldq,             // [n_atoms]
    __global const int* active      // [n_rep]
) {
    const int sid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= n_rep || active[sid] == 0) return;
    __global const float* qb  = q  + (size_t)sid * n_atoms;
    __global const float* q0b = q0 + (size_t)sid * n_atoms;
    __global const float* Gb  = G  + (size_t)sid * n_atoms * n_atoms;
    __global float* dqb = dq + (size_t)sid * n_atoms;
    __global float* Vb  = V  + (size_t)sid * n_atoms;

    for (int a = lid; a < n_atoms; a += lsz) {
        const float d = qb[a] - q0b[a];
        ldq[a] = d; dqb[a] = d;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int a = lid; a < n_atoms; a += lsz) {
        __global const float* row = Gb + (size_t)a * n_atoms;
        float sum = 0.0f;
        for (int b = 0; b < n_atoms; ++b) sum = fma(row[b], ldq[b], sum);
        Vb[a] = sum;
    }
}

// ------------------------------------------------------------------
// zhscc_batched — per flat (replica,k): H_scc = H0 + 1/2·S·(V_A + V_B)
//
// The SCC shift V is REAL per replica; S(k), H0(k) complex. Elementwise
// over n*n — one thread per element, reads V via rep = sid/nk.
// ------------------------------------------------------------------
__kernel void zhscc_batched(
    const int n,
    const int n_atoms,
    const int batch,                // n_sys = n_rep*nk
    const int nk,
    __global const float2* H0,      // [n_sys*n*n]
    __global const float2* S,       // [n_sys*n*n]
    __global const float* V,        // [n_rep*n_atoms]
    __global const int* orb_atom,   // [n_rep*n]
    __global float2* H,             // [n_sys*n*n] out
    __global const int* active      // [n_rep]
) {
    const int sid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch) return;
    const int rep = sid / nk;
    if (active[rep] == 0) return;
    __global const float2* H0b = H0 + (size_t)sid * n * n;
    __global const float2* Sb  = S  + (size_t)sid * n * n;
    __global const float*  Vb  = V  + (size_t)rep * n_atoms;
    __global const int*    oa  = orb_atom + (size_t)rep * n;
    __global float2* Hb = H + (size_t)sid * n * n;

    const int nn = n * n;
    for (int idx = lid; idx < nn; idx += lsz) {
        const int i = idx / n;
        const int j = idx - i * n;
        const float v = 0.5f * (Vb[oa[i]] + Vb[oa[j]]);
        Hb[idx] = H0b[idx] + v * Sb[idx];
    }
}

// ------------------------------------------------------------------
// zextract_diagonal_batched — flat: eig[sid*n+i] = Re(A[sid][i][i]).
// One thread per (system, orbital); no barriers, no local memory.
// ------------------------------------------------------------------
__kernel void zextract_diagonal_batched(
    const int n,
    const int batch,
    const int nk,
    __global const float2* a,
    __global float* diag,
    __global const int* active      // [n_rep]
) {
    const int gid = get_global_id(0);
    const int total = n * batch;
    if (gid >= total) return;
    const int sid = gid / n;
    if (active[sid / nk] == 0) return;
    const int i = gid - sid * n;
    diag[gid] = a[(size_t)sid * n * n + i * n + i].x;
}

// ------------------------------------------------------------------
// kpoint_occ_batched — per REPLICA occupation under PBC.
//
// Solves  sum_k w_k * sum_b f(eps_bk - mu) = n_occ  for the shared
// chemical potential mu, where f is the Fermi function (kT>0) or the
// step function (kT=0, integer occupation — bisection only). Stages all
// nk*n eigenvalues of the replica in local memory, runs the same
// safeguarded Newton (bisection fallback, f64 decisions) as the real
// kernel's R5 Jacobi tail, then writes
//   occ_w[(rep*nk+k)*n + b] = w_k * f_bk        (k-weight folded in)
//   mu[rep], and the energy scalars e_scal[rep*4 + {0,1}] =
//   {2*sum_k w_k sum_b f*eps, sum_k w_k sum_b [f ln f + (1-f) ln(1-f)]}
// so the band energy and -TS need no extra pass.
// ------------------------------------------------------------------
__kernel void kpoint_occ_batched(
    const int n,
    const int nk,
    const int n_rep,
    const float n_occ,              // occupied-band equivalents per cell
    const float kT,                 // 0 -> integer occupation (step)
    __global const float* eig_diag, // [n_rep*nk*n]
    __global const float* kw,       // [nk] k-point weights (sum_k w_k = 1)
    __global float* occ_w,          // [n_rep*nk*n] out
    __global float* mu_out,         // [n_rep] out
    __global double* e_scal,        // [n_rep*4] out: [0]=e_band [1]=mts
    __local float* le,              // [nk*n] staged eigenvalues
    __local double* red,            // [lsz] f64 reduction scratch
    __local double* red2,           // [lsz] second reduction column
    __local double* lohi,           // [4] {lo, hi, mu, done}
    __global const int* active      // [n_rep]
) {
    const int rep = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (rep >= n_rep || active[rep] == 0) return;
    const int nb = n * nk;
    __global const float* eb = eig_diag + (size_t)rep * nb;
    __global float* ob = occ_w + (size_t)rep * nb;

    for (int k = lid; k < nb; k += lsz) le[k] = eb[k];
    barrier(CLK_LOCAL_MEM_FENCE);

    const double kt = fmax((double)kT, 1.0e-12);
    // bracket [emin-32kT, emax+32kT] — same guaranteed bracket as the
    // real path; for kT=0 the 32kT padding still brackets the spectrum.
    double elo = 1.0e300, ehi = -1.0e300;
    for (int k = lid; k < nb; k += lsz) {
        const double e = (double)le[k];
        elo = fmin(elo, e); ehi = fmax(ehi, e);
    }
    red[lid] = elo;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = lsz >> 1; o > 0; o >>= 1) {
        if (lid < o) red[lid] = fmin(red[lid], red[lid + o]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    elo = red[0] - 32.0 * kt;
    barrier(CLK_LOCAL_MEM_FENCE);
    red[lid] = ehi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = lsz >> 1; o > 0; o >>= 1) {
        if (lid < o) red[lid] = fmax(red[lid], red[lid + o]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    ehi = red[0] + 32.0 * kt;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        const double mp = (double)mu_out[rep];   // warm-start from previous solve
        lohi[0] = elo; lohi[1] = ehi;
        lohi[2] = (mp > elo && mp < ehi) ? mp : 0.5 * (elo + ehi);
        lohi[3] = 0.0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Safeguarded Newton (bisection fallback): s(mu) = sum_k w_k sum_b f
    // is increasing in mu; kT=0 degenerates f to a step — the Newton step
    // then just falls back to bisection every iteration, still converging
    // to a mu inside the HOMO-LUMO gap.
    const double want = (double)n_occ;
    for (int it = 0; it < 48; ++it) {
        const double m = lohi[2];
        double s = 0.0, ds = 0.0;
        for (int k = lid; k < nb; k += lsz) {
            const double x = ((double)le[k] - m) / kt;
            const double f = (x > 40.0) ? 0.0 : ((x < -40.0) ? 1.0 : 1.0 / (1.0 + exp(x)));
            const double w = (double)kw[k / n];
            s += w * f;
            ds += (x > 40.0 || x < -40.0) ? 0.0 : w * f * (1.0 - f);
        }
        red[lid] = s; red2[lid] = ds;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int o = lsz >> 1; o > 0; o >>= 1) {
            if (lid < o) { red[lid] += red[lid + o]; red2[lid] += red2[lid + o]; }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            const double g = red[0] - want;
            if (fabs(g) < 1.0e-7 * fmax(want, 1.0)) {
                lohi[3] = 1.0;
            } else {
                if (red[0] > want) lohi[1] = fmin(lohi[1], m); else lohi[0] = fmax(lohi[0], m);
                double mn = m;
                if (red2[0] > 0.0) mn = m - g * kt / red2[0];   // ds/dmu = sum w f(1-f)/kT
                if (!(mn > lohi[0] && mn < lohi[1])) mn = 0.5 * (lohi[0] + lohi[1]);
                lohi[2] = mn;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lohi[3] > 0.5) break;
    }
    const double muf = lohi[2];
    if (lid == 0) mu_out[rep] = (float)muf;

    // Write occ_w = w_k*f and accumulate e_band + mts in the same pass.
    double eb_acc = 0.0, mt_acc = 0.0;
    for (int k = lid; k < nb; k += lsz) {
        const double x = ((double)le[k] - muf) / kt;
        const double f = (x > 40.0) ? 0.0 : ((x < -40.0) ? 1.0 : 1.0 / (1.0 + exp(x)));
        const double w = (double)kw[k / n];
        ob[k] = (float)(w * f);
        eb_acc += 2.0 * w * f * (double)le[k];
        if (f > 1.0e-15 && f < 1.0 - 1.0e-15)
            mt_acc += w * (f * log(f) + (1.0 - f) * log(1.0 - f));
    }
    red[lid] = eb_acc; red2[lid] = mt_acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = lsz >> 1; o > 0; o >>= 1) {
        if (lid < o) { red[lid] += red[lid + o]; red2[lid] += red2[lid + o]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        e_scal[4 * rep]     = red[0];    // e_band = 2 sum_k w_k sum_b f*eps
        e_scal[4 * rep + 1] = red2[0];   // mts for the -TS term
    }
}

// ------------------------------------------------------------------
// zsc_mulliken_batched — per flat (replica,k) partial Mulliken charges.
//
// Per-orbital population at this k:
//   qk_u = 2 * Re sum_t w_t C_ut * conj((S·C)_ut)
// derived from q_u = Re[(D·S)_uu] with D_ut = 2 w_t C_ut conj(C_vt):
//   Re sum_v D_uv S_vu = Re sum_v D_uv conj(S_uv)
//                      = 2 Re sum_t w_t C_ut conj((SC)_ut).
// The density matrix is never materialized — one zgemm for SC replaces
// the O(n^2 * n_occ) D build plus the D read in Mulliken.
// occ_w already carries w_k, so qk is k-weighted; kpoint_qreduce is a
// plain sum over the replica's k block.
// ------------------------------------------------------------------
__kernel void zsc_mulliken_batched(
    const int n,
    const int n_atoms,
    const int batch,                // n_sys
    const int nk,
    __global const float2* C,       // [n_sys*n*n] eigenvectors (AO basis)
    __global const float2* SC,      // [n_sys*n*n] S·C
    __global const float* occ_w,    // [n_sys*n] w_k*f_t
    __global const int* orb_atom,   // [n_rep*n]
    __global float* qk,             // [n_sys*n_atoms] out partial charges
    __local float* diag,            // [n] per-orbital population
    __global const int* active      // [n_rep]
) {
    const int sid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid / nk] == 0) return;
    __global const float2* Cb  = C  + (size_t)sid * n * n;
    __global const float2* SCb = SC + (size_t)sid * n * n;
    __global const float*  owb = occ_w + (size_t)sid * n;
    __global const int*    oa  = orb_atom + (size_t)(sid / nk) * n;
    __global float* qb = qk + (size_t)sid * n_atoms;

    // Phase 1: per-orbital population — row-wise contiguous reads.
    for (int mu = lid; mu < n; mu += lsz) {
        float s = 0.0f;
        const int rn = mu * n;
        for (int t = 0; t < n; ++t) {
            s = fma(owb[t], zre_mul_conj(Cb[rn + t], SCb[rn + t]), s);
        }
        diag[mu] = 2.0f * s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: per-atom reduction (same structure as real mulliken).
    if (lid < n_atoms) {
        float sum = 0.0f;
        for (int mu = 0; mu < n; ++mu) {
            if (oa[mu] == lid) sum += diag[mu];
        }
        qb[lid] = sum;
    }
}

// ------------------------------------------------------------------
// kpoint_qreduce_batched — per replica: q_new = sum_k qk (occ_w carried
// the k-weights, so this is a plain sum over the replica's k block).
// ------------------------------------------------------------------
__kernel void kpoint_qreduce_batched(
    const int n_atoms,
    const int nk,
    const int n_rep,
    __global const float* qk,       // [n_rep*nk*n_atoms]
    __global float* q_new,          // [n_rep*n_atoms] out
    __global const int* active      // [n_rep]
) {
    const int rep = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (rep >= n_rep || active[rep] == 0) return;
    __global const float* src = qk + (size_t)rep * nk * n_atoms;
    __global float* dst = q_new + (size_t)rep * n_atoms;
    for (int a = lid; a < n_atoms; a += lsz) {
        float s = 0.0f;
        for (int k = 0; k < nk; ++k) s += src[k * n_atoms + a];
        dst[a] = s;
    }
}

// ------------------------------------------------------------------
// zscale_eigenvectors_batched — flat: V_scaled[i][k] = V[i][k]·rsqrt(lam_k)
// for the S^{-1/2} = V·diag(rsqrt lam)·V^H path; also reports lambda_min.
// Eigenvalues sit on Re(diag A) after the Hermitian Jacobi.
// ------------------------------------------------------------------
__kernel void zscale_eigenvectors_batched(
    const int n,
    const int batch,
    const int nk,
    __global const float2* A,       // [n_sys*n*n] eigenvalues on Re(diag)
    __global const float2* V,       // [n_sys*n*n] eigenvectors
    __global float2* V_scaled,      // [n_sys*n*n] out
    __global float* lambda_min,     // [n_sys] out
    __global const int* active      // [n_rep]
) {
    const int sid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid / nk] == 0) return;
    __global const float2* gA = A + (size_t)sid * n * n;
    __global const float2* gV = V + (size_t)sid * n * n;
    __global float2* gVs = V_scaled + (size_t)sid * n * n;

    if (lid == 0) {
        float lmin = 1.0e30f;
        for (int k = 0; k < n; ++k) lmin = fmin(lmin, gA[k * n + k].x);
        lambda_min[sid] = lmin;
    }
    for (int idx = lid; idx < n * n; idx += lsz) {
        const int k = idx - (idx / n) * n;
        const float rlam = rsqrt(fmax(gA[k * n + k].x, ZLAMBDA_FLOOR));
        gVs[idx] = gV[idx] * rlam;
    }
}

// ------------------------------------------------------------------
// zsnormalize_batched — S(k)-metric column renorm of the warm complex AO
// basis: n_k = c_k^H S c_k (real for Hermitian S), c_k /= sqrt(n_k).
// Same role as real snormalize_batched — arrests f32 norm drift of the
// in-place-rotated basis. One workgroup per (system, column).
// loc layout: t_s[n] float2 | red[lsz] float
// ------------------------------------------------------------------
__kernel void zsnormalize_batched(
    const int n,
    const int batch,
    const int nk,
    __global float2* C,
    __global const float2* S,
    __local float2* loc,
    __global const int* active      // [n_rep]
) {
    const int gid = get_group_id(0);
    const int sid = gid / n;
    const int k = gid - sid * n;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid / nk] == 0) return;
    __local float2* t_s = loc;              // [n]
    __local float*  red = (__local float*)(loc + n);
    const size_t base = (size_t)sid * n * n;
    __global float2* col = C + base + k;
    __global const float2* Sb = S + base;
    for (int r = lid; r < n; r += lsz) {
        const int ro = r * n;
        float2 ss = (float2)(0.0f, 0.0f);
        for (int cc = 0; cc < n; ++cc) ss += zmul(Sb[ro + cc], col[cc * n]);
        t_s[r] = ss;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // c^H (S c): accumulate conj(c_r)*t_s[r]; imag cancels for Hermitian S.
    float2 acc = (float2)(0.0f, 0.0f);
    for (int r = lid; r < n; r += lsz) acc += zmul(zconj(col[r * n]), t_s[r]);
    red[lid] = acc.x;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = rsqrt(red[0]);
    for (int r = lid; r < n; r += lsz) col[r * n] *= inv;
}

// ------------------------------------------------------------------
// zbuild_density_batched — flat: D = 2 sum_t w_t C[:,t] C[:,t]^H
// Lower triangle + conjugate mirror. NOT on the production charge path
// (zsc_mulliken needs only C and SC) — retained for the EDM/forces step
// (W = 2 sum_t w_t rho_t C C^H) which needs a materialized matrix.
// ------------------------------------------------------------------
__kernel void zbuild_density_batched(
    const int n,
    const int batch,
    const int nk,
    __global const float2* C,
    __global const float* occ_w,    // [n_sys*n] weights w_k*f_t (x eig for W)
    __global float2* D,
    __global const int* active      // [n_rep]
) {
    const int sid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid / nk] == 0) return;
    __global const float2* Cb = C + (size_t)sid * n * n;
    __global const float*  owb = occ_w + (size_t)sid * n;
    __global float2* Db = D + (size_t)sid * n * n;

    const int nt = n * (n + 1) / 2;
    for (int idx = lid; idx < nt; idx += lsz) {
        int i = (int)(0.5f * (sqrt(8.0f * (float)idx + 1.0f) - 1.0f));
        while ((i + 1) * (i + 2) / 2 <= idx) i++;
        while (i * (i + 1) / 2 > idx) i--;
        const int j = idx - i * (i + 1) / 2;
        const int in = i * n;
        const int jn = j * n;
        float2 s = (float2)(0.0f, 0.0f);
        for (int t = 0; t < n; ++t)
            s += owb[t] * zmul(Cb[in + t], zconj(Cb[jn + t]));
        s *= 2.0f;
        Db[in + j] = s;
        Db[jn + i] = zconj(s);
    }
}

// ------------------------------------------------------------------
// zkpoint_energy_tail_batched — per replica f64 scalar merge.
// e_scal[rep*4+2] = dq·V ; e_scal[rep*4+3] = q0·V  (slots 0/1 were
// written by kpoint_occ_batched). One launch + one e_scal readback at
// finalize — mirrors the real path's W12 energy_reduce contract.
// ------------------------------------------------------------------
__kernel void zkpoint_energy_tail_batched(
    const int n_atoms,
    const int n_rep,
    __global const float* dq,
    __global const float* q0,
    __global const float* v,
    __global double* e_scal,        // [n_rep*4]
    __local double* red,
    __local double* red2,
    __global const int* active      // [n_rep]
) {
    const int rep = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (rep >= n_rep || active[rep] == 0) return;
    __global const float* dqb = dq + (size_t)rep * n_atoms;
    __global const float* q0b = q0 + (size_t)rep * n_atoms;
    __global const float* vb  = v  + (size_t)rep * n_atoms;
    double s1 = 0.0, s2 = 0.0;
    for (int a = lid; a < n_atoms; a += lsz) {
        s1 += (double)dqb[a] * (double)vb[a];
        s2 += (double)q0b[a] * (double)vb[a];
    }
    red[lid] = s1; red2[lid] = s2;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = lsz >> 1; o > 0; o >>= 1) {
        if (lid < o) { red[lid] += red[lid + o]; red2[lid] += red2[lid + o]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        e_scal[4 * rep + 2] = red[0];
        e_scal[4 * rep + 3] = red2[0];
    }
}
