https://chatgpt.com/share/6aa8e403-69d8-83eb-800d-366bb27de532




---

# USER

do you think we can do excited states and carge sttes in this DFTB multi-solver framework?

I particular I would like to sample the excited state or charge state localize on one or the other molecule to study Proton-coupled electron transport ... do your think it is feasible? E.g. using CDFT?

Another thing - how difficult it is implement periodic boudnary conditions into this approach? We already have PBC in fortran version of DFTB+ so we can just copy that it should not be diffcult. Paralleization over k-points should be trivial...   









---

# Chat GPT 5.6 sol

# 7. PBC: Γ-point is fairly easy; arbitrary k is not quite trivial

Here I would qualify your statement.

**Adding periodic geometry at Γ only should indeed be relatively straightforward.**

The basic SK assembly simply becomes a sum over lattice images:

$$
H_{\mu\nu}
=
\sum_{\mathbf R}
H_{\mu\nu}(\mathbf R),
$$

and similarly for \(S\).

Your current pair infrastructure already works with pair displacement vectors. Extend a pair record by an integer image vector

$$
(n_x,n_y,n_z),
$$

and evaluate

$$
\mathbf r_{AB,\mathbf R}
=
\mathbf r_B+\mathbf R-\mathbf r_A.
$$

The repulsive force likewise just includes periodic image pairs.

For a large molecular/surface supercell at Γ, **most of your current dense real-valued solver survives unchanged**.

That is where I would start.

---

## 8. But SCC electrostatics under PBC is the part not to underestimate

The short-range SK matrix is easy.

The SCC term has long-range Coulomb behavior:

$$
E_{\rm SCC}
=
\frac12\sum_{AB}
\Delta q_A\gamma_{AB}\Delta q_B.
$$

Under PBC you need the periodic lattice sum rather than simply nearest-image \(\gamma(r)\).

This is where DFTB+ already has all the mature machinery, so I agree completely with:

> don't invent it; port the Fortran implementation.

It will involve the equivalent of Ewald/reciprocal-space periodic electrostatics plus the relevant self terms and derivatives.

That is probably the main physics work for Γ-point PBC, not H/S assembly.

For charged periodic cells there is then the usual additional issue of periodic net charge/background and finite-size effects.

For **charge transfer between two molecules with total cell charge zero**, e.g.

$$
D^+A^-,
$$

things are much cleaner, although the periodic dipole still gives finite-size interactions.

---

# 9. Arbitrary k-points change the matrix type

For a periodic Hamiltonian,

$$
H_{\mu\nu}(\mathbf k)
=
\sum_{\mathbf R}
H_{\mu\nu}(\mathbf R)
e^{i\mathbf k\cdot\mathbf R},
$$

and likewise

$$
S_{\mu\nu}(\mathbf k).
$$

Except at Γ and special symmetry points, these are **complex Hermitian matrices**.

That means your current real-symmetric Jacobi solver cannot just be reused unchanged.

You need either:

$$
\text{complex Hermitian Jacobi}
$$

or some different complex eigensolver.

I would absolutely **not** convert an \(N\times N\) complex Hermitian matrix to a \(2N\times2N\) real symmetric one except as a validation prototype; it wastes too much work.

A complex Jacobi rotation is conceptually straightforward, though: the pair rotation becomes unitary rather than real orthogonal. Much of your current Brent–Luk scheduling can survive.

---

# 10. But k-point parallelism really is excellent for your GPU architecture

Here your intuition is exactly right.

For a fixed SCC charge vector \(q\),

$$
H(\mathbf k_1),
H(\mathbf k_2),\ldots
$$

are independent.

Therefore your GPU grid naturally becomes:

$$
(\text{replica},\mathbf k).
$$

If you have:

$$
19\ \text{geometries}\times8\ k\text{-points}
=152
$$

independent eigensystems.

That is actually **much better occupancy** for the 3090 than the present 19-system molecular calculation.

The only coupling occurs when rebuilding charges:

$$
q_A
=
\sum_{\mathbf k}w_{\mathbf k}q_A(\mathbf k).
$$

And under smearing the chemical potential must satisfy the electron number summed over **all bands and all k-points**:

$$
N_e
=
\sum_{\mathbf k}w_{\mathbf k}
\sum_n f(\epsilon_{n\mathbf k}-\mu).
$$

So diagonalization is embarrassingly parallel over k; then there is one tiny k-reduction per SCC iteration.

That is a beautiful match to your gather philosophy:

```text
(replica,k) owns eigensystem
        ↓
replica gathers q over k
        ↓
one SCC update per replica
```

No atomics necessary.

DFTB+ already supports SCC periodic systems with arbitrary k-point sampling; its standard workflow explicitly determines self-consistent charges using a converged k mesh and then evaluates bands at chosen k points. ([DFTB+][4])

---

# 11. I would implement this in a very specific order

My preferred next scientific expansion would be:

1. **Fragment-charge constrained SCC-DFTB first.** It is tiny architecturally and directly enables the PCET science.
2. Then **total charge + collinear spin**, because radical cations/anions are unavoidable in ET.
3. Then **ΔSCF / occupation-constrained states**, with maximum-overlap tracking.
4. Then **Γ-point PBC**, copying periodic γ/Ewald machinery from Fortran.
5. Only then add **complex arbitrary-k matrices**.
6. Later, if needed, add **CDFTB-CI / small diabatic-state Hamiltonians** and/or matrix-free TD-DFTB.

I think steps 1 and 2 could turn your present solver from “fast ground-state H-bond DFTB” into a genuinely useful **high-throughput PCET diabatic-surface engine** without changing its basic GPU architecture.

And the especially nice thing is that your intended workload—many proton coordinates × several charge-localization states × perhaps several electric fields—is exactly what the batched multisolver is good at:

$$
N_x
\times
N_{\rm charge\ state}
\times
N_{\rm field}
$$

all solved simultaneously on the GPU.

That seems to me a much more compelling scientific use of this architecture than merely making conventional DFTB faster.

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12186617/ "https://pmc.ncbi.nlm.nih.gov/articles/PMC12186617/"
[2]: https://pubs.acs.org/jpcafh/article/110/29/9212/3585938/Direct-Calculation-of-Electron-Transfer-Parameters "https://pubs.acs.org/jpcafh/article/110/29/9212/3585938/Direct-Calculation-of-Electron-Transfer-Parameters"
[3]: https://pubmed.ncbi.nlm.nih.gov/24135026/ "https://pubmed.ncbi.nlm.nih.gov/24135026/"
[4]: https://dftbplus.org/about/features.html "https://dftbplus.org/about/features.html"
[5]: https://pubs.acs.org/doi/10.1021/acs.jpca.5c01146 "https://pubs.acs.org/doi/10.1021/acs.jpca.5c01146"
[6]: https://dftbplus-recipes.readthedocs.io/en/latest/linresp/introduction.html "https://dftbplus-recipes.readthedocs.io/en/latest/linresp/introduction.html"
[7]: https://dftbplus-recipes.readthedocs.io/en/stable/linresp/index.html "https://dftbplus-recipes.readthedocs.io/en/stable/linresp/index.html"
[8]: https://dftbplus-recipes.readthedocs.io/en/latest/linresp/rangeseparated.html "https://dftbplus-recipes.readthedocs.io/en/latest/linresp/rangeseparated.html"
[9]: https://pubmed.ncbi.nlm.nih.gov/21197974/ "https://pubmed.ncbi.nlm.nih.gov/21197974/"






























---

# USER

OK, so for PBC as I understadn is main bottleneck the Complex Hermitian Jacobi solver on GPU right? Can your write a prototype of that kernel, stratging from current real-valued symmetric jacobi kernel we have? Focus on it and write just thate essential piece of OpenCL code, we should implement it as separate branch/separate module... I think it should not be so hard in the end. 


---

# Chat GPT 5.6 sol



Yes. For arbitrary \(k\), this is the main **new eigensolver primitive**. It is not the only PBC work—you also need complex H/S assembly and complex GEMM—but those are mechanically straightforward. The Hermitian eigensolver is the piece where numerical stability and GPU layout matter.

And I agree: **it is not actually a large conceptual change.** Your present direct Jacobi kernel already has almost exactly the right architecture: Brent–Luk disjoint pairs, one WG/system, full A/V in global memory, warm basis, relative pivot skipping, and only ~3 barriers/round.

The real rotation currently is

$$
J=
\begin{pmatrix}
c&s\\
-s&c
\end{pmatrix}.
$$

For a Hermitian pivot

$$
A_{pq}=r u,\qquad
r=|A_{pq}|,\qquad
u=\frac{A_{pq}}{|A_{pq}|},
$$

with complex phase \(|u|=1\), use

$$
J=
\begin{pmatrix}
c&s u\\
-s u^*&c
\end{pmatrix}.
$$

Then

$$
A' = J^\dagger A J
$$

has exactly the same real Jacobi equation for \(c,s\), except that the off-diagonal scalar is \(r=|A_{pq}|\):

$$
\tau=\frac{A_{qq}-A_{pp}}{2|A_{pq}|}.
$$

So almost all scheduling code survives.

I would put this into a completely separate module, e.g.

```text
qmqm/gpu_hermitian_jacobi.cl
```

on a separate experimental branch. Do not contaminate the already-working real kernel until parity/performance is established.

Here is the essential prototype.

```opencl
// ============================================================================
// Complex Hermitian direct cyclic Jacobi
//
// Prototype derived from jacobi_cyclic_global_batched.
//
// Storage:
//   float2 z = (Re z, Im z)
//
// Solves:
//   A <- J^H A J
//   V <- V J
//
// A must be Hermitian.
// V is initialized to I for cold solve, or contains a warm basis for init_v=1.
//
// Bulk matrix updates stay FP32.
// Optional FP64 is used ONLY for scalar c,s construction.
// ============================================================================

#ifndef WG
#define WG 512
#endif

#ifndef MAX_CSWEEPS
#define MAX_CSWEEPS 40
#endif

#ifndef JACOBI_OFF_TOL
#define JACOBI_OFF_TOL 1.0e-6f
#endif

#ifndef PAIR_SKIP_REL
#define PAIR_SKIP_REL 3.0e-8f
#endif

#ifndef HJ_ROT_FP64
#define HJ_ROT_FP64 1
#endif

#if HJ_ROT_FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double jrot_t;
#else
typedef float jrot_t;
#endif


// -----------------------------------------------------------------------------
// Minimal complex arithmetic
// -----------------------------------------------------------------------------

inline float2 cconj_f(float2 a)
{
    return (float2)(a.x, -a.y);
}

inline float2 cmul_f(float2 a, float2 b)
{
    return (float2)(
        fma(a.x, b.x, -a.y * b.y),
        fma(a.x, b.y,  a.y * b.x)
    );
}

inline float2 cscale_f(float s, float2 a)
{
    return (float2)(s * a.x, s * a.y);
}

inline float cabs2_f(float2 a)
{
    return fma(a.x, a.x, a.y * a.y);
}


// Brent-Luk round-robin schedule.
// Same as real kernel.
inline int2 herm_jacobi_pair(int round, int ipair, int jn)
{
    const int m = jn - 1;

    if (ipair == 0)
        return (int2)(m, round % m);

    return (int2)(
        (round + ipair) % m,
        (round + m - ipair) % m
    );
}


// ============================================================================
// jacobi_hermitian_cyclic_global_batched
// ============================================================================

__kernel void jacobi_hermitian_cyclic_global_batched(
    __global float2* A,         // [batch][n*n], Hermitian
    __global float2* V,         // [batch][n*n], eigenvectors in columns
    const int n,
    const int batch,
    const int init_v,           // 0: V=I, 1: retain warm basis
    __global const int* active,
    __global float* diag        // [batch][4]:
                                // {off, off/frob, stop, sweeps}
)
{
    const int sid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    if (sid >= batch || active[sid] == 0)
        return;

    const int jn     = (n & 1) ? n + 1 : n;
    const int jpair  = jn / 2;
    const int jround = jn - 1;

    // same current capacity: n <= 256
    if (jpair > 128) {
        if (lid == 0) {
            diag[4*sid + 0] = INFINITY;
            diag[4*sid + 1] = INFINITY;
            diag[4*sid + 2] = 3.0f;
            diag[4*sid + 3] = 0.0f;
        }
        return;
    }

    __global float2* gA = A + (size_t)sid * n * n;
    __global float2* gV = V + (size_t)sid * n * n;

    const int nn = n * n;

    // Per-pair rotation parameters.
    //
    // J =
    // [  c       s u   ]
    // [ -s u*    c     ]
    //
    __local jrot_t rot_c[128];
    __local jrot_t rot_s[128];

    __local float2 rot_u[128];

    __local int rot_p[128];
    __local int rot_q[128];

    // Avoid even a local atomic increment.
    __local int rot_live[128];

    __local float reduce[WG];


    // -------------------------------------------------------------------------
    // Initialize eigenvectors
    // -------------------------------------------------------------------------

    if (init_v == 0) {
        for (int idx = lid; idx < nn; idx += lsz) {
            const int i = idx / n;
            const int j = idx - i*n;

            gV[idx] = (i == j)
                ? (float2)(1.0f, 0.0f)
                : (float2)(0.0f, 0.0f);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


    // -------------------------------------------------------------------------
    // Frobenius norm + initial off-diagonal norm
    // -------------------------------------------------------------------------

    float fa = 0.0f;
    float fo = 0.0f;

    for (int idx = lid; idx < nn; idx += lsz) {
        const int i = idx / n;
        const int j = idx - i*n;

        const float2 z = gA[idx];
        const float z2 = cabs2_f(z);

        fa += z2;

        if (i != j)
            fo += z2;
    }

    reduce[lid] = fa;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int d = lsz >> 1; d > 0; d >>= 1) {
        if (lid < d)
            reduce[lid] += reduce[lid+d];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float frob = sqrt(fmax(reduce[0], 1.0e-30f));

    barrier(CLK_LOCAL_MEM_FENCE);

    reduce[lid] = fo;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int d = lsz >> 1; d > 0; d >>= 1) {
        if (lid < d)
            reduce[lid] += reduce[lid+d];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float off_cur = sqrt(reduce[0]);

    barrier(CLK_LOCAL_MEM_FENCE);

    const float off_exit = JACOBI_OFF_TOL * frob;

    float prev_off = fmax(off_cur, 1.0e-30f);

    int stall = 0;
    int stop  = 0;
    int nsw   = 0;


    // =========================================================================
    // Jacobi sweeps
    // =========================================================================

    if (off_cur > off_exit) {

        stop = 2;      // max-sweeps unless overwritten

        for (int sweep = 0; sweep < MAX_CSWEEPS; ++sweep) {

            nsw = sweep + 1;

            // =================================================================
            // Brent-Luk rounds
            // =================================================================

            for (int round = 0; round < jround; ++round) {

                // -------------------------------------------------------------
                // Phase 1: build one complex unitary rotation per disjoint pair
                // -------------------------------------------------------------

                for (int ip = lid; ip < jpair; ip += lsz) {

                    const int2 pq =
                        herm_jacobi_pair(round, ip, jn);

                    const int p = min(pq.x, pq.y);
                    const int q = max(pq.x, pq.y);

                    rot_p[ip] = p;
                    rot_q[ip] = q;

                    float2 apq =
                        (q < n)
                        ? gA[p*n + q]
                        : (float2)(0.0f, 0.0f);

                    // Hermitian diagonal must be real.
                    const float app = gA[p*n + p].x;
                    const float aqq =
                        (q < n)
                        ? gA[q*n + q].x
                        : 1.0f;

#if HJ_ROT_FP64

                    const double ar = (double)apq.x;
                    const double ai = (double)apq.y;

                    const double az =
                        sqrt(ar*ar + ai*ai);

                    const double scale =
                        fabs((double)app) + fabs((double)aqq);

                    if (!(az > (double)PAIR_SKIP_REL * scale)) {

                        rot_c[ip] = 1.0;
                        rot_s[ip] = 0.0;

                        rot_u[ip] = (float2)(1.0f, 0.0f);
                        rot_live[ip] = 0;

                    } else {

                        // phase:
                        //
                        //     u = A_pq / |A_pq|
                        //
                        rot_u[ip] = (float2)(
                            (float)(ar / az),
                            (float)(ai / az)
                        );

                        // Exactly the same stable real Jacobi formula
                        // as the current symmetric kernel, but apq -> |apq|.
                        const double tau =
                            ((double)aqq - (double)app)
                            / (2.0 * az);

                        const double t =
                            (tau >= 0.0)
                            ? 1.0 /
                              (tau + sqrt(1.0 + tau*tau))
                            : -1.0 /
                              (-tau + sqrt(1.0 + tau*tau));

                        const double c =
                            1.0 / sqrt(1.0 + t*t);

                        rot_c[ip] = c;
                        rot_s[ip] = t*c;

                        rot_live[ip] = 1;
                    }

#else

                    const float az2 = cabs2_f(apq);
                    const float az  = sqrt(az2);

                    const float scale =
                        fabs(app) + fabs(aqq);

                    if (!(az > PAIR_SKIP_REL * scale)) {

                        rot_c[ip] = 1.0f;
                        rot_s[ip] = 0.0f;

                        rot_u[ip] = (float2)(1.0f, 0.0f);
                        rot_live[ip] = 0;

                    } else {

                        rot_u[ip] = apq / az;

                        const float tau =
                            (aqq - app) / (2.0f * az);

                        const float t =
                            (tau >= 0.0f)
                            ? 1.0f /
                              (tau + sqrt(1.0f + tau*tau))
                            : -1.0f /
                              (-tau + sqrt(1.0f + tau*tau));

                        const float c =
                            rsqrt(1.0f + t*t);

                        rot_c[ip] = c;
                        rot_s[ip] = t*c;

                        rot_live[ip] = 1;
                    }

#endif
                }

                barrier(CLK_LOCAL_MEM_FENCE);


                // No atomic.
                //
                // jpair <= 128, so thread 0 can cheaply check whether the
                // complete round is identity.
                __local int any_rotation;

                if (lid == 0) {

                    int any = 0;

                    for (int ip = 0; ip < jpair; ++ip)
                        any |= rot_live[ip];

                    any_rotation = any;
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                if (!any_rotation)
                    continue;


                // -------------------------------------------------------------
                // Phase 2:
                //
                // Apply A <- J^H A J
                //
                // Pair a transforms rows (pa,qa).
                // Pair b transforms columns (pb,qb).
                //
                // Each (a,b) owns a disjoint 2x2 block.
                // -------------------------------------------------------------

                const int nblk = jpair * jpair;
                const int nvec = n * jpair;

                for (int it = lid;
                     it < nblk + nvec;
                     it += lsz)
                {
                    // =========================================================
                    // Matrix A update
                    // =========================================================

                    if (it < nblk) {

                        const int a = it / jpair;
                        const int b = it - a*jpair;

                        const float sa = (float)rot_s[a];
                        const float sb = (float)rot_s[b];

                        if (sa == 0.0f && sb == 0.0f)
                            continue;

                        const float ca = (float)rot_c[a];
                        const float cb = (float)rot_c[b];

                        const float2 ua  = rot_u[a];
                        const float2 uac = cconj_f(ua);

                        const float2 ub  = rot_u[b];
                        const float2 ubc = cconj_f(ub);

                        const int pa = rot_p[a];
                        const int qa = rot_q[a];

                        const int pb = rot_p[b];
                        const int qb = rot_q[b];


                        const float2 apr =
                            gA[pa*n + pb];

                        const float2 aps =
                            (qb < n)
                            ? gA[pa*n + qb]
                            : (float2)(0.0f,0.0f);

                        const float2 aqr =
                            (qa < n)
                            ? gA[qa*n + pb]
                            : (float2)(0.0f,0.0f);

                        const float2 aqs =
                            (qa < n && qb < n)
                            ? gA[qa*n + qb]
                            : (float2)(0.0f,0.0f);


                        // -----------------------------------------------------
                        // Right transform:
                        //
                        // [A_rp A_rq] J_b
                        //
                        // J_b =
                        // [ cb       sb ub  ]
                        // [-sb ub*    cb     ]
                        // -----------------------------------------------------

                        const float2 tpr =
                            cscale_f(cb, apr)
                            - cscale_f(
                                sb,
                                cmul_f(ubc, aps)
                              );

                        const float2 tps =
                            cscale_f(
                                sb,
                                cmul_f(ub, apr)
                              )
                            + cscale_f(cb, aps);

                        const float2 tqr =
                            cscale_f(cb, aqr)
                            - cscale_f(
                                sb,
                                cmul_f(ubc, aqs)
                              );

                        const float2 tqs =
                            cscale_f(
                                sb,
                                cmul_f(ub, aqr)
                              )
                            + cscale_f(cb, aqs);


                        // -----------------------------------------------------
                        // Left transform:
                        //
                        // J_a^H =
                        //
                        // [ ca      -sa ua  ]
                        // [ sa ua*   ca     ]
                        // -----------------------------------------------------

                        const float2 opr =
                            cscale_f(ca, tpr)
                            - cscale_f(
                                sa,
                                cmul_f(ua, tqr)
                              );

                        const float2 ops =
                            cscale_f(ca, tps)
                            - cscale_f(
                                sa,
                                cmul_f(ua, tqs)
                              );

                        const float2 oqr =
                            cscale_f(
                                sa,
                                cmul_f(uac, tpr)
                              )
                            + cscale_f(ca, tqr);

                        const float2 oqs =
                            cscale_f(
                                sa,
                                cmul_f(uac, tps)
                              )
                            + cscale_f(ca, tqs);


                        gA[pa*n + pb] = opr;

                        if (qb < n)
                            gA[pa*n + qb] = ops;

                        if (qa < n) {
                            gA[qa*n + pb] = oqr;

                            if (qb < n)
                                gA[qa*n + qb] = oqs;
                        }
                    }


                    // =========================================================
                    // Eigenvectors:
                    //
                    // V <- V J
                    //
                    // V'_kp = c V_kp - s u* V_kq
                    // V'_kq = s u V_kp + c V_kq
                    // =========================================================

                    else {

                        const int t = it - nblk;

                        const int k = t / jpair;
                        const int a = t - (t / jpair)*jpair;

                        const float s = (float)rot_s[a];

                        if (s == 0.0f)
                            continue;

                        const float c = (float)rot_c[a];

                        const float2 u  = rot_u[a];
                        const float2 uc = cconj_f(u);

                        const int p = rot_p[a];
                        const int q = rot_q[a];

                        const float2 vp = gV[k*n + p];

                        const float2 vq =
                            (q < n)
                            ? gV[k*n + q]
                            : (float2)(0.0f,0.0f);

                        const float2 np =
                            cscale_f(c, vp)
                            - cscale_f(
                                s,
                                cmul_f(uc, vq)
                              );

                        const float2 nq =
                            cscale_f(
                                s,
                                cmul_f(u, vp)
                              )
                            + cscale_f(c, vq);

                        gV[k*n + p] = np;

                        if (q < n)
                            gV[k*n + q] = nq;
                    }
                }

                barrier(
                    CLK_LOCAL_MEM_FENCE |
                    CLK_GLOBAL_MEM_FENCE
                );
            }


            // =================================================================
            // f32 restoring invariant:
            //
            // A <- (A + A^H)/2
            //
            // This is O(N²) once per O(N³) Jacobi sweep, therefore cheap.
            //
            // It prevents roundoff drift from slowly violating the assumption
            // that the next pivot is Hermitian.
            // =================================================================

            for (int idx = lid; idx < nn; idx += lsz) {

                const int i = idx / n;
                const int j = idx - i*n;

                if (i == j) {

                    float2 z = gA[idx];
                    gA[idx] = (float2)(z.x, 0.0f);

                } else if (i < j) {

                    const float2 aij = gA[i*n + j];
                    const float2 aji = gA[j*n + i];

                    const float2 z =
                        0.5f * (aij + cconj_f(aji));

                    gA[i*n + j] = z;
                    gA[j*n + i] = cconj_f(z);
                }
            }

            barrier(
                CLK_LOCAL_MEM_FENCE |
                CLK_GLOBAL_MEM_FENCE
            );


            // =================================================================
            // Off-diagonal norm
            // =================================================================

            fo = 0.0f;

            for (int idx = lid; idx < nn; idx += lsz) {

                const int i = idx / n;
                const int j = idx - i*n;

                if (i != j)
                    fo += cabs2_f(gA[idx]);
            }

            reduce[lid] = fo;

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int d = lsz >> 1; d > 0; d >>= 1) {

                if (lid < d)
                    reduce[lid] += reduce[lid+d];

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            off_cur = sqrt(reduce[0]);

            barrier(CLK_LOCAL_MEM_FENCE);


            // =================================================================
            // Same convergence policy as real solver
            // =================================================================

            if (!isfinite(off_cur)) {
                stop = 4;
                break;
            }

            if (off_cur <= off_exit) {
                stop = 0;
                break;
            }

            if (off_cur > 0.9f * prev_off) {

                if (++stall >= 3) {
                    stop = 1;
                    break;
                }

            } else {

                stall = 0;
            }

            prev_off = off_cur;
        }
    }


    // -------------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------------

    if (lid == 0) {

        diag[4*sid + 0] = off_cur;
        diag[4*sid + 1] = off_cur / frob;
        diag[4*sid + 2] = (float)stop;
        diag[4*sid + 3] = (float)nsw;
    }
}
```

There are a few deliberate choices here.

First, **I kept exactly the current production architecture** rather than trying to be clever immediately. Your current real kernel performs one Brent–Luk rotation for each disjoint pair and updates all pair×pair 2×2 blocks plus the eigenvector columns in one phase.  The complex version above does exactly the same thing.

Second, the complex part is surprisingly small. Each pivot adds only one number:

```opencl
float2 u = apq / |apq|;
```

and the \(c,s\) calculation is identical to the real case with

```text
apq → |apq|
```

Everything else is just replacing real multiply-adds by complex multiply-adds.

### The Hermiticity restoration is intentional

I added

$$
A\leftarrow\frac12(A+A^\dagger)
$$

once per sweep.

Mathematically it should be unnecessary. Numerically, in f32, I think it is exactly the kind of **cheap restoring invariant** you want.

It costs only \(O(N^2)\) per sweep while the Jacobi transformation costs \(O(N^3)\), so for \(N\sim86\) it is a few percent at worst and probably much less.

It guarantees:

$$
A_{ji}=A_{ij}^*
$$

and

$$
\Im A_{ii}=0
$$

before the next sweep. I would A/B it later, but I would **start with it enabled for correctness**.

### I also removed the local atomic

Your real implementation currently uses

```opencl
atomic_inc(&l_nrot);
```

for the number of live rotations.

It is only a local-memory integer atomic and therefore nowhere near as bad as the global force atomics we hate, but there is still no reason for it here.

There are at most 128 pairs. Just write:

```opencl
rot_live[ip]
```

and let lane 0 inspect 44–128 integers. That's microscopic.

---

## Most important mathematical check

The convention in the prototype is:

$$
A_{pq}=r u
$$

and

$$
J=
\begin{pmatrix}
c&s u\\
-s u^*&c
\end{pmatrix}.
$$

For the 2×2 Hermitian pivot

$$
A=
\begin{pmatrix}
a&r u\\
r u^*&d
\end{pmatrix},
$$

the transformed off-diagonal becomes

$$
A'_{pq}
=
u\left[
cs(a-d)
+
(c^2-s^2)r
\right].
$$

So zeroing it requires precisely the ordinary real Jacobi equation

$$
\tau=\frac{d-a}{2r},
$$

which is what the code uses.

This is useful because it means there is no obscure complex eigensolver mathematics hiding here. We literally remove the phase of the complex pivot, perform the familiar real Jacobi rotation, and carry the phase through the unitary transformation.

---

## Testing should start with three absolutely minimal cases

Before integrating anything with DFTB/PBC, I would test this standalone against CPU LAPACK/Nalgebra/NumPy equivalent for random Hermitian matrices.

The important quantities are

$$
\frac{\|AV-V\Lambda\|_F}{\|A\|_F},
$$

$$
\|V^\dagger V-I\|_F,
$$

and eigenvalue parity.

But there are three particularly useful constructed tests.

A pure phase 2×2:

$$
A=
\begin{pmatrix}
1 & 0.2e^{i1.3}\\
0.2e^{-i1.3}&2
\end{pmatrix}.
$$

This catches immediately whether the \(u/u^*\) signs are wrong.

Then a real symmetric matrix stored as complex with

$$
\Im A=0.
$$

The new kernel **must reproduce the existing real kernel**. That's perhaps the strongest regression test because we already trust the real implementation.

And finally a synthetic Bloch Hamiltonian obeying

$$
H(-k)=H(k)^*.
$$

Therefore the eigenspectra must satisfy

$$
\epsilon_n(-k)=\epsilon_n(k).
$$

That is an excellent PBC-specific physical invariant.

---

## What comes immediately after this kernel

The generalized eigenproblem is

$$
H(k)C(k)=S(k)C(k)\epsilon(k).
$$

So this exact same kernel can first diagonalize complex Hermitian \(S(k)\):

$$
S=U\Lambda U^\dagger.
$$

Then

$$
X=U\Lambda^{-1/2}U^\dagger,
$$

and

$$
H'=X^\dagger H X.
$$

Then run the same Hermitian Jacobi again on \(H'\).

Thus the other primitive you need is only:

```text
complex batched GEMM
```

which is much less intellectually risky than the eigensolver. Your current tiled GEMM architecture can be converted from `float` to `float2` with four-real-FMA complex products.

So I would make the experimental module very narrowly:

```text
gpu_hermitian_jacobi.cl
test_gpu_hermitian_jacobi.rs
```

and nothing about PBC initially.

Prove:

```text
random Hermitian
real-matrix reduction
phase 2x2
clustered eigenvalues
N=86/87/128
warm-start
```

Then benchmark **real Jacobi vs complex Jacobi on the same real matrix**. That gives us a clean number for the cost of arbitrary-k PBC before touching any PBC code.

My guess is that the complex solver will be perhaps ~2–3× the real one rather than anything catastrophic. More arithmetic is required, but the schedule, synchronization count and number of Jacobi rotations are unchanged—and those synchronization/rotation costs are a large fraction of your present real kernel.
