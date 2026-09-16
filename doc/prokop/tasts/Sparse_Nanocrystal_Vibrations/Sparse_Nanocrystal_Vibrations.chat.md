https://chatgpt.com/share/6a9ed2c7-b430-83ed-9d13-b0213f016bab

---

# USER

in this repo we recently made sparse density matrix purification GPU accelerated solver fr DFTB+ hamiltonian.
https://github.com/ProkopHapala/dftbplus

Now we want to use it for solving vibation speactra of silicon nanocrystals...


we should now focus how to optimize it further, but also on practicallity, what are the limits of single precission numerical accuracy, how we can use it for solving hessian (2nd derivative, wich is more sensitive for non-smooth numerical noise etc.), we must make sure we can converge the crystal sufficienly that there will not be some nasty imaginary frequencies and the whole fing will converged to reasonable spectra. I'm not obsessed with accuracy like I'm fine if our spectra have 5% error in the frequency or curvature of hamiltonina, but it mut be reasonable and physically menaingull, not vontain some imaginary frequecies or some wiggly noise. Wit that perhaps also the way we represent and intrpolate the absis function is critical, not sure if the B-spline or Hermite spline (cubic, quintic) is best, or of we should represent them by some polynominal globally ( like (1-c*x)^(2^m) basis is quite fast to evaluate and often represent slater-like exp(-bx) well with few terms but in the end I realized the splines are still usually more effiicent in pracitice if hogh accuracy is the goal)

these numerical aspect are one thing, the second thing is to define clear step by step testing prodocosl form basic test (building on what we already did for PAH) and the goin up to application production runs getting spectra fo large crystals and corelating vibration with different faces, groups etc (but this analysis is out of the scope it is already impleemnted in FireCore we do not nee to think about it now it is just for the context, bigger picture so you know where we are heading)

---

# Chat GPT 5.6 sol

Yes. I think this is a very promising application for the sparse solver, but I would substantially revise the current manifest before we start implementing the nanocrystal pipeline. The decisive numerical question is not simply “is f32 accurate enough?” It is:

$$
\boxed{\text{Can we make } \mathbf R \mapsto \mathbf F(\mathbf R)
\text{ sufficiently smooth, deterministic, and variationally consistent?}}
$$

If we do that, I think f32 for the expensive sparse matrix algebra is quite plausible for your target of roughly 5% frequency accuracy. If we do not, even f64 will not save us from spline knots, changing sparsity masks, incompletely converged SCC, or nested finite differences.

The manifest already has the right central validation idea: compare Hessians at **identical geometries first**, and only afterwards compare spectra at each implementation's own relaxed minimum.  The FireCore lesson later in the document—that the Hessian must be evaluated at the minimum of the same potential—is also exactly right. 

## Several things in the manifest should be changed first

The biggest conceptual correction concerns sparsity. Your kernel explicitly computes

$$
C=P_M(AB)
$$

with an externally prescribed output mask; blocks outside it are neither computed nor allocated.  Therefore a test such as

> does `nnz(K)` grow during purification?

doesn't really test anything anymore. **Structural nnz cannot grow.** It is fixed by construction.

The real physical question is:

$$
\boxed{\text{How large must }R_K\text{ and }R_Z\text{ be for the truncation error to be acceptable?}}
$$

I would distinguish at least three supports:

$$
R_{HS} < R_K,\qquad R_{HS}<R_Z
$$

where \(R_{HS}\) is the true Slater–Koster interaction range, \(R_K\) is density-kernel range, and \(R_Z\) is the range used for the approximate \(S^{-1}\). The product/intermediate mask may need to be somewhat wider still.

For an insulating H-passivated Si nanocrystal this is actually a favorable problem because the density matrix should decay exponentially. But locality will worsen when the HOMO-LUMO gap gets small—particularly for poorly passivated surfaces or dangling bonds. So **gap versus required \(R_K\)** is itself a useful numerical diagnostic.

Also, idempotency *inside the truncated mask* is not sufficient. One can have

$$
P_M(KSK-K)\approx0
$$

while the omitted components outside \(M\) are large. I would evaluate the residual periodically on a **larger validation mask**, say \(R_\mathrm{validate}=R_K+2\!-\!4\) Å, and measure leakage there.

There are several other corrections worth making:

| Current assumption                                  | What I would change                                                                                                                                                    |     |                                                        |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | ------------------------------------------------------ |
| Monitor `nnz(K_iter)` for densification             | Sweep \(R_K,R_Z\); monitor effective block norms and leakage outside the production mask                                                                               |     |                                                        |
| H can be padded to BSR4 with zero dummy orbitals    | **Not with \(S_\mathrm{dummy}=0\)**: then \(S\) is singular and Newton-Schulz \(S^{-1}\) is mathematically undefined                                                   |     |                                                        |
| Entire SCC is O(N)                                  | Sparse electronic part can be O(N), but dense \(\gamma\,\Delta q\) remains O(N²); for N≈1000 this is probably cheap enough to leave dense                              |     |                                                        |
| Every residual should decrease monotonically        | DIIS/mixing can legitimately oscillate; require final convergence and absence of sustained divergence instead                                                          |     |                                                        |
| No negative modes whatsoever                        | Project rigid modes and tolerate numerical (                                                                                                                           | \nu | \sim5-15) cm⁻¹; reject **significant** imaginary modes |
| `deltaXDiff ≈6e-5 Å` is the relevant phonon FD step | In the current Rust force implementation that tiny number is documented as the internal \(dH/dR,dS/dR\) finite-difference step, not a sensible f32 phonon displacement |     |                                                        |
| Phase 0 implements a Si nanocrystal builder         | This contradicts the later correct policy: use FireCore and only consume XYZs                                                                                          |     |                                                        |

There is also an important current implementation limitation: the production sparse entry point still constructs `n_orbs = 4*n_atom`, rejects a different orbital count, and starts from a full BSR4 mask.  So before discussing Si\(_{1000}\)H\(_m\), we need a genuinely sparse end-to-end route rather than only a sparse purification kernel embedded in a dense workflow.

---

## The H-passivation issue is actually blocking

The manifest currently suggests simply padding H's one \(1s\) orbital to four orbitals and accepting the waste.  The wasted memory is not the serious problem. The serious problem is that padding with three **zero-overlap** dummy orbitals makes \(S\) singular.

There are two sensible approaches.

For a quick prototype, I would keep BSR4 and embed H into four dimensions, but give the three dummy dimensions something like

$$
S_{dd}=1,\qquad H_{dd}=E_\mathrm{dummy},
$$

with zero physical couplings and \(E_\mathrm{dummy}\) safely above the occupied spectrum. Electron counts, Mulliken populations and physical residuals must ignore these dimensions. This lets the existing BSR4 machinery work, although a very high \(E_\mathrm{dummy}\) unnecessarily expands the spectral interval and can slow purification.

For production I would eventually prefer block types \(4\times4,4\times1,1\times4,1\times1\). Interestingly, your GPU Hamiltonian assembler already thinks in exactly these block types.  That suggests a natural path toward a variable-block sparse representation later.

I would not dodge the issue by initially using bare Si clusters for too long: dangling-bond surface states are precisely what can close the gap and make density-matrix locality much worse. A passivated Si crystal is numerically a much cleaner target.

---

# For phonons I would change the interpolation

This is probably the most important part of your question.

The current CPU SK representation is **cubic Hermite using values plus estimated first derivatives**. It is \(C^1\): value and first derivative are continuous, but its second derivative generally jumps at every grid knot.

For energies and forces that is quite reasonable.

For a Hessian,

$$
H_{\alpha\beta}
=-\frac{\partial F_\alpha}{\partial R_\beta},
$$

it is not ideal. The force is continuous but its slope has tiny discontinuities whenever an interatomic distance moves through a table knot. A finite difference straddling one of those knots can therefore produce exactly the sort of artificial curvature/wiggle you are worried about.

Curiously, you already have most of the solution: `spline_resample.rs` builds a **natural cubic spline and converts it into cubic B-spline control points**, while the GPU Hamiltonian uses a four-point cubic B-spline stencil.

I would standardize the whole codebase around a **C² cubic representation**.

In fact I don't care strongly whether we call it a cubic B-spline or precomputed piecewise cubic spline. After preprocessing, each interval can simply have

$$
V(t)=a+bt+ct^2+dt^3,
$$

so that

$$
V'=\frac{b+2ct+3dt^2}{\Delta r},
\qquad
V''=\frac{2c+6dt}{\Delta r^2}.
$$

That is extremely cheap. It gives us \(V,V',V''\) consistently from the exact same coefficients.

I would prefer this over quintic Hermite initially. Quintic Hermite gives you more smoothness, but now you need reliable second derivatives at every knot; estimating those from tabulated data creates another noise source. A well-constructed cubic \(C^2\) spline is almost certainly enough.

And I would strongly prefer it over a global basis such as your

$$
(1-cx)^{2^m}.
$$

That basis is attractive computationally and resembles exponentials surprisingly well, but global fitting makes curvature errors harder to localize, different SK channels have rather different shapes, and enforcing a clean finite cutoff with \(V=V'=V''=0\) becomes awkward. It would be a nice **compression experiment later**, not the phonon reference representation.

There is no strong reason to aggressively compress to 32 SK nodes for this application either. For the single-large-system sparse solver, SK interpolation is tiny compared with purification. If 128 or 256 nodes makes \(V''\) much better, use them. The correct test is not merely error in \(V\); measure

$$
\Delta V,\qquad \Delta V',\qquad \Delta V''
$$

over the actual Si-Si and Si-H bond-distance windows.

The spline tail deserves special attention. Ideally the cutoff should satisfy

$$
V(r_c)=V'(r_c)=V''(r_c)=0.
$$

The existing quintic `poly5_to_zero` idea is actually excellent for that purpose. I would use a C² cubic interior joined continuously to the quintic tail.

---

## Do not finite-difference the SK tables inside the force

This is another place where the code is currently in an intermediate state.

The existing top-level CPU force path still contains the DFTB+-style central difference of H/S blocks using approximately \(6.3\times10^{-5}\) Å.  Yet the repository also already contains analytic Slater–Koster rotation derivatives in `rotation.rs`.  So the analytic machinery exists but is not consistently wired into the current force path.

For the GPU sparse implementation I would absolutely not reproduce

$$
R\rightarrow
\frac{H(R+\delta)-H(R-\delta)}{2\delta}
\rightarrow F
\rightarrow
\frac{F(R+h)-F(R-h)}{2h}
\rightarrow H_\mathrm{nuclear}.
$$

That is a nested finite difference and is particularly nasty in f32.

Instead:

$$
V(r),V'(r)
\quad\longrightarrow\quad
\frac{\partial H_{\mu\nu}}{\partial R_{i\alpha}},
\frac{\partial S_{\mu\nu}}{\partial R_{i\alpha}}
$$

should be analytic, and the **only finite difference should be the final force → Hessian difference**.

That simultaneously makes the force faster and much smoother.

---

## There is a nice way to avoid diagonalization for the Pulay/EDM term

This deserves highlighting because it fits your sparse-purification architecture beautifully.

The force requires both the density matrix \(D\) and the energy-weighted density matrix \(W\). The dense code currently gets \(W\) from occupied eigenvectors/eigenvalues.

But if the purification kernel returns the spinless occupied projector \(K\),

$$
K=C_\mathrm{occ}C_\mathrm{occ}^{T},
\qquad KSK=K,
$$

then

$$
K H K
=
C_\mathrm{occ}
(C_\mathrm{occ}^{T}HC_\mathrm{occ})
C_\mathrm{occ}^{T}
=
C_\mathrm{occ}\epsilon C_\mathrm{occ}^{T}.
$$

Therefore, for closed shell,

$$
\boxed{D=2K,\qquad W=2KHK.}
$$

So the sparse analytic force can remain completely diagonalization-free. \(KHK\) is just sparse matrix multiplication, on the same mask infrastructure you already have.

That is, in my view, one of the key architectural pieces missing from the current manifest.

---

# What f32 really means for the Hessian

The manifest's statement that “f32 forces have about \(10^{-4}\) relative precision” is too pessimistic in one sense and too optimistic in another. IEEE f32 has machine epsilon

$$
\epsilon \approx1.19\times10^{-7},
$$

but the actual force error will be dominated by:

* density-kernel truncation,
* approximate \(S^{-1}\),
* SCC nonconvergence,
* spline/compression error,
* reduction error,
* different purification branch histories,
* and finally arithmetic roundoff.

So we should **measure force noise directly**, not derive it from machine epsilon.

For the central Hessian,

$$
H_{ji}
=
-\frac{F_j(R_i+h)-F_j(R_i-h)}{2h},
$$

a force error \(\sigma_F\) contributes roughly

$$
\sigma_H\sim \frac{\sigma_F}{h}.
$$

Meanwhile truncation error behaves as \(O(h^2)\). Therefore the total error has the familiar shape

$$
\epsilon_H(h)
\sim Ah^2+\frac{\sigma_F}{h}.
$$

There will be a broad optimum. For this reason I would not decide today that \(h=10^{-3}\) Å. I expect something more like **0.003–0.02 Å** may be optimal in f32, but we should determine it experimentally.

A very useful calibration is also the five-point stencil

$$
\frac{\partial F}{\partial x}
\approx
\frac{-F(x+2h)+8F(x+h)-8F(x-h)+F(x-2h)}
{12h}.
$$

It doubles the number of force evaluations but changes truncation from \(O(h^2)\) to \(O(h^4)\), so it permits a larger \(h\). I would not automatically use it in production, but it is an excellent reference for deciding whether the three-point Hessian is limited by curvature or by numerical noise.

### Mixed precision is probably the sweet spot

I would keep **all expensive SpGEMM/purification arithmetic f32**, but there is little reason to keep every tiny reduction in f32.

For N≤1000 I would use f64 for trace/residual reductions, SCC charge/potential accumulation, total-energy accumulation, pair-force accumulation if convenient, coordinates/displacements, Hessian assembly, mass weighting, and the final eigensolve.

Even on a GPU with terrible fp64 throughput this costs very little because the expensive part remains f32 matrix multiplication.

You already found a very concrete example of why this matters: the sparse inverse residual previously evaluated an algebraically equivalent expression with catastrophic f32 cancellation and falsely returned zero; it was fixed by accumulating the residual directly.  That is precisely the philosophy I would follow throughout the phonon path: **f32 matrices, numerically stable diagnostics.**

---

# Geometry convergence should be tied to the measured force floor

I would not initially decree \(F_\max<10^{-4}\) Ha/Å. That is about \(2.7\times10^{-3}\) eV/Å, which is reasonable, but only if the sparse/f32 force itself is reproducible significantly below that level.

First determine

$$
\sigma_F =
\left|\mathbf F_\mathrm{sparse32}
-\mathbf F_\mathrm{dense64}\right|
$$

at a collection of nearly relaxed geometries.

If the numerical floor is, say, \(3\times10^{-5}\) Ha/Å, then \(10^{-4}\) is a sensible optimizer target. If the floor is \(3\times10^{-4}\), attempting \(10^{-4}\) will just cause FIRE/L-BFGS to chatter.

For optimization I would use **FIRE initially and L-BFGS near the minimum**. FIRE is excellent for ugly initial structures; L-BFGS is much more efficient once we are in a near-quadratic basin.

And I would keep the sparse topology frozen near convergence. A block appearing/disappearing because a pair crossed \(R_K\) is a discontinuity in the numerical potential. Use a geometric skin, freeze the mask during the final optimization and the entire Hessian calculation, and distinguish:

$$
R_\mathrm{physical}
<
R_\mathrm{mask}.
$$

The `bsr4_drop_small` kernel in your uploaded code is explicitly diagnostic—it zeros numerical blocks but does not compact CSR.  I would keep it that way for phonons. Dynamic threshold-based sparsity is dangerous because it makes the force non-smooth.

---

# We should project rigid modes rather than demanding them to emerge perfectly

For a free nonlinear nanocrystal there are six exact zero modes. Numerical finite differences will mix tiny translation/rotation components into the Hessian.

I would first symmetrize

$$
H\leftarrow\frac12(H+H^T),
$$

while recording

$$
\eta_\mathrm{asym}
=
\frac{\|H-H^T\|_F}{\|H\|_F}
$$

as a very useful measure of numerical force noise.

Then explicitly construct the six mass-weighted rigid-body vectors, orthonormalize them into \(Q\), and project

$$
P=I-QQ^T,\qquad
H_\mathrm{phys}=P^T H P.
$$

This does **not** hide a real instability. It removes exactly the analytically known null space.

After projection I would not insist on “absolutely no negative eigenvalue.” Something corresponding to \(-3\) or \(-8\) cm⁻¹ is numerical zero. What we should reject is, for example, a \(-50\) or \(-100\) cm⁻¹ localized surface mode which does not exist in the f64 reference.

This also means the manifest's “six zero modes <10 cm⁻¹” should be recorded on the **unprojected Hessian as a diagnostic**, while the projected matrix is what we use for the physical spectrum.

---

# The sparse GPU kernel still has several high-value optimizations

Your symmetric-right SpGEMM already improves the inner product by intersecting sorted neighbor lists instead of binary-searching every \(A_{ik}\), which is the right direction.  For this application, though, the structure stays almost fixed for thousands of purification operations and thousands of displaced geometries. That makes **symbolic precomputation** exceptionally attractive.

Instead of intersecting neighbor lists during every matrix product, precompute for every output block \(C_{ij}\) something equivalent to

$$
[(a_1,b_1),(a_2,b_2),\ldots]
$$

giving exactly which stored \(A_{ik}\) and \(B_{kj}\) blocks contribute. The numerical kernel then becomes almost pure 4×4 FMAs.

This is more memory, but your coordination is small. I suspect it could be one of the largest remaining speedups.

The second obvious optimization is row-degree specialization. The kernel currently allocates local memory according to a compile-time `MAX_LEFT_BLOCKS`, even though a physical Si density kernel may have row degrees of only tens of atoms. I would build kernel variants roughly 32/64/128/256 and bucket rows. That improves occupancy and avoids reserving ~17 KiB when only a fraction is required.

I would also benchmark a more radical 4-thread-per-block design. Currently 16 lanes compute one scalar of a 4×4 output each. Four threads, each holding four accumulators in registers, could potentially reduce redundant right-block loads. I would treat this purely as a benchmark candidate, not assume it wins.

Before any of these microoptimizations, however, the larger architectural issues win: current geometric-mask construction is O(N²), product-mask construction is not particularly efficient, and some spectral-bound helpers still densify internally.  Those should disappear before we advertise end-to-end linear scaling.

---

# A testing protocol I would actually use

1. **Gate A — establish one mathematically consistent DFTB model.** Use Si\(_2\), a tiny Si cluster, and one small carbon/PAH system already known to work. Force CPU and GPU to use the **same C² SK spline representation**, same repulsive spline and same cutoff behavior. Scan every SK channel across \(r\), plotting errors in \(V,V',V''\) against the original table/reference. Numerically inspect every spline knot and the cutoff for derivative jumps. Do this before sparse/Hessian work.

2. **Gate B — analytic-force parity before f32 Hessians.** Wire the existing analytic SK/rotation derivatives into the actual production force path. Compare analytic f64 forces against central finite differences of total f64 energies, and against DFTB+ where possible. Then implement the sparse force using \(D=2K\) and \(W=2KHK\). Require force parity on the PAH tests first. The Hessian work should not begin while the force itself contains an internal finite difference.

3. **Gate C — establish density-kernel locality rather than “nnz preservation.”** For increasingly large insulating test systems, sweep \(R_Z\) and \(R_K\), for example by coordination shells or roughly 4, 6, 8, 10, 12 Å. At each point compare energy, Mulliken charges, forces, \(Tr(KS)\), idempotency, Hamiltonian commutator, and residual leakage on a larger validation mask against dense f64. Record wall time and the HOMO-LUMO gap. This produces the genuinely interesting plot: **accuracy and cost versus density-kernel radius**.

4. **Gate D — solve mixed Si/H basis cleanly.** Introduce the temporary nonsingular padded-H embedding or variable blocks, then reproduce the Gate-C tests on a small H-passivated Si nanocrystal. This must happen before claiming the approach works for realistic Si NCs. Also compare the required \(R_K\) for bare versus H-passivated structures; I expect the passivated system to be markedly better behaved.

5. **Gate E — characterize the force noise floor.** On one fixed near-equilibrium geometry, evaluate dense-f64 and sparse-f32 forces for a systematic displacement scan. For each coordinate use \(h=\{0.001,0.002,0.005,0.01,0.02,0.04\}\) Å. Track the antisymmetric Hessian norm, elementwise Hessian error and eigenvalue error. Compare three-point and five-point derivatives. There should be a visible plateau where changing \(h\) by ×2 changes the physically relevant frequencies by only ≈1–2%. That plateau, rather than a theoretical rule of thumb, determines production \(h\).

6. **Gate F — optimize a small Si/H crystal to its own minimum.** Use sparse f32 SCC and forces, initially FIRE and finally L-BFGS. Separately optimize the same small system using the dense f64 route. Do **not** compare Hessians across the two final geometries yet; first ensure each optimizer reaches a stationary point according to its own model. Near the claimed minimum, perform a few random \(\pm\delta\mathbf R\) probes and verify positive local curvature. Then compute a few lowest Hessian-vector modes before paying for the complete Hessian.

7. **Gate G — Hessian parity at identical geometry.** This is the test already correctly envisioned in the manifest.  Compute f64 dense and f32 sparse Hessians at exactly the same frozen coordinates. Compare not only max element error but \(\|ΔH\|_F/\|H\|_F\), antisymmetric component, frequencies, and mode overlaps. Near degeneracies, compare the subspace rather than pairing individual eigenvectors by index. A useful acceptance target is <5% for ordinary modes; for modes below perhaps 50–100 cm⁻¹ use an absolute tolerance instead.

8. **Gate H — spectra at their own minima.** Now optimize each method independently, compute the projected Hessian at its own minimum, and compare spectra. Tiny negatives below roughly 10–20 cm⁻¹ are numerical zero; anything appreciably imaginary should trigger diagnostics identifying the mode, atoms involved, force tolerance, SCC residual, \(R_K\), and \(h\). This is where we establish the cheapest settings that reliably produce a sensible spectrum.

9. **Gate I — scaling benchmark.** Use perhaps N≈60, 150, 300, 600, 1000, 1600 rather than only three points. Separately time H/S assembly, \(\gamma\Delta q\), \(S^{-1}\), K initialization, each SpGEMM, complete purification, SCC, analytic force and Hessian. Report H/S structural nnz, K/Z mask nnz, effective block norm distribution versus distance, symbolic-plan contribution count, iteration counts and memory. Fit the scaling exponent only after the small-N launch-overhead regime. The existing manifest correctly expects Hessian construction to become roughly quadratic if each force solve is linear. 

10. **Gate J — first real production calculation.** I would make ~300 atoms the first genuine success criterion, not Si\(_{1000}\). Optimize it, run a cheap low-mode Hessian-vector/Lanczos check to make sure there is no instability, then compute the full Hessian. Only after this is boringly reliable go to 800–1000 atoms. Every result should record the git commit, SK-set hash, spline representation/node count, \(R_{HS},R_K,R_Z\), SCC criterion, optimizer tolerance, Hessian \(h\), GPU and precision policy.

## One more potentially large optimization: reuse across Hessian displacements

The \(6N\) displaced calculations are not really 6000 unrelated calculations.

For \(R_{i\alpha}\pm h\):

* the sparse structure can be identical;
* symbolic SpGEMM plans are identical;
* only H/S blocks touching displaced atom \(i\) change;
* only one row/column of pairwise \(\gamma_{ij}\) distances changes;
* the central converged **charges** are an excellent SCC initial guess;
* \(S^{-1}\) from the central geometry is an excellent Newton-Schulz initial guess for the slightly perturbed \(S\).

This is a major opportunity.

One subtle point: I would **not** simply warm-start TC2 with the previous converged density projector and apply the same purification. A pure projector polynomial cannot in general rotate an old occupied subspace into the eigenspace of the new Hamiltonian. Charges can safely be warm-started; \(Z\) can be Newton-Schulz-updated because the current \(S\) enters the iteration; but K requires information from the **current H**.

Interestingly, your uploaded kernel already contains machinery pointing toward LNV/variational updates. A future optimization could therefore use the old K as a geometry-continuation guess and do several current-H LNV gradient steps before purification. That could be much better than rebuilding the projector from scratch, but I would put it **after** the robust baseline.

For the \(\pm h\) pair, always initialize both from the same central state—not `+h` from the center and then `-h` from `+h`. Otherwise solver hysteresis itself can create an antisymmetric Hessian.

So I would reorder the project around **force smoothness first, locality second, performance third**. The immediate implementation sequence I would give the coding agent is:

**unify on a C² SK spline → wire analytic SK derivatives into the real force path → derive sparse EDM as \(2KHK\) → fix Si/H representation → implement separate \(R_{HS},R_K,R_Z\) masks and expanded-mask diagnostics → do the \(h\)/f32 noise experiment → only then build the nanocrystal Hessian driver.**

That puts us in a much stronger position than the current manifest, because we will know *why* an imaginary mode appears rather than simply tuning SCC tolerance or \(h\) until it happens to disappear.


---

# USER


I think you give very good insights and I completely agree

* 

* Yes I was insitiing that the agent to proper optimized sparse kernel, but he always forget something have some supid dense operation and CPU turnedand which defeate the point of sparse PGU based algorhim, in the end the bottleneks are somehwre wlese than in the parrt we optmizie just because of ignoreacne, and stupidity of the coding agent which miss the pictore and ignore huge bottnecks on the rust harness side, instead of just OpenCL kernels

* add "Curiously, you already have most of the solution: spline_resample.rs builds a natural cubic spline and converts it into cubic B-spline control points, while the GPU Hamiltonian uses a four-point cubic B-spline stencil."
thank for info, that is serous problem we should correct, wemust fine B-spline coefs allowing us to fficne tevaluat besplin, this precalucation we can do on CPU in double-precision fast while evaluation of the optimized spline coefs is on GPU, wwe should aso spend some effort optimizing how we read the spline coefs to local memory in the kernel, we spend some time doing it in Grid_ocl. where we project from CLAO to grid (density and molecular orbitals), perhaps we can learn a lot form there, there can be also the quintic splines. (not sure maybe hermite splines are better than B-splinses for our purpose?

* YES! I stronly again numerical finite differences. It destroy both speed and accuracy (in single precission especislly!) I was teling it to the coding again mutliple times and that ignorant still kep doing it! thaks for info stress it!

* I agree we should avoid any diagonalize in the whole sparse path. Not sure I get you "avoid Pullay/EDM diagonalization" method, but I fully support it. Elaborte on it, maybe write the kernel or pseudocode direcly. 

* I agree that we should not use very samll dx for finite differences computing hessian as it compromise the arithmetic error of single precission. We should definitely use central derivative but with quite high step (maybe even 0.05-0.1A, I do not mind if it is already little bit anharmonic - which typically it is still quite fine at 0.1A around minimum in my experience. We are not trying to be formally rigorous, we are tring to be pragmatic and efficient!

* Mixed arithmetic for sensiteve reducation (where we sum up large number with oposite sign) is very promissing idea, but also quite chalangin and fragile to do right. We should keep it as one optin not immediately replace all our single-precission kernels. We should implement both and then compare it. But to help the coding agent plese be more concreate where exacly we should put this double precision (so that it does not compromise speed much, but significantly enhance robustness), or even you may consider those spacial error canclation sumation like https://en.wikipedia.org/wiki/Kahan_summation_algorithm, becuase I know on my gaming GPUs double prec is like 40x slower than single prec, so it is better to do mor work in single prec than do proper double prec. (basically emulate it). I already used it for some purposes. I would be happy if you help us properlky adress exacly these issues in more detail.


* ad "And I would keep the sparse topology frozen near convergence. A block appearing/disappearing because a pair crossed \(R_K\) is a discontinuity in the numerical potential. Use a geometric skin, freeze the mask during the final optimization and the entire Hessian calculation, and distinguish:"
yes! I think this is very usefull idea!


* ad "We should project rigid modes rather than demanding them to emerge perfectly" on one hand I agree, but on the other hand this porjection often density the hessian, and make it less trasparent to analyse. When computing spectra we should do it, but we should still compare and analyse hessian before this rigid-mode removal.

* ad "The sparse GPU kernel still has several high-value optimizations"
can you write the actuall kernels with comments? (not necessaily the CPU harnes, just specify what the agent need to precalculate on PU for that

* ad "One more potentially large optimization: reuse across Hessian displacements" yes, this is good idea, write clear pseudocode for that


In general I agree with moset of the ideas, now let help the coding agent to concretize the most critical parts in code, ither kernel code or pseudocode or small rust sniplets, becasue I do not thrust my agent he really understdn numerical math and physucs, he can code but is not scientest clearly


---

# Chat GPT 5.6 sol

Yes. I think at this point we should give the coding agent something much more prescriptive: not merely “optimize the sparse solver,” but a set of **numerical contracts and concrete kernels** which make it difficult to accidentally reintroduce a dense \(O(N^2)\) operation, a CPU round-trip, or a finite difference inside the force.

There are four pieces I would make the immediate core of the implementation:

1. **one \(C^2\) spline representation, precomputed in f64 on CPU and evaluated analytically on GPU;**
2. **density + energy-weighted density entirely from the purified \(K\), with no diagonalization;**
3. **precomputed symbolic SpGEMM plans so the GPU no longer searches sparse connectivity during every multiplication;**
4. **explicit precision variants: fast f32 baseline, compensated-f32 where cancellation matters, and tiny selective-f64 reductions only as a benchmark/reference.**

And yes: **finite differences inside the DFTB force must be forbidden in the production sparse path.** The repository already contains analytic SK rotation derivative machinery, while the displayed CPU force path still has the DFTB+-style tiny finite difference of pair blocks. That is exactly the kind of “local implementation convenience” which destroys the larger numerical design.

---

# 1. B-spline versus Hermite: I would standardize on the *function*, not the representation

The crucial requirement is

$$
V(r)\in C^2
$$

because we want

$$
V,\quad V',\quad V''
$$

to behave smoothly.

The current CPU Hermite representation is only \(C^1\), because the derivatives at the nodes are estimated separately.

But Hermite itself is not inherently the problem. If we solve the global cubic-spline problem first, the same \(C^2\) cubic spline can be represented as:

* cubic B-spline control points \(c_i\);
* values + **spline-consistent** endpoint slopes \(f_i,f_i'\), i.e. Hermite form;
* per-interval polynomial coefficients \(a_i,b_i,c_i,d_i\).

They are just different bases for the same piecewise cubic function.

For GPU, I slightly prefer the **cardinal cubic B-spline controls**, because it stores only one coefficient per node per channel. Your code already does almost exactly this: `spline_resample.rs` fits/resamples with a natural cubic spline and converts values into B-spline control points; the GPU Hamiltonian already evaluates a four-point cubic B-spline stencil.

So this is not a new subsystem. It is mostly a question of making this the **canonical** path instead of having separate Hermite/B-spline behavior.

## Exact cubic B-spline evaluator we want

For \(t\in[0,1)\), with four controls \(c_{i-1},c_i,c_{i+1},c_{i+2}\),

$$
V = \sum_{k=0}^3 B_k(t)c_k.
$$

The cubic basis is

$$
\begin{aligned}
B_0 &= (1-t)^3/6\\
B_1 &= (3t^3-6t^2+4)/6\\
B_2 &= (-3t^3+3t^2+3t+1)/6\\
B_3 &= t^3/6.
\end{aligned}
$$

More importantly, compute derivatives analytically from exactly the same controls:

$$
\frac{dV}{dr}
=
\frac1{\Delta r}\sum B'_k(t)c_k
$$

and

$$
\frac{d^2V}{dr^2}
=
\frac1{\Delta r^2}\sum B''_k(t)c_k.
$$

Concrete OpenCL:

```c
// Cubic cardinal B-spline: value, first and second radial derivative.
//
// c[0..3] are the four control coefficients surrounding the interval.
// t is fractional coordinate in [0,1).
//
// IMPORTANT:
//   These derivatives are analytic derivatives of exactly the same
//   interpolated function used for H/S. No numerical finite differences.

inline float3 bspline3_v_d1_d2(
    float c0, float c1, float c2, float c3,
    float t,
    float inv_dr
){
    const float t2 = t*t;
    const float t3 = t2*t;
    const float u  = 1.0f - t;

    // value basis
    const float b0 = (u*u*u)                          * (1.0f/6.0f);
    const float b1 = (3.0f*t3 - 6.0f*t2 + 4.0f)     * (1.0f/6.0f);
    const float b2 = (-3.0f*t3 + 3.0f*t2 + 3.0f*t
                      + 1.0f)                         * (1.0f/6.0f);
    const float b3 = t3                               * (1.0f/6.0f);

    // dB/dt
    const float d0 = -0.5f*u*u;
    const float d1 =  1.5f*t2 - 2.0f*t;
    const float d2 = -1.5f*t2 + t + 0.5f;
    const float d3 =  0.5f*t2;

    // d2B/dt2
    const float dd0 =  1.0f - t;
    const float dd1 =  3.0f*t - 2.0f;
    const float dd2 = -3.0f*t + 1.0f;
    const float dd3 =  t;

    float v = fma(c0,b0,
              fma(c1,b1,
              fma(c2,b2, c3*b3)));

    float d = fma(c0,d0,
              fma(c1,d1,
              fma(c2,d2, c3*d3))) * inv_dr;

    float dd = fma(c0,dd0,
               fma(c1,dd1,
               fma(c2,dd2, c3*dd3))) * inv_dr*inv_dr;

    return (float3)(v,d,dd);
}
```

For the DFTB force we only actually require \(V,V'\). But I would implement \(V''\) now because it is an extremely useful **diagnostic for spline smoothness** and costs essentially nothing.

---

# 2. CPU preprocessing of the B-spline controls

The expensive/fussy numerical step should absolutely happen once, in f64.

For uniform samples \(f_i\), cardinal cubic B-spline controls obey in the interior

$$
f_i = \frac{c_{i-1}+4c_i+c_{i+1}}{6}.
$$

Hence solve

$$
c_{i-1}+4c_i+c_{i+1}=6f_i.
$$

That is just a tridiagonal solve in double precision.

Your existing `function_to_bspline_control_points()` already does this.

I would, however, spend effort on the **boundaries/tail**, rather than blindly use a “natural spline everywhere.”

For the SK cutoff I want the final analytic representation to satisfy approximately

$$
V(r_c)=0,\qquad
V'(r_c)=0,\qquad
V''(r_c)=0.
$$

So I would keep the good DFTB+ idea of a quintic tail.

Something like:

```text
raw SK tabulation
        |
        v
double-precision C2 cubic fit
        |
        +---- interior ---> cubic B-spline control points
        |
        `---- final 0.5-1.0 Å ---> quintic tail constrained by
                                  V, V', V'' at join
                                  and V=V'=V''=0 at cutoff
```

That gives a globally \(C^2\) potential.

### Do we need quintic B-splines everywhere?

Probably not.

Interestingly FireCore already contains explicit quintic B-spline basis machinery (`basis5`, `dbasis5`, etc.).

But quintic means six coefficient loads instead of four.

For DFTB I would start with:

> **cubic C² interior + quintic cutoff tail**

rather than quintic everywhere.

Only if the \(V''\) comparison shows significant cubic interpolation error would I benchmark quintic globally.

---

# 3. Local-memory layout for the SK tables

This is an excellent place to borrow lessons from GridFF.

Your GridFF OpenCL implementation already uses B-spline data and prepares small index structures in local memory; importantly, it also contains an explicit warning against returning some work-items before a barrier, because rounded-up workgroups can deadlock.

For the DFTB SK case the problem is much simpler: it is only **1D radial interpolation**.

I would benchmark three variants:

```text
A. direct __global reads        -- baseline
B. entire species-pair table -> __local once per WG
C. read-only/constant cache     -- where supported
```

My guess is B will be good when a WG processes many atom pairs of the same species type.

Something like:

```c
// Each workgroup processes a chunk of pairs that share one species-pair SK table.
// Therefore load the whole small table once.
//
// Example:
//   n_grid = 64
//   n_channels = 10
//
// Only 64*10*4 = 2560 bytes local memory.

for(uint i = get_local_id(0);
         i < n_grid*n_channels;
         i += get_local_size(0))
{
    sk_local[i] = sk_global[table_offset + i];
}

barrier(CLK_LOCAL_MEM_FENCE);

// NO work-item may return before this barrier.
```

I would also reconsider the current 5-float stride.

For sp-sp, one atom pair needs several related channels together. Aligning node records to 16-byte vectors can be beneficial.

For example:

```c
typedef struct {
    float4 h0;   // ss, sp, pp_sigma, pp_pi
    float4 s0;   // ss, sp, pp_sigma, pp_pi
    float2 ps;   // H_ps, S_ps
    float2 pad;
} SkNode;
```

That's 48 bytes/node.

Or perhaps three `float4`s.

This wastes a little storage but may make:

* global loads aligned,
* local loads aligned,
* compiler vectorization obvious.

Do not assume this is faster; benchmark it. The entire table is so tiny that clarity may win.

---

# 4. The EDM/Pulay trick — derivation carefully

This is one of the nicest consequences of purification.

The generalized eigenproblem is

$$
HC = SC\varepsilon,
$$

with normalization

$$
C^TSC=I.
$$

Let \(C_o\) contain occupied orbitals.

For closed shell, conventional density is

$$
D = 2 C_o C_o^T.
$$

Define the spinless projector/density kernel

$$
K=C_oC_o^T.
$$

Then indeed

$$
KSK=K
$$

and

$$
\mathrm{Tr}(KS)=N_\mathrm{occ}.
$$

The energy-weighted density matrix used in the Pulay force is

$$
W
=
2C_o\varepsilon_o C_o^T.
$$

Now compute

$$
KHK
=
C_oC_o^T
H
C_oC_o^T.
$$

Because

$$
C_o^T H C_o
=
C_o^TSC_o\,\varepsilon_o
=
\varepsilon_o,
$$

we get

$$
KHK
=
C_o\varepsilon_oC_o^T.
$$

Therefore:

$$
\boxed{D=2K}
$$

and

$$
\boxed{W=2KHK}.
$$

**No eigenvectors. No eigenvalues. No diagonalization.**

This is exact for a converged projector, apart from your sparse truncation errors.

## How to use this in the existing force decomposition

Do not rewrite the force physics.

Keep exactly the same decomposition that your validated dense CPU force already uses:

```text
F_nonSCC
F_SCC_shift
F_SCC_dc
F_rep
```

but replace

```text
DM  <- 2 * K
EDM <- 2 * K * H_scc * K
```

instead of constructing DM/EDM from eigenvectors.

That gives a very clean parity test.

---

# 5. Sparse computation of EDM

We only need \(W_{ij}\) wherever \(dS_{ij}/dR\neq0\).

That is crucial.

There is no reason to compute an EDM on a huge density mask if the force only contracts it against the short-range physical SK mask.

So:

```text
T = K * H_scc

W_force = T * K
          projected directly onto M_HS
```

i.e.

$$
W_\mathrm{force}
=
2P_{M_{HS}}(K H_\mathrm{scc}K).
$$

Pseudocode:

```rust
// K has long density mask M_K
// H has short physical SK mask M_HS

// symbolic structure only once
let t_mask = symbolic_product(M_K, M_HS);

// output W only where dS/dR can be non-zero
let w_mask = M_HS;

// numerical operation every SCC/force evaluation
spgemm(K, Hscc, T, plan_KH);
spgemm(T, K,     W, plan_TK_to_HS);

scale(W, 2.0);

// DM does not even require a new matrix:
DM = 2*K;
```

Since \(H_\mathrm{scc}\) and \(K\) are symmetric, your existing symmetric-right machinery is applicable to both multiplications.

I would symmetrize `W` afterward as a tiny numerical safeguard.

---

# 6. Precompute the sparse multiplication itself

This is perhaps the clearest kernel optimization I would implement next.

Your current symmetric-right kernel walks two sorted sparse rows to find common \(k\)'s during **every multiplication**. 

But during:

* 20–50 TC2 iterations,
* many SCC iterations,
* hundreds of optimizer iterations,
* thousands of Hessian displacements,

the topology is identical.

There is no reason to repeatedly solve the symbolic sparse-multiplication problem.

## CPU preprocessing

For every requested output block \(C_{ij}\), precompute:

```text
which pairs of blocks A_ik and B_jk contribute?
```

Since \(B\) is symmetric,

$$
B_{kj}=B_{jk}^{T}.
$$

Create:

```rust
struct ProductPlan {
    plan_ptr: Vec<u32>,   // one range per output C block
    plan:     Vec<u32>,   // packed (a_local, b_block)
}
```

For each output block:

```rust
for i in rows {
    let a0 = A.row_ptr[i];
    let a1 = A.row_ptr[i+1];

    for cb in C.row_ptr[i] .. C.row_ptr[i+1] {
        let j = C.col_idx[cb];

        // intersection performed ONCE on CPU
        ia = A.row_ptr[i];
        ib = B.row_ptr[j];

        while ia < A.row_ptr[i+1] && ib < B.row_ptr[j+1] {
            ka = A.col_idx[ia];
            kb = B.col_idx[ib];

            if ka < kb { ia += 1; }
            else if kb < ka { ib += 1; }
            else {
                a_local = ia - A.row_ptr[i];

                // With MAX_LEFT_BLOCKS <= 256:
                //
                // low  8 bits : local A-block index
                // high 24 bits: global B-block index
                packed = (ib << 8) | a_local;

                plan.push(packed);

                ia += 1;
                ib += 1;
            }
        }

        plan_ptr[cb+1] = plan.len();
    }
}
```

Now the numeric GPU kernel contains **zero sparse searches**.

---

# 7. Concrete planned-SpGEMM kernel

This is directly analogous to your existing kernel, but all connectivity work disappears:

```c
#define BS    4
#define BS2  16

#ifndef WG
#define WG 128
#endif

#define TEAM   16
#define NTEAM  (WG/TEAM)

#ifndef MAX_LEFT_BLOCKS
#define MAX_LEFT_BLOCKS 128
#endif


// ============================================================================
// PREPLANNED BSR4 PRODUCT, SYMMETRIC RIGHT OPERAND
//
//       C_ij = sum_k A_ik * B_kj
//
// but B is symmetric:
//
//       B_kj = transpose(B_jk)
//
// CPU has ALREADY determined all contributing pairs.
//
// plan_ptr[cb]..plan_ptr[cb+1]:
//
//   packed low  8 bits = local index of A_ik inside row i
//   packed high 24 bits = global B block index for B_jk
//
// Therefore there are:
//   * no binary searches
//   * no sorted-list intersections
//   * no col-index reads in the hot inner loop
//
// One WG = one row i
// One 16-thread team = one output block C_ij
// One lane = one scalar element of the 4x4 output block
// ============================================================================

__attribute__((reqd_work_group_size(WG,1,1)))
__kernel void bsr4_spgemm_plan_Bsym(

    const uint n_active_rows,

    // allows degree-bucketed row launches
    __global const uint* row_ids,

    // A
    __global const uint*  A_row,
    __global const float* A,

    // symmetric B
    __global const float* B,

    // output C structure
    __global const uint* C_row,

    // symbolic plan
    __global const uint* plan_ptr,
    __global const uint* plan_packed,

    __global float* C
){
    const uint ig  = get_group_id(0);
    const uint lid = get_local_id(0);

    if(ig >= n_active_rows) return;

    const uint i = row_ids[ig];

    const uint a0 = A_row[i];
    const uint a1 = A_row[i+1];
    const uint na = a1-a0;

    // Host MUST guarantee this for the bucket.
    if(na > MAX_LEFT_BLOCKS) return;

    // We do not need A_col anymore.
    __local float lA[MAX_LEFT_BLOCKS*BS2];

    // cooperative cached load of the whole left row
    for(uint q=lid; q<na*BS2; q+=WG){
        lA[q] = A[a0*BS2 + q];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint team = lid >> 4;
    const uint lane = lid & 15;

    const uint r = lane >> 2;
    const uint c = lane & 3;

    const uint c0 = C_row[i];
    const uint c1 = C_row[i+1];

    for(uint cb=c0+team; cb<c1; cb+=NTEAM){

        float sum = 0.0f;

        const uint p0 = plan_ptr[cb];
        const uint p1 = plan_ptr[cb+1];

        for(uint p=p0; p<p1; ++p){

            const uint packed = plan_packed[p];

            const uint al = packed & 0xffu;
            const uint bb = packed >> 8;

            __local const float* Ab =
                lA + al*BS2;

            // Stored block is B_jk.
            // We need B_kj = transpose(B_jk).
            __global const float* Bjk =
                B + bb*BS2;

            sum = fma(Ab[4*r+0], Bjk[4*c+0], sum);
            sum = fma(Ab[4*r+1], Bjk[4*c+1], sum);
            sum = fma(Ab[4*r+2], Bjk[4*c+2], sum);
            sum = fma(Ab[4*r+3], Bjk[4*c+3], sum);
        }

        C[cb*BS2 + lane] = sum;
    }
}
```

For N≈1000 and sensible mask sizes, 24 bits for the B-block index is enormous headroom.

For a future huge problem, simply switch to `uint2`.

---

# 8. Degree-bucket the rows

The current kernel reserves local memory according to the maximum row size.

That means a row with 20 neighbors can pay for a compile-time capacity of 256.

I would bucket rows on CPU:

```text
bucket0: degree <=  32
bucket1: degree <=  64
bucket2: degree <= 128
bucket3: degree <= 256
```

and compile four variants:

```text
MAX_LEFT_BLOCKS=32
MAX_LEFT_BLOCKS=64
MAX_LEFT_BLOCKS=128
MAX_LEFT_BLOCKS=256
```

Then launch with corresponding `row_ids`.

This should improve occupancy substantially if the density kernel remains genuinely local.

It also makes the performance statistics physically useful:

```text
mean degree
P50 degree
P90 degree
max degree
```

rather than merely `nnz`.

---

# 9. Fuse the second TC2 product with the TC2 update

Currently conceptually:

```text
T = K*S
Q = T*K

if trace(KS)>Nocc:
    Knew = Q
else:
    Knew = 2K-Q
```

There is no reason to write `Q`, launch another kernel, then read `Q`.

The kernel computing \(Q=T K\) can directly write `Knew`.

It can also calculate the old idempotency residual

$$
R=Q-K
$$

for free.

Conceptually:

```c
float q = ... spgemm ...

float k = Kold[cb*16 + lane];

float knew =
    (traceKS > Nocc)
    ? q
    : 2.0f*k - q;

Knew[...] = knew;

float err = q-k;
res2 += err*err;
```

So one TC2 iteration becomes approximately:

```text
Kernel A:
    T = K*S
    + trace partials

tiny reduction:
    Tr(KS)

Kernel B:
    Knew = TC2( T*K, K )
    + idempotency residual partials

tiny reduction:
    RI

optional:
    symmetrize(Knew)

swap(K,Knew)
```

That is about the irreducible structure of the algorithm.

No extra `Q` matrix is required.

---

# 10. Precision: where I would NOT use f64

I would **not** initially put f64 into:

* BSR4 SpGEMM;
* B-spline evaluation;
* SK rotations;
* TC2 elementwise update;
* H/S matrices;
* K;
* Z;
* W/EDM;
* ordinary short-range electronic force contractions.

These are the bulk operations.

They should remain pure f32/FMA.

The expected error from summing perhaps a few dozen 4-term block products is much less frightening than the error from **long cancellation-heavy reductions**.

---

# 11. Where compensated or mixed arithmetic actually matters

This is the hierarchy I would test.

| Quantity                            | Problem                          | First choice                         |   |     |              |                    |
| ----------------------------------- | -------------------------------- | ------------------------------------ | - | --- | ------------ | ------------------ |
| 4×4 SpGEMM                          | modest accumulation              | ordinary f32 FMA                     |   |     |              |                    |
| (                                   |                                  | KSK-K                                |   | ^2) | positive sum | f32 tree reduction |
| (                                   |                                  | ZS-I                                 |   | ^2) | positive sum | f32 tree reduction |
| `Tr(KS)`                            | TC2 branch sensitive             | f32 partials + compensated/f64 final |   |     |              |                    |
| \(V_i=\sum_j\gamma_{ij}\Delta q_j\) | **large signed cancellation**    | compensated f32                      |   |     |              |                    |
| SCC energy                          | cancellation between large terms | f32 partials → f64 host final        |   |     |              |                    |
| SCC long-range force                | signed long sum                  | compensated f32                      |   |     |              |                    |
| short-range SK force                | few neighbors                    | ordinary f32                         |   |     |              |                    |
| \(F(+h)-F(-h)\)                     | subtraction of near values       | large \(h\); assemble Hessian f64    |   |     |              |                    |
| Hessian eigensolve                  | cheap, sensitive                 | f64 CPU                              |   |     |              |                    |

The previous false-zero Newton-Schulz residual is an excellent example of what we should avoid: an algebraically correct but cancellation-prone expression failed in f32, while computing the actual residual directly fixed it.

---

# 12. Kahan accumulation kernel for the SCC \(\gamma\) potential

This is perhaps the first place I would actually deploy Kahan.

For each atom,

$$
V_i=\sum_j \gamma_{ij}\Delta q_j.
$$

Terms can have opposite sign, and there may be ~1000 of them.

This is almost the canonical Kahan use case.

```c
typedef struct {
    float sum;
    float corr;
} KahanF32;

inline void kahan_add(
    __private KahanF32* a,
    const float x
){
    const float y = x - a->corr;
    const float t = a->sum + y;

    a->corr = (t - a->sum) - y;
    a->sum  = t;
}


// One work-item per atom.
// O(N^2), but for N~1000 this is only ~1e6 pair terms.
// Very likely negligible next to purification.
//
// IMPORTANT:
// Do NOT compile this particular kernel with aggressive FP reassociation
// (-cl-fast-relaxed-math), otherwise compensation may be optimized away.

__kernel void gamma_potential_kahan(
    const uint n,
    __global const float4* posU,  // xyz + Hubbard U
    __global const float* dq,
    __global float* V
){
    const uint i = get_global_id(0);
    if(i>=n) return;

    const float4 pi = posU[i];

    KahanF32 acc;
    acc.sum  = 0.0f;
    acc.corr = 0.0f;

    for(uint j=0; j<n; ++j){

        float gij;

        if(i==j){
            // whatever your canonical on-site gamma convention is
            gij = pi.w;
        }else{
            const float4 pj = posU[j];
            const float3 d  = pj.xyz - pi.xyz;
            const float r   = length(d);

            gij = gamma_full(r, pi.w, pj.w);
        }

        kahan_add(&acc, gij*dq[j]);
    }

    V[i] = acc.sum;
}
```

Benchmark against plain:

```c
sum = fma(gamma, dq[j], sum);
```

I would expect Kahan to matter more for difficult neutral/polar charge distributions than for a homogeneous Si NC.

---

# 13. A stronger all-f32 option: two-float accumulation

If Kahan isn't enough and fp64 really is 1/32–1/64 throughput, we can emulate an extended accumulator with `(hi,lo)` floats.

This does **not** magically make the underlying pair terms double precision.

But it strongly reduces summation/cancellation loss.

```c
inline float2 two_sum(float a, float b)
{
    float s  = a + b;
    float bb = s - a;

    float e =
        (a - (s - bb))
      + (b - bb);

    return (float2)(s,e);
}


inline void ds_add(
    __private float2* acc,
    float x
){
    float2 se = two_sum(acc->x, x);

    float lo = acc->y + se.y;

    float hi = se.x + lo;
    lo = lo - (hi - se.x);

    acc->x = hi;
    acc->y = lo;
}
```

Then:

```c
float2 acc = (float2)(0.0f,0.0f);

for(...) {
    ds_add(&acc, term);
}

float result = acc.x + acc.y;
```

For a validation build we can even compensate products:

```c
float p  = a*b;
float pe = fma(a,b,-p);  // recover product rounding error

ds_add(&acc,p);
ds_add(&acc,pe);
```

That is much more expensive, so I would **not** put it into production SpGEMM initially.

But it's a great experimental reference:

```text
SPGEMM_PRECISION = F32
SPGEMM_PRECISION = F32_COMPENSATED
```

If spectra are identical, we know multiplication roundoff was never the problem.

---

# 14. Selective real f64 can still be almost free

The fact that a gaming NVIDIA GPU has awful fp64 throughput does **not** mean every double is expensive.

If we reduce millions of f32 values into, say, 32 partial sums first, and then perform only 32 double additions:

```text
millions of terms: f32
       |
       v
32 float partials
       |
       v
32 fp64 adds
```

the 1/32 fp64 throughput is irrelevant.

Thirty-two double additions are nothing.

I would therefore implement three final-reduction variants:

```text
REDUCE_FAST       f32 tree
REDUCE_KAHAN      compensated f32
REDUCE_FP64_FINAL f32 partials + tiny fp64 tail
```

and actually benchmark them.

For `Tr(KS)` this is particularly attractive because the trace controls the TC2 branch.

---

# 15. Hessian step: I agree with being pragmatic

I agree with you that \(10^{-3}\) Å is probably exactly the wrong mindset here.

We don't care about formally asymptotic second derivatives at infinitesimal displacement.

We care about stable physical curvature.

I would explicitly test:

$$
h =
0.01,\ 0.02,\ 0.05,\ 0.10\ {\rm Å}.
$$

My **starting production guess would be about 0.05 Å**.

For Si-Si:

$$
0.05/2.35 \approx 2\%
$$

of the bond length.

That is still a fairly local perturbation.

0.1 Å is about 4%, and I agree it may still be perfectly satisfactory if our target is ~5% frequency accuracy.

For Si-H, 0.1 Å is a larger fractional displacement, so high-frequency H stretch modes may become visibly anharmonic. Therefore we could eventually use:

```text
Si coordinate: h = 0.05 Å
H  coordinate: h = 0.02-0.03 Å
```

There is nothing mathematically wrong with different finite-difference steps for different Hessian columns.

Then symmetrize afterward.

The selection criterion should simply be:

> Find the broad \(h\)-plateau where doubling/halving \(h\) changes ordinary frequencies by less than ~1–2%.

That is much more useful than chasing the formal \(h\to0\) limit.

---

# 16. Freeze *everything structural* during the final optimization/Hessian

This deserves turning into an explicit contract.

At the start of the final phase:

```text
final relaxed-ish geometry R0

choose h_max
choose mask skin

build masks from:
    cutoff + skin

freeze:
    H/S block topology
    K topology
    Z topology
    product masks
    symbolic SpGEMM plans
    row buckets
```

Then absolutely no:

```text
if norm(block)<eps:
    remove block
```

and no:

```text
if distance > R_K:
    rebuild CSR
```

during final relaxation or Hessian.

Values can go to zero.

**Structure cannot change.**

Your uploaded `bsr4_drop_small` is already only a value-threshold diagnostic and explicitly does not rebuild CSR; keep that philosophy. 

---

# 17. Hessian displacement reuse — explicit workflow

Here's how I would implement it.

## Central preparation

```text
R0 = tightly optimized geometry

freeze all sparse masks and symbolic multiplication plans

build H0(R0), S(R0)

Z0 = inverse_overlap(S0)

bounds0 = safe spectral bounds
          + deliberately add some safety margin

q0 = converged SCC charges at R0

K0_center = converged density kernel at R0
```

Now for each coordinate \(a=(i,\alpha)\):

```text
for sign in { -1, +1 } independently:

    R = R0
    R[i][alpha] += sign*h[i]

    ------------------------------------------------------------
    1. Update only geometry-dependent local H0/S blocks
    ------------------------------------------------------------

    copy H0_center -> H0

    copy S_center  -> S

    rebuild ONLY physical SK blocks touching atom i

    // all other atom pairs have exactly unchanged geometry


    ------------------------------------------------------------
    2. Update gamma geometry cheaply
    ------------------------------------------------------------

    gamma = gamma_center

    update ONLY row i and column i
    // because only distances involving moved atom i changed


    ------------------------------------------------------------
    3. Warm-start S^-1
    ------------------------------------------------------------

    Z = Z0

    // But iterate against the NEW S.
    //
    // Since S changed only slightly, Newton-Schulz should need very few
    // correction iterations.
    //
    // Z <- Z (2I - S Z)
    //
    newton_schulz_correct(Z, S)


    ------------------------------------------------------------
    4. Warm-start SCC charges
    ------------------------------------------------------------

    q = q0

    for scc_iter:

        V = gamma * dq

        Hscc = H0 + 0.5*S*(Vi+Vj)

        --------------------------------------------------------
        IMPORTANT:
        density must depend on CURRENT Hscc
        --------------------------------------------------------

        Kinit = build_projector_initial_guess(
                    Hscc,
                    S,
                    Z,
                    bounds0
                )

        K = TC2_purify(Kinit)

        q_new = mulliken(K,S)

        q = mix(q,q_new)

        if converged:
            break


    ------------------------------------------------------------
    5. Build diagonalization-free EDM
    ------------------------------------------------------------

    D = 2*K

    T = K*Hscc

    W = 2 * project_HS_mask(T*K)


    ------------------------------------------------------------
    6. Analytic forces
    ------------------------------------------------------------

    F = sparse_analytic_force(
            R,
            D,
            W,
            q,
            ...
        )

    save Fminus or Fplus
```

Then:

$$
H_{:,a}
=
-\frac{F_+-F_-}{2h_i}.
$$

And importantly:

```text
+ displacement starts from CENTER q0/Z0
- displacement starts from CENTER q0/Z0
```

Never:

```text
center -> +h -> -h
```

because that creates numerical hysteresis between the two sides.

---

# 18. What can be reused *inside the force* too

There is an additional possible optimization.

For every physical pair we need:

$$
dH/dx,\ dH/dy,\ dH/dz,\qquad
dS/dx,\ dS/dy,\ dS/dz.
$$

When atom \(i\) is displaced, only derivatives for pairs involving \(i\) change.

Therefore, for the center geometry, we can potentially precompute derivative blocks:

```text
pair_derivs[pair]:
    dHdx[16]
    dHdy[16]
    dHdz[16]
    dSdx[16]
    dSdy[16]
    dSdz[16]
```

and for each Hessian displacement update only pair derivatives touching displaced atom.

For ~1000 Si atoms and perhaps ~5000 physical SK pairs:

```text
5000 pairs
* 6
* 16 floats
* 4 bytes
≈ 1.9 MB
```

which is trivial.

Whether it is faster than evaluating the analytic spline/rotation on the fly is a benchmark question; the latter is cheap enough that memory bandwidth may dominate.

But the option is worth testing.

---

# 19. Important: do NOT warm-start TC2 with old K blindly

This subtlety is worth putting in the agent instructions.

Charges:

```text
YES: warm start q
```

Inverse overlap:

```text
YES: warm start Z and Newton-Schulz-correct against new S
```

Density kernel:

```text
NO: simply take K_old and run the same projector polynomial
```

Why?

TC2 is fundamentally a polynomial transformation of the matrix/eigenspace encoded in its input. A projector polynomial can sharpen occupations, but it cannot generally rotate an already-converged projector from the old Hamiltonian into the occupied eigenspace of the new Hamiltonian.

So baseline Hessian calculations should always construct \(K_\text{init}\) from the **current** \(H_\mathrm{scc}\).

Later, if we want stronger geometry continuation, then an LNV/variational density-matrix correction using current \(H\) can rotate the subspace. But that is a later optimization.

---

# 20. Raw Hessian first, rigid projection only in a separate copy/operator

I agree with your concern.

The raw Hessian is valuable physically and numerically.

Never overwrite it.

Store:

```text
H_raw
```

and first inspect:

### symmetry

$$
\eta_\mathrm{sym}
=
\frac{\|H-H^T\|_F}{\|H\|_F}.
$$

### translational sum rule

Ideally

$$
\sum_j H_{i\alpha,j\beta}=0.
$$

### rigid-mode response

Instead of projecting yet, construct the six translation/rotation vectors \(q_k\) and simply measure

$$
\|Hq_k\|
$$

and

$$
q_k^T H q_k.
$$

That tells us how badly translation/rotation invariance is violated without touching the Hessian.

### block locality

Plot

$$
\|H_{ij}\|_F
$$

versus \(r_{ij}\).

That is scientifically useful in itself.

Only afterwards make

```text
H_spectrum
```

for rigid-mode projection.

And if we want to avoid explicitly densifying by projection, we can apply

$$
P H P
$$

matrix-free:

```text
function projected_H_times(x):

    x1 = x - Q*(Q^T*x)

    y  = H*x1

    y -= Q*(Q^T*y)

    return y
```

So even the projected operator does not need to destroy the sparse/raw representation.

For N≈1000 and a final dense LAPACK eigensolve this distinction doesn't matter computationally, but conceptually it is much cleaner.

---

# 21. Protect the project from “optimized kernel, stupid harness”

I think this needs to become an explicit **performance contract** for the agent, because otherwise the same failure will keep recurring.

For the production sparse path:

```text
FORBIDDEN:
    to_dense()
    DMatrix allocation proportional to Norb^2
    CPU matrix multiplication
    CPU eigenproblem
    CPU sparse multiplication
    host copy of K/H/S per iteration
    rebuilding CSR inside SCC
    rebuilding symbolic SpGEMM plans inside SCC
    numerical finite differences of H/S
    numerical finite differences inside force
```

Allowed host transfers during an SCC iteration should be explicitly counted.

Something like:

```rust
struct SparsePerfStats {
    n_atom: usize,

    nnz_hs: usize,
    nnz_k:  usize,
    nnz_z:  usize,

    gpu_bytes: usize,

    n_kernel_launch: usize,
    n_host_read: usize,
    host_read_bytes: usize,

    t_hs: f64,
    t_ns: f64,
    t_tc2: f64,
    t_gamma: f64,
    t_force: f64,
}
```

And every performance run should end with:

```text
=== SPARSE EXECUTION AUDIT ===
N atoms                 1000
H/S blocks              ...
K blocks                ...
Z blocks                ...

largest dense allocation   0 bytes
host K transfers            0
host H/S transfers/SCC      0

SCC:
    GPU Hscc      ...
    GPU gamma     ...
    GPU TC2       ...
    host/misc     ...

TOTAL ...
```

If “host/misc” is suddenly 60% of runtime, it becomes impossible for the agent to celebrate a 2× SpGEMM improvement while ignoring the actual bottleneck.

Also, the current geometric mask construction itself loops over all \(i,j\), so even that is \(O(N^2)\) today.  For N≈1000 that's not disastrous as a one-time operation, but before claiming true linear scaling it should be replaced by a cell list.

---

# 22. What I would ask the coding agent to do next, in exact order

I would **not start the nanocrystal Hessian driver yet**.

The next implementation sequence should be:

**P0 — make the sparse path honestly sparse end-to-end.** Remove/ban hidden dense matrices, dense spectral helpers, dense production masks, per-iteration CPU operations, and instrument host/device time.

**P1 — make the C² spline canonical.** Use the existing CPU f64 spline/B-spline preprocessing, add GPU analytic \(V,V',V''\), add derivative-continuity tests at every knot and cutoff, and remove the current production Hermite-vs-B-spline inconsistency.

**P2 — production force must be 100% analytic.** Route the existing analytic Slater–Koster rotation derivative machinery into the sparse force. Add a test that searches the production sparse call graph for the tiny `DELTA_X_DIFF`/pair finite-difference path and fails if it is used.

**P3 — implement diagonalization-free EDM.**

$$
D=2K,\qquad W=2K H_\mathrm{scc} K.
$$

Compare the resulting force against the existing dense eigenvector-based force at the same geometry.

**P4 — precompute symbolic SpGEMM plans.** Implement the planned kernel above and benchmark it against the current sorted-row-intersection kernel. Then bucket row degrees.

**P5 — fuse TC2.** `KS + trace`, followed by `TK + TC2 update + idempotency residual`, plus cheap symmetrization.

**P6 — precision experiment.** Implement fast-f32, Kahan gamma/reductions, optional two-float compensated accumulation, and optional fp64-final-reduction variants. Benchmark all against dense f64 on a fixed geometry. Do **not** assume the most sophisticated one should become default.

**P7 — freeze-mask geometry optimization.** Only when the force is clean and deterministic.

**P8 — Hessian \(h\)-sweep.** \(0.01,0.02,0.05,0.10\) Å, compare raw Hessians and spectra, select the broad stable plateau.

Only then would I launch Si\(_{300}\), followed by Si\(_{1000}\).

That ordering is much safer because each stage answers one scientific/numerical question. If a \(-80\) cm\(^{-1}\) surface mode appears later, we can determine whether it came from geometry, SCC convergence, K truncation, arithmetic, or genuine physics rather than having fifteen simultaneous suspects.







---

# USER

we implemented most if this in this commit
https://github.com/ProkopHapala/dftbplus/commit/b269ab63eee65eb854bb21184793fed4d02aeebc
please review it focus on mathematical and physical rigot as well as numerical efficienency of the kernels and minimizing of the overhead, the coding agent often do stupid things, lazzy sub-optimal harness, and physical misconception, try to indetify them and say him how to do it properly. 

Focus just on the aspect related to your sub problem (the sparse solver, not the multi-small-system dense solver which is topic of the other tread)


---


# Chat GPT 5.6 sol

I went through the sparse-specific part of `b269ab63eee65eb854bb21184793fed4d02aeebc` in detail. There is substantial useful work here, but I would **not yet accept the commit's own statement that P0–P4 and Gates C–E are passed**. Several pieces are individually correct but are not wired together into the actual numerical path. In a few places the tests give a false sense of validation.

The biggest problem is exactly the one you were worried about: the agent has implemented several good kernels, but the harness still routes around them or adds enough redundant work to defeat them.

## 1. Most serious physics/numerics bug: the sparse force is still not differentiating the C² spline

This is the most important thing I found.

The new C² B-spline evaluator itself is good: the corrected cubic basis is there, including analytic \(V'\) and \(V''\).

But the actual `SkTableSp` still stores:

```rust
pub h: EqGridTable,
pub s: EqGridTable,
```

and `eval_shell_integrals_and_derivs_into()` explicitly calls:

```rust
self.h.eval_with_deriv_into(...)
self.s.eval_with_deriv_into(...)
```

which is the old **cubic Hermite** path.

And `EqGridTable` still says explicitly that it is C¹ Hermite with precomputed slopes. Worse, its tail derivative is still computed by a numerical finite difference of the Neville/tail interpolation:

```rust
let dr_step = 1.0e-4;
...
ddr[k] = (v_p[k] - v_m[k]) / (2.0 * dr_step);
```

So the commit currently has three different statements simultaneously:

```text
P1: C² B-spline canonical       <- documentation
P2: no FD inside production force <- documentation
actual sparse/CPU force:
        C¹ Hermite interior
        numerical FD derivative in tail
```

This is a **blocking correction before phonons**.

The right fix is not to add more tests around the existing functions. Change the data model so there is one canonical radial function.

Conceptually:

```rust
struct SkRadialTable {
    dr: f64,
    r_cut: f64,

    // canonical C² representation
    h_ctrl: Vec<[f32; NCHAN]>,
    s_ctrl: Vec<[f32; NCHAN]>,

    // optional explicit C² cutoff-tail coefficients
    ...
}
```

and both

```text
CPU H/S
CPU analytic derivative
GPU H/S
GPU analytic derivative
```

must evaluate **those exact same coefficients**.

`EqGridTable::eval_with_deriv_into()` should no longer be in the production SK-force call graph. Keep Hermite/Neville only as references.

A test like

```text
energy built with spline A
force derivative built with spline A'
```

is what matters. CPU/GPU parity of two isolated evaluators is not sufficient.

---

## 2. There is not yet a sparse SCC solver; there is a sparse purification attached to a dense SCC calculation

This is the largest architectural problem.

`rhai_run_sparse_purify_geom()`, which is described as the large-system production path, first retrieves an already converged dense `SccResult`. It then takes dense `scc.h_scc` and dense `scc.s` and converts them back into BSR4.

Therefore the current path is effectively:

```text
dense H/S construction
dense SCC
       ↓
dense H_scc
       ↓
dense -> sparse conversion
       ↓
sparse purification test
```

That is perfectly valid as a **purification parity test**, but it is not a sparse DFTB calculation.

More importantly, this matters physically, not merely for performance. In real SCC-DFTB the sparse K determines Mulliken charges,

$$
K \rightarrow q_A \rightarrow V_A
\rightarrow H_\mathrm{SCC}
\rightarrow K,
$$

and that loop must converge self-consistently.

At present the sparse K is being calculated for a Hamiltonian whose SCC charges came from the **dense solution**. Therefore a sparse force constructed from that K is not necessarily the derivative of a self-consistent sparse energy.

For a Hessian this variational consistency matters enormously.

The production path needs to become:

```text
sparse H0/S assembly
       ↓
q initial
       ↓
γ·Δq
       ↓
sparse Hscc
       ↓
sparse K0 + TC2
       ↓
Mulliken q
       ↓
mix
       └──────── repeat
```

with no dense `SccResult` anywhere in that loop.

The existing dense pathway should be renamed mentally and in code to something like:

```text
sparse_purification_reference_from_dense_hamiltonian()
```

so nobody mistakes it for the production implementation.

---

## 3. TC2 currently does nearly twice the necessary SpGEMM work

This one can give a very large immediate speedup.

The workspace computes:

$$
T_n=K_nS
$$

and

$$
Q_n=T_nK_n=K_nSK_n.
$$

It then updates \(K\).

After the swap, the convergence check recomputes \(KS\) and \(KSK\) for the new K.

And the current engine invokes:

```rust
tc2_purify_dev(..., check_every=1)
```

So a normal iteration effectively becomes:

```text
KS
KSK
update K
KS       <- again
KSK      <- again
residual
```

**Four sparse matrix multiplications per iteration instead of two.**

There is no reason for this.

For the current \(K_n\), you already have exactly what you need:

$$
R_I(K_n)=\|K_nSK_n-K_n\|
       =\|Q_n-K_n\|.
$$

So use the residual **before updating K**.

The loop should look like:

```text
for iter:

    T = K*S                       # SpGEMM 1
    trace = Tr(T)                # device reduction

    Q = T*K                       # SpGEMM 2

    if this is diagnostic/check iteration:
        RI = ||Q-K||             # reduction only, NO SpGEMM

        optionally read {trace,RI} to CPU

        if RI < tol:
            return current K

    # trace stays on device
    Knew = (trace > Nocc) ? Q : 2*K-Q

    maybe symmetrize
    swap(K,Knew)
```

The TC2 branch kernel already consumes `trace_buf` on the device, so **the CPU does not need `Tr(KS)` every iteration either**.

Currently `tc2_step_dev()` nevertheless reads that scalar back every iteration.

That should disappear.

Normal iteration:

```text
2 SpGEMMs
0 host synchronization
```

Diagnostic iteration:

```text
2 SpGEMMs
2 cheap reductions
1 combined scalar read
```

That is much closer to what we wanted originally.

---

## 4. P4 symbolic plans exist, but the actual solver does not use them

This is another classic agent problem: implemented feature ≠ integrated feature.

You added:

```rust
spgemm_plan_bsym_dev()
```

and the planned kernel.

But TC2 still calls:

```rust
spgemm_bsym_dev()
```

and `build_dw_sparse()` still calls the generic runtime-search path.

So P4 currently proves:

> “a symbolic-plan kernel can give the same answer.”

It does **not** make production TC2 faster.

The plan should be built for all recurring products at frozen topology:

```text
Z * S       -> T_ZS
T_ZS * Z    -> Znew

Z * H       -> T_ZH
T_ZH * Z    -> K0

K * S       -> T_KS
T_KS * K    -> Q/Knew

K * Hscc    -> T_KH
T_KH * K    -> W on M_HS
```

Notice also an important misconception in the present code: `build_k0()` says `Z*H` must use generic SpGEMM because `Z*H` is not symmetric.

That is irrelevant.

Your `Bsym` kernel requires the **right operand B** to be symmetric. H is symmetric.

Therefore:

```text
Z * H
```

can absolutely use the Bsym planned kernel.

The product C does not need to be symmetric.

---

## 5. The planned SpGEMM kernel itself still contains pointless work

The new kernel is a good direction, but it contains residue from the old intersection kernel.

It receives and loads:

```c
A_col
C_col
```

and allocates:

```c
__local uint lcol[MAX_LEFT_BLOCKS];
```

then fills `lcol`.

But the symbolic plan already tells it which A block to use. `lcol` is never used in the numerical loop. `C_col` is also unused.

Delete all three.

Then local memory goes approximately from:

```text
256 * (16 floats + 1 uint)
```

to:

```text
256 * 16 floats
```

and you remove useless global loads as well.

There is another easy improvement. The plan currently stores:

```rust
plan_a_idx: u32
plan_b_idx: u32
```

for every contribution.

But `MAX_LEFT_BLOCKS <= 256`, so `a_idx` needs only 8 bits.

Pack:

```text
bits  0..7   local A index
bits  8..31  B block index
```

so:

```c
uint p  = plan[t];
uint ia = p & 0xff;
uint bb = p >> 8;
```

This halves plan-term traffic from 8 B to 4 B.

Fail loudly during plan construction if:

```text
ia >= 256
bb >= 2^24
```

and use `uint2` only for extreme cases.

---

## 6. The current 16-lane mapping probably reloads B four times unnecessarily

Current mapping is:

```text
16 lanes = 16 scalar C_rc
```

Each lane performs four global B loads per block contribution.

For fixed `c`, the four different `r` lanes therefore request the same four B values. Hardware cache may rescue much of it, but logically the kernel asks for roughly four times as many B loads as necessary.

I would benchmark another mapping:

```text
16 lanes = (m,c)
```

Each lane reads one:

$$
B_{mc}
$$

and accumulates four output rows:

$$
C_{0c},C_{1c},C_{2c},C_{3c}.
$$

So per plan contribution:

```text
16 B global loads
64 local A loads
64 FMAs
```

instead of logically ~64 B loads.

You then reduce the four `m` contributions for each c.

I would **benchmark**, not assume, because NVIDIA caches may already make the present implementation quite good.

Also: the P4 performance test is currently an 8-atom chain and only 10 launches. That mostly benchmarks launch/cache behavior, not the nanocrystal kernel.

Use realistic matrices:

```text
N = 300, 600, 1000
degree distributions corresponding to RK ~ 5–10 Å
100–1000 repeated products
```

and profile actual kernel events.

---

## 7. Degree buckets were recommended but were not actually implemented

`MAX_LEFT_BLOCKS=256` means every workgroup reserves storage for 256 blocks regardless of whether its row contains 20, 50, or 80 blocks.

The code itself still says one remedy is “implement degree-bucketed kernels.”

I would actually do it:

```text
<=32
<=64
<=128
<=256
```

For Si density around 7 Å, I would expect 64/128 to be much more relevant than 256 for many rows.

This is particularly worthwhile because local-memory occupancy is one of the few clear limitations of this one-WG-per-row design.

---

## 8. `build_dw_sparse()` is currently a hot-path allocation machine

Mathematically,

$$
D=2K,\qquad W=2K H_\mathrm{SCC}K
$$

is the right result.

But every invocation currently:

* builds `m_tw` on CPU;
* allocates/uploads another GPU structure;
* allocates T;
* allocates W;
* performs two generic SpGEMMs;
* runs a separate kernel to multiply W by 2;
* runs a separate W symmetrization;
* allocates D;
* runs another kernel merely to create `D=2K`.

For Hessians this is exactly the overhead we must eliminate.

Create once:

```rust
struct SparseForceWorkspace {
    t_kh: GpuBsrMatrix,
    w:    GpuBsrMatrix,

    plan_k_h:  SpgemmPlanGpu,
    plan_th_k: SpgemmPlanGpu,

    ...
}
```

and reuse forever while topology is frozen.

Furthermore **do not create D at all**.

Pass:

```text
K + spin_factor=2
```

to the force contraction.

Similarly either fuse `2.0` into the second W product or, even simpler, let the force contraction know that W carries the spin factor.

So:

```text
no D allocation
no D scale kernel
no W scale kernel
```

---

## 9. W symmetrization should be a diagnostic, not a way of hiding errors

For symmetric H and K,

$$
KHK
$$

is symmetric.

And if the intermediate mask is the exact symbolic product support of the already truncated operands, the first multiplication should not introduce structural truncation.

Therefore significant asymmetry in W indicates something worth diagnosing.

I would measure:

$$
\eta_W=
\frac{\|W-W^T\|}{\|W\|}.
$$

If it is ~f32 roundoff, don't launch a full global symmetrization pass at all.

For force contraction you can cheaply use

$$
\frac{W_{ij}+W_{ji}^T}{2}
$$

on the pair if desired.

The same question applies to symmetrizing K after **every** TC2 iteration. Benchmark:

```text
every iteration
every diagnostic iteration
only when η exceeds threshold
```

rather than automatically paying the memory pass forever.

---

## 10. Gate C should be marked **not passed**

The current test is useful as an algebra smoke test, but not as a locality test.

It uses only five atoms on a 1.5 Å line. At \(R=7\) Å, the entire ~6 Å system is already inside the mask. So the reported:

```text
RK = 7
RZ = 7
```

“plateau” simply approaches the dense/full support.

There is an even clearer physics mistake:

```rust
for i in 0..n {
    h_full[i*n+i] += 2.0;
}
```

is described as making “a clear gap.”

It does not.

Adding \(2I\) shifts **every eigenvalue by exactly 2** and leaves every level spacing, including the HOMO-LUMO gap, unchanged.

So this test does not establish the relation between gap and density locality at all.

Use a real insulating system.

I would use actual DFTB H/S for something like:

```text
Si29H36
Si~80 H...
Si~150 H...
```

and then ask whether the **same** RK gives stable errors as N grows.

That is the actual linear-scaling question.

---

## 11. `R_leak` in Gate C is mathematically wrong

The present approximation is:

$$
R_\text{leak}
\approx
\|Q_\mathrm{val}\|-\|Q_K\|.
$$

That is not the norm of the omitted residual.

Even in the ideal case where `Q_K` is exactly the inside projection,

$$
\|Q_\mathrm{val}\|^2
=
\|Q_\mathrm{inside}\|^2
+
\|Q_\mathrm{outside}\|^2,
$$

so:

$$
\boxed{
R_\mathrm{leak}
=
\sqrt{
\|Q_\mathrm{val}\|^2-
\|Q_\mathrm{inside}\|^2
}
}
$$

not the difference of norms.

But I would implement something even cleaner.

Compute `Q=KSK` **once** on the validation support and tag each output block:

```text
inside MK     -> ignore
outside MK    -> accumulate block norm²
```

GPU reduction:

```c
if(!inside_MK[block]){
    sum += dot(Q_block,Q_block);
}
```

Then:

$$
R_\mathrm{leak}=\sqrt{\sum_{\text{outside }M_K}|Q_{ij}|^2}.
$$

No approximation.

---

## 12. RK and RZ are still being mixed together in the locality experiment

The test correctly constructs \(Z\) on \(M_Z\), but then does:

```rust
z = z_on_mz.project_to_mask(&m_k)
```

**before** constructing K0.

This destroys the independence of \(R_Z\) and \(R_K\).

The proper expression is

$$
K_0=
P_{M_K}
\left[
\frac{
\varepsilon_\max Z-ZHZ
}{
\varepsilon_\max-\varepsilon_\min
}
\right].
$$

Use the full available Z on \(M_Z\) in `ZHZ`.

Only the **final result** is projected to \(M_K\).

Otherwise when \(R_Z>R_K\), you throw away the extra inverse-overlap accuracy before it can contribute.

This is exactly the sort of subtle numerical-math issue I would spell out explicitly for the agent.

---

## 13. Gate E is currently a false positive and should also be marked **not passed**

This is perhaps the most misleading test in the commit.

It claims to characterize sparse-f32 force noise and find a Hessian-step plateau.

But the actual implementation does something else.

The “sparse force” is computed by finite differences of sparse **energies**:

```rust
forces[i][d] =
 -(E(R+h)-E(R-h))/(2h)
```

not by the new analytic sparse D/W force.

Then the Hessian h-sweep is performed with the **dense f64 non-SCC path only**.

And the Hessian is explicitly filled symmetrically:

```rust
hess[i][j] = h_ij;
hess[j][i] = h_ij;
```

after which the test measures “Hessian asymmetry.”

That asymmetry is identically zero by construction.

There is another tautology:

```text
h_ref = 0.05 Å
candidate h values include 0.05 Å
```

so the 0.05 entry has exactly zero difference from the chosen reference.

Therefore the test is practically guaranteed to find at least one “plateau.”

And finally, `band_energy()` sums occupied eigenvalues **without the closed-shell factor 2**, while sparse `Tr(KH)` similarly uses the spinless K. So those derived “forces” are half the physical electronic band-force magnitude. That is okay for an internal mathematical comparison, but it is not a physical force test.

The real Gate E must wait for the actual analytic sparse force.

---

## 14. What the real Gate E should do

Once the sparse analytic force bridge exists, use:

$$
F_\mathrm{sparse}(R)
$$

directly.

Then test same-coordinate repeatability under:

```text
cold q start
central q warm-start
small perturbed q start
tight vs loose SCC
tight vs loose TC2
```

and compare actual force components.

Then, without symmetrizing anything,

$$
H_{:,a}(h)=
-\frac{F(R+h e_a)-F(R-h e_a)}{2h}.
$$

Measure

$$
\frac{\|H-H^T\|}{\|H\|}
$$

on that raw matrix.

Do this for:

```text
0.01
0.02
0.05
0.10 Å
```

and compare to a dense-f64 analytic-force Hessian plus optionally a five-point reference.

That will finally tell us whether 0.02, 0.05, or 0.1 Å is sensible.

Right now the reported “h=0.02 Å plateau” has no evidential value.

---

## 15. The analytic angular derivative work itself looks good

There is good work in the commit.

The ss/sp/pp differentiation is mathematically the right form:

$$
\frac{\partial r}{\partial R_a}=u_a
$$

and

$$
\frac{\partial u_i}{\partial R_a}
=
\frac{\delta_{ia}-u_i u_a}{r},
$$

with the pp decomposition

$$
V_\pi\delta_{ij}
+
(V_\sigma-V_\pi)u_i u_j.
$$

The implementation follows those formulas and has tests against FD for many directions and distances.

The unit handling also appears internally consistent:

* SK radial derivative per Bohr;
* angular \(1/r\) with r in Bohr;
* final multiplication by `ANG2BOHR` to obtain derivative per Å.

So I would **keep this angular derivative machinery**.

The correction is to feed it radial \(V,V'\) from the canonical C² spline instead of `EqGridTable` Hermite.

---

## 16. Do not adapt the present dense force function directly for production sparse forces

`non_scc_electronic_force()` now has analytic derivatives, but as a performance implementation it is terrible for the nanocrystal use case.

Inside every pair it allocates multiple `Vec<f64>` objects:

```text
DM block
EDM block
H
S
dH/dx,y,z
dS/dx,y,z
```

That is fine as a CPU reference.

Do **not** build a “bridge” by repeatedly extracting BSR blocks into those Vecs.

Instead make a dedicated sparse-force GPU path.

I would use:

```text
kernel 1: one directed/undirected physical pair
          evaluate SK V,V'
          analytic angular derivatives
          read K_ij and W_ij
          output float4 pair_force[p]

kernel 2: one atom
          gather incident pair_force[]
          sum into F_atom
```

This avoids atomics and remains deterministic, which is especially desirable for Hessians.

Precompute once:

```text
pair i,j
BSR block index in M_HS
BSR transpose index
physical orbital counts ni,nj
species-pair table id
```

No CSR binary searches in the force loop.

---

## 17. Padded H is fine, but E_dummy should not control the physical TC2 condition number

The nonsingular padding idea is valid.

But the test hardcodes:

```rust
E_DUMMY = 2.0 Ha
```

A very high dummy energy unnecessarily widens the spectral interval used to initialize TC2.

There is a cleaner way.

Since dummy orbitals are exactly decoupled and explicitly unphysical:

```text
S_dummy = I
Z_dummy = I
K_dummy = 0
```

and \(K_\text{dummy}=0\) is an exact invariant of both TC2 branches.

Therefore exclude dummy orbitals from the physical spectral-bound calculation and explicitly initialize:

```text
K0 dummy rows/cols = 0
```

Then E_dummy only exists to make the padded generalized matrix nonsingular. It no longer dictates purification difficulty.

Also make sure the SCC H update does **not** accidentally turn:

```text
H_dummy = E_dummy
```

into:

```text
H_dummy = E_dummy + V_H
```

simply because `S_dd=1`.

Keep the dummy diagonal explicitly fixed.

An `active_orbital_mask` of four bits per atom is enough.

---

## 18. Newton-Schulz is “device resident” only inside the loop

`newton_schulz_inverse_dev()` starts by doing:

```rust
s.to_host(...)
```

to calculate `||S||∞`.

It then reads row/column CSR arrays back from GPU to construct the identity.

And at convergence it downloads Z back to the host.

For one standalone test this doesn't matter much.

For geometry optimization/Hessian it is the wrong architecture.

The host already owned the CSR pattern before upload. Never download it merely to reconstruct something known.

Keep:

```rust
struct BsrPattern {
    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    ...
    gpu: GpuBsrStructure,
}
```

The device structure should not be the sole owner of structural knowledge.

Likewise keep Z on GPU after convergence. For Hessian displacements:

```text
Z = central Z0
correct Z against new S with a few NS iterations
```

No download/reupload cycle.

---

## 19. K0 and spectral-bound construction still use the old allocation-heavy wrappers

`build_k0()` currently calls host-style `matmul_masked`, creates buffers, downloads matrices, reuploads them, then symmetrizes via another host-returning path. `spectral_bounds()` similarly materializes a host BSR matrix and runs CPU Gershgorin.

For one initial solve, CPU Gershgorin itself is not costly.

The bad part is moving the intermediate matrix through the CPU merely to do it.

For production:

```text
T_ZH = Z*H          persistent GPU buffer
```

and compute row-sum bounds directly with a GPU reduction yielding only:

```text
emin
emax
```

two scalars.

For Hessian/SCC calculations, either recompute cheap conservative bounds device-side or maintain a safely padded envelope.

---

## 20. The performance audit currently lies

I like the `SparsePerfStats` idea.

But:

```rust
let largest_dense = 0; // P0 contract
```

does not check anything.

The supposedly sparse engine actually has a dense `SccResult`, dense H/S, dense→BSR conversion, and eventually `K.to_dense()`, while the audit prints:

```text
largest dense alloc 0 bytes
```

So this metric actively gives false reassurance.

Also many statistics aren't actually filled by the caller, and `t_scc` plus `t_tc2` would be double-counted if `t_scc` is inclusive.

I would use real counters at the GPU wrapper/runtime level:

```text
buffer allocations count
current/peak GPU bytes
kernel launch count
blocking read count
blocking read bytes
queue finish count
structure upload bytes
plan upload bytes
```

and make stage timings explicitly either:

```text
exclusive
```

or:

```text
hierarchical/inclusive
```

but never sum overlapping timers.

The `sparse_firewall` is useful, but it is opt-in rather than default.  More importantly, it catches `Bsr4Matrix::to_dense()`, not a dense `DMatrix` created somewhere else. Type-level separation of the production sparse API is more robust.

---

## 21. Mask construction is still O(N²)

Despite P0 being marked done, `build_geometric_mask()` still does:

```rust
for i in 0..n {
    for j in 0..n {
        distance...
    }
}
```

For N=1000 once per frozen topology this isn't disastrous.

But it means “end-to-end O(N)” is not yet true.

Use a cell list or, even better in this application, consume the existing FireCore neighbor/topology information where convenient.

Also `build_product_mask()` still uses:

```rust
if !neighbors.contains(&j)
```

inside nested loops.

Replace with a stamping array:

```rust
marks[j] != stamp
```

which makes deduplication O(number of generated candidates).

Again: setup only, so not urgent—but easy.

---

## 22. GPU B-spline cutoff behavior deserves another test

The GPU evaluator itself now has the correct \(B_2\), which fixes the typo we found in the earlier manifest.

But `cubic_interp_params()` clamps the interval index:

```c
i = clamp(i, 1, n_grid - 3);
```

and therefore can extrapolate with \(t<0\) or \(t>1\) unless the caller has already explicitly rejected distances outside the table.

The CPU C² helper instead has explicit boundary behavior.

Do not validate merely:

```text
CPU Bspline evaluator == GPU Bspline evaluator
```

on interior points.

Test real Si-Si and Si-H SKF tables at:

```text
each knot ± epsilon
first physical sample
SK bond distances
start of cutoff tail
cutoff - epsilon
cutoff
cutoff + epsilon
```

for:

$$
V,\;V',\;V''.
$$

And verify the actual SKF physical grid origin. This commit fixed the one-grid-step problem in the GPU-prep path by prepending the r=0 dummy before resampling, which is the right idea. The canonical sparse H/S path must preserve that convention too.

---

# What I would tell the coding agent to do next

In this order:

1. **Unmark P1/P2/Gate C/Gate E as completed.** P2 angular differentiation is implemented, but P1 is not canonical in the sparse force call graph; Gate C and E do not yet test their advertised properties.

2. **Unify SK interpolation first.** Replace production `EqGridTable` Hermite evaluation with one C² spline representation used by H/S and analytic force. Delete the production Neville-FD tail derivative.

3. **Build a genuinely sparse SCC workspace**, not a sparse postprocessor of `SccResult`. H0/S, Hscc, K, q and force must form one self-consistent sparse calculation.

4. **Fix TC2 immediately:** residual before update, no recomputed KSK, no trace host read each iteration. This should reduce the normal iteration from ~4 SpGEMMs to 2.

5. **Integrate symbolic plans into TC2/NS/K0/W**, rather than leaving P4 as a standalone test.

6. **Make one persistent `SparseSystemWorkspace`** containing all frozen patterns, GPU structures, plans and scratch matrices. No `Buffer::builder`, `GpuBsrStructure::new`, product-mask construction, or matrix host roundtrip in SCC/force/Hessian hot loops.

7. **Implement the sparse analytic force directly on BSR blocks**, preferably pair-force + atom-gather, not through the allocation-heavy dense CPU force API.

8. **Redo Gate C on real passivated Si/H systems**, compute exact outside-mask leakage, and test whether a fixed RK/RZ remains accurate as N grows.

9. **Redo Gate E only after #7**, using actual SCC analytic sparse forces and unsymmetrized force-difference Hessians.

10. Only after those pass should the agent spend significant effort on secondary kernel tuning such as packed plans, degree buckets, alternative lane mapping and selective compensated reductions.

The commit has moved the project in the right direction, especially the analytic SK angular derivatives, the \(D/W\) formulation, and the symbolic-plan concept. But right now the largest gains are **not another clever OpenCL micro-optimization**. They are removing whole unnecessary matrix products, eliminating synchronization/allocation from the harness, and ensuring the force is genuinely the derivative of the same smooth self-consistent sparse model whose Hessian you intend to diagonalize.




---

# USER

OK, we updated it, based on comments in doc/prokop/reports/2026-09-10_gpu_accuracy_physics_performance_second_review.md
and added CLI unified engine using rhai script to avoid ad hoc test.
plese review it again now, both scientific rigor, performance, and f32 numerical acuracy issue, suggest imporvemnt which does not hamper speed while overcoming f32 limits. as wel as perforamnce improvement. Focus on the sparse large systems


---


# Chat GPT 5.6 sol

I reviewed the current sparse path at `e965ae00b7deff7ef4b0928f54a42cf35fd79039`, including the second-review document and the new Rhai-driven `SparseDftb` engine. This is a **substantial improvement** over the previous version. The architecture is finally beginning to look like a persistent solver rather than a collection of tests: `sparse_new()` constructs one `SparseDftb`, subsequent `sparse_scc`/`sparse_eval` calls reuse it, the TC2 loop now uses the precomputed symbolic plans for the dominant KS/KSK products, and the production SK evaluator really is the same C² B-spline for values and derivatives.

But I still would **not yet call it a large-system sparse DFTB solver**. There are three blockers more important than any further OpenCL micro-optimization: the persistent engine still carries dense \(N^2\) host matrices, the force path still contains an \(N^3\) dense host calculation, and the returned SCC state is not quite internally self-consistent. I also found what I think is the exact cause of the mysterious Newton–Schulz “N4” residual problem.

### Current status

| Priority    | Finding                                                                            | Consequence                                                      |
| ----------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **BLOCKER** | `SparseDftb` still builds/stores dense H/S/K arrays                                | memory/time cease to be linear before GPU sparse algebra matters |
| **BLOCKER** | sparse force builds \(W=2KHK\) by two dense CPU matmuls                            | \(O((4N)^3)\); impossible for 1000-atom Hessians                 |
| **BLOCKER** | final SCC repair stores \(q_\mathrm{fin}\) but retains \(H,V\) from previous \(q\) | energy/force are not exactly one stationary electronic state     |
| **HIGH**    | only one structural mask is used for H/S, K and Z                                  | destroys the intended independent \(R_K,R_Z\) locality control   |
| **HIGH**    | NS device residual misses a square root                                            | apparent “f32 NS failure” is probably just a contract bug        |
| **HIGH**    | persistent Z “identity reset” leaves stale off-diagonal entries                    | subsequent geometries do not actually start from \(\alpha I\)    |
| **HIGH**    | K is downloaded/densified every SCC iteration just to evaluate energy              | major synchronization + \(N^2\) CPU overhead                     |
| **MEDIUM**  | one unnecessary KS after every converged TC2 + duplicated ZH in K0                 | several avoidable SpGEMMs per SCC iteration                      |
| **MEDIUM**  | gamma functions are reevaluated \(O(N^2)\) every SCC iteration                     | expensive exponentials repeated although geometry is unchanged   |
| **MEDIUM**  | several supposedly persistent helpers still allocate GPU buffers                   | hidden `Buffer::builder()` remains inside SCC                    |
| **LATER**   | plan packing, degree buckets, multi-accumulator SpGEMM                             | worthwhile after the above is removed                            |

## 1. The biggest remaining architectural problem is still dense host storage

`SparseDftb` itself now persists correctly, but it stores

```text
h0_phys : f64[Norb²]
s_phys  : f64[Norb²]

h0_pad  : f32[(4N)²]
s_pad   : f32[(4N)²]
h_scc_pad : f32[(4N)²]
k_pad     : f32[(4N)²]
```

and `set_coords()` calls dense `HamiltonianBuilder::build_non_scc()`, copies the entire dense matrices, pads them to another dense representation, then finally extracts only the BSR blocks that the GPU actually needs.

For 1000 Si atoms, \(N_\mathrm{orb}=4000\). Those persistent arrays alone are approximately

$$
2(4000^2)(8)+4(4000^2)(4)
\simeq512\ {\rm MB},
$$

before counting the temporary dense `DMatrix` H and S. The sparse GPU matrix may be small while the host silently consumes close to a gigabyte.

I would fix this **before optimizing the SpGEMM further**. Implement direct sparse H/S assembly into BSR values. CPU first is completely sufficient:

```text
for physical pair (i,j) in frozen HS pair list:
    evaluate SK(rij)
    rotate 4x4 / 4x1 / 1x1
    write directly to BSR block b(i,j)

for atom i:
    write onsite diagonal block
    write dummy diagonal S=1, H=E_dummy
```

No dense `DMatrix`, no `h0_phys`, no `h0_pad`, no `fill_bsr_values_from_dense()`.

Then GPU assembly can replace the CPU pair loop later without changing the data architecture.

This is especially important for Hessians because only one atom moves. Once the central H/S is stored, displacement of atom \(a\) requires updating only blocks involving \(a\), i.e. \(O(n_\mathrm{neigh})\), rather than rebuilding the entire Hamiltonian.

## 2. The current “sparse analytic force” is still catastrophically dense

This is now probably the single worst runtime operation.

`SparseDftb::forces()` calls `sparse_analytic_forces()`. That calls `dw_from_k_padded()`, converts full padded K and H to f64, and explicitly performs

$$
T=KH,
\qquad
W=2TK
$$

with two dense three-loop CPU matrix multiplies.

For \(4N=4000\), one naive matrix multiplication has roughly

$$
4000^3=6.4\times10^{10}
$$

scalar multiply-add terms. You perform two. That completely overwhelms the sparse solver.

Ironically, you already implemented most of the correct replacement: `SparseDWWorkspace` has persistent T/W buffers and symbolic plans for sparse \(KH\) and \(TK\).

So the immediate solution is straightforward:

$$
T=P_{M_{KH}}(K H_\mathrm{SCC}),
$$

$$
W=P_{M_{HS}}(TK).
$$

Then contract only the physical H/S pair blocks.

I would not even construct `D=2K`. The force kernel knows that \(D=2K\); just multiply the K contribution by 2. Likewise W does not need a separate scaling kernel if the contraction knows its spin factor.

A good transitional architecture is:

```text
GPU:
    T = K Hscc
    W = T K             only M_HS output

download only W[M_HS] and K[M_HS]
CPU:
    analytic pair-force contraction
```

That already turns the catastrophic \(N^3\) force into sparse algebra while retaining your well-tested CPU force formula.

Production should then become:

```text
GPU T
GPU W
GPU pair force -> pair_force[p]
GPU atom gather -> F[i]
```

with no atomics.

Eventually you can fuse the second \(TK\) product and force contraction for H/S blocks so W never reaches global memory at all, but I would not do that until the simpler sparse force is validated.

## 3. There is still a real finite-SCC consistency bug

The finalization is better conceptually but not quite correct.

Currently `finalize_scc()` takes `q_new`, constructs

$$
V[q_\mathrm{new}],
\qquad
H[q_\mathrm{new}],
$$

purifies that Hamiltonian to K, and then obtains

$$
q_\mathrm{fin}=q[K].
$$

But after that it sets `self.q = q_fin` while `self.v` and `h_scc_pad` still correspond to `q_new`. `store_energy(q_fin)` then evaluates the SCC double-counting term using \(dq_\mathrm{fin}\) multiplied by \(V[q_\mathrm{new}]\). The force similarly receives `q_fin` together with \(H_\mathrm{SCC}[q_\mathrm{new}]\).

At exact convergence these are identical. At finite tolerance they are not.

For geometry optimization and especially a Hessian, I would make the state contract explicit:

$$
q_\mathrm{in}
\stackrel{H[q]}{\longrightarrow}
K
\stackrel{\rm Mulliken}{\longrightarrow}
q_\mathrm{out}.
$$

Finalization should continue until

$$
R_\mathrm{SCC}
=
\sqrt{\frac1N\sum_A(q_A^\mathrm{out}-q_A^\mathrm{in})^2}
$$

is below the final force tolerance.

Then store both `q_in` and `q_out` for diagnostics rather than silently replacing one with the other. The state used for H, K, W, V and force must be clearly identified.

This matters because

$$
W=2K H K
$$

has the desired EDM interpretation when K is the occupied projector of the **same H**. TC2 idempotency alone does not guarantee that.

## 4. Add a final Hamiltonian compatibility residual—but cheaply

The production `SparseDftb` still leaves `r_h = NaN`.

I would calculate \(R_H\) only after final SCC convergence, not in the inner TC2 loop:

$$
R_H=
\frac{\|HKS-SKH\|_F}
{\|HKS\|_F+\|SKH\|_F+\epsilon}.
$$

There is a useful simplification because H, K and S are symmetric:

$$
SKH=(HKS)^T.
$$

And you already have

$$
T=KS
$$

at the end of TC2.

Therefore compute only

$$
A=HT
$$

and measure

$$
\|A-A^T\|.
$$

That reduces a naive four-product commutator diagnostic to **one additional sparse multiplication** at final SCC only.

This is exactly the sort of numerical rigor we want: cheap enough to keep, strong enough to catch a beautifully idempotent projector onto the wrong subspace.

## 5. The single-mask design needs to be undone

`SparseDftb` currently builds one mask using approximately maximum SK cutoff + skin, then gives that same mask to H/S and K; workspace Z consequently also lives on that K mask.

Scientifically these are different locality problems:

$$
M_{HS}: \text{physical two-centre interaction support},
$$

$$
M_K: \text{density-matrix locality},
$$

$$
M_Z: \text{inverse-overlap locality}.
$$

There is no reason \(R_K\) or \(R_Z\) should equal the SK cutoff.

For passivated Si I expect K to be local because of the gap, but the whole point of Gate C is to **demonstrate** that. Production should therefore have independent:

```text
R_HS
R_K
R_Z
```

and corresponding product masks/plans.

Also, don't project Z down to \(M_K\) before evaluating \(ZHZ\). Use all available \(M_Z\) information and project only the resulting K0 onto \(M_K\).

Until this exists and is tested on actual Si/H clusters of increasing size, I would avoid claiming genuine linear-scaling locality.

## 6. I think the NS “N4” problem is now understood

This is an important finding.

`identity_residual_to_dev()` is documented and implemented as

$$
\|A-I\|_F^2.
$$

But `identity_residual_scalar_dev()` returns the resulting scalar directly without `sqrt()`. Then `newton_schulz_inverse_dev()` treats it as the norm and divides only by \(\sqrt{N}\).

So change the contract either to

```rust
Ok(host[0].sqrt())
```

or, better, rename the lower-level quantity explicitly to `identity_residual_sq`.

Then validate:

```text
same GPU T=ZS
device reduction
host f64 reduction of downloaded identical T
```

They should agree to ordinary f32 reduction accuracy.

Once this is fixed, **remove the current full T download every NS iteration**. `SparseSystemWorkspace::compute_z()` currently does exactly that as an N4 workaround.

This should leave one scalar diagnostic read rather than an entire sparse matrix transfer.

### There is a second NS bug nearby

`bsr4_build_identity_dev()` only writes the diagonal blocks. It does not zero off-diagonal blocks.

But Z is persistent. Therefore on the second geometry:

```text
Z contains previous inverse
build_identity_dev(Z)
```

overwrites only diagonal blocks; old off-diagonal Z values survive. Then `scale_dev()` scales **all blocks**.

So the resulting initial Z is neither \(\alpha I\) nor a sensible previous-Z warm start.

Do one of two things deliberately:

```text
cold start:
    one kernel writes every structural entry:
    diag physical -> 1
    everything else -> 0
```

or

```text
warm start:
    retain previous/central Z exactly
    Newton-Schulz correct it against new S
```

For Hessians the second option is much better. If \(Z\simeq S^{-1}\) and

$$
S'=S+\delta S,
$$

one NS step starting from the old Z gives approximately

$$
2Z-ZS'Z
\simeq
Z-Z\,\delta S\,Z,
$$

which is precisely the first-order inverse correction. For small ±h displacements I would expect only a few NS iterations.

Use the **same central Z** independently for +h and −h; don't warm-start −h from +h, which would introduce history asymmetry.

## 7. There are still several hidden buffer allocations in the “persistent” loop

The central TC2 part is genuinely much better. But two helpers violate the stated no-allocation contract.

`mulliken_dev()` creates a new GPU q buffer every call. `gershgorin_bounds_dev()` allocates `emin_partial`, `emax_partial` and further reduction buffers every call. Since spectral bounds are evaluated for every SCC H, these allocations are in the SCC outer loop.

Move these into `SparseSystemWorkspace`:

```text
q_atom
gersh_min
gersh_max
reduction scratch
two-scalar diagnostic buffer
```

Then absolutely no `Buffer::builder()` should occur between `SparseDftb::scc()` entry and exit.

This is easy to enforce with a runtime allocation counter.

## 8. There are three completely unnecessary sparse products/synchronizations

The TC2 redesign itself is now good: KS and KSK are computed once for the current K, and the residual is evaluated before the update. That fixes the previous factor-of-two SpGEMM waste.

But three opportunities remain.

First, when TC2 accepts K, `t_ks` already contains \(KS\). `mulliken_charges()` immediately calculates KS again. Reuse the existing final `t_ks`; that saves **one SpGEMM per SCC iteration**.

Second, `compute_k0_from_hscc()` calls `spectral_bounds_dev()`, which computes

$$
B=ZH,
$$

and then calls `build_k0_dev()`, which computes **the same \(B=ZH\) again** before \(BZ\).

Compute ZH once:

```text
ZH
├── Gershgorin bounds
└── ZH * Z -> ZHZ
```

Third, you already construct `plan_zh` and `plan_bz`, but the actual K0 route does not use them.

And the comment saying ZH requires the generic kernel because “ZH is not symmetric” repeats the earlier conceptual confusion: the Bsym kernel requires **the right operand H to be symmetric**, not the result ZH.

So `Z*H` is perfectly suitable for the planned Bsym kernel.

Those changes are likely more valuable than another 10% inside one SpGEMM kernel.

## 9. The energy calculation should become a very cheap high-precision island

Right now every SCC iteration downloads K, expands it to dense `k_pad`, and then `trace_ab()` performs the energy contraction over the full padded \(N^2\) array.

That is unnecessary because

$$
E_{H0}=2\operatorname{Tr}(K H_0),
$$

and \(H_0\) exists only on \(M_{HS}\).

Precompute a mapping

```text
HS block -> corresponding K transpose block
```

and reduce only those blocks.

This is also where I would use selective precision:

```text
GPU:
    each atom/row computes f32 block dot products using FMA
    -> one f32 partial per atom

CPU:
    sum ~1000 partials in f64
```

Only ~4 kB comes back for N=1000, and there is no expensive GPU fp64. You get essentially perfect final scalar accumulation of the already-f32 matrix data.

You don't even need energy every SCC iteration. Make it diagnostic/configurable; final energy is enough for production geometry work.

## 10. How I would handle f32—not by making SpGEMM f64

I agree with the current report that blanket f64 is the wrong response.

For your sparse workload, I would use the following arithmetic split:

**Keep pure f32 + FMA** for K/S/H/Z/W matrix storage, all main SpGEMMs, SK GPU evaluation, Hscc construction, and ordinary pair contractions. That's where consumer-GPU fp64 would be ruinous.

**Use compensated f32 selectively** for long signed reductions: most notably gamma matvec and perhaps the final force gather. Kahan/Neumaier there costs little because these kernels are tiny compared with TC2.

**Use host/tiny-device f64** for scalar energies, SCC mixing/DIIS algebra, Hessian assembly, frequencies, and final diagnostic reductions.

I would *not* introduce compensated SpGEMM everywhere yet.

There is a particularly attractive modification to the existing planned SpGEMM that can improve **both speed and accuracy**. Each output scalar currently has one long serial accumulator over all plan contributions:

```c
float sum = 0;
for(t=t0; t<t1; t++){
    sum = fma(..., sum);
    ...
}
```

Use 2 or 4 independent accumulators over successive plan entries:

```c
float s0=0, s1=0;

for(t=t0; t+1<t1; t+=2){
    s0 = dot4_fma(block(t  ), s0);
    s1 = dot4_fma(block(t+1), s1);
}
float sum = s0 + s1;
```

This shortens the dependency chain, exposes ILP to the GPU, and reduces serial roundoff depth. Four accumulators may be even better if register pressure remains modest.

I would benchmark that **before Kahan SpGEMM**.

For rows where real cancellation is demonstrated, a compensated final combination of the 2–4 accumulators is cheap.

## 11. Your absolute TC2 tolerance is problematic for large N

This is an important large-system/f32 point.

Current TC2 uses the raw Frobenius quantity

$$
R_I=\|KSK-K\|_F
$$

against approximately \(10^{-4}\).

Imagine N=1000 with about 50 BSR neighbors. You have roughly

$$
1000\times50\times16\approx8\times10^5
$$

stored scalars.

Demanding total Frobenius residual \(10^{-4}\) corresponds to an RMS error of order

$$
10^{-4}/\sqrt{8\times10^5}
\sim10^{-7}
$$

per stored scalar—already around binary32 machine precision.

Thus a tolerance perfectly reasonable on SiH4 becomes absurdly demanding simply because the system got larger.

Use a size-independent residual, for example

$$
r_I=
\frac{\|KSK-K\|_F}
{\max(\|K\|_F,\epsilon)}
$$

or at minimum normalize by \(\sqrt{N_\mathrm{occ}}\).

Likewise record

$$
r_N=
\frac{|\operatorname{Tr}(KS)-N_\mathrm{occ}|}
{\max(N_\mathrm{occ},1)}.
$$

Keep the raw values for debugging, but **convergence should use normalized quantities**.

This alone may save many useless late TC2 iterations and prevent misidentifying normal f32 saturation as “failure.”

NS already uses the correct idea by dividing its Frobenius residual by \(\sqrt{N_\mathrm{orb}}\).

## 12. I would not worry much about f32 trace reduction yet

At \(N_\mathrm{occ}\sim2000\), one f32 ULP is only about \(1.2\times10^{-4}\) electrons. Your current occupation acceptance width of 0.05 electrons is hundreds of ULPs.

So merely storing the final trace in f32 is **not currently the limiting factor**.

If branch jitter is actually observed, make the trace reduction compensated because it is cheap: only diagonal KS blocks are touched. A `float2 (hi,lo)` accumulator or Neumaier reduction would cost virtually nothing relative to the two SpGEMMs.

But first normalize the convergence criterion and determine whether the error originates in the KS matrix itself.

Also combine `trace` and `R_I²` into one tiny device result and do **one blocking read**, rather than two separate scalar reads at every diagnostic iteration.

And I would no longer default to `check_every=1` blindly. Test 2 or 3. Since TC2 itself is expensive, there is a balance between saved synchronization and overshooting convergence.

## 13. Gamma is an unusually good place to buy both speed and precision

Current sparse SCC calls `compute_intra_shifts()` every SCC iteration. That nested loop recomputes every \(R_{ij}\), every \(\gamma(R_{ij})\), including exponentials, every time even though geometry is fixed during SCC.

For N=1000, simply precompute

$$
G_{ij}=\gamma_{ij}(R)
$$

once per geometry.

A dense f64 \(1000\times1000\) gamma matrix is only **8 MB**. That's tiny compared with what the code currently allocates.

Then every SCC iteration becomes merely

$$
V=G\,\Delta q.
$$

Since q is already on the CPU as f64, I would initially do this matvec in **CPU f64**. A million multiply-adds per SCC iteration is cheap, requires no extra GPU synchronization, and completely sidesteps f32 cancellation issues in gamma.

This is one of the rare places where f64 can actually be *faster overall* because you're eliminating repeated transcendental functions.

For Hessian ±h it gets even better: only one atom moves, so relative to the central gamma matrix only one row/column changes. Update that row/column in \(O(N)\), rather than rebuilding \(O(N^2)\).

Only move gamma matvec to GPU if profiling later says the f64 CPU matvec matters.

## 14. Hscc should then be built on GPU from N atom potentials

Once V is known, don't rebuild a dense `h_scc_pad` on the CPU and upload all H blocks.

You already have `bsr4_build_Hscc`.

Modify it to respect the physical orbital count:

```c
bool active = (r < n_orb[i]) && (c < n_orb[j]);

H[p] = active
     ? fma(S[p], 0.5f*(Vi+Vj), H0[p])
     : H0[p];
```

Then dummy H orbitals retain exactly `E_dummy`, and per SCC iteration you upload only N atom potentials rather than Hscc.

That eliminates another \(N^2\) CPU operation and a large host→GPU transfer.

## 15. Mulliken should explicitly ignore dummy H orbitals

Current OpenCL Mulliken always sums all four diagonal elements of an atom block.

Today you rely on dummy occupations being almost zero. Better make the mathematics exact:

```text
Si: lanes 0,5,10,15
H : lane 0 only
```

Pass `n_orb_atom` or a 4-bit active mask.

Then separately monitor dummy occupation as a diagnostic.

Also tighten the charge-conservation test. Current acceptance allows

```text
|sum(q)-N_elec| < 0.5 electron
```

which is much too loose for detecting implementation errors.

A much stronger internal identity is

$$
\sum_A q_A \approx 2\,\mathrm{Tr}(KS).
$$

Those are computed from the same K and S and should agree very closely. Separately compare \(2\,\mathrm{Tr}(KS)\) with the requested electron number using the TC2 rank tolerance.

## 16. The mask “skin” implementation currently defeats the purpose of a skin

`set_coords()` rebuilds the geometric mask from scratch at every geometry and demands exact equality with the frozen mask.

`build_geometric_mask()` itself is \(O(N^2)\).

A Verlet-style skin should do the opposite.

Build structural support out to

$$
R_\mathrm{struct}=R_\mathrm{physical}+R_\mathrm{skin}.
$$

Store the coordinates at which that support was built. Rebuild only if

$$
2\max_i|\mathbf R_i-\mathbf R_i^\mathrm{build}|
>R_\mathrm{skin}.
$$

Inside the structural mask, H/S are simply zero whenever the present distance exceeds the physical cutoff.

For a Hessian, choose the skin to cover all ±h displacements and **never rebuild at all**.

That removes another \(O(N^2)\) operation from every geometry update and eliminates mask jitter by construction.

The new stamping implementation in `build_product_mask()` is a good fix, incidentally; the previous `Vec::contains` problem is gone.

## 17. Some CPU force infrastructure is also doing surprisingly expensive repeated setup

Until the force is moved fully to GPU, cache the CPU-side static objects.

`compute_forces_from_dw()` currently reconstructs `SystemContext`, `GammaTable`, neighbor list, parses all repulsive splines, and computes gamma shifts again.

You already have `repulsive_force_cached()` and `parse_all_repulsive()` specifically intended for initialization-time caching.

Yet `repulsive_energy()` reparses the SK repulsive sections every time `set_coords()` is called.

For a 6000-force-evaluation Hessian that sort of filesystem/string parsing is absurd overhead.

`SparseDftb::new()` should own:

```text
SystemContext/static atom metadata
repulsive spline tables
species pair indices
physical pair list
gamma species coefficients
```

and never reconstruct them.

There is also a unit/performance wart in the current CPU force neighbor construction: SK cutoff is in Bohr, coordinates in Å, but the raw SK cutoff is passed directly to `NeighborBuilder`, whose distance comparison uses coordinate units.

That makes the electronic neighbor sphere roughly 1.89× too large in radius. SK evaluation subsequently returns zero for the excess pairs, so it is mostly a performance rather than energy error, but in 3D it can mean several times too many neighbor pairs.

## 18. After the structural fixes, there are worthwhile SpGEMM optimizations

The planned kernel cleanup is correct: unused `A_col`, `C_col` and local `lcol` are gone.

After profiling N=300–1000 actual Si/H masks, I would test three things.

Pack

```text
plan_a_idx : 8 bits
plan_b_idx : 24 bits
```

into one `uint`. That halves per-plan-term index traffic.

Compile degree-specialized kernels around roughly 32/64/128/256 blocks so rows of degree 40 don't reserve local memory for 256 blocks.

And benchmark the two/four-accumulator scheme discussed above. I actually expect that one to be particularly attractive on the 3090 because it simultaneously shortens FMA dependency chains and improves numerical accumulation.

I would **not implement full Kahan inside every SpGEMM output scalar unless a frozen-input experiment shows it is needed**.

## 19. The remaining spline issue is now a boundary-condition problem, not an f32 problem

This part is much improved. Production value and derivative now really come from the same C² B-spline.

The remaining right-tail treatment—append several exact zero samples and refit—is explicitly documented as a stopgap. That's a reasonable temporary solution, but before final vibration work I would replace it with a constrained f64 preprocessing fit satisfying something like

$$
V(r_c)=0,\qquad
V'(r_c)=0,\qquad
V''(r_c)=0,
$$

while preserving the tabulated region.

That costs exactly zero additional runtime on GPU. So this is another case where we can improve Hessian smoothness without sacrificing performance.

## 20. The CLI itself is the right direction, but its current SiH4 script proves persistence, not sparsity

I like the Rhai design. The script does `sparse_new` once, SCC, repeats SCC to test warm reuse, then runs movement operations on the same object.

Keep this as the single user-facing path.

But SiH4 defaults to a full mask, so it tests persistence and physics plumbing, **not sparse scaling**.

After the blockers above, the benchmark should be the same CLI with real approximately

```text
Si~100H...
Si~300H...
Si~600H...
Si~1000H...
```

systems, and it should print actual stage timings, nnz/row-degree distributions, allocations, bytes transferred, TC2/SCC iteration counts and normalized residuals.

### What I would implement next

The implementation order I would give the coding agent is:

1. **Fix the NS squared-norm bug and stale-identity bug immediately.** Revalidate device NS, then remove the per-iteration T download and use the already-created NS plans.
2. **Remove dense matrices from `SparseDftb`.** Direct BSR H/S assembly; separate \(M_{HS},M_K,M_Z\); correct Verlet-skin semantics.
3. **Eliminate `k_to_dense_into()` from SCC.** Sparse energy reduction; reuse final KS for Mulliken; remove duplicate ZH and use `plan_zh/plan_bz`.
4. **Make SCC finalization genuinely stationary** and add final normalized \(R_H\).
5. **Replace dense CPU \(KHK\) force construction with `SparseDWWorkspace`**, then direct pair-force/gather GPU contraction.
6. **Precompute gamma and repulsive data per geometry/static lifetime.** For N≈1000 I would happily keep gamma matvec in CPU f64 initially because it is only ~\(10^6\) FMAs and q is already on the host.
7. Only then tune **f32 arithmetic**: normalized tolerances, multi-accumulator SpGEMM, compensated trace/gamma/force gather, f64 scalar energy and small SCC mixing algebra.
8. Finally benchmark packed plan indices, degree buckets and fused W→force kernels.

The encouraging part is that I don't currently see a reason to abandon the basic **f32 BSR4 + TC2 architecture**. Most of the apparent accuracy/performance limitations are still attributable to correctable contracts, unnecessary dense operations, convergence definitions, or long reductions—not to an inherent inability of f32 to solve this problem. The newly discovered NS square-root bug is a particularly clear example: a result that looked like a 100×–200× numerical-precision failure appears to be mostly a one-line norm-vs-norm-squared mistake.





---

# USER

please review the code in context of problem documented by the agents in this labbook, and suggest further path forward (focus on  sparse solver, let dense multi-system solver alone )

give clear instruction and best path forward to high speed and reasonable accuracy with f32 GPU arithmetic
https://github.com/ProkopHapala/dftbplus/commit/55ddb4c22867923810d2a950c8d9c046b8c7ab24





---


# Chat GPT 5.6 sol


I reviewed the sparse code at commit `55ddb4c22867923810d2a950c8d9c046b8c7ab24` against the newest labbook notes. I agree with the central principle now written into the repository: **f32 is the architecture**, so stability has to be designed around its floor rather than trying to converge an exact-arithmetic algorithm harder. The target remains a genuinely sparse, GPU-resident 300–1000+ atom solver with smooth forces and roughly 5% vibrational accuracy. 

My main conclusion is that the project has now crossed an important threshold: the basic sparse infrastructure is good enough. Direct BSR H/S assembly, persistent buffers, warm NS, sparse energy, sparse \(D/W\), analytic forces, DIIS, C² interpolation, and the multi-accumulator planned SpGEMM are all sound directions. The **remaining large error is primarily algorithmic, not an intrinsic f32 error**.

The 1648-atom result with \(R_I=7.9\times10^{-3}\) should not be accepted as the f32 floor. The labbook itself correctly identifies the much larger error source: truncating an intermediate product. 

## The most important correction: the source still contradicts the labbook

The present `SparseSystemWorkspace::new()` has

```rust
let m_t_ks = m_k.clone();
let m_t_zs = m_z.clone();
```

so it computes

$$
T=P_{M_K}(KS)
$$

and subsequently

$$
Q=P_{M_K}(TK).
$$

The source comment claims that the dropped pieces of \(KS\) are controlled by the same locality assumption as \(K\).

But your later experiment already disproved precisely that statement. An omitted \(T_{ik}\) can contribute to a perfectly valid in-mask \(Q_{ij}\), so truncating \(T\) is not equivalent to truncating \(Q\). The labbook estimates an \(O(10^{-3})\) perturbation from this mechanism and identifies it as the origin of the large idempotency floor. 

Exactly the same conceptual mistake remains in the current force path: `SparseDWWorkspace` stores

$$
T=P_{M_K}(KH)
$$

before computing \(W\). Its comment again says this is controlled by K truncation. It is not.

This needs to become a standing rule in the sparse implementation:

$$
\boxed{\text{truncate stored physical RESULT matrices; do not radius-truncate algebraic INTERMEDIATES.}}
$$

The literature agrees with where your experiments are pointing: sparse purification requires control of the error introduced by dropped products, and modern sparse purification work uses screening/error bounds for matrix-product contributions rather than assuming a spatial cutoff on an intermediate is harmless. ([PubMed][1])

## The Gate F regression is almost certainly not the intermediate mask

This can now be narrowed down substantially.

`gate_f_geom_opt` is SiH\(_4\), only five atoms.  And `SparseDftb` explicitly selects a full mask for \(N\le64\).

Therefore for Gate F,

$$
M_K=M_{HS}=M_Z=\text{full},
$$

so

```rust
m_t_ks = m_k.clone()
```

does **not truncate anything**. This exonerates suspect #1 in the labbook for this particular regression.

I would instead focus first on two interacting changes.

The TC2 loop now sometimes returns a “best plateau” projector with \(R_I\sim10^{-4}\), while the SCC mixer is allowed to extrapolate strongly. Then `SparseDftb::scc()` has this hard abort:

```rust
if it > 2 && rms > rms_prev * 2.0 && rms > 1e-2 {
    return Err(...);
}
```

DIIS residuals are not monotonic. A temporary jump from \(3.5\times10^{-3}\) to \(3.3\times10^{-2}\) is evidence that the proposed DIIS step was bad; it is **not yet evidence that the electronic fixed point does not exist**. That line turns a recoverable mixer overshoot directly into a fatal solver error.

For Gate F I would reproduce step 16→17 from saved coordinates/charges and compare exactly four variants: old TC2 versus new trace machinery, each with planned four-accumulator SpGEMM on/off. Print only \(R_I\), high-precision trace, selected TC2 polynomial, SCC RMS, DIIS step norm, and whether DIIS was accepted. Since the mask is full, this cleanly isolates arithmetic/control-flow effects.

## The current trace guard is useful containment, but it is not the invariant you actually need

The labbook correctly saw an eigenvalue just above one:

$$
\lambda=1+\delta
\quad\Rightarrow\quad
\lambda^2\simeq1+2\delta,
$$

which explains the observed doubling runaway. 

The new guard rescales

$$
K\leftarrow
\frac{N_\mathrm{occ}}{\operatorname{Tr}(KS)}K.
$$

That pins the trace. But it **does not guarantee**

$$
0\le\lambda_i(KS)\le1.
$$

You could have one eigenvalue at 1.01 and another deficient by 0.01 while the total trace is exactly perfect. So I would change the comments: trace rescaling is a useful emergency containment mechanism, not proof that the TC2 stability domain has been restored.

I also dislike this line conceptually:

```rust
trace = Nocc*(1+1e-6)  // force Q branch
```

It deliberately fabricates a trace value to choose one branch. That is another heuristic sitting exactly at the discrete decision that caused the catastrophe.

Trace-resetting purification is a real established family of methods, including Niklasson et al.'s TRS schemes, and is a better conceptual model for a closed-loop purifier than artificially biasing the TC2 branch. ([ResearchGate][2])

I would not immediately rewrite everything as TRS4, though. There is a simpler architectural change that I think is even more valuable.

# Purify \(P=KS\), not \(K\)

I think this deserves a dedicated experimental branch because it simultaneously attacks the **speed problem, the intermediate-mask problem, and the nonorthogonal algebra complexity**.

Define

$$
P=KS.
$$

For an exact occupied kernel \(K=C_oC_o^T\),

$$
P=C_oC_o^TS.
$$

Then

$$
P^2=P,
\qquad
\operatorname{Tr}P=N_\mathrm{occ}.
$$

And your Mulliken population is immediately

$$
q_A=2\,\operatorname{Tr}P_{AA}.
$$

Most importantly, instead of current metric TC2

$$
T=KS,
\qquad
Q=TK=KSK,
$$

which costs **two SpGEMMs per purification iteration**, ordinary TC2 on P is simply

$$
Q=P^2,
$$

$$
P_\mathrm{new}=
\begin{cases}
Q,&\operatorname{Tr}P>N_\mathrm{occ}\\
2P-Q,&\operatorname{Tr}P<N_\mathrm{occ}.
\end{cases}
$$

That is **one sparse matrix multiplication per iteration**.

There is also no wide \(KS\) intermediate anymore. \(P^2\) is computed directly and projected onto the structural P mask. That is ordinary result sparsification rather than chopping a hidden intermediate halfway through a contraction.

For the generalized eigenproblem

$$
HC=SC\epsilon,
$$

let

$$
Z\simeq S^{-1},
\qquad
A=ZH.
$$

Then the initial scaled projector becomes directly

$$
P_0=
\frac{\epsilon_{\max}I-A}
     {\epsilon_{\max}-\epsilon_{\min}},
$$

rather than building

$$
K_0=
\frac{\epsilon_{\max}Z-ZHZ}{\Delta\epsilon}.
$$

So you also eliminate the **\(ZHZ\)** multiplication from every SCC purification initialization.

After convergence, recover K only once:

$$
K=PZ.
$$

The SCC loop itself only needs P: charges come from its diagonal, and even the band energy can be written

$$
E_{H0}
=2\operatorname{Tr}(KH_0)
=2\operatorname{Tr}\left(PZH_0\right).
$$

This is a major simplification.

The catch is that P is not ordinarily symmetric, so your existing Bsym product plan cannot be reused for \(P^2\). But that is a very manageable problem: make a **generic symbolic product plan** storing `(a_block,b_block)` contributions for each output block. Then the hot kernel has the same attractive structure as the current planned Bsym kernel—no binary search or row intersection—just cached A blocks and FMAs.

I would expect the one-multiply P-TC2 to beat current two-multiply K-TC2 by much more than any 5–10% microkernel optimization.

I would prototype P-TC2 on SiH\(_4\), Si\(_{65}\), and then the ~300 atom crystal before replacing K-TC2.

## There is a similar simplification for W

Currently you calculate

$$
W=2KHK
$$

using

$$
T=KH,\qquad W=2TK,
$$

and unfortunately truncate T.

But with

$$
Z=S^{-1}
$$

and a converged occupied projector,

$$
ZHK
=
C_o\epsilon_o C_o^T
=
KHK.
$$

Therefore

$$
\boxed{W=2(ZH)K}.
$$

So if \(A=ZH\) is already available from the final SCC state, W can be generated with **one sparse multiplication**, directly onto \(M_{HS}\):

$$
W=2P_{M_{HS}}(AK).
$$

No \(KH\) intermediate at all.

This identity should first be parity-tested against \(2KHK\) on full-mask SiH\(_4\)/Si\(_{65}\), because your Z is approximate and truncated. But mathematically this is a much cleaner large-system force route.

It also gives a useful diagnostic:

$$
\|KHK-ZHK\|
$$

measures consistency among K, Z, H and the sparse approximations.

## Precision should be concentrated at decisions, not matrices

I would use this arithmetic policy:

| Quantity                                                | Arithmetic                        |
| ------------------------------------------------------- | --------------------------------- |
| H, S, Z, P/K, W storage                                 | f32                               |
| SK interpolation and rotations                          | f32 FMA                           |
| sparse matrix products                                  | f32 FMA                           |
| block product accumulation                              | 2–4 independent f32 accumulators  |
| Hscc construction                                       | f32 FMA                           |
| pair-force contractions                                 | f32 or CPU f64 while transitional |
| **TC2 trace / branch decision**                         | **host f64**                      |
| charge conservation                                     | host f64                          |
| energy final sum                                        | host f64                          |
| DIIS \(8\times8\) solve                                 | f64                               |
| Hessian construction / mass weighting / diagonalization | f64 CPU                           |
| wholesale GPU SpGEMM                                    | **never f64 by default**          |

Your four-independent-accumulator planned SpGEMM is exactly the sort of f32 optimization I like: it shortened the dependency chain, changed results only at ordinary reorder-noise level, and was already slightly faster even on the tiny benchmark. 

I would not add Kahan to every sparse dot product.

### But the trace branch really should become f64

At N=1648, reading one scalar f32 is not the optimal precision compromise.

Have a tiny kernel write one trace contribution per atom:

```text
trace_atom[i] = sum physical diagonal lanes of P_ii
```

or KS in the current architecture.

Read those N floats and sum them in host f64.

For 1648 atoms this is only about 6.6 kB per purification iteration. You already require a synchronization for the TC2 branch anyway, so the meaningful cost is synchronization latency, not those few kilobytes.

Then change the TC2 kernel from

```c
if(trace_KS[0] > Nocc)
```

to

```c
branch = host_decision;   // uint
```

The branch—the dangerous discrete event—is then based on f64. The matrices remain entirely f32.

This follows exactly the rule recorded in your labbook: cheap discrete decisions deserve high precision because the wrong decision can be catastrophic while a \(10^{-5}\) matrix-element error is usually harmless. 

Likewise the absolute

```rust
TC2_TRACE_TOL = 0.05
```

should disappear as a universal constant. A 0.05-electron tolerance might match the measured large-N f32 reduction jitter, but it is ridiculously loose for SiH\(_4\). Use a measured relative trace floor plus a tiny absolute floor.

## Do not return a bad projector as ordinary `Ok`

This is another thing I would change immediately.

The current TC2 accepts a best snapshot at exhaustion/divergence if roughly

```rust
best_r_i < 1e-2
```

and the trace is acceptable.

That is exactly why the 1648-atom calculation can proceed with

$$
R_I=7.87\times10^{-3}.
$$

The labbook itself correctly says this is a **bad ~0.8% idempotency floor and the resulting energy is not validated**. 

Returning it indistinguishably from true convergence violates the otherwise good “report the f32 floor honestly” philosophy.

Use an explicit result such as:

```rust
enum PurifyStatus {
    Converged,
    AcceptedAtNumericalFloor,
    Failed,
}
```

with `ri`, trace error, R_H, iteration count and floor reason.

`SparseDftb::scc()` can decide whether a given run mode accepts `AcceptedAtNumericalFloor`. A fast screening calculation may. A vibrational Hessian should generally not accept a 0.8% projector unless you have independently shown that it changes forces/frequencies within your accuracy target.

Do **not** encode scientific accuracy policy as `best_r_i < 1e-2`.

## The plateau detector should operate only in the endgame

The attempted “N consecutive non-improvements” detector failed because TC2 naturally oscillates while trace correction is still driving occupation. That observation in the labbook is correct. 

A better detector arms only after both

$$
|\mathrm{Tr}-N_\mathrm{occ}|/N_\mathrm{occ}
$$

and \(R_I\) are already in an endgame range.

Then maintain a short rolling history of \(R_I\) and detect either a 2-cycle or essentially zero negative slope in \(\log R_I\) over several branch cycles.

This says “we have reached the numerical/sparsity floor,” rather than “the last iteration happened not to improve.”

Also normalize the idempotency residual with a physically fixed scale. Current code computes `k_norm` once from K0 and uses that throughout. K0 is not necessarily a good normalization scale during the transient. In the proposed P representation,

$$
\|P\|_F\simeq\sqrt{N_\mathrm{occ}}
$$

for a projector, so

$$
r_I=
\frac{\|P^2-P\|_F}{\sqrt{N_\mathrm{occ}}}
$$

has a particularly clean system-size interpretation.

## Fix the SCC recovery rather than hard-failing on one DIIS overshoot

Since you asked me to leave the dense solver alone, I would **not modify the shared mixer globally**.

Put a sparse-specific safeguard around it, or create `SparseDiisMixer`.

Operate on

$$
\Delta q=q-q_0
$$

rather than the large neutral populations themselves, and project each residual onto the fixed-total-charge subspace. For a neutral system:

$$
\sum_A\Delta q_A=0.
$$

Then if a DIIS trial makes the *next measured residual* explode, restore the last accepted q, prune/reset the suspect DIIS history, take a damped linear step, and retry. Only fail after repeated rescue failure or nonfinite values.

That is a closed-loop mixer. The current

```rust
rms > 2*rms_prev
```

hard error is not.

Keeping DIIS history across small geometry moves was clearly valuable for the Si\(_{10}\)H\(_{16}\) Hessian, so I would retain that behavior, but make stale-history rejection automatic rather than assuming every carried history remains useful. 

## The spectral bound also needs to become deliberately conservative

The present K0 construction evaluates Gershgorin bounds on a **truncated \(ZH\)** and pads them by 10%.

As the labbook already recognizes, those are no longer guaranteed bounds. 

For purification, putting the initial spectrum inside the polynomial stability interval is more important than shaving two iterations.

I would initially use a conservative bound such as

$$
B=\|Z\|_\infty\,\|H\|_\infty,
\qquad
\epsilon_{\min}=-B,\quad
\epsilon_{\max}=B,
$$

or use that to enlarge—not shrink—the estimated Gershgorin interval. Matrix norm inequalities make this deliberately conservative.

Once large-system behavior is stable, you can make it tighter using measured parameter-set bounds or a cheap extremal iteration. First make K0 safe.

## For sparsity, τ-screening is the right next technology

Do not try to reach 64 neighbors by forcing \(R_K\) down to 6–8 Å. Your own Si65 tests already show that K/Z truncation is much more expensive than H/S truncation. 

The H/S situation is different. The measured SK decay and the energy sweep make the current matsci choice quite defensible:

$$
r_{HS}\approx8\ \text{Å}
$$

with the smooth 1 Å taper is a good accuracy-oriented default; 7 Å may be a good faster mode after vibration testing.  

For K/P/Z, keep a larger frozen structural super-mask—initially perhaps the 10–12 Å regime you are already testing—and make the **computational** sparsity depend on block magnitude.

A useful GPU implementation is to compute a Frobenius norm for every 4×4 block and let the symbolic product kernel skip a contribution when a conservative product bound says it cannot matter:

$$
\|A_{ik}B_{kj}\|_F
\le
\|A_{ik}\|_F\|B_{kj}\|_F.
$$

A naive per-term threshold is not rigorous because many small omitted terms can sum, so accumulate or budget the discarded bound per output block/row. That is essentially the direction used by sparse purification schemes with error-controlled submatrix-product screening. ([ScienceDirect][3])

For Hessians I would freeze the selected active pattern at the central geometry, or use strong hysteresis, so threshold crossings do not inject force discontinuities.

## The remaining safe performance gains are quite clear

The default `MAX_LEFT_BLOCKS=512` means the planned kernel reserves about 32 KiB just for cached 4×4 values per WG. The code already notes that this costs occupancy.

Your measured large-Si degrees are mostly much lower: at 8 Å max ~111, at 10 Å ~206. 

Compile degree buckets:

```text
MAX_LEFT = 64
MAX_LEFT = 128
MAX_LEFT = 256
MAX_LEFT = 512
```

and dispatch rows by degree. This is almost pure performance improvement with essentially zero numerical risk.

Likewise pack the symbolic plan indices. With `MAX_LEFT<=512`, `plan_a_idx` needs only 9 bits. One `uint` can hold local-A index plus ~23 bits of B-block index, enough for millions of blocks. That cuts plan-term index traffic roughly in half.

Keep the four independent f32 accumulators. Do not add full compensated multiplication until frozen-input tests show a meaningful remaining arithmetic error.

## Gamma is not where I would spend effort first

The current explicit dense f64 gamma matrix is aesthetically inconsistent with “GPU resident,” but for 1648 atoms it is only about 2.7 million elements, ~22 MB, and a few million f64 multiply-adds per SCC iteration. That may still be small compared with sparse purification. The code makes this explicit rather than silently falling back, which I prefer.

So first profile after fixing purification.

If it becomes important during thousands of Hessian geometries, then move it cleanly to GPU. I would pretabulate the short-range species-pair gamma correction in CPU f64 and evaluate gamma and gamma′ from the **same** smooth f32 spline on GPU, with the long-range \(1/r\) handled analytically. That preserves energy/force consistency and avoids the ill-conditioned close-U analytic formula at runtime.

## My implementation order

I would give the coding agent exactly this order:

1. **Restore Gate F first.** Recognize that full-mask SiH\(_4\) rules out intermediate truncation as the cause. Replace the SCC one-jump hard abort with rollback/rescue and AB-test trace guard × four-accumulator SpGEMM against the parent behavior. Do not touch the dense solver.
2. **Move TC2 trace and branch selection to host f64.** Read per-atom diagonal contributions, f64-sum them, pass an explicit branch flag. Replace the universal 0.05-electron contract by a measured size-scaled tolerance.
3. **Stop returning \(R_I<10^{-2}\) as ordinary success.** Add `Converged / NumericalFloor / Failed`, and make accuracy mode decide whether a floor is acceptable.
4. **Prototype the \(P=KS\) purifier.** One generic planned \(P^2\) SpGEMM per TC2 iteration, direct Mulliken from P, direct \(P_0=(\epsilon_\max I-ZH)/\Delta\), recover \(K=PZ\) only after SCC. Compare iterations, wall time, \(R_I\), force and energy against current K-TC2 on SiH\(_4\), Si65 and ~300 atoms.
5. **If P-TC2 works, make it the production baseline and then test trace-resetting/TRS-style purification.** This is the right place to put the rigorous restoring mechanism rather than bolting more heuristics onto K-TC2. Trace-resetting purification was designed precisely around predefined occupation and stable purification. ([ResearchGate][2])
6. **Remove the force intermediate truncation.** First test \(W=2(ZH)K\) against \(2KHK\); if parity is good, it gives W in one sparse product and removes the problematic \(KH\) intermediate.
7. **Implement error-budgeted τ screening**, keeping a larger frozen structural K/P/Z mask. Measure energy, forces and spectra versus τ; do not infer an error from radius alone.
8. **Then optimize GPU throughput:** degree-bucketed 64/128/256/512 SpGEMMs, packed plan indices, existing four-accumulator FMA kernel, fewer diagnostic synchronizations. Only afterward profile gamma and CPU pair-force contraction.

The most important strategic change I would make is therefore **not “better TC2 tuning.”** I would stop investing heavily in the present \(K\rightarrow KS\rightarrow KSK\) architecture because the wide intermediate is structurally at odds with the sparsity you want. The \(P=KS\) formulation looks unusually well matched to this application: it removes that intermediate, halves the purification SpGEMM count, makes charge and trace trivial, and naturally exposes where f64 should be spent—the scalar decisions rather than the matrices.

And I would keep the present 1648-atom 7.5 s run as a valuable **scaling/pipeline milestone**, not an accuracy milestone. The labbook is right to flag its \(R_I=7.9\times10^{-3}\) energy as unvalidated. 

[1]: https://pubmed.ncbi.nlm.nih.gov/18298139/?utm_source=chatgpt.com "Density matrix purification with rigorous error control - PubMed"
[2]: https://www.researchgate.net/publication/252723010_Trace_resetting_density_matrix_purification_in_ON_self-consistent-field_theory?utm_source=chatgpt.com "(PDF) Trace resetting density matrix purification in O(N) self-consistent-field theory"
[3]: https://www.sciencedirect.com/science/article/pii/S0021999121002497?utm_source=chatgpt.com "Sparse approximate matrix-matrix multiplication for density matrix purification with error control - ScienceDirect"






---

# USER


what you mean by this? 
"The fast SpGEMM kernel silently mis-computes rows wider than 512 blocks — needs a fallback path (S4)."

in general analyse everytwith with respect to the major goal, big picture. 

1) are we now doing just the neessary work? No CPU fallbacsk? No kernel location/kernel compilation inside SCC loop? No unnecessary CPU-GPU data tranfers?
2) what is still limiting the numerical floor? Did we impleemtned Kahan error compensating sum? Did we tried the hybrd f64 aproach to redocu this error
3) Are all the kernel writen in absolutely optimal way? Do they use local memory where possible ?what is the layout of workgroup size and local memory, and the tiling, maximum number of neighbors etc. 
4) what is the cutoff for interaction what is maximum number of neibhors?

do not code analyse, these are the critical things we must adress


---

# SWE-2.0

Let me check exactly what the kernel does on an over-wide row before I explain it:

There IS a host-side guard. Let me verify the production plan-based SpGEMMs (the ones used in the hot loop) actually call it:

Straight answers, big picture only. First — a correction to my earlier claim, because I verified the code just now:

**The 512-block row limit: it does NOT silently corrupt in the SpGEMM path.** Every production sparse multiply (`spgemm_*_dev`, all four entry points) checks the left matrix's max row width on the host *before* launching and fails loudly. What the kernel itself does on an over-wide row is a bare `return` — the row stays unwritten — but the driver guard fires first, so today it dies loudly, not silently. **Two real problems remain:** (a) there is *no* fallback path at all — a row wider than 512 blocks is a hard crash, the run just dies; (b) I found one kernel that is NOT guarded: `build_hscc_dev` (the H_scc assembly) caches the row's neighbor list in local memory and has the same `if(nb > MAX_LEFT_BLOCKS) return;` bail with **no host-side check** — that one really would silently produce a partially-stale H_scc. Today's masks are well under the limit so nothing has fired, but it's an unguarded hole.

Now your four questions:

## 1) Are we doing only necessary work?

- **No kernel builds/allocations inside the SCC loop** — verified by test (`F3`: zero buffer allocations between `scc()` entry and exit; all kernels/plans/buffers built once at workspace construction). ✅
- **No CPU fallback in production** — purification, inverse overlap, charges are all device-resident. The only host work in the loop is deliberate: per-iteration readback of ~N_atom floats (per-atom trace partials) so the TC2 branch decision happens in f64 on the host. That's a sync point per iteration — justified for correctness, but it is a measurable latency we haven't priced yet.
- **Force contraction still runs on CPU** — W and K are downloaded once per force call and the pair contraction is a CPU loop. Deliberate (no dense matrices involved), but it is a host path.
- **The real unnecessary work is iteration count, not plumbing**: ~80 purification iterations per SCC step on the big sphere. Each iteration is 2 sparse multiplies. Until we stop the floor-chasing (S2) and cut neighbors (S5), kernel micro-optimization is polishing the wrong thing — consistent with your "as few neighbors as possible" priority.

## 2) What still limits the numerical floor?

- **No Kahan/compensated summation anywhere.** Inside the SpGEMM, each 4×4 output block is a plain f32 accumulation over ~60–100+ block terms. That f32 accumulation *is* the floor (~1e-5 relative residual, which is what we now measure and honestly report as `NumericalFloor`).
- **Partial compensation exists only at the very end**: traces are computed as f32 per-atom partials, then summed in f64 on the host. That fixes the *decision* precision (branch on trace can't flip on noise) but does nothing for the matrix accumulation itself.
- **No hybrid f64 tried.** The work order explicitly forbids blanket-f64 as a substitute for a stable algorithm. The *targeted* candidates are: (a) compensated accumulation inside the SpGEMM inner loop (f32 Kahan or float2 pair-sum — costs ~2× flops in the inner loop, worth measuring); (b) the band-energy reduction `trace_kh0_dev` still collapses ~6600 orbitals to **one f32 scalar** — worst precision offender at scale, fixable cheaply with per-atom partials like the other trace (S6).
- **What we added this round:** the recovered-K diagnostics showed the Z≈S⁻¹ recovery loses ~1.6e-5 in trace — a *truncation* error (mask Z too narrow), not arithmetic noise. Different axis, needs mask-budget treatment (S2), not more precision.

## 3) Are the kernels optimal? (layout facts)

- **SpGEMM structure:** one work-group (128 lanes) per output matrix *row*; internally 8 teams of 16 lanes, each team produces 4×4 output blocks. The left row is cached in local memory (512 blocks × 64 B values + 2 KB column indices ≈ 34 KB per work-group).
- **That 34 KB is the occupancy killer:** on a typical NVIDIA SM (~48–100 KB local memory) you fit 1–2 work-groups per SM — very low occupancy, latency can't hide. The kernel source itself notes the fix: degree-bucketed variants (64/128/256/512) so short rows don't reserve the fat cache. Not implemented yet — S4.
- **Right operand is NOT tiled/cached** — every lane gathers B-blocks straight from global memory. A shared right-side tile (the output blocks of one row share many B rows) is a real optimization opportunity, also S4.
- **Symbolic-plan kernel** (precomputed contribution lists, no binary search at run time) exists and is used when plan build succeeds; fallback is the intersection kernel — one-time structural decision, not per-iteration.
- **Reductions** run at work-group 256 with per-atom partials — reasonable.

So: kernels are *functional and reasonably structured*, but not optimal — local memory is used but oversized per work-group (kills occupancy), the right operand isn't reused in local memory, and there's no degree bucketing. These are S4 items; they matter **after** S2 cuts the iteration count.

## 4) Cutoff / neighbor count

- **Physical interaction cutoff:** the SK table range — ~10.6 Å for matsci Si/H — optionally tapered to `r_trunc`. No screening by magnitude yet; every pair inside the geometric radius is computed.
- **Mask radii** (separate per matrix): M_HS = cutoff + 1 Å skin; K, Z, P each have their own mask radius (`r_k`, `r_z` config).
- **Measured:** at `r_k = 12 Å` on the 1648-atom Si sphere, ~61 block-neighbors per atom on average (exploratory histogram, not certified). An *untruncated* product row (K·S reaching r_k + r_hs ≈ 19 Å) would need ~1298 blocks/row — that's why intermediates are currently truncated to the target mask (a controlled approximation, and the 512 ceiling would be breached otherwise).
- **Your stated goal — minimum neighbors — is S5, not started.** Today the neighbor count is set purely by geometric radius; there is no magnitude-based dropping with an error budget. Everything about "keep the smallest neighbor set that preserves forces" is still to be designed and validated.

**Priority reading of where we stand:** the bookkeeping is now honest (S1 done, pending your acceptance). The two things that actually decide speed — iteration count to a *stated* accuracy (S2) and true neighbor reduction with error bounds (S5) — are untouched. The one new thing I'd flag from today's verification: the unguarded `build_hscc` row-degree bail — it's a one-line host check to close a real silent-corruption hole.


---

# USER

1) I think major problem is wee allo to many neighbors. I gave you clear question, how many neighs we allow? seems to me that more than 512 ?
We must agressively reduce maximum number of neighbors. I do not believe this sparse matixes are not more sparse, I do not belive Hij and Sij is so dense, and we cannot allow matrix to densitywy by multyplying them. That is absolutely most important thing to make this approach viable. When the matrixes become dense purification is useles (does not provide any preformance advantage over diagonalization)
2) what you mean " fallback is the intersection kernel — one-time structural decision, not per-iteration." ... we said there are not falbacks !!! why we need such fallback?

---

# SWE-2.0

Good questions — let me answer both precisely, then give you the layout scheme.

## 1) How many neighbors do we actually allow?

**The 512 is NOT a neighbor allowance — it's a memory limit of the kernel.** The kernel stashes the left matrix's row in fast on-chip memory (`__local`): 512 blocks × 64 bytes = 32 KB. A row wider than that simply doesn't fit — that's the whole meaning of 512. Two separate things that are easy to confuse:

- **Mask size (physics/cost):** how many atom pairs we store at all.
- **Kernel capacity (implementation):** 512 blocks per row, because of that 32 KB local cache.

**Now the actual neighbor counts, measured:**

- **H/S mask (M_HS):** every pair within the SK table range (~10.6 Å for matsci) + 1 Å skin ≈ **~200–400 neighbors for interior atoms** on a bulk Si sphere. *This is the fat mask* — and your instinct is exactly right: **H and S are not that dense physically.** Measured decay: Si–Si coupling is down to ~1e-4 Ha by ~7 Å; H–H is shorter. The 10.6 Å is where the SK *table ends*, not where the interaction dies. So the mask is 3–5× fatter than physics needs. A taper to `r_trunc ≈ 6–7 Å` (the option already exists, decay numbers support it) cuts M_HS to ~80–120 neighbors.
- **K mask (M_K):** r_k=12 Å gave ~61 neighbors/atom average on the 1648-atom sphere — and the density matrix of an insulator decays *faster* than H, so the honest r_k is likely smaller still (needs the error-budget measurement, S2/S5).
- **The densification you're worried about is real and already happening:** a product's support is the *sum* of the operands' radii. Untruncated K·S at 12+11.6 Å ≈ 24 Å reach ≈ **~1300 contributions per output block**. That exceeds 512 — which is why the code currently *truncates intermediates back to the target mask*. That truncation is an uncontrolled approximation (it deletes real contributions); the principled fix is smaller honest masks, not a bigger cache.
- **So: no, we're not running >512-row matrices — but only because products get truncated.** And yes, the masks allow far more neighbors than the physics needs. Your priority — minimum neighbor count with an error bound — is exactly S5, and the taper on H/S is the cheapest first cut available right now.

## 2) The "fallback" — you're right to flag it

It is **not a CPU fallback** — both paths are GPU BSR4 kernels. Two kernels exist for the same product:

- **Plan kernel** (fast): contribution lists precomputed once on the host — at run time each output block just reads "which pairs to multiply," no searching.
- **Intersection kernel** (slower): rediscovers matching index pairs at run time via merge/binary search — used when plan construction fails.

The violation is real though: plan-build failure currently prints a warning and *continues* on the slower kernel — a silent-ish performance fallback. Per the rules it should either fail loudly or be an explicit config choice. The genuinely missing piece is different: there is **no path at all** for a left row > 512 blocks — that's a hard error, not a fallback. Fine for fail-loud, useless for wide rows — that's what S4's bounded-memory kernel is for.

## 3) Kernel/algorithm layout scheme

```
=== ONE SCC ITERATION (all device-resident, f32) ===
1. build H_scc:  H_ij += ½·S_ij·(V_i+V_j)   physical lanes only
2. purify K  (TC2, ~N iterations):
     T = K·S              SpGEMM
     tr = Σ_A Tr(T_AA)    per-atom partials → host f64 → branch decision
     Q = T·K              SpGEMM
     [on check iters] r_I = ||Q−K||_F / ||K||_F   (device reduce → host)
     K ← squaring or complement branch; symmetrize
   (P-variant: Q=P·P only — 1 product/iter; K = P·Z recovered once at end)
3. q_A = 2·Tr((KS)_AA)    Mulliken diagonal read
4. V = γ·Δq               gamma matvec + host mixing decision
```

**SpGEMM kernel layout (the workhorse — both products above use it):**

```
grid:        one work-group per output ROW  (n_atom groups)
work-group:  128 lanes = 8 teams × 16 lanes
             16 lanes = the 16 scalar slots of one 4×4 block
             lane l ↔ element (r=l/4, c=l%4)
local mem:   lcol[512] u32          =  2 KB   (left-row column indices)
             lA[512×16] f32         = 32 KB   (left-row block values)
             total ≈ 34 KB → only 1–2 work-groups fit per SM  ← occupancy problem

per work-group:
    i = group_id
    if degree(A_i) > 512: host guard already hard-failed before launch
    cooperatively load row i of A into lcol/lA;  barrier
    for cb = team; cb < n_out_blocks(i); cb += 8:        # 8 teams stride the output row
        j = C_col[i][cb]
        acc = 0 (each lane holds one f32: acc[r,c])
        for each contribution (k-block) from plan/intersection:
            lane does: acc += Σ_{l=0..3} A[i,k](r,l)·B[k,j](l,c)   # 4 FMA
        C[i,j] = acc        # coalesced 64 B write per block
```

**Other kernels:**

```
trace_atom (Tr(T) per atom):   1 WG/atom, WG=256, no local mem → host f64 sum
idempotency/frob reduction:    flat element grid, WG=256        → 1 scalar each
build_hscc:                    1 WG/atom row, WG=128, ~4 KB local (V_j,n_orb_j cache)
                               ← SAME 512-row bail, NO host guard (hole found today)
mulliken / symmetrize / tc2_dev / axpby:  elementwise, no local mem
Newton–Schulz:                 T=Z·S (SpGEMM) + Z←Z(2I−T) + ||I−T||²/√N reduce
```

## What this means for the big picture

Your diagnosis is right and it's the crux: **the masks are set by table lengths and round radii, not by where the physics dies.** Honest numbers: taper H/S at ~6–7 Å and put K/Z at their measured decay length, and M_HS drops ~3×, product rows drop under ~150 contributions — the 512 ceiling stops mattering entirely, and each SpGEMM gets ~4–8× cheaper *because the inner loop over contributions shrinks*. Fewer neighbors = less work per product AND fewer products needed (narrower-banded iterates converge faster). That is the single highest-leverage change available — and it's S5's error-budgeted screening plus the existing `r_trunc` taper, not new machinery.

The three things still standing between here and there: (a) an honest error budget that says how small each mask can go (S2), (b) close the unguarded `build_hscc` row bail (one-line host check — trivial), (c) decide the plan-vs-intersection policy explicitly instead of warn-and-continue.


---

# USER

1) ad 
"Plan kernel (fast): contribution lists precomputed once on the host — at run time each output block just reads "which pairs to multiply," no searching.
Intersection kernel (slower): rediscovers matching index pairs at run time via merge/binary search — used when plan construction fails."

why we need that fallback? We must make sure we do not need such fallback, if wa want to make fast algorithm! it is not acceptable to randomlyswitch to algorithm wich is ,ay times slower!

2) ad "The 512 is NOT a neighbor allowance — it's a memory limit of the kernel." - but the size of the row memory depend on maximum number of neighbors, if we reduce number of neighbors we can use less maximum error alocated for ro of each matric, are we on the same page here?

3)add "The densification you're worried about is real and already happening:"  -  we cannot allow such desificaltion to happen, because every matrix multiplication doubles the bandwidth of the matrix, we quickly run out of local mmeory and get dense matrixes. We MUST truncate!!!!!!!!!! that is absolutely essential to do truancate after every matrix multiplication to restore limited bandwidth at lease that is what I think .Can you analyse it again with this goal in mind? 

---

Yes — we're on the same page on all three. Let me take them in order.

## 1) The fallback: agreed, it shouldn't exist

Why it exists at all: the intersection kernel is the *older* kernel; the plan kernel was added later (P4/A3) and the fallback was left as a safety net while plans were being proven. Today every production product builds a plan at workspace construction — the fallback only fires if plan *construction* fails, which with honest-sized masks shouldn't happen.

The right fix, matching your rule: **plan-build failure on a production path should be a hard error, not warn-and-continue.** Keep the intersection kernel only for unit tests/debugging where correctness-not-speed is being checked. That converts an invisible many-times-slower regime into a loud crash you can actually fix. Small change, and I agree with your reasoning completely — a solver that silently switches to the slow path is worse than one that fails.

## 2) Row memory = max neighbors: yes, exactly

The local-memory footprint is literally `max_left_degree × 64 B` (one 4×4 f32 block per left neighbor) plus `× 4 B` for column indices. So:

| max neighbors/row | local mem per WG | work-groups per SM (48–96 KB) |
|---|---|---|
| 512 (today's cap) | ~34 KB | 1–2 — occupancy starved |
| 128 | ~8.5 KB | ~5–8 |
| 64 | ~4.3 KB | ~10–15 |

**Fewer neighbors is a triple win, not a single one:** less work per product (shorter inner loop) AND less memory per work-group (higher occupancy, latency hiding) AND fewer SCC/purification iterations needed (narrower-banded matrices converge faster). Your instinct — aggressively cut neighbor count — is the highest-leverage knob in the whole design. The degree-bucketed kernel variants (compile 64/128/256/512 sizes, pick per-row) are exactly the mechanism to *pay only for the degree you actually have*.

## 3) Truncation after every multiply — re-analysis with that as the axiom

You're right, and this is not just your opinion — it's the established method in the purification literature (the chat file even cites Rubensson's "sparse approximate multiplication for density matrix purification **with error control**"). For an insulator, the density matrix decays exponentially with distance, so a product's *significant* elements stay inside a band far narrower than the geometric reach `r_A + r_B`. If you never truncate, two or three multiplications densify everything and purification loses its entire advantage — exactly as you say. **Bounded-bandwidth algebra is the only viable design.**

So the code's *structural* choice — every product writes onto a fixed target mask — is correct. The problems are in *how* the truncation is done and reported:

- **Today truncation is blind.** Contributions outside the mask are simply never computed, and nobody measures what was dropped. The fix that turns this into *your* strategy: since the plan already enumerates every (i,k,j) contribution, computing the **dropped mass** (sum of block norms excluded from the mask) costs almost nothing — do it once at plan build on the host, and optionally re-measure the kept-vs-dropped split on device during production. Then truncation is no longer a silent approximation — every product *reports* how much it threw away, and a truncation that drops real physics fails loudly instead of quietly polluting forces.
- **The mask must be sized by physics, not by round numbers.** For K (density matrix of an insulator) and Z=S⁻¹ (exponentially decaying when S is well-conditioned — the caveat: if S is ever nearly singular, Z has long-range content and any fixed mask breaks — worth a conditioning check) a modest radius captures almost all mass. The geometric mask is the *cheap* version of magnitude-thresholding: fixed structure means fixed buffers and static plans — the right trade for GPU-resident work. The missing piece is the error budget that proves the chosen radii keep the dropped mass under the accuracy target — that's precisely S2/S5.
- **Truncation restores structure, not values.** What stays is the sum of only the in-mask contributions — the dropped part is an error. Purification is partially self-correcting (it drives eigenvalues toward 0/1, cleaning small per-step truncation noise), which is why truncated purification works at all — but the same truncation applied to the force-path products (Z·H, K·S, W) lands directly in the forces with no self-correction. That's why the dropped-mass measurement matters most exactly where it isn't done yet.
- **Consequence for the 512 limit:** once every product is truncated to an honestly-sized mask (say ≤128 blocks/row), the kernel never sees a wide row — the ceiling stops being a correctness question and becomes purely a sizing parameter you set to match the mask. The bounded-memory SpGEMM (S4) is then insurance for the wide-mask *diagnostic* runs, not a production need.

**Bottom line:** truncate after every multiply — yes, mandatory, and already the architecture's intent. What must change: masks sized by measured decay + dropped-mass accounting on every product, so truncation is a *controlled* approximation instead of a silent one. That plus cutting H/S to its physical ~6–7 Å is the difference between a sparse solver that wins and one that's just a slower diagonalization.




---

# USER

please analyse the sparse code and kernels again. 1) I think critical think for the desing is to keep number of neighbors and band-width (sparcity patern) under control duting all multiplications, which means basically agresively pishing down the number of neighbors, and truncated elebemtns which would lead to desnification (even at the cost of accuracy, although we should attepnt for some compromise preseverve reasaoble accuracy while truncating), not sure if better apporach is set the sparcity pattern of the results fixed from start so we compute only the valid eleemnts (that is perhsp faster and makes easily kernel which can be better optimized - perhaps I slightly prefer that), or the compute more lements (halo) but then agresively set o zero smaller elements before passing the matrix to next matrix multiplication (that perhaps preserve more accuracy, but result in more complex code and redudant work) 2)the other quetion is where exacly we should apply Kahan error compensating sum or f64 sum in imporve accuracy without copromising performance on our cheap hardware (which is like 40x faster in f32 than f64) 
https://github.com/ProkopHapala/dftbplus/tree/main/rust_dftb/src/methods/sparse 
sparse_bsr4_purification.cl 
gpu_sparse.rs see discussion which I had with my coding agent ---

---

# Chat GPT 5.6 sol


Yes. After looking again at the current `main`, I think your instinct is basically right, but I would sharpen it into a more specific architectural rule:

> **The production sparse solver should use fixed, deliberately narrow sparsity patterns, and every matrix multiplication should write directly into its prescribed narrow output pattern. Do not let matrices structurally densify and then clean them up afterward.**

The important qualification is that we should avoid algorithms where a **wide intermediate is mathematically necessary to obtain an accurate narrow final result**. That is exactly why the new \(P=KS\) purifier is attractive: it turns purification into repeated

$$
P_{n+1}=f\!\left(P_M(P_nP_n)\right)
$$

with **one multiplication and one fixed mask**, instead of the old

$$
T=P_M(KS),\qquad Q=P_M(TK),
$$

where truncating \(T\) corrupts even in-mask elements of \(Q\).

That distinction is more important than “fixed mask vs thresholding” by itself.

## 1. I would choose fixed sparsity patterns for production

Your pasted discussion says that the solver must not allow the bandwidth to double after every multiplication. I agree completely with that design objective. 

The current OpenCL multiplication already has the right fundamental abstraction:

$$
C=P_{M_C}(AB),
$$

where `C_row/C_col` define the allowed result blocks and blocks outside \(M_C\) are never allocated or computed.

That is what I would keep.

I would **not** make the normal algorithm:

$$
A B \rightarrow \text{large halo matrix}
\rightarrow \text{measure blocks}
\rightarrow \text{drop small blocks}
\rightarrow \text{compress CSR}
\rightarrow \text{rebuild plan}.
$$

That destroys most of the advantages you are building: fixed GPU buffers, fixed plans, predictable local-memory requirements, deterministic work, smooth Hessians, and zero structural work inside SCC.

There is also a subtle but important point: if you compute a larger halo of the **final result** and then immediately throw it away, this does not improve the values of the elements inside your original mask at all. For one multiplication,

$$
C_{ij}=\sum_k A_{ik}B_{kj},
$$

computing extra \(C_{ij}\)'s outside the mask does nothing to improve the retained \(C_{ij}\)'s.

A halo is useful only when that wider result is subsequently used as an **intermediate in another multiplication**. And that is exactly the situation we should try to remove algorithmically rather than make the normal case.

So I would use:

$$
\boxed{\text{fixed input masks + fixed output mask + direct projected multiplication}}
$$

as the production algebra.

Dynamic halo/threshold multiplication should be primarily a **diagnostic/calibration tool**, or used for the few unavoidable intermediate products where experiments demonstrate that a narrow intermediate is inaccurate.

This is also consistent with the sparse-purification literature: thresholding/screening is essential to obtain linear scaling, but sophisticated implementations control the error of omitted submatrix products rather than allowing unrestricted fill-in. ([ScienceDirect][1])

---

# 2. The current code still does not impose the neighbor limit you actually want

This is probably the most important implementation-level issue.

At present `build_geometric_mask()` simply includes **every atom inside a radius**. There is no physical/performance contract such as

```text
max_degree <= 96
```

or

```text
max_degree <= 128
```

when the masks are constructed.

`MAX_LEFT_BLOCKS=512` is merely an emergency kernel capacity. It is not a sparsity policy.

Worse, `SparseDftb` constructs the GPU with

```rust
SparseBsr4Config::default()
```

whose `max_left_blocks` is 512.

So even if your real matrix has only, say, 70 neighbors per row, the planned kernel statically reserves

$$
512\times16\times4=32768\ {\rm bytes}
$$

of local memory for `lA` in every work-group. The plan kernel no longer stores the column indices, so its actual local-memory cost is ~32 KiB, not the ~34 KiB quoted in the pasted discussion; 34 KiB applies approximately to the intersection kernel, which also has `lcol[512]`.

This is backwards.

The sequence should be:

$$
\text{choose masks}
\rightarrow
\text{measure max degree}
\rightarrow
\text{choose MAX\_LEFT\_BLOCKS}
\rightarrow
\text{compile OpenCL once}.
$$

Conveniently, your current initialization order already allows this: masks are generated before `SparseBsr4Gpu::new()`.

I would initially use a **hard production ceiling around 128 blocks/row**, with an engineering goal that most matrices are closer to 64–96.

Then the plan kernel uses only:

$$
128\times16\times4=8\ {\rm KiB}
$$

or, for 64,

$$
4\ {\rm KiB}.
$$

That is a huge difference from 32 KiB.

I would actually **not implement four degree-bucket kernels yet**. That is more complexity than necessary.

For your single-large-system use case, compile one kernel at engine initialization with the smallest reasonable static capacity, e.g.

```text
actual max degree = 73
compile MAX_LEFT_BLOCKS = 96 or 128
```

and then run the entire simulation with it.

If an “accurate/reference” setup genuinely needs >128, make that explicit and compile 256. Never silently get 512 because that happened to be a historical default.

---

# 3. The current physical defaults are also still too permissive

`SparseDftbConfig` currently defaults to:

```rust
r_trunc_ang = None
r_k_ang     = None
r_z_ang     = None
r_skin_ang  = 1.0
```

and therefore H/S, K and Z default essentially to the full SK-table radius plus the skin.

The mask construction confirms this: H/S use the table cutoff unless `r_trunc_ang` is supplied; K and Z default to the same full table radius plus skin.

For the production large-system sparse solver I would invert that philosophy.

A large calculation should **not be allowed to start without an explicit sparsity budget**.

I would think in terms of separate degree budgets:

| Matrix              | Role                     | Desired policy                                                      |
| ------------------- | ------------------------ | ------------------------------------------------------------------- |
| \(H,S\)             | physical local operator  | very tight, species-dependent cutoff                                |
| \(P=KS\)            | hot purification iterate | **tightest important computational mask**, target ~64–96, hard ≤128 |
| \(K\)               | recovered density kernel | same structural scale as P, perhaps slightly wider if required      |
| \(Z\approx S^{-1}\) | inverse overlap          | can be somewhat wider, because evaluated once per geometry          |
| \(ZH\)              | auxiliary for P0/W       | modest fixed halo if needed                                         |
| \(W\)               | force result             | only \(M_{HS}\) needed                                              |

Do **not** assume \(K\) is more local than \(H\). The coding agent said something close to that in the pasted discussion, and that is not generally safe.

For an insulator the density matrix decays exponentially, yes, but the decay length is controlled by the electronic gap and basis. It can easily be longer-ranged than the finite-range Hamiltonian. In fact, published SP2 work explicitly reports density matrices becoming several times denser than the original Hamiltonian. ([ScienceDirect][2])

Your own previous numerical observations were already pointing in that direction.

So the goal is not:

> K must fit inside H because it “should decay faster.”

It is:

> K/P must fit inside the smallest degree budget that still gives acceptable energy/force/frequency errors.

That is a measured approximation.

---

# 4. The P purifier is much better matched to this design than legacy K-TC2

Current `main` already contains the experimental \(P=KS\) purifier. This is a very good development.

The hot operation is now simply

$$
Q=P_M(PP)
$$

followed by TC2.

Only **one SpGEMM per purification iteration**, with no \(KS\) intermediate whose support wants to expand. The implementation does exactly that and uses the new generic planned multiplication because P is nonsymmetric.

This is exactly the architecture I would optimize around.

The legacy path still does

$$
T=P_M(KS),
\qquad
Q=P_M(TK),
$$

and `m_t_ks=m_k`, so the intermediate is deliberately truncated. That keeps memory bounded, but it introduces the intermediate-truncation error we already diagnosed.

So I would make the medium-term goal:

$$
\boxed{\text{P-TC2 becomes production; K-TC2 becomes reference/legacy}}
$$

provided the parity tests remain acceptable.

Right now P-TC2 is still experimental/off by default; `SparseDftb::scc()` only uses it when the config/environment selects it.

I would concentrate validation there rather than spend much more time perfecting the two-multiply K-TC2 loop.

TRS4 is interesting for stability, but it requires \(P^2\) and \(P^4\), i.e. two sparse products per iteration again. I would therefore regard P-TC2 as the **speed baseline** and only use TRS if its greater stability actually reduces total work enough to compensate.

---

# 5. The new W path is also the right kind of simplification

Current `main` now supports

$$
W=2(ZH)K
$$

directly rather than

$$
T=KH,\qquad
W=2TK.
$$

That is another excellent change because it eliminates another dangerous truncated intermediate and cuts two products to one.

This is exactly the principle I want applied everywhere:

> Instead of making wide intermediates and inventing clever ways to truncate them, reformulate the algebra so the desired narrow matrix is produced directly whenever possible.

---

# 6. One place where a modest halo still makes sense: Z / Newton–Schulz

The inverse overlap is the awkward piece.

Newton–Schulz uses

$$
T=ZS,
$$

$$
Z' = 2Z-TZ.
$$

So there really is an intermediate.

If both \(T\) and \(Z\) are forced onto the same very narrow mask, some contributions to the final in-mask \(TZ\) are lost.

Here I would allow a **fixed, modest intermediate halo**, not a dynamically growing matrix.

For example conceptually:

$$
M_Z : \text{degree } 96,
$$

$$
M_{ZS} : \text{degree } 128,
$$

$$
Z_{\rm new} : M_Z.
$$

The important point is that 128 is an explicit fixed ceiling. It never becomes the full structural radius \(R_Z+R_S\).

NS is performed once per geometry and warm-started, so it can afford somewhat more work than the P-purification loop which is repeated in every SCC iteration.

This is where I would spend some accuracy budget.

---

# 7. Dynamic magnitude thresholding is useful mainly for choosing the fixed masks

I would not make threshold-induced CSR changes part of every multiplication.

For vibrations that is particularly dangerous: a block appearing/disappearing when its norm crosses a threshold creates numerical non-smoothness.

Instead, use a wider calculation occasionally to answer:

$$
\epsilon_M=
\frac{
\|C_{\rm wide}-P_M C_{\rm wide}\|_F
}{
\|C_{\rm wide}\|_F
}.
$$

Then choose a fixed mask satisfying the accuracy target.

You can make the mask magnitude-aware rather than purely radius-aware. For example, do one wider P/K calculation, compute a block score

$$
s_{ij}=\|P_{ij}\|_F,
$$

and retain the most important symmetric neighbor graph subject to

$$
d_i\le D_{\max}.
$$

This can outperform a purely spherical cutoff around defects/surfaces while preserving a completely static GPU layout.

A very important correction to the coding-agent discussion: it said that “dropped mass” could be calculated “almost for free” at **plan build time**. 

That is false for numerical dropped mass.

At plan-build time you know only structure. You can count omitted paths

$$
(i,k,j),
$$

but you do not know

$$
\|A_{ik}B_{kj}\|
$$

until you have the matrix values.

To get an error estimate you need either a wider numerical reference multiplication, or run-time bounds such as

$$
\|A_{ik}B_{kj}\|_F
\le
\|A_{ik}\|_F\,\|B_{kj}\|_F.
$$

That kind of submatrix-product screening is precisely what error-controlled sparse multiplication methods use. ([ScienceDirect][1])

But I would add it only after the simpler fixed-mask architecture is working.

---

# 8. Precision: no, Kahan is still not in the hot SpGEMM

The current planned kernel has four f32 accumulators:

```c
float s0, s1, s2, s3;
```

corresponding to the four orbital inner-index contributions. This is much better than one long serial FMA chain: dependency length is approximately the number of contributing blocks rather than four times that number. But it is **not compensated summation**.

There is no Kahan correction variable.

And I would **not jump directly to f64 accumulation inside SpGEMM** on a consumer NVIDIA GPU where FP64 throughput is catastrophically worse.

The first precision optimization I would test is actually cheaper than Kahan:

$$
\boxed{\text{increase independent f32 partial accumulators}}
$$

Currently each `s_m` accumulates every k contribution.

Instead use two accumulators per orbital channel, alternating terms:

```text
s00, s01
s10, s11
s20, s21
s30, s31
```

then pairwise combine at the end.

This doubles registers from 4 to 8 but adds essentially **no extra multiply-add work** and halves each serial summation chain.

If your degree is capped at ~64–96, this may already reduce the arithmetic floor enough.

Only after that would I benchmark Kahan.

---

# 9. Where Kahan and f64 actually make sense

The most important distinction is between **matrix arithmetic** and **small reductions / decisions**.

| Operation                           | My choice                                                |
| ----------------------------------- | -------------------------------------------------------- |
| P² / sparse SpGEMM early iterations | 4–8-way f32 FMA                                          |
| P² final purification iterations    | optional f32 Kahan variant                               |
| ZS / ZSZ                            | f32; possibly compensated only when NS stalls near floor |
| K=PZ final recovery                 | compensated f32 is attractive: done only once            |
| W=(ZH)K                             | compensated f32 worth testing: only once per force call  |
| Mulliken per atom                   | plain f32; only ≤4 diagonal terms                        |
| TC2 trace / branch                  | **f64 host**, as now                                     |
| TRS coefficients                    | **f64 host**, as now                                     |
| DIIS small solve                    | **f64 CPU**, keep                                        |
| idempotency residual                | f32 square + tree reduction + **f64 tail**               |
| NS residual                         | same: f32 partials + **f64 tail**                        |
| band energy \(2\mathrm{Tr}(KH_0)\)  | **f64 final reduction**                                  |
| spectral-bound decisions            | f32 matrix arithmetic but conservative/f64 scalar logic  |
| Hessian assembly/eigensolve         | f64 CPU                                                  |

The current trace handling is already exactly what I want: one per-atom f32 contribution is downloaded and summed in host f64.

That is a very cheap “f64 island”.

Do the same philosophy elsewhere.

### Idempotency and NS residuals

Currently the idempotency/Frobenius reductions are recursively reduced all the way down to **one f32 scalar** on the GPU.

These sums are of positive squares, so Kahan is unnecessary.

Instead stop the GPU tree when, say, 32–128 partial values remain, download those and sum them in host f64.

This costs almost nothing because these diagnostics already cause synchronization.

It makes the measured floor trustworthy without putting any FP64 in the big kernels.

### Band energy

`trace_hk_to_dev()` currently reduces the entire extensive band-energy sum to one f32 scalar.

That is a much better candidate for a f64 tail than SpGEMM itself because the energy is an extensive sum with cancellation.

Even better: I question whether you need band energy on **every SCC iteration** at all.

SCC convergence uses charge residuals, not energy.

Compute the accurate energy only when convergence is reached, plus optionally in verbose/debug mode. That simultaneously removes a reduction kernel and a host sync from most SCC iterations.

That is preferable to making the reduction more elaborate.

---

# 10. Kahan should probably be an endgame mode, not a permanent tax

This is the compromise I like most for your hardware.

Most purification iterations are far from the f32 floor. Running Kahan during those iterations achieves nothing.

So use the fast kernel until something like

$$
R_I\lesssim10^{-3}
$$

or until the residual starts flattening.

Then switch P² to a **compensated-f32 kernel** for the last few iterations.

If Kahan lowers

$$
10^{-5}\rightarrow10^{-6}
$$

while being, say, 1.5–2× slower, but is used for only 3 out of 25 iterations, the overall cost is tiny.

This is far more sensible than either:

> everything f32 forever

or

> everything f64 because f32 isn't accurate enough.

For Kahan inside the planned kernel I would retain the four independent orbital accumulators, each with its own compensation. Whether 8 simple accumulators outperform 4 Kahan accumulators is something the RTX 3090 benchmark can decide.

---

# 11. Aggressive sparsity also improves f32 accuracy

This connection is easy to miss.

If an output scalar accumulates \(n\) terms, ordinary sequential f32 error scales roughly with \(nu\) in the pessimistic case.

Reducing each row from 300 contributions to 64 does not only make the kernel faster; it makes the summation problem substantially better conditioned numerically.

Your current multi-accumulator layout plus a hard degree cap around 64–128 may push the residual floor down enough that Kahan becomes unnecessary.

So I would **reduce the degree before spending engineering effort on compensated arithmetic**.

---

# 12. Two current code issues I would fix regardless

`build_hscc_dev()` still has a real fail-silent hole.

The kernel contains

```c
if(nb > MAX_LEFT_BLOCKS) return;
```

but the Rust `build_hscc_dev()` launcher does not call the degree guard before enqueueing.

So if H/S ever exceeded the compile-time capacity, those rows would remain unwritten/stale.

Once we enforce max degree 128 this should never happen, but the guard should still be there.

The second issue is the plan fallback. Both `SparseSystemWorkspace` and `SparseDWWorkspace` still implement:

```rust
Some(plan) => fast_plan_kernel
None       => slower_intersection_kernel
```

and plan construction failures turn into `None`.

For production I would remove that semantics entirely.

The intersection kernel is useful as a reference test and AB benchmark. It should require an explicit diagnostic flag.

---

## What I would tell the coding agent to do next

1. **Make sparsity a hard solver contract.** Add explicit max-degree reporting for H/S, P/K, Z and auxiliary masks. Production target initially `max_degree <= 128`, aim for ~64–96. Fail initialization if it is exceeded rather than silently enlarging the kernel.

2. **Compile `MAX_LEFT_BLOCKS` from the actual production mask budget**, not the historical default 512. One program compile at engine initialization is fine. Do not implement degree buckets yet unless profiling later proves they are necessary.

3. **Make the plan kernel mandatory in production.** Plan build/upload failure is an error. Keep intersection only behind an explicit diagnostic/reference switch.

4. **Promote P-TC2 as the main performance candidate.** It has exactly the algebra we want: one `P_M(P²)` product per iteration and no densifying intermediate. Validate it on the wide-mask reference, then make it the default if energies/forces/frequencies pass.

5. **Keep `W=2(ZH)K` as the production force path.** Do not return to the `KH→KHK` intermediate except for parity testing.

6. **Shrink H/S aggressively and shrink the 1 Å skin.** The skin counts toward structural degree and plan work even where pair values are currently zero. For a Hessian with ±0.02 Å displacements the required frozen-topology skin is much smaller than 1 Å; geometry optimization can rebuild masks at explicit checkpoints instead of carrying a huge skin forever.

7. **Use a modest fixed halo only where algebra genuinely needs an intermediate**, primarily `ZS` in Newton–Schulz. Never permit unrestricted `R_A+R_B` product support. Give the halo its own hard degree cap.

8. **Precision order:** first test 8-way f32 partial accumulation; then add an optional compensated-f32 planned SpGEMM used only in purification endgame/final K/W products. Do not use GPU f64 for full matrix products. Convert scalar decision/reduction tails to host f64 instead.

9. **Make the measurements decisive:** for each max-degree target (64, 96, 128), report wall time, plan terms, local memory/WG, \(R_Z\), \(R_I\), SCC residual, ΔE/atom, max ΔF, and finally selected vibrational frequencies versus the widest reference. That directly answers the real question: **how little connectivity can we get away with before losing the ~5% physics target?**

The main architectural choice is therefore clearer to me now than in the previous review: **fixed sparse graphs win for this GPU solver**. Use threshold/halo calculations to *design and validate* those graphs, not as the routine representation of every intermediate. And the P-purifier is particularly attractive because it lets you enforce that rule without paying the large mathematical error that came from truncating \(KS\) halfway through \(KSK\).

[1]: https://www.sciencedirect.com/science/article/pii/S0021999121002497?utm_source=chatgpt.com "Sparse approximate matrix-matrix multiplication for density matrix purification with error control - ScienceDirect"
[2]: https://www.sciencedirect.com/science/article/abs/pii/S0010465518300365?utm_source=chatgpt.com "A sparse matrix–vector multiplication based algorithm for accurate density matrix computations on systems of millions of atoms - ScienceDirect"



---

# USER

the agent did some tests but I'm not happy about the result :-( it seems we cannot use less than several hundrets of neighbors which is non acceptable it defeates the purpose it make the sparse solver kinda useless it cannot compete with dense solver this way... resp. it can copete but it will not provide clear advantage in systems which are smaller than many throuands of atoms can we do something about it? maybe some smooth daping the tails? use specifically constrained basis? we must do something about it, maybe some smooth_step damping? try to look what basis set we have available which would have effectlively shorter cutoff? (it is not commit to git yet so do not look in code) ----


---

# Chat GPT 5.6 sol

Yes — I think there are still several serious ways forward, and I would **not** conclude from this sweep that sparse DFTB intrinsically needs 200–400 neighbors.

What the sweep establishes is that **blind radial truncation of the current `matsci` density kernel is not compressible below ~200 neighbors without large error**. At 10 Å / degree 212 the error is already only ~0.16 mHa/atom, while 8 Å / degree 108 jumps to ~0.8 mHa/atom and K-TC2 becomes unstable.  But that mixes together three things: the physical locality of the occupied subspace, the particular nonorthogonal atomic basis, and the very crude choice “keep every atom inside a sphere”.

I would attack all three before abandoning the idea.

## The most promising immediate experiment: use `pbc`, not `matsci`

I found a striking difference in the actual public Si parameter files.

Your current `matsci` Si basis uses a harmonic confinement radius \(r_0=4.2\,a_0\) for the Si \(s,p\) functions.  Its Si–Si SKF grid is

$$
\Delta r=0.05\,a_0,\qquad N=400,
$$

so the electronic tables extend to about

$$
20\,a_0 \simeq 10.6\ \text{Å}.
$$

The `pbc` Si parameters are **much more strongly confined**. The SKF documentation says the Si \(3s,3p\) wavefunctions use confinement radius \(3.3\,a_0\), versus 4.2 in matsci.  More dramatically, the Si–Si table has

$$
\Delta r=0.02\,a_0,\qquad N=520,
$$

so its entire electronic range is only

$$
10.4\,a_0\simeq5.5\ \text{Å},
$$

roughly **half the matsci range**.  The Si–H `pbc` file has the same 10.4-\(a_0\) tabulation range.

And this is not some inappropriate parameter set: the authors explicitly say the `pbc` Si parameters are good for **silicon clusters using an \(sp\) basis**.  DFTB.org lists `pbc` specifically as an SCC set for solids and surfaces containing Si/H. ([dftb.org][1])

That is exactly your problem.

So before doing anything elaborate I would repeat the **same R14 degree sweep with `pbc`**.

I would be very interested in whether the P/K block norm decay changes. The H/S matrix certainly becomes drastically sparser; whether P becomes sufficiently shorter-ranged is the key experiment.

### The available Si alternatives look like this

| set        | Si basis / purpose                          | electronic table range | suitability here                                                                  |
| ---------- | ------------------------------------------- | ---------------------: | --------------------------------------------------------------------------------- |
| **matsci** | \(sp\), \(r_0=4.2a_0\)                      |                ~10.6 Å | current baseline; rather diffuse                                                  |
| **pbc**    | \(sp\), wavefunction confinement \(3.3a_0\) |             **~5.5 Å** | **very promising**; explicitly used for Si clusters                               |
| **siband** | requires Si \(d\) orbitals                  |                 ~9.5 Å | bad for our objective: larger 9-orbital Si blocks and no real repulsive potential |
| **PTBP**   | modern minimal-basis solid-state set        |    parameter-dependent | worth testing later; also a good framework for custom reparameterization          |

`siband` is interesting electronically—it was designed to reproduce Si band structure and can give the experimental-like Si gap—but its documentation requires \(d\) orbitals on Si and explicitly says there is **no usable repulsive potential**, so it cannot be used for structural optimization as-is. ([GitHub][2]) It would make your GPU problem worse anyway: 9 orbitals/Si instead of 4.

`pbc` is the obvious first candidate.

---

## But there is another important possibility: perhaps **P is already local enough and K recovery is what kills you**

I would test this before trusting the present energy sweep too much.

Your P-TC2 calculation at 8 Å actually **converges nicely**, unlike K-TC2. The bad number comes from the final energy after recovering

$$
K=PZ
$$

and projecting K back onto the narrow K mask. The agent even found that K recovery inherits the finite \(ZS-I\) error and required a trace repair. 

So there are two separate questions:

$$
\text{Can }P=KS\text{ be represented accurately with 64–128 neighbors?}
$$

and

$$
\text{Can }K=PZ\text{ be represented accurately with 64–128 neighbors?}
$$

The second may be substantially harder because \(S^{-1}\) has an orthogonalization tail. There is good general evidence that the spectrum of the overlap matrix can severely reduce locality of quantities involving \(S^{-1}\). ([Taylor & Francis Online][3])

But we may **not need to store global K at all**.

For example, the band energy can be written directly using P:

$$
E_{\rm band}
=2\operatorname{Tr}(KH)
=2\operatorname{Tr}(P\,Z H).
$$

You already form

$$
A=ZH
$$

for P initialization.

Therefore test

$$
\boxed{E_P=2\operatorname{Tr}(P\,A)}
$$

directly, without ever constructing globally truncated \(K=PZ\).

Likewise Mulliken populations are already simply

$$
q_A=2\operatorname{Tr}P_{AA}.
$$

This is a **very important diagnostic**. Take the 12 Å wide reference, then purify P with 6/8/10 Å masks and compare:

$$
E_P=2\mathrm{Tr}(PZH)
$$

against the reference **before K recovery**.

If P at degree 108 is actually accurate but \(K=PZ\) at degree 108 is not, then the current conclusion “we need 200 neighbors” is misleading. The hot SCC purifier may still run at degree ~64–128; we would only construct the few K/W blocks needed for short-range force contractions afterward.

That would be a major win.

---

# Smooth damping: useful, but not the cure by itself

Your idea of a smooth step is reasonable, but I would be careful about what it can accomplish.

Instead of

$$
P_{ij}\rightarrow
\begin{cases}
P_{ij},&r<r_c\\
0,&r>r_c
\end{cases}
$$

use

$$
P_{ij}\rightarrow w(r_{ij})P_{ij},
$$

with for example

$$
w=1 \qquad r<r_0
$$

and for \(x=(r-r_0)/(r_c-r_0)\),

$$
w(x)=1-\left(6x^5-15x^4+10x^3\right),
$$

then \(w=0\) beyond \(r_c\).

That gives zero first and second derivative at both ends, which is exactly what I would want for Hessians.

But there is a fundamental issue:

> **At the same outer cutoff, smooth damping removes more density-matrix amplitude than a hard cutoff.**

So I expect it may improve numerical stability and geometry smoothness, but **not recover the missing 0.7 Ha at 8 Å**. It might even make the energy bias worse.

There is one situation where it becomes much more interesting: if after applying the localization window you **variationally re-optimize the retained matrix elements**.

That brings me to what I think may be the deeper algorithmic solution.

---

# Instead of truncating the exact projector, optimize the best projector *inside the sparse space*

Right now purification approximately does

$$
P_{\rm exact}\rightarrow P_M P_{\rm exact}
$$

repeatedly.

When the mask is too short, you're asking an impossible question:

> “Can the exact density matrix simply be chopped at 8 Å?”

Apparently not.

But what you really need is:

> “What is the **best variational density matrix constrained to this 8 Å / 64-neighbor space**?”

Those are not the same matrix.

There is a classic approach designed exactly around this idea: **Li–Nunes–Vanderbilt density-matrix minimization**. Its nonorthogonal extension was explicitly constructed so that **\(S\), but not \(S^{-1}\), appears in the functional**, and the variational density matrix remains localized. ([APS Journals][4])

One writes an auxiliary sparse matrix \(L\) and defines

$$
K(L)=3LSL-2LSLSL.
$$

Then instead of repeatedly purifying and truncating an exact solution, you minimize the electronic energy with respect to the **allowed sparse elements of \(L\)**. This is used in linear-scaling methods such as ONETEP; the density-kernel cutoff is a genuine variational constraint rather than an after-the-fact deletion. ([ResearchGate][5])

And there is very relevant history here: early LNV linear-scaling DFT was explicitly tested on **512-atom silicon**, with the density matrix constrained to a finite localization range, and reported rapid convergence of total energy as the localization range increased. ([APS Journals][6])

That makes me think this is not a crazy detour.

For you I would not replace everything. I would try:

$$
\text{P-TC2}
\rightarrow
\text{fast initial localized state}
$$

followed by perhaps 5–15 **variational localized-DM refinement steps** on the same fixed degree-64/96/128 mask.

Then the matrix inside the cutoff can adjust to compensate optimally for the missing tail.

That has a much better chance of making 6–8 Å useful than simply changing hard zero to smooth zero.

---

# Even cheaper than LNV: test whether radial masks are simply stupid

Before implementing any new solver, there is one experiment I consider essential.

Take your good wide reference at \(r=12\) Å. For each atom, calculate

$$
b_{ij}=\|P_{ij}\|_F
$$

for every 4×4 block.

Then construct masks containing the **largest** 32, 48, 64, 96, 128 blocks per row, symmetrized between \(i,j\), regardless of distance.

Then rerun the exact same calculation with those frozen masks.

This gives a crucial answer:

### Case A: top-64 or top-96 works

Excellent. The density matrix is compressible; **spherical radius was simply a terrible sparsifier**.

This is quite plausible in tetrahedral Si. A 10 Å sphere contains hundreds of atoms, but because of bonding geometry many of their 4×4 blocks may be tiny.

Then production masking can use a predictor based on:

$$
\|P_{ij}\|,\quad
\|H_{ij}\|,\quad
\text{bond/path connectivity},
$$

rather than Euclidean radius alone.

For a Hessian, determine that graph once at the central geometry and freeze it.

### Case B: even the optimum top-128 block mask fails

Then the occupied projector really is not sufficiently sparse in the present atomic basis.

At that point I would stop optimizing TC2 kernels. We need a different representation.

This one experiment is vastly more informative than another radius sweep.

---

# A more radical basis change may indeed be the ultimate solution

Silicon is a textbook case where the **occupied subspace has highly localized bond-centered Wannier functions** even though its density matrix in an atomic AO representation can have a rather long orthogonalization tail.

So your intuition about a “specifically constrained basis” is very relevant.

An on-site \(s,p\rightarrow sp^3\) rotation alone will **not** help your present BSR4 representation: a unitary transformation within each 4-orbital atom preserves

$$
\|P_{ij}\|_F,
$$

so the same atom blocks remain nonzero.

But **localized bond/support orbitals optimized over several neighboring atoms** are different. This is the philosophy behind CONQUEST/ONETEP and orbital-minimization methods.

For diamond Si, I would expect a basis of localized occupied bond orbitals to need perhaps a few bond shells rather than a 10–12 Å atomic density-kernel sphere.

That is a larger rewrite, but if top-128 magnitude sparsification fails, I would seriously consider it instead of accepting a 400-neighbor “linear-scaling” matrix.

---

# Another inexpensive knob: finite electronic temperature

There is also a mathematically clean way to shorten the density-matrix decay: finite electronic temperature.

The density matrix is exponentially localized for insulators and also at finite temperature; estimates show its range decreases with increasing gap and temperature. ([The Hebrew University of Jerusalem][7])

So if the Si sphere has small-gap surface or frontier states, a modest Fermi smearing could substantially improve locality.

For example, test dense/sufficiently-wide references at

$$
k_BT=0,\ 0.02,\ 0.05,\ 0.10\ {\rm eV}
$$

and measure the radial/block decay of P.

But this changes the formal problem: you should then minimize the Mermin free energy and include the electronic entropy consistently in energies/forces. Zero-temperature TC2 is no longer the correct occupation map.

So I rank this below the `pbc` and top-k-mask tests.

---

# A purpose-built “sparse DFTB” parameterization is actually plausible

If `pbc` demonstrates that stronger atomic confinement dramatically improves sparse locality while keeping acceptable Si/H physics, you could go one step further.

DFTB parameters are generated from confined atomic orbitals; changing the confinement is a legitimate parameterization degree of freedom. DFTB.org explicitly describes the usual basis as minimal and the two-center electronic integrals as being generated from confined atomic calculations. ([dftb.org][8]) Modern parameterization work explicitly optimizes wavefunction confinement parameters; the recent DSKO work even treats Si confinement as an optimization variable. ([American Chemical Society Publications][9])

I could imagine fitting a **sparsity-aware Si/H parameter set** with an objective like

$$
\chi^2 =
\chi^2_{\rm energy}
+\chi^2_{\rm forces}
+\chi^2_{\rm bands}
+\lambda_{\rm loc}\,\chi^2_{\rm locality}.
$$

The locality term could penalize long-range H/S and perhaps poor conditioning of S.

Keep the minimal \(sp\) Si basis—do **not** add the \(d\) shell used by siband/modern high-band-quality Si parameterizations, because that doubles the computational problem.

Fit against diamond Si, strained diamond, Si clusters, Si–H bonds and H-passivated surfaces.

`pbc` is already a proof that a much more confined \(sp\) Si DFTB parameterization can exist and be useful for clusters.

---

## What I would do next

I would **not implement Kahan or more GPU micro-optimization now**. The attached results show truncation/model representation dominates f32 arithmetic by orders of magnitude. 

I would run exactly these experiments, in this order:

1. **Repeat the R14 sweep using `pbc` Si/H.** Measure H/S degree, P block decay, degree required for ΔE/atom and forces. This is very cheap and could completely change the picture.
2. **On current matsci, evaluate P directly before K recovery:** compare \(2\mathrm{Tr}(PZH)\) for degree 52/108/212/386. Find out whether the long range is really P or mostly \(PZ\rightarrow K\).
3. **From the wide reference, construct optimal magnitude masks:** top-32/48/64/96/128 \(P_{ij}\) blocks per row. This tells us whether a 64–128-neighbor representation exists at all.
4. Only if top-k works poorly, test a **localized variational LNV refinement** on a fixed ~64–128-neighbor mask. Smooth radial damping can be included there as a localization penalty/window.
5. If even variational sparse DM fails, then I would move to **localized occupied/support orbitals**, not accept a 400-neighbor density matrix.

So I would not give up yet. In fact, the `pbc` discovery makes me considerably more optimistic: **5.5 Å versus 10.6 Å electronic range and \(r_\mathrm{wf}=3.3\) versus 4.2 bohr is a very large change in precisely the quantity we care about.** Whether it translates into P-locality needs one experiment, but it is much more promising than trying to squeeze another factor of two out of the current matsci projector with a smoothstep.

[1]: https://www.dftb.org/parameters/download.html "Download — dftb.org"
[2]: https://github.com/dftbparams/siband?utm_source=chatgpt.com "GitHub - dftbparams/siband: DFTB parameter set providing electronic parameters for accurate silicon and silicon dioxide band structures · GitHub"
[3]: https://www.tandfonline.com/doi/full/10.1080/00268976.2020.1765034?utm_source=chatgpt.com "Full article: The spectrum of the atomic orbital overlap matrix and the locality of the virtual electronic density matrix"
[4]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.50.17611?utm_source=chatgpt.com "Generalization of the density-matrix method to a nonorthogonal basis | Phys. Rev. B"
[5]: https://www.researchgate.net/publication/228927861_Density_kernel_optimization_in_the_ONETEP_code?utm_source=chatgpt.com "(PDF) Density kernel optimization in the ONETEP code"
[6]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.53.7147?utm_source=chatgpt.com "Linear-scaling density-functional-theory technique: The density-matrix approach | Phys. Rev. B"
[7]: https://cris.huji.ac.il/en/publications/sparsity-of-the-density-matrix-in-kohn-sham-density-functional-th/?utm_source=chatgpt.com "Sparsity of the density matrix in kohn-sham density functional theory and an assessment of linear system-size scaling methods - The Hebrew University of Jerusalem"
[8]: https://dftb.org/parameters/introduction.html?utm_source=chatgpt.com "Introduction — dftb.org"
[9]: https://pubs.acs.org/doi/10.1021/acs.jctc.6c00121?utm_source=chatgpt.com "DSKO: Dancing through DFTB Parametrization | Journal of Chemical Theory and Computation | ACS Publications"















---

# USER


OK, there are test with pbc basis instead of matsci. It is big success, but I think the agent is unnecessarily peisimisty about possibility to reduce the cutoff futher? What you think. There are certainly some ways wehich you pertially explored also in previous sugestions, can you explain it to hem more clearly?



OK, there are test with pbc basis instead of matsci. It is big success, but I think the agent is unnecessarily peisimisty about possibility to reduce the cutoff futher? What you think. There are certainly some ways wehich you pertially explored also in previous sugestions, can you explain it to hem more clearly?

---

# SWE-2

Let me read the full GPT-5.6 suggestion first:

Let me check whether the `pbc` parameter set is available locally:

`pbc-0-3` exists and Si–Si indeed tabulates only to 10.4 a₀ ≈ 5.5 Å. Check the sphere's composition and required SKFs:

All SKFs present, sphere is Si582H282 (H-passivated). First, write the GPT-5.6 plan into the manifest as a new work-order section:

Now write the pbc variant sweep (L1 — the user's priority):

**Huge result.** pbc at full table (5.5 Å) gives deg_k=95 — converges cleanly in **0.54 s** (matsci needed deg 386, 7.1 s — 13×). The confined basis genuinely shortens the DM range. Below 5.5 Å it still fails, but we no longer need it to. Let me find the actual floor — sweep between 5.0 and 5.5:

Odd — deg_k=95 at r_k=0 (full table, ~5.5 Å) vs deg 36 at r_k=5.4? That gap is too big for a 0.1 Å difference. Check how `sparse_new` interprets r_k=0:

r_k=0 → full SK radius, yet deg jumped 36→95 between r_k=5.4 and "full" — inconsistent with a 5.5 Å table end. Check what `r_full_ang` actually is for pbc:

`r_full` uses `t.cutoff()` — probably the *repulsive* spline cutoff, not the 5.5 Å electronic end. Check:

`r_full` = table end 5.5 Å + 1 Å skin = 6.5 Å. The working config was at 6.5, failures at ≤5.4 — the boundary is between. Also deg 95 at 6.5 Å is higher than pure-density estimate (~60) — H passivation adds surface rows. Bracket the transition:

Clear picture now. Let me update the manifest L1 entry with the measured result:

Done. Written into manifest §15.10 (the full GPT-5.6 rescue plan, L1–L8) and the report. **And I ran L1 — the result is genuinely good news:**

## pbc result (R14 sphere, 864 atoms)

| config | deg_k | deg_hs | outcome |
|--------|-------|--------|---------|
| pbc full mask (~6.5 Å) | **95** | 54 | **converges clean — 19 iters, 537 ms** |
| matsci reference (12 Å) | 386 | 168 | 7073 ms |
| pbc truncated (3.0–6.2 Å) | 6–52 | — | all fail (TC2 diverges, `R_I=inf`) |

**The ~200-neighbor floor was a matsci-basis property, not a sparse-solver limit.** The pbc confinement makes the density matrix genuinely short-ranged — deg 95 is already under the 128 budget, and it's ~13× faster than the matsci reference. The catch: it's all-or-nothing — you can't cut below pbc's own 5.5 Å table (Z on a narrower mask breaks the TC2 spectral bounds → divergence, not graceful degradation).

## Caveats before celebrating

1. **pbc physics unvalidated** — need a Fortran DFTB+ parity run (E/F) on R14 with pbc-0-3 to confirm the parameterization is good for this system.
2. **P-TC2 fails on pbc even at full mask** (r_I floor ~1.5e-2) — K-TC2 is the working purifier there; worth investigating.
3. E=−845.1 Ha can't be compared to matsci's −882.5 — different parameterizations.

## Next in the manifest order

L1 done → **L3** (P-locality diagnostic: is `2Tr(P·ZH)` accurate at low degree even when K=PZ isn't?) → **L4** (top-k magnitude masks — the decisive "does a 64-neighbor representation exist at all" test) → L5 (LNV variational refinement) → L6 (localized orbitals, last resort).




---

# Chat GPT 5.6 sol

Yes — I agree with you. The agent is drawing a stronger conclusion than the data justify.

The pbc result is actually much more encouraging than “95 neighbors is the floor.” What it demonstrates is that **locality is highly basis-dependent**: changing only the Si/H parameterization moved you from roughly 386 neighbors and 7 s to 95 neighbors and 0.54 s. That alone tells us that the previous “physics requires hundreds of neighbors” conclusion was wrong. The remaining question is whether we can push the **hot-loop** degree from ~95 toward perhaps 48–64. I think there are several plausible routes.

The key conceptual correction for the agent is this:

> **Failure of a calculation in which \(K\), \(Z\), and all intermediates are simultaneously chopped by the same spherical radius does not prove that the density matrix requires that radius. It only proves that this particular truncated algebra fails.**

In particular, `pbc full ≈6.5 Å` is not even the physical pbc electronic cutoff. The SKF electronic range is ~5.5 Å; the extra ~1 Å comes from your structural skin/default mask convention. The inverse overlap \(Z=S^{-1}\) and density kernel can legitimately extend beyond the H/S range. So the observation “r=6.2 fails, r=6.5 succeeds” may simply mean **\(Z\) or one intermediate requires the extra shell**, not that the repeatedly purified density representation must contain all ~95 neighbors.

There is another reason I would audit that transition carefully: diamond Si has discrete neighbor shells at approximately 2.35, 3.84, 4.50, 5.43, 5.92, then 6.65 Å. Moving a bulk radial cutoff from 6.2 to 6.5 Å does not cross a new bulk-Si shell. Therefore a reported maximum degree jump from ~52 to ~95 in that interval is probably dominated by surface H atoms, skin/mask semantics, or which matrix's degree is being quoted. It should not immediately be interpreted as a fundamental localization length.

## The first experiment should separate \(P/K\) from \(Z\)

I would do this before L3/L4 as currently ordered.

Keep

$$
R_Z \approx 6.5\text{–}7.0\ {\rm Å}
$$

and the \(ZS/ZH\) intermediate halo at least that wide, because Z is constructed **once per geometry**.

Then independently sweep only the matrix that lives in the repeated purification loop:

$$
R_P\ \text{or}\ R_K
=4.5,\ 5.0,\ 5.5,\ 6.0,\ldots
$$

The performance objective is not:

$$
\text{every matrix must have degree}<64.
$$

It is:

$$
\boxed{\text{the matrix multiplied 10–30 times per SCC iteration should have degree}\sim64.}
$$

If Z has degree 95 but is corrected once per geometry, that is cheap. If the final force construction briefly uses a 95-neighbor auxiliary matrix once after SCC convergence, that is also cheap.

This distinction is essential.

The same applies to \(K=PZ\). If P can be kept at degree 48–64, I would **not necessarily construct a full degree-95 K in every SCC iteration**.

Mulliken charge is already directly available from P:

$$
q_A=2\,\mathrm{Tr}\,P_{AA}.
$$

And the band energy can be written

$$
E_{\rm band}
=2\operatorname{Tr}(KH)
=2\operatorname{Tr}(P\,Z H).
$$

You already have \(A=ZH\). So test

$$
\boxed{E_{\rm band}=2\operatorname{Tr}(PA)}
$$

directly from the narrow P, **before recovering K**.

This is perhaps the single most important diagnostic now.

If degree-52 or degree-64 P gives a good \(2\mathrm{Tr}(PZH)\), but recovering \(K=PZ\) and truncating it makes the energy bad, then the long tail belongs mainly to the inverse-overlap transformation, not to the physically important occupied projector.

That would radically change the architecture: run SCC on narrow P; recover only those K blocks actually required for forces after convergence.

The literature on nonorthogonal bases gives a strong reason to suspect exactly this effect: inverse-overlap factors can substantially degrade locality. ([Taylor & Francis Online][1])

## Then do the “oracle top-N” experiment

I think the agent's L4 is actually more decisive than another radius sweep, but I would formulate it more strongly.

Take the converged good pbc reference and calculate for every atom pair

$$
b_{ij}=\|P_{ij}\|_F
$$

and separately

$$
k_{ij}=\|K_{ij}\|_F.
$$

Then create **oracle masks** retaining the strongest 32, 48, 64, 80 blocks per row, symmetrize the graph, freeze it, and rerun SCC self-consistently.

Do not merely truncate the reference matrix and measure its energy.

This answers:

> Does a 64-neighbor representation of the actual electronic state exist at all?

A radial mask is a very crude approximation. In a tetrahedral covalent solid, a farther atom connected through an important bonding pathway may matter more than a closer atom in an unimportant direction. There is no reason that “nearest 64 atoms” should equal “64 most important matrix blocks.”

Sparse purification literature does exactly this sort of magnitude/error-aware screening rather than relying only on geometric radius; there are schemes that bound errors of discarded block products explicitly. ([PubMed][2])

If top-64 works while radius-64 fails, we have essentially solved the conceptual problem. Then the task becomes predicting a good frozen graph without first computing the wide reference—using previous geometry, H/S connectivity, previous P, etc.

For Hessians this is especially convenient: build the graph once at the central geometry and freeze it for all ±h columns.

## Smooth damping is worth testing — but not as the main cure

Your smooth-step idea can help, particularly with stability.

A hard operation

$$
K_{ij}\rightarrow0
$$

at a radius boundary creates a discontinuous approximation and can badly disturb eigenvalues/idempotency. A compact taper

$$
K_{ij}\rightarrow w(r_{ij})K_{ij}
$$

could behave much better.

I would use something smoother than the usual cubic smoothstep because you care about Hessians. For example a \(C^2\) compact Wendland taper,

$$
w(x)=
(1-x)^4(1+4x),
\qquad
0\le x<1,
$$

and zero outside. Wendland functions can be chosen compactly supported and positive definite. ([Springer][3])

There is a nice mathematical bonus if this is applied to the **symmetric PSD K** rather than nonsymmetric P: if the taper matrix \(W_{ij}=w(r_{ij})\) is positive semidefinite, then

$$
K_{\rm tapered}=W\circ K
$$

remains positive semidefinite by the Schur-product theorem.

It will still violate \(KSK=K\), so purification/refinement must follow it. But it is considerably more principled than an arbitrary hard chop.

However, I would not expect tapering alone to magically make a physically important tail disappear. It can turn a catastrophic limit cycle into graceful degradation; it cannot recover information that was removed.

The more powerful version of your intuition is:

> **softly localize, then allow the remaining matrix elements to re-optimize.**

That leads directly to LNV.

## This is where LNV/variational refinement becomes genuinely interesting

The present purification approach effectively asks:

$$
\text{take an unconstrained projector}
\rightarrow
\text{throw away everything outside }M.
$$

But perhaps the best projector **inside \(M\)** is quite different.

For a fixed 48- or 64-neighbor graph, minimize the electronic energy with respect to the allowed density-kernel elements. The retained short-range elements can then rearrange to compensate for the missing tail.

This is exactly the motivation of the Li–Nunes–Vanderbilt family of linear-scaling methods. Importantly for us, Nunes and Vanderbilt explicitly generalized it to **nonorthogonal bases so that \(S\), rather than \(S^{-1}\), enters the functional**, and noted that the resulting density matrix remains localized. ([APS Journals][4])

And this is not hypothetical for silicon: variational finite-range density-matrix approaches were tested on crystalline Si systems up to 512 atoms and showed systematic convergence as the imposed density-matrix range was increased. ([APS Journals][5])

I therefore like this hybrid:

$$
\text{fast P-TC2}
\rightarrow
\text{reasonable localized initial P}
\rightarrow
5\text{–}10\text{ constrained variational refinement steps}.
$$

That is much less radical than replacing the entire implementation with an LNV solver.

## And pbc itself tells us we can probably push the basis further

The biggest lesson from this experiment should not be “pbc gives degree 95.”

It is:

$$
\boxed{\text{basis confinement changed the required degree by a factor of 4.}}
$$

That is enormous.

So a purpose-built sparse Si/H parameterization is now a credible long-term option. You could make the atomic \(sp\) orbitals slightly more confined than pbc, refit the repulsive potential, and benchmark against the same target physics.

I would not do that yet, because top-N masking and P-vs-K locality may already get us to 64. But pbc demonstrates that this lever is physically real rather than speculative.

---

Here is the instruction I would give the coding agent; I think this frames the next work more correctly.

Do **not** interpret the pbc result `deg≈95 works, smaller radial masks fail` as evidence that ~95 neighbors is a fundamental lower bound.

The pbc result actually disproves the previous conclusion that Si intrinsically requires ~200–400 neighbors: changing only basis localization reduced the working degree from ~386 to ~95 and SCC time by ~13×. Therefore locality is strongly representation-dependent, and we should now determine which part of the remaining degree is genuinely necessary.

The next experiments must separate the different matrices instead of shrinking `r_k=r_z=...` together.

1. **Separate hot-loop P/K locality from Z locality.**

   Keep `M_Z` and `M_TZS` at the successful full-pbc support (~6.5–7 Å initially). Z is computed only once per geometry, so its degree is not the main performance target.

   Independently sweep only `M_P` (and, for comparison, legacy `M_K`) to much smaller supports/degrees.

   The objective is not “every matrix ≤64 neighbors”. The objective is “the matrix multiplied repeatedly during purification ≤64 neighbors”. A degree-95 Z or one-off force auxiliary is acceptable.

2. **Measure P locality before recovering K.**

   For P-TC2, evaluate Mulliken charges directly from `P_AA`.

   Evaluate the band energy directly as

   $$
   E_\mathrm{band}=2\,\mathrm{Tr}[P(ZH)]
   $$

   using the existing `ZH` matrix.

   Compare this for narrow P masks against the wide pbc reference **before computing `K=PZ`**.

   This is critical: if narrow P is accurate but recovered/truncated K is not, then the apparent long-range requirement comes primarily from `S^{-1}`/orthogonalization tails. In that case SCC should stay in P representation and K should only be constructed locally after convergence for force contractions.

3. **Audit the strange 6.2→6.5 Å degree jump.**

   For each mask print:

   * average, p95, and maximum row degree;
   * species and atom index of maximum-degree row;
   * number of Si and H neighbors separately;
   * histogram of neighbor distances;
   * separate max degrees of `M_HS`, `M_P/K`, `M_Z`, `M_TZS`.

   Diamond-Si has no new bulk Si neighbor shell between roughly 5.92 and 6.65 Å, so a jump from ~52 to ~95 over 6.2→6.5 Å should be understood before treating it as physical.

4. **Run an oracle magnitude-mask test.**

   From the successful wide pbc calculation compute block norms

   $$
   b_{ij}=\|P_{ij}\|_F
   $$

   and optionally `||K_ij||_F`.

   Construct symmetric frozen masks retaining approximately the strongest 32, 48, 64, 80 blocks per row, independent of distance, and rerun the full SCC calculation on those masks.

   This is the decisive test of whether a degree-64 representation exists. A radial cutoff is only one very crude sparsifier.

   If top-64 works while radial degree-64 fails, then move to magnitude/error-aware graph selection rather than trying more radii.

5. **Test smooth compact tapering only as a stability/localization aid.**

   Do not expect a generic smoothstep to recover missing physics. Test a C² compact taper on K, preferably a positive-definite compact radial function such as a Wendland taper, then repurify/refine.

   Measure whether this converts abrupt divergence/limit cycles into a smooth accuracy-versus-range curve.

6. **If fixed top-64 remains inaccurate, test constrained variational refinement before abandoning the target.**

   A hard truncated purified projector is not necessarily the minimum-energy projector inside that sparsity space.

   Starting from P-TC2, perform a small number of LNV-like/local density-matrix minimization steps with the sparsity graph fixed. This allows retained near-field blocks to relax and compensate optimally for the omitted tail.

Only after these experiments should we decide that degree 64 is physically impossible.

Do **not** spend time on Kahan/ACC8 or further SpGEMM micro-optimization during this phase. The dominant question is representation/truncation, not f32 rounding.

The experiment I would prioritize above everything else is **wide Z + narrow P + direct \(2\mathrm{Tr}(PZH)\)**. It can tell us very quickly whether the remaining 95-neighbor requirement is real occupied-state nonlocality or mostly an artifact of forcing \(S^{-1}\) and recovered K into the same short radial mask.

[1]: https://www.tandfonline.com/doi/full/10.1080/00268976.2020.1765034?utm_source=chatgpt.com "Full article: The spectrum of the atomic orbital overlap matrix and the locality of the virtual electronic density matrix"
[2]: https://pubmed.ncbi.nlm.nih.gov/16170798/?utm_source=chatgpt.com "Systematic sparse matrix error control for linear scaling electronic structure calculations - PubMed"
[3]: https://link.springer.com/article/10.1007/s11071-025-11407-6?utm_source=chatgpt.com "Barrier Lyapunov Functions and Model Reference Adaptive Control with Uncertainty in Native Spaces | Nonlinear Dynamics | Springer Nature Link"
[4]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.50.17611?utm_source=chatgpt.com "Generalization of the density-matrix method to a nonorthogonal basis | Phys. Rev. B"
[5]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.51.10157?utm_source=chatgpt.com "Self-consistent first-principles technique with linear scaling | Phys. Rev. B"





---

# SWE-2

Discussion only — here are the precise answers:

## 1) The two blocking reads

Per TC2 iteration the host does:

| read | what's read | why it exists |
|---|---|---|
| `trace_ks_f64` | n_atom f32 partials → host f64 sum = Tr(KS) | **the branch decision** `Tr > nocc ? squaring : complement` is made on the host every iteration — this is a hard data dependency, not a diagnostic |
| `idempotency_to_f64` | ≤128 partials → host f64 → R_I = ‖KSK−K‖/‖K‖ | convergence/floor check (could run every k-th iter — `check_every` already exists but production passes 1) |

Each `read_buffer` on an in-order queue **drains everything enqueued so far** — including the just-launched KSK product. So the reads don't just cost bandwidth; they prevent any pipelining and the host sits idle ~0.3–0.7 ms/iter waiting.

**Can it be eliminated?** Yes, but it needs the branch decision on the device:

- A tiny kernel after `tc2.ks` computes Tr from partials, compares to nocc, writes `branch` + `guard_alpha` (= nocc/Tr when the guard condition holds, else 1.0 — folded into the existing `tc2_dev`/`scale_dev` so the rescale is unconditional-cheap).
- Per-iter `{Tr, R_I, branch, dev_rel}` appended to a device **log buffer**.
- Host runs M iterations blind, then one read of the log → decide continue / stop / restore-best. R_I can still be computed on-device every iter (cheap kernel) and only *read* at check points.

That's the dense side's W4 "chunked convergence" pattern transplanted to TC2. Cost: moderate refactor (decision kernel + log buffer + host checks every M iters instead of every iter). Payoff: zero pipeline drains inside the loop — products enqueue back-to-back; and the snapshot/best-K logic can trigger from the device log.

## 2) Why `ksk` is ~5× `ks` — yes, sparsity of the *inner* dimension

For `C_ij = Σ_k A_ik B_kj`, work per output block = number of k where **both** A_ik and B_kj exist:

- `T = K·S`: inner sum over k needs `S_kj ≠ 0` → only **deg_S ≈ 56** of K's ~175 neighbors contribute.
- `Q = T·K`: inner sum needs `K_kj ≠ 0` → **deg_K ≈ 175–330** contribute.

So KSK does ~3–6× more 4×4-block multiply-adds per output block — matching the measured 4.7×/8.8×. Your intuition is right that the *output* mask is identical (M_K); the cost asymmetry is entirely in the inner overlap `|N_left(i) ∩ N_right(j)|`. K's wide neighborhood is what's expensive — S's narrowness is what makes the KS product cheap. Consequence: the K-degree reduction directly attacks the dominant kernel.

## 3) Plateau waste — the honest detector

The old "N non-improving checks" detector was removed because it fired during *normal descent* (non-improving iters are routine mid-transient). The safe version that can't false-fire:

> Track `iter_of_best` (last time best_r_i was updated). During true descent, best updates every few iters. At the floor, it stops updating entirely. **Stop when `iter − iter_of_best > W` (~10–15) and `best_r_i < 1e-2` and trace valid → restore best K, return `NumericalFloor`.**

That distinguishes "descending slowly" (best still improving) from "at the floor" (best frozen for 10+ iters) — and it can't trigger before the first snapshot exists. From the R10 traces: floor reached ~iter 44, current code detects at 53 — the windowed detector would stop at ~55→ or actually detect no-improvement around 44+12≈56… the gain is bigger on the 80-iter exhaustions (several runs burned all 80). Expected saving ~30–40% of TC2 device time.

## 4) Fusing the reads / device-side reductions

Right — the GPU partial-sum kernels already exist (`trace_ks_f64` does device partials → host tail; `idempotency_to_f64` same). What's missing is doing the *decision* on-device so the host doesn't need the value every iter:

- **Short term (cheap):** run `check_every=4` for R_I (skip 3/4 of residual reads) — but trace read still needed every iter for the branch → saves ~half the drains. Also fuse both reductions into one kernel+one buffer+one read (they're independent, both computed post-KSK... actually trace is pre-KSK, residual is post — can't fuse into one read without moving the trace measurement to use the *previous* T or deferring the branch by one iteration — changes numerics).
- **Right fix (the architectural one, same as Q1):** device-side decision kernel + log buffer + host checks every M iters. One read per M iters total.

The two questions are really the same issue: as long as a *host* picks the branch each iteration, one drain per iteration is unavoidable. Moving the branch to device eliminates it.

**Proposed order:** (a) windowed-best plateau detector — few lines, ~40% of TC2 time; (b) `check_every` + device branch+guard kernel + log — kills all mid-loop syncs; (c) then the M_TKS halo experiment for accuracy. Shall I start with (a)?

---

# USER

OK, write these notes clearly and comprehensively (even more than here) into the report/labbook so we can discuss them in more details with the other LLM. say acatly at which line-range you added it. 

---

# SWE-2

Added to the report at **lines 2033–2082** (`## 2026-09-14 (cont. 3) — Sync architecture & per-kernel cost analysis`).

The section expands each of the four answers with exact numbers, code references, and design options:

- **Q1** — the two reads (`trace_ks_f64` for the branch decision, `idempotency_to_f64` for convergence), why each is a *pipeline drain* not a bandwidth cost, and the full fix: device-side TC2 driver (branch + guard_alpha + log buffer on GPU, host supervises in chunks — the dense side's W4 pattern).
- **Q2** — the ksk/ks asymmetry: identical output mask, but inner overlap `|N_left(i) ∩ N_right(j)|` is ~deg_S≈56 vs ~deg_K≈175–330 → measured 4.7–8.8× matches deg_K/deg_S plus locality. K's *stored* degree is the price.
- **Q3** — the windowed-best detector design (`iter − iter_of_best > W` with trace-valid snapshot), why it's safe where the removed "N non-improving" check wasn't.
- **Q4** — three levels: `check_every=4` now; one-drain-per-iter (numerics change, needs A/B); zero-drain device driver.

Plus a closing paragraph tying it to the §15.12 ordering for the reviewing LLM. Ready to continue with the plateau detector whenever you are.



---

# USER

OK, I just updated it in GIT
https://github.com/ProkopHapala/dftbplus/commit/df4361d929d871db9e458986e3fa9a6ad0a056cc

we fnally tried to compute hessian for large system, but I'm not happy with the resutls at all. It is very slow (for 330 atoms take >1 hour? what the hell that is slower than on CPU with full diagonalization. 

I think we are doing smething very stupid and wastefull. We should check if everything is computed on GPU (assembly of H and S as well) and if we use proper hot-start for density matrix when we do the small displacements. It seems we do not. Maybe there are other overheaps, plese analyze this thoroughly!

there is my preliminari doscussion with my coding agent suggesting where can be problems

# USER

thanks can you plot these also for the bigger crystals so we see the realistic system for which it make sense to use such sparse solver. Anslo in the @sparse_vibrations.md tutorial mention the performance - how long this calculations took for these test crystals

---

# SWE-2 devin

Two tasks: (a) run + plot a Hessian on a **big** crystal (R10-class, where sparse actually matters), (b) add measured wall times to the tutorial. First find the R10/R14 geometries and estimate the per-eval cost:

R10 is a pure-Si sphere (330 atoms, pbc-0-3) — a genuinely sparse showcase (deg175 at r_k=12 vs ~complete at deg330). Warm evals at deg175 should be ~0.2s → ~2000 evals ≈ 10 min. Writing the script and running it:

The R10 run is going (deg_k max=330, avg~175, nnz_k=57736 — the genuinely sparse setting). While it runs, update the tutorial with the measured timings:

Relax is progressing (FIRE 16/200, max|F| 3.6e-2→1.25e-2, surface reconstructing as expected for pure Si). Each warm SCC solve is ~17 mix iters — I'll keep polling:

The `timeout 60` I left in killed the run at ~FIRE 16 — my mistake. Restarting with a proper 2-hour limit:

Relax converging (FIRE 71, max|F|=1.06e-3 approaching 1e-3). Continuing to poll:

Relax done: 75 steps, 287s, max|F|=7.5e-4. Hessian now running — col 30/990, ~2s per eval → ~60 min. Polling periodically:

---

# USER

wait! what the hell! this takes so fucking LONG!!!! it seems slower than doing the same thing on CPU. How is that fuckinjg possible?

| si_sphere_R10 | 330 | deg~175/330 | 1980 | ~1–2 s | ~1 min | ~30–60 min | ~1 h |

what the fucking hell? 60 minutes? I can run it on CPU faster !!!!

---

# SWE-2 devin

You're right to be annoyed — let me look at where it actually is and why it's this slow, because the breakdown is revealing:

Now I see the real problem — and you're right, it's not just "big system":

**Root cause of the 4.5s/eval:** at deg175 the TC2 purifier **never converges** — it churns all ~55 iterations against the ~5.9e-4 mask floor *every single DIIS mix iteration*, and the floor noise makes DIIS need ~15–17 mix iters instead of ~5. So each displaced force eval ≈ 17 × 55 SpGEMMs ≈ 3–4.5s. The narrow mask that should be *faster per product* is *slower per eval* because the projected map has no stable fixed point — exactly the mechanism the frozen-operands diagnostic found last session.

And your CPU point is fair: at N=330 a dense CPU path (one eigendecomposition ~50ms per mix iter) is genuinely competitive — sparse+purification only wins asymptotically (R14+, ~1000 atoms) where O(N³) dies. 330 atoms is borderline.

The paradoxical fix: **deg330 warm evals were measured at 0.4s** — the purify actually converges (r_I→3e-5) instead of floor-churning, so fewer purify iters AND ~3 mix iters. Wider mask = more expensive per product but *faster per eval* here. Kill the deg175 run and restart at deg330 — faster AND more accurate:

---

# USER

wait this is absurd! I will not wait hors. I will not wait even 5 minutes. The whole point of this excercise wat som make these culaculations within seconds or minues. Identify what are the key bottleneks. Does it really spend all time in the kernels, or do we spend mosti time in some other bottlenecks? Do we need to recalculate purification? SCC? The hamiltonaina and density matrix around inimum should vary minimally. We should be able to compute these Hessain even without SCC with fixed DM (try it first), and when we do SCC clauclation to update DM it must converge extremely quicly as the hamiltonian is almost the same as the one in minimum. I think you are doing something conceptually stupid not exploiting these assumtpions.

Check if and then write clear report just about this, I will give it to analyse to my other clever LLM

---

# SWE-2 devin

You're right — let me actually find where the time goes instead of accepting ~60 min. First, check the deg330 run's pace (it's still running) and look at what each displaced eval really does:

---

# USER

I think I kiled the calculation it does not make any sense to wait, this is useless. Measure just few derivatives evaluations (like FD Hessian steps, perhaps just one for specific atom/direction vectro)

you said CPU bound parts like H/S building? We do this on CPU? Why we should not. We should do these O(n^2) things on GPU, we already implemented these on GPU in the dense multi solver


---

# SWE-2 devin



Let me find what set_coords/scc/forces actually do per displacement — where the CPU work hides:

set_coords is all CPU: BSR H0/S assembly, dense f64 γ matrix O(N²) per displacement, repulsive pairs, uploads — plus z_valid=false forces an NS re-run each eval. Let me check what script functions can drive a single-displacement micro-benchmark, and whether purify warm-starts K:

Now let me check whether the purifier warm-starts K or rebuilds it per call — that determines whether displaced-geometry SCCs should be cheap:

Found it — purify_hscc_trs calls compute_p0_from_hscc every mix iteration: it rebuilds P0 = (emax·I − ZH)/Δ from scratch and runs a full ~50-iter purification. No warm-start of K/P across ±h displacements or across mix iters. That's the conceptual waste you suspected — for a 0.02 Å displacement the previous converged K is essentially the answer already.

Check if the deg330 run is still alive, then measure the actual phase breakdown:

The deg330 process is stopped (T state) — cleaning it up, then running a per-phase micro-benchmark:

bench_eval_r10.rhai
+39
          "  forces=" + ftos(t3 - t2) + " s");
    d = -d;
}
print("=== done ===");
 

This is the smoking gun — full breakdown measured:



eval:  set_coords=4.7ms   scc=3.45s   forces=6.5ms     ← scc = 99%
prof:  tc2.tr=87%  tc2.res=9%  tc2.knorm=1.3%  (ksk+ks kernels ~0.5%)
set_coords (H0/S assembly + dense γ O(N²) + repulsive + upload) is only 5ms — CPU side is NOT the problem at 330 atoms. The scc call is 3.45s. Inside it: 4209 TC2 iterations across 77 purify calls (~55 iters each — running to tc2_max/plateau every time), ~15-16 DIIS mix iters per eval. The tc2.tr stage at 3.5ms/call is the per-iter blocking read that drains the whole queue including the speculative K·S·K.

Same measurement at deg330 to see if warm convergence actually engages:

Read shell
Now I have the complete picture — and deg330 is actually worse per eval (~5s vs 3.4s at deg175 — 11.75ms/iter ksk). The tc2.tr stage = 95% of host time = the blocking read that drains each iteration's K·S·K. Now let me check what forces() needs so I can try the frozen-DM (no-SCC) Hessian path you suggested:







---

# GPT 5.6 sol

Yes. I have enough now to give you a useful diagnosis, and your suspicion was correct: **the current Hessian workflow is conceptually wasting orders of magnitude of work.** The 330-atom result should not be accepted as “sparse purification is just slow.”

The commit itself describes a persistent GPU workspace, but the electronic solver is being used in a way that largely defeats persistence.

### The main problem: every SCC iteration throws away the density solution

For each SCC iteration, `SparseDftb::scc()` does:

```text
build H_scc(q)
→ purify_hscc(...)
→ Mulliken q
→ DIIS
→ build another H_scc
→ purify_hscc(...)
→ ...
```

and `purify_hscc()` begins by constructing a **new spectral initial matrix from the current H_scc**:

```rust
let (emin, emax) = self.compute_k0_from_hscc(0.1)?;
...
self.tc2_purify(...)
```

while P-TC2/TRS similarly call:

```rust
self.compute_p0_from_hscc(0.1)?;
```

before every purification.

So although geometry changes by only ~0.02 Å and the previous density matrix is essentially the new answer, we repeatedly perform a **cold projector construction**.

That explains the absurd profiler result you got:

```text
~15–17 SCC iterations / displaced geometry
× ~55 purification iterations / SCC iteration
≈ 800–1000 purification iterations / force evaluation
```

and in the observed bad case apparently ~4200 TC2 iterations because of retries/floor behaviour.

For a Hessian of 330 atoms:

$$
6N = 1980
$$

force evaluations. At even 1 second/evaluation that's already 33 minutes.

But **there is no physical reason for a tiny finite-difference displacement near an SCC minimum to require 15 full electronic solves, each containing 50 purification steps.**

---

## There are actually three separate issues

### 1. SCC hot-start exists, but density-matrix hot-start does not

Some state reuse is already implemented correctly.

Charges `q` survive geometry changes. DIIS state is retained. More importantly, the inverse overlap \(Z\approx S^{-1}\) is explicitly warm-started from the previous geometry:

```rust
if !self.z_valid {
    self.ws.compute_z(..., self.z_warm)?;
    ...
    self.z_warm = true;
}
```

So Newton–Schulz is **not** being completely restarted.

But the expensive part, the electronic projector \(K\) or \(P=KS\), *is* effectively restarted.

This is the first thing I would change.

There is one subtlety, however: we cannot simply feed the old converged \(P\) into TC2 and expect it to find the new projector.

TC2 does only

$$
P\rightarrow P^2
\quad\text{or}\quad
2P-P^2.
$$

These are polynomials of the input matrix and therefore **do not rotate its eigenvectors**. Starting TC2 directly from \(P_{\rm old}\) can clean eigenvalues but cannot properly follow the occupied subspace of \(H_{\rm new}\).

So the correct hot-start is not merely “TC2 from old P”. We need a **density-matrix update that responds to the new Hamiltonian**, followed by a few purification/idempotency corrections.

I would implement that separately rather than contaminating the cold TC2 solver.

---

### 2. Every TC2 iteration has a blocking CPU↔GPU synchronization

This is the second large architectural problem.

The code explicitly downloads per-atom trace contributions and sums them in host f64 during TC2:

```rust
self.gpu.read_f32(&self.trace_atom, &mut self.trace_atom_host)?;
let tr_now: f64 =
    self.trace_atom_host.iter().map(|&x| x as f64).sum();
```

The comment even says:

> “The ONE blocking read — the queue drains…”

P-TC2 does the same via `trace_ks_f64()`, and TRS4 can make **three separate trace reads per iteration**.

This explains your profiling result where `tc2.tr` appears to consume 87–95% of the time.

Important nuance: that does **not necessarily mean calculating the trace itself takes 95%**. The blocking read is where the host waits for everything queued before it, so GPU work gets attributed to `tc2.tr`.

But from a performance perspective the conclusion is the same:

> **the GPU pipeline is forcibly serialized at every purification iteration.**

We previously moved the trace decision to host f64 deliberately because branch mistakes were causing TC2 instability. That was the right correctness debugging choice, but it is a terrible final GPU architecture when executed 50 × 15 × 1980 times.

Eventually the branch decision should happen on-device, or several steps should be fused/batched without host intervention.

I would not solve this first, though. Reducing 800 purification iterations/eval to perhaps ~2–10 useful electronic correction steps is a much bigger win.

---

### 3. H/S are indeed assembled on CPU

Your suspicion here is also correct.

Current `set_coords()` does:

```rust
assemble_hs_bsr(...,
    &mut self.h_bsr.values,
    &mut self.s_bsr.values)?;

self.ws.upload_h0(&self.h_bsr)?;
self.ws.upload_s(&self.s_bsr)?;
```

So Slater–Koster interpolation/rotation and BSR H/S construction are on the CPU, followed by H and S uploads.

And a dense f64 \(\gamma_{ij}\) matrix is also rebuilt on the CPU for every geometry.

However, **this is not today's bottleneck**.

Your microbenchmark says approximately:

```text
set_coords : 4.7 ms
SCC        : 3450 ms
forces     : 6.5 ms
```

So currently:

$$
\frac{t_{\rm SCC}}{t_{\rm total}}\approx99.7\%.
$$

Even 1980 × 4.7 ms is only about **9 seconds for all geometry/H/S assembly**.

Yes, I absolutely would eventually move H/S assembly to GPU. It is almost ideal GPU work here: frozen pair list, one work item per atom pair, unique BSR locations, no atomics, write both transpose blocks. You already have very similar machinery in the dense-multi code.

But doing that now while SCC costs seconds would be optimizing ~0.1% of the runtime.

---

# What I think the Hessian workflow should actually look like

We should separate several levels of electronic relaxation. This will also tell us immediately what physics matters.

For equilibrium geometry \(R_0\), first obtain a good state:

$$
q_0,\quad Z_0,\quad K_0/P_0,\quad H_0,\quad W_0.
$$

Save it explicitly as a **central Hessian reference state**.

Then every \(R_0\pm h e_i\) evaluation should start from that reference, not from a random previous Hessian column.

I would implement four benchmark modes:

| mode                           | what changes                                                | purpose                         |
| ------------------------------ | ----------------------------------------------------------- | ------------------------------- |
| **A. frozen electronic state** | geometry/H/S derivatives only; reuse central \(K,W,q\)      | fastest diagnostic              |
| **B. fixed SCC charge**        | keep \(q=q_0\), update electronic projector for displaced H | orbital-relaxation contribution |
| **C. warm electronic + SCC**   | initialize from central electronic state and correct it     | intended production Hessian     |
| **D. cold SCC**                | existing implementation                                     | reference only                  |

Mode A is particularly important right now.

If its full 330-atom Hessian is not fast, something else is badly wrong.

Given your measured ~5 ms set_coords + ~6 ms force time, I expect an order like

$$
1980 \times O(10\,{\rm ms})
\sim 20\ {\rm seconds},
$$

not an hour.

That is the sanity baseline I want.

---

## Frozen density is not the exact Born–Oppenheimer Hessian

There is a physics distinction worth keeping explicit.

If you completely freeze the central density and energy-weighted density, you omit electronic response:

$$
\frac{\partial K}{\partial R}
$$

and charge response

$$
\frac{\partial q}{\partial R}.
$$

So the resulting Hessian is a **frozen-electronic Hessian**, not the fully relaxed DFTB Hessian.

But it is an extremely useful experiment.

If frequencies are already reasonable, we learn that we can use very cheap response approximations.

If differences are substantial, we know exactly how much orbital/SCC relaxation matters.

There is also a valuable intermediate mode:

> Freeze SCC charges \(q_0\), but recompute the occupied projector for each displaced non-self-consistent Hamiltonian.

That includes orbital relaxation but excludes SCC charge response. It is probably much closer to the final Hessian than completely freezing \(K\).

---

# How I would make the production Hessian fast

I would **not** immediately try to make cold TC2 20% faster.

The architecture should change.

### Stage 1 — central-state Hessian API

Add something conceptually like:

```rust
struct SparseElectronicState {
    q: Vec<f64>,
    K: device_buffer,
    P: device_buffer,
    Z: device_buffer,
    // maybe DIIS state separately
}
```

At the optimized geometry:

```text
reference = engine.snapshot_electronic_state()
```

For every Hessian displacement:

```text
restore(reference)
set_coords(R0 ± h ei)
...
```

This is also cleaner numerically: +h and -h start from exactly the same electronic state, avoiding FD asymmetry due to SCF history.

Do not use Hessian column 487 as the electronic initial condition of column 488.

---

### Stage 2 — implement `forces_frozen_density()`

No SCC.

No purification.

No Newton–Schulz unless the chosen force expression actually needs updated \(Z\).

Use the central electronic state and displaced geometry.

This should establish the lower-bound cost immediately.

**Target for R10/330 atoms: tens of milliseconds per force, therefore tens of seconds for the Hessian.**

If this is not achieved, stop and profile before doing anything more sophisticated.

---

### Stage 3 — fixed-\(q\), electronically relaxed

Use

$$
q=q_0
$$

so there is no SCC loop.

Build displaced

$$
H_{\rm SCC}(R;q_0).
$$

Now update the density matrix once.

Current code would still cold-purify it, which is wasteful, but this mode tells us separately:

```text
geometry cost
+ one electronic solve
```

rather than:

```text
15 SCC × one electronic solve.
```

Even with today's cold purifier, that should be roughly a **15× speedup**.

---

### Stage 4 — real warm density update

This is where I would invest algorithmic work.

For tiny \(\delta R\),

$$
H' = H+\delta H,
\qquad
S'=S+\delta S
$$

and

$$
P'=P+\delta P.
$$

We should update the occupied subspace rather than reconstructing it from a broad spectral map.

Possible approaches, from simplest to more sophisticated:

1. a few projected density-matrix minimization steps using \(H_{\rm new}\);
2. commutator-gradient correction based on \([H,P]\);
3. first-order density-matrix perturbation;
4. eventually a proper sparse density-matrix minimization method.

The crucial property is that the step contains \(H_{\rm new}\), so it can **rotate occupied ↔ virtual subspaces**. Pure TC2 of old \(P\) cannot.

For the Hessian displacement, I would expect only a handful of such corrections.

---

### Stage 5 — SCC should then require ~1–3 iterations, not 15–17

Start every displacement from \(q_0\).

For \(h=0.01\)–0.02 Å near equilibrium,

$$
q(R_0+h)\approx q_0 + O(h),
$$

so the initial SCC residual should already be small.

A sensible execution should look something like:

```text
central solution
    ↓
displaced H/S
    ↓
warm electronic correction
    ↓
new q
    ↓
maybe DIIS correction
    ↓
1–3 SCC iterations
```

not:

```text
displace
↓
cold 55-step purification
↓
charge update
↓
cold 55-step purification
↓
...
× 17
```

That is the conceptual stupidity currently dominating runtime.

---

# Do not confuse “purification floor” with a reason to run more iterations

The sparse mask case reaching

$$
R_I\sim5.9\times10^{-4}
$$

and then running to ~55 iterations every time is especially wasteful.

Once a structural truncation gives a numerical floor, additional iterations do not improve the result.

So for any iterative solver:

```text
converging → continue
reached measured numerical floor → STOP
diverging → fail/recover
```

Never:

```text
not below requested mathematical tolerance
→ blindly execute tc2_max again
```

This was one of our earlier design rules, and Hessians magnify violations by a factor of ~2000.

---

# Host synchronization should be attacked after that

Once we reduce the number of electronic iterations drastically, I'd change TC2/P purification control too.

Current architecture effectively does:

```text
GPU product
GPU reduction
CPU blocking read
CPU f64 branch
GPU product
GPU reduction
CPU blocking read
...
```

Ultimately I would use a hybrid:

```text
far from Nocc:
    device trace / device branch
    several completely GPU-resident iterations

near branch ambiguity or convergence:
    accurate host-f64 decision/check
```

Or use a tiny GPU double reduction if available.

Your sensible principle remains:

> **bulk matrices f32; precision only for tiny decisions.**

But “tiny decision in f64” does not necessarily require a PCIe synchronization after every matrix multiplication.

---

# Priority order I would give the coding agent

1. **Stop doing the full R10 Hessian.** Benchmark one atom / one Cartesian ± displacement only.
2. Add explicit timing counters for:
   `geom.hs`, upload, gamma, NS, Hscc, projector initialization, each SpGEMM, trace/read synchronization, Mulliken, DIIS, forces.
3. Implement **frozen-density force FD** and measure one ± pair.
4. Implement **fixed-q / no-SCC** evaluation and measure one ± pair.
5. Add a central electronic-state snapshot/restore specifically for Hessian calculations.
6. Ensure every ± displacement starts from the central solution.
7. Replace cold density reconstruction inside SCC by a genuine **Hamiltonian-aware warm density update**.
8. Require SCC near equilibrium to take only a few iterations; investigate anything >~3–5.
9. Stop purification at its measured mask/f32 floor instead of `tc2_max`.
10. Only then eliminate the per-TC2 host synchronization.
11. After SCC is cheap, port direct H/S pair assembly to GPU.
12. Profile gamma; port it only if it becomes material.

The current code confirms H/S assembly is CPU-side and uploaded afterwards, despite the rest of the sparse pipeline being persistent GPU-resident.  The workspace itself is already designed to keep its matrices, buffers and symbolic SpGEMM plans alive across geometry displacements, so the necessary infrastructure is mostly there.

### The performance target I would set

For the 330-atom R10 system, I would reject the implementation unless approximately:

```text
frozen-density force       ~ 5–20 ms
fixed-q warm electronic    ~ 10–50 ms
full warm SCC force        ~ 20–100 ms
```

is at least approached on a decent GPU.

Then 1980 force evaluations correspond roughly to:

```text
20 ms → 40 s
50 ms → 100 s
100 ms → 3.3 min
```

That is the scale this project should be aiming at.

**One hour means the algorithmic reuse is broken, not that the problem is intrinsically expensive.**

And I would keep the current R10 timing out of the tutorial as anything resembling representative sparse-solver performance. Right now it is primarily measuring **cold nested purification × SCC × finite differences**, which is exactly the workflow we should remove.








---

# USER

look there is some trace of some run, I think we reperatdly dance at numericall floor this is fucking nonsense! we should either make sure we cut the iteration before we hit the numerical floor, or introduce some correstions overcoming f32 accuracy (Kaan, f64 hybrid) and minimal performance cost, but we cannot affort to to 10 iterations without moving anywhere. I feel like you were saying this numerical floor is not major problem (that we have other more urgent issues to adress first), but it seems to me it is a major issue becasue you cannot rely on the numbers leading the algorithm to do many uselese iterations going nowhere.


---

# Chat GPT 5.6 sol

Yes. Based on this trace I would revise the priority: **the numerical-floor handling is a major performance bug right now.** You are absolutely right that it cannot be allowed to spend 30–50 iterations proving repeatedly that \(2\times10^{-5}\) is still \(2\times10^{-5}\).

The clearest example is SCC iteration 1. By TC2 iteration 27 it has already reached

$$
R_I=2.61\times 10^{-5},
$$

and at iteration 30 it reaches its eventual best,

$$
R_I=2.446\times10^{-5}.
$$

Then it continues all the way to **iteration 79**, wandering between roughly \(3\times10^{-5}\) and \(10^{-4}\), before restoring the K it already had at iteration 30. That is **~50 completely useless expensive iterations**. 

The same thing happens elsewhere: one SCC iteration reaches \(2.25\times10^{-5}\) at TC2 iter 32 and then wastes almost another 50 iterations before giving up at 80. 

And because each TC2 iteration contains expensive SpGEMMs **plus a blocking host synchronization for the trace**, this is catastrophic for runtime.

## I think there are two fixes, and we should do both

First, **make it impossible to waste iterations at the floor**. Second, try to **lower the floor cheaply**, because \(R_I\sim2\times10^{-5}\) may itself produce undesirable force noise for Hessians.

The first fix should happen immediately and independently of precision work.

### 1. The current plateau detector is conceptually wrong

There is already:

```rust
TC2_STAGNANT_MAX = 8
```

but the code comment says it counts iterations without **joint `(R_I, trace)` improvement**.

That is likely why it fails.

Look at the trace after convergence:

```text
R_I              Tr(KS)
2.4460e-5        459.000030
3.8484e-5        459.000025
6.2243e-5        459.000010
1.0224e-4        458.999990
3.6739e-5        459.000020
...
```

The trace is already effectively perfect:

$$
\operatorname{Tr}(KS)=459.00000\pm O(10^{-5}).
$$

Tiny random changes in trace should **not reset a stagnation counter**.

Once trace is locked onto \(N_{\rm occ}\), the optimization criterion should simply be:

$$
\boxed{\text{minimize }R_I}
$$

subject to trace remaining valid.

So I would change the state machine to:

```text
Before trace lock:
    normal TC2 logic

After:
    |Tr-Nocc|/Nocc < trace_lock
    AND R_I < endgame_threshold (~1e-3):

        save best K whenever R_I improves meaningfully

        trace only acts as validity/runaway guard;
        trace "improvements" NEVER reset plateau patience

        if best R_I did not improve by >~5% for 6–8 iterations:
            restore best K and STOP

        if current R_I > ~4 × best_RI:
            restore best K and STOP immediately
```

The word **meaningfully** matters. Do not reset patience because

$$
2.4510\times10^{-5}\rightarrow2.4499\times10^{-5}.
$$

For example:

```rust
if ri < best_ri * 0.95 {
    best_ri = ri;
    save_K();
    stagnant = 0;
} else {
    stagnant += 1;
}
```

Perhaps use 2–5%, but definitely not any floating-point decrease.

With the trace you supplied, this would usually stop around iteration 30–40 instead of 80.

That alone can cut the TC2 cost by approximately **2×** in these calls.

And importantly:

> `tc2_max=80` should be an emergency ceiling, not the normal mechanism by which we discover the f32 floor.

If a normal SCC repeatedly prints `TC2 exhausted 80 iters`, I would treat that as an algorithmic failure condition.

---

# 2. More importantly: the observed floor smells strongly like f32 accumulation

This particular experiment is very informative.

You use:

```text
r_k = 20 Å
deg_k = 330
deg_z = 330
```

for a 330-atom system. So K and Z essentially have **full atom support**.

Therefore the label

> `f32/mask floor`

is misleading here.

There is essentially no atom-block K truncation to blame. Yet the repeatable floor is

$$
R_I \sim 2-3\times10^{-5}.
$$

That points strongly toward **f32 arithmetic**, particularly SpGEMM accumulation.

And your current planned SpGEMM indeed accumulates each scalar as four f32 streams:

```c
float s0, s1, s2, s3;

for(t ...){
    s0 = fma(..., s0);
    s1 = fma(..., s1);
    s2 = fma(..., s2);
    s3 = fma(..., s3);
}

C = (s0+s1)+(s2+s3);
```

With degree \(\sim330\), each of those four streams sums approximately **hundreds of terms**.

A rough f32 accumulation error scale is

$$
n\epsilon_{\rm f32}
\sim330\times6\times10^{-8}
\sim2\times10^{-5}.
$$

That is almost suspiciously identical to the measured plateau.

So I no longer think Kahan/compensation is merely some later polishing step. **This run gives an actual numerical reason to try it now.**

---

## Conveniently, ACC8 is already implemented!

Your kernel already contains an ACC8 variant. It splits each of those four sums into two alternating accumulators:

```c
s0,s1,s2,s3
q0,q1,q2,q3
```

so each chain is roughly half as long, with no additional products.

But it is **OFF by default**:

```text
RUST_DFTB_SPGEMM_ACC8=1
```

enables it.

So before writing any new code, I would rerun exactly one of these SCC evaluations with:

```bash
RUST_DFTB_SPGEMM_ACC8=1
```

and compare:

```text
minimum R_I
TC2 iterations
SpGEMM kernel time
total SCC time
forces
```

This is a very cheap and extremely informative test.

If the floor changes from, say,

$$
2.3\times10^{-5}
\rightarrow1.0\times10^{-5}
$$

or below with negligible slowdown, **promote ACC8 to default immediately**.

The existing comment that ACC8 changes forces at the ulp level should not scare us away. Right now the 4-accumulator implementation is demonstrably preventing convergence to the requested tolerance. The relevant question is which result is more accurate against a higher-precision reference.

---

# I would also immediately try ACC16

It is almost free to implement.

Instead of two quads:

$$
s_m,\quad q_m
$$

use four quads:

$$
s_m,\ q_m,\ u_m,\ v_m.
$$

Distribute plan terms cyclically:

```text
t mod 4
```

and combine them pairwise at the end.

For degree 330:

```text
ACC4  : ~330 additions / accumulator
ACC8  : ~165
ACC16 : ~82
```

The arithmetic work—the number of multiplications/FMA products—is unchanged. Cost is primarily extra registers and a few final additions.

On the 3090, I would simply benchmark:

```text
ACC4
ACC8
ACC16
```

for one representative product.

There is a real possibility that ACC8/16 is **both faster and more accurate**, because shorter dependency chains expose more instruction-level parallelism.

Register pressure might eventually hurt occupancy, but that is exactly what a 30-second benchmark tells us.

---

# Then Kahan only for the endgame

If ACC8/ACC16 gets us close but still gives a floor, I would implement a second precision kernel using compensated f32 summation.

For each stream:

$$
y=p-c
$$

$$
t=s+y
$$

$$
c=(t-s)-y
$$

$$
s=t.
$$

This stays entirely f32. No horrific RTX-3090 fp64 penalty.

I would **not use Kahan for all TC2 iterations**.

The early TC2 stage has residuals

$$
10^{-1},10^{-2},10^{-3}
$$

where \(10^{-7}\) arithmetic precision is irrelevant.

Instead:

```text
fast ACC8/ACC16:
    while R_I > 1e-3

precise Kahan:
    once R_I < 1e-3
```

Then perhaps only ~3–10 expensive precision iterations ever occur.

That is exactly the kind of mixed-precision strategy we want.

---

# Before implementing Kahan, distinguish two different floors

There is one very useful diagnostic.

Take the **best K snapshot** at the apparent \(R_I\sim2.2\times10^{-5}\) floor and calculate

$$
R_I =
\frac{\|KSK-K\|_F}{\|K\|_F}
$$

once with a higher-accuracy reference.

For this debug experiment it is perfectly acceptable to download K and compute KSK in CPU f64, or implement one very slow f64 diagnostic. It happens only once.

Two possibilities:

### A. f64 says \(R_I\ll2\times10^{-5}\)

Then K itself is fine and **the residual measurement is noisy**.

In that case we should fix the diagnostic/reduction, not the purification.

### B. f64 also says \(R_I\approx2\times10^{-5}\)

Then the actual TC2 matrix multiplication/update is hitting the f32 floor.

ACC8/16/Kahan in SpGEMM is exactly the right intervention.

I strongly suspect B, because repeated KSK multiplication accumulates hundreds of terms, but this one diagnostic removes all uncertainty.

Note that your trace calculation is already effectively high precision at the final summation—the per-atom values are summed in host f64—and the trace is extremely stable. So **trace precision is not what needs fixing**.

---

# I would not use full f64 SpGEMM

I still stand by that.

On the RTX 3090 the f64 throughput penalty is enormous. Doing all sparse products in double precision is the wrong fix.

The hierarchy should be:

$$
\boxed{
\text{ACC8/16 f32}
\rightarrow
\text{selective Kahan f32}
\rightarrow
\text{tiny f64 reductions/diagnostics}
}
$$

not:

$$
\text{everything fp64}.
$$

A particularly attractive implementation is to compile two planned kernels:

```text
spgemm_fast      // ACC8 or ACC16
spgemm_precise   // f32 Kahan
```

and switch to `spgemm_precise` only in TC2 endgame.

---

# There is also a separate problem at the very end of this trace

The Hessian-specific `FIXQ + DMUPD` experiment is **not hitting the same numerical floor**.

It starts the displaced geometry at:

$$
R_I=1.35\times10^{-3}
$$

and then TC2 makes it progressively *worse*:

$$
1.35\times10^{-3}
\rightarrow2.67\times10^{-3}
\rightarrow5.36\times10^{-3}
\rightarrow1.08\times10^{-2}
\rightarrow2.21\times10^{-2}.
$$

The plateau detector correctly aborts after five iterations there.

That is the separate issue we discussed before: **starting from old K and applying TC2 does not rotate the occupied subspace toward the displaced Hamiltonian**. Pure polynomial purification of old K cannot serve as the Hamiltonian-aware density update.

And immediately afterwards it also hits the dummy-orbital occupation assertion.

So I would keep these bugs conceptually separate:

1. **cold SCC TC2:** real f32 floor around \(2\times10^{-5}\), and floor detection wastes tens of iterations;
2. **warm displaced DM update:** algorithmically wrong update, \(R_I\) diverges rather than stagnates;
3. **dummy orbital occupancy:** additional force-path correctness bug.

---

## What I would tell the coding agent now

The current TC2 numerical-floor behavior is now a P0 performance bug.

The supplied R10 trace shows repeated cases where the best idempotency residual is reached around iteration 25–35, but TC2 continues to iteration 60–80 and finally restores exactly the K it had tens of iterations earlier.

Example: one SCC iteration reaches `R_I=2.446e-5` at TC2 iter 30, then performs another 49 iterations without improvement and exits at iter 80. This must never happen in production.

### 1. Fix floor/stagnation detection first

After TC2 has entered the endgame:

* trace is locked:
  `abs(Tr(KS)-Nocc)/Nocc < TC2_TRACE_LOCK_REL`
* and `R_I < ~1e-3`

treat trace only as a validity/runaway constraint.

Do NOT reset stagnation because `Tr(KS)` improves by tiny f32 noise once it is already locked.

Track the best valid K by `R_I`.

Reset the stagnation counter only for a meaningful residual improvement, e.g.

`R_I < 0.95 * best_R_I`

Use approximately 6–8 iterations of patience.

Also exit immediately and restore best K if, after entering endgame,

`R_I > 4 * best_R_I`.

The normal path must not reach `tc2_max=80` after having already found a valid low-R_I state.

`tc2_max` is an emergency ceiling, not a floor detector.

### 2. Diagnose whether the 2e-5 floor is real

This R10 calculation has `deg_k=deg_z=330` for 330 atoms, so K/Z have effectively full atom support. Therefore the observed ~2e-5 plateau is unlikely to be primarily a geometric K-mask truncation floor.

For one best-K snapshot, compute

`||KSK-K||/||K||`

with a high-accuracy reference:

* CPU f64 is fine for this one diagnostic, or
* a deliberately slow f64/compensated reference kernel.

If the high-precision residual is much smaller than the f32 residual, fix the residual computation.

If it remains ~2e-5, the SpGEMM/update itself is hitting the f32 floor.

### 3. Immediately A/B the already implemented ACC8 kernel

Run the identical SCC calculation with

`RUST_DFTB_SPGEMM_ACC8=1`.

Measure:

* minimum R_I,
* TC2 iteration count,
* SpGEMM kernel time,
* SCC wall time,
* energy/forces versus ACC4 and high-precision reference.

ACC8 performs the same products but halves each serial f32 accumulation chain. If it lowers the floor significantly without material slowdown, promote it to production default.

### 4. Add ACC16 experiment

Implement four independent accumulator quads, distributing plan terms by `t mod 4`, followed by pairwise combination.

Compare ACC4 / ACC8 / ACC16.

No extra matrix products or FMAs should be introduced; only reassociation/register use changes.

Because degree is ~330, chain lengths become approximately:

* ACC4: 330 terms
* ACC8: 165 terms
* ACC16: 82 terms

The observed 2e-5 floor is quantitatively consistent with accumulation error over hundreds of f32 terms, so this experiment is high priority.

### 5. Add a separate compensated-f32 endgame SpGEMM only if needed

If ACC8/16 is insufficient, implement a planned SpGEMM variant with Kahan/compensated f32 accumulation.

Do not use it during the whole purification.

Use fast ACC8/16 while `R_I > ~1e-3`.

Switch to compensated SpGEMM only for the TC2 endgame.

This should provide near-f64 summation quality without the RTX-3090 f64 throughput penalty.

### 6. Do not use full fp64 sparse multiplication

Keep matrix storage and normal products f32.

Use f64 only for:

* small host/device reductions,
* convergence decisions,
* occasional reference diagnostics.

### 7. Keep the Hessian DM-update failure separate

The FIXQ/DMUPD displaced run is not merely at an f32 floor: its R_I grows from `1.35e-3` to `2.21e-2` in five TC2 steps.

That indicates the present warm-DM update is algorithmically unsuitable: TC2 applied to the previous projector cannot rotate the occupied subspace toward the new Hamiltonian.

Fix that separately with a Hamiltonian-aware density update.

Do not confuse this divergence with the ~2e-5 cold-TC2 numerical floor.

So my changed recommendation is: **fix the early floor exit immediately, then run ACC8 today before doing anything sophisticated.** You already have the code for it. If the floor is really dominated by accumulation length, ACC8/ACC16 could be an unusually cheap improvement in both accuracy **and** performance.










---

# USER

all these ACC4-ACC16 methos seems totally useless to me, they do not improve oscillation at the floor whatsoever!

---

Read it — GPT-5.6's plan is clear: fix floor detection, diagnose whether the 2e-5 floor is f32 accumulation (ACC4 chain length ~330 matches suspiciously), A/B the existing ACC8 kernel, try ACC16, Kahan endgame if needed. Let me inventory what exists:

Clear plan. `compute_k0_from_hscc` gives an identical cold seed for every variant — perfect for a controlled study.

Now ACC16 — config + kernel. Check the config:

Now the ACC16 kernel paths — bsym variant first:

Now the generic (non-symmetric) plan kernel:

Both TC2 products use the Bsym plan kernel — covered. Now build and write the study script:

Build clean. Check the R10 script setup to reuse:

Run the baseline (ACC4, no stop-window → full floor dance visible):

Baseline works — 150 iters at floor R_I=1.78e-5, ~12.4ms/iter. Now ACC8 and ACC16:

ACC8: floor 1.75e-5 (vs 1.78e-5) — marginal improvement, plateau detector cut at iter 119. Now ACC16:

ACC16: floor **1.32e-5** (vs 1.78e-5), plateau caught at iter 82, **0.51s — half the per-iter cost of ACC4** (shorter dependency chains → more ILP, as predicted). But the floor didn't drop 4× → accumulation is only part of it. Two more variants — stop-window comparison:

Now the plot — parse the last `# call` block from each CSV, log-scale R_I and trace deviation:

REVIEW: `debug/tc2_conv_r10.png` `@/home/prokop/git/dftbplus/debug/tc2_conv_r10.png`

The study is running and the picture is already informative:

**Cold purify on converged-regime H_scc (R10, 330 Si, deg~330), per-iter cost ~12ms→6ms:**

| variant | floor R_I | iters | time |
|---|---|---|---|
| ACC4 (default) | 1.78e-5 | 150 (max) | 1.86 s |
| ACC8 | 1.75e-5 | 119 (plateau cut) | 0.99 s |
| **ACC16 (new)** | **1.32e-5** | 81 (plateau cut) | **0.51 s** |
| ACC4 + W=8 | 1.78e-5 | 37 | 0.47 s |
| ACC16 + W=8 | 1.52e-5 | 36 | 0.23 s |

What the plot shows:

1. **All variants reach the floor by iter ~25-30**, then sawtooth-oscillate between ~1e-5 and ~7e-5 — the "dance" is branch alternation bouncing off the floor, exactly the waste GPT-5.6 flagged.

2. **ACC16 is both more accurate AND ~2× faster per iter** (6.2ms vs 12.4ms — shorter dependency chains = more ILP). But the floor only dropped 1.78→1.32e-5, not the ~4× chain-length scaling would predict → **f32 accumulation is only part of the floor**; ~1e-5 sits elsewhere (state truncation/K storage/residual measurement).

3. Trace locks at ~1e-8 by iter ~25 and never degrades — trace is not the problem; R_I oscillation is.

Next diagnostics queued by the notes: (a) f64 reference residual of best-K to separate "state at floor" vs "measurement noise", (b) McWeeny-endgame hybrid switch, (c) Kahan compensated SpGEMM for the endgame. Want me to continue with the McWeeny/static-switch hybrid variant next, or the f64 residual diagnostic?


---






# Chat GTP 5.6 sol

Yes — I agree with you. **ACC4/8/16 is basically a dead end as a cure for the floor oscillation.** ACC16 is worth keeping only because it is faster per iteration on this GPU, not because it fixes convergence.

The plot says something much more specific: all variants follow essentially the same trajectory until \(R_I\sim10^{-5}\), then enter the same sawtooth. Changing accumulation order shifts the minimum from roughly \(1.8\times10^{-5}\) to \(1.3\times10^{-5}\), but does not change the qualitative behavior at all. So the dominant problem is not summation-chain error.

What you are seeing is actually very characteristic of **TC2 near a finite numerical floor**.

For an occupied eigenvalue write

$$
\lambda_o=1-\delta,
$$

and for an unoccupied one

$$
\lambda_u=\epsilon.
$$

The two TC2 branches do opposite things:

$$
X\rightarrow X^2:
\qquad
\delta\rightarrow 2\delta,\qquad
\epsilon\rightarrow\epsilon^2
$$

while

$$
X\rightarrow2X-X^2:
\qquad
\delta\rightarrow\delta^2,\qquad
\epsilon\rightarrow2\epsilon.
$$

So one branch fixes the empty states while making the occupied-state error about twice as large; the other fixes occupied states while doubling the small empty-state error.

Far above machine/truncation noise, alternating these branches works beautifully because over two steps both errors become quadratic.

But once every multiplication injects some floor \(\eta\), you get roughly:

$$
\eta \rightarrow 2\eta \rightarrow \eta \rightarrow2\eta\ldots
$$

That is almost exactly the sawtooth in your figure.

So **the oscillation is an intrinsic failure mode of trace-correcting TC2 at finite precision**. Making the product slightly more accurate merely changes \(\eta\). It does not remove the oscillatory mechanism.

This also explains why the trace plot looks absurdly good while \(R_I\) dances: the trace is already locked to \(N_{\rm occ}\) at \(10^{-8}\)-relative accuracy, yet TC2 keeps choosing branches whose only remaining effect is to move numerical error between occupied and empty subspaces.

## Therefore I would change the strategy

Once

$$
|\mathrm{Tr}(KS)-N_{\rm occ}|/N_{\rm occ}
$$

is tiny and

$$
R_I\lesssim10^{-3}
$$

or perhaps \(10^{-4}\), **TC2 has finished its job**. It has established the correct rank and brought the spectrum close to \(\{0,1\}\).

At that point stop using the trace-correcting polynomial.

Switch to a polynomial that contracts **both sides simultaneously**.

The obvious one is McWeeny:

$$
P' = 3P^2-2P^3.
$$

Near 0:

$$
\epsilon' \approx 3\epsilon^2,
$$

and near 1:

$$
\delta' \approx 3\delta^2.
$$

So there is no alternating “fix occupied / damage empty / fix empty / damage occupied” behavior.

For your nonorthogonal K representation the corresponding expression is

$$
\boxed{
K' = 3KSK - 2KSKSK
}
$$

because multiplying by \(S\) gives the ordinary McWeeny polynomial for \(P=KS\).

That is exactly what I would test next.

And since it is only an **endgame correction**, I don't care if one McWeeny iteration costs twice as much as one TC2 iteration. Compare:

```text
current:
25 useful TC2
+ 30–100 useless TC2

proposed:
25 useful TC2
+ perhaps 1–4 McWeeny
```

That is clearly preferable.

---

There are actually two separate performance fixes I would make now.

**First: floor exit must become aggressive even without McWeeny.**

The current W=8 detector is still too timid conceptually. Once trace is locked, I would track the sequence of local minima of \(R_I\). Your plot shows an obvious two/three-step periodic pattern. If, after entering the endgame, two or three branch cycles fail to improve the best residual substantially, restore the best K and quit.

Something like:

```text
trace locked
R_I < 1e-3

best = current R_I

for each next iteration:
    if R_I < 0.9 * best:
        best = R_I
        save K
        n_cycle_no_improve = 0

    detect TC2 branch cycle

    if 2–3 complete branch cycles without 10% improvement:
        restore best K
        NumericalFloor
        STOP
```

I would prefer counting **complete TC2 branch cycles**, rather than arbitrary “8 iterations”.

With the curves in your plot, TC2 should terminate around iteration ~30–35, not 80 or 150.

**Second: switch to McWeeny instead of declaring the floor immediately.**

For example:

```text
TC2 until:
    trace locked
    R_I < 1e-3

then McWeeny:
    max 3–5 iterations

if McWeeny lowers R_I:
    continue until improvement stalls

otherwise:
    restore best state and stop
```

This is much more physically/mathematically sensible.

---

### I would do the f64 diagnostic before Kahan

The agent asked “McWeeny or f64 residual diagnostic?”

I would do the **f64 diagnostic first, but only as a very quick diagnostic**, then immediately test McWeeny.

Take the best ACC16 K around its \(1.3\times10^{-5}\) minimum and evaluate

$$
R_I=\frac{\|KSK-K\|}{\|K\|}
$$

once in CPU f64.

That tells us whether:

1. the **actual K** has residual \(1.3\times10^{-5}\), or
2. K is considerably better and the f32 evaluation of \(KSK-K\) itself produces an artificial \(10^{-5}\) floor.

This is worth knowing before implementing Kahan.

If f64 gives, say,

$$
R_I^{64}\sim10^{-7},
$$

then there is almost nothing wrong with K: the residual diagnostic is lying at the f32 floor. Then definitely do **not** spend time making purification more accurate.

If it gives

$$
R_I^{64}\sim10^{-5},
$$

then the state itself is at that floor, and McWeeny/Kahan/endgame precision can genuinely improve it.

This should be a single diagnostic calculation, not a new production pathway.

---

## I would not implement Kahan yet

ACC16 already provides an important clue.

If simple summation error were dominant, cutting the accumulator chain from ~330 to ~82 terms should have produced a much larger improvement than

$$
1.78\times10^{-5}
\rightarrow1.32\times10^{-5}.
$$

It didn't.

Therefore Kahan might improve the last factor 1.5–3, but **it probably will not remove the TC2 oscillation**, because the polynomial itself creates it.

So my current ordering is:

1. **f64 residual of one saved best-K** — establish what the real floor is.
2. **TC2→McWeeny hybrid** — this is the most important experiment.
3. **aggressive cycle-aware floor termination** regardless of precision.
4. Keep **ACC16** if its ~2× speedup is reproducible.
5. Only then test **Kahan endgame** if McWeeny still hits an unacceptably high true residual.

There is another diagnostic I would add almost for free: for the alternating endgame states, measure not only \(R_I\), but

$$
\Delta K_n =
\frac{\|K_{n+1}-K_n\|}{\|K_n\|},
$$

and especially the actual observables:

$$
\Delta E,\qquad
\Delta q_{\max},\qquad
\Delta F_{\max}.
$$

Because for the Hessian, **force stability is ultimately what matters**, not whether an abstract idempotency metric is \(1.3\times10^{-5}\) or \(3\times10^{-6}\).

If the force difference between the best TC2 states at iterations 28, 30, 32 is, say,

$$
10^{-7}\ {\rm Ha/\AA},
$$

then simply stopping at \(R_I\simeq1.5\times10^{-5}\) is entirely fine.

If it is

$$
10^{-4}\ {\rm Ha/\AA},
$$

then the floor really matters for vibrations and we need to lower it.

So I would tell the agent:

> **Stop exploring ACC variants. Keep ACC16 only as a performance optimization. Run one f64 best-K residual check, then implement a TC2→McWeeny endgame switch and measure \(R_I\), \(\Delta K\), energy and force stability. TC2 must never spend tens of iterations in the sawtooth.**

That is now much more strongly justified by your plot than Kahan is.




---

# USER

OK, now sure I quite understand this results:

---

# SWE-2

Fair — ACC8/16 moved the floor only ~25%, nowhere near useful. And you're right, I skipped Kahan which was the actual suggestion for the endgame. Let me answer the termination question, check the residual kernel's own accumulation, then implement Kahan.

Why ACC16 stopped earlier: the plateau detector fires when `r_i > 10×best`. The sawtooth spikes are ~1e-4 absolute on all variants, but ACC16's *best* dips lower (1.3e-5) → the spike ratio crosses 10× sooner. ACC4's best stays ~1.7e-5 so the same spikes are only ~4× — it never trips and burns to iter 150.

Check the residual-measurement kernel first — if it accumulates in plain f32, the *measured* floor is its own artifact:

Residual kernel is a strided partial + f64 host finish — the measurement is fine; the floor lives in the f32 products themselves. Implementing Kahan-compensated accumulation in both plan kernels (the real GPT suggestion I skipped):

Now the config flag:

Kahan: floor **9.7e-6** — only ~1.8× better, not 2 orders. Important negative result: accumulation length is NOT the dominant floor source. Let me check the curve shape and the last-call data:

The decisive facts now: Kahan cut the floor only 1.78e-5→9.7e-6 (~2×) — accumulation is a minor contributor. The remaining floor is the **TC2 map itself**: a period-2 limit cycle (sawtooth dips/spikes), i.e. a saddle the iteration can't sit on. That means the fix isn't more summation precision — it's a different endgame map. McWeeny is attracting rather than alternating, and `mcweeny_polish` already exists — let me wire a measure-and-polish path into the study.

Now add the McWeeny-endgame hybrid inside `tc2_purify` — env-gated switch once trace-locked:

REVIEW: `debug/tc2_conv_r10.png` `@/home/prokop/git/dftbplus/debug/tc2_conv_r10.png`

Full variant ladder (cold purify, same H_scc, deg~330):

| variant | min R_I | iters | character |
|---|---|---|---|
| ACC4 | 1.78e-5 | 150 | sawtooth dance |
| ACC8 | 1.75e-5 | 119 | sawtooth |
| ACC16 | 1.32e-5 | 81 | sawtooth, 2× faster/iter |
| Kahan | 9.7e-6 | 39 | floor + trace runaway → abort |
| **McWeeny endgame** | **5.4e-6** | 150 | **smooth monotone descent, no sawtooth** |
| Kahan+McW | 5.4e-6 | 150 | identical |
| ACC16+W8 | 1.52e-5 | 36 | early stop |

Honest conclusions:

1. **You're right — nothing here gives 2 orders.** The floor is ~5e-6–2e-5 regardless of summation scheme. Kahan proved it: near-exact products barely moved the floor → **the floor is a property of the map/state, not accumulation noise.**

2. **Why ACC16 stopped earlier**: the plateau detector fires at `R_I > 10×best`. Spikes are ~1e-4 absolute on all variants; ACC16's best dips lower (1.3e-5) so the ratio crosses 10× sooner. ACC4's shallower best never trips it → burns to iter 150. The detector measures ratios, not the floor.

3. **McWeeny endgame is qualitatively different**: once trace-locked, it descends monotonically (still slowly falling at 150 — hasn't converged, may reach ~4e-6). But its extra products use the *masked* kernel (no ACC/Kahan) and it doesn't conserve trace (flat ~2.5e-6 dev).

4. To actually reach ~1e-7 the iteration needs f64 in the state (K in f64 for the last few steps) or a different fixed-point entirely — not better f32 summation. The residual measurement itself is honest (f64 host finish).

Want me to (a) try an f64-state endgame on best-K (a few McWeeny/TC2 steps with K in f64 — diagnostic, RTX3090 f64 is slow but it's 3-5 iters), or (b) accept the ~5e-6 McWeeny floor and move back to the loop-level work?





---

# Chat GPT 5.6 sol

Yes. The plot is useful, but I think the agent’s interpretation mixes together several different things. In particular, I **do not agree with the conclusion that “to get \(10^{-7}\) we need f64 state.”** The data do not establish that yet.

The main message of the figure is simpler:

$$
\boxed{\text{TC2 is excellent for getting close to the projector, but bad as an endgame iteration in f32.}}
$$

And McWeeny looks like the right endgame map — but the McWeeny implementation/result is itself suspicious and needs one more diagnostic.

### What the curves actually say

Up to roughly iteration 22–24, every curve is almost identical. That is expected: TC2 is reducing a large spectral error from \(10^{-1}\) down toward \(10^{-4}\), and floating-point details are irrelevant there.

Then TC2 reaches the regime where

$$
\operatorname{Tr}(KS)\approx N_{\rm occ}
$$

essentially perfectly. From this point the branch alternates between

$$
K\rightarrow KSK
$$

and

$$
K\rightarrow2K-KSK.
$$

Near an ideal projector, write occupied eigenvalues as \(1-\delta\) and empty ones as \(\epsilon\). The two branches behave approximately as

$$
KSK:\qquad
\delta\rightarrow2\delta,\quad
\epsilon\rightarrow\epsilon^2,
$$

and

$$
2K-KSK:\qquad
\delta\rightarrow\delta^2,\quad
\epsilon\rightarrow2\epsilon.
$$

So once you have a finite arithmetic error \(\eta\), the endgame naturally becomes something like

$$
\eta \rightarrow 2\eta \rightarrow \eta \rightarrow2\eta\ldots
$$

That is your sawtooth.

So ACC4/8/16/Kahan cannot fundamentally fix this. They only change the size of \(\eta\).

That is exactly what the plot shows:

$$
1.8\times10^{-5}
\rightarrow1.3\times10^{-5}
\rightarrow9.7\times10^{-6}
$$

as summation gets better, **but the TC2 sawtooth survives**.

Therefore I agree with you: further ACC variants are not interesting for convergence. ACC16 may still be worth keeping because it happens to be faster, but that is a separate performance optimization.

---

## McWeeny is the genuinely interesting result

The brown/purple McWeeny curves behave qualitatively differently:

$$
R_I \text{ decreases monotonically}
$$

instead of oscillating.

That is precisely what we expected from

$$
K' = 3KSK-2KSKSK.
$$

Equivalently, for \(P=KS\),

$$
P'=3P^2-2P^3.
$$

Near 0 or 1 both errors are contracted simultaneously:

$$
\epsilon' \approx 3\epsilon^2,
\qquad
\delta' \approx3\delta^2.
$$

No trace-dependent branch alternation.

So I would now regard the intended algorithm as

$$
\boxed{
\text{TC2 for global spectral/rank convergence}
\quad\longrightarrow\quad
\text{McWeeny for the endgame}.
}
$$

TC2 should probably stop somewhere around \(R_I=10^{-3}\)–\(10^{-4}\), once trace/rank are firmly locked. It should **never enter that long \(10^{-5}\) sawtooth in production**.

---

# But there is something wrong with the McWeeny result too

This is the part I think the agent missed.

Look at McWeeny after the switch. It goes roughly

$$
10^{-5}\rightarrow8\times10^{-6}\rightarrow7\times10^{-6}\rightarrow\cdots
\rightarrow5\times10^{-6}
$$

over **more than 100 iterations**.

That is absolutely not what I would expect from an accurate McWeeny iteration.

If the projector error really were \(10^{-5}\), quadratic convergence should give approximately

$$
10^{-5}
\rightarrow
O(10^{-10})
$$

in one iteration, and then essentially machine precision.

Obviously the matrix residual isn't literally a single eigenvalue error, but the difference is much too large to explain the observed glacial convergence.

So:

> **McWeeny has removed the TC2 limit cycle, but each McWeeny step is apparently injecting a new error of order \(10^{-6}\)–\(10^{-5}\).**

We need to identify that error. I would not accept the \(5\times10^{-6}\) line as a fundamental f32-state floor.

---

## And I see a very plausible reason in the implementation you attached

Your new planned SpGEMM kernels have the ACC16/Kahan machinery.

But the McWeeny kernel itself is merely:

```c
Knew = 3*Q - 2*V;
```

with

$$
Q=KSK,\qquad V=KSKSK.
$$

So the precision is determined by **how Q and V are constructed**, not by `bsr4_mcweeny()`.

According to the agent's own report:

> “its extra products use the masked kernel (no ACC/Kahan)”

That means `Kahan+McW` being identical to `McW` is basically guaranteed!

It does **not** demonstrate that Kahan is irrelevant to McWeeny.

Your older masked Bsym SpGEMM uses a single accumulator:

```c
float sum = 0;

while (...) {
    sum = fma(..., sum);
    sum = fma(..., sum);
    sum = fma(..., sum);
    sum = fma(..., sum);
}
```

whereas your planned kernel has the 4/8/16/Kahan summation machinery.

So McWeeny may currently be doing its endgame through the **least accurate multiplication implementation in the whole solver**.

That is the first thing I would fix.

You already have the required plans:

$$
K S \rightarrow T,
$$

$$
T K \rightarrow Q,
$$

then exactly the same structural operations for

$$
Q S \rightarrow T_2,
$$

$$
T_2 K\rightarrow V.
$$

The same `plan_ks` / `plan_tk` type machinery can be reused.

Then test:

```text
McW + planned ACC16
McW + planned Kahan
```

for perhaps only 5–10 iterations.

If McWeeny is working properly, I would expect a dramatic drop within the first few steps, not 150 iterations.

---

# The statement “residual measurement is fine” is also too strong

The agent says:

> “Residual kernel is a strided partial + f64 host finish — the measurement is fine.”

Only the **final summation of the norm** is high accuracy.

To calculate

$$
R_I =
\frac{\|KSK-K\|}{\|K\|}
$$

you first need \(KSK\).

If \(KSK\) was generated by an f32 SpGEMM, then the residual contains the numerical error of that multiplication before the f64 reduction even begins.

A perfect f64 summation of inaccurate f32 terms does not tell you the true residual of the stored K.

So I still strongly want the one-off CPU-f64 check:

$$
K_{\rm f32}
\overset{\text{download}}{\longrightarrow}
(KSK-K)_{\rm CPU,f64}.
$$

No need for a production f64 solver. Just one diagnostic.

Suppose the GPU reports

$$
R_I^{GPU}=5\times10^{-6}.
$$

If CPU f64 says

$$
R_I^{64}=3\times10^{-7},
$$

the state is already good and we're mostly measuring SpGEMM noise.

If instead CPU f64 also gives

$$
R_I^{64}\approx5\times10^{-6},
$$

then the stored state itself contains that error.

That distinction is still unresolved.

---

# There is another suspicious clue: the dummy orbitals

I would definitely investigate this before reaching any conclusion about an f32 floor.

Your previous Hessian run ended with:

> `dummy |D_ii| sum=2.396e-5`

and aborted because dummy padded orbitals were getting occupied.

Notice the scale:

$$
2.4\times10^{-5}.
$$

That is remarkably close to the TC2 floor we have been arguing about.

Also, the earlier R10 log had 330 atoms but only 918 physical orbitals, so the plot title “330 Si” is misleading — this system contains hydrogen and therefore many padded dummy BSR4 lanes.

I would compute the idempotency residual decomposed into:

$$
R_I^{\rm physical}
$$

using only physical orbital lanes, and

$$
R_I^{\rm dummy}
$$

or simply the Frobenius contribution involving any dummy orbital.

If most of the \(10^{-5}\) floor lives in dummy lanes, then neither Kahan nor f64 is the fundamental problem. We have a **padding/algebra contamination bug**.

Given that the force code already detected dummy occupancy at essentially exactly this scale, this is now a high-priority diagnostic.

---

## The McWeeny trace drift is another clue

The lower panel shows that McWeeny develops approximately

$$
\frac{|\operatorname{Tr}P-N_{\rm occ}|}{N_{\rm occ}}
\sim10^{-6}.
$$

The agent treats that as “McWeeny doesn't conserve trace”.

Formally true, but near an already excellent projector the trace change of exact McWeeny should be second order.

Starting around \(R_I\sim10^{-5}\), I would expect much less than \(10^{-6}\) systematic drift unless numerical errors are being injected by the products.

So the trace plateau is another reason I think the present McWeeny multiplication is limiting the result.

Do **not** blindly rescale K every McWeeny step either. If

$$
K\rightarrow\alpha K
$$

to restore trace, that itself destroys exact idempotency by \(O(|\alpha-1|)\) and can establish a new floor.

For now just monitor trace and tolerate a tiny drift.

---

# There is an even cleaner production variant: McWeeny in P-space

Because you already implemented \(P=KS\), I think the final production form might be even cheaper.

Once TC2 has locked the occupation, you already have

$$
P=KS.
$$

Then McWeeny is simply

$$
Q=P^2,
$$

$$
R=QP=P^3,
$$

$$
\boxed{P_{\rm new}=3Q-2R}.
$$

That is only **two SpGEMMs per McWeeny step**, rather than the K-space form:

$$
KS,\quad(KS)K,\quad(KSK)S,\quad(KSKS)K
$$

which needs four.

And no metric algebra is inside the polynomial anymore.

After two or three P-McWeeny iterations:

$$
K=PZ
$$

once, for forces.

You already have the generic `plan_pp` infrastructure for P², so this is not a completely new solver.

I would probably first debug the existing K-McWeeny because it is easier to compare directly. But if it works, **P-space McWeeny is likely the better production endgame**.

---

## So I would interpret the table like this

| variant                                | what it actually tells us                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| ACC4/8/16                              | summation reassociation slightly changes the f32 error, but **does not cure TC2 endgame oscillation**         |
| Kahan                                  | f32 accumulation contributes perhaps a factor ~2, but is **not the main TC2 pathology**                       |
| McWeeny                                | **correct conceptual solution to TC2 sawtooth**                                                               |
| Kahan+McW identical                    | currently almost meaningless if McW products bypass the Kahan planned kernel                                  |
| McW slowly reaching \(5\times10^{-6}\) | **not satisfactory and not evidence of a fundamental f32 limit**; something injects error every McW iteration |

And I would give the agent this immediate order:

1. **Stop ACC studies.** Keep ACC16 only if its performance advantage is reproducible.
2. Run the one-shot **CPU-f64 residual of saved K**.
3. Decompose \(R_I\) into **physical and dummy-orbital contributions**.
4. Make McWeeny use the **planned SpGEMM path**, first ACC16 and then Kahan.
5. Run only ~5–10 McWeeny steps and plot each one.
6. If accurate McWeeny still decreases only linearly from \(10^{-5}\), there is an algebra/implementation bug; do not hide it by running 150 iterations.
7. Only after that decide whether any f64 endgame is needed.
8. If McWeeny works, consider the cheaper **P-space McWeeny** with two products/iteration.

The most important disagreement I have with the agent is therefore this sentence:

> “To actually reach ~1e-7 the iteration needs f64 in the state.”

**No — that is not established.** Proper McWeeny should converge much more aggressively than this. Before paying for f64, I would find out why your supposedly quadratically convergent endgame is crawling from \(10^{-5}\) to \(5\times10^{-6}\) over 100 iterations. That is the anomaly in this new plot.



---

# USER

the AI implemented what you said but perhaps badly TRS4 converge faster (less iterrations) to 1e-5 but never get significantly better.

Seems like there it no way aroun numerical accuracy 


Proceeding with the ordered list — F64-DIAG and DUMMY-DECOMP first (both cheap and decisive).

(Those were completed — the list now tracks remaining work.) Find how per-atom orbital counts are stored:

`atom_n_orb` is on the engine. Check `bsr_values_to_dense` signature, then write the diagnostic:

Add the call to the study script and rebuild:

Decisive result:

- **f64 true residual: 4.15e-5** — the stored state is ~2.3× *worse* than the f32 measurement showed (GPT's Case B: state at floor, not measurement noise).
- **dummy lanes: exactly 0.000e0** — no padding contamination; the floor is entirely physical.
- Trace exact: 459.000025.

So K can genuinely improve ~2 orders — the iterate noise is real and removable by a better map. Next: MCW-PLAN — the McWeeny products currently run on the masked kernel (no ACC/Kahan). `Q·S` and `(QS)·K` are structurally identical to `K·S` and `T·K` — the existing plans can be reused. Check the workspace structures/buffers:

The structures line up perfectly: `plan_ks` (M_K × S → M_TKS) applies to `Q·S`, `plan_tk` (M_TKS × M_K → M_K) applies to `U·K`. Even better — the current McWeeny truncates `Q·S` into an M_K buffer (`z_on_k`), which is wrong-shaped; using `t_ks` (M_TKS) is correct. Rewire the endgame branch:

Planned McWeeny fixed the trace drift (459.00003 vs 459.0004 — the M_K truncation of U was real) but R_I floor unchanged at 5.4e-6. Now planned + Kahan:

Planned + Kahan McWeeny: identical 5.35e-6 floor, f64 residual 1.42e-5. So GPT's masked-kernel hypothesis is also refuted — the glacial descent isn't product accuracy. 

There's a deeper explanation worth noting: **a polynomial of K can never rotate K's eigenbasis** — `K'=f(K)` preserves eigenvectors exactly. The residual floor is dominated by subspace-rotation error (off-diagonal occ↔unocc blocks), where the McWeeny linearization `E→P*E+EP*−2P*EP*` leaves E_ou/E_uo **unchanged** at linear order — marginal, contracting only through higher-order terms. That predicts exactly the observed glacial asymptote.

The last cheap variant is P-space McWeeny (`P'=3P²−2P³` via `plan_pp`). Check the P-purifier:

P-space purifier already exists with `plan_pp`, `q_p2`, `r_p4` buffers — P-McWeeny is a small addition. Check the P0 entry point:

Adding the same endgame switch + history recording to `tc2_purify_p` (P-space):

`ri_now` doesn't exist in the P variant — add it:

Make `purify_cold` dispatch on the purifier mode so the study covers P-space and TRS4 too:

TRS4 is also worth overlaying — check its loop for where to add hist recording:

Add history recording to TRS4 too:

P-space reaches R_I(P)=5.6e-6 but the **recovered K is 7e-5** — K=PZ recovery via approximate Z adds error. Now P+McWeeny and TRS4:

All variants collected. Regenerate the plot including P-space and TRS4:

The complete picture is now in. Update the docs with the results and the real mechanism, then report:

Done — the study is complete and the docs reflect it. Final plot: `@/home/prokop/git/dftbplus/debug/tc2_conv_r10.png`

**Results of the ordered list:**

| diagnostic | result |
|---|---|
| F64-DIAG | True K residual **4.1e-5** (TC2) / **1.4e-5** (K+McW) — f32 measurement *underestimates* 2.6×; state is genuinely at the floor |
| DUMMY-DECOMP | dummy lanes **exactly 0** — floor is 100% physical, no padding bug |
| MCW-PLAN | Products rerouted to planned kernels; **found+fixed a real bug**: U=Q·S was truncated onto M_K (now M_TKS) — trace drift 4.4e-7→7e-8. Floor unchanged → product accuracy isn't the limiter |
| P-space | P-TC2/TRS4 reach ~6e-6 on P, but **K=PZ recovery loses it** (7–8e-5). P-McWeeny drifts *upward* (non-normal iterate) — rejected |

**The mechanism (why nothing reaches 1e-7):** every polynomial purifier is *marginally stable in the occ↔unocc mixing direction* — linearizing at the target projector, E_ou satisfies L(E_ou)=E_ou exactly for both TC2 branches AND McWeeny. It contracts only through O(E²) coupling → the glacial tail. This is the same root cause as the failed warm-DM: **no polynomial of K can rotate the occupied subspace.**

So the options converge:

- **(a)** Accept the floor (~5e-6 measured / 1.4e-5 true) — physics cost already bounded at ~6 cm⁻¹ rms (fixq-vs-SCC), and return to loop-level work (batching, per-eval speed).
- **(b)** Implement the H-aware endgame — LNV/commutator descent minimizing ‖[K,H]‖. This is the *only* lever on E_ou and it's the same missing piece as G3 warm-DM — one design fixes both.

My recommendation: (b) is where both threads meet — a commutator-minimizing polish would fix the floor *and* enable the warm path. Want me to pursue it?


---





# Chat GPT 5.6 sol

Yes — this new result makes me much more willing to say that **there really is a numerical-precision floor in the current f32 representation**. TRS4 reaching essentially the same \(10^{-5}\)-class floor by a very different trajectory is strong evidence: changing the polynomial changes how quickly you arrive, but not where you end up.

However, I disagree with one important part of the agent’s explanation:

> “the remaining floor is occ↔unocc subspace-rotation error, and therefore H-aware commutator descent is the only fix”

That does **not** follow from these results, and mathematically it is suspicious.

For a projector \(P\), an infinitesimal occupied↔unoccupied rotation is a tangent motion along the manifold of projectors. A rotated projector can still satisfy

$$
P^2=P
$$

exactly. Linearizing the idempotency defect,

$$
(P+E)^2-(P+E)
\simeq PE+EP-E,
$$

an occ↔unocc block \(E_{ou}\) actually cancels at first order. So subspace rotation is exactly the kind of error that McWeeny does **not** correct — but it is also not what should produce a first-order idempotency residual.

In other words:

$$
\boxed{\text{stationarity error and idempotency error are different problems.}}
$$

An H-aware commutator/LNV update is absolutely needed for the **warm-DM problem**, because the old projector must rotate toward the eigenspace of the new displaced Hamiltonian. But it has not yet been demonstrated to be the cure for this cold-purification \(R_I\sim10^{-5}\) floor.

### What the new tests really establish

You now have several quite independent algorithms:

* TC2: sawtooth at the floor.
* TRS4: arrives faster, same floor.
* McWeeny: removes the sawtooth, but settles/crawls at roughly the same scale.
* ACC16: changes floor modestly.
* Kahan: changes floor only roughly another factor of two.
* Planned McWeeny / corrected intermediate support: fixes a real trace bug, but not the floor.
* CPU-f64 evaluation says the **stored f32 K itself** has a true residual \(1.4\times10^{-5}\) after McWeeny, not \(10^{-7}\).
* Dummy lanes contribute zero.

Taken together, I think the simplest hypothesis is now:

$$
\boxed{\text{the f32 matrix state + repeated f32 matrix algebra impose a }10^{-5}\text{-class floor for this large matrix.}}
$$

That is much more plausible now than it was two experiments ago.

And there is a very clean experiment that will settle it conclusively.

## Do one f64-state McWeeny diagnostic now

Take exactly the best stored f32 \(K\), convert it once to f64, use the same \(S\) values converted to f64, and perform:

$$
Q=KSK,
$$

$$
V=QSK,
$$

$$
K'=3Q-2V
$$

**entirely in f64**, including storing the new K in f64.

Do 3–5 iterations. No H, no SCC, no GPU optimization. CPU dense is perfectly fine for this diagnostic at 918 orbitals.

Plot the true f64

$$
R_I^{64}
$$

after each step.

There are only two interesting outcomes.

If you get something like

$$
1.4\times10^{-5}
\rightarrow10^{-9}
\rightarrow10^{-14},
$$

then the case is closed: **f32 precision is the floor**. No commutator descent is needed to explain cold purification.

If instead you get

$$
1.4\times10^{-5}
\rightarrow1.1\times10^{-5}
\rightarrow9\times10^{-6}\ldots
$$

then something much deeper is wrong with the algebra/matrix representation, because exact McWeeny should not behave like that on an ordinary near-projector.

I expect the first outcome.

---

### I would also measure \(R_H\) separately

You already have the stationarity diagnostic roughly of the form

$$
R_H=
\frac{\|HKS-SKH\|}{2\|HKS\|}.
$$

This is precisely the quantity that detects the occupied-subspace orientation relative to H.

So instead of trying to infer everything from \(R_I\), report both:

$$
R_I=\frac{\|KSK-K\|}{\|K\|},
\qquad
R_H=\frac{\|HKS-SKH\|}{\ldots}.
$$

Then the interpretation is clean:

| result                    | diagnosis                                         |
| ------------------------- | ------------------------------------------------- |
| \(R_I\) bad, \(R_H\) good | precision/idempotency problem                     |
| \(R_I\) good, \(R_H\) bad | wrong occupied subspace → H-aware rotation needed |
| both bad                  | both problems                                     |
| both good                 | stop iterating                                    |

This is especially important for the Hessian warm-start work.

---

## TRS4 may nevertheless be useful

The pink TRS4 curve is actually interesting. It gets from the initial state to the \(10^{-5}\) region much faster in terms of **iteration count** than TC2.

But don't compare iteration counts. Compare:

$$
\text{number of SpGEMMs}
$$

and actual GPU wall time.

If one TRS4 iteration costs two matrix products while TC2 costs one, then 12 TRS4 iterations versus 24 TC2 iterations may be a wash.

If it is still faster in wall time, then a sensible cold solver might simply be:

$$
\boxed{\text{TRS4 fast approach} \rightarrow \text{precision endgame}}
$$

rather than TC2 at all.

The important observation from the plot is that there is no reason to run **any** f32 polynomial purifier beyond approximately where TRS4 reaches its minimum. They all hit the same wall.

---

# If f64 is necessary, we still absolutely should not make the whole solver f64

This is the engineering point I would emphasize.

Suppose the diagnostic confirms that f32 bottoms out at \(10^{-5}\). That does **not** imply:

> convert sparse DFTB to double precision.

Instead use mixed precision in phases:

$$
\text{f32 fast purification}
\rightarrow
R_I\sim10^{-4}\text{--}10^{-5}
\rightarrow
\text{1--3 high-precision polish steps}.
$$

The expensive precision should only touch the last few matrix products.

On the RTX 3090 native fp64 is horrible, yes. But compare the costs correctly.

Current pathological solver:

$$
50\text{--}150
$$

f32 purification iterations.

Possible solver:

$$
10\text{--}20\ \text{f32 iterations}
+
1\text{--}3\ \text{f64 iterations}.
$$

Even if an fp64 product is 20–30× slower, that may or may not win — **measure it**. And on the eventually sparse degree ~64–100 matrices, an endgame double product is much cheaper than the current full-degree-330 diagnostic.

There are also intermediate possibilities if native f64 is too expensive:

* f32 storage + f64 accumulation;
* f64 endgame state but only on active physical blocks;
* double-single (`float2`) arithmetic for K;
* iterative refinement using a high-precision residual and f32 correction.

But I would not invent any of those until the simple f64-state experiment tells us what precision is actually required.

---

## One more experiment can separate storage error from arithmetic error

After the full-f64 diagnostic, if useful, do these two:

$$
\textbf{A: }\quad K_{\rm f32}\rightarrow
\text{f64 products}\rightarrow
K'_{\rm f32}
$$

versus

$$
\textbf{B: }\quad K_{\rm f64}\rightarrow
\text{f64 products}\rightarrow
K'_{\rm f64}.
$$

If A still stops near \(10^{-5}\) while B collapses, **f32 storage** itself is the limiting factor.

If both collapse dramatically, f32 storage is okay and the problem is matrix-product arithmetic.

That directly tells us what the cheapest production mixed-precision implementation should be.

---

### And for Hessians, don't automatically insist on \(R_I=10^{-7}\)

This is also important.

The quantity you actually care about is force noise:

$$
\delta F
$$

because Hessian noise is approximately

$$
\delta H \sim \frac{\delta F}{2h}.
$$

For \(h=0.02\) Å, measure forces on exactly the same geometry using projectors polished to, say,

$$
R_I=3\times10^{-5},
\quad
10^{-5},
\quad
10^{-6},
\quad
10^{-8}.
$$

Then compare

$$
\max|\Delta F|
$$

and frequency changes.

It may turn out that the f32 \(10^{-5}\) projector is already completely sufficient and the hour-long Hessian was caused almost entirely by **refusing to stop at that floor**, not by the floor itself.

That would be the best outcome.

---

So my instruction to the agent now would be:

> **Do not implement commutator/LNV yet as a response to the idempotency floor. First perform 3–5 full-f64 McWeeny iterations on the saved best K and report true f64 \(R_I\) after every step. Also report \(R_H\) separately. This distinguishes numerical precision from subspace stationarity unambiguously. If f64 McWeeny collapses the residual quadratically, accept that f32 is the cold-purifier floor and design a minimal mixed-precision polish. The H-aware commutator update remains important for the separate warm-displacement problem.**

I think you're now very close to knowing exactly what the limitation is. The TRS4 result does not say “purification fundamentally cannot do better”; it says **all of your f32 variants are converging to the same precision wall**.













====


---

# USER

New test. Analyse, suggest what to do next

Important correction from GPT-5.6 — my E_ou mechanism was wrong (subspace rotation produces *no* first-order idempotency residual; `(P+E)²−(P+E)` cancels E_ou). The plan now:

1. First update the docs (correcting my claim + new ordered list).
2. Then implement the decisive experiment: full-f64 McWeeny on saved best-K (CPU dense), reporting f64 R_I **and** R_H per step, plus the A/B storage-vs-arithmetic split.

Docs updated. Now implement the decisive diagnostic. First, look at the existing `sparse_ri_f64` (it already downloads K/S and does dense f64 products) and find an H_scc host-download path for R_H:

`ri_f64_diag` exists with the dense-f64 machinery. Now check how to download the device `h_scc` values for the R_H diagnostic:

Now let me check how the TC2 history CSV gets written and where `purify_now` lives, so the new diagnostic fits the existing pattern:

Now I'll write the diagnostic. First check whether rayon (or a host matmul helper) is available — dense 1320³ f64 needs to be fast enough to stay under the 30 s cap:

`ri_f64_diag` gives me the pattern. Check the rhai wrapper style and `ws.gpu()` accessor signature:

Add the `mcw_f64_diag` diagnostic after `ri_f64_diag`:

Now add the `matmul_f64_mt` helper — put it near other free functions in the file:

Now the rhai wrapper + registration:

Now build:

Build clean. Now look at the study script to wire the diagnostic after the McWeeny run (whose best-K is the stored state):

McWeeny endgame is `RUST_DFTB_TC2_MCW=1`. Now update the script — to stay under the 30 s cap I'll gate variants B/A on an env var so each invocation is bounded:

No env/args in rhai — simplest is both variants in the script with n_steps=4; each should be ~10 s threaded. Update the script:

Run the study with McWeeny endgame active (stored K = the best ~1.4e-5 state) and the f64 hist CSV:

**Decisive result — case closed:**

```
variant B (all-f64):  1.419e-5 → 4.00e-9 → 2.2e-15 → 8e-16   (quadratic collapse to machine eps)
variant A (f32 store, f64 products):  1.419e-5 → 2.0e-8 → 1.5e-8 → 1.3e-8  (stalls at f32 storage quantization)
```

This is exactly GPT's predicted outcome #1: **f32 precision is the floor** — the map is fine, McWeeny converges quadratically once arithmetic is clean. And the A/B split shows the stored-state floor (~1.4e-5) is *product-arithmetic* noise: with f64 products, f32 storage alone reaches 1.3e-8 — **~1000× below the current floor with just 1–2 polished products**. R_H=9.2e-2 is large but this H_scc only saw 4 SCC mixers (rms 5e-2) — subspace orientation can't be judged from this run.

Let me check the CSV and update the plot:

Add an optional third panel for the f64-MCW decisive test:

All CSVs are in /tmp. Regenerate the plot with the f64 panel:

The plot is decisive. Update the manifest/tasks with the measured results:

Merge the now-answered duplicate item:

Done. The decisive test answered the question GPT-5.6 posed — and it changes the plan slightly: **LNV/commutator descent is NOT the fix for this floor** (subspace rotation produces no first-order R_I — my earlier mechanism was wrong, corrected in the manifest). It's still needed for warm-DM, but the floor itself is pure f32 arithmetic.

**F64-MCW decisive test results** (`sparse_mcw_f64`, host f64 dense, stored best-K):

| step | B: all-f64 R_I | A: f32-store R_I |
|---|---|---|
| 0 | 1.42e-5 | 1.42e-5 |
| 1 | **4.0e-9** | 2.0e-8 |
| 2 | **2.2e-15** | 1.5e-8 |
| 4 | 7.9e-16 | 1.3e-8 |

REVIEW: `debug/tc2_conv_r10.png` (3rd panel) `@/home/prokop/git/dftbplus/debug/tc2_conv_r10.png`

**What this proves:**

1. **f32 precision IS the floor** — all-f64 McWeeny collapses quadratically to machine eps in 2 steps. The map and algebra are healthy; no bug.
2. **A/B split**: f32 storage alone supports ~1.3e-8 — the observed ~1e-5 floor is accumulated *product-arithmetic* noise, killed by ONE f64 product.
3. **Production implication**: `f32 purify to ~1e-5 → 1–2 high-precision-product McWeeny steps (keep f32 storage) → ~1e-8` — ~1000× improvement for the cost of a few endgame products. GPU-f64 or double-single (f32x2) accumulation in the planned kernel are the candidate implementations.
4. R_H=9.2e-2 here is meaningless for the subspace question — that H_scc had only 4 unconverged SCC mixers; needs a converged H_scc rerun before warm-DM work.

Docs updated: manifest §4.12.2 round-2 checklist + G6 tasks. New open items: MIXED-POLISH implementation, R_H on converged H_scc, TRS4-vs-TC2 cost, and the force-noise-vs-polish study (which may show 1e-5 already suffices for Hessians).

Next step candidates, your call: (a) implement the double-single/f64-product polish path, or (b) force-noise study first to decide if we even need it.

---

# Chat GPT 5.6 sol

This is now a genuinely decisive result. The third panel answers the precision question much more cleanly than all the ACC/Kahan/TRS4 experiments.

The key observation is:

$$
R_I:1.42\times10^{-5}
\overset{\rm all\ f64}{\longrightarrow}
4.0\times10^{-9}
\longrightarrow2\times10^{-15},
$$

whereas with **f32 storage but f64 products**

$$
1.42\times10^{-5}
\rightarrow2.0\times10^{-8}
\rightarrow1.3\times10^{-8}.
$$

So the diagnosis is:

$$
\boxed{\text{f32 storage is fine; ordinary f32 matrix products are the problem.}}
$$

That is actually an excellent outcome. We do **not** need an f64 density matrix, and we do not need to redesign the purification mathematics. One high-precision McWeeny step already buys almost three orders of magnitude.

There are two important caveats before implementing anything large.

First, this decisive experiment was deliberately done at **deg ≈ 330, essentially full support**. That isolates arithmetic very nicely, but it is not the solver we ultimately want. The next absolutely critical experiment is the same test on the real sparse pbc mask, around degree 64–100. There the result could instead be

$$
10^{-5}\rightarrow10^{-5}
$$

even in f64 because the **mask**, rather than arithmetic, has become the floor. We must distinguish these before spending effort on a mixed-precision GPU kernel.

Second, the large \(R_H\sim 0.09\) in this plot tells us nothing yet because, as the agent notes, this \(H_{\rm SCC}\) was not actually converged. Idempotency and Hamiltonian stationarity remain two separate issues.

## What I would do next

1. **Repeat the f64-McWeeny A/B experiment at the actual production sparsity.** Use pbc with the intended \(M_K\), e.g. degree ~95 and then perhaps ~64 if you are still pursuing that. This experiment should use exactly the same sparse mask in the CPU-f64 products, i.e. product → projection to \(M_K\) at every stage, so it measures the true sparse algebra rather than a dense reference. If it still gives \(10^{-5}\rightarrow10^{-8}\), mixed precision is unquestionably worth implementing. If it stalls around \(10^{-5}\), the remaining error is structural truncation and no amount of arithmetic precision will fix it.

2. **Before implementing a production GPU precision kernel, do the force-noise test with the existing CPU-f64 polish.** This is much more important than making \(R_I\) pretty. On one fully SCC-converged geometry, and then on perhaps 3–6 representative \(+h/-h\) displacements, compare forces from:

   * normal f32 state at \(R_I\sim10^{-5}\);
   * one f64-product/f32-store McWeeny polish, \(R_I\sim10^{-8}\);
   * optionally all-f64 polished reference.

   Compare

   $$
   \Delta F = F_{\rm polished}-F_{\rm f32}
   $$

   and, more importantly,

   $$
   \Delta H_i=
   -\frac{\Delta F(+h)-\Delta F(-h)}{2h}.
   $$

   Use one bulk Si, one surface Si, and one H if applicable. This tells us directly whether the \(10^{-5}\) projector floor matters for vibrations.

3. **If polishing matters, implement the simplest mixed-precision kernel first: f32 storage/input, f64 accumulation, f32 output.** Do not implement full-f64 matrices. Since every f32 number is represented exactly in f64 and a product of two f32 significands fits within f64 precision, this gets almost exactly the arithmetic used by your successful “A” diagnostic:

   ```c
   double s0=0, s1=0, s2=0, s3=0;
   s0 = fma((double)a0, (double)b0, s0);
   ...
   C[...] = (float)((s0+s1)+(s2+s3));
   ```

   Compile it as a separate **polish-only** planned SpGEMM kernel.

4. **Do not use that kernel throughout TC2/SCC.** The correct production scheme is roughly

   $$
   \text{fast f32 TRS4/TC2}
   \rightarrow R_I\sim10^{-4}\text{--}10^{-5}
   \rightarrow
   \boxed{\text{one mixed-precision McWeeny polish}}
   $$

   at the final electronic state. Your experiment says one polish may already be enough:

   $$
   1.4\times10^{-5}\rightarrow2\times10^{-8}.
   $$

   A second polish gives almost nothing because f32 storage then becomes the \(10^{-8}\) floor.

5. **Only if native fp64 accumulation is too slow on the 3090 should we implement float-float/double-single.** And in that case I would not use ordinary Kahan again. Kahan only compensates summation; the decisive test says we need a higher-precision **dot product**. A much better f32-only emulation is an error-free product plus a hi/lo accumulator:

   $$
   p=a b,\qquad
   p_{\rm lo}=\operatorname{fma}(a,b,-p),
   $$

   followed by `TwoSum`/double-single accumulation. Because the inputs are f32, that can recover almost all of the product information using fast fp32 FMA. That is much closer to your successful f64-product diagnostic than Kahan was.

The choice between native fp64 accumulation and float-float should be purely empirical. On an RTX 3090 fp64 throughput is poor, but this is only **one final polish**, and once you are back at degree ~95 the product is much cheaper anyway. I would absolutely benchmark the simple implementation before writing clever float-float code.

### TRS4 now has a clear role too

The pink TRS4 curve appears to reach the useful \(10^{-5}\) regime with fewer iterations. But compare **wall time / total SpGEMMs**, not iterations. We no longer care whether TRS4 itself can descend below \(10^{-5}\); it only needs to cheaply produce a sufficiently good seed for the mixed-precision McWeeny polish.

So the cold solver competition becomes:

$$
\text{TC2 + polish}
\quad\text{vs}\quad
\text{TRS4 + polish}.
$$

Whichever gets to \(10^{-4}\) or \(10^{-5}\) faster wins.

And I would now loosen the f32 purifier stopping criterion. There is no reason whatsoever to spend another 20 iterations trying to reach \(10^{-5}\) if one high-precision McWeeny step will clean \(10^{-4}\) to perhaps \(10^{-7}\)–\(10^{-8}\). Test exactly how early the polish can be switched on:

$$
R_I^{\rm switch}
=
10^{-2},10^{-3},10^{-4},10^{-5}.
$$

My guess is \(10^{-3}\) or \(10^{-4}\) may already be enough. If so, that saves more time than any SpGEMM micro-optimization.

---

The broader Hessian issue is therefore becoming nicely separated:

$$
\boxed{
\begin{array}{ll}
\text{cold idempotency floor} &\rightarrow \text{one mixed-precision McWeeny polish}\\[2mm]
\text{warm displaced geometry} &\rightarrow \text{H-aware subspace update}\\[2mm]
\text{SCC near minimum} &\rightarrow \text{reuse }q,K/P,Z,\ \text{few outer iterations}\\
\end{array}}
$$

Do **not** solve the first problem with LNV/commutator descent; the new experiment proves that was unnecessary. But we will still need an H-aware rotation/update for the Hessian warm start, because polishing \(K^2=K\) does not move it toward the eigenspace of a changed Hamiltonian.

My immediate choice would therefore be **not (a) or (b) exactly as the agent phrased it**. First run the f64 polish on the **degree~95 production mask**, then use that already-existing CPU polish for a **small force/Hessian-column noise study**. If the force difference is meaningful, implement the simple f32-input/f64-accumulator GPU polish. That sequence answers both “will it still work when truly sparse?” and “do we actually need it for vibrations?” before we write more GPU code.

---

# USER

1) I diagree here, we must first handle the accuracy properly, in the end we struggle get sprcity below 330 neighs, we will work on that once we can rely on the accuacy because otherwise we constantly bouce hor head agains the floor of f32 arithmetics limits, and I'm staring to get sick of your avoidance of properly adressing this!!!! why you avoid it? you seem to be biased again it for some reaosn.

2) what you mean by 
"One high-precision McWeeny step already buys almost three orders of magnitude."
how to implemen this? using f32 or f64? 
I'm getting lost here. I was thining what we want to do is some mixed arithmetic or kahan sum, or emulate f64 by two f32? Or what is the way how we can do these matrix products on our f32-gpu without loos of neigher accuracy nor performance? Explain it to me clearly


---

# Chat GPT 5.6 sol

Yes — I agree with point 1. I was being too conservative about avoiding higher precision because of the RTX 3090's poor native FP64 throughput. Your latest experiment settles this: **we should solve the arithmetic-accuracy problem properly first**, establish a reliable electronic solver, and only then return to aggressive sparsification. Otherwise every sparsity experiment is contaminated by an unrelated \(10^{-5}\) arithmetic floor.

And the good news is that the test tells us exactly what we need. We do **not** need to turn the whole solver into f64.

## What “one high-precision McWeeny step” actually means

Your decisive experiment was essentially:

$$
K_{\mathrm{f32}}
\quad\xrightarrow[\text{all products/intermediates high precision}]{\text{one McWeeny step}}\quad
K'_{\mathrm{f32}}.
$$

The input K is f32. The **final K is also stored as f32**. Only the arithmetic *inside the McWeeny step* is high precision.

And that gave:

$$
R_I:
1.42\times10^{-5}
\rightarrow
2.0\times10^{-8}
$$

after **one step**.

That is the crucial result. It proves that f32 storage of K is perfectly adequate: an f32 matrix can represent a projector with \(R_I\sim10^{-8}\). The \(10^{-5}\) floor comes from performing the sequence of matrix products in ordinary f32.

So the target production algorithm should be:

$$
\boxed{
\text{fast f32 purification}
\rightarrow
\text{ONE high-precision McWeeny polish}
\rightarrow
\text{f32 K}
}
$$

Maybe two polish steps if needed, but your experiment says one is already almost enough.

---

# What exactly has to be high precision?

This part is important. McWeeny in K-space is

$$
K'=3KSK-2KSKSK.
$$

So one step consists of roughly

$$
T=KS
$$

$$
Q=TK=KSK
$$

$$
U=QS
$$

$$
V=UK=KSKSK
$$

$$
K'=3Q-2V.
$$

It is **not enough** to do a Kahan sum inside each SpGEMM and then store `T`, `Q`, `U`, `V` as ordinary f32.

That repeatedly quantizes the intermediate matrices back to 24-bit precision.

This is likely an important reason your Kahan experiment only improved

$$
1.8\times10^{-5}\rightarrow9.7\times10^{-6}.
$$

To reproduce the successful diagnostic, the intermediate chain needs to retain extra precision:

```text
K32, S32
   │
   ├─ HP multiply ─→ T_HP
   │
   ├─ HP multiply ─→ Q_HP
   │
   ├─ HP multiply ─→ U_HP
   │
   └─ HP multiply ─→ V_HP
                         │
                  K32 = float(3 Q_HP - 2 V_HP)
```

Only **after the complete McWeeny polynomial** do we round back to f32.

That is fundamentally different from merely improving the summation inside an otherwise f32 pipeline.

---

# There are two sensible ways to implement HP arithmetic

I would implement both in this order.

### Option A — native `double` polish: simplest correctness implementation

Matrices remain f32 normally.

For the polish, create a couple of temporary double-precision BSR value buffers. Load the ordinary float matrices and cast on the fly:

```c
double a = (double)A[...];
double b = (double)B[...];
double sum = fma(a,b,sum);
```

Keep the intermediate results in `double`, not float.

You don't actually need four double scratch matrices. Two or three reusable buffers are enough. For example:

```text
T64 = K32 * S32
Q64 = T64 * K32

T64 = Q64 * S32          // reuse T64

final kernel:
    V = T64 * K32        // accumulate V in registers as double
    K32 = float(3*Q64 - 2*V)
```

So you only need roughly **two f64 sparse scratch matrices**.

This should reproduce the CPU diagnostic almost exactly:

$$
1.4\times10^{-5}\rightarrow O(10^{-8})
$$

in one polish step.

Yes, native FP64 throughput on the RTX 3090 is poor. But this is **not 50 f64 iterations**. It is four matrix multiplications, once.

I would implement this first because it gives us a clean, trustworthy solver immediately.

Then measure the cost.

---

# Option B — emulate ~f64 with two f32 values: probably the eventual fast solution

This is what I think you were getting at with “emulate f64 by two f32”.

Represent every high-precision intermediate scalar as

$$
x=x_{\rm hi}+x_{\rm lo},
$$

two ordinary floats.

This is usually called **double-single**, **float-float**, or compensated extended precision.

And GPUs like your 3090 are exactly where this can make sense: FP32 FMA is extremely fast while FP64 is slow.

The beautiful part is that for two f32 operands we can recover the multiplication error using FMA:

```c
float p  = a*b;
float pe = fma(a,b,-p);
```

Conceptually,

$$
ab=p+p_e.
$$

`p` contains the normal rounded f32 product and `pe` contains the missing low part.

Then instead of accumulating into one float

```text
sum += a*b
```

we accumulate the two pieces into a two-float accumulator

$$
(s_{\rm hi},s_{\rm lo})
$$

using error-free `TwoSum`/`FastTwoSum` operations.

Thus each output element is stored as:

```text
C_hi
C_lo
```

rather than a single float.

For the next multiplication, if A is `(hi,lo)` and B is ordinary float,

$$
A B=(A_{\rm hi}+A_{\rm lo})B,
$$

so accumulate both

$$
A_{\rm hi}B
$$

and

$$
A_{\rm lo}B
$$

into the extended accumulator.

Continue like that throughout the four McWeeny products.

Finally:

```c
Kout = (float)(Q_hi + Q_lo ...);
```

So the normal density matrix remains one f32 per element.

---

## This is much stronger than Kahan

There are three different ideas that got mixed together in our previous discussion:

| method                        | what it fixes                                             | your result                                               |
| ----------------------------- | --------------------------------------------------------- | --------------------------------------------------------- |
| ACC4/8/16                     | rearranges f32 additions / shorter chains                 | almost nothing                                            |
| Kahan                         | compensates loss during summation                         | ~2× improvement                                           |
| **float-float/double-single** | retains extra bits of **products + sums + intermediates** | should approximate your successful f64-product experiment |

Kahan is **not an emulated double**.

It still ultimately represents each intermediate matrix element as one float. It cannot carry all the information through

$$
KS\rightarrow KSK\rightarrow KSKS\rightarrow KSKSK.
$$

Double-single does.

That is why I think double-single is the right eventual GPU implementation if native FP64 polish is too expensive.

---

# How much precision do we actually need?

Not full 53-bit f64.

Your target is not

$$
R_I=10^{-15}.
$$

The experiment says f32 storage naturally bottoms out at approximately

$$
10^{-8}.
$$

That is already excellent.

So we need enough arithmetic precision that the McWeeny calculation doesn't introduce \(10^{-5}\) errors before rounding the final result to f32.

A double-single accumulator has roughly 40–48 useful bits depending on implementation. That should be **vastly more than necessary**.

We might even be able to simplify it substantially compared with a general-purpose IEEE double emulator.

We only need:

* multiplication of `float × float`;
* multiplication of `(hi,lo) × float`;
* extended summation of many terms;
* `3Q-2V`;
* no division, exp, sqrt, etc.

So this can be a very compact specialized kernel.

---

# And we do not need to pay this cost everywhere

This is the other essential point.

The final workflow should **not** become:

```text
everything double-single
```

It should be something like:

```text
TRS4 / TC2 in ordinary f32
     ↓
R_I ~ 1e-4 ... 1e-5
     ↓
ONE HP McWeeny step
     ↓
R_I ~ 1e-8
```

The high-precision arithmetic runs for perhaps **four SpGEMMs total**.

Compare that against what we've just been doing:

```text
30 useful f32 products
+ 50–120 completely useless f32 products dancing at the floor
```

So a slower HP kernel can actually make the entire solver **faster**.

That is the right way to think about performance.

---

# I would even test switching much earlier

McWeeny has quadratic convergence once K is sufficiently close.

You currently purify down to \(10^{-5}\) and then polish.

But perhaps we can do:

$$
R_I=10^{-3}
\overset{\rm HP\ McW}{\longrightarrow}
O(10^{-6})
\overset{\rm HP\ McW}{\longrightarrow}
O(10^{-8}).
$$

Then two HP McWeeny steps might replace twenty f32 endgame iterations.

So once HP-McW works, explicitly test switching at:

$$
10^{-2},\quad10^{-3},\quad10^{-4},\quad10^{-5}.
$$

I would not be surprised if

$$
\boxed{\text{TRS4 to }10^{-3} + 2\text{ HP McWeeny}}
$$

ends up faster than

$$
\text{TRS4 to }10^{-5} + 1\text{ HP McWeeny}.
$$

---

## Which implementation should we actually build?

I would give the agent this order:

**First implement the native-double polish correctly.** Not because I want the final code to depend on slow fp64, but because it is simple and gives us a trustworthy production-quality accuracy baseline.

Specifically:

```text
K, S remain f32.

Add:
    hp_T : f64 BSR values
    hp_Q : f64 BSR values

Implement planned mixed SpGEMMs:

    f32 × f32 → f64
    f64 × f32 → f64

Then:
    T64 = K32 S32
    Q64 = T64 K32
    T64 = Q64 S32

Final:
    for each output block:
        V64 = T64 K32        // double accumulation, need not store V
        K32 = float(3 Q64 - 2 V64)

Evaluate R_I with a high-precision diagnostic.
```

That is one complete HP-McWeeny step.

Do **not** insert f32 output between these four products.

Once that works on GPU and reproduces

$$
10^{-5}\rightarrow10^{-8},
$$

benchmark it.

Then implement the exact same dataflow using:

```text
hp_T_hi, hp_T_lo
hp_Q_hi, hp_Q_lo
```

instead of doubles.

Use f32 FMA-based `TwoProd` + `TwoSum` to implement the extended dot products.

Then compare:

```text
native FP64 polish
float-float polish
CPU-f64 reference
```

for both:

```text
R_I
energy
forces
wall time
```

I expect **float-float to be the final answer on a 3090**, but native double should be built first because it is the simplest way to guarantee we understand the arithmetic.

---

### One important distinction about “no loss of performance”

There is no way to get ~48-bit arithmetic for literally zero extra operations.

But we don't need zero overhead **per polish kernel**.

We need negligible overhead for the **whole electronic solve**.

If ordinary f32 takes, say,

```text
20 iterations × 6 ms = 120 ms
```

and instead we do

```text
12 f32 iterations × 6 ms = 72 ms
+ 2 HP steps × 15 ms = 30 ms
= 102 ms
```

then the more accurate solver is actually faster.

That is the metric I care about.

And after what we have just learned, **I would no longer accept a production purifier that ever spends tens of f32 iterations below \(R_I\sim10^{-4}\)**. Once it approaches the arithmetic floor, switch precision and finish decisively.

So yes: let's settle accuracy now. My preference is **f32 persistent matrices + temporary high-precision McWeeny arithmetic**, first native FP64 as the correctness baseline, then a two-float FP32 implementation to recover performance on the 3090.


---

# USER

this is what I worried. Doing 4 matrix multiplication in f64 means this will cost use 40x more than this iteration in f32, so consider that we typically converge in 20 iterations of f32 to 1e-5, and then we pay equavalent of 40x f32 iterations to go from 1e-5 to 1e-8 (if everything works), this basically makese it as slow as before where we were wasting 50iteration dancing on numerical floor (but now we actually go somewhre that is the difference)
for me this is still not acceptabl numecrical cost. I stronly plerefe the f32-f32 emulation of f64, although I'm not sure how well it works, or perhps we should get satisfied with 1e-5 floor. (which is simplest and fastes, but not sure how problamtic in production)

Anyway, write clearly the essntial kernel in the emulated f32-f32 and cler instruction how to integrate it in the whole algorithm...

ad"
RI​=10−3⟶HP McW​O(10−6)⟶HP McW​O(10−8).
"
you seem to ignore termendous cost of your HP iteration.
It is like that meme/joke parahprasing Interstelar movie - this little manuever will cost uss 50yers on Eath.
So here yor little HP-iteration would cost us 50 iterations in f32 

---

# chat GPT 5.6 sol


Yes — your Interstellar analogy is exactly right. I was undercounting the cost by talking about “one HP McWeeny step” as though it were one ordinary iteration. In K-space it is **four matrix multiplications**, and on an RTX 3090 four native-FP64 SpGEMMs could easily cost the equivalent of tens to >100 FP32 SpGEMMs. That is not an acceptable production solution.

So I would now make the design decision much sharper:

$$
\boxed{\textbf{Do not use native FP64 on the GPU for production.}}
$$

Use ordinary f32 for the whole solver, then use **float-float (`hi+lo`) arithmetic implemented entirely with FP32/FMA** for the final McWeeny polish. Native CPU/f64 remains only the correctness reference.

The important result from your diagnostic is that we do **not** need an f64 matrix state:

$$
K_{\rm f32}
\xrightarrow[\text{accurate products}]{\rm McWeeny}
K_{\rm f32}
$$

gave a true residual around \(10^{-8}\).

So all that is needed is to make the *temporary products* much more accurate than normal f32.

---

# What float-float means here

Represent one temporary high-precision number as

$$
x=x_{\rm hi}+x_{\rm lo}
$$

where both components are ordinary `float`.

This is sometimes called **double-single**, but I prefer `float-float` here because it is not IEEE double. It gives roughly 40–48 useful significand bits depending on the operations.

The crucial trick is that with FP32 FMA, the rounding error of an FP32 product can be recovered:

```c
float p  = a * b;
float pe = fma(a, b, -p);
```

Thus

$$
ab \simeq p+p_e
$$

with `p` the normal product and `pe` the missing low bits.

This is fundamentally stronger than Kahan.

Kahan says:

> “try not to lose small terms when adding many ordinary rounded products.”

Float-float says:

> “retain the missing bits of the products **and** carry them through all intermediate matrices.”

That latter point is why I expect this to reproduce your successful f64-product/f32-storage experiment much better than Kahan did.

---

## The essential arithmetic

I would use something close to this OpenCL code.

```c
// -----------------------------------------------------------------------------
// float-float number x = hi + lo
// -----------------------------------------------------------------------------

typedef float2 ff2;       // .x = hi, .y = lo


// Error-free transform of a+b:
//
//     a+b = s + e
//
// assuming normal IEEE fp32 operations and no fast-math reassociation.
inline ff2 two_sum_f32(float a, float b)
{
    float s  = a + b;
    float bb = s - a;
    float e  = (a - (s - bb)) + (b - bb);
    return (ff2)(s, e);
}


// Add two float-float values.
inline ff2 ff_add(ff2 a, ff2 b)
{
    ff2 s = two_sum_f32(a.x, b.x);
    ff2 t = two_sum_f32(a.y, b.y);

    ff2 u = two_sum_f32(s.x, s.y + t.x);
    ff2 v = two_sum_f32(u.x, u.y + t.y);

    return v;
}


// Exact-ish decomposition of float * float.
//
// For binary32 operands, FMA recovers the low product bits.
inline ff2 ff_mul_f32_f32(float a, float b)
{
    float hi = a * b;
    float lo = fma(a, b, -hi);

    return (ff2)(hi, lo);
}


// Multiply a float-float number by an ordinary float.
//
// (ahi+alo)*b
inline ff2 ff_mul_ff_f32(ff2 a, float b)
{
    float hi = a.x * b;

    // Error of hi product + contribution of low component.
    float lo = fma(a.x, b, -hi);
    lo = fma(a.y, b, lo);

    return two_sum_f32(hi, lo);
}


inline ff2 ff_madd_f32_f32(ff2 sum, float a, float b)
{
    return ff_add(sum, ff_mul_f32_f32(a, b));
}


inline ff2 ff_madd_ff_f32(ff2 sum, ff2 a, float b)
{
    return ff_add(sum, ff_mul_ff_f32(a, b));
}
```

There is no `double` anywhere.

There is also an extremely important compiler requirement:

```text
NO -cl-fast-relaxed-math
NO unsafe reassociation
```

for this kernel.

Those transformations can destroy `TwoSum` and `fma(a,b,-hi)` error recovery.

If your normal sparse kernels are built with fast-math, compile these polish kernels in a separate OpenCL `Program` with strict FP semantics.

---

# The SpGEMM kernel

You don't need general float-float × float-float multiplication.

That is another nice simplification.

For one McWeeny polish,

$$
K' =3KSK-2KSKSK,
$$

the right operand is always either the original f32 \(K\) or f32 \(S\).

The sequence is:

$$
T=KS,
$$

$$
Q=TK,
$$

$$
U=QS,
$$

$$
V=UK.
$$

Therefore:

```text
T : float-float = f32        × f32
Q : float-float = float-float × f32
U : float-float = float-float × f32
V : float-float = float-float × f32
```

We never need `ff × ff`.

That saves a lot of work.

The inner part of your existing planned Bsym kernel becomes essentially:

```c
ff2 s0 = (ff2)(0.0f, 0.0f);
ff2 s1 = (ff2)(0.0f, 0.0f);
ff2 s2 = (ff2)(0.0f, 0.0f);
ff2 s3 = (ff2)(0.0f, 0.0f);

for(uint t=t0; t<t1; ++t)
{
    uint ia = plan_a_idx[t];
    uint bb = plan_b_idx[t];

    __local const float* Ahi =
        lA_hi + ia*16;

    __local const float* Alo =
        lA_lo + ia*16;          // absent/zero for first KS product

    __global const float* B =
        Bval + bb*16;

    ff2 a0 = (ff2)(Ahi[4*r+0], Alo[4*r+0]);
    ff2 a1 = (ff2)(Ahi[4*r+1], Alo[4*r+1]);
    ff2 a2 = (ff2)(Ahi[4*r+2], Alo[4*r+2]);
    ff2 a3 = (ff2)(Ahi[4*r+3], Alo[4*r+3]);

    s0 = ff_madd_ff_f32(s0, a0, B[4*c+0]);
    s1 = ff_madd_ff_f32(s1, a1, B[4*c+1]);
    s2 = ff_madd_ff_f32(s2, a2, B[4*c+2]);
    s3 = ff_madd_ff_f32(s3, a3, B[4*c+3]);
}

ff2 x01 = ff_add(s0, s1);
ff2 x23 = ff_add(s2, s3);
ff2 x   = ff_add(x01, x23);

C_hi[cb*16 + lane] = x.x;
C_lo[cb*16 + lane] = x.y;
```

For the first multiplication \(K S\), use a specialized `f32×f32 -> ff` kernel so there is no pointless loading of an `A_lo=0` buffer.

So I would have just two generic building blocks:

```text
spgemm_ff_from_f32()
    f32 × f32 → (hi,lo)

spgemm_ff_from_ff()
    (hi,lo) × f32 → (hi,lo)
```

Both reuse your existing symbolic plans.

---

# The complete McWeeny polish

The dataflow should be exactly this:

```text
                        ordinary persistent matrices

                       K32              S32
                        │                │
                        └──────┬─────────┘
                               │
                f32 × f32 -> float-float
                               │
                         T_hi, T_lo        M_TKS
                               │
                               │ × K32
                               ▼
                         Q_hi, Q_lo        M_K
                               │
                               │ × S32
                               ▼
                         T_hi, T_lo        M_TKS
                         (reuse buffers)
                               │
                               │ × K32
                               ▼
                         V_hi, V_lo
                         (registers only)
                               │
                               │
                   Knew32 = round(
                       3 Q_ff - 2 V_ff
                            )
```

And crucially **do not store V**.

Fuse the last `U*K` product with the McWeeny combination.

So you need only four temporary float arrays:

```text
T_hi    on M_TKS
T_lo    on M_TKS

Q_hi    on M_K
Q_lo    on M_K
```

No f64 buffers.

No V buffers.

Memory overhead is modest.

---

## The final fused kernel

At the end of `U*K`, instead of writing V:

```c
ff2 V = ff_add(
    ff_add(s0,s1),
    ff_add(s2,s3)
);

ff2 Q = (ff2)(
    Q_hi[cb*16 + lane],
    Q_lo[cb*16 + lane]
);

// 3Q
ff2 q3 = ff_mul_ff_f32(Q,  3.0f);

// -2V
ff2 vm2 = ff_mul_ff_f32(V, -2.0f);

// McWeeny result still in float-float.
ff2 knew = ff_add(q3, vm2);

// ROUND ONCE, HERE.
Knew[cb*16 + lane] = knew.x + knew.y;
```

Then swap `K` and `Knew`.

This final conversion to f32 is intentional.

Your diagnostic proved that f32 storage itself supports roughly

$$
R_I\sim10^{-8}.
$$

---

# How to integrate this into the solver

I would give the coding agent these steps, in this order:

1. Keep the existing TC2/TRS4 path fully f32. Do **not** add Kahan/ACC16 complexity to it for accuracy. ACC16 can remain if it is genuinely faster.

2. Add four persistent polish scratch buffers:

   ```text
   ff_t_hi, ff_t_lo       // M_TKS
   ff_q_hi, ff_q_lo       // M_K
   ```

3. Add strict-FP planned kernels:

   ```text
   plan_bsym_f32_f32_to_ff
   plan_bsym_ff_f32_to_ff
   plan_bsym_ff_f32_mcweeny_final
   ```

   The last one performs `U*K`, combines with saved `Q`, and writes ordinary f32 `Knew`.

4. Reuse the existing symbolic multiplication plans:

   ```text
   plan_ks : K × S -> M_TKS
   plan_tk : T × K -> M_K
   ```

   So:

   ```text
   Tff = K32 * S32          plan_ks
   Qff = Tff * K32          plan_tk
   Tff = Qff * S32          plan_ks
   Knew32 = 3Qff-2(Tff*K32) plan_tk
   ```

5. Run this **once**, not iteratively, when ordinary f32 purification is already in the projector basin.

6. Initial switch condition should be conservative:

   $$
   R_I^{f32}<10^{-4}
   $$

   and trace locked.
   Do not waste time trying to hit \(10^{-5}\) with f32.

7. Validate the result with CPU-f64 only in debug mode. The acceptance gate is:

   ```text
   R10:
       f32 input RI ~1e-4...1e-5
       one FF32-McWeeny
       true CPU-f64 RI <= ~1e-7, ideally ~1e-8
   ```

   If it fails this test, the float-float kernel is wrong.

8. Once validated, the production path does **not** need a CPU/f64 residual after every polish. One deterministic polish after a calibrated entry threshold is enough.

This is the algorithm I would implement before touching sparsity again.

---

# How expensive will float-float be?

Not free. But it should be nowhere near native FP64.

One normal scalar accumulation is approximately:

```c
sum = fma(a,b,sum);
```

The robust float-float version uses several FP32 adds plus 1–2 FP32 FMAs per term.

So compute cost may perhaps be something like **3–8× a normal FP32 dot product**, depending strongly on instruction scheduling/register pressure.

But on the RTX 3090 these are all high-throughput FP32 instructions. We are not hitting the crippled FP64 units.

And a McWeeny polish is four SpGEMMs versus two per K-TC2 iteration.

So very crudely, if float-float costs 4× normal SpGEMM:

$$
4\ {\rm HP\ products}\times4
=
16\ {\rm f32\ products}
$$

which corresponds to about

$$
8\ {\rm TC2\ iterations}.
$$

That is the scale I find plausible and acceptable.

Not:

$$
50\text{--}100\ {\rm TC2\ iterations}
$$

as native FP64 could effectively cost.

And if it replaces the current endgame from, say, iteration 20–30 onward, total runtime may actually go **down**.

---

## We should also optimize the float-float kernel after correctness

The code above is deliberately robust, not minimal.

Once it reproduces the \(10^{-8}\) CPU reference, profile it.

If it is too expensive, the next optimization is **not going back to Kahan**. It is a cheaper compensated dot-product implementation.

Instead of fully renormalizing `(hi,lo)` after every contribution, accumulate:

```text
main sum
rounding-error sum
```

using `TwoProd` + `TwoSum`, and renormalize only periodically/end-of-dot-product.

That can substantially reduce operations.

But I would only optimize that after the full float-float implementation establishes the correct reference behavior.

---

# Should we perhaps simply accept \(10^{-5}\)?

I would keep that as a runtime option:

```text
fast/screening:
    f32 only
    stop at numerical floor ~1e-5

accurate/forces/vibrations:
    f32 purification
    + one float-float McWeeny polish
```

But I agree with you that **we should first make the accurate mode work**.

Right now we've spent too much time asking whether a sparsity change caused a \(10^{-5}\) error when the arithmetic itself already contributes \(10^{-5}\). That makes debugging ambiguous.

Once we have:

$$
\boxed{R_I\sim10^{-8}\ {\rm reproducibly}}
$$

with a reasonably cheap FP32-only polish, we finally have a clean numerical baseline from which to attack locality and degree again.

And yes: **no native FP64 “little maneuver that costs 50 years on Earth.”** The next implementation I would ask the agent for is specifically the `float2(hi,lo)` planned McWeeny path above.







---

# USER

OK we implemented something, but not sure it is optimal 

---

OK, this looks good but It does not look like practical scheme for running the actuall simuations. We must indenitfy (or anticipate) we hit the numerical floor of f32 TC2 and do few (max 5) FF32 FF32 McWeeny steps 

perhaps we can pu there fixed budget, that the f32 TC2 will be always stoped after max 30 iters (but may be stoped sooner if we identify it hit expected accuracy 1e-5 or 1e-6) and then depending on decision if we run HI-accuracy calculation or not we can run up to 5 FF32 McWeeny step. We are not goding to dance on the floor anymore, th performance is more improtant than squeezing aboslute numerical floor.

Also we should think how we can now optimize the FF32 kernel further to maybe puss it below 3x cost of f32 TC2, perhaomance is the priority!

do not code now analyse

---

# SWE-2

from my coding agent:

Let me look at the purify loop's printing/history/timing infrastructure and the env-flag reads:

Add a `t_ms` column to the TC2 hist CSV (call-start-relative wall clock):

Now the lean benchmark script:

Run the two benchmark passes — pure f32 vs FF-endgame, each with device-time profiling and per-iter history:

f32 baseline done: plateau at iter 29, R_I=1.78e-7 (device)/2.7e-7 (true), 14.6ms/iter. FF steps show 46ms — but that's mostly enqueue time (no sync in `mcw_ff`). For honest device cost, add a 10-step steady-state block where queue back-pressure forces real device time:

The 10-step block still only measures enqueue time if the queue never syncs. Check for an existing `finish`/`sync` on the sparse GPU handle:

Now run all three benchmark passes:

FF step ≈ **44 ms vs 14.5 ms per f32 iteration — only ~3×**, not 10×. Now the in-loop endgame run:

Important finding in the in-loop run: after `ff_done` hits the cap, f32 TC2 resumes and **re-pollutes** the polished state (best ended at 1.33e-7, not ~3e-8). The endgame must switch to FF *and stop*, not return to f32. Fix the loop to exit after the FF cap:

`tr_eff` is stale after the FF update — measure the trace of the returned K honestly:

Now the in-loop endgame is correct: **true R_I = 1.81e-8 after purify** (vs 2.7e-7 f32-only) — 15× better, at +0.24s total. Update the plot script for the t_ms column, FF-iter marking, and the FFSTEP series:

The plot tells the story well. Small refinements: add the f32-storage floor reference line and note r_k in the title:

One more comparison for the cost/benefit question — the f32 McWeeny endgame (`TC2_MCW=1`, ~same cost as a TC2 iter):

All three runs done. f32-McWeeny alone only reaches the f32 product floor (~1.4e-7 device / 2.4e-7 true) — FF is what breaks through. Regenerate the plot with all three curves:

Now also record the production-config (r_k=20) numbers for the report — that's where the representation floor shows:

r_k=20 result: the FF endgame lands at true R_I=3.1e-5 — the M_K storage tail dominates (as predicted by the 5.3e-6 tail measurement). Get the f32-only r_k=20 baseline for the comparison plot:

The title is hardcoded to r_k=40 — make it a parameter:

Both plots done. Now the report — check its current state:

Now the comprehensive report section — check the file end:

All done. Final state of the work:

## Summary

**FF32-POLISH (emulated double-f32 McWeeny endgame) — implemented, verified, benchmarked.**

### What was measured

| | f32 TC2 iter | FF32 McWeeny step |
|---|---|---|
| wall time | 14.5 ms | **~44 ms ≈ 3.0 f32 iters** (not 10×) |

**r_k=40 Å** (M_K covers whole cluster — zero truncation):
- f32 purify floor: true R_I = 2.7e-7
- +1 FF step → 3.96e-8; +2 → 2.92e-8; in-loop endgame (6 steps, terminal) → **1.81e-8**
- ≈ the f32-storage fixed point ~1.4e-8. Cost: +0.26 s on a 0.44 s purify → **15× residual for +55% wall**

**r_k=20 Å** (production mask): FF lands ~3e-5 — the M_K storage tail (measured 5.3e-6/update) dominates. Arithmetic can't beat mask; `r_k` is the accuracy knob.

### The last bug (interesting)

The fused product+combine kernel produced 1.7e-7 noise while every intermediate verified ≤1e-13 — the NVIDIA OpenCL compiler reassociated the TwoSum chain *in that kernel only*. `-cl-fp32-contract=off` is unsupported on this driver. Fixed by splitting: V through the verified product kernel + an elementwise combine written as a single explicit-`fma` chain (`fma(3,Qh, fma(-2,Vh, fma(3,Ql,-2·Vl)))`) — the lo parts are ~1e-8 linear corrections, so that's ~1 ulp and contraction-immune. GPU output is now bitwise-identical to the Rust emulation.

Also fixed: the in-loop FF endgame now **stops** after the step cap — returning to f32 TC2 was re-polluting the polished state to ~1e-7.

### Deliverables

- `debug/tc2_ff_endgame_rk40.png` / `rk20.png` — log-scale R_I vs iter and vs wall-time, f64-verified FF series, in-loop FF iters marked (◇)
- `bench_ff_endgame.rhai` (lean timing harness), `sparse_sync` rhai call, `t_ms` column in the TC2 hist CSV, `prof_tick`s on ff.* stages, debug prints gated behind `algebra_verbose`
- Report §15.14 written; manifest + tasks checkboxes updated with measured numbers

**Verdict on worth-it:** ~3× per step, 4–6 steps needed, applied only at the endgame → worth it where the mask supports it. On R10 that means r_k≈40 Å; for truly sparse big systems the M_K tail sets the floor regardless of arithmetic — that part of the story is now measured and documented rather than assumed.











---

# Chat GPT 5.6 sol

Now that I can see the actual implementation, I think the basic FF32 machinery is good, but the **control flow around it is still research-harness logic rather than the production algorithm you want**.

The most important observation is actually encouraging:

$$
t_{\rm TC2}^{f32}\approx14.5\ {\rm ms}
$$

contains **2 SpGEMMs**, whereas

$$
t_{\rm McW}^{FF}\approx44\ {\rm ms}
$$

contains **4 FF SpGEMMs**.

So per matrix multiplication,

$$
\frac{44/4}{14.5/2}\approx1.52.
$$

That means your float-float product is only about **1.5× the cost of an ordinary f32 product**. This is much better than “FF costs 3×”. Most of the 3× is simply because one K-space McWeeny step mathematically contains twice as many products as one TC2 iteration.

Therefore the absolute lower bound, without changing the algorithm, is already roughly

$$
2\times t_{\rm TC2}.
$$

Getting FF-McW from 3.0× to perhaps **2.3–2.5× TC2** looks realistic. Getting it to 1× is not: that requires reducing the number of matrix products, not just optimizing the FF kernel.

## The production algorithm should become two completely separate phases

I would **not** leave FF inside the TC2 `for iter` loop as it currently is.

Right now after one FF step, the next outer iteration starts again by doing ordinary f32

$$
T=KS,\qquad Q=KSK,
$$

plus trace/residual handling, before reaching the next FF step. The code then eventually stops when `ff_done == ff_max`. 

That doesn't corrupt K—the f32 result isn't applied before the next FF step—but it wastes essentially **one complete f32 TC2 diagnostic iteration between FF steps**.

So conceptually I want:

```text
PHASE 1: ordinary f32 TC2
------------------------
while iter < TC2_BUDGET:
    do one TC2 iteration

    if ordinary f32 target reached:
        FAST mode -> return

    if trace locked && R_I < FF_ENTRY:
        restore best valid K
        break

if not safely in projector basin:
    fail / robust fallback
    (do NOT blindly polish garbage)


PHASE 2: terminal FF32 McWeeny
------------------------------
for i = 0 .. N_FF-1:
    McWeeny_FF(K)

rebuild KS once
measure final trace once
return
```

No TC2 ever runs again after the precision transition.

Your code already recognizes that returning to f32 re-pollutes the polished state.  I would simply take the next logical step and make FF a distinct terminal phase rather than an alternate branch inside TC2.

## I agree with a hard f32 budget around 30

With one qualification: **30 means “stop spending f32 work”, not “whatever state exists at iteration 30 is acceptable.”**

A sensible production contract would be:

```text
hard f32 TC2 budget:       30
FF entry hard ceiling:     R_I <= 1e-3
preferred FF switch:       ~1e-4   (calibrate)
trace must be locked:      yes

fast mode:
    no FF
    return best valid f32 state

accurate mode:
    1–2 FF McWeeny steps normally
    hard maximum 5
```

If after 30 iterations you still have

```text
trace not locked
or
R_I >> 1e-3
```

then this is not an arithmetic-floor problem. Do **not** hide it by throwing five FF steps at it. Fail loudly or invoke an explicit robust/reference path.

The current code still contains the old windowed stagnation machinery and the `10×best` oscillation detector.   Those become much less important once the main algorithm has a strict budget and a terminal precision phase.

### I would not make five FF steps the normal setting

Your full-support measurements already say:

$$
R_I^{\rm true}
\simeq2.7\times10^{-7}
$$

for f32 alone,

then approximately

$$
4\times10^{-8}
$$

after one FF step, and roughly

$$
2\times10^{-8}
$$

after another.

So:

* **1 FF step** = normal accurate mode.
* **2 FF steps** = high-accuracy production.
* **5** = hard safety/debug ceiling.

Anything after step 2 is mostly polishing the f32 storage floor.

The code presently defaults to 2, which is reasonable. 

---

# The switch point should be chosen from cost, not aesthetics

This is where the measured 3× cost becomes useful.

One FF step costs roughly the same as **three ordinary TC2 iterations**.

Therefore if one FF step starting from \(R_I=10^{-4}\) already reaches the desired final state, it is pointless to spend more than ~3 f32 iterations getting from \(10^{-4}\) to \(10^{-6}\).

I would run one very specific calibration sweep:

$$
R_{\rm switch}
=
10^{-2},\ 3\times10^{-3},\
10^{-3},\ 3\times10^{-4},\
10^{-4},\ 3\times10^{-5}.
$$

For each:

```text
TC2 until R_switch
+ exactly ONE FF step
→ true f64 diagnostic R_I
```

Then choose the **earliest** switch that reliably lands below your production target.

If

$$
10^{-3}
\xrightarrow{\rm one\ FF}
<10^{-7},
$$

great: switch at \(10^{-3}\).

If it needs \(10^{-4}\), use \(10^{-4}\).

There is no reason to drive TC2 into its floor first.

---

# There are also three concrete issues in `sparse_system.rs`

### 1. Environment variables are read in the hot loop

`FF_STEPS`, `FF_SWITCH`, `FF`, `MCW`, and the stagnation setting are looked up with `std::env::var()` during purification.  

This is not your 14 ms bottleneck, but production code shouldn't do it.

Resolve the complete purification policy once before entering the loop:

```text
mode
tc2_budget
ff_enabled
ff_switch
ff_steps
```

and then run a simple state machine.

### 2. The returned `last_r_i` after FF is stale

After `mcweeny_ff_step()`, the code eventually prints/returns:

```text
device R_I≈{last_r_i}
```

but `last_r_i` was measured **before the FF update**. 

So that number has no well-defined meaning for the returned state.

Either:

* measure a post-FF residual once, or
* explicitly return “not measured” plus a state/status indicating FF polish.

Do not attach an old residual to the polished K.

### 3. `NumericalFloor` is the wrong semantic status for successful FF polish

If accurate mode intentionally performs FF polishing and satisfies its contract, that isn't “we failed to converge and hit a numerical floor”.

I'd use something like:

```text
ConvergedF32
PolishedFF
NumericalFloor
Failed
```

or put `precision_mode` separately in the result.

That will matter later when energies/forces/Hessians decide whether a state is acceptable.

---

# The kernel itself: I think there is real room from 3× → ~2.5×

The attached FF kernel is sensible. It uses `TwoProd` via FMA and `TwoSum`, carries `(hi,lo)` throughout, and only rounds after the McWeeny combination.

But I see one particularly expensive design choice.

## Optimization 1 — specialize the first `f32 × f32 → FF` product

Right now the same kernel handles both:

```text
f32 × f32
FF  × f32
```

via runtime `a_has_lo`.

And the kernel **always allocates both**

```c
__local float lA_hi[MAX_LEFT_BLOCKS*16];
__local float lA_lo[MAX_LEFT_BLOCKS*16];
```

even for the first product where `A_lo` does not exist.

This matters tremendously on your GPU.

For degree ~330:

$$
330\times16\times4
\approx21.1\,{\rm KiB}
$$

for one row.

Hi + lo:

$$
\approx42.2\,{\rm KiB}.
$$

Your 3090 gives 48 KiB local memory to the workgroup/SM configuration you've been using.

So the generic FF kernel nearly fills local memory and basically forces **one WG resident**, whereas the first `K32*S32` product needs only the hi row and could plausibly allow **two WGs resident**.

I would make two physically separate kernels:

```text
spgemm_f32_f32_to_ff
    cache A_hi only
    no a_has_lo
    no A_lo
    no alo*b operation

spgemm_ff_f32_to_ff
    cache hi+lo
```

This should be the first performance experiment.

It improves:

* local-memory occupancy,
* instruction count,
* branch elimination,
* memory traffic.

And it affects one of the four FF products.

---

## Optimization 2 — remove the runtime `a_has_lo` branch entirely

Even in the FF-input kernel, make lo support compile-time/static.

Your current accumulator contains conceptually:

```c
if (has_lo)
    pe = fma(alo,b,pe);
```

Uniform branching isn't disastrous, but it makes the compiler carry one generic kernel through two quite different cases.

Two kernels are cleaner and should generate better machine code.

---

## Optimization 3 — try **hi in local, lo from global/L1**

For FF×f32, the hi+lo cache is what consumes ~42 KiB.

Try an A/B kernel with:

```text
A_hi → local memory
A_lo → global load via plan_a_idx
```

This halves local-memory use back toward ~21 KiB and may allow twice the workgroup occupancy.

Yes, you add global loads for the low part. But:

* `A_lo` is read-only,
* access is structured,
* the same rows should get cache reuse,
* the 3090 has substantial L1/L2 bandwidth,
* you may gain 2× occupancy.

This is exactly the kind of GPU trade that should be benchmarked, not decided theoretically.

If it wins, it could substantially reduce the other **three** FF products.

---

## Optimization 4 — split the serial FF accumulator

You already have four FF accumulators corresponding to the four orbital indices \(m=0..3\).

But each `(hi,lo)` accumulator still has a long serial `TwoSum` dependency over every plan term.

Try two FF accumulators per \(m\):

```text
h0a,l0a   h0b,l0b
...
```

and alternate plan terms.

At the end:

```text
ff_add(acc_a, acc_b)
```

This is analogous to why ACC16 helped your ordinary f32 kernel: expose instruction-level parallelism.

The cost is register pressure, so test:

```text
FF1 accumulator
FF2 alternating accumulators
```

Don't go immediately to 4-way.

---

## Optimization 5 — revisit fused V + final combine, but only after the above

The Rust code comment says V is “fused into final combine / never stored”, but the actual current path does store:

```text
V_hi -> a_zhz
V_lo -> ff_v_lo
```

and then launches `mcw_ff_combine`. 

I understand why: the previous fused version triggered compiler reassociation and destroyed the compensation.

Now that the final expression has been reduced to a simple explicit FMA chain, I would eventually retry:

```text
last FF product
→ keep Vff in registers
→ read Q_hi/Q_lo
→ explicit FMA combine
→ write Knew
```

and verify **bitwise or against the host FF emulator**.

If the compiler breaks it again, abandon it. This is a smaller optimization than occupancy anyway.

---

# An important correction to the history of the “f32 floor”

The current code now uses the **true symbolic support**

$$
M_{TKS}=\mathrm{supp}(M_K M_{HS})
$$

instead of truncating \(KS\) immediately back to \(M_K\). 

And your code comments now explicitly say the old shortcut dropped roughly \(7\times10^{-6}\) worth of product error and that the famous \(1.4\times10^{-5}\) “f32 floor” was substantially an **intermediate-mask truncation floor**, not pure f32 arithmetic. 

This reconciles the new plots:

* with effectively full support, plain f32 gets to a few \(10^{-7}\);
* FF gets to a few \(10^{-8}\);
* with \(r_K=20\) Å, both hit the **mask** floor around \(10^{-5}\).

So we have finally separated the two effects properly.

I would still finish the accuracy machinery first, as you asked. But we should retain this distinction:

$$
R_{\rm observed}
\approx
\max(
R_{\rm arithmetic},
R_{\rm mask}
).
$$

The FF mode gives us a way to push \(R_{\rm arithmetic}\) low enough that later sparsity experiments measure \(R_{\rm mask}\) cleanly.

---

## My proposed final production policy

I would freeze the design around this:

```text
FAST
----
TC2 f32 only
max 30 iterations
stop earlier on normal convergence / reliable floor
return best valid K


ACCURATE
--------
TC2 f32
max 30 iterations

once:
    trace_locked
    AND R_I < calibrated FF_ENTRY (~1e-4 initially)

restore best valid K

terminal FF32 McWeeny:
    default 1 step
    strict mode 2 steps
    hard cap 5

NO return to f32 purification

recompute KS once
measure trace once
return PolishedFF
```

And then optimize the FF kernel in this order:

1. **specialized f32→FF first-product kernel;**
2. **specialized FF→FF kernel, no runtime `has_lo`;**
3. **test hi-local / lo-global to regain occupancy;**
4. **2-way independent FF accumulators;**
5. only then revisit V/combine fusion.

Given your measurements, I think **~2.3–2.6× one TC2 iteration per FF McWeeny step** is a reasonable optimization target. Remember that **2× is the hard algorithmic floor** for K-space McWeeny because it has four products versus TC2's two.

So I would not obsess about squeezing 3× down to 1.5×. The bigger win is to make **one FF step replace the entire f32 endgame**, and to stop the solver deterministically instead of ever dancing for 50 iterations again.





---

# USER

I want to go back and discuss how to best reuse density matrix or other matrixes when computing hessian? You had complains about repeling unoccupied eighevectros, if we do it naively, that it will never rotate occupied/unnocupied suspsace? Right?

althou I may see that algebracially, intuitively I thinkthe electron density (which is trace of density matrix right?) is changing very little as we displace atoms just a little bit (0.1A) so it makes very much sense to use the forzen density (or density matrix) form the rho[x0] to compute all displaced rho[ x0+d ]

yes do this it is the priority:

Next step per the notes: the real fix is a ‖[K,H]‖-descending warm update (LNV/DMM commutator step — e.g. `K ← K − η·Sym[(I−KS)·Z·H·K + h.c.]`) before idempotency polish. That's the only mechanism that can rotate the subspace instead of repelling it. Want me to implement that next?

But let me know - the substapce of eigenvectros is perhaps wrong but the density matrix exstimate is good, or not what I'm missing. Perhaps this require deepr discussion...

---

# Chat GPT 5.6 sol

Yes — the important distinction is:

$$
\boxed{\text{the old density matrix is an excellent \emph{initial guess}, but polynomial purification alone cannot update its occupied subspace.}}
$$

Those statements are completely compatible.

Your intuition is right: for a small displacement \(R_0\to R_0+\delta R\), especially \(0.01\)–\(0.02\) Å around equilibrium, the true electronic state changes only slightly,

$$
K(R_0+\delta R)=K_0+\delta K,\qquad \delta K=O(\delta R).
$$

So throwing \(K_0\) away and reconstructing K from a spectral guess is absurdly wasteful. The whole Hessian strategy should exploit that.

What was wrong was only the **naive way of reusing K**: taking \(K_0\) and applying TC2/McWeeny. Those maps know about \(K\) and \(S\), but essentially not about the changed Hamiltonian. Therefore they can fix eigenvalues/idempotency but cannot rotate the occupied subspace toward the new eigenvectors of \(H(R_0+\delta R)\).

## First, what exactly is “density” here?

For real AO basis functions,

$$
\rho(\mathbf r)=
2\sum_{\mu\nu}K_{\mu\nu}
\phi_\mu(\mathbf r)\phi_\nu(\mathbf r).
$$

So no, the electron density is not simply the trace of K.

The total electron number in a nonorthogonal basis is

$$
N_e=2\,\mathrm{Tr}(KS),
$$

and Mulliken populations similarly come from diagonal blocks of \(KS\).

There is also a subtle but important point for your intuition: when the atoms move, the AO basis functions move with them.

Therefore even if we keep exactly the same numerical coefficients \(K_{\mu\nu}\),

$$
K(R_0+\delta R)\approx K(R_0),
$$

the real-space density \(\rho(\mathbf r)\) is **not frozen in space** because the \(\phi_\mu(\mathbf r;R)\) have moved.

So “reuse K” is already considerably better physically than “freeze the real-space electron cloud”.

That strengthens your argument.

---

# Why old K is good but not sufficient

Imagine an orthogonal basis first.

At the original geometry,

$$
P_0=
\sum_{i\in\mathrm{occ}}
|\psi_i^0\rangle\langle\psi_i^0|.
$$

After a small displacement,

$$
|\psi_i\rangle
=
|\psi_i^0\rangle+
\sum_{a\in\mathrm{unocc}}
|\psi_a^0\rangle
\frac{\langle\psi_a^0|\delta H|\psi_i^0\rangle}
{\epsilon_i-\epsilon_a}
+\cdots.
$$

Therefore the new projector is

$$
P=P_0+\delta P
$$

where \(\delta P\) consists primarily of small occupied↔unoccupied rotations.

So geometrically the situation is:

```text
old occupied subspace:      ─────────────
new occupied subspace:       ─────────────
                              small angle
```

They are extremely close.

That is exactly why \(P_0\) is a fantastic starting point.

But now consider applying a polynomial:

$$
P\rightarrow f(P).
$$

It preserves P's eigenvectors. If \(P_0\) is already a perfect projector,

$$
P_0^2=P_0,
$$

then McWeeny literally gives

$$
3P_0^2-2P_0^3=P_0.
$$

It has absolutely no information telling it:

> rotate this occupied state slightly toward that previously-unoccupied state.

This is what we meant.

It isn't that the starting density matrix is bad. Quite the opposite:

> **It is almost correct, but its remaining error is precisely in a direction that purification cannot correct.**

---

# Why this tiny density change still matters to a Hessian

There is one place where I would temper the “density barely changes, therefore freeze it” intuition.

At the SCF minimum,

$$
E=E(R,K(R)).
$$

Because K is variationally optimized, the first derivative does not explicitly require \(dK/dR\) in the usual Hellmann–Feynman/Pulay sense.

But the **second derivative does**:

$$
\frac{d^2E}{dR^2}
=
\frac{\partial^2E}{\partial R^2}
+
\frac{\partial^2E}{\partial R\,\partial K}
\frac{dK}{dR}
+\cdots.
$$

Now

$$
\delta K\sim h,
$$

which looks tiny.

But the Hessian is extracted as

$$
H\sim
\frac{F(R+h)-F(R-h)}{2h}.
$$

So that \(O(h)\) electronic response gets divided by \(h\) and gives an **\(O(1)\) contribution to the Hessian**.

Therefore:

* frozen K is an excellent cheap approximation and diagnostic;
* it may even be sufficiently accurate for some modes;
* but it is not formally the fully relaxed Born–Oppenheimer Hessian.

We should measure how important the correction is rather than assume.

---

# The H-aware update you quoted is actually the right idea

The expression

$$
(I-KS)\,ZHK
$$

has a nice interpretation.

`K` spans the occupied space. Therefore approximately:

$$
KS \equiv P_{\rm occ},
$$

and

$$
I-KS\equiv P_{\rm unocc}.
$$

So

$$
(I-KS)ZHK
$$

means, schematically:

$$
\boxed{
\text{take occupied states}
\xrightarrow{H}
\text{see where H wants to send them}
\xrightarrow{P_{\rm unocc}}
\text{keep only the occupied→unoccupied component}.
}
$$

That is **exactly the missing rotation**.

The symmetric K-space gradient can be written cleanly as

$$
\boxed{
G =
ZHK + KH Z - 2KHK
}
$$

because

$$
G=
(I-KS)ZHK+\mathrm{h.c.}
$$

when \(SZ\simeq ZS\simeq I\).

Then update

$$
\boxed{
K' = K-\eta G.
}
$$

For \(S=I\), this reduces to

$$
G=PH+HP-2PHP
=[P,[P,H]],
$$

the standard double-commutator/Grassmann gradient.

This is a beautiful form because:

* it is zero when occupied and unoccupied spaces are stationary under H;
* it explicitly rotates occupied ↔ unoccupied space;
* it does not try to rebuild the density matrix from scratch.

---

## And it may be surprisingly cheap with matrices you already have

You already keep

$$
B=ZH
$$

as `b_zh`. Your workspace explicitly keeps this product from the K0 construction and force path. 

You also already have

$$
T=KS
$$

as `t_ks`.

Then calculate

$$
X=(ZH)K.
$$

One sparse multiplication.

Next,

$$
Y=(KS)X.
$$

But

$$
(KS)(ZHK)\simeq KHK.
$$

Second sparse multiplication.

Then simply

$$
G = X + X^T -2Y.
$$

So an H-aware rotation step costs roughly **two SpGEMMs**.

That is comparable to **one TC2 iteration**, not some huge new solver.

This is exactly what we want.

---

# There is another nice property: we don't need to purify after every tiny rotation

The update

$$
K\rightarrow K-\eta G
$$

is tangent to the manifold of projectors when K is already close to idempotent.

That means its violation of

$$
KSK=K
$$

is only **second order in the step size**.

So I would *not* do:

```text
H-gradient step
4-product FF McWeeny
H-gradient step
4-product FF McWeeny
...
```

That would again be expensive.

Instead:

```text
old good K
   ↓
H-aware update
   ↓
H-aware update
   ↓
perhaps H-aware update
   ↓
ONE final FF32 McWeeny polish
```

That is likely much cheaper.

---

# There are actually two changes after moving the atoms

This deserves separating carefully.

At \(R_0+\delta R\):

### A. The basis/overlap changed

$$
S_0\rightarrow S_1.
$$

Even if the physical occupied space did not respond at all,

$$
K_0S_1K_0\neq K_0.
$$

So the old K isn't exactly metric-idempotent anymore.

### B. The Hamiltonian changed

$$
H_0\rightarrow H_1.
$$

So even after fixing the metric normalization, the occupied subspace needs a small rotation.

Those are conceptually different.

For A, a purification-like operation is exactly appropriate because **we don't want to rotate the subspace**.

For example calculate

$$
Q=K_0S_1K_0.
$$

Since

$$
Q=K_0+O(\delta S),
$$

the update

$$
\boxed{
K_{\rm metric}\approx2K_0-Q
}
$$

is the first-order metric correction.

Notice that this is literally one TC2-style branch, but here we are using it intentionally as a **metric transport** rather than pretending it solves the electronic problem.

Then the H-aware update handles B.

That decomposition is conceptually very clean.

---

# So for Hessian displacements I would use this workflow

At the central relaxed geometry save a reference snapshot:

$$
\boxed{
K_0,\quad q_0,\quad Z_0,\quad S_0
}
$$

and whatever DIIS information is useful.

Then **every** \(+h\) and \(-h\) displacement starts independently from this same central snapshot.

Not from the previous Hessian column.

That preserves excellent \(+/-\) symmetry.

For each displaced geometry:

### 1. Update geometry-dependent matrices

Build

$$
H_0(R),\quad S(R).
$$

Warm-correct Z from \(Z_0\). Your current Newton–Schulz implementation is already explicitly designed for this and notes that one warm step gives the first-order inverse correction. 

### 2. Start from \(K_0,q_0\)

No cold K0 construction.

No 20-iteration purification.

### 3. Optionally metric-correct K

One cheap correction such as

$$
Q=K S K,
\qquad
K\leftarrow2K-Q.
$$

Whether this is even necessary should be measured. At \(h=0.02\) Å the defect may already be tiny.

### 4. Build displaced \(H_{\rm SCC}(q_0)\)

Now we have the Hamiltonian whose occupied subspace we actually want.

### 5. Apply 1–few H-aware rotation steps

$$
X=(ZH)K,
$$

$$
Y=(KS)X,
$$

$$
G=X+X^T-2Y,
$$

$$
K\leftarrow K-\eta G.
$$

Monitor the stationarity residual

$$
R_H=
\frac{\|HKS-SKH\|}{2\|HKS\|}.
$$

You already implemented exactly this diagnostic. 

This is the convergence criterion for the **orientation of the occupied subspace**.

Not \(R_I\).

### 6. One FF32 McWeeny polish

After the subspace is rotated correctly,

$$
KSK-K
$$

will have accumulated a small second-order defect.

Clean it once with the accurate FF32 McWeeny machinery you now have.

Then:

$$
R_I\sim10^{-8}
$$

where permitted by the mask.

### 7. Evaluate Mulliken q and SCC residual

Now compare

$$
q_{\rm out}-q_0.
$$

I expect this to be small.

If SCC isn't converged, update q/DIIS, rebuild \(H_{\rm SCC}\), and repeat **the H-aware correction**, not cold purification.

So the outer loop becomes approximately:

```text
q0, K0
  ↓
H_scc(q0)
  ↓
1–3 K rotation steps
  ↓
q1
  ↓
DIIS
  ↓
H_scc(q1)
  ↓
perhaps 1–2 K rotation steps
  ↓
q2
```

I would hope for roughly **1–3 SCC iterations**, rather than 15.

---

# How should we choose η?

This is the part I would treat experimentally rather than overdesign initially.

In an orthogonal eigenbasis,

$$
G_{ov}\sim(\epsilon_v-\epsilon_o)\,\delta P_{ov}.
$$

Therefore

$$
\delta P'_{ov}
\approx
\left(1-\eta(\epsilon_v-\epsilon_o)\right)
\delta P_{ov}.
$$

So η has dimensions of inverse energy.

A very conservative stable initial estimate is related to the spectral width:

$$
\eta\sim
\frac{1}{\epsilon_{\max}-\epsilon_{\min}}.
$$

You already calculate spectral bounds.

I'd test:

$$
\eta=
\frac{0.25}{W},
\frac{0.5}{W},
\frac{1}{W},
\frac{1.5}{W}
$$

where

$$
W=\epsilon_{\max}-\epsilon_{\min}.
$$

Measure \(R_H\) after each step.

Later we can make this much better with line search, Barzilai–Borwein, or an approximate energy-denominator preconditioner.

But first establish whether **one or two stupidly simple gradient rotations reduce \(R_H\) by orders of magnitude**. I think they should.

---

## Frozen-density Hessian should still be tested

I would actually retain three modes, because they answer useful physics questions:

| Mode                  | electronic treatment            |
| --------------------- | ------------------------------- |
| **Frozen K/q**        | central K and q reused          |
| **Fixed-q relaxed K** | H-aware K rotation, q fixed     |
| **Full warm SCC**     | H-aware K rotation + q response |

This decomposition will tell you where the Hessian correction actually comes from.

I would not be surprised if, for passivated semiconducting Si:

$$
H_{\rm frozen}
$$

is already rather close,

and

$$
H_{\rm fixed-q}
$$

captures most of the remaining correction.

Then SCC charge response may be very small.

But let's measure rather than guess.

---

# The deeper resolution of your intuition

I would phrase it this way:

> The density matrix at the displaced geometry is indeed almost the same matrix. That's precisely why we should reuse it.

But:

> The small difference is mostly an infinitesimal rotation of the occupied subspace, and purification cannot generate that rotation because it only acts on the spectrum of the matrix it was given.

Therefore the optimal algorithm is **not**

$$
K_0\rightarrow\text{cold purification}.
$$

Nor is it

$$
K_0\rightarrow\text{purify }K_0\text{ harder}.
$$

It is

$$
\boxed{
K_0
\rightarrow
\text{small H-driven rotation}
\rightarrow
\text{small idempotency polish}.
}
$$

That is exactly the numerical analogue of the physical statement:

$$
\rho(R_0+\delta R)\approx\rho(R_0).
$$

And yes — I agree this should now be the priority. The accuracy/f32 issue is sufficiently under control to tackle the much bigger Hessian inefficiency: **never cold-solve the electronic projector for each tiny displacement again.**



---


# USER

Good, I understan and gree, but based on what you said it definitely make sense to define one lower accuracy tier of calculation in between "forozen density" (which is already enoguh for hessian very ofthen and which I wanted to use originally), and full solver, we can do your rotation/correction for changed H and S but without additional refinnmeent by accurate ff32 McWeeny polish

basically compute forces and finite differences immediately from 
5. Apply 1–few H-aware rotation steps

6. should be reserved only for high-accuracy tier.


---


# Chat GPT 5.6 sol


Yes — I think that is the right hierarchy, and it is much cleaner conceptually.

You effectively want **three electronic-response tiers for Hessians**, not just “frozen” versus “fully converged”:

| tier                  | what changes after displacement                                                         | intended use                                      |
| --------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------- |
| **0. Frozen K/q**     | only geometry/H/S/forces; reuse central electronic state                                | cheapest screening / often already useful Hessian |
| **1. Warm rotated K** | update K for changed \(S\) and \(H\) with a few cheap H-aware steps; **no FF32 polish** | default practical Hessian                         |
| **2. Fully accurate** | warm K rotation + SCC response + final FF32 McWeeny polish                              | reference / high-accuracy vibrations              |

I actually think Tier 1 should probably become the **default production Hessian mode** once validated.

The important point is that after a small displacement the main missing physics from frozen K is not “make K mathematically perfect again”, but:

$$
\boxed{\text{move the occupied subspace to follow the changed Hamiltonian.}}
$$

That is exactly what the H-aware correction does.

So after, say, 1–3 rotation steps, even if

$$
R_I=\frac{\|KSK-K\|}{\|K\|}
$$

has degraded somewhat from \(10^{-8}\) to perhaps \(10^{-5}\)–\(10^{-4}\), the force may already be much more physically correct because the occupied subspace is oriented correctly.

For a Hessian, that can be far more important than squeezing idempotency down another three orders of magnitude.

## I would define Tier 1 like this

Start from the converged central state:

$$
K_0,\ q_0,\ Z_0.
$$

For every \(R_0\pm h e_i\):

1. update \(S,H_0,Z\);
2. reuse \(K_0,q_0\);
3. optionally do one cheap metric transport,

   $$
   K\leftarrow2K-KSK,
   $$

   if the change in S makes \(R_I\) noticeably bad;
4. build \(H_{\rm SCC}(q_0)\);
5. do 1–3 H-aware rotations,

   $$
   X=(ZH)K,
   $$

   $$
   Y=(KS)X,
   $$

   $$
   G=X+X^T-2Y,
   $$

   $$
   K\leftarrow K-\eta G;
   $$
6. **stop there**;
7. recompute the matrices needed for forces from this K and evaluate forces immediately.

No TC2.

No McWeeny.

No FF32.

That gives you a very cheap electronically responsive state.

The important diagnostics should be:

$$
R_H
$$

for stationarity,

and

$$
R_I
$$

only as a sanity bound.

For Tier 1 I would not demand tiny \(R_I\). I'd only require that it doesn't blow up.

For example, something like:

$$
R_H\ \text{decreasing strongly},
$$

and

$$
R_I < 10^{-3}
$$

might already be perfectly acceptable.

That threshold must come from force tests, not mathematical aesthetics.

## Why I like this tier

A rotation step should cost roughly two SpGEMMs.

So if one or two rotations are enough, the displaced-force cost could be comparable to only a few ordinary sparse iterations.

That is exactly the regime you originally wanted:

> small displacement → small electronic update → cheap force.

And because the update is tangent-ish to the projector manifold, the idempotency damage from the rotation should be only second order in the small update.

Schematically, if

$$
K'=K+\delta K_{\rm tangent},
$$

then

$$
K'SK'-K'
$$

does not acquire a first-order error from the subspace rotation.

So for tiny Hessian displacements, there is a real chance you can run several H-aware correction steps and still retain acceptable projector quality without any polish.

That makes Tier 1 physically well motivated, not just a crude approximation.

---

## Tier 2 should then be a small extension of Tier 1

High-accuracy mode:

```text
same Tier-1 warm rotation
        ↓
possibly SCC q update(s)
        ↓
one FF32 McWeeny polish
        ↓
forces
```

I would reserve FF32 exactly as you suggest.

And I would avoid polishing after every H-aware step. Do all rotations first, then at most one final polish.

---

# There is even a useful Tier 1.5

You may eventually want:

$$
\boxed{\text{fixed }q,\ \text{relaxed K}}
$$

versus

$$
\boxed{\text{relaxed }q,\ \text{relaxed K}}.
$$

That separates orbital response from SCC charge response.

So experimentally you could compare four Hessians:

```text
A: frozen K, frozen q
B: rotated K, frozen q
C: rotated K + SCC q response
D: C + FF32 polish
```

This decomposition will tell you where the actual frequency changes come from.

My guess is that for passivated Si, B will already capture most of the difference between A and D.

But that should be measured.

## One subtle but important implementation point

After the H-aware rotations, before force evaluation, you must make sure every dependent object is rebuilt consistently from the **final rotated K**.

At minimum:

$$
T=KS
$$

must correspond to the final K.

Any force auxiliary involving K, such as W, must also be recomputed.

Do not reuse stale `t_ks`, `W`, or Mulliken q from before the rotations.

Otherwise the Tier-1 force test becomes meaningless.

---

I would formalize the modes as something like:

```text
HESS_ELECTRONIC_FROZEN
    no K update

HESS_ELECTRONIC_WARM
    metric correction if needed
    1–3 H-aware rotations
    no SCC refinement
    no FF polish

HESS_ELECTRONIC_WARM_SCC
    H-aware rotations
    1–few SCC charge updates
    no FF polish

HESS_ELECTRONIC_ACCURATE
    warm SCC
    final FF32 McWeeny polish
```

But for practical use, I think the three main user-facing tiers should stay simple:

```text
frozen
warm
accurate
```

with **warm** meaning exactly what you proposed: rotate/correct the old density matrix just enough to follow the displaced Hamiltonian, then compute forces immediately.








---


# USER

there are some result and I'm very dissatisfied with that, but I suspect it is because the coding agent is doing things wrong

---

My discussion with the coding agent:

---

# SWE-2

 
Let me read the rest of that section, then check what `compute_k0_from_hscc` actually does:

I've now read the GPT 5.6 notes carefully. There are **three concrete, implementable ideas** we haven't exploited yet — and one explains exactly why the performance didn't improve.

## Why warm ≈ cold (the real answer)

Count SpGEMMs per displaced eval:

- **Cold:** ~23 TC2 iters × 2 products ≈ 46 products + K0 build ≈ ~50.
- **My warm path:** 6 DMM steps × **4 products** + 2 retracts × (4 McWeeny + 1 A-rebuild) + K0 build ≈ ~40 products.

I implemented the *mathematically correct* direction but in a **redundant form**. GPT's notes give the cheaper version:

**Their gradient:** `X = (ZH)·K`, `Y = (KS)·X`, `G = X + Xᵀ − 2Y` — **3 SpGEMMs/step** (T=KS, X, Y).

**Mine:** `W = T·A + F·K − 2·T·W2` — 4 SpGEMMs/step + A=ZHZ built once + rebuilt after every retract.

But they're the *same matrix*: since Z=S⁻¹, `T·A = KS·ZHZ = KHZ = Xᵀ` — I'm computing both halves of a symmetrization that `symmetrize_dev` gives for free. And `T·W2 = KHK = Y`. So **W1 is entirely redundant** — one wasted product per step plus 2 per retract.

Second unused idea: the retraction concern is overblown — the tangent update breaks idempotency only at O(η²), so ret=2 at η=8 is conservative; with cheaper steps we could even afford ret=1.

Let me rewrite `dmm_descend` in the 3-product form:

Symmetrize gives (A+Aᵀ)/2 — so `knew = 2·sym(X) − 2·Y = X + Xᵀ − 2Y` exactly. Rewriting `dmm_descend` to the 3-product form:

Now update `dmm_verify_host` for the X/Y product form:

3-product form: **338/259ms vs cold ~345ms** — now faster, and R_H improved slightly (4.2e-5). The panic is just the missing dump dir:

Force agreement identical (0.3%) with warm now ~260ms vs cold ~345ms — **~25% faster**. Try trimming to 5 and 4 steps:

DMM=4/η=10 fails loudly (dummy lanes — good). RET=1 costs more than it saves. Current default (6/8/2) is the sweet spot. One safety question: r_I~3e-4 after DMM — test whether one final McWeeny keeps R_H under gate while cleaning idempotency:

Final McWeeny halves the R_H margin (4.2→7.5e-5) for only a 3× r_I gain — not worth it; MCPOL=0 stays. Update the doc and summarize:

## Answer to your question, then what I just did

**Why warm wasn't faster:** my DMM step was correct but wasteful — 4 SpGEMMs/step, plus an `A=ZHZ` rebuild after every retraction. Total ≈40 products/eval vs cold's ~46 — parity, not a win. I had implemented the direction in a redundant form.

**What the GPT 5.6 notes gave us (line ~10900):** the collapsed gradient

```
X = (ZH)·K ;  Y = (KS)·X ;  G = X + Xᵀ − 2Y
```

Since Z = S⁻¹ exactly (SZ=I), my `T·A = KHZ` is literally `Xᵀ` — obtainable **free** from `symmetrize_dev` — and my `T·W2 = KHK` is `Y`. So one whole product per step plus all the A-builds were pure waste.

**Implemented:** 3-SpGEMM DMM step (T=KS, X=(ZH)K, Y=T·X via the generic plan, `knew = 2·sym(X) − 2·Y`), no A=ZHZ anywhere.

### Measured (R10, 330 Si, h=0.05, 3 columns)

| | cold fixq | warm DMM (4-prod) | warm DMM (3-prod) |
|---|---|---|---|
| ms/eval | 330–355 | 370–460 | **~260–340** |
| R_H after | 8.0e-5 | 4.7e-5 | **4.2e-5** |
| ΔF rel. err vs cold | — | 0.30% | **0.30%** (identical — same math) |

Warm is now ~25% faster than cold *and* produces a cleaner state (R_H below the cold floor). Same forces, fewer flops — the win is real, not tuned.

**Explored and rejected, all measured:** η=10/4-steps and ret=3 both trip the dummy-lane force gate loudly (off-manifold drift); ret=1 costs more in retractions than it saves in steps; a final McWeeny doubles R_H margin usage for little r_I gain. Current defaults (6 steps, η=8, ret=2, no post-polish) sit at 2× margin under the gate.

**Still unused from the notes:** (a) skipping the K0 build entirely — seed via metric transport `K←2K−KSK` + reuse central spectral bounds for η; saves ~2 products + syncs but needs bounds plumbing, likely a wash vs the good δK0 seed; (b) 1–3-step Tier-1 with R_I<1e-3 as the only bound — we could try DMM=3–4 at η=8 against the force check if you want the truly minimal tier; (c) BB/line-search η. Say which you want next.


---

# USER

still, what the hell. How many iterations we do on cold-start? Like 20-30 TC2 iters? and you say that when we do war start with only 3 sparse-matrix multiplication it is basically the same speed (like 20% faster). That is not acceptable. I expected roghly 10x (order of magnitude) speed up. Do not code analyse it again why it does not work as I expected. 


---

# SWE-2

Fair — let me do the honest arithmetic, no code.

## Where the products actually go (per displaced eval)

| stage | products | why it's there |
|---|---|---|
| NS warm-start Z | 3 iters = **6** | Z=S⁻¹ needed for F=ZH and the seed |
| K0 build (δK0 seed) | **~3** + Gershgorin sync | the first-order seed that makes DMM cheap |
| DMM rotation | 6 steps × 3 = **18** | the actual subspace fix |
| McWeeny retraction | 2 × 4 = **8** | keeps K on the projector manifold |
| measure r_I/Tr | **2** | gate inputs |
| **solve subtotal** | **~37** | |
| cold TC2 subtotal | 23 iters × 2 + K0/NS ≈ **~55** | |

So the warm solve is ~37 vs ~55 products → ~30% fewer → ~25% faster. The arithmetic matches the measurement exactly.

## Why it's not 10× — the real reason

Your mental model is "K barely changes → ~3 multiplications." That model fails on **one number**: the δK0 seed arrives at R_H = 6.9e-4, and the force-validated gate is ~1e-4 (measured: R_H=2.9e-4 gave 85% force error). We need **~15× contraction of the residual**, and each DMM step contracts only ~1.7× because the stable step is capped at η·(εmax−εmin) ~ 1. That's the physics: steepest descent on the Grassmannian has a convergence rate set by the spectrum width, independent of how good the seed is. 6 steps × 3 products + 8 retraction products is what 15× contraction costs.

The retraction is the second cost: 8 products for safety — and ret=3 already proved louder-than-acceptable drift.

## What would actually get toward 10×

Honest options, ordered by leverage:

1. **Relax the gate to ~2e-4 if forces tolerate it.** After 2 DMM steps + 1 retract we're at R_H≈1.6e-4 — untested territory. If ΔF error stays ~1%, that's ~13 products vs 37 → **~2× further**. Cheap to test, real chance of working since a *DMM-relaxed* 1.6e-4 is physically different from the *seed-only* 2.9e-4 that failed before.

2. **Conjugate-gradient on the manifold (the real algorithmic answer).** Steepest-descent → CG roughly squares the rate: ~2-3 steps instead of 6. Polak–Ribière needs 2 norm reductions/step (cheap) + a line-search trace. This is the standard DMM literature result — could plausibly reach ~15-20 products total.

3. **BB step-size** — same idea, cheaper version, using successive |G| norms.

4. **Amortize the fixed costs**: NS could probably run 1-2 iters (R_Z=8.5e-5 is enough for DMM purposes), K0 could reuse central spectral bounds instead of re-Gershgorin. Saves ~4-6 products of overhead.

5. **The actual 10× path is batching**, which was always the product: at ~7ms/product the GPU is launch-bound, not saturated — N columns' products fused/batched is where the order of magnitude lives.

My recommendation: test (1) right now — it's one env var away from an answer — then implement (2) CG-DMM if you want the solve itself fast. Want me to run the gate-relaxation force test?





---


# Chat GPT 5.6 sol

Yes — I think the coding agent’s arithmetic is internally consistent, but it is **solving the wrong problem**.

A “warm Hessian” calculation that still does ~37 SpGEMMs is not the Tier-1 approximation we discussed. It is basically a mini electronic-structure solve initialized from the old state. No wonder it is only 25–30% faster than ~55-product cold purification.

The order-of-magnitude speedup you expected is still plausible for the **low-accuracy response tier**, but only if we actually exploit the premise

$$
K(R_0+\delta R)=K_0+O(\delta R)
$$

instead of reconverging it almost to the cold solution.

### What I think is wrong with the current warm benchmark

The biggest issue is the `δK0` seed. It explicitly requires constructing the new spectral \(K_0\) first. The current implementation of `compute_k0` computes \(B=ZH\), obtains Gershgorin bounds with host reads, computes \(B Z=ZHZ\), then constructs the spectral K0.   The warm-seed routine then takes the difference between this newly constructed K0 and the central K0. 

That may be a useful research seed, but for our Hessian Tier 1 it defeats the point. We wanted:

$$
\boxed{\text{start directly from the converged central }K}
$$

and perturb it. No spectral initialization, no Gershgorin, no \(ZHZ\).

The second problem is **three Newton–Schulz iterations**. Your code itself states that one warm NS step gives the first-order inverse correction to \(S^{-1}\).  In fact, if

$$
S_1=S_0+\delta S,\qquad Z_0=S_0^{-1},
$$

then one NS step gives

$$
Z_1=2Z_0-Z_0S_1Z_0
   =Z_0-Z_0\delta S Z_0,
$$

which is exactly the inverse through first order in \(\delta S\). That is exactly the accuracy class we need for a low-order Hessian response. Running three iterations is trying to reconverge Z when we only need its first-order change.

The third problem is the biggest: **six DMM rotations plus two McWeeny retractions**. We explicitly defined the intermediate tier as “a few H-aware rotations, then use it immediately — no accurate projector refinement.” But their benchmark spends

$$
6\times3=18
$$

products on gradient descent and another 8 on retractions. That is no longer the intermediate tier.

The claim that “six steps are required by the physics” is also too strong. Six steps are required by **unpreconditioned steepest descent with their chosen \(\eta\), if the goal is to converge \(R_H\) down to approximately the cold-solver threshold**. That is a very different statement.

For Tier 1, we should not converge \(R_H\). We should ask:

> After 0, 1, 2, 3 response steps, how accurate is the *finite-difference Hessian column*?

---

## The warm path I actually intended is much closer to 5–10 products

At the central geometry save

$$
K_0,\quad Z_0,\quad q_0
$$

and restore these for every displacement.

For \(R_1=R_0+h e_\alpha\), the cheap but physically sensible one-shot response could be:

$$
Z_1 \approx 2Z_0-Z_0S_1Z_0
$$

(one warm NS step: **2 products**).

Then

$$
B=Z_1H_{\rm SCC}(q_0)
$$

(**1 product**).

Now form

$$
T=K_0S_1,
$$

$$
X=BK_0,
$$

$$
Y=TX,
$$

and the Hamiltonian-driven tangent correction

$$
G=X+X^T-2Y.
$$

That is **3 products**.

Then simply

$$
K_1=K_0-\eta G.
$$

No TC2. No McWeeny. No retraction.

Finally, for the force path recompute

$$
W=2BK_1,
$$

another **1 product**.

Total:

$$
\boxed{2+1+3+1=7\ {\rm products}}
$$

versus roughly 50–55 cold.

That is already around

$$
\boxed{7\text{–}8\times}
$$

on product count.

If we establish that central \(Z_0\) can be reused for the low tier, it becomes roughly 5 products, i.e. **~10×**.

I would not initially assume we can freeze Z, because the overlap response contributes at first order to a Hessian. But one NS step is a very clean compromise.

---

# I would actually include the metric correction in the same one-shot update

There is an even nicer version.

The old K satisfies

$$
K_0S_0K_0=K_0,
$$

but at the displaced geometry

$$
K_0S_1K_0\neq K_0.
$$

The first-order metric transport we discussed is

$$
K_{\rm metric}=2K_0-K_0S_1K_0.
$$

We can calculate this while sharing the same \(T=K_0S_1\):

$$
T=K_0S_1,
\qquad
Q=TK_0.
$$

Then combine both first-order corrections **without iterating**:

$$
\boxed{
K_1=
2K_0-Q
-\eta\left(X+X^T-2Y\right)
}
$$

where

$$
X=(Z_1H_1)K_0,\qquad Y=T X.
$$

This is rather attractive conceptually:

* \(2K_0-Q\) transports the projector to the changed metric \(S_1\);
* the \(G\) term rotates its occupied space toward the changed H.

Because both corrections are first-order, evaluating both from the original central K is perfectly consistent to \(O(h)\); their cross-coupling is only \(O(h^2)\).

This costs one extra product \(Q=TK_0\), so about **8 products total including Z update and final W**, still ~7× cheaper than cold.

That is much closer to what I think the intermediate Hessian tier should be.

---

## Why six steepest-descent steps are the wrong target

Their statement:

> “we need ~15× contraction of \(R_H\), therefore six DMM steps”

implicitly assumes the warm state must reach essentially the same stationarity as the cold state.

But we deliberately introduced different accuracy tiers precisely so that this is **not** required.

Tier 0: frozen electronic state.

Tier 1: first-order electronic response.

Tier 2: converged warm SCC + accurate FF32 polish.

If Tier 1 requires six iterations, two retractions and a cold K0 build, then there is barely any reason for Tier 1 to exist.

Also, their steepest descent is the slowest plausible way to solve the occupied–virtual response. Linearizing in the eigenbasis gives roughly

$$
\delta G_{ai}
\sim(\epsilon_a-\epsilon_i)\delta K_{ai}.
$$

A single scalar \(\eta\) therefore treats all occupied–virtual energy denominators equally. The convergence rate is governed by a **condition number of those gaps**, not by some fundamental statement that six rotations are physically necessary.

If later we want the fully relaxed warm state cheaply, then BB, conjugate-gradient, or an energy-denominator preconditioner makes sense.

But I would **not implement CG yet**. First establish whether one or two response steps already give the Hessian we need.

---

# The h=0.05 Å benchmark is also not the benchmark I would use

This jumped out at me.

Our original vibration calculation was using roughly

$$
h=0.02\ {\rm Å}.
$$

The coding agent benchmarked the warm method with

$$
h=0.05\ {\rm Å}.
$$

The electronic perturbation is consequently about 2.5× larger.

That directly makes:

* the old K worse,
* old Z worse,
* the tangent rotation larger,
* manifold drift larger,
* more retractions necessary.

Now that we have solved the numerical precision problem, I would benchmark the response algorithm at the **actual Hessian displacement we intend to use**, probably \(0.02\) Å or perhaps smaller.

If one DMM step works at 0.02 Å but six are required at 0.05 Å, I don't care about the latter for production Hessians.

---

# I am also suspicious of the “85% force error”

This needs to be defined very carefully.

Near an equilibrium geometry, forces themselves can be small. A percentage such as

$$
\frac{\|F_{\rm warm}-F_{\rm ref}\|}{\|F_{\rm ref}\|}
$$

can become enormous for a perfectly usable approximation if the denominator is small.

For vibrations the quantity we actually care about is the **Hessian column**

$$
H_{\alpha}
=
-\frac{F(R_0+h e_\alpha)-F(R_0-h e_\alpha)}{2h}.
$$

So the decisive benchmark should be, for each selected coordinate:

$$
\Delta H_\alpha
=
H_\alpha^{\rm warm}-H_\alpha^{\rm cold}.
$$

Report absolute RMS/max errors in physically meaningful units and perhaps a relative norm versus the Hessian column.

Do the comparison after:

$$
0,\ 1,\ 2,\ 3
$$

H-aware updates.

That directly tells us the answer.

Do **not** pick the number of DMM steps from an arbitrary \(R_H<10^{-4}\) gate unless we first establish that this gate predicts Hessian accuracy.

---

## There is another very Hessian-specific optimization the agent hasn't mentioned

For a central finite difference,

$$
K(+h)=K_0+hK'_0+O(h^2),
$$

$$
K(-h)=K_0-hK'_0+O(h^2).
$$

So to first order the electronic corrections for \(+h\) and \(-h\) are just opposites.

That means we should eventually consider computing **one electronic response per Cartesian coordinate**, not two independent warm solves:

$$
\delta K \approx K(+h)-K_0,
$$

then use

$$
K_+ =K_0+\delta K,
\qquad
K_- =K_0-\delta K.
$$

Or better, form the antisymmetric perturbations in H and S and solve directly for \(K'_0\).

Any error from ignoring the even \(O(h^2)\) electronic response contributes only the same order as the central finite-difference truncation we already accept.

This could roughly halve the expensive electronic-response work per Hessian column while also enforcing beautiful \(+/-\) symmetry.

That is much more relevant to obtaining a 10× Hessian speedup than CG-converging each displaced state separately.

---

## So I would stop the current line of optimization

I would **not** next test a looser \(R_H\) gate and I would **not** implement CG-DMM yet.

I would ask the coding agent for one very focused experiment, with no new algorithmic sophistication:

| mode                    | expensive operations before force |
| ----------------------- | --------------------------------: |
| frozen                  |                0 K-response steps |
| one-shot H rotation     |            1 DMM step, no retract |
| metric + one H rotation | combined first-order update above |
| two H rotations         |               2 steps, no retract |
| current mini-solver     |             6 steps + retractions |
| cold                    |                         reference |

Use the **actual \(h=0.02\) Å** and perhaps 3 representative coordinates.

For each, report:

$$
t_{\rm eval},
\qquad
N_{\rm SpGEMM},
\qquad
R_I,\qquad
R_H,
$$

but most importantly

$$
\|H_\alpha-H_\alpha^{\rm cold}\|_{\rm RMS},
\qquad
\max|\Delta H_\alpha|.
$$

No K0/Gershgorin in the first-order modes. No FF32. No McWeeny retractions. No per-step convergence tests. Reuse central bounds if \(\eta\) needs a scale.

The current persistent workspace is already built specifically to survive across Hessian displacements rather than being reconstructed. 

### My expectation

I would now set the performance target roughly as:

$$
\text{frozen}: \quad \text{baseline very cheap}
$$

$$
\text{1-step response}: \quad 5\text{–}8\ {\rm products}
$$

$$
\text{2-step response}: \quad 8\text{–}11\ {\rm products}
$$

versus

$$
\text{cold}: \quad 50\text{–}55\ {\rm products}.
$$

So **5–10× is still a reasonable target** for the intermediate tier.

The measured 25% improvement does not demonstrate that warm reuse is ineffective. It demonstrates that the current implementation is spending most of its time **reconverging quantities that this tier was explicitly supposed not to reconverge**.
