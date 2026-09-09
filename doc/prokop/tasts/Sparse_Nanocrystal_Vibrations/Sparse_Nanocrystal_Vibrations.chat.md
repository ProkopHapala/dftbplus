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

