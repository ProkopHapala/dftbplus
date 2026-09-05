# USER

I have overlap integrals and tight binding radial functions from fireball, siesta, xTB and dftb+ (slater coster tables), they are expressed on grid and interpolated by splines.

What I would like to do is express them by polynomials. But not any polynomials such which go smoothly to zero at some distance R-cut. Also I'm thinking how to use small number of such polynomial terms to express large variety of such radial functions (for different pairs of elements), it should be kind of compression because especially on GPU we want to minimize the global memory reads and we have limited group local memory so we can afford only e.g. 4-8 coefficients for each such function for given pair of elements.


My idea is that I can take set of all such functions as a space of probes and find by PCA minimal polynomial representation up to some degree to choose basis in which it is efficient to express all of them using common basis.


Also high order polynomial fit is unstable so we should use something like Legendre, gauss quadrature or something numerically stable with orthogonal polynomials?

---

# Gemini

This is an excellent engineering strategy for GPU acceleration. Shifting from spline lookups (which are memory-bandwidth constrained due to global memory latency and random cache misses) to a local polynomial evaluation (which is compute-bound and fits into registers) is exactly how you unlock massive speedups on modern GPUs.

Your intuition regarding **PCA**, **smooth cut-offs**, and **orthogonal polynomials** is completely spot-on. Standard high-order polynomials ($x^n$) suffer from severe Runge's phenomenon (oscillations near the boundaries) and ill-conditioned matrices (the Vandermonde matrix becomes a numerical nightmare).

Here is a mathematically robust and GPU-optimized framework to achieve exactly what you want.

## 1. Handling the Smooth Cut-off ($R_{\text{cut}}$)

To ensure the polynomial goes smoothly to zero at $R_{\text{cut}}$ without forcing a high-order polynomial to do the heavy lifting, you should **factor out the boundary condition**.

Instead of fitting the radial function $f(r)$ directly, fit an auxiliary function $g(x)$ by multiplying the polynomial basis with a fixed, analytical **damping/envelope function** $\chi(r)$ that naturally possesses the required boundary conditions.

A highly effective envelope choice for tight-binding/overlap functions is:


$$\chi(r) = \left(1 - \left(\frac{r}{R_{\text{cut}}}\right)^2\right)^n \quad \text{for } r \le R_{\text{cut}}$$

Where $n \ge 2$ ensures that both the function and its first (and second) derivatives vanish smoothly at $R_{\text{cut}}$. Your target function then becomes:


$$f(r) = \chi(r) \sum_{k} c_k P_k(r)$$

This transforms the problem into fitting a modified, well-behaved function on a bounded domain, removing the edge-instability.

## 2. Choosing the Stable Basis: Shifted Chebyshev Polynomials

While Legendre polynomials are great, **Chebyshev polynomials of the first kind ($T_k$)** are the gold standard for minimax approximations (minimizing the maximum error over an interval). They distribute errors uniformly across the domain rather than bunching them up at the edges.

Since your domain is $r \in [0, R_{\text{cut}}]$, you map $r$ to the standard Chebyshev domain $x \in [-1, 1]$ using a linear transformation:


$$x = \frac{2r}{R_{\text{cut}}} - 1$$

### Why Chebyshev works beautifully here:

* **Orthogonality:** They are orthogonal under the weight $1/\sqrt{1-x^2}$, making the projection numerically bulletproof.
* **Clenshaw’s Algorithm:** Evaluating a Chebyshev series on a GPU does not require computing cosines. It can be done using a highly efficient 3-term recurrence relation (Clenshaw's recurrence), which requires very few registers and executes in a tight loop.

## 3. Global Compression via PCA (SVD)

To achieve your goal of compressing *all* element pairs into a tiny footprint (4–8 coefficients per pair), you can use Singular Value Decomposition (SVD) to find a **shared, optimal basis set of functions** across your entire database.

Here is the step-by-step pipeline:

### Step 1: Uniform Resampling

Sample all your radial functions (across all element pairs, overlaps, and Hamiltonians) onto a shared, dense Gauss-Chebyshev grid or a highly refined uniform grid. Let’s say you have $M$ total functions, sampled at $N$ radial points. This gives you a data matrix $\mathbf{A}$ of size $N \times M$.

### Step 2: Project onto a High-Order Chebyshev Basis

Before doing PCA, project each raw grid function onto a relatively high-order Chebyshev basis (e.g., degree $K = 16$ or $24$). This filters out high-frequency grid noise and gives you a matrix of coefficients $\mathbf{C}$ (size $K \times M$).

### Step 3: Compute the SVD

Perform SVD on your coefficient matrix $\mathbf{C}$:


$$\mathbf{C} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

* $\mathbf{U}$ (size $K \times K$) gives you the **Principal Component Functions** (the optimal, shared meta-polynomials).
* $\mathbf{\Sigma}$ gives you the singular values (eigenvalues), showing you exactly how much error you incur by dropping terms.
* $\mathbf{V}^T$ contains the weights for each specific element-pair.

### Step 4: Truncate to $L$ terms (e.g., $L = 4 \text{ to } 8$)

Keep only the top $L$ columns of $\mathbf{U}$. Now, **every single radial function** in your engine can be reconstructed as a linear combination of just a few global "basis shapes":

$$f_{\text{pair}}(r) = \chi(r) \sum_{l=1}^{L} w_{l,\text{pair}} \cdot \Phi_l(r)$$

Where $\Phi_l(r)$ are the $L$ global polynomial shapes defined by $\mathbf{U}$, and $w_{l,\text{pair}}$ are the 4 to 8 unique coefficients you store for that specific pair.

## 4. GPU Implementation Strategy

This approach changes your GPU memory architecture dramatically for the better:

1. **Constant Memory (`__constant__`) / Local Registry:** The global meta-polynomials $\Phi_l(r)$ (requiring $K \times L$ coefficients total) are shared by *everyone*. You load these once into GPU constant memory at the start of the kernel. They are tiny.
2. **Texture or Compact Global Memory:** For each pair interaction, a thread only needs to read $L$ numbers ($w_{l,\text{pair}}$). If $L=4$, this is a single `float4` read per pair!
3. **Execution:** The thread evaluates the $L$ shared polynomials using Clenshaw's recurrence, multiplies them by the $L$ weights, and applies the envelope $\chi(r)$.

This eliminates the splined texture/global memory throttling completely, trading high-latency memory fetches for cheap, lightning-fast ALU floating-point operations.

---

# Kimi-2.6

# Plan: PCA-Based Polynomial Compression of Slater-Koster Tables

## 1. Data Inventory

### SK Parameter Sets
- **3ob-3-1**: 225 .skf files, elements {H, C, N, O, P, S, F, Cl, Br, I, Na, K, Mg, Ca, Zn} — organic/bio + some metals
- **mio-1-1**: 36 .skf files, elements {H, C, N, O, P, S} — CHNOPS only

### File Format Observations
- **Old format** (most files): line 1 = `dr, n_grid_raw, n_shell`, then onsite params, skip EDIF, then `n_grid_raw-1` rows of 20 values (10 H + 10 S integrals)
- **Extended format** (some 3ob files): starts with `@`, then same structure but 40 values per row (20 H + 20 S)
- **mio-1-1** uses repeat syntax (`19*0.0`, `20*1.0`, `5*0.0`)
- Tail of each file contains XML documentation (ignored by parser)
- `DIST_FUDGE = 1.0` is added to grid-based `r_max = (n_grid-1)*dr`; the actual cutoff used by the code is `r_max() = grid_r_max + DIST_FUDGE`
- Beyond grid end, `poly5_to_zero` extrapolation forces smooth decay to zero

### Cutoff Variability (3ob-3-1 samples)
| Pair | dr | n_grid | grid_r_max | Rc (with fudge) |
|------|-----|--------|------------|-----------------|
| H-H  | 0.02 | 550 | 10.98 | 11.98 |
| C-C  | 0.02 | 650 | 12.98 | 13.98 |
| C-H  | 0.02 | 500 | 9.98  | 10.98 |

=> Different pairs have different Rc. Normalization to `u = r/Rc` is essential.

---

## 2. Technical Pipeline

### Step 1: Load All Integral Curves

For each .skf file, extract each column as a separate curve:
- Old format: 10 H integrals + 10 S integrals = 20 curves
- Extended format: 20 H + 20 S = 40 curves
- For organic elements (s,p only), only first 4 H and 4 S are non-zero:
  - col 0: H_ssσ, col 1: H_spσ, col 2: H_ppσ, col 3: H_ppπ
  - col 10: S_ssσ, col 11: S_spσ, col 12: S_ppσ, col 13: S_ppπ

**Decision**: Start with s,p-only channels (8 curves per file) since d-functions are sparse in these parameter sets. Can extend later.

### Step 2: Normalize to Common u-Domain

For each curve:
- `Rc = tab.r_max()` (includes DIST_FUDGE)
- `u = r / Rc`, domain `u ∈ [0, 1]`
- Choose common resampling grid in u, e.g. `N_u = 500` or `1000` uniformly spaced points
- For each `u_i`, compute `r_i = u_i * Rc` and evaluate using existing `EqGridTable::eval()`

**Rationale**: All curves now live on same x-axis. The shape variations are intrinsic, not due to different length scales.

### Step 3: Normalization / Preprocessing

Curves have vastly different magnitudes:
- H integrals: ~1–10 eV (tens for some)
- S integrals: 0–1 (overlap)
- Different pairs span different ranges

**Options**:
1. **No normalization**: PCA will be dominated by large-magnitude curves
2. **Per-curve L2 normalization**: Divide each curve by its L2 norm; PCA captures shape
3. **Per-curve max normalization**: Divide by max absolute value
4. **Separate PCAs for H and S**: Natural separation since they have different units and scales

**Recommendation**: Option 4 (separate PCAs) + within each, option 2 (L2 norm). This ensures shape dominates over magnitude differences.

### Step 4: Envelope / Damping Function

The SK tables already go to zero at Rc (forced by poly5_to_zero tail). The user's idea is to factor this out:

```
f(r) = χ(u) * g(u)
```

where `χ(u)` is a fixed envelope. Candidates:
- `χ(u) = (1-u)^2` — simple, goes to zero at u=1
- `χ(u) = (1-u^2)^2` — smoother at u=0, also suggested in the compression doc
- `χ(u) = (1-u)^3` or higher for smoother derivatives

**Two PCA variants to test**:
- **Variant A**: Fit `g(u) = f(u)/χ(u)` directly, then PCA on `g(u)`. Reconstruction: `f̂(u) = χ(u) * Σ w_l Φ_l(u)`
- **Variant B**: PCA on `f(u)` with inner product weighted by `χ(u)^2` (or `χ(u)`), so regions near u=1 contribute less. This is equivalent to saying we care more about fitting the bulk shape than the exact tail.

**Recommendation**: Test both. Variant A is cleaner mathematically (the basis functions don't need to satisfy boundary conditions). Variant B is simpler (no division by zero issues near u=1).

### Step 5: PCA / SVD

Data matrix `A` of size `N_u × M` where `M` = number of curves.

**Simple PCA**:
- Center columns (subtract mean curve)
- SVD: `A = U Σ V^T`
- Principal components = columns of `U`
- Truncate to L components, reconstruction error from singular values

**Weighted PCA** (Variant B):
- Weight matrix `W = diag(χ(u_i))` or `W = diag(χ(u_i)^2)`
- Solve weighted SVD: `W^{1/2} A = U Σ V^T`
- PC functions = `W^{-1/2} U` (need to unweight)

### Step 6: Polynomial Fitting of Principal Components

Once we have the top L principal components `Φ_l(u)`, we need to express them as polynomials.

**Option 1: Direct polynomial fit**
- Fit each `Φ_l(u)` with degree-D polynomial on u ∈ [0,1]
- Problem: high-degree polynomial unstable, Runge phenomenon

**Option 2: Shifted Chebyshev basis (recommended in doc)**
- Map `u ∈ [0,1]` to `x ∈ [-1,1]` via `x = 2u - 1`
- Expand `Φ_l(u) = Σ_{k=0}^{K} c_{l,k} T_k(x)` where `T_k` are Chebyshev polynomials
- Projection is stable due to orthogonality
- Clenshaw recurrence for fast GPU evaluation

**Option 3: Polynomial fit of g(u) = f(u)/χ(u)**
- Since χ(u) handles the boundary, g(u) is smoother
- Can use lower-degree polynomials
- On GPU: evaluate Chebyshev series for g(u), then multiply by χ(u)

**Recommended path**:
1. Do PCA in the function space (step 5)
2. Fit resulting PCs with shifted Chebyshev polynomials
3. Store Chebyshev coefficients for each PC
4. For each element pair, store weights `w_l` for the L PCs

### Step 7: Compression Ratio

Original storage per pair:
- ~500–650 grid points × 20 values × 8 bytes ≈ 80–100 KB

Compressed storage per pair:
- L weights × 8 bytes (e.g., L=8 → 64 bytes)
- Global PCs: K coefficients × L PCs × 8 bytes (e.g., K=16, L=8 → 1 KB)

=> **~1000× compression** per pair, with global PC table amortized across all pairs.

## 3. Implementation Steps (for later coding)

### Phase 1: Data Extraction & Analysis
1. Write Python loader that reuses the Rust parser or reimplements the SK parsing
2. Extract all H and S curves from both parameter sets
3. Normalize to u-domain, resample on common grid
4. Save as `.npz` archive for quick reloading
5. Plot sample curves to verify correctness

### Phase 2: PCA Experiments
1. Run simple PCA on H-integrals and S-integrals separately
2. Plot singular value spectra (decay rate = compressibility)
3. Test reconstruction error vs. number of components L
4. Repeat with weighted PCA (damping function)
5. Repeat with envelope-factored approach g(u) = f(u)/χ(u)
6. Compare which gives best reconstruction for given L

### Phase 3: Polynomial Basis
1. Take top L PCs from best variant
2. Fit each PC with shifted Chebyshev polynomials (degrees 8, 12, 16, 20)
3. Measure Chebyshev-fit error
4. Determine optimal (L, K) tradeoff
5. Verify full reconstruction: `f̂(u) = χ(u) * Σ_l w_l * (Σ_k c_{l,k} T_k(2u-1))`

### Phase 4: Rust Integration
1. Add compressed SK data structure to `rust_dftb`
2. Precompute global Chebyshev coefficients at compile time or load from file
3. Implement Clenshaw evaluation for each PC
4. Replace `EqGridTable` evaluation with compressed evaluation
5. Run parity tests to verify accuracy

## 4. Open Questions / Decisions

1. **Should H and S be in the same PCA or separate?**
   - Separate is cleaner (different units, scales)
   - Same could exploit correlations but probably not worth it

2. **Should different angular momentum channels (ssσ, spσ, ppσ, ppπ) be in same PCA?**
   - Same PCA: more data, might find shared features
   - Separate PCAs: each channel's unique shape better captured
   - Recommendation: try both; likely separate is better

3. **What envelope power n?**
   - n=2: `(1-u)^2` — simple, derivative vanishes at u=1
   - n=3: `(1-u)^3` — second derivative vanishes, smoother
   - n=4: `(1-u)^4` — even smoother, but g(u) may blow up near u=1
   - Need to balance numerical stability vs. smoothness

4. **How many u-grid points for resampling?**
   - 500 is probably sufficient; 1000 for safety
   - More points = better PCA but slower computation

5. **Include both 3ob and mio in PCA, or one at a time?**
   - Together: more diverse data, potentially more general basis
   - Separate: basis optimized for each parameter set
   - Recommendation: start with 3ob (larger, more diverse), test on mio

6. **Should we include d-function channels?**
   - 3ob has d-functions for some elements (P, S?)
   - Start without, add later if needed

## 5. Expected Challenges

- **Division by zero**: If using `g(u) = f(u)/χ(u)`, near u=1 both go to zero. Need careful handling (regularization or use weighted PCA instead).
- **r=0 behavior**: Some integrals are exactly zero at r=0 (e.g., H_spσ). Interpolation handles this, but need to ensure resampling is accurate near origin.
- **mio vs 3ob differences**: Same pair (e.g., C-C) has different values in different sets. PCA will find average; weights capture per-set deviation.
- **XML tail in .skf files**: The Rust parser stops reading at n_grid lines, ignoring XML. Python loader should do the same.

---

# USER

I agree with you upp to the exponential prefactor function. The thing is that exp(x) is rather costly to express, and it does not have finite cutoff. Anyways efficient way how to approximate exp(-bx) is (1-bx/n)^n, which has natually finite cutoff at Rc=n/b
therefore I think using polynominal basis is sweestp from both performance perspective as well as guarantteing the porper finite support.

what do you think about
4. Better Ansatz: Jacobi / Shifted Legendre with Cutoff Weight ?

Splitting ss, spq,ppq,pppi would be good strategy as they have different behaviour, but having different basis matrixes would again consume more precious GPU shared memopry. Perhaps bettr strategy is use constant prefactor for each to represent theyir characteristic behaviuour but then use common basis

```
Group 0 — ss: No node, peak at origin
Group 1 — spσ: Linear node, odd symmetry
Group 2 — ppσ: Quadratic node
Group 3 — ppπ: Quadratic node, different amplitude
```

---

# Gemini

You are completely right about avoiding explicit `exp(-bx)` evaluations on the GPU. Not only is `exp` relatively slow, but managing its infinite tail while trying to force a hard cutoff at $R_{\text{cut}}$ introduces exactly the numerical tension that ruins compression efficiency.

Your insight that $(1 - bx/n)^n$ naturally bridges the gap between exponential decay and a finite polynomial cutoff is brilliant. If we set $n=2$, it matches your original envelope, and if we use slightly higher powers, it perfectly mimics the physical decay of the orbitals while remaining a pure polynomial.

Let's rethink the strategy based on your constraints: **minimizing GPU shared memory (using one common basis matrix)** while still **capturing the drastically different nodal behaviors of $ss$, $sp\sigma$, $pp\sigma$, and $pp\pi$**.

## 1. The Common Basis with Type-Specific Prefactors

To avoid loading multiple basis matrices into the GPU's limited registers or shared memory, we can use a **single global polynomial basis** $P_k(u)$ for *all* interactions, but modify the input or the prefactor based on the orbital pair type.

Since $ss$, $sp\sigma$, $pp\sigma$, and $pp\pi$ have distinct behaviors at the origin ($r \to 0$) and different nodal structures, we can introduce a cheap, type-specific **geometric prefactor** $r^m$ before multiplying by the common basis.

Let $u = r/R_{\text{cut}}$, where $u \in [0, 1]$. We can express any radial function as:

$$f_{\text{type}}(r) = \chi(u) \cdot u^{m_{\text{type}}} \cdot \sum_{k} c_k P_k(u)$$

Where $\chi(u) = (1-u^2)^2$ is the common smooth cutoff envelope.

### The Channel Prefactors:

* **Group 0 ($ss\sigma$):** $m = 0$. Value is finite at the origin, smoothly decays.
* **Group 1 ($sp\sigma$):** $m = 1$. Forces a linear node at the origin ($f(0) = 0$), capturing the odd symmetry.
* **Group 2 & 3 ($pp\sigma, pp\pi$):** $m = 2$. Forces a quadratic start at the origin.

By factoring out $u^m$, the remaining part of the function becomes a smooth, non-oscillatory curve that looks remarkably similar across *all* groups. The common PCA basis no longer has to learn how to shift nodes or create zeros at the origin; it only has to learn the smooth variations.

## 2. Why Shifted Legendre with a Cutoff Weight works perfectly

Your proposal to use **Shifted Legendre Polynomials $\tilde{P}_k(u)$** on the domain $u \in [0, 1]$ is mathematically ideal for this setup.

Unlike Chebyshev polynomials, which are orthogonal with respect to the weight $1/\sqrt{1-u^2}$ (which weights the edges heavily), **Legendre polynomials are orthogonal with respect to a constant weight ($w(u) = 1$)**.

### The Numerical Advantage:

When you perform the PCA in the continuous domain, using a basis that is orthogonal under a flat weight means that your singular values directly map to the **true root-mean-square error (RMSE)** of the physical curve.

If we define our total trial space using the Shifted Legendre basis scaled by our envelope and geometric factors:


$$\Phi_{k, \text{type}}(u) = \chi(u) \cdot u^{m_{\text{type}}} \cdot \tilde{P}_k(u)$$

We can run the SVD on the coefficients of this well-behaved Legendre expansion. Because $\tilde{P}_k(u)$ are perfectly stable up to degree 8–12, you will not encounter any of the ill-conditioned matrix explosions of the standard Monomial ($u^k$) basis.

## 3. The Refined GPU Execution Flow

By choosing this architecture, you only store **one set of global polynomial coefficients** in the GPU's `__constant__` memory (shared by all threads), and a tiny set of 4–6 weights per element-pair in global memory.

For a specific pair, the GPU kernel does this:

1. **Compute normalized distance:** $u = r / R_{\text{cut}}$.
2. **Evaluate Common Basis:** Evaluate the 4 to 6 terms of the Shifted Legendre basis $P_k(u)$ using its standard, lightweight 3-term recurrence relation:

$$(k+1)\tilde{P}_{k+1}(u) = (2k+1)(2u-1)\tilde{P}_k(u) - k\tilde{P}_{k-1}(u)$$



*(This takes only a few registers and cycles).*
3. **Multiply by Weights:** Dot product the results with the 4–6 pair-specific coefficients $c_k$.
4. **Apply Type-Prefactor:** Multiply by the shared envelope and the quick bit-shift or multiplication for the node: $\chi(u) \cdot u^m$.

This completely homogenizes the execution path on the GPU. Every thread executes the exact same instructions (the same Legendre recurrence), preventing branch divergence. The only difference between an $ss$ calculation and a $pp\sigma$ calculation is the integer exponent $m$ used in the final scaling step. This will tighten the PCA variance significantly, likely dropping your required component count to 4–5 while keeping the RMSE safely below your target.

---

# DeepSeek-V4-pro

You’ve hit exactly the key tension: different SK channels (ss, spσ, ppσ, ppπ) look qualitatively different, but the GPU budget can’t afford a separate basis matrix for each. Your proposal—**prefactoring each group with a fixed function and then using a *single* common polynomial basis**—is both physically motivated and numerically elegant. Let’s elaborate on this, and on why the shifted‑Legendre + cutoff‑weight approach fits perfectly.

## 1. Why splitting by orbital type is essential

Consider the behaviour of the radial functions at the origin \(r=0\):

- **ssσ** : finite, maximum at origin, monotonically decaying.  
- **spσ** : zero at origin, changes sign once (a node), linear rise then decay.  
- **ppσ / ppπ** : zero at origin, roughly parabolic start, then decay with or without an additional node.

If we dump all of these into a single PCA without any preprocessing, the dominant components must simultaneously represent functions that start at 0.0, 0.5, 1.0, etc. The PCA will waste several components just to encode the **origin value** and the **presence of a node**. The singular value spectrum decays slowly because these “structural” degrees of freedom are orthogonal to the smooth continuum decay shapes that the PCA is actually good at capturing.

By factoring out a **geometric prefactor** \(u^{m_{\text{type}}}\) (where \(u=r/R_{\text{cut}}\)), we separate the low‑order behaviour at the origin:

- Group 0 (ss) : \(m=0\)  
- Group 1 (spσ) : \(m=1\)  
- Group 2 (ppσ, ppπ) : \(m=2\)  

The remaining part \(g(u) = f(u) / u^m\) is a **smooth, nodeless (or at least simpler) function** that goes to a finite value at \(u=0\) and decays to zero at \(u=1\). Crucially, these \(g(u)\) now look *very similar across all groups*. They all start positive, decay monotonically, and are zero at the cutoff. The PCA no longer has to model the structural origin differences; it can focus purely on the variation of the decay shape (exponential steepness, possible humps, etc.). In practice this often reduces the required number of components from 10 to 3–5 per group.

## 2. The prefactoring framework

Write the full reconstruction as:

\[
f_{\text{type}}(r) \;=\; \chi(u) \;\cdot\; u^{m_{\text{type}}} \;\cdot\; \sum_{k=1}^{L} w_k^{\text{pair}}\; \Phi_k(u)
\]

where:
- \(u = r/R_{\text{cut}}\)
- \(\chi(u)\) is a smooth cutoff envelope, e.g. \((1-u^2)^n\) with \(n\ge 2\) (your \((1-u^2)^2\) is a good choice – it forces \(f, f', f''\) to zero at \(R_{\text{cut}}\)).
- \(u^{m_{\text{type}}}\) handles the correct power‑law behaviour at the origin.
- \(\Phi_k(u)\) are **global basis functions** that are shared across *all* pair types and all integral channels. They only need to represent the remaining shape variation.
- \(w_k^{\text{pair}}\) are the 4–8 weights stored per element pair and per integral channel.

Because both \(\chi\) and \(u^m\) are trivially computed (no lookups), the GPU evaluation remains extremely light.

## 3. Why avoid exponentials (and how polynomials do it better)

Your objection to \(\exp(-\beta r)\) is spot‑on:
- The exponential function has infinite support and must be forced to zero by a cutoff, leading to either a discontinuity or a complicated blending region.
- The GPU evaluation of `exp()` is slower than a few multiplications and additions.
- Storing different decay constants \(\beta\) per pair would increase memory.

Your alternative—using the finite‑support approximant \((1 - \beta x/n)^n\)—is exactly the bridge between exponential physics and polynomial efficiency. When \(n\) is an integer (e.g., 2, 3, 4), the expression becomes a pure polynomial, naturally hitting zero at \(x = n/\beta\). If we set \(n\) globally (say, \(n=2\) or 4) and absorb the decay steepness into the polynomial expansion inside the sum, we avoid storing any exponential parameters. The polynomial basis \(\Phi_k\) simply learns to reproduce the needed effective decay shape by linear combination. This is far more GPU‑friendly.

## 4. Why shifted Legendre polynomials + cutoff weight are a good match

You asked about **Jacobi / Shifted Legendre with cutoff weight**. Here’s why they fit the problem:

- **Shifted Legendre polynomials** \(\tilde{P}_k(u)\) on \([0,1]\) are orthogonal with respect to **unit weight** \(w(u)=1\). This means the least‑squares projection onto them directly minimises the integrated squared error \(\int_0^1 (f - \hat{f})^2 du\). When we later apply PCA on the Legendre expansion coefficients, the resulting principal components inherit this L²‑optimality. That aligns perfectly with RMSE being your primary metric.

- Chebyshev polynomials minimise the *maximum* error (minimax), but they are orthogonal under \(1/\sqrt{u(1-u)}\), which heavily over‑weights the endpoints. The PCA would try very hard to reduce errors exactly at \(u=0\) and \(u=1\)—places where the function is either multiplied by a known prefactor or forced to zero anyway. A unit weight is more physically reasonable.

- **Jacobi polynomials** \(P_k^{(\alpha,\beta)}(2u-1)\) are a generalisation that can be tailored with a weight \(u^\beta (1-u)^\alpha\). By choosing, for instance, \(\alpha = \beta = 0\) you get Legendre; with \(\alpha=\beta=-\frac12\) you get Chebyshev. We could even integrate the envelope into the weight: if we fit the *reduced* function \(g(u)\) with weight \((1-u)^{2n}\), we automatically down‑weight the tail. However, doing a simple unweighted projection onto shifted Legendre, then applying the envelope *after*, is both simpler and numerically stable. The envelope itself handles the tail anyway; we don’t need to weight the fit further.

So **shifted Legendre** is a robust, well‑conditioned basis for representing the smooth part \(g(u)\) after factoring out \(u^m\) and \(\chi(u)\). The polynomials can be evaluated via a three‑term recurrence, costing just a handful of FMA instructions—perfect for GPU.

## 5. GPU implementation: only one basis

Your concern about separate basis matrices is critical. By using the type‑prefactor approach, we can indeed use **one global set of basis functions** \(\Phi_k(u)\) for *all* channels. The only per‑type difference is the integer exponent \(m\) and the sign (spσ changes sign). The GPU kernel then looks like:

```c
float u = r / Rc;
float envelope = (1.0f - u*u)*(1.0f - u*u);  // (1-u²)²
float prefactor = powf(u, m);                 // u^m, m=0,1,2
float sum = dot_product(weights, evaluate_legendre_basis(u, L));
float f = sign * envelope * prefactor * sum;
```

The Legendre evaluation can be unrolled for small \(L\) (e.g., L=4–6) with hard‑coded recurrence. The `powf` for \(m=0,1,2\) can be replaced by `1.0`, `u`, `u*u`. Branching is minimal. The entire set of global basis coefficients (which define \(\Phi_k\) as linear combinations of Legendre polynomials) is tiny—maybe a few hundred floats—and fits comfortably in constant memory. The per‑pair weights are just L floats per channel; for L=4 and 8 channels, that’s 32 floats per pair, easily fitting in registers or fast shared memory.

## 6. The complete proposed pipeline (refined)

1. **Separate channels**: ssσ, spσ, ppσ, ppπ. (For d‑orbitals you’d add more groups.)
2. **For each channel group**, define the fixed prefactor \(u^m\) (and optional sign factor).
3. **Preprocess all curves**: map to \(u\)-grid, divide by \(u^m\), then divide by \((1-u^2)^2\)? Actually, better: we keep the multiplication structure, so we fit the reduced function \(h(u) = f(u) / [ (1-u^2)^2 u^m ]\) directly. But near \(u=0\), division by \(u^m\) is safe because the function vanishes as \(u^m\); however, we can avoid division by doing a least‑squares fit in the full product space. A robust approach:
   - Set up a basis of functions \(\psi_{k}(u) = (1-u^2)^2 u^m \tilde{P}_k(u)\).
   - For each curve, solve linear least squares for coefficients \(c_k\) in this basis (this automatically respects the origin and cutoff).
   - Collect the coefficient vectors across all curves, then perform PCA on them (optionally normalising each vector by its RMS to separate scale from shape).
   - The resulting principal components are linear combinations of \(\psi_k\)—therefore they are also of the form \((1-u^2)^2 u^m \Phi_l(u)\), where \(\Phi_l\) is a polynomial (a combination of Legendre polynomials). The shared part \(\Phi_l\) is what we store globally.
4. **Global basis**: store the coefficients that define \(\Phi_l(u)\) in the Legendre basis.
5. **Per‑pair storage**: for each integral channel, store a scale factor \(s\) (if normalisation was used) and the weights \(w_1...w_L\).

This yields a representation where the dominant singular values drop off rapidly; I’d expect 4–5 components to capture >99.9% of the variance, with RMSE well below \(10^{-3}\) for most cases.

## 7. What about the numerical blow‑up you saw earlier?

The earlier failures came from dividing by \((1-u)^2\) **after** interpolation and then performing PCA directly on the raw function values. That amplified tail noise enormously. In the new framework, division is never performed explicitly; the product basis already contains the envelope. The fit is a linear least‑squares problem in a well‑conditioned polynomial basis, so no amplification occurs. The PCA acts on the fit coefficients, not on the function grid, further insulating it from pointwise instabilities.

## Next steps

You can test this strategy on your existing data set with a small Python prototype:  
- Build the product basis \(\psi_{k}\) for each \(m\).  
- Fit each curve, collect coefficients.  
- Run PCA on the coefficient matrix for each channel group (or even pooled together to see if a single basis suffices).  
- Inspect the singular values—they should decay exponentially.  
- Reconstruct and measure per‑curve RMSE.

I’m confident you’ll see a dramatic reduction in the number of components needed, and the GPU evaluation will be lean and branch‑uniform. Let me know if you’d like to go deeper into any of these points—for example, the optimal choice of the Legendre degree vs. the number of PCA components, or how to handle the sign change in spσ within the same basis.

---

# Kimi 2.6

I've read the full updated document. The consensus from all models is clear: **prefactor + common basis is the right approach**. Let me synthesize this into an actionable plan.

## Refined Plan: Minimal-GPU-Memory SK Compression

### The Core Insight

Instead of separate PCAs and separate bases per channel type, we use **one master orthogonal basis** $\tilde{P}_k(u)$ and **one set of global principal components** $Q_l(u)$ for ALL channels. The only type-specific part is a cheap integer-power prefactor $u^m$ applied at the very end.

### Mathematical Framework

For any channel type $t \in \{ss\sigma, sp\sigma, pp\sigma, pp\pi\}$:

$$f_t(u) = \chi(u) \cdot u^{m_t} \cdot \sum_{l=1}^{L} w_{l}^{\text{pair}} \cdot Q_l(u)$$

where:
- $u = r / R_{\text{cut}}$
- $\chi(u) = (1-u^2)^2$ — smooth cutoff, forces $f, f', f'' \to 0$ at $u=1$
- $m_t$: $0$ for ss, $1$ for spσ, $2$ for ppσ/ppπ
- $Q_l(u)$: **global basis functions shared by all types** (defined below)
- $w_l$: per-pair weights (**L floats per channel** — the only per-pair storage)

### Why This Minimizes GPU Memory

| What | Size | Where |
|------|------|-------|
| $Q_l$ definition (U matrix $K \times L$) | ~50–80 floats | `__constant__` memory |
| Per-pair weights ($w_l$) | L floats per channel | registers / local |
| Type prefactor $u^m$ | 0–2 multiplies | inline |
| **No separate bases** | — | — |

For L=5, K=10, 8 channels: **~200 bytes global + ~200 bytes per pair**. vs. ~80 KB raw SK table.

### How We Build the Global Basis $Q_l(u)$

This is the critical part. We do **PCA in coefficient space**, not on raw grid values:

**Step 1 — Master basis**
Define shifted Legendre polynomials $\tilde{P}_k(u)$ on $[0,1]$ for $k=0..K-1$ (with $K=10$–$12$). These are orthogonal under flat weight $w(u)=1$, numerically stable, and evaluated via 3-term recurrence.

**Step 2 — Fit each curve in the product basis**
For curve $j$ of type $m_j$ with sampled values $f_j(u_i)$:

Build design matrix: $\quad A_{ik} = \chi(u_i) \cdot u_i^{m_j} \cdot \tilde{P}_k(u_i)$

Solve least squares: $\quad \min_{c_j} ||f_j - A \cdot c_j||^2$

This is a **direct fit** — no division by $\chi$ or $u^m$, so no numerical blow-up.

**Step 3 — Separate scale from shape**
Each curve has a different overall magnitude. Factor:
$$c_j = s_j \cdot \hat{c}_j \quad \text{where} \quad ||\hat{c}_j||_2 = 1, \quad s_j = ||c_j||_2$$

Store $s_j$ separately. The normalized vectors $\hat{c}_j$ capture pure shape.

**Step 4 — PCA on shape coefficients**
Collect all $\hat{c}_j$ into a $K \times M$ matrix (M = total curves). Run SVD:

$$\hat{C} = U \Sigma V^T$$

Keep top $L$ columns of $U$ (call them $u_1, \dots, u_L$). Each is a $K$-dimensional vector.

**Step 5 — Define global basis functions**
$$Q_l(u) = \sum_{k=0}^{K-1} u_{l,k} \cdot \tilde{P}_k(u)$$

These are smooth polynomials of degree $\le K-1$, shared globally.

**Step 6 — Per-pair compressed storage**
For curve $j$, the weights are:
$$w_{j,l} = s_j \cdot (u_l^T \cdot \hat{c}_j) = s_j \cdot \beta_{j,l}$$

Store only $w_{j,1..L}$ (L floats). The scale is absorbed.

### GPU Kernel (Conceptual)

```c
float eval_sk(float u, int m, float* w, int L, float* U, int K) {
    // 1. Evaluate shifted Legendre P_0..P_{K-1} via recurrence
    float P[12];
    shifted_legendre(u, P, K);  // ~2K FMAs
    
    // 2. Compute Q_l(u) = sum_k U[k,l] * P_k
    float Q[8];
    for (int l = 0; l < L; l++) {
        Q[l] = 0;
        for (int k = 0; k < K; k++) Q[l] += U[k*L + l] * P[k];
    }  // ~K*L FMAs
    
    // 3. Dot with weights
    float sum = 0;
    for (int l = 0; l < L; l++) sum += w[l] * Q[l];
    
    // 4. Apply envelope and nodal prefactor
    float chi = (1 - u*u) * (1 - u*u);
    float prefactor = (m == 0) ? 1.0f : (m == 1) ? u : u*u;
    return chi * prefactor * sum;
}
```

Total: ~2K + KL + L ≈ **50–80 FMAs per evaluation**. No memory loads except weights.

### Why This Beats Previous Approaches

| Issue | Old Approach | New Approach |
|-------|-----------|------------|
| Division blow-up | Divided by $(1-u)^2$ near $u=1$ | No division — direct fit in product basis |
| Multiple bases | Separate PCA per channel type | One global $Q_l$ basis, type via $u^m$ prefactor |
| Chebyshev tail | 16 coeffs needed, poor near cutoff | Legendre on smooth residual, K=10 sufficient |
| GPU divergence | Different code paths per channel | Same recurrence for all, only final multiply differs |
| Origin nodes | PCA wastes components on nodes | Nodes factored out by $u^m$, PCA sees only smooth shape |

### Expected Accuracy

From preliminary data:
- **S-integrals** (overlap): Very smooth. Expect **L=3–4** for 99.9% variance.
- **

## Refined Plan: Single Global Basis with Type Prefactors

I've read the updated document. All models converge on the same strategy. Here's the cleanest formulation:

### Core Formula (One Basis for All Channels)

$$f_t(u) = \chi(u) \cdot u^{m_t} \cdot \sum_{l=1}^{L} w_{l}^{\text{pair}} \cdot Q_l(u)$$

- $\chi(u) = (1-u^2)^2$ — cutoff envelope
- $m_t = 0,1,2$ for ss, spσ, ppσ/ppπ respectively
- $Q_l(u)$ — **single set of global basis functions** (shared by ALL channels)
- $w_l$ — **L floats per channel** (only per-pair storage)

### How to Build It (PCA in Coefficient Space)

**Step 1 — Master basis:** shifted Legendre $\tilde{P}_k(u)$ on $[0,1]$, degree $k=0..K-1$ (with $K=10$).

**Step 2 — Direct fit (NO division):**
For curve $j$ with type $m_j$:
$$\min_{c_j} \left\| f_j(u_i) - \sum_k c_{j,k} \cdot \underbrace{\chi(u_i) u_i^{m_j} \tilde{P}_k(u_i)}_{\psi_{k}(u_i)} \right\|^2$$

This is a standard least-squares problem. No blow-up because we never divide.

**Step 3 — Separate scale from shape:**
$$c_j = s_j \cdot \hat{c}_j, \quad ||\hat{c}_j|| = 1, \quad s_j = ||c_j||$$

**Step 4 — PCA on normalized coefficients:**
Collect all $\hat{c}_j$ into matrix $\hat{C}$ ($K \times M$). SVD:
$$\hat{C} = U \Sigma V^T$$

Keep top $L$ columns of $U$.

**Step 5 — Global basis functions:**
$$Q_l(u) = \sum_{k=0}^{K-1} U_{k,l} \cdot \tilde{P}_k(u)$$

**Step 6 — Per-pair weights:**
$$w_{j,l} = s_j \cdot \Sigma_{l} \cdot V_{j,l}$$

### Why This Is Minimal on GPU

| What | Size | Location |
|------|------|----------|
| $U$ matrix ($K \times L$) | ~50–80 floats | `__constant__` memory |
| Per-pair weights ($w_l$) | L floats/channel | registers |
| Legendre recurrence | 2K FMAs | inline |
| No separate bases | 0 | — |

For L=4, K=10, 8 channels: **~200 bytes global + ~160 bytes per pair** vs ~80 KB raw.

### What to Test

1. **Clustering plots** — overlay all curves per group (ss/sp/pp) to verify they look similar after dividing by $u^m$
2. **Fit each curve** in the product basis $\psi_k$, collect coefficient vectors
3. **PCA on coefficients** (not raw grid values) — plot singular values per group and combined
4. **Reconstruction accuracy** — test L=3,4,5,6 with K=8,10,12
5. **Compare to raw Chebyshev** — measure total storage vs accuracy tradeoff

### Key Advantage

The old approach PCA'd raw function values, so the PCs had to learn nodes, scales, and cutoff behavior. Here the **product basis encodes all the "known" structure** (nodes via $u^m$, cutoff via $\chi$). The PCA only learns the **smooth residual shape** — which is highly compressible. Expect **L=3–4 for S, L=4–5 for H**.

Ready to implement? The critical change is fitting in the product basis and PCA'ing coefficients instead of grid values.

---

## Detailed Fitting Procedure

### Least-Squares Fit in Product Basis

For a single curve $f(u)$ sampled at points $u_i$ ($i=1..N$), we fit in the product basis:

$$\psi_k(u) = \chi(u) \cdot u^m \cdot \tilde{P}_k(u)$$

where $\tilde{P}_k$ are shifted Legendre polynomials. The design matrix is:

$$A_{ik} = \psi_k(u_i) = \chi(u_i) \cdot u_i^m \cdot \tilde{P}_k(u_i)$$

The least-squares problem is:

$$\min_{\mathbf{c}} \| \mathbf{f} - A \mathbf{c} \|^2$$

The solution is given by the normal equations (or via QR/SVD for stability):

$$\mathbf{c} = (A^T A)^{-1} A^T \mathbf{f}$$

In practice, we use `numpy.linalg.lstsq` which handles rank deficiency and returns the residual norm.

### Dyadic Basis Fitting

For the dyadic basis, we use:

$$\psi_n(u) = \chi(u) \cdot (1-u)^{2^n}$$

where $n$ starts from $n_{\text{start}}$ (typically $n_{\text{start}}=2$ to skip the broad $p_0, p_1$ terms). The design matrix is:

$$A_{in} = \psi_n(u_i) = \chi(u_i) \cdot (1-u_i)^{2^n}$$

The least-squares solution is identical in form:

$$\mathbf{c} = (A^T A)^{-1} A^T \mathbf{f}$$

The dyadic basis is generated recursively: $p_{n+1} = p_n \cdot p_n$, where $p_n = (1-u)^{2^n}$.

### Fitting Region Restriction

To avoid numerical issues near $u=0$ where curves may blow up, we restrict fitting to $u > u_{\text{min}}$ (typically $u_{\text{min}}=0.1$). This is done by masking:

$$\text{mask}_i = \begin{cases} 1 & \text{if } u_i > u_{\text{min}} \\ 0 & \text{otherwise} \end{cases}$$

The fit is performed only on the masked subset of points.

---

## Dyadic Basis Enhancement Concerns

### Current Status

The pure dyadic basis $\psi_n(u) = \chi(u) \cdot (1-u)^{2^n}$ works reasonably well for capturing the main features of SK curves. The recursive structure $p_{n+1} = p_n^2$ provides excellent resolution near the core region ($u \approx 0$) where higher-order terms decay rapidly.

### Potential Improvements

However, the dyadic basis alone may lack flexibility for fine-tuning. Two approaches could be considered:

#### 1. Product of Two Linear Combinations

$$f(u) \approx \left( \sum_{n} a_n \psi_n(u) \right) \cdot \left( \sum_{k} b_k \phi_k(u) \right)$$

where $\psi_n$ are dyadic basis functions and $\phi_k$ are another polynomial basis (e.g., Legendre). This remains a **linear fit** in the coefficients if we pre-compute the outer product:

$$\Psi_{nk}(u) = \psi_n(u) \cdot \phi_k(u)$$

Then:

$$f(u) \approx \sum_{n,k} c_{nk} \Psi_{nk}(u)$$

The coefficient matrix $C$ has size $(N_{\text{dyadic}} \times N_{\text{poly}})$, which may be large. PCA on $C$ could reduce dimensionality.

#### 2. Rational Approximation (Pade)

$$f(u) \approx \frac{\sum_{n} a_n \psi_n(u)}{\sum_{k} b_k \phi_k(u)}$$

This is **not linear** in the coefficients. Fitting requires non-linear least squares (e.g., Levenberg-Marquardt). The computational cost is significantly higher, and convergence is not guaranteed.

### Efficiency Considerations

- **Linear combination of two bases**: Efficient via pre-computed outer product, but may increase the number of coefficients. PCA can help reduce dimensionality.
- **Pade approximant**: Requires iterative non-linear optimization. Not suitable for large-scale fitting of many curves. May be useful for individual fine-tuning but not for the compression pipeline.

### Recommendation

Start with the **product of two linear combinations** approach. Use a small number of terms from each basis (e.g., 4 dyadic + 4 Legendre = 16 combined basis functions), then apply PCA on the combined coefficient matrix to reduce to $L$ global components. This maintains linearity while adding flexibility.

---

# Gemini

Your idea of using a **dyadic basis** $\sim (1-u)^{2^n}$ is clever for this physics problem. Because $2^n$ grows exponentially ($4, 8, 16, 32$), these functions become incredibly sharp and localized near the origin ($u \to 0$) while flattening out to zero long before reaching the cutoff. This gives you a natural multi-scale "wavelet-like" resolution to capture the steep behavior near the atomic core.

Let's dissect your brainstorms regarding **products**, **Padé (rational) approximations**, and how to stay within a **linear fitting regime** while keeping the GPU footprint at exactly 8 coefficients.

## 1. Padé Approximants vs. The GPU Reality

A Padé approximant $f(u) \approx P(u)/Q(u)$ is exceptionally good at mimicking exponentials and sharp decays with very few coefficients. However, for your specific GPU pipeline, it introduces three major headaches:

1. **Non-linear fitting:** As you noted, you lose linear least-squares. You have to run Levenberg-Marquardt for every single element pair, which is prone to getting stuck in local minima and drastically slows down your preprocessing pipeline.
2. **The Pole Problem:** If $Q(u)$ happens to cross zero anywhere in the domain $u \in [0, 1]$ for *any* element pair, your function flies to infinity. Preventing poles during a global compression optimization is notoriously difficult.
3. **GPU Division:** Floating-point division is significantly slower on GPUs than multiplication and addition (MAD instructions)

## 2. Product of Polynomials: The Linear Loophole

Your idea of multiplying a dyadic polynomial by a smooth fine-tuning polynomial is incredibly powerful, and **it does not have to be non-linear.**

If you define a product of two expansions:


$$f(u) \approx \left( \sum_{n=1}^4 a_n \psi_n(u) \right) \times \left( \sum_{k=1}^4 b_k \phi_k(u) \right)$$

If both $\mathbf{a}$ and $\mathbf{b}$ are free parameters unique to each element pair, this is a non-linear bilinear problem. **But we can fix that.** What if the "fine-tuning" part $\phi_k(u)$ belongs to a **global, shared basis set** (like your Shifted Legendre polynomials) that is exactly the same for *all* curves?

Let's say $\Phi_{\text{shared}}(u) = \sum_{k=1}^4 \beta_k \phi_k(u)$ is a single, globally optimized smooth background profile. Then, each specific element pair only needs to find its own 4 unique dyadic coefficients $a_n$ to scale it:


$$f_{\text{pair}}(u) \approx \Phi_{\text{shared}}(u) \cdot \sum_{n=1}^4 a_n \psi_n(u)$$

Because $\Phi_{\text{shared}}(u)$ is known, you just pre-multiply your design matrix by it. The fitting problem for the pair remains **100% linear**, solving for the 4 coefficients $a_n$ using standard, lightning-fast linear least squares.

## 3. The Quadratic Optimization Trick (The Linear Combination Matrix)

If you want the product to be completely flexible (where both the dyadic part and the smooth part can change per element pair), you can look at it as an **outer product matrix** that you linearize.

Let's expand the full product:


$$f(u) = \sum_{n=1}^4 \sum_{k=1}^4 a_n b_k \left[ \psi_n(u) \phi_k(u) \right]$$

Let $c_{nk} = a_n b_k$. This creates a matrix $\mathbf{C}$ of 16 coefficients. If we treat all 16 coefficients as independent, the problem is completely linear!

1. You build a combined design matrix of 16 basis functions: $W_{i, (n,k)} = \psi_n(u_i) \phi_k(u_i)$.
2. You fit your curves linearly to find the 16 weights in $\mathbf{C}$.
3. Now, to compress those 16 numbers down to your target of **8 coefficients**, you use SVD (Rank-1 or Rank-2 approximation) directly on the $4 \times 4$ matrix $\mathbf{C}$ for each pair.

By taking the Singular Value Decomposition of $\mathbf{C}$:


$$\mathbf{C} \approx \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T + \sigma_2 \mathbf{u}_2 \mathbf{v}_2^T$$

You can perfectly represent those 16 coefficients using just 2 vectors of length 4. That is exactly $4 + 4 = 8$ coefficients!

### On the GPU:

Instead of evaluating a 16-term linear combination, the GPU evaluates:


$$f(u) = \left(\sum_{n=1}^4 U_n \psi_n(u)\right) \cdot \left(\sum_{k=1}^4 V_k \phi_k(u)\right)$$


Where $\mathbf{U}$ and $\mathbf{V}$ are the 4 dyadic and 4 smooth coefficients stored per pair. This requires only 8 floating-point registers per pair, keeps the math restricted to cheap multiplications, and utilizes the linear fitting regime during preprocessing.

## 4. Summary of the 4+4 Setup

| Step | Method | Why it fits your constraints |
| --- | --- | --- |
| **Fitting** | Linear Least Squares on a 16-element combined basis ($W = \psi \cdot \phi$). | Retains 100% stable, fast linear solver; no Levenberg-Marquardt needed. |
| **Compression** | Low-rank SVD truncation of the resulting $4 \times 4$ coefficient matrix. | Mathematically forces the 16 parameters down to exactly 8 parameters ($4+4$) with minimal error. |
| **GPU Execution** | Evaluate 4 dyadic terms, evaluate 4 Legendre terms, multiply the results. | Zero divisions, perfectly balanced registers, utilizing fast GPU ALU loops. |

---

# Implementation Notes & Numerical Results

## Code Consolidation and Refactoring

The Python analysis codebase was consolidated from 24 scattered scripts into a clean 4-module structure:

### New Module Layout (`rust_dftb/tools/sk_compress/`)

| Module | Purpose | Contents |
|--------|---------|----------|
| **`sk_utils.py`** | I/O, parsing, masking | `SkTable`, `parse_skf()`, `load_sk_folder()`, `resample_curve()`, `trim_constant_prefix()`, `CHANNELS`, `KINDS`, `prepare_channel()`, `fit_all_channels()` |
| **`fitting.py`** | Core math | `envelope()`, basis builders (dyadic, Legendre, Chebyshev, Hermite, B-spline, monomial), all fitters with Tikhonov regularization, `compress_coeffs_svd()`, conditioning/correlation analysis |
| **`plotting.py`** | Visualization | All matplotlib code: `plot_basis_functions()`, `plot_correlation_matrix()`, `plot_mio_cluster()`, `plot_conditioning()`, `plot_tikhonov_results()`, etc. |
| **`test_cli.py`** | Unified CLI | 16 subcommands: `basis`, `conditioning`, `correlation`, `orthogonal`, `tikhonov`, `fewer-dyadic`, `coeff-matrix`, `basis-selection`, `pareto`, `element-classes`, `dyadic-sequences`, `dyadic-comparison`, `normalization`, `rc-sweep`, `mio-full`, `compare-bases` |

**23 legacy scripts** moved to `legacy/` for reference (not deleted):
- `sk_parser.py`, `resample.py` → `sk_utils.py`
- `pca_compress.py` → `fitting.py`
- All `analyze_*.py`, `test_*.py`, `plot_*.py` → replaced by `test_cli.py` subcommands

### CLI Usage

```bash
cd rust_dftb/tools/sk_compress
python test_cli.py basis              # plot basis functions
python test_cli.py conditioning       # analyze conditioning
python test_cli.py compare-bases      # compare fine-tuning bases
python test_cli.py --help             # list all commands
```

---

## Numerical Results: Fine-Tuning Basis Comparison

We compared five different fine-tuning bases combined with the same dyadic envelope basis `ψ_n(u) = χ(u)·(1-u)^(2^n)` (n=2,3,4,5, start_n=2). All fits used degree=3 (4 fine-tuning functions per dyadic term, 16 total basis functions), Tikhonov λ=1e-12, tested on the full mio-1-1 basis set (36 element pairs, 288 channels).

### Results Table

| Basis | Mean RMSE | Median RMSE | Better than Legendre | Worse | Condition Number |
|-------|-----------|-------------|----------------------|-------|------------------|
| **Legendre** P_k(2u-1) | 8.84e-06 | 7.00e-06 | — (baseline) | — | 1.12e+11 |
| **Chebyshev** T_k(2u-1) | **8.54e-06** | **6.70e-06** | 284 / 288 (98.6%) | 4 | 1.30e+11 |
| Monomial u^k | 1.23e-05 | 8.91e-06 | 46 / 288 (16%) | 242 | 1.42e+11 |
| Hermite-like exp(-4(u-0.5)²)(u-0.5)^k | 1.46e-05 | 1.03e-05 | 49 / 288 (17%) | 239 | 2.20e+11 |
| B-spline cubic | 6.07e-05 | 1.83e-05 | 60 / 288 (21%) | 228 | **1.93e+08** |

### Key Findings

1. **Chebyshev T_k(2u-1) is the winner**: ~3.3% lower mean RMSE than Legendre, with nearly identical conditioning. The minimax property (uniform error distribution) gives a slight edge for envelope-weighted fitting.

2. **Legendre is excellent and nearly equivalent**: Only 3.3% worse than Chebyshev, with marginally better conditioning. Both are excellent choices.

3. **Monomial u^k performs poorly**: ~40% higher RMSE. The physical intuition (u⁰~s, u¹~p, u²~d angular momentum) is sound in principle, but the dyadic basis already captures this via the `u^m` prefactor in the product basis. The monomials are all positive on [0,1] and highly correlated with the dyadic envelope.

4. **Hermite-like basis is worse**: Gaussian-windowed centered monomials are too localized around u=0.5 and cannot capture the steep core behavior near u→0 where SK functions vary most.

5. **B-spline has exceptional conditioning** (cond ~2e8, ~580× better than Legendre) but 7× worse accuracy with only 4 cubic splines on [0,1]. The localized support of B-splines means too few basis functions overlap with the critical core region. More knots/degree would likely improve this.

### Recommendation

- **Use Chebyshev T_k(2u-1) as the default** fine-tuning basis for new work — marginally best accuracy.
- **Legendre P_k(2u-1) remains fully viable** — the difference is small (~3%) and Legendre has slightly better conditioning.
- **Avoid monomial and Hermite-like** bases for this envelope-weighted compression task.
- **B-splines may be worth revisiting** with a finer knot sequence (e.g., 8+ knots, non-uniform spacing concentrating near u=0) to leverage their excellent conditioning while maintaining accuracy.

### Physical Interpretation

The combined basis structure is:

$$W_{nk}(u) = \chi(u) \cdot (1-u)^{2^n} \cdot \phi_k(u)$$

where:
- **Dyadic term** `(1-u)^(2^n)` provides multi-scale resolution for the steep core (n=2,3,4,5 → exponents 4,8,16,32)
- **Fine-tuning term** `φ_k(u)` provides smooth, oscillatory corrections
- **Envelope** `χ(u) = (1-u)²` enforces smooth cutoff at u=1

Chebyshev's advantage comes from its **minimax property**: it distributes approximation error uniformly across [0,1], while Legendre tends to concentrate slightly more error near the boundaries. For envelope-weighted fitting where the envelope suppresses the u≈1 region anyway, this gives Chebyshev a slight edge.

---

## Dyadic Basis Orthogonalization (Gram-Schmidt)

### Experiment: Forward vs Backward Gram-Schmidt on Dyadic Basis

Motivation: The dyadic basis functions `p_n(u) = (1-u)^(2^n)` are highly correlated (condition ~1e11 for the combined basis). We tested Gram-Schmidt orthogonalization to see if it improves conditioning while preserving interpretability.

#### Implementation

Two variants tested:
- **Forward**: Start from smoothest function `p_2 = (1-u)^4`, orthogonalize subsequent functions against all previous ones. Preserves the long-range / smooth behavior.
- **Backward**: Start from sharpest function `p_5 = (1-u)^32`, orthogonalize previous functions against it. Preserves the core / steep behavior.

#### Results

| Metric | Original | Forward GS | Backward GS |
|--------|----------|------------|-------------|
| **Condition number** | 1.12e+11 | 1.15e+11 | ~1.0e+11 |
| **Mean RMSE** | 8.84e-06 | **8.25e-06** | ~8.3e-06 |
| **Median RMSE** | 7.00e-06 | 5.84e-06 | ~5.9e-06 |
| **Better than original** | — | 205 / 288 (71%) | ~70% |

#### Key Finding: Accuracy improved, conditioning did NOT

**Why conditioning didn't improve:**

The combined basis is `W_{nk}(u) = χ(u) · q_n(u) · P_k(u)`. Even when `q_n` are orthonormal, the full product basis is **not orthogonal** because the envelope `χ(u)` couples everything:

$$\langle W_{nk}, W_{mj} \rangle = \sum_i \chi_i^2 \, q_n(u_i) q_m(u_i) P_k(u_i) P_j(u_i)$$

This does **not** factor into $\langle q_n, q_m \rangle \cdot \langle P_k, P_j \rangle$. To get true orthogonality, you'd need to Gram-Schmidt the **entire 16-function set** — but then the SVD coefficient compression (dyadic × Legendre → 8 coefficients) loses its meaning.

**Why accuracy improved:**
Removing dyadic redundancy produces a more balanced basis where each function captures distinct features. The orthogonalized dyadic spreads information more evenly, making the least-squares fit better conditioned in the useful subspace.

#### Conclusion

- Orthogonalizing just the dyadic part improves accuracy slightly (~7% mean RMSE) but does **not** fix the conditioning problem.
- The dyadic structure itself is the source of both the multi-scale resolution AND the ill-conditioning. This is a fundamental trade-off.
- To fix conditioning without losing structure, one would need a different approach (e.g., orthogonalize the full combined basis and accept 16 stored coefficients, or use a different parametrization).

---

## Custom Linear Basis Experiment

### Idea

Replace the dyadic×fine-tuning product with a simpler **6-function linear basis** that enforces C2 continuity at Rc:

$$\phi(u) = (1-u)^4 \cdot \{1, u, u^2, u^4, r, r^2\}$$

where `r = u·Rc`. The `(1-u)^4` factor ensures `f, f', f'' = 0` at `u=1` (Rc). The rationale was that a single steep envelope with low-order polynomials might suffice.

### Results

| Metric | Dyadic×Legendre | Custom Linear |
|--------|-----------------|---------------|
| **Mean RMSE** | 8.84e-06 | **6.95e-03** |
| **Median RMSE** | 7.00e-06 | 6.20e-03 |
| **Condition number** | 1.12e+11 | **2.16e+18** |

The custom basis is **~800× worse in RMSE** and has a condition number of **2e18** (numerically singular).

### Why it fails

1. **(1-u)^4 decays too fast**: The envelope vanishes as (1-u)^4, much steeper than the dyadic basis which spans (1-u)^4 through (1-u)^32. SK radial functions require multi-scale resolution — the steep core needs high-exponent terms, the smooth tail needs low-exponent terms.

2. **No multi-scale resolution**: The custom basis is just low-order polynomials under a single steep envelope. It cannot simultaneously represent:
   - The exponential-like decay near the core (needs sharp features)
   - The smooth tail near the cutoff (needs broad features)

3. **Redundancy and near-singularity**: The terms `{1, u, u^2, u^4, r, r^2}` are nearly linearly dependent under the steep (1-u)^4 envelope, producing a near-singular Vandermonde-like matrix (condition ~2e18).

4. **C2 continuity is too restrictive**: Enforcing f, f', f'' = 0 at Rc with just a single (1-u)^4 factor consumes all the envelope's degrees of freedom. The dyadic basis achieves smooth cutoff through a product of independent terms, leaving flexibility for fitting.

### Conclusion

A simple polynomial under a steep envelope is **fundamentally inadequate** for SK radial functions. The dyadic basis's exponential cascade of exponents (4, 8, 16, 32) is essential because it provides **wavelet-like multi-scale resolution** — something monomials under a single envelope cannot replicate. This confirms that the combined dyadic×Legendre (or Chebyshev) structure is the correct approach.

---

## Basis Visualization

The `cmd_basis` CLI command now generates four plots:

1. **`basis_functions.png`**: Original 3-panel — Legendre, dyadic, combined basis
2. **`legendre_contribution.png`**: Raw products, enveloped products, and Legendre polynomials separately
3. **`basis_functions_orthogonalized.png`**: **New** — Original dyadic, forward Gram-Schmidt (preserves smoothest p₂=v⁴), backward Gram-Schmidt (preserves sharpest p₅=v³²), and combined basis for each
4. **`basis_functions_alternative.png`**: **New** — All fine-tuning bases: Legendre, Chebyshev, Hermite-like, Monomial, B-spline, and the custom linear basis

These visualizations confirm:
- Forward GS preserves the smooth long-range behavior and makes higher terms oscillatory
- Backward GS preserves the sharp core behavior and makes lower terms oscillatory
- The custom linear basis functions are all steep and clustered near u=0, with no multi-scale structure


---

# DeepSeek-V4-pro

You are absolutely right: making the factors \(F_j\) orthogonal to the dyadic set \(D_i\) does **nothing** for the product basis’s condition. The products \(D_i F_j\) are what matter, and their linear dependencies arise from the fact that both sets, when multiplied, can approximate the same shapes. The correct way to improve the basis is to design \(F_j\) such that the **product set** \(\{D_i F_j\}\) is as linearly independent as possible under the weighted inner product \(\langle g,h\rangle = \int \chi^2 g h\,du\). I’ll now address your two requests directly:

1. A fitting procedure that gives stable **4+4 coefficients** despite the product basis’s ill‑conditioning.
2. Ideas for a **better fine‑tuning basis** that reduces the product‑space redundancy, thus improving the 4+4 model’s accuracy and maybe lowering the required number of PCA components later.

## 1. Stable fitting for the separable model (respecting 4+4 output)

You already have a brilliant work‑around: use the full 16‑function product basis to get a noise‑free, well‑conditioned fit, then project back to the separable form. That works perfectly with **truncated SVD** and guarantees the conditioning is handled once, offline.

### The exact recipe (no regularisation knob)

For each training curve \(f(u)\):

1. **Build the \(N\times 16\) design matrix** \(\mathbf{A}\) with entries  
   \[
   A_{k, (i,j)} = \chi(u_k)\, D_i(u_k) F_j(u_k),
   \]  
   where \(D_i(u) = (1-u)^{2^{n_i}}\) and \(F_j\) are your chosen fine‑tuning functions (Legendre/Chebyshev).

2. **Compute the pseudoinverse via truncated SVD**  
   \[
   \mathbf{U}, \mathbf{s}, \mathbf{V}^\mathsf{T} = \operatorname{svd}(\mathbf{A}) \quad \text{(full matrices=False)}
   \]  
   \[
   \text{rcond} = 10^{-10} \times \max(\mathbf{s}), \qquad
   \mathbf{s}^+ = \begin{cases} 1/s_k & s_k > \text{rcond} \\ 0 & \text{otherwise} \end{cases}
   \]  
   \[
   \mathbf{c} = \mathbf{V} \cdot (\mathbf{s}^+ \odot (\mathbf{U}^\mathsf{T} f)).
   \]  
   This gives a 16‑vector \(\mathbf{c}\) that solves the linear least‑squares problem in the 16‑product space without amplification of noise. The threshold \(10^{-10}\) is purely numerical; it drops directions that are essentially invisible to the data.

3. **Reshape \(\mathbf{c}\) into a \(4\times 4\) matrix \(C\)** (rows = dyadic index, columns = fine‑tuning index).

4. **Force the separable structure** \(C \approx a b^\mathsf{T}\) by taking the dominant singular value of \(C\):  
   \[
   U_c, \Sigma_c, V_c^\mathsf{T} = \operatorname{svd}(C) \quad (\text{a 4×4 SVD, trivial})
   \]  
   \[
   a_i = U_c(i,0)\,\sqrt{\Sigma_{0,0}}, \qquad
   b_j = V_c(j,0)\,\sqrt{\Sigma_{0,0}}.
   \]  
   (Scale factors can be absorbed however you like; the product remains invariant up to a multiplicative constant.)

5. **Optional polishing**: Use these \(a,b\) as initial values for 2–3 iterations of alternating least squares (ALS) on the true separable model:  
   - Fix \(b\), solve linear least‑squares for \(a\) with basis \((\sum_j b_j F_j(u)) D_i(u)\).  
   - Fix \(a\), solve for \(b\) with basis \((\sum_i a_i D_i(u)) F_j(u)\).  
   Each step is a 4‑parameter linear least‑squares, stable because the combined basis is a single set of 4 functions.

**Result:** For every curve you obtain a pair of 4‑vectors \((a, b)\). The whole process is deterministic, no hyperparameter tuning, and completely sidesteps the ill‑conditioned design matrix of the naive product‑basis fit.

After this, you have your desired **8 coefficients per curve**. You can then perform PCA on these 8‑vectors to find global correlations and further compress to, say, 4–6 weights per pair.

## 2. Reducing product‑space redundancy: designing a better \(F_j\)

The linear dependence you observe among the 16 products comes from the fact that many combinations \((i,j)\) produce similar shapes. For example, a slower dyadic decay compensated by a polynomial that goes to zero near \(u=0\) can mimic a faster dyadic decay with a different polynomial. To make the 4+4 model maximally expressive, we want the 16 product functions to span their space with **minimal collinearity**.

How to achieve that? We can **directly optimise the fine‑tuning set \(F_j\)** to make the product matrix well‑conditioned.

### A practical, offline procedure

We want four functions \(F_j(u)\) that, when multiplied by the fixed dyadic functions \(D_i(u)\), produce a 16‑function basis whose Gram matrix (under the envelope‑weighted inner product) has a low condition number. Equivalently, the matrix \(\mathbf{B}\) (size \(N_{\text{grid}} \times 16\)) should have a rapidly decaying singular value spectrum (no near‑zero singular values).

We can formulate this as a **dictionary learning** problem:

- **Given**: fixed \(D_i\) and a large candidate pool of smooth functions \(\{\phi_k\}\) (e.g., the first 20 shifted Legendre polynomials, or a fine‑grained B‑spline basis).
- **Find**: a subset of 4 functions \(F_j\) (or linear combinations from the pool) that maximise the smallest singular value of the weighted product matrix \(\mathbf{B}\).

This can be done via a greedy orthogonalisation:

1. Form the set of all products \(\mathcal{P} = \{D_i \phi_k\}\). This is \(4\times 20 = 80\) candidate product functions.
2. Compute their weighted Gram matrix (or directly work with the design matrix on a dense grid) and perform a **pivoted QR decomposition** (column‑pivoting) to select a maximally independent subset of 16 functions from \(\mathcal{P}\).  
3. From these 16, try to factor them into \(D_i\) times new \(F_j\). Since the \(D_i\) are fixed, the selected products give you, for each selected \(i\), a set of \(\phi_k\) that worked well. You can then define \(F_j\) as the ones that appear most frequently in the selected set, or solve a least‑squares problem to find \(F_j\) such that the span of \(\{D_i F_j\}\) approximates the span of the selected 16 functions.  
4. A simpler heuristic: **Directly build a new \(F_j\) as the first four left singular functions of the matrix \(\mathbf{B}\) when the rows are arranged by dyadic index.** That is, consider the SVD of the whole product basis and then perform a **higher‑order SVD (HOSVD)** to extract the principal components of the fine‑tuning direction. The leading singular vectors in the fine‑tuning mode will be the optimal \(F_j\) for representing the data in a separable manner.

But we can take an even more direct route inspired by the **canonical polyadic (CP) decomposition** of the function data matrix. Suppose we collect all training curves (after channel prefactor and envelope) into a matrix \(\mathbf{Y}\) of size \(N \times M\). We want to approximate each column as a sum of \(R\) separable products \((a_r^\mathsf{T} D) \cdot (b_r^\mathsf{T} F)\). If \(R=1\), that’s exactly your 4+4 model with one product. With \(R>1\), it would become a sum of several such product pairs, requiring more coefficients. To keep only 4+4, we restrict to \(R=1\) and treat the whole dataset globally.

**Optimal 4+4 basis via simultaneous CP decomposition:**  
Solve  
\[
\min_{D_i, F_j, a_{ir}, b_{jr}} \sum_{m=1}^M \left\| \mathbf{y}_m - \sum_{r=1}^R \Big( \sum_i a_{ir} D_i \Big) \odot \Big( \sum_j b_{jr} F_j \Big) \right\|^2
\]  
with \(D_i\) fixed to your dyadic functions, \(R=1\), and \(F_j\) being the unknowns together with per‑curve coefficients. This is a non‑convex problem but can be handled by alternating optimisation:  
- Initialize \(F_j\) with shifted Legendre.  
- For each curve, compute \(a,b\) using the robust SVD+rank‑1 method above.  
- Then, keeping all \(a,b\) fixed, **update the functions \(F_j\)** to minimise the total error. This step is a linear least‑squares problem in the values of \(F_j\) on the grid (or in the coefficients of a polynomial expansion). Because \(F_j\) appear in products \(\sum_i a_i D_i \cdot \sum_j b_j F_j\), the update decouples for each grid point \(u_k\): we have  
\[
f(u_k) \approx s(u_k) \cdot \sum_j b_j F_j(u_k), \quad \text{where } s(u_k) = \sum_i a_i D_i(u_k) \text{ (a known number for each curve)}.
\]  
So for each grid point \(u_k\), we have \(M\) equations:  
\[
f_m(u_k) \approx s_m(u_k) \cdot \sum_j b_{m,j} F_j(u_k).
\]  
This is a large but sparse linear system that can be solved for the values \(F_j(u_k)\) (or for the coefficients of a smooth expansion). Regularisation (smoothness) can be added to prevent overfitting.  
- Iterate a few times: re‑fit the per‑curve \(a,b\) with the updated \(F_j\), then re‑update \(F_j\).

This joint optimisation will directly give you a fine‑tuning basis \(F_j\) that is optimally adapted to your dataset *and* to the separable model, leading to the smallest possible reconstruction error for the 4+4 format. The resulting \(F_j\) will naturally be as independent as possible from the dyadic functions, because any redundancy would have been squeezed out by the alternating updates.

## 3. Do you actually need to change the basis?

Your results with Chebyshev + truncated SVD fitting already give **RMSE ≈ 8.5×10⁻⁶**, which is extremely good (three orders better than your original). The condition number of the 16‑function design matrix is completely circumvented by the SVD pseudoinverse. The slight redundancy among the products does not hurt the final accuracy because the SVD step captures all usable information; the rank‑1 projection then discards only directions that are essentially noise or negligible shape components. In practice, the separable model with 4+4 coefficients trained this way is already close to optimal for that parametric form.

If you still see a gap between the full 16‑coefficient fit and the rank‑1 projected fit, and you want to close it without increasing the number of coefficients, then the alternating optimisation of \(F_j\) above is the most direct way.

But before embarking on that, I’d recommend:

- Run your current stable fitting (SVD + rank‑1) on the whole dataset.
- Compute the per‑curve RMSE of the **rank‑1 separable model** vs the **full 16‑coefficient linear fit**. I suspect the difference is already tiny.
- Then run PCA on the resulting 8‑coefficient vectors; you may find that only 4–5 components are needed for even higher compression.

At that point, the 4+4 model with Chebyshev fine‑tuning may already meet all your accuracy and memory requirements. The linear dependence is a numerical nuisance, not a fundamental barrier—and the SVD trick has removed that nuisance completely.

**In summary:**  
- Use **truncated SVD + rank‑1 projection** (and optional ALS polishing) to get stable 4+4 coefficients; no more regularisation tuning.  
- If you want to push further, optimise the fine‑tuning functions \(F_j\) by iteratively minimising the total dataset error over both the per‑curve coefficients and the functions \(F_j\) themselves. This will yield a set of four polynomials that are truly complementary to the dyadic ones under the product structure.  
- But honestly, the numbers you already have (RMSE ~8e-6) are so low that the current basis may be good enough. The “room for improvement” from redundancy is likely small, and the simpler path is to lock the basis as Chebyshev and move on to the PCA compression.