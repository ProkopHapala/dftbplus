
# USER

I just learned that xTB don't interpolate Skater Köster tables and instead have some analytical formulas for mixing atom involving their electronegativity and hardness.

Now I'm doing super fast AFM simulation engine. And there I realized electronegativity and hardness (in fact Fukui function) are very useful concepts to estimate interaction fast. 

Do you think it would be derive or at least justify this (put it in rigorous math ground) this approximative model as some limit (1st order weak interaction) of solution system with such xTB mixing Hamiltonian?

---

# Gemini-3.5-flash

Yes, you can absolutely ground this rigorously. What you are describing—using electronegativity, hardness, and Fukui functions to directly estimate intermolecular interactions—is the exact domain of **Conceptual Density Functional Theory (CDFT)**.

When you apply CDFT to a weak-interaction limit (like an AFM tip approaching a sample surface), you can derive these chemical reactivity indicators directly from a Taylor expansion of the electronic energy, which maps beautifully onto the mathematical structure of xTB.

Here is the mathematical and physical justification for how your AFM interaction model can be derived as a first-order approximation of an xTB-like Hamiltonian.

---

## 1. The Energy Expansion (The Foundation of CDFT)

In DFT, the total energy $E[N, v]$ is a functional of the number of electrons $N$ and the external potential $v(\mathbf{r})$ (exerted by the nuclei). If we perturb a system by changing the number of electrons ($\Delta N$) or changing the external potential ($\Delta v$), we can Taylor expand the energy:

$$\Delta E = \left( \frac{\partial E}{\partial N} \right)_v \Delta N + \frac{1}{2} \left( \frac{\partial^2 E}{\partial N^2} \right)_v (\Delta N)^2 + \int \left( \frac{\partial E}{\partial v(\mathbf{r})} \right)_N \Delta v(\mathbf{r}) d\mathbf{r} + \dots$$

By definition, these partial derivatives are the exact chemical quantities you mentioned:

* **Electronegativity ($\chi$):** $\chi = -\mu = -\left( \frac{\partial E}{\partial N} \right)_v$ (where $\mu$ is the chemical potential).
* **Chemical Hardness ($\eta$):** $\eta = \left( \frac{\partial^2 E}{\partial N^2} \right)_v$.
* **Fukui Function ($f(\mathbf{r})$):** $f(\mathbf{r}) = \left( \frac{\partial^2 E}{\partial N \partial v(\mathbf{r})} \right) = \left( \frac{\partial \rho(\mathbf{r})}{\partial N} \right)_v$, which dictates *where* the electron density $\rho(\mathbf{r})$ shifts when electrons are added or removed.

---

## 2. Connecting CDFT to the xTB Hamiltonian

In GFN-xTB, instead of pre-computed Slater-Koster tables, the Hamiltonian elements are built on-the-fly using analytical overlap integrals $S_{\mu\nu}$ multiplied by an average of the atomic environments.

The diagonal (on-site) terms of the xTB Hamiltonian are directly modulated by the charge fluctuations ($\Delta q$) on the atoms:


$$H_{\mu\mu} = \epsilon_\mu - \chi_A - \eta_A \Delta q_A - \sum_{B \neq A} V_{AB}(\Delta q_B)$$

If you treat the AFM tip ($T$) and the sample ($S$) as two distinct systems in the **weak-interaction limit**, they initially do not share electrons ($\Delta N = 0$ globally between them), but their local potentials perturb each other.

As the tip approaches, the external potential change felt by the sample is $\Delta v_S(\mathbf{r}) = v_{tip}(\mathbf{r})$. According to CDFT, the energy change of the sample to first order is:


$$\Delta E_S = \int \rho_S(\mathbf{r}) v_{tip}(\mathbf{r}) d\mathbf{r}$$


This is purely the classical electrostatic interaction (frozen densities).

### The Second-Order Response (Where Hardness and Fukui Enter)

As the tip gets closer, polarization occurs. The tip induces a charge rearrangement in the sample. The linear response of the sample's electron density to the tip's potential is governed by the **polarization kernel** $\chi(\mathbf{r}, \mathbf{r}')$.

Using a localized atom-centered basis (like xTB), this polarization energy simplifies to a charge-transfer and charge-equilibration problem between the tip atom ($T$) and the sample atom ($S$):


$$\Delta E_{pol} \approx -\frac{1}{2} \frac{(\chi_T - \chi_S)^2}{\eta_T + \eta_S - 2\gamma_{TS}(R)}$$


Where $\gamma_{TS}(R)$ is the analytical short-range damped Coulomb function used in xTB to interpolate between $1/R$ at long range and the average hardness $\frac{1}{2}(\eta_T + \eta_S)$ at short range.

---

## 3. Justifying your AFM Model: The Localized View

For a super-fast AFM engine, you can project these continuous functions onto atomic centers. The Fukui function becomes a set of atomic condensed Fukui indices ($f_A$), indicating how reactive/polarizable a specific atom on the surface is.

If you assume the AFM tip is a rigid probe with a fixed electronegativity $\chi_T$ and hardness $\eta_T$, the interaction energy with a surface atom $S$ can be modeled as:

$$E_{int}(R) = E_{rep}(R) + E_{vdW}(R) - \frac{1}{2} \frac{(\chi_T - \chi_S)^2}{\eta_T + \eta_S - 2\gamma(R)} \cdot (f_T \cdot f_S)$$

### Why this is mathematically rigorous:

1. **It matches the Taylor expansion:** It properly captures the second-order perturbation of the electronic energy without needing to diagonalize the full matrix.
2. **It naturally handles chemical specificity:** An oxygen lone pair on a surface will have a high Fukui index ($f_S$) for electrophilic attack, leading to a strong attractive polarization term with a sharp tip. A chemically inert gold surface atom will have a high hardness ($\eta_S$) and low Fukui index, correctly yielding a flatter, weaker polarization curve.
3. **It inherits xTB's scaling:** By using xTB's analytical $\gamma(R)$ function instead of a classical $1/R$ Coulomb law, your AFM engine will naturally avoid the "catastrophe" of infinite electrostatic attraction at zero distance.

## Summary for Implementation

You don't need to assemble a matrix or solve $H\psi = ES\psi$ at every pixel of your AFM scan.

Instead, you can pre-calculate or assign a triad of values to your surface atoms and tip: **$(\chi, \eta, f)$**. Your force field then evaluates a pairwise analytical formula where the electronic attraction is driven by the electronegativity mismatch $\Delta \chi$ and gated by the combined hardness and local Fukui values. This provides a rock-solid, quantum-mechanically justified bridge between empirical force fields and full tight-binding calculations.


---

# DeepSeek-V4-pro

This is an excellent question that sits at the intersection of **conceptual DFT**, **electronegativity equalisation**, and **tight‑binding SCF theory**. The short answer is **yes** – the approximate model you are thinking of *can* be rigorously derived as the first‑order (weak‑interaction) limit of a second‑order SCF Hamiltonian like xTB.

Below I build a systematic derivation path from the xTB energy expression down to a simple interaction model expressed purely in terms of atomic electronegativities, hardnesses, and Fukui functions.

### 1. The xTB Hamiltonian as a second‑order expansion

The GFN‑xTB family does **not** use tabulated Slater‑Koster integrals. Instead, the zero‑order Hamiltonian is an extended Hückel‑type expression:

\[
H_{\mu\nu}^{(0)} = \tfrac12 K_{AB} S_{\mu\nu} \bigl( h_\mu + h_\nu \bigr)
\]

where the **atomic orbital energies** \(h_\mu\) themselves depend on the **charge state of the atom** via a quadratic expansion:

\[
h_\mu(q_A) = h_\mu^{(0)} - \eta_A\, q_A + \dots
\]

Here \(h_\mu^{(0)}\) is the neutral‑atom orbital energy (related to Pauling electronegativity) and \(\eta_A\) is closely linked to the atomic chemical hardness (the second derivative of energy with respect to charge).  
Thus, the dependence of the Hamiltonian on local charges is already built in analytically.

The SCC correction adds an atom‑pair Coulomb term:

\[
E_{\text{SCC}} = \frac12 \sum_{A,B} \gamma_{AB} \, \Delta q_A \Delta q_B
\]

The total energy (up to second order in charges) then takes the form

\[
E(\{q_A\}) = \sum_A \Bigl( \chi_A q_A + \frac12 \eta_A q_A^2 \Bigr) + \frac12 \sum_{A\ne B} q_A q_B \gamma_{AB}
\]

where \(\chi_A\) is the **electronegativity** of the neutral atom and \(\eta_A\) the **hardness** (Parr–Pearson sense). This is exactly the energy functional of the **Electronegativity Equalization Method (EEM)**.

### 2. Weak‑interaction limit: two weakly coupled systems

Now imagine two subsystems – a tip (T) and a sample (S) – brought together at a distance where direct chemical bonding is negligible (the typical AFM regime). The total energy can be written as:

\[
E_{\text{tot}} = E_T(\{q_t\}) + E_S(\{q_s\}) + \sum_{t\in T}\sum_{s\in S} \gamma_{ts} \, q_t q_s
\]

If the coupling is weak (large distances, small \(\gamma_{ts}\)), we can treat the tip–sample Coulomb interaction as a perturbation. To first order in \(\gamma_{ts}\), the charge rearrangements are small and the energy shift can be computed using **linear response theory**.

### 3. Linear response and the Fukui function

In conceptual DFT, the **linear response function** of an isolated system is:

\[
\chi(\mathbf{r},\mathbf{r}') = \frac{\delta q(\mathbf{r})}{\delta v_{\text{ext}}(\mathbf{r}')}
\]

Its condensed‑atom version is intimately related to the **atom‑condensed Fukui functions** \(f_A^\pm\). The change in atomic charge upon a small external potential perturbation \(\delta v_B\) is:

\[
\delta q_A = -\sum_B f_A \,\kappa_{AB}^{-1} \, \delta v_B
\]

where \(\kappa_{AB} = \partial^2 E/\partial q_A \partial q_B\) is the **hardness kernel** (the inverse of the response function). For an isolated subsystem, \(\kappa_{AB}\) contains the atomic hardness \(\eta_A\) on its diagonal and the Coulomb interactions \(\gamma_{AB}\) off‑diagonal.

Now, the perturbation felt by the sample due to the tip’s charge distribution is a local potential \(\delta v_B \approx \gamma_{ts} q_t\). Inserting this into the linear response formula gives the **induced charge on the sample** as a linear function of the tip charges.

The resulting **interaction energy** to second order in the coupling is:

\[
\Delta E_{\text{int}} \approx \sum_{t,s} \gamma_{ts} \, q_t^{(0)} q_s^{(0)} \;-\; \frac12 \sum_{t,t'} \sum_{s,s'} \gamma_{ts} \, \chi_{ss'} \, \gamma_{s't'} \, q_t^{(0)} q_{t'}^{(0)}
\]

(and a similar term for the tip’s response). Here \(q_t^{(0)}\) is the charge of the isolated tip atom (often zero for neutral tips, but the formalism holds for ions too). The first term is the **direct Coulomb** interaction between unperturbed charges; the second term is the **charge‑polarisation** (or induction) energy.

### 4. Specialising to “only electronegativity and hardness”

For many AFM simulations, the tip is neutral (\(q_t^{(0)}=0\)) and the dominant contribution comes from the mutual polarisation of two initially neutral subsystems. The leading non‑zero term is the dispersion‑like induction energy, which depends on the **polarisability** of the atoms, which in turn is proportional to \(1/\eta_A\) (softness). In an atom‑condensed formulation, the interaction energy between two neutral but polarisable atoms becomes

\[
E_{\text{int}} \approx -\frac12 \alpha_t \alpha_s \, (\vec\nabla \gamma_{ts})^2
\]

where the atomic polarisability \(\alpha_A\) scales as \(f_A^2 / \eta_A^{\text{eff}}\). This is a well‑known result from **atom‑dipole dispersion models** but it emerges naturally from the SCC/EEM linear response. The Fukui function enters as the distribution of the charge response within the atom.

Thus, an effective AFM interaction model that uses only **local electronegativity** (to determine equilibrium charges) and **hardness** (to determine polarisability) can indeed be justified as the **first‑order weak‑coupling limit** of a full xTB or DFTB energy expression.

### 5. Steps to a rigorous mathematical derivation

A complete derivation would follow these steps:

1. Start from the full xTB/DFTB energy \(E[\{q_A\}]\) with a quadratic expansion.
2. Partition the system into two subsystems (tip and sample). Write the total energy as a sum of the isolated energies plus the cross‑term Coulomb interaction.
3. Expand the energy to second order in the cross‑term couplings \(\gamma_{ts}\) (or, equivalently, in the charge fluctuations around the ground state of the separated subsystems).
4. Use the **exact second‑order perturbation theory** for the SCF energy (as in DFTB’s response formulation or in conceptual DFT’s perturbation theory). This yields an effective interaction Hamiltonian that is a bilinear form in the charge distributions of the two subsystems.
5. Condense the atomic response into atom‑centered Fukui functions and hardness kernels. The interaction energy becomes a function solely of the atomic electronegativities (or effective chemical potentials) and hardnesses.
6. Show that in the limit of large separations, the leading term is the Coulomb interaction between the unperturbed charges (if any) plus a van‑der‑Waals‑like term proportional to the product of local softnesses.

This derivation is mathematically rigorous and has been carried out in various forms in the literature (e.g., **Mortier’s EEM + second‑order perturbation**, **Parr & Yang’s conceptual DFT**, or the **DFTB+ response method** for non‑adiabatic interactions).

### 6. Practical take‑away for your AFM engine

- You can pre‑compute or tabulate atomic **electronegativities** \(\chi_A\) and **hardnesses** \(\eta_A\) directly from the xTB parameter set (they are explicit in the GFN‑xTB paper).
- The **Fukui function** can be approximated as a simple spherical Gaussian distribution around each atom, normalised to 1, with width determined by the atomic radius. Its role is to distribute the charge response.
- The interaction at a given tip–sample atom pair can then be evaluated as:
  - Coulomb term (if charged atoms): \(E_{\text{Coul}} \approx \gamma(R) \, q_t q_s\).
  - Induction term: \(E_{\text{ind}} \approx -\tfrac12 \kappa \frac{q_t^2 \alpha_s}{R^4}\) (for a tip charge polarising a neutral sample), with \(\alpha_s \propto 1/\eta_s\) and \(\kappa\) a geometric factor.

This model is fast, physically transparent, and directly derivable from the same Hamiltonian you would otherwise use for a full quantum treatment.

**Bottom line:** The approximate model you envisage is **not just plausible** – it is the natural **first‑order weak‑interaction limit** of the xTB energy functional. You can write a rigorous derivation using conceptual DFT’s linear response theory, and the result will be an expression where the interaction is fully determined by atomic electronegativities, hardnesses, and Fukui functions. Your AFM engine would therefore be using a systematically improvable approximation: at short range you can turn on the full xTB Hamiltonian, while at long range the same parameters give you the fast analytic interaction.

---

# Kimi-2.6

Yes — and the justification is remarkably elegant. The xTB model (and DFTB SCC) can be placed on rigorous conceptual-DFT ground by recognizing that **the SCC loop is exactly the chemical-potential equalization principle**, and the **Hamiltonian electronegativity scaling is a semi-empirical ansatz for the first-order charge-transfer correction to covalent bonding**. Here is the systematic derivation.

## 1. The Rigorous Definitions (Conceptual DFT)

From density-functional theory, the chemical potential and hardness are *exact* derivatives of the energy with respect to particle number at fixed external potential:

$$
\mu = \left(\frac{\partial E}{\partial N}\right)_v = -\chi \qquad\text{(electronegativity)}
$$

$$
\eta = \left(\frac{\partial^2 E}{\partial N^2}\right)_v = \left(\frac{\partial \mu}{\partial N}\right)_v \qquad\text{(hardness)}
$$

The Fukui function is the density response to adding/removing electrons:

$$
f^{\pm}(\mathbf{r}) = \left(\frac{\partial \rho(\mathbf{r})}{\partial N}\right)_v^{\pm}
$$

These are not empirical fitting parameters — they are **rigorous reactivity descriptors**. The xTB model uses approximate numerical values for them (Mulliken/Pauling electronegativity, fitted hardness parameters), but the *mathematical structure* of xTB maps directly onto these exact quantities.

## 2. SCC as Chemical-Potential Equalization (The Core Theorem)

The xTB/DFTB total energy is expanded as a Taylor series in charge fluctuations $\Delta q_A = q_A - q_A^0$ around the neutral-atom reference density:

$$
E[\rho_0 + \delta\rho] \approx E^0[\rho_0] + \sum_A \left(\frac{\partial E}{\partial q_A}\right)_0 \Delta q_A + \frac{1}{2}\sum_{A,B} \left(\frac{\partial^2 E}{\partial q_A \partial q_B}\right)_0 \Delta q_A \Delta q_B + \dots
$$

Now identify the derivatives:
- **First derivative**: $\left(\frac{\partial E}{\partial q_A}\right)_0 = -\chi_A = \mu_A$ (the chemical potential / electronegativity of atom A)
- **Second derivative**: $\left(\frac{\partial^2 E}{\partial q_A \partial q_B}\right)_0 = \gamma_{AB}$ (the hardness matrix / charge-charge interaction kernel)

The SCC energy expression in xTB/DFTB is precisely this truncated expansion:

$$
E_{\text{SCC}} = \sum_A \chi_A \Delta q_A + \frac{1}{2}\sum_{A,B} \gamma_{AB} \Delta q_A \Delta q_B
$$

Minimizing this subject to charge conservation ($\sum_A \Delta q_A = 0$) via a Lagrange multiplier $\lambda$ gives:

$$
\frac{\partial \mathcal{L}}{\partial q_A} = \chi_A + \sum_B \gamma_{AB} \Delta q_B + \lambda = 0
$$

At the minimum, **all atoms have the same chemical potential**:

$$
\mu_A = \chi_A + \sum_B \gamma_{AB} \Delta q_B = -\lambda = \text{constant}
$$

**This is the Sanderson electronegativity equalization principle, derived from DFT variational principles.** The SCC loop in xTB/DFTB is simply an iterative solver for this exact condition. The $\gamma_{AB}$ matrix is the **hardness matrix** — in xTB it takes the Klopman-Ohno form $\gamma_{AB} = 1/\sqrt{R_{AB}^2 + \eta_{AB}^2}$, where $\eta_{AB}$ is the effective hardness.

## 3. The Hamiltonian Electronegativity Scaling as a Weak-Interaction Correction

Your specific question is about the **off-diagonal Hamiltonian** in xTB:

$$
H_{\mu\nu} = \frac{1}{2}(h_\mu + h_\nu) S_{\mu\nu} \cdot k_{AB}^{\text{eff}} \cdot \chi(\Delta\text{EN}) \cdot \Pi(R)
$$

where the electronegativity-dependent factor is typically:

$$
\chi(\Delta\text{EN}) = 1 + k_{\text{EN}}(\chi_A - \chi_B)^2
$$

This is **not** a random fitting function. It captures the **first-order charge-transfer perturbation to the covalent interaction** in the following sense:

When two atoms with very different electronegativity interact weakly (small overlap, large distance), the bonding character shifts from covalent to ionic. The **charge-transfer energy** in perturbation theory is:

$$
E_{\text{CT}}^{(1)} \approx -\frac{(\chi_A - \chi_B)^2}{2(\eta_A + \eta_B)} \cdot S_{\mu\nu}^2
$$

The xTB factor $\chi(\Delta\text{EN})$ is an empirical ansatz that mimics this physics: as $\Delta\chi \to 0$ (homonuclear), the factor $\to 1$ (pure covalent); as $\Delta\chi$ grows, the factor suppresses the covalent Hamiltonian element because the bond acquires ionic character and the covalent EHT formula becomes less appropriate.

**Rigorous justification**: In the limit of weak interaction ($S_{\mu\nu} \ll 1$), the interaction energy to first order is the electrostatic energy between unperturbed densities (computed by EEQ/SCC). The second-order term includes charge transfer, which scales as $(\Delta\chi)^2/\eta$. The xTB Hamiltonian scaling absorbs this into the **zeroth-order** Hamiltonian, making the model "pre-conditioned" for the ionic/covalent crossover without requiring explicit perturbation sums.

## 4. Weak-Interaction Perturbation Theory & AFM

For your AFM simulation, the tip-surface interaction spans three regimes:

| Regime | Distance | Dominant Physics | xTB Component |
|--------|----------|-----------------|---------------|
| **Long-range** | > 5 Å | Electrostatic, dispersion | EEQ charges + D3/D4 |
| **Medium-range** | 3–5 Å | Polarization, charge transfer | SCC γ-matrix + EN-scaling |
| **Close-range** | < 3 Å | Covalent bonding | Full EHT Hamiltonian + overlap |

In perturbation theory for weakly interacting fragments A (tip) and B (surface):

**First-order** (electrostatic):
$$
E^{(1)} = \int\int \frac{\rho_A^0(\mathbf{r})\rho_B^0(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|} d\mathbf{r}d\mathbf{r}' + V_{NN}
$$
This is exactly what the **EEQ model** computes in xTB — the self-consistent partial charges produce the first-order electrostatic energy without diagonalization.

**Second-order** (polarization + charge transfer):
$$
E^{(2)} = -\frac{1}{2}\sum_{i,j} \frac{|\langle i_A j_B | V | i_A j_B \rangle|^2}{\Delta E} + E_{\text{CT}}
$$
The polarization part is captured by the **hardness matrix** (via $\gamma_{AB}$ and the response of the density). The charge-transfer part is captured by the **electronegativity difference** driving $\Delta q$ between tip and surface atoms.

**The xTB model effectively resums these perturbation terms into a non-perturbative Hamiltonian**, but in the weak-interaction limit it correctly reduces to:
1. EEQ electrostatics (first-order)
2. Hardness-mediated polarization (second-order)
3. Electronegativity-driven charge transfer (second-order)

## 5. Fukui Function & AFM Reactivity

The Fukui function $f(\mathbf{r}) = (\partial\rho/\partial N)_v$ tells you **where** the system is most reactive to electron transfer. For AFM, this is extraordinarily useful:

- **f⁺(r)** (LUMO-like): sites that accept electrons from the tip
- **f⁻(r)** (HOMO-like): sites that donate electrons to the tip

In xTB, you can compute condensed Fukui functions approximately as:

$$
f_A^{\pm} \approx \pm \sum_{\mu \in A} \sum_{\nu} c_{\mu,\text{HOMO/LUMO}} c_{\nu,\text{HOMO/LUMO}} S_{\mu\nu}
$$

Or even faster, from the **frontier orbital coefficients** obtained from a single diagonalization. This gives you a **reactivity map** for the AFM tip without running full SCC at every scan point.

**Rigorous connection**: The Fukui function is the functional derivative of the chemical potential with respect to the external potential: $f(\mathbf{r}) = (\delta\mu/\delta v(\mathbf{r}))_N$. In the weak-interaction limit, the tip acts as a perturbation $\delta v(\mathbf{r})$, and the charge response is $\delta\rho(\mathbf{r}) \approx f(\mathbf{r}) \delta N$. The xTB SCC/EEQ procedure computes the *integrated* version of this response via the hardness matrix.

## 6. The Limitations (Where the Rigorous Ground Ends)

It is important to be honest about where the rigorous justification stops:

1. **The xTB Hamiltonian is semi-empirical**. The electronegativity scaling factor $k_{\text{EN}}$ and the polynomial damping $\Pi(R)$ are fitted, not derived from ab initio perturbation theory. The *functional form* is physically motivated, but the coefficients are empirical.

2. **The hardness matrix is approximated**. The exact hardness matrix in conceptual DFT is:
   $$
   \eta(\mathbf{r},\mathbf{r}') = \frac{\delta^2 F[\rho]}{\delta\rho(\mathbf{r})\delta\rho(\mathbf{r}')}
   $$
   which includes kinetic, Coulomb, and XC contributions. xTB replaces this with a simple Klopman-Ohno function. This is a **radical simplification** — justified by the observation that for charge-transfer between atoms, the Coulomb part dominates at long range and the XC part provides a short-range cutoff.

3. **No explicit relaxation**. The Fukui function in finite-difference form includes orbital relaxation, but xTB uses a minimal basis with frozen exponents. The condensed Fukui function from xTB is approximate.

## 7. Summary: The Answer to Your Question

**Can you justify the xTB mixing Hamiltonian as a first-order weak-interaction limit?**

**Yes — but with the right interpretation:**

- The **SCC/EEQ energy** is rigorously the **second-order Taylor expansion of the DFT energy in charge fluctuations**, truncated at the monopole level. The SCC loop solves the exact chemical-potential equalization condition. This is rigorous conceptual DFT.

- The **Hamiltonian electronegativity scaling** is an **empirical ansatz for the charge-transfer correction** to the covalent Extended Hückel interaction. It is not derived from first-principles perturbation theory, but it is **physically consistent** with the perturbative result that charge-transfer energy scales as $(\Delta\chi)^2/\eta$.

- In the **weak-interaction limit** (large tip-surface distance, small overlap), the xTB model correctly decouples into:
  1. **First-order electrostatics** (EEQ charges)
  2. **Second-order polarization/CT** (hardness matrix + electronegativity differences)
  3. **Fukui-function reactivity** (frontier orbital response)

**For your AFM engine**, this means you can rigorously justify using **EEQ alone** for fast electrostatics at large distances, and promoting to the **full xTB Hamiltonian** only when the tip enters the covalent-bonding regime. The electronegativity and hardness parameters give you the "switching function" between these regimes.

## 8. Practical Implementation for Your Rust/OpenCL Code

Given this theoretical framework, here is how you can structure your AFM engine:

```rust
// Phase 1: Fast EEQ (electronegativity equalization)
// Valid for tip-surface distances > 4-5 Å
// O(N²) but no diagonalization needed
fn compute_eeq_charges(atoms: &[Atom]) -> Vec<f64> {
    // Solve: χ_A + Σ_B γ_AB Δq_B = μ (equal for all atoms)
    // Linear system: (γ + η) · Δq = -Δχ
    // Returns partial charges for electrostatic force
}

// Phase 2: Fukui reactivity map
// Single diagonalization of surface, tip
// Identify reactive sites without full SCC
fn compute_fukui_functions(hamiltonian: &CsrMatrix) -> (Vec<f64>, Vec<f64>) {
    // f⁺ ≈ LUMO density, f⁻ ≈ HOMO density
    // Condensed per atom for AFM scan prediction
}

// Phase 3: Full xTB Hamiltonian
// For close approach (< 3 Å) or suspected bond formation
fn build_xtb_hamiltonian(atoms: &[Atom], charges: &[f64]) -> (CsrMatrix, CsrMatrix) {
    // H_μν = 0.5(h_μ + h_ν) S_μν · k_AB · (1 + k_EN·Δχ²) · Π(R)
    // Include SCC correction via γ_AB
}
```

The beauty of this approach is that **EEQ is the rigorous first-order limit** of the full xTB model, and you can derive the **convergence criterion** for switching from EEQ to full xTB based on the **maximum overlap** between tip and surface orbitals, or the **predicted charge transfer** from EEQ exceeding a threshold.

This gives you a **hierarchically justified** multi-scale AFM engine: fast EEQ for scanning, Fukui for reactivity prediction, and full xTB only where chemistry happens.