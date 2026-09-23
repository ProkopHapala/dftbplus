https://chatgpt.com/share/6ab3a286-fd08-83ed-8f13-9fc833c6facb

---

# USER

can you help me find decent SK-tables for DFTB for Si-H nanocrystals and eventually also Si-O-H and maybe even later Si-C-O-H (but that has leas priority) ?

I have currently two pbc-0-3 and matsci-0-3, but they does not seem so good especially matsci-0-3 has band gap 5eV which is nonsesne for silicion, I do not know what it is good for?



I want them for two purposes
1\) vibrations of Si nanocrystals pasivated hy hydrogn (maybe later with some oxidation but not priority now) - so optimize geometry and then get some hessian
2\) maby later some electronic structure (exicted states - something we were doing long time ago with fireball (https://journals.aps.org/prb/abstract/10.1103/PhysRevB.87.195420)
(again this is not priority now)

So mostly I want basis/SK tables which reporuce correcly hessian/vibration spectra or phonon-bands of Si (bulk) but also Si-H pasivation (surface) 

---

there are some notes 

matsci is not a silicon band-structure set that failed. It was built for something else, and elemental silicon is outside that job. For an Si–H particle in this code, pbc-0-3 is the set that was actually made for silicon.

The matsci file names its authors and its purpose: Frenzel, Oliveira, Jardillier, Heine and Seifert, TU Dresden, 2004–2009, “Slater-Koster tables … for materials science simulations.” The silicon table is stored as a partner of Si–O, Si–N, Si–C, Si–H, Si–Al, Si–P, Si–Na and Si–Cu. The repulsive note says it was not fitted to atomization or reaction energies. The target was inorganic materials, silica and related compounds, not the diamond lattice and not the silicon gap. A 2.24 Å bond and a 5 eV gap in pure silicon are what you get when you ask that table a question it was not fitted to answer. For Si–O chemistry it may still be the set people use. For elemental silicon it is the wrong tool.

Sets on disk that contain both Si and H:

| set | made for | relaxation | this solver |
|---|---|---|---|
| pbc-0-3 | Sieck, Paderborn, 2000. The file says “OK for bulk silicon and clusters,” and warns that the gap is wrong because the basis is only s and p. The extra d onsite was adjusted for the Si–O–Si angle in quartz. | yes | yes. Bulk Si–Si came out 2.364 Å against 2.352 Å, gap 1.68 eV against about 1.1 eV. |
| matsci-0-3 | inorganic materials, Si with O, N, C, … | yes, but the Si–Si minimum is 4.6% too short | yes, and the answer is not silicon |
| siband-1-1 | Markov et al., IEEE TED 62, 696 (2015): bands, dielectric response, transport in oxidised silicon. The repulsive spline is marked “DO NOT USE FOR RELAXATION.” The basis is 3s 3p 3d. | no | no. The sparse path stores four orbitals per silicon, s and p only. |

hyb-0-2 has Si–Si and no hydrogen. 3ob and mio have no silicon at all.

So for an Si–H system here, pbc-0-3 is the one possibility that is both documented for bulk silicon and clusters and able to relax in this solver. siband is the electronic-structure set, and it cannot be used for a relaxation or on the current four-orbital path.


---

# Chat GPT 5.6 sol

Yes. I searched specifically for parameterizations that make sense for **Si nanocrystals with H termination**, rather than just “a set containing Si and H.” There is one particularly interesting set I had not expected to find: **SiGeH**, distributed with BIOVIA Materials Studio, which was explicitly fitted to bulk Si *and* silicon hydrides. That may actually be the closest existing parameterization to what you want. ([Dassault Systèmes][1])

My current picture is:

| set                             | Si bulk geometry/phonons     | Si–H                         | Si–O–H                 | bands              | usable in your current 4-orbital Si code? |
| ------------------------------- | ---------------------------- | ---------------------------- | ---------------------- | ------------------ | ----------------------------------------- |
| **pbc-0-3**                     | **good candidate**           | questionable / needs testing | plausible              | mediocre           | **yes**                                   |
| **SiGeH**                       | **very promising**           | **very promising**           | no O                   | likely better      | **no: Si is sp³d⁵**                       |
| **matsci-0-3**                  | poor choice for elemental Si | not target                   | **yes, target domain** | not for Si         | probably yes                              |
| **siband**                      | no relaxation                | electronic Si–H only         | electronic Si/SiO₂     | **best candidate** | **no: Si has d**                          |
| old Frauenheim Si–H set (~2004) | apparently good              | **demonstrated vibrations**  | no                     | old/minimal        | apparently **yes, sp**                    |

So for what you want **right now**, I would actually keep `pbc-0-3` as the baseline, but investigate/replace the **Si–H part** rather than abandoning the whole set.

### 1. `pbc-0-3`: surprisingly defensible for the Si Hessian

The documentation says explicitly that its silicon parametrization is intended for **bulk Si and silicon clusters**, with an \(sp\) basis. It also explicitly warns that the electronic gap is wrong because of the minimal basis. ([GitHub][2])

More importantly for your purpose, I found a recent direct test. Tammo van der Heide's 2024 DFTB thesis uses `pbc-0-3` for bulk Si lattice dynamics and shows the **full silicon phonon dispersion against neutron-scattering experiment**. The text explicitly says previous work found pbc-0-3 to give “reasonable phonon dispersions and vibrational properties.” The comparison was made from a relaxed \(8\times8\times8\) Si supercell. ([HAL Browser][3])

That is a vastly more relevant test for you than the band gap.

In fact, this illustrates an important distinction:

$$
E_{\rm DFTB}
=
E_{\rm band}
+
E_{\rm SCC}
+
E_{\rm rep},
$$

while your object of interest is

$$
K_{I\alpha,J\beta}
=
\frac{\partial^2E_{\rm DFTB}}
{\partial R_{I\alpha}\partial R_{J\beta}}.
$$

A bad conduction band or bad \(E_g\) does **not** imply bad curvature of the ground-state Born–Oppenheimer surface. The repulsive potential was fitted precisely to repair total-energy/force behavior not captured correctly by the minimal electronic model.

So I would currently rate **pbc Si–Si as quite a reasonable choice for your vibration problem**.

### 2. The weak point is actually `pbc` Si–H

Here I found something less reassuring. The detailed metadata in the distributed PBC Si–H file says it was obtained from Zoltán Hajnal and:

> “Should be regarded as dummy file for saturating bonds.”



That is very different from saying “fitted to Si–H vibrational spectroscopy.”

So I would **not assume that the present `pbc-0-3/Si-H.skf` Hessian is trustworthy**, even though it may put H at approximately the correct bond length.

There is, however, a fascinating historical result.

### 3. There was an old DFTB Si–H parameterization that actually did vibrations quite well

Ranber Singh and coworkers used the Frauenheim DFTB scheme for hydrogenated amorphous silicon and Si clusters. They used

$$
{\rm Si}: sp,\qquad {\rm H}:s
$$

with interaction ranges approximately

$$
R_{\rm SiSi}=4.8\,a_0,\qquad
R_{\rm SiH}=3.2\,a_0.
$$

They explicitly validated small silanes before doing the nanostructures. ([ResearchGate][4])

For example, their calculated SiH\(_n\) frequencies are remarkably sensible. For SiH they obtained about \(1982\ {\rm cm^{-1}}\) versus \(2042\ {\rm cm^{-1}}\) experiment. For SiH\(_2\), roughly \(983,2008,2036\ {\rm cm^{-1}}\) versus experimental \(999,1993,1996\ {\rm cm^{-1}}\). For SiH\(_4\), their stretching modes were about \(2206-2238\ {\rm cm^{-1}}\), close to measured Si–H stretches around \(2190-2270\ {\rm cm^{-1}}\). They explicitly concluded that the SiH\(_n\) vibrational frequencies were in good agreement with experiment before applying the method to larger hydrogenated Si clusters. ([Electronics and Books][5])

Even better, the same parametrization was used for a **216-Si + 24-H amorphous-Si:H system**, including vibrational density of states, and produced the Si–H bond distribution centered around \(1.52\) Å. ([ResearchGate][6])

The annoying part is that **I have not yet found an official surviving download of those exact SKFs**. And there is evidence that they are *not exactly identical* to the current pbc Si–H file: the papers specify a Si–H range of \(3.2a_0\), whereas the current distributed file differs and labels itself a dummy saturation potential.

This old set is therefore something I think is **worth hunting down explicitly**. It is especially attractive for you because it uses precisely the \(sp_{\rm Si}+s_{\rm H}\) basis your present sparse code can handle.

### 4. The most interesting set I found: **SiGeH**

This may be the real prize if we are willing eventually to support \(d\) orbitals.

BIOVIA's own DFTB documentation describes a `SiGeH` set specifically for **semiconductors containing Si, Ge and H**. It uses an

$$
sp^3d^5
$$

basis for silicon and was fitted against:

* Si, Ge, and SiGe solids,
* Si\(_2\)H\(_6\), Si\(_2\)H\(_4\),
* Ge analogues,
* SiGeH\(_6\), SiGeH\(_4\),
* **SiH\(_4\)**,
* H\(_2\),

and then tested on **solids, nanowires, nanoclusters and Si vacancies**. ([SlideServe][7])

That training set is almost comically close to your application.

So for **Si–H nanocrystals**, among all sets I found, this is the one whose *training philosophy* I like best.

The catch is important: it is apparently a **Materials Studio/BIOVIA parameter library**, not one of the freely downloadable DFTB.org sets. I did not find an official public SKF download. BIOVIA still lists SiGeH as one of its supplied libraries. ([Dassault Systèmes][1])

And for our Rust sparse solver it immediately creates another problem:

$$
1s + 3p \quad\longrightarrow\quad 1s+3p+5d
$$

so Si goes from **4 to 9 orbitals**.

I would nevertheless try hard to obtain this set, even initially just to run it in standard DFTB+/Materials Studio as a benchmark. If it dramatically beats pbc for SiH\(_4\), bulk phonons, and a small H-terminated nanocrystal, then we know whether supporting \(d\) orbitals is actually worth doing.

### 5. `matsci`: I agree with your diagnosis, with one qualification

`matsci` is not really an elemental-silicon parameterization that happens to have failed. Its documented repulsive fits involve things such as

* Al–Si–O–H **imogolite**,
* Cu–Si–Al–Na–O–H **zeolites**,
* Si/P/N/O/C/H materials,

etc. ([GitHub][8])

So a ridiculous bulk-Si gap is not particularly surprising. More importantly for *your* problem, you already observe that its equilibrium Si–Si distance is substantially too short. **That**, unlike the electronic gap, directly tells us to distrust its Si force constants around diamond Si.

I therefore would not use matsci for the Si core.

For a later **oxidized Si surface**, however, I would bring it back into the comparison because Si–O environments are much closer to its fitting domain.

### 6. `siband`: useful later, but almost exactly the opposite of what you need now

`siband` is very interesting for your second application. It was constructed for the electronic/dielectric properties of

* bulk Si,
* SiO\(_2\),
* very thin oxidized Si,
* H-passivated Si(100),

and explicitly targets oxidized nanoscale Si devices. ([GitHub][9])

But:

* Si uses \(spd\),
* **there is no fitted repulsive potential for structural relaxation**,
* its O–H interaction is explicitly described as fictitious/not optimized.

So:

$$
\boxed{\text{siband: electronic structure yes; Hessian absolutely no.}}
$$

For your eventual optical/excited-state work, though, I would definitely benchmark it. It is much more conceptually appropriate there than pbc's minimal-\(sp\) Hamiltonian.

---

## What I would do now

I think we can turn this into a very small, decisive benchmark instead of continuing to browse parameter names.

**Stage A — bulk Si.** Compare `pbc-0-3`, `matsci-0-3`, and ideally SiGeH for \(a_0\), \(C_{11},C_{12},C_{44}\), and the phonon dispersion at least along \(\Gamma-X-L-\Gamma\). In particular the optical \(\Gamma\) mode should come out near the familiar \(\sim520\ {\rm cm^{-1}}\). I expect pbc to win over matsci here.

**Stage B — Si–H.** Do SiH, SiH\(_2\), SiH\(_3\), and especially SiH\(_4\). That gives us bond lengths, bends and stretches with essentially zero computational cost. The old DFTB paper already provides a very convenient target table. Then perhaps Si\(_2\)H\(_6\), because that simultaneously probes Si–Si and Si–H.

**Stage C — actual surface environment.** A tiny H-terminated Si cluster, perhaps Si\(_{10}\)H\(_{16}\), Si\(_{29}\)H\(_{36}\), or whatever cluster closest resembles the particles you care about. Run one PBE reference Hessian and compare **mode-projected force constants** rather than blindly RMS-matching ordered eigenfrequencies, because modes will mix.

This test would tell us within a few dozen calculations whether current `pbc` is already sufficient.

And if `pbc` gives good bulk phonons but bad Si–H stretches, I would **not immediately search for a whole new SK set**. I would seriously consider fitting a new

$$
V_{\rm rep}^{\rm Si-H}(R)
$$

on top of the existing pbc electronic SK tables.

That is quite attractive here. The Si–Si part is already good; we only need to repair the local surface passivation. Fit \(V_{\rm rep}^{SiH}\) simultaneously to PBE forces for SiH\(_4\), Si\(_2\)H\(_6\), several distorted hydrides, and a couple of H/Si-surface configurations. Include displaced geometries deliberately so that the fit constrains

$$
V'_{\rm rep}(R),\qquad V''_{\rm rep}(R),
$$

not merely equilibrium energy and bond length.

The limitation is that a pair repulsion can fix radial curvature very effectively but **cannot repair bad angular electronic physics**. So SiH\(_2\)/SiH\(_3\) bending modes are the diagnostic: if stretches are wrong but bends are sensible, refitting \(V_{\rm rep}^{SiH}\) is likely enough. If bends are badly wrong too, I would go after **SiGeH / an \(spd\) Hamiltonian** instead.

### For oxidation later

For Si–O–H I would benchmark **pbc versus matsci** rather than decide now. `magsil` does not really give us a new solution: DFTB.org says it *requires matsci* and essentially adds Mg interactions for chrysotile, so without Mg it does not solve our problem. ([dftb.org][10])

For eventual **Si–C–O–H**, `pbc` and `matsci` both at least have consistent coverage; DFTB.org explicitly warns that SK files from different parameter families should generally **not be mixed**. ([dftb.org][10]) So I definitely would not make a Frankenstein `pbc-SiSi + matsci-SiO + 3ob-OH` set.

The most promising hierarchy therefore looks like this:

$$
\boxed{
\begin{array}{ll}
\textbf{now, vibrations:} & \text{pbc-0-3 + carefully validate Si-H}\\[2mm]
\textbf{best Si-H set to acquire/test:} & \text{BIOVIA SiGeH}\\[2mm]
\textbf{interesting lost/legacy option:} & \text{2004--2008 Frauenheim }sp\text{ Si-H parameters}\\[2mm]
\textbf{oxidation later:} & \text{pbc vs matsci benchmark}\\[2mm]
\textbf{electronic structure later:} & \text{siband (and SiGeH)}
\end{array}}
$$

The **SiGeH discovery changes my view considerably**: before writing our own parameterization, I would first try to get those files and benchmark them. They were practically designed for “Si solid + hydrogenated nanowire/nanocluster.”

---

# Measured (DFTB+, 2026-09-23)

SiGeH is a BIOVIA Materials Studio library. There is no public `.skf` download. The 2004 Frauenheim Si–H files were not found as a separate archive either. The public set that was missing from disk is **PTBP** (Cui, Reuter, Margraf, [JCTC 2024](https://doi.org/10.1021/acs.jctc.4c00228)), now listed on dftb.org. The Si/H files are in `/home/prokop/SIMULATIONS/dftbplus/slakos/ptbp/`. The table is the 20-column s+p format (the file lists a `3d0` shell, but no d integrals are stored), so it can run in this solver. It was not run through the sparse path.

DFTB+ SCC, Si = p, H = s. Bulk: primitive diamond cell, lattice optimisation, 8×8×8 shifted k-mesh, then the Γ Hessian. SiH₄: conjugate-gradient minimum, then the Hessian. Frequencies from the mass-weighted Hessian. Experiment: Si a = 5.431 Å, nearest neighbour 2.352 Å, Γ optical phonon ~520 cm⁻¹. SiH₄ bond 1.480 Å; modes ν₄ 913, ν₂ 972, ν₁ 2187, ν₃ 2189 cm⁻¹.

| set | bulk a (Å) | Si–Si (Å) | Γ optical (cm⁻¹) | Si–H (Å) | SiH₄ bends T₂, E | SiH₄ stretches A₁, T₂ |
|---|---:|---:|---:|---:|---|---|
| experiment | 5.431 | 2.352 | ~520 | 1.480 | 913, 972 | 2187, 2189 |
| pbc-0-3 | 5.460 | 2.364 | 505 | 1.491 | 774, 882 | 2185, 2218 |
| matsci-0-3 | 5.183 | 2.244 | 1251 | 1.477 | 857, 991 | 2318, 2346 |
| PTBP | 5.550 | 2.403 | 475 | 1.489 | 776, 884 | 2030, 2069 |

pbc is the only one of the three that is close on both the silicon phonon and the Si–H stretch. Its bends are soft by about 100–140 cm⁻¹. matsci’s SiH₄ bond and bends look acceptable, and its Γ phonon is more than twice the measured 520 cm⁻¹, consistent with the 4.6% short bulk bond. PTBP is a whole-periodic-table baseline: the bulk bond is 2.2% long, the phonon is 475 cm⁻¹, and the stretches sit ~160 cm⁻¹ low. The pbc Si–H file does say it is a dummy saturation potential, and the stretch frequencies it produces are still the ones nearest experiment.

[DFTB.org parameter-set downloads](https://www.dftb.org/parameters/download.html?utm_source=chatgpt.com)
[Official pbc parameter repository](https://github.com/dftbparams/pbc?utm_source=chatgpt.com)
[Official siband parameter repository](https://github.com/dftbparams/siband?utm_source=chatgpt.com)
[Official matsci parameter repository](https://github.com/dftbparams/matsci?utm_source=chatgpt.com)

If you want, next I can **dig specifically for the actual SiGeH `.skf/.skflib` provenance and the old 2004 Si–H files**—authors, old DFTB archives, supplementary material, Git/SVN history, etc. That is probably more useful now than surveying another twenty generic DFTB sets.

[1]: https://www.3ds.com/assets/invest/2023-10/biovia-material-studio-dftb-datasheet.pdf?utm_source=chatgpt.com "HOW DOES MATERIALS STUDIO DFTB+ WORK?"
[2]: https://github.com/dftbparams/pbc?utm_source=chatgpt.com "GitHub - dftbparams/pbc: DFTB parameter set for calculating solids and surfaces in the self-consistent charges (SCC) model · GitHub"
[3]: https://media-api.suub.uni-bremen.de/api/core/bitstreams/8c231928-efc0-4f42-82fa-91f6b67a6d2d/content?utm_source=chatgpt.com "Hybrid Functionals for Periodic Systems in the Density Functional Tight-Binding Method"
[4]: https://www.researchgate.net/publication/231059206_Effect_of_hydrogen_on_ground_state_properties_of_silicon_clusters_Si_nHm_N_11-15_m_0-4_A_density_functional_based_tight_binding_study?utm_source=chatgpt.com "(PDF) Effect of hydrogen on ground state properties of silicon clusters (Si nHm; N = 11-15, m = 0-4): A density functional based tight binding study"
[5]: https://electronicsandbooks.com/edt/manual/Magazine/J/Journal%20of%20Physics%20Condensed%20Matter/2008%20Volume%2020/0953-8984_20_4_045226.pdf?utm_source=chatgpt.com "Effect of hydrogen on ground state properties of silicon clusters (SinHm; n=11-15, m=0-4):
a density functional based tight binding study"
[6]: https://www.researchgate.net/publication/239278671_Sample_dependence_of_the_structural_vibrational_and_electronic_properties_ofa-SiH_A_density-functional-based_tight-binding_study?utm_source=chatgpt.com "(PDF) Sample dependence of the structural, vibrational, and electronic properties ofa−Si:H: A density-functional-based tight-binding study"
[7]: https://www.slideserve.com/romeo/introduction-to-dftb?utm_source=chatgpt.com "PPT - Introduction to DFTB+ PowerPoint Presentation, free download - ID:1597418"
[8]: https://github.com/dftbparams/matsci?utm_source=chatgpt.com "GitHub - dftbparams/matsci: DFTB parameter set providing a collection of some sets used for various problems in materials science · GitHub"
[9]: https://github.com/dftbparams/siband?utm_source=chatgpt.com "GitHub - dftbparams/siband: DFTB parameter set providing electronic parameters for accurate silicon and silicon dioxide band structures · GitHub"
[10]: https://www.dftb.org/parameters/download.html?utm_source=chatgpt.com "Download — dftb.org"
