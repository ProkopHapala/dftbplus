#!/usr/bin/env python3
"""
Minimal diagnostic: single H atom STO.
- Places H at origin, projects 1s wavefunction onto a 1D grid along x-axis
- Compares: (1) direct analytic formula, (2) OCL kernel via project_orbital
- Plots: 2D image (XY at z=0) + 1D cut along x (linear + log scale)
- Purpose: verify radial exponent is correct in OCL kernel
"""
import sys, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pyBall.OCL.DFTBplusParser import parse_basis_hsd_ang, compute_sto_radial, BOHR2ANG
from pyBall.OCL.Grid import GridProjector

# ── parameters ────────────────────────────────────────────────────────────────
BOHR  = BOHR2ANG                     # 0.5292 Å
HSD   = Path(__file__).parent / 'dftb_h2o' / 'waveplot_in.hsd'
STEP  = 0.10                         # Å  — fine grid for clean 1D comparison
NPTS  = 64                           # grid points per axis
ORIGIN = np.array([-NPTS/2*STEP]*3)  # centred on atom

# ── OCL setup ─────────────────────────────────────────────────────────────────
species_list = parse_basis_hsd_ang(HSD)
print(f"Parsed basis from {HSD}:")
for sp in species_list:
    for o in sp['orbitals']:
        print(f"  {sp['name']} l={o['l']}  alpha(Å⁻¹)={o['exponents'].round(5)}  "
              f"coeff={o['coefficients'].ravel().round(5)}  cutoff={o['cutoff']:.3f} Å")

proj = GridProjector(fdata_dir=None, verbosity=0)
proj.load_basis_sto(species_list)
print(f"Basis grid: dr={proj.basis_meta['dr']:.6f} Å  n_nodes={proj.basis_meta['n_nodes']}")

# ── single H atom at origin ────────────────────────────────────────────────────
coords   = np.array([[0., 0., 0.]], dtype=np.float32)
atom_nz  = np.array([1], dtype=np.int32)
nz_to_rcut = {sp['atomic_number']: max(o['cutoff'] for o in sp['orbitals']) for sp in species_list}
rcut     = np.array([nz_to_rcut[1]], dtype=np.float32)
atoms    = {'pos': coords, 'Rcut': rcut, 'type': atom_nz}

# H 1s coefficient = 1.0, slot [px,py,pz,s] → [0,0,0,1]
coeffs   = np.array([[0., 0., 0., 1.]], dtype=np.float32)
norb_per = np.array([4], dtype=np.int32)

grid_spec = {
    'origin': ORIGIN,
    'dA': np.array([STEP, 0., 0.]),
    'dB': np.array([0., STEP, 0.]),
    'dC': np.array([0., 0., STEP]),
    'ngrid': np.array([NPTS, NPTS, NPTS], dtype=np.int32),
}

print(f"\nProjecting single H 1s onto {NPTS}³ grid (step={STEP} Å)...")
grid_ocl = proj.project_orbital(coeffs, norb_per, atoms, grid_spec)
print(f"  OCL grid max = {np.abs(grid_ocl).max():.6f}")

# ── analytic reference ─────────────────────────────────────────────────────────
# H 1s from HSD:  R(r) = coeff * r^0 * exp(-alpha * r),  psi = R(r) * Y_00 = R(r) / sqrt(4pi)
PREF_S = 1.0 / np.sqrt(4 * np.pi)   # 0.28209479
H_sp   = next(s for s in species_list if s['name'] == 'H')
H_orb  = H_sp['orbitals'][0]        # l=0
alpha  = float(H_orb['exponents'][0])
coeff  = float(H_orb['coefficients'].ravel()[0])
print(f"\nH 1s params (Ang): alpha={alpha:.6f}  coeff={coeff:.6f}  PREF_S={PREF_S:.6f}")
print(f"  psi(r=0) analytic = coeff*1*PREF_S = {coeff*PREF_S:.6f}")
print(f"  psi(r=0) OCL      = {grid_ocl[NPTS//2, NPTS//2, NPTS//2]:.6f}")

# 1D x-axis values
x_vals = ORIGIN[0] + np.arange(NPTS) * STEP   # Å
r_vals = np.abs(x_vals)                        # distance from origin along x

psi_analytic = coeff * np.exp(-alpha * r_vals) * PREF_S   # r^0, l=0

# ── extract OCL 1D cut along x at y=0, z=0 (iz=iy=NPTS//2) ──────────────────
iy0 = NPTS // 2
iz0 = NPTS // 2
cut_ocl = grid_ocl[:, iy0, iz0]           # 1D cut along x
cut_ref = psi_analytic                     # analytic (same x)

print(f"\n1D cut at iy={iy0}, iz={iz0}:")
print(f"  OCL max  = {np.abs(cut_ocl).max():.6f}  at x={x_vals[np.argmax(np.abs(cut_ocl))]:.3f} Å")
print(f"  Ref max  = {np.abs(cut_ref).max():.6f}  at x={x_vals[np.argmax(np.abs(cut_ref))]:.3f} Å")

# ── figures ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle(f"Single H atom 1s: alpha={alpha:.4f} Å⁻¹ (= {alpha*BOHR:.4f} Bohr⁻¹ = 0.967 Bohr⁻¹ expected)")

# 1. 2D image: OCL XY slice at iz=NPTS//2
ax = axes[0]
sl = grid_ocl[:, :, iz0]
clim = np.abs(sl).max() or 1e-6
im = ax.imshow(sl.T, origin='lower', cmap='RdBu_r', vmin=-clim, vmax=clim,
               extent=[ORIGIN[0], ORIGIN[0]+NPTS*STEP]*2, interpolation='bilinear')
plt.colorbar(im, ax=ax)
ax.set_title('OCL 2D (XY, z=0)')
ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)')
ax.axhline(0, c='k', lw=0.5, ls='--'); ax.axvline(0, c='k', lw=0.5, ls='--')

# 2. 2D image: analytic XY slice
ax = axes[1]
xx = np.arange(NPTS)*STEP + ORIGIN[0]
yy = np.arange(NPTS)*STEP + ORIGIN[0]
X, Y = np.meshgrid(xx, yy, indexing='ij')
R2D  = np.sqrt(X**2 + Y**2)
sl_ref = coeff * np.exp(-alpha * R2D) * PREF_S
im = ax.imshow(sl_ref.T, origin='lower', cmap='RdBu_r', vmin=-clim, vmax=clim,
               extent=[ORIGIN[0], ORIGIN[0]+NPTS*STEP]*2, interpolation='bilinear')
plt.colorbar(im, ax=ax)
ax.set_title('Analytic 2D (XY, z=0)')
ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)')
ax.axhline(0, c='k', lw=0.5, ls='--'); ax.axvline(0, c='k', lw=0.5, ls='--')

# 3. 1D linear
ax = axes[2]
ax.plot(x_vals, cut_ref, 'b-',  lw=2, label=f'Analytic α={alpha:.4f} Å⁻¹')
ax.plot(x_vals, cut_ocl, 'r--', lw=1.5, label='OCL kernel')
ax.axvline(0, c='k', lw=0.5, ls='--')
ax.set_xlabel('x (Å)'); ax.set_ylabel('ψ')
ax.set_title('1D cut along x (linear)')
ax.legend(fontsize=8)

# 4. 1D log scale — only positive x side (r > 0)
ax = axes[3]
mask = x_vals >= 0
ax.semilogy(x_vals[mask], np.abs(cut_ref[mask]), 'b-',  lw=2, label=f'Analytic α={alpha:.4f} Å⁻¹')
ax.semilogy(x_vals[mask], np.abs(cut_ocl[mask]).clip(1e-12), 'r--', lw=1.5, label='OCL kernel')
# Reference slope lines for comparison
for alpha_test, col, lab in [
    (alpha,           'b', f'slope α={alpha:.3f}'),
    (0.967,           'g', 'slope 0.967 (Bohr⁻¹, wrong)'),
    (0.967/BOHR,      'm', f'slope {0.967/BOHR:.3f} = 0.967/B (correct)'),
]:
    y0 = coeff * np.exp(-alpha_test * 0) * PREF_S
    ax.semilogy(x_vals[mask], y0 * np.exp(-alpha_test * x_vals[mask]), '--',
                color=col, lw=0.8, alpha=0.5, label=lab)
ax.set_xlabel('x (Å)'); ax.set_ylabel('|ψ| (log)')
ax.set_title('1D cut (log scale) — slope = -alpha')
ax.legend(fontsize=7)
ax.set_xlim(0, NPTS//2 * STEP)

fig.tight_layout()
out = Path(__file__).parent / 'waveplot_output' / 'sto_radial_test.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out), dpi=150)
print(f"\nSaved: {out}")
plt.show()
