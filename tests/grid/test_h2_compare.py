#!/usr/bin/env python3
"""
H2 molecule: rigorous 1D/2D comparison between WAVEPLOT cube and OCL.
Grid: identical origin/step/npts to waveplot_in.hsd.
1D cut: along z-axis through both H atoms (bond axis).
2D cut: XZ plane at y=0 (contains bond axis).
Plots both linear and log scale so exponent slope is directly visible.
"""
import sys, struct, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pyBall.OCL.DFTBplusParser import parse_basis_hsd_ang, BOHR2ANG
from pyBall.OCL.Grid import GridProjector

BOHR = BOHR2ANG
DIR  = Path(__file__).parent / 'dftb_h2'
OUT  = Path(__file__).parent / 'waveplot_output' / 'h2'
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Read WAVEPLOT cube files ───────────────────────────────────────────────
def read_cube(path):
    with open(path) as f: lines = f.readlines()
    natoms = int(lines[2].split()[0])
    origin = np.array([float(x) for x in lines[2].split()[1:4]]) * BOHR  # Bohr->Ang
    n = []; step = []
    for i in range(3):
        parts = lines[3+i].split()
        n.append(int(parts[0]))
        step.append(np.array([float(x) for x in parts[1:4]]) * BOHR)
    atom_pos = []
    for i in range(natoms):
        parts = lines[6+i].split()
        atom_pos.append([float(x)*BOHR for x in parts[2:5]])
    data_lines = lines[6+natoms:]
    vals = np.array([float(v) for line in data_lines for v in line.split()])
    # cube stores in x-major (ix outermost), z fastest
    grid = vals.reshape(n[0], n[1], n[2])
    return grid, origin, [s[i] for i,s in enumerate(step)], np.array(atom_pos)

print("Reading WAVEPLOT cube files...")
grid_wp1, origin, step_xyz, atom_pos = read_cube(DIR / 'wp-1-1-1-real.cube')
grid_wp2, _, _, _                    = read_cube(DIR / 'wp-1-1-2-real.cube')
NX, NY, NZ = grid_wp1.shape
print(f"  Grid: {NX}x{NY}x{NZ}  origin={np.round(origin,3)} Å  step={np.round(step_xyz,4)} Å")
print(f"  Atom positions (Å): {atom_pos}")
print(f"  WP MO1 max={np.abs(grid_wp1).max():.6f}  MO2 max={np.abs(grid_wp2).max():.6f}")

# ── 2. Read eigenvectors ──────────────────────────────────────────────────────
# H2: 2 atoms x 1 AO (1s) = 2 AOs, 2 MOs
with open(DIR / 'eigenvec.bin', 'rb') as f: raw = f.read()
identity = struct.unpack_from('i', raw, 0)[0]
norb, nstates = 2, 2
evecs = np.frombuffer(raw[4:], dtype=np.float64).reshape(nstates, norb)
print(f"\nEigenvectors (H2: H1_s, H2_s):")
for i in range(nstates):
    print(f"  MO{i+1}: {evecs[i]}")

# ── 3. Setup OCL projector from waveplot_in.hsd ───────────────────────────────
species_list = parse_basis_hsd_ang(DIR / 'waveplot_in.hsd')
print(f"\nSTO basis (Ang):")
for sp in species_list:
    for o in sp['orbitals']:
        print(f"  {sp['name']} l={o['l']}  alpha={o['exponents'].round(5)} Å⁻¹  "
              f"coeff={o['coefficients'].ravel().round(5)}  cutoff={o['cutoff']:.3f} Å")

proj = GridProjector(fdata_dir=None, verbosity=0)
proj.load_basis_sto(species_list)
print(f"  Basis grid: dr={proj.basis_meta['dr']:.5f} Å  n_nodes={proj.basis_meta['n_nodes']}")

# atoms: H1 at atom_pos[0], H2 at atom_pos[1]
coords   = atom_pos.astype(np.float32)
atom_nz  = np.array([1, 1], dtype=np.int32)
nz_to_rcut = {sp['atomic_number']: max(o['cutoff'] for o in sp['orbitals']) for sp in species_list}
rcut     = np.array([nz_to_rcut[1], nz_to_rcut[1]], dtype=np.float32)
atoms    = {'pos': coords, 'Rcut': rcut, 'type': atom_nz}
norb_per = np.array([4, 4], dtype=np.int32)

# DFTB AO order for H: just s. Kernel slot: [px,py,pz,s]
def make_coeffs(evec):
    c = np.zeros((2, 4), dtype=np.float32)
    c[0, 3] = float(evec[0])  # H1 s -> slot [3]
    c[1, 3] = float(evec[1])  # H2 s -> slot [3]
    return c

# Grid exactly matching cube
dA = np.array([step_xyz[0], 0., 0.])
dB = np.array([0., step_xyz[1], 0.])
dC = np.array([0., 0., step_xyz[2]])
grid_spec = {'origin': origin, 'dA': dA, 'dB': dB, 'dC': dC,
             'ngrid': np.array([NX, NY, NZ], dtype=np.int32)}

print(f"\nProjecting MO1 and MO2 via OCL...")
grid_ocl1 = proj.project_orbital(make_coeffs(evecs[0]), norb_per, atoms, grid_spec)
grid_ocl2 = proj.project_orbital(make_coeffs(evecs[1]), norb_per, atoms, grid_spec)
print(f"  OCL MO1 max={np.abs(grid_ocl1).max():.6f}  MO2 max={np.abs(grid_ocl2).max():.6f}")

# ── 4. Slice indices ──────────────────────────────────────────────────────────
# Bond axis is z. H atoms at z=±0.371 Å. Molecule centre at y=0,x=0.
ix0 = int(round((0.0 - origin[0]) / step_xyz[0]))  # x=0
iy0 = int(round((0.0 - origin[1]) / step_xyz[1]))  # y=0
iz0 = int(round((0.0 - origin[2]) / step_xyz[2]))  # z=0 (midpoint)
print(f"\nSlice indices: ix={ix0} iy={iy0} iz={iz0}")
print(f"  x at ix0={origin[0]+ix0*step_xyz[0]:.4f} Å  (want 0)")
print(f"  y at iy0={origin[1]+iy0*step_xyz[1]:.4f} Å  (want 0)")
print(f"  z at iz0={origin[2]+iz0*step_xyz[2]:.4f} Å  (want 0)")

# 1D cuts along z at x=0,y=0
z_vals  = origin[2] + np.arange(NZ) * step_xyz[2]
cut_wp1 = grid_wp1[ix0, iy0, :]
cut_wp2 = grid_wp2[ix0, iy0, :]
cut_oc1 = grid_ocl1[ix0, iy0, :]
cut_oc2 = grid_ocl2[ix0, iy0, :]

# 2D slices: XZ at y=iy0
x_vals = origin[0] + np.arange(NX) * step_xyz[0]
ext_xz = [z_vals[0], z_vals[-1], x_vals[0], x_vals[-1]]

# ── 5. Find ratio OCL/WP at atoms ─────────────────────────────────────────────
iz_H1 = int(round((atom_pos[0,2] - origin[2]) / step_xyz[2]))
iz_H2 = int(round((atom_pos[1,2] - origin[2]) / step_xyz[2]))
print(f"\n--- Amplitude comparison at atom positions ---")
for mo_idx, (cwp, coc) in enumerate([(cut_wp1,cut_oc1),(cut_wp2,cut_oc2)], 1):
    ratio1 = coc[iz_H1]/cwp[iz_H1] if abs(cwp[iz_H1])>1e-10 else float('nan')
    ratio2 = coc[iz_H2]/cwp[iz_H2] if abs(cwp[iz_H2])>1e-10 else float('nan')
    print(f"  MO{mo_idx}: WP@H1={cwp[iz_H1]:.6f}  OCL@H1={coc[iz_H1]:.6f}  ratio={ratio1:.4f}")
    print(f"         WP@H2={cwp[iz_H2]:.6f}  OCL@H2={coc[iz_H2]:.6f}  ratio={ratio2:.4f}")

# ── 6. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle("H2: WAVEPLOT vs OpenCL — MO1 (bonding) and MO2 (antibonding)", fontsize=13)

for mo_i, (grid_wp, grid_oc, cut_wp, cut_oc, label) in enumerate([
    (grid_wp1, grid_ocl1, cut_wp1, cut_oc1, "MO1 bonding"),
    (grid_wp2, grid_ocl2, cut_wp2, cut_oc2, "MO2 antibonding"),
]):
    row = axes[mo_i]

    # 2D XZ: WAVEPLOT
    sl_wp = grid_wp[:, iy0, :].T   # (NZ,NX)
    sl_oc = grid_oc[:, iy0, :].T
    clim = max(np.abs(sl_wp).max(), 1e-8)
    for ax, data, title in [(row[0],sl_wp,f"WAVEPLOT {label}"),
                             (row[1],sl_oc,f"OpenCL {label}")]:
        im = ax.imshow(data, origin='lower', cmap='RdBu_r', vmin=-clim, vmax=clim,
                       extent=ext_xz, interpolation='bilinear', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.04)
        ax.set_xlabel('z (Å)'); ax.set_ylabel('x (Å)')
        ax.set_title(title, fontsize=9)
        for zH in [atom_pos[0,2], atom_pos[1,2]]:
            ax.axvline(zH, c='k', lw=0.8, ls='--')

    # 1D linear
    ax = row[2]
    ax.plot(z_vals, cut_wp, 'b-',  lw=2,   label='WAVEPLOT')
    ax.plot(z_vals, cut_oc, 'r--', lw=1.5, label='OpenCL')
    for zH in [atom_pos[0,2], atom_pos[1,2]]:
        ax.axvline(zH, c='k', lw=0.8, ls=':')
    ax.set_xlabel('z (Å)'); ax.set_ylabel('ψ')
    ax.set_title(f'{label} — 1D cut along z (linear)', fontsize=9)
    ax.legend(fontsize=8)

    # 1D log  — only positive-valued half (bonding) or symmetric halves
    ax = row[3]
    eps = 1e-12
    # Use |ψ| but only where |ψ|>eps and on positive z side from H1
    z_pos = z_vals[z_vals >= atom_pos[0,2]]
    i0    = np.searchsorted(z_vals, atom_pos[0,2])
    y_wp  = np.abs(cut_wp[i0:]).clip(eps)
    y_oc  = np.abs(cut_oc[i0:]).clip(eps)
    ax.semilogy(z_pos, y_wp, 'b-',  lw=2,   label='WAVEPLOT')
    ax.semilogy(z_pos, y_oc, 'r--', lw=1.5, label='OpenCL')
    # Overlay expected slope from waveplot_in.hsd: alpha=0.967 Bohr^-1 = 1.827 Ang^-1
    alpha_ang = 0.967 / BOHR
    alpha_bohr_wrong = 0.967          # wrong: treating Bohr value as Ang
    y0_wp = float(y_wp[0])
    dz = z_pos - z_pos[0]
    ax.semilogy(z_pos, y0_wp*np.exp(-alpha_ang   *dz), 'g:',  lw=1, label=f'exp(-{alpha_ang:.3f}·z) correct')
    ax.semilogy(z_pos, y0_wp*np.exp(-alpha_bohr_wrong*dz), 'm:', lw=1, label=f'exp(-0.967·z) wrong Bohr as Ang')
    ax.axvline(atom_pos[0,2], c='k', lw=0.8, ls=':')
    ax.axvline(atom_pos[1,2], c='k', lw=0.8, ls=':')
    ax.set_xlabel('z (Å)'); ax.set_ylabel('|ψ| (log)')
    ax.set_title(f'{label} — log scale', fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xlim(z_pos[0], z_pos[-1])

# Row 3: ratio OCL/WP along z for both MOs
ax = axes[2, 0]
for cut_wp, cut_oc, lbl, col in [(cut_wp1,cut_oc1,'MO1','b'),(cut_wp2,cut_oc2,'MO2','r')]:
    mask = np.abs(cut_wp) > 1e-4 * np.abs(cut_wp).max()
    ratio = np.where(mask, cut_oc/cut_wp, np.nan)
    ax.plot(z_vals, ratio, color=col, lw=1.5, label=lbl)
ax.axhline(1.0, c='k', lw=1, ls='--', label='ratio=1')
for zH in [atom_pos[0,2], atom_pos[1,2]]:
    ax.axvline(zH, c='gray', lw=0.8, ls=':')
ax.set_xlabel('z (Å)'); ax.set_ylabel('OCL / WAVEPLOT')
ax.set_title('Ratio OCL/WAVEPLOT along z', fontsize=9)
ax.set_ylim(-0.5, 2.5); ax.legend(fontsize=8)

# Parity plot MO1
ax = axes[2, 1]
mask = np.abs(grid_wp1.ravel()) > 1e-4 * np.abs(grid_wp1).max()
ax.scatter(grid_wp1.ravel()[mask], grid_ocl1.ravel()[mask], s=0.5, alpha=0.3, c='b')
lim = np.abs(grid_wp1).max()*1.05
ax.plot([-lim,lim],[-lim,lim],'r--',lw=1)
ax.set_xlabel('WAVEPLOT'); ax.set_ylabel('OpenCL'); ax.set_title('Parity MO1 (all voxels)', fontsize=9)

# Parity plot MO2
ax = axes[2, 2]
mask = np.abs(grid_wp2.ravel()) > 1e-4 * np.abs(grid_wp2).max()
ax.scatter(grid_wp2.ravel()[mask], grid_ocl2.ravel()[mask], s=0.5, alpha=0.3, c='r')
lim = np.abs(grid_wp2).max()*1.05
ax.plot([-lim,lim],[-lim,lim],'r--',lw=1)
ax.set_xlabel('WAVEPLOT'); ax.set_ylabel('OpenCL'); ax.set_title('Parity MO2 (all voxels)', fontsize=9)

axes[2,3].axis('off')

fig.tight_layout()
out = OUT / 'h2_wp_vs_ocl.png'
fig.savefig(str(out), dpi=130)
print(f"\nSaved: {out}")
plt.show()
