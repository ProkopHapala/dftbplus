#!/usr/bin/env python3
"""Build the periodic quinoxaline/dihydroquinoxaline H-bond chain from
ASCII art (SPAMMM ascii_art_heterocycle), add capping H's, plot the
periodic tiling, and export the unit-cell geometry for GpuPbc/DFTB+.

Motif (user-specified ASCII, dimer format):

       N C
      | | |
       N C
       :
     C n
    | | |
     C n

Each block is a 2-ring molecule; the N's/n's sit at the top/bottom
vertices of one ring (pyrazine edge -> para diazine). ':' marks the
N...H-N bond between the bottom vertex of the upper molecule and the
top vertex of the lower one. Tiling along y repeats it.

Outputs:
  debug/qxhq_chain.png            — 3-cell plot for visual review
  data/xyz/qxhq_chain_cell.xyz    — one cell (2 molecules)
  data/xyz/qxhq_chain_cell.gen    — GenFormat for Fortran dftb+
"""
import os, sys
import numpy as np

sys.path.insert(0, "/home/prokop/git/SPAMMM")
from spammm.topology.ascii_art_heterocycle import (
    parse_ascii_art, _build_target_valence, resolve_hbond_pairs)
from spammm.topology.KekulePure import make_n_pi

ART = """
   N C
  | | |
   N C
   :
 C n
| | |
 C n
"""

HBOND = 2.9      # N...N target distance across ':' [Ang]
TILT_DEG = 25.0  # herringbone tilt of ring planes about the N-N spine axis
OUTDIR = "/home/prokop/git/dftbplus"


def build(hbond=HBOND, tilt_deg=TILT_DEG):
    atoms = parse_ascii_art(ART, hbond_length=hbond)
    atoms.neighs()
    n_pi0 = make_n_pi(atoms)
    tv = _build_target_valence(atoms, n_pi0)
    atoms.add_capping_h_sp2(target_valence=tv)
    atoms.neighs()
    resolve_hbond_pairs(atoms)
    if tilt_deg:
        tilt_molecules(atoms, tilt_deg)
    return atoms


def tilt_molecules(atoms, deg):
    """Rotate each molecule about the line through its two hetero
    vertices (the chain spine, parallel to +y). Opposite signs for the
    two molecules -> herringbone packing; N's and donor N-H's lie on
    the axis so the N...H-N geometry is invariant."""
    parent = list(range(atoms.natoms))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in atoms.bonds:
        parent[find(i)] = find(j)
    comp = {}
    for i in range(atoms.natoms):
        comp.setdefault(find(i), []).append(i)
    assert len(comp) == 2, f"expected 2 molecules, got {len(comp)}"
    th = np.radians(deg); c, s = np.cos(th), np.sin(th)
    for k, members in enumerate(comp.values()):
        Ns = [i for i in members if atoms.enames[i] == 'N']
        x0 = atoms.apos[Ns[0], 0]           # spine axis: x const, z=0
        sign = +1.0 if k == 0 else -1.0
        cs, sn = c, sign * s
        for i in members:
            dx = atoms.apos[i, 0] - x0; dz = atoms.apos[i, 2]
            atoms.apos[i, 0] = x0 + dx * cs - dz * sn
            atoms.apos[i, 2] = dx * sn + dz * cs


def cell_params(atoms):
    """Cell length along y: distance between the two N...N interfaces
    must be equal by symmetry of the motif -> L = 2*(yN_top - yN_bot)
    spacing measured from the two ':' bonds inside one drawing is not
    enough; derive L from the bottom-n -> next-cell top-N gap."""
    apos, enames = atoms.apos, atoms.enames
    N_up = [i for i in range(atoms.natoms) if enames[i] == 'N' and apos[i,1] > -3.0]
    N_lo = [i for i in range(atoms.natoms) if enames[i] == 'N' and apos[i,1] < -3.0 and apos[i,0] > 3.0]
    # donors (lowercase n -> 'N' after upper()); identify by H neighbors
    donors = [i for i in range(atoms.natoms) if enames[i] == 'N' and
              any(enames[j] == 'H' for j in atoms.ngs[i])]
    print("N_up:", [(i, np.round(atoms.apos[i],2)) for i in N_up])
    print("N_lo:", [(i, np.round(atoms.apos[i],2)) for i in N_lo])
    print("donors:", [(i, np.round(atoms.apos[i],2)) for i in donors])
    y_top_acc = max(apos[i,1] for i in N_up)   # top-vertex N of upper mol (accepts from prev cell)
    y_bot_don = min(apos[i,1] for i in donors) # bottom-vertex n of lower mol (donates to next cell)
    # next cell's top-N sits hbond below y_bot_don; equivalently the
    # drawing repeats with shift L where L = (top_acc - bot_don) + hbond
    L = (y_top_acc - y_bot_don) + HBOND
    return L


def main():
    atoms = build()
    apos = atoms.apos.copy()
    L = cell_params(atoms)
    print(f"cell length along y: {L:.3f} A")

    # clash check: min intermolecular H...H and X...H distances (with images)
    H = [i for i in range(atoms.natoms) if atoms.enames[i] == 'H']
    mol = [0] * atoms.natoms
    par = list(range(atoms.natoms))
    def _f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i, j in atoms.bonds: par[_f(i)] = _f(j)
    roots = sorted({_f(i) for i in range(atoms.natoms)})
    for i in range(atoms.natoms): mol[i] = roots.index(_f(i))
    worst = []
    for i in H:
        for j in H:
            if i >= j: continue
            for sh in (0.0, -L, L):
                d = np.linalg.norm(apos[i] - (apos[j] + [0, sh, 0]))
                if mol[i] != mol[j] or sh != 0.0:
                    worst.append((d, i, j, sh))
    worst.sort()
    print("min intermol H...H:", [(f'{d:.3f}', i, j, s) for d, i, j, s in worst[:4]])

    # tile 3 cells for the plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, az) = plt.subplots(1, 2, figsize=(11, 9), gridspec_kw={'width_ratios': [2, 1]})
    for c in (-1, 0, 1):
        for i, j in atoms.bonds:
            ax.plot([apos[i,0], apos[j,0]], [apos[i,1]+c*L, apos[j,1]+c*L],
                    'k-', lw=1.2, alpha=0.6 if c else 1.0)
            az.plot([apos[i,2], apos[j,2]], [apos[i,1]+c*L, apos[j,1]+c*L],
                    'k-', lw=1.2, alpha=0.6 if c else 1.0)
        for (e, p) in zip(atoms.enames, apos):
            col = {'N': 'b', 'C': 'gray', 'H': 'lightgray'}[e]
            sz = {'N': 90, 'C': 60, 'H': 30}[e]
            ax.scatter(p[0], p[1]+c*L, c=col, s=sz, zorder=3)
            ax.annotate(e, (p[0], p[1]+c*L), ha='center', va='center', fontsize=6)
            az.scatter(p[2], p[1]+c*L, c=col, s=sz, zorder=3)
    for yb in (0.0, -L):
        ax.axhline(yb, color='r', ls='--', lw=0.8, alpha=0.5)
        az.axhline(yb, color='r', ls='--', lw=0.8, alpha=0.5)
    ax.set_aspect('equal'); ax.set_title(f"QX-HQ chain cell L={L:.2f} A, tilt={TILT_DEG} deg")
    az.set_aspect('equal'); az.set_title("side view (z-y): herringbone tilt"); az.set_xlabel("z [A]")
    fn = os.path.join(OUTDIR, "debug", "qxhq_chain.png")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    fig.savefig(fn, dpi=150, bbox_inches='tight')
    print("REVIEW:", fn)

    # export unit cell: species + coords + lattice (chain along y)
    xlo, xhi = apos[:,0].min(), apos[:,0].max()
    zspan = max(apos[:,2].ptp(), 1.0)
    lx = (xhi - xlo) + 8.0     # vacuum margin
    lz = 15.0
    x0 = xlo - 4.0
    coords = apos.copy(); coords[:,0] -= x0; coords[:,2] += (lz - zspan)/2 - apos[:,2].min()
    coords[:,1] = np.mod(coords[:,1] - 0.5, L)  # wrap into [0,L), margin from boundaries
    lat = np.array([[lx, 0, 0], [0, L, 0], [0, 0, lz]])
    xyz = os.path.join(OUTDIR, "data/xyz/qxhq_chain_cell.xyz")
    with open(xyz, "w") as f:
        f.write(f"{atoms.natoms}\nqx-hq chain cell, Ly={L:.3f}\n")
        for e, p in zip(atoms.enames, coords):
            f.write(f"{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
    print("wrote", xyz)
    gen = os.path.join(OUTDIR, "data/xyz/qxhq_chain_cell.gen")
    sp = sorted(set(atoms.enames))
    with open(gen, "w") as f:
        f.write(f"{atoms.natoms} S\n{' '.join(sp)}\n")
        for i, (e, p) in enumerate(zip(atoms.enames, coords)):
            f.write(f"{i+1} {sp.index(e)+1} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
        f.write("0.0 0.0 0.0\n")
        for v in lat:
            f.write(f"{v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}\n")
    print("wrote", gen)
    # report the H-bond junctions for the scan (donor-N, H, acceptor-N)
    for (ih, iacc) in getattr(atoms, 'hbonds_ascii', []):
        don = [j for j in atoms.ngs[ih] if atoms.enames[j] != 'H'][0]
        print(f"hbond: N{don}-H{ih}...N{iacc}  r(N..N)={np.linalg.norm(apos[iacc]-apos[don]):.3f} A")


if __name__ == "__main__":
    main()
