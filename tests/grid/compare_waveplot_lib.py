#!/usr/bin/env python3
"""
Compare libwaveplot.so orbital grids vs reference WAVEPLOT cube files, and vs OpenCL.

General-purpose: works for any molecule given a DFTB+ run directory.
All system data (geometry, eigenvectors, basis) is read automatically from:
  <dftb-dir>/detailed.xml   -- geometry, nstates, norb, occupations
  <dftb-dir>/eigenvec.bin   -- eigenvectors (Fortran unformatted binary)
  <dftb-dir>/waveplot_in.hsd -- STO basis parameters

Usage:
  # Method 1: 3D grid comparison (needs wp-1-1-N-real.cube files)
  python compare_waveplot_lib.py --dftb-dir tests/grid/dftb_h2o --nmo 6

  # Method 2: 2D point evaluation on XY plane at z=2 Å
  python compare_waveplot_lib.py --dftb-dir tests/grid/dftb_h2o --points --plane2d xy --z-offset 2.0 --npoints 51

  # PTCDA, HOMO-4 to LUMO+4
  python compare_waveplot_lib.py --dftb-dir tests/grid/dftb_ptcda --points --plane2d xy --z-offset 2.0 --mo-range 66 75 --npoints 64
"""

import sys, os, struct, argparse
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pyBall.WavePlot.WavePlot import WavePlot
from pyBall.OCL.DFTBplusParser import parse_basis_hsd_ang

LIB_PATH   = str(REPO_ROOT / '_build' / 'app' / 'waveplot' / 'libwaveplot.so')
OUTPUT_DIR = Path(__file__).parent / 'waveplot_output' / 'comparison'
BOHR2ANG   = 0.5291772109


# ================================================================
# Parsers
# ================================================================

def parse_detailed_xml(xml_path):
    """
    Parse DFTB+ detailed.xml.
    Returns dict with: species_names, species_per_atom (0-based), coords_bohr (natoms,3),
                       nstates, norb, nkpoints, nspin, occupations (nstates,nkpoints,nspin)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # species names (strip quotes from XML text like '"O" "H"')
    tn_text = root.find('geometry/typenames').text.strip()
    species_names = [s.strip('"') for s in tn_text.split()]

    # atoms: type_index (1-based) + coords (Bohr)
    tc_text = root.find('geometry/typesandcoordinates').text.strip()
    species_per_atom = []
    coords = []
    for line in tc_text.splitlines():
        parts = line.split()
        if len(parts) == 4:
            species_per_atom.append(int(parts[0]) - 1)  # 0-based
            coords.append([float(x) for x in parts[1:4]])
    coords_bohr = np.array(coords)
    species_per_atom = np.array(species_per_atom, dtype=np.int32)

    nstates   = int(root.find('nrofstates').text)
    norb      = int(root.find('nroforbitals').text)
    nkpoints  = int(root.find('nrofkpoints').text)
    nspin     = int(root.find('nrofspins').text)

    # occupations
    occs = np.zeros((nstates, nkpoints, nspin))
    for ispin in range(nspin):
        spin_node = root.find(f'occupations/spin{ispin+1}')
        if spin_node is None: continue
        for ik in range(nkpoints):
            k_node = spin_node.find(f'k{ik+1}')
            if k_node is None: continue
            vals = [float(x) for x in k_node.text.split()]
            occs[:len(vals), ik, ispin] = vals

    return {
        'species_names':    species_names,
        'species_per_atom': species_per_atom,
        'coords_bohr':      coords_bohr,
        'natoms':           len(coords),
        'nstates':          nstates,
        'norb':             norb,
        'nkpoints':         nkpoints,
        'nspin':            nspin,
        'occupations':      occs,
    }


def parse_eigenvec_bin(bin_path, nstates, norb, nkpoints=1, nspin=1):
    """
    Parse eigenvec.bin: 4-byte int identity, then nstates*norb float64 values (row-major).
    Returns evecs[nstates, norb] float64 (k=0, spin=0).
    """
    with open(bin_path, 'rb') as f:
        raw = f.read()
    # identity: first 4 bytes
    identity = struct.unpack_from('i', raw, 0)[0]
    nfloats = (len(raw) - 4) // 8
    assert nfloats == nstates * norb, \
        f"eigenvec.bin: expected {nstates}*{norb}={nstates*norb} floats, got {nfloats}"
    evecs = np.frombuffer(raw[4:], dtype=np.float64).reshape(nstates, norb).copy()
    return evecs


def build_wp_basis(species_list_ang, species_names_ordered):
    """
    Convert parse_basis_hsd_ang() species list -> (basis_list, resoln_bohr) for WavePlot.
    species_names_ordered: unique species names in the order they appear in HSD.
    """
    sp_by_name = {sp['name']: sp for sp in species_list_ang}
    basis = []
    for sp_name in species_names_ordered:
        sp = sp_by_name[sp_name]
        angMoms   = [orb['l'] for orb in sp['orbitals']]
        cutoffs_b = [orb['cutoff'] / BOHR2ANG for orb in sp['orbitals']]
        occs      = [1.0] * len(sp['orbitals'])
        stos = []
        for orb in sp['orbitals']:
            l = orb['l']
            alpha_b = list(np.array(orb['exponents']) * BOHR2ANG)       # Å^-1 -> Bohr^-1
            coef_b  = np.array(orb['coefficients']) * (BOHR2ANG ** l)   # Å-normalised -> Bohr
            aa = coef_b.tolist() if hasattr(coef_b, 'tolist') else list(coef_b)
            stos.append({'alpha': alpha_b, 'aa': aa})
        basis.append({'angMoms': angMoms, 'cutoffs': cutoffs_b,
                      'occupations': occs, 'stos': stos})
    resoln_b = species_list_ang[0]['resolution'] / BOHR2ANG
    return basis, resoln_b


def evec_to_kernel_coeffs(evec_row, natoms, species_per_atom, species_names, species_list_ang):
    """
    Convert a single eigenvector row [norb] -> (natoms,4) float32 kernel coeffs [px,py,pz,s].
    Handles any combination of s and p orbitals.
    """
    sp_by_name = {sp['name']: sp for sp in species_list_ang}
    c = np.zeros((natoms, 4), dtype=np.float32)
    offset = 0
    for ia in range(natoms):
        sp_name = species_names[species_per_atom[ia]]
        for orb in sp_by_name[sp_name]['orbitals']:
            l  = orb['l']
            nm = 2*l + 1
            chunk = evec_row[offset:offset+nm]
            if l == 0:
                c[ia, 3] = float(chunk[0])           # s -> slot 3
            elif l == 1 and len(chunk) >= 3:
                c[ia, 1] = float(chunk[0])           # py
                c[ia, 2] = float(chunk[1])           # pz
                c[ia, 0] = float(chunk[2])           # px
            offset += nm
    return c


def read_cube(path):
    with open(path) as f: lines = f.readlines()
    natoms = int(lines[2].split()[0])
    ox, oy, oz = [float(x) for x in lines[2].split()[1:4]]
    nx, dx = int(lines[3].split()[0]), float(lines[3].split()[1])
    ny, dy = int(lines[4].split()[0]), float(lines[4].split()[2])
    nz, dz = int(lines[5].split()[0]), float(lines[5].split()[3])
    vals = np.fromstring(' '.join(lines[6+natoms:]), sep=' ')
    data = vals.reshape(nx, ny, nz)
    atom_Z     = np.array([int(l.split()[0]) for l in lines[6:6+natoms]])
    atom_coords = np.array([[float(x) for x in l.split()[2:5]] for l in lines[6:6+natoms]])
    return data, np.array([ox,oy,oz]), np.array([dx,dy,dz]), atom_coords, atom_Z


# ================================================================
# CLI
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dftb-dir',  type=str, default=None,
                   help='DFTB+ run directory containing detailed.xml, eigenvec.bin, waveplot_in.hsd')
    p.add_argument('--no-show',   action='store_true')
    p.add_argument('--dpi',       type=int, default=150)
    p.add_argument('--nmo',       type=int, default=6,
                   help='Number of MOs (Method 1, or default range if --mo-range not set)')
    p.add_argument('--mo-range',  type=int, nargs=2, default=None,
                   metavar=('START','END'), help='1-based inclusive MO index range')
    p.add_argument('--points',    action='store_true',
                   help='Method 2: evaluate at explicit points (libwaveplot + OpenCL)')
    p.add_argument('--plane2d',   type=str, choices=['xy','xz','yz'], default=None,
                   help='2D plane for --points; omit for 1D z-scan')
    p.add_argument('--z-offset',  type=float, default=0.0,
                   help='Fixed coordinate value for out-of-plane axis (Å)')
    p.add_argument('--xy-range',  type=float, nargs=2, default=None,
                   metavar=('MIN','MAX'), help='Coordinate range for 2D plane (Å); default: mol extent + 3 Å')
    p.add_argument('--z-range',   type=float, nargs=2, default=[-3.0, 3.0],
                   metavar=('ZMIN','ZMAX'), help='Range for 1D z-scan (Å)')
    p.add_argument('--npoints',   type=int, default=64, help='Grid points per axis')
    return p.parse_args()


# ================================================================
# MAIN
# ================================================================

def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dftb_dir = Path(args.dftb_dir) if args.dftb_dir else Path(__file__).parent / 'dftb_h2o'
    assert dftb_dir.exists(), f"Not found: {dftb_dir}"
    system_name = dftb_dir.name
    print("=" * 60)
    print(f"System: {system_name}  ({dftb_dir})")
    print("=" * 60)

    # ---- 1. Parse detailed.xml ----
    geo = parse_detailed_xml(dftb_dir / 'detailed.xml')
    atom_coords_b   = geo['coords_bohr']        # (natoms,3) Bohr
    species_per_atom = geo['species_per_atom']  # 0-based
    species_names    = geo['species_names']
    natoms  = geo['natoms']
    nstates_total = geo['nstates']
    norb_total    = geo['norb']
    nkpoints      = geo['nkpoints']
    nspin         = geo['nspin']
    occs_full     = geo['occupations']          # (nstates,nkpoints,nspin)
    occs_flat     = occs_full[:, 0, 0]          # k=1, spin=1
    print(f"  natoms={natoms}, species={species_names}, nstates={nstates_total}, norb={norb_total}")

    # ---- 2. Parse STO basis ----
    hsd_path = dftb_dir / 'waveplot_in.hsd'
    assert hsd_path.exists(), f"waveplot_in.hsd not found: {hsd_path}"
    species_list_ang = parse_basis_hsd_ang(hsd_path)
    sp_names_hsd = [sp['name'] for sp in species_list_ang]
    print(f"  HSD species: {sp_names_hsd}")

    # species array for libwaveplot: 1-based index into sp_names_hsd
    sp_name_to_idx = {name: i+1 for i, name in enumerate(sp_names_hsd)}
    species_wp = np.array([sp_name_to_idx[species_names[si]] for si in species_per_atom],
                           dtype=np.int32)

    wp_basis, resoln_b = build_wp_basis(species_list_ang, sp_names_hsd)
    print(f"  Basis: {len(wp_basis)} species, resoln={resoln_b:.4f} Bohr")

    # norb per atom (from basis)
    sp_by_name = {sp['name']: sp for sp in species_list_ang}
    norb_per_atom = np.array([sum(2*o['l']+1 for o in sp_by_name[species_names[si]]['orbitals'])
                               for si in species_per_atom], dtype=np.int32)
    assert norb_per_atom.sum() == norb_total, \
        f"norb mismatch: basis sums to {norb_per_atom.sum()}, XML says {norb_total}"

    # ---- 3. Parse eigenvectors ----
    # parse_eigenvec_bin returns (nstates, norb) directly
    evecs_full = parse_eigenvec_bin(dftb_dir / 'eigenvec.bin',
                                    nstates_total, norb_total)

    # ---- HOMO ----
    homo_idx = np.where(occs_flat > 0.5)[0]
    homo = int(homo_idx[-1]) + 1 if len(homo_idx) else nstates_total // 2

    # Read energies from band.out (optional, for labels)
    energies_ev = np.zeros(nstates_total)
    band_path = dftb_dir / 'band.out'
    if band_path.exists():
        with open(band_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        idx = int(parts[0])
                        if 1 <= idx <= nstates_total:
                            energies_ev[idx-1] = float(parts[1])
                    except ValueError:
                        pass

    # ---- MO range ----
    if args.mo_range:
        mo_start = max(1, args.mo_range[0])
        mo_end   = min(nstates_total, args.mo_range[1])
    else:
        mo_start = 1
        mo_end   = min(nstates_total, args.nmo)
    nstates  = mo_end - mo_start + 1
    evecs    = evecs_full[mo_start-1 : mo_end]
    energies = energies_ev[mo_start-1 : mo_end]
    occs     = occs_flat[mo_start-1 : mo_end]

    print(f"  HOMO = MO{homo} ({energies_ev[homo-1]:.3f} eV)")
    print(f"  Evaluating MOs {mo_start}–{mo_end}  ({nstates} total)")
    for i in range(min(nstates, 15)):
        tag = " <--HOMO" if (mo_start+i)==homo else (" <--LUMO" if (mo_start+i)==homo+1 else "")
        print(f"    MO{mo_start+i:4d}  E={energies[i]:8.3f} eV  occ={occs[i]:.1f}{tag}")

    # ================================================================
    # Method 2: point evaluation
    # ================================================================
    if args.points:
        print("\n" + "=" * 60)
        print(f"Method 2: {'2D '+args.plane2d.upper() if args.plane2d else '1D z-scan'}")
        print("=" * 60)

        atom_ang = atom_coords_b * BOHR2ANG
        if args.xy_range:
            rmin, rmax_r = args.xy_range
        else:
            rmin  = float(atom_ang.min()) - 3.0
            rmax_r = float(atom_ang.max()) + 3.0

        # indexing='ij': uu[i,j] = u[i] (x-axis), vv[i,j] = u[j] (y-axis)
        # reshape to (npoints,npoints) then imshow with no .T needed for [x,y] layout
        u = np.linspace(rmin, rmax_r, args.npoints)
        if args.plane2d == 'xy':
            uu, vv = np.meshgrid(u, u, indexing='ij')
            ww = np.full_like(uu, args.z_offset)
            points_ang = np.column_stack([uu.ravel(), vv.ravel(), ww.ravel()])
            ax_labels = ('x (Å)', 'y (Å)')
            plane_desc = f"XY plane  z={args.z_offset:.3f} Å"
        elif args.plane2d == 'xz':
            uu, vv = np.meshgrid(u, u, indexing='ij')
            points_ang = np.column_stack([uu.ravel(), np.full(uu.size, args.z_offset), vv.ravel()])
            ax_labels = ('x (Å)', 'z (Å)')
            plane_desc = f"XZ plane  y={args.z_offset:.3f} Å"
        elif args.plane2d == 'yz':
            uu, vv = np.meshgrid(u, u, indexing='ij')
            points_ang = np.column_stack([np.full(uu.size, args.z_offset), uu.ravel(), vv.ravel()])
            ax_labels = ('y (Å)', 'z (Å)')
            plane_desc = f"YZ plane  x={args.z_offset:.3f} Å"
        else:  # 1D z-scan
            z_vals = np.linspace(args.z_range[0], args.z_range[1], args.npoints)
            points_ang = np.column_stack([np.zeros(args.npoints), np.zeros(args.npoints), z_vals])
            ax_labels = ('z (Å)', 'ψ')
            plane_desc = "1D z-scan  x=0  y=0"

        npts = len(points_ang)
        print(f"  {npts} points, coord range [{rmin:.2f}, {rmax_r:.2f}] Å")
        print(f"  {plane_desc}")

        # --- libwaveplot ---
        points_bohr = points_ang / BOHR2ANG
        wp = WavePlot(LIB_PATH)
        wp.set_geometry(atom_coords_b, species_wp, is_periodic=False)
        wp.set_basis(wp_basis, resolution=resoln_b)
        wp.set_eigenvectors(evecs)

        wp_vals = np.zeros((nstates, npts))
        for i in range(nstates):
            wp_vals[i] = wp.orb2points(i+1, points_bohr)
        print(f"  libwaveplot done. max|ψ|={np.abs(wp_vals).max():.4e}")

        # --- OpenCL ---
        from pyBall.OCL.Grid import GridProjector
        from pyBall import elements as _el

        projector = GridProjector(fdata_dir=None, verbosity=0)
        projector.load_basis_sto(species_list_ang)

        atom_ang32 = atom_ang.astype(np.float32)
        nz_to_rcut = {sp['atomic_number']: max(o['cutoff'] for o in sp['orbitals'])
                      for sp in species_list_ang}
        atom_nz  = np.array([_el.ELEMENT_DICT[species_names[si]][0] for si in species_per_atom],
                              dtype=np.int32)
        rcut_arr = np.array([nz_to_rcut.get(int(nz), 7.0) for nz in atom_nz], dtype=np.float32)
        atoms_dict = {'pos': atom_ang32, 'Rcut': rcut_arr, 'type': atom_nz}

        print(f"  Using project_orbital_points for OCL (direct, no grid)")

        ocl_vals = np.zeros((nstates, npts))
        for i in range(nstates):
            coeffs_k = evec_to_kernel_coeffs(evecs[i], natoms, species_per_atom,
                                              species_names, species_list_ang)
            psi = projector.project_orbital_points(
                points_ang.astype(np.float32), coeffs_k, norb_per_atom, atoms_dict)
            ocl_vals[i] = psi.astype(np.float64)
        print(f"  OpenCL done. max|ψ|={np.abs(ocl_vals).max():.4e}")

        print("\n  RMS (libwaveplot vs OpenCL):")
        for i in range(nstates):
            diff = wp_vals[i] - ocl_vals[i]
            tag = " HOMO" if (mo_start+i)==homo else (" LUMO" if (mo_start+i)==homo+1 else "")
            print(f"    MO{mo_start+i:4d}{tag}  E={energies[i]:7.3f}eV  RMS={np.sqrt(np.mean(diff**2)):.3e}  max={np.abs(diff).max():.3e}")

        # ---- plot ----
        method_tag = "orb2points"   # explicit-point method — no grid slicing
        if args.plane2d:
            s2  = (args.npoints, args.npoints)
            ext = [rmin, rmax_r, rmin, rmax_r]
            ncols = 3
            fig, axes = plt.subplots(nstates, ncols, figsize=(5*ncols, 4*nstates))
            if nstates == 1: axes = axes[np.newaxis, :]
            for i in range(nstates):
                # reshape: indexing='ij' -> axis0=u (horizontal), axis1=v (vertical)
                # imshow(origin='lower'): rows=y, cols=x -> need .T
                wp2  = wp_vals[i].reshape(s2).T
                oc2  = ocl_vals[i].reshape(s2).T
                df2  = wp2 - oc2
                clim = max(np.abs(wp2).max(), np.abs(oc2).max()) or 1e-12
                tag  = " [HOMO]" if (mo_start+i)==homo else (" [LUMO]" if (mo_start+i)==homo+1 else "")
                plane_title = f"{plane_desc}  [{method_tag}]"
                for ax, dat, ttl, cm, vl, vh in [
                    (axes[i,0], wp2, f"libwaveplot MO{mo_start+i}{tag}\nE={energies[i]:.2f}eV  {plane_title}", 'RdBu_r', -clim, clim),
                    (axes[i,1], oc2, f"OpenCL MO{mo_start+i}{tag}\n{plane_title}",                              'RdBu_r', -clim, clim),
                    (axes[i,2], df2, f"diff (lib−OCL)\nRMS={np.sqrt(np.mean((wp_vals[i]-ocl_vals[i])**2)):.2e}", 'bwr', -clim*0.1, clim*0.1),
                ]:
                    im = ax.imshow(dat, origin='lower', cmap=cm, vmin=vl, vmax=vh,
                                   extent=ext, aspect='equal')
                    ax.set_xlabel(ax_labels[0]); ax.set_ylabel(ax_labels[1])
                    ax.set_title(ttl, fontsize=7)
                    plt.colorbar(im, ax=ax, fraction=0.046)
            suffix = f"{system_name}_{method_tag}_{args.plane2d}_z{args.z_offset:.2f}A_n{args.npoints}_MO{mo_start}-{mo_end}"
        else:
            fig, axes = plt.subplots(nstates, 2, figsize=(14, 3*nstates))
            if nstates == 1: axes = axes[np.newaxis, :]
            for i in range(nstates):
                tag = " [HOMO]" if (mo_start+i)==homo else (" [LUMO]" if (mo_start+i)==homo+1 else "")
                axes[i,0].plot(z_vals, wp_vals[i],  'b-',  lw=2,   label='libwaveplot')
                axes[i,0].plot(z_vals, ocl_vals[i], 'r--', lw=1.5, label='OpenCL')
                axes[i,0].axhline(0, c='gray', lw=0.5)
                axes[i,0].set_title(f"MO{mo_start+i}{tag} E={energies[i]:.2f}eV  [{method_tag}]  {plane_desc}", fontsize=8)
                axes[i,0].set_xlabel('z (Å)');  axes[i,0].set_ylabel('ψ')
                axes[i,0].legend(fontsize=7)
                mask = z_vals >= 0;  eps = 1e-12
                axes[i,1].semilogy(z_vals[mask], np.abs(wp_vals[i,mask]).clip(eps), 'b-', lw=2)
                axes[i,1].semilogy(z_vals[mask], np.abs(ocl_vals[i,mask]).clip(eps),'r--',lw=1.5)
                axes[i,1].set_title(f"MO{mo_start+i}{tag} — log  [{method_tag}]", fontsize=8)
                axes[i,1].set_xlabel('z (Å)');  axes[i,1].set_ylabel('|ψ|')
            suffix = f"{system_name}_{method_tag}_1d_n{args.npoints}_MO{mo_start}-{mo_end}"

        fig.suptitle(f"{system_name}: libwaveplot vs OpenCL  [{method_tag}]  {plane_desc}  (MO{mo_start}–{mo_end})", fontsize=9)
        fig.tight_layout()
        out_file = OUTPUT_DIR / f'comparison_points_{suffix}.png'
        fig.savefig(str(out_file), dpi=args.dpi)
        print(f"\nSaved: {out_file}")
        if not args.no_show: plt.show()
        return

    # ================================================================
    # Method 1: 3D grid comparison vs cube files
    # ================================================================
    print("\n" + "=" * 60)
    print("Method 1: 3D grid (libwaveplot vs WAVEPLOT cube)")
    print("=" * 60)

    cube_grids, origin_b, step_b, nPoints = [], None, None, None
    for imo in range(mo_start, mo_end+1):
        cube_path = dftb_dir / f'wp-1-1-{imo}-real.cube'
        if not cube_path.exists():
            print(f"  WARNING: {cube_path} not found — skipping Method 1")
            return
        data, _ori, _stp, _ac, _az = read_cube(cube_path)
        cube_grids.append(data)
        if origin_b is None:
            origin_b = _ori; step_b = _stp
            nPoints  = np.array(data.shape, dtype=np.int32)
        print(f"  MO{imo}: shape={data.shape}  |ψ|max={np.abs(data).max():.4e}")

    wp = WavePlot(LIB_PATH)
    wp.set_geometry(atom_coords_b, species_wp, is_periodic=False)
    wp.set_basis(wp_basis, resolution=resoln_b)
    wp.set_eigenvectors(evecs)

    gridVecs = np.diag(step_b)
    lib_grids = []
    for i in range(nstates):
        g = wp.orb2grid(i+1, origin_b, gridVecs, nPoints)
        lib_grids.append(g)
        print(f"  libwaveplot MO{mo_start+i}: |ψ|max={np.abs(g).max():.4e}")

    iz_mol   = int(np.clip(round(-origin_b[2]/step_b[2]), 0, nPoints[2]-1))
    z_slice_ang = (origin_b[2] + iz_mol * step_b[2]) * BOHR2ANG
    ext_xy = [origin_b[0]*BOHR2ANG, (origin_b[0]+nPoints[0]*step_b[0])*BOHR2ANG,
              origin_b[1]*BOHR2ANG, (origin_b[1]+nPoints[1]*step_b[1])*BOHR2ANG]
    method1_tag = "orb2grid"
    slice_desc  = f"XY slice  iz={iz_mol}  z={z_slice_ang:.3f} Å  [{method1_tag}]"
    print(f"  {slice_desc}")

    fig, axes = plt.subplots(nstates, 3, figsize=(13, 4*nstates))
    if nstates == 1: axes = axes[np.newaxis, :]
    all_stats = []
    for i in range(nstates):
        ref  = cube_grids[i]; lib  = lib_grids[i]; diff = lib - ref
        rs   = ref[:,:,iz_mol]; ls  = lib[:,:,iz_mol]; ds = diff[:,:,iz_mol]
        clim = max(np.abs(rs).max(), np.abs(ls).max()) or 1.0
        tag  = " [HOMO]" if (mo_start+i)==homo else (" [LUMO]" if (mo_start+i)==homo+1 else "")
        for ax, dat, ttl, cm, vl, vh in [
            (axes[i,0], rs, f"WAVEPLOT cube MO{mo_start+i}{tag}\nE={energies[i]:.2f}eV  {slice_desc}", 'RdBu_r', -clim, clim),
            (axes[i,1], ls, f"libwaveplot.so MO{mo_start+i}{tag}\n{slice_desc}",                        'RdBu_r', -clim, clim),
            (axes[i,2], ds, f"diff (lib−cube)\nRMS={np.sqrt(np.mean(diff**2)):.2e}",                    'bwr',    -clim*0.05, clim*0.05),
        ]:
            im = ax.imshow(dat.T, origin='lower', cmap=cm, vmin=vl, vmax=vh,
                           extent=ext_xy, interpolation='bilinear')
            ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)')
            ax.set_title(ttl, fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046)
        rms = np.sqrt(np.mean(diff**2)); rmax = np.abs(ref).max()
        all_stats.append((mo_start+i, rms, rms/rmax if rmax>1e-10 else float('nan'), rmax))

    fig.suptitle(f"{system_name}: libwaveplot vs WAVEPLOT cube  [{method1_tag}]  {slice_desc}  (MO{mo_start}–{mo_end})", fontsize=9)
    fig.tight_layout()
    out1 = OUTPUT_DIR / f'comparison_lib_vs_cube_{system_name}_MO{mo_start}-{mo_end}.png'
    fig.savefig(str(out1), dpi=args.dpi)
    print(f"\nSaved: {out1}")
    if not args.no_show: plt.show()
    plt.close(fig)

    print(f"\n{'MO':>5}  {'E(eV)':>8}  {'RMS':>12}  {'RelRMS':>10}  {'|ref|max':>12}")
    for mo, rms, rel, rmax in all_stats:
        i = mo - mo_start
        print(f"  {mo:3d}   {energies[i]:8.3f}   {rms:12.4e}   {rel:10.4e}   {rmax:12.4e}")
    print(f"\nOutputs in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
