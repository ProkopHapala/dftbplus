#!/usr/bin/env python3
"""
H2 masked STO component testing.
Test each individual STO component (coefficient term) to find perfect parity.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from pyBall.OCL.DFTBplusParser import parse_basis_hsd_ang, mask_sto_coefficients
from pyBall.OCL.Grid import GridProjector, setup_gridprojector_from_dftb, evaluate_mos_on_points
from pyBall.WavePlot.WavePlot import WavePlot, setup_waveplot_from_dftb, evaluate_mos_on_points as wp_evaluate_mos_on_points
from pyBall.WavePlot.TestUtils import compare_point_evaluations, print_comparison_results

REPO_ROOT = Path(__file__).parent.parent.parent
LIB_PATH = str(REPO_ROOT / '_build' / 'app' / 'waveplot' / 'libwaveplot.so')
OUTPUT_DIR = Path(__file__).parent / 'waveplot_output' / 'masked_tests'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOHR2ANG = 0.5291772109


def test_h2_single_component(pow_idx, alpha_idx, n_points=200):
    """
    Test H2 with only one STO component active.
    
    Args:
        pow_idx: Power index (0, 1, 2) -> r^(l + pow_idx)
        alpha_idx: Exponent index (0, 1, 2) -> exp(-alpha[alpha_idx] * r)
    """
    dftb_dir = Path(__file__).parent / 'dftb_h2'
    
    # Parse basis
    hsd_path = dftb_dir / 'waveplot_in.hsd'
    species_list_ang = parse_basis_hsd_ang(hsd_path)
    
    # Mask to single component
    masked_list, orig_coeff = mask_sto_coefficients(
        species_list_ang, 'H', orbital_idx=0, 
        active_pow=pow_idx, active_alpha=alpha_idx
    )
    
    print(f"\n{'='*60}")
    print(f"Testing H2: pow={pow_idx}, alpha={alpha_idx}, coeff={orig_coeff:.6f}")
    print(f"{'='*60}")
    
    # Parse geometry
    from pyBall.OCL.DFTBplusParser import parse_detailed_xml_custom, parse_eigenvec_bin_custom
    geo = parse_detailed_xml_custom(dftb_dir / 'detailed.xml')
    atom_coords_b = geo['coords_bohr']
    species_per_atom = geo['species_per_atom']
    species_names = geo['species_names']
    natoms = geo['natoms']
    
    # Set eigenvectors to unity (each atom has coefficient 1.0 in MO)
    # For H2: bonding state with equal coefficients on both H atoms
    nstates = 2
    norb = 2  # One s-orbital per H atom
    evecs = np.array([[1.0, 1.0],   # Bonding: both atoms +
                      [1.0, -1.0]])  # Antibonding: opposite signs
    
    # Bond line scan
    atom_ang = atom_coords_b * BOHR2ANG
    pos_0 = atom_ang[0]  # First H atom
    pos_1 = atom_ang[1]  # Second H atom
    bond_vec = pos_1 - pos_0
    bond_len = np.linalg.norm(bond_vec)
    bond_dir = bond_vec / bond_len
    
    # Scan range: -3Å to +3Å from bond center, along bond axis
    t_vals = np.linspace(-bond_len/2 - 3.0, bond_len/2 + 3.0, n_points)
    points_ang = np.array([(pos_0 + pos_1)/2 + t * bond_dir for t in t_vals])
    
    # --- libwaveplot ---
    # Build basis for libwaveplot
    from pyBall.OCL.DFTBplusParser import build_wp_basis
    sp_names_hsd = [sp['name'] for sp in masked_list]
    sp_name_to_idx = {name: i+1 for i, name in enumerate(sp_names_hsd)}
    species_wp = np.array([sp_name_to_idx[species_names[si]] 
                          for si in species_per_atom], dtype=np.int32)
    wp_basis, resoln_b = build_wp_basis(masked_list, sp_names_hsd)
    
    dftb_data_wp = {
        'coords_bohr': atom_coords_b,
        'species_wp': species_wp,
        'basis': wp_basis,
        'resolution': resoln_b,
        'evecs': evecs
    }
    wp = setup_waveplot_from_dftb(dftb_data_wp, LIB_PATH)
    mo_indices_wp = [1]  # Test MO1 (bonding)
    points_bohr = points_ang / BOHR2ANG
    wp_vals_list = wp_evaluate_mos_on_points(wp, mo_indices_wp, points_bohr)
    wp_vals = np.array(wp_vals_list)[0]
    
    # --- OpenCL ---
    dftb_data_ocl = {
        'coords_bohr': atom_coords_b,
        'species_per_atom': species_per_atom,
        'species_names': species_names
    }
    projector, atoms_dict = setup_gridprojector_from_dftb(
        dftb_data_ocl, masked_list, verbosity=0
    )
    
    # Build norb_per_atom from masked basis
    sp_by_name = {sp['name']: sp for sp in masked_list}
    norb_per_atom = np.array([
        sum(2*o['l']+1 for o in sp_by_name[species_names[si]]['orbitals'])
        for si in species_per_atom
    ], dtype=np.int32)
    
    mo_indices_ocl = [0]  # MO1
    ocl_vals_list = evaluate_mos_on_points(
        projector, mo_indices_ocl, points_ang.astype(np.float32),
        evecs, natoms, species_per_atom, species_names,
        masked_list, norb_per_atom, atoms_dict
    )
    ocl_vals = np.array(ocl_vals_list)[0]
    
    # --- Comparison ---
    diff = wp_vals - ocl_vals
    rms = np.sqrt(np.mean(diff**2))
    max_abs = np.max(np.abs(diff))
    
    print(f"\nResults:")
    print(f"  libwaveplot max|psi| = {np.abs(wp_vals).max():.6e}")
    print(f"  OpenCL max|psi| = {np.abs(ocl_vals).max():.6e}")
    print(f"  RMS error = {rms:.6e}")
    print(f"  Max abs error = {max_abs:.6e}")
    
    if rms < 1e-10:
        print(f"  [PASS] Perfect parity achieved!")
    elif rms < 1e-5:
        print(f"  [OK] Good agreement")
    else:
        print(f"  [FAIL] Significant discrepancy")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ax = axes[0]
    ax.plot(t_vals, wp_vals, 'b-', label='libwaveplot', linewidth=2)
    ax.plot(t_vals, ocl_vals, 'r--', label='OpenCL', linewidth=1)
    ax.set_xlabel('Distance along bond (Å)')
    ax.set_ylabel('ψ')
    ax.set_title(f'H2 pow={pow_idx} alpha={alpha_idx}\nCoeff={orig_coeff:.4f}')
    ax.legend()
    ax.grid(True)
    
    ax = axes[1]
    ax.plot(t_vals, diff, 'g-', label='diff (lib - OCL)')
    ax.set_xlabel('Distance along bond (Å)')
    ax.set_ylabel('ψ difference')
    ax.set_title(f'Error: RMS={rms:.2e}')
    ax.legend()
    ax.grid(True)
    
    ax = axes[2]
    ax.semilogy(t_vals, np.abs(diff) + 1e-16, 'g-', label='|diff|')
    ax.set_xlabel('Distance along bond (Å)')
    ax.set_ylabel('|ψ difference| (log)')
    ax.set_title('Absolute error (log scale)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    out_file = OUTPUT_DIR / f'h2_pow{pow_idx}_alpha{alpha_idx}.png'
    fig.savefig(str(out_file), dpi=150)
    print(f"  Saved: {out_file}")
    plt.close(fig)
    
    return rms, max_abs


def test_all_h2_components():
    """Test all 9 STO components for H (3 powers × 3 alphas)."""
    results = []
    
    for pow_idx in range(3):  # 0, 1, 2
        for alpha_idx in range(3):  # 0, 1, 2
            rms, max_err = test_h2_single_component(pow_idx, alpha_idx)
            results.append({
                'pow': pow_idx,
                'alpha': alpha_idx,
                'rms': rms,
                'max_err': max_err
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("H2 Masked Component Test Summary")
    print(f"{'='*60}")
    print(f"{'pow':>4} {'alpha':>6} {'RMS':>12} {'Max Err':>12} {'Status':>10}")
    print("-" * 60)
    
    for r in results:
        status = "PASS" if r['rms'] < 1e-10 else ("OK" if r['rms'] < 1e-5 else "FAIL")
        print(f"{r['pow']:>4} {r['alpha']:>6} {r['rms']:>12.2e} {r['max_err']:>12.2e} {status:>10}")
    
    perfect = sum(1 for r in results if r['rms'] < 1e-10)
    print(f"\n{perfect}/9 components achieved perfect parity (RMS < 1e-10)")


if __name__ == '__main__':
    test_all_h2_components()
