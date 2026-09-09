#!/usr/bin/env python3
"""
Utility to evaluate and visualize multi-zeta STO basis functions from wfc.*.hsd files.

This script:
1. Parses wfc.*.hsd files to extract multi-zeta STO parameters
2. Evaluates basis functions on a real-space grid
3. Generates PNG plots for selected atoms with overlays
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from pyBall.OCL.DFTBplusParser import parse_wfc_hsd, evaluate_sto_1d, evaluate_sto_2d
from pyBall.plotUtils import plot_sto_1d, plot_sto_2d, plot_sto_2d_separate, plot_sto_2d_overlay, plot_sto_radial_comparison


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot STO basis functions from wfc.*.hsd files')
    parser.add_argument('wfc_file', help='Path to wfc.*.hsd file')
    parser.add_argument('--species', nargs='+', required=True, help='Species to plot (e.g., H O)')
    parser.add_argument('--orbitals', nargs='+', type=int, default=[0], help='Orbital indices (default: 0 for s-orbital)')
    parser.add_argument('--plot-type', choices=['1d', '2d', 'overlay', 'radial'], default='radial',
                        help='Plot type (default: radial)')
    parser.add_argument('--grid-size', type=float, default=10.0, help='Grid size in Angstrom (default: 10.0)')
    parser.add_argument('--n-points', type=int, default=200, help='Number of grid points (default: 200)')
    parser.add_argument('--output', help='Output PNG file path')
    parser.add_argument('--origins', nargs='+', type=float, help='Atom origins as x1 y1 x2 y2 ... (default: 0 0 ...)')
    
    args = parser.parse_args()
    
    # Parse wfc file
    basis_data = parse_wfc_hsd(args.wfc_file)
    
    # Handle origins
    if args.origins:
        if len(args.origins) != 2 * len(args.species):
            raise ValueError(f"Expected {2*len(args.species)} origin values, got {len(args.origins)}")
        origins = [(args.origins[2*i], args.origins[2*i+1]) for i in range(len(args.species))]
    else:
        origins = None
    
    # Ensure orbital indices match species count
    if len(args.orbitals) < len(args.species):
        args.orbitals = args.orbitals * len(args.species)
    
    # Plot
    if args.plot_type == '1d':
        for species_name, orbital_idx in zip(args.species, args.orbitals):
            output = args.output if len(args.species) == 1 else None
            if output and len(args.species) > 1:
                output = output.replace('.png', f'_{species_name}.png')
            
            if species_name not in basis_data:
                print(f"Warning: Species {species_name} not found, skipping")
                continue
            
            species = basis_data[species_name]
            if orbital_idx >= len(species['orbitals']):
                print(f"Warning: Orbital index {orbital_idx} out of range for {species_name}, skipping")
                continue
            
            orbital = species['orbitals'][orbital_idx]
            l = orbital['AngularMomentum']
            exps = orbital['Exponents']
            coeffs = orbital['Coefficients']
            cutoff = orbital['Cutoff']
            
            # Create radial grid (in Bohr for STO evaluation, then convert to Angstrom for plotting)
            BOHR2ANG = 0.5291772109
            r_bohr = np.linspace(0, 5.0 / BOHR2ANG, 500)
            r_ang = r_bohr * BOHR2ANG
            
            # Evaluate STO (r in Bohr)
            sto = evaluate_sto_1d(r_bohr, l, exps, coeffs)
            
            # Plot (r in Angstrom, cutoff in Angstrom)
            cutoff_ang = cutoff * BOHR2ANG
            title = f'{species_name} orbital {orbital_idx} (l={l})'
            plot_sto_1d(r_ang, sto, l, cutoff_ang, title, output_path=output)
            print(f"Saved 1D plot to {output}")
            
    elif args.plot_type == '2d':
        n_atoms = len(args.species)
        if origins is None:
            origins = [(0, 0)] * n_atoms
        
        # Create 2D grid
        x = np.linspace(-args.grid_size/2, args.grid_size/2, args.n_points)
        y = np.linspace(-args.grid_size/2, args.grid_size/2, args.n_points)
        X, Y = np.meshgrid(x, y)
        
        sto_list = []
        l_list = []
        origin_list = []
        title_list = []
        
        for species_name, orbital_idx, origin in zip(args.species, args.orbitals, origins):
            if species_name not in basis_data:
                print(f"Warning: Species {species_name} not found, skipping")
                continue
            
            species = basis_data[species_name]
            if orbital_idx >= len(species['orbitals']):
                print(f"Warning: Orbital index {orbital_idx} out of range for {species_name}, skipping")
                continue
            
            orbital = species['orbitals'][orbital_idx]
            l = orbital['AngularMomentum']
            exps = orbital['Exponents']
            coeffs = orbital['Coefficients']
            
            # Evaluate STO on 2D grid
            sto = evaluate_sto_2d(X, Y, l, exps, coeffs, origin=origin)
            
            sto_list.append(sto)
            l_list.append(l)
            origin_list.append(origin)
            title_list.append(f'{species_name} orbital {orbital_idx} (l={l})')
        
        plot_sto_2d_separate(X, Y, sto_list, l_list, origin_list, title_list, output_path=args.output)
        print(f"Saved 2D plot to {args.output}")
        
    elif args.plot_type == 'overlay':
        n_atoms = len(args.species)
        if origins is None:
            origins = [(0, 0)] * n_atoms
        
        # Create 2D grid
        x = np.linspace(-args.grid_size/2, args.grid_size/2, args.n_points)
        y = np.linspace(-args.grid_size/2, args.grid_size/2, args.n_points)
        X, Y = np.meshgrid(x, y)
        
        sto_list = []
        l_list = []
        origin_list = []
        title_list = []
        
        for species_name, orbital_idx, origin in zip(args.species, args.orbitals, origins):
            if species_name not in basis_data:
                print(f"Warning: Species {species_name} not found, skipping")
                continue
            
            species = basis_data[species_name]
            if orbital_idx >= len(species['orbitals']):
                print(f"Warning: Orbital index {orbital_idx} out of range for {species_name}, skipping")
                continue
            
            orbital = species['orbitals'][orbital_idx]
            l = orbital['AngularMomentum']
            exps = orbital['Exponents']
            coeffs = orbital['Coefficients']
            
            # Evaluate STO on 2D grid
            sto = evaluate_sto_2d(X, Y, l, exps, coeffs, origin=origin)
            
            sto_list.append(sto)
            l_list.append(l)
            origin_list.append(origin)
            title_list.append(f'{species_name} (l={l})')
        
        plot_sto_2d_overlay(X, Y, sto_list, l_list, origin_list, title_list, output_path=args.output)
        print(f"Saved overlay plot to {args.output}")
        
    elif args.plot_type == 'radial':
        # Create list of (species_name, orbital_idx, label) tuples
        species_orbitals = []
        for species_name, orbital_idx in zip(args.species, args.orbitals):
            if species_name not in basis_data:
                print(f"Warning: Species {species_name} not found, skipping")
                continue
            
            species = basis_data[species_name]
            if orbital_idx >= len(species['orbitals']):
                print(f"Warning: Orbital index {orbital_idx} out of range for {species_name}, skipping")
                continue
            
            orbital = species['orbitals'][orbital_idx]
            l = orbital['AngularMomentum']
            l_label = 's' if l == 0 else 'p' if l == 1 else 'd' if l == 2 else f'l={l}'
            label = f'{species_name} {l_label}'
            species_orbitals.append((species_name, orbital_idx, label))
        
        if species_orbitals:
            plot_sto_radial_comparison(basis_data, species_orbitals, r_max=5.0, n_points=500, 
                                      output_path=args.output)
            print(f"Saved radial comparison plot to {args.output}")
        else:
            print("No valid species/orbitals to plot")


if __name__ == '__main__':
    main()
