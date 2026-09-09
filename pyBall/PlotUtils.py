"""
Simple plotting utilities for DFTB+ density and orbital projection.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_orbital_2d(values, extent, atom_coords, title, output_path, dpi=150, cmap='RdBu_r'):
    """Plot a single orbital in 2D.
    
    Args:
        values: (nx, ny) array of orbital values
        extent: [xmin, xmax, ymin, ymax] for matplotlib
        atom_coords: (natoms, 3) array in Angstrom
        title: plot title
        output_path: path to save figure
        dpi: figure DPI
        cmap: colormap name
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    clim = np.max(np.abs(values))
    im = ax.imshow(values, origin='lower', cmap=cmap, vmin=-clim, vmax=clim, extent=extent)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    
    # Add atoms
    ax.scatter(atom_coords[:, 0], atom_coords[:, 1], c='black', marker='.', s=10, alpha=0.5, zorder=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"  Saved: {output_path}")

def plot_density_2d(values, extent, atom_coords, title, output_path, dpi=150, cmap='viridis'):
    """Plot density in 2D.
    
    Args:
        values: (nx, ny) array of density values
        extent: [xmin, xmax, ymin, ymax] for matplotlib
        atom_coords: (natoms, 3) array in Angstrom
        title: plot title
        output_path: path to save figure
        dpi: figure DPI
        cmap: colormap name
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(values, origin='lower', cmap=cmap, extent=extent)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    
    # Add atoms
    ax.scatter(atom_coords[:, 0], atom_coords[:, 1], c='black', marker='.', s=10, alpha=0.5, zorder=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"  Saved: {output_path}")

def plot_orbitals_grid(mo_values, mo_indices, extent, atom_coords, output_path, dpi=150, cols=3):
    """Plot multiple orbitals in a grid layout.
    
    Args:
        mo_values: list of (nx, ny) arrays
        mo_indices: list of MO indices (1-based for display)
        extent: [xmin, xmax, ymin, ymax]
        atom_coords: (natoms, 3) array
        output_path: path to save figure
        dpi: figure DPI
        cols: number of columns
    """
    n = len(mo_values)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(n):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        clim = np.max(np.abs(mo_values[i]))
        im = ax.imshow(mo_values[i], origin='lower', cmap='RdBu_r', vmin=-clim, vmax=clim, extent=extent)
        ax.set_title(f'MO{mo_indices[i]}')
        plt.colorbar(im, ax=ax)
        ax.scatter(atom_coords[:, 0], atom_coords[:, 1], c='black', marker='.', s=5, alpha=0.5, zorder=10)
    
    # Hide unused subplots
    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"  Saved: {output_path}")

def plot_density_comparison(densities, titles, extent, atom_coords, output_path, dpi=150):
    """Plot multiple density calculations side by side.
    
    Args:
        densities: list of (nx, ny) arrays
        titles: list of titles
        extent: [xmin, xmax, ymin, ymax]
        atom_coords: (natoms, 3) array
        output_path: path to save figure
        dpi: figure DPI
    """
    n = len(densities)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1:
        axes = [axes]
    
    for i, (dens, title) in enumerate(zip(densities, titles)):
        im = axes[i].imshow(dens, origin='lower', cmap='viridis', extent=extent)
        axes[i].set_title(title)
        plt.colorbar(im, ax=axes[i])
        axes[i].scatter(atom_coords[:, 0], atom_coords[:, 1], c='black', marker='.', s=10, alpha=0.5, zorder=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"  Saved: {output_path}")
