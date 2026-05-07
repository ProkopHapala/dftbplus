#!/usr/bin/env python3
"""
Generate comparison summary for H2, O2, H2O using existing compare_waveplot_lib.py.
Reuses existing plotting utilities from plotUtils.py.
"""
import subprocess
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'waveplot_output' / 'comparison_summary'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MOLECULES = {
    'H2': {
        'dftb_dir': 'tests/grid/dftb_h2',
        'mo_range': [1, 2],
        'line_scan': True,
        'atoms': [0, 1],
        'plane2d': 'xz'
    },
    'O2': {
        'dftb_dir': 'tests/grid/dftb_o2',
        'mo_range': [1, 6],
        'line_scan': True,
        'atoms': [0, 1],
        'plane2d': 'xz'
    },
    'H2O_bond': {
        'dftb_dir': 'tests/grid/dftb_h2o_mio',
        'mo_range': [4, 5],
        'line_scan': True,
        'atoms': [0, 1],
        'plane2d': 'xz'
    },
    'H2O_xy_all': {
        'dftb_dir': 'tests/grid/dftb_h2o_mio',
        'mo_range': [1, 6],
        'line_scan': False,
        'atoms': None,
        'plane2d': 'xy'
    },
    'PTCDA_xy': {
        'dftb_dir': 'tests/grid/dftb_ptcda',
        'mo_range': [66, 75],
        'line_scan': False,
        'atoms': None,
        'plane2d': 'xy'
    }
}


def run_comparison(mol_name, config, mode='1d'):
    """Run compare_waveplot_lib.py for a molecule (1D or 2D)."""
    cmd = [
        'python3', 'tests/grid/compare_waveplot_lib.py',
        '--dftb-dir', config['dftb_dir'],
        '--points',
        '--npoints', '64',
        '--mo-range', str(config['mo_range'][0]), str(config['mo_range'][1]),
        '--no-show'
    ]
    
    if mode == '1d':
        cmd.extend(['--line-scan', 'bond'])
        cmd.extend(['--atoms', str(config['atoms'][0]), str(config['atoms'][1])])
    elif mode == '2d':
        cmd.extend(['--plane2d', config['plane2d']])
    
    print(f"Running {mode.upper()} comparison for {mol_name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error for {mol_name} ({mode}):")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def main():
    """Run comparisons for all molecules (1D and 2D)."""
    print("Running comparison summary...")
    print("=" * 60)
    
    results = {}
    for mol_name, config in MOLECULES.items():
        modes_to_run = []
        if config.get('line_scan', False):
            modes_to_run.append('1d')
        modes_to_run.append('2d')
        
        success = True
        for mode in modes_to_run:
            if not run_comparison(mol_name, config, mode=mode):
                success = False
        
        results[mol_name] = success
    
    print("=" * 60)
    print("Summary:")
    for mol_name, success in results.items():
        status = "✓" if success else "✗"
        modes = MOLECULES[mol_name].get('line_scan', False)
        mode_str = "(1D + 2D)" if modes else "(2D only)"
        print(f"  {status} {mol_name} {mode_str}")
    
    print(f"\nIndividual plots saved to: tests/grid/waveplot_output/comparison/")


if __name__ == '__main__':
    main()
