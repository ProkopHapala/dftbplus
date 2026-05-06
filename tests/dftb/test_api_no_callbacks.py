#!/usr/bin/env python
"""
Basic DFTB+ C API test - run calculation without callbacks (energy only)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pyBall.dftb_lib import DftbPlusCalculator

print("=== DFTB+ C API Test (No Callbacks) ===\n")

try:
    calc = DftbPlusCalculator()
    print("✓ Calculator initialized")
    
    calc.initialize(input_file="input.dftb")
    print(f"✓ Input loaded: {calc.nr_atoms} atoms, basis size {calc.basis_size}")
    
    # Try to get energy without registering callbacks
    energy = calc.calculate()
    print(f"✓ Energy calculated: {energy:.6f} Ha")
    
    calc.finalize()
    print("✓ Calculator finalized successfully")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== Test Complete ===")
