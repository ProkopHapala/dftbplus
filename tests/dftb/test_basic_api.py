#!/usr/bin/env python
"""
Basic DFTB+ C API test - initialize and finalize only (no callbacks)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pyBall.dftb_lib import DftbPlusCalculator

print("=== Basic DFTB+ C API Test ===\n")

try:
    calc = DftbPlusCalculator()
    print("✓ Calculator initialized with lib:", calc.lib_path)
    
    calc.initialize(input_file="input.dftb")
    print(f"✓ Input loaded: {calc.nr_atoms} atoms, basis size {calc.basis_size}")
    
    calc.finalize()
    print("✓ Calculator finalized successfully")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== Test Complete ===")
