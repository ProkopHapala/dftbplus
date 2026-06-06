#!/usr/bin/env python3
"""
Stub FdataParser for DFTB+ compatibility.

This is a minimal stub to allow Grid.py to be imported when using DFTB+.
The actual Fireball-specific functionality is not needed for DFTB+.
"""

class FdataParser:
    """Minimal stub for Fireball data parser - not used in DFTB+."""
    
    def __init__(self, fdata_dir):
        self.fdata_dir = fdata_dir
        # Stub - no actual parsing needed for DFTB+
        pass
    
    def parse(self):
        """Stub method - not used in DFTB+."""
        pass
