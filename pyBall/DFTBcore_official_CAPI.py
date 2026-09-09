"""
DFTBcore: Python interface to DFTB+ using the official C API with callbacks.

Uses libdftbplus.so with callback mechanism for matrix export.

Usage:
    from pyBall.DFTBcore import DFTBcore
    
    dftb = DFTBcore()
    dftb.init('h2o.hsd')
    energy = dftb.run_scf()
    
    # Get matrices via callbacks (stored automatically)
    H = dftb.get_h()
    S = dftb.get_s()
    dm = dftb.get_dm()
    
    dftb.finalize()
"""

import ctypes
import os
import numpy as np

# Find libdftbplus.so
_POSSIBLE_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', '..', '_build', 'src', 'dftbp', 'libdftbplus.so'),
    os.path.join(os.getcwd(), '_build', 'src', 'dftbp', 'libdftbplus.so'),
    os.path.expanduser('~/opt/dftbplus/lib/libdftbplus.so'),
]

_DEFAULT_LIB = None
for path in _POSSIBLE_PATHS:
    path = os.path.normpath(os.path.abspath(path))
    if os.path.exists(path):
        _DEFAULT_LIB = path
        break


# C handler structures (must match Fortran bind(C) types)
class c_DftbPlus(ctypes.Structure):
    _fields_ = [("instance", ctypes.c_void_p)]

class c_DftbPlusInput(ctypes.Structure):
    _fields_ = [("pDftbPlusInput", ctypes.c_void_p)]


# Callback function types
DMHS_CALLBACK = ctypes.CFUNCTYPE(
    None,  # return type
    ctypes.c_void_p,   # aux_ptr
    ctypes.c_int,      # iK
    ctypes.c_int,      # iS
    ctypes.POINTER(ctypes.c_int),  # blacs_descr
    ctypes.c_void_p,   # blacs_data
    ctypes.c_void_p    # matrix_descr
)


class MatrixDescr(ctypes.Structure):
    """Matrix descriptor from DFTB+ C API."""
    _fields_ = [
        ("matrix_type", ctypes.c_int),
        ("storage_type", ctypes.c_int),
    ]


class DFTBcore:
    """Wrapper around official DFTB+ C API with matrix callbacks."""

    def __init__(self, libpath=None):
        if libpath is None:
            if _DEFAULT_LIB is None:
                raise FileNotFoundError("libdftbplus.so not found. Build DFTB+ with -DBUILD_SHARED_LIBS=ON")
            libpath = _DEFAULT_LIB

        self._lib = ctypes.CDLL(libpath, mode=ctypes.RTLD_GLOBAL)
        self._handler = c_DftbPlus()
        
        # Storage for matrices collected via callbacks
        self._H = None
        self._S = None
        self._dm = None
        self._basis_size = 0
        
        # Keep references to callback functions to prevent garbage collection
        self._h_callback = DMHS_CALLBACK(self._h_callback_impl)
        self._s_callback = DMHS_CALLBACK(self._s_callback_impl)
        self._dm_callback = DMHS_CALLBACK(self._dm_callback_impl)
        
        self._setup_signatures()

    def _setup_signatures(self):
        lib = self._lib
        
        # dftbp_init(handler, output_path)
        lib.dftbp_init.argtypes = [ctypes.POINTER(c_DftbPlus), ctypes.c_char_p]
        lib.dftbp_init.restype = None
        
        # dftbp_final(handler)
        lib.dftbp_final.argtypes = [c_DftbPlus]
        lib.dftbp_final.restype = None
        
        # dftbp_get_input_from_file(handler, filename, input_handler)
        lib.dftbp_get_input_from_file.argtypes = [
            c_DftbPlus, ctypes.c_char_p, ctypes.POINTER(c_DftbPlusInput)
        ]
        lib.dftbp_get_input_from_file.restype = None
        
        # dftbp_process_input(handler, input_handler) - THIS RUNS THE CALCULATION
        lib.dftbp_process_input.argtypes = [c_DftbPlus, c_DftbPlusInput]
        lib.dftbp_process_input.restype = None
        
        # dftbp_input_final(input_handler)
        lib.dftbp_input_final.argtypes = [ctypes.POINTER(c_DftbPlusInput)]
        lib.dftbp_input_final.restype = None
        
        # dftbp_get_energy(handler, energy)
        lib.dftbp_get_energy.argtypes = [c_DftbPlus, ctypes.POINTER(ctypes.c_double)]
        lib.dftbp_get_energy.restype = None
        
        # dftbp_get_basis_size(handler) -> returns int
        lib.dftbp_get_basis_size.argtypes = [c_DftbPlus]
        lib.dftbp_get_basis_size.restype = ctypes.c_int
        
        # Callback registration
        lib.dftbp_register_h_callback.argtypes = [c_DftbPlus, DMHS_CALLBACK, ctypes.c_void_p]
        lib.dftbp_register_h_callback.restype = None
        
        lib.dftbp_register_s_callback.argtypes = [c_DftbPlus, DMHS_CALLBACK, ctypes.c_void_p]
        lib.dftbp_register_s_callback.restype = None
        
        lib.dftbp_register_dm_callback.argtypes = [c_DftbPlus, DMHS_CALLBACK, ctypes.c_void_p]
        lib.dftbp_register_dm_callback.restype = None

    def _h_callback_impl(self, aux_ptr, iK, iS, blacs_descr, data_ptr, matrix_descr):
        """Callback for Hamiltonian matrix export."""
        self._store_matrix(data_ptr, matrix_descr, 'H')

    def _s_callback_impl(self, aux_ptr, iK, iS, blacs_descr, data_ptr, matrix_descr):
        """Callback for overlap matrix export."""
        self._store_matrix(data_ptr, matrix_descr, 'S')

    def _dm_callback_impl(self, aux_ptr, iK, iS, blacs_descr, data_ptr, matrix_descr):
        """Callback for density matrix export."""
        self._store_matrix(data_ptr, matrix_descr, 'dm')

    def _store_matrix(self, data_ptr, matrix_descr_ptr, name):
        """Store matrix from callback data."""
        # Parse matrix descriptor
        descr = MatrixDescr.from_address(matrix_descr_ptr)
        n = descr.n_rows
        self._basis_size = n
        
        # Copy data from Fortran buffer
        # data_ptr points to n*n doubles in Fortran (column-major) order
        data = np.ctypeslib.as_array(
            ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_double)),
            shape=(n, n)
        ).copy()
        
        # Store in correct attribute
        if name == 'H':
            self._H = data
        elif name == 'S':
            self._S = data
        elif name == 'dm':
            self._dm = data

    def init(self, input_file):
        """Initialize DFTB+ from HSD input file."""
        # Initialize with output to stdout ("-" means stdout)
        self._lib.dftbp_init(ctypes.byref(self._handler), b"-")
        
        # Register callbacks for matrix collection
        self._lib.dftbp_register_h_callback(self._handler, self._h_callback, None)
        self._lib.dftbp_register_s_callback(self._handler, self._s_callback, None)
        self._lib.dftbp_register_dm_callback(self._handler, self._dm_callback, None)
        
        # Read input from file
        input_handler = c_DftbPlusInput()
        self._lib.dftbp_get_input_from_file(
            self._handler, 
            input_file.encode('utf-8'),
            ctypes.byref(input_handler)
        )
        
        # Process input (this runs the SCF calculation)
        self._lib.dftbp_process_input(self._handler, input_handler)
        
        # Finalize input handler
        self._lib.dftbp_input_final(ctypes.byref(input_handler))

    def get_energy(self):
        """Get total energy from calculation. Returns energy in Hartree."""
        energy = ctypes.c_double()
        self._lib.dftbp_get_energy(self._handler, ctypes.byref(energy))
        return energy.value

    def get_basis_size(self):
        """Return number of basis functions."""
        if self._basis_size == 0:
            self._basis_size = self._lib.dftbp_get_basis_size(self._handler)
        return self._basis_size

    def get_h(self):
        """Get Hamiltonian matrix (norb x norb)."""
        if self._H is None:
            raise RuntimeError("Hamiltonian not collected. Run SCF first.")
        return self._H

    def get_s(self):
        """Get overlap matrix (norb x norb)."""
        if self._S is None:
            raise RuntimeError("Overlap not collected. Run SCF first.")
        return self._S

    def get_dm(self):
        """Get density matrix (norb x norb)."""
        if self._dm is None:
            raise RuntimeError("Density matrix not collected. Run SCF first.")
        return self._dm

    def finalize(self):
        """Finalize DFTB+ and cleanup."""
        if self._handler.instance:
            self._lib.dftbp_final(self._handler)
            self._handler.instance = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finalize()
        return False


def run_dftb_calculation(input_file, libpath=None):
    """Convenience function to run DFTB+ and extract matrices."""
    with DFTBcore(libpath=libpath) as dftb:
        dftb.init(input_file)  # This runs the calculation
        energy = dftb.get_energy()
        return {
            'energy': energy,
            'basis_size': dftb.get_basis_size(),
            'h': dftb.get_h(),
            's': dftb.get_s(),
            'dm': dftb.get_dm()
        }
