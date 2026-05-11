
CODEMAP: DFTB+ Python Integration: Library Interfaces, Parsers, and OpenCL Grid Projection
https://windsurf.com/codemaps/1d6b4b7c-04de-49ef-b581-12cf5bfef54a-fe86ab10a43f3d18

# USER


this repo is for development of DFTB integration. The advantaage is that we have fotran DFTB backend and python frontend in one repo. But I would like to move (resp copy) some functionality in my other project. So basically I need to make detailed documentation how to do it. The python modules need to find precompiled .so ibraries of DFTB like libdftbcore.so and libwaveplot.so  and load them properly into python using CDLL, and then the functionality like relax geometry, comput density matrix, orbitals coefs, hamiltonian, and on to pof them compute projection on grid using pyOpenCL or using libwaveplot should be copied into different distanc folder in different repol

1) make list of python files which we should copy so that it is self-contained package without external dependencies. (perhaps also .cl file for pyopenbcl module)
2) provide intruction for migration (how to set paths etc. to make it robuts, list current existing hard coded path, and metion how to change them dynamically in robuts way)
3) also consider paths to basis set slakos and the @wfc.3ob-3-1.hsd @wfc.mio-1-1.hsd 

the most relevant python modules are those @Grid.py @DFTBcore.py @WavePlot.py @DFTBplusParser.py 

there are test which shows how to use them which we would like to also migrate to different project (which will not contina DFTB backeld .so lib but will call them from distant source)
@test_waveplot_dftbcore.py @compare_density_multizeta.py 

there is very relevant .cl file
@Grid.py 

now analyse all the code, and make detailed comprehesive notes about how to migrate it and what we should not forget (explore all dependencies and how to poer them)

write these notes here in the chat

do not code anything yet

---

# SWE-1.6

Based on my analysis of the codebase, here are comprehensive migration notes for extracting the DFTB-related Python modules:

# DFTB Python Modules Migration Documentation

## 1. Core Python Files to Copy

### Essential Modules (self-contained package):

**Primary modules:**
- [pyBall/OCL/Grid.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:0:0-0:0) - Main grid projection using pyOpenCL
- [pyBall/DFTBcore.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/DFTBcore.py:0:0-0:0) - ctypes interface to libdftbcore.so
- [pyBall/WavePlot/WavePlot.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/WavePlot/WavePlot.py:0:0-0:0) - ctypes interface to libwaveplot.so
- [pyBall/OCL/DFTBplusParser.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/DFTBplusParser.py:0:0-0:0) - Parser for DFTB+ output files (detailed.xml, eigenvec.bin, .hsd basis files)

**Required dependencies:**
- [pyBall/OCL/OpenCLBase.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/OpenCLBase.py:0:0-0:0) - Base class for OpenCL operations (imported by Grid.py)
- [pyBall/OCL/clUtils.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/clUtils.py:0:0-0:0) - OpenCL utilities (imported by OpenCLBase.py)

**OpenCL kernel files:**
- [pyBall/OCL/cl/Grid.cl](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/cl/Grid.cl:0:0-0:0) - OpenCL kernels for density/orbital projection

**Test files (for reference/migration):**
- [tests/grid/test_waveplot_dftbcore.py](cci:7://file:///home/prokop/git/dftbplus/tests/grid/test_waveplot_dftbcore.py:0:0-0:0) - Shows how to use DFTBcore + Grid projection
- [tests/grid/compare_density_multizeta.py](cci:7://file:///home/prokop/git/dftbplus/tests/grid/compare_density_multizeta.py:0:0-0:0) - Shows libwaveplot vs pyOpenCL comparison

### Optional/Utility Modules (not required for core functionality):
- [pyBall/OCL/plot_sto_basis.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/plot_sto_basis.py:0:0-0:0) - Visualization utilities (requires matplotlib)
- [pyBall/plotUtils.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/plotUtils.py:0:0-0:0) - Plotting utilities (requires matplotlib)

---

## 2. External Python Dependencies

### Required (no external compilation needed):
- `numpy` - Array operations
- `pyopencl` - OpenCL Python bindings
- `ctypes` - Standard library (for loading .so libraries)

### Optional (for visualization only):
- `matplotlib` - Plotting (not required for core computation)

---

## 3. Hard-Coded Paths Analysis

### DFTBcore.py (lines 42-62):

**Current hard-coded paths:**
```python
_POSSIBLE_PATHS = [
    # Relative to pyBall/DFTBcore.py
    os.path.join(os.path.dirname(__file__), '..', '..', '_build', 'app', 'dftbcore', 'libdftbcore.so'),
    # From current working directory
    os.path.join(os.getcwd(), '_build', 'app', 'dftbcore', 'libdftbcore.so'),
    # Installed location
    os.path.expanduser('~/opt/dftbplus/lib/libdftbcore.so'),
    os.path.expanduser('~/git/dftbplus/_build/app/dftbcore/libdftbcore.so'),
]
_INSTALLED_LIB = os.path.expanduser('~/opt/dftbplus/lib/libdftbcore.so')
```

**Migration strategy:**
- Replace with environment variable: `DFTBCORE_LIB_PATH`
- Or allow explicit `libpath` parameter (already supported in [__init__](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/clUtils.py:131:4-141:79))
- Fallback to system library search paths (e.g., [/usr/local/lib](cci:9://file:///usr/local/lib:0:0-0:0), [/usr/lib](cci:9://file:///usr/lib:0:0-0:0))

### WavePlot.py (lines 25-29):

**Current hard-coded path:**
```python
_DEFAULT_LIB = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '_build', 'app', 'waveplot', 'libwaveplot.so'
)
```

**Migration strategy:**
- Replace with environment variable: `WAVEPLOT_LIB_PATH`
- Or allow explicit `libpath` parameter (already supported in [__init__](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/clUtils.py:131:4-141:79))
- Fallback to system library search paths

### Grid.py (line 124):

**Current hard-coded path:**
```python
cl_path = os.path.join(os.path.dirname(__file__), "cl/Grid.cl")
```

**Migration strategy:**
- This is relative and should work if directory structure is preserved
- Alternatively, allow kernel path override via constructor parameter

---

## 4. Basis Set and HSD File Paths

### Basis set files referenced in tests:

**Official DFTB+ basis files:**
- `tests/grid/dftb_ptcda/wfc.3ob-3-1.hsd` - 3ob-3-1 parameter set
- `tests/grid/dftb_ptcda/wfc.mio-1-1.hsd` - mio-1-1 parameter set

**Runtime basis files (generated by DFTB+):**
- `waveplot_in.hsd` - Contains STO basis parameters for waveplot output
- `dftb_in.hsd` - Main DFTB+ input file

**Path handling in DFTBplusParser.py:**
- [parse_basis_hsd_ang()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/DFTBplusParser.py:609:0-761:23) function (line 610) handles `<<+ "filename.hsd"` include directives
- Automatically resolves relative paths for included wfc files
- Converts Bohr units to Angstrom for OpenCL kernels

**Migration strategy:**
- Use environment variable: `DFTB_BASIS_PATH`
- Or allow explicit path parameter to parser functions
- Document that basis files must be accessible relative to working directory

---

## 5. Dynamic Path Configuration Recommendations

### Proposed robust path resolution:

```python
# For DFTBcore.py
def get_lib_path(lib_name, env_var=None, default_paths=None):
    """Robust library path resolution."""
    # 1. Check explicit parameter
    if libpath is not None:
        return libpath
    
    # 2. Check environment variable
    if env_var and env_var in os.environ:
        return os.environ[env_var]
    
    # 3. Check default locations
    for path in default_paths:
        if os.path.exists(path):
            return path
    
    # 4. Check system library paths
    for lib_dir in ['/usr/local/lib', '/usr/lib', '/opt/lib']:
        candidate = os.path.join(lib_dir, lib_name)
        if os.path.exists(candidate):
            return candidate
    
    raise FileNotFoundError(f"{lib_name} not found")
```

### Environment variables to define:

- `DFTBCORE_LIB_PATH` - Path to libdftbcore.so
- `WAVEPLOT_LIB_PATH` - Path to libwaveplot.so
- `DFTB_BASIS_PATH` - Base directory for basis set files
- `OPENCL_KERNEL_PATH` - Optional override for Grid.cl location

---

## 6. Data Flow and Dependencies

### Typical workflow (from test files):

1. **Run DFTB+ calculation** → generates:
   - `detailed.xml` (geometry, occupations, k-points)
   - `eigenvec.bin` (eigenvectors)
   - `waveplot_in.hsd` (basis parameters)

2. **Parse output files** using DFTBplusParser:
   - [parse_detailed_xml_custom()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/DFTBplusParser.py:874:0-935:5) → geometry, species, occupations
   - [parse_eigenvec_bin_custom()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/DFTBplusParser.py:938:0-956:16) → eigenvectors
   - [parse_basis_hsd_ang()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/DFTBplusParser.py:609:0-761:23) → STO basis in Angstrom units

3. **Load basis into GridProjector**:
   - [load_basis_sto(species_list)](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:32:4-120:27) → precomputes radial functions on uniform grid

4. **Project to grid**:
   - [project_orbital()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:1238:4-1356:18) or [project()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:375:4-538:18) → density/orbital on 3D grid

### Key data transformations:

- **Units**: Bohr (DFTB+ internal) ↔ Angstrom (OpenCL kernels)
  - Conversion factor: `BOHR2ANG = 0.5291772109`
  - DFTBplusParser handles this automatically in [parse_basis_hsd_ang()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/DFTBplusParser.py:609:0-761:23)

- **Array ordering**: Fortran (column-major) ↔ C (row-major)
  - DFTBcore.get_eigvecs_dense() handles conversion
  - DFTBplusParser.parse_eigenvec_bin_custom() handles conversion

- **Coefficient ordering**: Fortran [s, py, pz, px] ↔ OpenCL [px, py, pz, s]
  - Handled by [evec_to_kernel_coeffs()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/DFTBplusParser.py:1051:0-1083:12) in DFTBplusParser

---

## 7. File Structure for Migration

### Recommended directory structure in target project:

```
target_project/
├── dftb_integration/
│   ├── __init__.py
│   ├── Grid.py              # from pyBall/OCL/Grid.py
│   ├── DFTBcore.py          # from pyBall/DFTBcore.py
│   ├── WavePlot.py          # from pyBall/WavePlot/WavePlot.py
│   ├── DFTBplusParser.py    # from pyBall/OCL/DFTBplusParser.py
│   ├── OpenCLBase.py        # from pyBall/OCL/OpenCLBase.py
│   ├── clUtils.py           # from pyBall/OCL/clUtils.py
│   └── cl/
│       └── Grid.cl          # from pyBall/OCL/cl/Grid.cl
├── basis_sets/              # Copy of .hsd files
│   ├── wfc.3ob-3-1.hsd
│   └── wfc.mio-1-1.hsd
└── tests/                   # Migrated test files
    ├── test_waveplot_dftbcore.py
    └── compare_density_multizeta.py
```

---

## 8. Critical Migration Steps

### Step 1: Update import statements

Change from:
```python
from pyBall.OCL.Grid import GridProjector
from pyBall.DFTBcore import DFTBcore
```

To:
```python
from dftb_integration.Grid import GridProjector
from dftb_integration.DFTBcore import DFTBcore
```

### Step 2: Replace hard-coded paths with environment variables

Modify [DFTBcore.__init__()](cci:1://file:///home/prokop/git/dftbplus/pyBall/DFTBcore.py:82:4-107:32):
```python
def __init__(self, libpath=None):
    if libpath is None:
        libpath = os.environ.get('DFTBCORE_LIB_PATH')
        if libpath and os.path.exists(libpath):
            self._lib = ctypes.CDLL(libpath, mode=ctypes.RTLD_LOCAL)
        else:
            # Try default paths...
```

### Step 3: Update relative path for OpenCL kernel

In [Grid._load_kernels()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:122:4-145:103):
```python
def _load_kernels(self):
    # Allow override via environment variable
    cl_path = os.environ.get('OPENCL_KERNEL_PATH')
    if cl_path is None:
        cl_path = os.path.join(os.path.dirname(__file__), "cl", "Grid.cl")
```

### Step 4: Test library loading

Create a simple test script:
```python
import os
os.environ['DFTBCORE_LIB_PATH'] = '/path/to/libdftbcore.so'
os.environ['WAVEPLOT_LIB_PATH'] = '/path/to/libwaveplot.so'

from dftb_integration import DFTBcore, WavePlot

dftb = DFTBcore()  # Should load from env var
wp = WavePlot()     # Should load from env var
```

---

## 9. Potential Issues and Solutions

### Issue 1: Library ABI compatibility
- **Problem**: .so files compiled with specific DFTB+ version may not be compatible
- **Solution**: Document required DFTB+ version and compile flags

### Issue 2: Basis set file paths
- **Problem**: Tests assume basis files in specific locations
- **Solution**: Use `DFTB_BASIS_PATH` environment variable or explicit path parameter

### Issue 3: OpenCL device selection
- **Problem**: Default device selection may not work on all systems
- **Solution**: Allow device selection via environment variable `OPENCL_DEVICE_INDEX`

### Issue 4: Array ordering confusion
- **Problem**: Fortran vs C ordering can cause bugs
- **Solution**: Document which functions expect which ordering, add assertions

---

## 10. Summary of Required Changes

1. **Copy files**: 7 Python files + 1 .cl file
2. **Update imports**: Change `pyBall.*` to new package name
3. **Replace hard-coded paths**: Use environment variables with fallbacks
4. **Document dependencies**: numpy, pyopencl, ctypes
5. **Update test files**: Change imports and paths
6. **Create configuration guide**: Document environment variables and required files

### Minimum viable package:
- Grid.py
- DFTBcore.py
- DFTBplusParser.py
- OpenCLBase.py
- clUtils.py
- cl/Grid.cl

### Optional additions:
- WavePlot.py (if using libwaveplot.so instead of pure OpenCL)
- plotUtils.py (if visualization needed)