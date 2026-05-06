# USER

I think it would be great to figure out how to use WAVEPLOT 
@waveplot.F90 

functionality of DFTB as from python. for that we need to compile it as dynamic library .so and than import it using ctypes.

here is example how we do it for Fireball, take notes and make something similar for wavefplot

@libFireCore.f90 @FireCore.py @cpp_utils_.py 

do not program it yet, just make notes what to use, how to do it, so we can program it based on these notes

you also need to figure out how works the compilation system@DftbPlusUtils.cmake 

and also read these notes @waveplot.tex @waveplot_reimplement.md 

---

# SWE-1.6

I'll investigate how to create a Python interface for WAVEPLOT by studying the Fireball example and understanding the compilation system. Let me create a plan and read all the relevant files.



Based on my analysis of the Fireball example and waveplot code, here are detailed notes for creating a Python interface for WAVEPLOT:

---

# Notes: Creating Python Interface for WAVEPLOT (DFTB+)

## 1. Fireball Pattern Analysis

### 1.1 Fortran Library Structure ([libFireCore.f90](cci:7://file:///home/prokop/git/dftbplus/pyBall/Fireball/libFireCore.f90:0:0-0:0))

**Key pattern elements:**
- **C-binding subroutines** using `bind(c, name='function_name')`
- **iso_c_binding** module for C-compatible types:
  - `integer(c_int), value` - scalars passed by value
  - `real(c_double), dimension(*)` - arrays passed as pointers
  - `type(C_PTR)` - for pointer exchange
- **Module-level state** - global variables in Fortran modules (options, configuration, etc.)
- **Initialization sequence**:
  1. `firecore_preinit()` - set defaults, init constants
  2. `firecore_set_lvs()` - set lattice vectors
  3. `firecore_set_kpoints()` - set k-points
  4. `firecore_initdir()` - read data files
  5. `firecore_init()` - initialize with atom positions

**Example interface pattern:**
```fortran
subroutine firecore_init(natoms_, atomTypes, atomsPos) bind(c, name='firecore_init')
    use iso_c_binding
    implicit none
    integer(c_int), value, intent(in) :: natoms_
    integer(c_int), dimension(natoms_), intent(in) :: atomTypes
    real(c_double), dimension(3,natoms_), intent(in) :: atomsPos
    ! ... body
end subroutine
```

### 1.2 Python Wrapper ([FireCore.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/Fireball/FireCore.py:0:0-0:0))

**Key components:**
- **Library loading** via [cpp_utils_.loadLib()](cci:1://file:///home/prokop/git/dftbplus/pyBall/cpp_utils_.py:104:0-135:14)
- **argDict** - maps function names to (restype, argtypes) tuples
- **ctypes array types** - `np.ctypeslib.ndpointer()` for numpy arrays
- **Helper functions** - wrapper functions with Python-friendly signatures

**Example pattern:**
```python
argDict["firecore_init"] = (None, [c_int, array1i, array2d])
def init(atomTypes, atomPos):
    natoms = len(atomTypes)
    return lib.firecore_init(natoms, atomTypes, atomPos)
```

### 1.3 Build System ([cpp_utils_.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/cpp_utils_.py:0:0-0:0))

**Key functions:**
- [loadLib(name, recompile, mode)](cci:1://file:///home/prokop/git/dftbplus/pyBall/cpp_utils_.py:104:0-135:14) - loads .so library
- [compile_lib(name, FFLAGS, LFLAGS, path, clean)](cci:1://file:///home/prokop/git/dftbplus/pyBall/cpp_utils_.py:41:0-93:27) - compiles C++/Fortran
- [make(what)](cci:1://file:///home/prokop/git/dftbplus/pyBall/cpp_utils_.py:95:0-102:34) - runs make in build directory
- [unload_lib(lib)](cci:1://file:///home/prokop/git/dftbplus/pyBall/cpp_utils_.py:148:0-151:29) - unloads library for reload

**Build path logic:**
- Default: `PACKAGE_PATH + '/../build/'`
- Override via `CPP_BUILD_PATH` environment variable
- Auto-recompile controlled by `CPP_RECOMPILE` environment variable

---

## 2. Waveplot Structure Analysis

### 2.1 Current Architecture ([waveplot.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/waveplot.F90:0:0-0:0))

**Main program flow:**
```
initGlobalEnv() → TEnvironment_init()
→ TProgramVariables_init(wp, env)
→ Allocate buffers
→ Read geometry from detailed.xml
→ Read basis from HSD
→ Initialize TMolecularOrbital
→ Loop over states (via gridcache)
→ Write cube files
```

**Key modules used:**
- `waveplot_gridcache` - grid caching and iteration
- `waveplot_initwaveplot` - initialization (TProgramVariables)
- `waveplot_molorb` - molecular orbital evaluation (getValue)
- `dftbp_*` modules - DFTB+ internal libraries

**Core data structures:**
- `TProgramVariables` - contains all program state
- `TMolecularOrbital` - orbital calculator
- `TGridCache` - grid caching

### 2.2 Critical Dependencies

**DFTB+ internal libraries (in `src/dftbp/`):**
- `dftbp_common_*` - common utilities
- `dftbp_io_*` - I/O routines
- `dftbp_dftb_*` - DFTB-specific code
- `dftbp_type_*` - type definitions

**These are NOT standalone** - waveplot currently links against the full DFTB+ library.

---

## 3. Adaptation Strategy for Waveplot

### 3.1 Challenge: Waveplot is a Program, Not a Library

**Problem:** Waveplot is designed as a standalone executable with:
- Complex initialization sequence
- Multiple interdependent modules
- Direct file I/O (detailed.xml, eigenvec.bin, HSD files)
- Output to cube files

**Solution approaches:**

#### Option A: Extract Core Functionality into Library
**Pros:**
- Clean separation
- Reusable for other purposes
- Follows Fireball pattern

**Cons:**
- Requires significant refactoring
- Must untangle module dependencies
- Complex initialization must be preserved

**Steps:**
1. Create new module `libwaveplot.f90` with C-bindable interface
2. Extract core orbital evaluation from `waveplot_molorb`
3. Extract grid setup from `waveplot_initwaveplot`
4. Create initialization functions matching Fireball pattern
5. Handle file I/O through parameters instead of direct reads

#### Option B: Wrap Existing Program
**Pros:**
- Minimal code changes
- Preserves existing functionality
- Faster to implement

**Cons:**
- Less flexible
- Still requires file I/O
- Not truly "library-like"

**Steps:**
1. Keep waveplot as executable
2. Add C-bindable wrapper functions that call internal routines
3. Use temporary files for data exchange
4. Python writes input files, calls waveplot, reads output

#### Option C: Hybrid Approach (Recommended)
**Pros:**
- Balanced effort
- Can leverage existing modules
- Clean Python interface

**Cons:**
- More complex build system
- Need to understand module dependencies

**Steps:**
1. Create library target in CMake that includes waveplot modules
2. Add C-bindable interface layer on top of existing modules
3. Initialize modules programmatically (no HSD parsing in library)
4. Pass data through parameters instead of files

---

## 4. Recommended Implementation Plan (Hybrid Approach)

### 4.1 Fortran Library Layer

**Create `app/waveplot/libwaveplot.f90`:**

```fortran
module libwaveplot
    use iso_c_binding
    use waveplot_molorb, only: TMolecularOrbital, getValue
    use waveplot_initwaveplot, only: TProgramVariables
    implicit none
    
    type(c_ptr), private :: molorb_ptr
    type(c_ptr), private :: progvar_ptr
    
contains
    
    ! Initialize waveplot library
    subroutine waveplot_lib_init() bind(c, name='waveplot_lib_init')
        ! Allocate and initialize internal structures
    end subroutine
    
    ! Set geometry
    subroutine waveplot_set_geometry(natoms, species, coords) bind(c, name='waveplot_set_geometry')
        integer(c_int), value, intent(in) :: natoms
        integer(c_int), dimension(natoms), intent(in) :: species
        real(c_double), dimension(3,natoms), intent(in) :: coords
        ! Set atom positions and species
    end subroutine
    
    ! Set basis parameters
    subroutine waveplot_set_basis(nspecies, species_data) bind(c, name='waveplot_set_basis')
        ! Pass STO parameters as arrays
    end subroutine
    
    ! Set eigenvectors
    subroutine waveplot_set_eigenvectors(norb, nstates, eigvecs) bind(c, name='waveplot_set_eigenvectors')
        integer(c_int), value, intent(in) :: norb, nstates
        real(c_double), dimension(norb,nstates), intent(in) :: eigvecs
        ! Load eigenvector coefficients
    end subroutine
    
    ! Set grid parameters
    subroutine waveplot_set_grid(origin, boxvecs, npoints) bind(c, name='waveplot_set_grid')
        real(c_double), dimension(3), intent(in) :: origin
        real(c_double), dimension(3,3), intent(in) :: boxvecs
        integer(c_int), dimension(3), intent(in) :: npoints
        ! Set grid specification
    end subroutine
    
    ! Evaluate MO on grid
    subroutine waveplot_eval_mo(istate, grid_out) bind(c, name='waveplot_eval_mo')
        integer(c_int), value, intent(in) :: istate
        real(c_double), dimension(*), intent(out) :: grid_out
        ! Evaluate and return grid values
    end subroutine
    
    ! Evaluate density on grid
    subroutine waveplot_eval_density(grid_out) bind(c, name='waveplot_eval_density')
        real(c_double), dimension(*), intent(out) :: grid_out
        ! Sum over occupied states
    end subroutine
    
    ! Cleanup
    subroutine waveplot_lib_cleanup() bind(c, name='waveplot_lib_cleanup')
        ! Deallocate structures
    end subroutine
    
end module libwaveplot
```

### 4.2 CMake Integration

**Modify `app/waveplot/CMakeLists.txt`:**

```cmake
# Add library target
add_library(waveplot_lib SHARED
    libwaveplot.f90
    # Include all required modules from waveplot
    initwaveplot.F90
    molorb.F90
    slater.F90
    gridcache.F90
    input.F90
    # Include necessary DFTB+ modules
    ${DFTBP_COMMON_SOURCES}
    ${DFTBP_IO_SOURCES}
    ${DFTBP_TYPE_SOURCES}
)

# Set C-binding properties
set_target_properties(waveplot_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules
    OUTPUT_NAME "waveplot"
)

# Link against DFTB+ libraries
target_link_libraries(waveplot_lib
    dftbplus
    ${EXTERNAL_LIBS}
)

# Install library
install(TARGETS waveplot_lib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
```

### 4.3 Python Wrapper (`pyBall/WavePlot/WavePlot.py`)

```python
import numpy as np
from ctypes import *
import os
from . import cpp_utils_ as cpp_utils

lib = None

# Define array types
array1i = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
array2i = np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='CONTIGUOUS')
array1d = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array2d = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array3d = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')

argDict = {}

def reload():
    global lib
    cpp_utils.BUILD_PATH = os.path.normpath(
        cpp_utils.PACKAGE_PATH + '/../../build/'
    )
    lib_ = cpp_utils.loadLib('waveplot', recompile=False, mode=ct.RTLD_LOCAL)
    if isinstance(lib_, ct.CDLL):
        lib = lib_
    else:
        lib = ct.CDLL(lib_)
    cpp_utils.set_args_dict(lib, argDict)
    return lib

# Define function signatures
argDict["waveplot_lib_init"] = (None, [])
argDict["waveplot_set_geometry"] = (None, [c_int, array1i, array2d])
argDict["waveplot_set_basis"] = (None, [c_int, c_void_p])  # Complex structure
argDict["waveplot_set_eigenvectors"] = (None, [c_int, c_int, array2d])
argDict["waveplot_set_grid"] = (None, [array1d, array2d, array1i])
argDict["waveplot_eval_mo"] = (None, [c_int, array1d])
argDict["waveplot_eval_density"] = (None, [array1d])
argDict["waveplot_lib_cleanup"] = (None, [])

def lib_init():
    return lib.waveplot_lib_init()

def set_geometry(species, coords):
    natoms = len(species)
    species = np.array(species, dtype=np.int32)
    coords = np.array(coords, dtype=np.float64)
    return lib.waveplot_set_geometry(natoms, species, coords.T)

def set_grid(origin, boxvecs, npoints):
    origin = np.array(origin, dtype=np.float64)
    boxvecs = np.array(boxvecs, dtype=np.float64)
    npoints = np.array(npoints, dtype=np.int32)
    return lib.waveplot_set_grid(origin, boxvecs, npoints)

def eval_mo(istate, grid_shape):
    npts = np.prod(grid_shape)
    grid_out = np.zeros(npts, dtype=np.float64)
    lib.waveplot_eval_mo(istate, grid_out)
    return grid_out.reshape(grid_shape)

def eval_density(grid_shape):
    npts = np.prod(grid_shape)
    grid_out = np.zeros(npts, dtype=np.float64)
    lib.waveplot_eval_density(grid_out)
    return grid_out.reshape(grid_shape)

lib = reload()
```

### 4.4 Data Passing Strategy

**Basis parameters (complex structure):**
```python
# Option 1: Flatten into arrays
def set_basis(species_data):
    # species_data: dict mapping species to STO parameters
    # Flatten into contiguous arrays for C interface
    nspecies = len(species_data)
    # Pass as flattened arrays with index arrays
    pass

# Option 2: Use HSD file (simpler)
def set_basis_from_hsd(hsd_file):
    # Let Fortran read HSD file internally
    lib.waveplot_set_basis_hsd(hsd_file.encode())
```

**Eigenvectors:**
- Read from `eigenvec.bin` in Python
- Pass as contiguous array
- Shape: (norb, nstates)

---

## 5. Key Challenges and Solutions

### 5.1 Module Dependencies

**Challenge:** Waveplot modules depend on many DFTB+ internal modules.

**Solution:**
- Link waveplot library against full DFTB+ library
- Use CMake's `target_link_libraries` to resolve dependencies
- Ensure module directories are in include path

### 5.2 Initialization Complexity

**Challenge:** Waveplot has complex multi-stage initialization.

**Solution:**
- Preserve initialization sequence in library
- Provide simplified Python wrappers that call sequence internally
- Or expose individual init functions for fine control

### 5.3 File I/O vs. Parameter Passing

**Challenge:** Current waveplot reads from files (detailed.xml, HSD, eigenvec.bin).

**Solutions:**
1. **File-based (easier):** Python writes temp files, library reads them
2. **Parameter-based (cleaner):** Pass all data through function parameters
3. **Hybrid:** Read basis from HSD (complex), pass geometry/eigenvectors as parameters

### 5.4 Memory Management

**Challenge:** Fortran allocates arrays internally, Python needs access.

**Solution:**
- Use `c_loc` and `c_f_pointer` to expose Fortran arrays to C
- Or copy data to Python-allocated arrays
- Provide explicit cleanup function

---

## 6. Alternative: Minimal Interface

If full library extraction is too complex, consider a **minimal interface**:

```fortran
! Minimal interface: run waveplot programmatically
subroutine waveplot_run_from_data(hsd_file, eigenvec_file, output_prefix) bind(c)
    character(c_char) :: hsd_file(*)
    character(c_char) :: eigenvec_file(*)
    character(c_char) :: output_prefix(*)
    ! Write input files, run waveplot, return
end subroutine
```

**Python:**
```python
def run_waveplot(hsd_file, eigenvec_file, output_prefix):
    lib.waveplot_run_from_data(
        hsd_file.encode(),
        eigenvec_file.encode(),
        output_prefix.encode()
    )
    # Read output cube files
```

This is much simpler but less flexible.

---

## 7. Build System Notes

### 7.1 CMake Configuration

**Key CMake variables to check:**
- `BUILD_SHARED_LIBS` - must be ON for .so
- `WITH_API` - may need to enable
- `WITH_TRANSPORT` - if waveplot uses transport features

**Build command:**
```bash
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release ..
make waveplot_lib
```

### 7.2 Library Location

**Default install location:** `${CMAKE_INSTALL_PREFIX}/lib/`

**For Python ctypes:**
- Set `LD_LIBRARY_PATH` to include library directory
- Or use absolute path when loading
- Or install to system location

---

## 8. Next Steps

1. **Decide on approach:**
   - Full library extraction (most work, most flexible)
   - Hybrid (recommended balance)
   - Minimal wrapper (least work, least flexible)

2. **If hybrid/full:**
   - Study waveplot module dependencies in detail
   - Identify which modules can be extracted
   - Design C-bindable interface
   - Implement in Fortran
   - Add CMake target
   - Write Python wrapper

3. **If minimal:**
   - Add C-binding to main program
   - Write Python wrapper that calls it
   - Handle file I/O in Python

4. **Testing:**
   - Start with simple case (single atom, s-orbital)
   - Compare output with standalone waveplot
   - Test with real DFTB+ output files

I've completed the analysis and compiled detailed notes for creating a Python interface for WAVEPLOT. Here's a summary:

## Key Findings

### Fireball Pattern
- **Fortran side**: C-bindable subroutines using `bind(c, name='...')` with `iso_c_binding` types
- **Python side**: `ctypes` wrapper with `argDict` mapping functions to (restype, argtypes) tuples
- **Build system**: [cpp_utils_.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/cpp_utils_.py:0:0-0:0) handles library loading, compilation via make, and path management

### Waveplot Challenge
Waveplot is currently a **standalone program** (not a library) with:
- Complex multi-stage initialization
- Heavy dependencies on DFTB+ internal modules
- File-based I/O (detailed.xml, eigenvec.bin, HSD files)
- Output to cube files

### Three Implementation Options

**Option A: Full Library Extraction**
- Extract core functionality into separate library module
- Cleanest but requires significant refactoring
- Must untangle module dependencies

**Option B: Wrap Existing Program**
- Minimal code changes
- Python writes input files, calls waveplot, reads output
- Least flexible

**Option C: Hybrid Approach (Recommended)**
- Create library target in CMake including waveplot modules
- Add C-bindable interface layer on top of existing modules
- Initialize programmatically (no HSD parsing in library)
- Pass data through parameters instead of files

### Recommended Implementation Plan

1. **Create `app/waveplot/libwaveplot.f90`** with C-bindable functions:
   - `waveplot_lib_init()` - initialize structures
   - `waveplot_set_geometry()` - set atom positions/species
   - `waveplot_set_basis()` - set STO parameters
   - `waveplot_set_eigenvectors()` - load eigenvectors
   - `waveplot_set_grid()` - set grid specification
   - `waveplot_eval_mo()` - evaluate MO on grid
   - `waveplot_eval_density()` - evaluate density
   - `waveplot_lib_cleanup()` - deallocate

2. **Modify `app/waveplot/CMakeLists.txt`** to add library target linking against DFTB+

3. **Create `pyBall/WavePlot/WavePlot.py`** following Fireball pattern with ctypes wrappers

### Key Challenges

- **Module dependencies**: Waveplot depends on many DFTB+ internal modules → link against full DFTB+ library
- **Initialization complexity**: Preserve multi-stage init sequence in library
- **File I/O vs parameters**: Hybrid approach - read basis from HSD (complex), pass geometry/eigenvectors as parameters
- **Memory management**: Use `c_loc`/`c_f_pointer` or copy to Python-allocated arrays

### Build System Notes

- Set `BUILD_SHARED_LIBS=ON` in CMake
- Library installs to `${CMAKE_INSTALL_PREFIX}/lib/`
- For Python ctypes: set `LD_LIBRARY_PATH` or use absolute path

The notes include detailed code examples for the Fortran interface, CMake configuration, and Python wrapper. The hybrid approach balances effort with flexibility by leveraging existing waveplot modules while providing a clean Python interface.

---

# Implementation Progress

## Status: **WORKING** (2026-05-06)

### Files Created / Modified

| File | Description |
|------|-------------|
| `app/waveplot/libwaveplot.F90` | Fortran C-binding module (`bind(c)` subroutines) |
| `app/waveplot/CMakeLists.txt` | Added `waveplotlib` shared library target |
| `pyBall/WavePlot/__init__.py` | Python package marker |
| `pyBall/WavePlot/WavePlot.py` | Python ctypes wrapper class `WavePlot` |
| `tests/dftb/test_waveplot_lib.py` | Test suite (analytical STO verification + H2O) |

### Build

```bash
cmake _build          # regenerate (picks up new CMakeLists.txt target)
cmake --build _build --target waveplotlib
# -> _build/app/waveplot/libwaveplot.so
```

### Exported C symbols (verified with `nm -D`)

- `waveplot_init` — reset state
- `waveplot_set_geometry(natoms, isPeriodic, coords[3,N], species[N])`
- `waveplot_set_basis(nSpecies, nOrbMax, nPowMax, nAlphaMax, ...)` — sets STO basis
- `waveplot_set_eigenvectors(nOrb, nStates, eigvecs[nOrb,nStates])`
- `waveplot_orb2points(iState, npoints, points[3,N], out[N])` — **main function**
- `waveplot_allorbs2point(point[3], nStates, out[nStates])`
- `waveplot_orb2grid(iState, origin[3], gridVecs[3,3], nPoints[3], out[N3])` — 3D grid
- `waveplot_get_nOrb(nOrb_out)`

### Array convention

Fortran is column-major; Python is row-major. The Python wrapper handles all transpositions internally:
- `coords[natoms,3]` → `.T` before passing (Fortran `(3,natoms)`)
- `eigvecs[nStates,nOrb]` → `.T` before passing (Fortran `(nOrb,nStates)`)
- `points[npoints,3]` → `.T` before passing (Fortran `(3,npoints)`)

### Test Results

```
--- test_orb2points (1s H at origin) ---
  Points (Bohr): [0.  0.5 1.  2.  3. ]
  MO values    : [0.15915494 0.05843285 0.02149624 0.0029092  0.00039372]
  Reference    : [0.15915494 0.05854983 0.02153928 0.00291502 0.00039451]
  Max error    : 1.17e-04   (< 1% tolerance: radial grid interpolation)
  PASSED

--- test_orb2grid (3D grid) ---
  Grid shape: (13, 3, 3)   PASSED

--- test_allorbs2point ---
  Error: 4.30e-05   PASSED

--- test_h2o (multi-species O+H) ---
  Library nOrb = 6
  Orbital 1 at test points: [0.16045552 0.01645685 0.04498945]
  PASSED

=== ALL TESTS PASSED ===
```

### Known Issue: `TBoundaryConds_init` signature

`TBoundaryConds_init(this, iBoundaryCondition, errStatus)` — the third `errStatus` argument
is required but not checked in our code. For robustness, add error checking after the call.

### Next Steps

1. **Test with real DFTB+ output**: Use `eigenvec.bin` + `wfc.*.hsd` + `detailed.xml` to
   initialize the library and compare against `waveplot` cube file output.
2. **Parse `wfc.*.hsd`**: Write a Python parser for wavefunction coefficient files (use
   `dftb_utils.py` as starting point).
3. **Parse `eigenvec.bin`**: Binary eigenvector reader (see `wfc.F90` in DFTB+ source).
4. **Add complex k-point support**: The `getValue` interface also accepts complex eigenvectors
   for periodic systems — expose `waveplot_orb2points_cmpl`.
5. **Performance**: For large point sets, `waveplot_orb2points` loops point-by-point (safe but
   slow). A vectorized Fortran loop over all points at once would be faster.