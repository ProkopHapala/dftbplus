# Grid Projector STO Modification Notes

## Current State (Grid.py + Grid.cl)

### Data Flow
1. **Host-side (Grid.py)**:
   - `load_basis_sto()` takes species_list with multi-zeta STO parameters
   - Evaluates STO analytically on uniform grid using `compute_sto_radial()`
   - Computes cubic spline second derivatives using `_spline_d2_uniform()`
   - Packs as `float2(value, d2)` per node
   - Layout: `packed_basis[n_species][max_shells][n_nodes][2]`
   - Uploads to GPU as single buffer

2. **GPU-side (Grid.cl)**:
   - `evaluate_radial()` does cubic B-spline interpolation:
     - Finds grid interval for given r
     - Interpolates using pre-computed values + spline derivatives
     - Formula: `psi = a*ylo + b*yhi + ((a^3-a)*d2lo + (b^3-b)*d2hi)*(h^2)/6`

### Current STO Evaluation (Host-side)
```python
# In DFTBplusParser.py::compute_sto_radial()
def compute_sto_radial(r, aa, alpha, ll):
    """
    Formula: R_l(r) = sum_{i=1}^{nAlpha} [ sum_{j=1}^{nPow} aa(j,i) * r^{l+j-1} ] * exp(-alpha_i * r)
    
    Args:
        aa: coefficient matrix (nPow, nAlpha)
        alpha: exponents (nAlpha,)
        ll: angular momentum
    """
    # This is the multi-zeta STO formula
    # Already supports multiple exponents and coefficients
```

### Key Parameters
- `max_shells = 2` (fixed: s and p only, H p-shell = zeros)
- `dr`: grid spacing (typically ~0.075 Å)
- `n_nodes`: typically ~50-100 nodes per shell
- `packed_basis`: shape (n_species, 2, n_nodes, 2)

## Old State (Grid_fireball.py + Grid_fireball.cl)

### Data Flow
1. **Host-side (Grid_fireball.py)**:
   - `load_basis()` reads numerical radial functions from .wf files
   - Each shell has its own grid (different dr, different rcutoff)
   - Resamples all shells to common uniform grid using cubic spline
   - Computes spline second derivatives
   - Packs as `float2(value, d2)` per node
   - Layout: `packed_basis[n_species][max_shells][n_nodes][2]`
   - Uploads to GPU

2. **GPU-side (Grid_fireball.cl)**:
   - `evaluate_radial()` - IDENTICAL to current Grid.cl
   - Same cubic B-spline interpolation

### Key Difference
- **Old**: Reads pre-computed numerical radial functions from disk
- **New**: Computes STO analytically from parameters (exponents, coefficients)
- **Both**: Use same GPU interpolation approach (cubic B-spline on pre-computed grid)

## Memory vs Compute Trade-offs

### Current Approach (Pre-computed Grid)
**Advantages:**
- Fast GPU evaluation (simple spline interpolation)
- Works well for single-zeta or multi-zeta STO
- GPU kernel is simple and efficient

**Disadvantages:**
- Memory: `n_species * max_shells * n_nodes * 2 * sizeof(float)` bytes
  - For 10 species, 2 shells, 100 nodes: ~16 KB (negligible)
  - For 100 species, 2 shells, 100 nodes: ~160 KB (still small)
- Grid resolution trade-off: finer grid = more memory but more accurate
- Must re-compute if basis parameters change

### On-the-fly STO Evaluation
**Advantages:**
- Minimal memory (only store exponents and coefficients)
- No grid resolution trade-off
- Flexible (can change parameters without re-computation)

**Disadvantages:**
- More GPU compute: evaluate exp() and power functions for each voxel
- More complex GPU kernel
- May be slower for dense grids

**Formula to evaluate:**
```
STO(r) = sum_{i=1}^{nAlpha} exp(-exps[i] * r) * sum_{j=1}^{nPow} coeffs[j,i] * r^{l + j - 1}
```

This requires:
- `nAlpha * nPow` multiplications per voxel
- `nAlpha` exp() calls per voxel
- Power terms: can be computed iteratively: `r^l, r^(l+1), r^(l+2), ...`

## Proposed Modifications

### Option 1: Keep Current Approach (Recommended)
**Rationale:** Current approach already supports multi-zeta STO correctly. Memory usage is negligible (~160 KB for 100 species). GPU kernel is simple and fast.

**Changes needed:**
- None! Current implementation already handles multi-zeta STO.
- Just ensure `load_basis_sto()` is called with correct multi-zeta parameters from wfc.*.hsd files.

### Option 2: Add On-the-fly STO Evaluation
**Rationale:** For flexibility and to avoid pre-computation. Could be useful for very large systems or when basis parameters change frequently.

**Changes needed:**

**Host-side (Grid.py):**
```python
def load_basis_sto_params(species_list):
    """
    Load STO parameters (exponents, coefficients) without pre-computation.
    Store as structured data for on-the-fly GPU evaluation.
    
    Data layout:
    - species_params[n_species][max_shells]
    - Each shell: {l, nAlpha, nPow, exponents[nAlpha], coeffs[nPow][nAlpha], cutoff}
    """
    # Pack into GPU buffer
    # Layout: [n_species][max_shells][max_nAlpha][max_nPow + 1]  # +1 for exponents
```

**GPU-side (Grid.cl):**
```c
// New kernel function for on-the-fly STO evaluation
float evaluate_sto_on_the_fly(
    float r, 
    int ityp, int ish,
    __global const float* sto_params,  // [n_species][max_shells][...]
    int max_nAlpha, int max_nPow
) {
    // Load parameters for this species/shell
    int base = (ityp * max_shells + ish) * (max_nAlpha * (max_nPow + 1));
    int l = sto_params[base];  // first element stores l
    int nAlpha = sto_params[base + 1];
    int nPow = sto_params[base + 2];
    float cutoff = sto_params[base + 3];
    
    if (r > cutoff) return 0.0f;
    
    // Load exponents and coefficients
    __global const float* exps = sto_params + base + 4;
    __global const float* coeffs = exps + max_nAlpha;  // coeffs follows exps
    
    // Compute r powers: r^l, r^(l+1), r^(l+2), ...
    float r_pow = (l == 0 && r < 1e-12f) ? 1.0f : pow(r, (float)l);
    float pows[MAX_NPOW];
    for (int j = 0; j < nPow; j++) {
        pows[j] = r_pow;
        r_pow *= r;
    }
    
    // Evaluate STO
    float sto = 0.0f;
    for (int i = 0; i < nAlpha; i++) {
        float radial_sum = 0.0f;
        for (int j = 0; j < nPow; j++) {
            radial_sum += coeffs[j * max_nAlpha + i] * pows[j];
        }
        sto += radial_sum * exp(-exps[i] * r);
    }
    
    return sto;
}
```

**Trade-offs:**
- Need to define `MAX_NALPHA` and `MAX_NPOW` compile-time constants
- For wfc.mio-1-1.hsd: H has 3 exponents, C/O/N have 4 exponents
- Could use `MAX_NALPHA = 4`, `MAX_NPOW = 4` (conservative)
- Memory: `n_species * 2 * 4 * (4+1) * sizeof(float)` = ~640 bytes for 10 species (tiny!)

### Option 3: Hybrid Approach
**Rationale:** Keep current pre-computed grid for performance, but add on-the-fly option for flexibility.

**Implementation:**
- Add flag to `load_basis_sto()`: `precompute=True/False`
- If `precompute=True`: use current approach (spline on grid)
- If `precompute=False`: use on-the-fly STO evaluation
- GPU kernel checks flag at runtime or uses different kernel

## Recommended Approach

**Keep current pre-computed grid approach** because:
1. Memory usage is negligible (~160 KB for 100 species)
2. GPU kernel is simple and fast (spline interpolation is cheap)
3. Already correctly implements multi-zeta STO
4. No changes needed - just use correct parameters from wfc.*.hsd

**If on-the-fly evaluation is desired later:**
- Can add as optional feature without breaking existing code
- Useful for very large systems or dynamic basis changes
- Implementation is straightforward (see Option 2 above)

## Implementation Steps (if Option 2 is chosen)

1. **Add compile-time constants to Grid.cl:**
```c
#define MAX_NALPHA 4
#define MAX_NPOW 4
```

2. **Modify Grid.py to support both modes:**
```python
def load_basis_sto(self, species_list, precompute=True, dr=None, rc_max=None):
    if precompute:
        # Current implementation
        self._load_basis_sto_precompute(species_list, dr, rc_max)
    else:
        self._load_basis_sto_params(species_list)
```

3. **Add on-the-fly evaluation function to Grid.cl**
4. **Modify GPU kernel to use appropriate evaluation method**
5. **Add tests for both modes**

## Key Caveats

1. **Units:** Current implementation uses Angstrom for GPU evaluation. STO parameters must be in Angstrom^-1 for exponents.

2. **Angular momentum:** Current implementation assumes max_shells=2 (s and p). For d-orbitals, need to increase max_shells.

3. **Normalization:** STO functions from wfc.*.hsd are not necessarily normalized. Current approach preserves whatever normalization is in the file.

4. **Cutoff:** STO functions should be zero beyond cutoff. Current spline interpolation may give small non-zero values beyond cutoff - need to clamp to zero.

5. **Performance:** On-the-fly evaluation will be slower. Benchmark both approaches before deciding.
