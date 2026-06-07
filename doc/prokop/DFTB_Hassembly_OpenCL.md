# USER

Look into DFTB in rust 
@hamiltonian.rs @interpolation.rs @rotation.rs @gamma.rs 
assembly of hamiltonian

and try to figure out how to efficient do this in OpenCL what should be the kernels, how many kernel we should have how should we map atom, how to use shared memory. 

we are goin to assemble many replicas of the same system. Each system will be one workgroup, each atom one thread he will iterate over his neighobrs. Or we can also parallelize over blocks (this is importaint desing decission) but I think it is probably better to paralleize over atoms. If we paralelize over blocs (pairs of atoms) we should create the maping of pairs on CPU in rust. For small systems block-based desing is better (we will utilize morh threads even for small systems)

think how to efficiently pre-load SK basis in local memory, how to do rotations etc. 
maybe what can help you is @Grid.cl 

you should not code now, rather analyse the desing and suggest optimal paht, eventually ask me to important desing decision from proposed variants


---

# Kimi-2.6

I'll analyze the Rust DFTB assembly code and the existing OpenCL patterns to propose a kernel design. Let me read all the relevant files first.

Let me also look at the SK data structures and the remaining parts of [Grid.cl](cci:7://file:///home/prokophapala/git/dftbplus/pyBall/OCL/cl/Grid.cl:0:0-0:0) for local memory patterns.

## OpenCL Hamiltonian Assembly: Design Analysis

### What the Rust code does

The [HamiltonianBuilder](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/hamiltonian.rs:127:0-129:1) loops over neighbor pairs, for each pair:
1. **SK interpolation**: [EqGridTable.eval_into()](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:41:4-46:5) → Neville 8-point interpolation on uniform grid (5 KB stack temporaries)
2. **Shell extraction**: [eval_shell_integrals_into()](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:97:4-139:5) → maps extended/old-format columns to shell pair `(ang1, ang2)`
3. **Rotation**: [Rotation::rotate_diatomic_block_into()](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/rotation.rs:188:4-254:5) → direction cosines `l,m,n` applied to ss/sp/ps/pp blocks
4. **Symmetrize**: writes block to `H[i,j]` and `H[j,i]` with transpose/sign rule

For sp-only: blocks are 4×4 (1 s + 3 p), max ~16 elements. Full spd could be 9×9 = 81 elements.

### Core Design Decision: Atom-Parallel vs Block-Parallel

| Approach | Thread = atom | Thread = atom-pair |
|---|---|---|
| **Workgroup** | 1 system (replica) | 1 system (replica) |
| **Thread work** | Iterate own neighbor list, accumulate to global H/S | 1 pair → compute block, atomic-add to H/S |
| **Local memory use** | Cache neighbor coords/species/orb_offsets | Cache coords + maybe SK tables |
| **Pros** | Natural scatter-add pattern; load atom data once; good load balancing if neighbor counts similar | Maximum parallelism even for tiny systems (e.g. 4 atoms = 6 pairs); no per-atom atomics |
| **Cons** | Low GPU utilization for small systems (<64 atoms); divergent neighbor counts cause warp divergence | Must build pair list on CPU; atomic-add contention on H/S matrix; more global writes |

**My recommendation**: **Hybrid**. For small systems (< ~64 atoms), use block-parallel with pair-list built on CPU. For larger systems, atom-parallel. If you must pick one, **block-parallel** is safer for replica ensembles because you likely have many small molecules. The CPU pair-list cost is negligible compared to GPU work.

---

### Kernel Design Proposal

#### Option A: Block-Parallel (preferred for replicas)

```c
// 1 workgroup = 1 replica
// thread      = 1 pair (or multiple pairs via striding)
__kernel void assemble_hamiltonian(
    __global const float4* coords,      // [n_replicas][n_atoms] xyz packed
    __global const int*    species,     // [n_replicas][n_atoms]
    __global const int*    orb_offset,  // [n_replicas][n_atoms+1]
    __global const Pair*   pairs,       // [total_pairs] {i,j, r, vec_ij[3]}
    __global const float*  sk_tables,   // flattened SK grid data
    __global const int*    sk_meta,     // per-pair-table: dr, n_grid, n_integ, offset
    __global float*        H,           // [n_replicas][n_orbs][n_orbs]
    __global float*        S,
    int n_pairs, int n_orbs, ...
)
```

- **No atomics needed if each pair writes 2 symmetric locations directly** (just 2 writes per element).
- Each thread processes `for (int p = tid; p < n_pairs; p += wg_size)`.

#### Option B: Atom-Parallel

```c
__kernel void assemble_hamiltonian_atom(
    __global const float4* coords,
    __global const int*    neighbors,   // [n_atoms][max_neigh] j indices
    __global const int*    n_neigh,     // [n_atoms]
    ...
)
// thread = atom i, loops j in neighbors[i]
```

### Local/Shared Memory Strategy

For block-parallel over **pairs**, the hot data per pair is:

| Data | Size | Strategy |
|---|---|---|
| Coords of atoms i,j | 2×`float4` | Already in registers from global read |
| Direction cosines `l,m,n` | 3 floats | Computed on-the-fly |
| SK table row (8-point stencil) | `n_integ × 8` floats | **Preload to local memory** if same species pair repeats |
| Rotated block (4×4 or 9×9) | ≤81 floats | Registers |

**Key insight**: In a single workgroup (1 replica), many pairs share the same species-pair type (e.g., C-C, C-H, H-H). Preloading the SK table for the active species-pairs into `__local` memory pays off.

Suggested local memory layout per workgroup:
```c
__local float l_sk_stencil[MAX_ACTIVE_TABLES * MAX_N_INTEG * N_INTER];
__local int   l_sk_info[MAX_ACTIVE_TABLES];   // species-pair → table index mapping
```

Build the active-table map on CPU (it's per-replica, constant during assembly). Pass it as a small `__constant` or `__global` array.

### Rotation in OpenCL

The [rotation.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/rotation.rs:0:0-0:0) code has hardcoded formulas:

- **ss**: `out[0] = sk[0]` — trivial
- **sp**: `out = [m*sp, n*sp, l*sp]` — 3 FMAs
- **ps**: same as sp, then transposed with sign `-1`
- **pp**: 9-element dense formula in `l,m,n` — 18 FMAs + 6 muls

These are cheap enough to compute per-pair in registers. **No local memory needed for rotation matrices.** Just compute `l,m,n` from `vec_ij / r` and apply the closed-form expressions inline.

For sp-only, the entire [fill_pairs_sp_only](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/hamiltonian.rs:304:4-391:5) logic collapses to ~50 FLOPs per pair after SK interpolation.

### SK Interpolation on GPU

The Neville interpolation in [interpolation.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:0:0-0:0) uses:
- Stack arrays: `xa[8]`, `yb[20][8]`, `cc[20][8]`, `dd[20][8]`, `delta[7]`
- Total ~5 KB temporaries

In OpenCL, this should be **registers / private memory** per thread. For sp-only, `n_integ <= 2` per shell pair, so the working set shrinks to ~`2 × 8 × 3 = 48` floats — trivial.

However, **random access into SK tables** (global memory) is the bottleneck. SK tables are 1D grids: `values[n_grid][n_integ]`. Access pattern is determined by `r` per pair, essentially random.

**Optimization**: If you tile over pairs that map to the same species-pair and similar `r`, you get coalesced SK reads. But with independent threads per pair, this is hard. Better: **convert SK tables to textures/images** or `__constant` if small enough. For many replicas, each thread reads its own `r`, so memory is naturally coalesced per warp if pairs are sorted by species-pair.

### Data Layout for Replicas

For `n_replicas` of the same system geometry (but different coords, same topology):

| Buffer | Layout | Notes |
|---|---|---|
| `coords` | `[n_replicas][n_atoms]` `float4` | SoA or AoS — `float4` per atom is fine |
| `species` | `[n_replicas][n_atoms]` `int` | If all replicas have identical species, pass once as `__constant` |
| `pairs` | `[n_replicas][n_pairs]` or global pool with offsets | If topology fixed, one neighbor list serves all replicas |
| `H, S` | `[n_replicas][n_orbs][n_orbs]` | Row-major; each workgroup writes one matrix |

If all replicas have **identical species and neighbor topology** (only coords differ), you only need:
- 1 species array
- 1 pair/neighbor list (offsets into coords per replica)

### Recommended Kernel Architecture

```
Kernels:
1. [CPU or GPU] build_neighbor_list() — only if coords change
2. [GPU] assemble_onsite()  — trivial, can fuse into main kernel
3. [GPU] assemble_pairs()     — main kernel
```

For the main `assemble_pairs` kernel (block-parallel):

```c
__kernel void assemble_pairs(
    __global const float4* coords,      // [n_replicas][n_atoms]
    __global const int*    atom_species,
    __global const int*    atom_orb_off,
    __global const int2*   pair_list,     // [n_pairs] (i,j)
    __global const float*  pair_r,      // [n_pairs] precomputed r
    __global const float4* pair_vec,    // [n_pairs] vec_ij packed
    __global const SkTable* sk_tables,  // metadata + pointers to data
    __global float* H_out,              // [n_replicas][n_orbs*n_orbs]
    __global float* S_out,
    int n_atoms, int n_orbs, int n_pairs_per_replica
)
{
    int replica = get_group_id(0);
    int tid     = get_local_id(0);
    int nwg     = get_local_size(0);

    // Each thread handles multiple pairs via striding
    for (int p = tid; p < n_pairs_per_replica; p += nwg) {
        // read pair
        int2 ij = pair_list[p];
        float4 vec = pair_vec[p];
        float  r   = pair_r[p];
        int si = atom_species[ij.x];
        int sj = atom_species[ij.y];

        // SK interpolation (private memory temporaries)
        float sk_h[4], sk_s[4];
        eval_sk_integrals(si, sj, r, sk_h, sk_s);

        // Rotation (private memory output)
        float block_h[16], block_s[16];  // or 81 for spd
        rotate_diatomic_sp(si, sj, vec, sk_h, sk_s, block_h, block_s);

        // Write symmetric blocks to global H/S
        int bi = atom_orb_off[ij.x];
        int bj = atom_orb_off[ij.y];
        int base = replica * n_orbs * n_orbs;
        write_block_symmetric(base, bi, bj, n_orbs, block_h, H_out);
        write_block_symmetric(base, bi, bj, n_orbs, block_s, S_out);
    }
}
```

### Questions for You

1. **System size range?** Are replicas small (<32 atoms) or large (>500 atoms)? This decides atom-parallel vs block-parallel.

2. **Do all replicas share the same species/topology?** If yes, we can pre-build one neighbor list and one SK active-table map on CPU.

3. **Precision?** `f64` or `f32`? OpenCL f64 support is limited on consumer GPUs. If `f32` is acceptable, throughput doubles.

4. **SCC or non-SCC?** SCC adds gamma-function evaluation and charge-dependent H updates, which changes whether H is rebuilt every SCC iteration.

5. **Do you want to fuse onsite + pair assembly into one kernel, or keep them separate?** Onsite is trivial and could be a small second kernel or the first `tid==0` work in each workgroup.

---

# USER

OK, lets focus on the pair-based desing and investigate it more. 
The main memory bottleneck are the SK tables. therefore it would be very beneficial to sort evaluantion by SK combination fist (combination of elements) packing together pairs from different sub-systems.such radical desin could save singiticant amout of memory bandwithd and shared memory. workgrou would preload only single species pair like C-C or C-H so thread in workgroup would iterate over blok of same size (4x4 or 1x4, we hsould chose is H if j or i index - major or minor index in the nested loop)

in pairwise assmebly the on-site terms should be perhaps independnet. But the electronstatics is only onsite (?), so then we should grop the onsite kernel and the electrostatic kenrel? 

precission MUST be single-point (32) since our GPU is very slow in double (10x slower)

ad topology/neighbor list - we can perhaps make pairwise kenrel agnostic to that, it is responsibilty of CPU to give each workgroup proper offset index to given atom position, and hamiltonian block dand this will be kept constant during calculation. 

later GPU can perhaps help with neighborlit culling (filter atoms which are beyond cutoff) but for the moment we will do it on CPU. First by brute force (good engouth form small systems) later maybe by some grid boxes. But do not overenginner it now (just put it as notes for future) focus on proper pairwise hamiltiona andssembly and sorting by species pairs, and then perhaps on the electrostatics and on-site kernel 

---

# Kimi-2.6

## Pair-Based Design: Species-Pair Sorted Kernels

### Core Idea

CPU pre-sorts **all pairs across all replicas** by species-pair type (e.g., C-C, C-H, H-H). Each species-pair bucket gets its own kernel launch. Every workgroup in that launch loads **one compact SK table** into `__local` memory and processes only pairs of that type.

This amortizes SK bandwidth across thousands of pairs and makes every thread in the workgroup execute the exact same code path (same block sizes, same rotation formulas).

---

### 1. CPU-Side Pair Bucketing (Rust)

For each replica, build the neighbor list, then scatter pairs into species-pair buckets:

```rust
struct PairEntry {
    replica: u32,      // which replica
    orb_i:   u16,      // orbital offset of atom i
    orb_j:   u16,      // orbital offset of atom j  
    r:       f32,      // distance
    l:       f32,      // direction cosine
    m:       f32,
    n:       f32,
}
```

CPU also builds **compact SK tables** per species-pair: only the columns actually needed for the shell combinations. For sp-only that's 4 integrals (`ss_sig`, `sp_sig`, `pp_sig`, `pp_pi`) instead of 10 or 20. Upload as two buffers: `sk_h[species_pair][n_grid][4]` and `sk_s[...]`.

### 2. OpenCL Kernel

```c
typedef struct {
    uint replica;
    ushort orb_i, orb_j;
    float r, l, m, n;
} PairEntry;

__kernel void assemble_pairs_sp(
    __global const PairEntry* pairs,      // [n_pairs] sorted by species-pair
    __global const float*   sk_h,       // compact table [n_grid][4]
    __global const float*   sk_s,
    const float dr, const int n_grid,
    __global float* H_out,                // [n_replicas][n_orbs][n_orbs]
    __global float* S_out,
    const int n_orbs
)
{
    const int tid = get_local_id(0);
    const int wg  = get_local_size(0);
    const int gid = get_global_id(0);
    const int n_pairs = ...; // passed as arg or uniform

    // --- LOCAL SK CACHE ---
    // 4 integrals * n_grid * 2 (H+S) floats
    // Example: n_grid=500 → 500*4*2*4 = 16 KB
    __local float l_sk_h[SK_GRID_MAX * 4];
    __local float l_sk_s[SK_GRID_MAX * 4];

    for (int i = tid; i < n_grid * 4; i += wg) {
        l_sk_h[i] = sk_h[i];
        l_sk_s[i] = sk_s[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid >= n_pairs) return;
    PairEntry p = pairs[gid];

    // --- SK INTERPOLATION (private registers) ---
    float sk_vals_h[4], sk_vals_s[4];
    interpolate_sk_compact(l_sk_h, l_sk_s, p.r, dr, n_grid, sk_vals_h, sk_vals_s);

    // --- ROTATION (private registers, 4x4 block) ---
    float blk_h[16], blk_s[16];
    rotate_sp_block(p.l, p.m, p.n, sk_vals_h, sk_vals_s, blk_h, blk_s);

    // --- WRITE SYMMETRIC BLOCKS ---
    const uint base = p.replica * (n_orbs * n_orbs);
    write_block_4x4_symmetric(H_out, base, n_orbs, p.orb_j, p.orb_i, blk_h);
    write_block_4x4_symmetric(S_out, base, n_orbs, p.orb_j, p.orb_i, blk_s);
}
```

**Key properties:**
- **Zero divergence**: every thread does exactly the same interpolation stencil size, same rotation formulas, same 4×4 block write.
- **Local memory**: one SK table load per workgroup, reused by all threads.
- **Memory**: each pair entry is 24 bytes. For 10k pairs × 100 replicas = small.
- **Writes**: each thread writes 32 floats (H block + transpose) + 32 floats (S). Scatter is unavoidable but coarsened by the bucket sort.

### 3. Interpolation in Registers

The Neville loop from [interpolation.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:0:0-0:0) becomes ~40 f32 registers per thread:

```c
inline void interpolate_sk_compact(
    __local const float* tab_h, __local const float* tab_s,
    float r, float dr, int n_grid,
    float out_h[4], float out_s[4]
)
{
    // Find stencil window (8 points)
    // Direct-index into __local tab — no global memory, no texture needed
    // ~200 FLOPs, all in registers
}
```

Because the table is in `__local`, even divergent `r` values per thread just hit fast shared memory. No need for image/texture tricks.

### 4. Onsite Kernel (separate)

```c
__kernel void assemble_onsite_sp(
    __global const int* atom_species,     // [n_atoms]
    __global const int* orb_offsets,      // [n_atoms]
    __global const float2* onsite_es_ep,  // [n_species] (e_s, e_p)
    __global float* H_out,
    const int n_atoms, const int n_orbs, const int n_replicas
)
{
    int idx = get_global_id(0);
    int total = n_replicas * n_atoms;
    if (idx >= total) return;

    int replica = idx / n_atoms;
    int atom    = idx % n_atoms;
    int base    = replica * n_orbs * n_orbs;
    int off     = orb_offsets[atom];
    float2 e    = onsite_es_ep[atom_species[atom]];

    // s orbital
    H_out[base + off * n_orbs + off] = e.x;
    // p orbitals (py, pz, px order if matching Fortran)
    H_out[base + (off+1)*n_orbs + (off+1)] = e.y;
    H_out[base + (off+2)*n_orbs + (off+2)] = e.y;
    H_out[base + (off+3)*n_orbs + (off+3)] = e.y;
}
```

This is ~4 writes per atom, trivially coalesced if you launch `n_replicas * n_atoms` threads.

### 5. What About SCC / Electrostatics?

The user asked: *"the electrostatics is only onsite (?), so then we should group the onsite kernel and the electrostatic kernel?"*

In **SCC DFTB**:
- `H = H0 + H1`, where `H0` is the non-SCC Hamiltonian (what we just built).
- `H1` comes from the γ-matrix: `H1_{μν} = 0.5 * S_{μν} * Σ_C (q_C * (γ_{AC} + γ_{BC}))` where atom A owns orbital μ and B owns ν.
- γ is computed per **atom pair**, not orbital pair, including off-diagonal γ(r).

So for SCC you need:
1. **`gamma_kernel`**: compute `γ[n_replicas][n_atoms][n_atoms]` from distances and Hubbard U values. This is atom-pair parallel, very similar structure to the H-pair kernel.
2. **`h1_update_kernel`**: atom-orbital parallel, updates H using pre-built S, γ, and atomic charges.

**Recommendation**: Keep `assemble_onsite` separate from electrostatics. In SCC, you will run:
```
assemble_pairs → assemble_onsite → gamma_kernel → h1_update_kernel
```

Fusing onsite with gamma doesn't make sense because gamma depends on charges that are only known after the first H/S build.

### 6. Sorting by `r` Within Bucket (optional future)

If you want to push further: within each species-pair bucket, sort pairs by `r`. Then threads in a warp access nearby SK grid indices → potential for warp-level cooperative caching if the table doesn't fit in `__local`. But with compact tables (16 KB), `__local` already solves this. Sorting by `r` is only useful if you later move to a direct global-memory stencil without `__local` preload.

### 7. Design Decisions to Confirm

| Question | My recommendation | Your call |
|---|---|---|
| **Block-parallel vs atom-parallel** | Block-parallel, sorted by species-pair | ✅ You confirmed |
| **f32 precision** | All kernels f32 | ✅ You confirmed |
| **Kernel count** | `assemble_pairs` (per species-pair) + `assemble_onsite` (global) | Confirm if acceptable |
| **Compact SK tables** | CPU extracts 4 columns per species-pair | Confirm if you want this |
| **Precompute `l,m,n` on CPU** | Yes, pass in `PairEntry` | Confirm |
| **SCC electrostatics** | Separate `gamma_kernel` + `h1_update_kernel` in future | Confirm scope |

If you agree with the pair-bucket design, the next step is to write the Rust CPU-side bucketing code and the OpenCL kernel skeleton. Should I proceed?

---

# USER

good, now we need to figure out what we need to prepare on the Rust host side. I do now want to introduce OpenCL dependency yet. What I want to do is prepare function producing the necessary arrays.

this means
1) B-spline data for Slater-Koster tables
2) properly ordered list of pairs (reshufled or sorted so that same type pairs are together so we can preload the SK data in local memory shared by the whole workgroup.
the input should be multiple systems (nSys,natoms) each witch each own geometry (maybe even types - not sure if having different types or number of atoms in each system breaks some optimization? if not keep it flexible). we should first find pairs within cutoff, compute the geometry parameters (distance eventually angle or cos-agnle which we need for interpolation - cos angle is faster comput by vectro without goniometric functions)

anything elese? think it through and then write detaield notes what to implement on CPU/rust


---

# Kimi-2.6

## What the Kernels Actually Need

### Kernel 1: `onsite_and_va`

| Buffer | Shape | Source |
|--------|-------|--------|
| `atom_species` | `[n_atoms]` | SystemContext.atom_species |
| `orb_offsets` | `[n_atoms]` | SystemContext.atom_orb_off |
| `onsite_es_ep` | `[n_species]` | AtomicParamsSp (e_s, e_p) |
| `charges` | `[n_replicas][n_atoms]` | SCC charges (or q0 for non-SCC) |
| `hubbard_u` | `[n_species]` | GammaTable.hubbard_u |
| `neigh_offsets` | `[n_atoms+1]` | CSR row ptr (gamma cutoff) |
| `neigh_j` | `[n_neigh_total]` | CSR col idx |
| `neigh_r` | `[n_neigh_total]` | CSR distances |
| `H_out` | `[n_replicas][n_orbs][n_orbs]` | Output (zeroed before kernel) |
| `V_out` | `[n_replicas][n_atoms]` | Output |

### Kernel 2: `assemble_pairs` (per bucket)

| Buffer | Shape | Source |
|--------|-------|--------|
| `pairs` | `[n_pairs]` | PairEntry structs, sorted by replica |
| `sk_h` | `[n_grid × n_sk_cols]` | B-spline table H |
| `sk_s` | `[n_grid × n_sk_cols]` | B-spline table S |
| `V_a` | `[n_replicas][n_atoms]` | Output from `onsite_and_va` |
| `H_out` | `[n_replicas][n_orbs][n_orbs]` | Output |
| `S_out` | `[n_replicas][n_orbs][n_orbs]` | Output |

## Critical Design Constraint

**All replicas must have identical `n_atoms` and `n_orbs`**. The kernels use:
```c
base_H = replica * n_orbs * n_orbs;
base_Q = replica * n_atoms;
```

`n_orbs` and `n_atoms` are scalar kernel args — one value for the entire launch.

**Options for multiple systems**:
1. **Require uniform replicas** (same system, different geometry/charges). Padded to max if needed. Best for MD/MC ensembles.
2. **Launch per-replica separately** with individual buffers. More flexible, more kernel launches.
3. **Add per-replica offset arrays** to kernels. More complex, but handles heterogeneous systems.

**Recommendation**: Start with option 1 (uniform replicas). For heterogeneous systems, pad to max `n_atoms` / `n_orbs` and use a mask. The kernel design is optimized for batching.

## Rust Functions to Implement

### 1. `GpuSkTable` — Compact SK Table for GPU

```rust
pub struct GpuSkTable {
    pub sk_h: Vec<f32>,       // [n_grid × n_sk_cols] packed
    pub sk_s: Vec<f32>,       // [n_grid × n_sk_cols] packed
    pub n_grid: usize,
    pub dr: f32,
    pub n_sk_cols: usize,     // 1, 2, or 4
    pub block_type: u8,       // 0, 1, 2
    pub species_i: u8,
    pub species_j: u8,
}
```

**Packing logic**:
- For each grid point `r_k`, evaluate shell integrals from [EqGridTable](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:16:0-19:1)
- Extract columns based on block type:
  - block_type 0 (s-s): `[ss_h, ss_s]` → but wait, sk_h and sk_s are separate arrays
  - Actually: `sk_h` stores only H integrals, `sk_s` stores only S integrals
  - For block_type 0: `sk_h[k] = ss_h`, `sk_s[k] = ss_s`
  - For block_type 1: `sk_h[2k] = ss_h`, `sk_h[2k+1] = sp_h`, `sk_s[2k] = ss_s`, `sk_s[2k+1] = sp_s`
  - For block_type 2: `sk_h[4k] = ss_h`, `sk_h[4k+1] = sp_h`, `sk_h[4k+2] = pp_sig_h`, `sk_h[4k+3] = pp_pi_h`

The kernel uses `__local float2* tab2 = (__local float2*)tab` for block_type 1, so columns must be interleaved as `(ss, sp)` pairs per grid point.

**Key question**: The current [EqGridTable](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:16:0-19:1) has `values[n_grid][n_integ]` where [n_integ](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:26:4-28:5) is 10 or 20 (all SK integrals). We need to extract the relevant shell integrals per grid point and pack them.

**Function signature**:
```rust
impl GpuSkTable {
    pub fn from_sk_table_sp(tab: &SkTableSp, sp1: &str, sp2: &str, sk_data: &SkData) -> Result<Self>;
}
```

For each grid point, call [tab.eval_shell_integrals_into(ang1, ang2, r, out_h, out_s)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:97:4-139:5) to get shell integrals, then pack into compact format.

**Note on `ang1`, `ang2`**: These come from [SpeciesOrbitals.ang_shells](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:177:4-182:5). For sp-sp, we need to evaluate shells (0,0), (0,1), (1,1) to get ss, sp, pp_sig, pp_pi. But the kernel's `interp_sk_4` expects a single float4 per grid point with `(ss, sp, pp_sig, pp_pi)`.

So we need to pre-evaluate all shell combinations and pack the resulting integrals. This is different from the current CPU approach which evaluates on-the-fly per pair.

**Implementation**:
```rust
pub fn pack_sk_tables(sk_data: &SkData, ctx: &SystemContext) -> Result<Vec<GpuSkTable>> {
    let mut tables = Vec::new();
    for (si, sp_i) in species.iter().enumerate() {
        for (sj, sp_j) in species.iter().enumerate() {
            if let Some(tab) = sk_data.get_pair(sp_i, sp_j) {
                let ang_i = sk_data.ang_shells(sp_i)?;
                let ang_j = sk_data.ang_shells(sp_j)?;
                let block_type = determine_block_type(ang_i, ang_j);
                
                let n_grid = tab.h.n_grid();
                let mut sk_h = vec![0.0f32; n_grid * n_sk_cols];
                let mut sk_s = vec![0.0f32; n_grid * n_sk_cols];
                
                for k in 0..n_grid {
                    let r = k as f64 * tab.h.dr;
                    // Evaluate shell integrals and pack into sk_h/sk_s
                }
                
                tables.push(GpuSkTable { ... });
            }
        }
    }
    Ok(tables)
}
```

Wait, but [ang_shells](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:177:4-182:5) returns `&[i32]` which can have multiple shells (e.g., `[0, 1]` for sp). For block type determination, we need the full angular momentum lists, but for the GPU kernels, we only support:
- block_type 0: both atoms have only s shell (ang = [0])
- block_type 1: one atom has only s, other has sp (ang = [0] and [0,1])
- block_type 2: both atoms have sp (ang = [0,1])

The current kernel doesn't handle arbitrary angular momentum combinations beyond these. So we need to restrict to these cases or add more block types.

Actually, looking at the kernel comment:
```
//!   0 : 1x1  (s-s, e.g. H-H)
//!   1 : 1x4  (s-sp, e.g. H-C)
//!   2 : 4x4  (sp-sp, e.g. C-C, C-O, N-O)
```

This is specifically for the sp-only approximation. For full DFTB with d orbitals, we'd need more block types. But for now, let's assume sp-only.

For sp-only:
- H has ang = [0] → n_orb = 1
- C, N, O have ang = [0, 1] → n_orb = 4

So the block type is determined by `n_orb_i` and `n_orb_j`:
- (1, 1) → block_type 0
- (1, 4) or (4, 1) → block_type 1
- (4, 4) → block_type 2

For the (4, 1) case, we need to swap i and j so that i is always the s atom. This means the pair list preprocessing needs to handle orientation.

**Direction cosine sign flip when swapping**: If we swap i↔j, the vector `vec_ij` becomes `vec_ji = -vec_ij`. The direction cosines `(l, m, n)` become `(-l, -m, -n)`. The kernel `rotate_1x4` uses `(l, m, n)` to compute `(0, m, n, l) * sp`. If we flip signs, the sp block gets negated, which is correct since it's antisymmetric.

Actually wait — the kernel `write_symmetric_1x4` has:
```c
M[base + orb_i * n_orbs + (orb_j + 1)] = -v.y;
```
This handles the sign for the transpose. If we swap i and j, the block itself would need to be transposed, but `write_symmetric_1x4` already writes both the block and its transpose with correct signs.

Hmm, let me think more carefully. For a sp-s pair (i=sp, j=s):
- The 4×1 block is at position (orb_i=sp, orb_j=s), meaning rows orb_i..orb_i+3, col orb_j
- The transpose is at (orb_j, orb_i..orb_i+3), which is 1×4 with sign flips

If we swap to (i=s, j=sp), then:
- The 4×1 block is at (orb_j=sp, orb_i=s), meaning rows orb_j..orb_j+3, col orb_i
- This is the same physical block, just transposed in storage

The kernel `write_symmetric_1x4` with `orb_i=s`, `orb_j=sp` writes:
- Direct: rows orb_j..orb_j+3, col orb_i (this is the 4×1 block at sp,s)
- Transpose: row orb_i, cols orb_j..orb_j+3 (this is the 1×4 block at s,sp with signs)

This is exactly correct! So swapping i and j for 4×1 pairs and using `write_symmetric_1x4` works perfectly, as long as we also flip the direction cosines sign.

But wait — the SK table lookup also changes. For (sp on i, s on j), the original code uses `tab_rev` (reversed species pair). If we swap i and j to make it (s on i, sp on j), we should use `tab_fwd` with the swapped species. But since the species pair is symmetric (C-H vs H-C), [sk_data.get_pair](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:150:4-155:5) handles this automatically.

So the rule is: **For block_type 1, always orient so that atom_i has s and atom_j has sp**. If the original pair is (sp, s), swap indices and flip direction cosines.

### 2. `GpuPairEntry` / `GpuPairList`

```rust
#[repr(C)]
pub struct GpuPairEntry {
    pub replica: u32,
    pub atom_i: u16,
    pub atom_j: u16,
    pub orb_i: u16,
    pub orb_j: u16,
    pub r: f32,
    pub l: f32,
    pub m: f32,
    pub n: f32,
}

pub struct GpuPairBucket {
    pub pairs: Vec<GpuPairEntry>,
    pub block_type: u8,
    pub sk_table_idx: usize,  // index into GpuSkTable list
    pub n_pairs: usize,
}
```

**Generation algorithm**:
```rust
pub fn build_pair_buckets(
    coords: &[&[[f64; 3]]],  // per replica
    species: &[&[String]],    // per replica
    ctx: &SystemContext,
    cutoff: f64,
) -> Result<Vec<GpuPairBucket>> {
    // For each replica
    for (rep, (coords_rep, species_rep)) in coords.iter().zip(species).enumerate() {
        // Build neighbor list
        let neigh = NeighborBuilder { cutoff }.build(coords_rep)?;
        
        // For each pair
        for p in &neigh.pairs {
            let si = ctx.atom_species[p.i];
            let sj = ctx.atom_species[p.j];
            let n_orb_i = ctx.atom_n_orb[p.i];
            let n_orb_j = ctx.atom_n_orb[p.j];
            
            let (atom_i, atom_j, orb_i, orb_j, l, m, n, swapped) = 
                if n_orb_i == 1 && n_orb_j == 4 {
                    // s on i, sp on j: natural orientation
                    (p.i, p.j, ctx.atom_orb_off[p.i], ctx.atom_orb_off[p.j],
                     p.vec_ij[0]/r, p.vec_ij[1]/r, p.vec_ij[2]/r, false)
                } else if n_orb_i == 4 && n_orb_j == 1 {
                    // sp on i, s on j: swap
                    (p.j, p.i, ctx.atom_orb_off[p.j], ctx.atom_orb_off[p.i],
                     -p.vec_ij[0]/r, -p.vec_ij[1]/r, -p.vec_ij[2]/r, true)
                } else {
                    // s-s or sp-sp: no swap
                    (p.i, p.j, ctx.atom_orb_off[p.i], ctx.atom_orb_off[p.j],
                     p.vec_ij[0]/r, p.vec_ij[1]/r, p.vec_ij[2]/r, false)
                };
            
            let block_type = match (n_orb_i, n_orb_j) {
                (1, 1) => 0,
                (1, 4) | (4, 1) => 1,
                (4, 4) => 2,
                _ => panic!("unsupported orbital count"),
            };
            
            let species_pair = (si.min(sj), si.max(sj)); // canonical
            let bucket_key = (block_type, species_pair);
            
            buckets.entry(bucket_key).or_default().push(GpuPairEntry {
                replica: rep as u32,
                atom_i: atom_i as u16,
                atom_j: atom_j as u16,
                orb_i,
                orb_j,
                r: p.r as f32,
                l, m, n,
            });
        }
    }
    
    // Sort within each bucket by replica for cache locality
    for bucket in buckets.values_mut() {
        bucket.sort_by_key(|p| p.replica);
    }
    
    Ok(buckets.into_values().collect())
}
```

### 3. `GpuGammaNeighborList` — CSR for `onsite_and_va`

```rust
pub struct GpuGammaNeighborList {
    pub neigh_offsets: Vec<i32>,  // [n_atoms + 1]
    pub neigh_j: Vec<i32>,        // [n_neigh_total]
    pub neigh_r: Vec<f32>,        // [n_neigh_total]
}
```

**Note**: The gamma cutoff is typically larger than the SK cutoff. Use `GammaTable.cutoffs` to determine per-species-pair cutoffs, or use [GammaTable.max_cutoff()](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/gamma.rs:97:4-100:5) as a global cutoff.

**Important**: The CSR list should include **all neighbors** (both directions) because `onsite_and_va` does a full gather: `V_A = sum_C q_C * gamma_AC`. Each atom needs to see all its neighbors.

```rust
pub fn build_gamma_neighbor_list(
    coords: &[[f64; 3]],
    species: &[u8],
    gamma_table: &GammaTable,
) -> GpuGammaNeighborList {
    let n = coords.len();
    let mut offsets = vec![0i32; n + 1];
    let mut neigh_j = Vec::new();
    let mut neigh_r = Vec::new();
    
    for i in 0..n {
        let si = species[i];
        let mut count = 0;
        for j in 0..n {
            if i == j { continue; } // or include self? gamma_full(r=0) handles it
            let sj = species[j];
            let cutoff = gamma_table.cutoff(si, sj);
            let dx = coords[j][0] - coords[i][0];
            let dy = coords[j][1] - coords[i][1];
            let dz = coords[j][2] - coords[i][2];
            let r = (dx*dx + dy*dy + dz*dz).sqrt();
            if r <= cutoff {
                neigh_j.push(j as i32);
                neigh_r.push(r as f32);
                count += 1;
            }
        }
        offsets[i + 1] = offsets[i] + count;
    }
    
    GpuGammaNeighborList { neigh_offsets: offsets, neigh_j, neigh_r }
}
```

Wait, the `onsite_and_va` kernel sums over all neighbors including self? Looking at the kernel:
```c
float v = q * hubbard_u[sp];  // self term
...
v += qb * g;  // neighbor terms
```

The self term is handled separately (`q * hubbard_u[sp]`). The neighbor list doesn't need to include self-pairs. Good.

But actually, [gamma_full(0, u1, u2) = 0.5*(u1+u2)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/gamma.rs:14:0-35:1). And `q_A * U_A` is the self term. If we include self in the neighbor list, [gamma_full(0) = 0.5*(u1+u2)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/gamma.rs:14:0-35:1) and `q_A * 0.5*(u1+u2)` would be added, but the self term should be `q_A * U_A`. So we should NOT include self in the neighbor list. The kernel handles the self term explicitly.

**Wait, actually**: For i == j, the gamma function returns `0.5*(u1+u2)`. But in the SCC expression, the self term is `q_A * U_A`, which is `q_A * gamma(0, U_A, U_A) = q_A * 0.5*(U_A + U_A) = q_A * U_A`. So including self in the sum with gamma(0) gives the correct self term!

But the kernel does:
```c
float v = q * hubbard_u[sp];  // q_A * U_A
...
for neighbors:
    v += qb * g;  // q_B * gamma_AB
```

If we include self in the neighbor list:
```c
v += q_A * gamma(0, U_A, U_A) = q_A * 0.5 * (U_A + U_A) = q_A * U_A
```

Then `v = q_A * U_A + q_A * U_A + sum_{B≠A} q_B * gamma_AB = 2*q_A*U_A + ...` which is wrong!

So the neighbor list should NOT include self. The kernel handles the self term explicitly with `q * hubbard_u[sp]`. Good.

### 4. `GpuSystemBatch` — Container for All Arrays

```rust
pub struct GpuSystemBatch {
    // Per-system static data
    pub atom_species: Vec<i32>,       // [n_atoms]
    pub orb_offsets: Vec<i32>,        // [n_atoms]
    pub onsite_es_ep: Vec<f32>,       // [n_species] float2 packed (e_s, e_p)
    pub hubbard_u: Vec<f32>,          // [n_species]
    
    // Per-replica dynamic data
    pub coords: Vec<Vec<[f64; 3]>>,   // [n_replicas][n_atoms]
    pub charges: Vec<Vec<f32>>,        // [n_replicas][n_atoms] (SCC)
    
    // Computed structures
    pub sk_tables: Vec<GpuSkTable>,
    pub pair_buckets: Vec<GpuPairBucket>,
    pub gamma_neigh: GpuGammaNeighborList,  // same for all replicas (if uniform geometry)
    
    // Dimensions
    pub n_replicas: usize,
    pub n_atoms: usize,
    pub n_orbs: usize,
    pub n_species: usize,
}
```

**Note on gamma neighbor list for multiple replicas**: If all replicas have the same geometry (e.g., same positions), one gamma neighbor list works for all. If geometries differ, we need per-replica neighbor lists. But `onsite_and_va` currently takes a single neighbor list for all replicas.

For now, assume uniform geometry across replicas (same positions). If geometries differ, we'll need per-replica neighbor lists and a kernel redesign.

### 5. `pack_onsite_es_ep`

```rust
pub fn pack_onsite_es_ep(ctx: &SystemContext) -> Vec<f32> {
    // Pack as float2: [e_s, e_p] for each species
    let mut out = vec![0.0f32; ctx.n_species * 2];
    for (si, onsite) in ctx.species_onsite.iter().enumerate() {
        out[2*si] = onsite.e_s as f32;
        out[2*si + 1] = onsite.e_p as f32;
    }
    out
}
```

### 6. `pack_charges`

```rust
pub fn pack_charges(charges: &[&[f64]], n_atoms: usize, n_replicas: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_replicas * n_atoms];
    for rep in 0..n_replicas {
        for a in 0..n_atoms {
            out[rep * n_atoms + a] = charges[rep][a] as f32;
        }
    }
    out
}
```

## Algorithm for `pack_sk_tables` (Detailed)

```rust
pub fn pack_sk_tables(
    sk_data: &SkData,
    ctx: &SystemContext,
) -> Result<Vec<GpuSkTable>> {
    let mut tables = Vec::new();
    let n_species = ctx.n_species;
    
    for si in 0..n_species {
        for sj in 0..n_species {
            let sp_i = /* map si back to species name */;
            let sp_j = /* map sj back to species name */;
            
            let tab = match sk_data.get_pair(&sp_i, &sp_j) {
                Some(t) => t,
                None => continue,
            };
            
            let ang_i = sk_data.ang_shells(&sp_i)?;
            let ang_j = sk_data.ang_shells(&sp_j)?;
            
            let block_type = determine_block_type(ang_i, ang_j);
            let n_sk_cols = match block_type {
                0 => 1,  // ss
                1 => 2,  // ss, sp
                2 => 4,  // ss, sp, pp_sig, pp_pi
                _ => unreachable!(),
            };
            
            let n_grid = tab.h.n_grid();
            let mut sk_h = vec![0.0f32; n_grid * n_sk_cols];
            let mut sk_s = vec![0.0f32; n_grid * n_sk_cols];
            
            for k in 0..n_grid {
                let r = k as f64 * tab.h.dr;
                
                match block_type {
                    0 => {
                        // s-s: only ss
                        let mut h = [0.0f64; 1];
                        let mut s = [0.0f64; 1];
                        tab.eval_shell_integrals_into(0, 0, r, &mut h, &mut s)?;
                        sk_h[k] = h[0] as f32;
                        sk_s[k] = s[0] as f32;
                    }
                    1 => {
                        // s-sp or sp-s: ss, sp
                        let mut h = [0.0f64; 1];
                        let mut s = [0.0f64; 1];
                        // For ss shell (ang1=0, ang2=0 or vice versa)
                        tab.eval_shell_integrals_into(0, 0, r, &mut h, &mut s)?;
                        sk_h[2*k] = h[0] as f32;
                        sk_s[2*k] = s[0] as f32;
                        
                        // For sp shell
                        let mut h2 = [0.0f64; 1];
                        let mut s2 = [0.0f64; 1];
                        tab.eval_shell_integrals_into(0, 1, r, &mut h2, &mut s2)?;
                        sk_h[2*k + 1] = h2[0] as f32;
                        sk_s[2*k + 1] = s2[0] as f32;
                    }
                    2 => {
                        // sp-sp: ss, sp, pp_sig, pp_pi
                        let mut h = [0.0f64; 2];
                        let mut s = [0.0f64; 2];
                        
                        // ss
                        tab.eval_shell_integrals_into(0, 0, r, &mut h[..1], &mut s[..1])?;
                        sk_h[4*k] = h[0] as f32;
                        sk_s[4*k] = s[0] as f32;
                        
                        // sp
                        tab.eval_shell_integrals_into(0, 1, r, &mut h[..1], &mut s[..1])?;
                        sk_h[4*k + 1] = h[0] as f32;
                        sk_s[4*k + 1] = s[0] as f32;
                        
                        // pp: returns 2 values (sigma, pi)
                        tab.eval_shell_integrals_into(1, 1, r, &mut h[..2], &mut s[..2])?;
                        sk_h[4*k + 2] = h[0] as f32;  // pp_sig
                        sk_s[4*k + 2] = s[0] as f32;
                        sk_h[4*k + 3] = h[1] as f32;  // pp_pi
                        sk_s[4*k + 3] = s[1] as f32;
                    }
                    _ => unreachable!(),
                }
            }
            
            tables.push(GpuSkTable {
                sk_h,
                sk_s,
                n_grid,
                dr: tab.h.dr as f32,
                n_sk_cols,
                block_type,
                species_i: si as u8,
                species_j: sj as u8,
            });
        }
    }
    
    Ok(tables)
}
```

Wait, there's an issue with [eval_shell_integrals_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:97:4-139:5). Looking at the code:

```rust
pub fn eval_shell_integrals_into(
    &self,
    ang1: i32,
    ang2: i32,
    r: f64,
    out_h: &mut [f64],
    out_s: &mut [f64],
) -> Result<<usize> {
```

This internally calls [self.h.eval_into(r, &mut h_all)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:41:4-46:5) which evaluates ALL integrals at distance `r` (10 or 20 values), then extracts the shell-specific ones. This is inefficient for precomputing on a grid — we re-evaluate the full interpolation for every shell combination at every grid point.

For the GPU preparation, we should instead:
1. For each grid point, evaluate the full `h_all` and `s_all` once
2. Extract all needed shell integrals from that evaluation
3. Pack them

This avoids redundant interpolation. But the current [EqGridTable](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:16:0-19:1) stores raw values, so [eval_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:41:4-46:5) does Neville interpolation. For the GPU, we want to precompute on the existing uniform grid and just read the raw values.

Actually, looking more carefully at [EqGridTable](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:16:0-19:1):
```rust
pub struct EqGridTable {
    pub dr: f64,
    pub values: Vec<Vec<f64>>, // [n_grid][n_integ]
}
```

The `values` are already on a uniform grid! The [eval_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:41:4-46:5) function just does interpolation to get values at arbitrary `r`. For grid point `k`, the raw values are `values[k]`.

So for packing, we should read the raw values directly from `values[k]`, not call [eval_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:41:4-46:5). This avoids interpolation entirely since we're already on the grid.

```rust
for k in 0..n_grid {
    let h_all = &tab.h.values[k];
    let s_all = &tab.s.values[k];
    
    // Extract shell integrals using the same logic as eval_shell_integrals_into
    // but without calling eval_into
}
```

But [eval_shell_integrals_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:97:4-139:5) also handles the extended vs old format and [sk_map](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:381:0-389:1). We should factor out the extraction logic.

Actually, the simplest approach: create a new function that extracts shell integrals from raw `h_all` / `s_all` arrays (already on grid):

```rust
fn extract_shell_integrals_from_raw(
    h_all: &[f64], 
    s_all: &[f64], 
    ang1: i32, 
    ang2: i32
) -> (Vec<f64>, Vec<f64>) {
    // Same logic as eval_shell_integrals_into but without eval_into call
}
```

Then for each grid point:
```rust
let h_all = &tab.h.values[k];
let s_all = &tab.s.values[k];
let (h_shell, s_shell) = extract_shell_integrals_from_raw(h_all, s_all, ang1, ang2);
```

Wait, but [eval_shell_integrals_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:97:4-139:5) handles the `ang1 <= ang2` swap and uses `tab_fwd` vs `tab_rev`. For the GPU, we need to be careful about which table to use.

Actually, for a given species pair (si, sj), we have [tab = sk_data.get_pair(sp_i, sp_j)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:150:4-155:5). This may be the forward or reverse table. The [eval_shell_integrals_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:97:4-139:5) handles `ang1 <= ang2` internally. But for the GPU table, we're packing for a specific species pair (si, sj), so we should use the shell integrals as if atom i has ang_i and atom j has ang_j.

For sp-sp (si, sj), ang1=1, ang2=1. [eval_shell_integrals_into(1, 1, r, ...)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:97:4-139:5) returns 2 values (pp_sig, pp_pi). This is correct.

For s-sp (si=s, sj=sp), ang1=0, ang2=1. [eval_shell_integrals_into(0, 1, r, ...)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:97:4-139:5) returns 1 value (sp). And [eval_shell_integrals_into(0, 0, r, ...)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:97:4-139:5) returns 1 value (ss). This is correct for block_type 1.

For sp-s (si=sp, sj=s), if we swap to s-sp orientation, we'd use the same tables but with swapped atoms. The GPU pair list handles this at the pair level.

So the packing function should just evaluate all shell combinations for each species pair and pack them.

## Full Implementation Plan

### New file: `rust_dftb/src/methods/dftb/gpu_prep.rs`

**Data structures**:
```rust
#[repr(C)]
#[derive(Debug, Clone)]
pub struct GpuPairEntry { ... }

pub struct GpuPairBucket { ... }

pub struct GpuSkTable { ... }

pub struct GpuGammaNeighborList { ... }

pub struct GpuSystemBatch { ... }
```

**Functions**:
1. `determine_block_type(ang_i: &[i32], ang_j: &[i32]) -> u8`
2. `pack_sk_tables(sk_data: &SkData, ctx: &SystemContext) -> Vec<GpuSkTable>`
3. `build_pair_buckets(coords: &[&[[f64; 3]]], ctx: &SystemContext, cutoff: f64) -> Vec<GpuPairBucket>`
4. `build_gamma_neighbor_list(coords: &[[f64; 3]], species: &[u8], gamma_table: &GammaTable) -> GpuGammaNeighborList`
5. `pack_onsite_es_ep(ctx: &SystemContext) -> Vec<f32>`
6. `pack_charges(charges: &[&[f64]], n_atoms: usize) -> Vec<f32>`
7. `assemble_gpu_batch(systems: &[SystemConfig], sk_data: &SkData) -> Result<GpuSystemBatch>`

### Modifications to existing files:

**[sk_data.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:0:0-0:0)**:
- Add `extract_shell_integrals_from_raw` (or make [eval_shell_integrals_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:97:4-139:5) public if already)
- The function already exists as [eval_shell_integrals_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:97:4-139:5)

Actually, [eval_shell_integrals_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:97:4-139:5) is already public. But it evaluates [h.eval_into(r)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:41:4-46:5) which does Neville interpolation. For precomputing on the grid, we should just read raw values. We could add:

```rust
impl SkTableSp {
    pub fn extract_shell_integrals_at_grid_point(
        &self,
        ang1: i32,
        ang2: i32,
        k: usize,  // grid point index
        out_h: &mut [f64],
        out_s: &mut [f64],
    ) -> Result<<usize> {
        let h_all = &self.h.values[k];
        let s_all = &self.s.values[k];
        // Extract using same logic as eval_shell_integrals_into but without interpolation
    }
}
```

But wait — [eval_shell_integrals_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:97:4-139:5) does:
```rust
self.h.eval_into(r, &mut h_all)?;
self.s.eval_into(r, &mut s_all)?;
```

Where [eval_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:41:4-46:5) does Neville interpolation. For the GPU, we want B-spline interpolation in the kernel, but the tabulated data should be the raw grid values. So yes, we should read `self.h.values[k]` directly.

However, there might be a subtlety: the current CPU code uses Neville interpolation with a specific stencil, while the GPU uses cubic B-spline. The B-spline requires the data to be the actual function values at grid points (which is what `values[k]` stores). So reading raw values is correct.

But there's another issue: [EqGridTable](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:16:0-19:1) has `n_grid_raw.saturating_sub(1)` points. The first point might be at r=0 or r=dr. We need to be careful.

Let me check [read_skf_all](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:289:0-349:1):
```rust
let n_grid = n_grid_raw.saturating_sub(1);
```

And the loop goes [for _ in 0..n_grid](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:22:4-24:5), reading grid points. So `values.len() == n_grid`.

The kernel's `cubic_interp_params` clamps `i` to `[1, n_grid-3]` and returns `base = i-1`. This requires `n_grid >= 4`. For small tables, we need to ensure this.

Also, `cubic_interp_params` checks `r >= (n_grid - 1) * dr`. The max valid r is `(n_grid - 1) * dr`. The pair cutoff from [tab.cutoff()](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:93:4-95:5) is `(n_grid - 1) * dr + DIST_FUDGE` where `DIST_FUDGE = 1.0`. So some pairs may have r slightly beyond the grid. The kernel returns 0 for out-of-range.

###

I'll provide a concise specification for the Rust host-side preparation module.


## Rust Host-Side Preparation (`gpu_prep.rs`)

### Design Constraint
**All replicas must have identical `n_atoms` and `n_orbs`**. Kernels use flat indexing (`replica * n_orbs * n_orbs`). For heterogeneous systems, pad to max and use a mask, or launch per-replica.


### Data Structures

```rust
#[repr(C)]
pub struct GpuPairEntry {
    pub replica: u32,
    pub atom_i: u16, pub atom_j: u16,
    pub orb_i: u16,  pub orb_j: u16,
    pub r: f32,
    pub l: f32, pub m: f32, pub n: f32,
}

pub struct GpuPairBucket {
    pub pairs: Vec<GpuPairEntry>,
    pub block_type: u8,       // 0, 1, 2
    pub sk_table_idx: usize,  // into GpuSkTable list
    pub n_pairs: usize,
}

pub struct GpuSkTable {
    pub sk_h: Vec<f32>,       // [n_grid * n_sk_cols] interleaved
    pub sk_s: Vec<f32>,
    pub n_grid: usize,
    pub dr: f32,
    pub n_sk_cols: usize,
    pub block_type: u8,
    pub species_i: u8,
    pub species_j: u8,
}

pub struct GpuGammaNeigh {
    pub offsets: Vec<i32>,    // [n_atoms + 1]
    pub neigh_j: Vec<i32>,    // [n_neigh]
    pub neigh_r: Vec<f32>,    // [n_neigh]
}

pub struct GpuBatch {
    pub atom_species: Vec<i32>,
    pub orb_offsets: Vec<i32>,
    pub onsite_es_ep: Vec<f32>,   // [n_species * 2] float2 packed
    pub hubbard_u: Vec<f32>,
    pub charges: Vec<f32>,         // [n_replicas * n_atoms]
    pub sk_tables: Vec<GpuSkTable>,
    pub pair_buckets: Vec<GpuPairBucket>,
    pub gamma_neigh: GpuGammaNeigh,
    pub n_replicas: usize,
    pub n_atoms: usize,
    pub n_orbs: usize,
    pub n_species: usize,
}
```

### Key Functions to Implement

#### 1. `pack_sk_tables(sk_data, ctx) -> Vec<GpuSkTable>`
For each species pair (si, sj):
- Determine `block_type` from angular shells: `(1,1)=0`, `(1,4)/(4,1)=1`, `(4,4)=2`
- Read raw grid values directly from `tab.h.values[k]` / `tab.s.values[k]` (avoid [eval_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/interpolation.rs:41:4-46:5) interpolation)
- Extract shell integrals using same [sk_map](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:381:0-389:1) logic as [eval_shell_integrals_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:97:4-139:5)
- Pack interleaved: `[ss, sp, pp_sig, pp_pi]` per grid point for block_type 2

**Critical**: Use raw grid values, not interpolated. The GPU kernel does B-spline interpolation.

#### 2. `build_pair_buckets(coords_per_replica, ctx, cutoff) -> Vec<GpuPairBucket>`
For each replica:
- Build neighbor list with [NeighborBuilder](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/core/neighbor.rs:17:0-19:1)
- For each pair, compute direction cosines `l,m,n = vec_ij / r`
- **Orient block_type 1 pairs**: atom_i must be the s-orbital atom. If swapped, flip `(l,m,n)` signs and swap indices
- Categorize by `(block_type, species_pair)` → bucket
- Sort each bucket by `replica` for cache locality

#### 3. `build_gamma_neigh(coords, species, gamma_table) -> GpuGammaNeigh`
- Use per-species-pair gamma cutoffs from `GammaTable.cutoffs`
- Build CSR neighbor list (exclude self — kernel adds self-term explicitly)
- Include both directions: if A sees B, B sees A

#### 4. `pack_onsite_es_ep(ctx) -> Vec<f32>`
Pack [AtomicParamsSp](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:78:0-82:1) as `(e_s, e_p)` float2 array per species.

#### 5. `pack_charges(charges_per_replica, n_atoms) -> Vec<f32>`
Flatten to `[n_replicas][n_atoms]`.

### What's Missing / TODOs for Host Code

| Item | Priority | Note |
|------|----------|------|
| `GpuBatch::from_systems()` | High | Orchestrates all packing functions |
| SK raw-value extraction | High | Factor out from [eval_shell_integrals_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/sk_data.rs:97:4-139:5) to avoid Neville interpolation |
| Padding for heterogeneous replicas | Medium | If systems have different `n_atoms`, pad and add `active_atoms` mask |
| Validation (all replicas same n_orbs) | Low | Assert or pad |

### File to Create

`rust_dftb/src/methods/dftb/gpu_prep.rs` — new module with all data structures and packing functions.

Shall I implement this file now?