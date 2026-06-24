# USER

OK now with this understaiding can you see some inefficiencies or non-elegant (overcomplicated code) in our rust implementation? Do you see some opportunities how to make it more efficient and streamlined? For example if we have this specialized version of only sp orbitals maybe we can exploint that? Do not code yet, we must make carefull analysis first.

---

# Kimi-2.6

## 🔴 Hot-Path Allocations (Biggest Issue)

The single biggest performance problem is **dynamic allocation inside the innermost loop**.

### 1. [rotate_diatomic_block](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:91:4-141:5) allocates full `DMatrix` for every neighbor pair

In `@/home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:104-105`:

```rust
let mut h_blk = DMatrix::<f64>::zeros(n_orb2, n_orb1);
let mut s_blk = DMatrix::<f64>::zeros(n_orb2, n_orb1);
```

For a 200-atom organic molecule with ~1000 neighbor pairs, this is **~1000 heap allocations per H build**, just for the diatomic blocks. Each allocation is small (4×4=16 f64s for sp), but the allocator overhead dominates.

**Worse:** these are deallocated immediately after [fill_pairs](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:83:4-117:5) copies the data into the global dense matrix.

### 2. [eval_shell_integrals](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:164:4-198:5) allocates `Vec<f64>` for every shell pair

In `@/home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:187-198`:

```rust
let h_all = tab.h.eval(r)?;   // allocates Vec<f64>
let s_all = tab.s.eval(r)?;   // allocates Vec<f64>
...
let h_shell = extract_shell_integrals_...(...); // allocates Vec<f64>
let s_shell = extract_shell_integrals_...(...); // allocates Vec<f64>
```

For an sp-sp pair, there are 4 shell pairs, so **8 `Vec<f64>` allocations per pair**. For 1000 pairs = **8000 allocations**.

### 3. [rotate_shell_pair_single](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:44:4-54:5) allocates tiny `DMatrix` for each sub-block

[rotate_ss](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:56:4-59:5) returns `DMatrix::from_row_slice(1, 1, ...)` — a heap-allocated 1×1 matrix. [rotate_sp](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:61:4-66:5) allocates 3×1, [rotate_pp](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:68:4-89:5) allocates 3×3.

**Impact:** For sp-sp, each pair triggers allocations for the (s,s), (s,p), (p,s), (p,p) sub-blocks, then the full 4×4 block, then the copy into the global matrix. That is **~7 allocations per pair** for rotation alone.

### 4. [extract_shell_integrals_old](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:343:0-366:1) rebuilds lookup table on every call

In `@/home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:348-367`:

```rust
let mut new_to_old = [0usize; 21];
for (old_idx, &new_col) in iSKInterOld.iter().enumerate() {
    new_to_old[new_col] = old_idx;
}
```

This 21-element table is recomputed **for every shell pair**. It should be a `const` or `static` lookup table.

## 🟡 HashMap Lookups in the Hot Loop

In [fill_pairs](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:83:4-117:5) (`hamiltonian.rs:92-115`), for every neighbor pair, the code performs:

1. [Rotation::rotate_diatomic_block(...)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:91:4-141:5) which internally calls:
   - [sk.ang_shells(sp1)?](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:157:4-162:5) → HashMap lookup
   - [sk.ang_shells(sp2)?](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:157:4-162:5) → HashMap lookup
   - [sk.eval_shell_integrals(...)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:164:4-198:5) → HashMap [get_pair](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:128:4-132:5) lookup (tries 2 keys!)
2. Back in [fill_pairs](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:83:4-117:5):
   - [self.sk.n_orb_species(sp1)?](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:150:4-155:5) → HashMap lookup
   - [self.sk.n_orb_species(sp2)?](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:150:4-155:5) → HashMap lookup

That is **5+ HashMap lookups per pair**, all using `String` keys. For 1000 pairs, the hashing cost is significant.

**The root cause:** [HamiltonianBuilder::build_non_scc](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:22:4-58:5) passes `species: &[String]`, but the [SkData](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:120:0-125:1) stores everything keyed by `String`. There is no precomputed per-atom lookup table.

## 🟡 `DMatrix` for Tiny Fixed-Size Blocks

`nalgebra::DMatrix` is a dynamically-sized matrix with runtime dimensions. For the diatomic blocks, the sizes are known at compile time for a given basis set:

| Basis | Block sizes |
|---|---|
| s-s | 1×1 |
| s-p | 3×1 |
| p-p | 3×3 |
| sp-sp | 4×4 |

Using `DMatrix` for these incurs:
- Runtime dimension checks on every index `[(i, j)]`
- No SIMD vectorization opportunities (compiler cannot unroll loops with unknown bounds)
- Heap allocation (as noted above)

`nalgebra::SMatrix<f64, N, M>` (static matrix) or even plain `[f64; N]` arrays would be **orders of magnitude faster** for these sizes.

## 🟡 Overcomplicated [poly_inter_uniform_vec](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/interpolation.rs:45:0-109:1)

The interpolation routine (`@/home/prokophapala/git/dftbplus/rust_dftb/src/interpolation.rs:46-110`) is a direct port of the Fortran `polyInterUniform` routine. It:

1. Accepts `yp: &[Vec<f64>]` — a **jagged** 2D structure (Vec of Vecs)
2. Clones the entire `yp` into `cc` and `dd` at the start
3. Allocates `delta`, `yy`, `dyy` on every call

For SK interpolation, [n_integ](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/interpolation.rs:20:4-22:5) is only 10 or 20, and `n` is 8 (interpolation points). The data is tiny but the function signature forces heap indirections.

**Streamlining opportunity:** For SK interpolation specifically, write a specialized `eval_sk_table(r)` that works on the transposed data (`[n_integ][n_grid]` → actually stored as `[n_grid][n_integ]` in your [EqGridTable](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/interpolation.rs:10:0-13:1)) and uses stack-allocated `[f64; 8]` work arrays. Do not clone the whole table.

## 🟡 Runtime Branching on SK Format

In [eval_shell_integrals](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:164:4-198:5) (`sk_data.rs:187-192`):

```rust
let is_extended = h_all.len() == 20;
let h_shell = if is_extended { ... } else { ... };
```

This branch is **per shell pair, per neighbor pair**. The format is a property of the [SkTableSp](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:106:0-111:1), not the evaluation. It should be stored as an enum in [SkTableSp](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:106:0-111:1) so the dispatch happens once at load time, not millions of times at runtime.

## 🟡 [get_pair](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:128:4-132:5) Tries Two Keys

In `@/home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:129-133`:

```rust
pub fn get_pair(&self, a: &str, b: &str) -> Option<&SkTableSp> {
    self.pairs
        .get(&(a.to_string(), b.to_string()))
        .or_else(|| self.pairs.get(&(b.to_string(), a.to_string())))
}
```

This constructs **two temporary `String` tuples** on every call, hashing both. For a symmetric pair store, you should normalize the key at insertion time (always store with `a <= b` lexicographically) and normalize at lookup time. Then you only do one lookup.

## 🟢 Architecture-Level Simplifications

### 6. No separation between "load time" and "run time" data

[SkData](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:120:0-125:1) stores raw parsed data (`HashMap<String, AtomicParamsSp>`, `HashMap<(String, String), SkTableSp>`). At runtime, you only need:

- Per-atom: species index, number of orbitals, orbital offset
- Per-species-pair: a direct pointer to the SK table

**Suggestion:** Add a `RuntimeSystem` struct that is built once from `SkData + species list + coords`:

```rust
struct RuntimeSystem {
    atom_species: Vec<u8>,          // species index per atom (no more String lookups)
    atom_n_orb: Vec<u8>,            // n_orb per atom
    atom_orb_offset: Vec<u16>,      // cumulative offset
    pair_tables: Vec<&SkTableSp>,   // indexed by (species_i, species_j) pair index
    ...
}
```

Then the hot loop becomes pure integer indexing into flat arrays.

## 🟢 SP-Only Specialization Opportunities

Since you asked specifically about exploiting an sp-only basis: yes, there are **massive** simplifications possible.

### Current sp-sp path (what the code does now)

For every neighbor pair:
1. [ang_shells("C")](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:157:4-162:5) → `[0, 1]`
2. Shell loop: (s,s), (s,p), (p,s), (p,p)
3. For each: [eval_shell_integrals](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:164:4-198:5) → interpolation → `Vec<f64>` allocation → `extract_shell_integrals_*` → [rotate_shell_pair](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:31:4-42:5) → `DMatrix` allocation
4. [rotate_diatomic_block](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:91:4-141:5) assembles 4 sub-blocks into a 4×4 `DMatrix`
5. [fill_pairs](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:83:4-117:5) copies the 4×4 `DMatrix` into the global dense matrix

That is ~10 allocations and ~5 HashMap lookups per pair for a case that is **entirely predictable**.

### Streamlined sp-sp path

Precompute at load time:
- A [SkTableSp](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:106:0-111:1) is always sp-sp (you know this from the species angular momenta)
- The 4 SK integrals are always at fixed positions in the interpolated array: `ssσ`, `spσ`, `ppσ`, `ppπ`

At runtime per pair:
1. Interpolate **once** to get `[ssσ, spσ, ppσ, ppπ]` for H and S (2 interpolations, not 8)
2. Compute the 4×4 block directly with closed-form formulas (no shell loops, no DMatrix, no `ang1 <= ang2` checks)
3. Write directly into the global dense matrix

The entire [rotate_diatomic_block](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:91:4-141:5) for sp-sp collapses to ~40 lines of arithmetic on `[[f64; 4]; 4]` stack arrays.

### The closed-form sp-sp block

For atom `i` (cols) and atom `j` (rows), with direction cosines `(l, m, n)`:

```
        |  s_i   py_i   pz_i   px_i  |
    ----|----------------------------|
 s_j    |  ssσ   spσ·m spσ·n spσ·l  |
 py_j   |        pp block (3×3)     |
 pz_j   |        (symmetric)          |
 px_j   |                             |
```

The 3×3 p-p block from [rotate_pp](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:68:4-89:5) is already a closed-form expression in `l, m, n, ppσ, ppπ`. You can write it as a single function:

```rust
fn rotate_sp_sp_block(ss: f64, sp: f64, pp_s: f64, pp_p: f64, dc: DirectionCosines) -> [[f64; 4]; 4]
```

This eliminates:
- All shell iteration
- All sub-block allocation
- All `ang1 <= ang2` branching
- The `extract_shell_integrals` indirection
- The `DMatrix` overhead

## 📋 Prioritized Action List

| Priority | Issue | Effort | Impact |
|---|---|---|---|
| **P0** | Replace `DMatrix` diatomic blocks with stack arrays (`SMatrix` or `[[f64; N]; M]`) | Low | **Huge** (eliminates 1000s of allocations) |
| **P0** | Precompute per-atom orbital info into flat `Vec`s (no HashMap in hot loop) | Low | **Huge** |
| **P1** | Make [extract_shell_integrals_old](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:343:0-366:1) use a `const` lookup table | Trivial | Medium |
| **P1** | Store `is_extended` as an enum in [SkTableSp](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:106:0-111:1), remove runtime branch | Low | Medium |
| **P1** | Normalize pair keys in `HashMap` to avoid double lookup + `to_string()` | Low | Medium |
| **P2** | Write an sp-sp specialized `rotate_sp_sp_block` | Medium | Large (for organic chemistry) |
| **P2** | Specialize interpolation to avoid `Vec<Vec<f64>>` cloning | Medium | Medium |
| **P3** | Add `RuntimeSystem` layer to separate load-time and run-time data | Medium | Large (cleaner architecture) |

Do you want me to proceed with designing the `RuntimeSystem` precomputation layer, or would you prefer to start with the lower-hanging fruit (stack-allocated blocks + const lookup tables)?

---

# USER

This is great analysis! I fully agree with you! We must follow High-perforamnce numerical data oriented desing.
1) minimize number of allocations / deallocation everything should be preallocated if possible, and teporary variable shouild be small lokal stack fixed size
2) we should use coninuous array and data oriented desing. We can use dict/hash/talbes and string if convenient especially in the IO part, but in performance core we should use numerical indexes where possible. It also make sense to precompute (bake) redindexing maps from the hashmaps beforehand, to potimize access later. Do not use strings and string hashing in performance core
3) yes, it is importaint to minimize on-the-fly work, what can be precalcualted and then reused many times in hot loop, that should be precalculated
4) I like your analysis of simplification by fixed size 4x4 sp blocks, we should keep it in mind, but this optimization will come lass fater we resolve the more urgent issues mentioned above. 

Now plese make definite and clear plan what should we change and how. 

---

# Kimi-2.6

Here is the concrete implementation plan. I have structured it into phases, with exact struct signatures, file targets, and before/after code patterns.

# Optimization Plan: Rust DFTB High-Performance Refactor

## Guiding Principles
1. **Zero allocations in the neighbor-pair hot loop**
2. **Zero string hashing in the neighbor-pair hot loop**
3. **Precompute all maps at system-setup time**
4. **Use fixed-size stack buffers for temporaries; preallocated slices for variable-size temporaries**
5. **Keep `DMatrix` only for the final global H/S containers**

---

## Phase 1: Precomputation Layer (`SystemContext`)

**Goal:** Convert all runtime string/HashMap lookups into flat integer-indexed arrays before entering the hot loop.

### 1.1 Normalize SK pair storage

**File:** [rust_dftb/src/sk_data.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:0:0-0:0)

**Current problem:** [get_pair](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:138:4-142:5) tries two string keys and hashes twice.

```rust
// CURRENT (sk_data.rs:129-133)
pub fn get_pair(&self, a: &str, b: &str) -> Option<&SkTableSp> {
    self.pairs
        .get(&(a.to_string(), b.to_string()))
        .or_else(|| self.pairs.get(&(b.to_string(), a.to_string())))
}
```

**Change:** Normalize the key at **insertion** time (in [load_sk_folder](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:213:4-235:5)). Always store with `a <= b` lexicographically. At lookup, normalize the query the same way.

```rust
// NEW: single lookup, no allocation
fn normalize_pair_key<'a>(a: &'a str, b: &'a str) -> (&'a str, &'a str) {
    if a <= b { (a, b) } else { (b, a) }
}
```

**Impact:** Cuts SK table lookups from 2 hashes + 2 `String` allocs to 1 hash + 0 allocs.

### 1.2 Add `SkFormat` enum to [SkTableSp](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:116:0-121:1)

**File:** [rust_dftb/src/sk_data.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:0:0-0:0)

**Current problem:** Runtime `h_all.len() == 20` branch inside [eval_shell_integrals](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:174:4-211:5).

```rust
// CURRENT (sk_data.rs:187-192)
let is_extended = h_all.len() == 20;
let h_shell = if is_extended { ... } else { ... };
```

**Change:** Store format at load time.

```rust
#[derive(Debug, Clone, Copy)]
pub enum SkFormat { Old, Extended }

pub struct SkTableSp {
    pub sp1: String,
    pub sp2: String,
    pub h: EqGridTable,
    pub s: EqGridTable,
    pub format: SkFormat, // <-- NEW
}
```

**Impact:** Removes per-shell-pair branch. Dispatch becomes `match tab.format` (often optimized to a jump table).

### 1.3 Build `SystemContext`

**File:** [rust_dftb/src/hamiltonian.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:0:0-0:0) (new module or top of file)

**New struct:**

```rust
/// Precomputed numerical-indexed system data.
/// Built once per [build_non_scc](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:22:4-69:5) call.
pub struct SystemContext<'a> {
    pub n_atoms: usize,
    pub n_species: usize,
    pub n_orbs: usize,

    // Per-atom data (index: atom_idx)
    pub atom_species: Vec<u8>,      // species index per atom
    pub atom_n_orb: Vec<u8>,        // n_orb per atom
    pub atom_orb_off: Vec<u16>,     // cumulative orbital offset

    // Per-species data (index: species_idx)
    pub species_n_orb: Vec<u8>,
    pub species_ang_shells: Vec<&'a [i32]>,
    pub species_onsite: Vec<&'a AtomicParamsSp>,

    // Pair lookup: flat 2D table [si * n_species + sj] -> Option<<usize>
    pub pair_lut: Vec<Option<<usize>>,
    pub pair_tables: Vec<&'a SkTableSp>,
}
```

**Build method:**

```rust
impl<'a> SystemContext<'a> {
    pub fn from_sk_data(
        sk: &'a SkData,
        species: &[String],
    ) -> Result<Self> { ... }
}
```

**Inside [build_non_scc](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:22:4-69:5):**
```rust
let ctx = SystemContext::from_sk_data(&self.sk, species)?;
// All HashMap/String lookups happen HERE, once.
```

**Impact:** Hot loop accesses only `Vec<u8>`/`Vec<u16>` by integer index. No HashMap, no `String`, no hashing.

## Phase 2: Workspace & Zero-Allocation Hot Loop

**Goal:** The [fill_pairs](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:94:4-133:5) loop must not allocate. All temporaries are either stack arrays or slices into preallocated workspace buffers.

### 2.1 Add `eval_into` to [EqGridTable](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/interpolation.rs:10:0-13:1)

**File:** [rust_dftb/src/interpolation.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/interpolation.rs:0:0-0:0)

**Current problem:** [eval(r)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/interpolation.rs:28:4-30:5) returns `Vec<f64>` (allocates every call).

```rust
// CURRENT (interpolation.rs:29-31)
pub fn eval(&self, r: f64) -> Result<Vec<f64>> {
    eval_eqgrid_new(self, r)
}
```

**Change:** Add an `eval_into` that writes into a caller-provided buffer.

```rust
pub fn eval_into(&self, r: f64, out: &mut [f64]) -> Result<()> {
    eval_eqgrid_new_into(self, r, out)
}
```

**Impact:** Eliminates interpolation result allocations.

### 2.2 Add `HWorkspace`

**File:** [rust_dftb/src/hamiltonian.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:0:0-0:0)

**New struct:**

```rust
/// Reusable scratch buffers for Hamiltonian assembly.
/// Created once per [build_non_scc](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:22:4-69:5) call.
pub struct HWorkspace {
    /// Max 20 integrals from interpolation (extended format)
    sk_buf_h: [f64; 20],
    sk_buf_s: [f64; 20],
    /// Max integrals per shell pair: mm=0..l_min, l_min<=3 => 4 max
    shell_buf_h: [f64; 4],
    shell_buf_s: [f64; 4],
    /// Reusable diatomic block (max 9x9=81 for spd; 16x16=256 for spdf)
    /// Preallocated Vec sized to max block in this system.
    block_h: Vec<f64>,
    block_s: Vec<f64>,
}

impl HWorkspace {
    pub fn new(max_block_size: usize) -> Self {
        Self {
            sk_buf_h: [0.0; 20],
            sk_buf_s: [0.0; 20],
            shell_buf_h: [0.0; 4],
            shell_buf_s: [0.0; 4],
            block_h: vec![0.0; max_block_size],
            block_s: vec![0.0; max_block_size],
        }
    }
}
```

### 2.3 Rewrite [rotate_diatomic_block](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:108:4-163:5) to write into slices

**File:** [rust_dftb/src/rotation.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:0:0-0:0)

**Current:** Returns two allocated `DMatrix<f64>`.

```rust
// CURRENT (rotation.rs:97)
pub fn rotate_diatomic_block(...) -> Result<(DMatrix<f64>, DMatrix<f64>)>
```

**New signature:**

```rust
pub fn rotate_diatomic_block_into(
    ctx: &SystemContext,
    tab: &SkTableSp,
    sp1_idx: u8,
    sp2_idx: u8,
    r: f64,
    dc: DirectionCosines,
    out_h: &mut [f64],  // caller provides buffer, size = n_orb2 * n_orb1
    out_s: &mut [f64],
) -> Result<<()>
```

**Internal changes:**
- `tab.h.eval_into(r, &mut workspace.sk_buf_h)?` (reuses 20-element stack array)
- `tab.s.eval_into(r, &mut workspace.sk_buf_s)?`
- Extract integrals into `workspace.shell_buf_h` / `workspace.shell_buf_s` (4-element stack arrays)
- Assemble block directly into `out_h` / `out_s` slices
- No `DMatrix`, no `Vec<f64>` allocations

### 2.4 Rewrite [fill_pairs](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:94:4-133:5) hot loop

**File:** [rust_dftb/src/hamiltonian.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:0:0-0:0)

**Before:**
```rust
for p in &neigh.pairs {
    let sp1 = &species[p.i];           // String ref
    let sp2 = &species[p.j];           // String ref
    let (h_blk, s_blk) = Rotation::rotate_diatomic_block(&self.sk, sp1, sp2, p.r, dc)?; // allocates DMatrix x2
    let ni = self.sk.n_orb_species(sp1)?; // HashMap
    ...
}
```

**After:**
```rust
let mut ws = HWorkspace::new(ctx.max_block_size);

for p in &neigh.pairs {
    let si = ctx.atom_species[p.i];
    let sj = ctx.atom_species[p.j];
    let pair_idx = ctx.pair_lut[(si as usize) * ctx.n_species + (sj as usize)]
        .ok_or_else(|| DftbError::InvalidInput("missing pair".into()))?;
    let tab = ctx.pair_tables[pair_idx];

    let dc = DirectionCosines::from_vec(p.vec_ij)?;
    let ni = ctx.atom_n_orb[p.i] as usize;
    let nj = ctx.atom_n_orb[p.j] as usize;
    let bi = ctx.atom_orb_off[p.i] as usize;
    let bj = ctx.atom_orb_off[p.j] as usize;

    let block_size = ni * nj;
    Rotation::rotate_diatomic_block_into(
        &ctx, tab, si, sj, p.r, dc,
        &mut ws.block_h[..block_size],
        &mut ws.block_s[..block_size],
    )?;

    // Write directly into global dense matrices
    for a in 0..nj {
        for b in 0..ni {
            let vh = ws.block_h[a * ni + b];
            let vs = ws.block_s[a * ni + b];
            h0[(bj + a, bi + b)] = vh;
            h0[(bi + b, bj + a)] = vh;
            s[(bj + a, bi + b)] = vs;
            s[(bi + b, bj + a)] = vs;
        }
    }
}
```

**Impact:** 
- 0 HashMap lookups in the loop
- 0 `String` operations
- 0 `Vec`/`DMatrix` allocations
- Only mutable slice indexing and arithmetic

## Phase 3: SK Extraction Hardening

**Goal:** Remove all per-call lookup table reconstruction and `Vec` returns from SK integral extraction.

### 3.1 Const lookup table for old format

**File:** [rust_dftb/src/sk_data.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:0:0-0:0)

**Current problem:** [extract_shell_integrals_old](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:356:0-379:1) rebuilds `new_to_old` on every call.

```rust
// CURRENT (sk_data.rs:348-355)
let mut new_to_old = [0usize; 21];
for (old_idx, &new_col) in iSKInterOld.iter().enumerate() {
    new_to_old[new_col] = old_idx;
}
```

**Change:** Make it a `const`.

```rust
const NEW_TO_OLD: [usize; 21] = {
    let mut arr =

Here is the condensed, actionable plan. I will keep it concrete but avoid full implementations to stay within limits.

# Definite Implementation Plan

## Phase 1: Precomputation Layer (`SystemContext`)

**Files:** [sk_data.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:0:0-0:0), new module `system.rs` (or top of [hamiltonian.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:0:0-0:0))

| Step | Change | File | Before | After |
|---|---|---|---|---|
| 1.1 | Normalize SK pair keys | [sk_data.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:0:0-0:0) | [get_pair](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:138:4-142:5) tries 2 string keys | Store with `a <= b` at load; [get_pair](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:138:4-142:5) does 1 lookup, 0 allocs |
| 1.2 | Add `SkFormat` enum | [sk_data.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:0:0-0:0) | Runtime `h_all.len() == 20` branch | `SkFormat::Old` / `Extended` stored in [SkTableSp](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:116:0-121:1) at load |
| 1.3 | Create `SystemContext` | [hamiltonian.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:0:0-0:0) | String/HashMap lookups in hot loop | Flat `Vec<u8>`/`Vec<u16>` arrays; all lookups done once at setup |

**`SystemContext` fields:**
```rust
pub struct SystemContext<'a> {
    n_atoms: usize, n_species: usize, n_orbs: usize,
    atom_species: Vec<u8>,      // per atom
    atom_n_orb: Vec<u8>,        // per atom
    atom_orb_off: Vec<u16>,     // per atom
    species_n_orb: Vec<u8>,     // per species
    species_ang: Vec<&'a [i32]>,// per species
    species_onsite: Vec<&'a AtomicParamsSp>,
    pair_lut: Vec<Option<<usize>>,// flat [si * n_species + sj]
    pair_tables: Vec<&'a SkTableSp>,
}
```

**Build location:** Inside [build_non_scc](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:22:4-69:5), before the hot loop:
```rust
let ctx = SystemContext::from_sk_data(&self.sk, species)?;
```

## Phase 2: Zero-Allocation Hot Loop

**Files:** [interpolation.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/interpolation.rs:0:0-0:0), [rotation.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:0:0-0:0), [hamiltonian.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:0:0-0:0)

| Step | Change | File | What it fixes |
|---|---|---|---|
| 2.1 | Add `eval_into` | [interpolation.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/interpolation.rs:0:0-0:0) | [eval(r)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/interpolation.rs:28:4-30:5) returns `Vec<f64>` → write into caller-provided `&mut [f64]` |
| 2.2 | Add `HWorkspace` | [hamiltonian.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:0:0-0:0) | Reusable stack/heap scratch: `sk_buf[20]`, `shell_buf[4]`, `block: Vec<f64>` (pre-sized once) |
| 2.3 | `rotate_diatomic_block_into` | [rotation.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:0:0-0:0) | No `DMatrix` return; writes into `&mut [f64]` slices |
| 2.4 | Rewrite [fill_pairs](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:94:4-133:5) | [hamiltonian.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:0:0-0:0) | 0 HashMap, 0 String, 0 allocations per pair |

**New `rotate_diatomic_block_into` signature:**
```rust
pub fn rotate_diatomic_block_into(
    ctx: &SystemContext,
    tab: &SkTableSp,
    si: u8, sj: u8,
    r: f64, dc: DirectionCosines,
    out_h: &mut [f64],
    out_s: &mut [f64],
    ws: &mut HWorkspace,
) -> Result<<()>
```

**Hot loop pattern after changes:**
```rust
let mut ws = HWorkspace::new(max_block_size);

for p in &neigh.pairs {
    let si = ctx.atom_species[p.i];
    let sj = ctx.atom_species[p.j];
    let tab_idx = ctx.pair_lut[(si as usize) * n_species + (sj as usize)]
        .ok_or(...)?;
    let tab = ctx.pair_tables[tab_idx];

    let dc = DirectionCosines::from_vec(p.vec_ij)?;
    let ni = ctx.atom_n_orb[p.i] as usize;
    let nj = ctx.atom_n_orb[p.j] as usize;
    let bi = ctx.atom_orb_off[p.i] as usize;
    let bj = ctx.atom_orb_off[p.j] as usize;

    Rotation::rotate_diatomic_block_into(
        &ctx, tab, si, sj, p.r, dc,
        &mut ws.block_h[..ni*nj],
        &mut ws.block_s[..ni*nj],
        &mut ws,
    )?;

    // Write into global dense matrices from preallocated slice
    for a in 0..nj {
        for b in 0..ni {
            let vh = ws.block_h[a * ni + b];
            let vs = ws.block_s[a * ni + b];
            h0[(bj + a, bi + b)] = vh;
            h0[(bi + b, bj + a)] = vh;
            s[(bj + a, bi + b)] = vs;
            s[(bi + b, bj + a)] = vs;
        }
    }
}
```

## Phase 3: SK Extraction Hardening

**File:** [sk_data.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:0:0-0:0)

| Step | Change | Before | After |
|---|---|---|---|
| 3.1 | Const lookup table | `new_to_old` rebuilt on every call | `const NEW_TO_OLD: [usize; 21]` |
| 3.2 | `extract_shell_integrals_into` | Returns `Vec<f64>` | Writes into `&mut [f64]` slice, length known from `l_min + 1` |
| 3.3 | `eval_shell_integrals_into` | Returns `(Vec<f64>, Vec<f64>)` | Uses `ws.sk_buf_h` / `ws.sk_buf_s` stack arrays, writes into `&mut [f64]` |

## Phase 4: Interpolation Streamlining (Deferred to Phase 2 or 3)

**File:** [interpolation.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/interpolation.rs:0:0-0:0)

The [poly_inter_uniform_vec](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/interpolation.rs:45:0-109:1) clones the entire jagged `Vec<Vec<f64>>` structure and allocates `delta`, `yy`, `dyy` on every call.

**Fix:** Write a specialized `eval_sk_table_into` that:
- Works directly on `EqGridTable.values` (stored as `[n_grid][n_integ]`)
- Uses stack work arrays `[f64; 8]` for interpolation points
- Writes result into caller-provided `&mut [f64]`

This removes all `Vec` cloning inside interpolation.

## Execution Order

I recommend implementing in this order, with a working test after each phase:

1. **Phase 1** (`SystemContext` + normalized keys + `SkFormat`) — lowest risk, immediately removes HashMap from hot loop
2. **Phase 2.1 + 2.2** (`eval_into` + `HWorkspace`) — adds infrastructure
3. **Phase 2.3 + 2.4** (`rotate_diatomic_block_into` + [fill_pairs](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:94:4-133:5) rewrite) — this is the big payoff; verify with parity tests
4. **Phase 3** (const tables + `*_into` functions) — clean up remaining allocations
5. **Phase 4** (interpolation specialization) — final polish

Which phase would you like me to start implementing?
