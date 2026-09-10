# Sparse DFTB — how to run a calculation

Same program as dense GPU DFTB: **one binary**, a calculation is **an input script**.

```text
dftb_engine --script my_sparse_job.rhai --sk-dir /path/to/matsci-0-3
```

If a job cannot be expressed as a script, the CLI is missing a function. Add that function to `dftb_engine`. Do **not** add `src/bin/…`, `examples/…`, or `tests/foo.rs` for a new molecule or nanocrystal.

Dense multi-system DFTB (`gpu_*`) is documented in [dftb_engine.md](dftb_engine.md). This page is the **sparse** product (`sparse_*` → `SparseDftb`). Do not mix the two solvers in one physical job.

---

## 1. What you get

`dftb_engine` loads Slater–Koster tables once, compiles OpenCL BSR4 kernels once, and then runs what the script asks for.

The production sparse path is **one system at a time** (`SparseDftb`):

1. load a geometry (XYZ, or atoms typed in the script)
2. create an engine (`sparse_new`) — SK pack, kernels, BSR buffers
3. SCC (`sparse_scc`) — Newton–Schulz Z once per geometry, then mix + TC2
4. energy, optionally forces (`sparse_eval`)
5. optional `sparse_fire_step` / `sparse_md_step` / `sparse_relax`

There is **no replica batch**. Copies of a molecule are sequential `sparse_new` calls (or `sparse_set_coords` on the same handle). That is different from dense `gpu_new(..., batch)`.

Coordinates are **Ångström**. Energies are **Hartree**. Forces are **Hartree / Å**.

GPU work is **f32** on an **NVIDIA** device. Do not treat PoCL/CPU OpenCL as a GPU run. Time only `--release` builds. Do not quote `cargo test` wall time as GPU performance.

Topology (which atom pairs exist in the BSR mask) is **frozen at `sparse_new`**. Full mask if the system has ≤ 64 atoms, else a geometric cutoff plus a skin. If a neighbor appears outside that mask, the engine **stops**. Call `sparse_new` again; do not grow CSR in the MD loop.

---

## 2. Build and run

From the repository root:

```bash
export RUST_DFTB_SK_DIR=/path/to/slakos/matsci-0-3   # folder with Si-Si.skf, H-H.skf, Si-H.skf, …
export RUST_DFTB_SPARSE_ALGEBRA_VERBOSE=0            # 1 = per-iter SCC dump

cargo run --release -p rust_dftb --bin dftb_engine -- \
    --script rust_dftb/scripts/test_sparse_dftb_sih4.rhai \
    --sk-dir "$RUST_DFTB_SK_DIR"
```

From `rust_dftb/`, the script path is `scripts/test_sparse_dftb_sih4.rhai`.

`--sk-dir` overrides `RUST_DFTB_SK_DIR`. Si/H sparse work needs a pack that actually contains **Si–H** (matsci-0-3 on this machine). **mio-1-1 does not.**

```bash
dftb_engine --help
```

prints the script functions (dense `gpu_*` and sparse `sparse_*` on the same list).

**Needs:** system OpenBLAS (`libopenblas-dev` on Debian/Ubuntu), an NVIDIA GPU with working OpenCL for `sparse_*` calls.

Worked example: distorted SiH₄ — `rust_dftb/scripts/test_sparse_dftb_sih4.rhai` (reuse SCC, one FIRE step, one MD step).

---

## 3. First script

Save as e.g. `rust_dftb/scripts/sih4.rhai`:

```rhai
const sk_dir = SK_DIR;

make_geom("sih4", "Si,H,H,H,H", [
    0.0, 0.0, 0.0,
    1.48, 0.0, 0.0,
    -0.4933036064009911,  1.3953678912429424,  0.0,
    -0.4933036064009911, -0.4650946033826280,  1.3155753729133621,
    -0.4933036064009911,  0.4650946033826280, -1.3155753729133621
]);

sparse_new("sih4", sk_dir);            // compile / allocate once
let rms = sparse_scc("sih4", 80, 1e-5);
let e = sparse_eval("sih4", true);     // true = also forces
let fmax = sparse_max_force("sih4");

print("E = " + ftos(e) + " Ha   rms = " + ftos(rms) + "   Tr(KS) = " + ftos(sparse_tr_ks("sih4")) + "   max|F| = " + ftos(fmax));
assert_finite(e, "SiH4 energy");
assert_close(sparse_tr_ks("sih4"), 4.0, 0.05, "Tr(KS) vs N_occ");
```

Run:

```bash
cargo run --release -p rust_dftb --bin dftb_engine -- --script rust_dftb/scripts/sih4.rhai --sk-dir "$RUST_DFTB_SK_DIR"
```

Load a file instead of typing atoms:

```rhai
load_xyz("nc", REPO_ROOT + "/data/xyz/your_nanocrystal.xyz");
sparse_new("nc", SK_DIR);
sparse_scc("nc", 80, 1e-5);
let e = sparse_eval("nc", true);
print("E = " + ftos(e));
```

The name `"nc"` is a handle: geometry and sparse engine share it. `sparse_new("nc", …)` looks up the geometry stored under `"nc"`.

---

## 4. Script language

Scripts are **Rhai** (small JS-like language). Same constants and checks as the dense guide:

| Name | Meaning |
|------|---------|
| `SK_DIR` | Slater–Koster folder (`--sk-dir` or `RUST_DFTB_SK_DIR`) |
| `REPO_ROOT` | Repository root |
| `A_CC` | Graphene C–C length (Å), for flake builders |

`print`, `ftos` / `itos`, `while`, `assert_finite` / `assert_close` / `die` work the same as in [dftb_engine.md](dftb_engine.md) §4.

Outputs from calculations go under `debug/<topic>/`, never into `scripts/` or `doc/`.

Errors **abort**. `NaN` is not a valid return.

---

## 5. Production functions (`sparse_*`)

Geometry first, then the engine. Geometry helpers are shared with dense DFTB.

### Geometry

| Function | Returns | |
|----------|---------|-|
| `load_xyz(name, path)` | n_atoms | Standard XYZ, coordinates in Å. Species currently in the table: C, H, N, B, O, F, Si, P, S, Cl. |
| `make_geom(name, species_csv, xyz_flat)` | n_atoms | e.g. `"Si,H,H,H,H"` and a flat `[x,y,z, …]` array in Å. |
| `save_xyz(name, path)` | bool | Write a stored geometry (updated after FIRE/MD). |

### Sparse DFTB

| Function | Returns | |
|----------|---------|-|
| `sparse_new(name, sk_dir)` | n_orbs | Compile/allocate once. **One system**, not a batch. NVIDIA required. |
| `sparse_scc(name, max_iter, tol)` | charge rms | Self-consistent charges. Cap at 100; typical request is 80 and 10⁻⁵. |
| `sparse_eval(name, want_forces)` | E (Ha) | Last SCC energy. `true` also builds analytic forces (CPU contract of D, W). |
| `sparse_max_force(name)` | max \|F\| | After `sparse_eval(name, true)` or a FIRE/MD step. Hartree/Å. |
| `sparse_set_coords(name, xyz_flat)` | n_atoms | Move atoms. Rebuilds H0/S values; **fails** if the frozen mask no longer covers neighbors. |
| `sparse_fire_step(name, f_tol)` | max \|F\| | One FIRE displacement, then H0/S refresh. Call `sparse_scc` after. |
| `sparse_md_step(name, dt)` | max \|F\| | One velocity-Verlet step (mass = 1, disp capped at 0.1 Å). Call `sparse_scc` after. |
| `sparse_relax(name, max_steps, f_tol, scc_tol)` | max \|F\| | SCC + FIRE loop until max\|F\| < `f_tol` or `max_steps`. |
| `sparse_n_atoms` / `sparse_n_orbs` / `sparse_scc_iters` | int | |
| `sparse_tr_ks(name)` | Tr(KS) | Occupied trace after the last SCC (SiH₄ target = 4). |
| `sparse_charges(name)` | csv | Mulliken populations after the last SCC. |

Typical loop:

```text
load_xyz / make_geom
sparse_new          ← once per system
sparse_scc
sparse_eval(true)   ← energy; forces if you asked
sparse_fire_step / sparse_md_step
sparse_scc          ← charges at the new geometry
```

`sparse_eval` does **not** re-solve SCC. After moving atoms, call `sparse_scc` before trusting energy.

### Checks and printing

| Function | |
|----------|-|
| `ftos(x)` / `itos(n)` | Number → string for `print`. |
| `assert_finite(x, msg)` | Abort if NaN/Inf. |
| `assert_close(a, b, tol, msg)` | Abort if \|a−b\| > tol. |
| `die(msg)` | Abort. |

---

## 6. SCC reuse and geometry steps

A second `sparse_scc` on the **same** geometry should warm-start from the previous charges (fewer mix iterations, energy unchanged to ~10⁻⁵ Ha). That is the check that the OpenCL objects are persistent, not rebuilt per call.

```rhai
sparse_scc("sih4", 80, 1e-5);
let e1 = sparse_eval("sih4", false);
sparse_scc("sih4", 80, 1e-5);
let e2 = sparse_eval("sih4", false);
assert_close(e1, e2, 1e-5, "reuse SCC");
```

FIRE / MD change coordinates and invalidate Z (overlap changed). The engine runs Newton–Schulz again on the next `sparse_scc`. Displacement per step is capped; a step that jumps Si–H out of a physical window is a bug, not a licence to clamp.

`sparse_relax` prints unbuffered FIRE progress. Use it for a short local relax, not as a substitute for looking at the numbers.

---

## 7. Also on the CLI (not the sparse product)

**Dense GPU DFTB** — `gpu_new` / `gpu_scc` / `gpu_eval`. Homogeneous replica batch. See [dftb_engine.md](dftb_engine.md).

**CPU dense DFTB** — `run_dftb_scc`, `run_dftb_nonscc`. f64 host reference.

**Sparse purification leftovers** — `run_sparse_purify`, `run_sparse_purify_geom`, `davidson_homo_lumo`. These take a **dense CPU SCC** result and purify K on the GPU. They are **not** `SparseDftb`. Do not use them as the template for a new nanocrystal, FIRE, or MD script. Example leftover scripts: `scripts/test_graphene_sparse.rhai`.

**Graphene builders** — `build_pah`, `build_flake`, `build_zigzag`.

---

## 8. Not on the CLI yet

These exist on `SparseDftb` in Rust or are known gaps. A user who needs them should get a `sparse_*` function, not a new binary.

| Missing / limited | Notes |
|-------------------|--------|
| Homogeneous replica batch | Sparse is one system per engine. Dense `gpu_*` has `batch`. |
| GPU-resident forces | Analytic F is still a CPU contract of D=2K, W=2KHK. |
| GPU H0/S / γ / Hscc | CPU build + upload each mix iteration. |
| Device Newton–Schulz residual | Host `‖I−T‖` of downloaded T is the production check. Device NS `_dev` is wrong — do not use it. |
| Species outside the `Element` table | Add the element (symbol, radius, valence) in `geometry/mod.rs`, then SK files. |
| Mixed sparse + dense in one launch | Sequential steps in one script only. |

---

## 9. What we are cleaning up

The user-facing surface is this binary + `rust_dftb/scripts/*.rhai`. Everything below is leftover scaffolding. Do not grow it.

| Leftover | Why it exists | Replacement |
|----------|---------------|-------------|
| `run_sparse_purify` / `run_sparse_purify_geom` | GPU purify of a dense SCC density | `sparse_*` on `SparseDftb` |
| `rust_dftb/tests/sparse_dftb.rs` | Compile/smoke, same SiH₄ physics | `scripts/test_sparse_dftb_sih4.rhai` |
| `rust_dftb/tests/gate_g3_energy.rs` etc. | Physics gates; **must call `SparseDftb`**, not a second SCC | CLI scripts for jobs; same object |
| New `src/bin` / `examples/` per nanocrystal | One binary per demo | `dftb_engine` + a `.rhai` |

Parity tests vs Fortran (`parity_*.rs`) stay. They are not user jobs.

---

## 10. Accuracy (so the CLI does not lie)

- **SiH₄ (distorted, matsci-0-3, NVIDIA, 2026-09-10):** NS `R_Z` ~ 6×10⁻⁸ (host `‖I−T‖`). SCC energy about **−2.764 Ha**, Tr(KS) ≈ 4. Reuse SCC dropped from ~17 mix iterations to ~4. That is a measured run, not a licence to hard-code the energy in a test.
- SCC charge rms on this molecule can sit around 10⁻⁵. Cap **100** iterations. If it does not converge, the engine **fails** (unlike dense GPU, which may stall and still return an energy).
- Analytic vs finite-difference forces on this geometry are a **gate**, not a user promise. See `doc/prokop/topical_audit/f32_floor_sparse.md`.
- Do not treat a green `cargo test --test sparse_dftb` as a GPU benchmark.

Details: `doc/prokop/topical_audit/f32_floor_sparse.md`, task notes in `doc/prokop/tasts/Sparse_Nanocrystal_Vibrations/`.

---

## 11. File map

| Path | Role |
|------|------|
| `rust_dftb/src/bin/dftb_engine.rs` | The CLI (Rhai host). Add user features here. |
| `rust_dftb/src/methods/sparse/sparse_dftb.rs` | Persistent sparse engine. |
| `rust_dftb/scripts/test_sparse_dftb_sih4.rhai` | User job / test for SiH₄. |
| `doc/prokop/userguide/dftb_engine.md` | Dense GPU CLI (same binary). |
| `debug/` | Run output. Do not commit. |
