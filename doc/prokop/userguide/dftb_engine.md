# `dftb_engine` — how to run a calculation

The program is **one binary**. A calculation is **an input script**, not a new Rust program.

```text
dftb_engine --script my_job.rhai --sk-dir /path/to/sk-pack
```

Two production engines share that binary:

| Script prefix | Engine | Typical SK | What it is for |
|---------------|--------|------------|----------------|
| `gpu_*` | `GpuDftb` | mio (C/H/N/O) | Dense GPU DFTB, **homogeneous replicas** |
| `sparse_*` | `SparseDftb` | matsci (Si/H, …) | Sparse BSR4 DFTB, **one system** |

Dense details: this page. Sparse details: [sparse_dftb.md](sparse_dftb.md).

If a job cannot be expressed as a script, the CLI is missing a function. Add that function to `dftb_engine`. Do **not** add `src/bin/…`, `examples/…`, or `tests/foo.rs` for a new molecule, scan, replica count, or nanocrystal.

---

## 1. What you get

`dftb_engine` loads Slater–Koster tables once, compiles OpenCL kernels once, and then runs what the script asks for.

The production path is **dense GPU DFTB** (`GpuDftb`):

1. load a geometry (XYZ, or atoms typed in the script)
2. create an engine (`gpu_new`) — SK pack, kernels, buffers
3. SCC (`gpu_scc`)
4. energy, optionally forces, in **one** call (`gpu_eval`)

Replicas are copies of **one** molecule in one GPU launch (`batch`). Different molecules are sequential steps in the same script (two `gpu_new` calls), not mixed in one launch.

Coordinates are **Ångström**. Energies are **Hartree**. Forces are **Hartree / Å**.

GPU work is **f32** on an **NVIDIA** device. Do not treat PoCL/CPU OpenCL as a GPU run. Time only `--release` builds.

---

## 2. Build and run

From `rust_dftb/`:

```bash
export RUST_DFTB_SK_DIR=/path/to/slakos/mio-1-1   # folder with H-H.skf, C-C.skf, …

cargo run --release --bin dftb_engine -- \
  --script scripts/test_gpu_dftb_molecules.rhai \
  --sk-dir "$RUST_DFTB_SK_DIR"
```

`--sk-dir` overrides `RUST_DFTB_SK_DIR`. One of them must point at a real mio (or other) SK folder.

```bash
dftb_engine --help
```

prints the script functions.

**Needs:** system OpenBLAS (`libopenblas-dev` on Debian/Ubuntu), an NVIDIA GPU with working OpenCL for `gpu_*` calls.

Worked example on this machine (2026-09-10): H2O, AT, GC, 7-azaindole dimer — see `rust_dftb/scripts/test_gpu_dftb_molecules.rhai`.

---

## 3. First script

Save as e.g. `rust_dftb/scripts/h2o.rhai`:

```rhai
const sk_dir = SK_DIR;

make_geom("h2o", "O,H,H", [
    0.0, 0.0, 0.0,
    -0.7580632005, 0.6358101311, 0.0,
    0.7580632005, 0.6358101311, 0.0
]);

gpu_new("h2o", sk_dir, 1);          // batch = 1 copy
let rms = gpu_scc("h2o", 100, 1e-6);
let e = gpu_eval("h2o", true);      // true = also forces
let fmax = gpu_max_force("h2o");

print("E = " + ftos(e) + " Ha   rms = " + ftos(rms) + "   max|F| = " + ftos(fmax));
assert_finite(e, "H2O energy");
```

Run:

```bash
cargo run --release --bin dftb_engine -- --script scripts/h2o.rhai --sk-dir "$RUST_DFTB_SK_DIR"
```

Load a file instead of typing atoms:

```rhai
load_xyz("at", REPO_ROOT + "/data/xyz/adenine-thymine.xyz");
gpu_new("at", SK_DIR, 1);
gpu_scc("at", 100, 1e-6);
let e = gpu_eval("at", true);
print("AT E = " + ftos(e));
```

The name `"at"` is a handle: geometry and GPU engine share it. `gpu_new("at", …)` looks up the geometry stored under `"at"`.

---

## 4. Script language

Scripts are **Rhai** (small JS-like language). You need very little of it:

| Construct | Example |
|-----------|---------|
| Constant from CLI | `const sk_dir = SK_DIR;` |
| String concat | `REPO_ROOT + "/data/xyz/foo.xyz"` |
| Print | `print("E=" + ftos(e));` |
| Loop | `while i < n { … i += 1; }` |
| Booleans | `gpu_eval("h2o", true);` |

**Built-in constants** (set by the binary, not the script):

| Name | Meaning |
|------|---------|
| `SK_DIR` | Slater–Koster folder (`--sk-dir` or `RUST_DFTB_SK_DIR`) |
| `REPO_ROOT` | Repository root (must contain `data/xyz/adenine-thymine.xyz`) |
| `A_CC` | Graphene C–C length (Å), for flake builders |

Outputs from calculations go under `debug/<topic>/`, never into `scripts/` or `doc/`.

Errors **abort**. `NaN` is not a valid return. `assert_finite` / `assert_close` / `die` exist so a script can fail loudly with a message.

---

## 5. Production functions (`gpu_*`)

This is the API to grow. Geometry first, then the engine.

### Geometry

| Function | Returns | |
|----------|---------|-|
| `load_xyz(name, path)` | n_atoms | Standard XYZ, coordinates in Å. Species in the geometry table: C, H, N, O, B, F, Si, P, S, Cl. |
| `make_geom(name, species_csv, xyz_flat)` | n_atoms | e.g. `"O,H,H"` and a flat `[x,y,z, x,y,z, …]` array in Å. |
| `save_xyz(name, path)` | bool | Write a stored geometry. |

### GPU DFTB

| Function | Returns | |
|----------|---------|-|
| `gpu_new(name, sk_dir, batch)` | n_orbs | Compile/allocate once. `batch` = number of **identical** copies. NVIDIA required. |
| `gpu_scc(name, max_iter, tol)` | charge rms | Self-consistent charges. Cap at 100; do not ask for 400. Production mixer = GPU DIIS (`mix=0`). |
| `gpu_scc_mixer(name, max_iter, tol, mix)` | charge rms | Same electronic kernels. `mix`: 0 GPU DIIS, 1 GPU simple α=0.3, 2 host f64 DIIS (batch=1). |
| `gpu_reset_q(name)` | | Reload `q0` and reset DIIS (fair mixer A/B / scan points). |
| `gpu_set_coords(name, xyz_flat)` | n_atoms | Per-geometry refill. Å. Resets DIIS, does not reset charges — call `gpu_reset_q` if you want `q0`. |
| `gpu_eval(name, want_forces)` | E of replica 0 (Ha) | **One** electronic finalize. `want_forces=false` → energy only. `true` → energy + forces (W stays on the GPU). |
| `gpu_measure(name, want_cpu)` | E of replica 0 (Ha) | Frozen-H + energy identities; if `want_cpu`, CPU f64 SCC+F and assembly `ΔH0`/`ΔS`. Batch=1. |
| `gpu_cpu_energy(name)` | E (Ha) | Independent CPU f64 SCC + repulsive at replica 0. |
| `gpu_energy_i(name, i)` | E of replica `i` | After `gpu_eval`. |
| `gpu_max_force(name)` | max \|F\| | After `gpu_eval(name, true)`. Hartree/Å. |
| `get_xyz(name)` | flat Å array | Geometry table (for scans). |
| `gpu_n_batch` / `gpu_n_orbs` / `gpu_n_atoms` / `gpu_scc_iters` / `gpu_scc_stalled` / `gpu_q_rms` | | |

Typical loop:

```text
load_xyz / make_geom
gpu_new          ← once per molecule template
gpu_scc
gpu_eval(true)   ← energy and forces together
```

Do **not** invent a second “energy then forces” sequence. That would re-solve the electronic problem.

### Checks and printing

| Function | |
|----------|-|
| `ftos(x)` / `itos(n)` | Number → string for `print`. |
| `assert_finite(x, msg)` | Abort if NaN/Inf. |
| `assert_close(a, b, tol, msg)` | Abort if \|a−b\| > tol. |
| `die(msg)` | Abort. |

---

## 6. Batch (many copies of one system)

`gpu_new("h2o", sk_dir, 4)` runs **four waters with the same geometry** in one GPU launch. After SCC, replica energies must match (they are the same physical system).

```rhai
make_geom("w", "O,H,H", [ /* … */ ]);
gpu_new("w", SK_DIR, 4);
gpu_scc("w", 100, 1e-6);
let e0 = gpu_eval("w", true);
assert_close(e0, gpu_energy_i("w", 1), 1e-4, "replica 1");
```

Adenine–thymine and guanine–cytosine together: two blocks, two names, two `gpu_new` — not `batch` of mixed species.

If a pair type appears that was not allocated at `gpu_new`, or the pair buffer overflows, the engine **stops**. Rebuild with `gpu_new`; do not silently grow buffers mid-run.

---

## 7. Also on the CLI (not the dense-GPU product)

**Sparse BSR4 DFTB (production for large / nanocrystal work)** — `sparse_new`, `sparse_scc`, `sparse_eval`, `sparse_fire_step`, `sparse_md_step`, `sparse_relax`. One system, not a replica batch. Si/H needs matsci (or another Si–H pack), not mio. User guide: [sparse_dftb.md](sparse_dftb.md). Example: `scripts/test_sparse_dftb_sih4.rhai`.

**Graphene builders** — `build_pah`, `build_flake`, `build_zigzag`.

**CPU dense DFTB** — `run_dftb_scc`, `run_dftb_nonscc`, then `get_energy`, `get_charges`, `get_eigenvalues`, `save_*`. f64 on the host. Fine for a single-system reference; not the batched GPU path.

**Sparse purification leftovers** — `run_sparse_purify`, `run_sparse_purify_geom`, `davidson_homo_lumo`, `compare_density`, `compare_charges`. These purify a **dense CPU SCC** result. They are not `SparseDftb`. Do not use them as the template for a new nanocrystal, FIRE, or MD script.

Example leftover scripts: `scripts/test_graphene_sparse.rhai`, `scripts/test_charges_homo_lumo.rhai`.

---

## 8. Not on the CLI yet

These exist on the Rust object `GpuDftb` but are **not** script functions. A user who needs them should get a `gpu_*` function, not a new binary.

| Missing script call | Already on `GpuDftb` |
|---------------------|----------------------|
| Move atoms / reload pairs | `set_coords` — **not** a `gpu_*` script call yet. Sparse already has `sparse_set_coords`. |
| FIRE / MD / relax | on `GpuDftb` in Rust, **not** `gpu_*` script calls yet. Sparse already has `sparse_fire_step` / `sparse_md_step` / `sparse_relax`. |
| Mixed molecules in one launch | not supported (homogeneous batch only) |
| Species beyond the `Element` table | add the element in `geometry/mod.rs` |

FIRE vs CPU force parity on this engine is still open. Do not advertise “production geometry optimization” until that is shown **through this CLI**.

---

## 9. What we are cleaning up

The user-facing surface is this binary + `rust_dftb/scripts/*.rhai`. Everything below is leftover scaffolding. Do not grow it. Prefer deleting or not calling it once the same job runs as a script.

| Leftover | Why it exists | Replacement |
|----------|---------------|-------------|
| `rust_dftb/examples/hbond_ref.rs`, `scan.rs`, `neb.rs`, `test_h2.rs` | One binary per demo | `dftb_engine` + a `.rhai` |
| `rust_dftb/src/bin/graphene_build.rs` | Extra binary | `build_pah` / `build_flake` in a script |
| New `rust_dftb/tests/*.rs` per molecule | Each file is another compile target | `scripts/<case>.rhai` |
| `tests/gpu_hbond_physics.rs` throwaway OpenCL | Physics bisect | `gpu_*` on `GpuDftb` |
| `GpuDriver` / `GpuForceDriver` / `gpu_scc.rs` one-shot | Early GPU bring-up | `GpuDftb` via `gpu_new` |
| `cargo test --test gpu_dftb` | H2O compile/smoke | Same physics in `test_gpu_dftb_molecules.rhai` |
| `cargo test --test sparse_dftb` | SiH₄ compile/smoke | Same physics in `test_sparse_dftb_sih4.rhai` |

Parity tests vs Fortran (`parity_*.rs`) stay. They are not user jobs.

---

## 10. Accuracy (so the CLI does not lie)

- **H2O (N=6):** GPU energy vs CPU is tight (~3×10⁻⁷ Ha in earlier checks).
- **AT/GC (N≈86–87):** SCC charge rms often plateaus around 10⁻⁵, not 10⁻⁶. Energy vs CPU is on the order of 10⁻⁵ Ha (~0.7 meV). That is the current f32 / eigensolver floor, not a licence to “fix” it by loosening a test.
- SCC **max 100** iterations. If it stalls, the script still gets an energy; check `gpu_scc_iters` and the printed rms. Do not raise the cap to 400.

Details: `doc/prokop/topical_audit/f32_floor_dense_hbond.md`.

---

## 11. File map

| Path | Role |
|------|------|
| `rust_dftb/src/bin/dftb_engine.rs` | The CLI (Rhai host). Add user features here. |
| `rust_dftb/src/qmqm/gpu_dftb.rs` | Persistent dense GPU engine. |
| `rust_dftb/src/methods/sparse/sparse_dftb.rs` | Persistent sparse GPU engine. |
| `rust_dftb/scripts/*.rhai` | User jobs and tests. |
| `doc/prokop/userguide/sparse_dftb.md` | Sparse CLI (same binary). |
| `data/xyz/` | Geometries. |
| `debug/` | Run output. Do not commit. |

Task notes (not a user manual): `doc/prokop/tasts/HBond_Relaxed_Scan_GPU/HBond_Relaxed_Scan_GPU.manifest..md` §0.5.
