---
type: Task
title: Integrate the resident BSR4 purification path
tags: [gpu, sparse, opencl, integration]
---

# Problem

The resident BSR4 structures, cached kernels, and `SparsePurifyWorkspace` already exist in `methods/sparse/gpu_sparse.rs`, but the Rhai large-system entry point still calls the legacy host-roundtrip solver. The legacy path rebuilds/uploads/reads sparse matrices during purification and computes redundant products. The integration must route the existing public resident Newton–Schulz and TC2 APIs into the entry point while preserving the current full-mask inputs and numerical behavior for this first step.

# Summary

Wire `rhai_run_sparse_purify` in `rust_dftb/src/bin/dftb_engine.rs` to the existing resident APIs. Keep the old path available as a clearly marked reference until parity and regression checks pass; do not duplicate the whole implementation in comments. Add fail-loud handling for nonconvergence and never return a best intermediate matrix as if it were converged. This change still consumes the existing dense DFTB result and must not claim a fully sparse SCC or linear-scaling end-to-end solver. Mixed H support, mask construction, geometry handling, and tolerances remain unchanged.

Validation is defined before implementation: run the existing release GPU sparse baseline, then compare resident and legacy outputs for energies/charges/status and check residuals, trace, finite values, and convergence status. Preserve the old path as the numerical reference during the transition. Do not run the full test suite concurrently with the baseline/regression agent.

## Agent dispatch checklist — copy/paste assignments

| Owner | Scope | Deliverable |
|---|---|---|
| `sparse_audit` | Own `gpu_sparse.rs` and `sparse_bsr4_purification.cl`; stabilize resident residual/reduction behavior, preallocation, and the two-product loop. | Narrow implementation patch with exact anchors and parity risks. |
| `alternatives_audit` | Establish the existing release GPU sparse baseline and prepare baseline/regression comparisons in `tests/gpu_sparse_bsr4.rs`. | Full unfiltered test output, timings, residuals, and numerical reference values. |
| `dense_audit` | Own this task spec and `rust_dftb/src/bin/dftb_engine.rs`; after baseline, integrate the resident Newton–Schulz + `SparsePurifyWorkspace` TC2 path. | Minimal integration patch, preserved reference path, fail-loud nonconvergence/status handling. |
| `root` | Review scope/API compatibility and coordinate worker findings. | Technical review notes; user acceptance remains outside the worker role. |
| `root` later | Update `OVERVIEW_Roadmap.md` and related status/report documents after validation. | Roadmap/status update only after evidence is available. |

## Integration constraints

- No API-breaking changes are expected.
- Preserve full masks and current numerical tolerances initially.
- Do not silently fall back to the legacy path if resident setup or convergence fails.
- Do not report a best-state or exhausted iteration as converged.
- Do not edit unrelated kernels, mixed H support, mask generation, geometry, or tolerances.
- Keep output status explicit: converged, nonconverged, or failed with residual context.

## Implementation status

- `dftb_engine.rs` now wires resident Newton–Schulz and `SparsePurifyWorkspace::tc2_purify_dev`; resident setup and nonconvergence fail loudly, with no legacy fallback.
- `cargo check --manifest-path rust_dftb/Cargo.toml --bin dftb_engine` passes. Release GPU baseline, resident-vs-legacy parity, and runtime convergence remain pending the GPU test agent; no full test suite was run here.
