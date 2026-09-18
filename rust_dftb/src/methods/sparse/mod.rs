//! Block-CSR (BSR4) sparse linear-scaling route for large DFTB systems.
//!
//! This module is **independent of the `qmqm` dense/fragment solver**. It
//! implements the sparse density-kernel purification architecture described
//! in `doc/prokop/chats/SparseLargeSystemOpenCL.chat.md`:
//!
//! - Atom-block CSR with 4×4 blocks (one atom = 4 orbitals for s,p DFTB).
//! - Non-orthogonal density kernel `K` with `KSK = K`, `N_occ = Tr(KS)`.
//! - Masked sparse matrix products `C = P_M(A·B)` that never densify.
//! - Generalized McWeeny (`3KSK − 2KSKSK`) and metric TC2 purifiers.
//! - Mulliken charges, trace, and idempotency residual directly from sparse
//!   `KS`.
//!
//! The OpenCL kernels live in `sparse_bsr4_purification.cl` and are compiled
//! via the shared `GpuRuntime` (no second OpenCL context is created).
//!
//! No `S^{-1}` or `S^{-1/2}` is ever constructed; matrix polynomials are
//! evaluated as sequences of masked sparse products so every intermediate
//! stays O(N·z) in memory.

pub mod bsr4;
pub mod davidson;
pub mod gpu_sparse;
pub mod harness;
pub mod scc;
pub mod sparse_dftb;
pub mod sparse_forces;
pub mod sparse_system;

pub use bsr4::{
    bsr_values_to_dense, build_full_mask, build_geometric_mask, build_identity, build_product_mask,
    build_spgemm_plan_bsym, diag_block_map, fill_bsr_values_from_dense, gershgorin_bounds,
    inf_norm, pad_physical_to_bsr4, pad_physical_to_bsr4_into, transpose_block_map, Bsr4Mask,
    Bsr4Matrix, SpgemmPlan,
};
pub use davidson::davidson_homo_lumo;
pub use gpu_sparse::{SparseBsr4Config, SparseBsr4Gpu, SparsePerfStats, SpgemmPlanGpu};
pub use scc::{
    apply_shift_padded, apply_shift_padded_into, energy_non_scc, eval_sparse_energy_forces,
    run_sparse_scc, SparseDftbEnergy,
};
pub use sparse_dftb::{valence_q0, SparseDftb, SparseDftbConfig, SparseDftbScc};
pub use sparse_forces::{dw_from_k_padded, sparse_analytic_forces, unpad_to_physical};
