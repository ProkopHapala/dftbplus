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
pub mod gpu_sparse;
pub mod davidson;
pub mod sparse_forces;

pub use bsr4::{
    build_geometric_mask, build_identity, build_product_mask, build_full_mask,
    diag_block_map, gershgorin_bounds, inf_norm, transpose_block_map,
    Bsr4Matrix, Bsr4Mask, SpgemmPlan, build_spgemm_plan_bsym,
};
pub use gpu_sparse::{SparseBsr4Config, SparseBsr4Gpu, SparsePerfStats, SpgemmPlanGpu};
pub use davidson::davidson_homo_lumo;
