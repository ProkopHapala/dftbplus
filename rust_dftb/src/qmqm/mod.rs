//! Multi-system QM/QM solver for non-covalently bonded fragments.
//!
//! Each fragment is diagonalized independently. Inter-fragment interactions are treated
//! as electrostatic perturbations via the DFTB γ-function. The global SCC loop uses a
//! single charge-vector with an Anderson/DIIS mixer.
//!
//! Design goals:
//! - Zero allocation in the hot SCC loop.
//! - Embarrassingly parallel fragment diagonalization.
//! - O(N) inter-fragment neighbor finding via cell lists.

pub use crate::methods::dftb::gamma;
pub mod fragment;
pub mod gpu_driver;
pub mod gpu_eigen;
pub mod gpu_matrix;
pub mod gpu_prep;
pub mod gpu_runtime;
pub mod gpu_scc;
pub mod gpu_scc_plan;
pub mod mixer;
pub mod neighbor;
pub mod shifts;
// Experimental complex-valued PBC path (arbitrary k, float2 Hermitian
// H(k)/S(k)) — kept separate from the production real path until the
// standalone validation in tests/gpu_hermitian_jacobi.rs is done.
// See doc/prokop/tasts/HBond_Relaxed_Scan_GPU/Dense_Multi_PBC.arch_notes.md.
pub mod gpu_dftb;
pub mod gpu_forces;
pub mod gpu_hermitian;
pub mod gpu_pbc;
pub mod gpu_pbc_plan;
pub mod gpu_purify;
pub mod pbc_cell;
// Constrained-DFT fragment-charge layer (Dense_Multi_CDFT spec) —
// optional attach on GpuSccPlan; None → zero cost on the solver.
pub mod gpu_cdft;
pub use crate::core::charges;
pub mod solver;

pub use fragment::{Fragment, FragmentTemplate};
pub use gamma::{gamma_full, GammaTable};
pub use gpu_cdft::{CdftReport, GpuCdft};
pub use gpu_dftb::{GpuDftb, GpuDftbEval, GpuDftbScc, SccStatus};
pub use gpu_forces::GpuForceDriver;
pub use gpu_scc_plan::GpuSccPlan;
pub use mixer::{DiisMixer, Mixer, SimpleMixer};
pub use neighbor::FragmentNeighborList;
pub use solver::MultiSystemSolver;
