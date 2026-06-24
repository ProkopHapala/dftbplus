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
pub mod mixer;
pub mod neighbor;
pub mod fragment;
pub mod shifts;
pub mod gpu_prep;
pub mod gpu_matrix;
pub use crate::core::charges;
pub mod solver;

pub use gamma::{GammaTable, gamma_full};
pub use mixer::{Mixer, SimpleMixer, DiisMixer};
pub use neighbor::FragmentNeighborList;
pub use fragment::{Fragment, FragmentTemplate};
pub use solver::MultiSystemSolver;
