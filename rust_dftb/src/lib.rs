//! Rust implementation of DFTB non-SCC Hamiltonian assembly
//! 
//! This library implements the core DFTB+ Hamiltonian assembly workflow
//! for sp basis sets, focusing on non-SCC components with parity checking
//! against the original DFTB+ implementation.

pub mod error;
pub mod sk_data;
pub mod interpolation;
pub mod rotation;
pub mod neighbor;
pub mod hamiltonian;
pub mod scc;
pub mod output;

pub use error::{DftbError, Result};
pub use sk_data::{SkData, SkTableSp};
pub use interpolation::{InterpolationMethod, EqGridTable};
pub use rotation::{Rotation, DirectionCosines};
pub use neighbor::{NeighborList, NeighborBuilder};
pub use hamiltonian::{HamiltonianBuilder, Hamiltonian};
pub use scc::Scc;
pub use output::{DftbOutput, OutputFormat};
