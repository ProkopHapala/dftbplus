//! Rust implementation of semi-empirical LCAO solvers (DFTB, xTB) and
//! multi-system QM/QM fragment solver.
//!
//! Module layout:
//! - `core/`    – method-agnostic primitives (errors, neighbor finding, charges)
//! - `methods/` – specific Hamiltonian methods (DFTB SK-tables, xTB analytical)
//! - `qmqm/`    – multi-fragment QM/QM solver (generic over Hamiltonian / Coulomb model)

pub mod core;
pub mod methods;
pub mod qmqm;

// --- Re-exports for backward compatibility ---
// These keep existing tests and callers working without changing their imports.

pub use core::error::{DftbError, Result};
pub use core::neighbor::{NeighborList, NeighborBuilder};
pub use core::charges;

pub use methods::dftb::sk_data::{SkData, SkTableSp, SpeciesOrbitals, AtomicParamsSp};
pub use methods::dftb::interpolation::{InterpolationMethod, EqGridTable};
pub use methods::dftb::rotation::{Rotation, DirectionCosines};
pub use methods::dftb::hamiltonian::{HamiltonianBuilder, Hamiltonian, SccResult};
pub use methods::dftb::gamma::{GammaTable, gamma_full};

pub use methods::traits::{H0Builder, CoulombModel};

pub mod io;

// Backward-compatible re-exports
pub use io::{DftbOutput, OutputFormat};
pub use io::{parse_xyz, parse_species, parse_coords, parse_f64_list, XyzMolecule};
pub use io::{max_abs_diff, max_abs_diff_vec, compare_matrices, compare_vecs, default_ang_map, load_sk_for_species, capitalize, permute_sp_per_atom};
