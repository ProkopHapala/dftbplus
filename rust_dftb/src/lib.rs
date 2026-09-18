//! Rust implementation of semi-empirical LCAO solvers (DFTB, xTB) and
//! multi-system QM/QM fragment solver.
//!
//! Module layout:
//! - `core/`    – method-agnostic primitives (errors, neighbor finding, charges)
//! - `methods/` – specific Hamiltonian methods (DFTB SK-tables, xTB analytical)
//! - `qmqm/`    – multi-fragment QM/QM solver (generic over Hamiltonian / Coulomb model)

pub mod core;
pub mod geometry;
pub mod methods;
pub mod qmqm;

// --- Re-exports for backward compatibility ---
// These keep existing tests and callers working without changing their imports.

pub use core::charges;
pub use core::error::{DftbError, Result};
pub use core::neighbor::{NeighborBuilder, NeighborList};

pub use methods::dftb::gamma::{gamma_full, GammaTable};
pub use methods::dftb::hamiltonian::{Hamiltonian, HamiltonianBuilder, SccResult, SystemContext};
pub use methods::dftb::interpolation::{EqGridTable, InterpolationMethod};
pub use methods::dftb::rotation::{DirectionCosines, Rotation};
pub use methods::dftb::sk_data::{AtomicParamsSp, SkData, SkTableSp, SpeciesOrbitals};

pub use methods::traits::{CoulombModel, H0Builder};

pub mod io;

// Backward-compatible re-exports
pub use io::{
    capitalize, compare_matrices, compare_vecs, default_ang_map, load_sk_for_species, max_abs_diff,
    max_abs_diff_vec, permute_sp_per_atom,
};
pub use io::{parse_coords, parse_f64_list, parse_species, parse_xyz, XyzMolecule};
pub use io::{DftbOutput, OutputFormat};
