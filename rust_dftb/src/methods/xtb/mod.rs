//! Analytical xTB Hamiltonian and parameters (GFN1/GFN2).

pub mod params;
pub mod params_gfn2;
pub mod basis;
pub mod integrals;
pub mod hamiltonian;
pub mod coulomb;
pub mod mulliken;
pub mod multipole_integrals;
pub mod scf;

pub use hamiltonian::{build_h0_s, build_h0_s_gfn2, XtbBuilder};
