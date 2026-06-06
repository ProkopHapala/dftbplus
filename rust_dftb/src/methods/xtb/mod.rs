//! Analytical xTB Hamiltonian and parameters (GFN1/GFN2).

pub mod params;
pub mod basis;
pub mod integrals;
pub mod hamiltonian;

pub use hamiltonian::{build_h0_s, XtbBuilder};
