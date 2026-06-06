//! Trait boundaries between the QM/QM solver and specific Hamiltonian methods.
//!
//! These two traits are the **only** cross-module references between `qmqm/`
//! and `methods/`.  DFTB and xTB each provide their own implementation.
//!
//! With generics Rust monomorphises at compile time; the hot `fill_pairs`
//! loop and `compute_shifts` loop have **zero** trait-object overhead.

use nalgebra::DMatrix;
use crate::core::error::Result;

/// Builds H0 and S.  Both DFTB and xTB implement this.
pub trait H0Builder: Clone {
    /// Spatial cutoff for pair interactions (Ångström).
    fn cutoff(&self) -> f64;

    /// Build a fragment template from species names and coordinates.
    /// `species` – e.g. `["H", "C", "O"]`.
    /// `coords`  – positions in Ångström.
    fn build_template(
        &self,
        species: Vec<String>,
        coords: Vec<[f64; 3]>,
    ) -> Result<(DMatrix<f64>, DMatrix<f64>, Vec<u8>, Vec<u8>, Vec<u16>, Vec<Vec<i32>>, Vec<u8>, Vec<f64>)>;
}

/// Computes SCC shifts.  DFTB and xTB provide different implementations.
pub trait CoulombModel: Clone {
    /// Evaluate γ(R, U_i, U_j) for a given distance and species pair.
    fn gamma(&self, r: f64, sp1: u8, sp2: u8) -> f64;

    /// Return the Hubbard U (or effective hardness) for a species code.
    fn u(&self, species: u8) -> f64;
}
