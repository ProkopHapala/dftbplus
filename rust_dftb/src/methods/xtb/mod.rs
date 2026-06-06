//! Analytical xTB Hamiltonian and parameters (GFN1/GFN2).
//!
//! This module is a placeholder for future xTB implementation.
//! It will contain:
//! - `params.rs` – static parameter tables from tblite
//! - `basis.rs`  – CGTO construction from STO exponents
//! - `integrals.rs` – analytical Cartesian Gaussian overlap / multipole
//! - `hamiltonian.rs` – `XtbBuilder` implementing `crate::methods::traits::H0Builder`
//! - `coulomb.rs` – `CoulombModel` for GFN1 (wraps gamma) and GFN2 (Klopman–Ohno)
