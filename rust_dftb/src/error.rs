//! Error types for DFTB implementation

use thiserror::Error;

pub type Result<T> = std::result::Result<T, DftbError>;

#[derive(Error, Debug)]
pub enum DftbError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Invalid SK file format: {0}")]
    InvalidSkFormat(String),

    #[error("Interpolation error: {0}")]
    Interpolation(String),

    #[error("Rotation error: {0}")]
    Rotation(String),

    #[error("Hamiltonian assembly error: {0}")]
    Hamiltonian(String),

    #[error("Index out of bounds: {0}")]
    IndexOutOfBounds(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}
