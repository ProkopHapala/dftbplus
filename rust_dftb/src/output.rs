use crate::core::error::{DftbError, Result};
use nalgebra::DMatrix;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    DftbSquare,
}

pub struct DftbOutput;

impl DftbOutput {
    /// Read DFTB+ `hamsqr*.dat` / `oversqr.dat` format.
    /// Format: header comments, then "T  nOrb  nKpoint", then kpoint index, then "# MATRIX", then data.
    pub fn read_square(path: &str) -> Result<DMatrix<f64>> {
        let content = std::fs::read_to_string(path)?;
        
        // Extract all numeric values, plus detect the dimension line
        let mut values = Vec::new();
        let mut dim_line_found = false;
        let mut n_orb = 0;
        
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            
            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            
            // Check if this is the dimension line: "T  14  1" or similar
            // It has 3 tokens: logical (T/F), nOrb, nKpoint
            if tokens.len() == 3 && !dim_line_found {
                if let (Ok(n), Ok(_k)) = (tokens[1].parse::<usize>(), tokens[2].parse::<usize>()) {
                    n_orb = n;
                    dim_line_found = true;
                    continue; // Don't add dimension values to data
                }
            }
            
            // Skip single-number lines (kpoint index)
            if tokens.len() == 1 && tokens[0].parse::<usize>().is_ok() {
                continue;
            }
            
            // Parse all numeric tokens as data
            for token in tokens {
                if let Ok(val) = token.parse::<f64>() {
                    values.push(val);
                }
            }
        }
        
        if !dim_line_found {
            return Err(DftbError::Parse("could not find matrix dimension line".into()));
        }
        
        let expected = n_orb * n_orb;
        if values.len() < expected {
            return Err(DftbError::Parse(format!(
                "not enough data: got {} values, expected {} for {}x{} matrix",
                values.len(), expected, n_orb, n_orb
            )));
        }
        
        // Take only the expected number of values (ignore any trailing)
        let data: Vec<f64> = values[..expected].to_vec();
        Ok(DMatrix::from_row_slice(n_orb, n_orb, &data))
    }

    pub fn write_square(path: &str, mat: &DMatrix<f64>) -> Result<()> {
        if mat.nrows() != mat.ncols() {
            return Err(DftbError::InvalidInput("matrix must be square".into()));
        }
        let n = mat.nrows();
        let mut f = File::create(path)?;
        writeln!(f, "{} {}", n, n)?;
        for i in 0..n {
            for j in 0..n {
                if j + 1 == n {
                    write!(f, "{:.16e}", mat[(i, j)])?;
                } else {
                    write!(f, "{:.16e} ", mat[(i, j)])?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
