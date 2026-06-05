use crate::error::{DftbError, Result};
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
    pub fn read_square(path: &str) -> Result<DMatrix<f64>> {
        let f = File::open(path)?;
        let mut it = BufReader::new(f).lines();
        let header = it
            .next()
            .ok_or_else(|| DftbError::Parse("empty matrix file".into()))??;
        let mut h = header.split_whitespace();
        let n: usize = h
            .next()
            .ok_or_else(|| DftbError::Parse("missing n".into()))?
            .parse()
            .map_err(|e: std::num::ParseIntError| DftbError::Parse(e.to_string()))?;
        let m: usize = h
            .next()
            .ok_or_else(|| DftbError::Parse("missing m".into()))?
            .parse()
            .map_err(|e: std::num::ParseIntError| DftbError::Parse(e.to_string()))?;
        if n != m {
            return Err(DftbError::Parse("expected square matrix".into()));
        }

        let mut data = Vec::with_capacity(n * n);
        for _ in 0..n {
            let line = it
                .next()
                .ok_or_else(|| DftbError::Parse("unexpected EOF".into()))??;
            let row: Vec<f64> = line
                .split_whitespace()
                .map(|x| x.parse::<f64>().map_err(|e: std::num::ParseFloatError| DftbError::Parse(e.to_string())))
                .collect::<Result<Vec<f64>>>()?;
            if row.len() != n {
                return Err(DftbError::Parse("row length mismatch".into()));
            }
            data.extend_from_slice(&row);
        }

        Ok(DMatrix::from_row_slice(n, n, &data))
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
