//! Intra-fragment SCC shift builder.
//!
//! Computes the electrostatic potential on each atom of a single fragment
//! arising from the other atoms *within the same fragment*:
//!
//! `V_intra(A) = Σ_{B≠A} γ(R_AB, U_A, U_B) · (q_B - q0_B)`
//!
//! For the self term (A=A) the contribution is `U_A · (q_A - q0_A)`.
//!
//! This module operates only on a single fragment; the inter-fragment coupling
//! is handled by `MultiSystemSolver::compute_v_ext`.

use crate::qmqm::gamma::GammaTable;

const ANG2BOHR: f64 = 1.889_726_133;

/// Compute intra-fragment shifts for all atoms in a fragment.
///
/// `coords`  – atom coordinates, length `n_atoms`.
/// `species` – species code per atom, length `n_atoms`.
/// `delta_q` – charge deviation `q - q0` per atom, length `n_atoms`.
/// `gamma`   – lookup table for Hubbard U values.
/// `out`     – pre-allocated shift vector, length `n_atoms` (zeroed on entry).
pub fn compute_intra_shifts(
    coords: &[[f64; 3]],
    species: &[u8],
    delta_q: &[f64],
    gamma_tbl: &GammaTable,
    out: &mut [f64],
) {
    let n = coords.len();
    assert_eq!(species.len(), n);
    assert_eq!(delta_q.len(), n);
    assert_eq!(out.len(), n);

    out.fill(0.0);

    for i in 0..n {
        let ui = gamma_tbl.u(species[i]);
        // Self term: U_i * delta_q_i
        // (our deltaQ = q_elec - q0, opposite to DFTB+'s deltaQ = q0 - q_elec,
        //  so the -gamma*deltaQ_DFTB+ becomes +gamma*deltaQ_ours)
        out[i] += ui * delta_q[i];

        for j in (i + 1)..n {
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt() * ANG2BOHR;
            let g = gamma_tbl.gamma(r, species[i], species[j]);

            out[i] += g * delta_q[j];
            out[j] += g * delta_q[i];
        }
    }
}

/// Same as `compute_intra_shifts` but only for a specific atom `iat`.
/// Useful when only one atom's shift is needed (rare).
pub fn compute_intra_shift_atom(
    iat: usize,
    coords: &[[f64; 3]],
    species: &[u8],
    delta_q: &[f64],
    gamma_tbl: &GammaTable,
) -> f64 {
    let n = coords.len();
    assert!(iat < n);

    let ui = gamma_tbl.u(species[iat]);
    let mut shift = ui * delta_q[iat];

    for j in 0..n {
        if j == iat {
            continue;
        }
        let dx = coords[iat][0] - coords[j][0];
        let dy = coords[iat][1] - coords[j][1];
        let dz = coords[iat][2] - coords[j][2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt() * ANG2BOHR;
        let g = gamma_tbl.gamma(r, species[iat], species[j]);
        shift += g * delta_q[j];
    }

    shift
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qmqm::gamma::gamma_full;

    #[test]
    fn intra_shift_self_only() {
        let coords = vec![[0.0, 0.0, 0.0]];
        let species = vec![0u8];
        let delta_q = vec![0.2];
        let gamma = GammaTable::from_hubbard_u(vec![0.5]);
        let mut out = vec![0.0];
        compute_intra_shifts(&coords, &species, &delta_q, &gamma, &mut out);
        assert!((out[0] - 0.1).abs() < 1e-12, "self shift = U·dq = 0.5·0.2 = 0.1, got {}", out[0]);
    }

    #[test]
    fn intra_shift_two_atoms() {
        // 1.0 Bohr in Ångström so that after ANG2BOHR conversion gamma sees r=1.0
        let coords = vec![[0.0, 0.0, 0.0], [1.0 / ANG2BOHR, 0.0, 0.0]];
        let species = vec![0u8, 0u8];
        let delta_q = vec![0.2, -0.1];
        let gamma = GammaTable::from_hubbard_u(vec![0.5]);
        let mut out = vec![0.0; 2];
        compute_intra_shifts(&coords, &species, &delta_q, &gamma, &mut out);

        // Self + cross
        // gamma(1.0, 0.5, 0.5) should be between 0.5 (onsite) and 1.0 (Coulomb)
        let g = gamma_full(1.0, 0.5, 0.5);
        let expected_0 = 0.5 * 0.2 + g * (-0.1);
        let expected_1 = 0.5 * (-0.1) + g * 0.2;
        assert!((out[0] - expected_0).abs() < 1e-10);
        assert!((out[1] - expected_1).abs() < 1e-10);
    }
}
