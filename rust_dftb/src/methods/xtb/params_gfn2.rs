//! Hard-coded GFN2-xTB parameters for select elements.
//!
//! Extracted from tblite's `gfn2.f90` for parity testing.

/// Conversion factor eV → Hartree used in tblite GFN2
pub const EVTOAU: f64 = 3.674932217695e-2;

/// Number of shells per element (Z=1..8 for now)
pub const nshell: &[usize] = &[1, 2, 2, 2, 2, 2, 2, 2];

/// Angular momentum of each shell [element][shell]
pub const ang_shell: &[[usize; 3]] = &[
    [0, 0, 0], // H  : 1s
    [0, 1, 0], // He : 1s, 2p
    [0, 1, 0], // Li : 2s, 2p
    [0, 1, 0], // Be : 2s, 2p
    [0, 1, 0], // B  : 2s, 2p
    [0, 1, 0], // C  : 2s, 2p
    [0, 1, 0], // N  : 2s, 2p
    [0, 1, 0], // O  : 2s, 2p
];

/// Principal quantum number of each shell [element][shell]
pub const principal_qn: &[[usize; 3]] = &[
    [1, 0, 0], // H
    [1, 2, 0], // He
    [2, 2, 0], // Li
    [2, 2, 0], // Be
    [2, 2, 0], // B
    [2, 2, 0], // C
    [2, 2, 0], // N
    [2, 2, 0], // O
];

/// Number of primitives per shell [element][shell]
pub const nprim: &[[usize; 3]] = &[
    [3, 0, 0], // H  : 3 for 1s
    [3, 4, 0], // He : 3 for 1s, 4 for 2p
    [4, 4, 0], // Li
    [4, 4, 0], // Be
    [4, 4, 0], // B
    [4, 4, 0], // C
    [4, 4, 0], // N
    [4, 4, 0], // O
];

/// Slater exponents zeta [element][shell]
pub const slater_zeta: &[[f64; 3]] = &[
    [1.230000, 0.0,      0.0], // H
    [1.669667, 1.500000, 0.0], // He
    [0.750060, 0.557848, 0.0], // Li
    [1.034720, 0.949332, 0.0], // Be
    [1.479444, 1.479805, 0.0], // B
    [2.096432, 1.800000, 0.0], // C
    [2.339881, 2.014332, 0.0], // N
    [2.439742, 2.137023, 0.0], // O
];

/// Self-energy (atomic level) in eV, before evtoau scaling [element][shell]
pub const selfenergy_ev: &[[f64; 3]] = &[
    [-10.707211,  0.000000, 0.0], // H
    [-23.716445, -1.822307, 0.0], // He
    [ -4.900000, -2.217789, 0.0], // Li
    [ -7.743081, -3.133433, 0.0], // Be
    [ -9.224376, -7.419002, 0.0], // B
    [-13.970922,-10.063292, 0.0], // C
    [-16.686243,-12.523956, 0.0], // N
    [-20.229985,-15.503117, 0.0], // O
];

/// CN-dependence kcn in eV, before evtoau scaling [element][ang]
/// GFN2 uses p_kcn(ang, izp) * evtoau directly (no extra 0.01)
pub const kcn_raw: &[[f64; 3]] = &[
    [-0.0500000,  0.0000000, 0.0], // H
    [ 0.2074275,  0.0000000, 0.0], // He
    [ 0.1620836, -0.0623876, 0.0], // Li
    [ 0.1187759,  0.0550528, 0.0], // Be
    [ 0.0120462, -0.0141086, 0.0], // B
    [-0.0102144,  0.0161657, 0.0], // C
    [-0.1955336,  0.0561076, 0.0], // N
    [ 0.0117826, -0.0145102, 0.0], // O
];

/// shpoly coefficients (GFN2 scales raw values by 0.01) [element][ang]
pub const shpoly: &[[f64; 3]] = &[
    [-0.953618 * 0.01,  0.000000 * 0.01, 0.0],              // H
    [-4.386816 * 0.01,  0.710647 * 0.01, 0.0],              // He
    [-4.750398 * 0.01, 20.424920 * 0.01, 0.0],               // Li
    [-7.910394 * 0.01, -0.476438 * 0.01, 0.0],              // Be
    [-5.183150 * 0.01, -2.453322 * 0.01, 0.0],              // B
    [-2.294321 * 0.01, -0.271102 * 0.01, 0.0],              // C
    [-8.506003 * 0.01, -2.504201 * 0.01, 0.0],              // N
    [-14.955291 * 0.01, -3.350819 * 0.01, 0.0],             // O
];

/// Atomic radii in Bohr (same source as GFN1)
pub const atomic_rad: &[f64] = &[
    0.32 * 1.889726133, // H
    0.37 * 1.889726133, // He
    1.30 * 1.889726133, // Li
    0.99 * 1.889726133, // Be
    0.84 * 1.889726133, // B
    0.75 * 1.889726133, // C
    0.71 * 1.889726133, // N
    0.64 * 1.889726133, // O
];

/// kdiag values for kshell computation
pub const kdiag: &[f64] = &[1.85, 2.23, 2.23, 2.23, 2.23];

/// enscale
pub const ENSCALE: f64 = 2.0e-2;

/// wexp for Slater overlap factor
pub const WEXP: f64 = 0.5;

/// Pauling electronegativity (same as GFN1)
pub const pauling_en: &[f64] = &[
    2.20, // H
    3.00, // He
    0.98, // Li
    1.57, // Be
    2.04, // B
    2.55, // C
    3.04, // N
    3.44, // O
];

/// Covalent radii in Bohr (same D3 source as GFN1)
pub const cov_rad: &[f64] = &[
    (4.0/3.0) * 0.32 * 1.889726133, // H
    (4.0/3.0) * 0.46 * 1.889726133, // He
    (4.0/3.0) * 1.20 * 1.889726133, // Li
    (4.0/3.0) * 0.94 * 1.889726133, // Be
    (4.0/3.0) * 0.77 * 1.889726133, // B
    (4.0/3.0) * 0.75 * 1.889726133, // C
    (4.0/3.0) * 0.71 * 1.889726133, // N
    (4.0/3.0) * 0.63 * 1.889726133, // O
];

/// Hubbard parameters (atomic hardnesses) for second-order electrostatics [element]
pub const hubbard_parameter: &[f64] = &[
    0.405771, // H
    0.642029, // He
    0.245006, // Li
    0.684789, // Be
    0.513556, // B
    0.538015, // C
    0.461493, // N
    0.451896, // O
];

/// Shell Hubbard scaling factors (1.0 + delta) [element][l=0,1,2]
pub const shell_hubbard: &[[f64; 3]] = &[
    [1.0, 1.0,         1.0], // H
    [1.0, 1.0,         1.0], // He
    [1.0, 1.1972612,   1.0], // Li
    [1.0, 1.9658467,   1.0], // Be
    [1.0, 1.3994080,   1.0], // B
    [1.0, 1.1056358,   1.0], // C
    [1.0, 1.1164892,   1.0], // N
    [1.0, 1.1497020,   1.0], // O
];

/// Third-order Hubbard derivatives (element-wise, already 0.1-scaled in Fortran) [element]
pub const hubbard_derivs: &[f64] = &[
    0.080000,  // H
    0.200000,  // He
    0.1303821, // Li
    0.0574239, // Be
    0.0946104, // B
    0.150000,  // C
    -0.0639780,// N
    -0.0517134,// O
];

/// Shell-resolved angular scaling of Hubbard derivatives [l]
pub const shell_hubbard_derivs: &[f64] = &[1.0, 0.5, 0.25, 0.25, 0.25];

/// Coulomb kernel exponent for GFN2 (Klopman-Ohno)
pub const GEXP: f64 = 2.0;

/// Reference occupation numbers by angular momentum [element][l]
pub const reference_occ: &[[f64; 3]] = &[
    [1.0, 0.0, 0.0], // H
    [2.0, 0.0, 0.0], // He
    [1.0, 0.0, 0.0], // Li
    [2.0, 0.0, 0.0], // Be
    [2.0, 1.0, 0.0], // B
    [1.0, 3.0, 0.0], // C
    [1.5, 3.5, 0.0], // N
    [2.0, 4.0, 0.0], // O
];

/// evtoau for GFN2
pub fn ev_to_au(ev: f64) -> f64 {
    ev * EVTOAU
}

/// Self-energy in Hartree for a given element and shell
pub fn selfenergy(elem_idx: usize, shell: usize) -> f64 {
    ev_to_au(selfenergy_ev[elem_idx][shell])
}

/// CN shift coefficient in Hartree for a given element and angular momentum
/// GFN2: no extra 0.01 scaling
pub fn kcn(elem_idx: usize, ang: usize) -> f64 {
    ev_to_au(kcn_raw[elem_idx][ang])
}

/// Compute kshell(jl, il) for GFN2
/// If one shell is d (l=2) and other is s/p (l=0,1), return 2.0
/// Otherwise return average of kdiag
pub fn kshell(jl: usize, il: usize) -> f64 {
    let is_d = |l: usize| l == 2;
    let is_sp = |l: usize| l == 0 || l == 1;
    if (is_d(jl) && is_sp(il)) || (is_d(il) && is_sp(jl)) {
        2.0
    } else {
        (kdiag[il] + kdiag[jl]) / 2.0
    }
}

/// Get pair parameter kpair for two elements (always 1.0 in GFN2)
pub fn kpair(_izp: usize, _jzp: usize) -> f64 {
    1.0
}

/// Multipole damping parameters
pub const MP_DMP3: f64 = 3.0;
pub const MP_DMP5: f64 = 4.0;
pub const MP_SHIFT: f64 = 1.2;
pub const MP_KEXP: f64 = 4.0;
pub const MP_RMAX: f64 = 5.0;

/// Dipole exchange-correlation kernel [element] (scaled by 0.01)
pub const dkernel: &[f64] = &[
    0.01 * 5.563889,  0.01 * -1.000000, 0.01 * -0.500000, 0.01 * -0.613341,
    0.01 * -0.481186, 0.01 * -0.411674, 0.01 * 3.521273,  0.01 * -4.935670,
];

/// Quadrupole exchange-correlation kernel [element] (scaled by 0.01)
pub const qkernel: &[f64] = &[
    0.01 * 0.027431, 0.01 * -0.337528, 0.01 * 0.020000, 0.01 * -0.058586,
    0.01 * -0.058228, 0.01 * 0.213583, 0.01 * 2.026786, 0.01 * -0.310828,
];

/// Valence coordination numbers for radii [element]
pub const vcn: &[f64] = &[
    1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 2.0,
];

/// Cutoff radii for multipole electrostatics [element]
pub const rad: &[f64] = &[
    1.4, 3.0, 5.0, 5.0, 5.0, 3.0, 1.9, 1.8,
];

