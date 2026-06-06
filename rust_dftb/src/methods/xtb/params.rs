//! Hard-coded GFN1-xTB parameters for select elements.
//!
//! Extracted from tblite's `gfn1.f90` for parity testing.

/// Conversion factor eV → Hartree used in tblite GFN1
pub const EVTOAU: f64 = 3.674932217695e-2;

/// Number of shells per element (Z=1..8 for now)
pub const nshell: &[usize] = &[2, 1, 2, 2, 2, 2, 2, 2];

/// Angular momentum of each shell [element][shell]
pub const ang_shell: &[[usize; 3]] = &[
    [0, 0, 0], // H  : 1s, 2s
    [0, 0, 0], // He : 1s
    [0, 1, 0], // Li : 2s, 2p
    [0, 1, 0], // Be : 2s, 2p
    [0, 1, 0], // B  : 2s, 2p
    [0, 1, 0], // C  : 2s, 2p
    [0, 1, 0], // N  : 2s, 2p
    [0, 1, 0], // O  : 2s, 2p
];

/// Principal quantum number of each shell [element][shell]
pub const principal_qn: &[[usize; 3]] = &[
    [1, 2, 0], // H
    [1, 0, 0], // He
    [2, 2, 0], // Li
    [2, 2, 0], // Be
    [2, 2, 0], // B
    [2, 2, 0], // C
    [2, 2, 0], // N
    [2, 2, 0], // O
];

/// Number of primitives per shell [element][shell]
pub const nprim: &[[usize; 3]] = &[
    [4, 3, 0], // H  : 4 for 1s, 3 for 2s
    [4, 0, 0], // He : 4 for 1s
    [6, 6, 0], // Li : 6 for 2s, 6 for 2p
    [6, 6, 0], // Be
    [6, 6, 0], // B
    [6, 6, 0], // C
    [6, 6, 0], // N
    [6, 6, 0], // O
];

/// Slater exponents zeta [element][shell]
pub const slater_zeta: &[[f64; 3]] = &[
    [1.207940, 1.993207, 0.0], // H
    [2.133698, 0.0, 0.0],      // He
    [0.743881, 0.541917, 0.0], // Li
    [0.876888, 1.104598, 0.0], // Be
    [1.667617, 1.495078, 0.0], // B
    [1.960324, 1.832096, 0.0], // C
    [2.050067, 2.113682, 0.0], // N
    [2.345365, 2.153060, 0.0], // O
];

/// Self-energy (atomic level) in eV, before evtoau scaling [element][shell]
pub const selfenergy_ev: &[[f64; 3]] = &[
    [-10.923452, -2.171902, 0.0], // H
    [-22.121015, 0.0, 0.0],       // He
    [-7.270105, -4.609277, 0.0],  // Li
    [-9.541494, -5.812621, 0.0],  // Be
    [-12.497913, -7.604923, 0.0], // B
    [-13.587210, -10.052785, 0.0],// C
    [-20.058000, -12.889326, 0.0],// N
    [-23.398376, -17.886554, 0.0],// O
];

/// CN-dependence kcn in eV*0.01, before evtoau*0.01 scaling [element][shell]
pub const kcn_raw: &[[f64; 3]] = &[
    [6.5540712, 1.3031412, 0.0], // H
    [13.2726090, 0.0, 0.0],      // He
    [0.0, 0.0, 0.0],             // Li
    [0.0, 0.0, 0.0],             // Be
    [0.0, 0.0, 0.0],             // B
    [8.1523260, -3.0158355, 0.0],// C
    [12.0348000, -3.8667978, 0.0],// N
    [14.0390256, -5.3659662, 0.0],// O
];

/// shpoly coefficients (GFN1 scales raw values by 0.01) [element][l]
pub const shpoly: &[[f64; 3]] = &[
    [0.0, 0.0, 0.0],              // H
    [0.08084149, 0.0, 0.0],       // He
    [-0.04102845, 0.09259276, 0.0], // Li
    [-0.12991482, -0.01308797, 0.0],// Be
    [-0.07088823, 0.00655877, 0.0], // B
    [-0.07082170, 0.00812216, 0.0], // C
    [-0.12745585, -0.01428367, 0.0],// N
    [-0.13729047, -0.04453341, 0.0],// O
];

/// Atomic radii in Bohr (from tblite_data_atomicrad, aatoau scaled)
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
pub const kdiag: &[f64] = &[1.85, 2.25, 2.0, 2.0, 2.0];

/// enscale
pub const ENSCALE: f64 = -7.0e-3;

/// kdiff
pub const KDIFF: f64 = 2.85;

/// kpair for H-H
pub const kpair_hh: f64 = 0.96;

/// Pauling electronegativity (from tblite)
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

/// Covalent radii in Bohr (tblite D3: 4/3 * covalent_rad_2009 * aatoau)
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
    0.470099, // H
    1.441379, // He
    0.205342, // Li
    0.274022, // Be
    0.340530, // B
    0.479988, // C
    0.476106, // N
    0.583349, // O
];

/// Shell Hubbard scaling factors (1.0 + delta) [element][l=0,1,2]
/// From tblite gfn1.f90 shell_hubbard(0:2, max_elem), elements 1-8
pub const shell_hubbard: &[[f64; 3]] = &[
    [1.0, 1.0,       1.0],   // H  (1): s=+0.0, p=+0.0
    [1.0, 1.0,       1.0],   // He (2): s=+0.0, p=+0.0
    [1.0, 0.9227988, 1.0],   // Li (3): s=+0.0, p=-0.0772012
    [1.0, 1.1113005, 1.0],   // Be (4): s=+0.0, p=+0.1113005
    [1.0, 1.0165643, 1.0],   // B  (5): s=+0.0, p=+0.0165643
    [1.0, 0.9528819, 1.0],   // C  (6): s=+0.0, p=-0.0471181
    [1.0, 1.0315090, 1.0],   // N  (7): s=+0.0, p=+0.0315090
    [1.0, 1.0374608, 1.0],   // O  (8): s=+0.0, p=+0.0374608
];

/// Third-order Hubbard derivatives [element]
pub const hubbard_derivs: &[f64] = &[
    0.000000, // H
    0.1500000, // He  (1.500000 * 0.1)
    0.1027370, // Li  (1.027370 * 0.1)
    0.0900554, // Be  (0.900554 * 0.1)
    0.1300000, // B   (1.300000 * 0.1)
    0.1053856, // C   (1.053856 * 0.1)
    0.0042507, // N   (0.042507 * 0.1)
    -0.0005102, // O  (-0.005102 * 0.1)
];

/// Coulomb kernel exponent for GFN1 (Klopman-Ohno)
pub const GEXP: f64 = 2.0;

/// evtoau for GFN1
pub fn ev_to_au(ev: f64) -> f64 {
    ev * EVTOAU
}

/// Self-energy in Hartree for a given element and shell
pub fn selfenergy(elem_idx: usize, shell: usize) -> f64 {
    ev_to_au(selfenergy_ev[elem_idx][shell])
}

/// CN shift coefficient in Hartree for a given element and shell
pub fn kcn(elem_idx: usize, shell: usize) -> f64 {
    ev_to_au(kcn_raw[elem_idx][shell] * 0.01)
}

/// Compute kshell(jl, il) for GFN1
pub fn kshell(jl: usize, il: usize) -> f64 {
    if (jl == 0 && il == 1) || (il == 0 && jl == 1) {
        2.08
    } else {
        (kdiag[il] + kdiag[jl]) / 2.0
    }
}

/// Get pair parameter kpair for two elements
pub fn kpair(izp: usize, jzp: usize) -> f64 {
    if izp == 0 && jzp == 0 {
        // H-H
        0.96
    } else {
        1.0
    }
}
