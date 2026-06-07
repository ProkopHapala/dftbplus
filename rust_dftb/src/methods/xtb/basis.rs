//! CGTO basis construction from Slater exponents using Stewart coefficients.

use crate::methods::xtb::params;
use crate::methods::xtb::params_gfn2;

/// Contracted Gaussian-type orbital
#[derive(Debug, Clone)]
pub struct Cgto {
    pub ang: usize,       // angular momentum l (0=s, 1=p, 2=d)
    pub nprim: usize,
    pub alpha: Vec<f64>, // primitive exponents
    pub coeff: Vec<f64>, // contraction coefficients (with normalization)
}

impl Cgto {
    pub fn new(ang: usize, nprim: usize) -> Self {
        Self {
            ang,
            nprim,
            alpha: vec![0.0; nprim],
            coeff: vec![0.0; nprim],
        }
    }
}

/// Build CGTOs for a single element from GFN1 parameters
pub fn build_element_cgtos(elem_idx: usize) -> Vec<Cgto> {
    build_element_cgtos_generic(
        elem_idx,
        &params::nshell,
        &params::ang_shell,
        &params::principal_qn,
        &params::nprim,
        &params::slater_zeta,
        true,
    )
}

/// Build CGTOs for a single element from GFN2 parameters (no orthogonalization)
pub fn build_element_cgtos_gfn2(elem_idx: usize) -> Vec<Cgto> {
    build_element_cgtos_generic(
        elem_idx,
        &params_gfn2::nshell,
        &params_gfn2::ang_shell,
        &params_gfn2::principal_qn,
        &params_gfn2::nprim,
        &params_gfn2::slater_zeta,
        false,
    )
}

/// Generic CGTO builder
pub fn build_element_cgtos_generic(
    elem_idx: usize,
    nshell_arr: &[usize],
    ang_shell_arr: &[[usize; 3]],
    principal_qn_arr: &[[usize; 3]],
    nprim_arr: &[[usize; 3]],
    slater_zeta_arr: &[[f64; 3]],
    orthogonalize: bool,
) -> Vec<Cgto> {
    let nsh = nshell_arr[elem_idx];
    let mut cgtos = Vec::with_capacity(nsh);
    let mut ang_idx = [0usize; 3]; // track first shell of each angular momentum

    // First pass: build raw CGTOs from Slater exponents
    for ish in 0..nsh {
        let l = ang_shell_arr[elem_idx][ish];
        let ng = nprim_arr[elem_idx][ish];
        let n = principal_qn_arr[elem_idx][ish];
        let zeta = slater_zeta_arr[elem_idx][ish];

        let mut cgto = slater_to_gauss(ng, n, l, zeta, true);

        if orthogonalize && ang_idx[l] > 0 {
            // Orthogonalize against previous shell with same angular momentum
            let prev = ang_idx[l] - 1;
            orthogonalize_cgto(&cgtos[prev], &mut cgto);
        } else {
            ang_idx[l] = ish + 1;
        }
        cgtos.push(cgto);
    }
    cgtos
}

/// Convert Slater function to contracted Gaussian
fn slater_to_gauss(ng: usize, n: usize, l: usize, zeta: f64, norm: bool) -> Cgto {
    let mut cgto = Cgto::new(l, ng);

    // ityp mapping: l=0(s): n, l=1(p): 4+n, l=2(d): 7+n
    let ityp = match l {
        0 => n,
        1 => 4 + n,
        2 => 7 + n,
        _ => panic!("Unsupported angular momentum {l}"),
    };

    let z2 = zeta * zeta;

    match ng {
        1 => {
            cgto.alpha[0] = pAlpha1(ityp) * z2;
            cgto.coeff[0] = 1.0;
        }
        2 => {
            let (a, c) = pAlpha2_coeff(ityp);
            for i in 0..2 {
                cgto.alpha[i] = a[i] * z2;
                cgto.coeff[i] = c[i];
            }
        }
        3 => {
            let (a, c) = pAlpha3_coeff(ityp);
            for i in 0..3 {
                cgto.alpha[i] = a[i] * z2;
                cgto.coeff[i] = c[i];
            }
        }
        4 => {
            let (a, c) = pAlpha4_coeff(ityp);
            for i in 0..4 {
                cgto.alpha[i] = a[i] * z2;
                cgto.coeff[i] = c[i];
            }
        }
        6 => {
            let (a, c) = pAlpha6_coeff(ityp);
            for i in 0..6 {
                cgto.alpha[i] = a[i] * z2;
                cgto.coeff[i] = c[i];
            }
        }
        _ => panic!("Unsupported number of primitives {ng}"),
    }

    if norm {
        let top = 2.0 / std::f64::consts::PI;
        let dfact = double_factorial(l);
        for i in 0..ng {
            let ai = cgto.alpha[i];
            let norm_factor = (top * ai).powf(0.75) * (4.0 * ai).powf(l as f64 / 2.0) / dfact.sqrt();
            cgto.coeff[i] *= norm_factor;
        }
    }

    cgto
}

/// Gram-Schmidt orthogonalization of cgto_b against cgto_a (same angular momentum)
/// Follows tblite's approach: append primitives from a to b with scaled coefficients
fn orthogonalize_cgto(cgto_a: &Cgto, cgto_b: &mut Cgto) {
    if cgto_a.ang != cgto_b.ang { return; }

    // Compute overlap S_ab = <a|b>
    let mut s_ab = 0.0;
    for i in 0..cgto_a.nprim {
        for j in 0..cgto_b.nprim {
            let a = cgto_a.alpha[i];
            let b = cgto_b.alpha[j];
            let c = cgto_a.coeff[i] * cgto_b.coeff[j];
            s_ab += c * primitive_overlap_s_s(a, b, 0.0);
        }
    }

    // Append primitives from a to b: |b'> = |b> - S_ab * |a>
    let old_nprim = cgto_b.nprim;
    cgto_b.nprim += cgto_a.nprim;
    cgto_b.alpha.resize(cgto_b.nprim, 0.0);
    cgto_b.coeff.resize(cgto_b.nprim, 0.0);
    for i in 0..cgto_a.nprim {
        cgto_b.alpha[old_nprim + i] = cgto_a.alpha[i];
        cgto_b.coeff[old_nprim + i] = -s_ab * cgto_a.coeff[i];
    }

    // Renormalize
    let mut norm = 0.0;
    for j in 0..cgto_b.nprim {
        for k in 0..cgto_b.nprim {
            norm += cgto_b.coeff[j] * cgto_b.coeff[k]
                  * primitive_overlap_s_s(cgto_b.alpha[j], cgto_b.alpha[k], 0.0);
        }
    }
    let norm_fact = 1.0 / norm.sqrt();
    for j in 0..cgto_b.nprim {
        cgto_b.coeff[j] *= norm_fact;
    }
}

/// Primitive s-s overlap at distance r
fn primitive_overlap_s_s(alpha: f64, beta: f64, r: f64) -> f64 {
    let gamma = alpha + beta;
    let pref = (std::f64::consts::PI / gamma).powf(1.5);
    let exp_term = (-alpha * beta * r * r / gamma).exp();
    pref * exp_term
}

fn double_factorial(n: usize) -> f64 {
    match n {
        0 => 1.0,
        1 => 1.0,
        2 => 1.0,
        3 => 3.0,
        4 => 15.0,
        5 => 105.0,
        6 => 945.0,
        _ => {
            let mut result = 1.0;
            for i in (1..=n).step_by(2) {
                result *= i as f64;
            }
            result
        }
    }
}

// Stewart STO-NG coefficients from tblite/basis/slater.f90

fn pAlpha1(ityp: usize) -> f64 {
    let table = [
        2.709498091e-1, 1.012151084e-1, 5.296881757e-2, 3.264600274e-2,
        2.216912938e-2, 1.759666885e-1, 9.113614253e-2, 5.578350235e-2,
        3.769845216e-2, 1.302270363e-1, 7.941656339e-2, 5.352200793e-2,
        1.033434062e-1, 6.952785407e-2, 8.565417784e-2,
    ];
    table[ityp - 1]
}

fn pAlpha2_coeff(ityp: usize) -> ([f64; 2], [f64; 2]) {
    let alpha = [
        [8.518186635e-1, 1.516232927e-1],
        [1.292278611e-1, 4.908584205e-2],
        [6.694095822e-1, 5.837135094e-2],
        [2.441785453e-1, 4.051097664e-2],
        [1.213425654e-1, 3.133152144e-2],
        [4.323908358e-1, 1.069439065e-1],
        [1.458620964e-1, 5.664210742e-2],
        [6.190052680e-2, 2.648418407e-2],
        [2.691294191e-1, 3.980805011e-2],
        [2.777427345e-1, 8.336507714e-2],
        [1.330958892e-1, 5.272119659e-2],
        [6.906014388e-2, 3.399457777e-2],
        [2.006693538e-1, 6.865384900e-2],
        [1.156094555e-1, 4.778940916e-2],
        [1.554531559e-1, 5.854079811e-2],
    ];
    let coeff = [
        [4.301284983e-1, 6.789135305e-1],
        [7.470867124e-1, 2.855980556e-1],
        [-1.529645716e-1, 1.051370110e+0],
        [-3.046656896e-1, 1.146877294e+0],
        [-5.114756049e-1, 1.307377277e+0],
        [4.522627513e-1, 6.713122642e-1],
        [5.349653144e-1, 5.299607212e-1],
        [8.743116767e-1, 1.513640107e-1],
        [-1.034227010e-1, 1.033376378e+0],
        [4.666137923e-1, 6.644706516e-1],
        [4.932764167e-1, 5.918727866e-1],
        [6.539405185e-1, 3.948945302e-1],
        [4.769346276e-1, 6.587383976e-1],
        [4.856637346e-1, 6.125980914e-1],
        [4.848298074e-1, 6.539381621e-1],
    ];
    (alpha[ityp - 1], coeff[ityp - 1])
}

fn pAlpha3_coeff(ityp: usize) -> ([f64; 3], [f64; 3]) {
    let alpha = [
        [2.227660584e+0, 4.057711562e-1, 1.098175104e-1],
        [2.581578398e+0, 1.567622104e-1, 6.018332272e-2],
        [5.641487709e-1, 6.924421391e-2, 3.269529097e-2],
        [2.267938753e-1, 4.448178019e-2, 2.195294664e-2],
        [1.080198458e-1, 4.408119382e-2, 2.610811810e-2],
        [9.192379002e-1, 2.359194503e-1, 8.009805746e-2],
        [2.692880368e+0, 1.489359592e-1, 5.739585040e-2],
        [4.859692220e-1, 7.430216918e-2, 3.653340923e-2],
        [2.389722618e+0, 3.499121109e-1, 1.683175469e-1],
        [1.410128298e+0, 5.077878915e-1, 1.847926858e-1],
        [5.868285913e+0, 1.530329631e+0, 5.475665231e-1],
        [2.488296923e+0, 7.981487853e-1, 3.311327490e-1],
        [4.634239420e+0, 1.341648295e+0, 2.209593028e-1],
        [8.820520428e-1, 3.410838409e-1, 9.204308840e-2],
        [1.357718039e+0, 5.004907278e-1, 2.296565064e-1],
    ];
    let coeff = [
        [1.543289673e-1, 5.353281423e-1, 4.446345422e-1],
        [-5.994474934e-2, 5.960385398e-1, 4.581786291e-1],
        [-1.782577972e-1, 8.612761663e-1, 2.261841969e-1],
        [-3.349048323e-1, 1.056744667e+0, 1.256661680e-1],
        [-6.617401158e-1, 7.467595004e-1, 7.146490945e-1],
        [1.623948553e-1, 5.661708862e-1, 4.223071752e-1],
        [-1.061945788e-2, 5.218564264e-1, 5.450015143e-1],
        [-6.147823411e-2, 6.604172234e-1, 3.932639495e-1],
        [-1.389529695e-1, 8.076691064e-1, 2.726029342e-1],
        [1.686596060e-1, 5.847984817e-1, 4.056779523e-1],
        [2.308552718e-1, 6.042409177e-1, 2.595768926e-1],
        [-2.010175008e-2, 5.899370608e-1, 4.658445960e-1],
        [1.737856685e-1, 5.973380628e-1, 3.929395614e-1],
        [1.909729355e-1, 6.146060459e-1, 3.059611271e-1],
        [1.780980905e-1, 6.063757846e-1, 3.828552923e-1],
    ];
    (alpha[ityp - 1], coeff[ityp - 1])
}

fn pAlpha4_coeff(ityp: usize) -> ([f64; 4], [f64; 4]) {
    let alpha = [
        [5.216844534e+0, 9.546182760e-1, 2.652034102e-1, 8.801862774e-2],
        [1.161525551e+1, 2.000243111e+0, 1.607280687e-1, 6.125744532e-2],
        [1.513265591e+0, 4.262497508e-1, 7.643320863e-2, 3.760545063e-2],
        [3.242212833e-1, 1.663217177e-1, 5.081097451e-2, 2.829066600e-2],
        [8.602284252e-1, 1.189050200e-1, 3.446076176e-2, 1.974798796e-2],
        [1.798260992e+0, 4.662622228e-1, 1.643718620e-1, 6.543927065e-2],
        [1.853180239e+0, 1.915075719e-1, 8.655487938e-2, 4.184253862e-2],
        [1.492607880e+0, 4.327619272e-1, 7.553156064e-2, 3.706272183e-2],
        [3.962838833e-1, 1.838858552e-1, 4.943555157e-2, 2.750222273e-2],
        [9.185846715e-1, 2.920461109e-1, 1.187568890e-1, 5.286755896e-2],
        [1.995825422e+0, 1.823461280e-1, 8.197240896e-2, 4.000634951e-2],
        [4.230617826e-1, 8.293863702e-2, 4.590326388e-2, 2.628744797e-2],
        [5.691670217e-1, 2.074585819e-1, 9.298346885e-2, 4.473508853e-2],
        [2.017831152e-1, 1.001952178e-1, 5.447006630e-2, 3.037569283e-2],
        [3.945205573e-1, 1.588100623e-1, 7.646521729e-2, 3.898703611e-2],
    ];
    let coeff = [
        [5.675242080e-2, 2.601413550e-1, 5.328461143e-1, 2.916254405e-1],
        [-1.198411747e-2, -5.472052539e-2, 5.805587176e-1, 4.770079976e-1],
        [-3.295496352e-2, -1.724516959e-1, 7.518511194e-1, 3.589627317e-1],
        [-1.120682822e-1, -2.845426863e-1, 8.909873788e-1, 3.517811205e-1],
        [1.103657561e-2, -5.606519023e-1, 1.179429987e+0, 1.734974376e-1],
        [5.713170255e-2, 2.857455515e-1, 5.517873105e-1, 2.632314924e-1],
        [-1.434249391e-2, 2.755177589e-1, 5.846750879e-1, 2.144986514e-1],
        [-6.035216774e-3, -6.013310874e-2, 6.451518200e-1, 4.117923820e-1],
        [-1.801459207e-2, -1.360777372e-1, 7.533973719e-1, 3.409304859e-1],
        [5.799057705e-2, 3.045581349e-1, 5.601358038e-1, 2.432423313e-1],
        [-2.816702620e-3, 2.177095871e-1, 6.058047348e-1, 2.717811257e-1],
        [-2.421626009e-2, 3.937644956e-1, 5.489520286e-1, 1.190436963e-1],
        [5.902730589e-2, 3.191828952e-1, 5.639423893e-1, 2.284796537e-1],
        [9.174268830e-2, 4.023496947e-1, 4.937432100e-1, 1.254001522e-1],
        [6.010484250e-2, 3.309738329e-1, 5.655207585e-1, 2.171122608e-1],
    ];
    (alpha[ityp - 1], coeff[ityp - 1])
}

fn pAlpha6_coeff(ityp: usize) -> ([f64; 6], [f64; 6]) {
    let alpha = [
        [2.310303149e+1, 4.235915534e+0, 1.185056519e+0, 4.070988982e-1, 1.580884151e-1, 6.510953954e-2],
        [2.768496241e+1, 5.077140627e+0, 1.426786050e+0, 2.040335729e-1, 9.260298399e-2, 4.416183978e-2],
        [3.273031938e+0, 9.200611311e-1, 3.593349765e-1, 8.636686991e-2, 4.797373812e-2, 2.724741144e-2],
        [3.232838646e+0, 3.605788802e-1, 1.717905487e-1, 5.277666487e-2, 3.163400284e-2, 1.874093091e-2],
        [1.410128298e+0, 5.077878915e-1, 1.847926858e-1, 1.061070594e-1, 3.669584901e-2, 2.213558430e-2],
        [5.868285913e+0, 1.530329631e+0, 5.475665231e-1, 2.288932733e-1, 1.046655969e-1, 4.948220127e-2],
        [5.077973607e+0, 1.340786940e+0, 2.248434849e-1, 1.131741848e-1, 6.076408893e-2, 3.315424265e-2],
        [2.389722618e+0, 7.960947826e-1, 3.415541380e-1, 8.847434525e-2, 4.958248334e-2, 2.816929784e-2],
        [3.778623374e+0, 3.499121109e-1, 1.683175469e-1, 5.404070736e-2, 3.328911801e-2, 2.063815019e-2],
        [2.488296923e+0, 7.981487853e-1, 3.311327490e-1, 1.559114463e-1, 7.877734732e-2, 4.058484363e-2],
        [4.634239420e+0, 1.341648295e+0, 2.209593028e-1, 1.101467943e-1, 5.904190370e-2, 3.232628887e-2],
        [8.820520428e-1, 3.410838409e-1, 9.204308840e-2, 5.472831774e-2, 3.391202830e-2, 2.108227374e-2],
        [1.357718039e+0, 5.004907278e-1, 2.296565064e-1, 1.173146814e-1, 6.350097171e-2, 3.474556673e-2],
        [1.334096840e+0, 2.372312347e-1, 1.269485744e-1, 7.290318381e-2, 4.351355997e-2, 2.598071843e-2],
        [8.574668996e-1, 3.497184772e-1, 1.727917060e-1, 9.373643151e-2, 5.340032759e-2, 3.057364464e-2],
    ];
    let coeff = [
        [9.163596280e-3, 4.936149294e-2, 1.685383049e-1, 3.705627997e-1, 4.164915298e-1, 1.303340841e-1],
        [-4.151277819e-3, -2.067024148e-2, -5.150303337e-2, 3.346271174e-1, 5.621061301e-1, 1.712994697e-1],
        [-6.775596947e-3, -5.639325779e-2, -1.587856086e-1, 5.534527651e-1, 5.015351020e-1, 7.223633674e-2],
        [1.374817488e-3, -8.666390043e-2, -3.130627309e-1, 7.812787397e-1, 4.389247988e-1, 2.487178756e-2],
        [2.695439582e-3, 1.850157487e-2, -9.588628125e-2, -5.200673560e-1, 1.087619490e+0, 3.103964343e-1],
        [7.924233646e-3, 5.144104825e-2, 1.898400060e-1, 4.049863191e-1, 4.012362861e-1, 1.051855189e-1],
        [-3.329929840e-3, -1.419488340e-2, 1.639395770e-1, 4.485358256e-1, 3.908813050e-1, 7.411456232e-2],
        [-1.665913575e-3, -1.657464971e-2, -5.958513378e-2, 4.053115554e-1, 5.433958189e-1, 1.204970491e-1],
        [1.163246387e-4, -2.920771322e-2, -1.381051233e-1, 5.706134877e-1, 4.768808140e-1, 6.021665516e-2],
        [2.020869128e-2, 1.321157923e-1, 3.911240346e-1, 4.779609701e-1, 1.463662294e-1, 1.463662294e-1],
        [-3.673711876e-3, 1.167122499e-1, 4.216476416e-1, 4.547673415e-1, 1.037803318e-1, 1.037803318e-1],
        [-3.231527611e-3, -2.434931372e-2, 3.440817054e-1, 5.693674376e-1, 1.511340183e-1, 1.511340183e-1],
        [1.999839052e-2, 1.395427440e-1, 4.091508237e-1, 4.708252119e-1, 1.328082566e-1, 1.328082566e-1],
        [-7.301193568e-4, 8.414991343e-2, 3.923683153e-1, 5.040033146e-1, 1.328979300e-1, 1.328979300e-1],
        [1.998085812e-2, 1.460384050e-1, 4.230565459e-1, 4.635699665e-1, 1.226411691e-1, 1.226411691e-1],
    ];
    (alpha[ityp - 1], coeff[ityp - 1])
}
