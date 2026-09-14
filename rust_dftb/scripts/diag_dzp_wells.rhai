// Isolate the 0.56 kcal corner offset on dzp_relaxed_mid:
// rep0: protons at relaxed DONOR positions (dzp_relaxed H20,H41)
// rep1: protons on-axis d=1.055 (grid donor point)
// rep2: protons at relaxed TRANSFERRED positions (dzpT H20,H41)
// rep3: protons on-axis d=1.995 (grid transferred point)
const BASE = "/home/prokop/git/dftbplus/data/xyz/dzp_relaxed_mid.xyz";
load_xyz("dzp", BASE);
let mid = get_xyz("dzp");
load_xyz("d0", "/home/prokop/git/dftbplus/data/xyz/dzp_relaxed.xyz");
let g0 = get_xyz("d0");
load_xyz("d1", "/home/prokop/git/dftbplus/data/xyz/dzpT_relaxed.xyz");
let g1 = get_xyz("d1");

const J1D = 9;  const J1P = 20; const J1A = 23;
const J2D = 30; const J2P = 41; const J2A = 2;

fn axis(xyz, dn, ac) {
    let ax = xyz[3*ac]-xyz[3*dn]; let ay = xyz[3*ac+1]-xyz[3*dn+1]; let az = xyz[3*ac+2]-xyz[3*dn+2];
    let n = sqrt(ax*ax+ay*ay+az*az);
    [ax/n, ay/n, az/n]
}
fn moved(xyz, dn, pr, d, a) {
    xyz[3*pr]   = xyz[3*dn]   + d*a[0];
    xyz[3*pr+1] = xyz[3*dn+1] + d*a[1];
    xyz[3*pr+2] = xyz[3*dn+2] + d*a[2];
    xyz
}

let a1 = axis(mid, J1D, J1A);
let a2 = axis(mid, J2D, J2A);
let n = mid.len();
let all = [];
// rep0: relaxed donor proton positions
let g = mid.map(|v| v);
g[3*J1P]=g0[3*J1P]; g[3*J1P+1]=g0[3*J1P+1]; g[3*J1P+2]=g0[3*J1P+2];
g[3*J2P]=g0[3*J2P]; g[3*J2P+1]=g0[3*J2P+1]; g[3*J2P+2]=g0[3*J2P+2];
for x in g { all.push(x); }
// rep1: on-axis d=1.055
g = moved(mid, J1D, J1P, 1.055, a1); g = moved(g, J2D, J2P, 1.055, a2);
for x in g { all.push(x); }
// rep2: relaxed transferred proton positions
g = mid;
g[3*J1P]=g1[3*J1P]; g[3*J1P+1]=g1[3*J1P+1]; g[3*J1P+2]=g1[3*J1P+2];
g[3*J2P]=g1[3*J2P]; g[3*J2P+1]=g1[3*J2P+1]; g[3*J2P+2]=g1[3*J2P+2];
for x in g { all.push(x); }
// rep3: on-axis d=1.995
g = moved(mid, J1D, J1P, 1.995, a1); g = moved(g, J2D, J2P, 1.995, a2);
for x in g { all.push(x); }

// also report off-axis distance of relaxed protons from the axis
for j in 0..2 {
    let dn = if j==0 { J1D } else { J2D };
    let pr = if j==0 { J1P } else { J2P };
    let ac = if j==0 { J1A } else { J2A };
    let a  = if j==0 { a1 } else { a2 };
    for tag in ["donor","transf"] {
        let src = if tag=="donor" { g0 } else { g1 };
        let px = src[3*pr]-mid[3*dn]; let py = src[3*pr+1]-mid[3*dn+1]; let pz = src[3*pr+2]-mid[3*dn+2];
        let par = px*a[0]+py*a[1]+pz*a[2];
        let off = sqrt(px*px+py*py+pz*pz - par*par);
        print(tag + " J" + itos(j+1) + ": d_par=" + ftos(par) + " off-axis=" + ftos(off) + " A");
    }
}

gpu_new("dzp", SK_DIR, 4);
gpu_set_coords("dzp", all);
gpu_smearing("dzp", 0.002);
gpu_reset_q("dzp");
gpu_scc("dzp", 100, 1.0e-5);
gpu_eval("dzp", false);
for b in 0..4 {
    print("E[" + itos(b) + "] = " + ftos(gpu_energy_i("dzp", b)));
}
