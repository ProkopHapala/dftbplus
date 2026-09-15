// Quinoxaline/dihydroquinoxaline PBC chain — 2D proton-transfer scan.
// Same recipe as scan2d_dzp.rhai but through GpuPbc (periodic, k-points).
// Cell: 34 atoms, periodic along y (L=11.480 A), herringbone tilt 25 deg
// (scripts/make_qxhq_chain.py -> data/xyz/qxhq_chain_cell.xyz).
// J1: N11-H27...N8          intracell, axis +y
// J2: N19-H33...N0(image y-L)  crosses the cell boundary, axis -y
const NP = 20;
const D_LO = 1.0;
const D_HI = 1.9;                  // rNN = 2.9 -> H..acc ~1.0 A at the far end
const BASE = "/home/prokop/git/dftbplus/data/xyz/qxhq_chain_cell.xyz";
const LY = 11.480;
const J1D = 11; const J1P = 27; const J1A = 8;
const J2D = 19; const J2P = 33; const J2A = 0;    // J2A sits in the NEXT cell

print("=== QX/HQ PBC chain 2D scan (batch=" + (NP*NP) + ") ===");
load_xyz("qx", BASE);
let xyz = get_xyz("qx");
let n_atoms = xyz.len() / 3;

// junction axes: donor -> acceptor unit vectors
let a1x = xyz[3*J1A]   - xyz[3*J1D];
let a1y = xyz[3*J1A+1] - xyz[3*J1D+1];
let a1z = xyz[3*J1A+2] - xyz[3*J1D+2];
// J2 acceptor is the periodic image of atom 0 one cell BELOW (y - LY)
let a2x = xyz[3*J2A]             - xyz[3*J2D];
let a2y = xyz[3*J2A+1] - LY      - xyz[3*J2D+1];
let a2z = xyz[3*J2A+2]           - xyz[3*J2D+2];
let n1 = sqrt(a1x*a1x + a1y*a1y + a1z*a1z);
let n2 = sqrt(a2x*a2x + a2y*a2y + a2z*a2z);
a1x /= n1; a1y /= n1; a1z /= n1;
a2x /= n2; a2y /= n2; a2z /= n2;
print("J1 r(N..N)=" + ftos(n1) + " A;  J2 r(N..N)=" + ftos(n2) + " A (J2 across cell boundary)");

let all = [];
for i1 in 0..NP {
    let d1 = D_LO + i1 * (D_HI - D_LO) / (NP - 1);
    for i2 in 0..NP {
        let d2 = D_LO + i2 * (D_HI - D_LO) / (NP - 1);
        for a in 0..n_atoms {
            if a == J1P {
                all.push(xyz[3*J1D]   + d1*a1x);
                all.push(xyz[3*J1D+1] + d1*a1y);
                all.push(xyz[3*J1D+2] + d1*a1z);
            } else if a == J2P {
                all.push(xyz[3*J2D]   + d2*a2x);
                all.push(xyz[3*J2D+1] + d2*a2y);
                all.push(xyz[3*J2D+2] + d2*a2z);
            } else {
                all.push(xyz[3*a]); all.push(xyz[3*a+1]); all.push(xyz[3*a+2]);
            }
        }
    }
}

// cell + k-points: 1D chain along y -> 4 k's along y (same set as the
// Fortran reference test), Gamma-centered.
let lat = [16.398276,0.0,0.0,  0.0,LY,0.0,  0.0,0.0,15.0];
let kpts = [0.0,0.0,0.0,  0.0,0.25,0.0,  0.0,-0.25,0.0,  0.0,0.5,0.0];
let kw = [0.25, 0.25, 0.25, 0.25];

pbc_new("qx", SK_DIR, lat, kpts, kw, NP*NP);
pbc_set_coords("qx", all);
pbc_smearing("qx", 0.002);

let t = clock();
let rms = pbc_scc("qx", 100, 1.0e-5);
print("SCC cold: rms=" + ftos(rms) + " t=" + ftos((clock()-t)*1000.0) + " ms");

let e0 = pbc_eval("qx");
print("eval done; E[0]=" + ftos(e0) + " Ha (incl. E_rep)");

print("=== ENERGY MAP (d1,d2,E) ===");
for i1 in 0..NP {
    for i2 in 0..NP {
        let b = i1*NP + i2;
        let d1 = D_LO + i1 * (D_HI - D_LO) / (NP - 1);
        let d2 = D_LO + i2 * (D_HI - D_LO) / (NP - 1);
        print(ftos(d1) + " " + ftos(d2) + " " + ftos(pbc_energy_i("qx", b)));
    }
}
print("=== done ===");
