// Pyridone dimer 2D proton-transfer scan — SCC only, batch=400.
// Base: relaxed lactam scaffold (pylac_relaxed.xyz).
// J1: N16-H20...O6   (d1 = r(N16-H20), axis N16->O6)
// J2: N5 -H23...O17  (d2 = r(N5-H23),  axis N5->O17)
const NP = 20;
const D_LO = 1.0;
const D_HI = 1.85;
const BASE = "/home/prokop/git/dftbplus/data/xyz/pylac_relaxed.xyz";
const J1D = 16; const J1P = 20; const J1A = 6;
const J2D = 5;  const J2P = 23; const J2A = 17;

print("=== pyridone dimer 2D scan (SCC only, batch=" + (NP*NP) + ") ===");
load_xyz("py", BASE);
let xyz = get_xyz("py");
let n_atoms = xyz.len() / 3;

let a1x = xyz[3*J1A]   - xyz[3*J1D];   let a1y = xyz[3*J1A+1] - xyz[3*J1D+1];   let a1z = xyz[3*J1A+2] - xyz[3*J1D+2];
let a2x = xyz[3*J2A]   - xyz[3*J2D];   let a2y = xyz[3*J2A+1] - xyz[3*J2D+1];   let a2z = xyz[3*J2A+2] - xyz[3*J2D+2];
let n1 = sqrt(a1x*a1x + a1y*a1y + a1z*a1z);
let n2 = sqrt(a2x*a2x + a2y*a2y + a2z*a2z);
a1x /= n1; a1y /= n1; a1z /= n1;
a2x /= n2; a2y /= n2; a2z /= n2;
print("J1 r(don..acc)=" + ftos(n1) + " A;  J2 r(don..acc)=" + ftos(n2) + " A");

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

gpu_new("py", SK_DIR, NP*NP);
gpu_set_coords("py", all);
gpu_smearing("py", 0.002);
gpu_reset_q("py");

let t = clock();
let rms = gpu_scc("py", 100, 1.0e-5);
print("SCC cold: rms=" + ftos(rms) + " iters=" + itos(gpu_scc_iters("py")) + " t=" + ftos((clock()-t)*1000.0) + " ms");

let e0 = gpu_eval("py", false);
print("eval done; E[0]=" + ftos(e0));

let t2 = clock();
for k in 0..3 { gpu_scc("py", 100, 1.0e-5); }
print("SCC warm: " + ftos((clock()-t2)/3.0*1000.0) + " ms/solve  iters=" + itos(gpu_scc_iters("py")));

print("=== ENERGY MAP (d1,d2,E) ===");
for i1 in 0..NP {
    for i2 in 0..NP {
        let b = i1*NP + i2;
        let d1 = D_LO + i1 * (D_HI - D_LO) / (NP - 1);
        let d2 = D_LO + i2 * (D_HI - D_LO) / (NP - 1);
        print(ftos(d1) + " " + ftos(d2) + " " + ftos(gpu_energy_i("py", b)));
    }
}
print("=== done ===");
