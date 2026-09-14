// Diazaphenalene dimer 2D scan — SCC only, batch=400, INTERPOLATED scaffold:
// heavy atoms blend between relaxed donor (s=0) and relaxed transferred (s=1)
// tautomers, s = mean transfer progress of the two protons.
const NP = 20;
const D_LO = 1.0;
const D_HI = 2.05;
const J1D = 9;  const J1P = 20; const J1A = 23;
const J2D = 30; const J2P = 41; const J2A = 2;

print("=== diazaphenalene dimer 2D scan, interpolated scaffold (batch=" + (NP*NP) + ") ===");
load_xyz("dzp", "/home/prokop/git/dftbplus/data/xyz/dzp_relaxed.xyz");
load_xyz("g1", "/home/prokop/git/dftbplus/data/xyz/dzpT_relaxed.xyz");
let x0 = get_xyz("dzp");
let x1 = get_xyz("g1");
let n_atoms = x0.len() / 3;

// junction axes on the donor scaffold
let a1x = x0[3*J1A]-x0[3*J1D]; let a1y = x0[3*J1A+1]-x0[3*J1D+1]; let a1z = x0[3*J1A+2]-x0[3*J1D+2];
let a2x = x0[3*J2A]-x0[3*J2D]; let a2y = x0[3*J2A+1]-x0[3*J2D+1]; let a2z = x0[3*J2A+2]-x0[3*J2D+2];
let n1 = sqrt(a1x*a1x+a1y*a1y+a1z*a1z); let n2 = sqrt(a2x*a2x+a2y*a2y+a2z*a2z);
a1x/=n1; a1y/=n1; a1z/=n1; a2x/=n2; a2y/=n2; a2z/=n2;
// transferred scaffold needs its own axes (donor atom positions differ slightly)
let b1x = x1[3*J1A]-x1[3*J1D]; let b1y = x1[3*J1A+1]-x1[3*J1D+1]; let b1z = x1[3*J1A+2]-x1[3*J1D+2];
let b2x = x1[3*J2A]-x1[3*J2D]; let b2y = x1[3*J2A+1]-x1[3*J2D+1]; let b2z = x1[3*J2A+2]-x1[3*J2D+2];
let m1 = sqrt(b1x*b1x+b1y*b1y+b1z*b1z); let m2 = sqrt(b2x*b2x+b2y*b2y+b2z*b2z);
b1x/=m1; b1y/=m1; b1z/=m1; b2x/=m2; b2y/=m2; b2z/=m2;
print("axes: donor rNN=" + ftos(n1) + "/" + ftos(n2) + "  transf rNN=" + ftos(m1) + "/" + ftos(m2));

let all = [];
for i1 in 0..NP {
    let d1 = D_LO + i1 * (D_HI - D_LO) / (NP - 1);
    for i2 in 0..NP {
        let d2 = D_LO + i2 * (D_HI - D_LO) / (NP - 1);
        // transfer progress of this replica, 0 = donor end, 1 = transferred end
        let s = ((d1 - 1.05) + (d2 - 1.05)) / (2.0 * 0.95);
        let s = if s < 0.0 { 0.0 } else if s > 1.0 { 1.0 } else { s };
        for a in 0..n_atoms {
            let bx = x0[3*a] + s*(x1[3*a]-x0[3*a]);
            let by = x0[3*a+1] + s*(x1[3*a+1]-x0[3*a+1]);
            let bz = x0[3*a+2] + s*(x1[3*a+2]-x0[3*a+2]);
            if a == J1P {
                // interpolated donor position + d1 * interpolated axis
                let ux = a1x + s*(b1x-a1x); let uy = a1y + s*(b1y-a1y); let uz = a1z + s*(b1z-a1z);
                let dx = x0[3*J1D] + s*(x1[3*J1D]-x0[3*J1D]);
                let dy = x0[3*J1D+1] + s*(x1[3*J1D+1]-x0[3*J1D+1]);
                let dz = x0[3*J1D+2] + s*(x1[3*J1D+2]-x0[3*J1D+2]);
                all.push(dx + d1*ux); all.push(dy + d1*uy); all.push(dz + d1*uz);
            } else if a == J2P {
                let ux = a2x + s*(b2x-a2x); let uy = a2y + s*(b2y-a2y); let uz = a2z + s*(b2z-a2z);
                let dx = x0[3*J2D] + s*(x1[3*J2D]-x0[3*J2D]);
                let dy = x0[3*J2D+1] + s*(x1[3*J2D+1]-x0[3*J2D+1]);
                let dz = x0[3*J2D+2] + s*(x1[3*J2D+2]-x0[3*J2D+2]);
                all.push(dx + d2*ux); all.push(dy + d2*uy); all.push(dz + d2*uz);
            } else {
                all.push(bx); all.push(by); all.push(bz);
            }
        }
    }
}

gpu_new("dzp", SK_DIR, NP*NP);
gpu_set_coords("dzp", all);
gpu_smearing("dzp", 0.002);
gpu_reset_q("dzp");

let t = clock();
let rms = gpu_scc("dzp", 100, 1.0e-5);
print("SCC cold: rms=" + ftos(rms) + " iters=" + itos(gpu_scc_iters("dzp")) + " t=" + ftos((clock()-t)*1000.0) + " ms");
gpu_eval("dzp", false);

print("=== ENERGY MAP (d1,d2,E) ===");
for i1 in 0..NP {
    for i2 in 0..NP {
        let d1 = D_LO + i1 * (D_HI - D_LO) / (NP - 1);
        let d2 = D_LO + i2 * (D_HI - D_LO) / (NP - 1);
        print(ftos(d1) + " " + ftos(d2) + " " + ftos(gpu_energy_i("dzp", i1*NP+i2)));
    }
}
print("=== done ===");
