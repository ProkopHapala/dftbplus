// 7-azaindole dimer 2D proton-transfer scan — SCC only, batch=400.
// J1: N6-H21...N10  (d1 = r(N6-H21), axis N6->N10)
// J2: N15-H27...N1  (d2 = r(N15-H27), axis N15->N1)
// replica i1*NP+i2: H21 at N6+d1*a1, H27 at N15+d2*a2.
const sk_dir = SK_DIR;
const NP = 20;
const D0 = 1.0;
const DSTEP = 0.05;

print("=== 7-azaindole 2D scan (SCC only, batch=" + (NP*NP) + ") ===");
load_xyz("az", "/home/prokop/git/dftbplus/data/xyz/azaindol_dimer.xyz");
let xyz = get_xyz("az");
let n_atoms = 30;

// junction axes (donor -> acceptor)
let a1x = xyz[3*10]   - xyz[3*6];   let a1y = xyz[3*10+1] - xyz[3*6+1];   let a1z = xyz[3*10+2] - xyz[3*6+2];
let a2x = xyz[3*1]    - xyz[3*15];  let a2y = xyz[3*1+1]  - xyz[3*15+1];  let a2z = xyz[3*1+2]  - xyz[3*15+2];
let n1 = sqrt(a1x*a1x + a1y*a1y + a1z*a1z);
let n2 = sqrt(a2x*a2x + a2y*a2y + a2z*a2z);
a1x /= n1; a1y /= n1; a1z /= n1;
a2x /= n2; a2y /= n2; a2z /= n2;
print("J1 N6..N10 = " + ftos(n1) + " A;  J2 N15..N1 = " + ftos(n2) + " A");

let all = [];
for i1 in 0..NP {
    let d1 = D0 + i1 * DSTEP;
    for i2 in 0..NP {
        let d2 = D0 + i2 * DSTEP;
        for a in 0..n_atoms {
            if a == 21 {
                all.push(xyz[3*6]   + d1*a1x);
                all.push(xyz[3*6+1] + d1*a1y);
                all.push(xyz[3*6+2] + d1*a1z);
            } else if a == 27 {
                all.push(xyz[3*15]   + d2*a2x);
                all.push(xyz[3*15+1] + d2*a2y);
                all.push(xyz[3*15+2] + d2*a2z);
            } else {
                all.push(xyz[3*a]); all.push(xyz[3*a+1]); all.push(xyz[3*a+2]);
            }
        }
    }
}

gpu_new("az", sk_dir, NP*NP);
gpu_set_coords("az", all);
gpu_smearing("az", 0.002);
gpu_reset_q("az");

let t = clock();
let rms = gpu_scc("az", 100, 1.0e-5);
print("SCC cold: rms=" + ftos(rms) + " iters=" + itos(gpu_scc_iters("az")) + " t=" + ftos((clock()-t)*1000.0) + " ms status=" + gpu_scc_status("az"));

let e0 = gpu_eval("az", false);
print("eval done; E[0]=" + ftos(e0) + "  (per-replica E via gpu_energy_i)");

// warm re-solve for timing
let t2 = clock();
for k in 0..3 { gpu_scc("az", 100, 1.0e-5); }
print("SCC warm: " + ftos((clock()-t2)/3.0*1000.0) + " ms/solve  iters=" + itos(gpu_scc_iters("az")));

// energy map (one line per replica, i1*NP+i2)
print("=== ENERGY MAP (d1,d2,E) ===");
for i1 in 0..NP {
    for i2 in 0..NP {
        let b = i1*NP + i2;
        print(ftos(D0 + i1*DSTEP) + " " + ftos(D0 + i2*DSTEP) + " " + ftos(gpu_energy_i("az", b)));
    }
}
print("=== done ===");
