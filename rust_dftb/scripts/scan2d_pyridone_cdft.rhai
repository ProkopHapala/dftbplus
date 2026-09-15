// Pyridone dimer 2D proton-transfer scan — NEUTRAL vs CDFT charge-transfer
// diabatic states, batch=400.
//
// Physics: closed-shell dimer, total charge conserved (ΣΔq=0), so ONE
// fragment constraint Q_1 = ±δ e gives the zwitterionic CT diabatics
//   Q_1 = -δ  →  M1^+δ M2^-δ  (partial electron removal from monomer 1)
//   Q_1 = +δ  →  M1^-δ M2^+δ
// NOTE: |δ|=1.0 is NOT reachable with Mulliken constraints on this dimer
// — fragment charge saturates at ~0.6 e (measured: bisection stalls at
// q_err~0.42). δ = 0.5 is a strong but reachable CT displacement.
// compare against the neutral (unconstrained) map = adiabatic ground state.
//
// Frag map: each atom joins the monomer of its nearest HEAVY atom
// (heavy1 = atoms 0..6, heavy2 = 11..17) — robust to atom ordering.
// Energies are gpu_cdft_energies() = E_DFTB of the constrained state
// (raw gpu_eval carries the +λ·Q_F offset — do NOT plot that).

const NP = 20;
const D_LO = 1.0;
const D_HI = 1.85;
const BASE = "/home/prokop/git/dftbplus/data/xyz/pylac_relaxed.xyz";
const J1D = 16; const J1P = 20; const J1A = 6;
const J2D = 5;  const J2P = 23; const J2A = 17;
const Q_TOL = 1.0e-3;      // fragment-charge tolerance (e)
const MAX_OUTER = 40;

print("=== pyridone 2D scan: neutral + CDFT CT states (batch=" + (NP*NP) + ") ===");
load_xyz("py", BASE);
let xyz = get_xyz("py");
let n_atoms = xyz.len() / 3;

// --- junction axes (same as scan2d_pyridone.rhai) ---
let a1x = xyz[3*J1A]   - xyz[3*J1D];   let a1y = xyz[3*J1A+1] - xyz[3*J1D+1];   let a1z = xyz[3*J1A+2] - xyz[3*J1D+2];
let a2x = xyz[3*J2A]   - xyz[3*J2D];   let a2y = xyz[3*J2A+1] - xyz[3*J2D+1];   let a2z = xyz[3*J2A+2] - xyz[3*J2D+2];
let n1 = sqrt(a1x*a1x + a1y*a1y + a1z*a1z);
let n2 = sqrt(a2x*a2x + a2y*a2y + a2z*a2z);
a1x /= n1; a1y /= n1; a1z /= n1;
a2x /= n2; a2y /= n2; a2z /= n2;

// --- fragment map: nearest heavy atom decides monomer membership ---
let frag = [];
for a in 0..n_atoms {
    let d1min = 1.0e30; let d2min = 1.0e30;
    for h in 0..7 {          // heavy atoms of monomer 1: 0..6
        let dx = xyz[3*a] - xyz[3*h]; let dy = xyz[3*a+1] - xyz[3*h+1]; let dz = xyz[3*a+2] - xyz[3*h+2];
        let d = dx*dx + dy*dy + dz*dz; if d < d1min { d1min = d; }
    }
    for h in 11..18 {        // heavy atoms of monomer 2: 11..17
        let dx = xyz[3*a] - xyz[3*h]; let dy = xyz[3*a+1] - xyz[3*h+1]; let dz = xyz[3*a+2] - xyz[3*h+2];
        let d = dx*dx + dy*dy + dz*dz; if d < d2min { d2min = d; }
    }
    frag.push(if d1min <= d2min { 0 } else { -1 });   // frag 0 = monomer 1
}
let nm1 = 0; for f in frag { if f == 0 { nm1 += 1; } }
print("monomer-1 atoms (frag 0): " + itos(nm1) + " / " + itos(n_atoms));

// --- 20x20 geometry set (identical to the neutral scan) ---
let all = [];
for i1 in 0..NP {
    let d1 = D_LO + i1 * (D_HI - D_LO) / (NP - 1);
    for i2 in 0..NP {
        let d2 = D_LO + i2 * (D_HI - D_LO) / (NP - 1);
        for a in 0..n_atoms {
            if a == J1P {
                all.push(xyz[3*J1D] + d1*a1x); all.push(xyz[3*J1D+1] + d1*a1y); all.push(xyz[3*J1D+2] + d1*a1z);
            } else if a == J2P {
                all.push(xyz[3*J2D] + d2*a2x); all.push(xyz[3*J2D+1] + d2*a2y); all.push(xyz[3*J2D+2] + d2*a2z);
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

fn emit_map(tag, np, lo, hi) {
    print("=== MAP " + tag + " ===");
    for i1 in 0..np {
        for i2 in 0..np {
            let b = i1*np + i2;
            let d1 = lo + i1 * (hi - lo) / (np - 1);
            let d2 = lo + i2 * (hi - lo) / (np - 1);
            print(ftos(d1) + " " + ftos(d2) + " " + ftos(gpu_energy_i("py", b)));
        }
    }
}

// ---------- state 0: neutral ground state (reference) ----------
let t = clock();
gpu_scc("py", 100, 1.0e-5);
print("neutral cold: iters=" + itos(gpu_scc_iters("py")) + " t=" + ftos((clock()-t)*1000.0) + " ms");
gpu_eval("py", false);
emit_map("NEUTRAL", NP, D_LO, D_HI);

// ---------- state 1: M1+0.5 M2-0.5  (partial electron removal) ----------
gpu_cdft("py", frag, [-1.0]);
let t1 = clock();
let e1 = gpu_cdft_scc("py", MAX_OUTER, 60, 1.0e-5, Q_TOL);
print("CT- (Q1=-1.0): q_err_max=" + ftos(e1) + " t=" + ftos((clock()-t1)*1000.0) + " ms");
gpu_cdft_energies("py");
emit_map("CT_M1p_M2m", NP, D_LO, D_HI);

// ---------- state 2: M1-0.5 M2+0.5  (partial electron addition) ----------
gpu_cdft("py", frag, [1.0]);
let t2 = clock();
let e2 = gpu_cdft_scc("py", MAX_OUTER, 60, 1.0e-5, Q_TOL);
print("CT+ (Q1=+1.0): q_err_max=" + ftos(e2) + " t=" + ftos((clock()-t2)*1000.0) + " ms");
gpu_cdft_energies("py");
emit_map("CT_M1m_M2p", NP, D_LO, D_HI);

// ---------- diagnostics: Q on monomer 2 should mirror (ΣΔq=0) ----------
print("check replica0: Q1=" + ftos(gpu_cdft_qf("py", 0, 0)) + " lam=" + ftos(gpu_cdft_lam("py", 0, 0)));
print("=== done ===");
