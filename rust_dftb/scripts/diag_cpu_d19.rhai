// Does the CPU f64 SCC converge at the far-transfer point d=1.9?
// (GPU DIIS fails there with rms~1.3e-2 — physics or GPU?)
const sk_dir = SK_DIR;
load_xyz("gc", "/home/prokop/git/dftbplus/data/xyz/guanine-cytosine.xyz");
let xyz = get_xyz("gc");
let dx = xyz[3*21]   - xyz[3*8];
let dy = xyz[3*21+1] - xyz[3*8+1];
let dz = xyz[3*21+2] - xyz[3*8+2];
let dNN = sqrt(dx*dx + dy*dy + dz*dz);
dx /= dNN; dy /= dNN; dz /= dNN;
let d = 1.9;
let sc = [];
for a in 0..29 {
    if a == 13 {
        sc.push(xyz[3*8]   + d * dx);
        sc.push(xyz[3*8+1] + d * dy);
        sc.push(xyz[3*8+2] + d * dz);
    } else {
        sc.push(xyz[3*a]); sc.push(xyz[3*a+1]); sc.push(xyz[3*a+2]);
    }
}
// Species MUST come from the loaded xyz — a hand-typed list here once had an
// extra H at index 16 (shifted elements 17-25 → wrong molecule, ~2.4 Ha off).
let sp = get_species("gc");
let species = "";
for s in 0..sp.len() { if s > 0 { species += ","; } species += sp[s]; }
make_geom("gc19", species, sc);
// Non-SCC H0 gap first (works even though SCC diverges)
let e0n = run_dftb_nonscc("gc19", sk_dir);
print("CPU d=1.9 H0 E_band=" + ftos(e0n) + "  HL: " + get_homo_lumo("gc19"));
let e = run_dftb_scc("gc19", sk_dir, 300, 1.0e-8);
print("CPU d=1.9 E=" + ftos(e) + "  HL: " + get_homo_lumo("gc19"));

// also the native point for comparison
let e0 = run_dftb_scc("gc", sk_dir, 300, 1.0e-8);
print("CPU native E=" + ftos(e0) + "  HL: " + get_homo_lumo("gc"));
