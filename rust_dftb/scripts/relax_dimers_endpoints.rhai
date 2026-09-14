// Relax the two limiting tautomers of each new dimer (batch=1 each).
// Prints relaxed coords (flat, Angstrom) for writing .xyz.
fn relax_one(tag, path, sk_dir) {
    let FTOL = 1.0e-3;
    let STOL = 1.0e-5;
    load_xyz(tag, path);
    let xyz = get_xyz(tag);
    gpu_new(tag, sk_dir, 1);
    gpu_set_coords(tag, xyz);
    gpu_smearing(tag, 0.002);
    gpu_scc(tag, 100, STOL);
    let e0 = gpu_eval(tag, false);
    let t = clock();
    let mf = gpu_relax(tag, 200, FTOL, STOL);
    print(tag + " relaxed: max|F|=" + ftos(mf) + " E0=" + ftos(e0) + " t=" + ftos(clock()-t) + " s");
    let e1 = gpu_eval(tag, false);
    print(tag + " E_relaxed=" + ftos(e1));
    print("XYZ_BEGIN " + tag);
    let c = gpu_get_coords(tag);
    for i in 0..c.len() { print(ftos(c[i])); }
    print("XYZ_END");
}

relax_one("pylac", "/home/prokop/git/dftbplus/data/xyz/pyridone_dimer.xyz", SK_DIR);
relax_one("pyiso", "/home/prokop/git/dftbplus/data/xyz/pyridone_isodimer.xyz", SK_DIR);
relax_one("dzp",   "/home/prokop/git/dftbplus/data/xyz/diazaphenalene_dimer.xyz", SK_DIR);
relax_one("dzpT",  "/home/prokop/git/dftbplus/data/xyz/diazaphenalene_transferred.xyz", SK_DIR);
