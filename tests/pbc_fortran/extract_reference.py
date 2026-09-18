#!/usr/bin/env python3
"""Extract Fortran DFTB+ reference values for the GPU PBC parity test.

Reads from a finished DFTB+ run directory:
  detailed.out  -> Mulliken gross charges, energies (Hartree)
  band.out      -> converged eigenvalues per k-point (eV -> Ha)

Writes reference.txt in a flat `key = value(s)` format that
gpu_pbc_fortran.rs parses with plain float/string ops (no deps).

Sections:
  kpoints     nk lines:  kx ky kz weight            (fractional)
  charges     nat lines: gross charge (e, net)      (atom order)
  eigenvalues nk*n lines, grouped per k in Fortran KPT order (Ha)
  energy_*    scalars (Ha): band, h0, scc, electronic, repulsive, total
"""
import sys, os

EV_PER_HA = 27.211386245988


def parse_detailed(path):
    charges, energy = [], {}
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if "Atomic gross charges" in ln:
            i += 2  # skip header "Atom  Charge"
            while True:
                p = lines[i].split()
                if len(p) == 2 and p[0].isdigit():
                    charges.append(float(p[1]))
                    i += 1
                else:
                    break
            continue
        for key, tag in [
            ("Band energy:", "energy_band"),
            ("Energy H0:", "energy_h0"),
            ("Energy SCC:", "energy_scc"),
            ("Total Electronic energy:", "energy_electronic"),
            ("Repulsive energy:", "energy_repulsive"),
            ("Total energy:", "energy_total"),
            ("Fermi level:", "fermi_level"),
        ]:
            if ln.strip().startswith(key):
                energy[tag] = float(ln.split()[len(key.split())])
        i += 1
    return charges, energy


def parse_bandout(path):
    """band.out: `KPT n SPIN s KWEIGHT w` header then `idx E_eV occ` lines."""
    eigs, weights = [], []
    for ln in open(path):
        p = ln.split()
        if len(p) >= 6 and p[0] == "KPT":
            weights.append(float(p[5]))
        elif len(p) == 3:
            eigs.append(float(p[1]) / EV_PER_HA)
    return eigs, weights


def main():
    work = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "reference.txt"
    charges, energy = parse_detailed(os.path.join(work, "detailed.out"))
    eigs, weights = parse_bandout(os.path.join(work, "band.out"))
    if not charges or not eigs:
        sys.exit("extract_reference: empty parse -- is the run complete?")
    with open(out, "w") as f:
        f.write("# Fortran DFTB+ reference for tests/gpu_pbc_fortran.rs\n")
        f.write(f"n_atoms = {len(charges)}\n")
        f.write(f"n_kpoints = {len(weights)}\n")
        f.write("kpoints_weights = " + " ".join(f"{w:.10f}" for w in weights) + "\n")
        f.write("charges_net = " + " ".join(f"{q:.10f}" for q in charges) + "\n")
        f.write("eigenvalues_ha = " + " ".join(f"{e:.10f}" for e in eigs) + "\n")
        for k, v in energy.items():
            f.write(f"{k} = {v:.10f}\n")
    print(f"wrote {out}: {len(charges)} atoms, {len(weights)} k-points, "
          f"{len(eigs)} eigenvalues")


if __name__ == "__main__":
    main()
