#!/usr/bin/env python3
"""mol2 -> xyz for the new dimers + automatic H-bond junction identification.

A junction = (donor_heavy, proton, acceptor_heavy) where donor is N/O bound
to an H, and acceptor is the nearest N/O of the OTHER monomer within 2.6 A
of the proton.  Prints junction table (0-based) and writes one .xyz per mol2
into data/xyz/.
"""
import numpy as np, sys, os

MOLS = {
    "pyridone_dimer":        "/home/prokop/git/dftbplus/data/mol/pyridone_dimer.mol2",
    "pyridone_isodimer":     "/home/prokop/git/dftbplus/data/mol/pyridone_isodimer.mol2",
    "diazaphenalene_dimer":  "/home/prokop/git/dftbplus/data/mol/diazaphenalene_dimer.mol2",
}
OUT = "/home/prokop/git/dftbplus/data/xyz"

def read_mol2(path):
    lines = open(path).read().splitlines()
    i_at = lines.index("@<TRIPOS>ATOM") + 1
    i_bd = lines.index("@<TRIPOS>BOND")
    el, xyz, res = [], [], []
    for ln in lines[i_at:i_bd]:
        t = ln.split()
        if len(t) < 6: continue
        el.append(t[5].split(".")[0])
        xyz.append([float(t[2]), float(t[3]), float(t[4])])
        res.append(int(t[6]))            # residue = monomer id
    bonds = []
    for ln in lines[i_bd + 1:]:
        t = ln.split()
        if len(t) < 4 or not t[0].isdigit(): break
        bonds.append((int(t[1]) - 1, int(t[2]) - 1))
    return el, np.array(xyz), res, bonds

for name, path in MOLS.items():
    el, xyz, res, bonds = read_mol2(path)
    n = len(el)
    # donors: N/O bound to H
    donors = {}
    for i, j in bonds:
        a, b = (i, j) if el[i] == "H" else (j, i) if el[j] == "H" else (None, None)
        if a is not None and el[b] in ("N", "O"):
            donors.setdefault(b, []).append(a)
    print(f"\n=== {name}: {n} atoms, donors (heavy->H): {donors}")
    acc = [i for i in range(n) if el[i] in ("N", "O") and i not in donors or
           (el[i] == "O" and i not in donors)]  # N/O not carrying H; O acceptor if it has no H
    junctions = []
    for dh, hs in donors.items():
        for h in hs:
            # nearest N/O of the other monomer (not bonded to this H)
            best, bd = None, 1e9
            for a in range(n):
                if el[a] not in ("N", "O") or res[a] == res[dh]: continue
                d = np.linalg.norm(xyz[h] - xyz[a])
                if d < bd: bd, best = d, a
            if best is not None and bd < 2.6:
                axis = xyz[best] - xyz[dh]
                rNN = np.linalg.norm(axis)
                junctions.append((dh, h, best))
                print(f"  J: {el[dh]}{dh}-H{h}...{el[best]}{best}  r(H..acc)={bd:.3f}  r(don..acc)={rNN:.3f} A")
            else:
                print(f"  {el[dh]}{dh}-H{h}: no acceptor within 2.6 A (nearest {bd:.2f})")
    with open(f"{OUT}/{name}.xyz", "w") as f:
        f.write(f"{n}\n{name} from {path}\n")
        for e, p in zip(el, xyz):
            f.write(f"{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
    print(f"  wrote {OUT}/{name}.xyz")
