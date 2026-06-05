#!/usr/bin/env python3
import os
import sys
from math import sqrt


def write_hsd(path, elem, r_bohr, vec):
    # vec is a direction unit vector in cart coords; place atoms at +/- r/2 along vec
    vx, vy, vz = vec
    n = sqrt(vx*vx + vy*vy + vz*vz)
    vx, vy, vz = vx/n, vy/n, vz/n
    dx, dy, dz = 0.5*r_bohr*vx, 0.5*r_bohr*vy, 0.5*r_bohr*vz

    sk_dir = os.environ.get("DFTB_SK_DIR", "/home/prokophapala/SIMULATIONS/dftbplus/slakos/matsci-0-3")

    with open(path, "w") as f:
        f.write("Geometry = {\n")
        f.write("  Periodic = No\n")
        f.write(f"  TypeNames = {{{elem}}}\n")
        f.write("  TypesAndCoordinates [Bohr] = {\n")
        f.write(f"    1  { -dx: .12f}  { -dy: .12f}  { -dz: .12f}\n")
        f.write(f"    1  {  dx: .12f}  {  dy: .12f}  {  dz: .12f}\n")
        f.write("  }\n")
        f.write("}\n\n")

        f.write("Hamiltonian = DFTB {\n")
        f.write("  SCC = No\n")
        f.write("  MaxAngularMomentum = {\n")
        f.write(f"    {elem} = \"p\"\n")
        f.write("  }\n")
        f.write("  SlaterKosterFiles = Type2FileNames {\n")
        f.write(f"    Prefix = \"{sk_dir}/\"\n")
        f.write("    Separator = \"-\"\n")
        f.write("    Suffix = \".skf\"\n")
        f.write("  }\n")
        f.write("}\n\n")

        # Avoid any extra output/stop
        f.write("Options {\n")
        f.write("}\n")


def main():
    if len(sys.argv) != 7:
        print("Usage: make_diatomic_hsd.py <out_hsd> <elem> <r_bohr> <vx> <vy> <vz>")
        sys.exit(2)

    out_hsd = sys.argv[1]
    elem = sys.argv[2]
    r_bohr = float(sys.argv[3])
    vx, vy, vz = map(float, sys.argv[4:7])

    write_hsd(out_hsd, elem, r_bohr, (vx, vy, vz))


if __name__ == "__main__":
    main()
