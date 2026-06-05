#!/usr/bin/env python3
import os
import sys
import numpy as np

# Make pyBall importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "git", "dftbplus")))

from pyBall.DFTBcore import DFTBcore


def write_square(path: str, a: np.ndarray):
    assert a.ndim == 2 and a.shape[0] == a.shape[1]
    n = a.shape[0]
    with open(path, "w") as f:
        f.write(f"{n} {n}\n")
        for i in range(n):
            f.write(" ".join(f"{a[i,j]:.16e}" for j in range(n)))
            f.write("\n")


def main():
    if len(sys.argv) != 4:
        print("Usage: run_dftbcore_dump.py <workdir> <ref_h> <ref_s>")
        sys.exit(2)

    workdir, out_h, out_s = sys.argv[1:4]

    # IMPORTANT: libdftbcore currently uses parseHsdInput(input) without passing filename,
    # so it effectively expects the default input file name in CWD.
    # We therefore chdir into workdir and require workdir/dftb_in.hsd to exist.
    os.chdir(workdir)

    dftb = DFTBcore()  # auto-finds libdftbcore.so in build tree
    dftb.init("dftb_in.hsd")
    dftb.enable_matrix_collection(dm=False, h=True, s=True)
    e = dftb.run_scf()
    H = dftb.get_h_dense()
    S = dftb.get_s_dense()
    dftb.finalize()

    write_square(out_h, H)
    write_square(out_s, S)

    print(f"Energy(Ha) = {e:.12f}")
    print(f"Wrote: {out_h}, {out_s}")


if __name__ == "__main__":
    main()
