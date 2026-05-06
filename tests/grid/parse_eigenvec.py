#!/usr/bin/env python3
"""Read eigenvec.bin directly from DFTB+ H2O calculation."""
import numpy as np, struct

work = 'dftb_h2o'
with open(f'{work}/eigenvec.bin', 'rb') as f:
    data = f.read()

print(f"File size: {len(data)} bytes")
# Expected: 1 identity record + 6 eigenvector records (6 doubles each)
# Fortran record: 4B marker | data | 4B marker
pos = 0
def read_record(f_bytes, pos):
    n = struct.unpack_from('i', f_bytes, pos)[0]; pos += 4
    if n % 8 == 0:
        vals = np.frombuffer(f_bytes, dtype=np.float64, count=n//8, offset=pos)
    else:
        vals = np.frombuffer(f_bytes, dtype=np.int32, count=n//4, offset=pos)
    pos += n
    end = struct.unpack_from('i', f_bytes, pos)[0]; pos += 4
    assert n == end, f"Record mismatch {n} != {end}"
    return vals, pos

# Identity
identity, pos = read_record(data, pos)
print(f"Identity: {identity}")

# 6 eigenvectors (1 spin, 1 kpt, 6 states, 6 orbs each)
evecs = []
for i in range(6):
    ev, pos = read_record(data, pos)
    evecs.append(ev)
    print(f"MO {i+1}: {ev}")

evecs = np.array(evecs)  # shape (nstates, norb)
print(f"\nEigenvec matrix (nstates=6, norb=6):\n{evecs}")
print(f"Orbital labels (DFTB order): O:s, O:py, O:pz, O:px, H1:s, H2:s")

# energies from band.out
energies = [-23.0459, -11.0052, -8.7497, -7.0553, 9.0574, 13.8781]
for i, (e, occ) in enumerate(zip(energies, [2,2,2,2,0,0])):
    print(f"MO{i+1} E={e:8.4f} eV  occ={occ}  coeffs: {evecs[i]}")
