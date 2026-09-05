#!/usr/bin/env python3
"""
geometry_engine.py — Build H-bonded molecular complexes from donor/acceptor vectors.

Uses lone-pair direction computation (ported from SPAMMM AtomicSystem.add_electron_pairs)
to find H-bond donor (X-H) and acceptor (X-lone-pair) vectors, then places molecules
so complementary vectors face each other.

Usage:
  python3 geometry_engine.py --azaindole data/xyz/azaindol.xyz --formic data/xyz/formic_dimer.xyz --out data/xyz/formic_azaindole_dimer.xyz --plot debug/hbond_geom.png
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── XYZ I/O ───────────────────────────────────────────────────────
def load_xyz(path):
    with open(path) as f:
        n = int(f.readline()); comment = f.readline().strip()
        atoms = []
        for _ in range(n):
            p = f.readline().split()
            atoms.append((p[0], np.array([float(p[1]), float(p[2]), float(p[3])])))
    return atoms, comment

def save_xyz(path, atoms, comment=""):
    with open(path, 'w') as f:
        f.write(f"{len(atoms)}\n{comment}\n")
        for s, c in atoms:
            f.write(f"{s:2s} {c[0]:12.6f} {c[1]:12.6f} {c[2]:12.6f}\n")

# ─── Bond finding (from atomicUtils.findBondsNP) ───────────────────
COV_RADII = {'H':0.31, 'C':0.76, 'N':0.71, 'O':0.66, 'S':1.05, 'P':1.07}

def find_bonds(atoms, factor=0.5):
    """Return list of (i,j) bonded pairs using covalent radii sum * factor."""
    bonds = []
    n = len(atoms)
    for i in range(n):
        for j in range(i+1, n):
            ri = COV_RADII.get(atoms[i][0], 0.7)
            rj = COV_RADII.get(atoms[j][0], 0.7)
            d = np.linalg.norm(atoms[i][1] - atoms[j][1])
            if d < (ri + rj) * 1.3:  # slightly generous
                bonds.append((i, j))
    return bonds

def neighbor_list(atoms, bonds):
    """Return list of sets of neighbor indices."""
    neighs = [set() for _ in atoms]
    for i, j in bonds:
        neighs[i].add(j)
        neighs[j].add(i)
    return neighs

# ─── Lone pair directions (ported from SPAMMM AtomicSystem.make_epair_geom) ──
# VALENCE_DICT: (nBond_total, nElectronPairs)
VALENCE_DICT = {'O': (2, 2), 'N': (3, 1)}

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def get_pi_direction(atoms, neighs, i):
    """Average normal to the plane of neighbors (for aromatic systems)."""
    neighbors = list(neighs[i])
    vecs = [normalize(atoms[j][1] - atoms[i][1]) for j in neighbors]
    d = np.zeros(3)
    for a, b in zip(vecs, vecs[1:] + [vecs[0]]):
        d += normalize(np.cross(a, b))
    return normalize(d)

def compute_lone_pairs(atoms, neighs, i):
    """
    Compute lone pair directions for atom i.
    Returns list of unit vectors pointing outward from atom i.
    Ported from SPAMMM AtomicSystem.make_epair_geom.
    
    Logic:
      nb     = total valence (O:2, N:3)
      nep    = number of electron pairs (O:2, N:1)
      nsigma = actual number of bonded neighbors
      npi    = nb - nsigma  (number of pi bonds)
    """
    ename = atoms[i][0]
    if ename not in VALENCE_DICT:
        return []
    
    nb_val = VALENCE_DICT[ename][0]  # total valence
    nep    = VALENCE_DICT[ename][1]  # number of electron pairs
    nsigma = len(neighs[i])
    npi    = nb_val - nsigma
    
    pos = atoms[i][1]
    neighbors = list(neighs[i])
    vs = [normalize(atoms[j][1] - pos) for j in neighbors]
    
    pairs = []
    
    if npi == 0:  # sp3-like
        if nsigma == 3:  # like NH3
            v1, v2, v3 = vs
            base = normalize(np.cross(v2 - v1, v3 - v1))
            if np.dot(base, v1 + v2 + v3) > 0:
                base = -base
            pairs.append(base)
        elif nsigma == 2:  # like H2O
            v1, v2 = vs
            m_c = normalize(v1 + v2)  # bisector
            m_b = normalize(np.cross(v1, v2))
            cc = 0.57735026919  # 1/sqrt(3)
            cb = 0.81649658092  # sqrt(2/3)
            pairs.append(normalize(m_c * -cc + m_b * cb))
            pairs.append(normalize(m_c * -cc - m_b * cb))
    elif npi == 1:  # sp2-like (one pi bond)
        if nsigma == 2:  # like =N- (pyridinic N)
            v1, v2 = vs
            m_c = normalize(v1 + v2)  # bisector
            pairs.append(m_c * -1.)  # lone pair opposite to bisector
        elif nsigma == 1:  # like =O (carbonyl)
            v1 = vs[0]
            # Get pi direction from the neighbor (the C of C=O)
            j = neighbors[0]
            m_b = get_pi_direction(atoms, neighs, j)
            m_c = normalize(np.cross(v1, m_b))
            pairs.append(normalize(v1 * -0.5 + m_c * 0.86602540378))
            pairs.append(normalize(v1 * -0.5 - m_c * 0.86602540378))
    elif npi == 2:  # sp-like (triple bond)
        if nsigma == 1:  # like ≡N
            pairs.append(vs[0] * -1.)
    
    return pairs

def find_hbonds(atoms, neighs):
    """
    Find H-bond donor and acceptor sites.
    
    Donors: X-H bonds where X is N or O (the H points outward).
    Acceptors: lone pairs on N or O (the lone pair points outward).
    
    Returns:
      donors:   list of (atom_idx, H_idx, direction) — direction is X→H (outward)
      acceptors: list of (atom_idx, lone_pair_idx, direction) — direction is outward
    """
    donors = []
    acceptors = []
    
    for i, (s, c) in enumerate(atoms):
        if s in ('N', 'O'):
            # Find H neighbors → donor sites
            for j in neighs[i]:
                if atoms[j][0] == 'H':
                    d = normalize(atoms[j][1] - c)
                    donors.append((i, j, d))
            # Compute lone pairs → acceptor sites
            lps = compute_lone_pairs(atoms, neighs, i)
            for k, lp in enumerate(lps):
                acceptors.append((i, k, lp))
    
    return donors, acceptors

# ─── Geometry transforms ───────────────────────────────────────────
def rotmat(axis, angle):
    """Rodrigues rotation matrix."""
    axis = normalize(axis)
    ca, sa = np.cos(angle), np.sin(angle)
    kx, ky, kz = axis
    v = 1 - ca
    return np.array([
        [kx*kx*v + ca,     kx*ky*v - kz*sa,  kx*kz*v + ky*sa],
        [kx*ky*v + kz*sa,  ky*ky*v + ca,     ky*kz*v - kx*sa],
        [kx*kz*v - ky*sa,  ky*kz*v + kx*sa,  kz*kz*v + ca   ]
    ])

def rotmat_align(v_from, v_to):
    """Rotation matrix to align unit vector v_from -> v_to."""
    v_from = normalize(v_from)
    v_to = normalize(v_to)
    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)
    if np.linalg.norm(cross) < 1e-10:
        return np.eye(3) if dot > 0 else np.diag([1, -1, -1])
    axis = normalize(cross)
    angle = np.arccos(np.clip(dot, -1, 1))
    return rotmat(axis, angle)

def transform_atoms(atoms, R, t):
    """Apply rotation R then translation t to all atoms."""
    return [(s, R @ c + t) for s, c in atoms]

# ─── H-bond complex builder ────────────────────────────────────────
def build_hbond_complex(mol_A, mol_B, hbond_dist=2.9, clash_cutoff=1.8):
    """
    Place mol_B near mol_A so that donor/acceptor vectors match.
    
    Strategy:
      1. Find all donors and acceptors in both molecules.
      2. For each (donor_A, acceptor_B) and (donor_B, acceptor_A) pair:
         a. Align donor X-H vector anti-parallel to acceptor lone-pair vector.
         b. Place molecules at hbond_dist apart.
         c. Check for clashes.
      3. If two H-bonds can form simultaneously, prefer that placement.
    
    Returns: (placed_atoms, info_dict) or (None, info)
    """
    bonds_A = find_bonds(mol_A)
    bonds_B = find_bonds(mol_B)
    neighs_A = neighbor_list(mol_A, bonds_A)
    neighs_B = neighbor_list(mol_B, bonds_B)
    
    donors_A, acceptors_A = find_hbonds(mol_A, neighs_A)
    donors_B, acceptors_B = find_hbonds(mol_B, neighs_B)
    
    print(f"Mol A: {len(donors_A)} donors, {len(acceptors_A)} acceptors")
    for i, h, d in donors_A:
        print(f"  donor: atom {i} ({mol_A[i][0]}) - H {h}, dir={d}")
    for i, k, d in acceptors_A:
        print(f"  acceptor: atom {i} ({mol_A[i][0]}), lp {k}, dir={d}")
    
    print(f"Mol B: {len(donors_B)} donors, {len(acceptors_B)} acceptors")
    for i, h, d in donors_B:
        print(f"  donor: atom {i} ({mol_B[i][0]}) - H {h}, dir={d}")
    for i, k, d in acceptors_B:
        print(f"  acceptor: atom {i} ({mol_B[i][0]}), lp {k}, dir={d}")
    
    # Generate all possible H-bond pairings:
    # donor_A → acceptor_B  (A donates H to B's lone pair)
    # donor_B → acceptor_A  (B donates H to A's lone pair)
    pairings = []
    for da in donors_A:
        for ab in acceptors_B:
            pairings.append(('A→B', da, ab))
    for db in donors_B:
        for aa in acceptors_A:
            pairings.append(('B→A', db, aa))
    
    print(f"\n{len(pairings)} possible H-bond pairings")
    
    # For single H-bond placement:
    # 1. Rotate mol_B so donor_X→H vector is anti-parallel to acceptor_lone_pair vector
    # 2. Translate so H...acceptor distance = hbond_dist
    
    best = None
    best_score = 999
    
    # Try all pairs of pairings (for double H-bond) — PREFER these
    print("\n--- Double H-bond search ---")
    best_double = None
    best_double_score = 999
    for i, p1 in enumerate(pairings):
        for j, p2 in enumerate(pairings):
            if j <= i:
                continue
            d1_atom = p1[1][0]; a1_atom = p1[2][0]
            d2_atom = p2[1][0]; a2_atom = p2[2][0]
            if d1_atom == d2_atom or a1_atom == a2_atom:
                continue
            placed, info = try_double_hbond(mol_A, mol_B, p1, p2, hbond_dist, clash_cutoff)
            if placed is not None:
                score = info['score']
                print(f"  pair ({i},{j}): {p1[0]}+{p2[0]} score={score:.3f} d1={info['d1_don_acc']:.2f} d2={info['d2_don_acc']:.2f} clash={info['min_inter']:.2f}")
                if score < best_double_score:
                    best_double_score = score
                    best_double = (placed, info)
            else:
                print(f"  pair ({i},{j}): {p1[0]}+{p2[0]} FAILED: {info.get('error','?')}")
    
    # Also try single H-bonds (fallback only)
    print("\n--- Single H-bond search (fallback) ---")
    for p1 in pairings:
        placed, info = try_single_hbond(mol_A, mol_B, p1, hbond_dist, clash_cutoff)
        if placed is not None:
            score = info['score']
            print(f"  single: {p1[0]} score={score:.3f} d={info['d_don_acc']:.2f} clash={info['min_inter']:.2f}")
            # Only use single if no double found, or single is much better
            if best_double is None and score < best_score:
                best_score = score
                best = (placed, info)
    
    if best_double is not None:
        best = best_double
        best_score = best_double_score
    
    if best is None:
        return None, {'error': 'No valid placement found'}
    
    return best

def try_single_hbond(mol_A, mol_B, pairing, hbond_dist, clash_cutoff):
    """
    Place mol_B to form one H-bond with mol_A.
    pairing: (direction, (donor_atom, H_atom, donor_dir), (acceptor_atom, lp_idx, acceptor_dir))
    """
    direction, (don_atom, h_atom, don_dir), (acc_atom, lp_idx, acc_dir) = pairing
    
    if direction == 'A→B':
        # A donates, B accepts
        # donor_dir is in mol_A frame, acc_dir is in mol_B frame
        # We need to rotate mol_B so acc_dir becomes anti-parallel to don_dir
        # Then place so H of A is at hbond_dist from acceptor atom of B
        
        # Target: acc_dir (in B) should point toward H of A (i.e., -don_dir direction)
        # So align acc_dir -> -don_dir
        R = rotmat_align(acc_dir, -don_dir)
        mol_B_rot = transform_atoms(mol_B, R, np.zeros(3))
        
        # Recompute acceptor atom position after rotation
        acc_pos_B = mol_B_rot[acc_atom][1]
        acc_dir_rot = normalize(acc_pos_B - mol_B_rot[acc_atom][1])  # this is wrong, need to recompute
        # Actually the lone pair direction rotates with the molecule:
        acc_dir_rot = R @ acc_dir
        
        # H position in A
        h_pos_A = mol_A[h_atom][1]
        don_atom_pos_A = mol_A[don_atom][1]
        
        # Place acceptor atom of B at distance hbond_dist from H of A
        # along the donor direction (X→H extended)
        target_acc_pos = h_pos_A + normalize(h_pos_A - don_atom_pos_A) * (hbond_dist - 1.0)
        # Actually: H...acceptor distance should be ~1.9 Å (typical H-bond H...X)
        # O...N distance ~2.9 Å, H...N ~1.9 Å (O-H is ~1.0 Å)
        h_to_acc_dist = hbond_dist - 1.0  # approximate H...acceptor distance
        target_acc_pos = h_pos_A + normalize(h_pos_A - don_atom_pos_A) * h_to_acc_dist
        
        t = target_acc_pos - acc_pos_B
        mol_B_placed = transform_atoms(mol_B_rot, np.eye(3), t)
        
    else:  # B→A
        # B donates, A accepts
        R = rotmat_align(don_dir, -acc_dir)
        mol_B_rot = transform_atoms(mol_B, R, np.zeros(3))
        
        don_dir_rot = R @ don_dir
        h_pos_B = mol_B_rot[h_atom][1]
        don_atom_pos_B = mol_B_rot[don_atom][1]
        
        acc_pos_A = mol_A[acc_atom][1]
        
        # Place H of B at ~1.9 Å from acceptor atom of A along acceptor lone pair direction
        target_h_pos = acc_pos_A + acc_dir * 1.9
        t = target_h_pos - h_pos_B
        mol_B_placed = transform_atoms(mol_B_rot, np.eye(3), t)
    
    # Check clashes
    all_atoms = mol_A + mol_B_placed
    min_inter = min(
        np.linalg.norm(mol_A[i][1] - mol_B_placed[j][1])
        for i in range(len(mol_A)) for j in range(len(mol_B_placed))
    )
    
    if min_inter < clash_cutoff:
        return None, {'error': f'clash {min_inter:.2f} < {clash_cutoff}'}
    
    # Compute H-bond quality
    if direction == 'A→B':
        d_h_acc = np.linalg.norm(mol_A[h_atom][1] - mol_B_placed[acc_atom][1])
        d_don_acc = np.linalg.norm(mol_A[don_atom][1] - mol_B_placed[acc_atom][1])
    else:
        d_h_acc = np.linalg.norm(mol_B_placed[h_atom][1] - mol_A[acc_atom][1])
        d_don_acc = np.linalg.norm(mol_B_placed[don_atom][1] - mol_A[acc_atom][1])
    
    score = abs(d_don_acc - hbond_dist) * 2 + abs(d_h_acc - 1.9)
    
    info = {
        'type': 'single',
        'direction': direction,
        'd_don_acc': d_don_acc,
        'd_h_acc': d_h_acc,
        'min_inter': min_inter,
        'score': score,
    }
    return mol_B_placed, info

def try_double_hbond(mol_A, mol_B, p1, p2, hbond_dist, clash_cutoff):
    """
    Try to place mol_B to satisfy two H-bonds simultaneously.
    Uses least-squares fitting of the two donor/acceptor vector pairs.
    """
    dir1, (don1, h1, don_dir1), (acc1, lp1, acc_dir1) = p1
    dir2, (don2, h2, don_dir2), (acc2, lp2, acc_dir2) = p2
    
    # Determine which mol is donor/acceptor in each pairing
    # For 'A→B': A donates (don_dir1 in A frame), B accepts (acc_dir1 in B frame)
    # For 'B→A': B donates (don_dir1 in B frame), A accepts (acc_dir1 in A frame)
    
    # Collect vectors in A frame and B frame
    # We want to find R, t such that:
    #   R @ B_vectors ≈ A_vectors (after translation)
    
    # Two anchor points: the donor H positions and acceptor atom positions
    
    if dir1 == 'A→B' and dir2 == 'A→B':
        # Both: A donates to B acceptors
        # Anchor in A: H atoms (h1, h2)
        # Anchor in B: acceptor atoms (acc1, acc2)
        # We want: H_A1 ... acc_B1 at hbond_dist, H_A2 ... acc_B2 at hbond_dist
        # And: don_dir1 (A) anti-parallel to acc_dir1 (B after rotation)
        
        # Step 1: align acc_dir1 (B) to -don_dir1 (A)
        R = rotmat_align(acc_dir1, -don_dir1)
        mol_B_rot = transform_atoms(mol_B, R, np.zeros(3))
        
        # Step 2: translate so acc1_B is at H1_A + don_dir1_extended * (hbond_dist - 1.0)
        h1_pos = mol_A[h1][1]
        don1_pos = mol_A[don1][1]
        don_h_dir = normalize(h1_pos - don1_pos)
        target_acc1 = h1_pos + don_h_dir * (hbond_dist - 1.0)
        t = target_acc1 - mol_B_rot[acc1][1]
        mol_B_placed = transform_atoms(mol_B_rot, np.eye(3), t)
        
        # Check second H-bond quality
        h2_pos = mol_A[h2][1]
        acc2_pos = mol_B_placed[acc2][1]
        d_h2_acc2 = np.linalg.norm(h2_pos - acc2_pos)
        
    elif dir1 == 'B→A' and dir2 == 'B→A':
        # Both: B donates to A acceptors
        R = rotmat_align(don_dir1, -acc_dir1)
        mol_B_rot = transform_atoms(mol_B, R, np.zeros(3))
        
        acc1_pos = mol_A[acc1][1]
        target_h1 = acc1_pos + acc_dir1 * 1.9
        t = target_h1 - mol_B_rot[h1][1]
        mol_B_placed = transform_atoms(mol_B_rot, np.eye(3), t)
        
        acc2_pos = mol_A[acc2][1]
        h2_pos = mol_B_placed[h2][1]
        d_h2_acc2 = np.linalg.norm(h2_pos - acc2_pos)
        
    else:
        # Mixed: one A→B, one B→A
        # Normalize: make p1 = A→B, p2 = B→A
        if dir1 == 'B→A':
            p1, p2 = p2, p1
            dir1, (don1, h1, don_dir1), (acc1, lp1, acc_dir1) = p1
            dir2, (don2, h2, don_dir2), (acc2, lp2, acc_dir2) = p2
        
        # p1: A donates (don_dir1 in A), B accepts (acc_dir1 in B)
        # p2: B donates (don_dir2 in B), A accepts (acc_dir2 in A)
        
        # We need to find R such that:
        #   R @ acc_dir1 ≈ -don_dir1  (B's acceptor faces A's donor)
        #   R @ don_dir2 ≈ -acc_dir2  (B's donor faces A's acceptor)
        
        # This is a Procrustes problem: find R that best aligns two vector pairs
        # B vectors: [acc_dir1, don_dir2]
        # A target vectors: [-don_dir1, -acc_dir2]
        
        B_vecs = np.array([acc_dir1, don_dir2])
        A_vecs = np.array([-don_dir1, -acc_dir2])
        
        # Kabsch algorithm for vectors (not points)
        H = B_vecs.T @ A_vecs
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # Ensure proper rotation (det = +1)
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        
        mol_B_rot = transform_atoms(mol_B, R, np.zeros(3))
        
        # Translation: place using first H-bond (A→B)
        h1_pos = mol_A[h1][1]
        don1_pos = mol_A[don1][1]
        don_h_dir = normalize(h1_pos - don1_pos)
        target_acc1 = h1_pos + don_h_dir * (hbond_dist - 1.0)
        t = target_acc1 - mol_B_rot[acc1][1]
        mol_B_placed = transform_atoms(mol_B_rot, np.eye(3), t)
        
        # Check second H-bond
        acc2_pos = mol_A[acc2][1]
        h2_pos = mol_B_placed[h2][1]
        d_h2_acc2 = np.linalg.norm(h2_pos - acc2_pos)
    
    # Clashes
    min_inter = min(
        np.linalg.norm(mol_A[i][1] - mol_B_placed[j][1])
        for i in range(len(mol_A)) for j in range(len(mol_B_placed))
    )
    if min_inter < clash_cutoff:
        return None, {'error': f'clash {min_inter:.2f}'}
    
    # H-bond distances
    if dir1 == 'A→B':
        d1_h_acc = np.linalg.norm(mol_A[h1][1] - mol_B_placed[acc1][1])
        d1_don_acc = np.linalg.norm(mol_A[don1][1] - mol_B_placed[acc1][1])
    else:
        d1_h_acc = np.linalg.norm(mol_B_placed[h1][1] - mol_A[acc1][1])
        d1_don_acc = np.linalg.norm(mol_B_placed[don1][1] - mol_A[acc1][1])
    
    if dir2 == 'A→B':
        d2_h_acc = np.linalg.norm(mol_A[h2][1] - mol_B_placed[acc2][1])
        d2_don_acc = np.linalg.norm(mol_A[don2][1] - mol_B_placed[acc2][1])
    else:
        d2_h_acc = np.linalg.norm(mol_B_placed[h2][1] - mol_A[acc2][1])
        d2_don_acc = np.linalg.norm(mol_B_placed[don2][1] - mol_A[acc2][1])
    
    score = abs(d1_don_acc - hbond_dist) + abs(d2_don_acc - hbond_dist) + \
            abs(d1_h_acc - 1.9) + abs(d2_h_acc - 1.9)
    
    info = {
        'type': 'double',
        'dir1': dir1, 'dir2': dir2,
        'd1_don_acc': d1_don_acc, 'd1_h_acc': d1_h_acc,
        'd2_don_acc': d2_don_acc, 'd2_h_acc': d2_h_acc,
        'min_inter': min_inter,
        'score': score,
    }
    return mol_B_placed, info

# ─── Plotting ──────────────────────────────────────────────────────
def plot_complex(mol_A, mol_B_placed, info, out_path, donors_A=None, acceptors_A=None,
                 donors_B=None, acceptors_B=None):
    """Plot the H-bonded complex in XY plane with labeled atoms and H-bonds."""
    colors = {'H': 'white', 'C': 'black', 'N': 'blue', 'O': 'red', 'E': 'green'}
    sizes = {'H': 50, 'C': 100, 'N': 130, 'O': 130, 'E': 40}
    
    all_atoms = mol_A + mol_B_placed
    n_A = len(mol_A)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw bonds
    for start, end, edge in [(0, n_A, 'blue'), (n_A, len(all_atoms), 'red')]:
        mol = all_atoms[start:end]
        for i in range(len(mol)):
            for j in range(i+1, len(mol)):
                d = np.linalg.norm(mol[i][1] - mol[j][1])
                if d < 1.7:
                    ax.plot([mol[i][1][0], mol[j][1][0]],
                            [mol[i][1][1], mol[j][1][1]], 'k-', alpha=0.3, linewidth=0.5)
    
    # Draw atoms
    for i, (s, c) in enumerate(all_atoms):
        is_A = i < n_A
        edge = 'blue' if is_A else 'red'
        ax.scatter(c[0], c[1], c=colors.get(s, 'gray'), s=sizes.get(s, 80),
                   edgecolors=edge, linewidths=1.5, zorder=5)
        ax.annotate(f'{i}:{s}', (c[0], c[1]), fontsize=5, xytext=(3, 3),
                    textcoords='offset points')
    
    # Draw lone pair directions (acceptors)
    if acceptors_A:
        for atom_i, lp_i, d in acceptors_A:
            pos = mol_A[atom_i][1]
            ax.arrow(pos[0], pos[1], d[0]*0.8, d[1]*0.8, head_width=0.08, head_length=0.05,
                     fc='cyan', ec='cyan', alpha=0.6, zorder=4)
    if acceptors_B:
        for atom_i, lp_i, d in acceptors_B:
            pos = mol_B_placed[atom_i][1]
            ax.arrow(pos[0], pos[1], d[0]*0.8, d[1]*0.8, head_width=0.08, head_length=0.05,
                     fc='orange', ec='orange', alpha=0.6, zorder=4)
    
    # Draw donor X-H directions
    if donors_A:
        for atom_i, h_i, d in donors_A:
            pos = mol_A[atom_i][1]
            h_pos = mol_A[h_i][1]
            ax.plot([pos[0], h_pos[0]], [pos[1], h_pos[1]], color='cyan', linewidth=1.5, alpha=0.6)
    if donors_B:
        for atom_i, h_i, d in donors_B:
            pos = mol_B_placed[atom_i][1]
            h_pos = mol_B_placed[h_i][1]
            ax.plot([pos[0], h_pos[0]], [pos[1], h_pos[1]], color='orange', linewidth=1.5, alpha=0.6)
    
    # Title with info
    title = f"H-bond complex (blue=A, red=B)\n"
    for k, v in info.items():
        if isinstance(v, float):
            title += f"{k}={v:.3f}  "
        else:
            title += f"{k}={v}  "
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {out_path}")

# ─── Main ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Build H-bonded molecular complexes")
    ap.add_argument('--azaindole', required=True, help='XYZ file for molecule A (azaindole)')
    ap.add_argument('--formic', required=True, help='XYZ file for molecule B (formic acid)')
    ap.add_argument('--out', required=True, help='Output XYZ file')
    ap.add_argument('--plot', default=None, help='Output plot PNG')
    ap.add_argument('--hbond-dist', type=float, default=2.9, help='Target O...N distance (Å)')
    ap.add_argument('--clash-cutoff', type=float, default=1.8, help='Min inter-molecular distance (Å)')
    args = ap.parse_args()
    
    mol_A, comment_A = load_xyz(args.azaindole)
    mol_B_raw, comment_B = load_xyz(args.formic)
    
    # If formic dimer, take first monomer (5 atoms)
    if len(mol_B_raw) == 10:
        print(f"Formic dimer detected, taking first monomer (atoms 0-4)")
        mol_B = mol_B_raw[:5]
    else:
        mol_B = mol_B_raw
    
    # Flatten both to XY plane (they should already be planar)
    mol_A = [(s, np.array([c[0], c[1], 0.0])) for s, c in mol_A]
    mol_B = [(s, np.array([c[0], c[2], 0.0])) for s, c in mol_B]  # formic acid is in XZ
    
    print(f"\nMol A ({args.azaindole}): {len(mol_A)} atoms")
    print(f"Mol B ({args.formic}): {len(mol_B)} atoms")
    
    # Compute donors and acceptors for plotting
    bonds_A = find_bonds(mol_A)
    bonds_B = find_bonds(mol_B)
    neighs_A = neighbor_list(mol_A, bonds_A)
    neighs_B = neighbor_list(mol_B, bonds_B)
    donors_A, acceptors_A = find_hbonds(mol_A, neighs_A)
    donors_B, acceptors_B = find_hbonds(mol_B, neighs_B)
    
    # Build complex
    print("\n=== Building H-bond complex ===")
    result, info = build_hbond_complex(mol_A, mol_B, args.hbond_dist, args.clash_cutoff)
    
    if result is None:
        print(f"FAILED: {info.get('error', 'unknown')}")
        sys.exit(1)
    
    print(f"\n=== Best placement ===")
    for k, v in info.items():
        if isinstance(v, float):
            print(f"  {k} = {v:.4f}")
        else:
            print(f"  {k} = {v}")
    
    # Save
    all_atoms = mol_A + result
    n_h = sum(1 for s, _ in all_atoms if s == 'H')
    n_heavy = sum(1 for s, _ in all_atoms if s in ('C', 'N', 'O'))
    n_orb = n_h * 1 + n_heavy * 4
    save_xyz(args.out, all_atoms, f"H-bond complex: {n_orb} orbitals")
    print(f"\nSaved {len(all_atoms)} atoms ({n_orb} orbitals) to {args.out}")
    
    # Plot
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or '.', exist_ok=True)
        # Recompute donors/acceptors for placed mol_B
        bonds_Bp = find_bonds(result)
        neighs_Bp = neighbor_list(result, bonds_Bp)
        donors_Bp, acceptors_Bp = find_hbonds(result, neighs_Bp)
        plot_complex(mol_A, result, info, args.plot,
                     donors_A, acceptors_A, donors_Bp, acceptors_Bp)

if __name__ == '__main__':
    main()
