//! Geometry generators for carbon nanostructures (graphene ribbons, sheets,
//! flakes, PAHs) on a honeycomb lattice.
//!
//! Ported from:
//! - SPAMMM `spammm/topology/MoleculeEditorBackend.py:honeycomb_ring_nodes` (L48-67)
//! - SPAMMM `spammm/topology/MoleculeEditorBackend.py:_build_strip_ribbon` (L2028-2138)
//! - NumericalMathPlayground `topics/LinearScalingQM/KekuleFluid/Graph.py:HoneycombGraph` (L59-360)
//!
//! All builders return a [`NanoStructure`] containing atom positions, element
//! types, and bond connectivity. Positions are in Angstroms, z=0 for 2D
//! systems. The honeycomb lattice uses the standard graphene convention:
//!   a1 = (3/2, √3/2)·a_CC,  a2 = (3/2, -√3/2)·a_CC
//!   δ0 = (1, 0)·a_CC,  δ1 = (-1/2, √3/2)·a_CC,  δ2 = (-1/2, -√3/2)·a_CC
//! where a_CC = 1.42 Å is the C-C bond length.

/// √3 constant (avoids unstable `std::f64::consts::SQRT_3`).
const SQRT_3: f64 = 1.73205080756887729353;

/// Default C-C bond length in graphene (Angstrom).
pub const A_CC: f64 = 1.42;

/// Element types in the structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    C,
    H,
    N,
    B,
    O,
    F,
    Si,
    P,
    S,
    Cl,
}

impl Element {
    pub fn symbol(&self) -> &'static str {
        match self {
            Element::C => "C",
            Element::H => "H",
            Element::N => "N",
            Element::B => "B",
            Element::O => "O",
            Element::F => "F",
            Element::Si => "Si",
            Element::P => "P",
            Element::S => "S",
            Element::Cl => "Cl",
        }
    }
    pub fn from_symbol(s: &str) -> Option<Element> {
        match s.trim() {
            "C" => Some(Element::C),
            "H" => Some(Element::H),
            "N" => Some(Element::N),
            "B" => Some(Element::B),
            "O" => Some(Element::O),
            "F" => Some(Element::F),
            "Si" => Some(Element::Si),
            "P" => Some(Element::P),
            "S" => Some(Element::S),
            "Cl" => Some(Element::Cl),
            _ => None,
        }
    }
    /// Covalent radius (Angstrom) for H passivation bond length estimation.
    pub fn covalent_radius(&self) -> f64 {
        match self {
            Element::C => 0.77,
            Element::H => 0.31,
            Element::N => 0.71,
            Element::B => 0.82,
            Element::O => 0.66,
            Element::F => 0.57,
            Element::Si => 1.11,
            Element::P => 1.07,
            Element::S => 1.05,
            Element::Cl => 0.99,
        }
    }
    /// s/p DFTB valence electrons. Not SK onsite `q0` (parser reads trailing fields).
    pub fn valence_electrons(&self) -> f64 {
        match self {
            Element::H => 1.0,
            Element::B => 3.0,
            Element::C => 4.0,
            Element::N => 5.0,
            Element::O => 6.0,
            Element::F => 7.0,
            Element::Si => 4.0,
            Element::P => 5.0,
            Element::S => 6.0,
            Element::Cl => 7.0,
        }
    }
}

/// A built nanostructure: atoms + bonds.
#[derive(Debug, Clone)]
pub struct NanoStructure {
    /// Element type per atom.
    pub elements: Vec<Element>,
    /// Cartesian positions [x, y, z] per atom (Angstrom).
    pub positions: Vec<[f64; 3]>,
    /// Bond list as pairs of atom indices.
    pub bonds: Vec<[usize; 2]>,
}

impl NanoStructure {
    pub fn new() -> Self {
        Self { elements: Vec::new(), positions: Vec::new(), bonds: Vec::new() }
    }
    pub fn natom(&self) -> usize { self.positions.len() }
    pub fn nbond(&self) -> usize { self.bonds.len() }

    /// Add an atom, return its index.
    pub fn add_atom(&mut self, el: Element, pos: [f64; 3]) -> usize {
        let i = self.positions.len();
        self.elements.push(el);
        self.positions.push(pos);
        i
    }

    /// Add a bond (no duplicate check).
    pub fn add_bond(&mut self, i: usize, j: usize) {
        self.bonds.push([i, j]);
    }

    /// Write to XYZ format (one molecule, no PBC).
    pub fn to_xyz(&self) -> String {
        let mut s = format!("{}\n\n", self.natom());
        for i in 0..self.natom() {
            let [x, y, z] = self.positions[i];
            s += &format!("{:2}  {:18.10}  {:18.10}  {:18.10}\n",
                self.elements[i].symbol(), x, y, z);
        }
        s
    }

    /// Write to extended XYZ with lattice vectors (for periodic systems).
    /// `lattice` is optional [a1x,a1y,a1z, a2x,a2y,a2z, a3x,a3y,a3z].
    pub fn to_xyz_periodic(&self, lattice: Option<&[[f64; 3]; 3]>) -> String {
        let mut s = format!("{}\n", self.natom());
        if let Some(lat) = lattice {
            let [a1, a2, a3] = lat;
            s += &format!(
                "Lattice=\"{:18.10} {:18.10} {:18.10} {:18.10} {:18.10} {:18.10} {:18.10} {:18.10} {:18.10}\" Properties=species:S:1:pos:R:3\n",
                a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a3[0], a3[1], a3[2]
            );
        } else {
            s += "Properties=species:S:1:pos:R:3\n";
        }
        for i in 0..self.natom() {
            let [x, y, z] = self.positions[i];
            s += &format!("{:2}  {:18.10}  {:18.10}  {:18.10}\n",
                self.elements[i].symbol(), x, y, z);
        }
        s
    }

    /// Find edge atoms (carbon atoms with < 3 carbon neighbors).
    pub fn find_edge_carbons(&self) -> Vec<usize> {
        let mut c_neighbors = vec![0usize; self.natom()];
        for &[i, j] in &self.bonds {
            if self.elements[i] == Element::C && self.elements[j] == Element::C {
                c_neighbors[i] += 1;
                c_neighbors[j] += 1;
            }
        }
        (0..self.natom())
            .filter(|&i| self.elements[i] == Element::C && c_neighbors[i] < 3)
            .collect()
    }

    /// Count atoms by element.
    pub fn count(&self, el: Element) -> usize {
        self.elements.iter().filter(|&&e| e == el).count()
    }

    /// Bounding box (min, max) in each dimension.
    pub fn bbox(&self) -> ([f64; 3], [f64; 3]) {
        if self.positions.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }
        let mut lo = self.positions[0];
        let mut hi = self.positions[0];
        for &p in &self.positions {
            for d in 0..3 {
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
        }
        (lo, hi)
    }

    /// Recenter all atoms so the centroid is at origin.
    pub fn recenter(&mut self) {
        let n = self.natom() as f64;
        let mut c = [0.0f64; 3];
        for &p in &self.positions {
            c[0] += p[0]; c[1] += p[1]; c[2] += p[2];
        }
        c[0] /= n; c[1] /= n; c[2] /= n;
        for p in &mut self.positions {
            p[0] -= c[0]; p[1] -= c[1]; p[2] -= c[2];
        }
    }
}

// ─── Honeycomb lattice primitives ───────────────────────────────────

/// Honeycomb lattice vectors (graphene convention).
/// a1 = (3/2, √3/2)·a_CC,  a2 = (3/2, -√3/2)·a_CC
pub fn lattice_vectors(a_cc: f64) -> [[f64; 3]; 2] {
    [[1.5 * a_cc, 0.5 * SQRT_3 * a_cc, 0.0],
     [1.5 * a_cc, -0.5 * SQRT_3 * a_cc, 0.0]]
}

/// Three nearest-neighbor bond vectors from sublattice A to B.
/// δ0 = (1, 0)·a_CC,  δ1 = (-1/2, √3/2)·a_CC,  δ2 = (-1/2, -√3/2)·a_CC
pub fn nn_deltas(a_cc: f64) -> [[f64; 3]; 3] {
    [[a_cc, 0.0, 0.0],
     [-0.5 * a_cc, 0.5 * SQRT_3 * a_cc, 0.0],
     [-0.5 * a_cc, -0.5 * SQRT_3 * a_cc, 0.0]]
}

/// Position key for deduplication (rounded to 4 decimals).
fn pos_key(p: &[f64; 3]) -> (i64, i64) {
    ((p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64)
}

/// 6 vertices of a hexagonal ring at axial coordinates (q, r).
/// Ported from SPAMM `honeycomb_ring_nodes(q, r, a_CC)` (L48-67).
pub fn honeycomb_ring_nodes(q: i32, r: i32, a_cc: f64) -> [[f64; 2]; 6] {
    let cx = a_cc * SQRT_3 * (q as f64 + r as f64 * 0.5);
    let cy = a_cc * 1.5 * r as f64;
    let mut nodes = [[0.0f64; 2]; 6];
    for i in 0..6 {
        let angle = std::f64::consts::PI * (i as f64 / 6.0 + 0.5); // start at 30°
        nodes[i] = [cx + a_cc * angle.cos(), cy + a_cc * angle.sin()];
    }
    nodes
}

// ─── Builders ───────────────────────────────────────────────────────

/// Build a zigzag graphene ribbon.
///
/// `width_chains`: number of atom rows across the ribbon width.
/// `length_cells`: number of unit cells along the ribbon length (x direction).
/// `passivate`: if true, add H atoms to undercoordinated edge carbons.
/// `periodic_x`: if true, ribbon is periodic along x (armchair edges wrap).
///
/// Ported from SPAMM `MoleculeEditorBackend._build_strip_ribbon` (L2028-2138)
/// and NMP `KekuleFluid/Graph.py:build_rect_patch` (L80-125).
pub fn build_zigzag_ribbon(
    width_chains: usize,
    length_cells: usize,
    passivate: bool,
    periodic_x: bool,
    a_cc: f64,
) -> NanoStructure {
    let mut st = NanoStructure::new();
    let xa = a_cc * std::f64::consts::FRAC_PI_6.cos(); // a_cc * cos(30°) = a_cc * √3/2
    let ya = a_cc * std::f64::consts::FRAC_PI_6.sin(); // a_cc * sin(30°) = a_cc / 2
    let yb = a_cc;
    let x_period = 2.0 * xa;

    // Determine strip types (A or B sublattice) for each row.
    // start_with_A = true (default in SPAMM).
    let strip_types: Vec<bool> = (0..width_chains)
        .map(|row| {
            let m = row % 4;
            m == 0 || m == 3
        })
        .collect();

    // Compute y positions for each row.
    let mut y_positions = vec![0.0f64];
    for r in 1..width_chains {
        let prev_a = strip_types[r - 1];
        let curr_a = strip_types[r];
        let dy = if prev_a && !curr_a {
            ya
        } else if !prev_a && !curr_a {
            yb
        } else if !prev_a && curr_a {
            ya
        } else {
            yb // both A
        };
        y_positions.push(y_positions[r - 1] + dy);
    }

    // Build atoms row by row.
    // Atom indexing: row * length_cells + i
    for row in 0..width_chains {
        let is_a = strip_types[row];
        let y = y_positions[row];
        let x_shift = if is_a { 0.0 } else { xa };
        for i in 0..length_cells {
            let x = i as f64 * x_period + x_shift;
            st.add_atom(Element::C, [x, y, 0.0]);
        }
    }

    // Build bonds between adjacent rows.
    for row in 1..width_chains {
        let is_a = strip_types[row];
        let prev_a = strip_types[row - 1];
        let row_start = row * length_cells;
        let prev_start = (row - 1) * length_cells;
        for i in 0..length_cells {
            let atom_idx = row_start + i;
            // Vertical bond to atom directly above/below
            st.add_bond(prev_start + i, atom_idx);
            // Diagonal bonds depending on sublattice pattern
            if is_a && !prev_a {
                // A below B: bond to prev row i-1
                if i > 0 {
                    st.add_bond(prev_start + (i - 1), atom_idx);
                }
            } else if !is_a && prev_a {
                // B below A: bond to prev row i+1
                if i + 1 < length_cells {
                    st.add_bond(prev_start + (i + 1), atom_idx);
                }
            }
        }
    }

    // For periodic x, add wrap-around bonds on the first/last column
    // when the sublattice pattern requires them.
    if periodic_x && length_cells > 1 {
        for row in 1..width_chains {
            let is_a = strip_types[row];
            let prev_a = strip_types[row - 1];
            let row_start = row * length_cells;
            let prev_start = (row - 1) * length_cells;
            if is_a && !prev_a {
                // A at (row, 0) bonds to B at (row-1, length-1) via PBC
                st.add_bond(prev_start + (length_cells - 1), row_start);
            } else if !is_a && prev_a {
                // B at (row, length-1) bonds to A at (row-1, 0) via PBC
                st.add_bond(prev_start, row_start + (length_cells - 1));
            }
        }
    }

    // In-row bonds (along x): atoms within the same row that are a_cc apart.
    // In the strip construction, consecutive atoms in a row are x_period apart
    // (2*xa = √3 * a_cc), which is NOT a nearest-neighbor bond.
    // The in-row nearest-neighbor bonds are the diagonal ones already added.
    // So no in-row bonds needed for the strip construction.

    if passivate {
        passivate_edges(&mut st, a_cc);
    }

    st
}

/// Build an armchair graphene ribbon.
///
/// Armchair edges run along x. The ribbon is built by rotating the zigzag
/// construction by 90° and swapping width/length roles.
///
/// `width_chains`: number of atom rows across the ribbon width.
/// `length_cells`: number of unit cells along the ribbon length.
/// `passivate`: if true, add H to edge carbons.
pub fn build_armchair_ribbon(
    width_chains: usize,
    length_cells: usize,
    passivate: bool,
    periodic_x: bool,
    a_cc: f64,
) -> NanoStructure {
    // Build a zigzag ribbon then rotate 90° so zigzag edges become armchair.
    let mut st = build_zigzag_ribbon(width_chains, length_cells, false, periodic_x, a_cc);
    // Rotate 90°: (x,y) -> (-y, x)
    for p in &mut st.positions {
        let [x, y, z] = *p;
        *p = [-y, x, z];
    }
    if passivate {
        passivate_edges(&mut st, a_cc);
    }
    st
}

/// Build a rectangular graphene sheet (supercell).
///
/// `nx`, `ny`: number of unit cells in each lattice direction.
/// `periodic`: if true, include PBC wrap-around bonds.
///
/// Ported from NMP `KekuleFluid/Graph.py:build_rect_patch` (L80-125).
pub fn build_sheet(
    nx: usize,
    ny: usize,
    periodic: bool,
    a_cc: f64,
) -> NanoStructure {
    let mut st = NanoStructure::new();
    let [a1, a2] = lattice_vectors(a_cc);
    let deltas = nn_deltas(a_cc);

    let mut pos_to_idx: std::collections::HashMap<(i64, i64), usize> = std::collections::HashMap::new();

    for n1 in 0..nx {
        for n2 in 0..ny {
            let a_pos = [
                n1 as f64 * a1[0] + n2 as f64 * a2[0],
                n1 as f64 * a1[1] + n2 as f64 * a2[1],
                0.0,
            ];
            let b_pos = [
                a_pos[0] + deltas[0][0],
                a_pos[1] + deltas[0][1],
                0.0,
            ];
            for (p, _is_a) in [(a_pos, true), (b_pos, false)] {
                let k = pos_key(&p);
                pos_to_idx.entry(k).or_insert_with(|| st.add_atom(Element::C, p));
            }
        }
    }

    // Build bonds: each A atom bonds to B atoms at A+δ_d
    let a_atoms: Vec<(usize, [f64; 3])> = (0..st.natom())
        .filter(|&i| is_sublattice_a(&st.positions[i], a_cc))
        .map(|i| (i, st.positions[i]))
        .collect();

    for (ia, pos_a) in &a_atoms {
        for d in 0..3 {
            let b_pos = [
                pos_a[0] + deltas[d][0],
                pos_a[1] + deltas[d][1],
                0.0,
            ];
            if let Some(&ib) = pos_to_idx.get(&pos_key(&b_pos)) {
                st.add_bond(*ia, ib);
            }
        }
    }

    if periodic {
        // Add wrap-around bonds: A atoms near the boundary bond to B atoms
        // on the opposite side via lattice vector shifts.
        let cell = [
            nx as f64 * a1[0] + ny as f64 * a2[0],
            nx as f64 * a1[1] + ny as f64 * a2[1],
            0.0,
        ];
        // For each A atom, check if A+δ_d wrapped by ±cell hits a B atom.
        for (ia, pos_a) in &a_atoms {
            for d in 0..3 {
                let b_pos = [
                    pos_a[0] + deltas[d][0],
                    pos_a[1] + deltas[d][1],
                    0.0,
                ];
                for sign in [-1.0, 1.0] {
                    let wrapped = [
                        b_pos[0] + sign * cell[0],
                        b_pos[1] + sign * cell[1],
                        0.0,
                    ];
                    if let Some(&ib) = pos_to_idx.get(&pos_key(&wrapped)) {
                        // Avoid duplicate bonds (check if ia-ib already exists)
                        if !st.bonds.iter().any(|&[i, j]| (i == *ia && j == ib) || (i == ib && j == *ia)) {
                            st.add_bond(*ia, ib);
                        }
                    }
                }
            }
        }
    }

    st
}

/// Build a hexagonal PAH (polycyclic aromatic hydrocarbon).
///
/// `n_shells`: number of ring shells around the central ring.
///   0 → benzene (6 atoms), 1 → coronene (24), 2 → circumcoronene (54), etc.
///
/// Ported from NMP `KekuleFluid/Graph.py:build_pah` (L127-262).
pub fn build_pah(n_shells: usize, a_cc: f64) -> NanoStructure {
    let mut st = NanoStructure::new();
    let [a1, a2] = lattice_vectors(a_cc);
    let deltas = nn_deltas(a_cc);

    let extent = (2 * n_shells + 3) as i32;
    let mut pos_to_idx: std::collections::HashMap<(i64, i64), usize> = std::collections::HashMap::new();

    // Step 1: Build a large rect patch
    for n1 in -extent..=extent {
        for n2 in -extent..=extent {
            let a_pos = [
                n1 as f64 * a1[0] + n2 as f64 * a2[0],
                n1 as f64 * a1[1] + n2 as f64 * a2[1],
                0.0,
            ];
            let b_pos = [a_pos[0] + deltas[0][0], a_pos[1] + deltas[0][1], 0.0];
            for p in [a_pos, b_pos] {
                let k = pos_key(&p);
                pos_to_idx.entry(k).or_insert_with(|| st.add_atom(Element::C, p));
            }
        }
    }

    // Build all bonds
    let a_atoms: Vec<(usize, [f64; 3])> = (0..st.natom())
        .filter(|&i| is_sublattice_a(&st.positions[i], a_cc))
        .map(|i| (i, st.positions[i]))
        .collect();
    for (ia, pos_a) in &a_atoms {
        for d in 0..3 {
            let b_pos = [pos_a[0] + deltas[d][0], pos_a[1] + deltas[d][1], 0.0];
            if let Some(&ib) = pos_to_idx.get(&pos_key(&b_pos)) {
                st.add_bond(*ia, ib);
            }
        }
    }

    // Step 2: Find complete hexagonal rings (6-atom cycles)
    let rings = find_hex_rings(&st, a_cc);

    if rings.is_empty() {
        return st;
    }

    // Step 3: Find central ring and select rings within n_shells
    let ring_centers: Vec<[f64; 2]> = rings.iter().map(|ring| {
        let n = ring.len() as f64;
        let mut c = [0.0f64; 2];
        for &ai in ring {
            c[0] += st.positions[ai][0];
            c[1] += st.positions[ai][1];
        }
        [c[0] / n, c[1] / n]
    }).collect();

    let central_idx = ring_centers.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a[0].powi(2).partial_cmp(&(b[0].powi(2) + b[1].powi(2))).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let central_pos = ring_centers[central_idx];
    let ring_spacing = SQRT_3 * a_cc;

    let mut selected_atoms: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (ri, center) in ring_centers.iter().enumerate() {
        let dx = center[0] - central_pos[0];
        let dy = center[1] - central_pos[1];
        let dist = (dx * dx + dy * dy).sqrt();
        if dist <= n_shells as f64 * ring_spacing + 0.01 * a_cc {
            for &ai in &rings[ri] {
                selected_atoms.insert(ai);
            }
        }
    }

    // Step 4: Rebuild with only selected atoms
    let mut new_st = NanoStructure::new();
    let mut old_to_new: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut new_pos_to_idx: std::collections::HashMap<(i64, i64), usize> = std::collections::HashMap::new();
    for i in 0..st.natom() {
        if !selected_atoms.contains(&i) { continue; }
        let new_i = new_st.add_atom(Element::C, st.positions[i]);
        old_to_new.insert(i, new_i);
        new_pos_to_idx.insert(pos_key(&st.positions[i]), new_i);
    }

    // Rebuild bonds among selected atoms
    for (ia, pos_a) in &a_atoms {
        if !selected_atoms.contains(ia) { continue; }
        let new_ia = old_to_new[ia];
        for d in 0..3 {
            let b_pos = [pos_a[0] + deltas[d][0], pos_a[1] + deltas[d][1], 0.0];
            if let Some(&new_ib) = new_pos_to_idx.get(&pos_key(&b_pos)) {
                new_st.add_bond(new_ia, new_ib);
            }
        }
    }

    // Step 5: Shift so central ring is at origin
    let shift = central_pos;
    for p in &mut new_st.positions {
        p[0] -= shift[0];
        p[1] -= shift[1];
    }

    new_st
}

/// Build a circular or hexagonal graphene flake.
///
/// `radius`: flake radius in Angstroms.
/// `shape`: "circle" or "hex".
/// `passivate`: if true, add H to edge carbons.
///
/// Ported from NMP `KekuleFluid/Graph.py:build_flake` (L264-327).
pub fn build_flake(
    radius: f64,
    shape: FlakeShape,
    passivate: bool,
    a_cc: f64,
) -> NanoStructure {
    let mut st = NanoStructure::new();
    let [a1, a2] = lattice_vectors(a_cc);
    let deltas = nn_deltas(a_cc);
    let extent = (radius / a_cc) as i32 + 3;
    let mut pos_to_idx: std::collections::HashMap<(i64, i64), usize> = std::collections::HashMap::new();

    for n1 in -extent..=extent {
        for n2 in -extent..=extent {
            let a_pos = [
                n1 as f64 * a1[0] + n2 as f64 * a2[0],
                n1 as f64 * a1[1] + n2 as f64 * a2[1],
                0.0,
            ];
            let b_pos = [a_pos[0] + deltas[0][0], a_pos[1] + deltas[0][1], 0.0];
            for p in [a_pos, b_pos] {
                let inside = match shape {
                    FlakeShape::Circle => (p[0].powi(2) + p[1].powi(2)).sqrt() <= radius,
                    FlakeShape::Hex => {
                        p[0].abs() <= radius
                            && (0.5 * p[0] + 0.5 * SQRT_3 * p[1]).abs() <= radius
                            && (0.5 * p[0] - 0.5 * SQRT_3 * p[1]).abs() <= radius
                    }
                };
                if inside {
                    let k = pos_key(&p);
                    pos_to_idx.entry(k).or_insert_with(|| st.add_atom(Element::C, p));
                }
            }
        }
    }

    // Build bonds
    let a_atoms: Vec<(usize, [f64; 3])> = (0..st.natom())
        .filter(|&i| is_sublattice_a(&st.positions[i], a_cc))
        .map(|i| (i, st.positions[i]))
        .collect();
    for (ia, pos_a) in &a_atoms {
        for d in 0..3 {
            let b_pos = [pos_a[0] + deltas[d][0], pos_a[1] + deltas[d][1], 0.0];
            if let Some(&ib) = pos_to_idx.get(&pos_key(&b_pos)) {
                st.add_bond(*ia, ib);
            }
        }
    }

    // Trim dangling atoms (fewer than 2 C-C bonds)
    trim_dangling(&mut st, a_cc);

    if passivate {
        passivate_edges(&mut st, a_cc);
    }
    st
}

/// Flake boundary shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlakeShape {
    Circle,
    Hex,
}

// ─── Internal helpers ───────────────────────────────────────────────

/// Determine if an atom is on sublattice A (vs B) by its position.
/// A atoms are at positions where (x/a_cc) maps to even lattice sites.
/// We use: A if round(x / (xa*2)) + round(y / (yb)) is even, where
/// xa = a_cc*√3/2, yb = a_cc. This is a heuristic; the builders above
/// track sublattice explicitly via the construction order.
///
/// For the lattice vectors a1=(3/2,√3/2)a, a2=(3/2,-√3/2)a and δ0=(1,0)a:
/// A atoms are at n1*a1 + n2*a2, B atoms at A + δ0.
/// We can distinguish by checking if the position is closer to a lattice
/// point (A) or a lattice point + δ0 (B).
fn is_sublattice_a(pos: &[f64; 3], a_cc: f64) -> bool {
    let xa = a_cc * SQRT_3 / 2.0;
    // A atoms have x that is a multiple of 1.5*a_cc (mod the lattice).
    // B atoms are shifted by +a_cc in x from their A partner.
    // Simple check: distance to nearest A site vs nearest B site.
    // A sites: n1*a1 + n2*a2 = (1.5*(n1+n2), √3/2*(n1-n2))*a_cc
    // The fractional coordinate along a1: f1 = (x*a2_y - y*a2_x) / det
    let det = 1.5 * a_cc * (-SQRT_3 / 2.0 * a_cc) - 1.5 * a_cc * (SQRT_3 / 2.0 * a_cc);
    let f1 = (pos[0] * (-SQRT_3 / 2.0 * a_cc) - pos[1] * 1.5 * a_cc) / det;
    let f2 = (pos[1] * 1.5 * a_cc - pos[0] * (SQRT_3 / 2.0 * a_cc)) / det;
    // f1, f2 are fractional coords along a1, a2. A atoms have integer f1,f2.
    // B atoms have f1,f2 such that position = A_site + δ0, i.e.
    // δ0 in fractional coords: δ0 = α1*a1 + α2*a2 where
    // α1 = (δ0 × a2) / det, α2 = (a1 × δ0) / det
    // δ0 = (a_cc, 0): α1 = (a_cc * (-√3/2 a_cc)) / det, α2 = (1.5 a_cc * 0 - ... ) / det
    // Actually simpler: just check if f1+f2 is close to an integer (A) or
    // close to integer + offset (B).
    let _ = xa; // suppress unused warning
    let frac_a1 = f1 - f1.round();
    let frac_a2 = f2 - f2.round();
    // A atom: both fracs ~0. B atom: fracs correspond to δ0 in fractional coords.
    // δ0 fractional: solve (a_cc, 0) = α1*a1 + α2*a2
    //   a_cc = α1*1.5*a_cc + α2*1.5*a_cc  =>  1 = 1.5*(α1+α2)
    //   0 = α1*√3/2*a_cc - α2*√3/2*a_cc  =>  α1 = α2
    //   => α1 = α2 = 1/3
    // So B atoms have frac_a1 ≈ 1/3, frac_a2 ≈ 1/3 (mod 1).
    let dist_a = (frac_a1.powi(2) + frac_a2.powi(2)).sqrt();
    let dist_b = {
        let d1 = (frac_a1 - 1.0 / 3.0).abs().min((frac_a1 + 2.0 / 3.0).abs());
        let d2 = (frac_a2 - 1.0 / 3.0).abs().min((frac_a2 + 2.0 / 3.0).abs());
        (d1 * d1 + d2 * d2).sqrt()
    };
    dist_a < dist_b
}

/// Find all complete hexagonal rings (6-atom cycles) in the structure.
fn find_hex_rings(st: &NanoStructure, a_cc: f64) -> Vec<Vec<usize>> {
    // Build adjacency
    let mut adj: Vec<std::collections::HashSet<usize>> = vec![std::collections::HashSet::new(); st.natom()];
    for &[i, j] in &st.bonds {
        adj[i].insert(j);
        adj[j].insert(i);
    }

    let bond_len = a_cc * 1.1; // tolerance
    let mut rings = Vec::new();
    let mut seen: std::collections::HashSet<[usize; 6]> = std::collections::HashSet::new();

    // For each atom, try to find a 6-cycle through it
    for start in 0..st.natom() {
        // BFS up to depth 3 to find hexagonal rings
        let neighbors: Vec<usize> = adj[start].iter().copied().collect();
        for &n1 in &neighbors {
            for &n2 in &adj[n1] {
                if n2 == start { continue; }
                for &n3 in &adj[n2] {
                    if n3 == n1 || n3 == start { continue; }
                    for &n4 in &adj[n3] {
                        if n4 == n2 || n4 == n1 || n4 == start { continue; }
                        for &n5 in &adj[n4] {
                            if n5 == n3 || n5 == n2 || n5 == n1 || n5 == start { continue; }
                            // Check if n5 connects back to start
                            if adj[n5].contains(&start) {
                                let mut ring = [start, n1, n2, n3, n4, n5];
                                ring.sort();
                                if seen.insert(ring) {
                                    rings.push(vec![start, n1, n2, n3, n4, n5]);
                                }
                            }
                        }
                    }
                }
            }
        }
        let _ = bond_len;
    }

    rings
}

/// Iteratively remove atoms with < 2 C-C bonds (dangling atoms).
fn trim_dangling(st: &mut NanoStructure, a_cc: f64) {
    let _ = a_cc;
    loop {
        // Count C-C bonds per atom
        let mut c_neighbors = vec![0usize; st.natom()];
        for &[i, j] in &st.bonds {
            if st.elements[i] == Element::C && st.elements[j] == Element::C {
                c_neighbors[i] += 1;
                c_neighbors[j] += 1;
            }
        }
        // Find atoms to remove
        let remove: std::collections::HashSet<usize> = (0..st.natom())
            .filter(|&i| st.elements[i] == Element::C && c_neighbors[i] < 2)
            .collect();
        if remove.is_empty() { break; }

        // Rebuild without removed atoms
        let mut new_st = NanoStructure::new();
        let mut old_to_new: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for i in 0..st.natom() {
            if remove.contains(&i) { continue; }
            old_to_new.insert(i, new_st.add_atom(st.elements[i], st.positions[i]));
        }
        for &[i, j] in &st.bonds {
            if let (Some(&ni), Some(&nj)) = (old_to_new.get(&i), old_to_new.get(&j)) {
                new_st.add_bond(ni, nj);
            }
        }
        *st = new_st;
    }
}

/// Add H atoms to undercoordinated carbon atoms (edge passivation) using
/// proper VSEPR sp² geometry (120° angles).
///
/// For graphene (all sp² carbon):
/// - 1 existing C-C bond → 2 H at ±120° from the bond (trigonal planar)
/// - 2 existing C-C bonds → 1 H along the negative bisector (completes the
///   trigonal triangle, 120° from each existing bond)
///
/// Ported from:
/// - SPAMMM `AtomicSystem._missing_sp2_direction()` (L1157-1193)
/// - FireCore `MMFFBuilderBase::makeConfGeom()` sp² branch (L932-998)
pub fn passivate_edges(st: &mut NanoStructure, a_cc: f64) {
    let ch_bond = 1.09; // C-H bond length in Angstrom
    let _ = a_cc;

    // Find C-C neighbors for each atom
    let mut c_neighbors: Vec<Vec<usize>> = vec![Vec::new(); st.natom()];
    for &[i, j] in &st.bonds {
        if st.elements[i] == Element::C && st.elements[j] == Element::C {
            c_neighbors[i].push(j);
            c_neighbors[j].push(i);
        }
    }

    let mut h_to_add: Vec<(usize, [f64; 3])> = Vec::new();
    for i in 0..st.natom() {
        if st.elements[i] != Element::C { continue; }
        let neighbors = &c_neighbors[i];
        let nb = neighbors.len();
        if nb >= 3 { continue; } // fully coordinated
        if nb == 0 { continue; } // isolated atom

        let pos_i = st.positions[i];

        // Compute missing sp² directions (120° from existing bonds)
        let dirs = missing_sp2_directions(pos_i, &st.positions, neighbors);

        for dir in &dirs {
            let h_pos = [
                pos_i[0] + dir[0] * ch_bond,
                pos_i[1] + dir[1] * ch_bond,
                pos_i[2] + dir[2] * ch_bond,
            ];
            h_to_add.push((i, h_pos));
        }
    }

    // Add H atoms and bonds
    for (c_idx, h_pos) in h_to_add {
        let h_idx = st.add_atom(Element::H, h_pos);
        st.add_bond(c_idx, h_idx);
    }
}

/// Compute the missing sp² hybrid directions for a carbon atom.
///
/// Given the atom position and its existing C-C neighbor positions, returns
/// unit vectors pointing toward where the missing H atoms should be placed
/// (at 120° from each existing bond, in the sp² plane).
///
/// Ported from SPAMMM `AtomicSystem._missing_sp2_direction()` (L1157-1193)
/// and FireCore `MMFFBuilderBase::makeConfGeom()` sp² branch (L932-998).
///
/// # Cases
/// - **1 existing bond** (`nb=1`): 2 missing H at ±120° from the bond.
///   `h = -0.5·v₁ ± (√3/2)·perp` where `perp ⊥ v₁` in the molecular plane.
/// - **2 existing bonds** (`nb=2`): 1 missing H along `-bisector(v₁, v₂)`.
///   If the two bonds are 120° apart, the H is 120° from each.
fn missing_sp2_directions(
    pos: [f64; 3],
    neighbor_pos: &[[f64; 3]],
    neighbors: &[usize],
) -> Vec<[f64; 3]> {
    let nb = neighbors.len();
    if nb == 0 { return Vec::new(); }

    // Get unit vectors from this atom toward each neighbor
    let vs: Vec<[f64; 3]> = neighbors.iter().map(|&j| {
        let npos = neighbor_pos[j];
        let d = [npos[0] - pos[0], npos[1] - pos[1], npos[2] - pos[2]];
        normalize3(d)
    }).collect();

    match nb {
        1 => {
            // One existing bond: two H at ±120° from v₁
            // Ported from SPAMMM _missing_sp2_direction (nb==1 case)
            // and FireCore makeConfGeom (nb==1, npi==1 case).
            let v1 = vs[0];
            // Find a vector perpendicular to v1.
            // For planar graphene (z=0), use the in-plane perpendicular.
            // For 3D, use get_some_ortho to find any ⊥ vector.
            let perp = get_some_ortho(&v1);
            let half = -0.5_f64;
            let s3_2 = 0.5 * SQRT_3; // √3/2 = sin(120°)
            let h1 = normalize3([
                half * v1[0] + s3_2 * perp[0],
                half * v1[1] + s3_2 * perp[1],
                half * v1[2] + s3_2 * perp[2],
            ]);
            let h2 = normalize3([
                half * v1[0] - s3_2 * perp[0],
                half * v1[1] - s3_2 * perp[1],
                half * v1[2] - s3_2 * perp[2],
            ]);
            vec![h1, h2]
        }
        2 => {
            // Two existing bonds: one H along the negative bisector
            // Ported from SPAMMM _missing_sp2_direction (nb>=2 case)
            // and FireCore makeConfGeom (nb==2, npi==1 case).
            let v1 = vs[0];
            let v2 = vs[1];
            let bisect = [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]];
            let bisect_norm = (bisect[0].powi(2) + bisect[1].powi(2) + bisect[2].powi(2)).sqrt();
            if bisect_norm < 1e-8 {
                // v1 ≈ -v2 (collinear bonds): use perpendicular in the plane
                let perp = get_some_ortho(&v1);
                vec![perp]
            } else {
                // Missing lobe = -bisector direction (completes the trigonal triangle)
                vec![normalize3([-bisect[0], -bisect[1], -bisect[2]])]
            }
        }
        _ => Vec::new(), // nb >= 3: no missing bonds
    }
}

/// Find a unit vector perpendicular to `v`, preferring the molecular plane.
///
/// For planar graphene (v in the xy-plane), returns the in-plane perpendicular
/// `[-v.y, v.x, 0]`. For 3D vectors, uses the "least-dominant axis" method:
/// cross `v` with the axis (x/y/z) where `v` has the smallest component.
///
/// Ported from FireCore `Vec3d::getSomeOrtho()`.
fn get_some_ortho(v: &[f64; 3]) -> [f64; 3] {
    // For planar systems (z ≈ 0), use the in-plane perpendicular
    if v[2].abs() < 1e-10 {
        let perp = [-v[1], v[0], 0.0];
        let n = (perp[0].powi(2) + perp[1].powi(2)).sqrt();
        if n > 1e-10 {
            return [perp[0] / n, perp[1] / n, 0.0];
        }
    }
    // General 3D: cross with the axis where v has the smallest component
    let ax = v[0].abs();
    let ay = v[1].abs();
    let az = v[2].abs();
    let axis = if ax <= ay && ax <= az {
        [1.0, 0.0, 0.0]
    } else if ay <= az {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let cross = cross3(*v, axis);
    let n = (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
    if n > 1e-10 {
        [cross[0] / n, cross[1] / n, cross[2] / n]
    } else {
        // v is along the chosen axis; try another
        [0.0, 1.0, 0.0]
    }
}

/// 3D vector cross product.
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalize a 3D vector. Returns zero vector if input is near-zero.
fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt();
    if n < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_ribbon_basic() {
        let st = build_zigzag_ribbon(4, 5, false, false, A_CC);
        // 4 rows × 5 cells = 20 atoms
        assert_eq!(st.natom(), 20);
        // All carbon
        assert_eq!(st.count(Element::C), 20);
        assert_eq!(st.count(Element::H), 0);
        // Should have bonds
        assert!(st.nbond() > 0);
        println!("zigzag 4×5: {} atoms, {} bonds", st.natom(), st.nbond());
    }

    #[test]
    fn test_zigzag_ribbon_passivated() {
        let st = build_zigzag_ribbon(4, 5, true, false, A_CC);
        let n_c = st.count(Element::C);
        let n_h = st.count(Element::H);
        assert_eq!(n_c, 20);
        assert!(n_h > 0, "should have H passivation");
        println!("zigzag 4×5 passivated: {} C, {} H, {} bonds", n_c, n_h, st.nbond());
    }

    #[test]
    fn test_sheet_basic() {
        let st = build_sheet(3, 3, false, A_CC);
        // 3×3 unit cells, 2 atoms per cell = 18 atoms
        assert_eq!(st.natom(), 18);
        assert_eq!(st.count(Element::C), 18);
        println!("sheet 3×3: {} atoms, {} bonds", st.natom(), st.nbond());
    }

    #[test]
    fn test_pah_benzene() {
        let st = build_pah(0, A_CC);
        // Benzene: 6 atoms, 6 bonds
        assert_eq!(st.natom(), 6);
        assert_eq!(st.nbond(), 6);
        println!("PAH n=0 (benzene): {} atoms, {} bonds", st.natom(), st.nbond());
    }

    #[test]
    fn test_pah_coronene() {
        let st = build_pah(1, A_CC);
        // Coronene: 24 atoms
        assert_eq!(st.natom(), 24);
        println!("PAH n=1 (coronene): {} atoms, {} bonds", st.natom(), st.nbond());
    }

    #[test]
    fn test_flake_circle() {
        let st = build_flake(5.0, FlakeShape::Circle, false, A_CC);
        assert!(st.natom() > 10, "flake should have reasonable size: {}", st.natom());
        println!("flake r=5 circle: {} atoms, {} bonds", st.natom(), st.nbond());
    }

    #[test]
    fn test_xyz_output() {
        let st = build_pah(0, A_CC);
        let xyz = st.to_xyz();
        assert!(xyz.starts_with("6\n"));
        assert!(xyz.contains("C "));
    }

    #[test]
    fn test_edge_detection() {
        let st = build_zigzag_ribbon(4, 5, false, false, A_CC);
        let edges = st.find_edge_carbons();
        assert!(!edges.is_empty(), "should have edge atoms");
        println!("zigzag 4×5: {} edge carbons", edges.len());
    }

    #[test]
    fn test_armchair_ribbon() {
        let st = build_armchair_ribbon(4, 5, false, false, A_CC);
        assert_eq!(st.count(Element::C), 20);
        println!("armchair 4×5: {} atoms, {} bonds", st.natom(), st.nbond());
    }

    /// Verify that H passivation uses 120° VSEPR angles, not 180°.
    /// For a =CH2 group (1 C-C bond, 2 H), the H-C-H angle should be ~120°
    /// and each H-C-C angle should be ~120°.
    #[test]
    fn test_vsepr_120_degrees() {
        let st = build_zigzag_ribbon(4, 5, true, false, A_CC);
        // Find a carbon with exactly 1 C neighbor and 2 H neighbors (=CH2 group)
        let mut c_neighbors: Vec<Vec<usize>> = vec![Vec::new(); st.natom()];
        let mut h_neighbors: Vec<Vec<usize>> = vec![Vec::new(); st.natom()];
        for &[i, j] in &st.bonds {
            if st.elements[i] == Element::C && st.elements[j] == Element::C {
                c_neighbors[i].push(j);
                c_neighbors[j].push(i);
            } else if st.elements[i] == Element::C && st.elements[j] == Element::H {
                h_neighbors[i].push(j);
            } else if st.elements[i] == Element::H && st.elements[j] == Element::C {
                h_neighbors[j].push(i);
            }
        }

        let mut found_ch2 = false;
        let mut max_angle_err = 0.0f64;
        for i in 0..st.natom() {
            if st.elements[i] != Element::C { continue; }
            if c_neighbors[i].len() != 1 || h_neighbors[i].len() != 2 { continue; }
            found_ch2 = true;
            let pos_c = st.positions[i];
            let pos_cc = st.positions[c_neighbors[i][0]];
            let pos_h1 = st.positions[h_neighbors[i][0]];
            let pos_h2 = st.positions[h_neighbors[i][1]];

            // Vectors from C
            let v_cc = normalize3([
                pos_cc[0] - pos_c[0],
                pos_cc[1] - pos_c[1],
                pos_cc[2] - pos_c[2],
            ]);
            let v_h1 = normalize3([
                pos_h1[0] - pos_c[0],
                pos_h1[1] - pos_c[1],
                pos_h1[2] - pos_c[2],
            ]);
            let v_h2 = normalize3([
                pos_h2[0] - pos_c[0],
                pos_h2[1] - pos_c[1],
                pos_h2[2] - pos_c[2],
            ]);

            // Angles (in degrees)
            let angle_cc_h1 = dot_angle_deg(v_cc, v_h1);
            let angle_cc_h2 = dot_angle_deg(v_cc, v_h2);
            let angle_h1_h2 = dot_angle_deg(v_h1, v_h2);

            println!("  =CH2 at atom {i}: C-C-H1={angle_cc_h1:.1}° C-C-H2={angle_cc_h2:.1}° H1-C-H2={angle_h1_h2:.1}°");

            // All angles should be ~120° (not 180°!)
            max_angle_err = max_angle_err.max((angle_cc_h1 - 120.0).abs());
            max_angle_err = max_angle_err.max((angle_cc_h2 - 120.0).abs());
            max_angle_err = max_angle_err.max((angle_h1_h2 - 120.0).abs());
        }

        assert!(found_ch2, "no =CH2 group found in passivated ribbon");
        println!("  max angle deviation from 120°: {max_angle_err:.2}°");
        assert!(max_angle_err < 5.0, "VSEPR angles not ~120°: max err = {max_angle_err:.2}°");
    }

    /// Verify that a =CH group (2 C-C bonds, 1 H) has the H at ~120° from
    /// each existing C-C bond.
    #[test]
    fn test_vsepr_ch_120_degrees() {
        let st = build_zigzag_ribbon(4, 5, true, false, A_CC);
        let mut c_neighbors: Vec<Vec<usize>> = vec![Vec::new(); st.natom()];
        let mut h_neighbors: Vec<Vec<usize>> = vec![Vec::new(); st.natom()];
        for &[i, j] in &st.bonds {
            if st.elements[i] == Element::C && st.elements[j] == Element::C {
                c_neighbors[i].push(j);
                c_neighbors[j].push(i);
            } else if st.elements[i] == Element::C && st.elements[j] == Element::H {
                h_neighbors[i].push(j);
            } else if st.elements[i] == Element::H && st.elements[j] == Element::C {
                h_neighbors[j].push(i);
            }
        }

        let mut found_ch = false;
        let mut max_err = 0.0f64;
        for i in 0..st.natom() {
            if st.elements[i] != Element::C { continue; }
            if c_neighbors[i].len() != 2 || h_neighbors[i].len() != 1 { continue; }
            found_ch = true;
            let pos_c = st.positions[i];
            let v_cc1 = normalize3([
                st.positions[c_neighbors[i][0]][0] - pos_c[0],
                st.positions[c_neighbors[i][0]][1] - pos_c[1],
                st.positions[c_neighbors[i][0]][2] - pos_c[2],
            ]);
            let v_cc2 = normalize3([
                st.positions[c_neighbors[i][1]][0] - pos_c[0],
                st.positions[c_neighbors[i][1]][1] - pos_c[1],
                st.positions[c_neighbors[i][1]][2] - pos_c[2],
            ]);
            let v_h = normalize3([
                st.positions[h_neighbors[i][0]][0] - pos_c[0],
                st.positions[h_neighbors[i][0]][1] - pos_c[1],
                st.positions[h_neighbors[i][0]][2] - pos_c[2],
            ]);
            let a1 = dot_angle_deg(v_cc1, v_h);
            let a2 = dot_angle_deg(v_cc2, v_h);
            println!("  =CH at atom {i}: C-C-H(1)={a1:.1}° C-C-H(2)={a2:.1}°");
            max_err = max_err.max((a1 - 120.0).abs()).max((a2 - 120.0).abs());
        }
        assert!(found_ch, "no =CH group found");
        println!("  max angle deviation from 120°: {max_err:.2}°");
        assert!(max_err < 5.0, "VSEPR CH angles not ~120°: max err = {max_err:.2}°");
    }
}

/// Angle between two unit vectors, in degrees.
fn dot_angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let dot = dot.clamp(-1.0, 1.0);
    dot.acos().to_degrees()
}
