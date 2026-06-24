//! GPU preparation: pack fragment data into flat arrays for OpenCL kernels.
//!
//! No OpenCL dependency here — only raw `Vec<f32>` / `Vec<i32>` arrays.
//! The actual kernel launch happens in a separate driver module.

use crate::core::error::{DftbError, Result};
use crate::methods::dftb::sk_data::SkData;
use crate::methods::dftb::spline_resample;
use crate::qmqm::fragment::Fragment;
use crate::qmqm::gamma::GammaTable;
use std::collections::HashMap;

const ANG2BOHR: f64 = 1.889726133;

/// Maximum SK grid size that can be loaded into GPU __local memory.
/// Must match SK_GRID_MAX in dftb_hamiltonian.cl.
/// With resampling to 64 points, actual usage is 64*4*8B = 2KB per table.
pub const SK_GRID_MAX: usize = 256;

// ------------------------------------------------------------------
// GPU-compatible structs (match OpenCL layout)
// ------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuFragment {
    pub n_atoms: i32,
    pub n_orbs: i32,
    pub atom_off: i32,
    pub h_base: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuPairEntry {
    pub replica: u32,
    pub atom_i: u16,
    pub atom_j: u16,
    pub orb_i: u16,
    pub orb_j: u16,
    pub r: f32,
    pub l: f32,
    pub m: f32,
    pub n: f32,
}

#[derive(Debug, Clone)]
pub struct GpuPairBucket {
    pub pairs: Vec<GpuPairEntry>,
    pub block_type: u8,
    pub sk_table_idx: usize,
    pub n_pairs: usize,
}

#[derive(Debug, Clone)]
pub struct GpuSkTable {
    pub sk_h: Vec<f32>,
    pub sk_s: Vec<f32>,
    pub n_grid: usize,
    pub dr: f32,
    pub n_sk_cols: usize,
    pub block_type: u8,
    pub species_i: u8,
    pub species_j: u8,
}

#[derive(Debug, Clone)]
pub struct GpuGammaNeigh {
    pub offsets: Vec<i32>,
    pub neigh_j: Vec<i32>,
    pub neigh_r: Vec<f32>,
}

// ------------------------------------------------------------------
// Main batch container
// ------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GpuBatch {
    pub n_frags: usize,
    pub n_global_species: usize,
    pub total_atoms: usize,
    pub total_h_elements: usize,

    pub hubbard_u: Vec<f32>,
    pub onsite_es_ep: Vec<f32>,
    pub sk_tables: Vec<GpuSkTable>,

    pub fragments: Vec<GpuFragment>,

    pub atom_species: Vec<i32>,
    pub atom_orb_off: Vec<i32>,
    pub charges: Vec<f32>,

    pub gamma_neigh: GpuGammaNeigh,
    pub pair_buckets: Vec<GpuPairBucket>,
}

// ------------------------------------------------------------------
// Build batch from fragments
// ------------------------------------------------------------------

impl GpuBatch {
    pub fn from_fragments(
        fragments: &[Fragment],
        sk_data: &SkData,
        gamma_table: &GammaTable,
    ) -> Result<Self> {
        if fragments.is_empty() {
            return Err(DftbError::InvalidInput("empty fragment list".into()));
        }

        // 1. Global species table
        let (global_species, species_to_global) = build_global_species(fragments, sk_data)?;
        let n_global_species = global_species.len();

        // 2. Per-fragment metadata + flattened atom arrays
        let mut gpu_frags = Vec::with_capacity(fragments.len());
        let mut atom_species = Vec::new();
        let mut atom_orb_off = Vec::new();
        let mut charges = Vec::new();
        let mut all_coords = Vec::new();

        let mut atom_cursor = 0i32;
        let mut h_cursor = 0i32;

        for frag in fragments {
            let tmpl = &frag.template;
            let n_atoms = tmpl.n_atoms as i32;
            let n_orbs = tmpl.n_orbs as i32;

            gpu_frags.push(GpuFragment {
                n_atoms,
                n_orbs,
                atom_off: atom_cursor,
                h_base: h_cursor,
            });

            for a in 0..tmpl.n_atoms {
                let local_sp = tmpl.atom_species[a];
                let global_sp = species_to_global[&tmpl.species[a]];
                atom_species.push(global_sp as i32);
                atom_orb_off.push(tmpl.atom_orb_off[a] as i32);
                charges.push(frag.charges[a] as f32);
                let c = frag.coords[a];
                all_coords.push([c[0] * ANG2BOHR, c[1] * ANG2BOHR, c[2] * ANG2BOHR]);
            }

            atom_cursor += n_atoms;
            h_cursor += n_orbs * n_orbs;
        }

        let total_atoms = atom_cursor as usize;
        let total_h_elements = h_cursor as usize;

        // 3. Global hubbard_u and onsite_es_ep
        let mut hubbard_u = vec![0.0f32; n_global_species];
        let mut onsite_es_ep = vec![0.0f32; n_global_species * 2];

        for (name, &idx) in &species_to_global {
            let sp_idx = idx as usize;
            let onsite = sk_data.onsite(name)?;
            hubbard_u[sp_idx] = gamma_table.u(idx) as f32;
            onsite_es_ep[2 * sp_idx] = onsite.e_s as f32;
            onsite_es_ep[2 * sp_idx + 1] = onsite.e_p as f32;
        }

        // 4. SK tables (compact, raw grid values)
        let sk_tables = pack_sk_tables(sk_data, &global_species, &species_to_global)?;

        // 5. Gamma neighbor list (flat CSR, inter-fragment)
        let gamma_neigh = build_gamma_neigh(&all_coords, &atom_species, gamma_table, &gpu_frags);

        // 6. Pair buckets (cross-fragment, SK cutoff)
        let pair_buckets = build_pair_buckets(fragments, &gpu_frags, &all_coords, &atom_species, sk_data, &sk_tables, &species_to_global)?;

        Ok(Self {
            n_frags: fragments.len(),
            n_global_species,
            total_atoms,
            total_h_elements,
            hubbard_u,
            onsite_es_ep,
            sk_tables,
            fragments: gpu_frags,
            atom_species,
            atom_orb_off,
            charges,
            gamma_neigh,
            pair_buckets,
        })
    }
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

fn build_global_species(
    fragments: &[Fragment],
    sk_data: &SkData,
) -> Result<(Vec<String>, HashMap<String, u8>)> {
    let mut names: Vec<String> = Vec::new();
    // Estimate capacity: max unique species across all fragments
    let mut max_species = 0usize;
    for frag in fragments {
        max_species += frag.template.species.len();
    }
    let mut map: HashMap<String, u8> = HashMap::with_capacity(max_species);

    for frag in fragments {
        for sp in &frag.template.species {
            if !map.contains_key(sp) {
                // Verify species exists in SK data
                let _ = sk_data.onsite(sp)?;
                let idx = names.len() as u8;
                map.insert(sp.clone(), idx);
                names.push(sp.clone());
            }
        }
    }
    Ok((names, map))
}

/// Target number of grid points after resampling SK tables.
/// 32-64 is sufficient for cubic spline representation of typical SK tables.
const SK_RESAMPLE_N: usize = 64;

fn pack_sk_tables(
    sk_data: &SkData,
    global_species: &[String],
    species_to_global: &HashMap<String, u8>,
) -> Result<Vec<GpuSkTable>> {
    let n = global_species.len();
    let mut tables = Vec::with_capacity(n * n);

    for si in 0..n {
        for sj in 0..n {
            let sp_i = &global_species[si];
            let sp_j = &global_species[sj];
            let tab = match sk_data.get_pair(sp_i, sp_j) {
                Some(t) => t,
                None => continue,
            };

            let ang_i = sk_data.ang_shells(sp_i)?;
            let ang_j = sk_data.ang_shells(sp_j)?;
            let block_type = determine_block_type(ang_i, ang_j);
            let n_sk_cols = match block_type {
                0 => 1,
                1 => 2,
                2 => 4,
                _ => unreachable!(),
            };

            let n_grid_orig = tab.h.n_grid();
            let dr_orig = tab.h.dr;

            // Extract columns from original grid into per-channel arrays
            let mut h_cols: Vec<Vec<f64>> = vec![vec![0.0; n_grid_orig]; n_sk_cols];
            let mut s_cols: Vec<Vec<f64>> = vec![vec![0.0; n_grid_orig]; n_sk_cols];

            for k in 0..n_grid_orig {
                let h_all = &tab.h.values[k];
                let s_all = &tab.s.values[k];

                match block_type {
                    0 => {
                        let (h, s, _) = extract_shell_old_or_new(h_all, s_all, 0, 0);
                        h_cols[0][k] = h[0];
                        s_cols[0][k] = s[0];
                    }
                    1 => {
                        let (h_ss, s_ss, _) = extract_shell_old_or_new(h_all, s_all, 0, 0);
                        h_cols[0][k] = h_ss[0];
                        s_cols[0][k] = s_ss[0];

                        let (h_sp, s_sp, _) = extract_shell_old_or_new(h_all, s_all, 0, 1);
                        h_cols[1][k] = h_sp[0];
                        s_cols[1][k] = s_sp[0];
                    }
                    2 => {
                        let (h_ss, s_ss, _) = extract_shell_old_or_new(h_all, s_all, 0, 0);
                        h_cols[0][k] = h_ss[0];
                        s_cols[0][k] = s_ss[0];

                        let (h_sp, s_sp, _) = extract_shell_old_or_new(h_all, s_all, 0, 1);
                        h_cols[1][k] = h_sp[0];
                        s_cols[1][k] = s_sp[0];

                        let (h_pp, s_pp, _) = extract_shell_old_or_new(h_all, s_all, 1, 1);
                        h_cols[2][k] = h_pp[0]; // sigma
                        s_cols[2][k] = s_pp[0];
                        h_cols[3][k] = h_pp[1]; // pi
                        s_cols[3][k] = s_pp[1];
                    }
                    _ => unreachable!(),
                }
            }

            // Resample each channel to SK_RESAMPLE_N points using cubic B-spline
            let n_grid = SK_RESAMPLE_N;
            let mut sk_h = vec![0.0f32; n_grid * n_sk_cols];
            let mut sk_s = vec![0.0f32; n_grid * n_sk_cols];
            let mut dr_new = 0.0f32;

            for col in 0..n_sk_cols {
                let (h_vals, dr) =
                    spline_resample::resample_sk_column(&h_cols[col], dr_orig, n_grid);
                let (s_vals, _) =
                    spline_resample::resample_sk_column(&s_cols[col], dr_orig, n_grid);
                dr_new = dr;

                for k in 0..n_grid {
                    sk_h[k * n_sk_cols + col] = h_vals[k];
                    sk_s[k * n_sk_cols + col] = s_vals[k];
                }
            }

            tables.push(GpuSkTable {
                sk_h,
                sk_s,
                n_grid,
                dr: dr_new,
                n_sk_cols,
                block_type,
                species_i: si as u8,
                species_j: sj as u8,
            });
        }
    }

    Ok(tables)
}

/// Extract shell integrals from raw SK table row (old or extended format).
/// Returns fixed-size stack arrays — zero allocation. Max output is 2 values (pp shell).
fn extract_shell_old_or_new(
    h_all: &[f64],
    s_all: &[f64],
    ang1: i32,
    ang2: i32,
) -> ([f64; 2], [f64; 2], usize) {
    use crate::methods::dftb::sk_data::sk_map;

    let (l_min, l_max) = if ang1 <= ang2 { (ang1, ang2) } else { (ang2, ang1) };
    let n_mm = (l_min + 1) as usize;

    let is_extended = h_all.len() == 20;
    let mut h_out = [0.0f64; 2];
    let mut s_out = [0.0f64; 2];

    if is_extended {
        for mm in 0..=l_min {
            let new_col = sk_map(mm, l_max, l_min) as usize;
            h_out[mm as usize] = h_all[new_col - 1];
            s_out[mm as usize] = s_all[new_col - 1];
        }
    } else {
        const NEW_TO_OLD: [usize; 21] = {
            let mut arr = [0usize; 21];
            let iSKInterOld: [usize; 10] = [8, 9, 10, 13, 14, 15, 16, 18, 19, 20];
            let mut i = 0;
            while i < 10 {
                arr[iSKInterOld[i]] = i;
                i += 1;
            }
            arr
        };
        for mm in 0..=l_min {
            let new_col = sk_map(mm, l_max, l_min) as usize;
            let old_idx = NEW_TO_OLD[new_col];
            h_out[mm as usize] = h_all[old_idx];
            s_out[mm as usize] = s_all[old_idx];
        }
    }

    (h_out, s_out, n_mm)
}

#[inline]
fn n_orb_from_ang(ang: &[i32]) -> usize {
    let mut sum = 0usize;
    for &l in ang {
        sum += (2 * l + 1) as usize;
    }
    sum
}

fn determine_block_type(ang_i: &[i32], ang_j: &[i32]) -> u8 {
    let n_orb_i = n_orb_from_ang(ang_i);
    let n_orb_j = n_orb_from_ang(ang_j);
    match (n_orb_i, n_orb_j) {
        (1, 1) => 0,
        (1, 4) | (4, 1) => 1,
        (4, 4) => 2,
        _ => panic!("gpu_prep: unsupported orbital count ({}, {})", n_orb_i, n_orb_j),
    }
}

fn build_gamma_neigh(
    all_coords: &[[f64; 3]],
    atom_species: &[i32],
    gamma_table: &GammaTable,
    frags: &[GpuFragment],
) -> GpuGammaNeigh {
    let total_atoms = all_coords.len();
    let mut offsets = vec![0i32; total_atoms + 1];
    // Rough capacity estimate: each fragment has at most n*(n-1) neighbors
    let mut max_neigh = 0usize;
    for frag in frags {
        let n = frag.n_atoms as usize;
        max_neigh += n * (n - 1);
    }
    let mut neigh_j = Vec::with_capacity(max_neigh);
    let mut neigh_r = Vec::with_capacity(max_neigh);

    for frag in frags {
        let off = frag.atom_off as usize;
        let n = frag.n_atoms as usize;
        for a in 0..n {
            let ga = off + a;
            let sp_a = atom_species[ga] as u8;
            let pos_a = all_coords[ga];
            let mut count = 0;

            for b in 0..n {
                if a == b {
                    continue;
                }
                let gb = off + b;
                let sp_b = atom_species[gb] as u8;
                let cutoff = gamma_table.cutoffs[sp_a as usize * gamma_table.n_species + sp_b as usize];
                let cutoff_sq = cutoff * cutoff;

                let dx = all_coords[gb][0] - pos_a[0];
                let dy = all_coords[gb][1] - pos_a[1];
                let dz = all_coords[gb][2] - pos_a[2];
                let r2 = dx * dx + dy * dy + dz * dz;

                if r2 > cutoff_sq {
                    continue;
                }

                let r = r2.sqrt();
                neigh_j.push(b as i32); // local index within fragment
                neigh_r.push(r as f32);
                count += 1;
            }
            offsets[ga + 1] = offsets[ga] + count;
        }
    }

    // Fix offsets: currently only set for intra-fragment neighbors.
    // Inter-fragment neighbors are handled by the qmqm solver, not by this kernel.
    // For now, gamma_neigh only covers intra-fragment (same as original onsite_and_va intent).
    // If inter-fragment gamma is needed, extend here.

    GpuGammaNeigh {
        offsets,
        neigh_j,
        neigh_r,
    }
}

fn build_pair_buckets(
    fragments: &[Fragment],
    gpu_frags: &[GpuFragment],
    all_coords: &[[f64; 3]],
    global_atom_species: &[i32],
    sk_data: &SkData,
    sk_tables: &[GpuSkTable],
    species_to_global: &HashMap<String, u8>,
) -> Result<Vec<GpuPairBucket>> {
    // Build flat SK lookup table: sk_lookup[sp_i * n_species + sp_j] = table index
    let n_species = species_to_global.len();
    let mut sk_lookup = vec![usize::MAX; n_species * n_species];
    for (idx, tab) in sk_tables.iter().enumerate() {
        let key = tab.species_i as usize * n_species + tab.species_j as usize;
        sk_lookup[key] = idx;
    }

    // Precompute global SK cutoff once (same for all fragments)
    let mut sk_cutoff = 0.0f64;
    for (_, tab) in &sk_data.pairs {
        let c = tab.cutoff();
        if c > sk_cutoff { sk_cutoff = c; }
    }
    let sk_cutoff_sq = sk_cutoff * sk_cutoff;

    // Flat bucket storage: index = block_type * n_species * n_species + sp_i * n_species + sp_j
    let n_bucket_slots = 3 * n_species * n_species;
    let mut buckets: Vec<Vec<GpuPairEntry>> = Vec::with_capacity(n_bucket_slots);
    for _ in 0..n_bucket_slots {
        buckets.push(Vec::new());
    }

    for fi in 0..fragments.len() {
        let frag = &fragments[fi];
        let tmpl = &frag.template;
        let atom_off = gpu_frags[fi].atom_off as usize;

        // Simple O(n²) pair finder (replace with NeighborBuilder if needed)
        let n = tmpl.n_atoms;
        for i in 0..n {
            let pos_i = all_coords[atom_off + i];
            let n_orb_i = tmpl.atom_n_orb[i] as usize;
            let orb_off_i = tmpl.atom_orb_off[i];

            for j in (i + 1)..n {
                let pos_j = all_coords[atom_off + j];
                let dx = pos_j[0] - pos_i[0];
                let dy = pos_j[1] - pos_i[1];
                let dz = pos_j[2] - pos_i[2];
                let r2 = dx * dx + dy * dy + dz * dz;

                if r2 > sk_cutoff_sq {
                    continue;
                }

                let r = r2.sqrt();
                let inv_r = 1.0 / r;

                let n_orb_j = tmpl.atom_n_orb[j] as usize;
                let orb_off_j = tmpl.atom_orb_off[j];

                let block_type = match (n_orb_i, n_orb_j) {
                    (1, 1) => 0u8,
                    (1, 4) | (4, 1) => 1u8,
                    (4, 4) => 2u8,
                    _ => continue,
                };

                // Orient block_type 1: s must be atom_i
                let (atom_i, atom_j, orb_i, orb_j, l, m, n_val) =
                    if block_type == 1 && n_orb_i == 4 && n_orb_j == 1 {
                        // swap so s is i
                        (
                            j as u16, i as u16,
                            orb_off_j, orb_off_i,
                            (-dx * inv_r) as f32, (-dy * inv_r) as f32, (-dz * inv_r) as f32,
                        )
                    } else {
                        (
                            i as u16, j as u16,
                            orb_off_i, orb_off_j,
                            (dx * inv_r) as f32, (dy * inv_r) as f32, (dz * inv_r) as f32,
                        )
                    };

                let sp_i = global_atom_species[atom_off + atom_i as usize] as u8;
                let sp_j = global_atom_species[atom_off + atom_j as usize] as u8;
                let sp_min = sp_i.min(sp_j);
                let sp_max = sp_i.max(sp_j);

                let bucket_idx = block_type as usize * n_species * n_species
                    + sp_min as usize * n_species + sp_max as usize;

                let entry = GpuPairEntry {
                    replica: fi as u32,
                    atom_i,
                    atom_j,
                    orb_i: orb_i as u16,
                    orb_j: orb_j as u16,
                    r: r as f32,
                    l,
                    m,
                    n: n_val,
                };

                buckets[bucket_idx].push(entry);
            }
        }
    }

    // Convert flat buckets to GpuPairBucket vec, sort by replica inside each
    let mut out = Vec::new();
    for block_type in 0u8..=2 {
        for sp_i in 0..n_species {
            for sp_j in sp_i..n_species {
                let bucket_idx = block_type as usize * n_species * n_species
                    + sp_i * n_species + sp_j;
                let entries = &mut buckets[bucket_idx];
                if entries.is_empty() {
                    continue;
                }
                entries.sort_by_key(|p| p.replica);
                let sk_key = sp_i as u8 * n_species as u8 + sp_j as u8;
                let sk_table_idx = sk_lookup[sk_key as usize];
                if sk_table_idx == usize::MAX {
                    return Err(DftbError::InvalidInput(format!(
                        "missing SK table for species pair ({}, {})",
                        sp_i, sp_j
                    )));
                }
                out.push(GpuPairBucket {
                    n_pairs: entries.len(),
                    pairs: std::mem::take(entries),
                    block_type,
                    sk_table_idx,
                });
            }
        }
    }

    Ok(out)
}
