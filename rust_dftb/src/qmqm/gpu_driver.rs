//! OpenCL driver for batched DFTB Hamiltonian/overlap assembly.
//!
//! Compiles `dftb_hamiltonian.cl`, uploads a `GpuBatch` to device buffers,
//! launches `onsite_diagonal`, `onsite_and_va` (V_A electrostatics), and one
//! `assemble_pairs` launch per species-pair bucket, then reads back H and S.
//!
//! Scope (Agent_1, Wave 1): non-SCC H0/S assembly + V_A only.
//! SCC loop, scan/NEB, and kernel edits are out of scope (other agents).
//!
//! Layout contract (frozen, master contract v1):
//!   H,S : `Buffer<f32>`, row-major, `[replica][i][j]` at
//!         `replica*N*N + i*N + j`, f32. Angstrom→Bohr done in `gpu_prep`.

use crate::core::error::{DftbError, Result};
use crate::qmqm::gpu_prep::{GpuBatch, GpuFragment, GpuPairEntry};
use ocl::prm::Float2;
use ocl::{flags, Buffer, Context, Device, Kernel, Platform, Program, Queue};

/// OpenCL source for the Hamiltonian assembly kernels.
const HAM_SOURCE: &str = include_str!("../methods/dftb/dftb_hamiltonian.cl");

// ---------------------------------------------------------------------------
// OclPrm impls for the GPU structs.
//
// `GpuFragment` and `GpuPairEntry` are `#[repr(C)]` in `gpu_prep.rs` (which is
// read-only for this agent). They already derive `Debug + Clone + Copy` and
// are `Send + Sync + 'static` (all fields are plain primitives). They are
// missing `Default` and `PartialEq`, which `OclPrm` requires as supertraits.
// Both the type (local to this crate) and the traits (`std::default::Default`,
// `std::cmp::PartialEq`, `ocl_core::OclPrm`) are implable here under the
// orphan rule (the type is local to our crate). The structs are `#[repr(C)]`
// with plain scalar fields, so the layout/alignment invariant required by the
// `unsafe impl OclPrm` is satisfied.
// ---------------------------------------------------------------------------

impl Default for GpuFragment {
    fn default() -> Self {
        Self { n_atoms: 0, n_orbs: 0, atom_off: 0, h_base: 0 }
    }
}
impl PartialEq for GpuFragment {
    fn eq(&self, o: &Self) -> bool {
        self.n_atoms == o.n_atoms
            && self.n_orbs == o.n_orbs
            && self.atom_off == o.atom_off
            && self.h_base == o.h_base
    }
}
unsafe impl ocl::OclPrm for GpuFragment {}

impl Default for GpuPairEntry {
    fn default() -> Self {
        Self {
            replica: 0, atom_i: 0, atom_j: 0, orb_i: 0, orb_j: 0,
            r: 0.0, l: 0.0, m: 0.0, n: 0.0,
        }
    }
}
impl PartialEq for GpuPairEntry {
    fn eq(&self, o: &Self) -> bool {
        self.replica == o.replica
            && self.atom_i == o.atom_i
            && self.atom_j == o.atom_j
            && self.orb_i == o.orb_i
            && self.orb_j == o.orb_j
            && self.r == o.r
            && self.l == o.l
            && self.m == o.m
            && self.n == o.n
    }
}
unsafe impl ocl::OclPrm for GpuPairEntry {}

/// OpenCL driver: holds context, queue, and compiled Hamiltonian program.
///
/// Reuses the same device selection strategy as `GpuMatrixContext` (first
/// available device on the default platform) so the two contexts can coexist
/// on the same GPU during Wave 1 (only one agent runs GPU tests at a time per
/// the master's GPU-serialization rule).
pub struct GpuDriver {
    context: Context,
    queue: Queue,
    program: Program,
}

impl GpuDriver {
    /// Initialize OpenCL: pick first device on default platform, compile
    /// `dftb_hamiltonian.cl`.
    pub fn new() -> Result<Self> {
        let platform = Platform::default();
        let device = Device::first(platform).map_err(map_ocl_err)?;
        let context = Context::builder()
            .platform(platform)
            .devices(device.clone())
            .build()
            .map_err(map_ocl_err)?;
        let queue = Queue::new(&context, device.clone(), None).map_err(map_ocl_err)?;
        let program = Program::builder()
            .devices(device)
            .src(HAM_SOURCE)
            .build(&context)
            .map_err(map_ocl_err)?;
        Ok(Self { context, queue, program })
    }

    /// Assemble H0/S for an entire batch of replicas in one set of kernel
    /// launches (one `onsite_diagonal`, one `onsite_and_va`, one
    /// `assemble_pairs` per species-pair bucket).
    ///
    /// Returns `(H_host, S_host)` as flat row-major `Vec<f32>` of length
    /// `batch.total_h_elements` each, in the layout described in the module
    /// docs (`[replica][i][j]` at `replica*N*N + i*N + j`).
    pub fn gpu_assemble_batched(&self, batch: &GpuBatch) -> Result<(Vec<f32>, Vec<f32>)> {
        let total_h = batch.total_h_elements;
        let total_atoms = batch.total_atoms;
        let n_frags = batch.n_frags;
        let n_global_species = batch.n_global_species;

        // --- Output buffers ---
        let buf_h = Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(total_h)
            .fill_val(0.0f32)
            .build()
            .map_err(map_ocl_err)?;

        // S must start as identity per replica (the `onsite_diagonal` kernel
        // only writes H; the CPU reference `build_non_scc` inits S = I). We
        // build a host vector with 1.0 on each replica's diagonal and upload
        // it as the initial S. `assemble_pairs` then fills the off-diagonal
        // blocks; the diagonal stays 1.0.
        let s_init = build_s_identity_init(&batch.fragments, total_h);
        let buf_s = Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(total_h)
            .copy_host_slice(&s_init)
            .build()
            .map_err(map_ocl_err)?;
        let buf_v = Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(total_atoms)
            .fill_val(0.0f32)
            .build()
            .map_err(map_ocl_err)?;

        // --- Per-atom / per-fragment / per-species buffers ---
        let buf_fragments = self.upload_fragments(&batch.fragments)?;
        let buf_atom_species = self.upload_i32(&batch.atom_species)?;
        let buf_orb_off = self.upload_i32(&batch.atom_orb_off)?;
        // Per-atom orbital count: needed by `onsite_diagonal` to avoid writing
        // e_p into positions that belong to the next atom (s-only species bug).
        let n_orb_per_atom = build_n_orb_per_atom(&batch.fragments, &batch.atom_orb_off);
        let buf_n_orb_per_atom = self.upload_i32(&n_orb_per_atom)?;
        // For non-SCC H0 assembly the SCC shift (H1 = 0.5*(V_A_i+V_A_j)*S) must
        // be zero. `onsite_and_va` computes V_A from `charges`; with zero
        // charges V_A = 0 and `assemble_pairs` writes pure H0 = sk_h. The
        // `GpuBatch.charges` field holds q0 (neutral valence counts, nonzero),
        // which would inject a spurious H1 term. Pass a zeroed charge buffer
        // instead. (Agent_4 / SCC will pass deltaQ here.)
        let zero_charges = vec![0.0f32; batch.charges.len()];
        let buf_charges = self.upload_f32(&zero_charges)?;
        let buf_hubbard_u = self.upload_f32(&batch.hubbard_u)?;
        let buf_onsite = self.upload_onsite(&batch.onsite_es_ep, n_global_species)?;

        // --- Kernel 0: onsite H0 diagonal ---
        // Each thread writes one atom's diagonal (up to 4 values).
        let wg_onsite = 64.min(total_atoms.max(1));
        let g_onsite = ((total_atoms + wg_onsite - 1) / wg_onsite) * wg_onsite;
        let k_onsite = Kernel::builder()
            .program(&self.program)
            .name("onsite_diagonal")
            .queue(self.queue.clone())
            .global_work_size(g_onsite)
            .local_work_size(wg_onsite)
            .arg(&buf_fragments)
            .arg(&buf_atom_species)
            .arg(&buf_orb_off)
            .arg(&buf_n_orb_per_atom)
            .arg(&buf_onsite)
            .arg(&buf_h)
            .arg(n_frags as i32)
            .arg(total_atoms as i32)
            .build()
            .map_err(map_ocl_err)?;
        unsafe { k_onsite.enq().map_err(map_ocl_err)?; }

        // --- Kernel 1: V_A (gamma electrostatics), one workgroup per fragment ---
        let gamma_neigh = &batch.gamma_neigh;
        let buf_neigh_off = self.upload_i32(&gamma_neigh.offsets)?;
        let buf_neigh_j = self.upload_i32(&gamma_neigh.neigh_j)?;
        let buf_neigh_r = self.upload_f32(&gamma_neigh.neigh_r)?;

        let local_va = 32usize;
        let global_va = n_frags * local_va;
        let k_va = Kernel::builder()
            .program(&self.program)
            .name("onsite_and_va")
            .queue(self.queue.clone())
            .global_work_size(global_va)
            .local_work_size(local_va)
            .arg(&buf_fragments)
            .arg(&buf_atom_species)
            .arg(&buf_charges)
            .arg(&buf_hubbard_u)
            .arg(&buf_neigh_off)
            .arg(&buf_neigh_j)
            .arg(&buf_neigh_r)
            .arg(&buf_v)
            .arg(n_frags as i32)
            .arg(n_global_species as i32)
            .build()
            .map_err(map_ocl_err)?;
        unsafe { k_va.enq().map_err(map_ocl_err)?; }

        // --- Kernel 2: assemble_pairs, one launch per species-pair bucket ---
        for bucket in &batch.pair_buckets {
            let sk_table = &batch.sk_tables[bucket.sk_table_idx];
            let n_pairs = bucket.n_pairs;
            if n_pairs == 0 {
                continue;
            }

            let buf_pairs = self.upload_pairs(&bucket.pairs)?;
            let buf_sk_h = self.upload_f32(&sk_table.sk_h)?;
            let buf_sk_s = self.upload_f32(&sk_table.sk_s)?;

            let local_pairs = 64usize;
            let global_pairs = ((n_pairs + local_pairs - 1) / local_pairs) * local_pairs;

            let k_pairs = Kernel::builder()
                .program(&self.program)
                .name("assemble_pairs")
                .queue(self.queue.clone())
                .global_work_size(global_pairs)
                .local_work_size(local_pairs)
                .arg(&buf_pairs)
                .arg(&buf_fragments)
                .arg(&buf_sk_h)
                .arg(&buf_sk_s)
                .arg(&buf_v)
                .arg(&buf_h)
                .arg(&buf_s)
                .arg(sk_table.dr)
                .arg(sk_table.n_grid as i32)
                .arg(n_pairs as i32)
                .arg(n_frags as i32)
                .arg(bucket.block_type as i32)
                .arg(sk_table.n_sk_cols as i32)
                .build()
                .map_err(map_ocl_err)?;
            unsafe { k_pairs.enq().map_err(map_ocl_err)?; }
        }

        self.queue.finish().map_err(map_ocl_err)?;

        // --- Read back H and S ---
        let mut h_host = vec![0.0f32; total_h];
        let mut s_host = vec![0.0f32; total_h];
        buf_h.read(&mut h_host).enq().map_err(map_ocl_err)?;
        buf_s.read(&mut s_host).enq().map_err(map_ocl_err)?;
        self.queue.finish().map_err(map_ocl_err)?;

        Ok((h_host, s_host))
    }

    // ------------------------------------------------------------------
    // Typed upload helpers
    // ------------------------------------------------------------------

    fn upload_f32(&self, data: &[f32]) -> Result<Buffer<f32>> {
        Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
            .len(data.len())
            .copy_host_slice(data)
            .build()
            .map_err(map_ocl_err)
    }

    fn upload_i32(&self, data: &[i32]) -> Result<Buffer<i32>> {
        Buffer::<i32>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
            .len(data.len())
            .copy_host_slice(data)
            .build()
            .map_err(map_ocl_err)
    }

    fn upload_fragments(&self, data: &[GpuFragment]) -> Result<Buffer<GpuFragment>> {
        Buffer::<GpuFragment>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
            .len(data.len())
            .copy_host_slice(data)
            .build()
            .map_err(map_ocl_err)
    }

    fn upload_pairs(&self, data: &[GpuPairEntry]) -> Result<Buffer<GpuPairEntry>> {
        Buffer::<GpuPairEntry>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
            .len(data.len())
            .copy_host_slice(data)
            .build()
            .map_err(map_ocl_err)
    }

    /// `onsite_es_ep` is stored in `GpuBatch` as a flat `Vec<f32>` of length
    /// `2 * n_global_species` (`[e_s_0, e_p_0, e_s_1, e_p_1, ...]`). The kernel
    /// declares it as `__global const float2*`, so we pack it into `Float2`
    /// pairs (one per species: `(e_s, e_p)`).
    fn upload_onsite(&self, flat: &[f32], n_species: usize) -> Result<Buffer<Float2>> {
        assert_eq!(flat.len(), 2 * n_species);
        let pairs: Vec<Float2> = (0..n_species)
            .map(|i| Float2::new(flat[2 * i], flat[2 * i + 1]))
            .collect();
        Buffer::<Float2>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
            .len(n_species)
            .copy_host_slice(&pairs)
            .build()
            .map_err(map_ocl_err)
    }
}

fn map_ocl_err(err: ocl::Error) -> DftbError {
    DftbError::InvalidInput(format!("OpenCL error: {err}"))
}

/// Build a flat S initialization vector with 1.0 on each replica's diagonal.
///
/// Each `GpuFragment` owns a contiguous `n_orbs × n_orbs` block starting at
/// `h_base` (flat row-major). We set `block[i*n_orbs + i] = 1.0` for every
/// replica and orbital; everything else stays 0.0. This matches the CPU
/// reference `HamiltonianBuilder::build_non_scc`, which initializes
/// `s = DMatrix::identity(n_orbs, n_orbs)` before filling off-diagonal pairs.
pub(crate) fn build_s_identity_init(fragments: &[GpuFragment], total_h: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; total_h];
    for frag in fragments {
        let n = frag.n_orbs as usize;
        let base = frag.h_base as usize;
        for i in 0..n {
            s[base + i * n + i] = 1.0;
        }
    }
    s
}

/// Build the per-atom orbital count array from fragment metadata.
///
/// `GpuBatch.atom_orb_off` holds the LOCAL orbital offset of each atom within
/// its fragment (length = total_atoms, NOT total_atoms+1). For atom `a` in
/// fragment `f`, the orbital count is:
///   - `atom_orb_off[a+1] - atom_orb_off[a]` if `a` is not the last atom
///   - `frag.n_orbs - atom_orb_off[a]` if `a` is the last atom in the fragment
///
/// This is needed by the `onsite_diagonal` kernel to avoid writing e_p into
/// diagonal slots that belong to the next atom (the original kernel
/// unconditionally wrote 4 entries per atom).
pub(crate) fn build_n_orb_per_atom(fragments: &[GpuFragment], atom_orb_off: &[i32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(atom_orb_off.len());
    for frag in fragments {
        let off = frag.atom_off as usize;
        let n = frag.n_atoms as usize;
        for a in 0..n {
            let n_orb_a = if a + 1 < n {
                atom_orb_off[off + a + 1] - atom_orb_off[off + a]
            } else {
                frag.n_orbs - atom_orb_off[off + a]
            };
            out.push(n_orb_a);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_constructs_if_gpu_present() {
        match GpuDriver::new() {
            Ok(_d) => {}
            Err(e) => eprintln!("Skipping: no OpenCL device ({e})"),
        }
    }
}
