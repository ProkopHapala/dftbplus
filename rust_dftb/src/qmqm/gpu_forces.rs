//! GPU analytic force kernel driver (Phase 4).
//!
//! Compiles `gpu_forces.cl` and launches `force_pairs` once per species-pair
//! bucket, accumulating non-SCC electronic forces via atomic_add.
//!
//! Force formula (matching CPU `non_scc_electronic_force` in forces.rs):
//!   F_i[a] += 2 * ANG2BOHR * Σ_{μ∈i,ν∈j} (DM[μ,ν]·dH[μ,ν]/dR_a - EDM[μ,ν]·dS[μ,ν]/dR_a)
//!   F_j[a] -= same
//!
//! Layout contract:
//!   dm, edm : `Buffer<f32>`, row-major `[replica][i][j]` at
//!             `replica*N*N + i*N + j`, f32 (same as H/S).
//!   forces  : `Buffer<f32>`, `[total_atoms][3]`, f32, Hartree/Å.
//!
//! R10: Uses the shared `GpuRuntime` (no separate context/queue/program).
//! Eliminates per-bucket `finish()` calls — in-order queue preserves
//! command order. Eliminates per-bucket uploads for fragments/dm/edm
//! (uploaded once per call). SK tables and pair data are uploaded per-bucket
//! (small, batch structure may change between calls).

use crate::core::error::{DftbError, Result};
use crate::qmqm::gpu_prep::{GpuBatch, GpuFragment, GpuPairEntry};
use crate::qmqm::gpu_runtime::{map_ocl_err as rt_map_err, GpuRuntime};
use ocl::{flags, Buffer, Kernel};

const FORCE_SOURCE: &str = include_str!("gpu_forces.cl");

/// GPU force driver: uses the shared `GpuRuntime` (R10: no separate context).
pub struct GpuForceDriver {
    /// Borrowed reference to the shared runtime. The driver does NOT own
    /// the runtime — the caller owns it and must keep it alive.
    /// We store the compiled program handle for kernel building.
    program: ocl::Program,
}

impl GpuForceDriver {
    /// Initialize using the shared `GpuRuntime`. Compiles `gpu_forces.cl`
    /// via the runtime's program cache (no separate context/queue).
    pub fn new(rt: &mut GpuRuntime) -> Result<Self> {
        let program = rt.build_program(FORCE_SOURCE)?;
        Ok(Self { program })
    }

    /// Compute non-SCC electronic forces for a batch of replicas.
    ///
    /// `dm` and `edm` are flat row-major f32 arrays of length
    /// `batch.total_h_elements`, layout `[replica][i][j]` at
    /// `replica*N*N + i*N + j`.
    ///
    /// Returns forces as `Vec<f32>` of length `3 * batch.total_atoms`,
    /// layout `[atom][3]`, in Hartree/Å.
    ///
    /// R10: No per-bucket `finish()` — in-order queue preserves order.
    /// Only one `finish()` at the end before reading back forces.
    pub fn gpu_force_batched(
        &self,
        rt: &GpuRuntime,
        batch: &GpuBatch,
        dm: &[f32],
        edm: &[f32],
    ) -> Result<Vec<f32>> {
        let total_atoms = batch.total_atoms;
        let n_frags = batch.n_frags;

        assert_eq!(dm.len(), batch.total_h_elements,
            "dm length {} != total_h_elements {}", dm.len(), batch.total_h_elements);
        assert_eq!(edm.len(), batch.total_h_elements,
            "edm length {} != total_h_elements {}", edm.len(), batch.total_h_elements);

        let queue = rt.queue();

        // Output force buffer, zero-initialized
        let buf_forces = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(3 * total_atoms)
            .copy_host_slice(&vec![0.0f32; 3 * total_atoms])
            .build()
            .map_err(map_ocl_err)?;

        // Upload once per call (not per-bucket)
        let buf_fragments = upload_fragments(queue, &batch.fragments)?;
        let buf_dm = upload_f32(queue, dm)?;
        let buf_edm = upload_f32(queue, edm)?;

        // One launch per species-pair bucket — no finish() between buckets
        for (bi, bucket) in batch.pair_buckets.iter().enumerate() {
            let sk_table = &batch.sk_tables[bucket.sk_table_idx];
            let n_pairs = bucket.n_pairs;
            if n_pairs == 0 { continue; }

            let buf_pairs = upload_pairs(queue, &bucket.pairs)?;
            let buf_sk_h = upload_f32(queue, &sk_table.sk_h)?;
            let buf_sk_s = upload_f32(queue, &sk_table.sk_s)?;

            let local_pairs = 64usize;
            let global_pairs = ((n_pairs + local_pairs - 1) / local_pairs) * local_pairs;

            let k = Kernel::builder()
                .program(&self.program)
                .name("force_pairs")
                .queue(queue.clone())
                .global_work_size(global_pairs)
                .local_work_size(local_pairs)
                .arg(&buf_pairs)
                .arg(&buf_fragments)
                .arg(&buf_sk_h)
                .arg(&buf_sk_s)
                .arg(&buf_dm)
                .arg(&buf_edm)
                .arg(&buf_forces)
                .arg(sk_table.dr)
                .arg(sk_table.n_grid as i32)
                .arg(n_pairs as i32)
                .arg(n_frags as i32)
                .arg(bucket.block_type as i32)
                .arg(sk_table.n_sk_cols as i32)
                .build()
                .map_err(map_ocl_err)?;
            unsafe { k.enq().map_err(map_ocl_err)?; }
            // R10: no finish() between buckets — in-order queue preserves order
        }

        // Single read+finish at the end
        let mut forces_host = vec![0.0f32; 3 * total_atoms];
        buf_forces.read(&mut forces_host).enq().map_err(map_ocl_err)?;
        queue.finish().map_err(map_ocl_err)?;
        Ok(forces_host)
    }

    /// Compute non-SCC electronic forces from GPU-resident DM/EDM buffers.
    /// R10: avoids host roundtrip when DM/EDM are already on the GPU.
    pub fn gpu_force_batched_dev(
        &self,
        rt: &GpuRuntime,
        batch: &GpuBatch,
        dm_buf: &Buffer<f32>,
        edm_buf: &Buffer<f32>,
    ) -> Result<Vec<f32>> {
        let total_atoms = batch.total_atoms;
        let n_frags = batch.n_frags;
        let queue = rt.queue();

        let buf_forces = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(3 * total_atoms)
            .copy_host_slice(&vec![0.0f32; 3 * total_atoms])
            .build()
            .map_err(map_ocl_err)?;

        let buf_fragments = upload_fragments(queue, &batch.fragments)?;

        for bucket in &batch.pair_buckets {
            let sk_table = &batch.sk_tables[bucket.sk_table_idx];
            let n_pairs = bucket.n_pairs;
            if n_pairs == 0 { continue; }

            let buf_pairs = upload_pairs(queue, &bucket.pairs)?;
            let buf_sk_h = upload_f32(queue, &sk_table.sk_h)?;
            let buf_sk_s = upload_f32(queue, &sk_table.sk_s)?;

            let local_pairs = 64usize;
            let global_pairs = ((n_pairs + local_pairs - 1) / local_pairs) * local_pairs;

            let k = Kernel::builder()
                .program(&self.program)
                .name("force_pairs")
                .queue(queue.clone())
                .global_work_size(global_pairs)
                .local_work_size(local_pairs)
                .arg(&buf_pairs)
                .arg(&buf_fragments)
                .arg(&buf_sk_h)
                .arg(&buf_sk_s)
                .arg(dm_buf)
                .arg(edm_buf)
                .arg(&buf_forces)
                .arg(sk_table.dr)
                .arg(sk_table.n_grid as i32)
                .arg(n_pairs as i32)
                .arg(n_frags as i32)
                .arg(bucket.block_type as i32)
                .arg(sk_table.n_sk_cols as i32)
                .build()
                .map_err(map_ocl_err)?;
            unsafe { k.enq().map_err(map_ocl_err)?; }
        }

        let mut forces_host = vec![0.0f32; 3 * total_atoms];
        buf_forces.read(&mut forces_host).enq().map_err(map_ocl_err)?;
        queue.finish().map_err(map_ocl_err)?;
        Ok(forces_host)
    }

    // ------------------------------------------------------------------
    // R6: SCC shift force (component 2)
    // ------------------------------------------------------------------

    /// Compute SCC shift forces for a batch of replicas.
    ///
    /// Formula: F_i[a] += 2*ANG2BOHR * Σ (0.5*(V_i+V_j) * dS/dR_a * DM)
    ///
    /// `dm` is the density matrix, `v_shift` is the SCC potential per atom
    /// (length `batch.total_atoms`).
    pub fn gpu_scc_shift_force_batched(
        &self,
        rt: &GpuRuntime,
        batch: &GpuBatch,
        dm: &[f32],
        v_shift: &[f32],
    ) -> Result<Vec<f32>> {
        let total_atoms = batch.total_atoms;
        let n_frags = batch.n_frags;
        let queue = rt.queue();

        assert_eq!(dm.len(), batch.total_h_elements,
            "dm length {} != total_h_elements {}", dm.len(), batch.total_h_elements);
        assert_eq!(v_shift.len(), total_atoms,
            "v_shift length {} != total_atoms {}", v_shift.len(), total_atoms);

        let buf_forces = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(3 * total_atoms)
            .copy_host_slice(&vec![0.0f32; 3 * total_atoms])
            .build()
            .map_err(map_ocl_err)?;

        let buf_fragments = upload_fragments(queue, &batch.fragments)?;
        let buf_dm = upload_f32(queue, dm)?;
        let buf_v_shift = upload_f32(queue, v_shift)?;

        for bucket in &batch.pair_buckets {
            let sk_table = &batch.sk_tables[bucket.sk_table_idx];
            let n_pairs = bucket.n_pairs;
            if n_pairs == 0 { continue; }

            let buf_pairs = upload_pairs(queue, &bucket.pairs)?;
            let buf_sk_s = upload_f32(queue, &sk_table.sk_s)?;

            let local_pairs = 64usize;
            let global_pairs = ((n_pairs + local_pairs - 1) / local_pairs) * local_pairs;

            let k = Kernel::builder()
                .program(&self.program)
                .name("force_pairs_scc_shift")
                .queue(queue.clone())
                .global_work_size(global_pairs)
                .local_work_size(local_pairs)
                .arg(&buf_pairs)
                .arg(&buf_fragments)
                .arg(&buf_sk_s)
                .arg(&buf_dm)
                .arg(&buf_v_shift)
                .arg(&buf_forces)
                .arg(sk_table.dr)
                .arg(sk_table.n_grid as i32)
                .arg(n_pairs as i32)
                .arg(n_frags as i32)
                .arg(bucket.block_type as i32)
                .arg(sk_table.n_sk_cols as i32)
                .build()
                .map_err(map_ocl_err)?;
            unsafe { k.enq().map_err(map_ocl_err)?; }
        }

        let mut forces_host = vec![0.0f32; 3 * total_atoms];
        buf_forces.read(&mut forces_host).enq().map_err(map_ocl_err)?;
        queue.finish().map_err(map_ocl_err)?;
        Ok(forces_host)
    }

    // ------------------------------------------------------------------
    // R6: Gamma derivative force (component 3)
    // ------------------------------------------------------------------

    /// Compute SCC double-counting (gamma derivative) forces.
    ///
    /// Formula: F_i[a] += -dq_i*dq_j*gamma'(r)/r * (coord_i - coord_j)_a * ANG2BOHR^2
    ///
    /// `coords` is [batch*n_atoms*3] in Å, `species_idx` is [batch*n_atoms],
    /// `delta_q` is [batch*n_atoms], `u_hub` is [n_species].
    pub fn gpu_gamma_deriv_force_batched(
        &self,
        rt: &GpuRuntime,
        n_atoms: usize,
        batch: usize,
        coords: &[f32],
        species_idx: &[i32],
        delta_q: &[f32],
        u_hub: &[f32],
        n_species: usize,
    ) -> Result<Vec<f32>> {
        let queue = rt.queue();
        let total_atoms = batch * n_atoms;

        assert_eq!(coords.len(), total_atoms * 3, "coords length mismatch");
        assert_eq!(species_idx.len(), total_atoms, "species_idx length mismatch");
        assert_eq!(delta_q.len(), total_atoms, "delta_q length mismatch");
        assert_eq!(u_hub.len(), n_species, "u_hub length mismatch");

        let buf_coords = upload_f32(queue, coords)?;
        let buf_species = upload_i32(queue, species_idx)?;
        let buf_dq = upload_f32(queue, delta_q)?;
        let buf_u_hub = upload_f32(queue, u_hub)?;
        let buf_forces = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(3 * total_atoms)
            .copy_host_slice(&vec![0.0f32; 3 * total_atoms])
            .build()
            .map_err(map_ocl_err)?;

        let wg = 256usize;
        let k = Kernel::builder()
            .program(&self.program)
            .name("force_gamma_deriv_batched")
            .queue(queue.clone())
            .global_work_size(batch * wg)
            .local_work_size(wg)
            .arg(n_atoms as i32).arg(batch as i32)
            .arg(&buf_coords).arg(&buf_species).arg(&buf_dq).arg(&buf_u_hub)
            .arg(n_species as i32).arg(&buf_forces)
            .build()
            .map_err(map_ocl_err)?;
        unsafe { k.enq().map_err(map_ocl_err)?; }

        let mut forces_host = vec![0.0f32; 3 * total_atoms];
        buf_forces.read(&mut forces_host).enq().map_err(map_ocl_err)?;
        queue.finish().map_err(map_ocl_err)?;
        Ok(forces_host)
    }

    // ------------------------------------------------------------------
    // R6: Repulsive pair force (component 4)
    // ------------------------------------------------------------------

    /// Compute repulsive pair-potential forces.
    ///
    /// Formula: F_i[a] += dE_rep/dr * (coord_j - coord_i)_a / r * ANG2BOHR
    ///
    /// `coords` is [batch*n_atoms*3] in Bohr, `species_idx` is [batch*n_atoms],
    /// `spline_offsets` is [n_species*n_species], `spline_data` is flat.
    pub fn gpu_repulsive_force_batched(
        &self,
        rt: &mut GpuRuntime,
        n_atoms: usize,
        batch: usize,
        coords: &[f32],
        species_idx: &[i32],
        spline_offsets: &[i32],
        spline_data: &[f32],
        n_species: usize,
        max_intervals: usize,
    ) -> Result<Vec<f32>> {
        let queue = rt.queue().clone();
        let total_atoms = batch * n_atoms;

        let buf_coords = upload_f32(&queue, coords)?;
        let buf_species = upload_i32(&queue, species_idx)?;
        let buf_offsets = upload_i32(&queue, spline_offsets)?;

        // Build kernel with REP_MAX_INTERVALS specialization
        let source = FORCE_SOURCE.replace("#define REP_MAX_INTERVALS 30",
            &format!("#define REP_MAX_INTERVALS {}", max_intervals));
        let prog = rt.build_program(&source)?;

        let buf_spline_data = upload_f32(&queue, spline_data)?;
        let buf_forces = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(3 * total_atoms)
            .copy_host_slice(&vec![0.0f32; 3 * total_atoms])
            .build()
            .map_err(map_ocl_err)?;

        let wg = 256usize;
        let k = Kernel::builder()
            .program(&prog)
            .name("force_repulsive_batched")
            .queue(queue.clone())
            .global_work_size(batch * wg)
            .local_work_size(wg)
            .arg(n_atoms as i32).arg(batch as i32)
            .arg(&buf_coords).arg(&buf_species).arg(&buf_offsets)
            .arg(n_species as i32).arg(&buf_spline_data).arg(&buf_forces)
            .build()
            .map_err(map_ocl_err)?;
        unsafe { k.enq().map_err(map_ocl_err)?; }

        let mut forces_host = vec![0.0f32; 3 * total_atoms];
        buf_forces.read(&mut forces_host).enq().map_err(map_ocl_err)?;
        queue.finish().map_err(map_ocl_err)?;
        Ok(forces_host)
    }
}

// ---- Upload helpers (use shared queue, no separate context) ----

fn upload_f32(queue: &ocl::Queue, data: &[f32]) -> Result<Buffer<f32>> {
    Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(data.len())
        .copy_host_slice(data)
        .build()
        .map_err(map_ocl_err)
}

fn upload_i32(queue: &ocl::Queue, data: &[i32]) -> Result<Buffer<i32>> {
    Buffer::<i32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(data.len())
        .copy_host_slice(data)
        .build()
        .map_err(map_ocl_err)
}

fn upload_fragments(queue: &ocl::Queue, data: &[GpuFragment]) -> Result<Buffer<GpuFragment>> {
    Buffer::<GpuFragment>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(data.len())
        .copy_host_slice(data)
        .build()
        .map_err(map_ocl_err)
}

fn upload_pairs(queue: &ocl::Queue, data: &[GpuPairEntry]) -> Result<Buffer<GpuPairEntry>> {
    Buffer::<GpuPairEntry>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(data.len())
        .copy_host_slice(data)
        .build()
        .map_err(map_ocl_err)
}

fn map_ocl_err(err: ocl::Error) -> DftbError {
    rt_map_err(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_driver_constructs_if_gpu_present() {
        match GpuRuntime::new() {
            Ok(mut rt) => {
                let _ = GpuForceDriver::new(&mut rt).expect("force driver should construct");
            }
            Err(e) => eprintln!("Skipping: no OpenCL device ({e})"),
        }
    }
}
