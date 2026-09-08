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

use crate::core::error::{DftbError, Result};
use crate::qmqm::gpu_prep::{GpuBatch, GpuFragment, GpuPairEntry};
use ocl::{flags, Buffer, Context, Device, Kernel, Platform, Program, Queue};

const FORCE_SOURCE: &str = include_str!("gpu_forces.cl");

// Reuse OclPrm impls from gpu_driver (same types, same crate).
// OclPrm is a trait — impls are coherent because the types are local.

/// GPU force driver: holds context, queue, and compiled force program.
pub struct GpuForceDriver {
    #[allow(dead_code)]
    context: Context,
    queue: Queue,
    program: Program,
}

impl GpuForceDriver {
    /// Initialize OpenCL: pick first device on default platform, compile
    /// `gpu_forces.cl`.
    pub fn new() -> Result<Self> {
        let platform = Platform::default();
        let device = Device::first(platform).map_err(map_ocl_err)?;
        let context = Context::builder()
            .platform(platform)
            .devices(device.clone())
            .build()
            .map_err(map_ocl_err)?;
        let queue = Queue::new(&context, device.clone(), None).map_err(map_ocl_err)?;
        eprintln!("GpuForceDriver: platform={:?} device={:?}",
            platform.name().unwrap_or_default(), device.name().unwrap_or_default());
        let program = Program::builder()
            .devices(device)
            .src(FORCE_SOURCE)
            .build(&context)
            .map_err(map_ocl_err)?;
        Ok(Self { context, queue, program })
    }

    /// Compute non-SCC electronic forces for a batch of replicas.
    ///
    /// `dm` and `edm` are flat row-major f32 arrays of length
    /// `batch.total_h_elements`, layout `[replica][i][j]` at
    /// `replica*N*N + i*N + j`.
    ///
    /// Returns forces as `Vec<f32>` of length `3 * batch.total_atoms`,
    /// layout `[atom][3]`, in Hartree/Å.
    pub fn gpu_force_batched(
        &self,
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

        // Output force buffer, zero-initialized
        let buf_forces = Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(3 * total_atoms)
            .copy_host_slice(&vec![0.0f32; 3 * total_atoms])
            .build()
            .map_err(map_ocl_err)?;

        let buf_fragments = self.upload_fragments(&batch.fragments)?;
        let buf_dm = self.upload_f32(dm)?;
        let buf_edm = self.upload_f32(edm)?;

        // One launch per species-pair bucket
        for (bi, bucket) in batch.pair_buckets.iter().enumerate() {
            let sk_table = &batch.sk_tables[bucket.sk_table_idx];
            let n_pairs = bucket.n_pairs;
            eprintln!("launching bucket {bi}: block_type={} n_pairs={} n_grid={} n_sk_cols={} dr={}",
                bucket.block_type, n_pairs, sk_table.n_grid, sk_table.n_sk_cols, sk_table.dr);
            if n_pairs == 0 { continue; }

            let buf_pairs = self.upload_pairs(&bucket.pairs)?;
            let buf_sk_h = self.upload_f32(&sk_table.sk_h)?;
            let buf_sk_s = self.upload_f32(&sk_table.sk_s)?;

            let local_pairs = 64usize;
            let global_pairs = ((n_pairs + local_pairs - 1) / local_pairs) * local_pairs;

            let k = Kernel::builder()
                .program(&self.program)
                .name("force_pairs")
                .queue(self.queue.clone())
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
            // Finish after each launch to isolate which bucket crashes
            if let Err(e) = self.queue.finish() {
                return Err(map_ocl_err(e));
            }
            eprintln!("  bucket {bi} OK");
        }

        self.queue.finish().map_err(map_ocl_err)?;

        let mut forces_host = vec![0.0f32; 3 * total_atoms];
        buf_forces.read(&mut forces_host).enq().map_err(map_ocl_err)?;
        self.queue.finish().map_err(map_ocl_err)?;
        Ok(forces_host)
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
}

fn map_ocl_err(err: ocl::Error) -> DftbError {
    DftbError::InvalidInput(format!("OpenCL error: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_driver_constructs_if_gpu_present() {
        match GpuForceDriver::new() {
            Ok(_d) => {}
            Err(e) => eprintln!("Skipping: no OpenCL device ({e})"),
        }
    }
}
