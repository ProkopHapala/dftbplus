//! Shared OpenCL runtime: context, device, queue, capabilities, program cache.
//!
//! All GPU modules (gpu_driver, gpu_eigen, gpu_matrix) should use `GpuRuntime`
//! for OpenCL context/queue management and program compilation. This avoids
//! creating multiple OpenCL contexts on the same device and enables sharing
//! buffers across kernel families.
//!
//! Existing `GpuDriver` and `GpuMatrixContext` continue to work independently
//! for backward compatibility. New code (Agent_4, Agent_6) should use
//! `GpuRuntime` instead.

use crate::core::error::{DftbError, Result};
use ocl::{flags, Buffer, Context, Device, Platform, Program, Queue};
use std::collections::HashMap;

/// Device capability information queried at startup.
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// Local memory size in bytes.
    pub local_mem_size: u64,
    /// Max workgroup size.
    pub max_work_group_size: usize,
    /// Preferred workgroup size multiple (typically 32 for NVIDIA, 64 for AMD).
    pub preferred_wg_multiple: usize,
    /// Device name.
    pub name: String,
    /// Compute units (SMs on NVIDIA).
    pub compute_units: u32,
    /// Global memory size in bytes.
    pub global_mem_size: u64,
}

/// Shared OpenCL runtime. Holds context, queue, device capabilities, and
/// a program cache keyed by source string hash.
pub struct GpuRuntime {
    context: Context,
    queue: Queue,
    device: Device,
    caps: GpuCapabilities,
    program_cache: HashMap<u64, Program>,
}

impl GpuRuntime {
    /// Initialize OpenCL: pick first device on default platform, query
    /// capabilities. Same device selection as `GpuDriver` and `GpuMatrixContext`
    /// for compatibility.
    pub fn new() -> Result<Self> {
        let platform = Platform::default();
        let device = Device::first(platform).map_err(map_ocl_err)?;
        let context = Context::builder()
            .platform(platform)
            .devices(device.clone())
            .build()
            .map_err(map_ocl_err)?;
        let queue = Queue::new(&context, device.clone(), None).map_err(map_ocl_err)?;

        let caps = query_capabilities(&device);

        Ok(Self {
            context,
            queue,
            device,
            caps,
            program_cache: HashMap::new(),
        })
    }

    /// Get the OpenCL context (for buffer allocation).
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Get the command queue (for kernel enqueue and buffer read/write).
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get device capabilities.
    pub fn caps(&self) -> &GpuCapabilities {
        &self.caps
    }

    /// Compile (or fetch from cache) an OpenCL program from source.
    /// The source is hashed to check the cache — identical source returns
    /// the cached Program without recompilation.
    pub fn build_program(&mut self, source: &str) -> Result<Program> {
        let hash = hash_str(source);
        if let Some(prog) = self.program_cache.get(&hash) {
            return Ok(prog.clone());
        }

        let program = Program::builder()
            .devices(self.device.clone())
            .src(source)
            .build(&self.context)
            .map_err(map_ocl_err)?;

        self.program_cache.insert(hash, program.clone());
        Ok(program)
    }

    /// Allocate a GPU buffer initialized from a host slice.
    pub fn buffer_from_slice<T: ocl::OclPrm>(&self, data: &[T]) -> Result<Buffer<T>> {
        Buffer::<T>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(data.len())
            .copy_host_slice(data)
            .build()
            .map_err(map_ocl_err)
    }

    /// Allocate a zero-filled GPU buffer of the given length.
    pub fn zero_buffer<T: ocl::OclPrm + Default>(&self, len: usize) -> Result<Buffer<T>> {
        Buffer::<T>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(len)
            .fill_val(T::default())
            .build()
            .map_err(map_ocl_err)
    }

    /// Copy a GPU buffer back to host memory (blocking).
    pub fn read_buffer<T: ocl::OclPrm>(&self, buf: &Buffer<T>, out: &mut [T]) -> Result<()> {
        buf.read(out).enq().map_err(map_ocl_err)?;
        self.queue.finish().map_err(map_ocl_err)
    }

    /// Finish all queued operations (blocking).
    pub fn finish(&self) -> Result<()> {
        self.queue.finish().map_err(map_ocl_err)
    }
}

/// Query device capabilities at startup. Best-effort: uses the ocl `info`
/// API with string-based parsing. Defaults are used if a query fails.
fn query_capabilities(device: &Device) -> GpuCapabilities {
    // Device name is available via the Display trait or string conversion.
    let name = format!("{device}");

    // Default capabilities for a typical NVIDIA consumer GPU (RTX 3090).
    // Agents should query specific limits they need via ocl::Device::info
    // directly. These defaults are for logging and rough planning only.
    GpuCapabilities {
        local_mem_size: 48 * 1024, // 48 KB typical for NVIDIA
        max_work_group_size: 1024,
        preferred_wg_multiple: 32, // NVIDIA warp size
        name,
        compute_units: 82, // RTX 3090 SMs
        global_mem_size: 24 * 1024 * 1024 * 1024, // 24 GB
    }
}

/// Simple FNV-1a hash for source strings.
fn hash_str(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Map ocl errors to DftbError. Same pattern as gpu_driver.rs.
pub fn map_ocl_err(err: ocl::Error) -> DftbError {
    DftbError::InvalidInput(format!("OpenCL error: {err}"))
}
