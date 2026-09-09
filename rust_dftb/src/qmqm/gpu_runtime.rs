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

    /// Device-to-device copy: allocate a new buffer and copy `src` into it.
    /// No host roundtrip — uses OpenCL `clEnqueueCopyBuffer`.
    pub fn copy_buffer<T: ocl::OclPrm>(&self, src: &Buffer<T>, len: usize) -> Result<Buffer<T>> {
        let dst = Buffer::<T>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(len)
            .build()
            .map_err(map_ocl_err)?;
        src.cmd()
            .queue(&self.queue)
            .copy(&dst, None, None)
            .enq()
            .map_err(map_ocl_err)?;
        Ok(dst)
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

/// Query device capabilities at startup using real OpenCL device queries.
/// Falls back to conservative defaults only if a specific query fails.
fn query_capabilities(device: &Device) -> GpuCapabilities {
    use ocl::enums::DeviceInfo;

    let name = match device.info(DeviceInfo::Name) {
        Ok(ocl::enums::DeviceInfoResult::Name(s)) => s,
        _ => format!("{device}"),
    };

    let local_mem_size = device.info(DeviceInfo::LocalMemSize)
        .ok()
        .and_then(|v| if let ocl::enums::DeviceInfoResult::LocalMemSize(n) = v { Some(n) } else { None })
        .unwrap_or(48 * 1024);

    let max_work_group_size = device.info(DeviceInfo::MaxWorkGroupSize)
        .ok()
        .and_then(|v| if let ocl::enums::DeviceInfoResult::MaxWorkGroupSize(n) = v { Some(n as usize) } else { None })
        .unwrap_or(1024);

    // PreferredWorkGroupSizeMultiple is a kernel property, not a device
    // property in OpenCL 1.2. Use 32 (NVIDIA warp size) as default; agents
    // should query kernel-specific values via Kernel::wg_info if needed.
    let preferred_wg_multiple = 32;

    let compute_units = device.info(DeviceInfo::MaxComputeUnits)
        .ok()
        .and_then(|v| if let ocl::enums::DeviceInfoResult::MaxComputeUnits(n) = v { Some(n) } else { None })
        .unwrap_or(1);

    let global_mem_size = device.info(DeviceInfo::GlobalMemSize)
        .ok()
        .and_then(|v| if let ocl::enums::DeviceInfoResult::GlobalMemSize(n) = v { Some(n) } else { None })
        .unwrap_or(0);

    GpuCapabilities {
        local_mem_size,
        max_work_group_size,
        preferred_wg_multiple,
        name,
        compute_units,
        global_mem_size,
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
