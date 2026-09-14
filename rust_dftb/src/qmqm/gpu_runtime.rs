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
use ocl::{flags, Buffer, Context, Device, Event, Platform, Program, Queue};
use ocl::enums::{ProfilingInfo, ProfilingInfoResult};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

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

/// Env-gated stage profiler (`RUST_DFTB_PROF`). Interior-mutable so the
/// `&self` sync paths can count without signature changes.
///   =mark  — `prof_tick` records wall time only (no finish): per-stage
///            HOST/enqueue time; the production sync points are unchanged.
///   =1|tick— each tick inserts a guarded `queue.finish()` first: per-stage
///            GPU-inclusive time. Off by default — benchmarking only.
///   =evt   — mark + one marker EVENT per tick (queue created with
///            CL_QUEUE_PROFILING_ENABLE). Timestamps are drained lazily at
///            natural sync points (finish/read/report) — never waited on
///            inline — giving TRUE per-stage GPU time next to host time.
#[derive(Debug, Default)]
pub struct Prof {
    pub enabled: bool,
    pub finish: bool,
    pub evt: bool,
    tick: Cell<Option<Instant>>,
    stages: RefCell<BTreeMap<&'static str, (u64, f64)>>, // name → (count, host sec)
    dev: RefCell<BTreeMap<&'static str, (u64, f64)>>,    // name → (count, gpu sec)
    marks: RefCell<Vec<(&'static str, Event)>>,          // pending stage-end markers
    prev_end: Cell<u64>,                                 // last drained marker end (ns)
    prev_ok: Cell<bool>,
    pub n_finish: Cell<u64>,   // queue.finish() calls (sync count)
    pub n_read: Cell<u64>,     // blocking device→host reads
}

/// Shared OpenCL runtime. Holds context, queue, device capabilities, and
/// a program cache keyed by source string hash.
pub struct GpuRuntime {
    context: Context,
    queue: Queue,
    device: Device,
    caps: GpuCapabilities,
    program_cache: HashMap<u64, Program>,
    pub prof: Prof,
    /// W10/I3: device buffer-alloc counter (buffer_from_slice/zero_buffer/
    /// copy_buffer/copy_into-target). Solver loops must never grow this —
    /// tests assert alloc_count deltas are zero inside iterations.
    pub alloc_count: std::sync::atomic::AtomicU64,
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
        let pv = std::env::var("RUST_DFTB_PROF").unwrap_or_default();
        let ktime = std::env::var("RUST_DFTB_KTIME").map(|v| v != "0" && !v.is_empty()).unwrap_or(false);
        // evt mode needs CL_QUEUE_PROFILING_ENABLE at queue creation; the
        // KTIME mode needs it for per-kernel START/END events.
        let qprops = if pv == "evt" || ktime {
            Some(flags::CommandQueueProperties::new().profiling())
        } else {
            None
        };
        let queue = Queue::new(&context, device.clone(), qprops).map_err(map_ocl_err)?;

        let caps = query_capabilities(&device);

        Ok(Self {
            context,
            queue,
            device,
            caps,
            program_cache: HashMap::new(),
            alloc_count: std::sync::atomic::AtomicU64::new(0),
            prof: Prof {
                enabled: !pv.is_empty() && pv != "0",
                finish: pv != "mark" && pv != "evt",
                evt: pv == "evt",
                tick: Cell::new(Some(Instant::now())),
                ..Prof::default()
            },
        })
    }

    /// Sparse/physics GPU tests must run on NVIDIA. PoCL/CPU OpenCL hides
    /// fence, atomic, and OOB bugs (review G0.2). Override with
    /// `RUST_DFTB_ALLOW_CPU_CL=1` only for an explicit CPU-OpenCL experiment.
    pub fn require_nvidia(&self) -> Result<()> {
        require_nvidia_device(&self.caps)
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
        self.alloc_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        self.alloc_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        self.alloc_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dst = Buffer::<T>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(len)
            .build()
            .map_err(map_ocl_err)?;
        self.copy_into(src, &dst, len)?;
        Ok(dst)
    }

    /// Device-to-device copy into an existing buffer. No allocation.
    pub fn copy_into<T: ocl::OclPrm>(&self, src: &Buffer<T>, dst: &Buffer<T>, len: usize) -> Result<()> {
        if src.len() < len || dst.len() < len {
            return Err(DftbError::InvalidInput(format!(
                "copy_into: len={len} exceeds src.len()={} or dst.len()={}", src.len(), dst.len()
            )));
        }
        src.cmd()
            .queue(&self.queue)
            .copy(dst, None, Some(len))
            .enq()
            .map_err(map_ocl_err)?;
        Ok(())
    }

    /// Copy a GPU buffer back to host memory (blocking).
    pub fn read_buffer<T: ocl::OclPrm>(&self, buf: &Buffer<T>, out: &mut [T]) -> Result<()> {
        self.prof.n_read.set(self.prof.n_read.get() + 1);
        buf.read(out).enq().map_err(map_ocl_err)?;
        self.prof.n_finish.set(self.prof.n_finish.get() + 1);
        let r = self.queue.finish();
        self.prof_drain_events();
        r.map_err(map_ocl_err)
    }

    /// Write host data into an existing buffer (no alloc).
    pub fn write_buffer<T: ocl::OclPrm>(&self, buf: &Buffer<T>, data: &[T]) -> Result<()> {
        buf.write(data).enq().map_err(map_ocl_err)?;
        Ok(())
    }

    /// Finish all queued operations (blocking).
    pub fn finish(&self) -> Result<()> {
        self.prof.n_finish.set(self.prof.n_finish.get() + 1);
        let r = self.queue.finish();
        self.prof_drain_events();
        r.map_err(map_ocl_err)
    }

    /// R3 (evt mode): drain completed marker events into the per-stage GPU
    /// table. Called only at natural sync points (finish/read_buffer/
    /// prof_report) — never waits on an event; incomplete markers stay
    /// pending for the next sync.
    fn prof_drain_events(&self) {
        if !self.prof.evt { return; }
        let mut mk = self.prof.marks.borrow_mut();
        if mk.is_empty() { return; }
        let mut prev_end = if self.prof.prev_ok.get() { Some(self.prof.prev_end.get()) } else { None };
        let mut i = 0usize;
        for (name, ev) in mk.iter() {
            let end = match ev.profiling_info(ProfilingInfo::End) {
                Ok(ProfilingInfoResult::End(t)) => t,
                _ => break,   // marker not complete yet — in-order queue: all later ones aren't either
            };
            let start = match ev.profiling_info(ProfilingInfo::Start) {
                Ok(ProfilingInfoResult::Start(t)) => t,
                _ => end,
            };
            let base = prev_end.unwrap_or(start);
            let dt = end.saturating_sub(base) as f64 * 1e-9;
            if !name.is_empty() {
                let mut dv = self.prof.dev.borrow_mut();
                let e = dv.entry(*name).or_insert((0, 0.0));
                e.0 += 1; e.1 += dt;
            }
            prev_end = Some(end);
            i += 1;
        }
        if let Some(e) = prev_end { self.prof.prev_end.set(e); self.prof.prev_ok.set(true); }
        if i > 0 { mk.drain(..i); }
    }

    /// Restart the stage clock (no sync). Call once before a measured region.
    pub fn prof_reset(&self) {
        if !self.prof.enabled { return; }
        if self.prof.evt {
            // Baseline marker: the next stage's device time is measured
            // from here (empty name = anchor only, not accumulated).
            if let Ok(ev) = self.queue.enqueue_marker(None::<&Event>) {
                self.prof.marks.borrow_mut().push(("", ev));
            }
        }
        self.prof.tick.set(Some(Instant::now()));
    }

    /// Close a measured stage and accumulate the elapsed wall time under
    /// `name`. In `mark` mode: no sync — measures host/enqueue time only.
    /// In `tick` mode: guarded `queue.finish()` first — GPU-inclusive.
    /// In `evt` mode: host time + a trailing marker event for GPU time.
    /// No-op when `RUST_DFTB_PROF` is unset.
    pub fn prof_tick(&self, name: &'static str) {
        if !self.prof.enabled { return; }
        if self.prof.finish {
            self.prof.n_finish.set(self.prof.n_finish.get() + 1);
            let _ = self.queue.finish();
            self.prof_drain_events();
        }
        if self.prof.evt {
            if let Ok(ev) = self.queue.enqueue_marker(None::<&Event>) {
                self.prof.marks.borrow_mut().push((name, ev));
            }
        }
        let dt = self.prof.tick.get().map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
        let mut st = self.prof.stages.borrow_mut();
        let e = st.entry(name).or_insert((0, 0.0));
        e.0 += 1; e.1 += dt;
        drop(st);
        self.prof.tick.set(Some(Instant::now()));
    }

    /// Print the accumulated stage table + sync counts.
    /// `evt` mode prints ONE merged table: per-stage host time (enqueue +
    /// host work) next to TRUE device time (marker-event spans), each with
    /// its own % column — the two sums differ because host time includes
    /// queue-drain waits while device spans include idle gaps.
    /// `RUST_DFTB_PROF_OUT=<path>` additionally appends the same report to
    /// a file (A/B diffing without scraping stderr).
    pub fn prof_report(&self, title: &str) {
        if !self.prof.enabled { return; }
        if self.prof.evt {
            let _ = self.queue.finish();      // final drain point
            self.prof_drain_events();
        }
        let mut out = String::new();
        use std::fmt::Write;
        let st = self.prof.stages.borrow();
        let mut names: Vec<&'static str> = st.keys().copied().collect();
        let dev_b = self.prof.dev.borrow();
        for k in dev_b.keys() { if !st.contains_key(*k) { names.push(*k); } }
        let get = |m: &BTreeMap<&'static str, (u64, f64)>, k: &str| m.get(k).copied().unwrap_or((0, 0.0));
        let host_total: f64 = st.values().map(|v| v.1).sum();
        let dev_total: f64 = dev_b.values().map(|v| v.1).sum();
        names.sort_by(|a, b| {
            let ka = get(&dev_b, a).1.max(get(&st, a).1);
            let kb = get(&dev_b, b).1.max(get(&st, b).1);
            kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
        });
        let _ = writeln!(out, "[prof] {title}: host={:.1} ms  dev={:.1} ms  finishes={} reads={}",
            host_total * 1e3, dev_total * 1e3, self.prof.n_finish.get(), self.prof.n_read.get());
        if self.prof.evt {
            let _ = writeln!(out, "[prof]   {:<26} {:>9} {:>9} {:>9} {:>7} {:>9} {:>6}", "stage", "host ms", "dev ms", "n", "h/call", "d/call", "dev%");
            for name in names {
                let (h_n, h_s) = get(&st, name);
                let (d_n, d_s) = get(&dev_b, name);
                let n = h_n.max(d_n);
                let _ = writeln!(out, "[prof]   {name:<26} {:9.3} {:9.3} {:>9} {:7.4} {:9.4} {:5.1}%",
                    h_s * 1e3, d_s * 1e3, n,
                    if n > 0 { h_s * 1e3 / n as f64 } else { 0.0 },
                    if d_n > 0 { d_s * 1e3 / d_n as f64 } else { 0.0 },
                    if dev_total > 0.0 { 100.0 * d_s / dev_total } else { 0.0 });
            }
        } else {
            let _ = writeln!(out, "[prof]   {:<26} {:>9} {:>9} {:>7} {:>6}", "stage", "host ms", "n", "ms/call", "host%");
            for name in names {
                let (h_n, h_s) = get(&st, name);
                let _ = writeln!(out, "[prof]   {name:<26} {:9.3} {:>9} {:7.4} {:5.1}%",
                    h_s * 1e3, h_n,
                    if h_n > 0 { h_s * 1e3 / h_n as f64 } else { 0.0 },
                    if host_total > 0.0 { 100.0 * h_s / host_total } else { 0.0 });
            }
        }
        eprint!("{out}");
        if let Ok(path) = std::env::var("RUST_DFTB_PROF_OUT") {
            if !path.is_empty() {
                if let Err(e) = std::fs::OpenOptions::new().create(true).append(true).open(&path)
                    .and_then(|mut f| std::io::Write::write_all(&mut f, out.as_bytes()))
                {
                    eprintln!("[prof] RUST_DFTB_PROF_OUT={path}: write failed: {e}");
                }
            }
        }
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

/// Abort unless `caps.name` looks like NVIDIA, or `RUST_DFTB_ALLOW_CPU_CL=1`.
pub fn require_nvidia_device(caps: &GpuCapabilities) -> Result<()> {
    let allow_cpu = std::env::var("RUST_DFTB_ALLOW_CPU_CL").ok().as_deref() == Some("1");
    if allow_cpu {
        eprintln!("[gpu] RUST_DFTB_ALLOW_CPU_CL=1 — accepting non-NVIDIA device '{}'", caps.name);
        return Ok(());
    }
    if !caps.name.to_uppercase().contains("NVIDIA") {
        return Err(DftbError::InvalidInput(format!(
            "NVIDIA GPU required, got '{}'. PoCL/CPU OpenCL hides fence/atomic/OOB bugs. \
             Set RUST_DFTB_ALLOW_CPU_CL=1 only for an explicit CPU-OpenCL experiment.",
            caps.name
        )));
    }
    Ok(())
}
