//! Fail-loud GPU / SK helpers for sparse tests (review Gate 0).
//!
//! Skip is allowed only when the ICD exposes **zero** OpenCL platforms.
//! A present device that errors, a PoCL CPU device, or missing SK files
//! are test failures — never `ok`.

use crate::core::error::Result;
use crate::methods::sparse::gpu_sparse::{SparseBsr4Config, SparseBsr4Gpu};
use crate::qmqm::gpu_runtime::{require_nvidia_device, GpuRuntime};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// Default SK set for Si/H sparse physics gates.
pub const DEFAULT_SIH_SK_DIR: &str = "/home/prokop/SIMULATIONS/dftbplus/slakos/matsci-0-3";

static GPU_BANNER: AtomicBool = AtomicBool::new(false);

fn is_no_platform_panic(msg: &str) -> bool {
    msg.contains("GetPlatformIdsPlatformListUnavailable")
}

fn panic_msg(payload: &(dyn std::any::Any + Send)) -> String {
    payload.downcast_ref::<String>().cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_else(|| "non-string panic".to_string())
}

fn print_gpu_banner(rt: &GpuRuntime) {
    if GPU_BANNER.swap(true, Ordering::Relaxed) {
        return;
    }
    let c = rt.caps();
    eprintln!(
        "[sparse GPU] device='{}'  local_mem={} B  max_wg={}  CUs={}  global={} B",
        c.name, c.local_mem_size, c.max_work_group_size, c.compute_units, c.global_mem_size
    );
}

/// Resolve `RUST_DFTB_SK_DIR` or the matsci-0-3 default. Missing dir/files panic.
pub fn require_sih_sk_dir() -> String {
    let dir = std::env::var("RUST_DFTB_SK_DIR").unwrap_or_else(|_| DEFAULT_SIH_SK_DIR.to_string());
    let p = Path::new(&dir);
    assert!(p.is_dir(),
        "SK directory missing: {dir}. Set RUST_DFTB_SK_DIR or install matsci-0-3 at {DEFAULT_SIH_SK_DIR}");
    let si_si = p.join("Si-Si.skf");
    let h_h = p.join("H-H.skf");
    let si_h = p.join("Si-H.skf");
    let h_si = p.join("H-Si.skf");
    assert!(si_si.is_file(), "Si-Si.skf missing in {dir}");
    assert!(h_h.is_file(), "H-H.skf missing in {dir}");
    assert!(si_h.is_file() || h_si.is_file(), "Si-H.skf / H-Si.skf missing in {dir}");
    eprintln!("[sparse SK] {dir}");
    dir
}

/// OpenCL runtime for sparse GPU tests. `None` only if there is no platform.
pub fn require_nvidia_runtime() -> Option<GpuRuntime> {
    match catch_unwind(AssertUnwindSafe(GpuRuntime::new)) {
        Ok(Ok(rt)) => {
            require_nvidia_device(rt.caps()).unwrap_or_else(|e| panic!("sparse GPU required (no skip): {e}"));
            print_gpu_banner(&rt);
            Some(rt)
        }
        Ok(Err(e)) => panic!("sparse GPU required (no skip on OpenCL Err): {e}"),
        Err(payload) => {
            let msg = panic_msg(&*payload);
            if is_no_platform_panic(&msg) {
                eprintln!("Skipping sparse GPU test: no OpenCL platform ({msg})");
                None
            } else {
                resume_unwind(payload)
            }
        }
    }
}

/// BSR4 GPU for sparse tests. `None` only if there is no OpenCL platform.
pub fn require_sparse_gpu() -> Option<SparseBsr4Gpu> {
    match catch_unwind(AssertUnwindSafe(|| SparseBsr4Gpu::new(SparseBsr4Config::default()))) {
        Ok(Ok(gpu)) => Some(gpu),
        Ok(Err(e)) => panic!("sparse GPU required (no skip on OpenCL Err): {e}"),
        Err(payload) => {
            let msg = panic_msg(&*payload);
            if is_no_platform_panic(&msg) {
                eprintln!("Skipping sparse GPU test: no OpenCL platform ({msg})");
                None
            } else {
                resume_unwind(payload)
            }
        }
    }
}

/// NVIDIA check used by `SparseBsr4Gpu::new` (production sparse path too).
pub fn check_sparse_device(rt: &GpuRuntime) -> Result<()> {
    require_nvidia_device(rt.caps())?;
    print_gpu_banner(rt);
    Ok(())
}
