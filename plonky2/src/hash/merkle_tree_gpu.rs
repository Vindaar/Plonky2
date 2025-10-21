#![cfg(all(feature = "gpu_merkle", target_arch = "wasm32"))]

//! WebGPU scaffolding for Merkle tree construction.
//!
//! The intent is to offload tree layer hashing to the GPU by dispatching the
//! compute kernel defined in `shaders/merkle_tree.wgsl`. Once the shader is
//! plugged in, `try_build_merkle_tree` can be extended to populate the digests
//! and Merkle cap buffers that the CPU fallback currently fills.

use std::sync::OnceLock;

use anyhow::{anyhow, Result};
use web_sys::console;
use wgpu::{ComputePipeline, Device, Queue};

use crate::hash::hash_types::RichField;
use crate::plonk::config::Hasher;

/// Static WebGPU context reused across Merkle tree constructions.
static GPU_CONTEXT: OnceLock<MerkleTreeGpuContext> = OnceLock::new();

/// Cached pipeline and device handles required to launch the Merkle compute kernel.
pub struct MerkleTreeGpuContext {
    pub device: Device,
    pub queue: Queue,
    pub pipeline: Option<ComputePipeline>,
}

impl MerkleTreeGpuContext {
    fn new(device: Device, queue: Queue, pipeline: Option<ComputePipeline>) -> Self {
        Self {
            device,
            queue,
            pipeline,
        }
    }
}

/// Initialize the global WebGPU context.
///
/// This should be invoked from WASM bindings before executing any proving flow
/// that relies on GPU acceleration. Subsequent calls are cheap no-ops.
pub async fn initialize() -> Result<()> {
    if GPU_CONTEXT.get().is_some() {
        return Ok(());
    }

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| anyhow!("failed to obtain WebGPU adapter"))?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Merkle Tree Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .map_err(|err| anyhow!("failed to create WebGPU device: {err}"))?;

    // Shader will be provided later; keep pipeline optional until then.
    let pipeline = load_merkle_pipeline(&device);

    GPU_CONTEXT
        .set(MerkleTreeGpuContext::new(device, queue, pipeline))
        .map_err(|_| anyhow!("GPU context already initialized"))?;

    console::info_1(&"Merkle GPU context initialized".into());
    Ok(())
}

const MERKLE_ENTRY_POINT: &str = "processMerkleTreeLayerWithCap";

fn load_merkle_pipeline(device: &Device) -> Option<ComputePipeline> {
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Merkle Tree Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "../../shaders/merkle_tree.wgsl"
        ))),
    });

    Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Merkle Tree Pipeline"),
        layout: None,
        module: &shader_module,
        entry_point: MERKLE_ENTRY_POINT,
        compilation_options: Default::default(),
        cache: None,
    }))
}

/// Returns `true` when the WebGPU context is ready for use.
pub fn is_initialized() -> bool {
    GPU_CONTEXT.get().is_some()
}

/// Attempt to build the Merkle tree using the GPU path.
///
/// While the shader is a stub, this simply acknowledges the initialized
/// context and falls back to the CPU implementation, returning `None`.
pub fn try_build_merkle_tree<F, H>(
    leaves: &[Vec<F>],
    cap_height: usize,
) -> Option<Result<GpuMerkleOutput<H::Hash>>>
where
    F: RichField,
    H: Hasher<F>,
{
    let _ = leaves;
    let _ = cap_height;

    let context = GPU_CONTEXT.get()?;

    if context.pipeline.is_none() {
        console::warn_1(
            &"Merkle GPU shader not loaded yet; using CPU fallback".into(),
        );
        return Some(Err(anyhow!("Merkle GPU shader stub is active")));
    }

    // Placeholder implementation. Once the WGSL kernel is available, populate
    // output buffers and construct digests and caps accordingly.
    Some(Err(anyhow!(
        "Merkle GPU execution path not implemented yet"
    )))
}

/// Placeholder output struct to be filled once the WGSL kernel writes digests.
pub struct GpuMerkleOutput<D> {
    pub digests: Vec<D>,
    pub cap: Vec<D>,
}
