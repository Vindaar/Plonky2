//! WebGPU scaffolding for Merkle tree construction.
//!
//! This module mirrors the CPU Merkle construction in `merkle_tree.rs` but offloads the heavy
//! Poseidon hashing to a WGSL compute shader. The shader expects Goldilocks field elements in
//! Montgomery form (`a * R mod p` with `R = 2^64`). We therefore convert all hash inputs and
//! constants to Montgomery before uploading them to the GPU and convert results back to the
//! canonical representation after read-back.

#![cfg(all(feature = "gpu_merkle", target_arch = "wasm32"))]

use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::rc::Rc;

use anyhow::{anyhow, ensure, Result};
use bytemuck::{Pod, Zeroable};
use futures::channel::oneshot;
use futures::future::{poll_fn, FutureExt};
use once_cell::unsync::OnceCell;
use web_sys::console;
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, Buffer, ComputePipeline, Device, Queue};

use crate::hash::hash_types::{HashOut, RichField, NUM_HASH_OUT_ELTS};
use crate::hash::poseidon::{self, Poseidon, SPONGE_WIDTH};
// Goldilocks modulus and Montgomery parameters.
const GOLDILOCKS_MODULUS: u64 = 0xFFFF_FFFF_0000_0001;
const MONTGOMERY_R: u128 = 1u128 << 64;
const MONTGOMERY_R_INV: u64 = 0xFFFF_FFFE_0000_0001;

const BIGINT_LIMBS: usize = 2;
const DIGEST_ELEMENTS: usize = NUM_HASH_OUT_ELTS;
const WORDS_PER_DIGEST: usize = DIGEST_ELEMENTS * BIGINT_LIMBS;
const BYTES_PER_DIGEST: usize = WORDS_PER_DIGEST * std::mem::size_of::<u32>();
const POSEIDON_WIDTH: usize = SPONGE_WIDTH;
const ROUND_CONSTANT_COUNT: usize = POSEIDON_WIDTH * poseidon::N_ROUNDS;
const WORKGROUP_SIZE: u32 = 64;

thread_local! {
    /// WebGPU context reused across Merkle tree constructions.
    static GPU_CONTEXT: OnceCell<Rc<MerkleTreeGpuContext>> = OnceCell::new();
}

/// Cached pipeline and device handles required to launch the Merkle compute kernel.
#[derive(Debug)]
pub struct MerkleTreeGpuContext {
    pub device: Rc<Device>,
    pub queue: Rc<Queue>,
    pub merkle_pipeline: Rc<ComputePipeline>,
    pub merkle_bind_group_layout: Rc<BindGroupLayout>,
    pub leaf_pipeline: Rc<ComputePipeline>,
    pub leaf_bind_group_layout: Rc<BindGroupLayout>,
    pub mds_circ: Buffer,
    pub mds_diag: Buffer,
    pub round_constants: Buffer,
}

impl MerkleTreeGpuContext {
    fn new(
        device: Device,
        queue: Queue,
        merkle_pipeline: ComputePipeline,
        merkle_bind_group_layout: BindGroupLayout,
        leaf_pipeline: ComputePipeline,
        leaf_bind_group_layout: BindGroupLayout,
        mds_circ: Buffer,
        mds_diag: Buffer,
        round_constants: Buffer,
    ) -> Self {
        Self {
            device: Rc::new(device),
            queue: Rc::new(queue),
            merkle_pipeline: Rc::new(merkle_pipeline),
            merkle_bind_group_layout: Rc::new(merkle_bind_group_layout),
            leaf_pipeline: Rc::new(leaf_pipeline),
            leaf_bind_group_layout: Rc::new(leaf_bind_group_layout),
            mds_circ,
            mds_diag,
            round_constants,
        }
    }
}

/// Parameters handed to the Merkle tree kernel for a single layer dispatch.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct MerkleTreeKernelArgs {
    cap_len: u32,
    layer: u32,
    src_layer_size: u32,
    dst_layer_size: u32,
    src_offset: u32,
    dst_offset: u32,
    write_to_cap: u32,
}

/// Montgomery helper: convert canonical Goldilocks element (as `u64`) into Montgomery form.
fn to_montgomery_u64(value: u64) -> u64 {
    let res = ((value as u128) * MONTGOMERY_R) % (GOLDILOCKS_MODULUS as u128);
    res as u64
}

/// Montgomery helper: convert Montgomery encoded Goldilocks element back to canonical form.
fn from_montgomery_u64(value: u64) -> u64 {
    let res = ((value as u128) * (MONTGOMERY_R_INV as u128)) % (GOLDILOCKS_MODULUS as u128);
    res as u64
}

fn field_to_montgomery_words<F: RichField>(value: &F) -> [u32; BIGINT_LIMBS] {
    let canonical = value.to_canonical_u64();
    let monty = to_montgomery_u64(canonical);
    [(monty & 0xFFFF_FFFF) as u32, (monty >> 32) as u32]
}

fn montgomery_words_to_field<F: RichField>(words: &[u32; BIGINT_LIMBS]) -> F {
    let monty = (words[1] as u64) << 32 | (words[0] as u64);
    let canonical = from_montgomery_u64(monty);
    F::from_canonical_u64(canonical)
}

fn montgomery_words_to_hash<F: RichField>(words: &[u32]) -> Result<HashOut<F>> {
    ensure!(
        words.len() == WORDS_PER_DIGEST,
        "expected {} words per digest, got {}",
        WORDS_PER_DIGEST,
        words.len()
    );
    let mut elements = [F::ZERO; NUM_HASH_OUT_ELTS];
    for (i, chunk) in words.chunks(BIGINT_LIMBS).enumerate() {
        let chunk: [u32; BIGINT_LIMBS] = chunk.try_into().unwrap();
        elements[i] = montgomery_words_to_field(&chunk);
    }
    Ok(HashOut { elements })
}

fn fields_to_montgomery_words<F: RichField>(values: &[F]) -> Vec<u32> {
    let mut out = Vec::with_capacity(values.len() * BIGINT_LIMBS);
    for value in values {
        out.extend_from_slice(&field_to_montgomery_words(value));
    }
    out
}

fn log(msg: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::console::log_1(&msg.into());
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("{msg}");
    }
}

// ============================================================================
// PROFILING HELPERS
// ============================================================================

/// Get high-resolution timestamp in milliseconds
fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window().unwrap().performance().unwrap().now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        // For native, use a simple placeholder
        0.0
    }
}

/// Log timing information to console
fn log_timing(label: &str, duration_ms: f64) {
    #[cfg(target_arch = "wasm32")]
    {
        console::log_1(&format!("⏱️  {}: {:.2}ms", label, duration_ms).into());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("⏱️  {}: {:.2}ms", label, duration_ms);
    }
}

/// Initialize the global WebGPU context. Subsequent calls are no-ops.
pub async fn initialize() -> Result<()> {
    if GPU_CONTEXT.with(|cell| cell.get().is_some()) {
        return Ok(());
    }

    log("Initializing GPU context");

    let inst_desc = wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    };
    let instance = wgpu::Instance::new(&inst_desc);

    log("Requesting adapter");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("failed to obtain WebGPU adapter");
    //.ok_or_else(|| anyhow!("failed to obtain WebGPU adapter"))?;

    log("Requesting device");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Merkle Tree Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            },
            //None,
        )
        .await
        .map_err(|err| anyhow!("failed to create WebGPU device: {err}"))?;

    log("Creating bind group layouts");

    let merkle_bind_group_layout = create_merkle_bind_group_layout(&device);
    let merkle_pipeline = create_merkle_pipeline(&device, &merkle_bind_group_layout)?;
    let leaf_bind_group_layout = create_leaf_hash_bind_group_layout(&device);
    let leaf_pipeline = create_leaf_hash_pipeline(&device, &leaf_bind_group_layout)?;
    let (mds_circ, mds_diag, round_constants) = create_poseidon_constant_buffers::<
        crate::field::goldilocks_field::GoldilocksField,
    >(&device);

    log("Creating Merkle GPU context");

    let context = Rc::new(MerkleTreeGpuContext::new(
        device,
        queue,
        merkle_pipeline,
        merkle_bind_group_layout,
        leaf_pipeline,
        leaf_bind_group_layout,
        mds_circ,
        mds_diag,
        round_constants,
    ));

    log("Setting RC cell content");

    GPU_CONTEXT.with(|cell| {
        cell.set(context)
            .map_err(|_| anyhow!("GPU context already initialized"))
    })?;

    console::info_1(&"Merkle GPU context initialized".into());
    Ok(())
}

fn create_merkle_pipeline(
    device: &Device,
    bind_group_layout: &BindGroupLayout,
) -> Result<ComputePipeline> {
    let shader_module =
        device.create_shader_module(wgpu::include_wgsl!("../../shaders/merkle_tree.wgsl"));

    log("creating Merkle tree compute pipeline");
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Merkle Tree Pipeline Layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    Ok(
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Merkle Tree Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("processMerkleTreeLayerWithCap"),
            compilation_options: Default::default(),
            cache: None,
        }),
    )
}

fn create_leaf_hash_pipeline(
    device: &Device,
    bind_group_layout: &BindGroupLayout,
) -> Result<ComputePipeline> {
    let shader_module =
        device.create_shader_module(wgpu::include_wgsl!("../../shaders/poseidon1_hash.wgsl"));

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Poseidon Leaf Hash Pipeline Layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    Ok(
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Poseidon Leaf Hash Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("poseidon1Hash"),
            compilation_options: Default::default(),
            cache: None,
        }),
    )
}

/// Returns `true` when the WebGPU context is ready for use.
pub fn is_initialized() -> bool {
    GPU_CONTEXT.with(|cell| cell.get().is_some())
}

fn create_merkle_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Merkle Tree Bind Group Layout"),
        entries: &[
            // 0: input hashed leaves
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 1: nodes buffer
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 2: cap buffer
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 3: per-layer uniforms
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(
                        std::mem::size_of::<MerkleTreeKernelArgs>() as u64
                    ),
                },
                count: None,
            },
            // 4: Poseidon MDS circulant
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(
                        (POSEIDON_WIDTH * BIGINT_LIMBS * std::mem::size_of::<u32>()) as u64,
                    ),
                },
                count: None,
            },
            // 5: Poseidon MDS diagonal
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(
                        (POSEIDON_WIDTH * BIGINT_LIMBS * std::mem::size_of::<u32>()) as u64,
                    ),
                },
                count: None,
            },
            // 6: round constants
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(
                        (ROUND_CONSTANT_COUNT * BIGINT_LIMBS * std::mem::size_of::<u32>()) as u64,
                    ),
                },
                count: None,
            },
        ],
    })
}

fn create_leaf_hash_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Poseidon Leaf Hash Bind Group Layout"),
        entries: &[
            // 0: output digests
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(BYTES_PER_DIGEST as u64),
                },
                count: None,
            },
            // 1: transposed input elements
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 2: number of leaves
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<i32>() as u64),
                },
                count: None,
            },
            // 3: elements per leaf
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<i32>() as u64),
                },
                count: None,
            },
            // 4: Poseidon MDS circulant
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(
                        (POSEIDON_WIDTH * BIGINT_LIMBS * std::mem::size_of::<u32>()) as u64,
                    ),
                },
                count: None,
            },
            // 5: Poseidon MDS diagonal
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(
                        (POSEIDON_WIDTH * BIGINT_LIMBS * std::mem::size_of::<u32>()) as u64,
                    ),
                },
                count: None,
            },
            // 6: round constants
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(
                        (ROUND_CONSTANT_COUNT * BIGINT_LIMBS * std::mem::size_of::<u32>()) as u64,
                    ),
                },
                count: None,
            },
        ],
    })
}

fn create_poseidon_constant_buffers<F: RichField + Poseidon>(
    device: &Device,
) -> (Buffer, Buffer, Buffer) {
    let mut mds_circ_words = Vec::with_capacity(POSEIDON_WIDTH * BIGINT_LIMBS);
    for &value in F::MDS_MATRIX_CIRC.iter().take(POSEIDON_WIDTH) {
        let field = F::from_canonical_u64(value);
        mds_circ_words.extend_from_slice(&field_to_montgomery_words(&field));
    }

    let mut mds_diag_words = Vec::with_capacity(POSEIDON_WIDTH * BIGINT_LIMBS);
    for &value in F::MDS_MATRIX_DIAG.iter().take(POSEIDON_WIDTH) {
        let field = F::from_canonical_u64(value);
        mds_diag_words.extend_from_slice(&field_to_montgomery_words(&field));
    }

    let mut rc_words = Vec::with_capacity(ROUND_CONSTANT_COUNT * BIGINT_LIMBS);
    for &value in poseidon::ALL_ROUND_CONSTANTS
        .iter()
        .take(ROUND_CONSTANT_COUNT)
    {
        let field = F::from_canonical_u64(value);
        rc_words.extend_from_slice(&field_to_montgomery_words(&field));
    }

    let mds_circ = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Poseidon MDS (circ)"),
        contents: bytemuck::cast_slice(&mds_circ_words),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let mds_diag = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Poseidon MDS (diag)"),
        contents: bytemuck::cast_slice(&mds_diag_words),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let round_constants = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Poseidon round constants"),
        contents: bytemuck::cast_slice(&rc_words),
        usage: wgpu::BufferUsages::STORAGE,
    });

    (mds_circ, mds_diag, round_constants)
}

fn host_layer_size(input_size: usize, layer: usize) -> usize {
    if layer == 0 {
        return input_size;
    }
    let mut size = input_size / 2;
    for _ in 1..layer {
        size = (size + 1) / 2;
    }
    size
}

fn host_layer_offset(input_size: usize, layer: usize) -> usize {
    let mut offset = 0;
    let mut size = input_size / 2;
    for _ in 1..layer {
        offset += size;
        size = (size + 1) / 2;
    }
    offset
}

#[derive(Debug)]
struct MerkleBuffers {
    input: Buffer,
    nodes: Buffer,
    cap: Buffer,
}

#[derive(Debug)]
enum MerkleGpuJobState<F: RichField> {
    Immediate(GpuMerkleOutput<F>),
    Deferred {
        context: Rc<MerkleTreeGpuContext>,
        buffers: MerkleBuffers,
        total_nodes: usize,
        cap_len: usize,
        num_leaves: usize,
        cap_height: usize,
        num_layers_to_cap: usize,
    },
}

#[derive(Debug)]
pub struct MerkleGpuJob<F: RichField> {
    state: MerkleGpuJobState<F>,
    _marker: PhantomData<F>,
}

impl<F> MerkleGpuJob<F>
where
    F: RichField + Poseidon + 'static,
{
    fn immediate(output: GpuMerkleOutput<F>) -> Self {
        Self {
            state: MerkleGpuJobState::Immediate(output),
            _marker: PhantomData,
        }
    }

    fn deferred(
        context: Rc<MerkleTreeGpuContext>,
        buffers: MerkleBuffers,
        total_nodes: usize,
        cap_len: usize,
        num_leaves: usize,
        cap_height: usize,
        num_layers_to_cap: usize,
    ) -> Self {
        Self {
            state: MerkleGpuJobState::Deferred {
                context,
                buffers,
                total_nodes,
                cap_len,
                num_leaves,
                cap_height,
                num_layers_to_cap,
            },
            _marker: PhantomData,
        }
    }

    async fn finish(self) -> Result<GpuMerkleOutput<F>> {
        match self.state {
            MerkleGpuJobState::Immediate(output) => Ok(output),
            MerkleGpuJobState::Deferred {
                context,
                buffers,
                total_nodes,
                cap_len,
                num_leaves,
                cap_height,
                num_layers_to_cap,
            } => {
                console::log_1(&"=== PHASE 4: GPU Completion & Readback ===".into());
                let readback_start = now_ms();

                // Wait for GPU to finish
                let wait_start = now_ms();
                wait_for_queue(context.queue.clone()).await?;
                log_timing("⚡ GPU execution (wait_for_queue)", now_ms() - wait_start);

                // Read leaf hashes
                let leaf_read_start = now_ms();
                let leaf_words = read_u32_buffer_async(
                    context.device.clone(),
                    context.queue.clone(),
                    &buffers.input,
                    num_leaves * WORDS_PER_DIGEST,
                )
                .await?;
                log_timing("Leaf hash readback", now_ms() - leaf_read_start);

                let leaf_convert_start = now_ms();
                let leaf_hashes: Vec<HashOut<F>> = leaf_words
                    .chunks(WORDS_PER_DIGEST)
                    .map(montgomery_words_to_hash::<F>)
                    .collect::<Result<Vec<_>>>()?;
                log_timing("Leaf Montgomery conversion", now_ms() - leaf_convert_start);

                // Read node hashes
                let node_read_start = now_ms();
                let node_hashes: Vec<HashOut<F>> = if total_nodes > 0 {
                    let node_words = read_u32_buffer_async(
                        context.device.clone(),
                        context.queue.clone(),
                        &buffers.nodes,
                        total_nodes * WORDS_PER_DIGEST,
                    )
                    .await?;
                    log_timing("Node hash readback", now_ms() - node_read_start);

                    let convert_start = now_ms();
                    let result = node_words
                        .chunks(WORDS_PER_DIGEST)
                        .map(montgomery_words_to_hash::<F>)
                        .collect::<Result<Vec<_>>>()?;
                    log_timing("Node Montgomery conversion", now_ms() - convert_start);
                    result
                } else {
                    Vec::new()
                };

                // Read cap
                let cap_read_start = now_ms();
                let cap_words = read_u32_buffer_async(
                    context.device.clone(),
                    context.queue.clone(),
                    &buffers.cap,
                    cap_len * WORDS_PER_DIGEST,
                )
                .await?;
                log_timing("Cap readback", now_ms() - cap_read_start);

                let cap_convert_start = now_ms();
                let cap_hashes: Vec<HashOut<F>> = cap_words
                    .chunks(WORDS_PER_DIGEST)
                    .map(montgomery_words_to_hash::<F>)
                    .collect::<Result<Vec<_>>>()?;
                log_timing("Cap Montgomery conversion", now_ms() - cap_convert_start);

                // CPU post-processing: reconstruct digest tree
                console::log_1(&"=== PHASE 5: CPU Post-processing ===".into());
                let postprocess_start = now_ms();

                let num_digests = 2 * (num_leaves - (1 << cap_height));
                let mut digests = if num_digests == 0 {
                    Vec::new()
                } else {
                    vec![HashOut::<F>::ZERO; num_digests]
                };

                if num_digests > 0 {
                    let accessor = LayerAccessor::new(
                        &leaf_hashes,
                        &node_hashes,
                        &cap_hashes,
                        num_leaves,
                        num_layers_to_cap,
                    );
                    let subtree_digests_len = num_digests >> cap_height;
                    let subtree_leaves_len = num_leaves >> cap_height;

                    log("Subtree business");
                    for (subtree_idx, subtree_buf) in
                        digests.chunks_mut(subtree_digests_len).enumerate()
                    {
                        let leaf_offset = subtree_idx * subtree_leaves_len;
                        let root_digest = fill_subtree_from_gpu(
                            subtree_buf,
                            &accessor,
                            leaf_offset,
                            subtree_leaves_len,
                        );
                        debug_assert_eq!(root_digest, cap_hashes[subtree_idx]);
                    }
                }

                log_timing("Digest tree reconstruction", now_ms() - postprocess_start);
                log_timing(
                    "📥 TOTAL READBACK + POST-PROCESSING",
                    now_ms() - readback_start,
                );

                Ok(GpuMerkleOutput {
                    digests,
                    cap: cap_hashes,
                })
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn await_async(self) -> Result<GpuMerkleOutput<F>> {
        let start = now_ms();
        let result = self.finish().await;
        log_timing("🏁 TOTAL await_async TIME", now_ms() - start);
        result
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(self) -> Result<GpuMerkleOutput<F>> {
        pollster::block_on(self.finish())
    }
}

fn create_buffers(
    ctx: &MerkleTreeGpuContext,
    input: Buffer,
    total_internal_nodes: usize,
    cap_len: usize,
) -> MerkleBuffers {
    let nodes = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("merkle-nodes"),
        size: (total_internal_nodes * BYTES_PER_DIGEST) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let cap = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("merkle-cap"),
        size: (cap_len * BYTES_PER_DIGEST) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    MerkleBuffers { input, nodes, cap }
}

async fn wait_for_queue(queue: Rc<Queue>) -> Result<()> {
    let (sender, receiver) = oneshot::channel();
    queue.on_submitted_work_done(move || {
        let _ = sender.send(());
    });
    receiver
        .await
        .map_err(|_| anyhow!("queue completion receiver dropped"))?;
    Ok(())
}

async fn read_u32_buffer_async(
    device: Rc<Device>,
    queue: Rc<Queue>,
    buffer: &Buffer,
    word_len: usize,
) -> Result<Vec<u32>> {
    log("read u32 buffer start");
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("merkle-readback"),
        size: (word_len * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("copy merkle buffer"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, staging.size());
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (map_sender, map_receiver) = oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = map_sender.send(res);
    });

    // Drive the device to make progress while we wait for the map to complete.
    let mut map_receiver = map_receiver.fuse();
    poll_fn(move |cx| {
        device.poll(wgpu::PollType::Poll);
        map_receiver.poll_unpin(cx)
    })
    .await
    .map_err(|_| anyhow!("failed to receive map result"))?
    .map_err(|err| anyhow!("failed to map buffer: {err}"))?;
    log::info!("Result received.");

    let data = slice.get_mapped_range();
    let words = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    log("...done read u32 buffer start");
    Ok(words)
}

fn transpose_leaves<F: RichField>(leaves: &[Vec<F>]) -> (Vec<F>, usize) {
    let num_leaves = leaves.len();
    let elements_per_leaf = leaves[0].len();

    // Now transpose knowing all are elements_per_leaf long
    let mut transposed = Vec::with_capacity(num_leaves * elements_per_leaf);
    for elem_idx in 0..elements_per_leaf {
        for leaf in leaves {
            transposed.push(leaf[elem_idx]);
        }
    }

    (transposed, elements_per_leaf)
}

fn hash_leaves_gpu<F>(ctx: &MerkleTreeGpuContext, leaves: &[Vec<F>]) -> Result<Buffer>
where
    F: RichField + Poseidon,
{
    let num_leaves = leaves.len();
    ensure!(num_leaves > 0, "Merkle tree requires at least one leaf");
    ensure!(
        num_leaves <= i32::MAX as usize,
        "Merkle GPU hashing expects leaf count to fit in i32, got {num_leaves}"
    );

    let elements_per_leaf = leaves[0].len();
    ensure!(
        leaves.iter().all(|leaf| leaf.len() == elements_per_leaf),
        "GPU Poseidon hashing requires leaves of uniform length"
    );

    // Time data conversion
    let convert_start = now_ms();
    let (transposed, elements_per_leaf) = transpose_leaves(leaves);
    ensure!(
        elements_per_leaf > 0,
        "GPU Poseidon hashing received empty leaves"
    );
    ensure!(
        elements_per_leaf <= i32::MAX as usize,
        "elements_per_leaf must fit in i32, got {elements_per_leaf}"
    );
    log_timing("  Leaf data transpose", now_ms() - convert_start);

    let transposed_words = fields_to_montgomery_words(&transposed);
    log_timing(
        "  Leaf data transpose + Montgomery conversion",
        now_ms() - convert_start,
    );

    // Time buffer creation
    let buffer_start = now_ms();
    let input_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("poseidon-leaf-input"),
            contents: bytemuck::cast_slice(&transposed_words),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("poseidon-leaf-output"),
        size: (num_leaves * BYTES_PER_DIGEST) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let num_leaves_i32 = num_leaves as i32;
    let num_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("poseidon-leaf-count"),
            contents: bytemuck::bytes_of(&num_leaves_i32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

    let elements_per_leaf_i32 = elements_per_leaf as i32;
    let elements_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("poseidon-leaf-width"),
            contents: bytemuck::bytes_of(&elements_per_leaf_i32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("poseidon-leaf-bind-group"),
        layout: &ctx.leaf_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: num_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: elements_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: ctx.mds_circ.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: ctx.mds_diag.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: ctx.round_constants.as_entire_binding(),
            },
        ],
    });
    log_timing(
        "  Leaf buffer creation + bind group",
        now_ms() - buffer_start,
    );

    // Time dispatch
    let dispatch_start = now_ms();
    let workgroup_size = WORKGROUP_SIZE;
    debug_assert!(workgroup_size <= 256);
    let workgroups_x = ((num_leaves as u32) + workgroup_size - 1) / workgroup_size;

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("poseidon-leaf-encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("poseidon-leaf-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.leaf_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
    ctx.queue.on_submitted_work_done(|| {});
    log_timing("  Leaf dispatch", now_ms() - dispatch_start);

    Ok(output_buffer)
}

/// Run the GPU Merkle tree pipeline, returning a job that resolves once GPU buffers are ready.
pub fn build_merkle_tree<F>(
    ctx: Rc<MerkleTreeGpuContext>,
    leaves: &[Vec<F>],
    cap_height: usize,
) -> Result<MerkleGpuJob<F>>
where
    F: RichField + Poseidon,
{
    let total_start = now_ms();

    let ctx_ref = ctx.as_ref();
    let num_leaves = leaves.len();
    ensure!(num_leaves > 0, "Merkle tree requires at least one leaf");
    ensure!(
        num_leaves.is_power_of_two(),
        "GPU Merkle currently expects a power-of-two leaf count"
    );

    let depth = num_leaves.trailing_zeros() as usize;
    ensure!(
        cap_height <= depth,
        "cap_height {cap_height} exceeds tree depth {depth}"
    );

    // PHASE 1: Leaf hashing setup and dispatch
    console::log_1(&"=== PHASE 1: Leaf Hashing ===".into());
    let leaf_start = now_ms();
    log("launching GPU Poseidon hashing");
    let leaf_buffer = hash_leaves_gpu(ctx_ref, leaves)?;
    log("queued GPU Poseidon hashing");
    log_timing("Leaf hash setup + dispatch", now_ms() - leaf_start);

    // PHASE 2: Buffer allocation
    console::log_1(&"=== PHASE 2: Buffer Creation ===".into());
    let num_layers_to_root = depth;
    let num_layers_to_cap = num_layers_to_root - cap_height;
    let cap_len = 1usize << cap_height;

    let total_nodes: usize = (1..num_layers_to_cap)
        .map(|layer| host_layer_size(num_leaves, layer))
        .sum();

    let buffer_start = now_ms();
    let buffers = create_buffers(ctx_ref, leaf_buffer, total_nodes, cap_len);
    log_timing("Buffer allocation", now_ms() - buffer_start);

    // PHASE 3: Layer processing
    console::log_1(
        &format!(
            "=== PHASE 3: Layer Processing ({} layers) ===",
            num_layers_to_cap
        )
        .into(),
    );
    let layer_setup_start = now_ms();

    for layer in 0..num_layers_to_cap {
        let layer_start = now_ms();

        log(&format!("layer: {}", layer));
        let src_layer_size = host_layer_size(num_leaves, layer);
        let dst_layer_size = host_layer_size(num_leaves, layer + 1);
        let src_offset = if layer == 0 {
            0
        } else {
            host_layer_offset(num_leaves, layer)
        };
        let dst_offset = if layer + 1 < num_layers_to_cap {
            host_layer_offset(num_leaves, layer + 1)
        } else {
            0
        };
        let write_to_cap = (layer + 1) == num_layers_to_cap;

        let args = MerkleTreeKernelArgs {
            cap_len: cap_len as u32,
            layer: layer as u32,
            src_layer_size: src_layer_size as u32,
            dst_layer_size: dst_layer_size as u32,
            src_offset: src_offset as u32,
            dst_offset: dst_offset as u32,
            write_to_cap: write_to_cap as u32,
        };

        // Time bind group creation
        let bind_start = now_ms();
        let args_buffer = ctx_ref
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("merkle-layer-args-{layer}")),
                contents: bytemuck::bytes_of(&args),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = ctx_ref
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("merkle-layer-bind-group-{layer}")),
                layout: &ctx_ref.merkle_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.nodes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.cap.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: args_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ctx_ref.mds_circ.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: ctx_ref.mds_diag.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: ctx_ref.round_constants.as_entire_binding(),
                    },
                ],
            });
        let bind_time = now_ms() - bind_start;

        let threads_per_block = WORKGROUP_SIZE as usize;
        let num_blocks = (dst_layer_size + threads_per_block - 1) / threads_per_block;
        let workgroups_x = num_blocks.max(1) as u32;

        // Time encoder creation and dispatch
        let encode_start = now_ms();
        let mut encoder = ctx_ref
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("merkle-layer-{layer}-encoder")),
            });

        log("execute layer");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("merkle-layer-{layer}-pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx_ref.merkle_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, 1, 1);
        }
        let encode_time = now_ms() - encode_start;

        // Time submission (should be fast)
        let submit_start = now_ms();
        ctx_ref.queue.submit(Some(encoder.finish()));
        ctx_ref.queue.on_submitted_work_done(|| {});
        let submit_time = now_ms() - submit_start;

        let layer_time = now_ms() - layer_start;
        console::log_1(
            &format!(
                "  Layer {}: total={:.2}ms (bind={:.2}ms, encode={:.2}ms, submit={:.2}ms)",
                layer, layer_time, bind_time, encode_time, submit_time
            )
            .into(),
        );
    }

    let total_layer_time = now_ms() - layer_setup_start;
    log_timing("Total layer setup + dispatch", total_layer_time);

    log("queued GPU Merkle buffers");

    let total_setup_time = now_ms() - total_start;
    log_timing(
        "🔧 TOTAL SETUP TIME (everything before GPU wait)",
        total_setup_time,
    );

    Ok(MerkleGpuJob::deferred(
        ctx,
        buffers,
        total_nodes,
        cap_len,
        num_leaves,
        cap_height,
        num_layers_to_cap,
    ))
}

fn fill_subtree_from_gpu<F: RichField>(
    digests_buf: &mut [HashOut<F>],
    accessor: &LayerAccessor<'_, F>,
    leaf_offset: usize,
    subtree_leaves_len: usize,
) -> HashOut<F> {
    if digests_buf.is_empty() {
        debug_assert_eq!(subtree_leaves_len, 1);
        return accessor.node(0, leaf_offset).clone();
    }

    let (left_buf, right_buf) = digests_buf.split_at_mut(digests_buf.len() / 2);
    let (left_digest_slot, left_recursive_buf) = left_buf.split_last_mut().unwrap();
    let (right_digest_slot, right_recursive_buf) = right_buf.split_first_mut().unwrap();

    let half = subtree_leaves_len / 2;
    let left_digest = fill_subtree_from_gpu(left_recursive_buf, accessor, leaf_offset, half);
    let right_digest =
        fill_subtree_from_gpu(right_recursive_buf, accessor, leaf_offset + half, half);

    *left_digest_slot = left_digest.clone();
    *right_digest_slot = right_digest.clone();

    let layer_index = subtree_leaves_len.trailing_zeros() as usize;
    let node_index = leaf_offset >> layer_index;
    accessor.node(layer_index, node_index).clone()
}

struct LayerAccessor<'a, F: RichField> {
    leaf_hashes: &'a [HashOut<F>],
    node_hashes: &'a [HashOut<F>],
    cap_hashes: &'a [HashOut<F>],
    layer_starts: Vec<usize>,
    num_layers_to_cap: usize,
}

impl<'a, F: RichField> LayerAccessor<'a, F> {
    fn new(
        leaf_hashes: &'a [HashOut<F>],
        node_hashes: &'a [HashOut<F>],
        cap_hashes: &'a [HashOut<F>],
        num_leaves: usize,
        num_layers_to_cap: usize,
    ) -> Self {
        let mut layer_starts = vec![0usize; num_layers_to_cap];
        let mut acc = 0;
        for layer in 1..num_layers_to_cap {
            layer_starts[layer] = acc;
            acc += host_layer_size(num_leaves, layer);
        }
        debug_assert_eq!(acc, node_hashes.len());
        Self {
            leaf_hashes,
            node_hashes,
            cap_hashes,
            layer_starts,
            num_layers_to_cap,
        }
    }

    fn node(&self, layer: usize, node_idx: usize) -> &HashOut<F> {
        match layer {
            0 => &self.leaf_hashes[node_idx],
            l if l < self.num_layers_to_cap => {
                let start = self.layer_starts[l];
                &self.node_hashes[start + node_idx]
            }
            l if l == self.num_layers_to_cap => &self.cap_hashes[node_idx],
            _ => panic!("layer {layer} out of bounds"),
        }
    }
}

/// Attempt to build the Merkle tree using the GPU path. Returns `None` if the GPU context has not
/// been initialised.
pub fn try_build_merkle_tree<F>(
    leaves: &[Vec<F>],
    cap_height: usize,
) -> Option<Result<MerkleGpuJob<F>>>
where
    F: RichField + Poseidon,
{
    let context = GPU_CONTEXT.with(|cell| cell.get().cloned());
    let context = match context {
        Some(ctx) => ctx,
        None => {
            log::info!("no cnotext!");
            return None;
        }
    };

    log::info!("not really here");
    Some(build_merkle_tree::<F>(context, leaves, cap_height))
}

/// GPU Merkle output that mirrors the CPU layout.
#[derive(Debug)]
pub struct GpuMerkleOutput<F: RichField> {
    pub digests: Vec<HashOut<F>>,
    pub cap: Vec<HashOut<F>>,
}
