//! WebGPU scaffolding for Merkle tree construction.
//!
//! This module mirrors the CPU Merkle construction in `merkle_tree.rs` but offloads the heavy
//! Poseidon hashing to a WGSL compute shader. The shader expects Goldilocks field elements in
//! Montgomery form (`a * R mod p` with `R = 2^64`). We therefore convert all hash inputs and
//! constants to Montgomery before uploading them to the GPU and convert results back to the
//! canonical representation after read-back.

#![cfg(all(feature = "gpu_merkle", target_arch = "wasm32"))]

use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;

use anyhow::{anyhow, ensure, Result};
use bytemuck::{Pod, Zeroable};
use once_cell::unsync::OnceCell;
use plonky2_maybe_rayon::{MaybeParIter, ParallelIterator};
use web_sys::console;
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, Buffer, ComputePipeline, Device, Queue};

use crate::hash::hash_types::{HashOut, RichField, NUM_HASH_OUT_ELTS};
use crate::hash::poseidon::{self, Poseidon, PoseidonHash, SPONGE_WIDTH};
use crate::plonk::config::Hasher;

/// Convenience alias for `HashOut<F>` once we have converted generics via `Into`/`From`.
type PoseidonHashOut<F> = HashOut<F>;

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
    pub pipeline: Rc<ComputePipeline>,
    pub bind_group_layout: Rc<BindGroupLayout>,
    pub mds_circ: Buffer,
    pub mds_diag: Buffer,
    pub round_constants: Buffer,
}

impl MerkleTreeGpuContext {
    fn new(
        device: Device,
        queue: Queue,
        pipeline: ComputePipeline,
        bind_group_layout: BindGroupLayout,
        mds_circ: Buffer,
        mds_diag: Buffer,
        round_constants: Buffer,
    ) -> Self {
        Self {
            device: Rc::new(device),
            queue: Rc::new(queue),
            pipeline: Rc::new(pipeline),
            bind_group_layout: Rc::new(bind_group_layout),
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

fn hash_to_montgomery_words<F: RichField>(hash: &PoseidonHashOut<F>) -> [u32; WORDS_PER_DIGEST] {
    let mut out = [0u32; WORDS_PER_DIGEST];
    for (i, element) in hash.elements.iter().enumerate() {
        let limbs = field_to_montgomery_words(element);
        out[i * BIGINT_LIMBS..(i + 1) * BIGINT_LIMBS].copy_from_slice(&limbs);
    }
    out
}

fn montgomery_words_to_hash<F: RichField>(words: &[u32]) -> Result<PoseidonHashOut<F>> {
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

    let bind_group_layout = create_bind_group_layout(&device);
    let pipeline = create_merkle_pipeline(&device, &bind_group_layout)?;
    let (mds_circ, mds_diag, round_constants) = create_poseidon_constant_buffers::<
        crate::field::goldilocks_field::GoldilocksField,
    >(&device);

    log("Creating Merkle GPU context");

    let context = Rc::new(MerkleTreeGpuContext::new(
        device,
        queue,
        pipeline,
        bind_group_layout,
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

/// Returns `true` when the WebGPU context is ready for use.
pub fn is_initialized() -> bool {
    GPU_CONTEXT.with(|cell| cell.get().is_some())
}

fn create_bind_group_layout(device: &Device) -> BindGroupLayout {
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
                    ty: wgpu::BufferBindingType::Uniform,
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

struct MerkleBuffers {
    input: Buffer,
    nodes: Buffer,
    cap: Buffer,
}

fn create_buffers(
    ctx: &MerkleTreeGpuContext,
    leaf_count: usize,
    total_internal_nodes: usize,
    cap_len: usize,
) -> MerkleBuffers {
    let input = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("merkle-input"),
        size: (leaf_count * BYTES_PER_DIGEST) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

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

fn write_words(queue: &Queue, buffer: &Buffer, data: &[u32]) {
    queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
}

fn read_u32_buffer(
    device: &Device,
    queue: &Queue,
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
    let map_result = Rc::new(RefCell::new(None));
    let map_clone = Rc::clone(&map_result);
    slice.map_async(wgpu::MapMode::Read, move |res| {
        *map_clone.borrow_mut() = Some(res);
    });

    while map_result.borrow().is_none() {
        device.poll(wgpu::PollType::Wait);
    }

    map_result
        .borrow_mut()
        .take()
        .expect("map_async result missing")
        .map_err(|err| anyhow!("failed to map buffer: {err}"))?;

    let data = slice.get_mapped_range();
    let words = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    log("...done read u32 buffer start");
    Ok(words)
}

/// Run the GPU Merkle tree pipeline.
pub fn build_merkle_tree<F>(
    ctx: &MerkleTreeGpuContext,
    leaves: &[Vec<F>],
    cap_height: usize,
) -> Result<GpuMerkleOutput<F>>
where
    F: RichField + Poseidon,
{
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

    log("cpu hashing");
    let leaf_hashes: Vec<PoseidonHashOut<F>> = leaves
        .par_iter()
        .map(|leaf| PoseidonHash::hash_or_noop(leaf))
        .collect();
    log("...done cpu hashing");

    if cap_height == depth {
        return Ok(GpuMerkleOutput {
            digests: Vec::new(),
            cap: leaf_hashes,
        });
    }

    let num_layers_to_root = depth;
    let num_layers_to_cap = num_layers_to_root - cap_height;
    let cap_len = 1usize << cap_height;

    let total_nodes: usize = (1..num_layers_to_cap)
        .map(|layer| host_layer_size(num_leaves, layer))
        .sum();

    let mut flattened_leaves = Vec::with_capacity(num_leaves * WORDS_PER_DIGEST);
    for hash in &leaf_hashes {
        flattened_leaves.extend_from_slice(&hash_to_montgomery_words(hash));
    }

    log("writing words");
    let buffers = create_buffers(ctx, num_leaves, total_nodes, cap_len);
    write_words(&ctx.queue, &buffers.input, &flattened_leaves);

    for layer in 0..num_layers_to_cap {
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

        let args_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("merkle-layer-args-{layer}")),
                contents: bytemuck::bytes_of(&args),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("merkle-layer-bind-group-{layer}")),
            layout: &ctx.bind_group_layout,
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

        let threads_per_block = WORKGROUP_SIZE as usize;
        let num_blocks = (dst_layer_size + threads_per_block - 1) / threads_per_block;
        let workgroups_x = num_blocks.max(1) as u32;

        let mut encoder = ctx
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
            pass.set_pipeline(&ctx.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, 1, 1);
        }

        ctx.queue.submit(Some(encoder.finish()));
        ctx.queue.on_submitted_work_done(|| {});
        ctx.device.poll(wgpu::PollType::Wait);
    }

    log("reading back nodes");
    let node_hashes: Vec<HashOut<F>> = if total_nodes > 0 {
        let node_words = read_u32_buffer(
            &ctx.device,
            &ctx.queue,
            &buffers.nodes,
            total_nodes * WORDS_PER_DIGEST,
        )?;
        node_words
            .chunks(WORDS_PER_DIGEST)
            .map(montgomery_words_to_hash::<F>)
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    log("reading back cap");

    let cap_words = read_u32_buffer(
        &ctx.device,
        &ctx.queue,
        &buffers.cap,
        cap_len * WORDS_PER_DIGEST,
    )?;
    let cap_hashes: Vec<HashOut<F>> = cap_words
        .chunks(WORDS_PER_DIGEST)
        .map(montgomery_words_to_hash::<F>)
        .collect::<Result<Vec<_>>>()?;

    let num_digests = 2 * (num_leaves - (1 << cap_height));
    let mut digests = if num_digests == 0 {
        Vec::new()
    } else {
        vec![HashOut::default(); num_digests]
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
        for (subtree_idx, subtree_buf) in digests.chunks_mut(subtree_digests_len).enumerate() {
            let leaf_offset = subtree_idx * subtree_leaves_len;
            let root_digest =
                fill_subtree_from_gpu(subtree_buf, &accessor, leaf_offset, subtree_leaves_len);
            debug_assert_eq!(root_digest, cap_hashes[subtree_idx]);
        }
    }

    Ok(GpuMerkleOutput {
        digests,
        cap: cap_hashes,
    })
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
) -> Option<Result<GpuMerkleOutput<F>>>
where
    F: RichField + Poseidon,
{
    let context = GPU_CONTEXT.with(|cell| cell.get().cloned());
    let context = match context {
        Some(ctx) => ctx,
        None => return None,
    };

    Some(build_merkle_tree::<F>(&context, leaves, cap_height))
}

/// GPU Merkle output that mirrors the CPU layout.
#[derive(Debug)]
pub struct GpuMerkleOutput<F: RichField> {
    pub digests: Vec<HashOut<F>>,
    pub cap: Vec<HashOut<F>>,
}
