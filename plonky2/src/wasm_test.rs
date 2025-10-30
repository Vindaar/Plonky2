//! WASM test for GPU-accelerated Merkle tree construction

use plonky2_field::types::{Field, PrimeField64};
use wasm_bindgen::prelude::*;
use web_sys::console;

use crate::field::goldilocks_field::GoldilocksField;
use crate::hash::merkle_tree::MerkleTree;
#[cfg(all(feature = "gpu_merkle", target_arch = "wasm32"))]
use crate::hash::merkle_tree_gpu;
use crate::hash::poseidon::PoseidonHash;
use crate::plonk::config::GenericHashOut;

/// Test data from test-vector-small.txt embedded as a constant
//const TEST_DATA: &str = include_str!("../../test-vector-small.txt");
//const TEST_DATA: &str = include_str!("../../test-vector-4096-2431.txt");
const TEST_DATA: &str = include_str!("../../test-vectors.txt");
const TEST_DATA_8192: &str = include_str!("../../test-vector-8192-input.txt");

/// Binary test data (524288 leaves × 86 elements per leaf)
/// Uncomment when the binary file is available
const TEST_DATA_BINARY: &[u8] = include_bytes!("../../test-vector-524288.bin");

#[wasm_bindgen]
pub async fn init_gpu_merkle() -> Result<(), JsValue> {
    #[cfg(all(feature = "gpu_merkle", target_arch = "wasm32"))]
    {
        merkle_tree_gpu::initialize()
            .await
            .map_err(|err| JsValue::from_str(&format!("GPU init failed: {err}")))?;
    }
    Ok(())
}

#[cfg(all(feature = "gpu_merkle", target_arch = "wasm32"))]
fn ensure_gpu_merkle_context() {
    if !merkle_tree_gpu::is_initialized() {
        console::warn_1(
            &"Merkle GPU context not initialized; call init_gpu_merkle() before proving".into(),
        );
    }
}

#[cfg(not(all(feature = "gpu_merkle", target_arch = "wasm32")))]
fn ensure_gpu_merkle_context() {}

/// Initialize panic hook and logging for better error messages in the browser console
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();

    // Initialize console logging to bridge log::info!() to browser console
    console_log::init_with_level(log::Level::Info).expect("logger should initialize");

    web_sys::console::log_1(&"WASM module initialized".into());
    log::info!("Console logging initialized - log::info!() will now appear in browser console");
}

/// Parse test vectors from a string
/// Format: [[elem1, elem2, ...], [elem3, elem4, ...], ...]
fn parse_test_vectors_from_str(data_str: &str) -> Vec<Vec<GoldilocksField>> {
    let data = data_str.trim();

    // Check if the data starts with [[ and ends with ]]
    if !data.starts_with("[[") || !data.ends_with("]]") {
        web_sys::console::error_1(
            &format!(
                "Invalid test data format. Expected [[...]], got: {}",
                &data[..100.min(data.len())]
            )
            .into(),
        );
        return Vec::new();
    }

    // Remove outer [[ and ]]
    let data = &data[2..data.len() - 2];

    // Split by "], [" to get individual arrays
    let arrays: Vec<Vec<GoldilocksField>> = data
        .split("], [")
        .filter_map(|array_str| {
            // Parse each element in the array
            let elements: Result<Vec<GoldilocksField>, _> = array_str
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<u64>()
                        .map(GoldilocksField::from_canonical_u64)
                })
                .collect();

            match elements {
                Ok(vec) if !vec.is_empty() => Some(vec),
                Ok(_) => {
                    web_sys::console::warn_1(&"Skipping empty array in test data".into());
                    None
                }
                Err(e) => {
                    web_sys::console::warn_1(&format!("Failed to parse array: {:?}", e).into());
                    None
                }
            }
        })
        .collect();

    web_sys::console::log_1(&format!("Parsed {} test vectors from file", arrays.len()).into());
    // for el in &arrays {
    //     web_sys::console::log_1(&format!("Vector length: {}", el.len()).into());
    // }

    arrays
}

/// Parse all test vectors from the test data
/// Format: [[elem1, elem2, ...], [elem3, elem4, ...], ...]
fn parse_test_vectors() -> Vec<Vec<GoldilocksField>> {
    parse_test_vectors_from_str(TEST_DATA)
}

/// Parse 8192 test vectors
fn parse_test_vectors_8192() -> Vec<Vec<GoldilocksField>> {
    parse_test_vectors_from_str(TEST_DATA_8192)
}

/// Parse binary test vector data into nested vector of GoldilocksField elements
///
/// # Arguments
/// * `data` - Raw binary data as a slice of bytes
/// * `elems_per_leaf` - Number of field elements per leaf (e.g., 86)
///
/// # Format
/// The binary data should contain field elements as 64-bit little-endian unsigned integers.
/// Each element is 8 bytes, and elements are grouped into leaves of size `elems_per_leaf`.
///
/// # Example
/// For 524288 leaves with 86 elements per leaf:
/// - Total elements: 524288 × 86 = 45,088,768
/// - Binary file size: 45,088,768 × 8 bytes = 360,710,144 bytes (~344 MB)
fn parse_binary_test_vectors(data: &[u8], elems_per_leaf: usize) -> Vec<Vec<GoldilocksField>> {
    const ELEMENT_SIZE: usize = 8; // GoldilocksField is u64 (8 bytes)

    let total_elements = data.len() / ELEMENT_SIZE;
    let expected_leaves = total_elements / elems_per_leaf;

    web_sys::console::log_1(
        &format!(
            "Parsing binary test vectors: {} bytes, {} elements, {} leaves ({}×{})",
            data.len(),
            total_elements,
            expected_leaves,
            expected_leaves,
            elems_per_leaf
        )
        .into(),
    );

    // Convert bytes to GoldilocksField elements
    let mut elements = Vec::with_capacity(total_elements);
    for chunk in data.chunks_exact(ELEMENT_SIZE) {
        // Parse as little-endian u64
        let bytes: [u8; 8] = chunk.try_into().expect("chunk is exactly 8 bytes");
        let value = u64::from_le_bytes(bytes);
        elements.push(GoldilocksField::from_canonical_u64(value));
    }

    // Warn if there are leftover bytes
    if data.len() % ELEMENT_SIZE != 0 {
        web_sys::console::warn_1(
            &format!(
                "Warning: {} leftover bytes in binary data (not a multiple of 8)",
                data.len() % ELEMENT_SIZE
            )
            .into(),
        );
    }

    // Reshape into leaves
    let mut leaves = Vec::new();
    for leaf_chunk in elements.chunks_exact(elems_per_leaf) {
        leaves.push(leaf_chunk.to_vec());
    }

    // Warn if there are leftover elements that don't form a complete leaf
    let leftover_elements = elements.len() % elems_per_leaf;
    if leftover_elements != 0 {
        web_sys::console::warn_1(
            &format!(
                "Warning: {} leftover elements (incomplete leaf with {} elements, expected {})",
                leftover_elements, leftover_elements, elems_per_leaf
            )
            .into(),
        );
    }

    web_sys::console::log_1(
        &format!("Parsed {} complete leaves from binary data", leaves.len()).into(),
    );

    leaves
}

/// Parse binary test vectors with 524288 leaves × 86 elements per leaf
/// Uncomment when TEST_DATA_BINARY is available
fn parse_test_vectors_binary() -> Vec<Vec<GoldilocksField>> {
    parse_binary_test_vectors(TEST_DATA_BINARY, 86)
}

/// Construct a Merkle tree using GPU acceleration and return statistics
#[wasm_bindgen]
pub async fn test_merkle_tree_construction() -> Result<JsValue, JsValue> {
    ensure_gpu_merkle_context();

    web_sys::console::log_1(&"Starting Merkle tree construction test...".into());

    // Parse test vectors
    let leaves = parse_test_vectors();
    let num_leaves = leaves.len();

    if num_leaves == 0 {
        return Err(JsValue::from_str("No test vectors found"));
    }

    web_sys::console::log_1(&format!("Parsed {} leaves", num_leaves).into());

    // Ensure we have a power of 2 number of leaves for the Merkle tree
    let log2_leaves = (num_leaves as f64).log2().ceil() as usize;
    let padded_size = 1 << log2_leaves;

    let mut padded_leaves = leaves.clone();

    // Pad with zero vectors if necessary
    while padded_leaves.len() < padded_size {
        padded_leaves.push(vec![GoldilocksField::ZERO; 32]);
    }

    web_sys::console::log_1(
        &format!("Padded to {} leaves (2^{})", padded_size, log2_leaves).into(),
    );

    // Use a cap height of 0 (single root)
    let cap_height = 0;

    web_sys::console::log_1(&"Calling MerkleTree::new_async()...".into());

    console::time_with_label("Merkle tree construction");
    // Construct the Merkle tree using the async (GPU) path
    let tree =
        MerkleTree::<GoldilocksField, PoseidonHash>::new_async(padded_leaves, cap_height).await;
    //MerkleTree::<GoldilocksField, PoseidonHash>::new(padded_leaves, cap_height);

    console::time_end_with_label("Merkle tree construction");
    web_sys::console::log_1(&"Merkle tree construction complete!".into());

    // Get the root hash
    let root = &tree.cap.0[0];
    let root_elements: Vec<String> = root
        .to_vec()
        .iter()
        .map(|x| x.to_canonical_u64().to_string())
        .collect();

    // Create result object
    let result = js_sys::Object::new();
    js_sys::Reflect::set(&result, &"success".into(), &JsValue::TRUE)?;
    js_sys::Reflect::set(&result, &"numLeaves".into(), &JsValue::from(num_leaves))?;
    js_sys::Reflect::set(&result, &"paddedSize".into(), &JsValue::from(padded_size))?;
    js_sys::Reflect::set(&result, &"capHeight".into(), &JsValue::from(cap_height))?;
    js_sys::Reflect::set(
        &result,
        &"rootHash".into(),
        &JsValue::from(root_elements.join(", ")),
    )?;
    js_sys::Reflect::set(
        &result,
        &"numDigests".into(),
        &JsValue::from(tree.digests.len()),
    )?;

    web_sys::console::log_1(&format!("Root hash: [{}]", root_elements.join(", ")).into());

    Ok(result.into())
}

/// Get the number of test vectors available
#[wasm_bindgen]
pub fn get_test_vector_count() -> usize {
    parse_test_vectors().len()
}

/// Test Merkle tree construction comparing CPU and GPU results
#[wasm_bindgen]
pub async fn test_merkle_tree_cpu_vs_gpu() -> Result<JsValue, JsValue> {
    ensure_gpu_merkle_context();

    web_sys::console::log_1(&"Starting CPU vs GPU Merkle tree comparison test...".into());

    // Parse test vectors
    //let leaves = parse_test_vectors_8192();
    //let leaves = parse_test_vectors();
    let leaves = parse_test_vectors_binary();
    let num_leaves = leaves.len();

    if num_leaves == 0 {
        return Err(JsValue::from_str("No test vectors found"));
    }

    web_sys::console::log_1(&format!("Parsed {} leaves from 8192 test vectors", num_leaves).into());

    // Ensure we have a power of 2 number of leaves
    let log2_leaves = (num_leaves as f64).log2().ceil() as usize;
    let padded_size = 1 << log2_leaves;

    let mut padded_leaves = leaves.clone();

    // Pad with zero vectors if necessary
    while padded_leaves.len() < padded_size {
        padded_leaves.push(vec![GoldilocksField::ZERO; 32]);
    }

    web_sys::console::log_1(
        &format!("Padded to {} leaves (2^{})", padded_size, log2_leaves).into(),
    );

    let cap_height = 0;

    // Build on CPU first (reference implementation)
    web_sys::console::log_1(&"Building Merkle tree on CPU (reference)...".into());
    console::time_with_label("CPU Merkle tree construction");
    let cpu_tree =
        MerkleTree::<GoldilocksField, PoseidonHash>::new(padded_leaves.clone(), cap_height);
    console::time_end_with_label("CPU Merkle tree construction");

    // Build on GPU
    web_sys::console::log_1(&"Building Merkle tree on GPU...".into());
    console::time_with_label("GPU Merkle tree construction");
    let gpu_tree =
        MerkleTree::<GoldilocksField, PoseidonHash>::new_async(padded_leaves.clone(), cap_height)
            .await;
    console::time_end_with_label("GPU Merkle tree construction");

    // Compare results
    web_sys::console::log_1(&"Comparing CPU and GPU results...".into());

    let mut differences = Vec::new();

    // Compare caps (roots)
    if cpu_tree.cap.0.len() != gpu_tree.cap.0.len() {
        differences.push(format!(
            "Cap length mismatch: CPU={}, GPU={}",
            cpu_tree.cap.0.len(),
            gpu_tree.cap.0.len()
        ));
    } else {
        for (i, (cpu_cap, gpu_cap)) in cpu_tree.cap.0.iter().zip(gpu_tree.cap.0.iter()).enumerate()
        {
            let cpu_vec = cpu_cap.to_vec();
            let gpu_vec = gpu_cap.to_vec();
            if cpu_vec != gpu_vec {
                differences.push(format!(
                    "Cap[{}] mismatch:\n  CPU: {:?}\n  GPU: {:?}",
                    i, cpu_vec, gpu_vec
                ));
            }
        }
    }

    // Compare digests
    if cpu_tree.digests.len() != gpu_tree.digests.len() {
        differences.push(format!(
            "Digests length mismatch: CPU={}, GPU={}",
            cpu_tree.digests.len(),
            gpu_tree.digests.len()
        ));
    } else {
        let mut digest_mismatches = 0;
        for (i, (cpu_digest, gpu_digest)) in cpu_tree
            .digests
            .iter()
            .zip(gpu_tree.digests.iter())
            .enumerate()
        {
            let cpu_vec = cpu_digest.to_vec();
            let gpu_vec = gpu_digest.to_vec();
            if cpu_vec != gpu_vec {
                digest_mismatches += 1;
                if digest_mismatches <= 10 {
                    // Only report first 10 to avoid overwhelming output
                    differences.push(format!(
                        "Digest[{}] mismatch:\n  CPU: {:?}\n  GPU: {:?}",
                        i, cpu_vec, gpu_vec
                    ));
                }
            }
        }
        if digest_mismatches > 10 {
            differences.push(format!(
                "... and {} more digest mismatches (total: {})",
                digest_mismatches - 10,
                digest_mismatches
            ));
        }
    }

    // Create result object
    let result = js_sys::Object::new();

    if differences.is_empty() {
        web_sys::console::log_1(&"✓ CPU and GPU results match perfectly!".into());
        js_sys::Reflect::set(&result, &"success".into(), &JsValue::TRUE)?;
        js_sys::Reflect::set(&result, &"match".into(), &JsValue::TRUE)?;
        js_sys::Reflect::set(
            &result,
            &"message".into(),
            &"CPU and GPU results match".into(),
        )?;
    } else {
        web_sys::console::error_1(&"✗ CPU and GPU results differ!".into());
        for diff in &differences {
            web_sys::console::error_1(&diff.into());
        }
        js_sys::Reflect::set(&result, &"success".into(), &JsValue::TRUE)?;
        js_sys::Reflect::set(&result, &"match".into(), &JsValue::FALSE)?;
        js_sys::Reflect::set(
            &result,
            &"message".into(),
            &format!("{} differences found", differences.len()).into(),
        )?;
        js_sys::Reflect::set(
            &result,
            &"differences".into(),
            &differences.join("\n\n").into(),
        )?;
    }

    js_sys::Reflect::set(&result, &"numLeaves".into(), &JsValue::from(num_leaves))?;
    js_sys::Reflect::set(&result, &"paddedSize".into(), &JsValue::from(padded_size))?;
    js_sys::Reflect::set(&result, &"capHeight".into(), &JsValue::from(cap_height))?;

    // Add root hashes for reference
    let cpu_root = &cpu_tree.cap.0[0];
    let gpu_root = &gpu_tree.cap.0[0];
    let cpu_root_str: Vec<String> = cpu_root
        .to_vec()
        .iter()
        .map(|x| x.to_canonical_u64().to_string())
        .collect();
    let gpu_root_str: Vec<String> = gpu_root
        .to_vec()
        .iter()
        .map(|x| x.to_canonical_u64().to_string())
        .collect();

    js_sys::Reflect::set(
        &result,
        &"cpuRootHash".into(),
        &cpu_root_str.join(", ").into(),
    )?;
    js_sys::Reflect::set(
        &result,
        &"gpuRootHash".into(),
        &gpu_root_str.join(", ").into(),
    )?;

    Ok(result.into())
}
