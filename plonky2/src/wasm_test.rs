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
const TEST_DATA: &str = include_str!("../../test-vectors.txt");
const TEST_DATA_8192: &str = include_str!("../../test-vector-8192-input.txt");

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
    let leaves = parse_test_vectors();
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

    let cap_height = 4;

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
