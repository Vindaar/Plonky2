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
const TEST_DATA: &str = include_str!("../../test-vector-small.txt");

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

/// Parse all test vectors from the test data
/// Format: [[elem1, elem2, ...], [elem3, elem4, ...], ...]
fn parse_test_vectors() -> Vec<Vec<GoldilocksField>> {
    let data = TEST_DATA.trim();

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
    arrays
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
