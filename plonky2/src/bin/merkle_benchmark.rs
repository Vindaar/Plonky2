//! Native benchmark for Merkle tree construction with binary test vectors
//!
//! This program builds a Merkle tree from the 524,288-leaf binary test vector.
//!
//! Usage:
//!   cargo run --release --bin merkle_benchmark

use core::slice;
use std::mem::MaybeUninit;
use std::time::Instant;

// Adjust these imports based on your crate structure
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::hash::hash_types::HashOut;
use plonky2::hash::merkle_tree::MerkleTree;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::plonk::config::{GenericHashOut, Hasher};
use plonky2_field::types::{Field, PrimeField64};
use plonky2_maybe_rayon::*;

/// Binary test data (524288 leaves × 86 elements per leaf)
/// Total size: ~344 MB (45,088,768 elements × 8 bytes each)
const TEST_DATA_BINARY: &[u8] = include_bytes!("../../../test-vector-524288.bin");

/// Parse binary test vector data into nested vector of GoldilocksField elements
///
/// # Arguments
/// * `data` - Raw binary data as a slice of bytes
/// * `elems_per_leaf` - Number of field elements per leaf (e.g., 86)
///
/// # Format
/// The binary data should contain field elements as 64-bit little-endian unsigned integers.
/// Each element is 8 bytes, and elements are grouped into leaves of size `elems_per_leaf`.
fn parse_binary_test_vectors(data: &[u8], elems_per_leaf: usize) -> Vec<Vec<GoldilocksField>> {
    const ELEMENT_SIZE: usize = 8; // GoldilocksField is u64 (8 bytes)

    let total_elements = data.len() / ELEMENT_SIZE;
    let expected_leaves = total_elements / elems_per_leaf;

    println!(
        "Parsing binary test vectors: {} bytes, {} elements, {} leaves ({}×{})",
        data.len(),
        total_elements,
        expected_leaves,
        expected_leaves,
        elems_per_leaf
    );

    // Convert bytes to GoldilocksField elements
    let parse_start = Instant::now();
    let mut elements = Vec::with_capacity(total_elements);

    for chunk in data.chunks_exact(ELEMENT_SIZE) {
        // Parse as little-endian u64
        let bytes: [u8; 8] = chunk.try_into().expect("chunk is exactly 8 bytes");
        let value = u64::from_le_bytes(bytes);
        elements.push(GoldilocksField::from_canonical_u64(value));
    }

    let parse_duration = parse_start.elapsed();
    println!("Parsed field elements in {:.2?}", parse_duration);

    // Warn if there are leftover bytes
    if data.len() % ELEMENT_SIZE != 0 {
        eprintln!(
            "Warning: {} leftover bytes in binary data (not a multiple of 8)",
            data.len() % ELEMENT_SIZE
        );
    }

    // Reshape into leaves
    let reshape_start = Instant::now();
    let mut leaves = Vec::new();
    for leaf_chunk in elements.chunks_exact(elems_per_leaf) {
        leaves.push(leaf_chunk.to_vec());
    }

    let reshape_duration = reshape_start.elapsed();
    println!("Reshaped into leaves in {:.2?}", reshape_duration);

    // Warn if there are leftover elements that don't form a complete leaf
    let leftover_elements = elements.len() % elems_per_leaf;
    if leftover_elements != 0 {
        eprintln!(
            "Warning: {} leftover elements (incomplete leaf with {} elements, expected {})",
            leftover_elements, leftover_elements, elems_per_leaf
        );
    }

    println!("Parsed {} complete leaves from binary data", leaves.len());

    leaves
}

/// Parse binary test vectors with 524288 leaves × 86 elements per leaf
fn parse_test_vectors_binary() -> Vec<Vec<GoldilocksField>> {
    parse_binary_test_vectors(TEST_DATA_BINARY, 86)
}

/// Build Merkle tree and return root hash and statistics
fn build_merkle_tree(
    leaves: Vec<Vec<GoldilocksField>>,
    cap_height: usize,
) -> MerkleTree<GoldilocksField, PoseidonHash> {
    let num_leaves = leaves.len();

    // Ensure we have a power of 2 number of leaves for the Merkle tree
    let log2_leaves = (num_leaves as f64).log2().ceil() as usize;
    let padded_size = 1 << log2_leaves;

    let mut padded_leaves = leaves;

    // Pad with zero vectors if necessary
    if padded_leaves.len() < padded_size {
        println!(
            "Padding from {} to {} leaves (2^{})",
            num_leaves, padded_size, log2_leaves
        );
        let padding_element = vec![GoldilocksField::ZERO; 86]; // Match the expected leaf size
        while padded_leaves.len() < padded_size {
            padded_leaves.push(padding_element.clone());
        }
    } else {
        println!(
            "Using {} leaves (2^{}) without padding",
            padded_size, log2_leaves
        );
    }

    // Construct the Merkle tree
    println!("\nBuilding Merkle tree with cap_height = {}...", cap_height);
    let tree_start = Instant::now();
    let tree = MerkleTree::<GoldilocksField, PoseidonHash>::new(padded_leaves, cap_height);
    let tree_duration = tree_start.elapsed();

    println!(
        "✓ Merkle tree construction complete in {:.2?}",
        tree_duration
    );
    println!("  - Tree has {} internal digests", tree.digests.len());
    println!("  - Cap has {} elements", tree.cap.0.len());

    tree
}

fn capacity_up_to_mut<T>(v: &mut Vec<T>, len: usize) -> &mut [MaybeUninit<T>] {
    assert!(v.capacity() >= len);
    let v_ptr = v.as_mut_ptr().cast::<MaybeUninit<T>>();
    unsafe {
        // SAFETY: `v_ptr` is a valid pointer to a buffer of length at least `len`. Upon return, the
        // lifetime will be bound to that of `v`. The underlying memory will not be deallocated as
        // we hold the sole mutable reference to `v`. The contents of the slice may be
        // uninitialized, but the `MaybeUninit` makes it safe.
        slice::from_raw_parts_mut(v_ptr, len)
    }
}

fn hash_leaves(leaves: &Vec<Vec<GoldilocksField>>) {
    let num = 1 << 19;
    let mut leaf = Vec::with_capacity(num);
    let leaf_buf = capacity_up_to_mut(&mut leaf, num);
    leaf_buf
        .par_iter_mut()
        .zip(leaves)
        .for_each(|(leaf_buf, leaf)| {
            leaf_buf.write(PoseidonHash::hash_or_noop(leaf));
        });
}

fn main() {
    println!("=== Merkle Tree Benchmark (Native) ===\n");

    let total_start = Instant::now();

    // Parse binary test vectors
    println!("Step 1: Parsing binary test vectors...");
    let leaves = parse_test_vectors_binary();
    let num_leaves = leaves.len();

    if num_leaves == 0 {
        eprintln!("Error: No test vectors found");
        std::process::exit(1);
    }

    println!("\n✓ Successfully parsed {} leaves\n", num_leaves);

    // Only perform hashing of the leaves to time that
    let hash_start = Instant::now();
    println!("Step 2: Hash leaves...");
    hash_leaves(&leaves);
    let hash_duration = hash_start.elapsed();
    println!("\n✓ Hash leaves execution time: {:.2?}", hash_duration);

    // Build Merkle tree with cap height 0 (single root)
    println!("Step 3: Building Merkle tree...");
    let cap_height = 4;
    let tree = build_merkle_tree(leaves, cap_height);

    // Display root hash
    println!("\n=== Results ===");
    let root = &tree.cap.0[0];
    let root_elements: Vec<String> = root
        .to_vec()
        .iter()
        .map(|x| x.to_canonical_u64().to_string())
        .collect();

    println!("Root hash: [{}]", root_elements.join(", "));
    println!("\nTotal leaves: {}", num_leaves);
    println!("Internal digests: {}", tree.digests.len());
    println!("Cap height: {}", cap_height);

    let total_duration = total_start.elapsed();
    println!("\n✓ Total execution time: {:.2?}", total_duration);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree_construction() {
        let leaves = parse_test_vectors_binary();
        assert!(!leaves.is_empty(), "Should parse at least one leaf");

        let tree = build_merkle_tree(leaves, 0);
        assert!(!tree.cap.0.is_empty(), "Tree should have a root");
        assert!(
            !tree.digests.is_empty(),
            "Tree should have internal digests"
        );
    }

    #[test]
    fn test_binary_parsing() {
        let leaves = parse_binary_test_vectors(TEST_DATA_BINARY, 86);

        // Should have exactly 524288 leaves for the full test vector
        // (or fewer if the file is smaller)
        assert!(!leaves.is_empty(), "Should parse at least one leaf");

        // Each leaf should have 86 elements
        for (i, leaf) in leaves.iter().enumerate().take(10) {
            assert_eq!(
                leaf.len(),
                86,
                "Leaf {} should have 86 elements, has {}",
                i,
                leaf.len()
            );
        }
    }
}
