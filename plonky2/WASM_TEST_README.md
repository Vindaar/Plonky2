# Plonky2 WASM GPU Merkle Tree Test

This test demonstrates GPU-accelerated Merkle tree construction in Plonky2 running in the browser via WebAssembly and WebGPU.

## Overview

This directory contains two WASM test implementations:

### 1. Basic GPU Merkle Tree Test (`wasm_test.html`)
- Loads test vectors from `test-vectors.txt` (embedded in the WASM binary)
- Constructs a Merkle tree using GPU acceleration via WebGPU
- Uses the Poseidon hash function over the Goldilocks field
- Displays the resulting Merkle root and construction statistics

### 2. CPU vs GPU Comparison Test (`test-gpu-compare.html`)
- Loads test vectors from `test-vectors.txt` or `test-vector-8192-input.txt`
- Constructs the **same** Merkle tree on both CPU and GPU
- Compares the outputs to detect any differences
- Reports mismatches in caps (roots) and digests with detailed output
- Useful for debugging and validating GPU implementation correctness

## Prerequisites

1. **Rust and Cargo**: Install from [rustup.rs](https://rustup.rs/)

2. **wasm-pack**: Install with:
   ```bash
   cargo install wasm-pack
   ```

3. **WebGPU-enabled browser**:
   - Chrome/Edge 113+ (stable)
   - Firefox Nightly with `dom.webgpu.enabled` flag
   - Safari Technology Preview 185+

## Building

From the `plonky2/plonky2` directory, run:

```bash
./build-wasm.sh
```

This will:
- Compile the Rust code to WebAssembly
- Enable the `gpu_merkle` feature
- Generate WASM bindings with `wasm-bindgen`
- Output the compiled module to the `pkg/` directory

## Running the Tests

1. Start a local web server from the `plonky2/plonky2` directory:
   ```bash
   python3 -m http.server 8000
   ```

   Or use any other static file server, e.g.:
   ```bash
   # Using Node.js
   npx serve .

   # Using Rust
   cargo install basic-http-server
   basic-http-server .
   ```

2. Open your WebGPU-enabled browser and navigate to:

   **For the basic GPU test:**
   ```
   http://localhost:8000/wasm_test.html
   ```
   Then click the "Run Merkle Tree Test" button.

   **For the CPU vs GPU comparison test:**
   ```
   http://localhost:8000/test-gpu-compare.html
   ```
   Then:
   - Click "Initialize GPU" to initialize the WebGPU context
   - Click "Run CPU vs GPU Test" to run the comparison

## What to Expect

### Basic GPU Test (`wasm_test.html`)
The test will:
1. Initialize the WASM module
2. Parse test vectors from the embedded `test-vectors.txt`
3. Construct a Merkle tree using GPU acceleration
4. Display results including:
   - Construction time
   - Number of leaves processed
   - Merkle root hash
   - Number of internal nodes (digests)

### CPU vs GPU Comparison Test (`test-gpu-compare.html`)
The test will:
1. Initialize the WASM module and WebGPU context
2. Parse test vectors (configurable to use `test-vectors.txt` or `test-vector-8192-input.txt`)
3. Construct the Merkle tree on **CPU** first (reference implementation)
4. Construct the **same** tree on **GPU**
5. Compare the results and display:
   - Whether CPU and GPU outputs match
   - Timing for both CPU and GPU construction
   - CPU and GPU root hashes
   - Detailed mismatch information if results differ (first 10 differences)
   - Total number of digest mismatches (if any)

All console output and debug messages will be shown in the browser's console and in the output section.

## Files

- **`src/wasm_test.rs`**: Main WASM test implementation containing:
  - `test_merkle_tree_construction()`: Basic GPU Merkle tree test
  - `test_merkle_tree_cpu_vs_gpu()`: CPU vs GPU comparison test
- **`wasm_test.html`**: HTML test harness for basic GPU test
- **`test-gpu-compare.html`**: HTML test harness for CPU vs GPU comparison
- **`build-wasm.sh`**: Build script
- **`test-vectors.txt`**: Default test data (embedded at compile time)
- **`test-vector-8192-input.txt`**: Larger test dataset with 8192 leaves (embedded at compile time)

## Troubleshooting

### "Failed to load WASM module"
- Ensure you built the WASM module with `./build-wasm.sh`
- Check that the `pkg/` directory exists and contains `plonky2.js` and `plonky2_bg.wasm`
- Verify you're serving the files via HTTP (not opening `file://` URLs)

### "WebGPU not supported"
- Ensure you're using a WebGPU-capable browser
- For Chrome/Edge: Check `chrome://gpu` to verify WebGPU is enabled
- For Firefox: Enable `dom.webgpu.enabled` in `about:config`

### Build errors
- Ensure the `gpu_merkle` feature is properly configured in `Cargo.toml`
- Check that all dependencies are up to date: `cargo update`
- Try a clean build: `rm -rf pkg target && ./build-wasm.sh`

## Performance Notes

GPU acceleration provides significant speedup for large Merkle trees. The basic test uses `test-vectors.txt` which contains a moderate number of leaves. For benchmarking larger trees, you can switch to `test-vector-8192-input.txt` in the code.

The async API (`MerkleTree::new_async()`) automatically falls back to CPU construction if GPU initialization fails or if an error occurs during GPU processing.

## Using the CPU vs GPU Comparison for Debugging

The comparison test is particularly useful for debugging GPU implementation issues:

1. **Detecting Correctness Issues**: If the GPU output differs from CPU, you have a bug in the GPU implementation.

2. **Identifying Problem Locations**: The test reports:
   - Mismatches in the Merkle cap (root hashes)
   - Mismatches in individual digests with their indices
   - The first 10 differences are shown in detail to help identify patterns

3. **Customizing the Test**: You can modify `src/wasm_test.rs:212` to switch between test datasets:
   ```rust
   // Use the default test vectors
   let leaves = parse_test_vectors();

   // Or use the 8192 test vectors
   let leaves = parse_test_vectors_8192();
   ```

4. **Adjusting Cap Height**: Modify the `cap_height` variable at line 235 to test different tree configurations.

5. **Interpreting Results**:
   - ✓ **Success**: CPU and GPU results match perfectly - GPU implementation is correct
   - ✗ **Failure**: Results differ - check the reported digest indices to narrow down the issue

## Development

To modify the tests:
1. Edit `src/wasm_test.rs` to change the Merkle tree construction logic or add new test functions
2. Edit `wasm_test.html` or `test-gpu-compare.html` to modify the UI or add additional visualizations
3. Rebuild with `./build-wasm.sh`
4. Refresh your browser (hard refresh may be needed: Ctrl+Shift+R)

### Adding New Test Functions

To add a new WASM-exposed test function:
1. Add your function to `src/wasm_test.rs` with the `#[wasm_bindgen]` attribute
2. Export it in the HTML file's JavaScript import statement
3. Create UI elements to trigger your new test function

## Additional Resources

- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [wasm-bindgen Documentation](https://rustwasm.github.io/wasm-bindgen/)
- [Plonky2 Repository](https://github.com/0xPolygonZero/plonky2)
