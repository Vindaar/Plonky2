# Plonky2 WASM GPU Merkle Tree Test

This test demonstrates GPU-accelerated Merkle tree construction in Plonky2 running in the browser via WebAssembly and WebGPU.

## Overview

The test:
- Loads test vectors from `test-vector-small.txt` (embedded in the WASM binary)
- Constructs a Merkle tree using GPU acceleration via WebGPU
- Uses the Poseidon hash function over the Goldilocks field
- Displays the resulting Merkle root and construction statistics

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

## Running the Test

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
   ```
   http://localhost:8000/wasm_test.html
   ```

3. Click the "Run Merkle Tree Test" button

## What to Expect

The test will:
1. Initialize the WASM module
2. Parse test vectors from the embedded `test-vector-small.txt`
3. Construct a Merkle tree using GPU acceleration
4. Display results including:
   - Construction time
   - Number of leaves processed
   - Merkle root hash
   - Number of internal nodes (digests)

All console output and debug messages will be shown in the browser's console output section.

## Files

- **`src/wasm_test.rs`**: Main WASM test implementation
- **`wasm_test.html`**: HTML test harness and UI
- **`build-wasm.sh`**: Build script
- **`test-vector-small.txt`**: Test data (embedded at compile time)

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

GPU acceleration provides significant speedup for large Merkle trees. The test uses `test-vector-small.txt` which contains a moderate number of leaves. For benchmarking larger trees, you can modify the test to use `test-vectors.txt` instead.

The async API (`MerkleTree::new_async()`) automatically falls back to CPU construction if GPU initialization fails or if an error occurs during GPU processing.

## Development

To modify the test:
1. Edit `src/wasm_test.rs` to change the Merkle tree construction logic
2. Edit `wasm_test.html` to modify the UI or add additional visualizations
3. Rebuild with `./build-wasm.sh`
4. Refresh your browser (hard refresh may be needed: Ctrl+Shift+R)

## Additional Resources

- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [wasm-bindgen Documentation](https://rustwasm.github.io/wasm-bindgen/)
- [Plonky2 Repository](https://github.com/0xPolygonZero/plonky2)
