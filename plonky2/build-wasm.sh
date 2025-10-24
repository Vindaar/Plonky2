#!/bin/bash

# Build script for Plonky2 WASM test with GPU acceleration

set -e

echo "Building Plonky2 WASM module with GPU Merkle tree support..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "Error: wasm-pack is not installed."
    echo "Please install it with: cargo install wasm-pack"
    exit 1
fi

# Build the WASM module with gpu_merkle and merkle_debug_print features enabled
wasm-pack build \
    --target web \
    --out-dir pkg \
    --features gpu_merkle \
    --features parallel \
    --features merkle_debug_print

echo ""
echo "✓ Build complete!"
echo ""
echo "To test the WASM module:"
echo "  1. Serve the plonky2 directory with a web server, e.g.:"
echo "     python3 -m http.server 8000"
echo "  2. Open http://localhost:8000/wasm_test.html in a WebGPU-enabled browser"
echo ""
echo "Note: WebGPU support is required. Tested in Chrome/Edge 113+ and Firefox Nightly."
