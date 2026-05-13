#!/bin/bash
# Simple build script for ROCm/HIP backend
set -e

echo "Building ds4 with ROCm backend..."
make BACKEND=rocm clean
make BACKEND=rocm -j$(nproc)

echo "Build complete. Executables: ds4, ds4-server, ds4-bench"
