### **Pull Request Description: Add AMD ROCm/HIP Support and Strix Halo Optimizations**

#### **Overview**
This PR introduces a complete AMD ROCm/HIP backend to DwarfStar 4, optimized specifically for hardware with unified memory architectures like the **AMD Strix Halo (gfx1151)**. It migrates the project from its original CUDA dependency to a portable HIP implementation while maintaining functional parity and performance.

#### **Key Changes**
1.  **ROCm/HIP Backend Migration**:
    *   Ported `ds4_cuda.cu` to `ds4_hip.cpp` and transitioned all symbol dependencies from CUDA/cuBLAS to HIP/hipBLAS.
    *   Updated the `Makefile` to detect and support the ROCm stack using `hipcc`.
2.  **Strix Halo / HSA Optimizations**:
    *   **Zero-Copy Memory Access**: Configured the engine to use HSA direct access (Zero-Copy) by default on AMD hardware. This avoids duplicating 83+ GiB of model weights in system RAM, significantly reducing memory overhead.
    *   **Vectorized Kernels**: Optimized F16 and F32 GEMV kernels using vectorized loads and warp-shuffle reductions for improved decoding throughput.
    *   **Hardware Intrinsics**: Replaced scalar loops with AMD-specific hardware dot-product intrinsics (`v_dot4_i32_i8`).
3.  **Unified Tooling**:
    *   Added **`build.sh`**: A one-click script for ROCm compilation.
    *   Added **`rocm_start_server.sh`**: A unified script that handles stale process cleanup, system cache flushing, and optimized server launch.
4.  **Verification**:
    *   Successfully validated with the `rocm-regression` long-context smoke test.
    *   End-to-end testing performed using DeepSeek-V4-Flash Q2-imatrix weights.

#### **Performance Benchmarks (AMD Strix Halo / Radeon Graphics)**
*   **Decoding Speed**: **8.09 – 13.24 tokens/sec** (Non-MTP, Zero-Copy mode).
*   **Prefill Latency**: **~4.45s** for short prompts (post-warmup).
*   **Startup**: ~16s weight warmup for 83.60 GiB mapping.

#### **How to Test**
1.  **Build**: `./build.sh`
2.  **Start**: `./rocm_start_server.sh`
3.  **Verify**:
    ```bash
    curl -X POST http://127.0.0.1:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "ds4flash",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 50
        }'
    ```

---

### **Summary of Work Done**
*   **Full Backend Port**: Replaced all CUDA/cuBLAS APIs with HIP/hipBLAS equivalents.
*   **Environmental Cleanup**: Renamed all CUDA-specific environment variables to `DS4_HIP_*` (e.g., `DS4_HIP_PREFILL_CHUNK`).
*   **Driver Compatibility**: Added robust `hipHostRegister` fallbacks for diverse ROCm driver environments.
*   **Unified Startup Flow**: Fused cleanup and server launch into a single, reliable maintenance script.
*   **Documentation Integrity**: Updated `README.md` with dedicated ROCm onboarding instructions.
