"""Check the production ROCm shared-X F16 pair/state fusion.

Host mode emulates the existing wave32 reduction and preloads shared X before
serial lane execution. It validates arithmetic/indexing, not GPU barriers.
--rocm runs unchanged kernels on HIP, including captured replays; --bench also
reports paired eager/graph timings. Builds and generated sources stay in /tmp.
"""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]


def source(native):
    common = (ROOT / "rocm/ds4_rocm_common.cuh").read_text()
    q8 = (ROOT / "rocm/ds4_rocm_q8.cuh").read_text()
    norm = (ROOT / "rocm/ds4_rocm_norm_rope.cuh").read_text()
    comp = (ROOT / "rocm/ds4_rocm_compressor.cuh").read_text()
    pair = extract_function(common, "__global__ static void matmul_f16_pair_f32_sharedx_warp_rows_w32_kernel(")
    fused = extract_function(common, "__global__ static void matmul_f16_pair_compressor_store_sharedx_w32_kernel(")
    # Sharing arithmetic is the correctness contract: deliberately fail if a
    # future edit changes only one side's products or reduction order.
    def arithmetic(body):
        return body[body.index("    extern __shared__"):body.index("        out1[row] = acc1;")]
    assert arithmetic(pair) == arithmetic(fused)
    assert pair.count("__syncthreads();") == fused.count("__syncthreads();") == 1
    assert pair.count("warp_sum_f32(") == fused.count("warp_sum_f32(") == 2
    helper = extract_function(common, "__device__ static float warp_sum_f32(float v) {") if native else ""
    bodies = [helper,
        extract_function(q8, "__device__ static float q8_0_scale_scalar("),
        extract_function(norm, "__device__ static float model_ape_value_dev("),
        pair, fused, extract_function(comp, "__global__ static void compressor_store_kernel(")]
    fixture = (ROOT / "tests/test_rocm_f16_compressor.cpp").read_text()
    return fixture.replace("// PRODUCTION_KERNELS", "\n".join(bodies))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--emit-hip", type=Path)
    args = parser.parse_args()
    if args.emit_hip:
        args.emit_hip.write_text(source(True))
        return
    if args.bench and not args.rocm:
        parser.error("--bench requires --rocm")
    with tempfile.TemporaryDirectory(prefix="ds4-rocm-f16-compressor-") as tmp:
        path = Path(tmp) / ("oracle.hip" if args.rocm else "oracle.cpp")
        binary = Path(tmp) / "oracle"
        path.write_text(source(args.rocm))
        if args.rocm:
            command = shlex.split(os.environ.get("HIPCC") or "hipcc")
            if not shutil.which(command[0]):
                parser.error("HIPCC and a visible AMD GPU are required for --rocm")
            flags = shlex.split(os.environ.get("ROCM_CFLAGS") or "-O3 -ffast-math -fno-finite-math-only --offload-arch=gfx1151")
            subprocess.run(command + flags + ["-std=c++17", "-DNATIVE_HIP",
                str(path), "-o", str(binary)], check=True)
            subprocess.run([str(binary)] + (["--bench"] if args.bench else []), check=True)
        else:
            command = shlex.split(os.environ.get("CXX") or "c++")
            for contraction in ("off", "fast"):
                subprocess.run(command + ["-std=c++17", "-O2", "-g",
                    "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                    "-ffp-contract=" + contraction, str(path), "-o", str(binary)], check=True)
                subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
