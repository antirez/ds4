"""Exercise production F16 compressor kernels against pair + state-store.

Host mode executes the one-barrier kernels lane 31 through lane 0, so the
serial reader sees every partial sum. It checks arithmetic and indexing, not
GPU synchronization or performance. Native mode keeps the kernels unchanged
and also checks graph replay with fresh inputs. All builds live in /tmp.
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
    backend = (ROOT / "ds4_cuda.cu").read_text()
    candidate = (ROOT / "cuda/ds4_f16_compressor.cuh").read_text()
    kernels = [extract_function(backend, signature) for signature in (
        "__device__ static float model_scalar_dev(",
        "__global__ static void matmul_f16_pair_ordered_chunks_kernel(",
        "__global__ static void compressor_store_kernel(")]
    kernels += [extract_function(candidate, "__global__ static void " + name + "(")
                for name in ("matmul_f16_pair_compressor_store_ordered_chunks_kernel",
                             "f16_pair_chunk32_repack_kernel",
                             "matmul_f16_pair_compressor_store_chunk32_prefetch8_kernel")]
    if not native:
        for body in kernels:
            # The host schedule below is valid only for this exact topology.
            assert body.count("__syncthreads();") <= 1
            if "__syncthreads();" in body:
                after = body.split("__syncthreads();")[1].lstrip()
                assert after.startswith("if (tid == 0")
    fixture = (ROOT / "tests/test_cuda_f16_compressor.cpp").read_text()
    return fixture.replace("// PRODUCTION_KERNELS", "\n".join(kernels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--emit-cuda", type=Path)
    args = parser.parse_args()
    if args.emit_cuda:
        args.emit_cuda.write_text(source(True))
        return
    if args.bench and not args.cuda:
        parser.error("--bench requires --cuda")
    with tempfile.TemporaryDirectory(prefix="ds4-f16-compressor-") as tmp:
        path = Path(tmp) / ("oracle.cu" if args.cuda else "oracle.cpp")
        binary = Path(tmp) / "oracle"
        path.write_text(source(args.cuda))
        if args.cuda:
            command = shlex.split(os.environ.get("NVCC") or "nvcc")
            if not shutil.which(command[0]):
                parser.error("NVCC and a visible NVIDIA GPU are required for --cuda")
            flags = shlex.split(os.environ.get("NVCCFLAGS") or "-O3 --use_fast_math -arch=sm_121")
            subprocess.run(command + flags + ["-std=c++17", "-DNATIVE_CUDA",
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
