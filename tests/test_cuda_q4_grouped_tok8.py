"""Host validation and CUDA source generation for an isolated Q4 grouped tok8 candidate.

Host checks compile the actual dot8 production helper, admission and schedule
with ASan/UBSan. They do not establish CUDA compilation, device rounding,
performance or model quality. --emit-cuda writes a TU for linking with MMQ_OBJS;
--cuda builds/runs it using already-built MMQ objects (NVCC/NVCCFLAGS supported).
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
FIXTURE = ROOT / "speed-bench/cuda_q4_grouped_tok8_bench.cu"
DEFAULT_OBJECTS = " ".join("cuda/mmq/" + name + ".o" for name in (
    "ds4_ggml_stubs", "ds4_mmq", "ds4_mmq_d2r", "quantize", "mmid", "mmvq", "ds4_repack"))
COMMON = r'''
#include <cstdint>
#include <cmath>
#include <cstring>
constexpr uint32_t CUDA_QK_K=256;
struct cuda_block_q4_K { uint16_t d,dmin; uint8_t scales[12],qs[CUDA_QK_K/2]; };
struct cuda_block_q8_K { float d; int8_t qs[CUDA_QK_K]; int16_t bsums[CUDA_QK_K/16]; };
static_assert(sizeof(cuda_block_q4_K)==144 && sizeof(cuda_block_q8_K)==292,"GGUF block ABI");
'''
HOST = r'''
static float dev_f16_to_f32(uint16_t h) {
    const unsigned exponent=(h>>10)&31u, fraction=h&1023u;
    float f=exponent ? (exponent==31 ? (fraction ? NAN : INFINITY) :
                       std::ldexp(float(1024u+fraction),int(exponent)-25)) :
                       std::ldexp(float(fraction),-24);
    return (h&0x8000u) ? -f : f;
}
static int32_t __dp4a(int32_t a,int32_t b,int32_t c) {
    for(unsigned i=0;i<4;++i)
        c+=int8_t(uint32_t(a)>>(8*i))*int8_t(uint32_t(b)>>(8*i));
    return c;
}
'''


def source(native):
    production = (ROOT / "ds4_cuda.cu").read_text()
    signatures = [
        "__device__ static void dev_q4_K_get_scale_min(",
        "__device__ __forceinline__ static int32_t dev_dot_q4_32(",
        "__device__ static void dev_dot_q4_K_q8_K_block8(",
    ]
    prefix = COMMON + HOST
    if native:
        prefix = "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n" + COMMON
        signatures = ["__device__ static float dev_f16_to_f32("] + signatures + [
            "__device__ static float dev_dot_q4_K_q8_K_block(",
            "__device__ static float quarter_warp_sum_f32(",
            "__global__ static void q8_K_quantize_kernel(",
        ]
    bodies = []
    for signature in signatures:
        body = extract_function(production, signature)
        if not native:
            body = body.replace("__device__ ", "").replace("__forceinline__ ", "")
        bodies.append(body)
    return prefix + "\n".join(bodies) + '\n#include "' + FIXTURE.as_posix() + '"\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-cuda", type=Path)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--link-objects", default=DEFAULT_OBJECTS)
    args = parser.parse_args()
    if args.emit_cuda:
        args.emit_cuda.write_text(source(True))
        return
    with tempfile.TemporaryDirectory(prefix="ds4-q4-tok8-") as directory:
        output = Path(directory) / "tok8"
        src = Path(directory) / ("tok8.cu" if args.cuda else "tok8.cpp")
        src.write_text(source(args.cuda))
        if args.cuda:
            compiler = shlex.split(os.environ.get("NVCC", "nvcc"))
            if not compiler or not shutil.which(compiler[0]):
                parser.error("--cuda requires NVCC and a visible CUDA device")
            objects = shlex.split(args.link_objects)
            if any(not (ROOT / obj).is_file() for obj in objects):
                parser.error("build CUDA MMQ_OBJS first, or provide --link-objects")
            flags = shlex.split(os.environ.get("NVCCFLAGS", "-O3 --use_fast_math"))
            subprocess.run(compiler + flags + ["-std=c++17", "-I", str(ROOT), str(src)] +
                           objects + ["-lcublas", "-lcublasLt", "-o", str(output)],
                           cwd=ROOT, check=True)
            command = [str(output), "--device", str(args.device)]
            if args.bench:
                command += ["--bench", "--tokens", str(args.tokens)]
            # Isolate benchmark defaults without changing compiler/build environment.
            native_env = {key: value for key, value in os.environ.items()
                          if not key.startswith("DS4_")}
            subprocess.run(command, env=native_env, check=True)
        else:
            compiler = shlex.split(os.environ.get("CXX", "c++"))
            for flags in (["-O2"], ["-O3", "-ffast-math", "-fno-finite-math-only"]):
                subprocess.run(compiler + flags + ["-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-Wno-unknown-pragmas", "-fsanitize=address,undefined",
                    "-fno-sanitize-recover=all", "-I", str(ROOT),
                    str(src), "-o", str(output)], check=True)
                subprocess.run([str(output)], check=True)


if __name__ == "__main__":
    main()
