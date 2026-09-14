"""Check the production aligned Q8 HC consumer against its GGUF consumer.

Host builds execute the actual integer-dot and kernel arithmetic with staged
warp reduction under ASan/UBSan. They do not establish CUDA compilation or
device scheduling. --cuda runs the unmodified kernels, activation quantizer,
canaries and three graph replays; --emit-cuda writes that native oracle.
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]


def production():
    source = (ROOT / "ds4_cuda.cu").read_text()
    names = ["load_i8x4_i32_aligned", "load_i8x4_i32_unaligned",
             "dot_i8x32_dp4a", "dot_i8_block", "dot_i8x32_aligned_int4"]
    bodies = [extract_function(source, "__device__ __forceinline__ static int32_t " + n + "(")
              for n in names]
    owned_signature = "__device__ static float moe_owned_packed_combine_row("
    # The first occurrence is a forward declaration.
    owned_source = source[source.index(owned_signature, source.index(owned_signature) + 1):]
    bodies.append(extract_function(owned_source, owned_signature))
    raw = extract_function(source, "__global__ static void matmul_q8_0_hc_expand_preq_warp8_kernel(")
    aligned = extract_function(source, "__global__ static void matmul_q8_0_hc_expand_aligned_preq_warp8_kernel(")
    dispatcher = extract_function(source, "static int cuda_matmul_q8_0_hc_expand_tensor_labeled(")
    assert dispatcher.count("ds4_cuda_launch_q8_0_quantize(") == 1
    assert "cuda_q8_quant_warp_reduce_enabled()" in dispatcher
    assert "float *xscale" in dispatcher and "sizeof(float)" in dispatcher
    assert "ds4_mmq_" not in dispatcher, "HC must retain canonical Q8_0 / F32 scales"
    warp = extract_function(source, "__device__ static float warp_sum_f32(")
    assert re.search(r"offset = 16; offset > 0; offset >>= 1", warp)
    assert "v += __shfl_down_sync(0xffffffffu, v, offset);" in warp
    return "\n".join(bodies), raw + "\n" + aligned, warp


def generate(native):
    helpers, kernels, warp = production()
    if not native:
        helpers = helpers.replace("__device__ __forceinline__ ", "").replace("__device__ ", "")
        kernels = kernels.replace("__global__ ", "")
        warp = "static float warp_sum_f32(float v) { return host_warp_sum(v); }"
    text = (ROOT / "tests/test_cuda_q8_hc_aligned.cpp").read_text()
    return text.replace("/* DS4_PRODUCTION_FUNCTIONS */", helpers + "\n" + warp + "\n" + kernels)


def policy_source():
    source = (ROOT / "ds4_cuda.cu").read_text()
    helper = extract_function(source, "static const char *cuda_q8_hc_aligned_weight_ptr(")
    dispatch = extract_function(source, "static int cuda_matmul_q8_0_hc_expand_tensor_labeled(")
    dispatch, launches = re.subn(r"<<<.*?>>>", "", dispatch, flags=re.S)
    assert launches == 2
    prefix = r"""
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <tuple>
static void require(bool b,const char *s) { if (!b) { fprintf(stderr,"Q8 HC policy: %s\n",s); exit(1); } }
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; int device; };
struct int4 { int x,y,z,w; };
using __half = uint16_t;
static int g_n_gpus=1,g_cuda_is_gb10[2]={1,0};
static bool g_decode_graph_capturing;
static unsigned char scratch[65536],raw[64],tensor_data[64];
static std::vector<char> artifact(20*1024*1024);
static void *g_cuda_tmp=scratch;
static uint64_t g_cuda_tmp_bytes=sizeof(scratch);
static struct { void *scratch; uint64_t scratch_bytes; } g_gpu[2];
static int enabled,dp4a,present,raw_present,allocation_fail,quant_fail,writer_fail;
static int lookups,allocations,quants,raw_writers,aligned_writers,error;
static int expected_k=4096,expected_m=4096;
enum { CUDA_DERIVED_Q8_0_ALIGNED_DENSE=101 };
static bool cuda_aligned_q8_enabled() { return enabled; }
static uint64_t ds4_mmq_q8_0_aligned_bytes(int m,int k) {
    const uint64_t blocks=uint64_t(m)*k/32;
    return ((blocks*2+63)&~63ull)+blocks*32;
}
static const char *cuda_derived_weight_ptr(const void *map,uint64_t offset,uint64_t bytes,
        int kind,uint64_t k,uint64_t m,uint64_t groups,uint64_t aligned_bytes) {
    ++lookups;
    require(map==raw && offset==128 && bytes==uint64_t(expected_m)*(expected_k/32)*34 &&
            kind==CUDA_DERIVED_Q8_0_ALIGNED_DENSE && k==unsigned(expected_k) && m==unsigned(expected_m) &&
            groups==1 && aligned_bytes==ds4_mmq_q8_0_aligned_bytes(expected_m,expected_k),"artifact registry key");
    return present ? artifact.data() : nullptr;
}
static int ds4_tensor_device_idx(const ds4_gpu_tensor *t) { return t->device; }
static const char *cuda_resolve_weight_ptr(const void *,uint64_t,uint64_t,int,const char *) {
    return raw_present ? reinterpret_cast<char *>(raw) : nullptr;
}
static void *cuda_tmp_alloc_on(int,uint64_t bytes,const char *) {
    ++allocations; require(bytes<=sizeof(scratch),"scratch bytes");
    return allocation_fail ? nullptr : scratch;
}
static int cuda_q8_use_dp4a() { return dp4a; }
static bool cuda_q8_quant_warp_reduce_enabled() { return true; }
static int cuda_decode_stream() { return 37; }
static int cudaGetLastError() { const int e=error;error=0;return e; }
static int cuda_ok(int e,const char *) { return e==0; }
static void ds4_cuda_launch_q8_0_quantize(bool,unsigned grid,int stream,int8_t *q,float *scale,
        const float *,uint64_t k,uint64_t blocks) {
    ++quants;require(grid==blocks && blocks==(k+31)/32 && stream==37,"quantize launch layout/stream");
    require(q==reinterpret_cast<int8_t *>(scratch) &&
            reinterpret_cast<char *>(scale)==reinterpret_cast<char *>(scratch)+((blocks*32+15)&~15ull),
            "canonical Q8_0 codes and F32 scale scratch");error=quant_fail;
}
template<typename... Args> static void matmul_q8_0_hc_expand_aligned_preq_warp8_kernel(Args... args) {
    const auto a=std::make_tuple(args...);
    const uint64_t scales_bytes=(uint64_t(expected_m)*(expected_k/32)*2+63)&~63ull;
    require(std::get<9>(a)==reinterpret_cast<const int4 *>(artifact.data()+scales_bytes) &&
            std::get<10>(a)==reinterpret_cast<const __half *>(artifact.data()), "aligned artifact scale/code offsets");
    require(std::get<11>(a)==reinterpret_cast<int8_t *>(scratch) &&
            std::get<12>(a)==reinterpret_cast<float *>(scratch+expected_k), "aligned activation pointers");
    ++aligned_writers; error=writer_fail;
}
template<typename... Args> static void matmul_q8_0_hc_expand_preq_warp8_kernel(Args...) {
    ++raw_writers; error=writer_fail;
}
"""
    suffix = r"""
static void defaults() {
    unsetenv("DS4_CUDA_NO_Q8_FUSED_ALIGNED");
    g_n_gpus=1; g_cuda_is_gb10[0]=1; g_decode_graph_capturing=false;
    g_cuda_tmp=scratch; g_cuda_tmp_bytes=sizeof(scratch);
    for (auto &gpu:g_gpu) { gpu.scratch=scratch;gpu.scratch_bytes=sizeof(scratch); }
    enabled=dp4a=present=raw_present=1;
    allocation_fail=quant_fail=writer_fail=0;
    lookups=allocations=quants=raw_writers=aligned_writers=error=0;
    expected_k=4096;expected_m=4096;
}
static int run(unsigned k=4096,unsigned m=4096,int tier=0) {
    expected_k=k;expected_m=m;
    ds4_gpu_tensor out{tensor_data,uint64_t(m)*16,tier},block{tensor_data,uint64_t(m)*4,tier},
        x{tensor_data,uint64_t(k)*4,tier},split{tensor_data,96,tier};
    return cuda_matmul_q8_0_hc_expand_tensor_labeled(&out,&block,raw,
        128+uint64_t(m)*((k+31)/32)*34,128,k,m,&x,nullptr,nullptr,nullptr,nullptr,nullptr,0,
        &out,&split,m,4,"policy test");
}
int main() {
    unsigned cases=0;
    for (unsigned test=0;test<24;++test) {
        defaults(); unsigned k=4096,m=4096;int tier=0,ok=1,q=1,a=1,r=0;
        switch (test) {
        case 0: break;
        case 1:g_decode_graph_capturing=true;break;
        case 2:g_cuda_is_gb10[0]=0;a=0;r=1;break;
        case 3:enabled=0;a=0;r=1;break;
        case 4:dp4a=0;a=0;r=1;break;
        case 5:present=0;a=0;r=1;break;
        case 6:g_n_gpus=2;a=0;r=1;break;
        case 7:g_n_gpus=2;tier=1;a=0;r=1;break;
        case 8:k=512;a=0;r=1;break;
        case 9:m=4095;a=0;r=1;break;
        case 10:g_decode_graph_capturing=true;g_cuda_tmp=nullptr;ok=q=a=0;break;
        case 11:g_decode_graph_capturing=true;g_cuda_tmp_bytes=4096;ok=q=a=0;break;
        case 12:g_n_gpus=2;tier=1;g_decode_graph_capturing=true;g_gpu[1].scratch=nullptr;ok=q=a=0;break;
        case 13:allocation_fail=1;ok=q=a=0;break;
        case 14:quant_fail=1;ok=a=0;break;
        case 15:writer_fail=1;ok=0;break;
        case 16:raw_present=0;ok=q=a=0;break;
        case 17:tier=-1;ok=q=a=0;break;
        case 18:tier=2;ok=q=a=0;break;
        case 19:k=2048;break;
        default: {
            const char *values[]={"","0","1","false"};
            setenv("DS4_CUDA_NO_Q8_FUSED_ALIGNED",values[test-20],1);a=0;r=1;break;
        }
        }
        require(run(k,m,tier)==ok,"return status");
        require(quants==q && aligned_writers==a && raw_writers==r,"one quantizer and selected writer; no retry after failure");
        if (test>=10 && test<=12) require(allocations==0 && lookups==0,"cold capture submitted allocation/lookup");
        ++cases;
    }
    defaults();
    for (uint64_t k : std::initializer_list<uint64_t>{0,512,uint64_t(INT_MAX)+1}) {
        require(!cuda_q8_hc_aligned_weight_ptr(raw,128,1,k,4096,0,1),"invalid K accepted");++cases;
    }
    for (uint64_t m : std::initializer_list<uint64_t>{0,129,uint64_t(INT_MAX)+1}) {
        require(!cuda_q8_hc_aligned_weight_ptr(raw,128,1,4096,m,0,1),"invalid M accepted");++cases;
    }
    require(!cuda_q8_hc_aligned_weight_ptr(nullptr,128,1,4096,4096,0,1),"null model accepted");
    require(lookups==0,"invalid shape reached registry");
    printf("PASS: %u production aligned-Q8 dispatch/graph/fault cases\n",cases+1);
}
"""
    return prefix + helper + dispatch + suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--emit-cuda", type=Path)
    args = parser.parse_args()
    if args.emit_cuda:
        args.emit_cuda.write_text(generate(True))
        return
    with tempfile.TemporaryDirectory(prefix="ds4-q8-hc-aligned-") as tmp:
        source = Path(tmp) / ("oracle.cu" if args.cuda else "oracle.cpp")
        binary = Path(tmp) / "oracle"
        source.write_text(generate(args.cuda))
        if args.cuda:
            compiler = shlex.split(os.environ.get("NVCC", "nvcc"))
            if not compiler or not shutil.which(compiler[0]):
                parser.error("NVCC and a visible NVIDIA GPU are required for --cuda")
            flags = shlex.split(os.environ.get("NVCCFLAGS", "-O3 --use_fast_math"))
            subprocess.run(compiler + flags + ["-std=c++17", "-I", str(ROOT),
                str(source), "-o", str(binary)], check=True)
            subprocess.run([str(binary), str(args.device)], check=True)
        else:
            compiler = shlex.split(os.environ.get("CXX", "c++"))
            for flags in (["-O2", "-ffp-contract=off"], ["-O3", "-ffast-math"]):
                subprocess.run(compiler + flags + ["-std=c++17", "-Wall", "-Wextra",
                    "-Werror", "-Wno-unknown-pragmas", "-fsanitize=address,undefined",
                    "-fno-sanitize-recover=all", str(source), "-o", str(binary)], check=True)
                subprocess.run([str(binary)], check=True)
                policy = Path(tmp) / "policy.cpp"
                policy.write_text(policy_source())
                subprocess.run(compiler + flags + ["-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=address,undefined", "-fno-sanitize-recover=all", str(policy), "-o", str(binary)], check=True)
                subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
