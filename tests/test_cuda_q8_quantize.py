"""Run the Q8_0 production warp reducer on a staged host shuffle model.

Strict/fast ASan+UBSan builds verify the actual helper, all 1..32 tails,
scale/rounding, and production quality/rollback policy. --cuda compiles the
production quantizers and runs bytewise ordinary/grouped/dual/graph oracles.
Host tests do not establish CUDA compilation, scheduling, or device math.
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
HEADER = ROOT / "cuda/ds4_q8_quantize.cuh"
FIXTURE = ROOT / "tests/test_cuda_q8_quantize.cpp"

HOST_PREFIX = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static void check_host(bool ok,const char *what) {
    if (!ok) { fprintf(stderr,"Q8 production helper: %s\n",what); exit(1); }
}
static uint32_t word(float x) { uint32_t b; memcpy(&b,&x,4); return b; }
static struct { unsigned x; } threadIdx;
static float snapshots[6][32];
static unsigned step, calls;
static float __shfl_down_sync(unsigned mask,float value,unsigned delta,int width) {
    const unsigned lane=threadIdx.x&31u;
    check_host(step<5 && delta==(16u>>step) && mask==0xffffffffu && width==32,"shuffle tree/mask");
    check_host(word(value)==word(snapshots[step][lane]),"shuffle pre-instruction value");
    const float other=snapshots[step][lane+delta<32 ? lane+delta : lane];
    ++step; return other;
}
static float __shfl_sync(unsigned mask,float value,unsigned source,int width) {
    check_host(step==5 && source==0 && mask==0xffffffffu && width==32,"lane-zero broadcast");
    check_host(word(value)==word(snapshots[5][threadIdx.x&31u]),"final lane value");
    ++calls; return snapshots[5][0];
}
#define __syncthreads() ((void)0)
'''

HOST_SUFFIX = r'''
static float production_host_amax(const float *x,unsigned n) {
    for (unsigned lane=0;lane<32;++lane) snapshots[0][lane]=lane<n ? fabsf(x[lane]) : 0.f;
    for (unsigned stage=0,stride=16;stride;stride>>=1,++stage) {
        memcpy(snapshots[stage+1],snapshots[stage],sizeof(snapshots[stage]));
        for (unsigned lane=0;lane<stride;++lane)
            snapshots[stage+1][lane]=fmaxf(snapshots[stage][lane],snapshots[stage][lane+stride]);
    }
    float shared[256]; for (float &v : shared) v=-1234.f;
    const unsigned warp=(calls/32u)%8u;
    for (unsigned lane=0;lane<32;++lane) {
        threadIdx.x=warp*32+lane; step=0;
        const float value=ds4_cuda_q8_0_warp_amax<true>(snapshots[0][lane],shared);
        check_host(word(value)==word(snapshots[5][0]),"broadcast amax bits");
    }
    for (float v : shared) check_host(v==-1234.f,"warp variant touched shared memory");
    return snapshots[5][0];
}
#define DS4_Q8_HOST_AMAX_EXTRACTED
'''


def host_source():
    header = HEADER.read_text()
    helper = extract_function(header, "__device__ __forceinline__ static float ds4_cuda_q8_0_warp_amax(")
    helper = "template<bool WARP_REDUCE>\n" + helper.replace("__device__ __forceinline__ ", "")
    source = (ROOT / "ds4_cuda.cu").read_text()
    policy = extract_function(source, "static inline bool cuda_q8_quant_warp_reduce_enabled(")
    refresh = extract_function(source, "static void cuda_decode_dispatch_env_refresh(")
    assignment = re.search(r'g_cuda_disable_q8_quant_warp_reduce\s*=\s*getenv\("DS4_CUDA_DISABLE_Q8_QUANT_WARP_REDUCE"\)\s*!=\s*NULL;',refresh)
    assert assignment, "production rollback initializer changed"
    for name in ("q8_0_quantize", "q8_0_group_slice_quantize", "q8_dual_quantize"):
        calls = re.findall(r"ds4_cuda_launch_"+name+r"\(\s*([^,]+),",source)
        assert calls and all(arg.strip()=="cuda_q8_quant_warp_reduce_enabled()" for arg in calls)
    assert not re.search(r"(?:quantize_q8_0_f32_kernel|quantize_q8_0_group_slice_rows_kernel|q8_K_q8_0_quantize_kernel)(?:\s*<[^>]+>)?\s*<<<",source)
    production_policy = "static int g_quality_mode,g_cuda_disable_q8_quant_warp_reduce;\n" + policy
    production_policy += '\nstatic void refresh_policy() { '+assignment.group()+' }\n'
    production_policy += r'''
static void production_policy_test() {
    for (int quality : {0,1}) {
        g_quality_mode=quality;
        unsetenv("DS4_CUDA_DISABLE_Q8_QUANT_WARP_REDUCE"); refresh_policy();
        check_host(cuda_q8_quant_warp_reduce_enabled()==!quality,"default/quality");
        for (const char *value : {"","0","1","false"}) {
            setenv("DS4_CUDA_DISABLE_Q8_QUANT_WARP_REDUCE",value,1); refresh_policy();
            check_host(!cuda_q8_quant_warp_reduce_enabled(),"presence rollback");
        }
    }
    // Exercise exceptional maximum operands without undefined lrintf(NaN).
    for (unsigned pattern=0;pattern<5;++pattern) {
        float values[32];
        for (unsigned i=0;i<32;++i) {
            uint32_t b=pattern==0 ? 0x7fc00001u+i : pattern==1 ? 0x7f800000u :
                       pattern==2 ? (i&1 ? 0xff800000u : 0x3f800000u) :
                       pattern==3 ? (i%7==0 ? 0x7fc00005u : 0x3f000000u) :
                                    (i&1 ? 0x80000000u : 0u);
            memcpy(&values[i],&b,4);
        }
        for (unsigned n=1;n<=32;++n) (void)production_host_amax(values,n);
    }
    puts("PASS: actual shuffle helper, NaN/Inf/zero max operands, quality and presence rollback");
}
'''
    # initializer_list is needed by the policy before the fixture's includes.
    return '#include <initializer_list>\n' + HOST_PREFIX + helper + HOST_SUFFIX + production_policy + FIXTURE.read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda",action="store_true")
    parser.add_argument("--device",type=int,default=0)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ds4-q8-warp-") as tmp:
        binary=Path(tmp)/"q8"
        if args.cuda:
            compiler=shlex.split(os.environ.get("NVCC","nvcc"))
            if not compiler or not shutil.which(compiler[0]):
                parser.error("NVCC and a visible device are required for --cuda")
            flags=shlex.split(os.environ.get("NVCCFLAGS","-O3 --use_fast_math"))
            subprocess.run(compiler+flags+["-x","cu","-std=c++17",str(FIXTURE),"-o",str(binary)],check=True)
            subprocess.run([str(binary),str(args.device)],check=True)
        else:
            source=Path(tmp)/"q8.cpp"; source.write_text(host_source())
            compiler=shlex.split(os.environ.get("CXX","c++"))
            for flags in (["-O2"],["-O3","-ffast-math","-fno-finite-math-only"]):
                subprocess.run(compiler+flags+["-std=c++17","-Wall","-Wextra","-Werror",
                    "-Wno-unknown-pragmas","-fsanitize=address,undefined",str(source),"-o",str(binary)],check=True)
                subprocess.run([str(binary)],check=True)


if __name__=="__main__":
    main()
