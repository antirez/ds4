"""Fault-inject the real ROCm fused F16 dispatcher without a GPU.

Kernel enqueue and weight resolution are observable stubs; actual arithmetic
is checked separately by test_rocm_f16_compressor.py.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]
STUBS = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
using __half = uint16_t;
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; int owner; };
static int g_quality_mode, graph_dump, gfx1151=1, fail_resolve, launch_error;
static unsigned writers, resolves, checks;
static int ds4_rocm_is_gfx1151() { return gfx1151; }
static struct config { int graph_dump; } cfg;
static const config *cuda_runtime_config() { cfg.graph_dump=graph_dump; return &cfg; }
static const char *cuda_model_range_ptr(const void *base, uint64_t offset, uint64_t, const char *) {
    return ++resolves == unsigned(fail_resolve) ? nullptr : (const char *)((uintptr_t)base+offset);
}
static int cudaGetLastError() { return launch_error; }
static int cuda_ok(int error, const char *) { return !error; }
template<class... T> static void matmul_f16_pair_compressor_store_sharedx_w32_kernel(T...) { ++writers; }
static void check(bool ok, const char *why) {
    ++checks;
    if (!ok) { fprintf(stderr,"FAIL ROCm F16 policy: %s\n",why); exit(1); }
}
'''
CASES = r'''
struct fixture {
    ds4_gpu_tensor out0{(void *)0x10000000,4096,0}, out1{(void *)0x10010000,4096,0};
    ds4_gpu_tensor state0{(void *)0x20000000,262144,0}, state1{(void *)0x30000000,262144,0};
    ds4_gpu_tensor x{(void *)0x40000000,16384,0};
    const void *model=(void *)0x100000000ull;
    uint64_t model_size=1ull<<30,w0=0,w1=16u<<20,ape=32u<<20,k=4096;
    uint32_t width=1024,ratio=4,type=0;
    int run(bool null_out=false) {
        return ds4_gpu_matmul_f16_pair_compressor_store_tensor(
            null_out?nullptr:&out0,&out1,&state0,&state1,model,model_size,
            w0,w1,ape,type,k,width,&x,ratio,UINT32_MAX);
    }
};
static void reset() {
    g_quality_mode=graph_dump=fail_resolve=launch_error=0; gfx1151=1; writers=resolves=0;
}
int main() {
    for(unsigned shape=0;shape<3;++shape) for(unsigned ape=0;ape<2;++ape) {
        reset(); fixture f; f.type=ape; f.width=shape==0?256:shape==1?1024:512; f.ratio=shape==2?128:4;
        check(f.run()==1&&writers==1&&resolves==3,"supported shape submits exactly one writer");
        check(f.run()==1&&writers==2&&resolves==6,"repeated decode remains one writer");
    }
    for(unsigned kind=0;kind<13;++kind) {
        reset(); fixture f;
        switch(kind) {
        case 0:g_quality_mode=1;break; case 1:graph_dump=1;break; case 2:gfx1151=0;break;
        case 3:f.width=512;break; case 4:f.width=256;f.ratio=128;break;
        case 5:f.k=2048;break; case 6:f.ratio=0;break; case 7:f.k=UINT64_MAX;break;
        case 8:f.width=UINT32_MAX;break; case 9:f.type=8;break; case 10:f.type=UINT32_MAX;break;
        case 11:f.width=0;break; case 12:f.ratio=UINT32_MAX;break;
        }
        check(f.run()==0&&writers==0&&resolves==0,"unsupported mode/shape/type declines before work");
    }
    for(unsigned kind=0;kind<18;++kind) {
        reset(); fixture f;
        switch(kind) {
        case 0:f.x.bytes=16383;break; case 1:f.out0.bytes=4095;break;
        case 2:f.out1.bytes=4095;break; case 3:f.state0.bytes=32767;break;
        case 4:f.state1.bytes=32767;break; case 5:f.model_size=f.w1+4096*1024*2-1;break;
        case 6:f.w0=UINT64_MAX;break; case 7:f.w1=UINT64_MAX;break; case 8:f.ape=UINT64_MAX;break;
        case 9:f.out0.ptr=nullptr;break; case 10:f.out1.ptr=nullptr;break; case 11:f.x.ptr=nullptr;break;
        case 12:f.state0.ptr=nullptr;break; case 13:f.state1.ptr=nullptr;break;
        case 14:f.model=nullptr;break; case 15:f.model_size=f.ape+4096*4-1;break;
        case 16:f.type=1;f.model_size=f.ape+4096*2-1;break;
        case 17:break;
        }
        check(f.run(kind==17)==-1&&writers==0&&resolves==0,"malformed buffers/ranges fail before work");
    }
    for(unsigned a=0;a<4;++a) for(unsigned b=a+1;b<4;++b) {
        reset(); fixture f; ds4_gpu_tensor *out[]={&f.out0,&f.out1,&f.state0,&f.state1};
        out[a]->ptr=(void *)((uintptr_t)out[b]->ptr+4);
        check(f.run()==0&&writers==0&&resolves==0,"all overlapping writers decline");
    }
    for(unsigned a=0;a<4;++a) for(unsigned reader=0;reader<4;++reader) {
        reset();fixture f; ds4_gpu_tensor *out[]={&f.out0,&f.out1,&f.state0,&f.state1};
        out[a]->ptr=reader==0?(void *)((uintptr_t)f.x.ptr+4):
            (void *)((uintptr_t)f.model+(reader==1?f.w0:reader==2?f.w1:f.ape)+4);
        check(f.run()==0&&writers==0,"writers cannot alias inputs, weights or APE");
        check(resolves==(reader==0?0u:3u),"alias refusal precedes writer");
    }
    for(int which=1;which<=3;++which) {
        reset();fixture f;fail_resolve=which;
        check(f.run()==-1&&writers==0,"unavailable weight/APE fails without enqueue");
    }
    reset();fixture f; launch_error=1;
    check(f.run()==-1&&writers==1,"launch failure cannot request a fallback replay");
    check(!cuda_f16_compressor_ranges_overlap((void *)100,4,(void *)104,4),"adjacent ranges do not overlap");
    check(cuda_f16_compressor_ranges_overlap((void *)(UINTPTR_MAX-3),4,(void *)(UINTPTR_MAX-1),2),"overlap avoids endpoint overflow");
    printf("PASS ROCm F16 compressor policy: %u checks\n",checks);
}
'''

def main():
    runtime=(ROOT/"rocm/ds4_rocm_runtime.cuh").read_text()
    backend=(ROOT/"rocm/ds4_rocm_matmul.cuh").read_text()
    bodies=[extract_function(runtime,s) for s in (
        "static int cuda_model_range_fits(","static int cuda_tensor_has_bytes(")]
    bodies += [extract_function(backend,s) for s in (
        "static bool cuda_f16_compressor_ranges_overlap(",
        'extern "C" int ds4_gpu_matmul_f16_pair_compressor_store_tensor(')]
    dispatcher=bodies[-1]
    assert dispatcher.count("<<<")==1
    bodies[-1]=re.sub(r"<<<[\s\S]*?\>\>\>","",dispatcher)
    with tempfile.TemporaryDirectory(prefix="ds4-rocm-f16-policy-") as tmp:
        path=Path(tmp)/"policy.cpp"; binary=Path(tmp)/"policy"
        path.write_text(STUBS+"\n".join(bodies)+CASES)
        command=shlex.split(os.environ.get("CXX") or "c++")
        subprocess.run(command+["-std=c++17","-O2","-g","-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",str(path),"-o",str(binary)],check=True)
        subprocess.run([str(binary)],check=True)

if __name__=="__main__":main()
