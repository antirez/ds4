"""Host oracle for the actual CUDA V4.1 HC kernel and public dispatcher.

Run both strict source arithmetic and explicit CUDA-style FMA. The extracted
legacy kernel plus an independent scalar oracle check the four HC outputs;
recorded launch boundaries test invalid views, devices and launch errors.
This does not establish CUDA execution, races, FTZ behavior, or performance.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]
HOST = r'''
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; int device_id; };
static struct { uint32_t x; } blockIdx, blockDim, threadIdx;
static int g_quality_mode=0, g_n_gpus=1;
static struct { int device_id; } g_gpu[1]={{3}};
static int current_device=3, query_error=0, launch_error=0, launches=0;
static int cudaGetDevice(int *d) { *d=current_device; return query_error; }
static int cudaGetLastError() { return launch_error; }
static int cuda_ok(int error,const char *) { return error==0; }
static int cuda_decode_stream() { return 7; }
static int ds4_tensor_device_idx(const ds4_gpu_tensor *t) { return t->device_id<0 ? 0 : t->device_id; }
static uint32_t __float_as_uint(float x) { uint32_t u; memcpy(&u,&x,4); return u; }
static float __uint_as_float(uint32_t u) { float x; memcpy(&x,&u,4); return x; }
static float __fadd_rn(float a,float b) { volatile float x=a+b; return x; }
static float __fmul_rn(float a,float b) { volatile float x=a*b; return x; }
'''
LAUNCH = r'''
static void launch_hc(unsigned grid,unsigned threads,unsigned shared,int stream,
        float *out,const float *block,const float *add,const float *res,
        const float *split,uint32_t rows) {
    ++launches;
    assert(grid==(uint64_t(rows)*5120+255)/256 && threads==256 && !shared && stream==7);
    if (launch_error) return;
    blockDim.x=threads;
    for (blockIdx.x=0;blockIdx.x<grid;blockIdx.x++)
        for (threadIdx.x=0;threadIdx.x<threads;threadIdx.x++)
            dsv41_hc_expand_bf16_kernel(out,block,add,res,split,rows);
}
'''
CHECKS = r'''
static float rounded(float x) {
    uint32_t u; memcpy(&u,&x,4);
    if ((u&0x7f800000u)!=0x7f800000u) u+=0x7fffu+((u>>16)&1u);
    u&=0xffff0000u;
    memcpy(&x,&u,4); return x;
}
static float oracle(float block,float post,const float *res,const float *mix) {
    float result=__fmul_rn(rounded(block),post);
    for (unsigned j=0;j<4;j++) {
#if HC_FMA
        result=fmaf(mix[j*4],res[j*5120],result);
#else
        result=__fadd_rn(result,__fmul_rn(mix[j*4],res[j*5120]));
#endif
    }
    return rounded(result);
}
int main() {
    constexpr unsigned E=5120, PAD=16;
    const float sentinel=123456;
    for (uint32_t rows : {1u,2u,8u,32u,65u,128u}) for (unsigned pattern=0;pattern<4;pattern++) {
        const size_t n=size_t(rows)*E, hn=n*4;
        std::vector<float> b(n),a(n),r(hn),s(rows*24),y(hn+2*PAD,sentinel),legacy(hn),rounded_b(n);
        auto fill=[&](std::vector<float> &v,unsigned seed) {
            for (size_t i=0;i<v.size();i++) {
                uint32_t x=uint32_t(i)*747796405u+seed*2891336453u; x^=x>>13;
                v[i]=(int(x%65537u)-32768)/8192.0f;
            }
        };
        fill(b,7);fill(a,8);fill(r,9);fill(s,10);
        if (pattern==1) {
            const uint32_t tie[]={0x3f807fff,0x3f808000,0x3f808001,0xbf818000,0,0x80000000};
            for (size_t i=0;i<n;i++) { b[i]=__uint_as_float(tie[i%6]);a[i]=i%2 ? 0x1p-16f : -0x1p-16f; }
        }
        if (pattern>=2) for (unsigned row=0;row<rows;row++) {
            for (unsigned i=0;i<24;i++) s[row*24+i]=0;
            for (unsigned dst=0;dst<4;dst++) {
                s[row*24+4+dst]=1;
                for (unsigned src=0;src<4;src++)
                    s[row*24+8+src*4+dst]=pattern==2 ? (src ? 0 : 1-0x1p-13f) : 1;
            }
            for (unsigned d=0;d<E;d++) {
                const size_t i=size_t(row)*E+d,h=size_t(row)*4*E+d;
                b[i]=pattern==2 ? -1 : (d%2 ? 1.00390625f : -1.01171875f);
                a[i]=pattern==2 ? 0 : (d%2 ? 0x1p-16f : -0x1p-16f);
                r[h]=pattern==2 ? 1+0x1p-13f : 8192;
                r[h+E]=pattern==2 ? 0 : -8192;
                r[h+2*E]=pattern==2 ? 0 : (d%2 ? 0x1p-8f : -0x1p-8f);
                r[h+3*E]=pattern==2 ? 0 : 0x1p-16f;
            }
        }
        ds4_gpu_tensor out={y.data()+PAD,hn*4,0},block={b.data(),n*4,0},
            add={a.data(),n*4,0},res={r.data(),hn*4,0},split={s.data(),s.size()*4,0};
        const auto original_b=b,original_a=a,original_r=r,original_s=s;
        for (unsigned mode=0;mode<3;mode++) {
            const ds4_gpu_tensor *ap=mode==2 ? &block : mode ? &add : nullptr;
            launches=0;
            assert(ds4_gpu_dsv41_hc_expand_bf16(&out,&block,ap,&res,&split,rows)==1 && launches==1);
            for (size_t i=0;i<n;i++) rounded_b[i]=rounded(mode ? __fadd_rn(b[i],mode==2 ? b[i] : a[i]) : b[i]);
            blockDim.x=256;
            for (size_t i=0;i<hn;i++) {
                blockIdx.x=i/256;threadIdx.x=i%256;
                hc_expand_kernel(legacy.data(),rounded_b.data(),nullptr,nullptr,r.data(),s.data()+4,s.data()+8,
                    E,4,rows,24,24,0,0);
            }
            for (unsigned row=0;row<rows;row++) for (unsigned h=0;h<4;h++) for (unsigned d=0;d<E;d++) {
                const size_t i=size_t(row)*E+d,o=(size_t(row)*4+h)*E+d;
                const float block_value=mode ? __fadd_rn(b[i],mode==2 ? b[i] : a[i]) : b[i];
                const float expected=oracle(block_value,s[row*24+4+h],r.data()+size_t(row)*4*E+d,s.data()+row*24+8+h);
                assert(__float_as_uint(y[PAD+o])==__float_as_uint(expected));
                assert(__float_as_uint(y[PAD+o])==__float_as_uint(rounded(legacy[o])));
                if (pattern==2 && mode<2) assert(y[PAD+o]==(HC_FMA ? -0x1p-26f : 0.0f));
            }
            assert(b==original_b && a==original_a && r==original_r && s==original_s);
            for (unsigned i=0;i<PAD;i++) assert(y[i]==sentinel && y[PAD+hn+i]==sentinel);
        }
        // One out-of-range thread may not access any pointer, including null.
        blockDim.x=256;blockIdx.x=unsigned(n/256);threadIdx.x=0;
        dsv41_hc_expand_bf16_kernel(nullptr,nullptr,nullptr,nullptr,nullptr,rows);
    }
    // Dispatcher validation does not dereference these small backing arrays.
    alignas(16) float storage[128]{};
    ds4_gpu_tensor out={storage,20480*4,0},block={storage+16,5120*4,0},
        add={storage+32,5120*4,0},res={storage+48,20480*4,0},split={storage+64,24*4,0};
    auto base=[](uintptr_t p) { return reinterpret_cast<void *>(p); };
    auto reset=[&]() {
        out={base(0x100000),20480*4,0};block={base(0x200000),5120*4,0};
        add={base(0x300000),5120*4,0};res={base(0x400000),20480*4,0};split={base(0x500000),24*4,0};
        g_quality_mode=0;g_n_gpus=1;current_device=3;query_error=0;launch_error=0;launches=0;
    };
    auto reject=[&](uint32_t rows=1) {
        assert(!ds4_gpu_dsv41_hc_expand_bf16(&out,&block,&add,&res,&split,rows) && launches==0);
    };
    reset();reject(0);reject(UINT32_MAX);
    for (unsigned j=0;j<5;j++) {
        reset();ds4_gpu_tensor *t[]={&out,&block,&add,&res,&split};t[j]->bytes--;reject();
        reset();t[j]->ptr=base(uintptr_t(t[j]->ptr)+4);reject();
        reset();t[j]->ptr=base(UINTPTR_MAX-15);reject();
        reset();t[j]->device_id=-2;reject();
        reset();t[j]->device_id=1;reject();
        reset();t[j]->ptr=nullptr;reject();
    }
    for (unsigned j=1;j<5;j++) for (unsigned offset : {0u,16u}) {
        reset();ds4_gpu_tensor *t[]={&out,&block,&add,&res,&split};out.ptr=base(uintptr_t(t[j]->ptr)+offset);reject();
    }
    reset();g_quality_mode=1;reject();
    reset();g_n_gpus=2;reject();
    reset();current_device=5;reject();
    reset();query_error=1;reject();
    reset();launch_error=1;
    assert(!ds4_gpu_dsv41_hc_expand_bf16(&out,&block,&add,&res,&split,1) && launches==1);
    // The launch boundary declines execution on injected errors, permitting a
    // safe admission check for shared read-only operands and default tags.
    reset();launch_error=1;block.device_id=-1;add=block;
    assert(!ds4_gpu_dsv41_hc_expand_bf16(&out,&block,&add,&res,&split,1) && launches==1);
    printf("CUDA HC host: actual kernel/legacy/CPU, FMA=%d, guards, aliases, devices and launch failures PASS\n",HC_FMA);
}
'''


GRAPH = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include "ds4_gpu_phase.h"
struct ds4_gpu_tensor { int id; };
struct ds41_gpu_graph { uint32_t tp_world; bool quality; void *imatrix; };
struct ds41_prefill_row {
    ds4_gpu_tensor *residual,*after_attn,*ffn_split,*attn_split,*routed,*block,*shared;
};
static constexpr uint32_t DS4_N_EMBD=5120,DS4_N_HC=4,DS4_V41_BF16=0;
static int g_n_gpus=1;
static ds4_gpu_execution_phase phase=DS4_GPU_PHASE_PREFILL;
ds4_gpu_execution_phase ds4_gpu_get_execution_phase(void) { return phase; }
static std::string operations;
static bool fused_ok=true;
static const ds4_gpu_tensor *seen[5];
static int ds4_gpu_dsv41_hc_expand_bf16(ds4_gpu_tensor *out,const ds4_gpu_tensor *block,
        const ds4_gpu_tensor *add,const ds4_gpu_tensor *res,const ds4_gpu_tensor *split,uint32_t rows) {
    assert(rows>0);operations+='F';seen[0]=out;seen[1]=block;seen[2]=add;seen[3]=res;seen[4]=split;
    return fused_ok;
}
static int ds4_gpu_tensor_copy(ds4_gpu_tensor *,uint64_t,const ds4_gpu_tensor *,uint64_t,uint64_t) {
    operations+='C';return 1;
}
static int ds4_gpu_add_tensor(ds4_gpu_tensor *,const ds4_gpu_tensor *,const ds4_gpu_tensor *,uint32_t) {
    operations+='A';return 1;
}
static int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *,uint32_t,uint32_t,uint32_t) {
    operations+='Q';return 1;
}
static int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *,const ds4_gpu_tensor *,
        const ds4_gpu_tensor *,const ds4_gpu_tensor *,uint32_t,uint32_t) {
    operations+='H';return 1;
}
'''
GRAPH_CHECKS = r'''
int main() {
    ds4_gpu_tensor t[7]{};
    ds41_prefill_row b={&t[0],&t[1],&t[2],&t[3],&t[4],&t[5],&t[6]};
    ds41_gpu_graph g={1,false,nullptr};
    for (int p=DS4_GPU_PHASE_AUTO;p<=DS4_GPU_PHASE_MIXED;p++)
        for (bool ffn : {false,true}) for (bool owner : {false,true}) {
            phase=static_cast<ds4_gpu_execution_phase>(p);operations.clear();
            assert(ds41_expand_batch(&g,&b,32,ffn,owner));
            if (phase==DS4_GPU_PHASE_PREFILL) {
                assert(operations=="F");
                assert(seen[0]==(ffn ? b.residual : b.after_attn));
                assert(seen[1]==(ffn ? b.routed : b.block));
                assert(seen[2]==(ffn && !owner ? b.shared : nullptr));
                assert(seen[3]==(ffn ? b.after_attn : b.residual));
                assert(seen[4]==(ffn ? b.ffn_split : b.attn_split));
            } else assert(operations==(ffn ? (owner ? "CQHQ" : "AQHQ") : "QHQ"));
        }
    phase=DS4_GPU_PHASE_PREFILL;
    for (unsigned exclusion=0;exclusion<5;exclusion++) {
        g={1,false,nullptr};g_n_gpus=1;uint32_t rows=32;
        if (exclusion==0) rows=1;
        if (exclusion==1) g.tp_world=2;
        if (exclusion==2) g.quality=true;
        if (exclusion==3) g.imatrix=&g;
        if (exclusion==4) g_n_gpus=2;
        operations.clear();assert(ds41_expand_batch(&g,&b,rows,true,false));
        assert(operations=="AQHQ");
    }
    g={1,false,nullptr};g_n_gpus=1;fused_ok=false;operations.clear();
    assert(!ds41_expand_batch(&g,&b,32,true,false) && operations=="F");
    puts("CUDA HC graph host: prefill-only eligibility, fallbacks, shared ownership and failure propagation PASS");
}
'''


def main():
    source = (ROOT / "ds4_deepseek41_cuda.cuh").read_text()
    common = "\n".join(extract_function(source, signature) for signature in (
        "static bool dsv41_has_floats(", "__device__ static float dsv41_bf16("))
    kernel = extract_function(source, "__global__ static void dsv41_hc_expand_bf16_kernel(")
    legacy = extract_function((ROOT / "ds4_cuda.cu").read_text(), "__global__ static void hc_expand_kernel(")
    wrapper = extract_function(source, 'extern "C" int ds4_gpu_dsv41_hc_expand_bf16(')
    wrapper, count = re.subn(r"dsv41_hc_expand_bf16_kernel<<<(.*?)>>>\s*\(", r"launch_hc(\1,", wrapper, flags=re.S)
    assert count == 1 and "<<<" not in wrapper, "instrument only the HC launch boundary"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    with tempfile.TemporaryDirectory(prefix="ds4-v41-hc-host-") as directory:
        for fma in (0, 1):
            bodies = kernel + legacy
            if fma:
                bodies, count = re.subn(r"acc \+= (comb\[.*?\]|comb_v) \* (r[0-3]|res_v);", r"acc = fmaf(\1, \2, acc);", bodies)
                assert count == 5, "contract exactly the four fused MACs and legacy loop MAC"
            text = HOST + common + bodies + LAUNCH + wrapper + CHECKS
            text = text.replace("__global__ ", "").replace("__device__ ", "")
            path = Path(directory) / f"hc-{fma}.cpp"
            binary = path.with_suffix("")
            path.write_text(text)
            subprocess.run(compiler + ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                "-ffp-contract=off", f"-DHC_FMA={fma}", "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all", str(path), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)
        graph = extract_function((ROOT / "ds4.c").read_text(), "static bool ds41_expand_batch(")
        path = Path(directory) / "graph.cpp"
        binary = path.with_suffix("")
        path.write_text(GRAPH + "\n#undef __APPLE__\n" + graph + GRAPH_CHECKS)
        subprocess.run(compiler + ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-I", str(ROOT), "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all", str(path), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
