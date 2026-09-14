"""Exercise production ROCm Q4 large-prefill dispatch without a GPU.

Only launches, device discovery and allocation are replaced. The production
selectors, slab launcher and grouped attention wrapper run under ASan/UBSan.
Native numerical and tail-store coverage lives in --prefill-long of the C++
fixture; these host checks cannot certify a HIP kernel or measure a speedup.
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
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CUDA_QK_K 256u
#define ROCM_Q4_PREFILL_TOKEN_TILE 8u
#define DS4_GPU_LOG_PREFIX "host-test: "
enum { ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE=-1,
       ROCM_Q4_PREFILL_WMMA_FALLBACK=0, ROCM_Q4_PREFILL_WMMA_USE=1 };
struct dim3 { unsigned x,y,z; dim3(unsigned a,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
struct cuda_block_q8_K { float d; int8_t qs[256]; int16_t bsums[16]; };
struct cuda_block_q4_K { unsigned char bytes[144]; };
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; };
struct quant_call { dim3 grid; unsigned threads; cuda_block_q8_K *out;
                    const float *x; uint32_t k,n; };
static std::vector<quant_call> quant_calls;
static std::vector<cuda_block_q8_K> scratch(65537u*2u);
static unsigned checks, tile_calls, wmma_calls, q8_calls, notes, allocations;
static uint64_t allocated_rows, allocated_blocks;
static int g_ssd_streaming_mode, g_quality_mode, gfx1151=1;
static unsigned fail_quant_call;
static int pending_error, fail_alloc, fail_tile, fail_wmma;
static dim3 last_tile_grid(0);
static uint64_t last_qstride, last_ostride;
static void check(bool ok,const char *why) {
    ++checks;
    if (!ok) { fprintf(stderr,"FAIL Q4 prefill dispatch: %s\n",why); exit(1); }
}
static int rocm_q4_attn_q_b_env_bool(const char *key) {
    const char *value=getenv(key); return value ? (strcmp(value,"0")!=0) : -1;
}
static int rocm_q4_qb_gfx1151_wave32_device() { return gfx1151; }
static int cudaGetLastError() { int rc=pending_error; pending_error=0; return rc; }
static int cuda_ok(int rc,const char *) { return rc==0; }
static void quantize_record(dim3 grid,unsigned threads,cuda_block_q8_K *out,
                            const float *x,uint32_t k,uint32_t n) {
    quant_calls.push_back({grid,threads,out,x,k,n});
    pending_error=quant_calls.size()==fail_quant_call;
}
static cuda_block_q8_K *rocm_q4_K_prequant_alloc(uint64_t rows,uint64_t blocks,const char *) {
    ++allocations; allocated_rows=rows; allocated_blocks=blocks;
    check(rows*blocks<=scratch.size(),"fixture scratch accommodates allocation");
    return fail_alloc ? nullptr : scratch.data();
}
static void rocm_q4_K_prefill_tile8_strided_launch(dim3 grid,float *,const char *,
        cuda_block_q8_K *,uint64_t,uint32_t,uint32_t,uint32_t,
        uint64_t qstride,uint64_t ostride) {
    ++tile_calls;last_tile_grid=grid;last_qstride=qstride;last_ostride=ostride;
    pending_error=fail_tile;
}
template<class... T> static int rocm_q4_K_prefill_wmma_launch(T...) {
    ++wmma_calls;return !fail_wmma;
}
template<class... T> static void rocm_q4_K_prefill_tile8_note(T...) { ++notes; }
static const char *cuda_model_range_ptr(const void *base,uint64_t offset,uint64_t,const char *) {
    return static_cast<const char *>(base)+offset;
}
template<class... T> static int ds4_gpu_matmul_q8_0_tensor(T...) { ++q8_calls;return 1; }
static void reset() {
    quant_calls.clear();tile_calls=wmma_calls=q8_calls=notes=allocations=0;
    fail_quant_call=0;pending_error=fail_alloc=fail_tile=fail_wmma=0;
    g_ssd_streaming_mode=g_quality_mode=0;gfx1151=1;
    unsetenv("DS4_ROCM_DISABLE_Q4_PREFILL_TILE8");
    unsetenv("DS4_ROCM_REQUIRE_Q4_PREFILL_TILE8");
    unsetenv("DS4_ROCM_ENABLE_Q4_PREFILL_WMMA");
    unsetenv("DS4_ROCM_DISABLE_Q4_PREFILL_WMMA");
    unsetenv("DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA");
}
'''
CASES = r'''
int main() {
    std::vector<float> x(65537u*512u), out(8192u*65u), low(8192u*256u);
    std::vector<char> model(1u<<20);
    for (uint32_t k : {256u,512u}) for (uint32_t n : {1u,8u,9u,8192u,32768u,65535u,65536u,65537u}) {
        reset();
        check(rocm_q4_K_q8_quantize_launch(scratch.data(),x.data(),k,n,"oracle")==1,"valid quantizer launch");
        const unsigned count=n<=65535u?1u:(n+32767u)/32768u;
        check(quant_calls.size()==count,"portable grid-y slab count");
        uint32_t row=0;
        for (const auto &call:quant_calls) {
            unsigned rows=n<=65535u?n:std::min(32768u,n-row);
            check(call.grid.x==k/256u&&call.grid.y==rows&&call.grid.z==1u&&call.threads==256u,"exact launch geometry");
            check(call.out==scratch.data()+uint64_t(row)*(k/256u),"Q8 pointer advances by blocks per row");
            check(call.x==x.data()+uint64_t(row)*k,"F32 pointer advances by elements per row");
            check(call.k==k&&call.n==n,"even the final one-row slab retains original reduction mode");
            row+=rows;
        }
        check(row==n,"every input row covered once");
    }
    for(unsigned fail=1;fail<=3;++fail) {
        reset();fail_quant_call=fail;
        check(!rocm_q4_K_q8_quantize_launch(scratch.data(),x.data(),256u,65537u,"fault"),"quantizer propagates launch error");
        check(quant_calls.size()==fail,"quantizer stops before subsequent slabs");
    }
    reset();
    check(!rocm_q4_K_q8_quantize_launch(nullptr,x.data(),256u,1u,"null"),"null output rejected");
    check(!rocm_q4_K_q8_quantize_launch(scratch.data(),nullptr,256u,1u,"null"),"null input rejected");
    for(uint32_t k:{0u,255u,257u}) check(!rocm_q4_K_q8_quantize_launch(scratch.data(),x.data(),k,1u,"shape"),"invalid K rejected");
    check(!rocm_q4_K_q8_quantize_launch(scratch.data(),x.data(),256u,0u,"empty")&&quant_calls.empty(),"empty batch does not enqueue");
    for(uint64_t n:{0ull,1ull,8ull,9ull,255ull,256ull,4095ull,4096ull,4097ull,8191ull,8192ull,8193ull,UINT64_MAX}) {
        reset();
        check(rocm_q4_K_prefill_tile8_scope(n)==(n>8&&n<=8192),"TILE8 boundary scope");
        check(rocm_q4_K_prefill_wmma_select(n,4096,65)==(n>=256&&n<=8192?1:0),"WMMA boundary scope");
    }
    for(unsigned mode=0;mode<4;++mode) {
        reset();
        if(mode==0)g_ssd_streaming_mode=1;
        if(mode==1)g_quality_mode=1;
        if(mode==2)gfx1151=0;
        if(mode==3)setenv("DS4_ROCM_DISABLE_Q4_PREFILL_WMMA","1",1);
        check(rocm_q4_K_prefill_wmma_select(8192,4096,65)==0,"existing WMMA eligibility restrictions preserved");
        setenv("DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA","1",1);
        check(rocm_q4_K_prefill_wmma_select(8192,4096,65)==-1,"strict WMMA cannot silently fall back");
    }
    auto batch=[&](uint32_t n,uint32_t btype=12u) {
        ds4_gpu_tensor tx{x.data(),x.size()*sizeof(float)},tl{low.data(),low.size()*sizeof(float)},to{out.data(),out.size()*sizeof(float)};
        return ds4_gpu_attention_output_q4_K_batch_tensor(&to,&tl,nullptr,nullptr,model.data(),model.size(),0,1u<<18,btype,256,32,8,65,&tx,n);
    };
    for(uint32_t n:{8191u,8192u}) {
        reset();setenv("DS4_ROCM_REQUIRE_Q4_PREFILL_TILE8","1",1);setenv("DS4_ROCM_DISABLE_Q4_PREFILL_WMMA","1",1);
        check(batch(n)==1,"strict eight-group batch reaches TILE8 at long chunk boundary");
        unsigned a_calls=n==8192?2:1;
        check(quant_calls.size()==a_calls+1&&tile_calls==2&&notes==1,"grouped A and B execute without rowwise fallback");
        check(quant_calls[0].n==n*8&&quant_calls[a_calls].n==n,"grouped A and B preserve distinct original row counts");
        check(last_tile_grid.x==3&&last_tile_grid.y==(n+7)/8&&last_tile_grid.z==1,"B token tail geometry");
        check(last_qstride==1&&last_ostride==65,"B token strides");
    }
    reset();setenv("DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA","1",1);
    check(batch(8192)==1&&wmma_calls==1&&tile_calls==1&&quant_calls.size()==1,"strict WMMA A at8192 plus TILE8 B");
    reset();setenv("DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA","1",1);
    check(batch(8192,8)==1&&wmma_calls==1&&q8_calls==1&&quant_calls.empty(),"strict WMMA A with Q8 B at8192");
    reset();check(batch(8193)==0&&quant_calls.empty()&&tile_calls==0&&wmma_calls==0,"8193 is outside public batch scope before any enqueue");
    setenv("DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA","1",1);
    check(batch(8193)==-1&&quant_calls.empty(),"strict8193 rejects without work");
    for(unsigned fail=1;fail<=3;++fail) {
        reset();setenv("DS4_ROCM_DISABLE_Q4_PREFILL_WMMA","1",1);fail_quant_call=fail;
        check(batch(8192)==-1,"grouped launch error never permits fallback replay");
        check(quant_calls.size()==fail&&tile_calls==(fail==3?1u:0u),"grouped error stops later quantizer and matmul launches");
    }
    reset();fail_alloc=1;setenv("DS4_ROCM_DISABLE_Q4_PREFILL_WMMA","1",1);
    check(batch(8192)==0&&quant_calls.empty()&&tile_calls==0,"pre-enqueue allocation failure may decline");
    reset();fail_tile=1;setenv("DS4_ROCM_DISABLE_Q4_PREFILL_WMMA","1",1);
    check(batch(8192)==-1&&tile_calls==1&&quant_calls.size()==2,"A matmul failure stops B");
    reset();fail_wmma=1;
    check(batch(8192)==-1&&wmma_calls==1&&tile_calls==0,"A WMMA failure stops B");
    printf("PASS ROCm Q4 prefill dispatch: %u checks\n",checks);
}
'''


def main():
    runtime = (ROOT / "rocm/ds4_rocm_runtime.cuh").read_text()
    backend = (ROOT / "rocm/ds4_rocm_q4.cuh").read_text()
    bodies = [extract_function(runtime, signature) for signature in (
        "static int cuda_u64_mul_checked(",
        "static int cuda_u64_mul3_checked(",
        "static int cuda_model_range_fits(",
    )]
    bodies += [extract_function(backend, signature) for signature in (
        "static int rocm_q4_K_prefill_tile8_scope(",
        "static int rocm_q4_K_prefill_tile8_requested(",
        "static int rocm_q4_K_prefill_tile8_required(",
        "static int rocm_q4_K_q8_quantize_launch(",
        "static int rocm_q4_K_prefill_wmma_requested_policy(",
        "static int rocm_q4_K_prefill_wmma_attention_a_requested_policy(",
        "static int rocm_q4_K_prefill_wmma_select(",
        "static int rocm_q4_K_prefill_tile8_quant_launch(",
        'extern "C" int ds4_gpu_attention_output_q4_K_batch_tensor(',
    )]
    source = "\n".join(bodies)
    source, count = re.subn(
        r"rocm_q4_q8_K_quantize_kernel<<<([^,]+),\s*([^>]+)>>>\s*\(",
        r"quantize_record(\1,\2,", source)
    assert count == 1 and "<<<" not in source
    with tempfile.TemporaryDirectory(prefix="ds4-rocm-q4-prefill-dispatch-") as tmp:
        path = Path(tmp) / "dispatch.cpp"
        binary = Path(tmp) / "dispatch"
        path.write_text(STUBS + source + CASES)
        compiler = shlex.split(os.environ.get("CXX") or "c++")
        subprocess.run(compiler + ["-std=c++17", "-O2", "-g",
            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
            str(path), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
