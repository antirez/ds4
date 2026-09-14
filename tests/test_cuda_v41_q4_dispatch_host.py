"""Exercise the extracted public V4.1 CUDA output dispatcher on the host.

Only the BF16 CUDA launch syntax is replaced with a call to a recording
boundary. Other backend dependencies record requests or inject failures.
This tests the real wrapper's dispatch, ownership and sequencing, not GPU math.
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
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
using std::min;
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; int device_id; };
struct cuda_block_q4_K { uint16_t d,dmin; uint8_t scales[12],qs[128]; };
struct cuda_block_q8_K { float d; int8_t qs[256]; int16_t bsums[16]; };
static_assert(sizeof(cuda_block_q4_K)==144 && sizeof(cuda_block_q8_K)==292,"block ABI");
static int g_n_gpus=1, g_cuda_test_q4_mmq_strict=0;
static struct { int device_id; } g_gpu[2]={{3},{5}};
static int current_device=3, set_device_fail=0, device_query_fail=0;
static int mmq_enabled=1, mmq_changes_device=1;
enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone, cudaStreamCaptureStatusActive, cudaStreamCaptureStatusInvalidated
};
static cudaStreamCaptureStatus capture_status=cudaStreamCaptureStatusNone;
static int capture_query_fail=0, mmq_queries=0;
static int cuda_decode_stream() { return 7; }
static int cudaStreamIsCapturing(int stream,cudaStreamCaptureStatus *status) {
    assert(stream==7);
    *status=capture_status;
    return capture_query_fail;
}
static int cudaGetDevice(int *d) { *d=current_device; return device_query_fail; }
static int cudaSetDevice(int d) {
    if (set_device_fail) return 1;
    current_device=d;
    return 0;
}
static int cuda_ok(int error,const char *) { return error==0; }
static int cuda_use_mmq() {
    ++mmq_queries;
    if (mmq_enabled && mmq_changes_device) current_device=0;
    return mmq_enabled;
}
static int ds4_tensor_device_idx(const ds4_gpu_tensor *t) {
    return t->device_id < 0 ? 0 : t->device_id;
}
struct Event {
    char op;
    uint32_t type, outputs, width, full_width, k0, groups, rows;
    const void *w, *x, *y, *scratch;
    bool mmq, bf16=false;
};
struct WeightRequest { uint64_t offset, bytes; int tier; };
static std::vector<Event> events;
static std::vector<WeightRequest> requests;
static std::vector<uint64_t> arena0(512*1024), arena1(512*1024);
static void *arena=arena0.data();
static void *scratch_override=nullptr;
static bool scratch_null=false, grow_in_scalar=false;
static uint64_t last_scratch_bytes=0;
static unsigned allocations=0, scalar_calls=0;
static int fail_resolve=0, misalign_resolve=0;
static char fail_stage=0;
static bool bf16_error=false;
static const char *cuda_resolve_weight_ptr(const void *map,uint64_t off,uint64_t bytes,
                                          int tier,const char *) {
    requests.push_back({off,bytes,tier});
    if (fail_resolve==int(requests.size())) return nullptr;
    return (const char *)map+off+(misalign_resolve==int(requests.size()) ? 2 : 0);
}
static void *cuda_tmp_alloc_on(int tier,uint64_t bytes,const char *) {
    assert(current_device==g_gpu[tier].device_id);
    ++allocations;
    last_scratch_bytes=bytes;
    return scratch_null ? nullptr : scratch_override ? scratch_override : arena;
}
static int record_projection(uint32_t type,float *out,const char *w,const float *x,
        void *scratch,uint32_t outputs,uint32_t width,uint32_t full_width,
        uint32_t k0,uint32_t groups,uint32_t rows,bool mmq) {
    const char op=outputs==1024 ? 'A' : 'B';
    assert(scratch==arena);
    events.push_back({op,type,outputs,width,full_width,k0,groups,rows,w,x,out,scratch,mmq});
    return fail_stage!=op;
}
static int dsv41_output_q4_rows(float *o,const char *w,const float *x,void *s,uint64_t bytes,
        uint32_t m,uint32_t k,uint32_t full,uint32_t k0,uint32_t groups,uint32_t rows,
        bool mmq,bool bf16) {
    assert(bytes==last_scratch_bytes);
    const int ok=record_projection(12,o,w,x,s,m,k,full,k0,groups,rows,mmq);
    events.back().bf16=bf16;
    return ok;
}
static int dsv41_output_q8_rows(float *o,const char *w,const float *x,void *s,
        uint32_t m,uint32_t k,uint32_t full,uint32_t k0,uint32_t groups,uint32_t rows) {
    return record_projection(8,o,w,x,s,m,k,full,k0,groups,rows,false);
}
static void dsv41_bf16_kernel(float *out,uint64_t count) {
    assert(!events.empty() && events.back().op=='A');
    assert(out==events.back().y);
    assert(count==uint64_t(events.back().rows)*events.back().groups*1024);
    events.push_back({'R',0,0,0,0,0,0,0,nullptr,nullptr,out,nullptr,false});
}
static int cudaGetLastError() { return bf16_error ? 1 : 0; }
static int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out,const void *,uint64_t,
        uint64_t,uint64_t width,uint64_t outputs,const ds4_gpu_tensor *in,uint64_t rows) {
    assert(rows==1 && width==8192 && outputs==5120);
    assert(in->bytes==width*4 && out->bytes==outputs*4);
    assert(!events.empty() && (events.back().op=='R' || events.back().op=='S' ||
        (events.back().op=='A' && events.back().bf16)));
    events.push_back({'S',8,uint32_t(outputs),uint32_t(width),8192,0,1,1,
                      nullptr,in->ptr,out->ptr,nullptr,false});
    if (grow_in_scalar && ++scalar_calls==1) arena=arena1.data();
    return fail_stage!='S';
}
static int ds4_gpu_dsv41_attention_output_batch(ds4_gpu_tensor *,ds4_gpu_tensor *,
        const void *,uint64_t,uint64_t,uint64_t,const ds4_gpu_tensor *,uint32_t rows) {
    events.push_back({'L',8,0,0,0,0,1,rows,nullptr,nullptr,nullptr,nullptr,false});
    return fail_stage!='L';
}
static int ds4_gpu_dsv41_attention_output_tp_batch(ds4_gpu_tensor *,ds4_gpu_tensor *,
        const void *,uint64_t,uint64_t,uint64_t,const ds4_gpu_tensor *,uint32_t rows,uint32_t rank) {
    events.push_back({'L',8,0,0,0,rank,2,rows,nullptr,nullptr,nullptr,nullptr,false});
    return fail_stage!='L';
}
'''

CHECKS = r'''
int main() {
    std::vector<uint32_t> model(32*1024*1024);
    std::vector<float> hs(65*32768), ls(65*8192), os(65*5120);
    const uint64_t size=model.size()*4, a=4096, b=64*1024*1024;
    ds4_gpu_tensor h{},l{},o{};
    auto reset=[&](uint32_t world=1,uint32_t rank=0) {
        events.clear(); requests.clear(); arena=arena0.data();
        scratch_override=nullptr; scratch_null=false; grow_in_scalar=false;
        allocations=0; scalar_calls=0; fail_resolve=0; misalign_resolve=0;
        fail_stage=0; bf16_error=false; set_device_fail=0; device_query_fail=0;
        capture_status=cudaStreamCaptureStatusNone; capture_query_fail=0; mmq_queries=0;
        g_n_gpus=world; g_cuda_test_q4_mmq_strict=0;
        current_device=g_gpu[rank].device_id; mmq_enabled=world==1;
        mmq_changes_device=1;
        h={hs.data(),hs.size()*4,int(rank)};
        l={ls.data(),ls.size()*4,int(rank)};
        o={os.data(),os.size()*4,int(rank)};
    };
    auto call=[&](uint32_t at=12,uint32_t bt=12,uint32_t world=1,uint32_t rank=0,
                  uint32_t rows=65,uint64_t ao=4096,uint64_t bo=64*1024*1024) {
        return ds4_gpu_dsv41_attention_output_typed_batch(&o,&l,model.data(),size,
            ao,bo,at,bt,&h,rows,world,rank);
    };
    for (uint32_t world : {1u,2u}) for (uint32_t rank=0;rank<world;++rank)
        for (uint32_t at : {8u,12u}) for (uint32_t bt : {8u,12u}) {
            reset(world,rank);
            grow_in_scalar=world==1 && at==12 && bt==8;
            assert(call(at,bt,world,rank)==1);
            assert(requests.size()==2);
            const uint64_t ar=at==8 ? 4352 : 2304, br=bt==8 ? 8704 : 4608;
            const uint32_t groups=8/world, low=groups*1024, heads=groups*4096;
            assert(requests[0].offset==a+rank*uint64_t(low)*ar);
            assert(requests[0].bytes==uint64_t(low)*ar && requests[0].tier==int(rank));
            assert(requests[1].offset==b && requests[1].bytes==5120*br);
            if (at==8 && bt==8) {
                assert(events.size()==1 && events[0].op=='L' && events[0].rows==65);
                assert(events[0].groups==world && events[0].k0==rank);
                continue;
            }
            assert(current_device==g_gpu[rank].device_id && allocations==2);
            assert(last_scratch_bytes==uint64_t(64)*groups*16*292);
            size_t event=0;
            for (uint32_t first : {0u,64u}) {
                const uint32_t count=first==0 ? 64 : 1;
                const auto &ae=events.at(event++);
                assert(ae.op=='A' && ae.type==at && ae.rows==count && ae.groups==groups);
                assert(ae.outputs==1024 && ae.width==4096 && ae.full_width==4096 && ae.k0==0);
                assert(ae.w==(char *)model.data()+requests[0].offset);
                assert(ae.x==hs.data()+uint64_t(first)*heads);
                assert(ae.y==ls.data()+uint64_t(first)*low);
                assert(ae.scratch==(grow_in_scalar && first ? arena1.data() : arena0.data()));
                if (at==8) assert(events.at(event++).op=='R');
                else assert(ae.bf16);
                if (world==1 && bt==8) {
                    for (uint32_t row=0;row<count;++row) {
                        const auto &se=events.at(event++);
                        assert(se.op=='S');
                        assert(se.x==ls.data()+uint64_t(first+row)*low);
                        assert(se.y==os.data()+uint64_t(first+row)*5120);
                    }
                } else {
                    const auto &be=events.at(event++);
                    assert(be.op=='B' && be.type==bt && be.rows==count && be.groups==1);
                    assert(!be.bf16);
                    assert(be.outputs==5120 && be.width==low && be.full_width==8192);
                    assert(be.k0==rank*low && be.w==(char *)model.data()+b);
                    assert(be.x==ls.data()+uint64_t(first)*low);
                    assert(be.y==os.data()+uint64_t(first)*5120);
                }
            }
            assert(event==events.size());
        }
    auto rejected=[&] { assert(call()==0 && events.empty()); };
    reset(); h.bytes=1; rejected();
    reset(); l.ptr=(char *)l.ptr+1; rejected();
    reset(); o.ptr=l.ptr; rejected();
    reset(); h.ptr=(char *)model.data()+a; rejected();
    reset(); h.ptr=nullptr; rejected();
    reset(); h.device_id=-2; rejected();
    reset(); h.device_id=1; rejected();
    reset(); current_device=0; rejected();
    reset(); device_query_fail=1; rejected();
    reset(); set_device_fail=1; rejected();
    for (auto status : {cudaStreamCaptureStatusActive,cudaStreamCaptureStatusInvalidated}) {
        reset(); capture_status=status; rejected();
        assert(requests.empty() && allocations==0 && mmq_queries==0);
    }
    reset(); capture_query_fail=1; rejected();
    assert(requests.empty() && allocations==0 && mmq_queries==0);
    reset(); fail_resolve=1; rejected();
    reset(); fail_resolve=2; rejected();
    reset(); misalign_resolve=1; rejected();
    reset(); misalign_resolve=2; rejected();
    reset(); scratch_null=true; rejected();
    reset(); scratch_override=(char *)arena+1; rejected();
    reset(); scratch_override=l.ptr; rejected();
    reset(); scratch_override=(char *)model.data()+a; rejected();
    reset(); mmq_enabled=0; g_cuda_test_q4_mmq_strict=1; rejected();
    reset(); assert(!call(12,12,1,0,0) && events.empty());
    reset(); assert(!call(12,12,3,0) && events.empty());
    reset(); assert(!call(12,12,1,1) && events.empty());
    reset(); assert(!call(1,12) && events.empty());
    reset(); assert(!call(12,1) && events.empty());
    reset(); assert(!call(12,12,1,0,65,size, b) && events.empty());
    reset(); assert(!call(12,12,1,0,65,a, size) && events.empty());
    for (char failure : {'A','B','S'}) {
        reset(); fail_stage=failure;
        assert(!call(12,failure=='S' ? 8 : 12));
        assert(events.front().op=='A' && events.back().op==failure);
        assert(events.size()==(failure=='A' ? 1u : 2u));
    }
    reset(); bf16_error=true;
    assert(!call(8,12) && events.size()==2 && events.back().op=='R');
    reset(); bf16_error=true;
    assert(call(12,12)); // Q4-A owns its rounding; no standalone launch.
    puts("CUDA V4.1 Q4 dispatcher host: preflight, TP, chunks, BF16 order, scratch and device PASS");
}
'''


def main():
    source = (ROOT / "ds4_deepseek41_cuda.cuh").read_text()
    helpers = "\n".join(extract_function(source, signature) for signature in (
        "static bool dsv41_has_floats(",
        "static bool dsv41_output_range(",
        "static bool dsv41_output_overlap(",
    ))
    wrapper = extract_function(source, 'extern "C" int ds4_gpu_dsv41_attention_output_typed_batch(')
    wrapper, count = re.subn(r"dsv41_bf16_kernel<<<.*?>>>", "dsv41_bf16_kernel", wrapper, flags=re.S)
    assert count == 1 and "<<<" not in wrapper, "only the BF16 launch may be instrumented"
    with tempfile.TemporaryDirectory(prefix="ds4-v41-q4-dispatch-") as directory:
        src = Path(directory) / "check.cpp"
        binary = Path(directory) / "check"
        src.write_text(HOST + helpers + wrapper + CHECKS)
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        subprocess.run(compiler + ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
            str(src), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
