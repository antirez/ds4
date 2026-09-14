"""Test the production HC4096 split kernels and launch policy.

Host mode stages CUDA barriers to check the actual arithmetic bodies, then
compiles the actual dispatcher with allocation/launch fault injection. It runs
strict source arithmetic and explicit CUDA-style FMA under fast host math,
both with ASan/UBSan. This does not simulate CUDA races, FTZ, or device math.
--cuda runs the unmodified kernels against the fused production reference on
the selected GPU; --emit-cuda writes that oracle.
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
SOURCE = (ROOT / "ds4_cuda.cu").read_text()
PREFIX = "hc_split_weighted_sum_norm_fused_"
NAMES = [PREFIX + "kernel"] + [PREFIX + p + "4096_kernel"
                                     for p in ("partial", "reduce", "store")]
BODIES = [extract_function(SOURCE, "__global__ static void " + n + "(")
          for n in NAMES]
SPLIT = extract_function(SOURCE, "__device__ static void hc4_split_one(")

COMMON = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
static void check(bool ok, const char *message) {
    if (!ok) { fprintf(stderr, "FAIL HC4096: %s\n", message); exit(1); }
}
enum { WIDTH = 4096, GUARD = 16, CASES = 128 };
static const float sentinel = 123456.0f;
struct inputs {
    std::vector<float> mix, residual, scale, base, norm_w;
    inputs(unsigned seed) : mix(24), residual(4*WIDTH), scale(3),
                           base(24), norm_w(WIDTH) {
        uint32_t state = seed + 17;
        auto word = [&]() { state ^= state << 13; state ^= state >> 17;
                           state ^= state << 5; return state; };
        for (float &v : mix) v = (int(word()%257)-128) / 16.0f;
        for (float &v : base) v = (int(word()%65)-32) / 32.0f;
        for (float &v : scale) v = (int(word()%17)-8) / 8.0f;
        for (unsigned i=0; i<residual.size(); ++i) {
            residual[i] = ldexpf(float(int(word()%1025)-512), int(i%17)-12);
            if (seed%8 == 0) residual[i] = 0.0f;
            if (seed%8 == 1) residual[i] = i%2 ? 1e-10f : -1e-10f;
            if (seed%8 == 2) residual[i] = i%2 ? 1e10f : -1e10f;
        }
        for (float &v : norm_w) v = (int(word()%129)-64) / 32.0f;
    }
};
struct outputs {
    std::vector<float> out, norm, split, scale;
    outputs() : out(WIDTH+2*GUARD,sentinel), norm(WIDTH+2*GUARD,sentinel),
                split(24+2*GUARD,sentinel), scale(1+2*GUARD,sentinel) {}
};
static void guards(const std::vector<float> &v) {
    for (unsigned i=0; i<GUARD; ++i)
        check(v[i]==sentinel && v[v.size()-1-i]==sentinel, "output guards");
}
static void compare(const outputs &ref, const outputs &got) {
    for (const auto *v : {&got.out,&got.norm,&got.split,&got.scale}) guards(*v);
    check(!memcmp(ref.out.data(),got.out.data(),ref.out.size()*4), "weighted sum bits");
    check(!memcmp(ref.norm.data(),got.norm.data(),ref.norm.size()*4), "normalized output bits");
    check(!memcmp(ref.split.data(),got.split.data(),ref.split.size()*4), "Sinkhorn split bits");
}
'''

HOST = r'''
#include <setjmp.h>
static struct { unsigned x; } threadIdx, blockIdx, blockDim = {256};
static bool collecting;
static float leaves[256], total;
static jmp_buf collected;
static float rsqrtf(float x) { return 1.0f / sqrtf(x); }
static float reduce_stage(float sum) {
    if (collecting) { leaves[threadIdx.x] = sum; longjmp(collected, 1); }
    return total;
}
static void finish_reduce() {
    // Lockstep equivalent of the production shared-memory tree.
    for (unsigned stride=128; stride; stride>>=1)
        for (unsigned lane=0; lane<stride; ++lane) {
            volatile float value = leaves[lane] + leaves[lane+stride];
            leaves[lane] = value;
        }
    total = leaves[0];
}
#define __syncthreads() ((void)0)
'''


def host_body(body):
    if "__shared__ float partial[256];" in body:
        if "float sum = 0.0f;" in body[body.index("__shared__ float partial[256];"):]:
            body = body.replace("    __shared__ float partial[256];\n", "")
            start = body.index("    partial[d] = sum;")
        else:
            start = body.index("    __shared__ float partial[256];")
        end = body.index("    if (d == 0u)", start) if "reduce4096" in body else body.index("    const float norm_scale", start)
        tree = body[start:end]
        assert "partial[d] += partial[d + stride]" in tree
        assert "stride = blockDim.x >> 1" in tree
        body = body[:start] + "    const float reduced = reduce_stage(sum);\n" + body[end:]
        body = body.replace("partial[0]", "reduced")
    return body.replace("__global__ ", "").replace("__shared__", "static")


HOST_CASES = r'''
static void run_case(unsigned seed) {
    inputs in(seed); const inputs original = in;
    outputs ref, got;
    const unsigned iterations[] = {0,1,2,20};
    const unsigned iters = iterations[seed%4];
    const float eps = seed&1 ? 1e-6f : 1e-5f;
    auto baseline = [&]() {
        hc_split_weighted_sum_norm_fused_kernel(ref.out.data()+GUARD,
            ref.norm.data()+GUARD,ref.split.data()+GUARD,in.mix.data(),
            in.residual.data(),in.scale.data(),in.base.data(),in.norm_w.data(),
            WIDTH,4,1,iters,eps,eps);
    };
    blockIdx.x=0; collecting=true;
    for (threadIdx.x=0; threadIdx.x<256; ++threadIdx.x)
        if (!setjmp(collected)) baseline();
    finish_reduce(); collecting=false;
    for (threadIdx.x=0; threadIdx.x<256; ++threadIdx.x) baseline();
    for (unsigned order=0; order<16; ++order) {
        blockIdx.x = seed&1 ? 15-order : order;
        // The first barrier publishes sp from lane zero to the whole CTA.
        for (threadIdx.x=0; threadIdx.x<256; ++threadIdx.x)
            hc_split_weighted_sum_norm_fused_partial4096_kernel(
                got.out.data()+GUARD,got.split.data()+GUARD,in.mix.data(),
                in.residual.data(),in.scale.data(),in.base.data(),iters,eps);
    }
    auto reduce = [&]() { hc_split_weighted_sum_norm_fused_reduce4096_kernel(
        got.out.data()+GUARD,got.scale.data()+GUARD,eps); };
    blockIdx.x=0; collecting=true;
    for (threadIdx.x=0; threadIdx.x<256; ++threadIdx.x)
        if (!setjmp(collected)) reduce();
    finish_reduce(); collecting=false;
    for (threadIdx.x=0; threadIdx.x<256; ++threadIdx.x) reduce();
    for (blockIdx.x=0; blockIdx.x<16; ++blockIdx.x)
        for (threadIdx.x=0; threadIdx.x<256; ++threadIdx.x)
            hc_split_weighted_sum_norm_fused_store4096_kernel(
                got.out.data()+GUARD,got.norm.data()+GUARD,in.norm_w.data(),
                got.scale.data()+GUARD);
    compare(ref,got);
    for (auto pair : {std::make_pair(&in.mix,&original.mix),
                      std::make_pair(&in.residual,&original.residual),
                      std::make_pair(&in.norm_w,&original.norm_w)})
        check(*pair.first==*pair.second,"inputs unchanged");
}
'''

POLICY = r'''
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; int device_id; };
typedef int cudaStream_t;
static int g_n_gpus=1, g_decode_graph_capturing=0;
static void *g_cuda_tmp;
static uint64_t g_cuda_tmp_bytes;
static struct { void *scratch; uint64_t scratch_bytes; } g_gpu[2];
static int launches, allocations, fallbacks, fail_launch, launch_error, alloc_fail;
static std::vector<unsigned> grids;
static int cuda_decode_stream() { return g_decode_graph_capturing ? 7 : 0; }
static int ds4_tensor_device_idx(const ds4_gpu_tensor *t) { return t->device_id; }
static int cudaGetLastError() { int rc=launch_error; launch_error=0; return rc; }
static int cuda_ok(int error, const char *) { return error==0; }
static void *cuda_tmp_alloc_on(int tier, uint64_t bytes, const char *) {
    check(launches==0,"scratch before writer");
    check(bytes==sizeof(float),"scratch size"); ++allocations;
    return alloc_fail ? nullptr : g_n_gpus==1 ? g_cuda_tmp : g_gpu[tier].scratch;
}
static const char *cuda_resolve_weight_ptr(const void *model,uint64_t off,
        uint64_t,int,const char *) { return (const char *)model+off; }
template<class... Args> static int ds4_gpu_hc_split_weighted_sum_tensor(Args...) {
    ++fallbacks; return 1;
}
template<class... Args> static int ds4_gpu_rms_norm_weight_rows_tensor(Args...) {
    ++fallbacks; return 1;
}
template<class... Args> static void launch_record(unsigned grid,unsigned threads,
        unsigned shared,cudaStream_t stream,Args...) {
    check(threads==256 && shared==0,"launch geometry");
    check(stream==cuda_decode_stream(),"capture stream");
    grids.push_back(grid); ++launches;
    launch_error = launches==fail_launch;
}
'''


def policy_source():
    helper = extract_function(SOURCE, "static bool cuda_hc_split4096_disjoint(")
    wrapper = extract_function(SOURCE, 'extern "C" int ds4_gpu_hc_split_weighted_sum_norm_tensor(')
    wrapper = re.sub(r"\b(?:" + "|".join(NAMES) + r")<<<(.*?)>>>\(",
                     r"launch_record(\1,", wrapper, flags=re.S)
    assert "<<<" not in wrapper
    return POLICY + helper + "\n" + wrapper + POLICY_CASES


POLICY_CASES = r'''
static void run_policy_cases() {
    std::vector<float> o(2*WIDTH), n(2*WIDTH), s(48), m(48), r(8*WIDTH);
    std::vector<float> model(27+WIDTH); float scratch=0;
    ds4_gpu_tensor out{o.data(),WIDTH*4,0}, norm{n.data(),WIDTH*4,0},
        split{s.data(),96,0}, mix{m.data(),96,0}, residual{r.data(),4*WIDTH*4,0};
    auto reset = [&]() {
        launches=allocations=fallbacks=fail_launch=launch_error=alloc_fail=0;
        grids.clear(); g_n_gpus=1; g_decode_graph_capturing=0;
        g_cuda_tmp=&scratch; g_cuda_tmp_bytes=4;
        out={o.data(),WIDTH*4,0}; norm={n.data(),WIDTH*4,0};
        split={s.data(),96,0}; mix={m.data(),96,0}; residual={r.data(),4*WIDTH*4,0};
        unsetenv("DS4_CUDA_NO_HC_SPLIT_NORM_SPLIT4096");
        unsetenv("DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED");
    };
    auto call = [&](unsigned width=WIDTH) {
        return ds4_gpu_hc_split_weighted_sum_norm_tensor(&out,&norm,&split,&mix,
            &residual,model.data(),model.size()*4,0,12,108,width,4,20,1e-6f,1e-6f);
    };
    reset(); check(call()==1 && grids==std::vector<unsigned>({16,1,16}),"default split4096");
    for (int fail=1; fail<=3; ++fail) {
        reset(); fail_launch=fail;
        check(call()==0 && launches==fail && fallbacks==0,"no replay after writer failure");
    }
    for (const char *value : {"","0","1"}) {
        reset(); setenv("DS4_CUDA_NO_HC_SPLIT_NORM_SPLIT4096",value,1);
        check(call()==1 && launches==1 && allocations==0,"presence rollback");
    }
    reset(); alloc_fail=1;
    check(call()==1 && launches==1 && grids[0]==1,"pre-writer allocation fallback");
    reset(); g_decode_graph_capturing=1;
    check(call()==1 && launches==3,"capture reuses warm scratch");
    reset(); g_decode_graph_capturing=1; g_cuda_tmp=nullptr; g_cuda_tmp_bytes=0;
    check(call()==1 && launches==1 && allocations==0,"cold capture does not allocate");
    reset(); norm.ptr=out.ptr;
    check(call()==1 && launches==1,"in-place norm retains reference");
    reset(); norm.ptr=(float *)out.ptr+1;
    check(call()==1 && launches==1,"partial overlap retains reference");
    reset(); split.ptr=mix.ptr;
    check(call()==1 && launches==1,"split/mix alias retains reference");
    reset(); g_cuda_tmp=out.ptr;
    check(call()==1 && launches==1,"scratch/output alias retains reference");
    for (auto *t : {&norm,&split,&mix,&residual}) {
        reset(); t->device_id=1;
        check(call()==0 && launches==0 && allocations==0,"cross-device inputs rejected");
    }
    reset(); g_n_gpus=2; out.device_id=norm.device_id=split.device_id=mix.device_id=residual.device_id=1;
    g_gpu[1].scratch=&scratch; g_gpu[1].scratch_bytes=4;
    check(call()==1 && launches==3,"tier-local scratch");
    reset(); mix.ptr=nullptr;
    check(call()==0 && launches==0,"null input rejected");
    reset(); norm.bytes-=4;
    check(call()==0 && launches==0,"short output rejected");
    reset(); out.bytes=norm.bytes=1024*4;
    check(call(1024)==1 && launches==1 && allocations==0,"other width retains reference");
    reset(); out.bytes=norm.bytes=2*WIDTH*4; mix.bytes=split.bytes=192; residual.bytes=8*WIDTH*4;
    check(call()==1 && launches==0 && fallbacks==2,"multi-row fallback preserved");
    reset(); setenv("DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED","1",1);
    check(call()==1 && launches==0 && fallbacks==2,"global rollback preserved");
    reset();
    puts("PASS: production dispatcher rollback, alias/device guards, capture, allocation and launch faults");
}
int main() {
    for (unsigned i=0; i<CASES; ++i) run_case(i);
    run_policy_cases();
    printf("PASS: %u production HC4096 arithmetic cases, bitwise fused/split parity and guards\n",CASES);
}
'''

NATIVE = r'''
#define CUDA_CHECK(call) do { cudaError_t err=(call); if (err!=cudaSuccess) { \
    fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(err)); exit(1); } } while(0)
static float *upload(const std::vector<float> &v) {
    float *p=nullptr; CUDA_CHECK(cudaMalloc(&p,v.size()*4));
    CUDA_CHECK(cudaMemcpy(p,v.data(),v.size()*4,cudaMemcpyHostToDevice)); return p;
}
static void download(std::vector<float> &v,const float *p) {
    CUDA_CHECK(cudaMemcpy(v.data(),p,v.size()*4,cudaMemcpyDeviceToHost));
}
static void run_case(unsigned seed) {
    inputs in(seed); outputs ref,got;
    float *mix=upload(in.mix),*residual=upload(in.residual),*scale=upload(in.scale),
          *base=upload(in.base),*weight=upload(in.norm_w),*out=upload(got.out),
          *norm=upload(got.norm),*split=upload(got.split),*norm_scale=upload(got.scale);
    const unsigned iterations[]={0,1,2,20};
    const float eps=seed&1 ? 1e-6f : 1e-5f;
    hc_split_weighted_sum_norm_fused_kernel<<<1,256>>>(out+GUARD,norm+GUARD,
        split+GUARD,mix,residual,scale,base,weight,WIDTH,4,1,iterations[seed%4],eps,eps);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    download(ref.out,out); download(ref.norm,norm); download(ref.split,split);
    CUDA_CHECK(cudaMemcpy(out,got.out.data(),got.out.size()*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(norm,got.norm.data(),got.norm.size()*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(split,got.split.data(),got.split.size()*4,cudaMemcpyHostToDevice));
    // Non-default stream and repeated graph replay exercise scratch lifetime
    // and ordering of the reduction/store across the three real kernels.
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
    auto launch = [&]() {
        hc_split_weighted_sum_norm_fused_partial4096_kernel<<<16,256,0,stream>>>(
            out+GUARD,split+GUARD,mix,residual,scale,base,iterations[seed%4],eps);
        CUDA_CHECK(cudaGetLastError());
        hc_split_weighted_sum_norm_fused_reduce4096_kernel<<<1,256,0,stream>>>(out+GUARD,norm_scale+GUARD,eps);
        CUDA_CHECK(cudaGetLastError());
        hc_split_weighted_sum_norm_fused_store4096_kernel<<<16,256,0,stream>>>(out+GUARD,norm+GUARD,weight,norm_scale+GUARD);
        CUDA_CHECK(cudaGetLastError());
    };
    // Resolve lazy module loading before capture, then clear every writer's
    // destination so replay cannot pass using warmup results.
    launch(); CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(out,got.out.data(),got.out.size()*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(norm,got.norm.data(),got.norm.size()*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(split,got.split.data(),got.split.size()*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(norm_scale,got.scale.data(),got.scale.size()*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    launch();
    cudaGraph_t graph; cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for (unsigned i=0;i<3;++i) CUDA_CHECK(cudaGraphLaunch(exec,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    download(got.out,out); download(got.norm,norm); download(got.split,split); download(got.scale,norm_scale);
    compare(ref,got);
    inputs after=in; download(after.mix,mix); download(after.residual,residual); download(after.norm_w,weight);
    check(in.mix==after.mix && in.residual==after.residual && in.norm_w==after.norm_w,"native inputs unchanged");
    CUDA_CHECK(cudaGraphExecDestroy(exec)); CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (float *p : {mix,residual,scale,base,weight,out,norm,split,norm_scale}) CUDA_CHECK(cudaFree(p));
}
int main(int argc,char **argv) {
    int count=0; CUDA_CHECK(cudaGetDeviceCount(&count)); check(count>0,"visible CUDA device required");
    const int device=argc>1 ? atoi(argv[1]) : 0; check(device>=0 && device<count,"device ordinal");
    CUDA_CHECK(cudaSetDevice(device));
    for (unsigned i=0;i<CASES;++i) run_case(i);
    printf("PASS CUDA device %d: %u HC4096 fused/split bitwise cases, guards and graph replay\n",device,CASES);
}
'''


def native_source():
    return "#include <cuda_runtime.h>\n" + COMMON + SPLIT + "\n" + "\n".join(BODIES) + NATIVE


def host_source(fma):
    split = SPLIT.replace("__device__ ", "")
    bodies = "\n".join(host_body(b) for b in BODIES)
    if fma:
        # Host compilers can contract the reference's register expression
        # differently from the split kernel's reload, even without reduction
        # reassociation. Explicit FMA models the CUDA rounding DAG; this is
        # deliberately not a claim about the PTX chosen by an untested NVCC.
        bodies, count = re.subn(
            r"acc \+= (residual_hc\[[^;]+?\]) \* sp\[h\];",
            r"acc = fmaf(\1, sp[h], acc);", bodies)
        assert count == 2
        assert bodies.count("sum += acc * acc;") == 1
        assert bodies.count("sum += out[col] * out[col];") == 1
        bodies = bodies.replace("sum += acc * acc;", "sum = fmaf(acc, acc, sum);")
        bodies = bodies.replace("sum += out[col] * out[col];", "sum = fmaf(out[col], out[col], sum);")
        # Give both host paths the same compiled Sinkhorn arithmetic instead
        # of letting host-only global/shared stand-ins affect specialization.
        split = split.replace("static void", "__attribute__((noinline)) static void", 1)
    return COMMON + HOST + split + "\n" + bodies + HOST_CASES + policy_source()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-cuda", type=Path)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.emit_cuda:
        args.emit_cuda.write_text(native_source())
        print(f"Wrote native oracle to {args.emit_cuda}; no CUDA execution")
        return
    with tempfile.TemporaryDirectory(prefix="ds4-hc4096-") as tmp:
        if args.cuda:
            compiler = shlex.split(os.environ.get("NVCC", "nvcc"))
            if not compiler or not shutil.which(compiler[0]):
                parser.error("NVCC is required for --cuda; use --emit-cuda for source inspection")
            source, binary = Path(tmp)/"hc.cu", Path(tmp)/"hc"
            source.write_text(native_source())
            flags = shlex.split(os.environ.get("NVCCFLAGS", "-O3 --use_fast_math"))
            subprocess.run(compiler + flags + ["-std=c++17", str(source), "-o", str(binary)],check=True)
            subprocess.run([str(binary),str(args.device)],check=True)
        else:
            source, binary = Path(tmp)/"hc.cpp", Path(tmp)/"hc"
            compiler = shlex.split(os.environ.get("CXX", "c++"))
            # NVCC --use_fast_math enables FMA/approximate device math, not
            # arbitrary reduction reassociation or a ban on -INFINITY. Keep
            # those host-only assumptions off when testing the fixed DAG.
            for fma, flags in ((False, ["-O2"]),
                               (True, ["-O3", "-ffast-math", "-fno-associative-math",
                                       "-fno-finite-math-only"])):
                print("Testing host " + ("explicit CUDA FMA / fast math" if fma
                                         else "strict source arithmetic"), flush=True)
                source.write_text(host_source(fma))
                subprocess.run(compiler + flags + ["-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-Wno-unknown-pragmas", "-fsanitize=address,undefined", str(source), "-o", str(binary)],check=True)
                subprocess.run([str(binary)],check=True)


if __name__ == "__main__":
    main()
