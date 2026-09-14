#!/usr/bin/env python3
"""Host arithmetic oracle for real CUDA/ROCm HC norm/mix preparation.

Stage the actual 256-lane reduction tree and 32-lane ordered projection. Run
strict arithmetic and explicit GPU-style FMA with fast math, both ASan/UBSan.
This is not a native GPU test: it cannot validate scheduling, FTZ or cuBLAS.
The separate test_gpu_hc_norm_mix_native.cpp exercises the real backend APIs.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from kernel_source import extract_function

ROOT=Path(__file__).resolve().parents[1]

POLICY_MOCKS=r'''
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; int device_id; };
static bool g_quality_mode, supported;
static int g_decode_graph_capturing, allocations, launches, blas_calls, pending_error;
static int fail_norm, fail_project, fail_blas, fail_alloc;
static void *g_cuda_tmp,*allocation_result;
static uint64_t g_cuda_tmp_bytes;
static const char *resolved_override;
static bool override_weight;
static thread_local ds4_gpu_execution_phase phase_state=DS4_GPU_PHASE_AUTO;
extern "C" ds4_gpu_execution_phase ds4_gpu_get_execution_phase(){return phase_state;}
extern "C" ds4_gpu_execution_phase ds4_gpu_exchange_execution_phase(ds4_gpu_execution_phase p){auto old=phase_state;phase_state=p;return old;}
static int ds4_gpu_hc_rms_norm_mix_f16_available(){return supported;}
static int rocm_hc_norm_mix_device_supported(){return supported;}
static int ds4_tensor_device_idx(const ds4_gpu_tensor *t){return t->device_id;}
static int cuda_decode_stream(){return g_decode_graph_capturing?7:0;}
static int cudaGetLastError(){int result=pending_error;pending_error=0;return result;}
static bool cuda_ok(int status,const char *){return status==0;}
static bool cublas_ok(int status,const char *){return status==0;}
static const char *cuda_model_range_ptr(const void *model,uint64_t off,uint64_t,const char *){
    return override_weight?resolved_override:static_cast<const char *>(model)+off;
}
static const char *cuda_resolve_weight_ptr(const void *model,uint64_t off,uint64_t bytes,int tier,const char *label){
    check(tier==0,"weight tier");return cuda_model_range_ptr(model,off,bytes,label);
}
static void *cuda_tmp_alloc(uint64_t count,const char *){
    check(launches==0,"scratch preparation precedes writers");++allocations;
    if(fail_alloc)return nullptr;
    g_cuda_tmp=allocation_result;g_cuda_tmp_bytes=count;return allocation_result;
}
static void *cuda_tmp_alloc_on(int tier,uint64_t count,const char *label){
    check(tier==0&&count==ds4_hc_norm_mix::half_bytes,"CUDA scratch shape");
    return cuda_tmp_alloc(count,label);
}
static void cuda_normalize_launch(unsigned grid,unsigned threads,unsigned shared,int stream,
        __half *,const float *,uint32_t width,uint32_t rows,float){
    check(grid==1&&threads==256&&shared==0&&stream==cuda_decode_stream()&&width==16384&&rows==1,"CUDA norm launch");
    ++launches;pending_error=fail_norm;
}
static void rocm_scale_launch(unsigned grid,unsigned threads,float *,const float *,uint32_t width,float){
    check(grid==1&&threads==256&&width==16384,"ROCm norm launch");++launches;pending_error=fail_norm;
}
static void rocm_project_launch(unsigned grid,unsigned threads,float *,const __half *,const float *,const float *,uint64_t n,uint64_t m){
    check(grid==24&&threads==32&&n==16384&&m==24,"ROCm ordered launch");++launches;pending_error=fail_project;
}
using cublasStatus_t=int;
enum { CUBLAS_OP_T=1,CUBLAS_OP_N=2,CUDA_R_16F=3,CUDA_R_32F=4,CUBLAS_GEMM_DEFAULT=5 };
static int cuda_cublas_for_tier(int tier){check(tier==0,"BLAS tier");return 19;}
static cublasStatus_t cublasGemmEx(int handle,int op_a,int op_b,int m,int n,int k,const float *alpha,
        const __half *,int a_type,int lda,const __half *,int b_type,int ldb,const float *beta,
        void *,int c_type,int ldc,int compute,int algorithm){
    check(handle==19&&op_a==CUBLAS_OP_T&&op_b==CUBLAS_OP_N&&m==24&&n==1&&k==16384,
          "original cuBLAS transpose/dimensions/handle");
    check(*alpha==1&&*beta==0&&a_type==CUDA_R_16F&&b_type==CUDA_R_16F&&c_type==CUDA_R_32F&&
          lda==16384&&ldb==16384&&ldc==24&&compute==CUDA_R_32F&&algorithm==CUBLAS_GEMM_DEFAULT,
          "original cuBLAS types/scalars/strides/compute");
    check(launches==1,"BLAS after successful producer");++blas_calls;return fail_blas;
}
'''

POLICY_CASES=r'''
static void wrapper_cases(){
    namespace p=ds4_hc_norm_mix;
    std::vector<float> xv(16384),ov(24),scratch(16384);
    std::vector<uint16_t> model(p::weight_bytes/2+8);
    ds4_gpu_tensor x{xv.data(),p::x_bytes,0},out{ov.data(),p::out_bytes,0};
    unsigned cases=0;
    for(bool hip:{false,true}){
        auto reset=[&](){
            supported=true;g_quality_mode=false;g_decode_graph_capturing=0;phase_state=DS4_GPU_PHASE_DECODE;
            allocations=launches=blas_calls=pending_error=fail_norm=fail_project=fail_blas=fail_alloc=0;
            g_cuda_tmp=nullptr;g_cuda_tmp_bytes=0;allocation_result=scratch.data();override_weight=false;
            x={xv.data(),p::x_bytes,0};out={ov.data(),p::out_bytes,0};
        };
        auto call=[&](float eps=1e-6f){return hip?
            hc_norm_mix_rocm_test(&out,&x,model.data(),model.size()*2,16,16384,24,eps):
            hc_norm_mix_cuda_test(&out,&x,model.data(),model.size()*2,16,16384,24,eps);};
        auto declined=[&](){check(call()==0&&launches==0&&blas_calls==0,"pre-writer decline");++cases;};
        reset();check(call()==1&&allocations==1&&launches==(hip?2:1)&&blas_calls==(hip?0:1),"actual wrapper success");++cases;
        reset();fail_norm=1;check(call()==-1&&launches==1&&blas_calls==0,"producer error is -1, no consumer");++cases;
        reset();if(hip)fail_project=1;else fail_blas=1;
        check(call()==-1&&launches==(hip?2:1)&&blas_calls==(hip?0:1),"consumer error is -1 without fallback");++cases;
        reset();fail_alloc=1;declined();check(allocations==1,"allocation failure reached");
        reset();supported=false;declined();check(allocations==0,"unsupported device is preflight");
        for(auto phase:{DS4_GPU_PHASE_PREFILL,DS4_GPU_PHASE_VERIFY,DS4_GPU_PHASE_BATCH_DECODE,DS4_GPU_PHASE_MIXED}){
            reset();phase_state=phase;declined();check(allocations==0,"phase rejected before scratch");
        }
        reset();phase_state=DS4_GPU_PHASE_AUTO;check(call()==1,"AUTO direct API compatible");++cases;
        reset();g_quality_mode=true;declined();
        reset();out.bytes--;declined();reset();x.bytes--;declined();
        reset();out.ptr=nullptr;declined();reset();x.ptr=nullptr;declined();
        for(float eps:{0.f,-1.f,INFINITY,-INFINITY,NAN}){
            reset();check(call(eps)==0&&allocations==0&&launches==0,"invalid epsilon preflight");++cases;
        }
        reset();out.ptr=x.ptr;declined();check(allocations==0,"input alias before scratch");
        reset();override_weight=true;resolved_override=nullptr;declined();
        reset();override_weight=true;resolved_override=reinterpret_cast<const char *>(out.ptr);declined();
        for(void *alias:{out.ptr,x.ptr,static_cast<void *>(model.data()+8)}){
            reset();g_cuda_tmp=alias;g_cuda_tmp_bytes=4;declined();
            check(allocations==0,"old scratch alias rejected before growth/free");
            reset();allocation_result=alias;declined();check(allocations==1,"new scratch alias pre-writer");
        }
        if(!hip){
            reset();x.device_id=1;declined();reset();out.device_id=1;declined();
            reset();g_decode_graph_capturing=1;declined();check(allocations==0,"cold graph capture does not allocate");
            reset();g_decode_graph_capturing=1;g_cuda_tmp=scratch.data();g_cuda_tmp_bytes=p::half_bytes;
            check(call()==1&&launches==1&&blas_calls==1,"warm graph capture uses capture stream");++cases;
        }
    }
    std::printf("PASS: %u actual CUDA/ROCm wrapper cases: dispatch, unchanged BLAS arguments, capture, aliases, bounds, allocation/producer/consumer faults and tri-state results.\n",cases);
}
'''

def wrapper_source(cuda):
    rocm=(ROOT/'rocm/ds4_rocm_current_api_compat.cuh').read_text()
    parts=[]
    for source,backend in ((cuda,'cuda'),(rocm,'rocm')):
        wrapper=extract_function(source,'extern "C" int ds4_gpu_hc_rms_norm_mix_f16_tensor(')
        wrapper=wrapper.replace('ds4_gpu_hc_rms_norm_mix_f16_tensor(',f'hc_norm_mix_{backend}_test(')
        for name,replacement in (('rms_norm_plain_f16_batch8_kernel','cuda_normalize_launch'),
                ('ds4_hc_rms_scale_kernel','rocm_scale_launch'),
                ('ds4_hc_f16_project_scaled_ordered_kernel','rocm_project_launch')):
            wrapper=re.sub(name+r'<<<(.*?)>>>\(',replacement+r'(\1,',wrapper,flags=re.S)
        assert '<<<' not in wrapper
        parts.append(wrapper)
    return POLICY_MOCKS+'\n'+'\n'.join(parts)+POLICY_CASES

def staged_norm(body):
    start=body.index('    __shared__ float partial[256];')
    marker='    const float scale' if '    const float scale' in body else '    float scale'
    if 'if (threadIdx.x == 0u) scale_out[0]' in body:
        marker='    if (threadIdx.x == 0u) scale_out[0]'
    end=body.index(marker,start)
    tree=body[start:end]
    assert 'stride = blockDim.x >> 1' in tree
    assert 'partial[threadIdx.x] += partial[threadIdx.x + stride]' in tree
    assert 'partial[threadIdx.x] = sum;' in tree
    return (body[:start]+'    const float staged_sum = reduce_stage(sum);\n'+body[end:]).replace('partial[0]','staged_sum')

def production_source(fma):
    cuda=(ROOT/'ds4_cuda.cu').read_text()
    rocm=(ROOT/'rocm/ds4_rocm_norm_rope.cuh').read_text()
    common=(ROOT/'rocm/ds4_rocm_common.cuh').read_text()
    hc=(ROOT/'cuda/ds4_hc_norm_mix.cuh').read_text()
    functions=[]
    for source,name in ((rocm,'rms_norm_plain_kernel'),
                        (cuda,'rms_norm_plain_batch8_kernel'),
                        (cuda,'rms_norm_plain_f16_batch8_kernel'),
                        (hc,'ds4_hc_rms_scale_kernel')):
        functions.append(staged_norm(extract_function(source,'__global__ static void '+name+'(')))
    # Execute the explicit F32 multiplication fence already supplied for CUDA
    # by the production helper. The host shim is a volatile rounded multiply;
    # this models the HIP register fence, not PTX/GCN compiler validation.
    helper=extract_function(hc,'__device__ __forceinline__ static float ds4_hc_normalized_f32(')
    functions.append('#define __CUDA_ARCH__ 1\n'+helper+'\n#undef __CUDA_ARCH__')
    functions.append(extract_function(common,'__global__ static void matmul_f16_ordered_chunks_kernel('))
    functions.append(extract_function(hc,'__global__ static void ds4_hc_f16_project_scaled_ordered_kernel('))
    source='\n'.join(functions).replace('__global__ ','').replace('__device__ ','').replace('__forceinline__ ','').replace('__shared__','static')
    if fma:
        source,count=re.subn(r'sum \+= ([^;\n]+?) \* ([^;\n]+?);',r'sum = fmaf(\1, \2, sum);',source)
        assert count==20, f'unexpected accumulation count {count}'
    return source+'\n'+wrapper_source(cuda)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit-host',type=Path)
    args=parser.parse_args()
    template=(ROOT/'tests/test_gpu_hc_norm_mix.cpp').read_text()
    assert template.count('// PRODUCTION_SOURCE')==1
    if args.emit_host:
        args.emit_host.write_text(template.replace('// PRODUCTION_SOURCE',production_source(False)))
        return
    with tempfile.TemporaryDirectory(prefix='ds4-hc-norm-mix-') as tmp:
        path,binary=Path(tmp)/'test.cpp',Path(tmp)/'test'
        compiler=shlex.split(os.environ.get('CXX','c++'))
        if not compiler:parser.error('CXX must name a C++ compiler')
        for fma,flags in ((False,['-O2','-ffp-contract=off']),
                          (True,['-O3','-ffast-math','-fno-associative-math','-fno-finite-math-only'])):
            print('HC norm/mix host '+('explicit FMA / fast math' if fma else 'strict source arithmetic'),flush=True)
            path.write_text(template.replace('// PRODUCTION_SOURCE',production_source(fma)))
            subprocess.run(compiler+flags+['-std=c++17','-Wall','-Wextra','-Werror','-Wno-unknown-pragmas',
                '-fsanitize=address,undefined','-I'+str(ROOT),str(path),'-o',str(binary)],check=True)
            subprocess.run([str(binary)],check=True)

if __name__=='__main__':main()
