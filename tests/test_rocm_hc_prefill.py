#!/usr/bin/env python3
"""Check real ROCm RMS-to-FP16 source and preflight/error contracts.

Host strict/explicit-FMA fast variants use ASan/UBSan and staged 256-lane
barriers; they do not validate HIP scheduling, FTZ or device assembly. --rocm
builds/runs the actual operand kernels; --rocm --bench also times both complete
operand-preparation routes. Full public-API/BLAS tests are a separate target.
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

LT_MOCKS = r'''
namespace lt_test {
static int g_hipblaslt=17,g_hipblaslt_ready=1,g_hipblaslt_prefill_state;
static int solution,version_result,revision_result,matmul_result;
static bool plan_available,version_matches;
static unsigned matmuls,plans;
enum {HIP_R_32F=4};
struct cuda_hipblaslt_gemm_plan {int desc=2,a_desc=3,b_desc=4,c_desc=5,d_desc=6,algo=7;};
static cuda_hipblaslt_gemm_plan plan;
static int hipblaslt_prefill_solution_index(uint32_t,uint32_t,uint32_t) {return solution;}
static bool hipblaslt_ok(int status,const char *) {return status==0;}
static int hipblasLtGetVersion(int handle,int *out) {check(handle==17,"Lt handle");*out=version_matches?100401:1;return version_result;}
static int hipblasLtGetGitRevision(int,char *out) {std::strcpy(out,"8d1ae90e");return revision_result;}
static cuda_hipblaslt_gemm_plan *hipblaslt_gemm_plan_get(uint32_t m,uint32_t n,uint32_t k,const char *,int dtype,int chosen) {
    check(m==24&&n==2048&&k==16384&&dtype==HIP_R_32F&&chosen==solution,"unchanged Lt plan shape/type");
    ++plans;return plan_available?&plan:nullptr;
}
static int hipblasLtMatmul(int handle,int desc,const float *alpha,const __half *,int ad,
        const __half *,int bd,const float *beta,float *c,int cd,float *d,int dd,
        int *algo,void *workspace,size_t size,int stream) {
    check(handle==17&&desc==2&&*alpha==1.f&&*beta==0.f&&ad==3&&bd==4&&cd==5&&dd==6&&
        c==d&&*algo==7&&!workspace&&!size&&!stream,"unchanged Lt submission arguments");
    ++matmuls;return matmul_result;
}
'''
LT_CASES = r'''
static void cases() {
    float output=0;__half input=__float2half(1.f),weight=__float2half(2.f);
    unsigned count=0;
    for(bool strict:{false,true}) {
        auto reset=[](){g_hipblaslt_ready=1;g_hipblaslt_prefill_state=0;solution=10;
            version_result=revision_result=matmul_result=0;plan_available=version_matches=true;matmuls=plans=0;};
        auto call=[&](){return hipblaslt_gemm_tn_f16_out_f32_prefill(&output,&weight,&input,24,2048,16384,strict);};
        reset();check(call()==1&&matmuls==1&&plans==1&&g_hipblaslt_prefill_state==1,"actual Lt success");++count;
        reset();matmul_result=1;check(call()==(strict?-1:0)&&matmuls==1&&g_hipblaslt_prefill_state==-1,"Lt submission failure distinct from no plan");++count;
        reset();plan_available=false;check(call()==0&&!matmuls&&plans==1,"unavailable Lt plan permits first consumer fallback");++count;
        reset();version_result=1;check(call()==0&&!matmuls&&!plans,"Lt version query failure precedes submission");++count;
        reset();revision_result=1;check(call()==0&&!matmuls&&!plans,"Lt revision failure precedes submission");++count;
        reset();version_matches=false;check(call()==0&&!matmuls&&!plans,"unsupported Lt version precedes submission");++count;
        reset();solution=-1;check(call()==0&&!matmuls&&!plans,"unsupported Lt shape precedes submission");++count;
        reset();g_hipblaslt_ready=0;check(call()==0&&!matmuls&&!plans,"unready Lt precedes submission");++count;
        reset();g_hipblaslt_prefill_state=-1;check(call()==0&&!matmuls&&!plans,"disabled Lt precedes submission");++count;
    }
    std::printf("PASS: %u actual hipBLASLt helper cases; preflight/plan-unavailable vs post-Matmul failure and unchanged non-strict contract.\n",count);
}
} // namespace lt_test
'''

GRAPH_MOCKS = r'''
namespace graph_test {
enum {DS4_N_HC=4,DS4_TENSOR_F16=1};
static constexpr float DS4_RMS_EPS=1e-6f;
struct ds4_gpu_tensor {int id;};
struct ds4_model {const void *map;uint64_t size;};
struct ds4_tensor {uint32_t type;uint64_t abs_offset;};
static ds4_gpu_tensor output{1},normalized{2},input{3};
static ds4_tensor weight{DS4_TENSOR_F16,128};
static ds4_model model{&weight,2000000};
static unsigned trace;
static int folded,norm_result,project_result;
static bool norm_written;
static uint64_t dimension;
static uint32_t rows;
static int ds4_gpu_matmul_f16_rms_fold_tensor(ds4_gpu_tensor *out,const void *map,uint64_t bytes,
        uint64_t offset,uint64_t n,uint64_t m,const ds4_gpu_tensor *x,uint64_t tokens,float eps) {
    check(out==&output&&map==model.map&&bytes==model.size&&offset==weight.abs_offset&&
        n==dimension&&m==24&&x==&input&&tokens==rows&&eps==DS4_RMS_EPS,"graph folded argument propagation");
    trace=trace*10+1;return folded;
}
static int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out,const ds4_gpu_tensor *x,
        uint32_t n,uint32_t tokens,float eps) {
    check(out==&normalized&&x==&input&&n==dimension&&tokens==rows&&eps==DS4_RMS_EPS,"graph reference RMS operands");
    trace=trace*10+2;norm_written=norm_result!=0;return norm_result;
}
static int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out,const void *map,uint64_t bytes,
        uint64_t offset,uint64_t n,uint64_t m,const ds4_gpu_tensor *x,uint64_t tokens) {
    check(out==&output&&map==model.map&&bytes==model.size&&offset==weight.abs_offset&&
        n==dimension&&m==24&&x==&normalized&&tokens==rows&&norm_written,"graph reads scratch only after successful reference RMS");
    trace=trace*10+3;return project_result;
}
'''
GRAPH_CASES = r'''
static void cases() {
    unsigned count=0;
    for(uint64_t n:{UINT64_C(16384),UINT64_C(28672)})
    for(uint32_t tokens:{2u,128u})
    for(unsigned f16=0;f16<2;++f16)
    for(folded=-1;folded<=1;++folded)
    for(norm_result=0;norm_result<2;++norm_result)
    for(project_result=0;project_result<2;++project_result) {
        dimension=n;rows=tokens;weight.type=f16?DS4_TENSOR_F16:7;trace=0;norm_written=false;
        const bool result=metal_graph_hc_rms_scale_project(&output,&normalized,&model,&weight,&input,dimension,rows);
        if(f16&&folded!=0)check(result==(folded>0)&&trace==1&&!norm_written,"graph success/failure must not replay fallback or read norm scratch");
        else check(result==bool(norm_result&&project_result)&&
            trace==(norm_result?(f16?123u:23u):(f16?12u:2u)),"graph fallback occurs only on decline and stops after failed norm");
        ++count;
    }
    auto bad=[&](ds4_gpu_tensor *out,ds4_gpu_tensor *norm,const ds4_model *m,
            const ds4_tensor *w,const ds4_gpu_tensor *x,uint64_t n) {
        trace=0;check(!metal_graph_hc_rms_scale_project(out,norm,m,w,x,n,128)&&trace==0,"graph invalid arguments must not dispatch");++count;
    };
    bad(nullptr,&normalized,&model,&weight,&input,16384);
    bad(&output,nullptr,&model,&weight,&input,16384);
    bad(&output,&normalized,nullptr,&weight,&input,16384);
    bad(&output,&normalized,&model,nullptr,&input,16384);
    bad(&output,&normalized,&model,&weight,nullptr,16384);
    bad(&output,&normalized,&model,&weight,&input,UINT64_C(1)+UINT32_MAX);
    std::printf("PASS: %u actual HC prefill graph tri-state/fallback/scratch and invalid-input cases.\n",count);
}
} // namespace graph_test
'''


def staged_norm(body):
    start = body.index('    __shared__ float partial[256];')
    end = body.index('    float scale =', start)
    tree = body[start:end]
    assert 'stride = blockDim.x >> 1' in tree
    assert 'partial[threadIdx.x] += partial[threadIdx.x + stride]' in tree
    assert 'partial[threadIdx.x] = sum;' in tree
    return (body[:start] + '    const float staged_sum = reduce_stage(sum);\n' +
            body[end:]).replace('partial[0]', 'staged_sum')


def production_source(native, fma=False):
    rms = (ROOT / 'rocm/ds4_rocm_norm_rope.cuh').read_text()
    common = (ROOT / 'rocm/ds4_rocm_common.cuh').read_text()
    candidate = (ROOT / 'rocm/ds4_rocm_rms_f16.cuh').read_text()
    reference = extract_function(rms, '__global__ static void rms_norm_plain_kernel(')
    cast = extract_function(common, '__global__ static void f32_to_f16_kernel(')
    if native:
        # The production header supplies the unchanged HIP candidate + fence.
        return reference + '\n' + cast
    fused = extract_function(candidate, '__global__ static void rocm_hc_rms_norm_f16_kernel(')
    hc = (ROOT / 'cuda/ds4_hc_norm_mix.cuh').read_text()
    fence = extract_function(hc, '__device__ __forceinline__ static float ds4_hc_normalized_f32(')
    source = '#define __CUDA_ARCH__ 1\n' + fence + '\n#undef __CUDA_ARCH__\n'
    source += staged_norm(reference) + '\n' + staged_norm(fused) + '\n' + cast
    source = source.replace('__global__ ', '').replace('__device__ ', '').replace('__forceinline__ ', '')
    if fma:
        source, count = re.subn(r'sum \+= v \* v;', 'sum = fmaf(v, v, sum);', source)
        assert count == 2, f'unexpected RMS accumulation count {count}'
    wrapper = (ROOT / 'rocm/ds4_rocm_current_api_compat.cuh').read_text()
    stream = extract_function(wrapper, 'static bool rocm_hc_prefill_stream_ready(')
    actual = extract_function(wrapper, 'extern "C" int ds4_gpu_matmul_f16_rms_fold_tensor(')
    actual, count = re.subn(r'rocm_hc_rms_norm_f16_kernel<<<(.*?)>>>\(',
        r'mock_norm_launch(\1,', actual, flags=re.S)
    assert count == 1 and '<<<' not in actual
    lt = extract_function((ROOT / 'rocm/ds4_rocm_hipblaslt.cuh').read_text(),
                          'static int hipblaslt_gemm_tn_f16_out_f32_prefill(')
    graph = extract_function((ROOT / 'ds4.c').read_text(),
                             'static bool metal_graph_hc_rms_scale_project(')
    graph = ('\n#ifdef __APPLE__\n#define DS4_TEST_RESTORE_APPLE 1\n#undef __APPLE__\n#endif\n' +
             graph + '\n#ifdef DS4_TEST_RESTORE_APPLE\n#define __APPLE__ 1\n#undef DS4_TEST_RESTORE_APPLE\n#endif\n')
    return (source + '\n#define __HIP_PLATFORM_AMD__ 1\n' + stream +
            '\n#undef __HIP_PLATFORM_AMD__\n' + actual + LT_MOCKS + lt + LT_CASES +
            GRAPH_MOCKS + graph + GRAPH_CASES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rocm', action='store_true')
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--emit-host', type=Path)
    parser.add_argument('--emit-rocm', type=Path)
    args = parser.parse_args()
    if args.bench and not args.rocm:
        parser.error('--bench requires --rocm; host timing is not GPU performance')
    template = (ROOT / 'tests/test_rocm_hc_prefill.cpp').read_text()
    assert template.count('// PRODUCTION_SOURCE') == 1
    if args.emit_host or args.emit_rocm:
        path = args.emit_rocm or args.emit_host
        path.write_text(template.replace('// PRODUCTION_SOURCE', production_source(bool(args.emit_rocm))))
        return
    with tempfile.TemporaryDirectory(prefix='ds4-rocm-hc-prefill-') as tmp:
        path, binary = Path(tmp)/'test.cpp', Path(tmp)/'test'
        if args.rocm:
            compiler = shlex.split(os.environ.get('HIPCC', 'hipcc'))
            if not compiler or not shutil.which(compiler[0]):
                parser.error('HIPCC must name an installed HIP compiler')
            path.write_text(template.replace('// PRODUCTION_SOURCE', production_source(True)))
            subprocess.run(compiler + ['-std=c++17', '-O3', '-ffast-math', '-DDS4_HC_PREFILL_NATIVE',
                '-I'+str(ROOT), str(path), '-o', str(binary)], check=True)
            subprocess.run([str(binary)] + (['--bench'] if args.bench else []), check=True)
            return
        compiler = shlex.split(os.environ.get('CXX', 'clang++'))
        if not compiler:
            parser.error('CXX must name a C++ compiler')
        for fma, flags in ((False, ['-O2', '-ffp-contract=off']),
                           (True, ['-O3', '-ffast-math', '-fno-associative-math', '-fno-finite-math-only'])):
            path.write_text(template.replace('// PRODUCTION_SOURCE', production_source(False, fma)))
            subprocess.run(compiler + flags + ['-std=c++17', '-Wall', '-Wextra', '-Werror',
                '-Wno-unknown-pragmas', '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                '-I'+str(ROOT), str(path), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)
            print('PASS: '+('explicit FMA fast' if fma else 'strict')+' actual source, ASan/UBSan', flush=True)


if __name__ == '__main__':
    main()
