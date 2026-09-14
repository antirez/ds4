#!/usr/bin/env python3
"""Real-source indexer operand preparation and unchanged-score A/B fixture.

Default: checked policy plus extracted F16 producer/staging, strict and fast
ASan/UBSan host runs. --cuda/--rocm build native WMMA score comparisons;
--bench includes preparation on EVERY candidate invocation. Native builds
need their respective SDK, headers and an admitted GPU; host results are not
native correctness or performance evidence.
"""
import argparse
import difflib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]
BASE_SIGNATURE = '__global__ static void indexer_scores_wmma128_kernel('
CANDIDATE_SIGNATURE = '__global__ static void indexer_scores_wmma128_prepared_kernel('


def block_at(source, needle, start=0):
    begin = source.index(needle, start)
    opening = source.index('{', begin)
    end, depth = opening + 1, 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[begin:end]


def normalize(source):
    source = re.sub(r'/\*.*?\*/|//[^\n]*', '', source, flags=re.S)
    return re.findall(r'\w+|[^\w\s]', source)


def source_check(backend):
    baseline_path = 'ds4_cuda.cu' if backend == 'cuda' else 'rocm/ds4_rocm_indexer.cuh'
    candidate_path = f'{backend if backend == "cuda" else "rocm"}/ds4_{backend if backend == "cuda" else "rocm"}_indexer_prepared.cuh'
    baseline = extract_function((ROOT / baseline_path).read_text(), BASE_SIGNATURE)
    candidate = extract_function((ROOT / candidate_path).read_text(), CANDIDATE_SIGNATURE)
    rows = 32 if backend == 'cuda' else 16
    k_original = block_at(baseline, 'for (uint32_t i = tid; i < 128u * 128u;')
    q_original = block_at(baseline, f'for (uint32_t i = tid; i < {rows}u * 128u;', baseline.index('for (uint32_t h ='))
    normalized = candidate
    stages = []
    for name, original in (('K', k_original), ('Q', q_original)):
        expression = rf'// DS4_INDEXER_PREPARED_{name}_BEGIN.*?// DS4_INDEXER_PREPARED_{name}_END'
        found = re.findall(expression, candidate, flags=re.S)
        if len(found) != 1:
            raise AssertionError(f'{candidate_path}: expected one {name} staging marker pair')
        stages.append(block_at(found[0], 'for (uint32_t i ='))
        normalized = re.sub(expression, lambda _: original, normalized, flags=re.S)
    normalized = normalized.replace('indexer_scores_wmma128_prepared_kernel', 'indexer_scores_wmma128_kernel')
    normalized = normalized.replace('const __half *q,', 'const float *q,').replace('const __half *index_comp,', 'const float *index_comp,')
    normalized = normalized.replace('__align__(32) ', '')
    expected, actual = normalize(baseline), normalize(normalized)
    if expected != actual:
        difference = '\n'.join(list(difflib.unified_diff(expected, actual))[:90])
        raise AssertionError(f'{candidate_path}: non-staging arithmetic/control-flow changed:\n{difference}')
    print(f'PASS: {backend} consumer source equals baseline outside its two packed staging loops.', flush=True)
    return baseline, stages, rows


def host_stages(backend, stages):
    k, q = stages
    return f'''
namespace {backend}_staging {{
static void k(__half *b_sh,const __half *index_comp,uint32_t n_comp,uint32_t tile_c,uint32_t tid) {{
    const uint32_t head_dim=128;
    {k}
}}
static void q(__half *a_sh,const __half *q,uint32_t n_tokens,uint32_t n_head,uint32_t h,uint32_t tile_t,uint32_t tid) {{
    const uint32_t head_dim=128;
    {q}
}}
}}
'''


STAGING_CASES = r'''
using q_stage = void(*)(__half*,const __half*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
using k_stage = void(*)(__half*,const __half*,uint32_t,uint32_t,uint32_t);
static void staging_case(q_stage q_load,k_stage k_load,unsigned rows,unsigned stride,
        uint32_t nc,uint32_t nt,unsigned pattern) {
    constexpr uint32_t nh=64,hd=128;
    const size_t nq=size_t(nt)*nh*hd,nk=size_t(nc)*hd;
    const auto q=input(nq,pattern,91),k=input(nk,pattern,73);
    std::vector<__half> qh(nq+2*guard,half_poison()),kh(nk+2*guard,half_poison());
    prepare_host(qh.data()+guard,kh.data()+guard,q.data()+guard,k.data()+guard,nq,nk,3);
    for(uint32_t tile_t:{0u,((nt+rows-1)/rows-1)*rows})
    for(uint32_t tile_c:{0u,((nc+127)/128-1)*128})
    for(uint32_t h:{0u,1u,63u}) {
        std::vector<__half> a(rows*stride+2*guard,half_poison()),b(128*stride+2*guard,half_poison());
        for(unsigned tid=0;tid<256;++tid) {
            q_load(a.data()+guard,qh.data()+guard,nt,nh,h,tile_t,tid);
            k_load(b.data()+guard,kh.data()+guard,nc,tile_c,tid);
        }
        for(unsigned axis=0;axis<2;++axis) {
            const auto &v=axis?b:a;const unsigned nr=axis?128:rows;
            for(size_t i=0;i<v.size();++i) {
                uint16_t expected=half_bits(half_poison());
                if(i>=guard&&i<guard+nr*stride) {
                    const unsigned row=unsigned(i-guard)/stride,d=unsigned(i-guard)%stride;
                    if(d<hd)expected=axis?
                        (tile_c+row<nc?half_reference(k[guard+size_t(tile_c+row)*hd+d]):0):
                        (tile_t+row<nt?half_reference(q[guard+(size_t(tile_t+row)*nh+h)*hd+d]):0);
                }
                check(half_bits(v[i])==expected,"actual packed LDS staging preserves row/head order, zero tails, padding and guards");
            }
        }
        ++stage_cases;
    }
    verify_half(qh,q,nq);verify_half(kh,k,nk);
}
static void staging_cases() {
    for(auto shape:{std::pair<uint32_t,uint32_t>{1,1},{31,7},{128,16},{129,17},{513,33}})
        for(unsigned pattern=0;pattern<5;++pattern) {
            staging_case(cuda_staging::q,cuda_staging::k,32,136,shape.first,shape.second,pattern);
            staging_case(hip_staging::q,hip_staging::k,16,128,shape.first,shape.second,pattern);
        }
}
'''

FAULT_MOCKS = r'''
using ds4_indexer_prepared_stream_t=unsigned;
struct dim3 {unsigned x,y,z;dim3(unsigned a,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
using mock_StreamCaptureStatus=int;
enum {mock_Success=0,mock_StreamCaptureStatusNone=0,mock_StreamCaptureStatusInvalidated=2};
static unsigned producers,consumers,queries,peeks,error_reads;
static int fail_capture,capture_status,existing_error,producer_error,consumer_error,pending_error;
static ds4_indexer_plan_request request;
static ds4_indexer_prepared_caps caps;
static ds4_indexer_prepared_buffers buffers;
static ds4_indexer_prepared_plan expected;
static int mock_StreamIsCapturing(unsigned stream,int *status) {
    check(stream==37,"wrapper propagates requested stream to capture query");++queries;*status=capture_status;return fail_capture;
}
static int mock_PeekAtLastError() {++peeks;return existing_error;}
static int mock_GetLastError() {++error_reads;const int result=pending_error;pending_error=0;return result;}
static void mock_prepare(unsigned blocks,unsigned threads,unsigned shared,unsigned stream,
        __half *qh,__half *kh,const float *q,const float *k,uint64_t nq,uint64_t nk) {
    check(blocks==expected.prepare_blocks&&threads==256&&!shared&&stream==37&&
        uintptr_t(qh)==buffers.scratch&&uintptr_t(kh)==buffers.scratch+expected.k_offset&&
        uintptr_t(q)==buffers.q&&uintptr_t(k)==buffers.keys&&nq==expected.q_elements&&nk==expected.k_elements,
        "actual wrapper forwards checked producer geometry, stream and contiguous scratch pointers");
    ++producers;pending_error=producer_error;
}
static void mock_score(bool registers,dim3 grid,unsigned threads,unsigned shared,unsigned stream,
        float *out,const __half *qh,const float *weights,const __half *kh,
        uint32_t nc,uint32_t nt,uint32_t pos,uint32_t heads,uint32_t dim,uint32_t ratio,float scale,int causal) {
    check(producers==1&&!producer_error&&error_reads==1,"consumer only after successful producer error check");
    check(registers==caps.register_scores&&grid.x==expected.score_grid_x&&grid.y==expected.score_grid_y&&grid.z==1&&
        threads==256&&!shared&&stream==37&&uintptr_t(out)==buffers.scores&&uintptr_t(qh)==buffers.scratch&&
        uintptr_t(kh)==buffers.scratch+expected.k_offset&&uintptr_t(weights)==buffers.weights&&
        nc==request.n_comp&&nt==request.n_tokens&&pos==request.pos0&&heads==request.n_head&&dim==request.head_dim&&
        ratio==request.ratio&&bits(scale)==bits(-.1f)&&causal==1,"actual wrapper preserves score arguments and selected consumer");
    ++consumers;pending_error=consumer_error;
}
'''

FAULT_CASES = r'''
static void cases() {
    unsigned count=0;
    auto reset=[]() {
        producers=consumers=queries=peeks=error_reads=0;
        fail_capture=capture_status=existing_error=producer_error=consumer_error=pending_error=0;
        request={DS4_INDEXER_PREPARED_BACKEND,DS4_GPU_PHASE_PREFILL,513,129,1024,64,128,4,false};
        caps={true,false,false,false,true,1024,65536,UINT32_MAX,65535,UINT64_MAX};
        check(ds4_indexer_prepared_build(&request,&caps,&expected),"fault fixture valid plan");
        buffers={0x10000000,0x20000000,0x30000000,0x40000000,0x50000000,
            expected.base.q_bytes,expected.base.index_bytes,expected.base.weight_bytes,
            expected.base.score_bytes,expected.scratch_bytes};
    };
    auto call=[](){return ds4_indexer_prepared_try(&request,&caps,&buffers,-.1f,1,37);};
    auto decline=[&](){check(call()==0&&!producers&&!consumers&&!queries&&!peeks&&!error_reads,"wrapper declines before stream/GPU work");++count;};
    for(unsigned reg=0;reg<2;++reg) {
        reset();caps.register_scores=reg;
        if(reg&&request.backend!=DS4_INDEXER_BACKEND_HIP){decline();continue;}
        check(ds4_indexer_prepared_build(&request,&caps,&expected),"candidate plan");
        check(call()==1&&producers==1&&consumers==1&&queries==1&&peeks==1&&error_reads==2,"one producer + one consumer without retry");++count;
        reset();caps.register_scores=reg;check(ds4_indexer_prepared_build(&request,&caps,&expected),"producer-fault plan");producer_error=1;
        check(call()==-1&&producers==1&&!consumers&&error_reads==1,"failed producer cannot submit consumer or fallback");++count;
        reset();caps.register_scores=reg;check(ds4_indexer_prepared_build(&request,&caps,&expected),"consumer-fault plan");consumer_error=1;
        check(call()==-1&&producers==1&&consumers==1&&error_reads==2,"failed consumer returns -1 without retry");++count;
    }
    reset();request.phase=DS4_GPU_PHASE_AUTO;check(call()==1,"AUTO preserves admitted path");++count;
    for(auto phase:{DS4_GPU_PHASE_DECODE,DS4_GPU_PHASE_VERIFY,DS4_GPU_PHASE_BATCH_DECODE,DS4_GPU_PHASE_MIXED}) {reset();request.phase=phase;decline();}
    reset();request.quality=true;decline();reset();caps.native_mxf4=true;decline();reset();caps.capturing=true;decline();
    reset();caps.device_supported=false;decline();reset();caps.max_shared_bytes=1;decline();reset();request.n_tokens=1;decline();
    reset();request.backend=request.backend==DS4_INDEXER_BACKEND_CUDA?DS4_INDEXER_BACKEND_HIP:DS4_INDEXER_BACKEND_CUDA;decline();
    reset();--buffers.scratch_bytes;decline();reset();buffers.scores=buffers.q;decline();reset();buffers.scratch=buffers.keys;decline();
    reset();++buffers.q;decline();reset();buffers.weights=0;decline();
    reset();fail_capture=1;check(call()==-1&&queries==1&&!peeks&&!producers&&!consumers,"capture query failure propagates before writer");++count;
    reset();capture_status=1;check(call()==0&&queries==1&&!peeks&&!producers&&!consumers,"actual captured stream declines before writer");++count;
    reset();capture_status=2;check(call()==-1&&queries==1&&!peeks&&!producers&&!consumers,"invalidated capture propagates error without fallback");++count;
    reset();existing_error=1;check(call()==-1&&queries==1&&peeks==1&&!error_reads&&!producers&&!consumers,"existing runtime error is not swallowed or overwritten");++count;
    reset();check(ds4_indexer_prepared_try(nullptr,&caps,&buffers,-.1f,1,37)==0&&!queries,"null request preflight");++count;
    reset();check(ds4_indexer_prepared_try(&request,nullptr,&buffers,-.1f,1,37)==0&&!queries,"null capabilities preflight");++count;
    reset();check(ds4_indexer_prepared_try(&request,&caps,nullptr,-.1f,1,37)==0&&!queries,"null buffers preflight");++count;
    std::printf("PASS: %u actual %s pipeline cases; phase/resource/alias decline, actual stream capture, prior-error and producer/consumer fault propagation.\n",count,
        DS4_INDEXER_PREPARED_BACKEND==DS4_INDEXER_BACKEND_HIP?"HIP":"CUDA");
}
'''


def fault_source(backend):
    body = extract_function((ROOT / 'ds4_indexer_prepared_launch.cuh').read_text(),
                            'static int ds4_indexer_prepared_try(')
    replacements = (
        ('ds4_indexer_prepare_f16_kernel<<<plan.prepare_blocks, 256, 0, stream>>>(',
         'mock_prepare(plan.prepare_blocks, 256, 0, stream,'),
        ('ds4_rocm_indexer_scores_registers_kernel<true><<<score_grid, 256, 0, stream>>>(',
         'mock_score(true, score_grid, 256, 0, stream,'),
        ('indexer_scores_wmma128_prepared_kernel<<<score_grid, 256, 0, stream>>>(',
         'mock_score(false, score_grid, 256, 0, stream,'),
    )
    for before, after in replacements:
        if body.count(before) != 1:
            raise AssertionError(f'expected exactly one production launch: {before}')
        body = body.replace(before, after)
    prefix = '#define __HIPCC__ 1\n' if backend == 'hip' else ''
    suffix = '#undef __HIPCC__\n' if backend == 'hip' else ''
    return (prefix + f'namespace {backend}_fault {{\n'
            '#define DS4_INDEXER_PREPARED_GPU(name) mock_##name\n'
            f'#define DS4_INDEXER_PREPARED_BACKEND DS4_INDEXER_BACKEND_{backend.upper()}\n'
            + FAULT_MOCKS + body + FAULT_CASES +
            '\n#undef DS4_INDEXER_PREPARED_GPU\n#undef DS4_INDEXER_PREPARED_BACKEND\n}\n' + suffix)


def compiler_command(parser, variable, fallback):
    compiler = shlex.split(os.environ.get(variable, fallback))
    if not compiler or not shutil.which(compiler[0]):
        parser.error(f'{variable} must name an available compiler ({fallback} by default)')
    return compiler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument('--cuda', action='store_true')
    backend.add_argument('--rocm', action='store_true')
    parser.add_argument('--bench', action='store_true')
    args = parser.parse_args()
    if args.bench and not (args.cuda or args.rocm):
        parser.error('--bench requires --cuda or --rocm; no simulated GPU timings')
    sources = {name: source_check(name) for name in ('cuda', 'hip')}
    template = (ROOT / 'tests/test_gpu_indexer_prepared.cpp').read_text()
    producer = extract_function((ROOT / 'cuda/ds4_indexer_prepare.cuh').read_text(),
                                '__global__ static void ds4_indexer_prepare_f16_kernel(')
    with tempfile.TemporaryDirectory(prefix='ds4-indexer-prepared-') as tmp:
        source, binary = Path(tmp) / 'fixture.cpp', Path(tmp) / 'fixture'
        if args.cuda or args.rocm:
            chosen = 'hip' if args.rocm else 'cuda'
            source.write_text(template.replace('// PRODUCTION_SOURCE', sources[chosen][0]))
            if args.rocm:
                compiler = compiler_command(parser, 'HIPCC', 'hipcc')
                print('Running native rocWMMA coordinate diagnostics before score validation or timing.', flush=True)
                subprocess.run([sys.executable, '-B', str(ROOT / 'tests/test_rocm_indexer_registers.py'), '--rocm'], check=True)
                flags = shlex.split(os.environ.get('ROCM_CFLAGS', '-O3 --offload-arch=gfx1151'))
                flags = [f for f in flags if f not in ('-ffast-math', '-fno-fast-math', '-fno-finite-math-only') and not f.startswith('-ffp-contract=')]
                command = compiler + flags + ['-std=c++17', '-x', 'hip', '-DTEST_HIP', '-DTEST_NATIVE']
                modes = (('strict', ['-fno-fast-math', '-ffp-contract=off']),
                         ('production-fast', ['-ffast-math', '-fno-finite-math-only']))
            else:
                compiler = compiler_command(parser, 'NVCC', 'nvcc')
                flags = shlex.split(os.environ.get('CUDA_CFLAGS', '-O3'))
                flags = [f for f in flags if f != '--use_fast_math' and not f.startswith(('--fmad=', '--ftz=', '--prec-div=', '--prec-sqrt='))]
                if not any(f.startswith(('-arch', '--gpu-architecture', '-gencode', '--generate-code')) for f in flags):
                    flags += ['-arch=' + (os.environ.get('CUDA_ARCH', '').strip() or 'native')]
                command = compiler + flags + ['-std=c++17', '-x', 'cu', '-DTEST_NATIVE', '-Xptxas=-v']
                modes = (('strict', ['--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true']),
                         ('production-fast', ['--use_fast_math']))
            for label, math in modes:
                complete = command + math + ['-I', str(ROOT), str(source), '-o', str(binary)]
                print(f'Building/running native {chosen} {label}: {shlex.join(complete)}', flush=True)
                subprocess.run(complete, check=True)
                subprocess.run([str(binary)] + (['--bench'] if args.bench and label == 'production-fast' else []), check=True)
        else:
            compiler = compiler_command(parser, 'CXX', 'clang++')
            bodies = producer.replace('__global__ ', '') + '\n'
            bodies += '\n'.join(host_stages(name, item[1]) for name, item in sources.items())
            bodies += '\n' + fault_source('cuda') + '\n' + fault_source('hip')
            source.write_text(template.replace('// PRODUCTION_SOURCE', bodies).replace('// HOST_STAGING_CASES', STAGING_CASES)
                              .replace('policy_cases();', 'policy_cases();cuda_fault::cases();hip_fault::cases();'))
            for label, math in (('strict', ['-ffp-contract=off']),
                                ('fast', ['-ffast-math', '-fno-finite-math-only'])):
                print(f'Building/running host {label} with ASan/UBSan.', flush=True)
                subprocess.run(compiler + ['-std=c++17', '-O2', '-g', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-I', str(ROOT)] + math +
                    [str(source), '-o', str(binary)], check=True)
                subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
