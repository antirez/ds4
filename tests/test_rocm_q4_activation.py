#!/usr/bin/env python3
"""Check extracted Q-B F16 conversion, staging and dispatch; --rocm runs WMMA.

Host strict/fast ASan/UBSan validates addressing and launch/error contracts.
Native --rocm [--bench] compares the unchanged F32 K128 input against conversion
plus reused F16 input. A native run requires gfx1151 and never silently skips.
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


def phase_implementation(source):
    declaration = re.search(r'^static thread_local ds4_gpu_execution_phase .*;$',
                            source, re.M)
    assert declaration, 'production TLS declaration missing'
    return declaration[0] + '\n' + '\n'.join(extract_function(source, name) for name in (
        'extern "C" ds4_gpu_execution_phase ds4_gpu_get_execution_phase(',
        'extern "C" ds4_gpu_execution_phase ds4_gpu_exchange_execution_phase(',
    ))


def test_backend_phases(compiler, directory):
    """Compile actual CUDA/ROCm TLS bodies without requiring either GPU SDK."""
    for backend in ('cuda', 'rocm'):
        source = (ROOT / f'ds4_{backend}.cu').read_text()
        body = '#include "ds4_gpu_phase.h"\n#include <cassert>\n#include <thread>\n'
        body += '#include <cstdint>\n#include <cstring>\n' + phase_implementation(source)
        graph_test = ''
        if backend == 'cuda':
            body += '\nusing cudaGraphExec_t = void *;\n'
            body += extract_function(source, 'struct ds4_decode_graph_key {') + ';\n'
            body += extract_function(source, 'struct cuda_decode_graph_entry {') + ';\n'
            for name in ('LAYERS', 'ISLANDS', 'VARIANTS'):
                definition = re.search(r'^#define CUDA_DECODE_GRAPH_' + name + r'\s+\d+u$', source, re.M)
                assert definition
                body += definition[0] + '\n'
            body += '''static cuda_decode_graph_entry
                g_decode_graphs[CUDA_DECODE_GRAPH_LAYERS][CUDA_DECODE_GRAPH_ISLANDS]
                               [CUDA_DECODE_GRAPH_VARIANTS];\n'''
            body += extract_function(source, 'static cuda_decode_graph_entry *cuda_decode_graph_find(')
            graph_test = '''
                ds4_decode_graph_key key{};
                auto *automatic = cuda_decode_graph_find(&key, DS4_GPU_PHASE_AUTO);
                assert(automatic); automatic->state = 1;
                auto *decode = cuda_decode_graph_find(&key, DS4_GPU_PHASE_DECODE);
                assert(decode && decode != automatic); decode->state = 1;
                assert(cuda_decode_graph_find(&key, DS4_GPU_PHASE_AUTO) == automatic);
                assert(cuda_decode_graph_find(&key, DS4_GPU_PHASE_DECODE) == decode);
                auto *prefill = cuda_decode_graph_find(&key, DS4_GPU_PHASE_PREFILL);
                assert(prefill && prefill != decode); prefill->state = 1;
                auto *verify = cuda_decode_graph_find(&key, DS4_GPU_PHASE_VERIFY);
                assert(verify && verify != prefill); verify->state = 1;
                assert(!cuda_decode_graph_find(&key, DS4_GPU_PHASE_BATCH_DECODE));
                assert(!cuda_decode_graph_find(&key, DS4_GPU_PHASE_MIXED));
            '''
        body += '''
            int main() {
                assert(ds4_gpu_get_execution_phase() == DS4_GPU_PHASE_AUTO);
                for (int value = DS4_GPU_PHASE_AUTO; value <= DS4_GPU_PHASE_MIXED; ++value) {
                    const auto phase = static_cast<ds4_gpu_execution_phase>(value);
                    assert(ds4_gpu_exchange_execution_phase(phase) == DS4_GPU_PHASE_AUTO);
                    assert(ds4_gpu_get_execution_phase() == phase);
                    assert(ds4_gpu_execution_phase_allows_prefill(phase) ==
                           (phase == DS4_GPU_PHASE_AUTO || phase == DS4_GPU_PHASE_PREFILL));
                    std::thread worker([] {
                        assert(ds4_gpu_get_execution_phase() == DS4_GPU_PHASE_AUTO);
                        assert(ds4_gpu_exchange_execution_phase(DS4_GPU_PHASE_VERIFY) == DS4_GPU_PHASE_AUTO);
                        assert(ds4_gpu_get_execution_phase() == DS4_GPU_PHASE_VERIFY);
                    });
                    worker.join();
                    assert(ds4_gpu_get_execution_phase() == phase);
                    assert(ds4_gpu_exchange_execution_phase(DS4_GPU_PHASE_AUTO) == phase);
                }
        ''' + graph_test + '\n}\n'
        generated = directory / f'{backend}_phase.cpp'
        generated.write_text(body)
        binary = directory / f'{backend}_phase'
        subprocess.run(compiler + ['-O2', '-std=c++17', '-Wall', '-Wextra', '-Werror',
            '-pthread', '-fsanitize=address,undefined', '-I', str(ROOT),
            str(generated), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
        print(f'PASS actual {backend.upper()} TLS phase isolation' +
              (' and phase-keyed graph lookup' if backend == 'cuda' else ''), flush=True)


def build_source(native):
    q4 = (ROOT / 'rocm/ds4_rocm_q4.cuh').read_text()
    common = (ROOT / 'rocm/ds4_rocm_common.cuh').read_text()
    convert = extract_function(common, '__global__ static void f32_to_f16_kernel(')
    definitions = [phase_implementation((ROOT / 'ds4_rocm.cu').read_text()), convert]
    names = [
        'static int rocm_q4_K_prefill_wmma_load4_compatible(',
        'static void rocm_q4_K_prefill_wmma_k128_enqueue(',
        'static int rocm_q4_K_prefill_wmma_k128_half_enqueue(',
        'extern "C" int ds4_rocm_bench_q4_K_wmma_k128_enqueue(',
        'extern "C" int ds4_rocm_bench_q4_K_wmma_k128_half_enqueue(',
    ]
    if native:
        signature = '__global__ static void\nrocm_matmul_q4_K_prefill_wmma_k128_p144_rowtile_strided_kernel('
        body = extract_function(q4, signature)
        begin = q4.rfind('template <', 0, q4.index(signature))
        definitions.append(q4[begin:q4.index(signature)] + body)
    else:
        names += [
            'static int rocm_q4_K_prefill_wmma_load2_compatible(',
            'static int rocm_q4_K_prefill_wmma_k64_control_policy(',
            'static int rocm_q4_K_prefill_wmma_k128_policy(',
            'static int rocm_q4_K_prefill_wmma_k128_half_try(',
            'static int rocm_q4_K_prefill_wmma_launch(',
        ]
    definitions += [extract_function(q4, name) for name in names]
    if not native:
        body = extract_function(q4, '__global__ static void\nrocm_matmul_q4_K_prefill_wmma_k128_p144_rowtile_strided_kernel(')
        stage = extract_function(body, 'for (uint32_t j = tid * 4u;')
        definitions.append('''static void stage_f32_baseline(_Float16 *lds_x, const float *x,
                uint32_t n_tok, uint32_t tok0, uint32_t k0, uint32_t tid) {
            const uint32_t group=0, group32_base=k0/32u;
            const uint64_t x_token_stride=1024, x_group_stride=0;
        ''' + stage + '\n}')
    text = '\n'.join(definitions)
    if not native:
        text = text.replace('__global__', '')
        text, count = re.subn(r'f32_to_f16_kernel<<<(.*?)>>>\(',
                             r'conversion_launch(dim3(\1), ', text, flags=re.S)
        assert count == 1, count
        text, count = re.subn(
            r'rocm_matmul_q4_K_prefill_wmma_k128_p144_rowtile_strided_kernel<\s*'
            r'256u, 16u, 1u(, __half)?><<<grid, 512u>>>\(',
            lambda match: ('wmma_half_launch(' if match[1] else 'wmma_float_launch(') + 'grid, ',
            text)
        assert count == 2, count
    fixture = (ROOT / 'tests/test_rocm_q4_activation.cpp').read_text()
    return ('#define TEST_NATIVE\n' if native else '') + fixture.replace('// PRODUCTION_SOURCE', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rocm', action='store_true')
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if args.bench and not args.rocm:
        parser.error('--bench requires --rocm')
    with tempfile.TemporaryDirectory(prefix='ds4-q4-activation-') as tmp:
        source = Path(tmp) / 'activation.cpp'
        source.write_text(build_source(args.rocm))
        binary = Path(tmp) / 'activation'
        if args.rocm:
            compiler = shlex.split(os.environ.get('HIPCC', 'hipcc'))
            if not compiler or not shutil.which(compiler[0]):
                parser.error('HIPCC and a gfx1151 AMD GPU are required')
            flags = shlex.split(os.environ.get('ROCM_CFLAGS', '-O3 --offload-arch=gfx1151'))
            subprocess.run(compiler + flags + ['-x', 'hip', '-std=c++17', '-I', str(ROOT),
                str(source), '-o', str(binary)], check=True)
            subprocess.run([str(binary), str(args.device)] + (['--bench'] if args.bench else []), check=True)
        else:
            compiler = shlex.split(os.environ.get('CXX', 'clang++'))
            test_backend_phases(compiler, Path(tmp))
            for flags in (['-O2'], ['-O3', '-ffast-math', '-fno-finite-math-only']):
                print('Q4 activation host ' + ' '.join(flags), flush=True)
                subprocess.run(compiler + flags + ['-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-Wno-unknown-pragmas', '-fsanitize=address,undefined', '-I', str(ROOT),
                    str(source), '-o', str(binary)], check=True)
                subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
