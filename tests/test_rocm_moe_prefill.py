#!/usr/bin/env python3
"""Validate extracted ROCm stable-routing and Q2 WMMA staging kernels.

Default runs strict/fast host address, ordering and dequantization oracles with
ASan/UBSan. --rocm builds the actual HIP kernels, checks guarded baseline/K32
outputs and reports alternating GPU timings. Host tests do not execute WMMA.
"""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]


def source(native):
    kernels = (ROOT / 'rocm/ds4_rocm_moe.cuh').read_text()
    routing = extract_function(kernels, '__global__ static void moe_scatter_sorted_pairs_deterministic_kernel(')
    half = extract_function(kernels, '__device__ static float dev_f16_to_f32(')
    pack = extract_function(kernels, '__device__ __forceinline__ static uint32_t dev_pack_half2_bits(')
    dequant = extract_function(kernels, '__device__ __forceinline__ static void q2_K_dequant_pair_tile_half_rowwise_staged(')
    dequant = 'template <int BN, int BK>\n' + dequant
    wmma = extract_function(kernels, '__global__ static void moe_down_q2K_hotlist_wmma_n2_kernel(')
    wmma = ('template <int MTILES=8, int BM=16, int BN=16, int BK=16, bool MID_F16=false, '
            'bool OUT_F16=false, bool SLOT_MAJOR=false, int STAGE_K=BK>\n' + wmma)
    # C aliases A only after every wave has finished reading the final K
    # stage. Keep these lifetime fences explicit when evolving the kernel.
    inner = extract_function(wmma, '        for (uint32_t krel = 0;')
    assert inner[:-1].rstrip().endswith('__syncthreads();'), 'C/A alias needs the final K-stage fence'
    outer = extract_function(wmma, '    for (uint32_t kb = 0;')
    epilogue = wmma[wmma.index(outer) + len(outer):]
    assert not any(name in epilogue for name in ('shA', 'shB0', 'shB1', 'shW')), 'epilogue must not read dead K scratch'
    first_store = epilogue.index('acc0, BN, rocwmma::mem_row_major)')
    second_store = epilogue.index('acc1, BN, rocwmma::mem_row_major)')
    assert epilogue[first_store:second_store].count('__syncthreads();') >= 2, 'C reuse needs publish and read-completion fences'
    # Exercise the actual cooperative A load body on the host, independently
    # of GPU execution; do not rewrite its loop or address expressions.
    begin = wmma.index('            if (MID_F16) {')
    end = wmma.index('            q2_K_dequant_pair_tile_half_rowwise_staged', begin)
    stage = '''template <bool MID_F16, int STAGE_K>
static void stage_a(half *shA, const uint32_t *shPair, const float *mid,
                   const half *mid_h, uint32_t expert_mid_dim, uint32_t k0, uint32_t tid) {
    constexpr int MTILES=4, BM=16;
''' + wmma[begin:end] + '\n}\n'
    mma_begin = wmma.index('            if (wave < MTILES) {', end)
    mma = extract_function(wmma[mma_begin:], '            if (wave < MTILES) {')
    mma = '''template <int STAGE_K>
static void mma_stage(half *shA, half *shB0, half *shB1, uint32_t wave) {
    constexpr int MTILES=4, BM=16, BK=16;
    MockFragment a{0}, b0{1}, b1{2}, acc0{3}, acc1{4};
''' + mma + '\n}\n'
    layout_begin = wmma.index('    static_assert(BK == 16')
    layout_end = wmma.index('    const uint32_t hot_idx', layout_begin)
    layout = wmma[layout_begin:layout_end].replace('    extern __shared__ __align__(16) unsigned char raw_sh[];\n', '')
    layout = '''template <int STAGE_K>
static void check_layout(unsigned char *raw_sh, size_t bytes) {
    constexpr int MTILES=4, BM=16, BN=16, BK=16;
''' + layout + '''
    check(reinterpret_cast<unsigned char*>(shA)==raw_sh, "A starts at LDS base");
    check(reinterpret_cast<unsigned char*>(shB0)==raw_sh+64*STAGE_K*2, "B0 LDS offset");
    check(reinterpret_cast<unsigned char*>(shB1)==raw_sh+(64+16)*STAGE_K*2, "B1 LDS offset");
    check(reinterpret_cast<unsigned char*>(shW)+32*84<=raw_sh+bytes, "raw Q2 LDS bounds");
    check(reinterpret_cast<unsigned char*>(shC)+64*16*4<=raw_sh+bytes, "C LDS bounds");
    check(STAGE_K==32 ? reinterpret_cast<void*>(shC)==shA : reinterpret_cast<void*>(shC)==shW+32*21,
          "only K32 aliases epilogue C to A");
    for(size_t i=0;i<64*16;++i) shC[i]=float(i); // verify alias writable within exact scratch
}\n'''
    text = '\n'.join((routing, half, pack, dequant))
    if native:
        text += '\n' + wmma
    else:
        text = text.replace('__global__', '').replace('__device__', '').replace('__forceinline__', 'inline')
        text += '\n' + stage + '\n' + mma + '\n' + layout
    fixture = (ROOT / 'tests/test_rocm_moe_prefill.cpp').read_text()
    return ('#define TEST_NATIVE\n' if native else '') + fixture.replace('// PRODUCTION_SOURCE', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rocm', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--bench', action='store_true', help='native full-shape alternating kernel timings')
    args = parser.parse_args()
    if args.bench and not args.rocm:
        parser.error('--bench requires --rocm')
    with tempfile.TemporaryDirectory(prefix='ds4-rocm-moe-prefill-') as tmp:
        src = Path(tmp) / 'moe.cpp'
        src.write_text(source(args.rocm))
        binary = Path(tmp) / 'moe'
        if args.rocm:
            compiler = shlex.split(os.environ.get('HIPCC', 'hipcc'))
            if not compiler or not shutil.which(compiler[0]):
                parser.error('HIPCC and an AMD GPU are required; native validation cannot be skipped')
            flags = shlex.split(os.environ.get('ROCM_CFLAGS', '-O3'))
            subprocess.run(compiler + flags + ['-x', 'hip', '-std=c++17', str(src), '-o', str(binary)], check=True)
            subprocess.run([str(binary), str(args.device)] + (['--bench'] if args.bench else []), check=True)
        else:
            compiler = shlex.split(os.environ.get('CXX', 'clang++'))
            for flags in (['-O2'], ['-O3', '-ffast-math', '-fno-finite-math-only']):
                print('ROCm MoE host ' + ' '.join(flags), flush=True)
                subprocess.run(compiler + flags + ['-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-Wno-unknown-pragmas', '-fno-strict-aliasing', '-fsanitize=address,undefined',
                    str(src), '-o', str(binary)], check=True)
                subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
