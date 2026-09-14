"""Exercise MMQ's actual dense write-back and shared-ID initialization as C.

The CUDA DP4A and MMA producer epilogues and 16x8 accumulator mapping are
extracted from production. This checks destination bits, indexing, tails and
identity-table elision. The gfx1151 Y loaders are also exercised with exact
input allocations; this does not execute MMA instructions or GPU barriers.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]
MMQ = (ROOT / 'cuda/mmq/mmq.cuh').read_text()
MMA = (ROOT / 'cuda/mmq/mma.cuh').read_text()


def translated_y_loads():
    process = extract_function(MMQ, 'static __device__ __forceinline__ void mul_mat_q_process_tile(')
    seed = extract_function(process, 'if (fast_y_loads && !full_y_tile) {')
    marker = 'if constexpr (fast_y_loads) {'
    assert process.count(marker) == 2
    output = ''
    for stage, body in enumerate((seed,) + tuple(
            extract_function(marker + part, marker)
            for part in process.split(marker)[1:])):
        output += (f'static void y_stage_{stage}(int *tile_y, const int *by0, '
                   'int nthreads, int linear_tid, int full_y_tile, int valid_y_words) {\n'
                   '    const int fast_y_loads = 1;\n'
                   '    enum { MMQ_TILE_Y_K = 36 };\n' + body + '\n}\n')
    return output.replace('if constexpr', 'if').replace('#pragma unroll', '')


def translated_functions():
    kernel = extract_function(MMQ, 'static __global__ void mul_mat_q(')
    # All three producer calls must preserve the nullable mapping through
    # ordinary tiling, complete Stream-K tiles and trailing partial tiles.
    calls = re.findall(r'mul_mat_q_process_tile<[^>]+>\s*\(([^;]+)\);', kernel)
    assert len(calls) == 3
    for call in calls:
        assert call.split(',')[3].strip() == 'tile_ids'
    process = extract_function(MMQ, 'static __device__ __forceinline__ void mul_mat_q_process_tile(')
    assert 'write_back(sum, nullptr, tmp_fixup' in process
    # Shared-ID storage remains disjoint from both MMA input tiles.
    assert 'int * tile_y = data_mul_mat_q + mmq_x;' in process
    assert 'int * tile_x = tile_y + GGML_PAD(' in process
    start = kernel.index('    const int32_t * tile_ids =')
    stop = kernel.index('    // On non-CDNA AMD', start)
    init = kernel[start:stop]
    assert init.count('__syncthreads();') == 1 and 'if (ids_dst)' in init
    init_fn = ('static const int32_t * initialize_ids(const int32_t *ids_dst) {\n' +
               '    const int nwarps = mmq_get_nwarps_device();\n' +
               '    const int warp_size = ggml_cuda_get_physical_warp_size();\n' +
               init + '    return tile_ids;\n}\n').replace('nullptr', 'NULL')

    # The first generic tile specialization's final branch is NVIDIA's
    # I-major mapping. Keep its actual expressions for a 16x8 accumulator.
    tile = MMA[MMA.index('struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {'):]
    nvidia = tile[tile.index('#else\n        static constexpr int ne = I * J / 32;'):]
    nvidia = nvidia[:nvidia.index('#endif // defined(GGML_USE_HIP)')]
    mapping = ''
    for axis in ('i', 'j'):
        func = extract_function(nvidia, f'static __device__ __forceinline__ int get_{axis}(')
        func = func.replace(f'get_{axis}(', f'tile_{axis}(')
        func = func.replace(f'tile<16, 8, T>::tile_{axis}(', f'tile_{axis}(')
        mapping += func.replace('if constexpr', 'if') + '\n'
    output = init_fn + mapping
    for name in ('dp4a', 'mma'):
        func = extract_function(MMQ, f'static __device__ __forceinline__ void mmq_write_back_{name}(')
        func = func.replace('constexpr', 'const')
        func = func.replace('typedef tile<16, 8, int> tile_C;', '')
        func = func.replace('static_assert(', '_Static_assert(')
        for old, new in [('tile_C::I', '16'), ('tile_C::J', '8'),
                         ('tile_C::ne', '4'), ('tile_C::get_i', 'tile_i'),
                         ('tile_C::get_j', 'tile_j')]:
            func = func.replace(old, new)
        # mmq_y and nwarps are test geometry variables instead of template
        # constants in the C host shim; the same equality is checked at runtime.
        func = func.replace('_Static_assert(nwarps*16 == mmq_y,', 'check(nwarps*16 == mmq_y,')
        func = func.replace('nullptr', 'NULL')
        output += func + '\n'
    # The C shim sweeps runtime tile sizes rather than template constants;
    # GPU unroll directives do not apply to this host address-only oracle.
    return output.replace('#pragma unroll', '')


PREAMBLE = r'''
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define __device__
#define __forceinline__ inline
#define TURING_MMA_AVAILABLE
#define NO_DEVICE_CODE abort()
#define GGML_UNUSED(x) (void)(x)
enum {I = 16, J = 8, GUARD = 32, LIMIT = 128};
static struct { int x, y; } threadIdx;
static int mmq_x, mmq_y = 128, need_check, barriers;
static int ids_dst_shared[LIMIT];
static void __syncthreads(void) { ++barriers; }
#define __syncthreads() __syncthreads()
static int ggml_cuda_get_physical_warp_size(void) { return 32; }
static int mmq_get_nwarps_device(void) { return 8; }
static int mmq_get_granularity_device(int width) { return width >= 48 ? 16 : 8; }
static void check(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL MMQ dense IDs: %s\n", what); exit(1); }
}
'''

DRIVER = r'''
static void test_y_loads(void) {
    const int widths[] = {32, 48, 64, 80, 96, 112, 128};
    const int thread_counts[] = {64, 128, 256};
    const int poison = 0x5a1930fb;
    unsigned cases = 0;
    for (unsigned w = 0; w < sizeof(widths) / sizeof(widths[0]); ++w)
    for (unsigned t = 0; t < sizeof(thread_counts) / sizeof(thread_counts[0]); ++t)
    for (int tail = 0; tail < 4; ++tail) {
        mmq_x = widths[w];
        const int nthreads = thread_counts[t], words = mmq_x * 36;
        const int padded = (words + nthreads - 1) / nthreads * nthreads;
        const int cols = tail == 0 ? 0 : tail == 1 ? 1 : tail == 2 ? mmq_x-1 : mmq_x;
        const int valid = cols * 36, full = cols == mmq_x;
        // No readable input padding: ASan catches a full-tile overread even
        // when the production pool would mask it with an oversized allocation.
        int *input = valid ? malloc((size_t)valid * sizeof(int)) : NULL;
        int *storage = malloc((size_t)(padded + 2*GUARD) * sizeof(int));
        check(storage && (!valid || input), "Y load allocation");
        int *tile = storage + GUARD;
        for (int i = 0; i < padded + 2*GUARD; ++i) storage[i] = poison;
        for (int tid = 0; tid < nthreads; ++tid)
            y_stage_0(tile, input, nthreads, tid, full, valid);
        // Alternate both actual K-stage loaders over changing input. Partial
        // tiles must retain the suffix initialized once before the K loop.
        for (int stage = 0; stage < 6; ++stage) {
            for (int i = 0; i < valid; ++i) input[i] = 7919 * stage + i + 1;
            for (int tid = 0; tid < nthreads; ++tid) {
                if (stage & 1) y_stage_2(tile, input, nthreads, tid, full, valid);
                else y_stage_1(tile, input, nthreads, tid, full, valid);
            }
            for (int i = 0; i < padded; ++i) {
                const int expected = i < valid ? input[i] : i < words || full ? 0 : poison;
                check(tile[i] == expected, "Y values, zero suffix and shared padding");
            }
            for (int i = 0; i < GUARD; ++i)
                check(storage[i] == poison && storage[GUARD+padded+i] == poison,
                      "Y shared prefix and suffix guards");
        }
        free(storage); free(input); ++cases;
    }
    printf("PASS gfx1151 Y loads host: %u exact-allocation shapes, both production K stages, guards and initialization.\n", cases);
}

static void initialize_test(void) {
    for (int routed = 0; routed < 2; ++routed) {
        for (int j = 0; j < LIMIT; ++j) ids_dst_shared[j] = -701;
        barriers = 0;
        for (threadIdx.y = 0; threadIdx.y < 8; ++threadIdx.y)
        for (threadIdx.x = 0; threadIdx.x < 32; ++threadIdx.x) {
            const int32_t *ptr = initialize_ids(routed ? ids_dst_shared : NULL);
            check(ptr == (routed ? ids_dst_shared : NULL), "nullable pointer admission");
        }
        check(barriers == (routed ? 256 : 0), "dense barrier removed; routed publication retained");
        for (int j = 0; j < LIMIT; ++j)
            check(ids_dst_shared[j] == (routed && j < mmq_x ? j : -701),
                  "dense table untouched; routed identity and guards");
    }
}
static void write_back(int mma, const float *sum, const int *ids, float *dst,
                       int stride, int last_row, int last_col) {
    if (mma) mmq_write_back_mma(sum, ids, dst, stride, last_row, last_col);
    else mmq_write_back_dp4a(sum, ids, dst, stride, last_row, last_col);
}
static void run_case(int mma, int rows, int cols, int grouped_stride, unsigned salt) {
    const int stride = grouped_stride ? 8*mmq_y+17 : mmq_y;
    const size_t values = (size_t)stride * mmq_x + 2*GUARD;
    const uint32_t poison = 0x6a981234;
    float *old = malloc(values * 4), *dense = malloc(values * 4), *routed = malloc(values * 4);
    int *identity = malloc((size_t)cols * sizeof(int)), *mapping = malloc((size_t)cols * sizeof(int));
    check(old && dense && routed && identity && mapping, "allocate");
    for (size_t i = 0; i < values; ++i) {
        memcpy(old + i, &poison, 4); memcpy(dense + i, &poison, 4); memcpy(routed + i, &poison, 4);
    }
    for (int j = 0; j < cols; ++j) { identity[j] = j; mapping[j] = cols - 1 - j; }
    need_check = rows < mmq_y;
    float sums[LIMIT*128/256];
    const int per_thread = mmq_x*mmq_y/256;
    for (threadIdx.y = 0; threadIdx.y < 8; ++threadIdx.y)
    for (threadIdx.x = 0; threadIdx.x < 32; ++threadIdx.x) {
        for (int j = 0; j < per_thread; ++j) {
            // Preserve all payload bits including signed zeros, NaNs and
            // infinities: write-back must only select an address.
            uint32_t value = 0x3f000000u + 7919u * (salt + j + 97u*threadIdx.y + threadIdx.x);
            if ((j + threadIdx.x) % 17 == 0) value = 0x7fc12345u;
            if ((j + threadIdx.x) % 19 == 0) value = 0x80000000u;
            if ((j + threadIdx.x) % 23 == 0) value = 0x7f800000u;
            memcpy(sums+j, &value, 4);
        }
        write_back(mma, sums, identity, old+GUARD, stride, rows-1, cols-1);
        write_back(mma, sums, NULL, dense+GUARD, stride, rows-1, cols-1);
        write_back(mma, sums, mapping, routed+GUARD, stride, rows-1, cols-1);
    }
    check(!memcmp(old, dense, values*4), "dense nullptr is bitwise identity");
    for (int j = 0; j < mmq_x; ++j)
    for (int i = 0; i < stride; ++i) {
        const size_t at = GUARD + (size_t)j*stride + i;
        if (j < cols && i < rows) {
            uint32_t word; memcpy(&word, dense+at, 4);
            check(word != poison, "complete destination coverage");
            check(!memcmp(old+at, routed+GUARD+(size_t)mapping[j]*stride+i, 4),
                  "routed scatter preserved");
        } else {
            check(!memcmp(dense+at, &poison, 4), "dense row/column tail");
            check(!memcmp(routed+at, &poison, 4), "routed row/column tail");
        }
    }
    for (int i = 0; i < GUARD; ++i) {
        check(!memcmp(dense+i, &poison, 4) && !memcmp(dense+values-1-i, &poison, 4), "dense guards");
        check(!memcmp(routed+i, &poison, 4) && !memcmp(routed+values-1-i, &poison, 4), "routed guards");
    }
    free(mapping); free(identity); free(routed); free(dense); free(old);
}
int main(void) {
    test_y_loads();
    const int tiles[] = {8, 16, 24, 32, 48, 64, 80, 96, 112, 128};
    const int row_tails[] = {1, 17, 63, 127, 128};
    unsigned cases = 0;
    for (unsigned t = 0; t < sizeof(tiles)/sizeof(tiles[0]); ++t) {
        mmq_x = tiles[t];
        initialize_test();
        for (int mma = 0; mma < 2; ++mma)
        for (int grouped = 0; grouped < 2; ++grouped)
        for (unsigned r = 0; r < sizeof(row_tails)/sizeof(row_tails[0]); ++r)
        for (int c = 0; c < 3; ++c) {
            int cols = c == 0 ? 1 : c == 1 ? mmq_x-1 : mmq_x;
            run_case(mma, row_tails[r], cols, grouped, cases++);
        }
    }
    printf("PASS MMQ dense IDs host: %u actual DP4A/MMA epilogue cases, initializer, "
           "three tile dispatch sites, scatter, tails and guards. GPU unverified.\n", cases);
}
'''


def main():
    with tempfile.TemporaryDirectory(prefix='ds4-mmq-dense-ids-') as tmp:
        source, binary = Path(tmp) / 'ids.c', Path(tmp) / 'ids'
        source.write_text(PREAMBLE + translated_functions() + translated_y_loads() + DRIVER)
        for flags in (['-O2'], ['-O3', '-ffast-math']):
            subprocess.run(shlex.split(os.environ.get('CC', 'cc')) + flags +
                           ['-std=c11', '-Wno-unknown-pragmas', '-fsanitize=address,undefined',
                            str(source), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
