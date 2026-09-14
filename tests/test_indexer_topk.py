#!/usr/bin/env python3
"""Host-check real Metal compact merges, causal/shuffle leaves and geometry."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]


def actual_leaf_source(source):
    leaf = extract_function(source, 'kernel void kernel_argsort_f32_i32(')
    start = leaf.index('    const int width =')
    end = leaf.index('    const int work_width =', start)
    width = leaf[start:end].replace('return;', 'return {};')
    start = leaf.index('    for (int k = 2;')
    end = leaf.index('    const int64_t i0 =', start)
    network = leaf[start:end]
    # Simulate each SIMD shuffle from a complete snapshot of the previous
    # lane registers. Keep the actual comparison/exchange statements, and
    # complete the shared-memory reload for every lane before the snapshot.
    shuffle = extract_function(network, 'if (shuffle && j < 32)')
    reload = extract_function(shuffle, 'if (k > 32 && j == 16)')
    exchange = shuffle[shuffle.index('{')+1:shuffle.rindex('continue;')]
    exchange = exchange.replace(reload, '')
    exchange = exchange.replace('simd_shuffle_xor(reg_idx, j)', 'before_indices[col ^ j]')
    exchange = exchange.replace('simd_shuffle_xor(reg_value, j)', 'before_values[col ^ j]')
    registers = 'int &reg_idx = reg_indices[col]; float &reg_value = reg_values[col];'
    network = network.replace(shuffle, '''if (shuffle && j < 32) {
                for (int col = 0; col < int(nth); ++col) {
                    ''' + registers + reload + '''
                }
                const auto before_indices = reg_indices;
                const auto before_values = reg_values;
                for (int col = 0; col < int(nth); ++col) {
                    ''' + registers + exchange + '''
                }
                continue;
            }''')
    publish = extract_function(network, 'if (shuffle && k >= 32)')
    publish_body = publish[publish.index('{')+1:publish.rindex('}')]
    publish_body = publish_body.replace('reg_idx', 'reg_indices[col]')
    publish_body = publish_body.replace('threadgroup_barrier(mem_flags::mem_threadgroup);', '')
    network = network.replace(publish, 'if (shuffle && k >= 32) {\n'
        '            for (int col = 0; col < int(nth); ++col) {' + publish_body + '\n            }\n        }')
    # Compare pairs are disjoint within each stage. Execute all real lane
    # bodies before advancing to the next GPU barrier.
    network = network.replace('            int ixj = col ^ j;',
        '            for (int col = 0; col < int(nth); ++col) {\n'
        '            int ixj = col ^ j;')
    network = network.replace('            threadgroup_barrier(mem_flags::mem_threadgroup);',
                              '            }')
    assert 'threadgroup_barrier' not in network and 'simd_shuffle_xor' not in network
    return r'''
enum { DS4_SORT_ORDER_ASC, DS4_SORT_ORDER_DESC };
#define SWAP(a,b) std::swap(a,b)
static std::vector<int32_t> actual_leaf(const float *scores, uint32_t n, uint32_t begin, uint32_t nth,
        bool shuffle = false, uint32_t causal_start = 0, uint32_t causal_ratio = 0, uint32_t row = 0) {
    const struct { int ne00; uint32_t causal_start, causal_ratio; } args = {int(n),causal_start,causal_ratio};
    const struct { int x; } ntg = {int(nth)};
    constexpr auto order = DS4_SORT_ORDER_DESC;
    const bool causal = causal_ratio != 0;
    const int i00 = int(begin), i01 = int(row);
''' + width + r'''
    std::vector<int32_t> shmem_i32(nth);
    std::vector<float> shmem_f32(nth, std::numeric_limits<float>::quiet_NaN());
    std::vector<int32_t> reg_indices(nth);
    std::vector<float> reg_values(nth, 0.0f);
    for (uint32_t i = 0; i < nth; ++i) {
        shmem_i32[i] = reg_indices[i] = int32_t(begin+i);
        if (begin+i < uint32_t(width)) shmem_f32[i] = reg_values[i] = scores[begin+i];
    }
''' + network + r'''
    if (shuffle) shmem_i32 = std::move(reg_indices);
    shmem_i32.resize(std::min(nth,uint32_t(width)-begin));
    return shmem_i32;
}
'''


def main():
    source = (ROOT / "metal/argsort.metal").read_text()
    name = "kernel_argsort_merge_f32_i32_desc_compact"
    kernel = extract_function(source, 'kernel void ' + name + '(')
    body = kernel[kernel.index('{'):].replace("device ", "")
    constants = extract_function(source, 'struct ds4_metal_args_argsort_compact_merge {') + ';\n'
    generated = """// Extracted production Metal body; only address spaces are removed.
static void actual_compact_merge(const ds4_indexer_topk_merge_args &args,
    const char *src0, const int32_t *tmp, int32_t *dst,
    uint3 tgpig, ushort3 tpitg, ushort3 ntg)
""" + body + "\n" + constants + actual_leaf_source(source)
    compiler = shlex.split(os.environ.get("CXX", "clang++"))
    if not compiler:
        raise SystemExit("CXX must name a C++ compiler")
    with tempfile.TemporaryDirectory(prefix="ds4-indexer-topk-") as directory:
        temp = Path(directory)
        (temp / "indexer_topk_actual.h").write_text(generated)
        for label, flags in (("strict", ["-ffp-contract=off"]),
                             ("fast", ["-ffast-math", "-fno-finite-math-only"])):
            exe = temp / label
            command = compiler + ["-std=c++17", "-O2", "-g", "-Wall", "-Wextra", "-Werror",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer"] + flags + [
                "-I", str(temp), "-I", str(ROOT), str(ROOT / "tests/test_indexer_topk.cpp"),
                "-o", str(exe)]
            subprocess.run(command, check=True)
            subprocess.run([str(exe)], check=True)
            print(f"PASS: {label} actual-source merge + geometry + causal/shuffle leaves, ASan/UBSan", flush=True)


if __name__ == "__main__":
    main()
