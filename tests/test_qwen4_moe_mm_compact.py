#!/usr/bin/env python3
"""CPU regression test for the Qwen MoE MM active-expert dispatch plan.

Extracts the real host/shader structs, mapping helpers, and empty guards.
Compiles a CPU-only C++ oracle in a temporary directory; never loads Metal,
opens a model, or touches a running server. Run with --sanitize for ASan/UBSan.
"""
from pathlib import Path
import argparse
import json
import os
import re
import shlex
import subprocess
import tempfile


def extract(text, marker):
    start = text.index(marker)
    begin = text.index('{', start)
    depth = 1
    for end in range(begin + 1, len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if depth == 0:
            return text[start:end + 1]
    raise AssertionError(marker)


MAPPED_BODIES = (
    "static inline void qwen4_moe_mm_mid_impl(",
    "kernel void kernel_qwen4_moe_mm_down(",  # shared float/half template
    "kernel void kernel_qwen4_moe_mm_mid_nax_t(",
    "kernel void kernel_qwen4_moe_mm_down_nax_t(",
)
MID_WRAPPERS = (
    ("kernel void kernel_qwen4_moe_mm_mid(", "float", "false", "nullptr"),
    ("kernel void kernel_qwen4_moe_mm_mid_f16(", "half", "true", "mid_shadow"),
)


def check_shader_mapping(metal):
    # Ignore comments: a stale example must not satisfy a dispatch safety check.
    metal = re.sub(r'/\*.*?\*/|//[^\n]*', '', metal, flags=re.S)
    for marker in MAPPED_BODIES:
        body = extract(metal, marker)
        mapping = re.search(r'\be\s*=\s*qwen4_moe_mm_expert\(args, tgpig\.y\)', body)
        guard = re.search(r'if\s*\(e\s*>=\s*args\.n_expert\)\s*return\s*;', body)
        reads = list(re.finditer(r'\bcounts\s*\[([^]]+)\]', body))
        assert body.count('qwen4_moe_mm_expert(') == 1, marker
        assert mapping and guard and reads, marker
        assert mapping.end() < guard.start() < guard.end() < reads[0].start(), marker
        assert all(read[1].strip() == 'e' for read in reads), marker

    # Both entry points must reach that same guarded implementation, with
    # the original compact-dispatch args/grid and the proper shadow operand.
    for marker, scalar, shadow_enabled, shadow in MID_WRAPPERS:
        body = extract(metal, marker)
        calls = re.findall(r'qwen4_moe_mm_mid_impl\s*<([^>]+)>\s*\(([^;]+)\);', body)
        assert len(calls) == 1, marker
        template, arguments = calls[0]
        assert [part.strip() for part in template.split(',')] == ['NT', scalar, shadow_enabled], marker
        expected = ['args', 'gate_base', 'up_base', 'lists', 'counts', 'x',
                    'mid', shadow, 'Ag', 'Au', 'Bs', 'Cs', 'tgpig', 'tid', 'sgitg']
        assert [part.strip() for part in arguments.split(',')] == expected, marker
        assert not re.search(r'\bcounts\s*\[|qwen4_moe_mm_expert\(', body), marker

    # Half down instantiates the shared guarded down template, not another
    # kernel body. Keep all 8/16/32/64-token instances on that implementation.
    down_half = metal.split('#define QWEN4_MOE_MM_DOWN_F16_INSTANCE(NT_)', 1)[1]
    down_half = down_half.split('#undef QWEN4_MOE_MM_DOWN_F16_INSTANCE', 1)[0]
    assert 'kernel_qwen4_moe_mm_down<NT_, half>(' in down_half
    assert re.findall(r'QWEN4_MOE_MM_DOWN_F16_INSTANCE\((\d+)\)', down_half) == ['1', '2', '4', '8']


def check_scanner_negative_controls(metal):
    """Ensure realistic dispatch regressions are rejected by the source check."""
    rejected = 0

    def reject(original, replacement, label):
        nonlocal rejected
        assert original in metal and replacement != original, label
        mutant = metal.replace(original, replacement, 1)
        try:
            check_shader_mapping(mutant)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError(f"scanner accepted mutation: {label}")

    for marker in MAPPED_BODIES:
        body = extract(metal, marker)
        mapping = 'qwen4_moe_mm_expert(args, tgpig.y)'
        guard = 'if (e >= args.n_expert) return;'
        count = 'const uint count = (uint)counts[e];'
        reject(body, body.replace(mapping, 'tgpig.y', 1), f'{marker}: dense expert ID')
        reject(body, body.replace(guard, '', 1), f'{marker}: missing bounds guard')
        late_guard = body.replace(guard, '', 1).replace(count, count + '\n' + guard, 1)
        reject(body, late_guard, f'{marker}: late bounds guard')
        reject(body, body.replace('counts[e]', 'counts[tgpig.y]', 1), f'{marker}: compact counts index')
    for marker, scalar, shadow_enabled, shadow in MID_WRAPPERS:
        body = extract(metal, marker)
        reject(body, body.replace('lists, counts, x', 'counts, lists, x', 1), f'{marker}: swapped buffers')
        reject(body, body.replace('Cs, tgpig, tid', 'Cs, uint3(0), tid', 1), f'{marker}: lost grid index')
        reject(body, body.replace(f'<NT, {scalar}, {shadow_enabled}>', '<NT, float, true>', 1),
               f'{marker}: wrong template arguments')
        wrong_shadow = 'mid_shadow' if shadow == 'nullptr' else 'nullptr'
        reject(body, body.replace(f'mid, {shadow}, Ag', f'mid, {wrong_shadow}, Ag', 1),
               f'{marker}: wrong shadow buffer')
    reject('kernel_qwen4_moe_mm_down<NT_, half>(', 'kernel_qwen4_moe_mm_down<NT_, float>(',
           'half down: wrong template')
    for nt in (1, 2, 4, 8):
        reject(f'QWEN4_MOE_MM_DOWN_F16_INSTANCE({nt})', '', f'half down: missing NT={nt}')
    return rejected


def extract_cpu_code(host, metal):
    host_args = re.search(r'typedef struct \{\n    uint32_t n_tokens, n_slots, n_out, in_dim, out_rows, weight_type, row_bytes, list_cap;.*?\} qwen4_moe_mm_args;', host, re.S).group(0)
    metal_args = extract(metal, 'struct ds4_metal_args_qwen4_moe_mm') + ';'
    helper = extract(host, 'static uint32_t qwen4_moe_mm_copy_active(')
    dispatch = extract(host, 'static uint32_t qwen4_moe_mm_dispatch_experts(')
    gpu_map = extract(metal, 'static inline uint qwen4_moe_mm_expert(')
    gpu_block = extract(metal, 'static inline uint2 qwen4_moe_mm_block(')
    mid_wrapper = extract(host, 'int ds4_gpu_qwen4_moe_mm_mid_tensor(')
    down_wrapper = extract(host, 'int ds4_gpu_qwen4_moe_mm_down_tensor(')
    mid_empty = extract(mid_wrapper, 'if (!dispatch_experts)')
    down_empty = extract(down_wrapper, 'if (!dispatch_experts)')

    code = r'''// Extracted from candidate host and shader; no GPU API is used.
    #include <cstdint>
    #include <cstddef>
    #include <climits>
    using uint = uint32_t;
    #define constant const
    struct uint2 { uint x, y; uint2(uint a, uint b) : x(a), y(b) {} };
    struct uint3 { uint x, y, z; };
    HOST_ARGS
    METAL_ARGS
    struct stream_table_stub { uint32_t n_total_expert; };
    struct half_pass_stub { const void *mid; };
    struct qwen4_stream_weights {
        const stream_table_stub *table;
        const uint32_t *frequency;
        const int32_t *mm_pass_counts;
        half_pass_stub *half_pass;
    };
    static qwen4_stream_weights *g_qwen4_stream_weights;
    HOST_HELPER
    HOST_DISPATCH
    GPU_MAP
    GPU_BLOCK
    static const void *g_qwen4_nax_half_mid_for;
    static uint64_t g_qwen4_nax_half_mid_count;
    static int empty_mid_guard(uint32_t dispatch_experts) {
    MID_EMPTY
        return 0;
    }
    static int empty_down_guard(uint32_t dispatch_experts) {
    DOWN_EMPTY
        return 0;
    }
    '''
    for key, value in [('HOST_ARGS', host_args), ('METAL_ARGS', metal_args),
                       ('HOST_HELPER', helper), ('HOST_DISPATCH', dispatch),
                       ('GPU_MAP', gpu_map), ('GPU_BLOCK', gpu_block),
                       ('MID_EMPTY', mid_empty), ('DOWN_EMPTY', down_empty)]:
        code = code.replace(key, value)

    check_shader_mapping(metal)
    for wrapper in (mid_wrapper, down_wrapper):
        assert "dispatch_experts == UINT32_MAX" in wrapper
        assert wrapper.index("if (!dispatch_experts)") < wrapper.index("qwen4_dispatch(")
        for line in wrapper.splitlines():
            if "qwen4_moe_mm_grid(" in line or "tail_grid = MTLSizeMake(" in line:
                assert "dispatch_experts" in line, line
        assert "&args, sizeof(args)" in wrapper
    assert "setBytes:args length:args_len atIndex:0" in host
    return code


CPU_ORACLE = r'''
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <array>

static_assert(sizeof(qwen4_moe_mm_args) == 2112);
static_assert(sizeof(qwen4_moe_mm_args) <= 4096);
static_assert(sizeof(qwen4_moe_mm_args) == sizeof(ds4_metal_args_qwen4_moe_mm));
#define SAME_FIELD(F) static_assert(offsetof(qwen4_moe_mm_args, F) == offsetof(ds4_metal_args_qwen4_moe_mm, F));
SAME_FIELD(n_tokens) SAME_FIELD(n_slots) SAME_FIELD(n_out) SAME_FIELD(in_dim)
SAME_FIELD(out_rows) SAME_FIELD(weight_type) SAME_FIELD(row_bytes) SAME_FIELD(list_cap)
SAME_FIELD(expert_bytes) SAME_FIELD(n_expert) SAME_FIELD(tiles_per_launch)
SAME_FIELD(tail_base) SAME_FIELD(expert_major) SAME_FIELD(n_active_expert) SAME_FIELD(active_expert)
static_assert(offsetof(qwen4_moe_mm_args, n_active_expert) == 56);
static_assert(offsetof(qwen4_moe_mm_args, active_expert) == 60);

static uint64_t mapping_cases, scheduled_cases, schedule_events, lifetime_cases, empty_state_cases, half_state_cases;
static stream_table_stub table{512};

struct mapped_args { ds4_metal_args_qwen4_moe_mm args; uint32_t dispatch_count; };
static mapped_args encode(const uint32_t *frequency, const int32_t *pass, uint32_t n = 512) {
    table.n_total_expert = n;
    qwen4_stream_weights scope{&table, frequency, pass, nullptr};
    g_qwen4_stream_weights = &scope;
    qwen4_moe_mm_args host{};
    host.n_expert = n;
    const uint32_t count = qwen4_moe_mm_dispatch_experts(&host);
    mapped_args captured{};
    // Simulate the immediate full-byte copy made by Metal setBytes.
    memcpy(&captured.args, &host, sizeof(host));
    captured.dispatch_count = count;
    g_qwen4_stream_weights = nullptr;
    return captured;
}

static void check_map(const uint32_t *frequency, const int32_t *pass, uint32_t n = 512) {
    uint32_t frequency_before[512];
    memcpy(frequency_before, frequency, n * sizeof(uint32_t));
    const mapped_args mapped = encode(frequency, pass, n);
    assert(mapped.dispatch_count != UINT32_MAX);
    std::vector<uint32_t> expected, actual;
    for (uint32_t e = 0; e < n; ++e) if (frequency[e] && (!pass || pass[e])) expected.push_back(e);
    for (uint32_t y = 0; y < mapped.dispatch_count; ++y) actual.push_back(qwen4_moe_mm_expert(mapped.args, y));
    assert(actual == expected);
    if (mapped.dispatch_count && mapped.dispatch_count < n)
        assert(qwen4_moe_mm_expert(mapped.args, mapped.dispatch_count) == n);
    if (mapped.dispatch_count == n) assert(mapped.args.n_active_expert == 0);
    assert(!memcmp(frequency_before, frequency, n * sizeof(uint32_t)));
    mapping_cases++;
}

using event = std::array<uint32_t, 5>; // original expert, row-block, tile index, list start, list count
static std::vector<event> schedule(mapped_args mapped, const int32_t *counts,
                                   uint32_t rows, uint32_t row_tile, uint32_t tt,
                                   uint32_t tiles, uint32_t tail_base, bool expert_major,
                                   bool nax) {
    auto &args = mapped.args;
    args.out_rows = rows; args.tiles_per_launch = tiles; args.expert_major = expert_major;
    const uint32_t rb_count = (rows + row_tile - 1u) / row_tile;
    const uint32_t grid_x = expert_major ? rb_count * tiles : rb_count;
    const uint32_t grid_z = expert_major ? 1 : tiles;
    std::vector<event> result;
    for (uint32_t y = 0; y < mapped.dispatch_count; ++y) {
        const uint32_t e = qwen4_moe_mm_expert(args, y);
        assert(e < args.n_expert);
        const uint32_t count = counts[e];
        if (!count) continue;
        uint32_t work = count, start = 0;
        if (tail_base) {
            const uint32_t remainder = count % tail_base;
            const uint32_t tail_tt = nax ? (remainder <= 32 ? 32 : 64) :
                (remainder <= 8 ? 8 : remainder <= 16 ? 16 : remainder <= 32 ? 32 : 64);
            if (tt < tail_base) {
                if (!remainder || tail_tt != tt) continue;
                start = count - remainder; work = remainder;
            } else if (remainder && tail_tt < tt) work = count - remainder;
        }
        for (uint32_t z = 0; z < grid_z; ++z) for (uint32_t x = 0; x < grid_x; ++x) {
            uint32_t rb, tile0;
            if (!nax) {
                const uint2 block = qwen4_moe_mm_block(args, uint3{x,y,z});
                rb = block.x; tile0 = block.y;
            } else {
                rb = expert_major ? x % rb_count : x;
                tile0 = expert_major ? x / rb_count : z;
            }
            for (uint32_t tile = tile0; tile * tt < work; tile += tiles) {
                result.push_back(event{e, rb, tile, start + tile * tt, std::min(tt, work - tile * tt)});
            }
        }
    }
    return result;
}

int main() {
    uint32_t full[512]{};
    int32_t resident[512]{}, missing[512]{}, empty[512]{};
    const uint32_t counts[] = {1,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129};
    for (uint32_t i = 0; i < 16; ++i) {
        const uint32_t e = i == 15 ? 511 : i * 31;
        full[e] = counts[i];
        (i & 1 ? resident : missing)[e] = counts[i];
    }
    check_map(full, nullptr); // cold/unsplit selected experts
    check_map(full, resident); check_map(full, missing); // mixed independent passes
    check_map(full, empty); // an all-cold resident pass or all-resident missing pass
    int32_t warm[512]; for (uint32_t e = 0; e < 512; ++e) warm[e] = full[e];
    check_map(full, warm); // all resident
    uint32_t zero[512]{}; check_map(zero, nullptr); check_map(zero, empty);
    uint32_t dense[512]; int32_t dense_pass[512];
    for (uint32_t e = 0; e < 512; ++e) dense[e] = dense_pass[e] = e + 1;
    check_map(dense, nullptr); check_map(dense, dense_pass);
    for (uint32_t n = 1; n <= 512; ++n) check_map(dense, dense_pass, n);

    // Every subset of an eight-expert fixture, including empty and all-active.
    for (uint32_t mask = 0; mask < 256; ++mask) {
        uint32_t f[512]{}; int32_t r[512]{}, m[512]{};
        for (uint32_t e = 0; e < 8; ++e) { f[e] = e + 1; (mask & (1u << e) ? r : m)[e] = f[e]; }
        check_map(f, r, 8); check_map(f, m, 8);
    }

    // Reject malformed CPU pass metadata rather than dispatching truncated
    // list prefixes or experts that have no validated routing frequency.
    int32_t bad[512]{};
    bad[0] = -1; assert(encode(full, bad).dispatch_count == UINT32_MAX);
    bad[0] = 2; assert(encode(full, bad).dispatch_count == UINT32_MAX);
    bad[0] = 0; bad[1] = 1; assert(encode(full, bad).dispatch_count == UINT32_MAX);
    uint32_t scratch[512]; assert(qwen4_moe_mm_copy_active(scratch, full, nullptr, 513) == UINT32_MAX);
    // Nonstreamed calls retain the identity mapping and never inspect GPU
    // counts. A missing frequency or a mismatched table also remains dense.
    qwen4_moe_mm_args direct{}; direct.n_expert = 512;
    assert(qwen4_moe_mm_dispatch_experts(&direct) == 512 && direct.n_active_expert == 0);
    assert(qwen4_moe_mm_copy_active(nullptr, full, nullptr, 512) == UINT32_MAX);
    assert(qwen4_moe_mm_copy_active(scratch, nullptr, nullptr, 512) == UINT32_MAX);
    table.n_total_expert = 512;
    qwen4_stream_weights no_frequency{&table, nullptr, nullptr, nullptr};
    g_qwen4_stream_weights = &no_frequency;
    assert(qwen4_moe_mm_dispatch_experts(&direct) == 512 && direct.n_active_expert == 0);
    no_frequency.frequency = full; table.n_total_expert = 511;
    assert(qwen4_moe_mm_dispatch_experts(&direct) == 512 && direct.n_active_expert == 0);
    g_qwen4_stream_weights = nullptr;

    // Execute the actual extracted empty-return blocks from both public
    // wrappers. An empty pass must invalidate prior NAX and streamed half
    // associations; a nonempty pass preserves both. Cover absent streaming
    // and absent half storage as well as the active half producer record.
    int marker;
    for (auto guard : {empty_mid_guard, empty_down_guard}) {
        for (uint32_t mode = 0; mode < 3; ++mode) {
            half_pass_stub half{&marker};
            qwen4_stream_weights scope{&table, full, nullptr, mode == 2 ? &half : nullptr};
            g_qwen4_stream_weights = mode ? &scope : nullptr;
            g_qwen4_nax_half_mid_for = &marker; g_qwen4_nax_half_mid_count = 123;
            assert(guard(0) == 1);
            assert(g_qwen4_nax_half_mid_for == nullptr && g_qwen4_nax_half_mid_count == 0);
            assert(half.mid == (mode == 2 ? nullptr : &marker));
            empty_state_cases++;
            if (mode == 2) half_state_cases++;
            half.mid = &marker;
            g_qwen4_nax_half_mid_for = &marker; g_qwen4_nax_half_mid_count = 123;
            assert(guard(1) == 0);
            assert(g_qwen4_nax_half_mid_for == &marker && g_qwen4_nax_half_mid_count == 123);
            assert(half.mid == &marker);
            empty_state_cases++;
            if (mode == 2) half_state_cases++;
            g_qwen4_stream_weights = nullptr;
        }
    }

    // Retain three fully copied argument snapshots, then destroy/overwrite
    // all CPU routing inputs before mapping their expert indices.
    uint32_t *heap_frequency = (uint32_t *)malloc(sizeof(full));
    int32_t *heap_resident = (int32_t *)malloc(sizeof(resident));
    int32_t *heap_missing = (int32_t *)malloc(sizeof(missing));
    memcpy(heap_frequency, full, sizeof(full)); memcpy(heap_resident, resident, sizeof(resident));
    memcpy(heap_missing, missing, sizeof(missing));
    auto a = encode(heap_frequency, heap_resident), b = encode(heap_frequency, heap_missing), c = encode(heap_frequency, nullptr);
    memset(heap_frequency, 0xCC, sizeof(full)); memset(heap_resident, 0xDD, sizeof(resident)); memset(heap_missing, 0xEE, sizeof(missing));
    free(heap_frequency); free(heap_resident); free(heap_missing);
    for (const auto *mapped : {&a, &b, &c}) {
        for (uint32_t y = 0; y < mapped->dispatch_count; ++y) {
            const uint32_t e = qwen4_moe_mm_expert(mapped->args, y);
            assert(e < 512 && full[e]);
        }
        lifetime_cases++;
    }

    for (const int32_t *pass : {resident, missing, empty, warm}) {
        const auto compact = encode(full, pass);
        mapped_args original{}; original.args.n_expert = 512; original.dispatch_count = 512;
        for (bool nax : {false, true}) for (uint32_t rows : {1u, 640u, 641u, 2560u}) {
            for (uint32_t tt : {8u,16u,32u,64u}) {
                if (nax && tt < 32) continue;
                for (uint32_t tiles : {1u,2u,4u,8u,32u}) for (bool major : {false,true}) {
                    for (uint32_t tail : {0u,16u,32u,64u}) {
                        if (nax && tail != 0 && tail != 64) continue;
                        const auto old = schedule(original, pass, rows, nax ? 64 : 32, tt, tiles, tail, major, nax);
                        const auto now = schedule(compact, pass, rows, nax ? 64 : 32, tt, tiles, tail, major, nax);
                        assert(old == now); scheduled_cases++; schedule_events += now.size();
                    }
                }
            }
        }
    }
    printf("{\"status\":\"pass\",\"argument_bytes\":%zu,\"mapping_cases\":%llu,\"schedule_cases\":%llu,"
           "\"matching_schedule_events\":%llu,\"destroyed_pointer_lifetime_cases\":%llu,"
           "\"nax_empty_state_cases\":%llu,\"stream_half_state_cases\":%llu,\"gpu_runs\":false}\n",
           sizeof(qwen4_moe_mm_args), (unsigned long long)mapping_cases, (unsigned long long)scheduled_cases,
           (unsigned long long)schedule_events, (unsigned long long)lifetime_cases,
           (unsigned long long)empty_state_cases, (unsigned long long)half_state_cases);
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--host", type=Path, default=root / "ds4_metal.m")
    parser.add_argument("--metal", type=Path, default=root / "metal" / "qwen4.metal")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "clang++"))
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    host, metal = args.host.read_text(), args.metal.read_text()
    code = extract_cpu_code(host, metal)
    negative_controls = check_scanner_negative_controls(metal)
    flags = ["-std=c++17", "-O2", "-Wall", "-Wextra"]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    with tempfile.TemporaryDirectory(prefix="qwen-moe-mm-compact-") as directory:
        source = Path(directory) / "test.cpp"
        executable = Path(directory) / "test"
        source.write_text(code + CPU_ORACLE)
        subprocess.run(shlex.split(args.cxx) + flags + [str(source), "-o", str(executable)], check=True)
        run = subprocess.run([str(executable)], capture_output=True, text=True, check=True)
        if run.stderr:
            raise AssertionError(run.stderr)
        result = json.loads(run.stdout)
        result["sanitizers"] = args.sanitize
        result["scanner_negative_controls"] = negative_controls
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
