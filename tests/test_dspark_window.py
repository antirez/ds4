#!/usr/bin/env python3
"""Exercise the production DSpark target-window helpers with checked row ranges.

Requires Python and a C compiler; no model or GPU is needed. Optional --output retains the emitted C and build/run evidence; otherwise temporary files are removed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


HELPERS = (
    "metal_graph_dspark_cache_reset",
    "metal_graph_dspark_cache_window_valid",
    "metal_graph_dspark_cache_current_window_valid",
    "metal_graph_dspark_cache_set_window",
    "metal_graph_dspark_cache_crop_to_prefix",
    "metal_graph_dspark_cache_ends_at",
    "metal_graph_dspark_cache_merge_target_range",
    "metal_graph_dspark_cache_target_prefix",
)


def extract(source, name):
    matches = list(re.finditer(r"static (?:void|bool) " + re.escape(name) + r"\s*\(", source))
    if len(matches) != 1:
        raise ValueError("require exactly one production helper: " + name)
    start = matches[0].start()
    opening = source.index("{", start)
    token = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]', re.S)
    depth = 0
    for match in token.finditer(source, opening):
        if match.group() == "{":
            depth += 1
        elif match.group() == "}":
            depth -= 1
            if depth == 0:
                return source[start:match.end()]
    raise ValueError("unterminated production helper: " + name)


PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static uint32_t model_window = 128;
#define DS4_N_SWA model_window
typedef struct {
    uint32_t dspark_cache_cap, dspark_cache_start;
    uint32_t dspark_cache_token_start, dspark_cache_len;
} ds4_gpu_graph;
'''

TESTS = r'''
static unsigned checks = 0;
#define REQUIRE(x) do { checks++; assert(x); } while (0)
static void expect(const ds4_gpu_graph *g, uint32_t first, uint32_t len) {
    REQUIRE(g->dspark_cache_token_start == (len ? first : 0));
    REQUIRE(g->dspark_cache_len == len);
    REQUIRE(g->dspark_cache_start == (len ? first % g->dspark_cache_cap : 0));
    REQUIRE(metal_graph_dspark_cache_current_window_valid(g));
}
int main(void) {
    ds4_gpu_graph g = {.dspark_cache_cap = 4352};
    /* Cold prefill clips visibility, retaining the physical raw-cap modulus. */
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 0, 4096));
    expect(&g, 3968, 128);
    REQUIRE(metal_graph_dspark_cache_target_prefix(&g, 4095));
    expect(&g, 3968, 127);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 4095, 1));
    expect(&g, 3968, 128);
    /* Overlapping accepted verifier capture preserves the preceding history. */
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 4095, 7));
    expect(&g, 3974, 128);
    REQUIRE(metal_graph_dspark_cache_target_prefix(&g, 4101));
    expect(&g, 3974, 127);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 4101, 1));
    expect(&g, 3974, 128);
    /* A declined proposal followed by an ordinary seed retains its target row. */
    REQUIRE(metal_graph_dspark_cache_target_prefix(&g, 4102));
    expect(&g, 3975, 127);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 4102, 1));
    expect(&g, 3975, 128);
    /* Repeated/overlapping capture never retains the old speculative future. */
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 4098, 3));
    expect(&g, 3975, 126);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 4101, 2));
    expect(&g, 3975, 128);
    /* A gap resets; one successfully written target initializes an empty ring. */
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 4110, 4));
    expect(&g, 4110, 4);
    REQUIRE(metal_graph_dspark_cache_target_prefix(&g, 4120));
    expect(&g, 0, 0);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 4120, 1));
    expect(&g, 4120, 1);
    /* Rewind past the retained prefix cannot bridge to stale future features. */
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 50, 4));
    expect(&g, 50, 4);
    REQUIRE(metal_graph_dspark_cache_target_prefix(&g, 52));
    expect(&g, 50, 2);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 52, 1));
    expect(&g, 50, 3);
    /* Physical wrap and temporary draft exclusion: only claimed target is visible. */
    g = (ds4_gpu_graph){.dspark_cache_cap = 256};
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 190, 128));
    expect(&g, 190, 128);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 317, 7));
    expect(&g, 196, 128);
    REQUIRE(metal_graph_dspark_cache_target_prefix(&g, 323));
    expect(&g, 196, 127);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 323, 1));
    expect(&g, 196, 128);
    REQUIRE(g.dspark_cache_token_start + g.dspark_cache_len == 324);
    /* Simulate physical writes across a small ring, including scratch draft rows. */
    {
        int row[8] = {0};
        model_window = 4;
        g = (ds4_gpu_graph){.dspark_cache_cap = 8};
        for (int p = 5; p < 9; p++) row[p % 8] = p;
        REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 5, 4));
        for (int p = 8; p < 11; p++) row[p % 8] = p;
        REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 8, 3));
        expect(&g, 7, 4);
        for (uint32_t p = 7; p < 11; p++) REQUIRE(row[p % 8] == (int)p);
        for (int p = 11; p < 19; p++) row[p % 8] = p;
        REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 11, 8));
        expect(&g, 15, 4);
        for (uint32_t p = 15; p < 19; p++) REQUIRE(row[p % 8] == (int)p);
        REQUIRE(metal_graph_dspark_cache_target_prefix(&g, 18));
        expect(&g, 15, 3);
        row[18 % 8] = 18;
        for (int p = 19; p < 22; p++) row[p % 8] = -p;
        REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 18, 1));
        expect(&g, 15, 4);
        for (uint32_t p = 15; p < 19; p++) REQUIRE(row[p % 8] == (int)p);
    }
    /* Model window is dynamic, including the W=1 boundary. */
    g = (ds4_gpu_graph){.dspark_cache_cap = 256};
    model_window = 128;
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 196, 128));
    model_window = 32;
    REQUIRE(metal_graph_dspark_cache_target_prefix(&g, 324));
    expect(&g, 293, 31);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 324, 1));
    expect(&g, 293, 32);
    model_window = 1;
    REQUIRE(metal_graph_dspark_cache_target_prefix(&g, 325));
    expect(&g, 0, 0);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 325, 1));
    expect(&g, 325, 1);
    ds4_gpu_graph saved = g;
    REQUIRE(!metal_graph_dspark_cache_merge_target_range(&g, UINT32_MAX, 1));
    REQUIRE(!metal_graph_dspark_cache_merge_target_range(&g, 0, 257));
    REQUIRE(!metal_graph_dspark_cache_merge_target_range(&g, 0, 0));
    REQUIRE(memcmp(&saved, &g, sizeof(g)) == 0);
    model_window = 0;
    REQUIRE(!metal_graph_dspark_cache_target_prefix(&g, 326));
    REQUIRE(!metal_graph_dspark_cache_merge_target_range(&g, 326, 1));
    REQUIRE(memcmp(&saved, &g, sizeof(g)) == 0);
    printf("PASS %u metadata assertions\n", checks);
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "ds4.c")
    parser.add_argument("--output", type=Path, help="exclusive-new report directory; temporary files are removed when omitted")
    args = parser.parse_args()
    raw = args.source.read_bytes()
    helpers = "\n\n".join(extract(raw.decode("utf-8"), name) for name in HELPERS)
    code = PRELUDE + helpers + TESTS
    compiler = shlex.split(os.environ.get("CC", "cc"))
    if not compiler:
        raise ValueError("empty CC")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="ds4-window-contract-") as temporary:
        root = (args.output or Path(temporary)).resolve()
        native_path, binary = root / "metadata.c", root / "ds4-host-window-contract"
        native_path.write_text(code)
        command = compiler + ["-std=c99", "-O2", "-UNDEBUG", "-fno-fast-math", "-Wall", "-Wextra", "-Werror", str(native_path), "-o", str(binary)]
        build = subprocess.run(command, capture_output=True, text=True, timeout=60)
        report = {
            "scope": "CPU execution of extracted production metadata helpers and simulated physical row tags; no GPU arithmetic or model inference",
            "source": str(args.source.resolve()),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "extracted_helpers": list(HELPERS),
            "emitted_c_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "compile_command": command,
            "build_returncode": build.returncode,
            "build_stdout": build.stdout,
            "build_stderr": build.stderr,
            "exit_code": build.returncode,
        }
        if build.returncode == 0:
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
            report.update(exit_code=run.returncode, result=run.stdout.strip(), stderr=run.stderr,
                          binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
        if args.output:
            (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
