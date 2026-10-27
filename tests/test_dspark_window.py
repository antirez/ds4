#!/usr/bin/env python3
"""Exercise the production DSpark target-window helpers with checked row ranges.

Also replays the proposal driver's window sequence (catch up a pending capture,
crop to h[L-1], then draft or ring-maintain) against the official DSpark history
of the last 128 target positions, and checks structural contracts in ds4.c
that keep the reference alignment common to every GPU backend.

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
    matches = list(re.finditer(r"static (?:void|bool|int) " + re.escape(name) + r"\s*\(", source))
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

/* Driver-order replay. Reference visibility for a proposal at L (seed t[L],
 * target row h[L-1] at L-1): every target position max(0, L-128) .. L-1. The
 * support window must therefore be exactly [max(0, L-128), L-1) with true
 * physical rows, and the target row then joins it. */
static uint32_t rng_state = 12345u;
static uint32_t rnd(uint32_t n) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (rng_state >> 8) % n;
}
static int ring_tag[4352];
static void ring_write(const ds4_gpu_graph *g, uint32_t pos, int tag) {
    ring_tag[pos % g->dspark_cache_cap] = tag;
}
static void expect_history(const ds4_gpu_graph *g, uint32_t L) {
    const uint32_t first = L > 128u ? L - 128u : 0u;
    REQUIRE(metal_graph_dspark_cache_current_window_valid(g));
    REQUIRE(g->dspark_cache_len == L - 1u - first);
    if (g->dspark_cache_len) REQUIRE(g->dspark_cache_token_start == first);
    for (uint32_t p = first; p + 1u < L; p++) REQUIRE(ring_tag[p % g->dspark_cache_cap] == (int)p);
}
/* Seeding a prefill/verifier capture writes true rows, then merges them. */
static void catch_up(ds4_gpu_graph *g, uint32_t start, uint32_t n) {
    for (uint32_t p = start; p < start + n; p++) ring_write(g, p, (int)p);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(g, start, n));
}
static void replay_driver(unsigned seed_value) {
    /* Production caps the ring at raw_cap (>= the largest prefill chunk). */
    ds4_gpu_graph g = {.dspark_cache_cap = 512};
    model_window = 128;
    rng_state = seed_value;
    memset(ring_tag, 0xff, sizeof(ring_tag));
    uint32_t L = 1u + rnd(300u);              /* prompt length, one chunk */
    uint32_t pend_start = 0, pend_n = L;       /* prefill capture [0, L) */
    for (int cycle = 0; cycle < 400; cycle++) {
        if (pend_n && pend_start + pend_n == L) catch_up(&g, pend_start, pend_n);
        pend_n = 0;
        REQUIRE(metal_graph_dspark_cache_target_prefix(&g, L - 1u));
        expect_history(&g, L);
        const uint32_t event = rnd(5u);
        if (event == 0) {
            /* Scheduler skip: ring maintenance adds h[L-1], then decode t[L]. */
            ring_write(&g, L - 1u, (int)(L - 1u));
            REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, L - 1u, 1u));
            L += 1u;
            continue;
        }
        /* Proposal: target row at L-1, temporary drafts at L..L+4. */
        ring_write(&g, L - 1u, (int)(L - 1u));
        for (uint32_t d = 0; d < 5u; d++) ring_write(&g, L + d, -1);
        REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, L - 1u, 1u));
        if (event == 1) {
            L += 1u;                           /* declined: ordinary seed decode */
        } else if (event == 2) {
            /* Short fallback / non-seed path: decode t[L] alone, then verify
             * drafts from L+1; that capture starts at the seed's h[L]. */
            const uint32_t committed = 1u + rnd(5u);
            pend_start = L;
            pend_n = committed + 1u;
            L += 1u + committed;
        } else {
            const uint32_t committed = 1u + rnd(6u);   /* seed plus 0..5 drafts */
            pend_start = L - 1u;               /* verifier capture includes h[L-1] */
            pend_n = committed + 1u;
            L += committed;
        }
    }
}
static void verify_then_skip_regression(void) {
    /* A multi-row verify followed directly by a scheduler skip: cropping to
     * h[L-1] before merging the pending capture would drop all history. */
    ds4_gpu_graph g = {.dspark_cache_cap = 256};
    model_window = 128;
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 0, 200));
    REQUIRE(metal_graph_dspark_cache_target_prefix(&g, 199));
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 199, 1));
    ds4_gpu_graph crop_first = g;
    REQUIRE(metal_graph_dspark_cache_target_prefix(&crop_first, 203));
    REQUIRE(crop_first.dspark_cache_len == 0);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 199, 5));
    REQUIRE(metal_graph_dspark_cache_target_prefix(&g, 203));
    REQUIRE(g.dspark_cache_token_start == 76 && g.dspark_cache_len == 127);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 203, 1));
    REQUIRE(g.dspark_cache_token_start == 76 && g.dspark_cache_len == 128);
}

/* Prefill history. Seeding writes only the trailing min(window, n) rows of a
 * capture and merges them (equal to merging the whole capture). */
static void seed_tail(ds4_gpu_graph *g, uint32_t start, uint32_t n) {
    const uint32_t rows = n > 128u ? 128u : n;
    for (uint32_t p = start + n - rows; p < start + n; p++) ring_write(g, p, (int)p);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(g, start + n - rows, rows));
}
/* Decode-style prefill and continued-prefill catch-up merge one row h[pos]. */
static void merge_row(ds4_gpu_graph *g, uint32_t pos) {
    REQUIRE(metal_graph_dspark_cache_target_prefix(g, pos));
    ring_write(g, pos, (int)pos);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(g, pos, 1u));
}
static void proposal_sees(ds4_gpu_graph *g, uint32_t L) {
    REQUIRE(metal_graph_dspark_cache_target_prefix(g, L - 1u));
    expect_history(g, L);
}
/* Chunked prefill [start, end) in chunks of cap tokens; per-chunk tail merge
 * only when the chunk can still reach the final window (as the driver does). */
static void chunked_prefill(ds4_gpu_graph *g, uint32_t start, uint32_t end, uint32_t cap) {
    for (uint32_t p0 = start; p0 < end; ) {
        uint32_t chunk = end - p0 < cap ? end - p0 : cap;
        if (start != 0 && p0 % cap) {
            const uint32_t to_boundary = cap - p0 % cap;
            if (to_boundary < chunk) chunk = to_boundary;
        }
        const uint32_t p1 = p0 + chunk;
        if (p1 < end && end - p1 < 128u) seed_tail(g, p0, chunk);
        else if (p1 == end) seed_tail(g, p0, chunk);   /* first proposal's catch-up */
        p0 = p1;
    }
}
static void prefill_tail_regression(void) {
    model_window = 128;
    const uint32_t cases[][2] = {
        {2048u + 50u, 2048u}, {2048u + 127u, 2048u}, {2048u + 128u, 2048u},
        {4096u + 1u, 2048u}, {130u, 100u}, {129u, 128u}, {60u, 2048u},
    };
    for (uint32_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        ds4_gpu_graph g = {.dspark_cache_cap = 4352};
        memset(ring_tag, 0xff, sizeof(ring_tag));
        chunked_prefill(&g, 0, cases[c][0], cases[c][1]);
        proposal_sees(&g, cases[c][0]);
    }
    /* Trailing seeding equals seeding the whole long capture. */
    {
        ds4_gpu_graph a = {.dspark_cache_cap = 4352}, b = a;
        REQUIRE(metal_graph_dspark_cache_merge_target_range(&a, 300, 50));
        b = a;
        REQUIRE(metal_graph_dspark_cache_merge_target_range(&a, 350, 2000));
        REQUIRE(metal_graph_dspark_cache_merge_target_range(&b, 2350 - 128, 128));
        REQUIRE(memcmp(&a, &b, sizeof(a)) == 0);
    }
    /* Continued prefill after a verifier commit: the pending capture
     * [L-1, E) merges before the append, then the append's tail. */
    {
        ds4_gpu_graph g = {.dspark_cache_cap = 4352};
        memset(ring_tag, 0xff, sizeof(ring_tag));
        seed_tail(&g, 0, 900);
        proposal_sees(&g, 900);
        ring_write(&g, 899, 899);
        REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 899, 1));  /* stage chain */
        seed_tail(&g, 899, 6);                 /* verifier capture [899, 905) */
        chunked_prefill(&g, 905, 905 + 20, 2048);
        proposal_sees(&g, 925);
    }
    /* Continued prefill after an ordinary decode: merge the decoded row. */
    {
        ds4_gpu_graph g = {.dspark_cache_cap = 4352};
        memset(ring_tag, 0xff, sizeof(ring_tag));
        seed_tail(&g, 0, 400);
        proposal_sees(&g, 400);
        ring_write(&g, 399, 399);
        REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 399, 1));  /* declined draft */
        merge_row(&g, 400);                    /* seed t[400] decoded, h[400] */
        chunked_prefill(&g, 401, 401 + 10, 2048);
        proposal_sees(&g, 411);
    }
    /* Continued prefill crossing a chunk boundary keeps prior history. */
    {
        ds4_gpu_graph g = {.dspark_cache_cap = 4352};
        memset(ring_tag, 0xff, sizeof(ring_tag));
        seed_tail(&g, 0, 2000);
        merge_row(&g, 2000);
        chunked_prefill(&g, 2001, 2001 + 80, 2048);   /* splits 47 + 33 */
        proposal_sees(&g, 2081);
    }
}
static void short_prefill_regression(void) {
    /* A prompt shorter than the window exposes every prompt row. */
    ds4_gpu_graph g = {.dspark_cache_cap = 256};
    model_window = 128;
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 0, 60));
    REQUIRE(metal_graph_dspark_cache_target_prefix(&g, 59));
    REQUIRE(g.dspark_cache_token_start == 0 && g.dspark_cache_len == 59);
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 59, 1));
    REQUIRE(g.dspark_cache_len == 60);
    /* One-token prompt: no history, then h[0] alone. */
    g = (ds4_gpu_graph){.dspark_cache_cap = 256};
    REQUIRE(metal_graph_dspark_cache_merge_target_range(&g, 0, 1));
    REQUIRE(metal_graph_dspark_cache_target_prefix(&g, 0));
    REQUIRE(g.dspark_cache_len == 0);
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
    verify_then_skip_regression();
    short_prefill_regression();
    prefill_tail_regression();
    for (unsigned seed = 1; seed <= 64; seed++) replay_driver(seed * 2654435761u);
    printf("PASS %u metadata assertions\n", checks);
    return 0;
}
'''


def find_body(source, name):
    try:
        return extract(source, name)
    except ValueError:
        return ""


def contracts(source):
    """Structural guards: the reference alignment is the only DSpark path."""
    results = []

    def check(name, ok):
        results.append({"check": name, "ok": bool(ok)})

    for gone in ("gfx1151_reference_alignment", "align_rocm", "claim_appended_row"):
        check("no " + gone, gone not in source)
    driver = find_body(source, "ds4_session_prepare_dspark_draft_impl")
    # The TP branch chooses DRAFT/MAINTAIN before entering its own cache
    # helper. The local path below it must still merge before checking skip.
    local_start = driver.find("const uint32_t feature_pos = pos - 1u;")
    local = driver[local_start:] if local_start >= 0 else ""
    order = [local.find(token) for token in (
        "metal_graph_seed_dspark_initial_cache_from_prefill(",
        "metal_graph_dspark_cache_target_prefix(",
        "ds4_session_dspark_scheduler_should_skip(",
        "metal_graph_dspark_ring_maintain(")]
    check("local driver merges pending capture, crops, then skips or drafts",
          min(order) >= 0 and order == sorted(order))
    tp = find_body(source, "ds4_session_dspark_tp_step")
    order = [tp.find(token) for token in (
        "metal_graph_seed_dspark_initial_cache_from_prefill(",
        "metal_graph_dspark_cache_target_prefix(",
        "if (mode == DS4_TP_DSPARK_MAINTAIN)",
        "metal_graph_dspark_ring_maintain(",
        "metal_graph_eval_dspark_stage_chain(")]
    check("TP step merges pending capture and crops before maintaining or drafting",
          min(order) >= 0 and order == sorted(order))
    check("TP driver runs the cache helper for both draft and maintain",
          "tp_draft && !skip ? DS4_TP_DSPARK_DRAFT : DS4_TP_DSPARK_MAINTAIN" in driver and
          "ds4_session_dspark_tp_step(s, token, pos, mode," in driver and
          "if (tp_draft && !skip)" not in driver)
    check("driver pairs t[pos] with h[pos-1]", "const uint32_t feature_pos = pos - 1u;" in driver)
    initial = find_body(source, "metal_graph_seed_dspark_initial_cache_from_prefill")
    check("prefill/verifier capture merges into history",
          "metal_graph_dspark_cache_merge_target_range(" in initial and
          "metal_graph_dspark_cache_set_window(" not in initial)
    target = find_body(source, "metal_graph_seed_dspark_stage_target_cache")
    check("target KV is wkv(main_x) without stage HC/attn_norm",
          "metal_graph_batch_ffn_norm(g)" in target and "hc_attn_fn" not in target and
          "attn_norm" not in target)
    for name in ("metal_graph_eval_dspark_stage_block", "metal_graph_eval_dspark_stage_chain"):
        check(name + " ropes the target row at pos-1",
              "const uint32_t feature_pos = pos - 1u;" in find_body(source, name))
    chain = find_body(source, "metal_graph_eval_dspark_stage_chain")
    check("stage chain commits the target row",
          "metal_graph_dspark_cache_merge_target_range(g, feature_pos, 1u)" in chain)
    ring = find_body(source, "metal_graph_dspark_ring_maintain")
    check("ring maintenance merges from any window",
          "metal_graph_dspark_cache_merge_target_range(g, pos, 1u)" in ring and
          "dspark_cache_len == 0" not in ring)
    check("both setup blocks place h[pos-1] at pos-1",
          source.count("positions[0] = (int32_t)(pos - 1u);") == 2)
    support = find_body(source, "ds4_session_prepare_support_draft")
    check("DSpark never drafts after the seed's evaluation",
          support != "" and "ds4_session_prepare_dspark_draft(" not in support)
    seed = find_body(source, "metal_graph_seed_dspark_initial_cache_from_prefill")
    check("capture seeding projects only the trailing window",
          "batch_start + skip" in seed and "metal_graph_eval_dspark_stage0_batch(g," in seed)
    chunked = find_body(source, "metal_graph_prefill_chunked_range")
    check("continued prefill merges the live capture first",
          "metal_graph_dspark_merge_before_prefill(g, start);" in chunked)
    check("chunked prefill keeps each chunk tail that can reach the window",
          "metal_graph_dspark_merge_batch_capture(g, pos0, chunk);" in chunked)
    check("drafting sessions expose the support model to prefill",
          "s->graph.dspark_model = &e->mtp_model;" in source)
    return results


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
        structural = contracts(raw.decode("utf-8"))
        report = {
            "scope": "CPU execution of extracted production metadata helpers and simulated physical row tags; no GPU arithmetic or model inference",
            "structural_contracts": structural,
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
        if report["exit_code"] == 0 and not all(item["ok"] for item in structural):
            report["exit_code"] = 1
        if args.output:
            (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
