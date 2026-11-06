#!/usr/bin/env python3
"""Check that DSpark confidence reads the pre-norm hc_head rows (official forward_head).

Extracts the production confidence0 and lazy-confidence functions from ds4.c and
runs them on a host stub graph whose pre-norm rows (batch_ffn_cur) and normalized
rows (batch_ffn_norm) yield very different confidence logits. Requires Python and
a C compiler; no model or GPU is needed. Optional --output retains the emitted C
and build/run evidence; otherwise temporary files are removed.
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
    "dspark_argmax_f32",
    "dspark_eval_confidence0_runtime",
    "dspark_apply_markov_confidence_lazy_runtime",
)
# Also accept sources that inline the confidence-row selection.
OPTIONAL_HELPERS = ("metal_graph_dspark_confidence_hidden",)
DRIVER = "ds4_session_prepare_dspark_draft_impl"


def find(source, name):
    pattern = r"static (?:inline )?(?:void|bool|uint32_t|ds4_gpu_tensor \*)\s*" + re.escape(name) + r"\s*\("
    return list(re.finditer(pattern, source))


def extract(source, name):
    matches = find(source, name)
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


def probe_read_contract(source):
    """The CPU probe path reads its hidden rows inside the proposal driver."""
    body = extract(source, DRIVER)
    begin = body.find("if (confidence_ready) {")
    end = body.find("dspark_eval_confidence_probe(", begin)
    if begin < 0 or end < 0:
        return {"ok": False, "reason": "probe confidence block not found"}
    block = body[begin:end]
    ok = "metal_graph_dspark_confidence_hidden(" in block and "metal_graph_batch_ffn_norm(" not in block
    return {"ok": ok, "reason": "probe hidden rows come from the pre-norm selector" if ok
            else "probe hidden rows do not use the pre-norm selector"}


PRELUDE = r'''
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DS4_N_EMBD 8u
#define DS4_N_VOCAB 16u
#define DS4_DSPARK_MAX_BLOCK_SIZE 16u
#define DS4_DSPARK_MAX_STAGES 8u
#define RANK 4u
#define BLOCK 3u
enum { DS4_TENSOR_F32 = 0, DS4_TENSOR_Q8_0 = 8 };
typedef struct ds4_gpu_tensor { float *data; uint64_t bytes; } ds4_gpu_tensor;
typedef struct { const void *map; uint64_t size; } ds4_model;
typedef struct { uint32_t type; uint64_t abs_offset; int id; } ds4_tensor;
typedef struct { ds4_tensor *markov_w1, *markov_w2, *confidence_proj; } ds4_dspark_stage_weights;
typedef struct {
    uint32_t n_stages, block_size, markov_rank;
    ds4_dspark_stage_weights stage[DS4_DSPARK_MAX_STAGES];
} ds4_dspark_weights;
typedef struct {
    ds4_gpu_tensor *spec_logits, *dspark_draft_tokens, *ffn_cur, *ffn_norm;
} ds4_gpu_graph;
enum { T_W1 = 1, T_W2 = 2, T_PROJ = 3 };
static float g_w1[DS4_N_VOCAB][RANK];
static float g_proj[DS4_N_EMBD + RANK];
static unsigned g_reads_cur, g_reads_norm;
static ds4_gpu_graph *g_graph;
__attribute__((unused)) static ds4_gpu_tensor *metal_graph_batch_ffn_cur(const ds4_gpu_graph *g) { return g->ffn_cur; }
__attribute__((unused)) static ds4_gpu_tensor *metal_graph_batch_ffn_norm(const ds4_gpu_graph *g) { return g->ffn_norm; }
static uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t ? t->bytes : 0; }
static int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t off, void *dst, uint64_t n) {
    if (!t || off > t->bytes || n > t->bytes - off) return 0;
    if (t == g_graph->ffn_cur) g_reads_cur++;
    if (t == g_graph->ffn_norm) g_reads_norm++;
    memcpy(dst, (const char *)t->data + off, n);
    return 1;
}
__attribute__((unused)) static ds4_gpu_tensor *ds4_gpu_tensor_view(ds4_gpu_tensor *t, uint64_t off, uint64_t n) {
    (void)t; (void)off; (void)n; return NULL;
}
__attribute__((unused)) static void ds4_gpu_tensor_free(ds4_gpu_tensor *t) { (void)t; }
__attribute__((unused)) static int ds4_gpu_dspark_markov_argmax_tensor(ds4_gpu_tensor *o, ds4_gpu_tensor *l,
        const void *m, uint64_t s, uint64_t a, uint64_t b, uint32_t p, uint32_t v, uint32_t r) {
    (void)o; (void)l; (void)m; (void)s; (void)a; (void)b; (void)p; (void)v; (void)r; return 0;
}
static bool dspark_markov_probe_ready(const ds4_dspark_weights *dw) { return dw != NULL; }
static bool dspark_confidence_probe_ready(const ds4_dspark_weights *dw) { return dw != NULL; }
static bool dspark_markov_bias_disabled(void) { return false; }
static bool dspark_disable_fused_cpu_markov_argmax(void) { return true; }
__attribute__((unused)) static bool dspark_markov_q8_0_argmax(uint32_t *o, const ds4_model *m,
        const ds4_tensor *w, const float *s, const float *l) {
    (void)o; (void)m; (void)w; (void)s; (void)l; return false;
}
static float sigmoid_stable(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
}
static bool dspark_dense_row_to_f32(float *out, const ds4_model *m, const ds4_tensor *t, uint32_t row) {
    (void)m;
    if (!t || t->id != T_W1 || row >= DS4_N_VOCAB) return false;
    memcpy(out, g_w1[row], sizeof(g_w1[row]));
    return true;
}
__attribute__((noinline)) static void matvec_any(float *out, const ds4_model *m, const ds4_tensor *t, const float *x) {
    (void)m;
    if (t->id == T_PROJ) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < DS4_N_EMBD + RANK; i++) acc += g_proj[i] * x[i];
        out[0] = acc;
    } else {
        assert(t->id == T_W2);
        /* Zero Markov bias keeps the greedy proposals equal to the logits argmax. */
        for (uint32_t i = 0; i < DS4_N_VOCAB; i++) out[i] = 0.0f;
    }
}
'''

TESTS = r'''
static unsigned checks = 0;
#define REQUIRE(x) do { checks++; if (!(x)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)

static float pre[BLOCK][DS4_N_EMBD], post[BLOCK][DS4_N_EMBD], logits[BLOCK][DS4_N_VOCAB];

static float conf_of(const float *hidden, int prev) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) acc += g_proj[i] * hidden[i];
    for (uint32_t r = 0; r < RANK; r++) acc += g_proj[DS4_N_EMBD + r] * g_w1[prev][r];
    return acc;
}

int main(void) {
    /* Pre-norm hc_head rows are large; RMSNorm brings them to unit scale.
     * The projection rewards large positive hidden features, so the pre-norm
     * read is confident at every position while the normalized one is not. */
    for (uint32_t k = 0; k < BLOCK; k++) {
        double ss = 0.0;
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
            pre[k][i] = 6.0f + (float)(i % 3) + (float)k;
            ss += (double)pre[k][i] * pre[k][i];
        }
        const float rs = 1.0f / sqrtf((float)(ss / DS4_N_EMBD) + 1e-6f);
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) post[k][i] = pre[k][i] * rs;
        for (uint32_t v = 0; v < DS4_N_VOCAB; v++) logits[k][v] = (float)((v * 7u + k * 3u) % DS4_N_VOCAB);
    }
    /* Hidden weight 0.3 gives about +18 from pre-norm rows and +2.4 from
     * normalized rows; the Markov feature adds a fixed -6 (W1[v][0] = -6 with
     * projection weight 1), so sigmoid is ~1 pre-norm and ~0.03 post-norm. */
    for (uint32_t v = 0; v < DS4_N_VOCAB; v++) {
        g_w1[v][0] = -6.0f;
        for (uint32_t r = 1; r < RANK; r++) g_w1[v][r] = 0.01f * (float)((v + r) % 5);
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) g_proj[i] = 0.3f;
    g_proj[DS4_N_EMBD] = 1.0f;
    for (uint32_t r = 1; r < RANK; r++) g_proj[DS4_N_EMBD + r] = 0.0f;

    ds4_gpu_tensor cur = {&pre[0][0], sizeof(pre)}, norm = {&post[0][0], sizeof(post)};
    ds4_gpu_tensor spec = {&logits[0][0], sizeof(logits)}, tokens = {NULL, 0};
    ds4_gpu_graph g = {&spec, &tokens, &cur, &norm};
    g_graph = &g;
    ds4_tensor w1 = {DS4_TENSOR_F32, 0, T_W1}, w2 = {DS4_TENSOR_F32, 0, T_W2},
               proj = {DS4_TENSOR_F32, 0, T_PROJ};
    ds4_dspark_weights dw = {.n_stages = 1, .block_size = BLOCK, .markov_rank = RANK};
    dw.stage[0] = (ds4_dspark_stage_weights){&w1, &w2, &proj};
    ds4_model model = {0};
    const int seed = 5;
    const float threshold = 0.7f;

    /* Expected proposals: zero Markov bias, so each draft is the row argmax. */
    int32_t want[BLOCK];
    for (uint32_t k = 0; k < BLOCK; k++) {
        uint32_t best = 0;
        for (uint32_t v = 1; v < DS4_N_VOCAB; v++) if (logits[k][v] > logits[k][best]) best = v;
        want[k] = (int32_t)best;
    }
    float conf_pre[BLOCK], conf_post[BLOCK];
    for (uint32_t k = 0; k < BLOCK; k++) {
        const int prev = k == 0 ? seed : want[k - 1];
        conf_pre[k] = conf_of(pre[k], prev);
        conf_post[k] = conf_of(post[k], prev);
        REQUIRE(sigmoid_stable(conf_pre[k]) >= threshold);
        REQUIRE(sigmoid_stable(conf_post[k]) < threshold);
        REQUIRE(fabsf(conf_pre[k] - conf_post[k]) > 1.0f);
    }

    float features[DS4_N_EMBD + RANK];
    float c0 = -1e9f;
    g_reads_cur = g_reads_norm = 0;
    REQUIRE(dspark_eval_confidence0_runtime(&g, &model, &dw, seed, features,
                                            DS4_N_EMBD + RANK, &c0));
    REQUIRE(fabsf(c0 - conf_pre[0]) <= 1e-5f * (1.0f + fabsf(conf_pre[0])));
    REQUIRE(g_reads_cur == 1 && g_reads_norm == 0);

    for (int reuse = 0; reuse < 2; reuse++) {
        float lbuf[DS4_N_VOCAB], bias[DS4_N_VOCAB];
        int32_t proposal[DS4_DSPARK_MAX_BLOCK_SIZE];
        uint32_t plen = 99, clen = 99, cprefix = 99;
        float conf0 = reuse ? conf_pre[0] : -1e9f;
        g_reads_cur = g_reads_norm = 0;
        REQUIRE(dspark_apply_markov_confidence_lazy_runtime(&g, &model, &dw, seed, threshold,
                lbuf, bias, features, DS4_N_EMBD + RANK, proposal, &plen, &clen, &cprefix,
                reuse != 0, &conf0));
        /* Every position passes only if every read used the pre-norm rows. */
        REQUIRE(plen == BLOCK && clen == BLOCK && cprefix == BLOCK);
        for (uint32_t k = 0; k < BLOCK; k++) REQUIRE(proposal[k] == want[k]);
        REQUIRE(fabsf(conf0 - conf_pre[0]) <= 1e-5f * (1.0f + fabsf(conf_pre[0])));
        REQUIRE(g_reads_norm == 0);
        REQUIRE(g_reads_cur == (reuse ? BLOCK - 1u : BLOCK));
    }

    /* Undersized pre-norm rows are rejected instead of read out of bounds. */
    ds4_gpu_tensor short_cur = {&pre[0][0], sizeof(pre[0]) * (BLOCK - 1u)};
    g.ffn_cur = &short_cur;
    {
        float lbuf[DS4_N_VOCAB], bias[DS4_N_VOCAB], conf0 = 0.0f;
        int32_t proposal[DS4_DSPARK_MAX_BLOCK_SIZE];
        uint32_t plen = 0, clen = 0, cprefix = 0;
        REQUIRE(!dspark_apply_markov_confidence_lazy_runtime(&g, &model, &dw, seed, threshold,
                lbuf, bias, features, DS4_N_EMBD + RANK, proposal, &plen, &clen, &cprefix,
                false, &conf0));
    }
    printf("PASS %u confidence assertions pre=%.4f post=%.4f\n", checks, conf_pre[0], conf_post[0]);
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "ds4.c")
    parser.add_argument("--output", type=Path, help="exclusive-new report directory; temporary files are removed when omitted")
    args = parser.parse_args()
    raw = args.source.read_bytes()
    source = raw.decode("utf-8")
    present = [name for name in OPTIONAL_HELPERS if len(find(source, name)) == 1]
    names = present + list(HELPERS)
    helpers = "\n\n".join(extract(source, name) for name in names)
    contract = probe_read_contract(source)
    code = PRELUDE + helpers + TESTS
    compiler = shlex.split(os.environ.get("CC", "cc"))
    if not compiler:
        raise ValueError("empty CC")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="ds4-dspark-confidence-") as temporary:
        root = (args.output or Path(temporary)).resolve()
        native_path, binary = root / "confidence.c", root / "ds4-dspark-confidence-input"
        native_path.write_text(code)
        command = compiler + ["-std=c99", "-O2", "-UNDEBUG", "-fno-fast-math", "-Wall", "-Wextra", "-Werror",
                              str(native_path), "-o", str(binary), "-lm"]
        build = subprocess.run(command, capture_output=True, text=True, timeout=60)
        report = {
            "scope": "CPU execution of extracted production confidence readers on a stub graph; no GPU arithmetic or model inference",
            "source": str(args.source.resolve()),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "extracted_helpers": names,
            "probe_read_contract": contract,
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
        if report["exit_code"] == 0 and not contract["ok"]:
            report["exit_code"] = 1
        if args.output:
            (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
