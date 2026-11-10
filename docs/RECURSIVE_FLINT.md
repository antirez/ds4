# Recursive Flint (fork, in progress)

[README](../README.md) · [fork maintenance](FORK_MAINTAINING.md) ·
[Qwen3.8 27B](QWEN38_27B.md) · [Bonsai 2 27B](BONSAI2_27B.md) ·
[Qwen3.8 Flash Next](QWEN38_FLASH_NEXT.md)

This branch (`recursive-flint`) adds
[RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), a 40B-A2.5B
mixture-of-experts coding model, to the Metal backend. It is not part of
upstream ds4; upstream issue
[#1260](https://github.com/antirez/ds4/issues/1260) (2026-10-23, no comments,
no PR) requests it.

The port follows Recursive's own engine,
[rec](https://github.com/recursive-oss/rec) (Apache 2.0, installed locally as
`~/.local/bin/rec` 0.1.1), closely: its weight layout, its decode megakernel
and its prefill schedule are the template, and rec is the ground truth for
tokens, logits and speed. The point of having it in ds4 rather than using rec
is **SSD streaming of the routed experts**: rec needs the whole 22.6 GB
checkpoint resident and refuses Macs under 48 GB; ds4 already streams
routed experts from disk through a bounded RAM cache for DeepSeek, GLM and
Qwen, and gets sessions, disk KV checkpoints, the agent and the server with
it.

Scope, for now:

- **One weight format**: Recursive's 4-bit release
  [RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4)
  (`mlx-affine-int4`, the file `rec pull` downloads), loaded bit for bit.
  No other quant until this is done.
- **One reference**: rec. Decode must reproduce rec's megakernel bit for
  bit on identical inputs; prefill within rec's own spread.
- **Metal first.** CUDA is step 7.

The reference machine is an M4 Pro with 48 GB (16-core GPU, 273 GB/s), where
rec reports more than 125 tok/s averaged over a 128K-token session.

## Status

| Step | State |
|---|---|
| 0. Scaffolding: fork files from `bonsai2`; converter from rec's checkpoint (rec's tensor layout, per-expert packs); shape, loader, tokenizer, server id | **done 2026-11-07.** Family `DS4_MODEL_FAMILY_FLINT = 200` / variant 202 with fork scaffolding; converter writes 22.574 GB in 56 s, `--check-all` byte-identical on all 1375 tensors and the nibble layout confirmed against `mx.dequantize`; `--inspect` opens, binds and validates the GGUF; tokenizer bit-identical to HF tokenizers on 1069 cases; `golden.json` has 8 rec cases (incl. a 20277-token story; rec decodes ~120 tok/s, prefill ~550 tok/s on the M4 Pro); `rec_logits.py` dumps prefill and per-step logits, 50×2048 bf16 traces and per-layer KV from rec's runtime (loads in 10 s). Descriptor and address tables move to step 1 with the kernel. |
| 1. Decode megakernel ported: per-layer residual trace bit-identical to rec on identical KV; one token per launch, CPU sampling | **done 2026-11-07.** `metal/flint.metal` + `ds4_metal_flint.m` + `ds4_flint.inc`; `trace_compare.py` reports every trace row (50 × 2048) and all 150144 logits identical to rec on 56 decode steps over four prefills: 5 tokens, 559 (ring wrapped), 1401 (above the 1024-part boundary, 8-head path, 128 parts) and 8445 (256 parts). Warm decode 7.3 ms/token at 5 tokens, 8.0 at 559, 8.4 at 1401, 10.0 at 8445 (one token per launch with logits readback); a cold first step pays about 200 ms of page-ins. Not yet exercised: count exactly 1024 (64 parts), 512 parts, and contexts past 65536 (multi-block KV). |
| 2. Prefill (chunked, window attention, sorted-expert GEMM), sessions, live KV reuse, disk checkpoints; golden cases within rec's spread | **done 2026-11-09.** `metal/flint_prefill.metal` (own kernels, see Design: Prefill), `ds4_flint_session.inc`; `make test-flint-kernels` passes. Against rec (`prefill_compare.py`): the last row's logits agree to KL 1e-8 to 1e-9 on 12-, 578- and 2556-token prompts; per-layer error grows like rounding noise (relative mean 0.003 to 0.12 by layer 48) with a few rows diverging at routing near-ties (578-token prompt: 55 of 578 rows have a sixth/seventh expert margin under 1%). Golden teacher-forced (`check_golden.py`, 128 tokens): argmax = rec's 94.5 to 100%, abs delta logprob mean 0.011 to 0.039, max 0.60; rec's own decode-vs-prefill at identical positions is mean 0.05 to 0.32, max 4.5, agreement 88 to 100% (3 dumps, 40 positions). Greedy paths part after 15 to 66 tokens on the short cases and run 64/64 on the tool case. Gate 2 (`reuse_check.py`): a follow-up turn on decode KV versus a fresh prefill differs along 64 steps by KL mean 0.001 to 0.005, max 0.064, abs delta logprob mean 0.011 to 0.032, top-1 same 97 to 100%: reuse stays. `ds4_test --server`, `--session-snapshot` and `--long-context` (29624 tokens, 100K context, two KV blocks) pass. Speed (not tuned): prefill 170 tok/s at 20K (rec 410 to 425 on this Mac), the first-cut GEMM; decode through sessions 10.3 to 11.4 ms/token short, 21.3 at 20K. |
| 3. Measure against rec: decode at 4K/32K/128K, prefill, launch overhead; decide on 8-token resident blocks | **done 2026-11-10** (mains, High Power Mode, M4 Pro 48 GB, `tests/flint/bench_vs_rec.py`: both servers through one streaming client, 128 new tokens, thinking on). Decode ms/token rec vs ds4: 7.3 vs 7.6 at 256 tokens, 8.5 vs 8.8 at 4K, 10.9 vs 11.5 at 16K, 13.3 vs 13.9 at 28K (ds4 within 4 to 5% everywhere). The launch floor (`DS4_FLINT_LAYER_COUNT=0`: embedding, head and readback only) is 0.92 ms per step, so one launch per token costs what the gap shows; **8-token resident blocks would buy at most 5% and are not worth the in-kernel sampling: decision no.** Prefill tok/s rec vs ds4: 398 vs 201 at 256 (warm), 659 vs 433 at 4K, 569 vs 234 at 16K, 463 vs 164 at 28K: ds4's prefill falls 2.6x from 4K to 28K where rec's falls 1.4x, which names the attention kernel (64 x 32 tiles, one head per threadgroup, no double buffering) as the first target; the GEMM gap at 4K is 1.5x. A cold first prefill pays about 8 s of page-ins for the 22.6 GB mapping. 128K not measured (no 128K prompt file yet; rec's 75 tok/s at 28K says its "125 tok/s over a 128K session" is weighted toward the short part). |
| 4. Expert streaming: routing-trace study, cache and address table, in-kernel slot wait (or per-layer split), prefill staging, budget; streamed output bit-identical to resident | **done 2026-11-09** (`ds4_flint_stream.c`, see Design: Streaming). The visibility experiment failed usefully: a CPU store reaches a running kernel only after 0.3 to 10.7 s and a GPU store reaches the CPU about 30 ms after launch (caches flush at command-buffer boundaries), so misses are handled by **abort and resume**: the megakernel aborts at the first layer whose expert is not cached, publishes the layer's ids, the host reads the packs and relaunches from that layer. Hits are the unchanged resident path. `stream_check.py`: streamed and resident outputs are bit-identical on `short_01`, `multi_01` and `tool_01` with a 1 GB cache (606 slots; every decode layer misses, 4.8 of 6 experts), under all three prefill policies. `make test-flint-stream` (model-free, sparse file) passes; `ds4_test --server` and `--session-snapshot` pass with `DS4_TEST_SSD_STREAMING=1` and a 2 GB cache. Cost of a relaunch about 0.3 ms plus the reads (35 ms/token with a 1 GB cache and the file in the page cache, against 9 resident). Routing logs over 1197 decoded tokens (`route_sim.py`, plain LRU): miss fraction 0.65 at 2 GB, 0.21 at 8 GB, 0.09 at 12 GB, 0.03 at 16 GB, 0.007 at 20 GB. Not measured: cold-SSD bandwidth (the file sits in this Mac's page cache and `purge` needs root) and the 20K case in streaming mode (deferred to an AC-powered session). |
| 5. Server and agent: Flint template, `<tool_call>` syntax, reasoning parsing, `ds4-agent` session | **done 2026-11-09.** `ds4_flint_server.inc`: the server renders Flint prompts from rec's `chat_template.jinja` (roles in order, `flint_text` escaping, the `# Tools` preamble before the first non-system turn, `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>` with string values raw and other values as Python JSON, `<|observation|><tool_response>` results); `tests/flint/check_template.py` renders nine message sets through the live server and through rec's template under HF's jinja settings: all byte-identical, and a live request with tools returns a structured `tool_calls` entry plus `reasoning_content`. Flint reports GLM's tool syntax, so the GLM reply parser and the live tool-call tracker are reused unchanged. Thinking: rec's grammar forces `<think>` as the first token, so the generation prompt is `<|assistant|><think>`; `enable_thinking: false`, `reasoning_effort: none` and the `flint-chat` / `flint-no-think` aliases prefill `<think></think>` instead (`flint-reasoner` forces it on); `/v1/models` lists `flint`, `flint-chat`, `flint-reasoner`. Sampling defaults from `generation_config.json` (temperature 1.0, top-p 0.95, top-k 20) for knobs a request leaves out; the agent uses 1.0 / 0.95. `ds4-agent` with Flint: thinking, bash and read tool calls and the final answer work in one turn on a file task; its KV save needed `ds4_engine_routed_quant_bits` to report 4 for the packs. Not done: the Terminal-Bench style comparison with `rec launch` (needs the AC-powered session). |
| 6. Streaming speed: prefetch and prediction, popularity preload, cache policy | open |
| 7. Later: CUDA (DGX Spark), session batching, other quants | open |

Each step records what it measured here when it lands. Steps 0-2 are
correctness against rec, 3 the resident baseline, 4 the reason for the
branch, 5 product, 6 streaming speed. Nothing in 3-6 starts before the
trace check of step 1 passes.

## Model

Published 2026-10-21 under Apache 2.0, trained from scratch, built with
Recursive's "Eureka Machine" architecture search from a 35B/3B baseline. Text
only; no vision tower, no MTP head. From `config.json` and the model card:

- 49 layers: 2 dense SwiGLU layers (6144 wide) then 47 MoE layers. Hidden
  2048, vocabulary 150144, untied embedding and output head, RMSNorm
  eps 1e-6, no biases.
- Attention: 32 query heads, 4 KV heads, head dim 128, scale 1/sqrt(128),
  per-head QK RMSNorm (weight of 128) before RoPE, and a learned per-head
  attention sink (`softmax_offset`, 32 values: one extra softmax logit
  whose mass is dropped).
- 35 sliding-window layers (window 513: the token and 512 before it) with
  NeoX RoPE, theta 10000, full 128-dim rotation; 14 full-attention layers
  with **no positional encoding** (layers 0, 1, 5, 9, ..., 45, 48). From
  layer 2 the pattern is three sliding, one full. Context 262144.
- MoE: 256 routed experts of width 512, 6 per token, plus one always-on
  shared expert of width 1024. Router: `sigmoid(x W^T)`, top-6 (plus a zero
  `expert_bias`), weights renormalised to sum 1, scale 1.0, no groups.
- Active parameters per token (from the shapes):

| Part | Params/token | Share |
|---|---:|---:|
| attention q/k/v/o, 49 layers | 925M | 37% |
| routed experts, 47 × 6 × 3 × 2048 × 512 | 887M | 35% |
| output head 2048 × 150144 | 307M | 12% |
| shared experts, 47 × 3 × 2048 × 1024 | 296M | 12% |
| dense FFN (2 layers) + routers | 100M | 4% |
| **total** | **2.52B** | |

Routed experts are 95% of the 39.8B stored weights (21.3 GB of the 22.6 GB
file) but only 35% of what a token reads. Everything else, about 1.3 GB,
is read every token and stays resident in every mode. This is the shape
SSD streaming likes: a small hot set, a large cold set, 282 expert
selections per token at 1.69 MiB each.

Tokenizer: byte-level BPE, 150000 merged tokens plus 144 specials from
150000 (`<|begin_of_text|>` 150000, `<|end_of_text|>` 150001, `<|system|>`,
`<|user|>` 150003, `<|assistant|>` 150004, `<|observation|>` 150005,
`<think>`/`</think>` 150006-7, `<tools>`, `<tool_call>`, `<arg_key>`,
`<arg_value>`, `<tool_response>`, `<|developer|>` 150018). No BOS is added.
End-of-generation ids: 150001, 150003, 150005, 150018; the model ends a turn
by emitting the next role tag, as GLM does. The pre-tokenizer is three
sequential splits (`\p{N}{1,3}`, CJK runs, then a GPT-4o-style word pattern
with uppercase/lowercase alternation); rec's tokenizer (HF `tokenizers`) is
the arbiter, llama.cpp's `tokenizer.ggml.pre = "flint"` lists the regexes.

Chat template (`chat_template.jinja`): `<|system|>…`, `<|developer|>…`,
`<|user|>…`, `<|assistant|>` then `<think>…</think>` and content, tool
results as `<|observation|><tool_response>…</tool_response>`, generation
prompt `<|assistant|>`. Tools go in a `<|system|># Tools` preamble with one
JSON schema per line inside `<tools>`. Tool calls are single-line:
`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`,
strings raw, other values JSON. User text has role tags HTML-escaped.
Sampling defaults: temperature 1.0, top-p 0.95, top-k 20.

### The INT4 checkpoint

`rec pull` stores it under `~/.local/share/rec/models/<digest>/` with the HF
file names (five `model-0000N.safetensors`, 22.6 GB, `config.json`,
tokenizer, template). `config.json`: `quantization = {bits: 4, group_size:
64}`, `mlx_export = {bits: 4, embedding_bits: 8, routed_expert_bits: 4,
scale_rule: "mse"}`; the manifest: `weight_format: mlx-affine-int4,
kv_format: bf16`. MLX affine quantisation:

- `w = scale * q + bias` per group of 64 along the input dimension;
  `scales` and `biases` are BF16 `[out, in/64]`; `weight` is `U8 [out, in/2]`,
  MLX's packed `uint32 [out, in/8]` viewed as bytes, element `i` of a word
  in bits `4i..4i+3`. 4.5 bits per weight.
- Every `nn.Linear` is 4-bit g64, including `lm_head` (`[150144, 1024]` U8)
  and the routed experts (`switch_mlp.{gate,up,down}_proj.weight`
  `[256, 512, 1024]` and `[256, 2048, 256]` U8, scales `[256, 512, 32]` and
  `[256, 2048, 8]`).
- `embed_tokens` is 8-bit g64 (`U8 [150144, 2048]`, scales `[150144, 32]`).
- Norm weights, `softmax_offset` and the router (`mlp.gate.weight`
  `[256, 2048]`) are BF16.

## How rec runs it

From `engine/src/rec_mlx/` at rec 0.1.1. This is the specification for
steps 1 and 2; the source files are the reference when this summary is not
enough.

### Weights (`kernels/megakernel.py`, `FullModel.__init__`)

Thirteen "banks" of four layers each (the last also holds the final norm,
embedding and head), each a flat byte buffer with every tensor 256-byte
aligned in the order weight, scales, biases. A 64-word descriptor per layer
gives, for each projection, the three offsets plus `n` (rows) and `k`
(inputs); words 0-9 hold the bank id, `is_global`, `is_sparse`, the norm and
sink offsets and the router; words 61-62 the KV cache offset and capacity.
The router is kept as BF16 (`router_bf16`, checked to round-trip the fp32
values). Routed experts live inside the bank as `[256][rows][k/2]`, indexed
as `expert * n + row`. A `silu` lookup table over all 65536 bf16 bit
patterns serves the shared expert's activation.

### Decode (`flint_decode`, `_HEADER` + `_BODY`)

One persistent Metal kernel per token: grid `workers × 256` threads (M4
Pro/Max, M5 Pro: 36 workers, 32 "MLP workers", 32 head workers; M5 Max:
76/64/64), 8 simdgroups per threadgroup. Threadgroups synchronise with
device-wide atomic barriers (`sync_all`, a monotonic counter with a
10M-spin watchdog that aborts the kernel with a diagnostic instead of
hanging). Every threadgroup keeps the residual `h` (8 bf16 per thread) in
registers and recomputes each RMSNorm into threadgroup memory. Per layer:

1. `rms_256_w4(h) → n` (threadgroup), fp32 statistics, bf16 output.
2. Q/K/V: 40 units of one 128-row head each (32 q, 4 k, 4 v); `q4_local<2048>`
   is the 4-bit matvec: each lane loads 16 bf16 activations as fp32
   pre-scaled by 1, 1/16, 1/256, 1/4096 (`load16`), so a 16-bit word of
   four nibbles multiplies without shifts (`dot16`: `s*Σ(x·masked) + b·Σx`);
   fp32 accumulation per lane over k, `simd_sum`, bf16 store. Then QK
   RMSNorm and, on sliding layers, RoPE with `theta = pos · 2^(-pair/64 ·
   log2 10000)` in `fast::cos/sin`; K and V are written to the cache.
   The "KV teams" transform (`kernels/kv_teams.py`) splits this across
   `workers/4` teams with per-team barriers.
3. Attention (`attention_partial_path_w53` + `attention_merge_vec4_w2`):
   split-K over `parts` partitions (32 for sliding layers and short global
   contexts; 64 to 1024 for global contexts of 1024 tokens and more, where
   partial accumulators are rounded to bf16), one simdgroup per
   (KV head, `HEADS` query heads, part), online softmax with the sink as
   the initial max/denominator of part 0, bf16 K/V, fp32 scores, merge over
   parts with fp32 statistics. `scale = 0.08838834764831845`.
4. `q4<4096>` output projection, residual add in bf16, `rms_256_w4` again.
5. Router: `route_dot_bf16` (bf16 weights, fp32 accumulate, the dot
   rounded to bf16 before the sigmoid), then `w3_route_own_rank`: each
   expert team computes its own rank of the top-6 (ties to the lower
   index), renormalised weights `score/(Σ+1e-20)`.
6. Expert teams (`kernels/expert_teams.py`): of the 32 MLP workers, four
   threadgroups per routed role (6 × 4 = 24) and eight for the shared
   expert. `expert_team_gateup` computes gate and up together (two rows
   per simdgroup step), activation `bf16(bf16(g·σ(g))·u)` with the sigmoid
   in `fast::exp` for routed experts and the lookup table for the shared
   one; `expert_team_handoff` is a per-team barrier (the shared expert's
   eight groups and each role's four groups wait only for themselves);
   `expert_team_down` (`dot16_w197_bias23`, the exact-integer-float nibble
   decode). Dense layers 0-1 use `q4_local` gate/up, a plain SiLU, `q4<6144>`
   down.
7. Mix: `Σ_r bf16(down_r · bf16(w_r))` with bf16 rounding after every
   addition (MLX's order), plus the shared expert, plus the residual, all
   in bf16. Group 0 writes the residual to `trace[(layer+1)·2048]`.

After layer 48: final norm, `q4_local<2048>` over the 150144-row head by the
head workers, bf16 logits. `resident_block.py` wraps the body in a loop of
8 decode steps per launch with an in-kernel argmax (or the in-kernel
samplers in `sampling.py`/`masked_sampling.py` for temperature, top-p,
top-k, presence penalty and grammar masks), so the CPU sees tokens in
packets of 8. The transforms in `kernels/*.py` (expert teams, KV teams,
rank routing, mixed bits, fast activation, M4 Max loads) are string
rewrites of the base shader; for the port they are resolved once, for the
M4 Pro profile, into plain Metal.

### KV cache (`cache.py`, `m4_max.py`)

bf16, layout `[kv_head][capacity][128]` per layer. Full-attention layers
share banks of 65536 tokens (grown on demand; the kernel picks the bank by
`position >> 16`); sliding layers are 513-row rings with `insert = pos mod
513` once the ring is full. Metadata words 61-62 give each layer its offset
and capacity.

### Prefill (`quantized.py`, `m4_max_prefill.py`, `prefix.py`)

Not fused: the native MLX graph (`model.py`) in chunks of 2048 tokens
(4096 on M5), `SwitchGLU` (MLX `gather_qmm`) for experts, SDPA with sinks,
last-row logits only. On M4 Max: a steel GEMM over tokens sorted by expert
(`flint_prefill_gate_up`), dequantised dense GEMMs above 1024 rows, and
window attention in 512-row blocks. The prompt cache keeps the last chunk
boundary's sliding rings as a checkpoint for prefix divergence. **Decode KV
is never reused as a prefix**: after a reply, rec re-prefills the generated
tokens in the background (`prepare_prefix`) so the next turn's cached state
has batched-prefill arithmetic ("its arithmetic differs ... and changes
probabilities").

### API

`rec serve` (port 11500, key from `rec auth token`): OpenAI chat completions
and Anthropic messages; `logprobs: true` with `stream: false` returns the
chosen token's logprob; `top_logprobs` is unsupported. `--metrics FILE`
appends per-request JSON with prefill and decode seconds, token ids and
cache statistics. Full logits and the per-layer `trace` come from the
runtime directly (`~/.local/share/rec/runtimes/<sha>/`, extracted on first
serve): `FullModel.forward(state, token)` returns `(logits, trace)`.

## What ds4 already has

| Need | In the tree | Gap |
|---|---|---|
| mmap-backed GGUF weights as Metal buffers, model shape struct, fork variant pattern | upstream loader; Bonsai's `PQ2_0` type and fork files | `flint` metadata, three fork tensor types, per-layer descriptors built from GGUF offsets |
| Persistent kernel with device-wide barriers | none; every ds4 graph is one dispatch per op | the megakernel is new ground for ds4; port rec's `sync_all` and team barriers as they are |
| GQA attention with sinks, 128-wide | `flash_attn.metal` has sinks and a 128/128 typedef | not used: the port takes rec's split-K attention inside the megakernel; `flash_attn` 128/128 serves prefill |
| Routed-expert SSD streaming | Qwen3.8's bounded expert cache, `pread` into owned buffers, GPU address tables (`qwen4_expert_rows`), selected-id readback through an `MTLSharedEvent` handshake (#1328), overlap of misses with cached work, per-layer prefill staging, budget sizing from RAM and the Metal working set, failure semantics (skip zero entries, exact retry) | the same mechanisms applied to one kernel instead of a per-layer dispatch sequence (see Streaming) |
| Prefill GEMM | Bonsai's `kernel_bonsai2_gemm` family with decoded-fragment operands; `ds4_metal_matmul_f16_buffer` | an affine-int4 operand decoder; the sorted-expert GEMM from rec's M4 Max prefill |
| Tokenizer | GLM4, Qwen3.5 and DeepSeek pre-tokenizers in C with the Qwen unicode tables | `bpe_tokenize_text_flint` |
| Chat template, tool calls | GLM renderer and `<arg_key>/<arg_value>` parser | Flint variant (no `[gMASK]<sop>`, no newlines in `<tool_call>`, `<|developer|>`, no effort preamble) |
| Sessions, live KV reuse, disk checkpoints, agent | per-family plumbing in `ds4.c` | family decision below; ring + linear caches in the payload |
| Reference tooling | `tests/bonsai2/check_reference.py`, `tests/qwen38_27b/kld.c` | rec as the source; a trace comparison tool |

## Design

Fork rules apply ([FORK_MAINTAINING.md](FORK_MAINTAINING.md), copied from
`bonsai2` in step 0): Flint code in fork-owned files, one- to three-line
hooks in upstream files, upstream merged, never rebased. rec's shader code
is Apache 2.0: the ported kernels carry its notice in `licenses/` and in
`metal/flint.metal`.

### Family decision (step 0): a new family

Decided 2026-11-07. Flint is its own model family,
`DS4_MODEL_FAMILY_FLINT = 200` with `DS4_VARIANT_FLINT = 202` (fork
families and variants start at 200 so they never collide with upstream
additions; 200 and 201 are Bonsai's and the 27B branch's variants), GGUF
architecture string `flint`. The probe that preceded the decision (commit
`353abd7`) showed that a Qwen3.8-family variant was viable with about 35
hooks, and the GLM family was ruled out because what it shares is MLA/DSA
attention, a compact latent cache and the indexer, none of which Flint has
(its GLM-shaped chat format is a handful of product hooks either way). The
new family was chosen over the variant for cleanliness:

- **Nothing of Flint runs through the Qwen driver.** The variant route
  would have run every Flint session through `qwen4_session_*` with the
  GDN, PLE, MTP and indexer machinery switched off by shape values, and
  Flint's payload, KV rings and streaming reads hidden behind redirects
  inside Qwen functions. Upstream changes to Qwen internals could then break
  Flint silently, which `FORK_MAINTAINING.md` lists as Bonsai's main
  maintenance risk. With its own family every hook is a first-line
  `if (ds4_model_is_flint()) return flint_...(...)` and the driver lives in
  fork files.
- **The shape is honest.** No `n_full_attn_interval = 1` trick, no inert
  PLE layer index; `DS4_SHAPE_FLINT` carries only Flint's numbers
  (`n_swa = 513`, the per-layer global table lives in the shape include).
- **Family-wide predicates stay correct.** `ds4_model_is_qwen4()` keeps
  meaning Qwen3.8 in the 39 places it is tested, and the 16
  `ds4_session_is_qwen4()` branches keep their semantics.

The price is more hook sites, each still one to three lines. Shared
dispatch points a new family must touch, from the audit of upstream
`b3f5018` (function names as in `ds4.c`):

| Area | Upstream functions | Hooks |
|---|---|---|
| shape and arch | family/variant enums; `#include "ds4_flint_shape.inc"` after `ds4_qwen4_layer_is_nextn`; `config_validate_model` arch `flint` | 3 |
| GGUF types and weights | `gguf_types[143..145]`; `weights_have_output_head`, `weights_have_partial_output_head`, `weights_layer_has_required`, `weights_validate_layout`, `weights_bind_output`, `weights_bind_layer`, `weights_bind` (three family tests), `model_map_span_vec_include_layer_decode_static`/`_decode` | about 11 |
| tokenizer and chat | `bpe_tokenize_text`, `vocab_load`, `vocab_token_is_generation_stop`, `encode_chat_prompt`, `ds4_chat_begin`, `ds4_chat_append_think_prefix`, `ds4_chat_append_message`, `ds4_chat_append_assistant_prefix`, `ds4_chat_append_multimodal_message` (refuse) | 9 |
| engine | `ds4_engine_open_internal` (`flint_engine_supported` gate, graph setup branch, inspect/first-token branch, four "not Qwen" predicates), `ds4_engine_close` release, `ds4_engine_routed_quant_bits` (4), `ds4_engine_is_flint` in `ds4.h`, `engine_compute_entry_bytes`, `engine_per_tier_graph_overhead_bytes`, `ds4_context_memory_estimate_with_prefill_mode` (two sites), `ds4_engine_generate_argmax` (SSD branch), `generate_metal_graph_raw_swa`, `ds4_engine_mtp_draft_tokens` | about 15 |
| sessions | `struct ds4_session` gains `flint_graph` and its flags; `ds4_session_create` (CPU refusal and graph setup), `ds4_session_free`, `ds4_session_sync_internal` (prefill), `ds4_session_eval_internal` (decode), `ds4_session_payload_bytes`, `ds4_session_save_payload`, `ds4_session_load_payload`, `ds4_session_rewind`; `ds4_sessions_eval_batch_metal_supported`, `ds4_sessions_eval_batch_native`, the three speculative entry points and the two CUDA batch entry points refuse or skip Flint until step 7 | about 15 |
| streaming (step 4) | `streaming_layer_routed_expert_bytes`, `streaming_layer_gate_down_expert_bytes`, `ds4_streaming_cacheable_expert_count`, `weights_model_map_decode_static_supported`, the admission and staging estimates | about 5 |
| `ds4_server.c`, `ds4_agent.c`, `ds4_cli.c` | `server_model_syntax_for_engine`, `server_model_id_from_engine`, alias list, `send_models`, GLM renderer prefix and effort block; `agent_tool_syntax_for_engine`, compaction `<\|user\|>`; CLI rendered-prompt detection already lists `<\|user\|>` | about 8 |

About 65 sites against the variant's 35. The session driver
(`flint_session_create/free/prefill/decode/payload/rewind`) is fork code in
`ds4_flint_session.inc`, included after `struct ds4_session` like Bonsai's
batch file; the graph, KV cache and streaming orchestration are in
`ds4_flint.inc`. The generic pieces the family reuses unchanged are the
GGUF reader and mmap, `ds4_ssd.c` budgets and locks, the Metal expert cache
entries (`ds4_gpu_stream_expert_cache_*`), the `pread` task runner,
`ds4_gpu_buffer_address`, the session and checkpoint containers, the
tokenizer core (BPE merges, specials) and the server and agent loops. The
GLM streaming path's per-expert address-table mode
(`glm_stream_expert_cache_addr_layout_supported`) is the closest existing
code to the step-4 design and is read before writing Flint's.

### Fork-owned files

| File | Contents |
|---|---|
| `ds4_flint_shape.inc` | `DS4_SHAPE_FLINT`, `ds4_model_is_flint()`, per-layer global/sparse tables |
| `ds4_flint_model.inc` | `flint` metadata and tensor validation, binding, the per-layer descriptor table, engine gate |
| `ds4_flint.inc` | graph: decode launch, prefill schedule, KV cache (rings + banks), payload spans, streaming orchestration |
| `ds4_flint_session.inc` | session driver: create, free, prefill, decode, payload save/load, rewind (included after `struct ds4_session`) |
| `ds4_flint_gpu.h`, `ds4_metal_flint.m`, `metal/flint.metal` | kernel API, wrappers, the ported megakernel and prefill kernels |
| `ds4_flint_tokenizer.inc`, `ds4_server_flint.inc` | pre-tokenizer; template, tool syntax, model ids `flint`, `recursive/flint` |
| `gguf-tools/flint_int4_to_gguf.py` | rec checkpoint → GGUF, bit-exact, `--check` against `mx.dequantize` |
| `tests/flint/` | kernel tests, `trace_compare.py`, `golden.json`, `check_reference.py`, `rec_logits.py`, `route_trace.py`, `kld` |
| `docs/RECURSIVE_FLINT.md` | this file |

### Weights in GGUF

rec's layout is kept so the kernels port unchanged, with one change for
streaming: routed experts are stored **one pack per expert**.

- Non-expert tensors keep MLX's three arrays: `*.weight` as the fork type
  `FLINT_U4 = 143` (raw packed nibbles, `[out, in/2]` bytes) or `FLINT_U8 =
  144` for the embedding, `*.scales` and `*.biases` as BF16 `[out, in/64]`.
  llama.cpp's names (`blk.N.attn_q`, `attn_output`, `attn_q_norm`,
  `attn_k_norm`, `attn_sinks`, `ffn_norm`, `ffn_gate_inp` for the router,
  `ffn_gate_shexp`, ..., `token_embd`, `output_norm`, `output`) with the
  suffixes `.weight`, `.scales`, `.biases`. `general.alignment = 256`, so
  the mmap'd file is rec's bank. The per-layer 64-word descriptor is
  computed at load from the tensor offsets; the GGUF carries no offsets of
  its own.
- Routed experts: `blk.N.ffn_exps.pack`, fork type `FLINT_EXPERT_PACK =
  145`, shape `[256, 1769472]` bytes: per expert, in order, `gate.weight`
  (524288), `gate.scales` (32768), `gate.biases` (32768), the same for `up`,
  then `down.weight` (524288), `down.scales` (32768), `down.biases`
  (32768). 1.6875 MiB, already 256-aligned. One `pread` per expert in
  streaming mode; resident mode maps the same bytes.
- Metadata, written by `gguf-tools/flint_int4_to_gguf.py` and checked
  field by field in `config_validate_flint_model`: `flint.block_count`,
  `context_length`, `embedding_length`, `vocab_size`, the three
  feed-forward lengths, expert counts, `leading_dense_block_count`, the
  attention head counts and lengths, `attention.sliding_window`,
  `attention.layer_norm_rms_epsilon`, `rope.dimension_count`,
  `rope.freq_base`, `full_attention_layers` (the 14 global layers),
  `eog_token_ids`, `weight_format = mlx-affine-int4`, `quant.bits`,
  `quant.group_size`, `quant.embedding_bits`, `expert_pack.layout`,
  `expert_pack.offsets`, `expert_pack.bytes`, and the checkpoint SHA-256s
  from rec's manifest under `flint.checkpoint.sha256.<file>`. The tokenizer
  is byte-level BPE from the checkpoint's `tokenizer.json` (150144 tokens,
  149744 merges, `tokenizer.ggml.pre = flint`, no BOS, EOS 150001) and the
  chat template travels along in `tokenizer.chat_template`.

The kernel's expert addressing changes from `bank + offset + expert·n·k/2`
to `pack_base(expert) + fixed offset`, where `pack_base` comes from a
per-layer table of 256 GPU addresses. In resident mode the table holds
`mmap + tensor offset + expert · 1769472`; in streaming mode it holds cache
slot addresses or zero. That table is the whole interface between the
megakernel and the streaming code.

### Decode

`metal/flint.metal` holds rec's `_HEADER` and `_BODY` with the M4 Pro
profile's transforms applied once (expert teams for 32 MLP workers, KV
teams for 36 workers, own-rank routing, mixed-bits rounding, fast
activation), with the following deliberate differences, each checked by
the trace test:

- Weight addressing through the descriptor table built at load and the
  expert address table (above).
- One token per launch, logits read back, ds4's CPU samplers, grammar and
  tool parsing as for every other model. rec's 8-token resident blocks are
  a step 3 decision: at 7-8 ms per token a launch plus readback costs a few
  percent; streaming wants the CPU between tokens anyway.
- The `trace` and `diagnostic` outputs stay: the watchdog's diagnostic is
  how a lost barrier surfaces, and the trace is the test interface.

Launch envelope: 36 threadgroups of 256 must be co-resident or the barriers
deadlock into the watchdog. rec's `DeviceProfile` encodes what each chip
sustains; ds4 reads the same device name and refuses others until profiled.

### KV cache and sessions

bf16 as rec: 14 linear caches (2 KB per position per layer; 7.3 GB at 262K)
and 35 rings of 513 rows (37 MB total). Disk checkpoints and live KV reuse
serialise both. Following rec's rule that decode KV is not reused as a
prefix is the open question of step 2: ds4's sessions reuse decode KV
everywhere, and whether the bf16 difference between a decoded row and a
prefilled one matters for Flint is measured, not assumed (gate 2).

### Prefill

Chunks of 2048 tokens as rec on M4 Pro, through Flint's own kernels in
`metal/flint_prefill.metal` rather than Bonsai's GEMM (the fork rules want
Flint out of ds4_metal.m, and the MLX arithmetic to follow is specific):

- `flint_gemm` (templated): 64 x 64 x 32 tiles, four simdgroups of 32 x 32,
  bf16 simdgroup matrices with f32 accumulation; the int4-affine operand is
  dequantized to bf16 with one rounding (`bf16(scale * q + bias)`, MLX's
  `QuantizedBlockLoader`) and a bf16 variant serves the router. Epilogues:
  store, residual add, GLU (`bf16(silu(bf16(g)) * bf16(u))` with the silu
  table) and the routed combine (sequential bf16 sum of the six weighted
  expert outputs, then the shared expert, then the residual: MLX's `sum`
  over a bf16 axis rounds after every add, as the decode kernel does).
- Routed experts: `flint_route_topk` (sigmoid, own-rank top-6 with ties to
  the lower index, weights `bf16(p / (sum + 1e-20))`, per-expert counts),
  `flint_route_offsets` (prefix sums, a tile table of 64 rows per expert,
  indirect dispatch arguments), `flint_route_scatter`, then
  `flint_moe_gateup` and `flint_moe_down` over the tile table, reading each
  expert's pack through the same GPU-address table the decode kernel uses
  (the model buffer must be made resident with `useResource`).
- `flint_attention`: 64 queries x 1 head per threadgroup, key tiles of 32,
  f32 online softmax in base 2 with the scale folded in, sinks as the
  initial running maximum with sum 1, probabilities rounded to bf16 before
  the PV product, as `mx.fast.scaled_dot_product_attention`'s steel kernel;
  masks causal or the 513 window; keys from up to four KV blocks (global
  layers) or from a window buffer (sliding layers: the previous 512
  positions gathered from the ring, then the chunk).
- `flint_head_norm_rope`, `flint_rmsnorm`, `flint_embed`, `flint_ring_copy`
  (ring gather before, scatter after a chunk: the ring is re-laid for
  position + n, which becomes rec's `initial`), `flint_head_row` (the
  last row's logits with the decode kernel's vector arithmetic).

The MLX facts these follow were measured in the venv: `sum` over a bf16
axis rounds per add; `quantized_matmul` with 64 rows matches an f32
reference with single-rounded bf16 weights on 99.99% of elements (max 1
ulp); the hardware lane layout of 8 x 8 simdgroup matrices is MLX's
`BaseMMAFrag::get_coord` (the epilogues depend on it; the kernel test
checks it through the GEMM). Prefill cannot be bit-identical to MLX's
tiles; it is held to rec's own spread (Status, step 2). Short continuations
go through the same chunk path (rows pad to 64 inside the tiles).

Session state beyond the token checkpoint: the KV blocks, `ring_initial`
(the prefill length the rings are laid out for), and the previous step's
logits so a one-token rewind is exact; longer rewinds invalidate and the
next sync rebuilds (rings hold only the last 513 positions). Payloads carry
the header (tag `FLNT`), tokens, logits, `ring_initial`, the global layers'
rows and the whole rings.

### Streaming (step 4)

What ds4 does for Qwen3.8 on Metal, adapted to a megakernel that cannot
be told anything while it runs. Built in `ds4_flint_stream.c` (plain C, no
Metal) with the GPU side in `ds4_metal_flint.m`:

1. **Resident bank.** In streaming mode every tensor the descriptors
   reference except the expert packs (1.19 GiB) is copied into an owned
   Metal buffer with compact offsets, and the tensor region of the file
   mapping is unmapped; the file is reopened for `pread` with `F_NOCACHE`
   (`DS4_FLINT_STREAM_CACHED=1` keeps the page cache). Streaming is on
   with `--ssd-streaming`, or automatically when the file is larger than
   the GPU's recommended working set.
2. **Cache.** Expert slots of 1.6875 MiB in shared buffers of up to 1024
   slots, reached through GPU addresses (every encoder marks them
   resident). The kernel's `(layer, expert)` table holds the slot's address
   or zero. Exact LRU over the slots; the selection being fetched is
   pinned. Budget: `--ssd-streaming-cache-experts N|<GB>`, else ds4's plan
   (86% of the recommended working set minus the bank, the prefill
   scratch, the two staging buffers, the sessions' KV at the requested
   context and a 1 GiB reserve). At least 64 slots.
3. **Decode misses: abort and resume.** The first experiment of the step
   settled the in-kernel wait: a kernel spinning on a shared word sees a
   CPU store only after 0.3 to 10.7 s, and the CPU sees a GPU store about
   30 ms after launch (10 of 10 runs, checksums correct, 36 threadgroups);
   Apple's GPU caches flush at command-buffer boundaries, not before. So
   after routing each routed team looks its expert up in the table; a zero
   entry publishes the layer's six ids in the request buffer, sets rec's
   abort word and returns, and every other group leaves at its next
   barrier. The host reads the request words, has the stream module
   `pread` the missing packs (a pool of 6 threads, in parallel) into free or
   evicted slots, writes the addresses, and relaunches the kernel with
   `layer_start = L`: the residual entering layer L is the trace row the
   previous layer wrote, the layer's KV rows are rewritten identically.
   Hits run the unchanged resident code, so streamed logits equal resident
   logits bit for bit. A relaunch costs about 0.3 ms on top of the reads.
   The request buffer also records every sparse layer's selection, which
   `DS4_FLINT_ROUTE_LOG=FILE` appends per token for the cache studies.
4. **Prefill staging** (policy by chunk length, `DS4_FLINT_PREFILL_CACHE_MAX`
   32 and `DS4_FLINT_PREFILL_FULL_MIN` 128). After a sparse layer's routing
   the host waits, reads the per-expert counts, and fills a per-parity
   table of 256 addresses for the layer's grouped GEMMs: chunks under 32
   tokens take the needed experts through the cache (unless they would
   fill more than half of it); longer chunks stage the needed experts in
   one of two layer-sized buffers (432 MiB each); from 128 tokens the whole
   layer is read as one contiguous range split across the pool, and the
   next layer is read in the background while this one computes (the
   routing wait proves the buffer's previous user has finished).
5. **Exactness.** `make test-flint-stream` runs the module over a sparse
   temporary file shaped like the pack region (signatures at both ends of
   every pack): cold misses, hits without reads, LRU with pins, touch, the
   three prefill policies and the read-ahead, a failed read returning its
   slot, dense-layer requests, a too-small cache. `tests/flint/stream_check.py`
   runs the golden cases resident and streamed with a capped cache and
   compares every step's argmax and log-probabilities exactly.
6. **Speed** (step 6). Routing logs of three 400-token generations
   (code, prose, an agent-style debugging answer), replayed through an
   exact LRU (`route_sim.py`, warm-up a quarter of the tokens; the three
   logs touch 98% of all experts):

| Cache | Slots | Miss fraction | MB/token | Layers that relaunch | tok/s at 5 GB/s |
|---:|---:|---:|---:|---:|---:|
| 2 GB | 1213 | 0.645 | 322 | 46.9 | ~12 |
| 4 GB | 2427 | 0.437 | 218 | 45.5 | ~15 |
| 8 GB | 4854 | 0.206 | 103 | 35.2 | ~26 |
| 12 GB | 7281 | 0.089 | 45 | 20.2 | ~44 |
| 16 GB | 9709 | 0.030 | 15 | 7.8 | ~78 |
| 20 GB | 12136 | 0.007 | 3.5 | 1.9 | ~114 |

   The estimate is 7.5 ms compute plus the read time plus 0.3 ms per
   relaunching layer. A 16 GB Mac (about 8 GB of cache) lands near 26
   tok/s with plain LRU, a 24 GB Mac (14 GB) near 60, a 32 GB Mac holds
   everything after warm-up. The miss fraction is what step 6's prefetch
   (predict the next layers' experts from the previous token, fill slots
   that were free at launch) and popularity preload have to cut; the SSD's
   cold random-read bandwidth is still unmeasured here because the file
   lives in the page cache of the 48 GB development machine.

Memory targets, with 1.19 GiB of resident weights, 0.84 GiB of staging,
2 GB runtime reserve and context at 8K (0.3 GB): a 16 GB Mac holds about
an 8 GB cache (37% of experts), 24 GB about 14 GB (65%), 32 GB the whole
21.3 GB. On the 48 GB development machine streaming is tested with
`--ssd-streaming-cache-experts` capped to those sizes.

## Speed model

Resident decode, bytes per step at 4.5 bits: 1.42 GB; at the 226 GB/s ds4's
matvecs sustain on this chip, about 160 tok/s before attention and barrier
costs. rec's megakernel on the same chip is the measured ceiling (step 3);
the port should land within a few percent of it, since it runs the same
kernel with one launch per token instead of one per eight.

| Term | At 4K | At 32K | At 128K |
|---|---:|---:|---:|
| weights | 6.3 ms | 6.3 ms | 6.3 ms |
| KV reads, 14 global layers, bf16 (2 KB/position/layer) | 0.5 ms | 4.2 ms | 16.6 ms |
| 35 sliding layers, 513 rows | <0.1 ms | <0.1 ms | <0.1 ms |
| **bound** | **147 tok/s** | **95 tok/s** | **44 tok/s** |

rec's 125 tok/s "over a 128K session" is above this bound at 128K; step 3
measured rec at 137 tok/s at 256 tokens, 118 at 4K, 91 at 16K and 75 at
28K (ds4 within 5% of each), so the figure is weighted toward the short
part of a session. The bound above is roughly right.

## Verification

- **Converter**: `--check` dequantises every tensor and matches
  `mx.dequantize` on the source bit for bit; expert packs are checked
  against the source slices byte for byte.
- **Trace test** (`tests/flint/trace_compare.py`, step 1's gate): for a set
  of tokens and positions, run rec's `FullModel.forward` in its runtime and
  ds4's decode on the same KV contents (both caches written from the same
  prefill, exported from rec), and compare the 50 × 2048 residual trace and
  the logits. Decode must be bit-identical; a difference names the layer
  and the phase.
- **Kernel tests** (`make test-flint-kernels`), model-free: `dot16` and
  `load16` against a double reference over all nibble patterns, RMSNorm,
  head norm and RoPE, split-K attention with sinks at the part-count
  boundaries (1023/1024, 8192, 32768), own-rank routing with ties, expert
  teams against the unfused expert loop, the sorted GEMM against a plain
  dequantised matmul, the barrier watchdog.
- **Golden reference** (`make test-flint-reference`): `golden.json` from
  `rec serve` (`temperature 0` is argmax in `engine.py`; `logprobs: true`,
  `stream: false`): Bonsai's six case shapes plus a 20K case; tokens must
  match, logprobs within rec's own spread across prefill chunk sizes.
- **Streaming**: `tests/flint/test_streaming` as above; a full-model run
  with a capped cache must reproduce the resident logits exactly.
- **Sessions**: `ds4_test --server`, checkpoint save/restore and rewind
  across a ring wrap, the 48K story test.
- **Speed**: `ds4-bench` 2K-128K against `rec serve --metrics`, same
  machine, AC power, High Power Mode, never both engines at once.

## Quality

Same weights and, for decode, the same arithmetic. Gates:

1. **Trace and streaming exactness** as above, after every kernel change.
2. **KLD against rec** (`tests/flint/kld`, rec's full logits from
   `rec_logits.py` over 3072 teacher-forced positions) for everything that
   is not bit-exact: the prefill path, and decode continuing from ds4's own
   prefill or from reused decode KV. The tolerance is rec's own spread
   between prefill chunk sizes, measured first.
3. **Official continuations** (`gguf-tools/quality-testing`): the repo's
   prompts collected from the hosted model (OpenRouter
   `recursive/flint-v1.0:free` if it returns top logprobs, else Recursive's
   API), scored by ds4, with rec's greedy first tokens on the same prompts
   as the comparison. This catches template and tokenizer drift that the
   token-id-based gates cannot.

## Steps

### 0. Scaffolding and converter

Copy `fork.mk`, `ds4_metal_internal.h`, `docs/FORK_MAINTAINING.md` and the
CI workflow from `bonsai2`, renamed. Write `flint_int4_to_gguf.py` and run
it on the pulled checkpoint; `--check` passes. Loader: shape, metadata
validation (types 143-145, BF16, nothing else), binding, descriptor table,
address table in resident mode, pre-tokenizer, server id. Family decision
recorded. Deliverable: `--dump-tokens` matches rec's tokenizer on 20
strings (CJK, digits, contractions, escaped role tags, specials);
`golden.json` collected; rec's runtime found and `rec_logits.py` dumping
logits and traces.

### 1. Decode megakernel

Port the resolved shader, the launch, the readback, the KV banks and rings.
Deliverable: `trace_compare.py` reports zero differing elements on 64
positions across sliding and global layers, at contexts below and above
the 1024-part boundary.

### 2. Prefill and sessions

Deliverable: golden check passes on all cases; `ds4_test --server` passes;
48K story test; the decode-KV-reuse question measured with gate 2.
Done (Status). Left for later: prefill speed (step 3), the tool preamble
and thinking control of the template (step 5), batching (step 7).

### 3. Resident measurements

Done (Status). Decode is settled: ds4 is within 5% of rec at every
context up to 28K and the one-launch-per-token floor is 0.9 ms, so the
8-token resident blocks stay out. Prefill is the open gap: 1.5x at 4K
growing to 2.8x at 28K. Order of work: the attention kernel (larger key
tiles, two heads or more queries per threadgroup, double-buffered loads),
then the GEMM (double buffering, 128-wide tiles), measured with the same
harness after each change. 128K needs a longer prompt file
(`tests/generate_long_context_story_prompt.py`).

### 4. Streaming

Done (Status and Design: Streaming). The visibility experiment ruled out
the in-kernel wait; abort-and-resume replaced both it and the per-layer
split. Left for later: cold-SSD measurements and the capped-cache speed
table at 8/14/22 GB (needs the page cache purged, root), the 20K case in
streaming mode, and prefetch (step 6).

### 5. Server and agent

Done (Status). rec's `chat.py` has no thinking switch: the model always
thinks (the grammar forces `<think>`), so ds4's `<think></think>` prefill
is ds4's addition; the empty think block is what the template renders for
an assistant turn without reasoning content, so it is in distribution.
Left: the Terminal-Bench style task against `rec launch`.

### 6. Streaming speed

Prefetch from the previous token's routing, popularity preload at startup,
cache policy variants, each kept only with a measured gain on the
routing-trace workloads and unchanged logits.

### 7. Later

CUDA on DGX Spark (the megakernel is Metal; CUDA would start from the Qwen
expert kernels), `--batched-session`, 8-token resident blocks with
in-kernel sampling if step 3 says they pay, other quants.

## Open questions

- Resolved (step 4): CPU-to-GPU visibility inside a running kernel takes
  seconds on Apple GPUs; the kernel aborts on a miss and resumes from the
  layer instead.
- Resolved (step 2, gate 2): reusing decode KV as a prefix moves the
  distribution by an order of magnitude less than rec's own decode-vs-
  prefill difference; ds4 keeps reusing it.
- Where rec's 125 tok/s at 128K comes from with bf16 KV (step 3).
- Resolved (step 5): thinking off prefills the template's empty think
  block; rec itself cannot switch thinking off.
- The M4 Pro's sustained SSD read bandwidth under the streaming access
  pattern (1.7 MiB random reads), which sets the whole streaming table.

## References

- Announcement: https://www.recursive.com/updates/introducing-recursive-flint
- Model card and config: https://huggingface.co/RecursiveSI/flint;
  INT4 checkpoint: https://huggingface.co/RecursiveSI/flint-int4
- rec: https://github.com/recursive-oss/rec, `engine/src/rec_mlx/`:
  `kernels/megakernel.py` (shader and packing), `kernels/{expert_teams,
  kv_teams,rank_routing,mixed_bits,resident_block,activation_optimization,
  m4_max,device}.py` (transforms and profiles), `quantized.py`, `cache.py`,
  `m4_max.py`, `m4_max_prefill.py`, `kernels/m4_max_prefill.py`, `prefix.py`,
  `engine.py`, `server.py`, `scheduler.py`
- Out of scope for now: `RecursiveSI/flint-int4-gguf` (Q4_1),
  `RecursiveSI/flint-nvfp4-gguf`, `unsloth/flint-GGUF`; llama.cpp PR #30928
- Hosted: https://openrouter.ai/recursive/flint-v1.0:free
- ds4: issue #1260, PR #1328 (streamed decode handshake);
  `docs/QWEN38_FLASH_NEXT.md` (SSD streaming behaviour and tests),
  `docs/QWEN38_27B.md` (method), `docs/FORK_MAINTAINING.md` (rules),
  `tests/test_qwen4_ssd_experts.c`, `tests/bonsai2/check_reference.py`,
  `tests/qwen38_27b/kld.c` (tooling to adapt)
