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
| 1. Decode megakernel ported: per-layer residual trace bit-identical to rec on identical KV; one token per launch, CPU sampling | **done 2026-11-07.** `metal/flint.metal` + `ds4_metal_flint.m` + `ds4_flint.inc`; `trace_compare.py` reports every trace row (50 × 2048) and all 150144 logits identical to rec on 56 decode steps over four prefills: 5 tokens, 559 (ring wrapped), 1401 (above the 1024-part boundary, 8-head path, 128 parts) and 8445 (256 parts). Warm decode 7.3 ms/token at 5 tokens, 8.0 at 559, 8.4 at 1401, 10.0 at 8445 (one token per launch with logits readback); a cold first step pays about 200 ms of page-ins. Not yet exercised: count exactly 1024 (64 parts), 512 parts, and contexts past 65536 (multi-block KV). **2026-11-14:** the worker count is a shader macro (8 to 36, step 4) and a runtime choice: pinned, profiled (36) or probed at start-up on other Apple GPUs; traces byte-identical at every count, streaming bit-identical at 16 and 8; 134 / 137 / 124 / 112 / 67 tok/s at 36 / 32 / 24 / 16 / 8 workers on the M4 Pro (Design: Decode). |
| 2. Prefill (chunked, window attention, sorted-expert GEMM), sessions, live KV reuse, disk checkpoints; golden cases within rec's spread | **done 2026-11-09.** `metal/flint_prefill.metal` (own kernels, see Design: Prefill), `ds4_flint_session.inc`; `make test-flint-kernels` passes. Against rec (`prefill_compare.py`): the last row's logits agree to KL 1e-8 to 1e-9 on 12-, 578- and 2556-token prompts; per-layer error grows like rounding noise (relative mean 0.003 to 0.12 by layer 48) with a few rows diverging at routing near-ties (578-token prompt: 55 of 578 rows have a sixth/seventh expert margin under 1%). Golden teacher-forced (`check_golden.py`, 128 tokens): argmax = rec's 94.5 to 100%, abs delta logprob mean 0.011 to 0.039, max 0.60; rec's own decode-vs-prefill at identical positions is mean 0.05 to 0.32, max 4.5, agreement 88 to 100% (3 dumps, 40 positions). Greedy paths part after 15 to 66 tokens on the short cases and run 64/64 on the tool case. Gate 2 (`reuse_check.py`): a follow-up turn on decode KV versus a fresh prefill differs along 64 steps by KL mean 0.001 to 0.005, max 0.064, abs delta logprob mean 0.011 to 0.032, top-1 same 97 to 100%: reuse stays. `ds4_test --server`, `--session-snapshot` and `--long-context` (29624 tokens, 100K context, two KV blocks) pass. Speed (not tuned): prefill 170 tok/s at 20K (rec 410 to 425 on this Mac), the first-cut GEMM; decode through sessions 10.3 to 11.4 ms/token short, 21.3 at 20K. |
| 3. Measure against rec: decode at 4K/32K/128K, prefill, launch overhead; decide on 8-token resident blocks | **done 2026-11-11** (mains, High Power Mode, M4 Pro 48 GB, `tests/flint/bench_vs_rec.py`: both servers through one streaming client, thinking on; the harness accounts for rec sending its tokens in bursts of 8, which had undercounted rec's decode time by 5% and overcounted its prefill by 7 decode steps). Decode ms/token rec vs ds4: 7.7 vs 7.6 at 256 tokens, 9.0 vs 8.8 at 4K, 11.6 vs 11.6 at 16K, 14.0 vs 14.0 at 28K, 23.7 vs 23.0 at 64K, 39.4 vs 38.1 at 128K (148K-token prompt, 64 new tokens): the same kernel at the same speed. The launch floor (`DS4_FLINT_LAYER_COUNT=0`) is 0.92 ms per step; a 64K step is 23.2 ms GPU against 23.4 ms wall (`DS4_FLINT_DECODE_PROFILE=1`), so **8-token resident blocks would buy at most 1 to 4%: decision no.** Prefill tok/s rec vs ds4 after the attention rewrite and the MoE tile heights (step 3): 449 vs 589 at 256 (warm), 684 vs 852 at 1K, 671 vs 856 at 4K, 573 vs 708 at 16K, 501 vs 578 at 28K, 349 vs 400 at 64K, 220 vs 239 at 128K: ds4 is 9 to 31% ahead at every length, with prefill output bit-identical to the first kernels. A cold first prefill pays about 8 s of page-ins for the 22.6 GB mapping. **Decode attention loads (2026-11-11):** one 8-byte vector load per lane for K and for V per token instead of eight scalars (bit-identical on all four trace dumps) takes decode to 7.6 / 8.6 / 13.2 / 21.4 / 34.2 ms at 256 / 4K / 28K / 64K / 128K (rec 7.7 / 9.0 / 14.0 / 23.7 / 39.4); deeper prefetch, launch overlap and MoE register prefetch measured no gain (step 3). |
| 4. Expert streaming: routing-trace study, cache and address table, in-kernel slot wait (or per-layer split), prefill staging, budget; streamed output bit-identical to resident | **done 2026-11-09** (`ds4_flint_stream.c`, see Design: Streaming). The visibility experiment failed usefully: a CPU store reaches a running kernel only after 0.3 to 10.7 s and a GPU store reaches the CPU about 30 ms after launch (caches flush at command-buffer boundaries), so misses are handled by **abort and resume**: the megakernel aborts at the first layer whose expert is not cached, publishes the layer's ids, the host reads the packs and relaunches from that layer. Hits are the unchanged resident path. `stream_check.py`: streamed and resident outputs are bit-identical on `short_01`, `multi_01` and `tool_01` with a 1 GB cache (606 slots; every decode layer misses, 4.8 of 6 experts), under all three prefill policies. `make test-flint-stream` (model-free, sparse file) passes; `ds4_test --server` and `--session-snapshot` pass with `DS4_TEST_SSD_STREAMING=1` and a 2 GB cache. Cost of a relaunch about 0.3 ms plus the reads (35 ms/token with a 1 GB cache and the file in the page cache, against 9 resident). Routing logs over 1197 decoded tokens (`route_sim.py`, plain LRU): miss fraction 0.65 at 2 GB, 0.21 at 8 GB, 0.09 at 12 GB, 0.03 at 16 GB, 0.007 at 20 GB. Not measured: cold-SSD bandwidth (the file sits in this Mac's page cache and `purge` needs root) and the 20K case in streaming mode (deferred to an AC-powered session). |
| 5. Server and agent: Flint template, `<tool_call>` syntax, reasoning parsing, `ds4-agent` session | **done 2026-11-09.** `ds4_flint_server.inc`: the server renders Flint prompts from rec's `chat_template.jinja` (roles in order, `flint_text` escaping, the `# Tools` preamble before the first non-system turn, `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>` with string values raw and other values as Python JSON, `<|observation|><tool_response>` results); `tests/flint/check_template.py` renders nine message sets through the live server and through rec's template under HF's jinja settings: all byte-identical, and a live request with tools returns a structured `tool_calls` entry plus `reasoning_content`. Flint reports GLM's tool syntax, so the GLM reply parser and the live tool-call tracker are reused unchanged. Thinking: rec's grammar forces `<think>` as the first token, so the generation prompt is `<|assistant|><think>`; `enable_thinking: false`, `reasoning_effort: none` and the `flint-chat` / `flint-no-think` aliases prefill `<think></think>` instead (`flint-reasoner` forces it on); `/v1/models` lists `flint`, `flint-chat`, `flint-reasoner`. Sampling defaults from `generation_config.json` (temperature 1.0, top-p 0.95, top-k 20) for knobs a request leaves out; the agent uses 1.0 / 0.95. `ds4-agent` with Flint: thinking, bash and read tool calls and the final answer work in one turn on a file task; its KV save needed `ds4_engine_routed_quant_bits` to report 4 for the packs. Not done: the Terminal-Bench style comparison with `rec launch` (needs the AC-powered session). **2026-11-13:** live KV reuse for clients that drop `reasoning_content` (tau2 re-prefilled every turn): Flint's own thinking-visible key; array/object/number arguments were passed to clients as JSON strings (tau2 retail 0.333): the parser now follows the template's `tojson` rule per tool schema; rerun retail 0.867, airline 0.733, equal to Bonsai 2, see step 5. |
| 6. Streaming speed: prefetch and prediction, popularity preload, cache policy | **done 2026-11-11** (Design: Streaming, point 6). The listed ideas were studied on routing traces first and none pays: consecutive tokens share 16% of a layer's experts (the previous token's experts are already cached), popularity preload only helps the cold start, SLRU saves at most 4% of misses and LFU loses, and lookahead prediction of later layers from the current residual (recall 0.84 of the six at P=8 one layer ahead) is a wash at 8 GB in simulation (the wrong reads cost the bandwidth the avoided relaunches save) and about 10% at 14 GB. The measured cost was elsewhere: per-launch profiling (`DS4_FLINT_STREAM_PROFILE=1|2|3`) showed every streaming launch running 1.7x slower than the same layers in resident decode, with commit-to-start and completion latencies doubled, because the fetch gaps (0.3 to 1 ms, about 30 per token at 8 GB) let the GPU power manager drop the clocks. A keep-alive like ds4's TP one (`flint_keepalive`: one threadgroup, 2000 FMA pairs, back-to-back on a side queue while a streaming session decodes) fixes it: 300-token agent-style generation, greedy, 8 GB cache 23 -> 36 tok/s, 14 GB 27 -> 41, 22 GB (no steady-state misses) 22 -> 38; per-launch GPU time back to the resident 0.142 ms/layer, queue 0.07 -> 0.04 ms, wake 0.06 -> 0.04 ms. Output bit-identical with and without it (`stream_check.py`). Not kept: a same-queue heater sized to the fetch (restores GPU speed but slows the reads), long keep-alive dispatches (command buffers on two queues do not overlap on this Mac: 30000 iterations cost 5 tok/s, 1.2M made each launch wait 10 to 15 ms). Open: lookahead prefetch for caches of 14 GB and up (in-kernel prediction from the residual, ~10% simulated). |
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
   parts with fp32 statistics. `scale = 0.08838834764831845`. Each lane
   reads its four dims of K and of V as one 8-byte vector per token (rec
   reads eight scalars; same values, same arithmetic); `FLINT_KV_PREFETCH`
   (1, 2 or 4 tokens in flight, env `DS4_FLINT_KV_PREFETCH`) is 1: the
   loop is issue-bound, deeper prefetch measured no faster.
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

Launch envelope: the worker threadgroups must be co-resident or the
barriers never complete; each barrier spins at most 10M iterations, then
the group writes the phase into the diagnostic buffer and returns, and the
host reports `GPU synchronization failed: group G timed out in phase P`.
rec's `DeviceProfile` (`rec_mlx/kernels/device.py`) lists what each chip
sustains: M4 Pro and M4 Max (36 workers, 32 MLP, 32 head), M5 Pro (the
same), M5 Max (76 / 64 / 64); every other Metal device raises
`Unsupported resident kernel device`. That is about two threadgroups per
GPU core (16 cores on the M4 Pro, 40 on the M5 Max), so a 10-core base
M4 would need 3.6 per core at 36.

**Worker count as a parameter (2026-11-14).** The port's shader was rec's
generated text for 36 workers, with the derived numbers (KV teams of 9
groups, 72 simdgroups per team in the attention loops, expert teams of 4
plus 8 shared workers, 392 QKV tickets) as literals. They are now macros
derived from `FLINT_WORKERS`, following rec's generators (`kv_teams.py`,
`expert_teams.py`, `rank_routing.py`): four KV teams of W/4 groups, expert
teams of W/8 groups (rec's 4 from 32 up) for the six routed experts and a
double team for the shared one, up to 32 head workers; W is any multiple
of 4 from 8 to 36 (rec's 76-worker M5 Max shape, which uses the other
KV-team code path, is not ported, so an M5 Max runs 36). Every output row
is still produced by one simdgroup whatever the count, so the arithmetic
is the same, and it is: 128 greedy tokens at 32, 24, 16 and 8 workers give
the same text and a byte-identical 50 × 2048 residual trace per token as
36, and streaming with a 1 GB cache (the abort-and-resume path with
2-group and 1-group expert teams) is bit-identical to resident on all four
`stream_check.py` cases at 16 and at 8 workers, the 20K story included.
Speed on the M4 Pro, 300 greedy tokens: 134 tok/s at 36 workers, 137 at
32, 124 at 24, 112 at 16, 67 at 8 (resident); with an 8 GB streaming
cache 42 at 36 and 35 at 16. So the count is a bandwidth knob that costs
little down to 16 and halves at 8.

The host compiles the shader with the count (`ds4_metal_flint.m`,
`flint_choose_workers`): `DS4_FLINT_WORKERS=N` pins it on any chip, the
profiled chips (M4 Pro, M4 Max) get 36, and an unprofiled chip is probed
at start-up. The probe launches the decode kernel with `layer_count` 0
and `parameters[6]` = 256 extra grid-wide barriers (embedding, 257
barriers, head), for every candidate from 36 down to 8 in steps of 4: two
warm-up launches, then the best of three timed ones; a candidate whose
groups do not all pass a barrier within the spin budget is skipped, and
the largest candidate within 10x of the fastest wins, printed as a table
(`ds4: Flint: probe 36 workers: 1.51 ms per launch ...`) and a choice line
with the pin to reuse. The 10x slack is because resident counts differ
only by the head's share of the launch and by clock noise (1.1 to 1.8 ms
warm, up to 5 ms on a cold GPU across 8 to 36 on this Mac), while a count
the GPU could only time-slice would pay a scheduler slice per barrier, two
orders of magnitude. What the M4 Pro actually does when over-committed
(one-off build with the cap raised, `DS4_METAL_FLINT_SOURCE` on a shader
with the cap raised): 64 and 128 groups all pass the 256 barriers in under
3 ms, so the 20-core GPU keeps at least 128 groups of this kernel resident
and rec's 36 is a speed choice there, not the residency limit; 256 groups
time out cleanly ("GPU synchronization failed: group 0 timed out in phase
1", about 3 s per launch), which is what the probe relies on for a GPU
that cannot hold a count. Decoding at 64 or 128 crawls (2.8 and 1.05
tok/s, already 130 ms per token on the two dense layers alone), a kernel
effect above the supported range that was not chased. The probe therefore
matters on the 7- to 10-core GPUs (the pipeline reports 1024 threads per
threadgroup, so at least four groups per core fit: 28 on a 7-core M1, 32
on an 8-core, 40 on a 10-core M4) and is expected to say 36 on most
others. Each count compiles in 0.3 to 0.4 s the first time and comes from
the system shader cache afterwards, so a cold sweep adds about 3 s to the
first start. `DS4_FLINT_FORCE_PROBE=1` runs the sweep on a profiled chip
(diagnostic); the sweep's first start on an unprofiled chip is the first
thing to report from such a machine (Field measurements).

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
  `flint_route_offsets` (prefix sums, tile tables, indirect dispatch
  arguments), `flint_route_scatter`, then `flint_moe_gateup` and
  `flint_moe_down` over the tile tables, reading each expert's pack
  through the same GPU-address table the decode kernel uses (the model
  buffer must be made resident with `useResource`). Tiles come in four
  heights: an expert with c rows gets c / 64 tiles of 64 rows and one
  remainder tile of 64, 32, 16 or 8 rows, each height its own table and
  indirect dispatch (`flint_moe_gateup_64` .. `_8`; 64 = 2 x 2 simdgroups
  of 32 x 32, smaller = 1 x 4 of height x 16). A tile's MMA work is
  proportional to its height while its weight reads are not, and with 6
  of 256 experts per token an expert sees about 6 rows per 256-token
  chunk, so the 64-row tiles made the MoE GEMMs cost the same at 256
  tokens as at 2048 (825 ms of a 1.0 s prefill); with the four heights
  218 ms. Every output element accumulates its k products in the same
  order in every height, so rows are bit-identical whichever tile holds
  them (`DS4_FLINT_MOE_TILES=64` restores the single height for A/B
  runs).
- `flint_attention`: 64 queries x 1 head per threadgroup of 8 simdgroups
  (8 query rows each), Q in threadgroup memory and re-read per d step, K
  and V sharing one 32-key tile, f32 online softmax in base 2 with the
  scale folded in, sinks as the initial running maximum with sum 1,
  probabilities rounded to bf16 before the PV product, as
  `mx.fast.scaled_dot_product_attention`'s steel kernel; masks causal or
  the 513 window, evaluated only on tiles touching an edge; keys from up to
  four KV blocks (global layers) or from a window buffer (sliding layers:
  the previous 512 positions gathered from the ring, then the chunk). The
  kernel is a template (`fp_attention_body<simdgroups, rows, Q in
  registers, key tile, shared K/V>`); the first instance, 4 simdgroups of
  16 rows with Q in registers, ran at 0.7 TFLOPS because its 16-row output
  tile spills registers, the production instance runs at 4.6 TFLOPS (MLX's
  sdpa on the same shape: 4.7). `tests/flint/bench_flint_attention` times
  the instances in `tests/flint/flint_attention_variants.metal` against
  each other; `bench_flint_gemm` does the same for the projections
  (`flint_gemm` 5.0 TFLOPS, `mx.quantized_matmul` 5.0).
  `DS4_FLINT_PREFILL_PROFILE=1` prints the GPU time per kernel group of a
  prefill (each group then runs in its own command buffer).
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

   That estimate (7.5 ms compute plus reads plus 0.3 ms per relaunch) was
   wrong about where the time goes. Measured on 2026-11-11 (M4 Pro 48 GB,
   mains, High Power Mode, F_NOCACHE reads, 300-token greedy generation of
   an agent-style answer, caches capped to what 16 / 24 / 32 GB Macs hold):

   - **SSD.** Random 1.77 MB `pread`s run at 2.6 to 3.6 GB/s on one
     thread (0.34 to 0.5 ms per pack) and plateau at 6.5 to 6.8 GB/s from
     6 threads; a batch of 2 packs takes 0.5 ms, 6 packs 1.1 ms; splitting
     a pack into slices gains nothing. Read time varies by run (8 to 17 ms
     per token at 8 GB for the same tokens), so timed runs are bracketed
     with a read benchmark.
   - **Launch anatomy** (resident decode, probe launches): 0.133 ms per
     layer, 0.09 ms for a layer that aborts after routing, no fixed cost
     for the launch, residency or prologue; commit-to-start 0.06 ms,
     completion-to-wake 0.05 ms. The misses of a token spread over about
     30 layers at 8 GB (48 misses in 30 relaunches), 13 at 14 GB, 8 at
     22 GB warm.
   - **The clock.** Per-launch logging (`DS4_FLINT_STREAM_PROFILE=3`)
     showed streaming launches at 0.21 to 0.31 ms per layer plus 0.25 to
     0.6 ms per launch, 1.7x the resident cost, intermittent (the fastest
     tenth of launches ran at resident speed), with queue and wake
     latencies doubled. A standalone test reproduced it: a memory-bound
     kernel launched after 0.4 to 1 ms of GPU idleness runs 2x slower,
     with or without `pread`s in the gap. ds4's tensor-parallel path
     documents the same effect ("the GPU power manager halves the clocks
     within a second") and runs a keep-alive kernel on a side queue.
   - **Keep-alive** (`flint_keepalive`, `ds4_metal_flint.m`): one
     threadgroup of 2000 FMA pairs (about 30 us) dispatched back-to-back
     on its own queue by its own thread while a streaming session decodes,
     paused during prefill. Streaming decode tok/s without / with it:
     8 GB cache 23.0 / 36.3, 14 GB 26.7 / 40.9, 22 GB 21.7 / 37.9; GPU
     time per token 24.8 / 11.8 ms at 8 GB (resident decode is 7.2),
     per-launch cost back to 0.13 + 0.142 ms/layer, queue 0.04 ms, wake
     0.04 ms. 500 to 8000 iterations are equivalent; 30000 cost 5 tok/s and
     ds4's 1.2M made every launch wait 10 to 15 ms, because command
     buffers on different queues do not overlap on this Mac (a long spinner
     on a second queue serialised the main queue in an earlier test); four
     threadgroups gain nothing. A same-queue heater committed before each
     fetch and sized to its expected length restored the GPU speed but
     slowed the reads by 4 ms per token (not kept).
     CPU cost: the process goes from about 35% to 70% of one core while
     decoding (the keep-alive thread and Metal's completion threads), a
     few GPU watts. `DS4_FLINT_STREAM_KEEPALIVE=0` turns it off, `=N` sets
     threadgroups, `DS4_FLINT_STREAM_KEEPALIVE_ITERS` the loop.
   - **Prefetch and policy studies** (`DS4_FLINT_ROUTE_LOG`, six 399-token
     generations: code, prose, agent, math, chat, JSON; `DS4_FLINT_TRACE_DUMP`
     for the residual rows): consecutive tokens share 16% of a layer's
     experts, so prefetching the previous token's selection adds nothing
     (they are the cache's most recent entries); the most popular 40% of
     experts cover 76% of selections, so a popularity preload only
     shortens the cold start; SLRU saves at most 4% of misses, LFU loses.
     Predicting layer L+k's routing from the residual entering layer L
     (the router of L+k applied to it): recall of the true six among the
     top 8 is 0.94 at k=0, 0.84 at k=1, 0.77 at k=2, 0.72 at k=3 (all six
     predicted: 0.71 / 0.38 / 0.23 / 0.15). Simulated with the measured
     read costs, fetching the predicted misses of the next layer together
     with the current ones is a wash at 8 GB (relaunches 32 -> 22 per
     token, reads 61 -> 89 packs) and about 10% at 14 GB; not built.
   - **Prefill** in streaming mode reads the union of the prompt's experts
     (10 GiB for a short agent prompt) through the staging path; unchanged.

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
part of a session. At 128K itself (step 3, 2026-11-11) ds4 decodes at
38.1 ms/token (26 tok/s) and rec at 39.4 ms (25 tok/s), both below the
44 tok/s bound: the KV read of 3.8 GB per token ran at about 120 GB/s,
half the matvec rate, on both (the same kernel). With the vector K/V
loads (step 3, "Decode attention loads") ds4 decodes 128K at 34.2
ms/token (29 tok/s, 142 GB/s over the KV read); the partial path is
issue-bound (per token and lane: two loads, eight `simd_sum`s, 64 FMAs
and an exp), so what is left is instruction count, not latency.

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
- **Worker counts** (2026-11-14): `DS4_FLINT_TRACE_DUMP` at 32, 24, 16 and
  8 workers against 36 on 128 greedy tokens, byte-identical traces and
  text; `stream_check.py --cache 1GB` at 16 and at 8 workers, all four
  cases IDENTICAL (Design: Decode, worker count).
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
8-token resident blocks stay out.

Prefill was 1.5x behind rec at 4K and 2.8x at 28K. The model-free
micro-benchmarks (`make bench-flint-kernels`, GPU timestamps, one
2048-query chunk) located all of it in the attention kernel: 0.70 TFLOPS
against 16K keys (700 ms per global layer and chunk) where MLX's sdpa
does 4.7, while `flint_gemm` already matched `mx.quantized_matmul` (5.0
TFLOPS both on every projection shape). The cause was the output tile:
four simdgroups of 16 rows hold 64 f32 output fragments per lane plus Q
in registers and spill. Variants measured on the same harness (M4 Max,
eval traffic on the GPU, so absolute numbers are a little low):

| Variant (64 queries x 1 head unless noted) | 16K keys | TFLOPS |
|---|---:|---:|
| old: 4 simdgroups x 16 rows, Q in registers | 731 ms | 0.70 |
| 4 x 16 rows, Q in threadgroup memory | 678 ms | 0.76 |
| 4 x 16 rows, Q in registers, masks on edge tiles only | 717 ms | 0.72 |
| 4 x 8 rows (32 queries), Q in registers | 141 ms | 3.66 |
| 4 x 8 rows (32 queries), Q in threadgroup memory | 117 ms | 4.40 |
| 4 x 8 rows (32 queries), 64-key tiles | 119 ms | 4.35 |
| 8 x 8 rows, Q in registers | 130 ms | 3.97 |
| 8 x 8 rows, Q in registers, 64-key tiles | 126 ms | 4.09 |
| **8 x 8 rows, Q in threadgroup memory, shared 32-key K/V tile** | **113 ms** | **4.57** |
| MLX `scaled_dot_product_attention`, same shape | 109 ms | 4.73 |

The winner is bit-identical to the old kernel (same fragment order), also
at 4K keys (22 ms, 4.72 TFLOPS) and on the 513 window (4.0 ms, 4.28; MLX
with an explicit window mask takes 17.6 ms there). `make
test-flint-kernels` passes.

On the model (2026-11-11, mains, High Power Mode, GPU otherwise idle,
`bench_vs_rec.py`, 128 new tokens, best of 2; 64K and 128K from five
copies of the story file, 148K tokens, 64 new tokens, one run):

| Prompt tokens | rec prefill tok/s | ds4 before | ds4 after | rec decode ms | ds4 decode ms |
|---:|---:|---:|---:|---:|---:|
| 256 | 449 | 201 | 212 | 7.7 | 7.6 |
| 4096 | 671 | 433 | 758 | 9.0 | 8.8 |
| 16384 | 573 | 234 | 634 | 11.6 | 11.6 |
| 28000 | 501 | 164 | 543 | 14.0 | 14.0 |
| 65536 | 349 | | 370 | 23.7 | 23.0 |
| 131072 | 220 | | 239 | 39.4 | 38.1 |

(rec's columns re-measured 2026-11-11 with the burst-aware harness, see
below; its first measurement had read 7.3/8.5/10.9/13.3/20.5/34.8 ms and
398/659/569/463/348/226 tok/s.) ds4's prefill went from 1.5 to 2.8x
behind rec to ahead at every length, with the model output unchanged: `prefill_compare.py` on rec's
2556-token dump prints the same table as before the rewrite (argmax
agreement 92.25%, mean KL 0.153), and `stream_check.py` reports all eight
cases identical between resident and streaming, the 20K story included
(resident prefill 33.8 s, streaming 34.4 s, was about 100 s). The profile
of that 2556-token prefill (`DS4_FLINT_PREFILL_PROFILE=1`, 4.06 s GPU in
total): MoE gate/up 36%, MoE down 16%, q/k/v GEMM 13%, o proj 10%,
attention 8% (global layers half of it), shared GLU 7%, head 7%. The
next prefill target, if one is wanted, is the MoE GEMM (6 of 256 experts
per token gives about 48 rows per expert per 2048-token chunk, fewer
with uneven routing, against 64-row tiles of width 512), not attention.

Short prompts: at 256 tokens ds4 took 1.21 s where rec takes 0.64 s,
and 1024 tokens took 1.67 s. The profile named it: the two MoE GEMMs
took 825 ms at 256 tokens, as much as at 2048, because every expert
paid a 64-row tile for its 6 rows. With the four tile heights (Design:
Prefill, `DS4_FLINT_MOE_TILES=64` for the old behaviour) the GPU time of
a 256-token prefill fell from 997 to 393 ms (1024: 1609 to 1140; 2048:
2525 to 2266), rows stay bit-identical (`prefill_compare.py` unchanged),
and through the server:

| Prompt tokens | rec prefill tok/s | ds4 64-row tiles | ds4 four heights |
|---:|---:|---:|---:|
| 256 | 449 | 212 | 589 |
| 1024 | 684 | 615 | 852 |
| 4096 | 671 | 758 | 856 |
| 16384 | 573 | 634 | 708 |
| 28000 | 501 | 543 | 578 |

What remains of the MoE GEMMs at 2048 tokens (940 ms of 2.27 s) is the
tiles' dequantization and smem traffic at 48 rows per expert; a 16-column
variant or a sorted two-expert tile would be the next step there.

Decode at 64K and 128K first read 10 to 12% behind rec. `DS4_FLINT_DECODE_PROFILE=1`
put the 64K step at 23.2 ms GPU against 23.4 ms wall, so the kernel
itself, not the launch, and the model in an owned buffer instead of the
file mapping ran at the same speed (8.24 ms per step either way), so not
the memory backing. The cause was the harness: rec sends its tokens in
bursts of 8 (eight events within 5 ms, then a gap of 8 decode steps),
so the first event arrives 7 steps after the first token; timing from
it to the last event over n - 1 tokens undercounts rec's decode by 5%
at 128 new tokens (12% at 64) and overcounts its prefill by 7 steps.
`bench_vs_rec.py` now times decode from the end of the first burst over
the tokens after it, and takes the burst's extra steps out of the prefill
time. Back to back on the same day (ds4, rec, ds4 at 256 and 64K) the
two engines decode within 2% of each other everywhere, ds4 a hair ahead
at 64K and 128K.

**Decode attention loads (2026-11-11).** The partial path read each lane's
four dims of K and of V as eight scalar 16-bit loads per token, each
through the KV view's block select. One 8-byte vector load each for K
and V (same values, same arithmetic; `trace_compare.py` identical on all
four dumps) cuts the step at 64K from 23.4 to 21.4 ms and at 128K from
38.1 to 34.2 ms:

| Context | before ms/token | after | change |
|---:|---:|---:|---:|
| 256 | 7.6 | 7.6 | 0 |
| 4096 | 8.8 | 8.6 | -2% |
| 28000 | 14.0 | 13.2 | -6% |
| 65536 | 23.0 | 21.4 | -7% |
| 131072 | 38.1 | 34.2 | -10% |

Deeper prefetch was the first hypothesis and is wrong: with named slots
(`FLINT_KV_PREFETCH` 2 or 4) the 64K step is 24.8 ms against 23.4 for
one token in flight, and 21.7 against 21.4 with the vector loads, so the
loop is bound by instruction issue, not by load latency (per token and
lane: the loads, eight `simd_sum`s over the heads, 64 FMAs, one exp, two
shuffles per head). Two pitfalls on the way: an indexed ring
`next_key[D][4]`, even under `#pragma unroll`, went to private memory
and ran 2 to 4x slower (depth 8 even miscompared), hence the
preprocessor-expanded slots; and computing the KV block and row offset
by hand instead of through the view's `address()` was 3% slower.

Two other candidates from the same list were tried individually on their
own branches and measured no gain; they stay there as records:

- `flint-launch-overlap`: encode the next step's command buffer (own
  queue, parameters in a buffer) while the current step runs. 7.8 and
  8.9 ms/token at 256 and 4K with and without it, GPU time per step
  unchanged: encoding was never on the critical path. The host costs
  about 0.3 ms per token in all (0.17 ms commit-to-start and completion
  latency inside the decode call, the rest sampler, detokenizer and SSE).
- `flint-moe-prefetch`: stage the next k tile of the MoE GEMMs in
  registers while the current one multiplies (the expert weights are
  read from DRAM once, so the hypothesis was exposed load latency). One
  2048-token chunk: gate/up 603 ms before, 722 with A rows and q4 words
  staged, 714 with the q4 words only; down 280, 302, 296. Output
  byte-identical (prefill traces of pf_mid and pf_long). The extra live
  registers cost more than the overlap gains; the MoE GEMM at 48 rows
  per expert remains the open prefill item.

### 4. Streaming

Done (Status and Design: Streaming). The visibility experiment ruled out
the in-kernel wait; abort-and-resume replaced both it and the per-layer
split. Left for later: cold-SSD measurements and the capped-cache speed
table at 8/14/22 GB (needs the page cache purged, root) and prefetch
(step 6). The 20K case in streaming mode is identical to resident
(`stream_check.py`, 2026-11-11: 34.4 s streaming vs 33.8 s resident
prefill, decode 41.8 vs 12.9 ms/token with the 1 GB cache).

### 5. Server and agent

Done (Status). rec's `chat.py` has no thinking switch: the model always
thinks (the grammar forces `<think>`), so ds4's `<think></think>` prefill
is ds4's addition; the template renders `<|assistant|>` and the content
with no think block at all for a history turn without reasoning content,
so the empty block only ever appears at the generation prompt.

**Live KV reuse with reasoning dropped (2026-11-13).** The tau2 pilot on
the agent bench logged `request reuses nothing` on every one of 599
requests and re-prefilled the whole conversation each turn (5 to 10
minutes at 90K to 116K tokens). tau2, like most OpenAI-style harnesses,
echoes content and tool calls byte for byte and drops `reasoning_content`,
so the next prompt parts from the live KV at the `<think>` of the
generation prompt. ds4's thinking-visible key (the Qwen path: remember the
bytes the next request will render for the current frontier, continue
from live KV with only the new suffix tokenized, never rebuild the KV)
was gated off for Flint (tools present, GLM syntax) and would have built
the wrong bytes (`<think></think>` plus trimmed content). Flint now
builds its own key in `ds4_flint_server.inc`: the generation prompt
without its trailing `<think>` or `<think></think>`, the content as
returned, and for tool-call turns the sampled call text that the renderer
replays by id; the stop token is not in the KV, so the suffix (`<|user|>`
or `<|observation|>`) evaluates it. Replaying five tau2 turns from the
pilot's `requests.jsonl` against the server: every follow-up continued
from live KV (`thinking live continuation match=visible-prefix`),
prefilling 12 to 376 new tokens instead of 4.9K to 5.8K, 0.8 to 2.3 s per
turn instead of 5 to 7 s. Unit test `test_flint_visible_checkpoint`
(`ds4_test --server`). Left: the Terminal-Bench style task against
`rec launch`.

**Typed tool-call arguments (2026-11-13).** The same pilot's retail and
airline tasks ended in `too_many_errors` on 8 of 15 and 7 of 15 tasks
(retail 0.333 against Bonsai's 0.867) because array parameters reached
the tools as JSON strings: `{"item_ids": "[\"8538875209\"]"}`. The
template writes a tool-call argument as `v | tojson if v is not string
else v`, so the model emits strings raw and arrays, objects, numbers and
booleans as JSON text; the upstream GLM reply parser takes every
`<arg_value>` as a string. `flint_glm_arg_is_string` inverts the template
rule the way the Qwen parser does: a parameter the tool schema declares
as `string` stays text whatever it looks like (zip codes, ids), any other
value that parses as one complete non-string JSON value is passed on as
JSON, and anything else stays text. The parser gets the request's tool
schemas from the post-generation parse (the streaming completeness check
and the repair pass run without schemas and are only used as predicates).
Replaying pilot record 102 against the rebuilt server returns
`"item_ids": ["7602931732","9570044148"]`; 2135 of the pilot's 15107
tool calls carried such a string. Unit test `test_flint_typed_arguments`.
Rerun of the 30 retail and airline tasks on the rebuilt server
(`agent-bench/results/flint-int4/tau2-20261113T125333Z-pilot`, 51 min
for 30 tasks): retail 0.867 and airline 0.733, the same scores as local
Bonsai 2, with no `too_many_errors` termination (15 of 30 before the fix;
airline's one `infrastructure_error` is counted as 0). Caveat: the history renderer writes a non-string value with the client's
bytes, so a client that re-serializes an array differently from the
model's `["a", "b"]` spacing renders a tool turn that no longer matches
the live KV key and that turn re-prefills (Python's `json.dumps` default
matches).

### 6. Streaming speed

Done (Status and Design: Streaming, point 6). The listed ideas (prefetch
from the previous token, popularity preload, cache policy variants) were
measured on routing traces and none pays; the gain came from keeping the
GPU clocked through the fetch gaps (23 -> 36 tok/s at 8 GB, 27 -> 41 at
14 GB), with logits unchanged. Instrumentation kept:
`DS4_FLINT_STREAM_PROFILE=1` (per token), `=2` (per launch: host, encode,
queue, GPU, wake, fetch; GPU split by aborted and final launches), `=3`
(one line per launch), `DS4_FLINT_TRACE_DUMP=FILE` (the 50 x 2048 bf16
residual rows per token), `DS4_FLINT_STREAM_THREADS`, `DS4_FLINT_WORKERS=N`
(decode threadgroups, Design: Decode), `DS4_FLINT_FORCE_PROBE=1`. Left: lookahead
prefetch from in-kernel prediction for caches of 14 GB and up (about 10%
simulated), and a check of the keep-alive on a 16 GB Mac (here it costs
about a third of a CPU core and a few GPU watts).

### 7. Later

CUDA on DGX Spark (the megakernel is Metal; CUDA would start from the Qwen
expert kernels), `--batched-session`, 8-token resident blocks with
in-kernel sampling if step 3 says they pay, other quants.

## Field measurements (how to help)

Everything above was measured on one M4 Pro with 48 GB, where streaming
is only ever simulated with a capped cache and the SSD is the one Apple
ships in that machine. What the branch needs now are numbers from other
Macs, above all the 16, 24 and 32 GB ones streaming was built for. Each
contribution is one row in the table at the end of this section plus the
raw log lines, sent as a PR against `recursive-flint` on
`github.com/mschneider/ds4` (or an issue there with the logs attached).

**Which Macs can run it.** The decode megakernel runs a fixed number of
threadgroups that must all be resident on the GPU at once (Design:
Decode). rec profiled 36 on M4 Pro and M4 Max (and M5 Pro and M5 Max) and
refuses every other chip; ds4 uses 36 on M4 Pro and M4 Max and on any
other Apple GPU probes at start-up, for every count from 36 down to 8 in
steps of 4, whether the GPU holds it and how fast it passes a run of
barriers, then decodes with the largest count that is resident and fast
(the log prints one `ds4: Flint: probe N workers: X ms per launch` line
per count and a choice line). On the M4 Pro the GPU holds far more than
36 groups, so a base M4 or an M1, M2 or M3 with 8 or 10 GPU cores is
expected to be told 36 too and to run at the speed its memory bandwidth
allows; a 7-core M1 may land on 28 or 24. The three things to report from
such a machine are the probe table, the chosen count and the resulting
tok/s (Rules for a timed run). If decode later stops with `GPU
synchronization failed: group G timed out in phase P`, or runs at a crawl
(under 10 tok/s resident), rerun with `DS4_FLINT_WORKERS=N` one step
lower and report both. A 48 GB or larger M4 Pro/Max contributes by
capping the cache (`--ssd-streaming-cache-experts 8GB`).

### Setup

1. `git clone -b recursive-flint git@github.com:mschneider/ds4.git && cd ds4
   && make ds4 ds4-server ds4-bench` (Xcode command line tools). Run every
   binary from the repo directory: the Metal sources are found relative to
   the working directory.
2. The model: `rec pull` (rec 0.1.1, https://github.com/recursive-oss/rec)
   downloads the INT4 checkpoint to `~/.local/share/rec/models/<digest>/`
   on any Mac, whether or not rec will run there; then
   `python3 gguf-tools/flint_int4_to_gguf.py --out gguf/flint/flint-int4.gguf`
   (numpy; `--check` also needs mlx) writes the 22.6 GB GGUF in about a
   minute. Copying a converted `flint-int4.gguf` from another machine is
   equivalent (the converter is bit-exact).
3. Smoke test, which also shows the plan ds4 made for your memory:

   ```
   ./ds4 -m gguf/flint/flint-int4.gguf --ssd-streaming -n 64 --temp 0 -p "Write a haiku about SSDs."
   ```

   The line `ds4: Flint SSD streaming: resident weights ... expert cache
   X GiB (N of 12032 experts) ... KV reserve ...` is the first thing to
   report. Streaming switches itself on when the file is larger than the
   GPU's recommended working set, so on Macs under 48 GB `--ssd-streaming`
   is redundant but harmless.
4. Python for the exactness check: a venv with `numpy` (`tokenizers` as well
   for `bench_vs_rec.py`).

### Rules for a timed run

- Mains power and High Power Mode: `pmset -g batt | head -1` says
  `AC Power`, `pmset -g | awk '/powermode/ {print $2}'` prints `2` (on
  Macs without the setting, note that). Charging from a low battery
  throttles the GPU by 10 to 20x.
- Nothing else on the GPU or the SSD: no other model process (`pgrep -fl
  'ds4|rec|llama|mlx'`), no Time Machine or Spotlight indexing run in
  progress, no browser video. One ds4 process at a time (the single-instance
  lock refuses a second); never rec and ds4 together.
- Watch for swapping: `sysctl vm.swapusage` before and after a run; a cache
  that pushes the machine into swap is not a valid measurement, shrink it.
- Every run prints its own `Flint SSD streaming:` line; keep it with the
  numbers. Run each configuration twice and report the second (the first
  pays the cold-start reads).

### What to measure

**A. Streaming decode against cache size.** The 300-token agent-style
greedy generation used in step 6, with the per-token profile on:

```
DS4_FLINT_STREAM_PROFILE=1 ./ds4 -m gguf/flint/flint-int4.gguf --ssd-streaming \
    --ssd-streaming-cache-experts 8GB -n 300 --seed 1 --temp 0 \
    -p "You are debugging a Rust web service that returns 502 errors under load. Walk through, step by step, which commands you would run, what the likely causes are, and how you would fix each one."
```

Once with ds4's own plan (no `--ssd-streaming-cache-experts`), then at
8 GB and 14 GB where they fit, then at the largest cap the machine takes
without swapping (grow in 2 GB steps). Report the last `Flint streaming
profile at position ...` line (ms/token, launches, misses, ms reading per
token) and the `prefill: ... generation: ... t/s` line. The reading time
per token is the SSD measurement; nothing else is needed for it.

**B. The keep-alive on a small Mac.** Same command at the 8 GB (or auto)
cap with `DS4_FLINT_STREAM_KEEPALIVE=0`, and the generation tok/s of both.
On the 48 GB machine the keep-alive is worth 23 -> 36 tok/s at 8 GB and
costs a third of a CPU core; whether the GPU clock behaves the same on an
M1/M2/M3 or a base M4, and what it does to a 16 GB machine's fan, is not
known. If you can, note the package power from `sudo powermetrics
--samplers gpu_power -i 1000 -n 10` during decode with and without it.

**C. Long context.** `ds4-bench` runs Flint in streaming mode and sweeps
prompt lengths; `tests/long_context_story_prompt.txt` holds about 40K
tokens, so concatenate copies for the long frontiers:

```
for i in 1 2 3 4; do cat tests/long_context_story_prompt.txt; done > /tmp/story_x4.txt
./ds4-bench -m gguf/flint/flint-int4.gguf --ssd-streaming --prompt-file /tmp/story_x4.txt \
    --ctx-start 2048 --ctx-max 131072 --step-mul 2 --gen-tokens 128 --csv flint_field.csv
```

Add `--ssd-streaming-cache-experts` when you want a fixed cache; without it
the plan shrinks the cache as `--ctx-max` grows, because the KV reserve
takes 28 KB per token of context (0.9 GiB at 32K, 3.5 GiB at 128K) and is
the right thing to see on a small machine. Report the CSV (`prefill_tps`,
`gen_steady_tps` per frontier) and the streaming banner. 128K needs
`--ctx-max 131072` plus the KV to fit; on 16 GB stop where the plan
refuses (`cannot size the expert cache`) and say where that was.

**D. Exactness, once per machine.** Streamed output must equal resident
output bit for bit, and the model-free module test must pass:

```
make test-flint-stream
venv/bin/python tests/flint/stream_check.py --model gguf/flint/flint-int4.gguf --cache 1GB --max-tokens 32
```

The resident run of `stream_check.py` needs 48 GB; on a smaller Mac the
"resident" side streams too, under ds4's own plan, so the check becomes
auto plan against a 1 GB cache. That is still the test that matters
(every decode layer misses at 1 GB): the two must be IDENTICAL on all
eight golden cases, the 20K story included. Also `diff` the stdout of two
runs of A at different caps: greedy decode must give the same text.

**E. Against rec (48 GB and up).** `tests/flint/bench_vs_rec.py` as in
step 3: `rec serve --context 40000`, then `--engine rec`; `rec stop`; then
`./ds4-server -m gguf/flint/flint-int4.gguf --ctx 40000 --port 18089` and
`--engine ds4`. The frontiers default to 256 / 4K / 16K / 28K.

### What to report

One row here, plus the raw lines in the PR description:

| Date | Mac (chip, GPU cores, RAM, SSD model) | macOS | Commit | Cache (plan / caps) | A: tok/s at each cap, ms reading per token | B: keep-alive off / on | C: gen tok/s at 4K / 32K / 128K | D |
|---|---|---|---|---|---|---|---|---|
| 2026-11-11 | M4 Pro 16c GPU, 48 GB, AP2048Z | 26.7 | 35a0428 | caps 8 / 14 / 22 GB | 36 / 41 / 38, 8 to 17 ms at 8 GB | 23 / 36 at 8 GB | see step 3 (resident) | identical |

`sysctl -n hw.memsize machdep.cpu.brand_string`, `system_profiler
SPDisplaysDataType | grep Cores`, `system_profiler SPNVMeDataType | grep
Model` and `sw_vers -productVersion` give the machine columns.

## Open questions

- Resolved (step 4): CPU-to-GPU visibility inside a running kernel takes
  seconds on Apple GPUs; the kernel aborts on a miss and resumes from the
  layer instead.
- Resolved (step 2, gate 2): reusing decode KV as a prefix moves the
  distribution by an order of magnitude less than rec's own decode-vs-
  prefill difference; ds4 keeps reusing it.
- Resolved (step 3): rec's 125 tok/s at 128K is a session average; at
  128K rec decodes at 39.4 ms/token and ds4 at 38.1 (Speed model).
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
