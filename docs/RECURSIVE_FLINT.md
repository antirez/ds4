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
| 0. Scaffolding: fork files from `bonsai2`; converter from rec's checkpoint (rec's tensor layout, per-expert packs); shape, loader, tokenizer, server id | open |
| 1. Decode megakernel ported: per-layer residual trace bit-identical to rec on identical KV; one token per launch, CPU sampling | open |
| 2. Prefill (chunked, window attention, sorted-expert GEMM), sessions, live KV reuse, disk checkpoints; golden cases within rec's spread | open |
| 3. Measure against rec: decode at 4K/32K/128K, prefill, launch overhead; decide on 8-token resident blocks | open |
| 4. Expert streaming: routing-trace study, cache and address table, in-kernel slot wait (or per-layer split), prefill staging, budget; streamed output bit-identical to resident | open |
| 5. Server and agent: Flint template, `<tool_call>` syntax, reasoning parsing, `ds4-agent` session | open |
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

### Family decision (step 0)

Bonsai 2 and the 27B run as variants of the Qwen3.8 family because their
layer geometry matches it. Flint shares no layer type with any ds4 family.
Two options:

1. **Qwen3.8-family variant** (`DS4_VARIANT_FLINT = 201`, Bonsai's pattern):
   known contact points (about 45 lines in `ds4.c`), sessions, disk KV,
   server and the SSD-streaming engine plumbing (`--ssd-streaming`, cache
   budget options, the expert cache structure) come along. Risk: the shared
   session and payload code may not accept zero-width recurrent state.
2. **New family**: clean, but about 60 `ds4_model_is_qwen4()` sites need a
   Flint branch, and the streaming plumbing would be duplicated.

The streaming reuse tips this toward option 1. Probe it first in step 0
(inert `n_lin_*`, `n_hc = 1`, `n_ple_layer = DS4_MAX_LAYER`; own KV cache
in the fork graph; one session checkpoint round-trip); fall back to option
2 only if more than a handful of hooks are needed. Record the outcome here.

### Fork-owned files

| File | Contents |
|---|---|
| `ds4_flint_shape.inc` | `DS4_SHAPE_FLINT`, `ds4_model_is_flint()`, per-layer global/sparse tables |
| `ds4_flint_model.inc` | `flint` metadata and tensor validation, binding, the per-layer descriptor table, engine gate |
| `ds4_flint.inc` | graph: decode launch, prefill schedule, KV cache (rings + banks), payload spans, streaming orchestration |
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
  llama.cpp's names (`blk.N.attn_q`, `ffn_gate_shexp`, ...) with the
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
- Metadata: `flint.*` as llama.cpp writes it (block count, heads, window,
  the two 49-entry patterns, expert counts, `expert_gating_func = 2`), plus
  `flint.weight_format = mlx-affine-int4` and the checkpoint SHA-256s from
  rec's manifest.

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

Chunks of 2048 tokens as rec on M4 Pro. Dense projections through Bonsai's
GEMM with an affine-int4 operand decoder; experts through a port of rec's
sorted-token gate/up GEMM (`flint_prefill_gate_up`: tokens sorted by
expert, binary-searched expert boundaries, a steel-style block MMA with
`QuantizedBlockLoader` 4-bit g64 operands) and a matching down GEMM; sliding
attention in 512-row blocks against the ring, global attention with
`flash_attn` 128/128 and sinks; QK norm and RoPE as kernels. Prefill cannot
be bit-identical to MLX's `gather_qmm`; it is held to rec's own spread.
Short continuations (a few tokens after a long prefix) go through the
decode kernel with all-row logits, as ds4's other models do.

### Streaming (step 4)

What ds4 does for Qwen3.8 on Metal, moved into the megakernel:

1. **Cache.** Expert slots of 1.6875 MiB in owned buffers sized by the
   existing budget logic (87.5% of RAM or the Metal working set, minus
   context, prefill scratch, the 1.3 GB of resident weights and the
   runtime reserve). The address table maps `(layer, expert)` to a slot or
   zero. Eviction LRU with the layer's current selection pinned.
2. **Decode miss handling.** Routing happens inside the kernel, so the
   selected ids are known to the GPU first. The expert team of a missing
   expert publishes its request (`layer, expert, slot`) to a shared buffer
   and spins on the slot's ready word; the shared expert's eight groups and
   the teams whose experts are cached run meanwhile, which is the overlap
   the Qwen path gets from ordering kernels. A CPU thread polls the request
   buffer, `pread`s the pack into a free slot, updates the address table
   entry and the ready word. The spin reuses rec's watchdog so a failed
   read ends the kernel with a diagnostic and ds4's existing exact-retry
   path reruns the token.

   This depends on a CPU store to a shared buffer becoming visible to a
   running kernel's device-scope atomic loads. It is the first experiment
   of step 4, a 50-line kernel and a thread, before anything else is built.
   If it does not hold on Apple GPUs, the fallback is the per-layer split:
   the megakernel body ends after routing, publishes the ids through the
   `MTLSharedEvent` handshake of #1328, the CPU reads, and a second kernel
   runs the experts and the rest of the layer. 49 launches a token cost
   about half a millisecond, acceptable for streaming mode only.
3. **Prefill staging.** A 2048-token chunk touches nearly every expert of a
   layer, so prefill reads whole layers (432 MiB each) into staging while
   the previous layer computes, as the Qwen and GLM paths do; the sorted
   GEMM then runs from staging. Short appends use the cache.
4. **Exactness.** Streamed and resident runs must produce bit-identical
   logits: the kernels do the same arithmetic from different addresses.
   `tests/flint/test_streaming` follows `test_qwen4_ssd_experts`: a sparse
   temporary file, cold and warm selections, eviction, prefill staging,
   failed reads with exact recovery, and a `pread` probe counting bytes.
5. **Speed** (step 6). The miss cost is exposed: a layer's experts are
   known only when that layer runs. Per token, 282 selections; a miss
   fraction `m` costs `m × 0.5 GB` of reads. At the SSD's measured
   bandwidth (to be measured; M4 Pro internal drives read 5-7 GB/s):

| Miss fraction | Bytes/token | Added ms at 6 GB/s | tok/s with 7 ms compute |
|---:|---:|---:|---:|
| 0 (resident) | 0 | 0 | ~140 |
| 0.05 | 25 MB | 4 | ~90 |
| 0.1 | 50 MB | 8 | ~65 |
| 0.25 | 125 MB | 21 | ~36 |
| 0.5 | 250 MB | 42 | ~20 |
| 1.0 (cold) | 500 MB | 83 | ~11 |

   The miss fraction for a given cache size is a property of Flint's routing
   on real agent transcripts, so step 4 starts with `route_trace.py`:
   routing ids per layer from rec's runtime over a few long agent sessions,
   then an offline LRU simulation per cache size (8, 12, 16, 20 GB). That
   table decides whether plain LRU is enough or step 6 needs prefetch (issue
   the next layer's likely reads from the previous token's routing, the
   fork's expert-prediction work on Qwen) and popularity preload.

Memory targets, with 1.3 GB resident weights, 2 GB runtime reserve and
context at 8K (0.3 GB): a 16 GB Mac holds about an 8 GB cache (37% of
experts), 24 GB about 14 GB (65%), 32 GB the whole 21.3 GB. On the 48 GB
development machine streaming is tested with `--ssd-streaming-cache-experts`
capped to those sizes.

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

rec's 125 tok/s "over a 128K session" is above this bound at 128K, so the
number is either weighted toward the short part of the session or its
split-K attention reads less than modelled; step 3 measures rec at fixed
contexts and settles it.

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

### 3. Resident measurements

ds4 and rec at 4K/32K/128K, prefill at 2K and 32K, launch and readback
overhead; the resident-block decision. The numbers go here.

### 4. Streaming

The visibility experiment; routing traces and the LRU table; the cache,
address table and in-kernel wait (or the split); prefill staging; budget
sizing; `test_streaming`; capped-cache full-model exactness and speed at
8/14/22 GB.

### 5. Server and agent

Flint renderer and tool syntax, reasoning in `reasoning_content`, thinking
on by default with `<think></think>` prefill to disable (confirm against
rec's `chat.py`), `/v1/models` aliases, `ds4-agent` session save, the
model's sampling defaults, one Terminal-Bench style task end to end
compared with `rec launch`.

### 6. Streaming speed

Prefetch from the previous token's routing, popularity preload at startup,
cache policy variants, each kept only with a measured gain on the
routing-trace workloads and unchanged logits.

### 7. Later

CUDA on DGX Spark (the megakernel is Metal; CUDA would start from the Qwen
expert kernels), `--batched-session`, 8-token resident blocks with
in-kernel sampling if step 3 says they pay, other quants.

## Open questions

- Family variant or new family (step 0).
- CPU-to-GPU visibility inside a running kernel (step 4's first
  experiment); the per-layer split is the fallback.
- Whether reusing decode KV as a prefix changes Flint's probabilities
  enough to matter, given that rec refuses to (step 2, gate 2).
- Where rec's 125 tok/s at 128K comes from with bf16 KV (step 3).
- Thinking control without a template switch (step 5).
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
