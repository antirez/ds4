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

Scope, for now:

- **One weight format**: Recursive's own 4-bit release,
  [RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4)
  (`mlx-affine-int4`, 22.6 GB), the file `rec pull` downloads. ds4 loads
  these exact weights, converted to GGUF without requantisation. No recipe
  search, no Q4_K, no other quants until this one is done.
- **Ground truth is rec**, Recursive's engine, installed locally
  (`~/.local/bin/rec`, 0.1.1). Correctness means ds4 reproduces rec's
  tokens and log-probabilities on the same weights; speed means ds4 is
  measured against rec on the same machine. llama.cpp is not a reference
  on this branch.

The goal is the fastest local decode on Apple Silicon at rec's quality. The
reference machine is an M4 Pro with 48 GB (16-core GPU, 273 GB/s), where rec
reports more than 125 tok/s averaged over a 128K-token session.

## Status

| Step | State |
|---|---|
| 0. Scaffolding: fork files from `bonsai2`; `flint` GGUF with the affine-int4 tensor types; converter from rec's checkpoint; shape, loader, tokenizer, server id | open |
| 1. Decode graph: greedy tokens and logprobs match rec on the golden cases | open |
| 2. Prefill graph, chunked prefill, sessions, live KV reuse, disk checkpoints | open |
| 3. Measure: time outside weight streaming; decode at 4K/32K/128K against rec | open |
| 4. Quality parity: KLD against rec's full logits, teacher-forced; official continuations from the hosted model for both engines | open |
| 5. Fused decode: attention block and MoE block as single kernels; KV precision for the 14 full layers if long context needs it | open |
| 6. Server and agent: Flint template, `<tool_call>` syntax, reasoning parsing, `ds4-agent` session | open |
| 7. Later: CUDA (DGX Spark), session batching, other quants | open |

Each step records what it measured here when it lands. Steps 0-2 are
correctness, 3-5 speed, 4 quality, 6 product. Nothing in 3-5 starts before
the golden check of step 1 passes.

## Model

Published 2026-10-21 under Apache 2.0, trained from scratch, built with
Recursive's "Eureka Machine" architecture search from a 35B/3B baseline. Text
only; no vision tower, no MTP head. The BF16 checkpoint is 80 GB of
safetensors with an SGLang kit and no Transformers code. Recursive's local
engine [rec](https://github.com/recursive-oss/rec) is MLX-based Python with
Metal megakernels for M4 Pro/Max and M5 Pro/Max (48 GB minimum); the Go CLI
bundles the Python runtime and extracts it under
`~/.local/share/rec/runtimes/` on first `rec serve`, which is also where a
script can import `rec_mlx` to dump logits.

From `config.json` and the model card:

- 49 layers: 2 dense SwiGLU layers (6144 wide) then 47 MoE layers. Hidden
  2048, vocabulary 150144, untied embedding and output head, RMSNorm
  eps 1e-6, no biases.
- Attention: 32 query heads, 4 KV heads, head dim 128, scale 1/sqrt(128),
  per-head QK RMSNorm (weight of 128) before RoPE, and a learned per-head
  attention sink (`softmax_offset`, 32 values: one extra logit in the
  softmax whose probability mass is dropped).
- 35 sliding-window layers (window 513: the token and 512 before it) with
  NeoX RoPE, theta 10000, full 128-dim rotation; 14 full-attention layers
  with **no positional encoding** (layers 0, 1, 5, 9, ..., 45, 48). From
  layer 2 the pattern is three sliding, one full. Context 262144 documented,
  512K claimed; the RoPE geometry is why extension costs nothing.
- MoE: 256 routed experts of width 512, 6 per token, plus one always-on
  shared expert of width 1024. Router: `sigmoid(x W^T)` in fp32, top-6 by
  score (plus a zero `expert_bias`), weights renormalised to sum 1
  (`norm_topk_prob`), scale 1.0, no expert groups, no shared-expert gate.
- Active parameters per token (computed from the shapes):

| Part | Params/token | Share |
|---|---:|---:|
| attention q/k/v/o, 49 layers | 925M | 37% |
| routed experts, 47 × 6 × 3 × 2048 × 512 | 887M | 35% |
| output head 2048 × 150144 | 307M | 12% |
| shared experts, 47 × 3 × 2048 × 1024 | 296M | 12% |
| dense FFN (2 layers) + routers | 100M | 4% |
| **total** | **2.52B** | |

Routed experts are 95% of the 39.8B stored weights but only 35% of what a
token reads. Decode speed is set by the attention projections, shared
experts and head as much as by the experts, which is why the INT4 release
quantises everything, the head included.

Tokenizer: byte-level BPE, 150000 merged tokens plus 144 specials from
150000 (`<|begin_of_text|>` 150000, `<|end_of_text|>` 150001, `<|system|>`,
`<|user|>` 150003, `<|assistant|>` 150004, `<|observation|>` 150005,
`<think>`/`</think>` 150006-7, `<tools>`, `<tool_call>`, `<arg_key>`,
`<arg_value>`, `<tool_response>`, `<|developer|>` 150018). No BOS is added.
End-of-generation ids: 150001, 150003, 150005, 150018; the model ends a turn
by emitting the next role tag, as GLM does. The pre-tokenizer is three
sequential splits (`\p{N}{1,3}`, CJK runs, then a GPT-4o-style word pattern
with uppercase/lowercase alternation); llama.cpp's `tokenizer.ggml.pre =
"flint"` holds the exact regexes, and rec's tokenizer (HF `tokenizers`) is
the arbiter.

Chat template (`chat_template.jinja`, 2.3 KB): `<|system|>…`,
`<|developer|>…`, `<|user|>…`, `<|assistant|>` followed by `<think>…</think>`
and content, tool results as `<|observation|><tool_response>…</tool_response>`,
generation prompt `<|assistant|>`. Tools go in a `<|system|># Tools` preamble
with one JSON schema per line inside `<tools>`. Tool calls are single-line:
`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`,
strings raw, other values JSON. User text has role tags HTML-escaped.
Recommended sampling: temperature 1.0, top-p 0.95, top-k 20 (rec's
defaults), at least 32K output tokens for hard problems.

### The INT4 checkpoint

`rec pull` stores it under `~/.local/share/rec/models/<digest>/` with the
HF file names (five `model-0000N.safetensors`, 4.4 + 3 × 4.4 + 3.5 GiB,
`config.json`, tokenizer, template). `config.json` carries
`quantization = {bits: 4, group_size: 64}` and `mlx_export = {bits: 4,
embedding_bits: 8, routed_expert_bits: 4, scale_rule: "mse"}`; the manifest
says `weight_format: mlx-affine-int4, kv_format: bf16`. The format is MLX
affine quantisation:

- `w = scale * q + bias` per group of 64 along the input dimension;
  `scales` and `biases` are BF16 `[out, in/64]`; `weight` is `U8 [out, in/2]`,
  the view of MLX's packed `uint32 [out, in/8]`, element `i` of a word in
  bits `4i..4i+3` (low nibble first). 4.5 bits per weight.
- Every `nn.Linear` is 4-bit g64, including `lm_head` (`[150144, 1024]` U8)
  and the routed experts (`switch_mlp.{gate,up,down}_proj.weight`,
  `[256, 512, 1024]` and `[256, 2048, 256]` U8, scales `[256, 512, 32]` and
  `[256, 2048, 8]`).
- `embed_tokens` is 8-bit g64 (`U8 [150144, 2048]`, scales `[150144, 32]`).
- Norm weights, `softmax_offset` and the router (`mlp.gate.weight`
  `[256, 2048]`) are BF16; rec casts the router and sinks to fp32 and bf16
  respectively at load time.

ds4 gets two fork tensor types for this, beside Bonsai's `PQ2_0 = 142`:
`AFF4_64 = 143` (row = 64 nibbles + bf16 scale + bf16 bias, 36 bytes per
64) and `AFF8_64 = 144` (64 bytes + 4, for the embedding). The converter
(`gguf-tools/flint_int4_to_gguf.py`) interleaves scale and bias with each
group's packed weights so a kernel reads one contiguous 36-byte block, keeps
the values bit for bit, writes llama.cpp's tensor names and `flint.*`
metadata, and records the checkpoint's SHA-256s. A `--check` mode
dequantises every tensor with numpy and compares against MLX's
`mx.dequantize` on the source file. Expert tensors are written as one
3-D tensor per projection (`[256, rows, cols]`), as llama.cpp does.

### What ds4 already has

| Need | In the tree | Gap |
|---|---|---|
| GQA attention with sinks | `metal/flash_attn.metal` (ggml port) has `has_sinks` function constants in the prefill and decode kernels; the `vec` typedef is already written for head dim 128 | only 512/512 and 256/256 host instantiations exist; add `dk128_dv128` for both kernels |
| Sliding window | `n_swa` in the shape (DeepSeek Flash uses 128) and banded masks in the DeepSeek graph | a 513-row ring KV cache per sliding layer and a window mask for prefill; decode reads at most 513 rows |
| No-RoPE layers, QK norm | QK RMSNorm and NeoX RoPE kernels in the Qwen3.8 graph (head dim 256) | per-layer RoPE on/off; head dim 128 variants |
| MoE routing and expert scheduling | Flash Next's routed-expert kernels (selected-slot gather, gate/up then down, ordered reduction) and DeepSeek's sigmoid top-k | kernels for `AFF4_64` experts at width 512 with 6 slots; the Flash Next kernels are the structural template, not reusable as is (Q4_K/MXFP4/IQ2 decoders) |
| 4-bit dense matvec | `kernel_bonsai2_mv_q4k`/`q6k` and their half-activation multi-row variants on `qwen38-27b`; `kernel_mul_mv_ext_q4_f32_disp` | an `AFF4_64` matvec (simpler decode than Q4_K: one scale, one bias, no 6-bit sub-scales) and an 8-bit embedding gather |
| Prefill GEMM | `ds4_metal_matmul_f16_buffer`, Bonsai's `kernel_bonsai2_gemm` family with decoded-fragment operands | an `AFF4_64` operand decoder in the Bonsai GEMM |
| Tokenizer | GLM4, Qwen3.5 and DeepSeek pre-tokenizers in C with the Qwen unicode tables | `bpe_tokenize_text_flint` with the three-stage split |
| Chat template, tool calls | GLM renderer and `<arg_key>/<arg_value>` parser (`SERVER_MODEL_SYNTAX_GLM`) | Flint variant: no `[gMASK]<sop>`, no newlines inside `<tool_call>`, `<|developer|>` role, no reasoning-effort preamble |
| Sessions, live KV reuse, disk checkpoints, agent | family plumbing in `ds4.c` (per family) | see the family decision below |
| Reference tooling | `tests/bonsai2/check_reference.py` (golden file), `tests/qwen38_27b/kld.c` (teacher-forced KLD), `gguf-tools/quality-testing` | rec as the source instead of llama.cpp |

Nothing from Qwen3.8's recurrent state, hyper-connections, n-grams, indexer
or MTP applies.

## How rec runs it

What ds4 must reproduce, from `engine/src/rec_mlx/` (0.1.1):

- `model.py`: the reference forward (360 lines). bf16 activations
  throughout; router scores and the sink softmax in fp32; `RMSNorm` from
  `mlx.nn`; `nn.RoPE(128, traditional=False, base=10000)` on sliding
  layers; `scaled_dot_product_attention(..., sinks=softmax_offset)`;
  `SwitchGLU` for the experts; weights `scores/(sum+1e-20)`.
- `quantized.py`: loads the checkpoint with `nn.quantize(group_size=64,
  mode="affine")`, keeps the router in fp32; `FixedKVCache` (bf16) for the
  14 full layers and `FixedSlidingKVCache(window=513)` rings for the rest.
- `kernels/megakernel.py` (1300 lines): one Metal kernel per decode step
  over all 49 layers with bf16 workspaces, fp32 partials and statistics,
  `router_bf16`, expert "teams" and KV "teams" of threadgroups.
- Prefill in chunks (`prefill_chunk_size`), last-row logits, prompt-prefix
  cache with rollback. Sampling defaults temperature 1.0, top-p 0.95,
  top-k 20; `max_tokens` 1024 in chat.
- The HTTP API (`rec serve`, port 11500, key from `rec auth token`) is
  OpenAI chat completions plus Anthropic messages. `logprobs: true` with
  `stream: false` returns the chosen token's logprob per generated token;
  `top_logprobs` is not supported. Full logit vectors come from the runtime
  directly: `Model.__call__(tokens, cache)` returns logits for all rows.

Arithmetic differences to expect against ds4's fp32 activations: bf16
rounding in every projection input and in the KV cache (rec), fp32 in ds4.
The 27B saw llama.cpp's half-rounded activations move top-5 logprobs by up
to 0.32 while tokens stayed identical; rec's bf16 should be in the same
range. The golden tolerance is therefore set from rec's own spread (step 1)
rather than assumed.

## Speed model and prediction

Bytes per decode step at 4.5 bits per weight for everything read:
2.52B × 4.5 / 8 = 1.42 GB. An M4 Pro sustains about 226 GB/s in matvec
kernels (measured on the 27B), so the weight stream alone allows about
160 tok/s. Then:

| Term | At 4K context | At 32K | At 128K |
|---|---:|---:|---:|
| weights | 6.3 ms | 6.3 ms | 6.3 ms |
| KV reads, 14 full layers, bf16/F16 (2 KB per position per layer) | 0.5 ms | 4.2 ms | 16.6 ms |
| KV reads, same at K8/V4 | 0.2 ms | 1.6 ms | 6.2 ms |
| 35 sliding layers, 513 rows | <0.1 ms | <0.1 ms | <0.1 ms |
| **bound, 16-bit KV** | **147 tok/s** | **95 tok/s** | **44 tok/s** |
| **bound, K8/V4 KV** | **154 tok/s** | **127 tok/s** | **80 tok/s** |

rec's 125 tok/s "averaged over a 128K session" with a bf16 KV cache sits
above the 16-bit-KV bound at 128K, so either its number is weighted toward
the short part of the session or it reads less KV than this model assumes;
step 3 measures rec at fixed contexts and settles it.

Dispatch overhead decides the rest. A straightforward ds4 graph runs 12-15
kernels per layer (norms, q/k/v, QK norm and RoPE, attention, output,
router, expert gate/up, expert down, shared expert, residual adds): 600-700
dispatches per token. At the 5-10 µs each costs on Apple GPUs that is
3-7 ms, as much as the weights. This is why rec fuses the whole decode pass
into one kernel and why step 3 measures time outside weight streaming
before any kernel work, as the 27B did. The prediction for this machine:

| M4 Pro | Prediction |
|---|---:|
| step 1 graph, separate kernels | 80-100 tok/s at 4K |
| fused attention and MoE blocks (step 5) | 120-140 tok/s at 4K |
| K8/V4 cache for the full layers | about 125 tok/s averaged over 128K |

Prefill: 2.5B active parameters per token is small; the expert GEMMs are
6 × 256-way scatter at width 512, where Flash Next's batched expert
scheduling already runs. The sliding layers attend to at most 513 keys, so
only 14 layers pay full attention cost. No prefill target until rec is
measured (step 3).

## Design

Fork rules apply ([FORK_MAINTAINING.md](FORK_MAINTAINING.md), to be copied
from `bonsai2` in step 0): Flint code lives in fork-owned files, upstream
files get one- to three-line hooks, upstream is merged, never rebased.

### Family decision (step 0)

Bonsai 2 and the 27B run as variants of the Qwen3.8 family because their
layer geometry matches it; they reuse its sessions, payloads and kernels.
Flint shares no layer type with any ds4 family: it is a plain GQA + MoE
transformer. Two options:

1. **Qwen3.8-family variant** (`DS4_VARIANT_FLINT = 201`, Bonsai's pattern).
   Known contact points (about 45 lines in `ds4.c`); the session, disk-KV
   and server plumbing come for free. Risk: the shared session and payload
   code walks the Qwen graph struct (GDN state, conv history, indexer spans)
   and may not accept zero-width recurrent state; Bonsai kept GDN layers,
   Flint has none.
2. **New family** (`DS4_MODEL_FAMILY_FLINT`). Clean, but every
   `ds4_model_is_qwen4()` fork in `ds4.c` (about 60 sites: prefill
   chunking, session payloads, server aliases, agent save) needs a Flint
   branch or a shared predicate.

Recommendation: try option 1 first, as a time-boxed probe in step 0: set
`n_lin_*`, `n_hc = 1`, `n_ple_layer = DS4_MAX_LAYER` inert as Bonsai does, give
the fork graph its own KV cache, and see whether a session round-trips a
checkpoint. Fall back to option 2 only if the shared code needs more than a
handful of hooks to tolerate an attention-only model. Record the outcome here.

### Fork-owned files

| File | Contents |
|---|---|
| `ds4_flint_shape.inc` | `DS4_SHAPE_FLINT`, `ds4_model_is_flint()`, the per-layer sliding/RoPE/dense tables |
| `ds4_flint_model.inc` | `flint` metadata and tensor validation (types 143/144 and F32/BF16 only), weight binding, engine gate |
| `ds4_flint.inc` | Metal forward graph: decode, prefill, KV cache (ring for sliding layers, linear for full), payload spans |
| `ds4_flint_gpu.h`, `ds4_metal_flint.m`, `metal/flint.metal` | kernel API, wrappers, kernels: `AFF4_64` matvec and GEMM operand, `AFF8_64` gather, router, attention 128 with sinks and window, expert gate/up/down, fused blocks |
| `ds4_flint_tokenizer.inc` | `bpe_tokenize_text_flint` |
| `ds4_server_flint.inc` | template renderer, tool-call syntax, model ids `flint`, `recursive/flint` |
| `gguf-tools/flint_int4_to_gguf.py` | rec checkpoint → GGUF, bit-exact, with `--check` |
| `tests/flint/` | kernel tests (model-free), `golden.json` from rec, `check_reference.py`, `rec_logits.py`, `kld` |
| `docs/RECURSIVE_FLINT.md` | this file |

Contact points, to be listed exactly when they land: `gguf_types[143..144]`,
arch dispatch in `config_validate_model`, `weights_bind_*`, graph
alloc/forward redirects, `ds4_engine_routed_quant_bits` (4), tokenizer
dispatch, server model id and syntax, `fork.mk` include,
`ds4_gpu_full_source` kernel sources.

### Graph

Per token, per layer (decode):

```
x    = rmsnorm(h) * attn_norm
q,k,v = x Wq, x Wk, x Wv                         (4096, 512, 512 outputs)
q,k  = rmsnorm_head(q) * q_norm, rmsnorm_head(k) * k_norm
q,k  = rope_neox(q,k, pos)                        sliding layers only
cache[l][pos % cap] = k, v                        cap = 513 sliding, ctx full
a    = softmax([q·K / sqrt(128), sink_h]) V       mask: window or causal
h   += a Wo
x    = rmsnorm(h) * ffn_norm
h   += dense_swiglu(x)                            layers 0, 1
h   += sum_i w_i expert_i(x) + shared(x)          layers 2..48
```

then `output_norm` and the head. Decisions:

- KV cache: F16 K and V, 4 heads × 128, as rec's bf16 cache in precision.
  Sliding layers use a 513-row ring (37 MB for all 35 layers at any
  context); full layers a linear cache (2 KB per position per layer; 7.3 GB
  at 262K for all 14). Live KV reuse and disk checkpoints serialise both; a
  rewind into a sliding ring older than 513 tokens replays the retained
  prefix, as Qwen's recurrent state does.
- Attention kernels: `kernel_flash_attn_ext_vec` at 128/128 with sinks for
  decode, `kernel_flash_attn_ext` at 128/128 for prefill; both already take
  sinks via function constants. The window is a mask in prefill and a row
  range in decode. GQA ratio 8 maps to the kernel's head grouping.
- Router: one kernel per layer, 256 sigmoids in fp32, top-6, renormalise;
  writes slot ids and weights for the expert kernels.
- Experts: `AFF4_64` gate/up and down kernels with 6 slots and width 512,
  built on Flash Next's selected-slot scheduling; the shared expert is a
  dense 2048→1024→2048 SwiGLU on the same activations. Step 5 fuses router,
  gate/up, down and shared into one kernel per layer.
- Dense projections: `AFF4_64` matvec, fp32 activations for single-row
  decode (the 27B found half activations exact enough for multi-row verify,
  which this model does not need).
- Prefill: Bonsai's GEMM with an `AFF4_64` operand decoder for the dense
  parts, Flash Next's batched expert path with the new decoder, chunked as
  upstream does (`--prefill-chunk`).

## Verification

- **Converter**: `flint_int4_to_gguf.py --check` dequantises every tensor
  and matches `mx.dequantize` on the source bit for bit; the GGUF records
  the checkpoint SHA-256s from rec's manifest.
- **Kernel tests** against double-precision CPU references, model-free, in
  `tests/flint/` (`make test-flint-kernels`): `AFF4_64` dequantisation and
  matvec, QK-norm + RoPE, attention with sinks and window at several
  lengths and ring wrap-around, router top-6 with ties, expert gate/up/down
  at width 512, GEMM operand decode, fused kernels against the unfused ones
  bit for bit.
- **Golden reference** (`make test-flint-reference`): `tests/flint/golden.json`
  holds rendered prompts, rec's prompt token ids, rec's greedy tokens and
  per-token logprobs (from `rec serve` with `logprobs: true`,
  `temperature 0`, `stream: false`; verify that rec samples greedily at 0)
  on Bonsai's six case shapes (five short chats including a tool call, one
  4K-token passage) plus one 20K-token case that crosses many window
  boundaries. ds4 must tokenise identically and reproduce the tokens; the
  logprob tolerance is set by rec's own spread, measured by running each
  case twice with different prefill chunk sizes.
- **Sessions**: `ds4_test --server`, checkpoint save/restore and rewind
  across a sliding-window wrap, live KV reuse on a multi-turn agent
  transcript, server story test at 48K.
- **Speed**: `ds4-bench` sweeps 2K-128K, back to back with `rec serve` on
  the same machine (`--metrics` JSONL gives rec's per-request timings),
  AC power, High Power Mode, no other model process
  ([benchmark rules](../CONTRIBUTING.md#speed-regression-tests)). rec and
  ds4 never run at the same time: both want the whole GPU and 22.6 GB.

## Quality

Same weights, so quality is agreement with rec, not a quantisation bar.

1. **KLD against rec** (`tests/flint/kld`, the 27B tool with rec as the
   base): `tests/flint/rec_logits.py` runs inside rec's extracted runtime,
   teacher-forces the same 3072 positions (Italian prose, an English story,
   `rax.c`) through `Model.__call__` with rec's caches, and writes full
   log-probabilities per position in the `kld` base format. ds4 runs the
   positions through its single-token decode path. Report KL(rec || ds4)
   mean, p99, max; top-1 agreement; perplexity of both. Pass: mean KLD and
   top-1 within what rec shows against itself when prefill chunking changes
   (measured first), and no position with a top-1 flip whose rec logprob
   gap exceeds the tolerance.
2. **Official continuations** (`gguf-tools/quality-testing`, the repo's
   release gate): 100 short prompts and the 12 long archive/code prompts,
   collected from the hosted model (OpenRouter
   `recursive/flint-v1.0:free` if it returns top logprobs, else Recursive's
   API), scored by ds4; rec cannot be scored by `score_official`, so its
   greedy first tokens on the same prompts (via `rec serve`) are compared
   instead. This gate catches template and tokenizer drift that gate 1,
   which starts from token ids, cannot.
3. **Exactness**: `make test-flint-reference` after every graph or kernel
   change; a fused kernel must reproduce the unfused logits bit for bit or
   state the arithmetic difference and re-pass gate 1.

A speed number is only valid for a graph that passes these gates.

## Steps

### 0. Scaffolding and converter

Copy `fork.mk`, `ds4_metal_internal.h`, `docs/FORK_MAINTAINING.md` and the
CI workflow from `bonsai2`, renamed for Flint, so this branch and the other
fork branches can later merge into one fork main. Write
`flint_int4_to_gguf.py` and run it on rec's pulled checkpoint
(`~/.local/share/rec/models/*/`), `--check` passing. Recognise the GGUF:
shape, metadata validation (reject anything but types 143/144, F32, BF16),
tensor binding, the pre-tokenizer, server model id `flint`. Decide the
family question above and record it. Deliverable: `./ds4 -m
gguf/flint/flint-int4.gguf --dump-tokens` matches rec's tokenizer on 20
strings including CJK, digits, contractions, HTML-escaped role tags and the
special tokens; `golden.json` collected from `rec serve`.

### 1. Decode graph

Single-token forward on Metal: the `AFF4_64` matvec, 128-wide attention
instantiations with sinks, router, ring-cache writes, `AFF4_64` expert
kernels. Golden check passes on the short cases with the tolerance from
rec's own spread.

### 2. Prefill and sessions

GEMM operand decoder, chunked prefill with window masks, full-row logits
for the long golden cases, sessions, live KV reuse, disk checkpoints with
both cache kinds, rewind replay across the ring. Golden check passes on all
cases; `ds4_test --server` passes; a 48K story test retrieves its facts.

### 3. Measure

`DS4_FLINT_TIMING` stage timing as `DS4_QWEN4_TIMING` does. Report ms per
token at 4K/32K/128K split into weights, attention, overhead, and rec's
decode and prefill at the same contexts from its `--metrics` file, both on
AC power with nothing else on the GPU. This table decides how much of
step 5 is needed and whether KV precision matters at the contexts the
agent actually uses.

### 4. Quality parity

Gate 1 and gate 2 above on the step 2 graph; the numbers go in this file.
Any later kernel change re-runs gate 1.

### 5. Fused decode

In order of expected gain, each kept only with a measured win and
bit-identical or re-gated output: (a) attention block kernel (norm, q/k/v,
QK norm, RoPE, cache write, attention with sinks, output projection) per
layer; (b) MoE block kernel (norm, router, gate/up, down, shared,
residual); (c) K8/V4 cache for the full layers if the 32K-128K numbers from
step 3 say so (gate 1 again: it changes arithmetic and moves away from
rec's bf16 cache); (d) one persistent kernel for the whole layer loop only
if (a)-(b) leave more than 1 ms of dispatch overhead.

### 6. Server and agent

Flint renderer and tool-call syntax, reasoning in `reasoning_content`,
thinking on by default with `<think></think>` prefill to disable (check
against rec's `chat.py` and the SGLang reasoning parser; the template has
no switch), `/v1/models` aliases, `ds4-agent` session save with a sliding
ring, `--temp 1.0 --top-p 0.95 --top-k 20` as the model defaults.
Terminal-Bench style smoke: one agent task end to end, compared with the
same task on `rec launch`.

### 7. Later

CUDA on DGX Spark (Flash Next's expert kernels and `flash_attn` ports with
the `AFF4_64` decoder), `--batched-session`, other quants (NVFP4 for
Blackwell, Q4_K/unsloth files) once the INT4 path is complete. SSD
streaming is not needed (the file fits); MTP does not exist for this model.

## Open questions

- Family variant or new family (step 0 decides; see above).
- Whether rec's `temperature: 0` is greedy or clamps (its `Sampling` class
  validates ranges); if not, collect golden tokens with `top_k: 1`.
- Where rec's 125 tok/s over 128K comes from with a bf16 cache (step 3).
- Thinking control: the template has no `enable_thinking`; confirm the
  `<think></think>` prefill convention before building the server switch
  on it.
- Context above 262144 (Recursive says 512K works, recommends 256K): ds4
  needs no RoPE change, only cache memory; leave the cap at 262144.
- Whether the batched expert prefill path can keep fp32 activations at
  width 512 without losing bandwidth, or needs the half-activation variant
  (then gate 1 decides).

## References

- Announcement: https://www.recursive.com/updates/introducing-recursive-flint
- Model card and config: https://huggingface.co/RecursiveSI/flint;
  INT4 checkpoint: https://huggingface.co/RecursiveSI/flint-int4
- Engine: https://github.com/recursive-oss/rec (`engine/src/rec_mlx/model.py`,
  `quantized.py`, `cache.py`, `engine.py`, `server.py`, `kernels/megakernel.py`)
- Other quants, out of scope for now: `RecursiveSI/flint-int4-gguf` (Q4_1,
  24.2 GiB), `RecursiveSI/flint-nvfp4-gguf` (21.3 GiB), `unsloth/flint-GGUF`
  (BF16, Q8_0 39.4 GiB, UD-Q4_K_XL 23.3 GiB, imatrix); llama.cpp PR #30928
  (arch `flint`, b11803), #30929 (Jinja hex escapes), #30861
- Hosted: https://openrouter.ai/recursive/flint-v1.0:free
- ds4: issue #1260; `docs/QWEN38_27B.md` (method), `docs/FORK_MAINTAINING.md`
  (rules), `tests/qwen38_27b/kld.c`, `tests/bonsai2/check_reference.py`
  (tooling to adapt)
