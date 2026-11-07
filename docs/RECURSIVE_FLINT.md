# Recursive Flint (fork, in progress)

[README](../README.md) · [fork maintenance](FORK_MAINTAINING.md) ·
[Qwen3.8 27B](QWEN38_27B.md) · [Bonsai 2 27B](BONSAI2_27B.md) ·
[Qwen3.8 Flash Next](QWEN38_FLASH_NEXT.md)

This branch (`recursive-flint`) adds
[RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), a 40B-A2.5B
mixture-of-experts coding model, to the Metal backend. It is not part of
upstream ds4; upstream issue
[#1260](https://github.com/antirez/ds4/issues/1260) (2026-10-23, no comments,
no PR) requests it. The goal is the fastest local decode on Apple Silicon at
the quality of Recursive's own 4-bit release, measured, one change at a time.

The reference machine is an M4 Pro with 48 GB (16-core GPU, 273 GB/s).
Recursive's engine reports more than 125 tok/s on this chip; llama.cpp runs
the model at roughly half that (rec's README: 1.4-2.6x llama.cpp). Nothing
below 100 tok/s on this machine is worth shipping.

## Status

| Step | State |
|---|---|
| 0. Fork scaffolding: `fork.mk`, trampolines, fork rules copied from `bonsai2`; `flint` GGUF recognised, shape, loader, tokenizer, server id | open |
| 1. Decode graph on existing kernels: greedy tokens and top-5 logprobs match llama.cpp on the golden cases | open |
| 2. Prefill graph, chunked prefill, sessions, live KV reuse, disk checkpoints | open |
| 3. Measure: time outside weight streaming, decode at 4K/32K/128K against llama.cpp and rec on the same machine | open |
| 4. 4-bit recipe at the official INT4's quality: KLD against Q8_0, official continuations from the hosted model | open |
| 5. Fused decode: attention block and MoE block as single kernels, then KV precision for the 14 full layers | open |
| 6. Server and agent: Flint template, `<tool_call>` syntax, reasoning parsing, `ds4-agent` session | open |
| 7. Later: CUDA (DGX Spark), session batching | open |

Each step records what it measured here when it lands. Steps 0-2 are
correctness, 3-5 speed and quality, 6 product. Nothing in 3-5 starts before
the golden check of step 1 passes.

## Model

Published 2026-10-21 under Apache 2.0, trained from scratch, built with
Recursive's "Eureka Machine" architecture search from a 35B/3B baseline. Text
only; no vision tower, no MTP head. The HF repository holds BF16 safetensors
(80 GB), `config.json`, tokenizer and chat template, and an SGLang kit;
Transformers modeling code is not included. Recursive's own local engine
[rec](https://github.com/recursive-oss/rec) is MLX-based Python with Metal
megakernels for M4 Pro/Max and M5 Pro/Max (48 GB minimum).

From `config.json` and the model card:

- 49 layers: 2 dense SwiGLU layers (6144 wide) then 47 MoE layers. Hidden
  2048, vocabulary 150144, untied embedding and output head, RMSNorm
  eps 1e-6, no biases.
- Attention: 32 query heads, 4 KV heads, head dim 128, scale 1/sqrt(128),
  per-head QK RMSNorm (weight of 128) before RoPE, and a learned per-head
  attention sink (`softmax_offset`, 32 floats: one extra logit in the softmax
  whose probability mass is dropped).
- 35 sliding-window layers (window 513: the token and 512 before it) with
  NeoX RoPE, theta 10000, full 128-dim rotation; 14 full-attention layers
  with **no positional encoding** (layers 0, 1, 5, 9, ..., 45, 48). The
  pattern from layer 2 is three sliding, one full. Context 262144 documented,
  512K claimed; the RoPE geometry explains why extension is free.
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
experts and head as much as by the experts, which is why Recursive quantised
the head to 4 bits "against common practice". A recipe that keeps the dense
parts at Q8_0 (ds4's habit for DeepSeek and Qwen) reads 2.2 GB per token
instead of 1.4 GB and caps at about 100 tok/s on this machine.

Tokenizer: byte-level BPE, 150000 merged tokens plus 144 specials from
150000 (`<|begin_of_text|>` 150000, `<|end_of_text|>` 150001, `<|system|>`,
`<|user|>` 150003, `<|assistant|>` 150004, `<|observation|>` 150005,
`<think>`/`</think>` 150006-7, `<tools>`, `<tool_call>`, `<arg_key>`,
`<arg_value>`, `<tool_response>`, `<|developer|>` 150018). No BOS is added.
End-of-generation ids: 150001, 150003, 150005, 150018; the model ends a turn
by emitting the next role tag, as GLM does. The pre-tokenizer is three
sequential splits (`\p{N}{1,3}`, CJK runs, then a GPT-4o-style word pattern
with uppercase/lowercase alternation); llama.cpp's `tokenizer.ggml.pre =
"flint"` holds the exact regexes.

Chat template (`chat_template.jinja`, 2.3 KB, needs Jinja `\x00` hex
escapes): `<|system|>…`, `<|developer|>…`, `<|user|>…`, `<|assistant|>`
followed by `<think>…</think>` and content, tool results as
`<|observation|><tool_response>…</tool_response>`, generation prompt
`<|assistant|>`. Tools go in a `<|system|># Tools` preamble with one JSON
schema per line inside `<tools>`. Tool calls are single-line:
`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`,
strings raw, other values JSON. User text has role tags HTML-escaped.
Recommended sampling: temperature 1.0, top-p 0.95, top-k 20, at least 32K
output tokens for hard problems. Scores (Recursive): Terminal-Bench 2.1
54.9%, SWE-bench Pro v2 89.7%, SWE-bench Verified 74.2%.

### What ds4 already has

| Need | In the tree | Gap |
|---|---|---|
| GQA attention with sinks | `metal/flash_attn.metal` (ggml port) has `has_sinks` function constants in the prefill and decode kernels; the `vec` typedef is already written for head dim 128 | only 512/512 and 256/256 host instantiations exist; add `dk128_dv128` for both kernels |
| Sliding window | `n_swa` in the shape (DeepSeek Flash uses 128) and banded masks in the DeepSeek graph | a 513-row ring KV cache per sliding layer and a window mask for prefill; decode reads at most 513 rows |
| No-RoPE layers, QK norm | QK RMSNorm and NeoX RoPE kernels in the Qwen3.8 graph (head dim 256) | per-layer RoPE on/off; head dim 128 variants |
| MoE with Q4_K/MXFP4/Q8_0 experts | Qwen3.8 Flash Next's routed-expert kernels and SSD streaming; DeepSeek's sigmoid top-k | a 256-wide sigmoid top-6 with sum renormalisation and no groups is a small kernel; check the expert kernels accept width 512 and 6 slots (Flash Next: 640, 10) |
| 4-bit dense projections | `kernel_bonsai2_mv_q4k`/`q6k` and the half-activation multi-row variants on `qwen38-27b`; `kernel_mul_mv_ext_q4_f32_disp` | bring the Q4_K/Q6_K matvec family over; Q8_0 is the upstream default |
| Tokenizer | GLM4, Qwen3.5 and DeepSeek pre-tokenizers in C with the Qwen unicode tables | `bpe_tokenize_text_flint` with the three-stage split |
| Chat template, tool calls | GLM renderer and `<arg_key>/<arg_value>` parser (`SERVER_MODEL_SYNTAX_GLM`) | Flint variant: no `[gMASK]<sop>`, no newlines inside `<tool_call>`, `<|developer|>` role, no reasoning-effort preamble |
| Sessions, live KV reuse, disk checkpoints, agent | family plumbing in `ds4.c` (per family) | see the family decision below |
| Quality tooling | `tests/qwen38_27b/kld.c`, `tests/bonsai2/check_reference.py`, `gguf-tools/quality-testing` | fixtures for Flint |

Nothing from Qwen3.8's recurrent state, hyper-connections, n-grams, indexer
or MTP applies.

## Ecosystem (2026-11-07)

| Source | What | Notes |
|---|---|---|
| [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) | BF16 safetensors, 15 shards, ~80 GB | the reference weights |
| [RecursiveSI/flint-int4-gguf](https://huggingface.co/RecursiveSI/flint-int4-gguf) | `flint-int4-q4_1.gguf`, 24.21 GiB, everything Q4_1 except F32 norms, sinks, router and `token_embd` | Recursive's Mac release for llama.cpp; ds4 has no Q4_1 kernels |
| [RecursiveSI/flint-nvfp4-gguf](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf) | 21.29 GiB NVFP4, `lm_head` 4-bit | the benchmarked quant ("within 0.7 points of 16-bit"); distilled from the BF16 teacher |
| [RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4) | MLX affine 4-bit g64, embeddings 8-bit, 22.6 GB | what rec runs; K cache 8-bit, V cache 4-bit, g64 |
| [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF) | BF16 (3 shards), Q8_0 39.40 GiB, UD-Q4_K_XL 23.33 GiB, UD-Q3/Q5/Q6_K_XL, IQ1-IQ3, imatrix (0.21 GiB, 687 chunks) | Q4_K_XL mixes Q5_K down experts, Q8_0 attention; BF16 GGUF is the converter input |
| llama.cpp [#30928](https://github.com/ggml-org/llama.cpp/pull/30928) | arch `flint`, merged 2026-10-21 (b11803), by Recursive | graph = Qwen2-MoE + sinks + SWA/RoPE patterns; author notes 10-30% decode loss from per-expert activation scaling, "to be addressed" |
| [rec](https://github.com/recursive-oss/rec) | MLX engine, `engine/src/rec_mlx/model.py` is a 360-line reference forward; `kernels/megakernel.py` fuses the decode pass | 125+ tok/s on M4 Pro over a 128K session, prefill excluded |
| [OpenRouter](https://openrouter.ai/recursive/flint-v1.0:free) | `recursive/flint-v1.0:free`, one provider | candidate source for official continuations; logprob support to verify |
| ds4 [#1260](https://github.com/antirez/ds4/issues/1260) | support request | no plan or PR upstream |

The GGUF layout (`general.architecture = flint`, 779 tensors) is llama.cpp's
standard naming: `attn_{q,k,v,output,q_norm,k_norm,sinks}`, `ffn_{gate,up,down}`
on blocks 0-1, `ffn_gate_inp` and `ffn_{gate,up,down}_{exps,shexp}` on 2-48.
Metadata: `flint.attention.sliding_window_pattern` and `flint.attention.rope_pattern`
(49 booleans each), `flint.expert_gating_func = 2` (sigmoid),
`flint.expert_weights_norm = true`, `flint.expert_shared_count = 2` (llama.cpp
encodes the shared width as 1024/512; ds4 should read `expert_shared_feed_forward_length`
instead), `tokenizer.ggml.eog_token_ids` with the four ids above.

## Speed model and prediction

Bytes per decode step at 4.5 bits per weight (Q4_K) for everything read:
2.52B × 4.5 / 8 = 1.42 GB. An M4 Pro sustains about 226 GB/s in matvec
kernels (measured on the 27B), so the weight stream alone allows about
160 tok/s. Then:

| Term | At 4K context | At 32K | At 128K |
|---|---:|---:|---:|
| weights, all 4-bit | 6.3 ms | 6.3 ms | 6.3 ms |
| KV reads, 14 full layers, F16 (2 KB per position per layer) | 0.5 ms | 4.2 ms | 16.6 ms |
| KV reads, same at K8/V4 like rec | 0.2 ms | 1.6 ms | 6.2 ms |
| 35 sliding layers, 513 rows | <0.1 ms | <0.1 ms | <0.1 ms |
| **bound, F16 KV** | **147 tok/s** | **95 tok/s** | **44 tok/s** |
| **bound, K8/V4 KV** | **154 tok/s** | **127 tok/s** | **80 tok/s** |

Dispatch overhead decides the rest. A straightforward ds4 graph runs 12-15
kernels per layer (norms, q/k/v, QK norm and RoPE, attention, output, router,
expert gate/up, expert down, shared expert, residual adds): 600-700
dispatches per token. At the 5-10 µs each costs on Apple GPUs that is 3-7 ms,
as much as the weights. This is why rec fuses the whole decode pass into one
kernel and why step 3 measures time outside weight streaming before any
kernel work, as the 27B did. The prediction for this machine:

| M4 Pro | Prediction |
|---|---:|
| step 1 graph, Q8_0 dense parts, separate kernels | 50-70 tok/s |
| all-4-bit recipe, separate kernels | 80-100 tok/s |
| fused attention and MoE blocks (step 5) | 120-140 tok/s at 4K |
| K8/V4 cache for the full layers | about 125 tok/s averaged over 128K |

Prefill: 2.5B active parameters per token is small; the expert GEMMs are
6 × 256-way scatter at width 512, where Flash Next's batched expert kernels
already run. The sliding layers attend to at most 513 keys, so only 14 layers
pay full attention cost. No prefill target yet; measure llama.cpp and rec
first (step 3).

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
   Known contact points (about 45 lines in `ds4.c`), the session, disk-KV
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
| `ds4_flint_model.inc` | `flint` metadata and tensor validation, weight binding, accepted types, engine gate |
| `ds4_flint.inc` | Metal forward graph: decode, prefill, KV cache (ring for sliding layers, linear for full), payload spans |
| `ds4_flint_gpu.h`, `ds4_metal_flint.m`, `metal/flint.metal` | kernel API, wrappers, kernels |
| `ds4_flint_tokenizer.inc` | `bpe_tokenize_text_flint` |
| `ds4_server_flint.inc` | template renderer, tool-call syntax, model ids `flint`, `recursive/flint` |
| `gguf-tools/flint_quantize.py` | BF16 GGUF → ds4 recipe, imatrix aware (from `qwen38_27b_quantize.py`) |
| `tests/flint/` | kernel tests (model-free), `golden.json`, `check_reference.py`, KLD fixtures |
| `docs/RECURSIVE_FLINT.md` | this file |

Contact points, to be listed exactly when they land: arch dispatch in
`config_validate_model`, `weights_bind_*`, graph alloc/forward redirects,
`ds4_engine_routed_quant_bits`, tokenizer dispatch, server model id and
syntax, `fork.mk` include, `ds4_gpu_full_source` kernel sources.

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

- KV cache: F16 K and V, 4 heads × 128. Sliding layers use a 513-row ring
  (37 MB for all 35 layers at any context); full layers a linear cache
  (2 KB per position per layer; 7.3 GB at 262K for all 14). Live KV reuse and
  disk checkpoints serialise both; a rewind into a sliding ring older than
  513 tokens replays the retained prefix, as Qwen's recurrent state does.
- Attention kernels: `kernel_flash_attn_ext_vec` at 128/128 with sinks for
  decode, `kernel_flash_attn_ext` at 128/128 for prefill, both already
  support sinks via function constants. The window is a mask in prefill and
  a row range in decode. GQA ratio 8 maps to the kernel's head grouping.
- Router: one kernel per layer, 256 sigmoids, top-6, renormalise; writes
  slot ids and weights for the expert kernels. fp32 as in rec.
- Experts: Flash Next's routed gate/up and down kernels with 6 slots and
  width 512; the shared expert is a dense 2048→1024→2048 SwiGLU on the same
  activations. Step 5 fuses router, gate/up, down and shared into one kernel
  per layer.
- Dense projections: Q4_K/Q6_K matvecs from `qwen38-27b` and Q8_0 from
  upstream; fp32 activations for single-row decode.
- Prefill: `ds4_metal_matmul_f16_buffer` GEMMs for the dense parts, Flash
  Next's batched expert path, chunked as upstream does (`--prefill-chunk`).

### Target GGUF recipe

From unsloth's BF16 GGUF (same names as the INT4 file) with the repository
quantizer and an imatrix (unsloth's 687-chunk file, or our own from
`llama-imatrix` on Q8_0 with the repo's calibration prompts):

| Tensors | Type | Bytes/token |
|---|---|---:|
| routed `ffn_{gate,up,down}_exps` | Q4_K (MXFP4 as the alternative; native on Blackwell) | 0.50 GB |
| `attn_{q,k,v,output}` | Q4_K; Q6_K if the gates demand it (+0.24 GB) | 0.52 GB |
| `ffn_{gate,up,down}_shexp`, dense `ffn_*` | Q4_K | 0.21 GB |
| `output` | Q4_K (Recursive ships it at 4 bits) | 0.17 GB |
| `token_embd` | Q4_K or Q8_0 (row gather) | 0 |
| `ffn_gate_inp`, norms, `attn_sinks` | F32 (router BF16 in the file is fine too) | 0.01 GB |

About 23 GB on disk, 1.4 GB per token. Expected sensitive classes, from the
27B: attention projections, then the head. Measure per class before
spending bits (step 4).

## Verification

- **Kernel tests** against double-precision CPU references, model-free, in
  `tests/flint/` (`make test-flint-kernels`): QK-norm + RoPE, attention with
  sinks and window at several lengths and ring wrap-around, router top-6
  with ties, expert gate/up/down at width 512, fused kernels against the
  unfused ones bit for bit.
- **Golden reference** (`make test-flint-reference`): greedy tokens and
  top-5 logprobs from llama.cpp (b11803 or later, `--jinja`) on the *same*
  ds4 recipe GGUF, on Bonsai's six case shapes (five short chats including
  a tool call, one 4K-token passage) plus one 20K-token case that crosses
  many window boundaries. Tokens must match exactly; the logprob tolerance
  is set by llama.cpp's own run-to-run spread, as for the 27B (0.4 there,
  because llama.cpp rounds few-row activations to half).
- **Sessions**: `ds4_test --server`, checkpoint save/restore and rewind
  across a sliding-window wrap, live KV reuse on a multi-turn agent
  transcript, server story test at 48K.
- **Speed**: `ds4-bench` sweeps 2K-128K, back to back with llama.cpp and
  rec on the same machine, AC power, High Power Mode, no other model
  process ([benchmark rules](../CONTRIBUTING.md#speed-regression-tests)).

## Quality

The bar is Recursive's INT4 release (`flint-int4-q4_1.gguf`): the file
Recursive ships to Mac users. Its KLD against Q8_0 is measured with
llama.cpp (`llama-perplexity --kl-divergence`) on the same texts as ds4's
KLD tool, so the bar needs no Q4_1 kernels in ds4. NVFP4's distance is
measured the same way for reference.

Gates, in the order they are run while iterating:

1. **KLD** against unsloth's Q8_0 (39.4 GiB; runs mmap-backed on 48 GB,
   slowly) with `tests/flint/kld` (the 27B tool, teacher-forced through the
   single-token decode path): full-vocabulary KL, top-1 agreement,
   perplexity over 3072 positions of prose, an English story and `rax.c`.
   Pass: KLD ≤ the INT4 file's KLD, top-1 ≥ its top-1.
2. **Official continuations** (`gguf-tools/quality-testing`): 100 short
   prompts and the 12 long archive/code prompts, collected from the hosted
   model (OpenRouter `recursive/flint-v1.0:free` if it returns logprobs,
   else Recursive's API or the SGLang kit on a rented H200 serving BF16),
   with the model's template, no thinking, temperature 0, 64 tokens, top-5
   logprobs. Pass per the release criteria: no first-token drop, no clear
   NLL regression, no material API top-1 or pair-order regression against
   the INT4 file scored by llama.cpp on the same fixtures.
3. **Exactness**: `make test-flint-reference` after every graph or kernel
   change; the fused kernels must reproduce the unfused logits bit for bit
   or state the arithmetic difference and re-pass gate 1.

A speed number is only valid for a file and a graph that pass these gates.

## Steps

### 0. Scaffolding

Copy `fork.mk`, `ds4_metal_internal.h`, `docs/FORK_MAINTAINING.md` and the
CI workflow from `bonsai2`, renamed for Flint, so this branch and the other
fork branches can later merge into one fork main. Recognise `flint` GGUFs:
shape, metadata validation (reject anything but the recipe's types),
tensor binding, the pre-tokenizer, `--dump-tokens` parity with llama.cpp on
the tokenizer test strings, server model id `flint`. Decide the family
question above and record it. Deliverable: `./ds4 -m flint.gguf
--dump-tokens` matches llama.cpp on 20 strings including CJK, digits,
contractions and the special tokens.

### 1. Decode graph

Single-token forward on Metal with existing kernels plus the three new ones
(128-wide attention instantiations, router, ring-cache writes). First on a
Q8_0-dense/Q4_K-expert file to separate graph bugs from kernel bugs, then
on the all-4-bit recipe. Golden check passes on the short cases.

### 2. Prefill and sessions

Chunked prefill with window masks, full-row logits for the golden long
cases, sessions, live KV reuse, disk checkpoints with both cache kinds,
rewind replay across the ring. Golden check passes on all cases;
`ds4_test --server` passes; a 48K story test retrieves its facts.

### 3. Measure

`DS4_FLINT_TIMING` stage timing as `DS4_QWEN4_TIMING` does. Report: ms per
token at 4K/32K/128K split into weights, attention, overhead; same sweep
for llama.cpp (b11803+, Q4_1 INT4 and our recipe) and rec (INT4 MLX). This
table decides how much of step 5 is needed.

### 4. Recipe

Per-class KLD table (one class at Q4_K, rest Q8_0; then calibrated), the
INT4 bar, candidates C1.. as on the 27B, official fixtures for the chosen
file. Deliverable: one GGUF, its `flint_quantize.py` command, both gates
passed, size and bytes per token recorded.

### 5. Fused decode

In order of expected gain, each kept only with a measured win and
bit-identical or re-gated output: (a) attention block kernel (norm, q/k/v,
QK norm, RoPE, cache write, attention with sinks, output projection) per
layer; (b) MoE block kernel (norm, router, gate/up, down, shared, residual);
(c) K8/V4 cache for the full layers if the 32K-128K numbers from step 3 say
so (gate 1 again: it changes arithmetic); (d) one persistent kernel for the
whole layer loop only if (a)-(b) leave more than 1 ms of dispatch overhead.

### 6. Server and agent

Flint renderer and tool-call syntax, reasoning in `reasoning_content`,
thinking on by default with `<think></think>` prefill to disable (verify
against the hosted model that this is what Recursive does; the template
has no switch), `/v1/models` aliases, `ds4-agent` session save with a
sliding ring, `--temp 1.0 --top-p 0.95 --top-k 20` as the model defaults.
Terminal-Bench style smoke: one agent task end to end.

### 7. Later

CUDA on DGX Spark (Flash Next's expert kernels and `flash_attn` ports),
`--batched-session`, SSD streaming is not needed (the file fits), MTP does
not exist for this model.

## Open questions

- Family variant or new family (step 0 decides; see above).
- Whether Flash Next's expert kernels take width 512 and 6 slots without
  a variant, and whether their Q4_K path keeps fp32 accumulation at this
  width (the 27B found half activations fine for multi-row verify but kept
  fp32 for single-row decode).
- Quality bar source: whether OpenRouter's free endpoint returns top
  logprobs; otherwise which hosted or rented BF16 serves the fixtures.
- Thinking control: the template has no `enable_thinking`; confirm the
  `<think></think>` prefill convention with the SGLang kit's reasoning
  parser before building the server switch on it.
- Context above 262144 (Recursive says 512K works, recommends 256K): ds4
  needs no RoPE change, only cache memory; leave the cap at 262144.
- Whether llama.cpp's per-expert activation scaling (#30861) is needed for
  Q8 activations at width 512 with Flint's activation ranges; ds4's expert
  kernels use fp32 activations for decode, so this applies only to the
  batched prefill path if it quantises activations.

## References

- Announcement: https://www.recursive.com/updates/introducing-recursive-flint
- Model card and config: https://huggingface.co/RecursiveSI/flint
- Engine: https://github.com/recursive-oss/rec (`engine/src/rec_mlx/model.py`,
  `kernels/megakernel.py`, `quantized.py`)
- llama.cpp: PR #30928 (model), #30929 (Jinja hex escapes), #30861
  (per-expert activation scaling); `src/models/flint.cpp`,
  `conversion/flint.py`, `src/llama-vocab.cpp` (`LLAMA_VOCAB_PRE_TYPE_FLINT`)
- ds4: issue #1260; `docs/QWEN38_27B.md` (method), `docs/FORK_MAINTAINING.md`
  (rules), `tests/qwen38_27b/kld.c`, `tests/bonsai2/check_reference.py`,
  `gguf-tools/qwen38_27b_quantize.py` (tooling to adapt)
