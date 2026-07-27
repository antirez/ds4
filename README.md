# Pulsar

**Pulsar** is a small native inference engine built for **DeepSeek V4
Flash** on a single **NVIDIA GB10 (DGX Spark, ~128 GB unified memory)**. It is
intentionally narrow: not a generic GGUF runner, not a wrapper around another
runtime: it is completely self-contained. Other than running the model in a
correct and fast way, the project goal is to provide DeepSeek specific loading,
prompt rendering, tool calling, and KV state handling (RAM and on-disk) behind an
OpenAI- and Anthropic-compatible server API, ready to work with local coding
agents. The shipped release is a single binary — **`pulsar-server`** — that serves
that API; that is the product. There are also offline tools for GGUF and imatrix
generation, plus development tools for quality and speed testing (see the
Development tools note under Building).

This tree is the **CUDA-only fork** of
[antirez/ds4](https://github.com/antirez/ds4) targeting **NVIDIA CUDA on
Linux**, with special care for the **DGX Spark (GB10)** and its ~128 GB of
unified memory. The upstream Metal (macOS) and ROCm backends were removed in
this fork; see [AGENTS.md](AGENTS.md) for the source layout, supported weight
formats, and validation notes.

This project would not exist without **llama.cpp and GGML**, make sure to read
the acknowledgements section, a big thank you to Georgi Gerganov and all the
other contributors.

This fork's target is deliberately narrow: **DeepSeek-V4-Flash on a single
GB10**. Flash is the right fit for a 128 GB box — quasi-frontier capability,
2-bit-tolerant experts, and a compressed KV cache that makes million-token
contexts practical. The bet is one model, one class of machine, done end to
end rather than a generic runner. If a materially better open-weight model
lands for this size class, the engine can follow it; the model is a target,
not a permanent commitment.

## Acknowledgements to llama.cpp and GGML

Pulsar does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, host-side quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

This fork also stands on:

- **[antirez/ds4](https://github.com/antirez/ds4)** — the upstream DeepSeek V4
  inference engine this project is forked from.
- **DeepSeek** — for the DeepSeek V4 Flash / PRO open-weight models and their
  compressed-KV architecture.
- **[PrismaQuant](https://github.com/RobTand/prismaquant)** (Rob Tand) — the
  measured-KL mixed-precision quant allocation approach used to build this
  fork's GGUFs (see `gguf-tools/prisma/`).
- **[REAP](https://arxiv.org/abs/2510.13999)** (Router-weighted Expert
  Activation Pruning) — the expert-pruning method behind the production build's
  25%-pruned routed experts.
- **[eouya2](https://huggingface.co/eouya2/DeepSeek-V4-Flash-REAP25-LCB50-DS4)**
  — the REAP-25 (LiveCodeBench-50-calibrated) DeepSeek-V4-Flash expert-survivor
  set, vendored as index metadata in `gguf-tools/reap/` and used by the v5mx
  build's REAP transplant.

## Status

This is **beta quality**, but a first release has shipped: the
measured-allocation DeepSeek-V4-Flash GGUF (the `v5mx` build) and the
`pulsar-server` engine that serves it. The inference path, the MXFP4/MXFP8/IQ2
quantization, and speculative decoding are validated against the tests in this
tree and the `tool-eval-bench` quality suite — a 4-seed run of the
shipping build averages ~90/100 with hardmode included (range 87–92 across seeds).
v0.3.1 was re-verified **quality-neutral** versus v0.2.3 on the full 84-scenario
hardmode suite (every committed change — the f32→BF16 residual carrier included —
within seed noise, no regression cluster).
**v0.3.1** is the current release — the first shippable v0.3.x. It brings the v0.3
engine work (Tier-2 batched multi-session decode, KV overcommit with proactive
eviction, the head-grouped decode-attention kernel, and the f32→BF16 residual
carrier) together with the pooled-routing and KV round-trip fixes that stabilized
the line: a divergent prompt match now routes to a fresh bank instead of
clobbering a live one (plus a live-lock follow-up), and two round-trip fixes
eliminate deep cold re-prefills for agentic clients that replay reasoning
(opencode) or strip it (openwebui). It also adds the
`--served-model-id`/`--served-model-name` flags and a default bank-pool cap of 5
(4 concurrent conversations plus the pinned base).
Model serving is
a large surface, so rough edges remain; we keep the project usable and are
actively hardening it. If you hit a problem, run `pulsar-server --trace
/tmp/ds4-trace.txt` to capture the session and open an issue with the full
trace.


## More Documentation

If you are looking for very specific things, we have other
sub-README files. Otherwise for normal usage keep reading the
next sections.

- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md): correctness and speed regression testing
  guide for contributors. **Read this before sending a pull request**.
- [gguf-tools/README.md](gguf-tools/README.md): offline GGUF generation,
  imatrix collection, quantization tooling, and quality checks.
- [gguf-tools/imatrix/README.md](gguf-tools/imatrix/README.md): how the
  routed-MoE imatrix is collected and used.
- [gguf-tools/imatrix/dataset/README.md](gguf-tools/imatrix/dataset/README.md):
  how the calibration prompt corpus is generated.
- [gguf-tools/prisma/README.md](gguf-tools/prisma/README.md): measured-KL
  per-layer expert-format allocation, used to build the mixed-quant GGUFs.
- [gguf-tools/reap/README.md](gguf-tools/reap/README.md): the vendored REAP-25
  expert-survivor map and the Path-B transplant tool.
- [REPRODUCE.md](REPRODUCE.md): the full end-to-end build DAG for the shipped
  v5mx GGUF (templates → oracle → REAP transplant → imatrix → allocation →
  mixed-quant → DSpark merge).
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md): synopsis of the official DeepSeek
  V4 model card, with the architecture details that matter for Pulsar.
- [gguf-tools/quality-testing/README.md](gguf-tools/quality-testing/README.md):
  how local GGUFs are scored against official DeepSeek V4 Flash/PRO continuations.
- [dir-steering/README.md](dir-steering/README.md): directional steering data,
  vector generation, and usage.
- [speed-bench/README.md](speed-bench/README.md): benchmark commands, charts,
  and CSV generation.
- [tests/test-vectors/README.md](tests/test-vectors/README.md): official
  continuation vectors used for regression checks.

## Model Weights

This implementation is not a general GGUF loader: the engine binds one known
DeepSeek V4 tensor layout and rejects everything else. **This fork's weight
binder is stricter than upstream's.** It accepts exactly:

| Tensor group | Accepted formats |
| --- | --- |
| Attention projections, shared experts | MXFP8 (FP8 E4M3 values + per-32 E8M0 scales), run on the FP8 tensor-core GEMM path |
| Routed experts gate/up/down | Per layer: gate/up as a matching pair in `IQ2_XXS`/`Q2_K`/`MXFP4`, down independently in `IQ2_XXS`/`Q2_K`/`MXFP4`; or all three in the CUTLASS tensor-core `MXFP4` layout |
| Output head | `BF16` or MXFP8 |
| Norms, embeddings, indexer, HC | `F32`/`F16` |

Legacy `Q4_K` and `Q8_0` weight tensors are **rejected at load** with a clear
error. This means the GGUFs published for upstream ds4 do not load in this
fork: they use `Q8_0` attention/shared-expert/output tensors (and `Q4_K` in
the q4 and PRO variants). GGUFs for this fork are produced offline with the
tools in [gguf-tools/](gguf-tools/README.md), starting from the original FP8
safetensors (`deepseek4-quantize`, driven per tensor by a `--format-map`
manifest from the prisma allocator). The asymmetric-quantization idea is
unchanged from upstream: only the routed MoE experts — the large majority of
the model bytes — are pushed to 2 or 4 bits, while everything else stays high
precision.

A few things this fork's GGUFs do beyond upstream:

- **Per-layer expert formats.** The expert combo is validated per layer, so
  one GGUF can mix formats across layers. The production allocation is
  measured, not guessed: the prisma pipeline measures real per-layer KL and
  promotes whole layers from the `IQ2_XXS`/`Q2_K` floor to `MXFP4` under a
  byte budget. The `MXFP4` expert tensors are a byte-lossless re-encode of
  the original Hugging Face expert weights, so promoted layers pay zero
  requantization loss. Gate/up and down formats are chosen independently per
  layer; the CUTLASS tensor-core layout is the one exception, since its
  grouped GEMM runs the whole expert FFN in one dispatch, it applies to all
  three tensors or none. The measured-allocation build is the **shipped
  release**: in the hardmode `tool-eval-bench` bake-off that selected it, it
  scored 92/100 against 84/100 for a uniform all-`Q2_K` build, so the measured
  build ships and the all-`Q2_K` build was dropped. (A fresh 4-seed
  re-measure of the shipping build averages ~90/100 with hardmode, range 87–92; the weakest
  area is Category K safety/refusal — including one cross-turn prompt-injection
  scenario the model does not resist — a model-alignment limitation, not a
  quantization artifact.) Recommended sampling for the shipped build is
  temperature 0.95, top-p 0.38 (send these as the `temperature`/`top_p` API
  parameters).
- **REAP expert pruning.** Production GGUFs are REAP expert-pruned: expert
  tensors are dense-trimmed to a per-layer survivor count while the router
  stays padded to the full expert count. This trades a small quality margin
  for significant residency headroom on the GB10. The survivor set is the
  REAP-25 (LiveCodeBench-50-calibrated) prune published by
  [eouya2](https://huggingface.co/eouya2/DeepSeek-V4-Flash-REAP25-LCB50-DS4),
  vendored and transplanted per `gguf-tools/reap/` (full build DAG in
  [REPRODUCE.md](REPRODUCE.md)).
- **Merged DSpark drafter.** The speculative drafter's tensors ship inside
  the same GGUF file (spliced by `gguf-tools/merge_dspark_gguf.py`); see the
  speculative decoding section below.

`download_model.sh` fetches this fork's shipped GGUF from our release repo. It
has a single target:

```sh
./download_model.sh v5mx   # measured-allocation release build, ~91 GB on disk
```

This downloads `ds4flash-v5mx-reap25-type40-mxfp8lt-dspark-v1.gguf`
(91,321,404,640 bytes, sha256
`000974720296f2cad17ac0525796f4bb9ceaac9f4015ed61af3fba445dfb1039`) from
<https://huggingface.co/twaggs88/DeepSeek-V4-Flash-REAP25-DSpark-ds4-GGUF>,
stores it under `./gguf/`, and updates `./ds4flash.gguf` to point at it. The
script prefers the Xet-aware Hugging Face CLI (`hf download`, chunk-deduplicated
and resumable) when present and falls back to `curl -C -` otherwise. The repo is
public, so authentication is optional; `--token TOKEN`, `HF_TOKEN`, or the local
Hugging Face token cache are used when present.

If you want to regenerate GGUF files or collect a new imatrix, see
[gguf-tools/README.md](gguf-tools/README.md). Those tools are meant for offline
model-building work and can take a long time on the full DeepSeek V4 Flash
weights. Flash GGUF generation is supported by the local tools. PRO GGUF
production currently still depends on the external `llama.cpp`-based workflow;
native tooling can be added later.

Then build (the `cutlass/` submodule is required):

```sh
git submodule update --init cutlass
make cuda-spark            # Linux CUDA, DGX Spark / GB10 (sm_120f)
```

`./ds4flash.gguf` is the default model path used by `pulsar-server`. Pass `-m` to
select another supported GGUF from `./gguf/`. Run `./pulsar-server --help` for the
full flag list, and start serving with:

```sh
./pulsar-server -m ds4flash.gguf --ctx 100000
```

## Speed

Performance is measured on this fork's target hardware — a **DGX Spark GB10**
running the ~91 GB REAP-pruned Flash build. The artifact unifies every
MXFP4 routed-expert layer onto the CUTLASS tensor-core **type-40 W4A8 path**
(4-bit weights, E4M3 activations) and stores all MXFP8 workhorse weights in the
**type-41 MXFP8_LT** swizzle (zero-copy, no runtime repack) — both since v0.2.3.
Numbers below are the **v0.3.1** build. Prefill is measured
**cold** (`cache_prompt` off, unique-prefix probe — a warm cache reports
arbitrarily high prompt-processing rates and is not a real prefill number);
decode is measured with the merged DSpark speculative drafter, single stream, on
the server's own wall-clock. All numbers below are the median of three runs at
real depth-varying prompts.

| Context (actual) | Prefill (t/s, cold) | TTFT (cold) | Decode structured (t/s) | Decode prose (t/s) |
| ---: | ---: | ---: | ---: | ---: |
| ~0.3k | — | ~1.2 s | ~25.7 | ~23.0 |
| ~2.3k | ~393 | ~6.2 s | ~24.9 | ~23.6 |
| ~9.3k | ~363 | ~25.9 s | ~21.8 | ~20.6 |

Decode figures are the production speculative path (drafter on, thinking on),
greedy (`temp:0`), single stream. Measured on the v0.3.1 build. Decode improved
over v0.2.3: an under-occupied-kernel fix recovered ~+4.6% at shallow depth
(bit-exact), and the recalibrated yield-quench no longer false-quenches, so the
prose rows in particular sit a few percent higher than the prior build (which
read prose 18.1/20.7/19.6 at the same depths). Prefill is flat versus v0.2.3
(~393 t/s at ~2.3k) and **declines** with depth to ~363 t/s at ~9.3k — cold
prefill slows as attention-at-depth grows, and speculative-decode acceptance
falls, which is the physically expected shape. (The v0.2.3 table's "413 rising at
9.3k" and a 9.3k structured cell above its 2.3k value were non-monotonic
measurement artifacts; these v0.3.1 numbers are internally consistent across the
log estimator, the server `timings{}` block, and a unique-nonce cold probe.) The
build is **fidelity-neutral to source** — the f32→BF16 residual carrier and all
quant paths were re-verified quality-neutral on the full 84-scenario hardmode
suite (see Status). (Speculative decode at `temp:1.0` is comparable and on
several depths faster but is stochastic run-to-run, so the greedy rows are the
reported baseline; the one genuinely noisy cell above is 9.3k structured, where
acceptance forks run-to-run — the median is defensible, a single run is not.)

The MXFP4 rich-expert layers (~4.25 bpw versus the ~2 bpw floor) read ~2x the
weight bytes and set the prefill rate; the type-40 grouped W4A8 GEMM runs those
layers on the tensor cores. An expert-tiled big-batch kernel keeps each decoded
weight resident across a wide token tile so its decode cost amortizes across the
batch. (A uniform 2-bit build prefills faster but scores lower on quality — the
measured allocation trades some prefill for the tool-use win, beating the uniform
2-bit build on hard-mode tool-eval-bench.)

Draft acceptance varies with workload and depth: measured α runs ~81-91% on
structured/tool output and ~60-68% on general prose, so the speedup is largest on
structured/tool workloads where acceptance is highest. The default draft depth is
**3** (measured optimum for this build — see Speculative decoding); the
autoregressive drafter's per-position cost means deeper chains stop paying, and
shallower drafting especially helps at long context, where the old depth-5
default lost the most to verify-row cost. Plain (non-speculative) decode is
roughly flat and bandwidth-bound; speculation is a speedup layered on top. On
shallow, low-acceptance requests speculation can run slightly *slower* than
plain, so **its downside is capped to a few percent by the yield-quench safety
net** described under Speculative decoding — not "speculation never hurts." It
never changes the output distribution — exact sampled acceptance at all
temperatures — so it is a pure speed knob, not a quality one. (Prefill is
measured cold with `cache_prompt` off and a unique-prefix probe to defeat the
prefix cache; decode is measured on the server's own wall-clock. Speculative t/s
is stochastic run-to-run under sampling; the numbers above are `--temp 0`.)

## Model residency

Pulsar is a fully resident engine for **weights**: the CUDA path makes the
whole model (~85 GiB of tensor spans + merged DSpark drafter) resident in
GPU-addressable memory at load time. A model that does not fit is rejected at
startup with a clear error instead of silently degrading — there is no
partial-residency or streaming fallback for weights, so a successful load is a
residency guarantee for the whole run. Since v0.2.3, MXFP8 workhorse weights are
stored in the type-41 **MXFP8_LT** swizzle and loaded zero-copy (no runtime
repack cache), and the type-40 W4A8 expert path runs the rich layers on the
tensor cores.

**KV state, in v0.3.1, is overcommitted rather than fully charged.** The default
context is **1M tokens** with **overcommit ON**: a small pool of banks (auto-sized,
up to 5 as of v0.3.1 — 4 concurrent conversations plus a pinned base bank) comes
up "grow-to-1M-capable," but only the *eager floor* (raw sliding
window + state lanes + drafter) is charged at admission — the ctx-scaled
compressed-KV/indexer term is virtual-address-only, made physical on touch. A
short session pays only for the KV it actually uses. Banks that genuinely grow
toward 1M are bounded by a **proactive LRU-idle eviction guard** (task #55): at
each batched decode-quantum, if the live banks' projected touched-KV would breach
the physical budget, the guard spills the least-recently-used idle bank's KV to
disk and `cudaFree`s its physical slabs, reloading it (bit-identically) on its
next turn. This was stress-validated — under sustained concurrent multi-bank
growth the guard fired repeatedly, bounded memory itself (an independent
out-of-memory watchdog never engaged), and every restore was byte-identical.

Two boundaries are worth knowing: the proactive guard runs **only in the batched
decode lane** (≥2 concurrent decoding banks — the overcommit default's target
regime); a purely sequential single-stream workload leans on admission plus a
live-`MemAvailable` floor backstop instead. And the guard is **decode-quantum
granular**, so a bank's *prefill* growth between checks is not proactively
bounded — the layered defenses (admission floor + MemAvailable backstop) catch
that case. An operator who wants a hard per-session cap passes a smaller `--ctx`;
`DS4_OVERCOMMIT=0` reverts to classic full-charge admission (1M ⇒ a single
session).

### Startup warmup and session admission

At startup, before the listener accepts requests, the server logs the overcommit
fit table and runs a short warmup generation that materializes the
first-generation CUDA working set (~7.2 GiB). Warmup lets admission measure the
*actual* memory cost of a slot rather than estimate it. At the 1M default the
boot log reads, for example:

```text
Tier-2 pool fit table (budget 14.0 GiB, cap 5 banks):
  ctx  131072: fits 5 bank(s) (1-bank 4.57 GiB, 5-bank 7.41 GiB)
  ctx 1048576: fits 2 bank(s) (1-bank 9.59 GiB, 2-bank 13.89 GiB)
Tier-2 OVERCOMMIT auto-sized to 5 bank(s) for --ctx 1048576: eager floor
  0.20 GiB/bank charged; demand-paged VA 4.11 GiB/bank reserved (physical on
  touch, NOT charged); shared 5.28 GiB; admission est 6.26 GiB (batching ON)
warmup generation: 4160 prompt + 12 decode tokens in 10.8 s;
  MemAvailable 16.08 -> 12.13 GiB (first-generation working set materialized)
Tier-2 shared pool ACTIVE: 5 banks, spec_max_live=1, pool resident 26.80 GiB
Tier-2 2b guard ENABLED: touched budget 7.69 GiB (kv budget 13.95 - eager 6.26),
  spill dir ./ds4-spill
```

Without overcommit the 1M context would admit a single session (the full-charge
fit table shows only 2 banks fit even at the 1M ceiling); with overcommit the
same box brings up all 5 banks at the eager floor and lets each grow into its
touched budget under the eviction guard. Admission is driven by this measured
ledger, not a fixed slot count. (Prefill numbers are effectively post-warmup —
the first real request already has its working set resident, which is why cold
and process-warm prefill agree within run-to-run noise.)

Long-context KV state is kept compact by default: the compressed-attention
cache uses a bit-exact packed value layout (`DS4_ATTN_PACK`, on by default),
and the ratio-4 indexer cache — the dominant KV term at very long context —
is stored MXFP4-packed (`DS4_IDX_FP4`, on by default). Set either to `0` only
for debugging comparisons against the plain f32 layouts.

## Speculative decoding (DSpark)

Pulsar has exactly one drafter: **DSpark**. The earlier MTP drafter path
was removed rather than kept as a semi-supported variant. The drafter's
tensors ship merged inside the main model GGUF and the engine auto-enables
speculative decoding when it detects them at load — no extra file or flag
needed. `--no-dspark` opts out, and `--dspark PATH` still loads a split
drafter file. The `pulsar-bench` development tool intentionally measures the
plain-decode baseline.

Acceptance is **exact sampled acceptance at all temperatures**: a draft token
is accepted with its filtered target probability, and on rejection the engine
samples from the residual distribution. The speculative path filters logits
exactly like the plain sampler (temperature scale, top-k, min-p, top-p), so
the output distribution is provably identical to plain sampling — greedy
decoding is just the point-mass special case of the same walk. Speculative
decoding is purely a speedup, never a quality knob, and it now applies to the
sampling recipes people actually use, not only temperature 0.

This guarantee is only meaningful because decode and prefill numerics are
run-to-run deterministic: accumulation orders are fixed everywhere
(no atomic-order races, no split-K reduction schemes), so the same prompt,
sampling parameters, and seed reproduce the same output across runs, and the
speculative verify pass sees exactly the numerics plain decode would.

### Yield-quench safety net

A draft that is rarely accepted costs more than it saves, so on shallow or
low-acceptance requests speculation can be a net slowdown. This release pairs the
drafter with a **yield-quench** controller: it tracks each request's realized
draft economics (an EWMA of per-step speedup and an accumulated debt) and, once a
request is clearly losing, latches *that request* to plain decode for the
remainder — bounding the worst case. The controller has to run a short mandatory
pre-latch window (`WARMUP=3` + `MINEV=8` steps) speculatively before it can fire,
so on very short or low-α generations a small residual cost survives. Measured in
this release's sweep, quench caps the worst case from **~−26%** (unquenched
speculation on the historically-losing prose / `temp:1.0` / thinking-off cell)
to a **−2% to −6%** residual on the shallow, low-acceptance cells. The honest
claim is therefore **"speculation's downside is now capped to a few percent,"**
not "speculation never hurts." When quench fires it logs, e.g.:

```text
pulsar: dspark yield-quench pos=330 steps=13 debt=5.92 ewma=-0.60 -> plain decode for request remainder
```

The min-p draft prefilter on the speculative path is bit-exact at the production
sampling parameters (`top_p=1.0`), so both quench and the prefilter change only
speed, never the sampled output distribution.

## Prefill chunking

Prefill chunking is configurable and affects the KV checkpoint/logit path.
Sessions prefill long prompts in 4096-token chunks by default; use the server's
`--prefill-chunk N` (or `DS4_CUDA_PREFILL_CHUNK=N`) to compare another chunk
size, for example `2048` to match the strict official-vector checkpoint path,
or `0` to prefill a prompt as one whole batch when memory allows. Changing the
chunk changes the KV checkpoint/logit path, so compare it as an explicit run
configuration. Chunked GPU prefill reuses the same range-capable layer-major
graph for each chunk, preserving absolute compressor/indexer boundaries while
avoiding the old per-layer chunk dispatch path.

## Server

Start a local OpenAI/Anthropic-compatible server:

```sh
./pulsar-server --ctx 100000
```

The disk KV cache is on by default (see [Disk KV Cache](#disk-kv-cache));
add `--kv-disk-dir DIR` / `--kv-disk-space-mb N` only to relocate or resize it,
or `--no-kv-disk` to turn it off.

Use `--chdir /path/to/pulsar` when launching `pulsar-server` from another directory,
so relative runtime paths such as the default `./ds4flash.gguf` model and
`dir-steering/` data resolve from the project tree.

Two flags (v0.3.1) control how the model is *named* to clients, independently:
`--served-model-id ID` sets the id reported in `/v1/models` (`id` and `root`),
`/version`, and the `/metrics` label — default `deepseek-v4-flash`. Point it at an
HF repo path (e.g. `--served-model-id org/Model`) so HF-convention benchmark
tooling that treats the model id as the tokenizer id (llama-benchy et al.)
resolves the tokenizer with no extra config. `--served-model-name NAME` sets the
human display name (`/v1/models` `name`, `/version` `model_name`) — free-form, may
contain spaces, and is never used as a tokenizer id. Neither affects the KV cache
(keyed by gguf path + numeric model id) or request acceptance (the incoming
`model` field is not validated).

The server keeps one mutable backend/KV checkpoint in memory,
so stateless clients that resend a longer version of the same prompt can reuse
the shared prefix instead of pre-filling from token zero.

Request parsing and sockets run in client threads; inference runs on one graph
worker. The server serves **multiple concurrent sessions** from a small pool of
banks admitted against a measured KV/VRAM budget. As of **v0.3.1**, concurrently
decoding banks share **one batched decode step** (Tier-2 batched multi-session
decode), so aggregate decode throughput now *scales* with concurrency rather than
being the single-stream rate divided across sessions; concurrent prefills overlap
as well. The default context is the model's **1M-token** limit with overcommit
on, so a bank is admitted at its eager floor and grows into its touched budget
under the eviction guard rather than being charged the full 1M up front (see
Model residency). A returning client whose bank was evicted resumes from a disk
KV snapshot instead of re-prefilling, and requests beyond the admission budget
queue or are refused rather than driving the box out of memory.

One property of batched decode is worth noting for reproducibility: an M≥2
co-batched forward differs by ~1 ULP from the single-sequence path (different
GEMM tiling / reduction order). This is invisible under sampling (`temp` > 0) and
distribution-preserving, but at **greedy `temp:0`** it can flip a near-tie argmax,
so a request's greedy continuation can depend on what else is co-scheduled at the
moment — the same class of numerical nondeterminism as batched inference in other
engines. Likewise, v0.3.1's head-grouped decode-attention kernel is not
bit-identical to v0.2.3 at depth, so a given greedy prompt's continuation may
differ from the prior release (quality-verified neutral; set
`DS4_CUDA_NO_INDEXED_DECODE_HEADS8=1` to restore the prior kernel).
Two further guardrails keep a misbehaving client from wedging the server:
concurrent client connections are capped (64; connections over the cap get an
immediate 503 instead of piling up threads), and a whole-request read deadline
(30 seconds) drops clients that never finish sending their request.

Supported endpoints:

- `GET /health` — readiness + live status: `{status, version, model, uptime_s, slots{total,running,waiting}, kv_cache_usage}`; **200** when serving, **503 `{"status":"draining"}`** once shutdown is requested (so a load balancer stops routing)
- `GET /healthz`, `GET /ping` — liveness (always **200** while the process runs, independent of drain state)
- `GET /version` — `{version, engine, cuda_arch, model, model_name, context}`
- `GET /` — banner with version + endpoint list
- `GET /v1/models`
- `GET /v1/models/{id}`
- `POST /v1/chat/completions` — OpenAI-compatible
- `POST /v1/completions` — OpenAI-compatible
- `POST /v1/responses` — OpenAI Responses (Codex CLI)
- `POST /v1/messages` — Anthropic-compatible (Claude Code style clients)
- `GET /metrics` — Prometheus counters (spec-decode acceptance, token totals)

`GET /v1/models/{id}` accepts any id (`deepseek-v4-flash`, `deepseek-v4-pro`)
as a compatibility alias: it reports the model actually loaded from the GGUF
passed with `-m`; the id does not select a different model.

Use `GET /health` for readiness/status (or `GET /v1/models`, which also
returns 200 when the model is loaded and serving).

`/v1/chat/completions` accepts the usual OpenAI-style `messages`,
`max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`,
`seed`, `stream`, `stream_options.include_usage`, `tools`, and `tool_choice`.
Tool schemas are rendered into DeepSeek's DSML tool format, and generated DSML
tool calls are mapped back to OpenAI tool calls.

`/v1/responses` accepts OpenAI Responses-style `input`, `instructions`,
`tools`, `tool_choice`, `max_output_tokens`, `temperature`, `top_p`, `stream`,
and `reasoning`. It is the preferred endpoint for Codex CLI. The server keeps
Responses continuations bound to live state when possible, and can fall back to
the same DSML rendering and KV prefix reuse used by chat completions.

`/v1/messages` is the Anthropic-compatible endpoint used by Claude Code style
clients. It accepts `system`, `messages`, `tools`, `tool_choice`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `stream`, `stop_sequences`, and thinking
controls. Tool uses are returned as Anthropic `tool_use` blocks.

Default sampled API generation uses `temperature=1`, `top_p=1`, and
`min_p=0.05`, so the default filter is relative probability rather than
nucleus mass. Those defaults apply only to parameters the request leaves
absent (in thinking and non-thinking mode alike); any sampling knob the client
sends explicitly is respected as-is, including `temperature=0` to select
greedy decode (which lets DSpark speculative decoding engage). During
structured tool-call output the server still forces greedy decode regardless
of the request's sampling parameters. `min_p` is range-validated at parse time:
an out-of-range value disables the min-p filter (the same convention `top_p`
uses out of range) rather than being silently clamped.

The chat, Responses, and Anthropic endpoints support SSE streaming. In thinking
mode, reasoning is streamed in the native API shape instead of being mixed into
final text. This holds even when generation stops inside an unterminated
`<think>` block (token cap or stop sequence): the partial reasoning is
returned on the reasoning channel and never as visible content, on both the
streaming and non-streaming paths. OpenAI chat streaming
also streams tool calls as soon as the DSML invocation is recognized: the tool
header is sent first, then parameter bytes are forwarded as
`tool_calls[].function.arguments` deltas while generation continues. The
Anthropic endpoint streams thinking and text live, then emits structured
`tool_use` blocks when the generated tool block is complete.
The Responses endpoint streams the Responses event lifecycle expected by Codex,
including `response.output_text.delta`, function-call argument events, and
terminal `response.completed` / `response.incomplete` / `response.failed`
events.

For browser JavaScript clients served from another origin, start the server with
`--cors` to emit `Access-Control-Allow-*` headers. This only changes HTTP
headers; it is independent of which interfaces the server binds.

> **Network exposure.** `pulsar-server` binds **`0.0.0.0` (all interfaces) by
> default** and has **no authentication** — any host that can reach the port can
> use the model, the agent tools, and `/metrics`. This is a deliberate
> single-node, trusted-LAN design: run it only on a network you control. To
> restrict it to the local machine, pass `--host 127.0.0.1`. Do not expose the
> port directly to the public internet; put it behind your own authenticating
> reverse proxy if you need remote access.

### Tool call handling and canonicalization

DeepSeek V4 emits tool calls as [DSML text](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/README.md). Agent clients do not send that
same text back on the next request: they send normalized OpenAI/Anthropic JSON
tool-call objects. **If the server re-rendered those objects slightly
differently, the rendered byte prefix would no longer match the live KV
checkpoint** and the next turn would have to be rebuilt.

The first line of defense is exact replay. Every tool call gets an unguessable
API tool ID, and the server remembers `tool id -> exact sampled DSML block` in
a bounded in-memory map backed by radix trees. When the client later sends that
tool ID back, the prompt renderer uses the exact DSML bytes the model sampled,
not a freshly formatted approximation. This map can also be saved inside KV
cache files, so exact replay survives server restarts for cached histories.

**Canonicalization is only the backup path**. If the exact DSML block is missing,
or exact replay is disabled with `--disable-exact-dsml-tool-replay`, the server
renders a deterministic DSML form from the JSON tool object. After a tool-call
turn, it compares the live sampled token stream with the prompt that the next
client request will render. If needed, it rewrites the live checkpoint, or
falls back to an older disk KV snapshot and replays only the suffix. This keeps
the model continuation aligned with the stateless API transcript.

During generation, the server also treats DSML syntax differently from payload.
When the model is emitting stable protocol structure such as DSML tags,
parameter headers, JSON punctuation, or closing markers, sampling is forced to
`temperature=0` so the tool call stays parseable. This greedy mode does **not**
apply to argument payloads: `string=true` parameter bodies and JSON string
values, including file contents and edit text, use the request's normal sampling
settings. That separation is important: deterministic decoding is helpful for
syntax, but can create repeated text when applied to long code or file bodies.

Minimal OpenAI example:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash",
    "messages":[{"role":"user","content":"List three Redis design principles."}],
    "stream":true
  }'
```

### Agent Client Usage

`pulsar-server` can be used by local coding agents that speak OpenAI-compatible
chat completions. Start the server first, and set the client context limit no
higher than the `--ctx` value you started the server with:

```sh
./pulsar-server --ctx 100000
```

You can use larger context and larger cache if you wish, up to the model's
1M-token context limit. The default MXFP4-packed indexer cache keeps
long-context KV state far below its old f32 cost, but configure a context
which makes sense in your system: on the GB10 with ~128GB of unified memory,
the ~91GB Flash build takes most of the memory, so size the context window to
the headroom you actually have and to the prefill time you are willing to pay.

The `384000` output limit below avoids token caps since the model is able
to generate very long replies otherwise (up to 384k tokens). The server
still stops when the configured context window is full.

For **opencode**, add a provider and agent entry to
`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "pulsar": {
      "name": "pulsar (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "dsv4-local"
      },
      "models": {
        "deepseek-v4-flash": {
          "name": "DeepSeek V4 Flash (pulsar local)",
          "limit": {
            "context": 100000,
            "output": 384000
          }
        }
      }
    }
  },
  "agent": {
    "pulsar": {
      "description": "DeepSeek V4 Flash served by local pulsar-server",
      "model": "pulsar/deepseek-v4-flash",
      "temperature": 0
    }
  }
}
```

For **Pi**, add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "pulsar": {
      "name": "pulsar local",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "dsv4-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "deepseek",
        "requiresReasoningContentOnAssistantMessages": true
      },
      "models": [
        {
          "id": "deepseek-v4-flash",
          "name": "DeepSeek V4 Flash (pulsar local)",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh"
          },
          "input": ["text"],
          "contextWindow": 100000,
          "maxTokens": 384000,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Optionally make it the default Pi model in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "pulsar",
  "defaultModel": "deepseek-v4-flash"
}
```

For **Codex CLI**, use the Responses wire API:

```toml
[model_providers.pulsar]
name = "Pulsar"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1000000
```

Then run:

```sh
codex --model deepseek-v4-flash -c model_provider=pulsar
```

For **Claude Code**, use the Anthropic-compatible endpoint. A wrapper like this
matches the local `~/bin/claude-ds4` setup:

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="${DS4_ANTHROPIC_BASE_URL:-http://127.0.0.1:8000}"
export ANTHROPIC_AUTH_TOKEN="${DS4_API_KEY:-dsv4-local}"
export ANTHROPIC_MODEL="deepseek-v4-flash"

export ANTHROPIC_CUSTOM_MODEL_OPTION="deepseek-v4-flash"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="DeepSeek V4 Flash local pulsar"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="pulsar local GGUF"

export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec "$HOME/.local/bin/claude" "$@"
```

Claude Code may send a large initial prompt, often around 25k tokens, before it
starts doing useful work. Keep the default-on disk KV cache enabled (do not pass
`--no-kv-disk`): after the first expensive prefill, the disk KV cache lets later
continuations or restarted sessions reuse the saved prefix instead of
processing the whole prompt again.

## Thinking Modes

DeepSeek V4 Flash has distinct non-thinking, thinking, and Think Max modes.
The server defaults to thinking mode. `reasoning_effort=max` requests Think
Max, but it is only applied when the context size is large enough for the model
card recommendation; smaller contexts fall back to normal thinking. OpenAI
`reasoning_effort=xhigh` still maps to normal thinking, not Think Max.

For direct replies, use `thinking: {"type":"disabled"}`, `think:false`, or a
non-thinking model alias such as `deepseek-chat`.

## Disk KV Cache

Chat/completion APIs are stateless: agent clients usually resend the whole
conversation every request. `pulsar-server` first tries the cheap exact token-prefix
check, then falls back to comparing rendered prompt bytes with decoded
checkpoint bytes. The live in-memory checkpoint covers the current session; the
disk KV cache makes useful prefixes survive session switches and server
restarts.

For RAM reasons there is currently only one live KV cache in memory. When a new
unrelated session replaces it, the old checkpoint can only be resumed without
re-processing if it was written to the disk KV cache. In other words, memory
cache handles the active session; disk cache is the resume mechanism for
different sessions.

The disk KV cache is **on by default**. When `--kv-disk-dir` is not given, the
server resolves `$XDG_CACHE_HOME/ds4/kv-<model>` (else `~/.cache/ds4/kv-<model>`),
where `<model>` is the symlink-resolved gguf basename — so two different model
artifacts never share a default directory, while restarts of the same model
find their checkpoints again. The resolved directory and disk budget are logged
at startup. If the directory cannot be created or is not writable, the server
logs that once and keeps serving with the disk cache disabled — it never fails
to start over cache placement.

```sh
./pulsar-server                                        # default-on, default dir
./pulsar-server --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192   # custom
./pulsar-server --no-kv-disk                           # opt out (or --kv-disk-dir "")
```

The disk budget defaults to 4096 MiB (`--kv-disk-space-mb`); least-valuable
checkpoints are evicted when the budget fills. Snapshot behavior is unchanged
by the default-on switch: files are written with ordinary write + fsync + rename
into place, and the store validates the model variant and the rendered byte
prefix before restoring (quant bits are recorded and logged, but mismatches are
refused only with `--kv-cache-reject-different-quant`; the per-artifact default
directory is what keeps different builds' checkpoints apart). Note that on unified-memory hosts (GB10) the
checkpoint writes pass through the host page cache, which competes with GPU
memory — the usual `sync; echo 3 > /proc/sys/vm/drop_caches` discipline before
a model load still applies; pass `--no-kv-disk` if you want none of that
pressure.

The cache key is the SHA1 of the rendered byte prefix, and files are named
`<sha1>.kv`. The Pulsar payload still stores the exact token IDs and graph state
for that prefix. This matters for continued chats: the model may have generated
one token whose decoded text is later sent back by a client as two canonical
prompt tokens. A rendered byte-prefix hit can still reuse the checkpoint and
tokenize only the new suffix.
The file is intentionally written with ordinary `read`/`write` I/O, not
`mmap`, so restoring cache entries does not add more VM mappings to a process
that already maps the model.

Tool calls also keep a bounded exact-DSML replay map keyed by unguessable tool
IDs, so client JSON history can be rendered back to the exact sampled text. The
RAM map keeps up to 100000 IDs by default; tune it with `--tool-memory-max-ids`.
Use `--disable-exact-dsml-tool-replay` to disable this and fall back to
canonical JSON-to-DSML rendering.

On disk, a cache file is:

```text
KVC fixed header, 48 bytes
u32 rendered_text_bytes
rendered_text_bytes of UTF-8-ish token text
Pulsar session payload, payload_bytes from the KVC header
optional tool-id map section
```

The fixed header is little-endian:

```text
0   u8[3]  magic = "KVC"
3   u8     version = 1
4   u8     routed expert quant bits, currently 2 or 4
5   u8     save reason: 0 unknown, 1 cold, 2 continued, 3 evict, 4 shutdown
6   u8     extension flags, bit 0 = appended tool-id map
7   u8     reserved
8   u32    cached token count
12  u32    hit count
16  u32    context size the snapshot was written for
20  u8[4]  reserved
24  u64    creation Unix time
32  u64    last-used Unix time
40  u64    Pulsar session payload byte count
```

The rendered text is the tokenizer-decoded text for the cached token prefix.
It is both the human-inspectable prefix and the lookup identity: its SHA1 is
the filename, and a file is reusable only when those bytes are a prefix of the
incoming rendered prompt. After load, the exact checkpoint tokens from the Pulsar
payload remain authoritative, and only the incoming text suffix after the cached
bytes is tokenized.

The optional tool-id map is present only when header extension bit 0 is set.
Appended sections use fixed bit order, so future extension bits can add fields
without ambiguity. The map stores unguessable API tool call IDs back to the
exact DSML block the model sampled. Only mappings whose DSML block is present
in the rendered cached text are stored. This lets restarted servers render
later client history byte-for-byte like the original model output, even if the
client reorders JSON arguments.

The current tool-id map section is:

```text
0   u8[3]  magic = "KTM"
3   u8     version = 1
4   u32    entry count

For each entry:
0   u32    tool id byte length
4   u32    sampled DSML byte length
8   bytes  tool id
... bytes  exact sampled DSML block
```

The section is auxiliary replay memory, not model state. A cache hit restores
the session payload first, then loads the map if present. Before rendering a
request, the server can also scan cache files for the tool IDs present in the
client history and load just those mappings, so an exact DSML replay can survive
server restarts even when the matching KV snapshot is not the one ultimately
used for the rendered-prefix hit.

The Pulsar session payload starts with thirteen little-endian `u32` fields:

```text
0   magic = "DSV4"
1   payload version = 2
2   saved context size
3   prefill chunk size
4   raw KV ring capacity
5   raw sliding-window length
6   compressed KV capacity
7   checkpoint token count
8   layer count
9   raw/head KV dimension
10  indexer head dimension
11  vocabulary size
12  live raw rows serialized below
```

Then it stores:

- `u32[token_count]` checkpoint token IDs.
- `float32[vocab_size]` logits for the next token after that checkpoint.
- `u32[layer_count]` compressed attention row counts.
- `u32[layer_count]` ratio-4 indexer row counts.
- For every layer: the live raw sliding-window KV rows, written in logical
  position order rather than physical ring order.
- For compressed layers: live compressed KV rows and compressor frontier
  tensors.
- For ratio-4 compressed layers: live indexer compressed rows and indexer
  frontier tensors.

The logits are raw IEEE-754 `float32` values from the host `pulsar_session`
buffer. They are saved immediately after the checkpoint tokens so a loaded
snapshot can sample or continue from the exact next-token distribution without
running one extra decode step. Speculative draft state is not persisted; after
loading a disk checkpoint it is invalidated and rebuilt by normal generation.

The tensor payload is Pulsar-specific KV/session state, not a generic inference
graph dump. It is expected to be portable only across compatible Pulsar
builds for this model layout.

The cache stores checkpoints at four moments:

- `cold`: after a long first prompt reaches a stable prefix, before generation.
- `continued`: when prefill or generation reaches the next absolute aligned frontier.
- `evict`: before an unrelated request replaces the live in-memory session.
- `shutdown`: when the server exits cleanly.

Cold saves intentionally trim a small token suffix and align down to a prefill
chunk boundary. This avoids common BPE boundary retokenization misses when a
future request appends text to the same prompt. The defaults are conservative:
store prefixes of at least 512 tokens, cold-save prompts up to 30000 tokens,
trim 32 tail tokens, and align to 2048-token chunks. The important knobs are:

Continued saves use the same alignment and are written only when the live graph
naturally reaches an absolute frontier. With the defaults this means roughly
every 10k tokens, independent of where the first cold checkpoint landed, so long
generations leave restart points behind without persisting the fragile final few
tokens.

- `--kv-cache-min-tokens`
- `--kv-cache-cold-max-tokens`
- `--kv-cache-continued-interval-tokens`
- `--kv-cache-boundary-trim-tokens`
- `--kv-cache-boundary-align-tokens`
- `--tool-memory-max-ids`
- `--disable-exact-dsml-tool-replay`

By default, checkpoints may be reused across the 2-bit and 4-bit routed-expert
variants if the rendered prefix matches. Use `--kv-cache-reject-different-quant`
when you want strict same-quant reuse only.

The cache directory is disposable. If behavior looks suspicious, stop the
server and remove it. You can investigate what is cached with hexdump as
the kv cache files include the verbatim prompt cached.

## Building

Pulsar is **CUDA-only**; the whole model runs as a single CUDA-graph
inference path. Plain `make` prints the available build targets instead of
selecting one implicitly. For the DGX Spark / GB10 (the validated production
target) build with:

```sh
make cuda-spark
```

The CUTLASS MXFP4 expert path requires the **`sm_120f`** family arch for its
block-scaled tensor-core MMA, so `cuda-spark` builds with `CUDA_ARCH=sm_120f`.
This fork targets the GB10 only; there is a single build target, and
`make cuda-spark` builds only the shipped binary, `pulsar-server`.

If a binary is run on a GPU whose architecture it was not built for, it **fails
fast at startup**: every translation unit's compiled arch list is checked at
init, and a mismatch aborts with rebuild instructions instead of crashing later
inside a kernel launch.

### Development tools

The tree also builds `pulsar` (the CLI), `pulsar-bench` (speed microbench), `pulsar-eval`
(embedded capability check), and `pulsar-agent` (native coding agent) via
`make <name>` — for example `make pulsar-bench`. These are development tools for
contributors; they are not part of the release and are not built by
`make cuda-spark`. Some are untested after the pre-release cleanup.

## Steering

This project supports steering with single-vector activation directions; see the
`dir-steering` directory for more information. This follows the core idea of the
[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
paper. You can use it to make the model more or less verbose, less likely to
answer programming questions if it is a chatbot for your car rental web site,
and so forth, much faster than fine-tuning.
This is also useful for cybersecurity researchers who want to reduce a model's
willingness to provide dual-use or offensive security guidance.

`pulsar-server` exposes steering directly: point it at a direction file and set the
per-path scales (`--dir-steering-ffn` after FFN outputs, `--dir-steering-attn`
after attention outputs):

```sh
./pulsar-server -m ds4flash.gguf --ctx 100000 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -4.0
```

With no steering file or zero scales the server follows the normal inference
path.

## Test Vectors

`tests/test-vectors` contains short and long-context continuation vectors
captured from the official DeepSeek V4 Flash API. The requests use
`deepseek-v4-flash`, greedy decoding, thinking disabled, and the maximum
`top_logprobs` slice exposed by the API. Local vectors are generated with
`./pulsar --dump-logprobs` (the `pulsar` development CLI, built with `make pulsar`) and
compared by token bytes, so tokenizer/template or
attention regressions show up before they become long generation failures. The
C runner pins `DS4_CUDA_PREFILL_CHUNK=2048` for this strict API-vector
comparison.

All project tests are driven by the C runners:

```sh
make test                  # pulsar_test
./pulsar_test --logprob-vectors
./pulsar_test --server
```

`pulsar_test` loads a model (`DS4_TEST_MODEL`, default `./ds4flash.gguf`);
`pulsar_agent_test` and the eval self-test run without one. `make
cuda-regression` runs a GPU kernel smoke test that needs no model.

## Debugging Notes

When a generation looks wrong, a few small tools are usually enough to get a
first answer. `pulsar-server --trace` is the one that works against the shipped
binary; the `--dump-*` probes live on the `pulsar` development CLI (`make pulsar`):

```sh
./pulsar-server --trace /tmp/ds4-trace.txt ...
./pulsar --dump-tokens -p "..."
./pulsar --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./pulsar --dump-logits /tmp/logits.json --cuda --nothink --prompt-file prompt.txt
```

- `--dump-tokens` tokenizes the `-p` or `--prompt-file` string exactly as
  written, recognizes Pulsar protocol specials, and then exits before inference
  starts. For example, the DSML tool close marker starts as two tokens: `</`
  and `｜DSML｜`.
- `--dump-logprobs` stores a greedy continuation with the top local
  alternatives at each step, which helps separate sampling choices from
  logit/model issues.
- `pulsar-server --trace` writes the rendered prompts, cache decisions, generated
  text, and tool-parser events for a whole agent session.

## Roadmap

This is the current development direction. Near-term items are planned for
the v1.x series; longer-term items are directional and may be reordered or
dropped as measurements come in.

### Near term (v1.x)

- **Temperature-matched draft sampling.** Today drafts are proposed greedily
  and verified with exact sampled acceptance against the target distribution.
  The next step is to sample drafts from a real q-distribution built at the
  request's sampling parameters, with a fused GPU p/q rejection verify:
  accept with probability `min(1, p/q)`, sample the `(p - q)+` residual on
  rejection. This is expected to roughly double draft acceptance at
  temperature compared to greedy drafts. Prior art: the p/q scheme from the
  DSpark paper, and Marco Palaferri's independent GB10 fork
  ([xangel82/DS4-GB10-GX10-DSpark-CUDA](https://github.com/xangel82/DS4-GB10-GX10-DSpark-CUDA),
  MIT) as a reference implementation. Our version will keep the
  exact-output-distribution property under top-k/top-p/min-p filtering, and
  the statistical oracle used to validate sampled acceptance will be
  extended to the p/q case.
- **Fused K-row GPU accept kernel.** The sampled verify currently reads
  per-row logits back to the host; a fused accept kernel removes that
  readback from the decode loop.
- **Online drafter confidence calibration.** Replace the static
  confidence schedule that trims low-confidence draft tails with an
  adaptive threshold calibrated from observed acceptance.
- **Agent-loop speculative decoding.** Wire speculative decoding into the
  native agent loop, with rewind-on-forced-token handling so injected
  protocol tokens do not desynchronize the drafter.

The measured-allocation release artifact has **shipped** (the `v5mx` build; see
Model Weights and Speed above); it is no longer a near-term roadmap item.

### Longer term

- **Batched multi-session decode.** Concurrent sessions are served today by
  time-slicing one engine lane (a session pool with round-robin scheduling,
  KV-budget admission control, and LRU eviction to a disk KV cache). The next
  step is a batched decode step so co-scheduled sessions share a single read of
  the weights: decode is weight-bandwidth-bound, so aggregate throughput should
  scale toward ~N× (early engine-level measurements show ~1.7× at three
  sessions before it saturates).
- **MXFP6 tensor-core paths**, if measured demand holds. MXFP6 (E2M3) is
  expected to beat `Q6_K` on quality-per-byte for FP8-source tensors.
- **MoE verify-microbatch routing audit.** Verify batches are only a few
  rows but currently pass through prefill-shaped routing machinery; audit
  it and skip what a small batch does not need.
- **Grouped MXFP4 prefill GEMM.** Prefill dispatches the CUTLASS MXFP4
  expert GEMM per expert behind a blocking per-layer offset readback. A
  single ptr-array grouped launch (available in the SM120 blockscaled
  builder without a CUTLASS bump) would remove that readback and the
  per-expert launch overhead — a prefill-only win, worth profiling first
  since prefill is already competitive.
- **Retire the remaining CPU compute paths.** The engine is CUDA-only; the
  leftover host-side decode code inherited from upstream should go. In the
  same pass, revisit the Q8_K activation quantization used by the MoE
  kernels.
- **Upstream quantization-pipeline fixes.** Offer the fixes developed here
  for the measured-KL allocation pipeline — Fisher-proxy normalization,
  footprint accounting, solver corrections, DP decision units — to the
  [PrismaQuant](https://github.com/RobTand/prismaquant) project.
