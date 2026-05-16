# DwarfStar 4

DwarfStar 4 is a small native inference engine specific for **DeepSeek V4 Flash**. It is
intentionally narrow: not a generic GGUF runner, not a wrapper around another
runtime: it is completely self-contained. Other than running the model in a
correct and fast way, the project goal is to provide DS4 specific loading,
prompt rendering, tool calling, KV state handling (RAM and on-disk), and server
API, all ready to work with coding agents or with the provided CLI interface.
There are also tools for GGUF and imatrix generation, and for quality and
speed testing.

We support the following backends:
* **Metal** is our primary target. Starting from MacBooks with 96GB of RAM.
* **NVIDIA CUDA** with special care for the DGX Spark.
* **AMD ROCm** is only supported in the [rocm](https://github.com/antirez/ds4/tree/rocm) branch. It is kept separate from main since I (antirez) don't have direct hardware access, so the community rebases the branch as needed.

This project would not exist without **llama.cpp and GGML**, make sure to read
the acknowledgements section, a big thank you to Georgi Gerganov and all the
other contributors.

## Motivations

Now, back at this project. Why we believe DeepSeek v4 Flash to be a pretty special
model deserving a standalone engine? Because after comparing it with powerful smaller
dense models, we can report that:

1. DeepSeek v4 Flash is faster because of less active parameters.
2. In thinking mode, if you avoid *max thinking*, it produces a thinking section that is a lot shorter than other models, even 1/5 of other models in many cases, and crucially, the thinking section length is **proportional to the problem complexity**. This makes DeepSeek v4 Flash usable with thinking enabled when other models are practically impossible to use in the same conditions.
3. The model features a context window of **1 million tokens**.
4. Being so large, it knows more things if you go sampling at the edge of knowledge. For instance asking about Italian show or political questions soon uncovers that 284B parameters are a lot more than 27B or 35B parameters.
5. It writes much better English and Italian. It *feels* a quasi-frontier model.
6. The KV cache is incredibly compressed, allowing long context inference on local computers and **on disk KV cache persistence**.
7. It works well with 2-bit quantization, if quantized in a special way (read later). This allows to run it in MacBooks with 128GB of RAM (and many people reported it working with 96GB as well, even at 250k context window!).
8. We expect DeepSeek to release **updated versions of v4 Flash** in the future, even better than the current one.

That said, a few important things about this project:

* The local inference landscape contains many excellent projects, but new models are released continuously, and the attention immediately gets captured by the next model to implement. This project takes a deliberately narrow bet: one model at a time, official-vector validation (logits obtained with the official implementation), long-context tests, and enough agent integration to know if it really works. The exact model may change as the landscape evolves, but the constraint remains: local inference credible on high end personal machines or Mac Studios, starting from 96/128GB of memory.
* This software is developed with **strong assistance from GPT 5.5** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.
* This implementation is based on the idea that compressed KV caches like the one of DeepSeek v4 and the fast SSD disks of modern MacBooks should change our idea that KV cache belongs to RAM. **The KV cache is actually a first-class disk citizen**.
* Our vision is that local inference should be a set of three things working well together, out of the box: A) inference engine with HTTP API + B) GGUF specially crafted to run well under a given engine and given assumptions + C) testing and validation with coding agents implementations. This inference engine only runs with the GGUF files provided. It gets tested against officially obtained logits at different context sizes. This project exists because we wanted to make one local model feel finished end to end, not just runnable. However this is just alpha quality code, so probably we are not still there.
* The optimized graph path targets **Metal on macOS** and **CUDA on Linux**. The CPU path is only for correctness checks and model/tokenizer diagnostics. For CPU-only Linux builds, use `make cpu`; it builds the normal `./ds4` and `./ds4-server` binaries without CUDA or Metal. On macOS, **warning: current macOS versions have a bug in the virtual memory implementation that will crash the kernel** if you try to run the CPU code. Remember? Software sucks. It was not possible to fix the CPU inference to avoid crashing, since each time you have to restart the computer, which is not funny. Help us, if you have the guts.

## Acknowledgements to llama.cpp and GGML

`ds4.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 Flash-specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Status

The code and GGUF files are to be considered of **alpha quality** because
inference and model serving is a complicated matter and all this exists
only for a few days. It will take months to reach a more stable form.
However, we try to keep the project in a usable state, and we are making
progresses. If you have issues, make sure to use `--trace` to log the
sessions, and open issues including the full trace.

## More Documentation

If you are looking for very specific things, we have other
sub-README files. Otherwise for normal usage keep reading the
next sections.

- [CONTRIBUTING.md](CONTRIBUTING.md): correctness and speed regression testing
  guide for contributors. **Read this before sending a pull request**.
- [gguf-tools/README.md](gguf-tools/README.md): offline GGUF generation,
  imatrix collection, quantization tooling, and quality checks.
- [gguf-tools/imatrix/README.md](gguf-tools/imatrix/README.md): how the
  routed-MoE imatrix is collected and used.
- [gguf-tools/imatrix/dataset/README.md](gguf-tools/imatrix/dataset/README.md):
  how the calibration prompt corpus is generated.
- [gguf-tools/quality-testing/README.md](gguf-tools/quality-testing/README.md):
  how local GGUFs are scored against official DeepSeek V4 Flash continuations.
- [dir-steering/README.md](dir-steering/README.md): directional steering data,
  vector generation, and usage.
- [speed-bench/README.md](speed-bench/README.md): benchmark charts, Metal
  Tensor candidate gates, drift checks, comparator probes, and local artifact
  indexing.
- [tests/test-vectors/README.md](tests/test-vectors/README.md): official
  continuation vectors used for regression checks.

## Model Weights

This implementation only works with the DeepSeek V4 Flash GGUFs published for
this project. It is not a general GGUF loader, and arbitrary DeepSeek/GGUF files
will not have the tensor layout, quantization mix, metadata, or optional MTP
state expected by the engine. The 2 bit quantizations provided here are not
a joke: they behave well, work under coding agents, call tools in a reliable way.
The 2 bit quants use a very asymmetrical quantization: only the routed MoE
experts are quantized, up/gate at `IQ2_XXS`, down at `Q2_K`. They are the
majority of all the model space: the other components (shared experts,
projections, routing) are left untouched to guarantee quality.

Download one main model. **Prefer the imatrix versions.**

```sh
./download_model.sh q2-imatrix   # 96/128 GB RAM machines, imatrix-tuned q2
./download_model.sh q4-imatrix   # >= 256 GB RAM machines, imatrix-tuned q4
```

Legacy GGUF files are still available if you specifically need the older
non-imatrix quants:

```sh
./download_model.sh q2           # 96/128 GB RAM machines, legacy non-imatrix
./download_model.sh q4           # >= 256 GB RAM machines, legacy non-imatrix
```

The script downloads from `https://huggingface.co/antirez/deepseek-v4-gguf`,
stores files under `./gguf/`, resumes partial downloads with `curl -C -`, and
updates `./ds4flash.gguf` to point at the selected q2-imatrix/q4-imatrix/q2/q4
model. The plain q2 XXS weights are produced with the weights importance vector
only, without an imatrix. The imatrix variants are preferred.
Authentication is optional for public downloads, but `--token TOKEN`,
`HF_TOKEN`, or the local Hugging Face token cache are used when present.

If you want to regenerate GGUF files or collect a new imatrix, see
[gguf-tools/README.md](gguf-tools/README.md). Those tools are meant for offline
model-building work and can take a long time on the full DeepSeek V4 Flash
weights.

`./download_model.sh mtp` fetches the optional speculative decoding support
GGUF. It can be used with q2-imatrix, q4-imatrix, q2, and q4, but must be
enabled explicitly with `--mtp`. The current MTP/speculative decoding path is
still experimental: it is correctness-gated and currently provides at most a
slight speedup, not a meaningful generation-speed win.

Then build:

```sh
make                  # macOS Metal
make cuda-spark       # Linux CUDA, DGX Spark / GB10
make cuda-generic     # Linux CUDA, other local CUDA GPUs
make cpu              # CPU-only diagnostics build
```

`./ds4flash.gguf` is the default model path used by both binaries. Pass `-m` to
select another supported GGUF from `./gguf/`. Run `./ds4 --help` and
`./ds4-server --help` for the full flag list.

## Speed

These are single-run Metal CLI numbers with `--ctx 32768`, `--nothink`, greedy
decoding, and `-n 256`. The short prompt is a normal small Italian story
prompt. The long prompts exercise chunked prefill plus long-context decode.
Q4 requires the larger-memory machine class, so M3 Max Q4 numbers are `N/A`.

| Machine | Quant | Prompt | Prefill | Generation |
| --- | ---: | ---: | ---: | ---: |
| MacBook Pro M3 Max, 128 GB | q2 | short | 58.52 t/s | 26.68 t/s |
| MacBook Pro M3 Max, 128 GB | q2 | 11709 tokens | 250.11 t/s | 21.47 t/s |
| MacBook Pro M3 Max, 128 GB | q4 | short | N/A | N/A |
| MacBook Pro M3 Max, 128 GB | q4 | long | N/A | N/A |
| MacBook Pro M5 Max, 128 GB | q2 | short | 87.25 t/s | 34.27 t/s |
| MacBook Pro M5 Max, 128 GB | q2 | 11707 tokens | 463.44 t/s | 25.90 t/s |
| Mac Studio M3 Ultra, 512 GB | q2 | short | 84.43 t/s | 36.86 t/s |
| Mac Studio M3 Ultra, 512 GB | q2 | 11709 tokens | 468.03 t/s | 27.39 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | short | 78.95 t/s | 35.50 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | 12018 tokens | 448.82 t/s | 26.62 t/s |
| DGX Spark GB10, 128 GB | q2 | 7047 tokens | 343.81 t/s | 13.75 t/s |

![M3 Max t/s](speed-bench/m3_max_ts.svg)

## Benchmarking

`ds4-bench` measures instantaneous prefill and generation throughput at context
frontiers instead of reporting one whole-run average. It loads the model once,
walks a fixed token sequence to frontiers such as 2048, 4096, 6144, and uses
incremental prefill so each row measures only the newly-added token interval.
After each frontier it saves the live KV state to memory, generates a fixed
greedy non-EOS probe, restores the memory snapshot, and continues prefill.

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

The example file is a cleaned public-domain Project Gutenberg text of
Alessandro Manzoni's *I Promessi Sposi* (ebook #45334), with the Gutenberg
header and footer removed: <https://www.gutenberg.org/ebooks/45334>.

Use `--step-incr N` for different linear spacing, or `--step-mul F` for
exponential sweeps. Output is CSV with one row per frontier: latest prefill
interval tokens/sec, generation tokens/sec at that frontier, and
`kvcache_bytes`.

Sessions prefill long prompts in 4096-token chunks by default. Set
`DS4_METAL_PREFILL_CHUNK=N` to compare another chunk size, for example `2048`
to match the strict official-vector checkpoint path, or
`DS4_METAL_PREFILL_CHUNK=0` to prefill a prompt as one whole batch when memory
allows. Changing the chunk changes the KV checkpoint/logit path, so compare it
as an explicit run configuration.
Chunked Metal prefill reuses the same range-capable layer-major graph for each
chunk, preserving absolute compressor/indexer boundaries while avoiding the old
per-layer chunk dispatch path.

## Capability Evaluation

`ds4-eval` is a small real-model integration benchmark. It is not a leaderboard
runner and should not be reported as an official GPQA, SuperGPQA, AIME, or
security benchmark score: the questions are an embedded 92-item subset chosen
to make local regression testing useful and visually inspectable. The program
loads the real GGUF,
renders DS4 chat prompts, streams sampled tokens in a split-screen TUI, grades
the final answer, and prints a per-question report with prompt tokens,
generated tokens, pass/fail state, the model answer, and the correct answer.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
```

The default run uses `--tokens 16000`, thinking mode enabled, and a soft/hard
`</think>` budget cutoff so the model has room to produce a visible answer.
`ds4-eval` sizes the context internally from the largest selected prompt plus
the generation budget, and refuses runs that would need more than 1M context
tokens. Press `p` to pause, `q` to exit and print the report, Up/Down to
inspect or select another question, and Enter to run the selected question next.
`--plain` disables the TUI.

The first 75 embedded questions are interleaved as 25 GPQA Diamond, 25 audited
SuperGPQA, and 25 AIME 2025 problems. The final 17 are an audited COMPSEC
subset of reduced single-function C/C++ vulnerability-localization questions.
The model is asked for the single best source line, or the smallest exact line
set only when the bug cannot be localized to one line; the scorer accepts small
audited ranges only when adjacent lines are equivalent locations for the same
bug. The order is
intentionally progressive: early questions are useful smoke tests, while later
questions are hard enough that a strong reasoning model should still miss some
of them. The SuperGPQA slice is curated rather than blind: upstream rows with
wrong keys, missing figures, or underspecified prompts are replaced with cleaner
rows.

For a model like DeepSeek V4 Flash, the set should be treated as a hard
capability regression suite rather than a pass/fail unit test:

- **GPQA Diamond** contributes graduate-level science questions with
  multiple-choice answers. DeepSeek's model card reports strong Flash results
  on full GPQA Diamond in thinking mode, but individual items still require
  careful physics, chemistry, or biology reasoning and are easy to lose with a
  small prompt/rendering or sampling regression.
- **SuperGPQA** contributes broad specialist knowledge and domain-transfer
  questions. The model-card SuperGPQA number is much lower than GPQA Diamond,
  so these items are expected to be uneven: some look mundane, others require
  niche professional knowledge or exact interpretation of a translated-style
  exam question.
- **AIME 2025** contributes exact-answer contest math. These are often the most
  unforgiving items in the set: no multiple-choice prior, no partial credit, and
  a single arithmetic or algebraic slip changes the grade.
- **COMPSEC** contributes single-function C/C++ security reasoning items
  reduced from public CVE writeups. These are not exploit prompts: the task is
  to identify the best source line where the defensive code flaw is introduced,
  or return `0` for a safe function.

In practice this means `ds4-eval` should not be expected to produce a perfect
92/92 run. It is meant to answer a more useful engineering question: after a
kernel, quantization, prompt-rendering, KV-cache, or tool-streaming change, does
DeepSeek V4 Flash still solve a representative mix of hard science, broad
knowledge, exact math, and security-code problems while using the same inference
path users run?

## Metal 4 and M5 Neural Accelerators

The current production path is still hand-written Metal compute kernels over
`MTLBuffer` storage. That is intentional: DS4's hot path is dominated by
quantized routed-MoE matvec/matmul, sparse compressed attention, and mmap-backed
model views, which do not map cleanly to a whole-model Core ML package.

Metal 4 is the right next target, but it should be introduced as a feature-gated
kernel backend rather than a rewrite. On macOS 26+ with `MTLGPUFamilyMetal4`,
Apple exposes tensor resources, cooperative tensor primitives, and Metal 4
command infrastructure that can run machine-learning work on the same timeline
as compute work. The Apple Neural Engine path is exposed through Metal 4
machine-learning passes over Core ML packages; it is separate from DS4's current
hand-written compute-shader path over mmap-backed GGUF weights. For this branch,
`DS4_METAL_MEMORY_REPORT=1` reports the device, Metal 4 family support, MTL4
queue availability, and whether the device looks like an M5 Neural Accelerator
target, but that diagnostic is not proof that a custom DS4 shader dispatched on
the ANE.

The implementation follows the same conservative shape used by llama.cpp's
current Metal backend: the tensor API is disabled by default on pre-M5/pre-A19
devices, can be forced with `DS4_METAL_TENSOR_ENABLE=1`, and can always be
disabled with `DS4_METAL_TENSOR_DISABLE=1`. At startup ds4 compiles a tiny
Metal Performance Primitives tensor matmul probe before it lets the main Metal
shader source see `DS4_METAL_HAS_TENSOR`, so unsupported SDK/device
combinations fall back to the legacy kernels.

Metal Tensor policy is explicit and guarded. Use `-mt auto` or `--mt auto` for
the default route policy, `-mt on` to force Tensor routes where the Metal tensor
path is available, and `-mt off` for the legacy Metal reference path. The old
`--mpp` spelling remains accepted as a compatibility alias. Auto currently
enables the F16 compressor Tensor path, enables attention-output low Tensor in
all layers, and runs routed-MoE Tensor only in its conservative layer window
while preserving same-top1/same-greedy agreement. The dense Q8_0 prefill path
remains on the legacy hand-written Metal simdgroup kernel; the experimental
Tensor Q8_0 route was removed after M5 drift bisection showed it was the
drift-prone path.

The next prefill optimization target is therefore not a re-enable of the removed
Q8_0 Tensor route. It is a new, isolated quantized prefill matmul experiment
that targets the high-impact routed-MoE and dense-attention shapes with Metal 4
cooperative matrix primitives, while keeping the legacy
dequantization/reduction behavior close enough to pass the five-fixture quality
gate before it can become part of `-mt auto`. Any Apple Neural Engine work
should be a separate Core ML/Metal 4 machine-learning pass investigation; it is
not something the current custom compute shaders get automatically by changing
their matrix instructions.

The environment controls `DS4_METAL_MPP_ENABLE` and
`DS4_METAL_MPP_DISABLE` accept `1/true/yes/on` and `0/false/no/off`;
`DS4_METAL_MPP_ENABLE=0` disables Tensor routes instead of enabling them by mere
presence. Passing `--quality` also disables Tensor routes so strict/debug runs
stay on the legacy Metal kernels. Set `DS4_METAL_MPP_FAST=1` to opt into the
current throughput diagnostic profile: it uses the routed-MoE all-layer
diagnostic window. This profile is not the default because its top-k overlap is
weaker than auto in the current full-model suite.

The default safe-window policy uses the direct-RHS tensor layout for Tensor
routes; set `DS4_METAL_MPP_DIRECT_RHS=0` to compare against the older staged-RHS
layout. Attention-output direct-RHS supports both 32-token and 64-token Tensor
tiles, and auto defaults it to 64-token tiles. Set
`DS4_METAL_MPP_ATTN_OUT_TILE_N=32` to force the narrower layout. The
route-specific `DS4_METAL_MPP_F16_DIRECT_RHS=1` and
`DS4_METAL_MPP_ATTN_OUT_DIRECT_RHS=1` switches isolate that layout without
turning on every direct-RHS route at once when the global
`DS4_METAL_MPP_DIRECT_RHS=0` override is set.

The isolated `./ds4_test --metal-kernels` regression reports
small/medium/model-ish kernel deltas; the full-model
`./ds4_test --metal-tensor-equivalence` diagnostic compares default auto
against `-mt off`. The old `--metal-mpp-equivalence` spelling remains accepted
as a compatibility alias. Set `DS4_TEST_MPP_EQ_FORCE_ON=1` to compare forced
Tensor against `-mt off` while working on a route.
`DS4_TEST_MPP_EQ_CASE=<case-id-substring>` limits the diagnostic to one prompt,
and `DS4_TEST_MPP_EQ_MATRIX=1` prints
separate auto, fast-profile, attention-output-only, MoE gate/up/down-only, and
full-forced summary rows. The equivalence gate requires finite logits, the same
top-1 token, and matching greedy continuation; it also reports top-5/top-20
overlap, top-20 rank displacement, top-20 logit deltas, and whole-vocab RMS/max
drift so route changes can be judged beyond pass/fail.

Full-graph route localization is available with
`DS4_METAL_MPP_COMPARE_ROUTE=attn_out|moe_gate|moe_up|moe_down|flash_attn`
and optional `DS4_METAL_MPP_COMPARE_MAX=N`. The comparator snapshots the
candidate Tensor output, runs the legacy Metal route on the same tensor input,
and reports the first comparison that exceeds the kernel target, including
module/layer context, shape, max absolute error, RMS, and the largest element
deltas. Set `DS4_METAL_MPP_COMPARE_VERBOSE=1` to print passing comparisons as
well.
Set `DS4_METAL_Q8_PREFILL_PROFILE=1` while profiling a prompt to time the
current legacy Q8_0 prefill matmul by module/layer context without changing the
dispatch. Add `DS4_METAL_Q8_PREFILL_PROFILE_FILTER=<substring>` to limit the
rows to dense Q8_0 contexts such as `attn_q_a`, `attn_kv`, or `attn_q_b`.
Set `DS4_METAL_Q8_COMPARE=1` to run a local dense Q8_0 ref-vs-candidate
comparison using the same comparator output format, and
`DS4_METAL_Q8_COMPARE_FILTER=<substring>` to focus it on one context such as
`attn_q_b` or `attn_out`. This is a diagnostic hook for future default-off Q8
kernel prototypes; the current production path still uses the legacy Q8_0
prefill kernel.
Set `DS4_METAL_FLASH_ATTN_COMPARE=1` with
`DS4_METAL_MPP_COMPARE_ROUTE=flash_attn` to compare static-mixed prefill head
outputs against the existing generic masked FlashAttention path. Use
`DS4_METAL_FLASH_ATTN_COMPARE_FILTER=<substring>` to limit the comparison by
shape label before testing a default-off static-mixed attention kernel.
Routed-MoE gate/up/down uses the specialized routed-MoE profiler below instead
of this dense wrapper. Use both profilers to choose the first default-off Metal 4
matmul prototype target; current profile data points first at early routed-MoE
matmuls, then at dense attention `attn_q_b`.

Set `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` to run a default-off routed-MoE
matmul candidate that moves the existing Metal 4 cooperative/tensor MoE matmul
window to the first layer, without changing dense Q8_0 dispatch. This is meant
for timing and drift-gate experiments only. `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=N`
can narrow that candidate before promotion, and the existing MoE route filters,
route disables, comparator, and stage profiler still apply.

Current Tensor route status balances drift with prefill throughput: `auto`
enables F16 compressor, attention-output low projection, and routed-MoE Tensor.
Attention-output low projection is enabled for all layers by default, and
routed-MoE Tensor uses the lower-drift conservative default window: down from
layer 12 and gate/up from layer 15. This gives up some of the all-layer
routed-MoE prefill speedup to avoid the larger drift seen with layer-0
routed-MoE Tensor windows while keeping the dense Q8_0 prefill route on the
legacy kernel. The attention-output low Tensor kernels stage activation tiles
through half to match the legacy Metal matmul input path, which removes the
first attention-output comparator breach. The current auto policy uses
direct-RHS Tensor inputs and 64-token tiles for attention-output low projections.
The F16 compressor route did not introduce measurable drift in the current
prompt set.

The `DS4_METAL_MPP_FAST=1` profile is the measured high-throughput diagnostic
profile under the relaxed same-top1/same-greedy gate. In the current prompt
suite it keeps top-1 and greedy continuations stable, but reports weaker top-k
overlap than auto. It remains diagnostic-only because it widens routed-MoE
Tensor to layer 0, which produces the largest full-suite drift.
The current fastest default-off eval candidate keeps the fast gate/up window but
excludes the largest local `moe_down` comparator outliers:

```
DS4_METAL_MPP_FAST=1 \
DS4_METAL_MPP_MOE_DOWN_FILTER=layer=0-25,layer=27-28,layer=31-42
```

If generation steadiness matters more than maximum short-context prefill, add
`DS4_METAL_MOE_MID_F32=1` to the same env. That balanced variant still passes
the five-fixture drift gate, keeps the same Tensor-vs-standard drift summary,
and reduces the compact-generation timing swings seen in the fastest variant.
In the 128-token long sweep it remains prefill-positive through 65k context,
but gives up the strongest long-context prefill gains and has a -2.7%
generation point at 65k. Neither variant is promoted to the default policy; use
them only for explicit eval runs.

The routed-MoE Tensor projections are enabled by default from layer 12 for down
and layer 15 for gate/up. For route isolation, use
`DS4_METAL_MPP_MOE_GATE_ENABLE/DISABLE`,
`DS4_METAL_MPP_MOE_UP_ENABLE/DISABLE`, and
`DS4_METAL_MPP_MOE_DOWN_ENABLE/DISABLE`; `DS4_METAL_MPP_MOE_DISABLE=1`
disables all routed-MoE Tensor projections. Set the common
`DS4_METAL_MPP_MOE_FILTER` or route-specific
`DS4_METAL_MPP_MOE_GATE_FILTER`, `DS4_METAL_MPP_MOE_UP_FILTER`, and
`DS4_METAL_MPP_MOE_DOWN_FILTER` to `all`, `late_safe`, `none`, or
comma-separated full-graph context substrings to localize safe layer windows.
Use `layer=N` for an exact layer match or `layer=A..B` for an inclusive layer
range when testing sparse Tensor windows. The same `<substring>@layer=A..B`
syntax can restrict a context substring to a layer window.
Set `DS4_METAL_MOE_STAGE_PROFILE=1` to split routed-MoE prefill into timed
`map`, `gate`, `up`, `gate_up_pair`, `activation_weight`, `down`, and `sum`
stages. Add `DS4_METAL_MOE_STAGE_PROFILE_FILTER=<substring>` to print only
matching stages or layer context while still flushing every stage for correct
timing.
Set `DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1` to split prefill FlashAttention into
copy, mask, block-map, pad, attention, and reduce stages; add
`DS4_METAL_FLASH_ATTN_STAGE_PROFILE_FILTER=<substring>` to limit printed rows
while still flushing every stage.
Set `DS4_METAL_MPP_MOE_TILE_N=64` to test the experimental wider routed-MoE
Tensor token tile for performance against the default `32`. The routed-MoE
Tensor path uses the faster first-PR threadgroup tensor layout by default inside
the active routed-MoE windows; set `DS4_METAL_MPP_MOE_FAST_LAYOUT=0` to compare
against the newer staged layout. Set
`DS4_METAL_MPP_MOE_START_LAYER=N`, or the route-specific
`DS4_METAL_MPP_MOE_GATE_START_LAYER`,
`DS4_METAL_MPP_MOE_UP_START_LAYER`, and
`DS4_METAL_MPP_MOE_DOWN_START_LAYER`, to test routed-MoE Tensor start layers; the
resolved start layer also defines the route's default `late_safe` filter. Set
`DS4_METAL_MPP_MOE_PAIR_GATE_UP=1` only to profile the experimental fused
gate/up Tensor dispatch; it passes the current equivalence gate but is not a
default path because it is slower than separate gate and up dispatches.

For the common six-routed-expert prefill shape, the down-projection expert
outputs are summed with a single Metal kernel instead of five chained add
passes. Set `DS4_METAL_MOE_SUM6_DISABLE=1` to compare or temporarily disable
that fused sum route.

Long-context decode uses the indexed mixed-attention kernel once ratio-4
compressed rows exceed the dense-attention window. The default decode
specialization stages sixteen selected rows per threadgroup block; set
`DS4_METAL_INDEXED_ATTN_RB4=1` to compare the older four-row staging variant.
Set `DS4_METAL_DECODE_INDEXER_TOP_K=64`, `128`, `256`, or `512` to cap the
decode indexer candidate count for speed/quality diagnostics. The normal
non-quality decode path keeps the legacy dense-attention window until there are
more than `1024` compressed rows, then selects `256` rows in sparse indexed
attention. Set `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` to `64`, `128`,
`256`, `512`, `1024`, `2048`, or `4096` to tune the sparse-decode crossover
separately. `--quality` keeps the full `512` candidate path unless this
environment override is set explicitly.

The attention-output low-projection Tensor route applies to full 32-token
multiples in all layers by default, using a 64-token Tensor tile by default and
falling back to the existing indexed simdgroup kernel for shorter or
non-32-multiple tails. Set
`DS4_METAL_MPP_ATTN_OUT_ENABLE=1` or `DS4_METAL_MPP_ATTN_OUT_DISABLE=1` to
isolate this route. Set `DS4_METAL_MPP_ATTN_OUT_FILTER=all`, `late_safe`,
`none`, or a comma-separated list of full-graph context substrings such as
`layer=42` to localize layer windows; `late_safe` keeps the old 32..42 default
window for comparison. Layer filters are exact, and `layer=A..B` matches an
inclusive range. Set
`DS4_METAL_MPP_ATTN_OUT_TILE_N=32` to compare against the narrower Tensor token
tile.
The ratio-2 F16 compressor route can similarly be controlled with
`DS4_METAL_MPP_F16_ENABLE=1` or `DS4_METAL_MPP_F16_DISABLE=1`.
`DS4_METAL_MPP_F16_PAIR=1` tests a paired KV/gate compressor dispatch that keeps
the standard simdgroup F16 matmul accumulation shape. It passes the current
full-model equivalence gate, but the measured long-code prefill change was
within noise (`~0.4%`), so it remains opt-in. `DS4_METAL_MPP_F16_WIDE=1` tests
wider 512/1024-column compressor Tensor, including the paired Tensor route when both
variables are set. The wide route is diagnostic only: the current long-code
prompt fails full-model equivalence with wide F16 Tensor (`rms ~= 0.569`,
`top20_max_abs ~= 1.48`), so it is not enabled by `auto`.

## CLI

One-shot prompt:

```sh
./ds4 -p "Explain Redis streams in one paragraph."
```

No `-p` starts the interactive prompt:

```sh
./ds4
ds4>
```

The interactive CLI is a real multi-turn DS4 chat. It keeps the rendered chat
transcript and the live graph KV checkpoint, so each turn extends the previous
conversation. Useful commands are `/help`, `/think`, `/think-max`, `/nothink`,
`/ctx N`, `/read FILE`, and `/quit`. Ctrl+C interrupts the current generation
and returns to `ds4>`.

The CLI defaults to thinking mode. Use `/nothink` or `--nothink` for direct
answers. `--mtp MTP.gguf --mtp-draft 2` enables the optional MTP speculative
path; it is useful only for greedy decoding, currently uses a confidence gate
(`--mtp-margin`) to avoid slow partial accepts, and should be treated as an
experimental slight-speedup path.

## Server

Start a local OpenAI/Anthropic-compatible server:

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Use `--chdir /path/to/ds4` when launching `ds4-server` from another directory,
so relative runtime files such as `metal/*.metal` resolve from the project tree.

The server keeps one mutable backend/KV checkpoint in memory,
so stateless clients that resend a longer version of the same prompt can reuse
the shared prefix instead of pre-filling from token zero.

Request parsing and sockets run in client threads, but inference itself is
serialized through one graph worker. The current server does not batch multiple
independent requests together; concurrent requests wait their turn on the single
live graph/session.

Supported endpoints:

- `GET /v1/models`
- `GET /v1/models/deepseek-v4-flash`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/messages`

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
nucleus mass. In thinking mode DS4 uses those fixed sampling defaults and
ignores client sampling knobs, matching DeepSeek's fixed-thinking API behavior.

The chat, Responses, and Anthropic endpoints support SSE streaming. In thinking
mode, reasoning is streamed in the native API shape instead of being mixed into
final text. OpenAI chat streaming
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
headers; it does not expose the server on the LAN. Use `--host 0.0.0.0`
explicitly when remote machines should be able to connect.

### Tool call handling and canonicalization

DeepSeek V4 Flash emits tool calls as [DSML text](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/README.md). Agent clients do not send that
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

`ds4-server` can be used by local coding agents that speak OpenAI-compatible
chat completions. Start the server first, and set the client context limit no
higher than the `--ctx` value you started the server with:

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

You can use larger context and larger cache if you wish. Full context of
1M tokens is going to use more or less 26GB of memory (compressed indexer
alone will be like 22GB), so configure a context which makes sense in
your system. With 128GB of RAM you would run the 2-bit quants, which are
already 81GB, 26GB are going to be likely too much, so a context window
of 100~300k tokens is wiser. However users reported being able to run 2bit
quants with 250k ctx window in a Macs with just 96GB of system memory: make sure
to kill processes that use too much memory, if you plan doing so ;)

The `384000` output limit below avoids token caps since the model is able
to generate very long replies otherwise (up to 384k tokens). The server
still stops when the configured context window is full.

For **opencode**, add a provider and agent entry to
`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ds4": {
      "name": "ds4.c (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "dsv4-local"
      },
      "models": {
        "deepseek-v4-flash": {
          "name": "DeepSeek V4 Flash (ds4.c local)",
          "limit": {
            "context": 100000,
            "output": 384000
          }
        }
      }
    }
  },
  "agent": {
    "ds4": {
      "description": "DeepSeek V4 Flash served by local ds4-server",
      "model": "ds4/deepseek-v4-flash",
      "temperature": 0
    }
  }
}
```

For **Pi**, add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "ds4": {
      "name": "ds4.c local",
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
          "name": "DeepSeek V4 Flash (ds4.c local)",
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
  "defaultProvider": "ds4",
  "defaultModel": "deepseek-v4-flash"
}
```

For **Codex CLI**, use the Responses wire API:

```toml
[model_providers.ds4]
name = "DS4"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1000000
```

Then run:

```sh
codex --model deepseek-v4-flash -c model_provider=ds4
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
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="DeepSeek V4 Flash local ds4"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="ds4.c local GGUF"

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
starts doing useful work. Keep `--kv-disk-dir` enabled: after the first expensive
prefill, the disk KV cache lets later continuations or restarted sessions reuse
the saved prefix instead of processing the whole prompt again.

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
conversation every request. `ds4-server` first tries the cheap exact token-prefix
check, then falls back to comparing rendered prompt bytes with decoded
checkpoint bytes. The live in-memory checkpoint covers the current session; the
disk KV cache makes useful prefixes survive session switches and server
restarts.

For RAM reasons there is currently only one live KV cache in memory. When a new
unrelated session replaces it, the old checkpoint can only be resumed without
re-processing if it was written to the disk KV cache. In other words, memory
cache handles the active session; disk cache is the resume mechanism for
different sessions.

Enable it with:

```sh
./ds4-server --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

The cache key is the SHA1 of the rendered byte prefix, and files are named
`<sha1>.kv`. The DS4 payload still stores the exact token IDs and graph state
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
DS4 session payload, payload_bytes from the KVC header
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
40  u64    DS4 session payload byte count
```

The rendered text is the tokenizer-decoded text for the cached token prefix.
It is both the human-inspectable prefix and the lookup identity: its SHA1 is
the filename, and a file is reusable only when those bytes are a prefix of the
incoming rendered prompt. After load, the exact checkpoint tokens from the DS4
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

The DS4 session payload starts with thirteen little-endian `u32` fields:

```text
0   magic = "DSV4"
1   payload version = 1
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

The logits are raw IEEE-754 `float32` values from the host `ds4_session`
buffer. They are saved immediately after the checkpoint tokens so a loaded
snapshot can sample or continue from the exact next-token distribution without
running one extra decode step. MTP draft logits/state are not persisted; after
loading a disk checkpoint the draft state is invalidated and rebuilt by normal
generation.

The tensor payload is DS4-specific KV/session state, not a generic inference
graph dump. It is expected to be portable only across compatible `ds4.c`
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

## Backends

The default graph backend is Metal on macOS and CUDA in CUDA builds:

```sh
./ds4 -p "Hello" --metal
./ds4 -p "Hello" --cuda
```

On Linux, plain `make` prints the available build targets instead of selecting a
CUDA target implicitly. Use `make cuda-spark` for DGX Spark / GB10. It omits an
explicit `nvcc -arch` because that is currently the fastest path on GB10. Use
`make cuda-generic` for a normal local CUDA build, or set `CUDA_ARCH` explicitly
when cross-building or when you need a known target:

```sh
make cuda CUDA_ARCH=sm_120
make cuda CUDA_ARCH=native
```

There is also a CPU reference/debug path:

```sh
./ds4 -p "Hello" --cpu
make cpu
./ds4
./ds4 -p "Hello"
```

Do not treat the CPU path as the production target. The CLI and `ds4-server`
support the CPU backend for reference/debug use and share the same KV session
and snapshot format as Metal and CUDA, but normal inference should use Metal or
CUDA.

## Steering

This project supports steering with single-vector activation directions; see the
`dir-steering` directory for more information. This follows the core idea of the
[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
paper. You can use it to make the model more or less verbose, less likely to
answer programming questions if it is a chatbot for your car rental web site,
and so forth, much faster than fine-tuning.
This is also useful for cybersecurity researchers who want to reduce a model's
willingness to provide dual-use or offensive security guidance.

## Test Vectors

`tests/test-vectors` contains short and long-context continuation vectors
captured from the official DeepSeek V4 Flash API. The requests use
`deepseek-v4-flash`, greedy decoding, thinking disabled, and the maximum
`top_logprobs` slice exposed by the API. Local vectors are generated with
`./ds4 --dump-logprobs` and compared by token bytes, so tokenizer/template or
attention regressions show up before they become long generation failures. The
C runner uses the standard Metal path and pins `DS4_METAL_PREFILL_CHUNK=2048`
for this strict API-vector comparison; Tensor route drift is checked separately
by `--metal-tensor-equivalence` and the five-fixture drift gate.

All project tests are driven by the C runner:

```sh
make test                  # ./ds4_test --all
./ds4_test --logprob-vectors
./ds4_test --metal-tensor-equivalence
./ds4_test --server
```

## Debugging Notes

When a generation looks wrong, three small tools are usually enough to get a
first answer:

```sh
./ds4 --dump-tokens -p "..."
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ds4 --dump-logits /tmp/q2-off.json --metal -mt off --nothink --prompt-file prompt.txt
python3 speed-bench/compare_logit_drift.py /tmp/q2-off.json /tmp/q2-mt.json /tmp/q4-off.json --labels q2_mt q4_off
./ds4-server --trace /tmp/ds4-trace.txt ...
```

- `--dump-tokens` tokenizes the `-p` or `--prompt-file` string exactly as
  written, recognizes DS4 protocol specials, and then exits before inference
  starts. For example, the DSML tool close marker starts as two tokens: `</`
  and `｜DSML｜`.
- `--dump-logprobs` stores a greedy continuation with the top local
  alternatives at each step, which helps separate sampling choices from
  logit/model issues.
- `ds4-server --trace` writes the rendered prompts, cache decisions, generated
  text, and tool-parser events for a whole agent session.
