<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="220">
</p>

**DwarfStar** is a small native inference engine optimized first for
**DeepSeek V4 Flash**. It also supports **GLM 5.2** and, on very high-memory
machines, **DeepSeek V4 PRO**. It is self-contained and deliberately narrow,
not a general GGUF runner. Model loading, prompt rendering, tool calls, KV
state, the HTTP server, and the coding agent are built and tested together.
The repository also includes tools and data for GGUF, imatrix, quality, and speed.

Supported backends:

* **Metal**, the primary target, on Macs with 96 GB or more. Smaller machines
  can use SSD streaming.
* **NVIDIA CUDA**, including multi-GPU systems and DGX Spark.
* **ROCm** on Strix Halo systems such as the Framework Desktop.

This project would not exist without **llama.cpp and GGML**, make sure to read
the acknowledgements section, a big thank you to Georgi Gerganov and all the
other contributors.

Model support is intentionally opportunistic. The project follows the best open
weights for useful local machine sizes, especially 128 GB laptops and 512 GB
workstations. A model may be removed when a better replacement arrives.

# So, what can I do with this software?

* You can run a very capable models in your consumer hardware, a MacBook, a DGX Spark, or a Strix Halo for example. Even if you have not enough RAM, with SSD streaming, you can run it at a decent speed.
* Using the CUDA multi-GPU support and with ds4-server micro batching of decoding and generation, you can turn a server with old-ish CUDA cards (Ada Lovelace architecture), no longer supported for new models by vLLM, into a multi-user LLM server for your company. We tested this setup with 8xL40S NVIDIA cards and multiple sessions with very good results. 120 t/s aggreated generation, 2000 t/s prefill.
* Using two MacBook M5 Max / M3 Ultra RDMA, you can run 4 bit DeepSeek Flash or GLM 5.2 with tensor parallelism.
* You can also use pipeline paralellism to glue together multiple systems to sum their RAM and run larger models.

## Motivations

* Capable open-weight models now fit on high-end personal machines.
* DeepSeek V4 Flash and PRO, GLM 5.2, tolerate aggressive routed-expert quantization.
* Compressed KV caches and fast local SSDs make long contexts practical.
* The idea of an inference system specialized for a few models.

# AI full disclosure

* This software is developed with **strong assistance from GPT 5.5, 5.6, Claude Fable** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.

## Acknowledgements to llama.cpp and GGML

`ds4.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Status

The software is currently very fast changing. Consider it beta quality.
Before each release, a big QA run is executed, however instabilities
are definitely possible.

## More Documentation

If you are looking for very specific things, we have other
sub-README files. Otherwise for normal usage keep reading the
next sections.

- [CONTRIBUTING.md](CONTRIBUTING.md): correctness and speed regression testing
  guide for contributors. **Read this before sending a pull request**.
- [QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md): the complete release test
  matrix, including the remote Metal, CUDA, and ROCm machines.
- [gguf-tools/README.md](gguf-tools/README.md): offline GGUF generation,
  imatrix collection, quantization tooling, and quality checks.
- [gguf-tools/imatrix/README.md](gguf-tools/imatrix/README.md): how the
  routed-MoE imatrix is collected and used.
- [gguf-tools/imatrix/dataset/README.md](gguf-tools/imatrix/dataset/README.md):
  how the calibration prompt corpus is generated.
- [gguf-tools/quality-testing/README.md](gguf-tools/quality-testing/README.md):
  how local GGUFs are scored against official DeepSeek V4 Flash/PRO continuations.
- [dir-steering/README.md](dir-steering/README.md): directional steering data,
  vector generation, and usage.
- [speed-bench/README.md](speed-bench/README.md): benchmark commands, charts,
  and CSV generation.
- [tests/test-vectors/README.md](tests/test-vectors/README.md): official
  continuation vectors used for regression checks.

## Model Weights

This implementation only works with the DeepSeek V4 and GLM 5.2 GGUFs listed
below. It is not a general GGUF loader, and arbitrary GGUF files will not have
the tensor layout, quantization mix, metadata, or optional MTP state expected by
the engine. The 2 bit quantizations provided here are verified to be actually
high quality: they behave well, work under coding agents, call tools in a reliable way.

The 2 bit quants use a very asymmetrical quantization: only the routed MoE
experts are quantized, up/gate at `IQ2_XXS`, down at `Q2_K`. They are the
majority of all the model space: the other components (shared experts,
projections, routing) are left untouched to guarantee quality.

Download one main model. **Prefer the imatrix versions.**

```sh
./download_model.sh q2-imatrix   # 96/128 GB RAM machines, imatrix-tuned q2
./download_model.sh q2-q4-imatrix  # 96/128 GB RAM machines, q2 with last 6 layers q4
./download_model.sh q4-imatrix   # >= 256 GB RAM machines, imatrix-tuned q4
./download_model.sh pro-q2-imatrix  # 512 GB RAM machines, PRO q2 imatrix quant
```

For the full PRO Q4 distributed run, download one half on each machine:

```sh
./download_model.sh pro-q4-layers00-30      # first half of PRO Q4 split
./download_model.sh pro-q4-layers31-output  # second half of PRO Q4 split
```

The script downloads from `https://huggingface.co/antirez/deepseek-v4-gguf`,
stores files under `./gguf/`, resumes partial downloads with `curl -C -`, and
updates `./ds4flash.gguf` to point at the selected main model.
The `pro-q4-layers00-30`, `pro-q4-layers31-output`, and `pro-q4-split` targets
download distributed PRO Q4 pieces and do not update `./ds4flash.gguf`.
Authentication is optional for public downloads, but `--token TOKEN`,
`HF_TOKEN`, or the local Hugging Face token cache are used when present.

If you want to regenerate GGUF files or collect a new imatrix, see
[gguf-tools/README.md](gguf-tools/README.md). Those tools are meant for offline
model-building work and can take a long time on the full DeepSeek V4 Flash
weights. Flash GGUF generation is supported by the local tools. PRO GGUF
production currently still depends on the external `llama.cpp`-based workflow;
native tooling can be added later.

`./download_model.sh mtp` fetches the optional speculative decoding support
GGUF for Flash. It can be used with q2-imatrix, q2-q4-imatrix, and q4-imatrix,
but must be enabled explicitly with `--mtp`. The current MTP/speculative
decoding path is still experimental: it is correctness-gated and currently
provides at most a slight speedup, not a meaningful generation-speed win.

GLM 5.2 support is limited to the GGUF files tested by this branch:

```sh
./download_model.sh glm-unsloth-q4  # Unsloth UD-Q4_K_XL, 11 shards
./download_model.sh glm-antirez-iq2xxs  # antirez routed IQ2_XXS single-file GGUF
./download_model.sh glm-antirez-q2  # antirez routed Q2_K single-file GGUF
./download_model.sh glm-antirez-q4  # antirez routed Q4_K single-file GGUF
```

The supported GLM layout keeps dense/model-control tensors in the existing
Q8/F32 paths and supports routed expert gate/up tensors in `Q2_K`, `Q4_K`, or
`Q5_K`; routed expert down tensors are supported in `Q2_K`, `Q4_K`, `Q5_K`, or
`Q6_K`. Other GLM GGUF quant layouts should be treated as unsupported until they
are added deliberately and scored against the official 100-case fixture.

These formats do not all support the same execution modes. The Q4 files work
for normal Metal and CUDA inference. Two-Mac tensor parallelism currently
requires an ownership-aware IQ2_XXS or Q2_K routed layout; a routed Q4 GLM
must be rejected before evaluation.

GLM's MTP block is part of the main GGUF; it does not use the separate Flash
MTP file. Ordinary decode remains the default. `--glm-mtp` enables experimental
greedy speculation. `--glm-mtp-timing` also enables it and prints acceptance
and timing counters:

```sh
./ds4 -m gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --glm-mtp-timing --temp 0
```

GLM inference uses the Metal, CUDA, or ROCm graph backend. Directional steering,
`--power` below 100, an explicit `--prefill-chunk`, and the external `--mtp`
file are not supported for GLM yet.

Then build:

```sh
make                  # macOS Metal
make cuda-spark       # Linux CUDA, DGX Spark / GB10
make cuda-generic     # Linux CUDA, other local CUDA GPUs
make strix-halo       # Linux ROCm, AMD Strix Halo
make cpu              # CPU-only diagnostics build
```

`./ds4flash.gguf` is the default model path used by both binaries. Pass `-m` to
select another supported GGUF from `./gguf/`. Run `./ds4 --help` and
`./ds4-server --help` for the full flag list.

## DSpark Speculative Decoding

DSpark is an auxiliary draft model released by DeepSeek for DeepSeek V4 Flash.
It reads hidden states from the main model and proposes up to five future
tokens. DwarfStar checks those proposals with the main Flash model and commits
only the accepted prefix. The main model remains authoritative; a rejected or
low-confidence suffix falls back to ordinary target decoding.

The possible gain is faster generation: when several proposed tokens are
accepted, one target verification pass advances the stream by several tokens.
It does not accelerate prefill, and the draft and verification work is not
free. Predictable continuations, especially code, tend to benefit most;
low-yield prompts can be no faster or even slower. DSpark is therefore still
experimental and explicitly opt-in.

The released DSpark checkpoint is packaged here as a separate support GGUF of
about 5.6 GiB. It is not a standalone model. Download it once:

```sh
./download_model.sh dspark-support
```

The same support file can be used with the Flash `q2-imatrix`,
`q2-q4-imatrix`, and `q4-imatrix` models listed above. For now **DeepSeek
V4 PRO** is not supported. On Metal, the main model may be resident or use
`--ssd-streaming`; the support model still adds its own weights and runtime
state to the memory requirement. DSpark replaces the legacy one-stage MTP
support model for that run rather than stacking with it.

Run it with greedy decoding:

```sh
./ds4 -m ds4flash.gguf \
  --mtp gguf/DeepSeek-V4-Flash-DSpark-support.gguf \
  --dspark --temp 0
```

`--mtp` supplies the support GGUF, while `--dspark` selects the DSpark runtime.
The default confidence threshold is `0.9`; it prunes suffixes that are unlikely
to repay their verification cost. `--dspark-confidence 0` forces fixed
five-token blocks and is intended for diagnostics. Sampled decoding does not
use DSpark proposals. `--quality` and `--dspark-strict` also keep target-only
decoding, which is useful for comparisons and correctness checks.

## Speed

*Warning: some of those numbers may no longer be updated, because of the optimization
efforts that improved the runtime speed without updating the benchmark
results.*

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
| Mac Studio M3 Ultra, 512 GB | PRO q2 | 32768 tokens | 138.82 t/s | 9.56 t/s |
| DGX Spark GB10, 128 GB | q2 | 7047 tokens | 343.81 t/s | 13.75 t/s |

![M3 Max t/s](speed-bench/m3_max_ts.svg)
![PRO model M3 Ultra t/s](speed-bench/pro_model_m3_ultra_ts.svg)

## Running models larger than RAM

The normal Metal path tries to make the model resident in GPU-addressable
memory. This is the fastest path and should remain your default when the model
fits. DwarfStar also has an **SSD streaming** capacity mode on Metal and for
GLM 5.2 on ROCm. In this mode the non-routed model weights stay resident, while
routed MoE experts are kept in an in-memory cache and loaded from the GGUF file
on cache misses.

Streaming is not as fast as fitting the full model in RAM. It still needs memory
for non-routed weights, KV cache, graph scratch, activations, and the routed
expert cache. It is useful because routed experts dominate model size and modern
Mac SSDs are fast enough to make cache misses tolerable. Long prefills can still
be fast; generation is more sensitive to cache misses because every new token
routes through experts again.

Start with the automatic cache budget:

```sh
./ds4 -m ./ds4flash.gguf --ssd-streaming
```

If startup reports that the expert cache is too large, or if you want to reserve
more memory for context, set the routed expert cache explicitly:

```sh
./ds4 -m ./ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 32GB
```

The `32GB` value is a routed-expert memory budget, not a generic byte cache.
DwarfStar first reserves headroom for the two full routed layers used by
overlapped streaming prefill, then converts the remaining bytes to the number of
dynamic cached experts that fit for the current GGUF. Explicit `NGB` budgets may
also be capped after context/KV accounting so the backend working set stays out of
the slow pressure zone. A plain number such as
`--ssd-streaming-cache-experts 4000` is different: it means exactly 4000 dynamic
expert slots, with no extra accounting. Non-routed weights, KV cache, graph
scratch, and activations need additional memory. The automatic cache budget takes
80% of the backend's recommended working set, subtracts non-routed weights, then
applies the same routed-prefill headroom before sizing the dynamic cache. Leave
the hot expert preload enabled for normal use; use `--ssd-streaming-cold` and
`--ssd-streaming-preload-experts N` only for measurements.

### Practical SSD streaming examples

On 64GB MacBooks, start with the 2-bit Flash GGUF and a moderate expert cache:

```sh
./download_model.sh q2-imatrix

./ds4 \
  -m ./ds4flash.gguf \
  --ssd-streaming \
  --ssd-streaming-cache-experts 32GB \
  --ctx 32768 \
  --nothink
```

On 128GB MacBooks, PRO q2 streaming is experimental but usable for inspection
and occasional work when you accept slow generation. Start with `--nothink`:

```sh
./download_model.sh pro-q2-imatrix

./ds4 \
  -m gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf \
  --ssd-streaming \
  --ctx 32768 \
  --nothink
```

On an M5 Max with 128GB of RAM, a short PRO q2 streaming decode benchmark found
the automatic budget best: it selected about `59GB` of routed expert cache.
Manual `64GB` to `75GB` caches were close on that machine. Prefer the automatic
budget; if setting the cache manually on this class of machine, start around
`48GB` to `64GB`, then increase only while the machine remains responsive and
the startup log shows the requested dynamic cache. Once the machine is stable,
re-enable thinking with a conservative generation limit:

```sh
./ds4 \
  -m gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf \
  --ssd-streaming \
  --ctx 32768 \
  --think \
  --tokens 1500
```

GLM 5.2 uses the same option. Its streaming path keeps the largest full-layer
prefix that fits resident, then uses the remaining budget for a dynamic expert
cache. Start with the automatic budget:

```sh
./ds4 \
  -m gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --ssd-streaming \
  --ctx 32768
```

The important startup line is the cache report. Start conservative, then
increase the cache if the machine has headroom.

On a 128GB Strix Halo, use the routed Q2_K model and a 4096-token context as the
starting point. The automatic cache budget leaves room for the GLM graph and KV
state:

```sh
./download_model.sh glm-antirez-q2
make strix-halo
./ds4 --rocm -m gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf \
  --ssd-streaming --ctx 4096
```

## Distributed inference with pipeline parallelism

Pipeline parallelism lets DwarfStar **run a model that is too large for one machine** by
splitting transformer layers across multiple machines. The main example is the
full 4-bit Flash quant across two 128 GB MacBooks: each process maps only its
own layer slice, activations are sent over TCP, and the coordinator keeps normal
CLI/API behavior.

Pipeline parallelism can also **speed up prefill** by
using multiple GPUs at the same time to process different micro-batches at
different layers, like in an assembly line. Only prefill can be accelerated this
way. Generation is purely autoregressive: each token must finish across the
route before the next token can start. The model work is the same as a single
process, plus coordination latency, so distributed generation is slower.

To build an initial mental model, here are the high level concepts:

1. You put the GGUF on every machine, but each one loads just a subset. `--layers` controls which tensors are mapped, so a worker with `--layers 20:output` does not load the earlier layers.
2. Layer ranges are inclusive: `10:20` means layers 10, 11, ..., 20. `N:output` means layer `N` through the final layer plus the output head.
3. You assign one of the machines the role of `coordinator`, the others the roles of `workers`. Workers will connect to the coordinator and will tell they are there and which layers they are able to process.
4. Each worker keeps its slice of the KV cache.
5. Communication is worker-to-worker, there is no need to use the coordinator as relay, so if your coordinator is `A`, and you make a request, activations will flow in `A -> B -> C -> back to A`.

### How it works and how to configure it

The prefill path is pipelined (this is why it can go faster than in a single machine).
For large prompts the coordinator can run its
slice on chunk N+1 while the worker is running its slice on chunk N. The
distributed rows below were measured with two M5 Max 128 GB MacBooks connected
by Thunderbolt 5, using the Q4 Flash GGUF and the default 4096-token
distributed prefill chunk. The single-process column is a reference run with
the Q2 GGUF on a single machine, so it actually is a bit faster since
the routed MoEs are smaller.

| Prompt | Single-process reference | Two MacBooks | Speedup |
| ---: | ---: | ---: | ---: |
| 9421 tokens | 421.70 t/s | 582.22 t/s | 1.38x |
| 28684 tokens | 405.30 t/s | 674.16 t/s | 1.66x |
| 63819 tokens | 353.62 t/s | 654.79 t/s | 1.85x |

Generation is different. **It is strictly autoregressive**: token N+1 cannot start
until token N has produced logits and sampling has selected the next token. That
means distributed generation cannot use the long prefill pipeline. It pays at
least one cross-machine activation hop per generated token, so generation is
slower than a single local process. On the same two-Mac Thunderbolt setup, a
12k-context control run with the 91 GB Flash quant went from 30.59 t/s
single-process to 24.67 t/s distributed, a 19.4% loss. Distributed inference is
therefore mainly for fitting larger models and speeding up long prefills, not
for making decode faster.

### Full DeepSeek V4 PRO Q4 on two Mac Studios

The full-size PRO Q4 GGUF can be run across two 512 GB Mac Studio M3 Ultra
machines by giving the coordinator layers `0:30` and the worker
`31:output`. Use the split GGUF files so each side maps only the tensors it
needs:

```sh
# Coordinator machine.
./download_model.sh pro-q4-layers00-30

# Worker machine.
./download_model.sh pro-q4-layers31-output
```

The two files are:

```text
gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf
gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf
```

This is a capacity use case: each process maps only its own half of the model,
while the worker owns the output head and returns logits.

The current PRO Q4 Metal path uses queue-resident exact expert tables for the
large routed experts. This avoids the broad multi-GiB routed-tensor bindings
that made early distributed PRO Q4 attempts either run very slowly or hit Metal
memory accounting limits. In a short greedy smoke test over the direct
`192.168.0.182` / `192.168.0.183` link, the model generated coherent text and
measured 11.47 t/s generation after startup. Per-token telemetry was balanced:
local layers were around 39-43 ms, remote layers around 44-49 ms, for total
token times around 84-92 ms. Expect a slow startup while each side maps and
makes its half of the model resident. Long-context PRO Q4 prefill and decode
performance still needs separate benchmarking.

The measurements above use a Thunderbolt 5 cable. The implementation is plain
TCP and also works over slower links, including WiFi, but fast Ethernet or
Thunderbolt networking is strongly recommended. Slow links mostly hurt
generation latency and short prefills; large prefills can still benefit when
the layer split is balanced. In the normal performance path, the last worker
owns the output head and returns logits directly.

Minimal two-host configuration:

```sh
# Machine A: coordinator, owns tokenization, sampling, the prompt, and layers 0..30.
./ds4 \
  -m gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf \
  --role coordinator \
  --layers 0:30 \
  --listen 169.254.43.68 1234

# Machine B: worker, connects to A and owns layers 31..output.
./ds4 \
  -m gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf \
  --role worker \
  --layers 31:output \
  --coordinator 169.254.43.68 1234
```

Normally the final worker should own the output head too, for example
`--layers 20:output`. This avoids returning a full final hidden-state batch
after prefill and lets the final worker produce the logits directly. On very
slow or metered links, `--layers 20:42` is also supported: the coordinator will
load the output head and compute logits locally, trading extra coordinator work
for smaller per-token replies.

### Network Link Comparison

The table below shows the same two M5 Max hosts, the same 91 GB Flash quant,
coordinator `--layers 0:19`, worker `--layers 20:output`, an 8192-token prompt
from `speed-bench/promessi_sposi.txt`, and 128 generated tokens. WiFi and
Internet numbers vary with local conditions, but the shape is the important
part: high latency hurts generation directly, while lower bandwidth also pulls
down long-prefill speed.

| Link | Addresses | Ping avg | Prefill | Generation |
| --- | --- | ---: | ---: | ---: |
| Thunderbolt 5 | `169.254.43.68` -> `169.254.12.245` | 0.45 ms | 582.99 t/s | 25.09 t/s |
| WiFi | `192.168.1.57` -> `192.168.1.95` | 77.20 ms | 250.70 t/s | 10.70 t/s |
| Internet / VPN | `10.77.0.4` -> `10.77.0.3` | 152.10 ms | 114.88 t/s | 3.63 t/s |

The Internet/VPN case is not meant to be a good interactive experience. It is
still useful for collective testing: multiple people can temporarily combine
machines to run a larger model that would not fit on any single host, accepting
slow decode in exchange for being able to inspect the model at all.

Use the coordinator exactly like normal `./ds4`: interactive chat, `/read`,
and ordinary generation go through the same high-level session API. The same
distributed options are also wired into `ds4-agent`, `ds4-eval`, and
`ds4-bench`. For benchmarks, workers should already be running; `ds4-bench`
waits until a complete route is available.

Useful tuning and diagnostics:

```sh
./ds4-bench \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 32768 \
  --ctx-max 65536 \
  --step-incr 32768 \
  --gen-tokens 0 \
  --role coordinator \
  --layers 0:19 \
  --listen 169.254.43.68 1234 \
  --debug
```

`--debug` on the coordinator prints route formation and per-hop telemetry:
layer range, token span, local evaluation time, downstream wait time, socket
send time, and input/output byte counts. This is the current profiling tool for
deciding whether a split is balanced. `--dist-prefill-window N` controls how
many prefill chunks may be in flight end-to-end; the default is conservative
and bounded. `--dist-prefill-chunk N` exists for experiments, but the default
4096-token chunk is the canonical setting and should be used unless you are
explicitly validating a different chunk size.

By default DwarfStar sends hidden-state activations as 32-bit floats. To reduce
traffic, pass `--dist-activation-bits 16` or `--dist-activation-bits 8` on the
coordinator. This changes only the transport format between machines, not the
model weights or KV cache. 16-bit transport halves activation traffic and is the
first option to try on Ethernet or WiFi. 8-bit transport is more aggressive and
should be treated as an approximate/experimental mode unless you have validated
the output for your use case. However experimentally reduction activation
size didn't provide a significant improvement, so this option may be removed
in the future.

**If a worker disconnects, the coordinator removes that worker from the active
route**. The request already in flight can fail, and later calls report an
incomplete route until a compatible worker reconnects and sends a new
registration. For live sessions, the coordinator keeps the token history and can
rebuild worker KV state by replaying the prefix when the route is available
again. Workers also validate a rolling 64-bit token-prefix hash on every work
item, so a restarted worker at position 0 cannot silently accept work for
position N; it reports the mismatch and the coordinator replays the current
transcript. Ctrl+C in the CLI and agent is cooperative: DwarfStar waits for the
current distributed token or prefill chunk to drain before returning control,
which avoids coordinator-caused KV splits. Saved agent/server sessions use the
same KV file format as single-machine sessions: during save the coordinator
fetches worker-owned layer tensors and serializes one normal payload; during
load it splits that payload over the currently registered route.

### Distributed protocol overview

At the protocol level there are two kinds of connections. Workers keep a
control TCP connection open to the coordinator and send a `HELLO` with their
model ID, model family, quant profile, layer slice, context capacity, and data
port. The coordinator uses these registrations to build a route that covers all
layers. Work then moves over low-latency TCP data connections: the coordinator
computes the first slice, sends a `WORK` frame with session ID, token positions,
rolling token-prefix hashes before and after the span, route information, and
hidden-state payload, and each worker computes its slice. Middle workers can
forward directly to the next worker. The final worker returns logits to the
coordinator, or ACKs for non-final prefill chunks so the prefill pipeline can
stay full. `RESULT` frames echo the request ID and the post-span hash. A worker
status error is handled differently from a socket failure: KV/hash mismatch can
be recovered by replaying the token history on the same route, while transport
failure drops the route and waits for a replacement worker. For persistent KV,
the coordinator opens worker data connections and sends snapshot save/load
messages for each worker-owned layer range; the disk payload remains a single
agent/server cache file. The protocol has no
encryption or authentication, and is not release-stable yet; coordinator and
workers should be built from the same commit and used on trusted machines and
trusted networks.

## Tensor Parallelism over RDMA

Tensor parallelism runs a single decode across two Macs connected with a
Thunderbolt 5 cable, splitting the heavy per-layer work between the two
GPUs and exchanging 16-24KB partial sums at synchronization gates inside the
graph (RDMA over Thunderbolt when available, a dedicated TCP socket
otherwise). Unlike the pipelined distributed mode above, both
machines work on the *same token at the same time*, so it reduces
per-token latency instead of just fitting a bigger model.

Each machine keeps one contiguous half of the routed experts resident. Dense,
attention, shared-expert, embedding, and output weights remain replicated.
This lets a model whose routed experts do not fit on one machine run fully
resident across the pair; routed kernels never touch the peer's expert half.

### Running GLM 5.2 across two 128 GB MacBooks

One-time setup per boot, on **both** machines:

```sh
# Let the GPU wire ~117 GB (default cap is ~75% of RAM; the resident
# expert shard needs ~97.5 GiB plus KV/scratch).
sudo sysctl iogpu.wired_limit_mb=120000

# RDMA over Thunderbolt needs an IPv4 address directly on the cabled
# member interface (the bridge IP does not count). Use the interface
# that is 'active' in ifconfig, e.g. en1 on one side and en6 on the
# other. Skip this if you are fine with the TCP fallback.
sudo ifconfig en1 inet 10.99.0.2/30 alias     # machine A
sudo ifconfig en6 inet 10.99.0.1/30 alias     # machine B
```

Check the verbs device before loading the model:

```sh
rdma_ctl status
ibv_devinfo -v
```

The device must be active and expose the IPv4-mapped GID for the address above,
for example `::ffff:10.99.0.2`. A working IP ping does not prove that RDMA is
active.

Both machines need the same tree, commit, and GGUF path. Tensor parallelism is
always a 50/50 split with one worker, so do not pass `--layers`. Start the worker
first; it retries while the coordinator loads. The worker must dial the address
on the Thunderbolt member interface, not the bridge address:

```sh
MODEL=gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf

# Machine B: worker.
./ds4 -m "$MODEL" --tensor-parallel --role worker \
  --coordinator 10.99.0.2 9911 --transport rdma

# Machine A: coordinator.
./ds4 -m "$MODEL" --tensor-parallel --role coordinator \
  --listen 10.99.0.2 9911 --transport rdma -c 8192 \
  -p "Tell me something about the sea."
```

The active verbs device and IPv4-mapped GID are selected automatically. If that
is ambiguous, add `--rdma-device rdma_en6 --rdma-gid-index 1` on the worker and
the matching `rdma_en1` flags on the coordinator. Use `--transport tcp` on both
sides to force TCP. Tensor parallel roles are currently exposed by the `ds4`
CLI, not by `ds4-server` or `ds4-agent`.

Startup takes about 9 seconds per machine: each rank pre-faults its
~100 GiB shard from SSD and pins it through a Metal residency set.
DeepSeek V4 Flash works the same way with its own GGUF on both machines.
DeepSeek gate vectors are 16 KB and ride as one RDMA message. GLM's
6144-wide 24 KB vectors are split into two ordered RDMA messages.

Measured on two M5 Max 128 GB MacBooks (GLM 5.2, IQ2_XXS, 188 GiB):

| | two Macs, tensor parallel | one Mac, SSD streaming |
|---|---|---|
| decode | ~16.8 t/s (15.4 at 4k context) | ~4.8 t/s |
| prefill (4096 tokens) | ~94 t/s | ~3-5 t/s |
| residency | fully memory-resident | streams experts from SSD |

Notes: the coordinator mirrors every prompt sync and eval to the worker, so
both KV caches stay in lockstep; prompt processing splits both the
routed-expert GEMMs (by expert ownership) and the attention heads (a
contiguous half per machine) with one bulk partial-sum exchange per
layer per stage (`--tensor-parallel-token-prefill` selects a slower
token-by-token prefill that exactly matches the single-machine arithmetic).
The split graph is deterministic, but its changed floating-point reduction
order is not generally byte-identical to single-machine execution.

## Tensor Parallelism across CUDA GPUs

On a single CUDA server, `--cuda-tensor-parallel` splits DeepSeek V4 Flash
tensor and routed-expert work across an even number of GPUs. This is separate
from the Mac-to-Mac mode above: it does not use `--role`, RDMA, or the
distributed layer pipeline. GPU placement and memory budgets are selected with
the normal `--gpu-devices` and `--gpu-vram` options.

The device order is significant. With `N` devices, the first `N/2` logical
tiers are contiguous layer-pipeline homes and the second `N/2` tiers are their
tensor-parallel partners. Specify all homes first and then all partners, with
the closest P2P pair at matching positions. For example, the tested L40S host
uses physical pairs `(0,1)`, `(2,3)`, `(4,5)`, and `(6,7)`, expressed as
`0,2,4,6,1,3,5,7`. Each pair stores a 50/50 split of the routed experts, and
the vocabulary head is row-sharded across the participating output tiers.
Those large tensors are not duplicated. Dense attention, router, and shared
expert weights are replicated within each pair.

For maximum throughput on eight 48 GB L40S cards, use the imatrix Q4 model.
Its routed `Q4_K` layout has the native grouped multi-session kernels; the Q2
model is the lower-memory choice (including tested four-card runs), but its
unsupported grouped routed shapes use the exact fallback and have lower
aggregate serving throughput. Download and build the L40S target with:

```sh
./download_model.sh q4-imatrix
make cuda CUDA_ARCH=sm_89
```

This is the interactive-agent setup used on the eight-L40S server:

```sh
MODEL=gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf

./ds4-agent --cuda --cuda-tensor-parallel \
  --gpu-vram auto \
  --gpu-devices 0,2,4,6,1,3,5,7 \
  --model "$MODEL" \
  --ctx 100000
```

For serving, keep multiple KV sessions resident so decode rows can be grouped
across requests. The tested host is configured for up to 16 resident sessions:

```sh
./ds4-server --cuda --cuda-tensor-parallel \
  --gpu-vram auto \
  --gpu-devices 0,2,4,6,1,3,5,7 \
  --model "$MODEL" \
  --ctx 100000 \
  --batched-session 16 \
  --host 0.0.0.0
```

The equivalent local launchers are `./run-nvidia-tp-agent.sh` and
`./run-nvidia-tp-server.sh`. The server launcher also enables the on-disk KV
cache. Reduce the session count or context size if the requested resident KV
caches do not fit after model loading. CUDA TP, half-resident expert ownership,
output sharding, pipelined prefill, and compatible grouped decode are selected
by `--cuda-tensor-parallel`; no `DS4_CUDA_*` environment tuning is required.
Without an explicit `--prefill-chunk`, this mode uses 2048-token chunks so the
tested 16-session, 100k-context layout retains enough VRAM for resident KV
caches. An explicit `--prefill-chunk` remains an override for other topologies.

Any even card count that can hold the selected model and graph scratch is a
valid topology. On this class of 48 GB card, the useful measured endpoints are
Q2 on four cards (two pipeline stages) and Q4 on eight cards (four stages).
For a four-card PIX-paired subset such as physical GPUs `0,1,4,5`, the ordered
list is `0,4,1,5`. Two cards do not have enough memory for these Flash models.

This mode currently requires DeepSeek V4 Flash and an even multi-GPU
placement. GLM 5.2 instead uses normal layer placement across the selected
CUDA devices. DGX Spark is a single-GPU target and must not be started with
`--cuda-tensor-parallel`.

## Reducing heat, power usage and fan noise

Long local inference runs can keep the GPU busy for extended periods. If you
care more about heat, fan noise, battery life on MacBooks, or reducing thermal
stress on the hardware than about maximum throughput, use `--power N`.

`--power 100` is the default and means full speed. Lower values ask DwarfStar to target
that percentage of GPU usage: `--power 70` targets about 70%, `--power 50`
targets about half usage, and so forth. DwarfStar does this by measuring GPU work time
and inserting small sleeps between work units: during prefill it sleeps between
layers, and during generation it sleeps between decoded tokens. This reduces
sustained load without changing model output.

The option is available on the CLI, server, agent, eval, and benchmark tools
for DeepSeek models. GLM 5.2 currently accepts only `--power 100`. For example:

```sh
./ds4 --power 50
./ds4-agent --power 70
./ds4-server --power 40 --ctx 100000
```

## Native agent

DwarfStar features a native coding agent that works in a different way
than most other systems: the inference is controlled from within the agent
itself, without socket/API boundaries, so the session is represented
by the on-disk KV cache itself. Moreover the tools and the system prompt
are all designed vertically for DeepSeek v4 Flash and PRO. This provides a
few advantages:

* Low latency experience, bounded mainly by the prefill speed limits. Displaying of generated text, tool calling, start of a new session are always instantaneous.
* Live progress bar during prefill time.
* No DSML tool calling conversion, the tools are handled natively in the LLM format.
* KV cache mismatch are impossible by construction, the current state is always the truth.
* Everything is tuned for this model.
* Ability to switch saved sessions with `/list` and `/switch`; full KV sessions resume without a prefill stage.

Agent sessions are stored in `~/.ds4/kvcache`. Use `/save` to persist the
current session, `/list` to show saved sessions sorted by recent update time,
and `/switch <sha>` to resume one of them. The session ID is stable across
future saves and is derived from the first user prompt and creation time.
`/del <sha>` removes a saved session. `/strip <sha>` keeps the rendered
conversation text and title but removes the heavy KV payload; switching to a
stripped session rebuilds the KV cache by prefilling the saved text.

Use `--chdir /path/to/ds4` when launching `ds4-agent` from another directory,
so relative runtime files such as `metal/*.metal` resolve from the project tree.

However while the system already works, there is a lot of work to do
in order to make it ready for prime time. When finally the agent will reach
the wanted shape, we will *likely* split the server and the client creating a stateful
session-based protocol that can recreate all that in a client-server way.

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

Sessions prefill long prompts in 4096-token chunks by default. Use
`--prefill-chunk 2048`, for example, to match the strict official-vector
checkpoint path. Changing the chunk changes the KV checkpoint/logit path, so
compare it as an explicit run configuration.
Chunked Metal prefill reuses the same range-capable layer-major graph for each
chunk, preserving absolute compressor/indexer boundaries while avoiding the old
per-layer chunk dispatch path.

## Capability Evaluation

`ds4-eval` is a small real-model integration benchmark. It is not a leaderboard
runner and should not be reported as an official GPQA, SuperGPQA, AIME, or
security benchmark score: the questions are an embedded 92-item subset chosen
to make local regression testing useful and visually inspectable. The program
loads the real GGUF, renders DeepSeek chat prompts, streams sampled tokens in a split-screen TUI, grades
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

Use `--regrade-trace /path/to/trace.txt` to replay the current answer
extractor and scorer against a prior `--trace` file without loading the model
or regenerating tokens. This is useful when auditing evaluator changes: it
shows which cases changed, the old picked answer, the new picked answer, and a
pass/fail summary.

For inference changes that can affect generation drift, keep this deterministic
q1..q4 token-count gate in the test plan:

```sh
./ds4-eval \
  -m ds4flash.gguf \
  --plain \
  --questions 4 \
  --tokens 2048 \
  --temp 0 \
  --seed 1
```

The generated-token counts must stay aligned with the baseline:

| Question | Expected state | Expected generated tokens | Expected given/correct |
|---:|---|---:|---|
| 1 | `PASSED` | 2048 | `B` / `B` |
| 2 | `PASSED` | 438 | `C` / `C` |
| 3 | `PASSED` | 666 | `70` / `70` |
| 4 | `FAILED` | 2048 | `A` / `C` |

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

The set should be treated as a hard capability regression suite rather than
a pass/fail unit test.

- **GPQA Diamond** contributes graduate-level science questions with
  multiple-choice answers. DeepSeek's model card reports strong results
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

The interactive CLI is a real multi-turn chat. It keeps the rendered chat
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

By default the server keeps one mutable backend/KV checkpoint in memory, so
stateless clients that resend a longer version of the same prompt can reuse the
shared prefix instead of pre-filling from token zero.

`--batched-session N` preallocates `N` independent resident KV sessions. Ready
decode steps are evaluated together, while long prefills alternate in bounded
chunks so one request does not block every decoder. Requests beyond `N` wait
for a resident slot. If disk KV caching is enabled, an idle slot is persisted
before reuse and can be restored when that conversation returns; an active
request is never evicted. Choose `N` and `--ctx` so all resident KV allocations
fit in GPU memory. Without this option, inference retains the original
single-session behavior.

Batching is exact: when a native batched kernel is unavailable, DwarfStar runs
the affected rows in a fixed order and returns the same full logits as separate
session evaluations. The current backend behavior is:

| Backend and model | Session execution |
| --- | --- |
| Metal, resident DeepSeek Flash | Native shared-expert and QKV batching from two rows upward when supported; ordered fallback otherwise. |
| Metal, GLM 5.2 | Ordered exact fallback. |
| CUDA, DeepSeek Flash on a supported multi-GPU TP/EP layout | Native decode and mixed prefill/decode, with exact fallbacks for unsupported kernel shapes. |
| CUDA single GPU, including DGX Spark | Ordered exact fallback. |

`N` resident sessions allocate `N` KV states, so a context size that fits once
may not fit eight times. Native batching can improve aggregate throughput; an
ordered fallback provides concurrency and fairness, but not the same speedup.
MTP speculative decoding is disabled while native session batching is active.

Supported endpoints:

- `GET /v1/models`
- `GET /v1/models/deepseek-v4-flash`
- `GET /v1/models/deepseek-v4-pro`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/messages`

The Flash and PRO model endpoints are compatibility aliases. They both report
the model currently loaded from the GGUF passed with `-m`; the endpoint name does
not select a different model.

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
nucleus mass. In thinking mode DwarfStar applies those fixed sampling defaults
to any knob the request omits, matching DeepSeek's fixed-thinking API behavior,
but sampling parameters set explicitly in the request always win: a
`temperature=0` request is greedy through the whole reasoning phase, so
benchmark harnesses get deterministic thinking-mode output.

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

export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_AUTH_TOKEN="dsv4-local"
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

The logits are raw IEEE-754 `float32` values from the host `ds4_session`
buffer. They are saved immediately after the checkpoint tokens so a loaded
snapshot can sample or continue from the exact next-token distribution without
running one extra decode step. MTP draft logits/state are not persisted; after
loading a disk checkpoint the draft state is invalidated and rebuilt by normal
generation.

Distributed coordinator sessions use the same `DSV4` payload. Worker-owned
layer tensors are pulled during save and merged into the normal layer-ordered
tensor stream; during load the coordinator splits that stream into the current
route and pushes the relevant layer tensors back to the workers. The saved file
does not retain the distributed topology.

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

CUDA builds accept `--gpu-vram N[,N,...]` and `--gpu-devices N[,N,...]` in the
CLI, server, agent, and benchmark. VRAM values are per-device GiB budgets;
`--gpu-vram auto` uses the free memory reported by CUDA. The device list controls
the placement order and must have the same number of entries as an explicit
budget list. Placement reserves graph and KV memory for the requested context
and refuses to start if model layers would spill to the CPU.

Without `--cuda-tensor-parallel`, CUDA uses normal layer placement across the
listed devices. This is also the supported multi-GPU layout for GLM 5.2:

```sh
./ds4 -m gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf \
  --gpu-vram auto --gpu-devices 0,2,4,6,1,3,5,7 \
  --ctx 32768 -p "Hello"
```

For the DeepSeek Flash tensor/expert-parallel layout, see
"Tensor Parallelism across CUDA GPUs" above.

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
C runner pins a 2048-token prefill chunk for this strict API-vector comparison.

The core local tests are driven by the C runner, with a small `ds4-eval`
extractor self-test run first:

```sh
make test                  # ./ds4-eval --self-test-extractors && ./ds4_test --all
./ds4_test --logprob-vectors
./ds4_test --server
```

The batching tests are model-backed and must run on the matching GPU backend:

```sh
# Metal, with DS4_TEST_SESSION_COUNT set to 2, 4, 8, and 16.
DS4_TEST_MODEL=/path/to/model.gguf DS4_TEST_SESSION_COUNT=4 \
  make test-metal-session-batch

# CUDA multi-GPU Flash.
DS4_TEST_MODEL=/path/to/model.gguf make test-cuda-session-batch
DS4_TEST_MODEL=/path/to/model.gguf make test-cuda-mixed-batch
```

For GLM, run the same Metal session test with a GLM GGUF and run
`tests/glm_long_context_smoke.sh /path/to/model.gguf`. The official 100-case
quality scorers, two-Mac TCP/RDMA tests, CUDA matrix, and manual agent checks are
release gates rather than quick local tests; follow
[QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md).

## Debugging Notes

When a generation looks wrong, three small tools are usually enough to get a
first answer:

```sh
./ds4 --dump-tokens -p "..."
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ds4 --dump-logits /tmp/logits.json --metal --nothink --prompt-file prompt.txt
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

## Logo

The DwarfStar logo was designed by hand by Salvatore Sanfilippo, made more
graphical with AI, and manually reworked by Ben Gnomino, whose human touch made
it rock.
