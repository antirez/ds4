<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="220">
</p>

**DwarfStar** is a small native inference engine optimized first for
**DeepSeek V4 Flash** (including the experimental vision model).
It also supports **GLM 5.2 and 5.3**, **GLM 5.3 Flash**, and
**DeepSeek V4 PRO**. It is self-contained and
deliberately narrow, not a general GGUF runner. Model loading, prompt rendering,
tool calls, KV state, the HTTP server, and the coding agent are built and tested together.
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

The project has first class support for SSD streaming of weights, so it is
possible to run models bigger than RAM while often still getting decent
performances, and even running very large models (like the full GLM 5.3 or DeepSeek v4 PRO)
on systems with just 128GB of RAM at a slower speed, but fast enough for
QA-style chats.

# So, what can I do with this software?

* You can run a very capable models in your consumer hardware, a MacBook, a DGX Spark, or a Strix Halo for example. Even if you have not enough RAM, with SSD streaming, you can run it at a decent speed.
* You can use multiple CUDA cards as a multi-user LLM server. Ada Lovelace, including L40S, is supported: newer models can run here even when their other inference implementations require newer GPUs. Our eight-L40S Flash setup has reached about 126 t/s aggregate generation with 16 sessions.
* Using two 128 GB Macs connected with RDMA, you can run 4-bit DeepSeek Flash or GLM 5.3 Flash with tensor parallelism. Larger GLM 5.2 quants need larger machines, such as Mac Studios.
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

# How to use this project?

I (Salvatore) believe that the way projects should be shipped and used changed because of AI. The main differences today are:

1. With AI, users can modify the software in significant ways with low efforts, costs, and even lacking deep domain knowledge about the task they want to accomplish. For instance, a DwarfStar user with a specific hardware setup can ask a coding agent to improve the inference speed of this software for the specific hardware setup, asking the model to reach the maximum prefill and generation speed without impacting correctness, and also asking to do a deep QA pass.
2. Similiarly, because of "1", software may be shipped in a different way than before. It must be more a working template for the biggest use cases, without trying to cover every possible setup. If DwarfStar showcases a few good implementations of tensor parallel execution, the code will work as a rail for implementing the same feature in specific conditions, for a new model, and so forth.

So, while this project attempts to be usable for the featured models and the most common hardware setups, I ask you, if you have access to coding agents, to consider using coding agents as an interface to discover the project, make modifications, create personalized setups. This way you can likely do more than what we ship, and certain things that are not documented or implemented, and that you require, are potentially very easy to achieve.

## Start Here

```sh
git clone https://github.com/antirez/ds4.git
cd ds4
```

Choose your build. The platform guides cover prerequisites, memory sizing,
and hardware-specific setups:

| Platform guide | Build |
| --- | --- |
| [Metal on Apple Silicon](docs/METAL.md) | `make` |
| [DGX Spark](docs/DGX_SPARK.md) | `make cuda-spark` |
| [Strix Halo / Framework Desktop](docs/STRIX_HALO.md) | `make strix-halo` |
| [One or more CUDA cards, including Ada/L40S](docs/CUDA_MULTI_GPU.md) | `make cuda-generic` |

For a first run on a 96 or 128 GB machine, download DeepSeek V4 Flash Q2:

```sh
./download_model.sh ds4f-q2
```

Downloads go in `gguf/`. Repeat the command to resume an interrupted download.
Leave memory for the context and runtime buffers as well as the model.
See [other models](docs/MODELS.md) or use [SSD streaming](docs/SSD_STREAMING.md)
on a smaller Mac.

## Everyday Use

Once built and with a model downloaded:

```sh
./ds4
./ds4 -p "Explain Redis streams in one paragraph."
./ds4-agent
./ds4-server --ctx 32768
```

The default model is `ds4flash.gguf`, a link updated by main-model downloads.
Pass `-m FILE` to choose explicitly. Commands normally run from the repository
root; use `--chdir /path/to/ds4` when launching elsewhere.

The server listens at `http://127.0.0.1:8000` by default; see [serving](docs/SERVER.md)
for API access and multiple sessions.

The interactive CLI keeps a multi-turn conversation. Use `/help`, `/read FILE`,
`/ctx N`, and `/quit`. Ctrl+C interrupts generation and returns to the prompt.
Run each binary with `--help` for its full options.

### Native coding agent

`ds4-agent` runs inference directly, without a separate HTTP server. It keeps
the token history and live model state together, shows prefill progress, and
uses the model's native tool format. DeepSeek and GLM have their own templates.

Sessions are stored in `~/.ds4/kvcache`:

| Command | Action |
| --- | --- |
| `/save` | Save the current session |
| `/list` | List saved sessions |
| `/switch <sha>` | Resume a session |
| `/del <sha>` | Delete a saved session |
| `/strip <sha>` | Keep text and title, removing the large KV payload |

Compatible local KV snapshots avoid rebuilding the prompt. Stripped sessions
and network TP restores require prefill. Sessions containing images cannot yet
be saved. Saved conversations and traces may contain private information.

For Pi, OpenCode, Codex CLI, or Claude Code, use `ds4-server` instead and follow
the [client setup guide](docs/CLIENTS.md).

### Models, images, and speculation

[Models and vision](docs/MODELS.md) lists the supported downloads and memory
requirements. DeepSeek Vision Experimental uses a different checkpoint from
Flash 0731; GLM 5.3 Flash adds vision to the same text model.

With the matching encoder passed as `--vision FILE`, use `/read image.png`
in the CLI or `view_image` in the native agent.

Speculative decoding is opt-in. GLM uses `--mtp`; Flash DSpark needs a matching
support GGUF. It can improve generation, but not every workload benefits.
Read [speculative decoding](docs/SPECULATIVE_DECODING.md) for setup and the
difference between default opportunistic sampling and `--mtp-exact-sampling`.

### Output and power

Thinking is enabled by default. Use `--nothink` or `/nothink` for direct
answers, and `--think` or `/think` to enable it again.
The normal sampling defaults are temperature 1, top-p 1, and min-p 0.05;
`--temp 0` selects greedy output.

For DeepSeek, `--power N` trades throughput for lower sustained GPU load.
The default is 100. GLM currently requires `--power 100`.

DeepSeek Flash and GLM 5.3 Flash also support directional steering. Load a
vector with `--dir-steering-file FILE`; `/steer F` adjusts its scale for
subsequent tokens in a local CLI or agent session, without rebuilding the
existing KV cache. See [steering documentation](dir-steering/README.md).

`--prefix-file FILE` preloads complete `USER:` / `ASSISTANT:` pairs before
the live conversation. A turn marker must start a line, roles must alternate,
and the last turn must be `ASSISTANT:`.

## Capability Evaluation

`ds4-eval` runs embedded capability regression tests against a real GGUF.
These are DwarfStar integration checks, not official leaderboard scores.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
./ds4-eval -m ds4flash.gguf --suite hard-smoke
./ds4-eval -m ds4flash.gguf --suite hard --retry-incomplete
```

The default suite is `core`; `--suite all` runs core and hard cases.
`--list-cases` lists tests without loading a model. `--plain` selects
non-interactive output, and `--regrade-trace FILE` scores an existing trace
without generating again. Sources and licenses are in [EVAL_DATA.md](EVAL_DATA.md).
For inference correctness and release checks, read [testing](docs/TESTING.md).

## Speed

This recorded DeepSeek V4 Flash Q2 sweep uses an M5 Max with 128 GB RAM,
2048-token continued-prefill intervals, and 128 greedy generation tokens per
frontier. It is a baseline, not a fresh benchmark of every commit.

![M5 Max Flash Q2 throughput](speed-bench/m5_max_ts.svg)

See [performance and benchmarking](docs/PERFORMANCE.md) for the full numbers,
DGX Spark results, comparison conditions, and benchmark commands.

## Detailed Guides

- [Models and vision](docs/MODELS.md): Flash, PRO, GLM, and matching encoders.
- [SSD streaming](docs/SSD_STREAMING.md): run larger than RAM and size the cache.
- [Inference across machines](docs/DISTRIBUTED.md): two-Mac TP/RDMA and layer pipelines.
- [Speculative decoding](docs/SPECULATIVE_DECODING.md): DSpark, GLM MTP, and sampling.
- [Serving](docs/SERVER.md): APIs, images, batching, and disk KV caches.
- [Coding agent clients](docs/CLIENTS.md): Pi, OpenCode, Codex CLI, and Claude Code.
- [Performance](docs/PERFORMANCE.md): reproducible measurements and recorded baselines.
- [Testing and development](docs/TESTING.md): regression tests, debugging, and model-building tools.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request.

## Logo

The DwarfStar logo was designed by hand by Salvatore Sanfilippo, made more
graphical with AI, and manually reworked by Ben Gnomino, whose human touch made
it rock.

## Qwen3.8 Flash Next

Qwen3.8-Flash-Next (`qwen4exp` in GGUF terms) runs on its own Metal graph:
36 gated delta-net layers and 12 gated GQA layers with the QSA block-sparse
indexer, four-stream hyper-connections, the hashed per-layer n-gram table,
512-expert MoE with a shared expert, and the built-in MTP block. Build the
GGUF from the Hugging Face checkpoint with the converter in `gguf-tools/`;
the stock `ggml-org` Q8_0 GGUF also loads once its two parts are merged with
`llama-gguf-split --merge`:

```sh
python gguf-tools/qwen4_exp_convert.py --src /path/to/Qwen3.8-Flash-Next \
  --out gguf/Qwen3.8-Flash-Next-Q8.gguf --outtype q8_0            # about 192 GB
python gguf-tools/qwen4_exp_convert.py --src /path/to/Qwen3.8-Flash-Next \
  --out gguf/Qwen3.8-Flash-Next-MXFP4.gguf --outtype q8_0 --experts mxfp4  # about 126 GB
llama-quantize --allow-requantize --tensor-type hc_=f16 --tensor-type ffn_gate_exps=Q4_K \
  --tensor-type ffn_up_exps=Q4_K --tensor-type per_layer_token_embd=Q4_0 \
  gguf/Qwen3.8-Flash-Next-Q8.gguf gguf/Qwen3.8-Flash-Next-Q4K.gguf Q8_0  # about 124 GB
llama-quantize --allow-requantize --tensor-type hc_=f16 --tensor-type ffn_gate_exps=Q2_K \
  --tensor-type ffn_up_exps=Q2_K --tensor-type ffn_down_exps=MXFP4 \
  --tensor-type per_layer_token_embd=Q4_0 \
  gguf/Qwen3.8-Flash-Next-Q8.gguf gguf/Qwen3.8-Flash-Next-Q2K.gguf Q8_0   # about 83 GB
llama-quantize --imatrix imatrix.gguf --allow-requantize --tensor-type hc_=f16 \
  --tensor-type blk.48.ffn_gate_exps=MXFP4 --tensor-type blk.48.ffn_up_exps=MXFP4 \
  --tensor-type ffn_gate_exps=IQ2_XXS --tensor-type ffn_up_exps=IQ2_XXS \
  --tensor-type ffn_down_exps=MXFP4 --tensor-type per_layer_token_embd=Q4_0 \
  gguf/Qwen3.8-Flash-Next-Q8.gguf gguf/Qwen3.8-Flash-Next-IQ2.gguf Q8_0   # about 77 GB
```

The Q8 file keeps every weight at 8 bits except the QSA indexer projections,
which stay at the released BF16. The MXFP4 file keeps the routed experts in
native MXFP4 blocks, the Q4K file requantizes the expert gate/up projections
to Q4_K, and the two-bit files take them to Q2_K or, with an importance
matrix from `llama-imatrix`, IQ2_XXS; the 640-wide expert down projections
are too narrow for 256-value blocks and go to MXFP4. The first matching
`--tensor-type` wins, and the MTP block's experts (`blk.48`) stay MXFP4
because an importance matrix collected with llama.cpp never sees them. Keep
the `hc_=f16` override: the Q8_0 base type would otherwise requantize the
hyper-connection mixers, which slows prefill. Q2K is the better two-bit
choice unless the last few GB matter. All need a Mac with more unified memory
than the file size.

```sh
./ds4 -m gguf/Qwen3.8-Flash-Next-Q8.gguf --ctx 32768
./ds4 -m gguf/Qwen3.8-Flash-Next-Q8.gguf --mtp --temp 0
./ds4-server -m gguf/Qwen3.8-Flash-Next-Q8.gguf --ctx 65536 --kv-disk-dir ~/.ds4/server-kv
./ds4-server -m gguf/Qwen3.8-Flash-Next-Q8.gguf --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
./ds4-agent -m gguf/Qwen3.8-Flash-Next-MXFP4.gguf
./ds4-server -m gguf/Qwen3.8-Flash-Next-Q4K.gguf --mtp --mtp-exact-sampling --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
```

The MTP block is inside the same GGUF; `--mtp` enables it and speeds up
greedy decoding. At non-zero temperature it keeps target-matching greedy
drafts, like the GLM path, which skews sampled output toward the greedy
choice; `--mtp-exact-sampling` preserves the ordinary sampling distribution
at a smaller speedup. Prefill runs in 8192-token chunks
(`DS4_QWEN4_PREFILL_CHUNK` overrides it; the transient buffers scale with the
chunk size). For A/B checks, `DS4_QWEN4_NO_FUSE=1` selects the unfused decode
kernels, `DS4_QWEN4_NO_IDX_SELECT=1` the argsort top-k and
`DS4_QWEN4_NO_ATTN_MM=1` the per-token attention kernel for prefill batches;
`DS4_QWEN4_TIMING=1` prints stage timings.

`--batched-session N` keeps N sessions resident; their decode steps run one
after another rather than as one grouped batch, so it buys concurrency, not
throughput. Thinking is on by default with the model's `xhigh` reasoning
instruction; `reasoning_effort` `low`, `medium` or `xhigh` selects the model
card's levels (`chat_template_kwargs` with `enable_thinking` and
`reasoning_effort` is accepted too), the server's `qwen3.8-flash-next-chat`
alias disables thinking and `qwen3.8-flash-next-reasoner` forces it. Tool
calls use the model's native `<tool_call><function=...><parameter=...>` format
in both the server and the agent. Disk KV checkpoints and live prefix reuse
work as for the other models; the recurrent state travels with the checkpoint,
so a session cannot be rewound to an arbitrary earlier position and a shorter
prompt is prefilled again.

The model's native window is 262144 tokens. For longer prompts set
`DS4_QWEN4_YARN_FACTOR=4`, which
applies the static YaRN scaling from the model card, up to about 1M tokens;
use a factor of 2 for 512k. Static YaRN costs a little accuracy on short
prompts, so leave it off otherwise.

Images go through the model's Qwen3-VL vision tower. Pass llama.cpp's mmproj
file (`ggml-org/Qwen3.8-Flash-Next-GGUF` ships a Q8_0 one, or run
`convert_hf_to_gguf.py --mmproj --outtype f16` on the checkpoint) with
`--vision` and send `image_url` parts as usual. Each image is resized to
multiples of 32 pixels within 64 to 1024 tokens (`DS4_QWEN4_IMAGE_MAX_TOKENS`
raises the cap), encoded on the GPU, and takes the model's 3D rope positions;
live KV reuse keys on the image fingerprints. `make test-qwen4-vision`
compares the tower with the Hugging Face implementation.

Metal only for now. The Metal graph accepts Q8_0, Q4_0, F16, BF16 and F32
dense weights, Q8_0/MXFP4/Q4_0/Q4_K/Q2_K/IQ2_XXS experts, F16/F32/Q8_0
hyper-connection mixers and a Q8_0/Q4_0/MXFP4/F16/F32 n-gram table.
Multi-node tensor parallelism is not implemented yet.

## Qwen3.8 Flash Next (campaign branch merge)

This branch carries the qwen4-exp engine adoption merge. The native fast-pack
qwen4 engine developed during the campaign is superseded; its history, notes
and performance handoff live in [QWEN38_FLASH_NEXT.md](QWEN38_FLASH_NEXT.md)
and [QWEN38_PERF_HANDOFF.md](QWEN38_PERF_HANDOFF.md), and its source is kept
in-tree (`ds4_qwen4.c`/`ds4_qwen4.h`, unreferenced by the build).

The canonical runtime model is the Q4_K-imatrix `pleext` build run with the
external PLE sidecar, for example:

```
./ds4 -m Qwen3.8-Flash-Next-Q4KImatrix-qwen4exp-pleext.gguf \
      --ple Qwen3.8-Flash-Next-PLE-Q4_1.gguf --metal
```

Models still in the old fast-pack format can be converted with
`gguf-tools/qwen4_pack_to_qwen4exp.py` and then run the same way.
