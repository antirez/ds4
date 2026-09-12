# Models and Vision

[README](../README.md) | [Getting started](../README.md#start-here)

DwarfStar is not a general GGUF runner. Use the download targets below: other
GGUFs may have unsupported tensor layouts, metadata, or quantization mixes.
Run `./download_model.sh --help` for filenames and all available targets.

Main-model downloads update `ds4flash.gguf`. Encoders, draft models, packaged
FP8 weights, and split PRO pieces do not. Pass `-m FILE` to avoid depending on
which model was downloaded last.

Some downloads require the Hugging Face CLI; the script prints installation
instructions when needed. Authentication is optional for public weights;
your cached Hugging Face token or `HF_TOKEN` is used when present.

## DeepSeek V4

| Target | Use |
| --- | --- |
| `ds4f-q2` | Flash 0731, about 81 GiB; starting point for 96/128 GB systems |
| `ds4f-q2-q4` | Mostly Q2, with the last six routed-expert layers at Q4; needs more memory |
| `ds4f-q4` | Flash 0731 Q4; larger-memory or distributed systems |
| `ds4f-mxfp4` | Native MXFP4 routed experts; larger-memory or distributed systems |
| `pro-q2-imatrix` | PRO 0813; 512 GB resident target, or SSD streaming |
| `pro-q4-split` | Both PRO Q4 pieces for pipeline execution |

```sh
./download_model.sh ds4f-q2
./ds4
```

The Flash Q2 recipe spends most of its compression on routed experts:
IQ2_XXS gate/up and Q2_K down. Other components use higher precision, including
Q8 projections, shared experts and output, plus F16/F32 tensors. They are not
all untouched source weights. The imatrix guides the routed quantization.

The MXFP4 recipe preserves DeepSeek's released MXFP4 routed experts without
requantizing them. Metal and CUDA support it. Blackwell uses native FP4 matrix
instructions and FP4 activations for batched expert work; CUDA decode and
other CUDA architectures use Q8 activations. ROCm also has a resident MXFP4
path, including pipeline execution for models too large for one host.

To build weights rather than download them, see [GGUF tools](../gguf-tools/README.md).

## DeepSeek V4.1 Flash

V4.1 Flash text and vision inference work on Metal. It needs its own GGUF,
tokenizer and inference graph; V4 Flash weights and DSpark
support files are not interchangeable with it.

| Target | File size | Main weights |
| --- | ---: | ---: |
| `ds41f-q2` | 341 GiB | 152 GiB |
| `ds41f-q4` | 483 GiB | 294 GiB |

Both use imatrix-calibrated routed experts and include 189 GiB of Engram
tables. Engram rows are read directly from the file as needed in every mode,
never loaded as a resident table. Keep the GGUF on a fast local SSD.

On one 128 GB Mac, use SSD streaming. Leave the expert cache budget automatic:

```sh
./download_model.sh ds41f-q2
./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --ssd-streaming --ctx 32768
```

Use `ds4-agent` or `ds4-server` with the same model and memory options.
`--think-level 25` sets reasoning effort explicitly; the range is 1 to 100,
with 0 disabling thinking. `/think 25` changes it in the CLI or native agent.
`--think` selects 75 and `--think-max` selects 100.

For two 128 GB Macs, follow the [TP/RDMA setup](DISTRIBUTED.md), passing this
GGUF with `-m` on both ranks and omitting `--ssd-streaming`. Each rank holds
about 81 GiB of main weights, plus context and runtime buffers. Both machines
need the complete GGUF on disk. A 256 GB or larger Mac can instead hold all
main weights; full residency has been tested on an M3 Ultra with 512 GB.

Q4 needs SSD streaming on smaller Macs, or a 512 GB Mac for full residency.
It does not fit resident TP across two 128 GB Macs. Download it with
`./download_model.sh ds41f-q4` and select `gguf/DeepSeek-V4.1-Flash-Q4.gguf`.
The download comes in two parts; the script joins and verifies them automatically.
Allow another 37 GiB of free disk space while joining. Rerun the command to resume an
interrupted download or join.

Large SSD prefills process layers in wide batches, overlapping computation
with the next layer's reads. Short appends keep using the expert cache.
Resident and TP inference also batch continued prefills automatically.

For concurrent serving, see [session batching](SERVER.md#multiple-sessions).
Each slot needs its own context memory; start with `--ctx 4096` before
increasing both context and slot count. DSpark, pipeline execution and
non-Metal backends are not implemented for V4.1.

Scalar, batched and tensor-parallel execution are not numerically identical.
Q4 batched prefill shows a small probability-score loss on the short official
continuation test, with unchanged overall top-token agreement. See the
[QA results](../QA_BEFORE_RELEASES.md#17-deepseek-v41-flash-metal) for details
and remaining differences.

For images, download the matching encoder and add it to the same command:

```sh
./download_model.sh ds41f-vision
./ds4-agent -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --ssd-streaming --vision gguf/DeepSeek-V4.1-Flash-Vision.gguf
```

Vision works with SSD streaming, full residency and two-Mac TP. Pass the encoder
on both TP ranks. Use `/read image.png` in `ds4`, `view_image` in `ds4-agent`,
or the [server image API](SERVER.md#images). V4 Flash vision encoders do not
work with V4.1. See [conversion](../gguf-tools/README.md#convert-deepseek-v41-flash)
to build the GGUFs from safetensors.

## GLM 5.3 Flash

| Target | Approximate file size | Use |
| --- | ---: | --- |
| `glm53-q2` | 90 GiB | One 128 GB Mac or DGX Spark; ROCm also supported |
| `glm53-q4` | 178 GiB | Larger Mac, two 128 GB Macs, or SSD streaming |
| `glm53-fp8` | 305 GiB | Packaged native weights only; inference not implemented |

```sh
./download_model.sh glm53-q2
./ds4 -m gguf/GLM-5.3-Flash-Q2.gguf --ctx 32768
```

GLM 5.3 Flash has recurrent KDA layers, sparse DSA attention, hyper-connections,
and a built-in MTP block. The Q2 file uses imatrix-guided IQ2_XXS gate/up and
Q2_K down experts. Q4 is the higher-precision alternative.

Q2 is close enough to a 128 GB machine's memory budget that other workloads
and context size matter. Follow the [Metal](METAL.md), [Spark](DGX_SPARK.md),
or [Strix Halo](STRIX_HALO.md) starting configuration for your host.

Ordinary decode is the default. Enable the embedded draft block with `--mtp`:

```sh
./ds4-agent -m gguf/GLM-5.3-Flash-Q2.gguf --mtp --ctx 50000
```

No second model file is needed. See [sampling behavior](SPECULATIVE_DECODING.md)
before choosing between the default opportunistic mode and exact sampling.

## Full GLM 5.3 and GLM 5.2

Full GLM 5.3 Q2 is about 197 GiB. Use a sufficiently large machine or streaming:

```sh
./download_model.sh glm53-full-q2
./ds4 --ssd-streaming
```

GLM 5.2 downloads are `glm-antirez-iq2xxs`, `glm-antirez-q2`,
`glm-antirez-q4`, and the 11-shard `glm-unsloth-q4`. They are much larger
than GLM 5.3 Flash; choose memory capacity before choosing the quantization.

GLM runs on Metal, CUDA and ROCm. The routed paths include IQ2_XXS, Q2_K, and
Q4_K, with additional mixed layouts supported by the tested GGUFs. Two-Mac
ownership-aware TP accepts IQ2_XXS, Q2_K, and Q4_K gate/up layouts; this does
not mean every Q4 model fits two 128 GB machines. Use a tested artifact, not
an arbitrary combination of supported tensor types.

GLM uses graph-selected prefill chunks and does not accept `--prefill-chunk`
or an external `--mtp-model`. It currently requires `--power 100`.
Directional steering is supported for GLM 5.3, not GLM 5.2.

## Vision

PNG and JPEG input works in the CLI, native agent, and HTTP server on Metal,
single-GPU CUDA, and ROCm. The encoder must match the model.
V4.1 Flash is currently Metal-only; its setup is [above](#deepseek-v41-flash).

### DeepSeek Flash Vision Experimental

Vision Experimental is a different language checkpoint from Flash 0731.
The main download includes its encoder:

```sh
./download_model.sh ds4f-vision-q2
./ds4 --vision gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf
```

Larger targets are `ds4f-vision-q2-q4` and `ds4f-vision-mxfp4`.
`ds4f-vision-encoder` downloads just the encoder when the language GGUF is
already present. Vision Experimental has its own [DSpark drafter](SPECULATIVE_DECODING.md).

### GLM 5.3 Flash

The text GGUF stays the same. Download and add the encoder explicitly:

```sh
./download_model.sh glm53-vision
./ds4 -m gguf/GLM-5.3-Flash-Q2.gguf \
  --vision gguf/GLM-5.3-Flash-Vision-Encoder.gguf
```

Use `/read image.png` in `ds4`, or start `ds4-agent` with the same `--vision`
argument to enable `view_image`. Agent sessions containing images cannot yet
be saved with `/save`.

For two-Mac TP, pass the same encoder on both ranks. The coordinator encodes
the image and sends the projected visual tokens to the worker.
For HTTP image formats and limits, see [serving](SERVER.md#images).
