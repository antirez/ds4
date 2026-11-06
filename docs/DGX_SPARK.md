# DGX Spark

[README](../README.md) | [Getting started](../README.md#start-here)

Use the NVIDIA driver and CUDA development
toolkit for the machine; the build needs `nvcc` and cuBLAS.
Check that `nvidia-smi` sees the GPU before building.

## Build and run

```sh
make cuda-spark
./download_model.sh ds4f-q2
./ds4 --cuda
```

The build targets GB10 (`sm_121a`) and enables the Blackwell-specific kernels.
Do not use `--cuda-tensor-parallel` on a single Spark.
Stop other inference services before loading a model so they do not compete
for memory. Restore any services you stopped when finished.

## DeepSeek V4.1 Flash

Q2 needs SSD streaming on one 128 GB Spark:

```sh
./download_model.sh ds41f-q2
./ds4 --cuda -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --ssd-streaming --ctx 32768
```

The file is 341 GiB, including 189 GiB of disk-only Engram tables. Keep it on
a fast local SSD. The expert cache is sized automatically; leave memory for
other programs and the context. Use the same options with `ds4-agent` or
`ds4-server`.

Two Sparks can instead keep half the experts each, using
[network tensor parallelism](DISTRIBUTED.md#tensor-parallelism-between-two-sparks).
Both need the complete GGUF on their local SSD. Single-host V4.1
[DSpark](SPECULATIVE_DECODING.md#deepseek-v41-flash-dspark) also works with SSD
streaming; TP speculation and CUDA vision are not supported for V4.1.
The [model guide](MODELS.md#deepseek-v41-flash) covers
thinking levels and the separate Metal configurations.

September 13-14, 2026, Q2 SSD with automatic cache sizing: a 3,241-token
`/read README.md` prefill reached 93-96 t/s, up from 49-54 t/s, with 32K
allocated context. No extra flag is needed. A separate 64 GiB cache-budget
run with 64K allocated context reached 384 t/s for a 32K initial prefill and
88 t/s for a 3.2K append. These are SSD-streamed V4.1 results, not resident
V4 Flash numbers; speed depends on prompt length and the expert cache.
The automatic-cache 256-token reply test decoded about 5% slower, but
prefill plus the reply fell from about 87 to 59 seconds.
See the [QA record](../QA_BEFORE_RELEASES.md#cuda-ssd-streaming) for conditions
and longer-context measurements.

On two Sparks over the direct RoCE link, Q2 measured **21.8 t/s** for one
session on October 2, 2026: a 1K prompt, 64K allocated context and 2,048
teacher-forced decode tokens, without speculation.

| Work | Speed |
| --- | ---: |
| Fresh 32K prefill | 411 t/s |
| Add 8K after 32K | 314 t/s |

Both runs used 64K allocated context and 512 decode tokens per frontier;
decode at these longer contexts was about 18 t/s. Engram tables stayed on disk.

Decode batching starts at five ready sessions; smaller groups run in order.
The earlier eight-session reference is about **28 aggregate t/s**, not per
client. See the [network TP QA record](../QA_BEFORE_RELEASES.md#cuda-network-tensor-parallelism)
for the workloads and memory limits.

## Flash 0731 on two Sparks

[Network tensor parallelism](DISTRIBUTED.md#tensor-parallelism-between-two-sparks)
also supports Flash 0731 Q2 and MXFP4, including DSpark with the matching
support file. Each host holds about 45 GiB of Q2 weights or 77 GiB of MXFP4,
plus context and runtime buffers. Download `ds4f-mxfp4` on both for MXFP4.

October 2, 2026, Q2, direct 200 Gb/s RoCE, 64K allocated context: a 105-token
English prose prompt decoded 512 tokens at **26.0 t/s** without speculation.
A fresh 32K prefill reached **618 t/s**; appending 8K reached **610 t/s**.
Ordinary decode at those longer frontiers was about 19 t/s. These are single
measurements, not release medians.

October 4, 2026, matching 0731 DSpark support, default speculation settings,
temperature zero, 64K allocated context and 512 generated tokens. Both formats
use the drafter on both ranks over RDMA. These are medians of two runs per case
(four for MXFP4 prose), including the first token or block. Benchmark processes
ran on the performance cores (`taskset -c 5-9,15-19`):

| Weights | Prompt | DSpark |
| --- | --- | ---: |
| Q2 | English prose, 105 tokens | 44.0 t/s |
| Q2 | C hash table, 31 tokens | 83.3 t/s |
| MXFP4 | English prose, 105 tokens | 41.7 t/s |
| MXFP4 | C hash table, 31 tokens | 75.6 t/s |

On one Spark, the same Q2 prose and coding prompts reached 23.4 and 44.0 t/s
with DSpark. These are workload-specific measurements, not a promised speedup;
poor draft acceptance can still make it slower. The measured settings are
automatic once DSpark is enabled; speculation itself remains opt-in.

## Qwen3.8 Flash Next

Both Q2 and Q4 fit on one Spark. After building:

```sh
./download_model.sh qwen38-q2
./ds4 --mtp --ctx 32768
```

Q2 keeps 41.73 GiB of weights resident; Q4 keeps 69.74 GiB. Both GGUFs also
contain 95.37 GiB of BF16 n-grams, read directly from the SSD. Leave room
for context and runtime buffers. Vision, the native agent, server APIs and
text-session checkpoints work on CUDA; see [Qwen setup](QWEN38_FLASH_NEXT.md).

September 15, 2026, Q2 on one Spark: a 16K prefill reached 250 t/s, extending
it to 30K reached 243 t/s, and ordinary decode stayed around 17.3 t/s.
These are single runs with 8,192-token prefill chunks and 32 decode tokens.
A separate 256-token prose run measured 17.1 t/s ordinarily and 19.9 t/s with
MTP, at temperature zero and 8K allocated context. Speculation is opt-in;
its benefit depends on the prompt.

## GLM 5.3 Flash

Q2 is the resident target for one Spark:

```sh
./download_model.sh glm53-q2
./ds4 --cuda -m gguf/GLM-5.3-Flash-Q2.gguf --ctx 16384
```

Q4 does not fit resident. GLM Spark-to-Spark tensor parallelism is not
implemented; the two-Mac RDMA instructions do not apply to GLM on CUDA.

## Vision and speculative decoding

DeepSeek Vision Experimental needs its matching text model and encoder:

```sh
./download_model.sh ds4f-vision-q2
./ds4 --cuda --vision gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf
```

Use `/read image.png` in the CLI. GLM vision also works on this backend; see
[models and vision](MODELS.md#vision).

For Flash 0731, [DSpark](SPECULATIVE_DECODING.md) uses the separate 0731 support
file. Vision Experimental has a different drafter. GLM uses its built-in MTP
block with `--mtp`. None is enabled by default.

## Flash Q2 with DSpark

September 6, 2026, fully resident Flash 0731: median generation speed from
three runs of 256 tokens, 4K allocated context and a 512-token prefill chunk.

| Prompt | Temperature | Ordinary decode | DSpark |
| --- | ---: | ---: | ---: |
| C hash table | 0 | 19.72 t/s | 31.41 t/s |
| C hash table | 1 | 19.53 t/s | 29.98 t/s |
| Unpredictable prose | 1 | 19.53 t/s | 18.81 t/s |

Longer coding runs reached about 33 t/s. Poor draft acceptance can still make
DSpark slower. Temperature 1 uses the default opportunistic policy; these are
not exact-sampling results. DSpark does not accelerate prefill.

To reproduce the greedy coding case after downloading `ds4f-q2` and
`ds4f-dspark`:

```sh
./ds4 --cuda -m ds4flash.gguf \
  --dspark --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --ctx 4096 --prefill-chunk 512 --nothink --temp 0 --seed 12345 -n 256 \
  -p 'Write a complete C hash table implementation with string keys, insert, find, delete, and a test main. Output only C code.'
```

For the temperature-1 rows, use `--temp 1 --top-p 0.95 --min-p 0.05`.
The [QA record](../QA_BEFORE_RELEASES.md#16-speed-regression) includes the previous
implementation, continued-context checks and quality comparisons.

## Larger models and serving

CUDA also has [SSD streaming](SSD_STREAMING.md) paths for larger weights.
Memory fit and speed depend on the model layout; start with Q2 for normal use.

V4.1 Q2 SSD serving batches up to eight decode rows together. An eight-session
test reached 11.0 aggregate t/s versus 8.4 with ordered execution; this is
total throughput, not speed per client. Other single-Spark model paths retain
their existing scheduling.
See [serving](SERVER.md) and the recorded [benchmarks](PERFORMANCE.md).
