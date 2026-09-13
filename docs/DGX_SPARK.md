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

The build selects `sm_121` and enables the Blackwell-specific kernels.
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
Both need the complete GGUF on their local SSD. V4.1 CUDA vision and DSpark
are not supported. The [model guide](MODELS.md#deepseek-v41-flash) covers
thinking levels and the separate Metal configurations.

September 12, 2026, Q2 SSD with automatic cache sizing and 64K allocated
context: two runs reached 332 and 359 t/s for a 32K initial prefill, and
183 and 187 t/s when adding 8K at that frontier. Generation was about
7-8 t/s, measured over 32 teacher-forced tokens. Smaller prompts have lower
prefill throughput. These are SSD-streamed V4.1 results, not resident V4 Flash numbers.
See the [QA record](../QA_BEFORE_RELEASES.md#cuda-ssd-streaming) for conditions
and longer-context measurements.

On two Sparks over RoCE, September 13 measurements averaged **21.9 t/s** for
one session: Q2, a 1K prompt, 64K allocated context and 2,048 teacher-forced
decode tokens. This does not use speculative decoding. Separately, eight
ready sessions reached about **28 aggregate t/s**, not 28 t/s per client. Decode batching
starts at five ready sessions; smaller groups run in order. Separate long
runs measured about 400 t/s for a 32K prefill and 290 t/s for an 8K append. The
[network TP QA record](../QA_BEFORE_RELEASES.md#cuda-network-tensor-parallelism)
has the timing controls and memory limits.

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
