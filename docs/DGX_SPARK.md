# DGX Spark

[README](../README.md) | [Getting started](../README.md#start-here)

This is the single-GPU GB10 path. Use the NVIDIA driver and CUDA development
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

## GLM 5.3 Flash

Q2 is the resident target for one Spark:

```sh
./download_model.sh glm53-q2
./ds4 --cuda -m gguf/GLM-5.3-Flash-Q2.gguf --ctx 16384
```

Q4 does not fit resident. Spark-to-Spark RDMA tensor parallelism is not
implemented; the two-Mac RDMA instructions do not apply here.

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

Single-GPU session serving currently uses ordered decode rather than the
native grouped decode used by the supported multi-GPU Flash configurations.
See [serving](SERVER.md) and the recorded [benchmarks](PERFORMANCE.md).
