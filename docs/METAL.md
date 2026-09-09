# Metal on Apple Silicon

[README](../README.md) | [Getting started](../README.md#start-here)

## Build and run

Install Apple's command-line developer tools if needed:

```sh
xcode-select --install
```

From the repository root:

```sh
make
./download_model.sh ds4f-q2
./ds4
```

The same build supports M3 and M5 Macs. Hardware-specific fast paths are
selected automatically; no environment variable is needed to enable them.
Leave other GPU and memory-heavy applications idle when comparing performance.

## Choose a model

| Memory | Starting point |
| --- | --- |
| 64 GB | Flash Q2 with `--ssd-streaming` |
| 96 GB | Flash Q2; leave room for the context and other applications |
| 128 GB | Flash Q2, or GLM 5.3 Flash Q2 with modest initial context |
| 256 GB | Flash Q4/MXFP4 or GLM 5.3 Flash Q4 |
| 512 GB | Larger models, including PRO Q2 |

These are starting points, not guarantees that every context or session count
will fit. GLM 5.3 Flash Q2 is about 90 GiB before runtime allocations.
Stop other memory-heavy workloads before loading it resident.

```sh
./download_model.sh glm53-q2
./ds4 -m gguf/GLM-5.3-Flash-Q2.gguf --ctx 32768
```

For a model larger than RAM, start with automatic cache sizing:

```sh
./ds4 --ssd-streaming
```

See [SSD streaming](SSD_STREAMING.md) before increasing the expert cache.
Do not bypass the memory guard just to make an oversized resident model start.

## Two Macs

Two 128 GB Macs can run a larger model fully resident with a 50/50 routed-expert
split. Thunderbolt RDMA is the low-latency option; TCP is also supported.
Follow [tensor parallel setup](DISTRIBUTED.md#tensor-parallelism-between-two-macs).
For more than two machines, use [pipeline parallelism](DISTRIBUTED.md#pipeline-parallelism).

## Next steps

- [Vision and GLM models](MODELS.md)
- [DSpark and GLM MTP](SPECULATIVE_DECODING.md)
- [Batched serving](SERVER.md#multiple-sessions)
- [Benchmarking](PERFORMANCE.md)

For DeepSeek, `--power 70` trades throughput for lower sustained GPU load.
GLM currently requires `--power 100`.
