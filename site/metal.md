# Apple Silicon (Metal)

## Build and Run

```sh
xcode-select --install
make
./download_model.sh ds4f-q2
./ds4
```

The same build supports M3 and M5 Macs. Hardware fast paths are automatic.

## Memory Guide

| RAM | Recommended |
| --- | --- |
| 64 GB | Flash Q2 with `--ssd-streaming` |
| 96 GB | Flash Q2 |
| 128 GB | Flash Q2 or GLM 5.3 Flash Q2 |
| 256 GB | Flash Q4/MXFP4 or GLM 5.3 Flash Q4 |
| 512 GB | PRO Q2, full GLM 5.2 |

For models larger than RAM:

```sh
./ds4 --ssd-streaming
```

## Two Macs

Two 128 GB Macs can run larger models with tensor parallelism. Thunderbolt RDMA is recommended.

Follow the [distributed setup](/distributed).

## Power Management

```sh
./ds4 --power 70    # Lower GPU load, V4 Flash only
./ds4 --power 100   # Default (required for V4.1 and GLM)
```