# DGX Spark

## Build and Run

```sh
make cuda-spark
./download_model.sh ds4f-q2
./ds4 --cuda
```

The build selects `sm_121` with Blackwell-specific kernels.

## DeepSeek V4.1 Flash

```sh
./download_model.sh ds41f-q2
./ds4 --cuda -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --ssd-streaming --ctx 32768
```

Two Sparks over RoCE: ~21.9 t/s per session, ~28 t/s aggregate for 8 sessions.

## Qwen3.8 Flash Next

```sh
./download_model.sh qwen38-q2
./ds4 --mtp --ctx 32768
```

16K prefill: 250 t/s. Decode: ~17 t/s, MTP: ~20 t/s.

## GLM 5.3 Flash

```sh
./download_model.sh glm53-q2
./ds4 --cuda -m gguf/GLM-5.3-Flash-Q2.gguf --ctx 16384
```

## DSpark (Speculative Decoding)

```sh
./download_model.sh ds4f-dspark
./ds4 --cuda --dspark \
  --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf
```

Performance on DGX Spark with Flash DSpark: up to 31.4 t/s (C code, greedy).