# Qwen3.8 Flash Next

## Downloads

| Target | Size |
| --- | --- |
| `qwen38-q2` | 42 GB weights + 95 GB n-grams on disk |
| `qwen38-q4k` | 70 GB weights + 95 GB n-grams |
| `qwen38-vision` | Optional encoder |

```sh
./download_model.sh qwen38-q2
```

## First Run

```sh
./ds4 --ctx 8192 --prefill-chunk 1024
```

## Speculative Decoding (MTP)

```sh
./ds4 --mtp --ctx 32768
```

MTP weights included in the GGUF — no separate download.

## CUDA Performance

On DGX Spark with Q2:

| Metric | Value |
| --- | --- |
| 16K prefill | 250 t/s |
| Ordinary decode | ~17.3 t/s |
| MTP decode | 19.9 t/s |

## Vision

```sh
./download_model.sh qwen38-vision
./ds4 --vision gguf/Qwen3.8-Flash-Next-Vision-Encoder.gguf
```

## Validation

```sh
./ds4_test --qwen4-kernels
./ds4_test --qwen4-ngrams
./ds4_test --qwen4-ngram-state
./ds4_test --qwen4-vision
```