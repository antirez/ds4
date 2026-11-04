# Jetson AGX Thor

[README](../README.md) | [Getting started](../README.md#start-here)

Tested on the Jetson AGX Thor Developer Kit (p4071-0000 + p3834-0008):
128 GB unified LPDDR memory, one Blackwell GPU (sm_110), CUDA 13.0.
There is no `nvidia-smi` on Jetson; the GPU shares system memory and is
visible to CUDA directly.

## Build and run

```sh
make cuda CUDA_ARCH=sm_110
./download_model.sh ds4f-q2
./ds4 --cuda
```

Use the MAXN power mode for full speed; it measured about 12-13% faster
prefill and 8% faster decode than the default 120W mode:

```sh
sudo nvpmodel -m 0 && sudo jetson_clocks
```

Stop other inference services before loading a model. Q2 Flash is 80.8 GiB
and fits resident in one 128 GB Thor; no SSD streaming is needed.

## DeepSeek V4 Flash Q2, resident

Planned memory at server startup (model 80.76 GiB + KV + buffers):

| Allocated context | Total planned | Notes |
| --- | ---: | --- |
| 32,768 | 81.79 GiB | KV 0.78 GiB |
| 131,072 | 83.80 GiB | KV 2.04 GiB |
| 1,048,576 | 102.56 GiB | KV 13.79 GiB; managed KV cache mode |

The 1M-context configuration serves normally; a 101,186-token real-text
prompt prefilled at ~403 t/s and decoded afterward (see below).

`ds4-bench`, greedy, MAXN, ctx sweep (128 generated tokens per frontier):

| Context | Prefill t/s | Generation t/s |
| ---: | ---: | ---: |
| 2,048 | 423.8 | 12.65 |
| 4,096 | 431.9 | 12.50 |
| 6,144 | 420.0 | 12.39 |
| 8,192 | 415.0 | 12.24 |

Streaming completions over the HTTP API (OpenAI-compatible
`/v1/completions`, real prose prompts, measured October 29, 2026):

| Prompt tokens | TTFT | Implied prefill t/s | TPOT after TTFT |
| ---: | ---: | ---: | ---: |
| 1,039 | 2.72 s | ~430 | 70.3 ms |
| 4,141 | 9.70 s | ~427 | 82.1 ms |
| 33,132 | 74.41 s | ~445 | 89.8 ms |
| 101,186 | 250.80 s | ~403 | 107.5 ms |

Short-context decode is about 67-70 ms/token (~14-15 t/s) and stays flat
as the context grows to 8K. Single-GPU CUDA serving uses the ordered
fallback: requests decode one at a time, per-request TPOT does not degrade
under concurrent load, and TTFT grows with queue depth instead.

## Benchmarking notes

When driving the server with external tools, send text prompts. Two
pitfalls were observed on October 29, 2026:

- Integer-array prompts (`"prompt": [123, 456, ...]`, the OpenAI token-ID
  form) are not decoded as token IDs; a 1,000-int prompt was read as 9
  tokens.
- Random-token synthetic text re-tokenizes very differently under the
  GGUF tokenizer than under the HuggingFace tokenizer of the NVFP4
  release (a 100,000-token HF text measured 77 tokens server-side), so
  length-targeted random datasets under-report the real prompt size.
  Real prose round-trips within about 1% (100,000 → 100,775).
