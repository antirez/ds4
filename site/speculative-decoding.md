# Speculative Decoding

DwarfStar supports three speculative decoding approaches:

| Approach | Models | Flag |
| --- | --- | --- |
| DSpark | DeepSeek V4 Flash 0731 | `--dspark` |
| GLM MTP | GLM 5.3 Flash | `--mtp` |
| Qwen MTP | Qwen3.8 Flash Next | `--mtp` |

## DSpark

```sh
./download_model.sh ds4f-dspark
./ds4 --dspark --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf
```

| Prompt | Temp | Ordinary | DSpark |
| --- | ---: | ---: | ---: |
| C hash table | 0 | 19.72 t/s | 31.41 t/s |
| C hash table | 1 | 19.53 t/s | 29.98 t/s |
| Prose | 1 | 19.53 t/s | 18.81 t/s |

Poor draft acceptance can make DSpark slower. Does not accelerate prefill.

## GLM MTP

Built into GLM 5.3 Flash:

```sh
./ds4 --mtp
```

No separate download needed.

## Qwen MTP

Built into Qwen3.8 Flash Next:

```sh
./ds4 --mtp --ctx 32768
```

On DGX Spark: 17.1 t/s ordinary → 19.9 t/s with MTP (temp 0).

## Exact Sampling

```sh
./ds4 --mtp-exact-sampling
```

Matches non-speculative output token-by-token.