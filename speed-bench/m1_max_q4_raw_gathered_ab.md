# M1 Max Q4 raw-gathered attention A/B

Date: 2026-08-22

Hardware: Apple M1 Max, 32 GiB RAM. Backend: Metal with SSD streaming.
Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ4-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`.

The control is the new default raw-gathered path. The candidate sets
`DS4_METAL_DISABLE_DECODE_RAW_GATHERED_ATTN=1` and restores the legacy raw-only
attention path. Both use the same decode split schedule.

## Correctness

- 16-step greedy top-20 logprob dumps are byte-identical.
- Both files have SHA-256
  `7ee7b8a119f8f93c4ea91fb27867eb4563037e34fc8c61ca795a0953ace61738`.
- Each alternating run compared 145 full-vocabulary frontiers: 18,745,600
  float logits and 144 non-EOS selections were bit-identical.

## Alternating same-process results

| Prefix | Variant | Steady tokens/s | Delta |
|---:|---|---:|---:|
| 128 | raw-gathered | 6.7360 | +1.52% |
| 128 | legacy raw | 6.6350 | baseline |
| 2048 | raw-gathered | 5.4325 | +0.66% |
| 2048 | legacy raw | 5.3967 | baseline |

Command shape:

```sh
speed-bench/metal_decode_schedule_bench \
  -m /path/to/model.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --prefix-tokens 128 --ctx 300 \
  --warmup 16 --tokens 128 \
  --candidate-env DS4_METAL_DISABLE_DECODE_RAW_GATHERED_ATTN \
  --include-selection --ssd-streaming
```

Forcing `DS4_METAL_ENABLE_GATHERED_KV_STAGE=1` on the M1 was neutral:
6.4763 tokens/s versus 6.4769 tokens/s control (-0.01%), with exact logits.
It therefore remains automatic only on its existing device policy.

The packed32 raw extension is retained for eligible devices, but the M1 does
not arm the inverse-RoPE fusion required by that kernel, so it was not selected
in these measurements.
