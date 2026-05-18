# GB10 no-MTP vs exact MTP speed-bench

Measured on DGX Spark / GB10 with the q2-imatrix DeepSeek V4 Flash GGUF and
the Q4_K MTP GGUF.  The sweep uses `speed-bench/promessi_sposi.txt`, 2048-token
frontiers through 65536 context tokens, and 128 greedy generation tokens per
frontier.

No-MTP:

```sh
./ds4-bench --cuda \
  -m /home/ent/models/antirez-q2/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv speed-bench/mtp-compare-2026-05-14/gb10_nomtp.csv
```

Exact MTP:

```sh
DS4_CUDA_MTP_TOP2=1 \
DS4_CUDA_MTP_VERIFY_TOP2=1 \
./ds4-bench --cuda \
  -m /home/ent/models/antirez-q2/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --mtp /home/ent/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft 2 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv speed-bench/mtp-compare-2026-05-14/gb10_exact_mtp.csv
```

Summary:

| Run | Prefill mean | Generation mean | Generation first | Generation last |
| --- | ---: | ---: | ---: | ---: |
| no-MTP | 342.20 t/s | 12.68 t/s | 14.00 t/s | 11.63 t/s |
| exact MTP | 341.02 t/s | 12.62 t/s | 12.53 t/s | 11.68 t/s |

Exact MTP is effectively tied with no-MTP in this full-context sweep. It is
behind on the first cold row and slightly ahead at the largest contexts.
