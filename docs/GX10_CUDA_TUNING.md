# DGX Spark / GB10 CUDA tuning notes

These notes record the CUDA tuning pass run on an ASUS Ascent GX10 / NVIDIA
GB10 with another GPU workload (`sd-server`) left resident. The goal was to
improve generated tokens/sec without materially hurting prefill or increasing
memory-crash risk.

## Baseline

The baseline used `ds4-bench` with the public-domain *I Promessi Sposi* prompt:

```sh
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 32768 --step-incr 2048 \
  --gen-tokens 128 \
  --csv tuning/gx10-20260514/baseline_default.csv
```

With `sd-server` using about 10.5 GiB GPU memory, the baseline averaged:

- prefill: 366.99 t/s
- generation: 13.16 t/s

## Accepted changes

- GB10/sm_121 now uses the regular unordered one-token F16 matmul path by
  default. The F16 pair helper keeps the same GB10 choice and fuses the two
  one-token projections into a single unordered CUDA launch. The previous
  ordered F16 decode kernels remain available with
  `DS4_CUDA_FORCE_ORDERED_F16_MATMUL=1`.
- Several CUDA hot-path environment switches are cached during backend
  initialization instead of being read repeatedly during decode.
- CUDA uses `DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=0` by default. The old split
  default is kept on Apple/Metal, where it is useful for command scheduling.

The full env-equivalent sweep for the unordered F16 path improved average
generation from 13.16 t/s to 13.89 t/s while keeping prefill effectively flat.
After the F16 pair helper was aligned with the same GB10 decision, the local
8k/64 check measured 14.44 t/s generation. A post-rebase check on the updated
main measured 14.26 t/s generation at the same 8k/64 frontier. Fusing the
unordered F16 pair path then measured 14.50 t/s versus 14.31 t/s for the
two-launch fallback at 8k/64, with the same 392 t/s-class prefill.

## Rejected knobs

The following local knobs were tested and were slower or not useful for this
machine:

- `DS4_CUDA_SERIAL_ROUTER=1`
- `DS4_CUDA_NO_WARP_ROUTER_SELECT=1`
- `DS4_CUDA_MOE_NO_DECODE_LUT_GATE=1`
- `DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6=1`
- `DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR=1`
- `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=72`
- `DS4_CUDA_DIRECT_MODEL=1`
- `DS4_CUDA_DISABLE_QKV_RMS_FUSED=1`
- `DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN=5`
- `DS4_CUDA_Q8_F16_ALL=1`
- `DS4_CUDA_Q8_F32_LARGE=1`
- one-token logits top-k/argmax readback instead of full logits readback
- asynchronous CUDA `end_commands` without a per-token synchronize
- a decode Q/KV pair projection that reused one Q8 input quantization for
  `attn_q_a` and `attn_kv`; it measured 14.54 t/s versus 14.65 t/s for the
  existing separate projections at 8k/64
- an experimental one-token Q8->F16 cuBLAS decode path for the output logits;
  limiting it to `out_dim >= 65536` measured 13.57 t/s generation and was
  slower than the native Q8 matvec path

## 20 t/s assessment

The current release path is not one micro-kernel away from 20 t/s on this GB10
setup. With `sd-server` stopped, a short CLI run measured about 16.3 t/s and
`DS4_METAL_GRAPH_TOKEN_PROFILE=1` reported steady decode tokens around 60-62 ms:

- encode/enqueue: 21-26 ms
- GPU drain/synchronize: 35-39 ms
- logits readback: about 0.02 ms

That makes full-logit readback and sampling irrelevant for throughput. It also
means kernel-local changes must remove about 10 ms/token to reach 20 t/s, while
the largest individual CUDA hot spots are spread across routed MoE,
attention-output projection, Q path, and the generic Q8 matvecs. The local
attempts above did not find a safe kernel-level change with that scale of gain.

The plausible route to 20 t/s is reducing per-token launch/enqueue overhead,
most likely with CUDA Graph replay or an equivalent persistent decode executor.
This is a larger architectural change, not a small backend knob: the current
decode tape passes token, position, raw-cache row, compressed-cache frontiers,
and pointer-swapped HC buffers as host-side control state, and compressed
layers change behavior at ratio frontiers. A graph implementation would need to
move those dynamic values into device-resident parameter/state buffers, or
maintain a small set of graph variants for HC-buffer parity and compression
frontiers. If that work can cut the 21-26 ms encode/enqueue component roughly in
half, 20 t/s is realistic; without that class of change, the measured ceiling of
the current design is closer to the mid/high 16 t/s range in the interactive
CLI and mid 14 t/s in the conservative 8k/64 bench.

## Experimental graph-decode branch

The `gx10-cuda-graph-decode` branch starts from the stable tuning branch and is
reserved for heavier CUDA decode work. The first accepted experiment is a
GB10-only cached-input variant of the one-token Q8 matvec for wide projections
(`out_dim >= 1024`). It copies the already-quantized activation row and scales
to shared memory once per CUDA block, then lets the eight row warps reuse that
block-local copy. This targets repeated `xq` traffic in `attn_q_a`, `attn_q_b`,
and the output logits projection without touching Metal or batched prefill.

Local 8k/64 checks measured:

- cached wide Q8 default on GB10: 14.49-14.53 t/s generation
- disabled with `DS4_CUDA_NO_Q8_CACHE_X=1`: 14.42-14.46 t/s generation

Rejected variants from the same pass:

- applying cached `xq` to the Q8 pair matvec used by shared gate/up; it measured
  14.42 t/s versus 14.46 t/s for the existing pair path
- applying cached `xq` to the HC-expand Q8 path; it was effectively neutral
- a 16-row half-warp Q8 matvec variant; it measured 14.43 t/s and 16.08 t/s in
  the CLI check, both below the existing warp-per-row path
- a session sampling scratch buffer to avoid recomputing default `top_p=1`
  probabilities; it did not move the sampled CLI throughput enough to keep

## Fork scan

The relevant fork ideas were:

- `spmurrayzzz/ds4`: CUDA decode fast-path experiments. The CUDA no-split idea
  was safe to port; the larger patches target an older baseline and overlap
  with current upstream work.
- `reffdev/ds4` branch `fused-matmul-mtp`: MTP/tiny-batch fused matmul work.
  It is useful background for MTP, but not for the current no-MTP greedy decode
  bottleneck.
- `Entrpi/ds4-on-spark`: useful GX10/Spark methodology notes, including the
  difference between first-token-inclusive CLI/bench numbers and pure
  steady-state decode measurements.

## MTP

The optional MTP GGUF loaded and ran, but did not help on this setup. With
`--mtp-draft 2`, a 4k/64 CLI session test measured 3.66 t/s with speculative
decode disabled and 3.70 t/s with speculative decode enabled. Timing logs showed
frequent first-draft misses or one-token commits and verifier costs around
72-139 ms per speculative step, so MTP is not recommended for GB10 production
use yet.

## Verification

The branch was verified with:

```sh
make cuda-spark CUDA_HOME=/usr/local/cuda-13.0
make ds4_test CUDA_HOME=/usr/local/cuda-13.0
./ds4_test --server
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 8192 \
  --gen-tokens 64
```
