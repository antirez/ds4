# gfx1151 DeepSeek V4 Flash

## Scope

- Hardware: Framework Desktop, AMD Strix Halo `gfx1151`, 128 GB unified memory
- Runtime: ROCm 7.14
- Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
- Prompt: `speed-bench/promessi_sposi.txt`
- Workload: pure prefill, 2048-token increments, ending at 4096 context

## Result

| Build | 2K prefill | Warm 4K prefill |
|---|---:|---:|
| Clean DS4 | - | 187.26 tok/s |
| Tuned gfx1151 path | 231.91 tok/s | 292.86 tok/s |

The warm 4K result is 56.4% faster than the clean baseline. The final run had
all tuning selectors unset; gfx1151 now selects this path by default.

## Retained changes

- Use the tuned x64/y64 IQ2 MMQ gate/up path and bound each expert by the input
  token count.
- Use shape-specific wave32 rocWMMA kernels for the hot Q8 and F16 projections.
- Use 32-head, 80-key rocWMMA attention tiles with vectorized F32-to-F16
  staging, plus the specialized attention-output B projection.
- Vectorize Q8 activation staging and stage each Q2_K weight block once per
  256-value K slab.

The environment variables remain as per-path debug overrides: setting one to
`0` disables that path. They are not required for normal gfx1151 execution.

## Validation

- The 512, 1024, 2048, and 4096 frontier logits are byte-identical to the
  previously validated 80-key attention reference.
- That reference preserves top-1 at all four frontiers against the earlier
  accepted stack. Its worst full-logit difference is max-abs 4.446 and RMSE
  0.770, inside the accepted max-abs 5.14 and RMSE 0.871 envelope.
- ROCm 7.14 builds and links `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and
  `ds4-agent`; all five pass runtime help smoke tests.
- Answer-extractor self-tests pass. Q4_K and MXFP4 dot tests pass 4/4 each.

## ROCm 10.0 DSpark scheduler follow-up

The adaptive scheduler is enabled by default on gfx1151. It probes four speculative cycles and requires an average of four accepted tokens per cycle. Below that floor, the rest of the request bypasses the support model and uses ordinary target decode; the next prompt sync resets the decision.

| Workload | Aggregate generation | Steady generation | Scheduler result |
|---|---:|---:|---|
| 2K, low acceptance, 128 tokens | 15.40 tok/s | 15.91 tok/s | Bypass after four cycles |
| 16K, 100% acceptance, 128 tokens | 25.71 tok/s | 26.77 tok/s | Keep DSpark active |

The ordinary 2K control is 15.57 tok/s. The low-acceptance bypass therefore limits DSpark overhead to 1.1% while preserving the high-acceptance speedup.

The experimental compact IQ2 worklist was removed after a small-context GPU memory fault. The retained stock launch with the x64/y64 tile measures 294.38 tok/s at 4K and 268.51 tok/s at 16K in the 2K-step pure-prefill sweep.

`ds4-bench` restores local DSpark sweep frontiers from session snapshots. Restored and pure-prefill frontier logits are byte-identical at 2K and 4K; the restored 4K interval is 295.26 tok/s versus 295.27 tok/s pure.

## November 2026 prefill pass (ROCm 7.14, hipBLASLt)

Same workload as above: pure prefill in 2048-token steps, measured with
`ds4-bench --rocm --prompt-file speed-bench/promessi_sposi.txt --gen-tokens 0
--ctx-start 2048 --ctx-max 16384 --step-incr 2048`, model
`DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf` (text
only), inside the `ds4:rocm-gfx1151` container.

| Context after the step | Before | After |
|---:|---:|---:|
| 2048 (cold first chunk) | 160.41 tok/s | 347.28 tok/s |
| 4096 | 272.14 tok/s | 427.01 tok/s |
| 8192 | 264.60 tok/s | 420.84 tok/s |
| 16384 | 250.90 tok/s | 409.54 tok/s |

### Retained changes

- Every prefill-sized dense projection (q_a, q_b, kv, shared expert,
  attention output A/B, hc mix, indexer and compressor projections) runs as
  an F16 hipBLASLt GEMM.  The library's heuristic order is not reliable on
  gfx1151 (its first F32-output choice can be 4x slower than the best of the
  first eight), so each shape class times the candidates once and keeps the
  fastest; the choice is shared across power-of-two token buckets.  The
  rocBLAS `GemmEx` F32-output path (4-5 TFLOPS) and the custom Q8/F16 WMMA
  row-tile kernels (5-10 TFLOPS) are gone; hipBLASLt reaches 20-27 TFLOPS
  with F32 outputs and 33-42 with F16 outputs at these shapes.
- The F16 staging buffer is allocated on ROCm too, so q_b, the attention
  output and the shared-expert down projection write F16 straight into the
  half-input norm/rope and hyper-connection kernels.  The transposed F16
  weight cache and its 74 ms per-layer cold dequant are removed.
- One 32-head WMMA attention kernel serves every prefill chunk: the first
  chunk of a prompt used to run the scalar two-pass kernel (40 ms per layer).
  The kernel stages each 32-key tile once for both Q.K^T and P.V, keeps Q in
  registers, splits Q.K^T over all 16 waves, keeps the P.V accumulators in
  registers with per-row online-softmax rescaling, and prefetches the next
  tile.  Indexed layers: 32.6 -> 19.7 ms per 2048 tokens.
- The Q2_K routed down projection runs as a tiled F16 WMMA kernel (64 tokens
  x 128 rows per block, 64-wide K chunks, raw superblocks staged in LDS,
  activations prefetched); the SwiGLU writes its F16 input directly.  The
  gate/up pair stays on the vendored MMQ kernels without the redundant
  sanitize passes.  Down: 32 -> 18 ms per layer.
- Indexer scores keep the ReLU-weighted head sum in the WMMA accumulator
  registers and stage four heads per barrier: 6.9 -> 1.2 ms per layer at 4K
  context, and the cost no longer dominates long contexts.
- The deterministic sorted-pair scatter runs 256 threads per expert.

### Validation

- Frontier logits at 2048, 4096 and 6144 tokens against the previous image's
  binary: top-1 identical at all three frontiers, top-5 identical at two and
  4/5 at the third, max-abs 0.8-1.7 and RMSE 0.15-0.30 — inside the accepted
  max-abs 5.14 / RMSE 0.871 envelope.  Builds that only changed the WMMA
  attention or down tiles differ from each other by the same amount because
  the hipBLASLt algorithm choice can differ between processes.
- The vendored MMQ Q2_K kernel was tried for the down projection and
  rejected: 258 ms per layer on gfx1151.  `ROCM_MMQ_Y=128` gave no gain
  over 64.
