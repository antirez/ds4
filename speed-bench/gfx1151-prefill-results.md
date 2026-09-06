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
