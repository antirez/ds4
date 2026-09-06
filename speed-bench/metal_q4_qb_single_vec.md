# Metal Q4 Q-b: experimental single-token vector weights

Base: `96f5b463`, Apple M1 Max / 32 GiB, 2026-09-05. **Opt-in only.**
The isolated kernel is faster, but the measured SSD model runs do not establish
a throughput improvement. No CUDA/ROCm machine was contacted.

## Scope

The token-pair implementation is instantiated with one token per threadgroup:
explicit `ushort4` packed-weight loads and `half2` scale loads, one set of
accumulators, and the classic scalar dot/reduction order. No conversion,
weight sidecar, extra SSD read, threadgroup barrier or larger workgroup is
introduced. The existing two-token instantiation remains the default for its
established N=2..8 scope.

Runtime admission is standalone Q4_K K=1024/M=32768/N=1, eight-byte-aligned
weights, M1 family, quality off and no active TP service (including suspended
expert sharding during an MTP draft). Measurements cover **M1 Max only**.
Other calls retain their prior dispatch; unavailable pipelines fall back.

- `DS4_METAL_ENABLE_Q4_QB_SINGLE_VEC=1`: request the candidate.
- `DS4_METAL_DISABLE_Q4_QB_SINGLE_VEC=1`: dominant rollback.

Both are presence controls: even `0` or an empty value counts as set. The
classic-matvec disable flag remains authoritative. The new controls do not
alter N=2..8 token pairing. Test-only atomic counters record candidate host
submissions, not device completion.

## Kernel measurements

Two independent 32768x1024 projections per logical invocation, one token,
two SIMDgroups and two rows per SIMDgroup. Both arms use two dispatches.
Times below are **microseconds per pair of projections**, not per token of
the model. Sixteen adjacent AB/BA samples per arm, 64 logical invocations per
sample, two warmup samples per arm, GPU command-buffer timestamps.

| Fixture | Original median | Vector median | Time change |
| --- | ---: | ---: | ---: |
| Pilot, repeated 36 MiB weights | 244.042 | 214.227 | -12.22% |
| Eight rotating sets, 288 MiB | 236.726 | 208.240 | -12.03% |
| Rotating sets, re-poison/verify every sample | 235.046 | 207.408 | -11.76% |

The final fixture poisons all timed outputs before every sample, checks every
set against the independent untimed legacy-pair oracle after completion,
and rechecks input/weight hashes. The earlier two measurements had bitwise
and guard checks, but not per-sample re-poisoning.

## Model-backed SSD measurements: not an accepted speedup

Checkpoint: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ4-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`.
It has Q4 attention projections, IQ2/Q2 routed experts and Q8 shared/output
weights; this is **not** an all-Q4 versus all-Q8 comparison. Both arms keep
768 dynamic expert slots (5.06 GiB), the same checkpoint and cache policy.

The same-engine oracle used `promessi_sposi.txt`, a 128-token prefix, ctx=300,
16 warmup and 128 measured decode steps per arm, split schedule 2/32 and
selection included. In each process all 145 full-vocabulary frontiers
(18,745,600 floats) and 144 selected tokens matched bitwise. Candidate
coverage was 6,192 calls versus zero for rollback.

| Process | Original t/s | Vector t/s | Change |
| --- | ---: | ---: | ---: |
| Enable flag, normal labels | 6.3884 | 6.5296 | +2.21% |
| Disable flag, reversed labels | 5.3168 | 5.2027 | -2.15% |

The two sessions share an expert cache. Alternating arms does not eliminate
the interaction between token-dependent misses and which session runs first.
These runs are the exact-logit gate, **not sufficient SSD performance proof**.

Four additional fresh CLI processes used A1/B1/B2/A2 order, ctx=512,
prefill-chunk=32, temperature=0, seed=1, `--nothink`, and `-n 128` with:

> Spiega in dettaglio come progettare gli indici di un database con molte letture e memoria limitata. Includi esempi e compromessi.

The prompt was 44 input tokens. All four stdout files were byte-identical.

| Arm | Prefill t/s | Generation t/s |
| --- | ---: | ---: |
| A1 original | 4.84 | 4.51 |
| B1 vector | 4.58 | 4.44 |
| B2 vector | 5.03 | 4.74 |
| A2 original | 4.91 | 4.98 |

Generation means are 4.745 original and 4.590 vector (-3.27%). These are only
two pairs and CLI rates are rounded; they do not certify a regression across
workloads, but clearly do **not** justify enabling the candidate by default.
Prefill differences are noise for this fixture: neither 32 nor 12 tokens
selects the single-token candidate. No SSD bandwidth or temperature was measured.

Raw model and rotating-kernel logs are in
`/private/tmp/ds4-q4-single-vec.ZAVjvl/` on the development Mac; temporary
files are not shipped. The model timing binary `ds4` SHA-256 was
`98594c5dc3df51394aca53e2ccf54f34d08f5d42e67598d3760e634eb305564a`.
The final rebuild additionally excludes suspended TP; ordinary non-TP
arithmetic/admission is unchanged, but it is not that exact binary hash.

## Native reproduction

```sh
make test-metal-q4-qb-token-pair
make bench-metal-q4-single-vec
make metal-decode-schedule-bench
./speed-bench/metal_decode_schedule_bench \
  -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --prefix-tokens 128 --ctx 300 --warmup 16 --tokens 128 \
  --candidate-env DS4_METAL_ENABLE_Q4_QB_SINGLE_VEC \
  --include-selection --print-step-times \
  --ssd-streaming --ssd-streaming-cache-experts 768
```

Unset the new DISABLE flag for this command. The harness rejects zero
candidate coverage. For the inverse comparison, keep ENABLE=1 and use
`--candidate-env DS4_METAL_DISABLE_Q4_QB_SINGLE_VEC`; **control then means
vector and candidate means original**. No Python is needed.

The final runtime oracle passes 720 cases, including coexistence with token
pairing, quality, SSD spans, stream switching, normal/suspended TP and the
presence-valued rollbacks. The kernel run passes 213 cases, including the
production-size rotating fixture and existing Q4/Q8 coverage. Physical odd
weight rows in generic oracle cases retain legacy-loader fixture padding;
this does not establish safety of that old loader on unpadded odd rows.
