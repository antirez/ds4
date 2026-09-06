# ROCm Q4_K decode K1024: four-lane candidate

**Experimental, opt-in; no AMD speedup or GPU parity demonstrated.**
Base reviewed: `96f5b463`. Local validation used an Apple Silicon Mac without
HIP. No remote tester was contacted.

## Scope and arithmetic

Only standalone Q4_K dense calls with **K=1024, M=32768, N=1..8** on AMD
wave32/wave64, outside quality mode, can select the new kernel. The public
entry is `ds4_gpu_matmul_quant_tensor` with weight type 12. Metal, CUDA,
prefill, paired, grouped and MoE kernels are unchanged.

The candidate maps 256 threads to 64 four-lane rows instead of 32 eight-lane
rows: 512 rather than 1024 workgroups per token. This is geometry, not a
measured throughput gain. Q8_K quantization, scratch, default stream, block
dot, per-lane accumulation and K-loop order remain unchanged. No SSD I/O,
weight preload, F16 sidecar or quantization change is introduced.

The output-lane reduction remains `(v0 + v2) + (v1 + v3)`, preceded by
the canonical offset-4 addition of +0. The device helper explicitly uses
`__fadd_rn(v, 0.0f)`: signed zero and signaling NaNs must not be treated as
identities. The [HIP 7.2 math API](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/reference/math_api.html)
specifies round-to-nearest, ties-to-even for `__fadd_rn`; that API contract
does not establish fast-math codegen or GPU bitwise parity.
Remaining shuffle offsets are 2 and 1, width 4. Masks are formed
in 64 bits before shifting, including wave64 bits 60..63.

GPU compiler contraction, operand order, denormals and NaN payloads still
require verification. Run the bitwise oracle with production and strict-math
builds; inspect generated instructions, registers and spills. A failing
bitwise test is not grounds to relax the oracle or promote the candidate.

## Controls

All three flags use **presence semantics**, including empty and `0`:

- `DS4_ROCM_ENABLE_Q4_DECODE_LANE4`: opt in for eligible calls; otherwise
  retain the existing path. Unset is off.
- `DS4_ROCM_DISABLE_Q4_DECODE_LANE4`: authoritative rollback.
- `DS4_ROCM_REQUIRE_Q4_DECODE_LANE4`: opt in and require selection.
  Disabled, excluded shape, quality mode or unavailable runtime returns 0.

REQUIRE is resolved before weight lookup, scratch allocation or quantizer
enqueue. Existing tensor/model bounds validation remains. Only REQUIRE is
read outside the narrow shape; environment values are not cached.
Quality mode conservatively uses legacy with ENABLE and fails with REQUIRE.

REQUIRE is a **focused standalone-call diagnostic**, not a model-wide flag:
other Q4 shapes, including prefill, intentionally fail. Use ENABLE/DISABLE
with REQUIRE unset for full-model A/B.

The same-thread launch counter increments only on successful candidate
submission. No new atomics or counter updates occur on default/rollback.
It attests selection, not GPU completion or graph replay; synchronized
readback verifies output separately.

## Validation status

- Host: **PASS**, 36,960 policies, 16 complete N/wave geometries and 28,561
  reduction cases, also under ASan/UBSan.
- Focused oracle: **PASS, host syntax-only**, warnings treated as errors.
  HIP-only event/device branches and device kernels remain uncompiled locally.
- ROCm test/bench targets: **SKIP without hipcc**, not a GPU pass.
- HIP compilation, AMD bitwise parity, timing and model t/s: **pending**.

The host model uses one non-inlined scalar addition for both reduction trees
to avoid host vectorization choosing different NaN payloads. This models the
tree only; the GPU oracle still requires equality of every output bit.

The focused oracle covers all N=1..8 with finite and signed-zero activations;
N=1/8 also use exceptional Q4 weight scales (zero/subnormal/Inf/NaN).
Non-finites enter via weights, not undefined CPU casts in a quantizer model.
It checks prefix/suffix guards, overwritten output bodies, unchanged inputs,
launch counts, actual default, REQUIRE-only, rollback precedence, quality,
short tensor/model ranges and excluded N=9/K=768/M=32767.

## Tester commands

Use the actual tester architecture; gfx1151 below is only an example.
Test both wave32 and wave64 hardware before broadening or promoting.

```sh
make test-rocm-q4-decode-host
make test-rocm-q4-decode-lane4 ROCM_ARCH=gfx1151 DS4_TEST_REQUIRE_ROCM_DEVICE=1
make bench-rocm-q4-decode-lane4 ROCM_ARCH=gfx1151 DS4_TEST_REQUIRE_ROCM_DEVICE=1

# Custom native benchmark, without Python:
make tests/test_rocm_q4_decode_lane4 ROCM_ARCH=gfx1151
DS4_TEST_REQUIRE_ROCM_DEVICE=1 ./tests/test_rocm_q4_decode_lane4 --bench \
  --tokens 1,2,3,4,5,6,7,8 --samples 32 --sets 4 --warmup 4 --iterations 1
```

The binary returns 77 on missing HIP/device, or 1 with REQUIRE_DEVICE=1.
The Make targets report SKIP or enforce failure accordingly.

## Benchmark interpretation

`--bench` skips the broad domain/exceptional suites. It prechecks each
requested N/set against forced rollback using bitwise output and guards,
then warms both arms. ABBA/BAAB quartets rotate weight sets, with both orders
for each set; samples must be a multiple of 2*sets. Defaults: N=1,2,4,8,
8 quartets, 4 sets, 4 warmup rounds, 1 API call per timed arm.

HIP events bracket public API calls on the unchanged default stream.
Environment updates, counter queries, input/output readback and guard checks
are outside timing. Every measured arm is independently poisoned and checked.
The API retains its selection/launch overhead and B's thread-local increment;
event times include Q8_K quantization and can include host-submission idle
time. These are not matmul-only, SSD or end-to-end token-throughput timings.

Raw rows include order/set/N/arm, milliseconds and selected launches.
Summaries show mean/median latency, paired geometric A-time/B-time (>1 favors
B), wins and pair min/max; min/max is not a confidence interval.
More iterations reuse weights within an interval and change cache warmth.
Require repeatable model A/B gains as well as strict device parity.
