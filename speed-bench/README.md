## Benchmarking

For the single-GB10 tiled Q8_0 activation producer used by eligible Q4/Q8
prefill, see [the native three-way oracle and timing protocol](cuda_q8_prefill_tiled.md).
The new prefill kernel preserves decode dispatch; GPU validation is pending.

For the gfx1151 Q4 F32 `attn_q_b` RMSNorm/RoPE epilogue candidate, see
[scope, native parity tests and default/rollback timing](rocm_q4_qb_f32_epilogue.md).
The change is default-on only for the documented F32-GEMM prefill scope;
GPU validation and model-throughput evidence are still pending.

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

### Metal decode schedule A/B

Build the balanced, same-engine Metal decode comparison with:

```
make metal-decode-schedule-bench
./speed-bench/metal_decode_schedule_bench \
  -m ds4flash.gguf \
  --include-selection
```

The harness prefills two sessions and alternates both variant order and
variant-to-session assignment. It aborts unless every full-vocabulary logit
row is bit-identical and, with `--include-selection`, both variants select the
same non-EOS token. Use `--candidate-env NAME` to measure a rollback control,
or `--help` to compare explicit split schedules. Pass `--ssd-streaming` for a
model larger than RAM; the harness then skips full-weight warmup while keeping
both variants in the same engine and expert cache. SSD runs can use
`--ssd-streaming-cold`, `--ssd-streaming-cache-experts N`, and
`--ssd-streaming-preload-experts N` to hold the cache policy constant.

This paired harness is the exact-logit gate. Since its two sessions share the
expert cache, confirm SSD throughput separately with one process and engine per
variant before promoting a scheduling change. Environment variables consumed
while the engine opens, including the Metal streaming `F_NOCACHE` controls,
also require separate processes: `--candidate-env` changes them too late to
create a different model descriptor inside this harness.

The experimental single-token Q4 Q-b vector loader has a native kernel test:

```sh
make bench-metal-q4-single-vec
make test-metal-q4-qb-token-pair
```

The kernel benchmark rotates eight pairs of production-size weights (288 MiB),
checks bitwise outputs, and compares adjacent AB/BA samples. Runtime selection
requires `DS4_METAL_ENABLE_Q4_QB_SINGLE_VEC=1`; any defined
`DS4_METAL_DISABLE_Q4_QB_SINGLE_VEC` rolls it back. The same-engine benchmark
checks dispatch counts for either flag used with `--candidate-env` and can
print per-step durations with `--print-step-times`. When using the DISABLE
flag, keep ENABLE set and remember that the labels are reversed: `control`
is the new kernel, `candidate` the rollback. See
[the measurements and limitations](metal_q4_qb_single_vec.md) before treating
kernel latency as an SSD tokens/s gain.

To compare the default pre-M5 ratio-4 compressor pack/transpose fusion with the
legacy decode path, including token selection, use:

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION \
  --include-selection \
  --tokens 1024
```

### Metal prefill variant A/B

Build the balanced prefill comparison. To compare the default resident pre-M5
MXFP4 pair tail-SIMDgroup cull against the original pair kernel, make the
rollback path the candidate:

```
make metal-prefill-variant-bench
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL
```

To isolate the default routed-down tail-SIMDgroup cull from the retained pair
default, use its down-specific rollback as the candidate:

```
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_DOWN_TAIL_SIMDGROUP_CULL
```

The harness uses one Metal engine and fresh sessions for every run. It warms
both variants with at least 32 tokens, alternates control/candidate order in
ABBA and BAAB blocks, poisons host logit buffers before copying, and aborts
unless every final full-vocabulary logit row is bit-identical. Defaults are an
8192-token prefix, an automatically sized 8193-token context, and two repeats;
use `--help` to override them. SSD-prefill variants can add `--ssd-streaming`,
`--ssd-streaming-cold`, `--ssd-streaming-cache-experts N`, and
`--ssd-streaming-preload-experts N`. Since those paired runs intentionally
share one engine and expert cache, confirm any SSD throughput win separately
with one process and engine per variant before promotion.
Environment variables consumed while the engine opens, including the Metal
streaming `F_NOCACHE` controls, cannot be compared with `--candidate-env` in
this harness and likewise require separate processes.

### Metal Q4_K generic-MM tail cull

Build and run the GGUF-free kernel-only comparison with:

```
make metal-q4-mm-tail-cull-bench
./speed-bench/metal_q4_mm_tail_cull_bench
```

The default `4096 -> 1024` shape models a Flash Q-A projection. Use
`--in-dim 1024 --out-dim 32768` to measure `attn_q_b`. Both arms dispatch the
checked-in production Metal kernels with resident rotating Q4_K weights and
GPU timestamps. The harness covers `N=9,16,17,31,33,47,63,65`, alternates
ABBA/BAAB, and requires bit-exact outputs plus intact input/output canaries.
No GGUF access, SSD I/O, upload, readback, or CPU wall time is included in a
measured command buffer.

### Metal Q4_K attention output-A direct routing

Build and run the production-shape, resident GPU comparison with:

```
make metal-q4-attn-out-a-direct-bench
./speed-bench/metal_q4_attn_out_a_direct_bench \
  --n 512,1024,2048,4096 --samples 8 --warmup 2
```

The fixed `4096 -> 1024 x 8` geometry compares the current map-plus-routed
path, the routed kernel with a prebuilt map, and the fixed-route direct kernel
on the production Apple M1–M4 and N=512–4096 scope. All three pairwise
comparisons are scheduled in balanced ABBA/BAAB blocks and timed only with Metal
GPU start/end timestamps. The harness requires all three full outputs to be
bit-identical, hashes immutable weights, heads, ids, and the prebuilt map, and
checks prefix/suffix canaries around every allocation. Fixture construction,
map prebuilding, and validation are outside the samples. There is no GGUF,
SSD I/O, model upload, GPU readback, or CPU wall timing in measured command
buffers, so these numbers isolate the kernel and routing overhead rather than
SSD-streaming noise.

### Metal resident IQ2/Q2 routed MoE

Build the production top-6 tail-cull fixture or the GLM-shape top-8 pair-fusion
fixture with:

```
make metal-iq2-moe-tail-cull-bench
./speed-bench/metal_iq2_moe_tail_cull_bench --samples 12 --warmup-cycles 2

make metal-iq2-moe-top8-pair-bench
./speed-bench/metal_iq2_moe_top8_pair_bench --samples 12 --warmup-cycles 2
```

Both fixtures keep synthetic IQ2_XXS gate/up and Q2_K down weights resident,
alternate variants in ABBA/BAAB order, and report per-stage GPU timestamps.
The top-8 fixture matches the `4096 -> 2048 -> 4096`, 288-expert GLM routed
MoE geometry. Its baseline launches gate and up separately; its candidate
uses the grouped pair-SwiGLU kernel. It requires bit-exact F16 mid rows, F32
expert rows and final output, plus intact allocation canaries. The measured
stages contain no GGUF loading, file-backed model mapping, application SSD
reads, uploads, readback, or CPU wall timing. The anonymous fixture is fully
touched before oracle and warm-up, so a change in these GPU timestamps is
attributable to the kernel path rather than SSD streaming.

The automatic top-8 dispatch is intentionally limited to the measured
M1 Max, resident, 4096-token shape. SSD streaming and other batch/device
shapes keep the separate gate/up path until they have their own A/B data.

### Resident ROCm Q4_K prefill

Build the production-dispatch A/B harness on a ROCm host with:

```
make rocm-q4-prefill-bench ROCM_ARCH=gfx1151
./speed-bench/rocm_q4_prefill_bench
```

The fixture copies four rotating sets of synthetic GGUF-layout Q4_K weights
to device memory before warmup and forces SSD streaming off. HIP events then
measure only activation conversion/quantization and projection kernels. The
row-geometry and activation-loader comparisons use resident pointers plus an
explicit enqueue-only hook, excluding environment parsing, model lookup, and
policy selection. The
comparisons are:

- `lds`: original versus scalar-streaming LDS copy/fence schedule, with
  vector copies disabled on both arms;
- `lds_vector`: scalar streaming versus aligned four-word copies at K4096
  and the K1024 `q_b` shape, retaining the same barriers and tile geometry.
  See [the vector-copy tester notes](rocm_q4_prefill_lds_vector.md).
- `dense`: legacy versus TILE8 at the Flash Q-A `K=4096,M=1024` shape;
- `pair`: two TILE8 calls versus the fused Q-A/KV
  `K=4096,M=(1024+512)` path;
- `qb`: TILE8 versus TILE4 at the production `attn_q_b`
  `K=1024,M=32768` shape.
- `outb`: TILE8 versus the compressed direct-Q4 WMMA kernel at the
  production `output_b` `K=8192,M=4096` shape;
- `output`: the complete grouped `output_a` plus `output_b` production API,
  comparing an all-TILE8 rollback with the production A-WMMA/B-TILE8
  pipeline.

On gfx1151 wave32, `dense` and `qb` also emit a direct-Q4 WMMA comparison for
`N>=256`. The candidate keeps Q4_K weights compressed, rounds each transient
32-value weight group and the activation tile to F16 in the kernel, accumulates
through WMMA in F32, and avoids both Q8_K activation scratch and persistent F16
weight sidecars. Its shape-selected row tile uses 64 rows below output dimension
1024, 128 below 8192, and 256 otherwise; the wider variants stage two adjacent
F32 activations into one F16 pair, matching the established Q8 WMMA loader.
`dense`, `outb`, and `output` also emit a bit-exact 64-row versus shape-selected
scalar-loader A/B. The large `q_b` shape instead reports adjacent scalar-loader
64→128 and 128→256 comparisons, so
register pressure in the 512-thread candidate cannot hide a better midpoint.
These direct comparisons measure the net effect of the broader row geometry,
including its changed workgroup and occupancy contract, separately from the
candidate's arithmetic change versus TILE8. `q_b` additionally compares scalar
versus two-wide activation staging at fixed 128- and 256-row geometry.
The direct hook receives both tile and loader explicitly, so every arm attests
its own configuration. Eligible standalone resident calls and attention-output
A use direct-Q4 WMMA by default; set `DS4_ROCM_DISABLE_Q4_PREFILL_WMMA=1` to
opt out. Attention-output B remains on Q8_K+TILE8. The production comparison
sets `DS4_ROCM_ENABLE_Q4_PREFILL_WMMA=1` without REQUIRE, which explicitly
retains that same A-WMMA/B-TILE8 policy. The hard B oracle replays TILE8 over
the same WMMA-low intermediate and requires bitwise-identical output. The composed
A-WMMA/B-TILE8 versus all-TILE8 delta remains visible as a non-gating
diagnostic because it includes A's deliberate F16 boundary rather than
isolating B correctness. Raw two-stage direct-WMMA row-geometry and K32/K64
comparisons remain diagnostic-only kernel measurements.

Eligible aligned 256-row direct-Q4 WMMA launches use the q_b-focused K128/P144
stage by default. It groups four adjacent Q4_K qgroups behind one barrier pair
and uses float4 activation loads, targeting a 2x reduction in synchronization
and activation-load instructions over K64 while retaining the same F16
conversions and accumulation order. Set
`DS4_ROCM_DISABLE_Q4_PREFILL_WMMA_K128=1` to roll the same launch back to
K64/P80; unset or explicitly false keeps K128. Incompatible alignment or
64/128-row geometry automatically uses K64.

Aligned 128/256-row K64 launches now have a float4 activation loader with
`DS4_ROCM_DISABLE_Q4_PREFILL_WMMA_K64_LOAD4=1` as the presence-based rollback
(even `0` or empty disables). This changes neither the staging geometry nor
WMMA arithmetic. The separate `--case wmma_load4` benchmark isolates LOAD2 vs
LOAD4, including grouped output-A, at fixed K64 geometry. Existing direct
K32/K64 benchmark hooks retain the original loaders. GPU validation and
timing are pending; see [scope and native tester commands](rocm_q4_prefill_wmma_load4.md).

K64/P80 stages two adjacent 32-value Q4_K groups and a 64-value activation
slice in one padded P80 LDS tile. Leave
`DS4_ROCM_ENABLE_Q4_PREFILL_WMMA_K64` unset or true to permit both K64 and the
eligible K128 default; set it to `0`, `false`, `no`, or `off` to restore K32
staging. `DS4_ROCM_DISABLE_Q4_PREFILL_WMMA=1` rolls the whole direct-WMMA path
back to Q8_K plus TILE8/TILE4. A persistent or automatic transient F16 q_b
projection bypasses the direct-Q4 candidates.

The q_b microbenchmark contains a strict, same-process
`q_b_wmma_k64_k128` A/B with bitwise output and canary checks:

```
./speed-bench/rocm_q4_prefill_bench \
  --case qb --tokens 256,512,1024,2048,2049,4096 \
  --sets 4 --warmup 4 --samples 12
```

Use the production 2048-token sweep to exercise default K128/P144 without
SSD-streaming or policy-selection noise. The same process also reports the
strict K64/P80 versus K128/P144 A/B above:

```
./speed-bench/rocm_q4_prefill_bench \
  --case all --tokens 2048 --sets 4 --warmup 4 --samples 12
```

The benchmark always keeps SSD streaming disabled. Runtime SSD experiments
need the separate `DS4_ROCM_ENABLE_Q4_PREFILL_WMMA_SSD=1` gate and only accept
projection ranges already backed by physical device memory. That gate opts
attention-output A into WMMA but retains TILE8 for B; only the strict,
diagnostic `DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA=1` selects direct WMMA for B.
Model I/O is therefore never folded into the kernel result.

The default token set is `9,17,33,128,256,257,512`, covering the first row
after the small TILE8 boundaries plus both the exact 256-token occupancy case
and the first token after a 64-token WMMA boundary. Use `--full` for
`9,16,17,31,32,33,128,256,257,512,4096`, or select a focused run such as:

```
./speed-bench/rocm_q4_prefill_bench \
  --case qb --tokens 256,257,512,1024,2048,2049,4096 \
  --sets 4 --warmup 4 --samples 12
```

Every case rotates identical resident weight sets between arms, alternates
ABBA/BAAB, and verifies allocation guards both before and after timing.
Comparisons among the integer Q4 paths remain bit-exact. Direct-Q4 WMMA has a
deliberate F16 arithmetic boundary, so single-stage comparisons require finite
results within an explicit absolute/relative smoke tolerance. For the chained
production output, the B-stage same-input replay is bit-exact while the
end-to-end delta is informational; release validation must also include the
model-logit or prompt oracle that guards the automatic standalone and
attention-output-A policy.
Fixture creation, the host-to-device residency copy, warmup, oracle readback,
and environment-gate changes are outside the reported HIP-event intervals.
`candidate_delta_pct` is negative when the candidate is faster; the companion
`speedup_pct` reports the positive speedup convention.

### Resident CUDA Q4_K prefill

Build the production-API kernel harness on a CUDA host with an explicit
architecture:

```
make cuda-q4-prefill-bench CUDA_ARCH=sm_121
./speed-bench/cuda_q4_prefill_bench --path mmq
```

The default `dense`, `pair`, `qb`, and GB10-only `outa` cases use production
shapes and include both sides of the 128-column MMQ tail. The synthetic
GGUF-layout weights are copied by
the backend into a `cudaMalloc` allocation before warmup. A CUDA test hook
checks the backend-owned pointer provenance and device attributes of every
dense, KV, q_b, output-A, and minimal output-B range in every rotating weight
set; a global free-memory
delta is printed only as a diagnostic and is not accepted as residency proof.
CUDA events measure the production backend GPU interval, including
stream-ordered scratch allocation/free, tail clears, activation quantization,
Q4 projection, and output sanitization. Model uploads, output poisoning, full
finite-output scans, sampled CPU Q4_K oracles, canary checks, and warmups are
outside the event interval.

`cuda_use_mmq()` caches its first decision for the life of the process, so the
dense and `attn_q_b` MMQ-versus-Q8_K comparison deliberately uses separate
processes. The MMQ process also enables a test-only strict control, so an MMQ
rejection fails the run instead of silently measuring the Q8_K fallback. Run
both orders to balance thermal/order drift:

```
# ABBA
./speed-bench/cuda_q4_prefill_bench --path legacy --case dense
./speed-bench/cuda_q4_prefill_bench --path mmq    --case dense
./speed-bench/cuda_q4_prefill_bench --path mmq    --case dense
./speed-bench/cuda_q4_prefill_bench --path legacy --case dense

# BAAB
./speed-bench/cuda_q4_prefill_bench --path mmq    --case dense
./speed-bench/cuda_q4_prefill_bench --path legacy --case dense
./speed-bench/cuda_q4_prefill_bench --path legacy --case dense
./speed-bench/cuda_q4_prefill_bench --path mmq    --case dense
```

Repeat with `--case qb` for `K=1024,M=32768`. Each result records the immutable
path as `path=legacy` or `path=mmq`; do not toggle `DS4_CUDA_MMQ` around calls
inside another harness. The default MMQ `pair` case is a true in-process
ABBA/BAAB comparison between two public dense calls and the public fused pair
API, with bit-exact pair outputs. Prefill pair is skipped under `--path legacy`
because the CUDA pair API intentionally returns control to two independent
dense projections for `N>8` when MMQ is disabled.

To isolate the experimental Q4_K m128n128 16-warp GEMM from activation
quantization and every storage effect, run the focused bitwise/canary oracle
and then the resident prequantized A/B benchmark:

```
make test-mmq-q4-16warp-cuda CUDA_ARCH=sm_121
make cuda-q4-prefill-bench CUDA_ARCH=sm_121
./speed-bench/cuda_q4_prefill_bench \
  --path mmq --kernel-16warp --case dense \
  --tokens 512,1024,2048,2049,4096,6144,8192 \
  --sets 4 --samples 16 --warmup 4
./speed-bench/cuda_q4_prefill_bench \
  --path mmq --kernel-16warp --case pair \
  --tokens 512,1024,2048,2049,4096,6144,8192 \
  --sets 4 --samples 16 --warmup 4
DS4_CUDA_MMQ_X_MAX=128 \
./speed-bench/cuda_q4_prefill_bench \
  --path mmq --kernel-16warp --case outb \
  --tokens 2048 \
  --sets 4 --samples 16 --warmup 4
```

The benchmark quantizes X to canonical Q8_1 DS4 once, before timing, and uses
CUDA events around only the production-policy Stream-K reference or the
16-warp kernel with the same partition/fixup policy. Both arms therefore
include fixup whenever the canonical dispatcher would use it, including the
GB10 `M=1024,N=4096` case. One guarded fixup allocation is created before the
samples and reused by both A/B arms and by the two pair legs, so CUDA pool
allocation/free nodes cannot bias the kernel delta.
The dense case is the production Q-A shape `K=4096,M=1024`; the pair case
reuses that same prequantized activation across the asymmetric Q-A/KV shapes
`M0=1024,M1=512` and times two GEMMs in each arm. It alternates the
arms ABBA/BAAB over resident Q4_K weight sets, compares every complete output
bit-for-bit, checks finite values and independent output canaries, and samples
a CPU Q4_K oracle before and after timing. Results attest
`timing=kernel_only_prequant`; SSD streaming, model upload, Q8_1 quantization,
allocation, host copies, and oracle work are outside the samples. Use
`--case qb` for the additional `K=1024,M=32768` large-projection datapoint and
`--case outb` for the real attention output-B `K=8192,M=4096` geometry. The
focused oracle covers that wider K with complete N128 tiles; the resident
benchmark retains the real `N=2048` prefill geometry for performance claims.
The focused test also invokes the real standalone dense and pair dispatchers in
required mode at `N=4096`, so fallback fails instead of producing a misleading
canonical-path pass. A negative pair case verifies that an ineligible 384-row
leg returns `DS4_MMQ_NOT_APPLICABLE` without touching either guarded output.
The CLI accepts contexts through 8192 tokens; the default kernel-only sweep
adds 6144 and 8192 to expose long-context scaling. A custom token tail must
also make the canonical picker select `m128n128`; the harness checks the real
device selector up front and reports `SKIP` instead of running a mismatched
Stream-K partition.

The production path remains opt-in until the NVIDIA oracle passes and the
paired median is a repeatable win. `DS4_CUDA_Q4_MMQ_16WARP=1` requests it and
falls back on ineligible shapes. For an attested production benchmark,
`DS4_CUDA_REQUIRE_Q4_MMQ_16WARP=1` also prevents a Q8_K/MMQ fallback;
`DS4_CUDA_NO_Q4_MMQ_16WARP=1` is the value-aware rollback. The strict path
requires Ampere or newer, complete 128x128 output tiles, `N>=512`,
`1024<=K<=8192`, the default 128-column MMQ selector, and at least 80% final-wave
grid efficiency. Single dense admission starts at `M=1024`, including the
`K=8192` output-B projection; the Q-A/KV pair admission also accepts its
`M1=512` leg when both projections select the 16-warp path, and remains bounded
to `K<=4096`.

On GB10, isolate the production Flash attention-output A geometry
(`groups=8`, `K=4096`, `rank=1024`) and its 127/128/129 token tails with:

```
./speed-bench/cuda_q4_prefill_bench \
  --path mmq --case outa --tokens 127,128,129,257,512,2048 \
  --samples 16 --warmup 4
./speed-bench/cuda_q4_prefill_bench \
  --path mmq --case outa --grouped-q81-kernel \
  --tokens 512,1024,2048,4096,6144,8192 \
  --sets 4 --samples 16 --warmup 4
```

This is an in-process ABBA/BAAB comparison between the current eight-group
pack/MMQ/unpack rollback and the default, strictly required direct-strided
grouped-prefill dispatch with one canonical MMQ grid per group. Add
`--grouped-single-grid` to instead compare the grouped eight-grid path with the
experimental grid.z submission; do not mix that experiment into the default
promotion measurement.
`--grouped-q81-kernel` holds those same eight MMQ grids constant and compares
the canonical strided Q8_1 producer with the default `K=4096`, `groups=8`
eight-warp kernel. The candidate is required and the reference is forced with
its narrow rollback, so neither arm can silently time the other quantizer. The
complete output remains bitwise checked; run
`make test-mmq-q4-grouped-q81-cuda CUDA_ARCH=sm_121` first for direct
byte-level Q8_1 parity and canary coverage.
The public API must also execute output-B, so the fixture uses a valid Q4_K
`K=8192,M=256` output-B common to both arms. It is 6.25% of output-A's MACs;
the result prints `focus_macs_per_token` and `common_macs_per_token` separately
and checks the two arms bit-for-bit. SSD streaming is forced off and every
measured weight range must resolve to backend-owned device storage.

### Resident IQ2/Q2 MoE prefill on ROCm and CUDA

The backend-neutral fixture uses the production `N=4096`, 256-expert, top-6
IQ2_XXS/Q2_K geometry and a deterministic routing distribution containing
every 32-row tail from 1 through 31. Weights and tensors are resident and SSD
streaming is disabled, so the reported GPU-event intervals isolate kernel
work rather than storage throughput. The fixture needs roughly 4 GiB of
explicit host/device storage in addition to backend runtime overhead.

On a wave32 ROCm host, build and run the real balanced A/B with:

```
make rocm-iq2-moe-prefill-bench ROCM_ARCH=gfx1151
./speed-bench/gpu_iq2_moe_prefill_bench_rocm
```

The baseline sets the dominant rollback
`DS4_ROCM_DISABLE_IQ2_MOE_WMMA_TAIL_CULL=1`; the candidate sets
`DS4_ROCM_ENABLE_IQ2_MOE_WMMA_TAIL_CULL=1`. The harness alternates ABBA/BAAB,
requires bit-exact intermediate scratch and final tensors, verifies allocation
canaries, and prints `cudaEvent` time for only the IQ2 gate/up and Q2 down
rocWMMA kernels. The candidate remains opt-in until real-hardware results show
a repeatable win. A wave64 device takes the scalar fallback and therefore
cannot produce a valid candidate timing.

On CUDA, build and run the measurement-only current path with an explicit
architecture:

```
make cuda-iq2-moe-prefill-bench CUDA_ARCH=sm_121
./speed-bench/gpu_iq2_moe_prefill_bench_cuda
```

CUDA prints `DS4_CUDA_MOE_PROFILE` stage times for the resident IQ2 MMQ path
and performs a structural/canary oracle. Every marked call must produce exactly
one completed fast-path profile record or the harness fails. It intentionally
does not claim an A/B result: CUDA's cooperative D2R CTA has different
synchronization and tail semantics, so the ROCm/Metal wave-cull selector cannot
be copied safely.
On a discrete CUDA GPU with enough VRAM, prefix the run with
`DS4_CUDA_COPY_MODEL=1` to keep the raw expert weights in device memory and
remove mapped-host/PCIe stalls from the measured kernel interval; this adds
about 1.7 GiB of device storage. Leave it unset on GB10/UMA when measuring the
normal aligned-artifact production selector, because forcing a raw model copy
changes that residency path.
