# Kimi K3 on Metal

Kimi K3.1 Pro text inference runs on Metal with Moonshot’s original
SafeTensors. Point `-m` at the directory containing the shards, config and
tokenizer. The full release has 96 shards and uses 1.56 TB of storage.

```sh
hf download moonshotai/Kimi-K3.1-Pro --revision 6d0a0202956796543364d666ab19c6ad1a494ee1 --local-dir models/Kimi-K3.1-Pro
make
./ds4 -m models/Kimi-K3.1-Pro --ctx 4096 -p "Explain Redis streams."
```

During SafeTensors prefill, Metal starts computing ready experts while the
SSD reads the remaining selections. Large batches alternate between halves
of the existing expert buffers, so the reader can start the next group while
the GPU finishes the current one.

K3.1 uses a learned index in 24 attention layers. Each query attends to all
available positions through 2,048 tokens and to its selected 2,048 positions
thereafter. The original K3 SafeTensors and its existing mixed-Q2 GGUF also
run on this branch. K3 vision, CUDA, ROCm, speculative decoding, steering, and reduced-power execution are not supported.

```sh
./ds4 -m models/Kimi-K3-IQ2_XXS-Q2_K.gguf --ctx 4096 -p "Explain Redis streams."
./ds4-agent -m models/Kimi-K3-IQ2_XXS-Q2_K.gguf --ctx 32768
./ds4-server -m models/Kimi-K3-IQ2_XXS-Q2_K.gguf --ctx 32768
```

The original K3 GGUF is 799.783 GiB. It streams from SSD automatically when its local
weights cannot fit safely in memory. The router still selects all 16 experts;
streaming changes where the weights live, not the computation.

On a 128 GiB Mac, the runtime keeps the non-routed weights resident when they
fit, then uses the remaining safe budget for an expert cache. With less free
memory it keeps only some layers' non-routed weights resident. This is automatic;
no cache or residency flags are needed.

During single-token GGUF decoding, the runtime uses the next layer’s router
and the current hidden state to predict four likely experts. While Metal
runs the current layer, macOS reads ahead from the GGUF for guesses that
are missing from the expert cache. The next layer’s actual router then
selects all 16 experts. A wrong guess only costs I/O. The runtime skips
prediction across an AttnRes block boundary or when the next layer’s static
weights are on disk. Set `K3_ROUTE_PREDICT=0` to benchmark without lookahead.

After the actual router selects its 16 experts, Metal can start work on experts
already in its cache or whose reads have finished. The other reads continue
while the GPU computes. If the first experts in router order are ready, Metal
can also start their output sum early and resume it in the original order.

Routed gate/up weights use IQ2_XXS and routed down weights use Q2_K.
Shared experts remain MXFP4, latent up projections use Q8_0, and attention,
embeddings, the output head, routers, and norms retain higher precision.

K3 uses its native XTML chat and tool format. Its sampling defaults are
temperature 1, top-p 0.95, and min-p 0. Saved sessions rebuild their inference
state by replaying the saved tokens.

## Optional SSD streaming setup for a 32 GiB M1 Max

For the original K3 mixed-Q2 GGUF, aligned static staging and tiled output-head
reads reuse two bounded buffers. An optional lossless static cache stores
16 MiB blocks using byte-plane shuffling and zstd, with raw blocks when
compression does not save space. It changes no tensor values or model math.

The cache requires `libzstd` discoverable through `pkg-config`; the normal
build still works without this optional dependency. These commands build the
tools and create a new cache without installing packages:

```sh
k3_model_file="/absolute/path/to/Kimi-K3-IQ2_XXS-Q2_K.gguf"
k3_static_cache_file="/absolute/path/to/Kimi-K3-IQ2_XXS-Q2_K.static-zstd-v1.cache"
pkg-config --exists libzstd && make ds4 k3-static-cache-build
./k3-static-cache-build "$k3_model_file" "$k3_static_cache_file"
```

Allow about 59 GiB of additional disk space for this GGUF. The builder reads
its static spans, compares every decoded block byte-for-byte with the source,
and publishes the completed cache atomically without replacing an existing
file. The original GGUF remains unchanged and is still required for experts,
embeddings and the output head. The cache is tied to that file's identity,
size and timestamps; the loader checks header/index CRCs and each decoded
block's CRC, and rejects changed or corrupt data.

A reproducible short generation check uses a 10 GiB expert cache, a 12 GiB
host reserve, and approximately 131 MiB of additional decoder workspace
accounted for before allocation. All existing memory checks remain active.

```sh
K3_MEMORY_RESERVE_GIB=12 K3_RESIDENT_STATIC=0 \
K3_GGUF_STATIC_STAGING=1 K3_GGUF_STATIC_ALIGNED=1 \
K3_GGUF_HEAD_STAGING=1 K3_GGUF_STATIC_CACHE="$k3_static_cache_file" \
K3_STATIC_IO_THREADS=4 K3_LAYER_PAGEIN=0 K3_ROUTE_PREDICT=1 \
K3_BF16_MV_UNROLL=16 K3_BF16_MV_NSG=4 \
DS4_METAL_STREAMING_EXPERT_SLAB_MB=256 \
DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD=1 K3_IO_PROFILE=1 \
./ds4 -m "$k3_model_file" --ssd-streaming \
  --ssd-streaming-cache-experts 10GB --prefill-chunk 64 --ctx 1024 \
  --nothink --temp 0 --seed 1 -n 9 -p "Ciao, come stai?"
```

To also stage multi-token prefill from the same lossless cache, add
`K3_GGUF_PREFILL_STAGING=1` to the command environment. This optional path
reuses the existing two static buffers and decoder workspace; no cache rebuild
or extra weight buffers are needed. Expert selection, the expert cache and
FP32 computation retain their existing paths. The flag requires the lossless
cache configuration above and is disabled by default; unset it or set it to
`0` to restore the original prefill path while keeping cached decode.
The earlier decode measurements below describe the command above without
this additional flag.

Leave both `K3_GGUF_STATIC_CACHE` and `K3_GGUF_PREFILL_STAGING` unset to use
the original GGUF static reads.
An invalid cache causes an explicit error rather than a silent fallback.
`make test-k3-static-cache` runs the CPU format and decoder fixtures.
On the 32 GiB M1 Max, one controlled comparison of eight greedy decode steps
for this prompt measured 41.488 to 28.379 seconds per step: 46.2% more tokens
per second, or 31.6% less decode time. The reference already used GGUF static
staging (`8ba3fac3`); these numbers measure the combined settings above.
Prefill was 181.156 versus 183.668 seconds. Peak task footprint was 13.827
versus 13.949 GiB, with normal memory pressure and no new swap in either run.
Generated text matched, all 5,309 cache blocks were roundtrip verified, and
separate KDA/MLA boundary and full output-head checks were bitwise identical.
This is one measured pair with a short prompt, not a long-session benchmark;
filesystem cache, external load and temperature were not controlled.


With `K3_GGUF_PREFILL_STAGING=1`, four alternating full-model runs on the
same M1 Max compared commit `bb4ae3e9` with this prefill extension. Each run
used the prompt above (44 formatted tokens), chunk64, ctx1024 and the same
10 GiB expert cache, 12 GiB host reserve and lossless-cache configuration.
`--dump-logits` measured one complete prefill without generation:

| Prefill path | Run 1 | Run 2 | Median |
| --- | ---: | ---: | ---: |
| Reference `bb4ae3e9` | 183.589 s | 178.048 s | 180.819 s |
| Lossless prefill staging | 103.380 s | 102.212 s | 102.796 s |

The median prefill time fell 43.15%, equivalent to 75.90% more tokens per
second. Comparing the slowest optimized run with the fastest reference still
reduced time by 41.94%. All 163,840 final FP32 logits were bitwise identical
across all four runs. A separate real-model KDA/MLA slice check compared all
931,840 boundary values after 9-token prefill, 3-token continuation and one
decode step, also bitwise identical. Peak task footprint was 13.747 GiB for
the optimized path versus 13.629 GiB for the reference, with normal memory
pressure and no new swap in every run. This measures the short prompt above;
longer contexts were not benchmarked, and filesystem cache, temperature and
external load were not controlled.

## Two 512 GiB Macs

Put the same GGUF on both machines. This is a contiguous-layer pipeline, not
tensor parallelism. The split below assigns about 401.16 GiB to the coordinator
and 398.62 GiB to the worker, before context and runtime buffers.

If needed, raise the Metal wired-memory limit on each **512 GiB** machine:

```sh
sudo sysctl iogpu.wired_limit_mb=480000
```

Do not use that limit on a smaller Mac.

```sh
# Machine A
./ds4 -m models/Kimi-K3-IQ2_XXS-Q2_K.gguf --ctx 32768 \
  --role coordinator --layers 0:46 --listen 0.0.0.0 9000

# Machine B
./ds4 -m models/Kimi-K3-IQ2_XXS-Q2_K.gguf --ctx 32768 \
  --role worker --layers 47:output --coordinator MACHINE_A 9000
```

The runtime pins each local slice only when it fits the live memory budget.
The pipeline transfers the complete FP32 AttnRes state and negotiates the
prefill chunk size between nodes. `K3_GGUF_RESIDENT=0` disables residency
for diagnostic comparisons.

## Conversion

With the K3.1 Pro checkpoint in `models/Kimi-K3.1-Pro`:

```sh
make -C gguf-tools k3-quantize
./gguf-tools/k3-quantize --model models/Kimi-K3.1-Pro \
  --out models/Kimi-K3.1-Pro-IQ2_XXS-Q2_K.gguf --threads 16
```

Use `--resume` after an interrupted conversion. The converter checks the
partial file and its progress journal before continuing.

`make test-k3-gguf` checks the completed GGUF and its Metal layer path.
`make test-k3-gguf-source` compares retained tensors with the matching source checkpoint. For a K3.1 conversion set both `K3_MODEL` and `K3_GGUF` to its source and output paths.
