# Models Larger Than RAM

[README](../README.md)

Use resident inference when the model and runtime fit: it is faster.
SSD streaming keeps a bounded cache of routed experts and reads missing
experts from the GGUF. It trades speed for capacity; it does not remove the
memory needed for other weights, activations, scratch, and the context.

Metal supports streaming for DeepSeek, GLM and [Qwen3.8](QWEN38_FLASH_NEXT.md).
CUDA has streaming paths too, and ROCm supports GLM 5.2/5.3 streaming.
Do not infer support for every model
and tensor layout from the existence of the flag.

## Start with the automatic budget

```sh
./ds4 -m ds4flash.gguf --ssd-streaming
```

The startup report shows the effective cache, resident layers where applicable,
and memory requirements. Prefer the automatic budget for a first run. Use a
local SSD and leave memory for the OS and other applications.

Examples:

```sh
# Flash Q2 on a smaller Mac.
./download_model.sh ds4f-q2
./ds4 --ssd-streaming --ctx 32768 --nothink

# GLM 5.3 Flash Q4 on a 128 GB Mac.
./download_model.sh glm53-q4
./ds4 --ssd-streaming --ctx 4096

# PRO Q2 on a 128 GB Mac: usable for inspection, but slow.
./download_model.sh pro-q2-imatrix
./ds4 --ssd-streaming --ctx 32768 --nothink
```

Generation is usually more sensitive to cache misses than prefill. A large
model that starts successfully can still be too slow for interactive work.
Use a short generation before committing to a long task.

## Adjusting the cache

To leave more room for context or other sessions:

```sh
./ds4 --ssd-streaming --ssd-streaming-cache-experts 32GB
```

A byte budget is a target, not a guaranteed allocation. DwarfStar reserves
routed-prefill headroom and fits the cache to the remaining model, graph,
context, and backend budget. The effective value may be smaller than requested.
Non-routed weights and KV state are additional to that expert-cache budget.

A plain number, such as `--ssd-streaming-cache-experts 4000`, requests dynamic
expert slots rather than a byte budget. It is also subject to memory limits.

By default, GLM spends the cache budget on selected experts across all layers.
`--ssd-streaming-full-layers N` reserves full routed prefix layers instead.
Metal also keeps the non-routed weights resident when it can; they are needed
by every token, and paging them back in delays generation after a prompt.
An oversized expert cache can displace those weights and slow decoding.
More cache helps only while the rest of the working set still fits.

More context and more server sessions consume more memory. Metal GLM budgeting
includes planned and already resident sessions; a request can still be refused
when its mandatory allocations cannot fit. Disabling the memory guard is not
a remedy for insufficient RAM.

Leave expert preloading enabled for normal use. `--ssd-streaming-cold` and
`--ssd-streaming-preload-experts N` are mainly useful for controlled measurements.
See [benchmarking](PERFORMANCE.md) and the [release QA guide](../QA_BEFORE_RELEASES.md).

## Seeding the cache from a routing hotlist

A short prompt leaves the prompt-driven expert seed nearly empty, so its first
decode turns pay a cache miss (a disk read) for most routed experts. To cover
that, a session can preload a hotlist: a table of (layer, expert) pairs ranked
by how often that expert was selected in recorded runs. DeepSeek V4, GLM, and
V4.1 all ship built-in tables; the older models seed around their prefill
layer sweeps, and GLM demand-fills by default (it seeds only with an explicit
count or auto cap). V4.1 seeds from its table at session create - currently
Metal only, with the CUDA and ROCm paths keeping their own cache policies -
and deliberately seeds the whole table, clamped to the expert-cache budget,
rather than GLM's 4096-entry auto cap: full coverage is where the short-prompt
win is measured. The seed fills the budget once, roughly seven seconds of
sequential reads on an M5 Max, a good trade for long-lived sessions.

The switches:

| Switch | Effect |
| --- | --- |
| `DS4_METAL_STREAMING_EXPERT_HOTLIST=file` | Seed from a recorded file instead of the built-in table |
| `--ssd-streaming-preload-experts N` | Cap how many hotlist entries are preloaded (the same count the GLM seeder honors) |
| `DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP=N` | Cap the default (unset count) seed, for file and built-in sources alike - the same brake the GLM seeder honors |
| `DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST=1` | Skip the seed entirely |
| `--ssd-streaming-cold` | Keep measuring the empty cache |

The built-in tables are generated from real routing, and the same procedure
reproduces them or builds one for a new model. Record selections while
running a representative mix of workloads, accumulating across runs with
`_MERGE=1` (run from the repo root, and start with no existing
`hotlist.txt`). The corpus below - two bench sizes and three one-shot
prompts - is the one the shipped V4.1 table was recorded from, so these
commands reproduce `ds4_streaming_hotlist_v41.inc` on the same GGUF:
DeepSeek V4.1 Flash Q2, the `ds41f-q2` download
(`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`). Routing depends on
the weights, so a different GGUF produces a different table.

```sh
DS4_MOE_RECORD_SELECTED_HOTLIST=hotlist.txt \
DS4_MOE_RECORD_SELECTED_HOTLIST_MERGE=1 \
  ./ds4-bench -m MODEL --ssd-streaming \
    --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start 1024 --ctx-max 1024 --gen-tokens 128

DS4_MOE_RECORD_SELECTED_HOTLIST=hotlist.txt \
DS4_MOE_RECORD_SELECTED_HOTLIST_MERGE=1 \
  ./ds4-bench -m MODEL --ssd-streaming \
    --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start 64 --ctx-max 64 --gen-tokens 96

for P in "Why is the sky blue? Answer in two sentences." \
         "Explain in exactly two sentences why sunsets look red." \
         "Summarize the theory of evolution in two sentences."; do
  DS4_MOE_RECORD_SELECTED_HOTLIST=hotlist.txt \
  DS4_MOE_RECORD_SELECTED_HOTLIST_MERGE=1 \
    ./ds4 -m MODEL --ssd-streaming --ctx 2048 \
      --think-level 0 --temp 0 -n 48 -p "$P"
done
```

Each routed-layer call counts its six selections in memory; the file is
written when the process exits. Short prefills are recorded too, but a
large chunked prefill reads whole layers through the batch path and is not
recorded — the 1024-token bench above contributes only its decode tokens.
`_MERGE` makes every run load the previous file first, so one hotlist keeps
growing (add `_FRESH=1` to start clean). Record with the same engine
options as the intended deployment — the corpus determines the table.

Convert the result into a compile-time table:

```sh
python3 gguf-tools/hotlist_to_inc.py hotlist.txt ds4_streaming_hotlist_v41.inc
```

The generated header records its selection total and pair count; a
regeneration should match them (the shipped table's are 117840 and 9448).

The pairs keep the recorder's hits-descending order, which is what the
loader's rank-based priority is derived from. To ship it as a built-in for a
new model, add the generated `.inc` to the build, include it next to the
existing tables in `ds4.c`, and add the model's variant to
`metal_graph_streaming_expert_hotlist_load_default`. For V4.1, seeding from
the recorded hotlist raised steady decode of a 64-token prompt from 19.8 to
26.7 tokens/s on an M5 Max, with byte-identical greedy output.

## CUDA expert reads

DeepSeek V4.1 CUDA prefill reads selected experts in file order and combines
adjacent reads by default, instead of prefetching every expert in the next
layer. Single-token decode retains its original read order, which avoids a
slowdown with small expert caches. The weight formats, routing, and cache
budget are unchanged; no extra flags are needed.

For comparisons on other storage or workloads,
`DS4_CUDA_ENABLE_SSD_PREFETCH=1` restores whole-next-layer prefetch.
`DS4_CUDA_DISABLE_SSD_SORT_SELECTED_READS=1` restores the original selected-read
order. Set both to reproduce the previous policy.
`DS4_CUDA_DISABLE_SSD_PREFETCH=1` still overrides prefetch enablement.
For experiments, `DS4_CUDA_SSD_SORT_SELECTED_READS=1` also orders single-token
loads; the explicit sort-disable switch takes precedence.
Unset that old experimental sort flag to use the new prefill-only default.
These switches use presence semantics: unset them to return to the default;
setting a switch to `0` still enables it.

## M5 Max measurements

On a 128 GB M5 Max, with automatic cache sizing and no speculative decoding:

| Model | Initial prefill | Continued prefill | Generation after each | Runs |
| --- | ---: | ---: | ---: | --- |
| GLM 5.3 Flash Q4_K, 177.77 GiB | 121 t/s | 104 t/s | 11.9 / 14.9 t/s | Three-run median |
| DeepSeek Flash Vision Exp MXFP4, 145.26 GiB | 300 t/s | 263 t/s | 11.9 / 19.3 t/s | Single run |

September 6, 2026. GLM used a 2K prompt and a 1K append; DeepSeek used 8K
and a 4K append. Both generated 128 tokens at each frontier. DeepSeek's latest
run uses an 86.2 GiB expert cache and updated eviction priorities; it is not
directly comparable to the earlier 64-token measurement. Generation includes
the first-token wait. Cache reuse depends on the prompt, so these are
workload references, not a speed
guarantee for every model larger than RAM.

Full GLM 5.3 IQ2_XXS (196.58 GiB) also benefits from caching short tool-result
prompts. With an 8K context and a 61.35 GiB effective expert cache, a 16-token
append fell from 30.8 to 2.9 seconds; generation afterward rose from 4.09 to
5.11 t/s (three-run medians, 64 generated tokens). All compared logits and
generated text were identical. This is a short-append improvement, not an
equivalent gain in initial prefill or every generation workload.
