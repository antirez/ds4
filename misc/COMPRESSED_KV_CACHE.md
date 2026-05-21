# Compressed Disk KV Cache

Durable notes for the LZ4 payload codec in `ds4_kvstore.c`.  Format
spec, key constants, and the measurements behind their defaults.

## On-disk format

`KV_CACHE_VERSION` is 2.  Two previously-reserved bytes of the
48-byte fixed header become `codec` (offset 7) and `chunk_size`
(offset 20).  When `codec == DS4_KVSTORE_CODEC_LZ4` the payload region is:

```text
u64 uncompressed_total
u32 chunk_count
repeat chunk_count times:
    u32 raw_size
    u32 comp_size
    u8[comp_size]    LZ4_compress_default block
```

`raw_size` equals `chunk_size` for all chunks except possibly the
last, where it is the remainder.  `chunk_count` therefore equals
`ceil(uncompressed_total / chunk_size)`; the reader enforces this.

`codec == DS4_KVSTORE_CODEC_NONE` is a raw payload, byte-identical to v1.
The loader accepts v1 and v2; the writer always writes v2.

## Defaults and bounds

| Constant | Value | Why |
|---|---|---|
| `KV_CACHE_DEFAULT_CHUNK_BYTES` | 16 MiB | Saturates ~8 P-cores on a 1–2 GiB payload (64–128 chunks); per-chunk preamble overhead is then noise. |
| `KV_CACHE_MAX_CHUNK_BYTES` | 64 MiB | Hard cap on chunk_size read from the header so a tampered file can't ask for gigabyte allocations. |
| default threads | `min(8, online_cpus)` | Matches `ds4_threads_init` in `ds4.c`.  8 saturates lz4 -1 on both M1 Ultra and M4 Pro (see below). |
| `--kv-cache-compression-threads N` cap | 64 | Internal writer/reader cap; the CLI clamps at parse time so the startup log matches actual behavior. |

## Save / load cost, 1.5 GB checkpoint, M1 Ultra

| Codec | Save wall | On disk | Single-thread load |
|---|---|---|---|
| Uncompressed (v1 path) | ~750 ms | 1500 MB | ~210 ms (mmap) |
| lz4 -1 -T8 (this codec) | ~410 ms | 1010 MB | ~400 ms |

Save is faster with lz4 -1 because disk I/O dominates and half the
bytes go to disk.  Single-thread decompress on load costs ~190 ms
versus warm mmap; cold-disk loads come out ahead either way because
reads halve.  The chunked format permits parallel decompress to
close the warm gap, but that path is not yet measured.

## Compress scaling (lz4 -1, 2.1 GB file)

| Threads | M1 Ultra | M4 Pro |
|---|---|---|
| -T1 | 4.62 s | 3.34 s |
| -T4 | 1.39 s | 0.97 s |
| **-T8** | **0.88 s** | **0.59 s** |
| -T0 (all P-cores) | 0.86 s | 0.59 s |

Both saturate by 8 P-cores.  Going wider wastes cores that could
run inference.  `min(8, online_cpus)` is the right default on both.

## Streaming, not snapshot

Compression goes through a `funopen`/`fopencookie` wrapper around
`ds4_session_save_payload` / `load_payload`.  Peak extra RAM during
a save is `2 * threads * chunk_size` (~256 MiB at the defaults),
independent of context length — the worker never holds the full
1–16 GiB uncompressed buffer.

Engine APIs in `ds4.h` are unchanged.
