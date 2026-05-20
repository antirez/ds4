# Compressed Disk KV Cache

Durable notes for the LZ4 payload codec in `ds4_kvstore.c`.  Format
spec, key constants, and the measurements behind their defaults.

## On-disk format

`KV_CACHE_VERSION` is bumped to 2.  The codec lives in two
previously-reserved bytes of the 48-byte fixed header:

| Offset | Bytes | Field |
|---:|:---:|---|
| 21 | 1 | `codec` (`DS4_KVSTORE_CODEC_NONE`=0, `DS4_KVSTORE_CODEC_LZ4`=1) |
| 22 | 1 | `chunk_size_log2` — `chunk_size = 1 << log2`.  Only meaningful when `codec == LZ4`.  Default 24 (= 16 MiB). |
| 23 | 1 | reserved (must be zero) |

Version-1 files wrote zero into bytes 21–23, so the reader still accepts
them and they round-trip as `codec=NONE` / `chunk_size=0` with no special
case.  The bump exists for the other direction: version-1 binaries rewrite
the fixed header in place when they refresh a file's trailer, and they
would zero the codec byte of a compressed file — relabeling LZ4 bytes as a
raw payload.  Rejecting unknown versions makes old builds treat version-2
files as a miss instead.

When `codec == DS4_KVSTORE_CODEC_LZ4` the payload region is:

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

`codec == DS4_KVSTORE_CODEC_NONE` is a raw payload, byte-identical to the
pre-codec format.

## Defaults and bounds

| Constant | Value | Why |
|---|---|---|
| `DS4_KVSTORE_DEFAULT_CHUNK_BYTES` | 16 MiB (`log2 = 24`) | Saturates ~8 P-cores on a 1–2 GiB payload (64–128 chunks); per-chunk preamble overhead is then noise. |
| `DS4_KVSTORE_MAX_CHUNK_BYTES` | 64 MiB (`log2 = 26`) | Hard cap on chunk_size read from the header so a tampered file can't ask for gigabyte allocations. |
| default threads | `min(8, online_cpus)` | Matches `ds4_threads_init` in `ds4.c`.  8 saturates lz4 -1 on both M1 Ultra and M4 Pro (see below). |
| `--kv-cache-compression-threads N` cap | 64 | Internal writer/reader cap; the CLI clamps at parse time so the startup log matches actual behavior. |

## Measured ratios and timings

`ds4-server` on M1 Ultra (Metal, ctx=400000, IQ2XXS+w2Q2K quant), three
prompt sizes sent twice each (cold then warm):

| Prompt | On-disk size | Compression ratio | Save | Load (warm hit) |
|---:|---:|:---:|---:|---:|
|  3.5 K tok |  37 MiB | **1.86×** |  59 ms |  25 ms |
|   17 K tok | 140 MiB | **1.77×** | 138 ms |  72 ms |
|   47 K tok | 354 MiB | **1.81×** | 359 ms | 180 ms |

End-to-end TTLB cold → warm:

| Prompt | Cold (prefill + save) | Warm (load + reply) |
|---:|---:|---:|
|  3.5 K tok |  15.88 s | **0.45 s** |
|   17 K tok |  73.89 s | **0.59 s** |
|   47 K tok | 225.49 s | **0.91 s** |

The chunked format also permits parallel decompress, but the current
load path is single-threaded — the per-load decompress time above is
already small relative to the matching cold prefill, so the parallel
path is left for a follow-up.

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
