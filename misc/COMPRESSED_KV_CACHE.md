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
    u8[comp_size]    LZ4_compress_HC(.., level=1) block
                     applied to the chunk after a byte-4 transpose
```

The byte-4 transpose collects the bytes at each position-mod-4
within a chunk together, so a 16 MiB chunk becomes four 4 MiB
streams of "first bytes," "second bytes," etc.  Real KV state is
laid out in 4-byte aligned blocks, so the transposed streams have
much longer match runs and LZ4 finds them at HC1 cost.  Reader is
symmetric: decompress chunk, then untranspose into the engine's
payload buffer.

NEON `vld4q_u8` / `vst4q_u8` on AArch64 does the 4-way transpose in
a single instruction; SSSE3 `pshufb` builds the same permutation on
x86_64; a portable scalar fallback covers everything else.

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
| default threads | `min(8, online_cpus)` | Matches `ds4_threads_init` in `ds4.c`.  8 P-cores saturate the codec at production chunk sizes. |
| `--kv-cache-compression-threads N` cap | 64 | Internal writer/reader cap; the CLI clamps at parse time so the startup log matches actual behavior. |

## Measured ratios and timings

`ds4-server` on M1 Ultra (Metal, ctx=600000, IQ2XXS+w2Q2K quant),
three prompt sizes sent twice each (cold then warm):

| Prompt | On-disk size | Compression ratio | Save | Load (warm hit) |
|---:|---:|:---:|---:|---:|
|  3.5 K tok |  28 MiB | **2.74×** |  80 ms |  23 ms |
|   17 K tok | 117 MiB | **2.50×** | 221 ms |  87 ms |
|   47 K tok | 340 MiB | **2.38×** | 583 ms | 219 ms |

Default thread count is `min(8, online_cpus)`, matching
`ds4_threads_init` in `ds4.c`; 8 P-cores saturate the codec at
production chunk sizes, and going wider takes cores away from
inference.

## Why HC level 1

Codec sweep over a real 994 MiB transposed payload (47 K-token IQ2XXS
checkpoint, 63 × 16 MiB chunks, 8 threads, M1 Ultra):

| Codec | Compress | Ratio | Decompress |
|---|---:|:---:|---:|
| `LZ4_compress_default` | 6.9 GB/s | 2.16× | 33.9 GB/s |
| fast, acceleration 8 | 9.6 GB/s | 2.01× | 35.7 GB/s |
| **HC level 1** | **3.2 GB/s** | **2.38×** | **24.1 GB/s** |
| HC level 2 | 3.2 GB/s | 2.38× | 23.9 GB/s |

The transposed streams are match-rich, so HC1 runs at ~400 MB/s/core —
nothing like its cost on generic data — and still beats fast mode by
~10% on disk.  At 3.2 GB/s the compressor is no longer the bottleneck
of a save (the write side is), so trading ratio for more speed buys
little; HC2+ costs the same and gains nothing.

## Streaming, not snapshot

Compression goes through a `funopen`/`fopencookie` wrapper around
`ds4_session_save_payload` / `load_payload`.  Peak extra RAM during
a save is `2 * threads * chunk_size` (~256 MiB at the defaults),
independent of context length — the worker never holds the full
1–16 GiB uncompressed buffer.

Engine APIs in `ds4.h` are unchanged.
