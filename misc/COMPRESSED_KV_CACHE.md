# Compressed Disk KV Cache

The shared disk KV store losslessly compresses checkpoint payloads with LZ4-HC
level 1 after a byte-4 transpose. Server and agent disk checkpoints use this
path; live inference KV buffers and `ds4-bench` memory snapshots do not.

## Format and compatibility

Compressed files use cache format version 2 and two previously reserved bytes
of the 48-byte header:

| Offset | Field |
|---:|---|
| 21 | Codec: 0 = raw, 2 = LZ4 with chunk checksums |
| 22 | Chunk size log2; 24 means 16 MiB, meaningful only for LZ4 |
| 23 | Reserved |

Raw files are written as version 1, exactly the layout older binaries read and
refresh in place. Only compressed files are version 2, which older binaries
reject, so their header and trailer updates cannot clear a codec byte.
Refreshing an entry rewrites its version from its codec, so raw entries end as
version 1, including raw version-2 files from earlier builds. The reader
accepts version 1 with codec 0 and version 2 with either codec, and rejects
version 1 carrying a codec. Codec 1 was an unreleased layout without checksums;
it is rejected, and refreshing the server index removes such files. The agent
never deletes files, so sessions saved in that layout drop out of its session
list and must be removed by hand.

An LZ4 payload contains little-endian framing followed by chunk records:

```text
u64 uncompressed_total
u32 chunk_count
repeat chunk_count times:
    u32 raw_size
    u32 compressed_size
    u32 xxh32_of_raw_bytes
    u8[compressed_size] LZ4 block of byte-4-transposed input
```

Every chunk except the last has the declared chunk size, and raw sizes sum to
the declared total. Decoding reverses the transpose and verifies each chunk's
XXH32 before its bytes reach the engine. Framing checks reject empty
compressed frames, inconsistent counts, impossible expansion, truncated
records and invalid chunk lengths; the checksum catches content mutations
those checks cannot. Trailers remain outside the compressed region, and the
payload loader rejects a payload size larger than the rest of the file before
it seeks.
Raw payloads still carry no content checksum.

## Resources and cache policy

The default is `min(8, online CPUs)` workers and 16 MiB chunks. The server flag
`--kv-cache-compression-threads N` overrides
`DS4_KV_CACHE_COMPRESSION_THREADS`; zero writes raw files. The worker limit is
64; accepted chunk sizes are bounded by 64 MiB, and a size that is not a power
of two uses the default. Compression does not depend on cookie streams: the
writer reads the staged payload directly. Failure to allocate a writer falls
back to raw storage, and so does an encoded region that saves less than 1/64 of
the payload: it is rewritten raw in place and the file truncated, since such a
file would pay chunk decoding on every load for no useful space. Qwen3.8 Flash
Next checkpoints take this path, since their dense float state and 16-bit
key/value caches compress to about 1.00x; they still pay the encode attempt,
so `--kv-cache-compression-threads 0` saves that time.

Each writer worker holds raw, shuffled, and encoded buffers: approximately
`3 * workers * chunk_size`, or 384 MiB at defaults, plus codec and stream state.
Reader pools are limited to the actual number of chunks and allocate no more
than the smaller of chunk size and total decoded size per raw/shuffle slot.

The engine first stages a complete raw payload into a temporary file. The codec
then streams that file into the cache's temporary output; it does not require
an additional full-payload RAM buffer. Temporary disk usage includes raw staging
and the new output while old cache files still exist.

Admission uses the actual encoded size, including text, trailers and 1% safety
headroom. Raw writes can reject a known oversize payload early. Raw fallback can
exceed the budget, so the final size check applies to both formats. Only after successful close and atomic rename does the
store evict entries using actual file sizes, protecting the newly admitted
checkpoint. Failed writes or publication leave existing entries intact, including
a same-key file incompatible with the current model or context. This
requires free temporary disk space beyond the configured cache budget; an
ENOSPC failure does not justify deleting working cache entries speculatively.

A payload is removed only when its stored bytes are proven wrong, so a clean
recomputation can replace it: bad framing or chunk records, a checksum
mismatch, or a size past the end of the file. Every other load failure keeps
the file, including engine, GPU, allocation and stream I/O failures that report
no errno. Files in the retired codec-1 layout are removed when the index is
refreshed, since nothing can read or evict them; files with any other unknown
codec or version are left alone.

Where cookie streams are available the engine reads through the decoder
directly. Elsewhere the payload is decoded into a temporary file first, which
needs space equal to the uncompressed payload in the C library's temporary
directory; on Linux that is usually `/tmp`, which may be RAM-backed. A decode that
fails for lack of that space keeps the entry and is retried on the next lookup.

Cold checkpoints can be saved during prefill. Logged `save_ms` excludes the
initial raw staging, so it must not be described as total checkpoint overhead
or as work that always happens after the user receives a response.

## Real V4.1 server measurements

M1 Ultra, 128 GiB, Metal SSD streaming, 32 GiB expert-cache target,
`DeepSeek-V4.1-Flash-Q2.gguf`, context 69,632. Default prefill chunk 8,192;
cache boundary alignment 2,048. Measured on local revision `d83b4ba` before the
subsequent admission/recovery fixes; the payload encoding is unchanged.

For each size: empty raw directory, HTTP request, capture the cold file before
shutdown, restart, repeat and verify disk reuse. Repeat with eight compression
workers and a separate empty directory. Requests contain mixed prose and source
code, temperature zero, and at most 64 generated tokens.

| Cached tokens | Prompt tokens | Raw file | LZ4 file | Ratio | Raw / LZ4 logged load |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 4,141 | 35.52 MiB | 12.15 MiB | 2.92x | 7.5 / 16.5 ms |
| 16,384 | 16,429 | 110.61 MiB | 36.08 MiB | 3.07x | 23.9 / 31.9 ms |
| 63,488 | 65,494 | 398.43 MiB | 127.53 MiB | 3.12x | 92.5 / 111.7 ms |

All twelve response messages matched. Independently decoding each compressed
checkpoint with upstream LZ4 and a separate inverse transpose reproduced every
raw payload byte. These are actual filesystem sizes of matching checkpoints,
not directory totals or API write counters. The largest replay recomputed 2,006
prompt tokens beyond the saved aligned prefix.

The chat endpoint did not return requested log probabilities. An initial audit
incorrectly compared absent values; the corrected audit reports them unavailable.
The response check is not a numerical-logit comparison or a long-horizon quality
assessment. Initial timing runs did not clear OS page cache or balance run order
and therefore establish no inference-speed improvement or no-regression bound.

V4.1 stores BF16/FP8/FP4-rounded values in float-addressable buffers. In two
16 MiB samples of the largest checkpoint, the low two bytes of every four-byte
word were zero. Byte shuffling groups these redundant bytes into runs. One
sample compressed 2.56x directly and 3.00x with the shuffle, without introducing
additional numerical loss. Ratios depend on payload representation and workload.

## Server measurements on M1 Ultra and M5 Max

Base `9139e2a`, macOS 26.6.2. Each prompt was sent through `ds4-server`
twice, first with `--kv-cache-compression-threads 0` and then with LZ4, so the
first request stores a checkpoint and the second loads it. M1 Ultra, 128 GiB:
V4 Flash `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
and `DeepSeek-V4.1-Flash-Q2.gguf` over SSD streaming with 16 and 24 GiB expert
caches, because a separate 34 GiB process was running; `Qwen3.8-Flash-Next-Q2.gguf`
resident. M5 Max, 128 GiB: the same V4 Flash file resident and
`DeepSeek-V4.1-Flash-Q4.gguf` over SSD streaming with a 32 GiB expert cache.

| Machine | Model | Tokens | Raw | LZ4 | Ratio |
|---|---|---:|---:|---:|---:|
| M1 Ultra | V4 Flash | 4,096 | 76.66 MiB | 27.96 MiB | 2.74x |
| M1 Ultra | V4 Flash | 20,480 | 291.77 MiB | 116.43 MiB | 2.51x |
| M5 Max | V4 Flash | 4,096 | 76.66 MiB | 26.43 MiB | 2.90x |
| M5 Max | V4 Flash | 20,480 | 291.77 MiB | 108.79 MiB | 2.68x |
| M5 Max | V4 Flash | 40,960 | 560.66 MiB | 212.19 MiB | 2.64x |
| M5 Max | V4 Flash | 60,746 | 820.39 MiB | 321.64 MiB | 2.55x |
| M1 Ultra | V4.1 Q2 | 4,096 | 35.50 MiB | 12.20 MiB | 2.91x |
| M1 Ultra | V4.1 Q2 | 21,878 | 144.27 MiB | 46.69 MiB | 3.09x |
| M5 Max | V4.1 Q4 | 4,096 | 35.52 MiB | 12.85 MiB | 2.76x |
| M5 Max | V4.1 Q4 | 21,878 | 144.18 MiB | 47.31 MiB | 3.05x |

Checkpoints stored at the end of a request include the reply and compress 2.0x
to 2.6x on V4 Flash. With a fixed `--kv-disk-space-mb`, six ~10K-token prompts
kept 9 to 10 LZ4 files against 3 raw on V4 Flash at 600 MiB, and 11 against 3
on V4.1 at 280 MiB, on both machines.

The request column is the wall time of the request that stored the checkpoint,
including its prefill.

| Machine | Model | Tokens | Save raw / LZ4 | Load raw / LZ4 | Request |
|---|---|---:|---:|---:|---:|
| M1 Ultra | V4 Flash | 4,096 | 74 / 85 ms | 30.0 / 34.6 ms | 36-49 s |
| M1 Ultra | V4 Flash | 20,480 | 287 / 312 ms | 115.6 / 119.6 ms | 128-153 s |
| M5 Max | V4 Flash | 4,096 | 29 / 36 ms | 11.2 / 15.1 ms | 7 s |
| M5 Max | V4 Flash | 20,480 | 50 / 156 ms | 39.6 / 63.8 ms | 31 s |
| M5 Max | V4 Flash | 40,960 | 71 / 292 ms | 75.1 / 138.4 ms | 85-90 s |
| M5 Max | V4 Flash | 60,746 | 106 / 393 ms | | 46-49 s |
| M5 Max | V4.1 Q4 | 4,096 | 16 / 62 ms | 11.2 / 20.0 ms | 67-71 s |
| M5 Max | V4.1 Q4 | 21,878 | 57 / 99 ms | | 88-94 s |

File I/O dominates on the M1 Ultra, so LZ4 adds a few milliseconds there. The
M5 Max reads and writes raw files faster, so the codec's share is larger; the
largest cost, 287 ms on a 60,746-token save, is 0.6% of the request that saved
it. V4.1 Q2 on the M1 Ultra loaded LZ4 checkpoints in 31 to 53 ms at 4K to 20K
tokens, with no matching raw phase.

Flipping 8 bytes in the middle of a stored V4 Flash or V4.1 checkpoint on
either machine made the next load report a corrupt compressed payload in 14.7
to 19.1 ms; the entry was removed, recomputed and stored again.

Qwen3.8 Flash Next checkpoints compress to 0.996x to 1.002x and are stored raw
under the 1/64 rule. At 10K tokens a save took 615 to 651 ms with compression
enabled and 287 to 442 ms with `--kv-cache-compression-threads 0`; loads took
100 and 126 ms.

## What ds4-bench measures

`ds4-bench` reuses live KV during incremental prefill. Between frontiers it can
save and restore an uncompressed session snapshot through `fmemopen`; snapshot
save/restore is outside both throughput timing windows. Its `kvcache_bytes`
column reports that memory snapshot's payload size, not a disk checkpoint size.
It also has an expert-weight cache in SSD streaming mode, a separate mechanism.

The executable does not link `ds4_kvstore.o`, `lz4.o`, or `lz4hc.o` and never reads
the compression-thread setting. An upstream-versus-PR benchmark is an inference
regression control. It cannot measure disk compression, save/load overhead,
cache-budget effects or time to first token after restoring a disk checkpoint.
Those need actual server restart tests and/or a separately labeled codec replay
benchmark. Changing the compression environment variable for `ds4-bench` is not
an on/off experiment for this feature.

## Regressions

```sh
make test-kv-lz4 test-kv-lz4-nofwrap
make ds4_test ds4_agent_test
./ds4_test --server
./ds4_agent_test
```

The codec suite covers byte-shuffle inversion, XXH32 against `xxhsum`
reference values, chunk boundaries, per-chunk checksums, malformed regions,
header versions through refresh, trailer positioning and raw fallback, and
asserts that no fuzzed region decodes to full-length wrong bytes. Store regressions replace only
the engine boundary and exercise actual staging, compression, admission,
eviction and publication with deterministic payloads. They cover fitting
compressed files, unnecessary eviction, write/rename failures, raw fallback
exceeding budget, incompressible payloads stored raw and loaded back, protection of an admitted file,
corrupt-file and checksum-mismatch replacement, removal of the retired codec-1
layout while other unknown codecs are kept, retention after engine failures
that report no errno and after stream read errors in the framing, a chunk
record or a block, byte-exact delivery to the engine,
retention and retry after reported allocation/I/O failures, and atomic
replacement of same-key incompatible files. `test-kv-lz4-nofwrap` builds both
suites without cookie streams: every codec test runs there too, and the store
cases load compressed entries through the temporary-file path, including a
temporary-file failure that must keep the entry. These fixtures
are not model-generated KV compression-ratio evidence.

## V4.1 throughput through 256K

M1 Ultra / 128 GiB / macOS 26.6.2, Metal SSD streaming, Q2 V4.1 model, 32 GiB expert-cache target, Engram disk-only, prefill chunk 8,192. One process incrementally prefills the same prose corpus to each frontier, generates 64 tokens, and restores the prompt snapshot before continuing. Generation includes the first token; steady generation excludes it.

| Context tokens | Tokens added | Prefill tokens/s | Generation tokens/s | Steady generation tokens/s |
|---:|---:|---:|---:|---:|
| 16,384 | 16,384 | 236.91 | 4.85 | 5.50 |
| 32,768 | 16,384 | 204.91 | 5.17 | 5.64 |
| 65,536 | 32,768 | 244.39 | 5.06 | 5.42 |
| 131,072 | 65,536 | 259.51 | 4.85 | 5.25 |
| 262,144 | 131,072 | 243.16 | 4.42 | 4.95 |

These are single measurements, not repeated-run confidence intervals. Prefill rates apply to the **tokens added** column, not a fresh prompt from zero at every row. The context allocation is 262,209; the planned memory footprint was 51.74 GiB. The complete run, including startup/snapshots/generation, took 1,148 seconds. System-wide VM counters recorded 169.75 MiB of swap-ins and zero swap-outs during the run; these counters do not identify which process caused the traffic. No simultaneous model, compilation or stress test was launched by this task.

All five frontier vectors contain 129,280 finite values. The six same-configuration upstream/PR controls match bit-for-bit at 4K and 16K, including signed zero. The smaller-chunk experiment also matches those vectors. The validator rejects missing entries, nulls and NaNs and detects a deliberately changed finite value. This establishes the tested numerical invariants; it is not an official-reference long-context quality evaluation.

The CSV's final `kvcache_bytes=0` means the final frontier needs no saved snapshot. It does not mean KV memory is absent. At 128K the uncompressed memory snapshot was 850,388,020 bytes (811.0 MiB); this is not a compressed disk-file measurement.

```sh
mkdir -p /tmp/v41-long-logits
./ds4-bench -m /path/to/DeepSeek-V4.1-Flash-Q2.gguf \
  --ssd-streaming --ssd-streaming-cold --ssd-streaming-cache-experts 32GB \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 16384 --ctx-max 262144 --step-mul 2 --gen-tokens 64 \
  --csv /tmp/v41-long.csv --dump-frontier-logits-dir /tmp/v41-long-logits
```
