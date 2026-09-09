# DeepSeek V4 Pro 0813 Conversion

This is the runbook for producing the DS4 Q2 quant of the August 2026
DeepSeek V4 Pro checkpoint.

## Current State

- The official weights were published as
  `deepseek-ai/DeepSeek-V4-Pro-0813` on 2026-08-13. The conversion is pinned to
  revision `72e1d3230f6c080a530b0a1d46f8eb4602340597`: 92 files totaling
  892,762,497,859 bytes, including 66 safetensor shards totaling
  892,744,322,880 bytes. The download to `mac-m5max-us` completed and passed
  the pinned size manifest on 2026-08-15.
- `/Users/antirez/ds4-pro-0813/monitor_download.py` checks the download every
  five minutes. It writes `logs/READY_FOR_CONVERSION` only after the downloader
  exits successfully and all 92 file sizes, the 66-shard index, and incomplete
  download state validate against the pinned manifest. Do not start conversion
  without that marker.
- The new 100-case official continuation fixture is complete in
  `gguf-tools/quality-testing/data/pro-0813`. All 2,275 returned tokens include
  top-20 log probabilities and report the fingerprint
  `fp_v4pro_20260812_prod0820_fp8_kvcache_20260402`.
- The June preview fixture remains in `gguf-tools/quality-testing/data/pro`.
  Only 12 of the 100 new continuations are byte-identical to it, so the two
  checkpoints must never share a quality oracle.
- The provisional Q2 conversion completed on 2026-08-15 in 12,817 seconds.
  It has the planned size of 464,627,334,240 bytes and passed structural and
  SSD-streamed inference checks. Its log and exit status are in
  `logs/provisional-quantize.log` and `logs/provisional-quantize.exit`.
- The final imatrix Q2 conversion completed on 2026-08-15 in 13,810 seconds.
  The artifact is 464,627,334,560 bytes and passed structural checks, two
  SSD-streamed generation checks, and the complete 100-case 0813 quality
  comparison. Its log and exit status are in `logs/final-0813-quantize.log`
  and `logs/final-0813-quantize.exit`.
- Its SHA-256 is
  `c4d997ab9894b6c78b759f7869fe1726b6314b6515f6ff82607df3797c5eb193`.
- It was published in `antirez/deepseek-v4-gguf` as
  `DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix-0813.gguf`
  by Hub commit `86bb38ce2ba7a98ab0e550359fec5f48859dc723`.
- The imatrix quant improved average NLL by 8.8% over the provisional quant.
  It matched the API's top token on 89.67% of the 2,275 scored tokens, compared
  with 88.75% for the provisional file, and increased the average greedy
  prefix from 5.14 to 7.76 tokens.

## Rules

- Do the download, conversion, imatrix collection, and validation only on
  `mac-m5max-us`.
- The target is the GA checkpoint reported by the DeepSeek API as
  `DeepSeek-V4-Pro-0813`.
- Pin the exact source revision before downloading any weights. Record the
  repository, revision, publication time, file count, and total byte count.
- Do not use the older `deepseek-ai/DeepSeek-V4-Pro` weights unless that
  repository has received a new weight revision. Its preview revision at the
  start of this work was `b5968e9190ef611bbf34a7229255be88a0e937c1`, dated
  2026-06-22. Rebuilding it would only reproduce the existing quant.
- Never put Hugging Face, DeepSeek, or OpenRouter credentials in this file,
  command logs, manifests, or commits.

## Remote Safety: Avoid OOM

The M5 Max has 128 GB of unified memory; the PRO Q2 is about 465 GB. Never run
`ds4`, `ds4-bench`, `score_official`, or a model test with a PRO GGUF unless
the command explicitly includes both `--ssd-streaming` and
`--ssd-streaming-cache-experts 48GB`. An accidental resident load can exhaust
unified memory, make the host unresponsive, and force a reboot.

Before every PRO execution:

1. Re-read the complete command and verify both streaming flags are present.
2. Confirm that no other `ds4`, scorer, benchmark, or model process is running.
3. Start with a 4,096-token context and at most a 512-token imatrix probe.
4. Watch memory pressure and swap during startup and the first layer. Stop the
   process immediately if memory pressure stops being green or swap grows
   continuously.
5. Run only one PRO process at a time. Do not point broad test targets at the
   PRO file because they may invoke a binary without forwarding streaming
   options.

The initial 48 GB expert cache is deliberately conservative. Raise it to 64 GB
only after a short probe leaves at least 24 GB of memory headroom with stable
swap. Do not begin with an 80 GB cache on this machine. Quantization does not
execute the model and does not need the runtime streaming flags, but it must
still be the only large job running on the host.

## Paths On The M5 Max US

```text
source:       /Users/antirez/ds4-pro-0813/hf/DeepSeek-V4-Pro-0813
work:         /Users/antirez/ds4-pro-0813/work
logs:         /Users/antirez/ds4-pro-0813/logs
DS4 tree:     /Users/antirez/ds4-pro-0813/ds4-src
old template: /Users/antirez/ds4/gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf
final GGUF:   /Users/antirez/ds4/gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix-0813.gguf
```

At preparation time the machine had about 1.7 TiB free. The old PRO template
is 464,627,334,560 bytes and is also available in
`antirez/deepseek-v4-gguf`. Its local SHA-256 matched the hosted LFS object,
`a0314d9c0e16122cd60071079124a2d17185d317c55a8f95ecb3ed3506278a96`.
It was removed only after the provisional 0813 GGUF passed metadata and
inference checks and a fresh full imatrix was complete.

## 1. Detect And Pin The Release

The release appeared in the official Hugging Face namespace as
`deepseek-ai/DeepSeek-V4-Pro-0813`. Use only revision
`72e1d3230f6c080a530b0a1d46f8eb4602340597` for this conversion. A model name
in an API price list is not by itself evidence that open weights exist.

Before downloading:

1. Save the model API response in `logs/source-model-info.json`.
2. Save the recursive file listing in `logs/source-files.json`.
3. Record the pinned revision in `logs/source-revision.txt`.
4. Confirm that the index and safetensor object hashes differ from the June
   preview, not only README or inference-code files.
5. Sum all source file sizes and check peak disk use before starting.

Use the isolated downloader already installed at
`/Users/antirez/ds4-pro-0813-venv/bin/hf`. Pass the pinned revision explicitly
and download to the source path above. Preserve the downloader log and verify
that all files named by `model.safetensors.index.json` exist at their declared
sizes before conversion.

## 2. Build The Provisional Q2

Update `/Users/antirez/ds4` to the intended DS4 revision and build the runtime
and quantizer without warnings:

```sh
cd /Users/antirez/ds4
make clean && make
make -C gguf-tools clean all
```

The existing PRO GGUF provides the tokenizer, metadata, tensor order, shapes,
and output tensor types. The first pass uses the same tensor policy as the
published Q2 but uses the quantizer's deterministic weight-energy importance
for `IQ2_XXS` because no 0813 activation imatrix exists yet.

The 0813 `compress_ratios` array has 64 entries, compared with 62 in the June
template, because the new checkpoint contains three DSpark stages. The
quantizer must replace this one metadata record from the pinned HF
`config.json`; copying all template metadata verbatim would leave stale
checkpoint information. The 61 target-model entries are unchanged.

Run:

```sh
gguf-tools/deepseek4-quantize \
  --hf /Users/antirez/ds4-pro-0813/hf/DeepSeek-V4-Pro-0813 \
  --template gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf \
  --out /Users/antirez/ds4-pro-0813/work/DeepSeek-V4-Pro-0813-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-provisional.gguf \
  --threads 24
```

The expected policy is:

- routed gate and up experts: `IQ2_XXS`;
- routed down experts: `Q2_K`;
- attention projections, shared experts, and output: `Q8_0`;
- embeddings, router, indexer, compressor, HC, norms, biases, and other
  auxiliary tensors: the same types as the published PRO template.

Before committing to the full write, run `--dry-run` and regenerate at least
one routed gate, routed down, attention, shared-expert, and output tensor with
`--compare-tensor`. Shape or name mismatches mean the old template is not
compatible and must be handled explicitly, not ignored.

The 0813 preflight passed with 1,894 tensors, no output type changes, 384
experts, 64 config-derived compression ratios, and a planned size of
464,627,334,240 bytes. The five representative regenerated tensors matched
the template byte sizes exactly. Their checksums differ, as expected for a new
checkpoint.

The provisional conversion completed in 12,817 seconds with exit status zero
and the exact planned size. Its tensor schema and offsets match the old
template. The only ordinary metadata change is the intended compression-ratio
array; stale template imatrix provenance is absent. A one-token SSD-streamed
smoke test returned `OK` in 15.49 seconds with 37.35 GB peak resident memory,
no swap operations, and the required 48 GB cache and 4,096-token context.

After the provisional write, inspect its metadata and tensor-type counts, run
the loader checks, and generate a short coherent continuation with
`--ssd-streaming --ssd-streaming-cache-experts 48GB`. Do not delete the old
template until these checks pass.

## 3. Collect A Fresh 0813 Imatrix

PRO cannot reside in the M5 Max memory, so imatrix collection must use Metal
SSD streaming. SSD streaming is slow enough that the calibration budget must
be selected from a measurement, not copied blindly from the old run.

Start with one representative prompt and a small token cap. Use a warm expert
cache and preserve elapsed time, processed token count, cache settings, and
prefill speed:

```sh
cd /Users/antirez/ds4
./ds4 \
  -m /Users/antirez/ds4-pro-0813/work/DeepSeek-V4-Pro-0813-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-provisional.gguf \
  --ssd-streaming \
  --ssd-streaming-cache-experts 48GB \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out /Users/antirez/ds4-pro-0813/work/DeepSeek-V4-Pro-0813-imatrix-probe.dat \
  --imatrix-max-prompts 1 \
  --imatrix-max-tokens 512 \
  --ctx 4096
```

If the probe is healthy, repeat with 1,024 or 2,048 tokens to obtain a stable
rate. Estimate the wall time for 4K, 8K, 16K, and 32K calibration tokens. Use
the largest budget that is practical without turning calibration into a
multi-day blind run. Prefer prompt diversity over one long prompt: a light
matrix should cover several thinking, non-thinking, code, prose, Italian, and
tool-use prompts when time allows. Record the chosen prompt count, token count,
cache budget, measured speed, and estimated duration in this file before the
final run.

The old documented PRO budget was 16 prompts and 32,768 tokens. It is a target,
not a requirement. A smaller fresh 0813 activation matrix is preferable to
reusing the preview matrix or relying only on synthetic weight importance.

The 512-token probe completed in 44.07 seconds. The final calibration used all
16 selected prompts, totaling 25,954 available tokens, and completed in
760.28 seconds. It contains 183 unique routed entries, 976 layer-prompt
chunks, and 9,499,164 routed expert observations. Every entry has the expected
384 expert slices, the quantizer accepted all values as finite, peak resident
memory was 38.51 GB, and the run performed no swap operations.

Write the selected matrix to:

```text
/Users/antirez/ds4-pro-0813/work/DeepSeek-V4-Pro-0813-Instruct-routed-moe-ds4.dat
```

Require entries for all 61 PRO routed layers and all three routed tensor
classes. Check that each packed entry has the expected 384 expert slices and
that the file is complete before quantization.

## 4. Build The Final Q2

Use the validated provisional GGUF as the template so that the final metadata
comes from the 0813 conversion, not from the preview file:

```sh
cd /Users/antirez/ds4-pro-0813/ds4-src
gguf-tools/deepseek4-quantize \
  --hf /Users/antirez/ds4-pro-0813/hf/DeepSeek-V4-Pro-0813 \
  --template /Users/antirez/ds4-pro-0813/work/DeepSeek-V4-Pro-0813-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-provisional.gguf \
  --out /Users/antirez/ds4/gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix-0813.gguf \
  --imatrix /Users/antirez/ds4-pro-0813/work/DeepSeek-V4-Pro-0813-Instruct-routed-moe-ds4.dat \
  --imatrix-strict \
  --threads 24
```

The first final pass stopped at `blk.59.ffn_down_exps.weight`. One expert in
that tensor had a collapsed weighted search range: the quantizer accepted an
initial zero score and then divided by `max - min`, which was also zero. The
fix is to stop the Q2_K refinement loop when `max <= min`. Focused tests
reproduced the failure, passed after the fix, and confirmed that an ordinary
expert tensor remained byte-identical. The successful second pass emitted all
1,894 tensors with the planned schema and offsets.

If peak space becomes tight, remove only artifacts already verified as
downloadable from `antirez` Hugging Face. The old PRO template may be removed
after the provisional file is validated and its hosted size/hash have been
recorded. Never remove the source weights or provisional file before the final
GGUF is complete and structurally valid.

## 5. Capture 0813 Official Continuations

Keep the preview fixtures in `gguf-tools/quality-testing/data/pro`. Store the
new official DeepSeek API responses separately in:

```text
gguf-tools/quality-testing/data/pro-0813
```

Collect the same 100 prompts and 24 output tokens with top-20 log probabilities:

```sh
DEEPSEEK_API_KEY=... \
python3 gguf-tools/quality-testing/collect_official.py \
  --model deepseek-v4-pro \
  --endpoint https://api.deepseek.com/chat/completions \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/pro-0813 \
  --count 100 \
  --max-tokens 24 \
  --top-logprobs 20 \
  --thinking disabled \
  --reasoning-effort omit
```

Inspect every response for API errors and empty content. Record the collection
date and the API-reported model or system fingerprint in a small README inside
the fixture directory. The public model alias must resolve to 0813 at collection
time. Never overwrite `data/pro`.

The 0813 collection completed on 2026-08-12 UTC. It contains 100 valid
responses and 2,275 output tokens. Every token has 20 API alternatives. The
API returned `fp_v4pro_20260812_prod0820_fp8_kvcache_20260402`; the preserved
preview fixture reports `fp_9954b31ca7_prod0820_fp8_kvcache_20260402`.

## 6. Validation

All model work and model execution in this runbook remains on
`mac-m5max-us`.

1. Record final byte size and SHA-256.
2. Compare metadata, tensor names, dimensions, and tensor-type counts with the
   provisional file and published preview Q2. Expected weight changes are not
   byte differences in tensor layout.
3. Run malformed-file and loader sanity checks from `QA_BEFORE_RELEASES.md`.
4. Run short thinking and non-thinking prompts with
   `--ssd-streaming --ssd-streaming-cache-experts 48GB`; require coherent
   output and no missing tensors, cache failures, or Metal errors.
5. Score the final model against only the 0813 fixture:

   ```sh
   gguf-tools/quality-testing/score_official \
     /Users/antirez/ds4/gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix-0813.gguf \
     gguf-tools/quality-testing/data/pro-0813/manifest.tsv \
     /Users/antirez/ds4-pro-0813/logs/pro-0813-quality.tsv \
     4096 \
     --ssd-streaming \
     --ssd-streaming-cache-experts 48GB
   ```

6. Score the provisional model against the same fixture. Compare average NLL,
   first-token matches, greedy prefix length, API target MAE, top-1 agreement,
   top-N recall, and pair-order agreement. The imatrix result should not show a
   material quality regression; otherwise keep the provisional file and
   investigate the calibration budget before publishing.
7. Preserve the conversion log, imatrix log, scorer output, checksums, source
   revision, and disk-cleanup record under `/Users/antirez/ds4-pro-0813/logs`.

The final and provisional scorers both completed all 100 cases without swap
operations. The direct comparison is preserved in
`logs/provisional-vs-final-0813.txt`:

| Metric | Provisional | Final imatrix |
| --- | ---: | ---: |
| Average NLL | 0.331358 | 0.302183 |
| First-token matches | 56 | 64 |
| Average greedy prefix | 5.140 | 7.760 |
| API top-1 agreement | 88.75% | 89.67% |
| API top-N recall | 76.53% | 76.64% |
| API pair-order agreement | 98.88% | 98.97% |

The final scorer took 1,882 seconds and peaked at 37.41 GB resident memory.
The non-thinking smoke produced a correct C function at 2.64 generated tokens
per second. The thinking smoke correctly calculated a 15% discount on EUR 120
at 3.28 generated tokens per second. Both used a 48 GB expert budget and a
4,096-token context, and neither swapped.

Do not upload or change public download aliases until the final file passes
these checks.

The first Xet upload transferred 441 GB of new data, then failed during Hub
finalization with `timed out reading request body`. The retry reused all
content-addressed chunks, rescanned the file at about 1.1 GB/s, and committed
successfully without retransmitting the payload. Hub metadata reports the
exact local size and SHA-256. An official `hf download --dry-run` resolves the
dated file as one 464.6 GB download.
