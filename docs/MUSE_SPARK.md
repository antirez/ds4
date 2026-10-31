# Muse Spark Open on Apple Silicon

This branch adds experimental Metal support for Meta's Muse Spark Open model,
including its 816B total / 42B active mixture-of-experts architecture. The
current target is the six-shard UD-IQ2_XXS release, which maps about 210 GiB of
weights and therefore requires a Mac with at least 256 GiB of unified memory.

Model weights are not included in this repository. Supply the six GGUF shards
locally, beginning with `Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-00001-of-00006.gguf`.

## Run the model

Build the native server and coding agent:

```sh
make -j8 ds4-server ds4-agent
```

Point `-m` at the first shard; DwarfStar discovers the other five shards
automatically. A 16K context is a practical starting point for the 256 GB M3
Ultra configuration.

Start the OpenAI-compatible server at `http://127.0.0.1:8000`:

```sh
MUSE=/path/to/UD-IQ2_XXS/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-00001-of-00006.gguf

./ds4-server --metal -m "$MUSE" --ctx 16384 \
  --host 127.0.0.1 --port 8000
```

Or start the native terminal coding agent directly. It operates in the current
working directory and does not require the server:

```sh
./ds4-agent --metal -m "$MUSE" --ctx 16384
```

### Install the `muse` command

After building `ds4-agent`, run the repository's portable launcher once with
the first GGUF shard:

```sh
./muse --install /path/to/UD-IQ2_XXS/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-00001-of-00006.gguf
```

This saves the model location in `~/.config/ds4/muse-model` and creates
`~/bin/muse`. If `~/bin` is not already on `PATH`, the installer prints the
single line to add to the shell profile. After that, run Muse from any project:

```sh
cd /path/to/project
muse
```

The command launches DwarfStar's native chat agent in the current directory
with Metal and a 16K context. Override the context with `CTX=8192 muse`, or
temporarily override the saved model with `MUSE_MODEL=/path/to/model.gguf muse`.

Prompt lookup is enabled automatically and can accelerate repetitive code. To
measure ordinary single-token decoding, prefix either command with
`DS4_MUSE_NO_PROMPT_LOOKUP=1`.

## Current performance

Measured on an Apple M3 Ultra with 256 GiB of unified memory:

| Mode | Result |
| --- | ---: |
| Plain decode, 128-token prompt | 18.30 tok/s steady |
| Prompt prefill | about 47.9 tok/s |
| Prompt lookup on repetitive code | 27.7-29.0 tok/s |

The plain decode number uses 24 generated tokens with prompt lookup disabled.
Prompt lookup only reaches the higher range when generated text repeats tokens
already present in the prompt; ordinary generation remains near 18 tok/s.

```sh
make -j8 ds4-bench

MUSE=/path/to/UD-IQ2_XXS/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-00001-of-00006.gguf
PROMPT=/path/to/prompt.txt

DS4_MUSE_NO_PROMPT_LOOKUP=1 ./ds4-bench --metal \
  -m "$MUSE" \
  --chat-prompt-file "$PROMPT" \
  --ctx-start 128 --ctx-max 128 --ctx-alloc 160 --gen-tokens 24
```

## Implemented work

- Muse model loading, graph construction, tokenizer/chat defaults, and Metal
  kernels.
- A parallel router path that reduced router time across all layers from about
  11.0 ms to 1.4 ms per generated token. This raised plain decode from about
  15 tok/s to more than 17.6 tok/s while preserving exact logits.
- GPU argmax and prompt lookup decoding.
- Exact grouped IQ2 verification for prompt lookup candidates.
- Paired IQ2 gate/up processing, which improved prefill by about 21 percent and
  was neutral for single-token decode.
- Fused weighted RMS processing with exact output and neutral decode speed.
- A converter for Muse DSpark draft checkpoints and an optimized proposal path.

The current DSpark draft does not accept enough proposed tokens on ordinary
prompts to beat plain decoding. In measured runs it produced about 16.7-17.0
tok/s, so prompt lookup and plain decode remain the useful modes.

## Current decode costs

A representative trace attributes the largest costs per token to:

| Work | Time |
| --- | ---: |
| Dense Q8 projections | about 21.1 ms |
| IQ2 expert mid/down projections | about 17.9 ms |
| Dense Q6 projections, including output head | about 14.1 ms |
| Host boundary | about 1.4 ms |

## Experiments that did not produce a repeatable gain

These changes were measured and then reverted:

- IQ2 threadgroup-size sweeps, packed-word table lookup, and four-way
  accumulators.
- Copying IQ2 lookup tables to threadgroup memory. It preserved exact logits
  but measured 18.05 tok/s against an 18.24 tok/s baseline.
- Aligned and repacked IQ2 prototypes.
- Paired Q8 K/V dispatch: 18.31 tok/s against an 18.30 tok/s baseline.
- Q8 and Q6 row-count/threadgroup sweeps, with no repeatable improvement.
- An alternate Q6 kernel, expanded Q6 caching, and attempted command overlap.
- Grouped MoE prefill changes that improved prefill but changed numerical
  behavior or reduced decode speed.

## Useful contribution targets

1. Distill or retrain the Muse draft model against this exact IQ2 target to
   improve speculative-decoding acceptance.
2. Develop a Muse-specific IQ2 expert decode kernel while preserving the same
   values and left-to-right accumulation order.
3. Improve the dense Q8 and Q6 kernels with an end-to-end decode gain.
4. Reproduce the benchmark on other Apple Silicon systems with 256 GiB or more.

For kernel changes, compare all 202,048 output logits with the current exact
path and report the end-to-end benchmark above. Microbenchmarks alone can hide
dispatch, memory, or synchronization costs.

## DSpark conversion

The converter accepts a Muse DSpark safetensors checkpoint and writes GGUF v3:

```sh
python3 gguf-tools/muse_dspark_convert.py \
  /path/to/model.safetensors \
  /path/to/muse-spark-dspark.gguf
```

Pass the resulting file with `--mtp-model` and enable DSpark with `--dspark`.
The draft checkpoint is optional and is not included in this repository.
