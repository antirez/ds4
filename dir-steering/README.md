# Directional Steering

Directional steering is a runtime activation edit for DS4. A steering file is a
flat `f32` matrix with one normalized hidden-width direction per normal
transformer layer. During inference, ds4 can apply the edit after attention
outputs, FFN outputs, or both:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

Positive scale removes the represented direction. Negative scale amplifies it.
With no steering file or zero scales, ds4 follows the normal inference path.

The file shape depends on the model:

- DeepSeek V4 Flash: `43 x 4096`.
- GLM 5.3 Flash: `45 x 4096`. The separate MTP predictor layer is omitted.

GLM 5.2 steering is not implemented.

`--dir-steering-file` also accepts a **GLP file** — the same directions in a
GGUF container that states what they are. See [GLP files](#glp-files) below.

## Runtime Options

```text
--dir-steering-file FILE   one direction per normal model layer: a GLP .gguf,
                           or the raw f32 blob described above
--dir-steering-ffn F       apply steering after FFN outputs; default is 1 when a file is provided
--dir-steering-attn F      apply steering after attention outputs; default is 0
--dir-steering-info FILE   print a GLP vector's metadata and exit; loads no model
--dir-steering-allow-hook-mismatch
                           apply a GLP vector at a hook it was not calibrated for
```

The format is chosen by sniffing the file, not by a flag, so every existing raw
vector keeps working unchanged.

The FFN output is usually the best first target because it is late enough in
each layer to represent behavior, style, and topic signals. Attention steering
is available for experiments, but it can be more fragile.

## GLP files

The raw format is a headerless blob. That is fine on the machine that produced
it and unsafe to hand to anyone else, because it cannot say four things that
are each silently wrong when they are wrong:

**The operation.** llama.cpp has shipped control vectors since 2024 with the
tensor convention this format reuses exactly — tensors named `direction.<N>`,
fp32, 1-D — but llama.cpp *adds* them:

```text
h <- h + v                      ADD:      steer towards a direction
y -= scale * v * dot(v, y)      PROJECT:  delete the component along v  (ds4)
```

These are different operations, and the difference is not a scale factor. A
file loads into either runtime with the right tensor names, the right dtype and
the right shapes, raises no error, and produces wrong output in one of them: an
additive apply of a projective direction pushes every token *along* the
direction instead of removing it.

**The hook point.** ds4 steers the block writers — `ffn_out = moe + shared`, or
the attention output — before the residual / hyper-connection fold. llama.cpp's
`build_cvec()` and the weightless vLLM overlay steer the post-layer residual.
Measured on the same direction, the same layers and the same alpha, the writer
site left 34.0% refusal against 3.8% post-layer: 9x weaker, and not an error.

**The layer map.** `direction.N` applies at layer `N`, with no offset. A
one-layer shift does not fail, it degrades — adjacent layers' refusal
directions have cosine similarity 0.555–0.979 — so it survives a smoke test.

**The base checkpoint.** A direction is tied to the exact revision it was
derived from. Applying one elsewhere is undefined, and the shapes still match.

GLP (GGUF Layer Projection) is standard GGUF v3 with llama.cpp's tensor
convention unchanged, plus a `glp.*` block that states all four. ds4 refuses a
file rather than misapplying it: wrong operation, a hook it was not calibrated
for, tensor names disagreeing with the declared layer list, `rank > 1`, a
non-F32 direction, or a width that is not this model's `n_embd`. Full spec:
[weightless/spec/GLP.md](https://github.com/msuiche/weightless/blob/main/spec/GLP.md).

Inspect a vector without loading the model it belongs to — the first thing to
reach for when one behaves oddly:

```sh
./ds4 --dir-steering-info dir-steering/out/verbosity-GLP.gguf
```

```text
GLP vector: dir-steering/out/verbosity-GLP.gguf
  mode           project
  spec_version   1
  hook_point     ffn_out_pre_residual
  alpha_default  2
  rank           1
  coverage       42 directions, n_embd=4096, layers 1..42
  direction norm 1.000000 .. 1.000000
  base model     DeepSeek-V4-Flash-0731
  method         paired_difference_of_means
  contrast       succinct.txt_vs_verbose.txt
  content sha256 2fc93b0a...
```

When the file's hook matches the site ds4 is about to steer, `glp.alpha_default`
becomes the default `--dir-steering-ffn`, so a published vector runs at the
strength it was calibrated at with no flags:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 160 \
  --dir-steering-file verbosity-GLP.gguf \
  -p "Explain why databases use indexes."
```

An explicit `--dir-steering-ffn` always wins, and alpha is never adopted from a
file whose hook does not match: projection is quadratic in the direction's
norm and the same direction at a different site measured 9x weaker, so an alpha
from elsewhere is not a better default than 1.0, only a more confident wrong
one.

### Publishing a vector

`f32_to_glp.py` packages a raw vector built by `build_direction.py`, reading the
sidecar JSON so the shape and the hook point cannot disagree with what was
measured:

```sh
python3 dir-steering/tools/f32_to_glp.py \
  dir-steering/out/verbosity.f32 dir-steering/out/verbosity-GLP.gguf \
  --meta dir-steering/out/verbosity.json \
  --layers 1-42 \
  --base-model DeepSeek-V4-Flash-0731 \
  --base-org deepseek-ai \
  --base-revision <hf commit> \
  --alpha 2.0
```

It normalises each direction to unit length, writes `glp.content_sha256` over
the tensor bytes only — `glp.created` makes the file non-reproducible, so the
hash is what lets two people confirm they hold the same direction — and reads
the result back to assert the layer ids landed verbatim and the values survived
exactly. No third-party Python packages.

**Layer 0 cannot be expressed.** `direction.N` applies at layer `N` and
`direction.0` is invalid, so the container has no slot for layer 0 — a
difference from the raw format, which has a row for it. `build_direction.py`
fills every layer, so a full-coverage vector needs `--layers 1-<last>` and
loses layer 0. Vectors derived over a mid-stack range (ours cover L10–38) are
unaffected. Coverage is the lever that matters most here — 6 layers 18%, 16
layers 3.8%, 29 layers 0.0% on our refusal suites, while alpha saturates above
about 4 — so losing the first layer of a 43-layer stack is not what decides a
vector's strength.

## GLM 5.3 Example

Build a GLM 5.3 direction from paired target and control prompt lists:

```sh
python3 dir-steering/tools/build_direction.py \
  --profile glm-5.3-flash \
  --ds4 ./ds4 \
  --model gguf/GLM-5.3-Flash-Q2.gguf \
  --good-file /path/to/target-prompts.txt \
  --bad-file /path/to/control-prompts.txt \
  --out dir-steering/out/glm53-direction.json \
  --component ffn_out \
  --ctx 512
```

Generated `.f32` vectors are local artifacts and are not stored in the
repository. GLM 5.3 steering works with `--mtp`, `ds4-server`, native session
batching, and two-Mac tensor parallelism. For tensor parallelism, pass the same
steering file and scales to both the worker and coordinator.

## Verbosity Example

The bundled example builds a style direction from 100 paired prompts. Each pair
asks for the same information in two ways:

- `examples/succinct.txt`: terse target prompts.
- `examples/verbose.txt`: detailed contrast prompts.

Because the extracted direction is `succinct - verbose`, negative FFN scales
make answers shorter, while positive FFN scales tend to make answers longer and
more explanatory.

Build the vector:

```sh
python3 dir-steering/tools/build_direction.py \
  --profile deepseek-v4-flash \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --good-file dir-steering/examples/succinct.txt \
  --bad-file dir-steering/examples/verbose.txt \
  --out dir-steering/out/verbosity.json \
  --component ffn_out \
  --ctx 512
```

This writes:

```text
dir-steering/out/verbosity.json
dir-steering/out/verbosity.f32
```

Try a terse run:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 160 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1 \
  -p "Explain why databases use indexes."
```

Try a verbose run:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 220 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn 2 \
  -p "Explain why databases use indexes."
```

The same vector can be used in either direction. The sign is the important part:

- negative scale amplifies the succinct target direction;
- positive scale suppresses that direction and usually gives the model more room
  to elaborate.

## Evaluating Scales

Use the sweep helper to test several strengths on a fixed prompt set:

```sh
python3 dir-steering/tools/run_sweep.py \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --direction dir-steering/out/verbosity.f32 \
  --prompts dir-steering/examples/eval_prompts.txt \
  --scales "-1,-0.5,0,0.5,1,2" \
  --tokens 180 \
  --nothink
```

Start with FFN scales between `-1` and `2`. If the model becomes repetitive,
ignores the prompt, or starts losing factual content, the scale is too strong.
For this example, `-1` is a good first terse setting and `2` is a good first
verbose setting. Strong negative scales such as `-2` or `-3` can over-amplify
the terse direction and collapse into repetition on some prompts.

## Observed Effect

With the 100-pair vector built from the commands above, local greedy checks
showed the expected behavior:

- Prompt: `Explain why databases use indexes.`
- `--dir-steering-ffn -1`: 67 words, one compact paragraph.
- `--dir-steering-ffn 0`: 136 words, structured explanation.
- `--dir-steering-ffn 1`: 140 words, structured explanation with more detail.

On a prompt that the unsteered model already answered briefly, positive steering
made the expansion more visible:

- Prompt: `What does DNS do?`
- `--dir-steering-ffn 0`: 44 words.
- `--dir-steering-ffn 2`: 171 words, with sections and step-by-step detail.

## Building Other Directions

The extractor compares two prompt sets:

- `good-file`: target prompts for the direction you want to represent.
- `bad-file`: contrast prompts that should be separated from the target.

It captures DS4 activations from the same local GPU graph used for inference,
averages target minus contrast, normalizes one vector per layer, and writes both
metadata JSON and the runtime `.f32` file.

Concept removal:

1. Put concept-heavy prompts in `good-file`.
2. Put neutral prompts in `bad-file`.
3. Run with a positive FFN scale.

Concept amplification:

1. Put desired concept prompts in `good-file`.
2. Put neutral prompts in `bad-file`.
3. Run with a negative FFN scale.

Style control:

1. Put prompts for the target style in `good-file`.
2. Put contrasting style prompts in `bad-file`.
3. Use negative scale to amplify the target style, positive scale to reduce it.

The method is not a fine-tune. It is a low-rank runtime edit, so it works best
for coarse behavior, topic, or style directions that are consistently present in
the activation captures.
