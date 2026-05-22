# Directional Steering

Directional steering is a runtime activation edit for DS4. A steering file is a
flat `f32` matrix with one normalized 4096-wide direction per layer. During
inference, ds4 can apply the edit after attention outputs, FFN outputs, or both:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

Positive scale removes the represented direction. Negative scale amplifies it.
With no steering file or zero scales, ds4 follows the normal inference path.

## Runtime Options

```text
--dir-steering-file FILE   load a 43 x 4096 f32 direction file
--dir-steering-ffn F       apply steering after FFN outputs; default is 1 when a file is provided
--dir-steering-attn F      apply steering after attention outputs; default is 0
--dir-steering-policy MODE server-only policy: final-answer, decoding, always, or off; default is final-answer
```

The FFN output is usually the best first target because it is late enough in
each layer to represent behavior, style, and topic signals. Attention steering
is available for experiments, but it can be more fragile.

For tool-using agents, `ds4-server` defaults to `--dir-steering-policy
final-answer`. This keeps prompt prefill, thinking tokens, and DSML tool-call
tokens unsteered. Steering is re-enabled only after generation has clearly
entered final natural-language answer text. This avoids letting a
behavior/style vector perturb tool-call grammar while still allowing the final
prose to use the configured direction.

`--dir-steering-policy decoding` is a middle ground for experiments that should
leave prompt/prefill activations untouched but steer every generated token,
including thinking and tool-call syntax. `always` restores the original
always-on behavior, and `off` disables directional steering at the server policy
layer.

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
