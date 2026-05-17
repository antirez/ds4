# DeepSeek v4 model card synopsis

This document extracts the most important information from the official
DeepSeek-V4-Flash Hugging Face model card, with emphasis on facts that matter
for local inference, DS4 development, and benchmark interpretation.

Source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash

## Model Family

DeepSeek-V4 is a preview model family with two Mixture-of-Experts language
models:

| Model | Total parameters | Active parameters | Context length |
|---|---:|---:|---:|
| DeepSeek-V4-Flash | 284B | 13B | 1M tokens |
| DeepSeek-V4-Pro | 1.6T | 49B | 1M tokens |

Flash is the smaller and more efficient model. The model card says Flash-Max can
approach Pro reasoning performance when given a larger thinking budget, while
remaining behind Pro on pure knowledge and the most complex agentic tasks.

## Architecture

DeepSeek-V4 uses long-context compressed attention. The model card calls the
hybrid design Compressed Sparse Attention (CSA) plus Heavily Compressed
Attention (HCA). In DS4 terms, each layer keeps a raw sliding-window KV cache
for the latest 128 tokens. This is the high-resolution local context.

After that raw window, the model uses layer-dependent compressed KV rows:

| 0-based layer indexes | DS4 ratio | Extra state | Meaning |
|---|---:|---|---|
| 0, 1 | none | none | Raw 128-token sliding window only |
| even layers from 2 onward | 4 | compressed KV + indexer KV | One compressed row per 4 tokens, with an indexer selecting visible compressed rows |
| odd layers from 3 onward | 128 | compressed KV | One compressed row per 128 tokens |

So, after the first two layers, the model alternates ratio-4 and ratio-128
compressed attention. A token in a compressed layer attends over both the raw
latest-128-token window and the older compressed history. The compression here
is time-axis compression: several token positions are pooled into one KV row.
The attention rows still use the model attention/value dimensions, so raw and
compressed rows can be consumed by the same mixed-attention computation.

Ratio-4 layers are the selective compressed-attention layers. They maintain a
second compressed stream for the indexer, and when the compressed history is
larger than the configured top-k, DS4 scores the compressed rows and selects up
to 512 of them for attention. Ratio-128 layers are the heavily compressed path:
they do not have the indexer stream and use the available ratio-128 compressed
rows directly.

DS4 validates these details from the GGUF metadata. The relevant fixed
implementation constants are:

- Layers: 43
- Raw sliding-window attention: 128 tokens
- Indexer heads: 64
- Indexer head dimension: 128
- Indexer top-k: 512

This is the practical reason the model can expose a 1M-token context without a
standard full KV cache for every token in every layer. The model card reports
that, at 1M tokens, DeepSeek-V4-Pro needs much less single-token inference
compute and KV cache than DeepSeek-V3.2.

The family also uses:

- Manifold-Constrained Hyper-Connections (mHC), intended to improve signal
  propagation stability across layers.
- The Muon optimizer, used for faster convergence and training stability.
- A post-training pipeline with domain expert cultivation followed by unified
  consolidation via on-policy distillation.

## Precision And Weights

Official download entries include:

| Model | Precision |
|---|---|
| DeepSeek-V4-Flash-Base | FP8 Mixed |
| DeepSeek-V4-Flash | FP4 + FP8 Mixed |
| DeepSeek-V4-Pro-Base | FP8 Mixed |
| DeepSeek-V4-Pro | FP4 + FP8 Mixed |

For the instruct models, the model card describes FP4 + FP8 Mixed as using FP4
for MoE expert parameters and FP8 for most other parameters.

## Reasoning Modes

The instruct models support three reasoning-effort modes:

| Mode | Intended behavior | Output shape |
|---|---|---|
| Non-think | Fast, intuitive replies | `</think>` summary |
| High | Deliberate reasoning for harder tasks | `<think>... </think>` summary |
| Max | Largest reasoning budget | Special system prompt plus thinking and summary |

The model card recommends using at least a 384K-token context window for Think
Max.

## Important Flash Benchmarks

### DeepSeek-V4-Flash Across Reasoning Modes

| Benchmark | Non-Think | High | Max |
|---|---:|---:|---:|
| GPQA Diamond Pass@1 | 71.2 | 87.4 | 88.1 |
| MMLU-Pro EM | 83.0 | 86.4 | 86.2 |
| SimpleQA-Verified Pass@1 | 23.1 | 28.9 | 34.1 |
| Chinese-SimpleQA Pass@1 | 71.5 | 73.2 | 78.9 |
| HLE Pass@1 | 8.1 | 29.4 | 34.8 |
| LiveCodeBench Pass@1 | 55.2 | 88.4 | 91.6 |
| HMMT 2026 Feb Pass@1 | 40.8 | 91.9 | 94.8 |
| IMOAnswerBench Pass@1 | 41.9 | 85.1 | 88.4 |
| SWE Verified Resolved | 73.7 | 78.6 | 79.0 |
| Terminal Bench 2.0 Acc | 49.1 | 56.6 | 56.9 |
| MCPAtlas Pass@1 | 64.0 | 67.4 | 69.0 |
| Toolathlon Pass@1 | 40.7 | 43.5 | 47.8 |

### DeepSeek-V4-Flash-Base

The base-model table reports these Flash-Base scores:

| Benchmark | Shots | Score |
|---|---:|---:|
| SuperGPQA EM | 5-shot | 46.5 |
| MMLU EM | 5-shot | 88.7 |
| MMLU-Pro EM | 5-shot | 68.3 |
| Simple-QA verified EM | 25-shot | 30.1 |
| HumanEval Pass@1 | 0-shot | 69.5 |
| GSM8K EM | 8-shot | 90.8 |
| LongBench-V2 EM | 1-shot | 44.7 |

The model card reports SuperGPQA for the base model table, not in the instruct
reasoning-mode comparison table.

## Chat Template And Encoding

The release does not use a Jinja chat template as the source of truth. The
official prompt renderer is the Python code in
`encoding/encoding_dsv4.py`, with examples and tests in the same `encoding`
directory:

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/encoding/encoding_dsv4.py
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/encoding/test_encoding_dsv4.py

The important special tokens are:

| Purpose | Token |
|---|---|
| Beginning of sequence | `<｜begin▁of▁sentence｜>` |
| End of assistant turn | `<｜end▁of▁sentence｜>` |
| User turn prefix | `<｜User｜>` |
| Assistant turn prefix | `<｜Assistant｜>` |
| Latest reminder prefix | `<｜latest_reminder｜>` |
| Thinking start | `<think>` |
| Thinking end / non-thinking marker | `</think>` |
| DSML tool markup marker | `｜DSML｜` |

The renderer accepts `system`, `user`, `assistant`, `tool`,
`latest_reminder`, and `developer` roles. The `developer` role is described in
the Python comments as an internal search-agent role, not as a normal public
chat role.

Normal chat mode starts with the BOS token, then system text if present, then
alternating user and assistant markers. In non-thinking chat mode, a new
assistant generation is opened with:

```text
<｜Assistant｜></think>
```

That immediate `</think>` tells the model to skip hidden reasoning and produce
the visible answer. In thinking mode, a new assistant generation is opened with:

```text
<｜Assistant｜><think>
```

Completed assistant thinking turns are rendered as reasoning content inside
`<think>...</think>`, followed by the visible answer and the EOS token.

By default, the Python renderer drops earlier assistant reasoning content before
the last user message. If tools are present on any message, it disables that
reasoning drop and keeps the full reasoning/tool context. `reasoning_effort=max`
also prepends a special high-effort instruction prefix before the first rendered
message in thinking mode.

Tool definitions are passed in OpenAI-compatible function schema form, but the
model is instructed to emit DSML. A tool call is rendered as a DSML
`tool_calls` block containing one or more `invoke` entries, each with named
parameters. Parameters carry a `string="true"` flag for raw strings and
`string="false"` for JSON values such as numbers, booleans, arrays, or objects.

DeepSeek-V4 does not render standalone `tool` role messages. The Python
preprocessor converts tool results into user content blocks and renders each
result as:

```text
<tool_result>...</tool_result>
```

Tool-result bodies are rendered as raw text. Literal `<`, `>`, and `&` from
file contents or shell output are preserved; only the exact closing sentinel
`</tool_result>` is escaped so the wrapper cannot be terminated by data.

When there are multiple tool results, the renderer sorts them to match the
order of the preceding assistant tool calls.

The same script also defines special task tokens for internal quick tasks such
as title generation, search-query generation, action selection, authority
classification, domain classification, and URL-read decisions. Those are
separate from normal chat/tool rendering.

## Local Running Notes

The model card lists vLLM and SGLang examples for OpenAI-compatible serving.
For local deployment, it recommends:

- `temperature = 1.0`
- `top_p = 1.0`
- At least 384K context for Think Max

These are deployment recommendations from the model card, not necessarily the
same settings used for deterministic benchmarking. DS4 keeps `top_p=1.0` but
adds a local `min_p=0.05` default to avoid sampling tokens whose probability is
far below the best token.

## Licensing

The repository and model weights are licensed under the MIT License.

## Citation

The model card cites:

```bibtex
@misc{deepseekai2026deepseekv4,
      title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
      author={DeepSeek-AI},
      year={2026},
}
```
