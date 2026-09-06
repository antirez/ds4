# Qwen chat tool-turn continuation

The visible-checkpoint optimization proposed in PR #2 is incorporated with
an assistant-boundary correction. Qwen tool generation stops at
`</tool_call>`, before evaluating `<|im_end|>`. The saved visible key therefore
excludes `<|im_end|>`; a reasoning-omitting client's next request supplies
that token in its new suffix while preserving the exact sampled KV state.
Repaired/truncated calls and calls recovered from unclosed reasoning do not
use this shortcut. The change is restricted to Qwen chat completions without
images; existing Responses and Anthropic paths are retained.

Validation on September 7, 2026, M3 Ultra, 512 GiB:

- `make -j8 ds4-server ds4_test` succeeded. The existing format warning at
  `tests/ds4_test.c:310` remains; no new server warning was introduced.
- `./ds4_test --server` passed, including the new renderer regression covering
  thinking on/off, with/without visible content, reasoning replay, images,
  Responses, truncation, and unclosed reasoning. The continuation must begin
  with `<|im_end|>` followed by the user/tool-response turn.
- Two live three-request weather-tool loops passed on Q4_K-imatrix with
  external PLE: one non-streaming and one streaming. Both replayed tool calls
  without reasoning, looked up Paris then Rome, and returned the supplied
  Rome temperature (19 C). Both successive tool turns reused live KV.

| Transport | Continuation | Cached tokens | Effective prompt | New tokens |
| --- | --- | ---: | ---: | ---: |
| Non-streaming | First tool result + next query | 383 | 420 | 37 |
| Non-streaming | Second tool result | 468 | 492 | 24 |
| Streaming | First tool result + next query | 391 | 428 | 37 |
| Streaming | Second tool result | 481 | 505 | 24 |

Live settings: Metal, 8K context, 1024-token prefill chunk, temperature 0,
low reasoning effort, 1024 output tokens maximum, no MTP or images. Tests ran
serially against a temporary localhost server, which was stopped afterward.
Raw requests/results, logs and the original failing boundary probe remain in
the local ignored `OUT/pr2-review` directory.

These short loops establish suffix reuse and successful tool continuation;
they do not reproduce the PR author's 75K-context speed measurement or
validate interleaved batched sessions, disk-cache restoration, or MTP.
