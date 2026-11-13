# Maintaining the Flint fork

[Recursive Flint](RECURSIVE_FLINT.md)

This repository is a long-lived community fork of
[antirez/ds4](https://github.com/antirez/ds4) that adds Recursive Flint as
its own model family. The rules below keep the cost of following upstream
small. They are the Bonsai fork's rules (branch `bonsai2`), applied here.

## Layout rule: fork code lives in fork-owned files

| File | Contents |
|---|---|
| `ds4_flint_shape.inc` | `DS4_SHAPE_FLINT`, `ds4_model_is_flint()`, the global-layer mask, Flint-only constants |
| `ds4_flint_model.inc` | GGUF metadata validation, engine gate, `ds4_engine_is_flint()`; later weight binding, layout checks, the per-layer descriptor table |
| `ds4_flint.inc` | descriptor and expert-offset tables, silu table, engine open/close (resident, or streaming: the owned bank of resident weights, the cache budget, the unmapping of the expert region), the decode trace check |
| `ds4_flint_stream.h`, `ds4_flint_stream.c` | SSD streaming of the routed experts (plain C, linked on every backend): slot cache with exact LRU and pins, pread pool, decode fetch for the abort-and-resume loop, prefill staging with whole-layer read-ahead, statistics (`DS4_FLINT_STREAM_STATS=1`) |
| `ds4_flint_session.inc` | session driver: create, free, chunked prefill (sync), decode (eval), one-token rewind, payload save/load (disk checkpoints and snapshots), and the `--flint-prefill-trace` / `--flint-generate` check modes |
| `ds4_flint_gpu.h`, `ds4_metal_flint.m`, `metal/flint.metal`, `metal/flint_prefill.metal` | GPU API; its own strict-math Metal libraries, model buffer, KV blocks, launch and readback; the ported decode megakernel (`licenses/rec-NOTICE.txt`); the prefill kernels (GEMM, grouped expert GEMMs, routing, attention, norms, RoPE, embedding, head) and the chunk orchestration with its engine-wide scratch |
| `ds4_flint_tokenizer.inc` | pre-tokenizer; the chat template's roles, stop tokens and the assistant prefix with thinking on (`<think>`) or off (`<think></think>`) for the CLI and the agent |
| `ds4_flint_server.inc`, `ds4_flint_server_test.inc` | the server's Flint prompt renderer after rec's `chat_template.jinja` (tool preamble with Python-style JSON, tool calls, observations, live tool tail), sampling defaults, model aliases; its unit test (runs in `ds4_test --server`) |
| `ds4_metal_internal.h` | trampolines exposing `ds4_metal.m` statics to out-of-tree kernel files (identical to `bonsai2`'s) |
| `fork.mk` | build rules and fork test targets (included from `Makefile`) |
| `gguf-tools/flint_int4_to_gguf.py`, `gguf-tools/gen_flint_unicode.py` | rec's INT4 checkpoint to GGUF; the pre-tokenizer's Unicode class tables (`ds4_flint_unicode.inc`) |
| `tests/flint/` | `check_tokenizer.py` (parity with HF tokenizers; needs the `tokenizers` package), `collect_golden.py` + `golden.json` (from `rec serve`; `golden_prompt_ids.py` adds rec's prompt ids), `rec_logits.py` / `rec_prefill_dump.py` (logits, traces, KV and prefill activations from rec's runtime; they re-exec themselves in `~/.local/share/rec/runtimes/<sha>/python`), `trace_compare.py` (decode, bit-exact), `prefill_compare.py` (prefill, layer by layer), `check_golden.py` (teacher-forced and greedy against rec serve), `reuse_check.py` (gate 2), `stream_check.py` (streamed output identical to resident), `route_sim.py` (LRU miss fractions from `DS4_FLINT_ROUTE_LOG` logs), `check_template.py` (server prompts byte-identical to rec's jinja; needs `jinja2`), `bench_vs_rec.py` (rec serve and ds4-server through one streaming client; step 3), `test_flint_kernels.m` (`make test-flint-kernels`, model-free), `test_flint_stream.c` + `flint_pread_probe.h` (`make test-flint-stream`, model-free), `bench_flint_attention.m` + `flint_attention_variants.metal` and `bench_flint_gemm.m` (`make bench-flint-kernels`: model-free kernel timings with GPU timestamps) |
| `.github/workflows/flint.yml` | build, warning check, model-free tests (kernels, streaming module) |
| `docs/RECURSIVE_FLINT.md`, this file | |

Upstream files get only **contact points**: one- to three-line hooks that
include or dispatch into the files above. When new Flint code is needed, put
it in a fork file and add a hook. Never edit shared logic inline. Because
Flint is its own family, a hook is almost always a first-line
`if (ds4_model_is_flint()) return flint_...(...)`: no Flint code runs inside
another family's driver.

## Contact points

Check this list after every upstream merge (`git diff upstream/main -- <file>`).
If a hook's surrounding function was refactored upstream, re-place the hook.
Don't fork the refactored code. The plan's Design section lists the sites
each later step adds; this table tracks what is in the tree.

| File | Hook |
|---|---|
| `Makefile` | `include fork.mk` after the platform `CORE_OBJS` lists |
| `ds4.c` | `DS4_MODEL_FAMILY_FLINT = 200` and `DS4_VARIANT_FLINT = 202`. Fork families and variants start at 200 so they never collide with upstream additions or share a KV-cache model id (200 and 201 are other fork branches' variants). |
| | `#include "ds4_flint_shape.inc"` after `ds4_qwen4_layer_is_nextn` |
| | `#include "ds4_flint_model.inc"` before `config_validate_model`, plus its `flint` arch dispatch |
| | `ds4_engine_open_internal`: `flint_engine_supported` gate after `config_validate_model` |
| | `gguf_types[143..145]` and `DS4_TENSOR_FLINT_U4/U8/PACK` |
| | `#include "ds4_flint_weights.inc"` before `weights_have_output_head`; first-line redirects in `weights_have_output_head`, `weights_have_partial_output_head`, `weights_layer_has_required`, `weights_validate_layout`, `weights_bind_output`, `weights_bind_layer` |
| | `#include "ds4_flint.inc"` before `ds4_engine_open_internal`; a three-line Flint block before the Qwen block of `ds4_engine_open_internal` (inspect returns, else `flint_engine_open_tail`); `flint_engine_close()` after `qwen4_state_pool_free` in `ds4_engine_close` |
| | `#include "ds4_flint_tokenizer.inc"` before `bpe_tokenize_text`, plus its first-line dispatch; `flint_vocab_setup` branch at the top of the family section of `vocab_load`; first-line redirect of `special_token_at` to `flint_special_token_at` (HF matches every added token literally, including the reserved ones); first-line Flint branches in `encode_chat_prompt`, `ds4_chat_append_message`, `ds4_chat_append_assistant_prefix`, `chat_push_think_prefix` (no effort prefix) and `vocab_token_is_generation_stop` |
| | sessions: `void *flint` in `struct ds4_session` with three static forward declarations after it; `#include "ds4_flint_session.inc"` right after `ds4_flint.inc`; `ds4_session_create` (CPU refusal, and `return flint_session_create(...)` after the graph-path allocation), `ds4_session_free` (first branch of the GPU chain), `ds4_session_sync_internal` (after the distributed branch), `ds4_session_eval_internal` (after `e` is taken), `ds4_session_eval_argmax`, `ds4_session_payload_bytes`, `ds4_session_save_payload`, `ds4_session_load_payload` (before the V4.1 branch), `ds4_session_rewind` (`state_ok` from `flint_session_rewind`), `ds4_sessions_eval_batch_metal_supported` (false), `ds4_session_eval_speculative_argmax_impl` (plain eval, with the CPU branch), `ds4_engine_generate_argmax` (sessions path) |
| `ds4.h` | `ds4_engine_is_flint()`, `ds4_flint_trace_check()`, `ds4_flint_prefill_trace()`, `ds4_flint_generate_check()` |
| `ds4.c` (misc) | `ds4_engine_routed_quant_bits` returns 4 for Flint (the agent's KV save checks it); `ds4_chat_append_assistant_prefix` calls `flint_chat_append_assistant_prefix` |
| `ds4_metal.m` | include of `ds4_metal_internal.h`, fork sources appended to `required_sources`, trampoline block at the end |
| `ds4_server.c` | model id `flint` in `server_model_id_from_engine`; a Flint branch in `send_models` (`flint`, `flint-chat`, `flint-reasoner`); a four-line block before `model_alias_disables_thinking` (`g_flint_server` and three forward declarations) with first-line redirects in `model_alias_disables_thinking`, `model_alias_enables_thinking`, `server_model_alias_known`; `server_model_syntax_for_engine` (Flint sets `g_flint_server` and reports `SERVER_MODEL_SYNTAX_GLM`); `#include "ds4_flint_server.inc"` before `render_chat_prompt_text_for_syntax` with first-line redirects there and in `render_live_tool_tail_for_syntax`; one line after the thinking sampling defaults in the generation loop (`flint_server_sampling_defaults`); first-line redirects in `should_remember_thinking_checkpoint` and `build_thinking_visible_text` and a `g_flint_server ? remember_flint_tool_turn_visible_checkpoint(...) :` prefix on the `remember_qwen_tool_turn_visible_checkpoint` call after generation (live KV reuse when clients drop the reasoning); `parse_glm_generated_message_ex` takes a `tool_schema_orders` parameter (NULL from the syntax dispatch) and chooses the value type through `flint_glm_arg_is_string` when `g_flint_server`, with a `g_flint_server ?` branch in the post-generation parse that passes the request's schemas; `#include "ds4_flint_server_test.inc"` before `test_model_alias_thinking_controls` and `test_flint_render_prompt()` in the test runner |
| `ds4_agent.c` | `agent_tool_syntax_for_engine` (Flint uses `AGENT_TOOL_SYNTAX_GLM`), `agent_apply_model_sampling_defaults` (Flint like GLM) |
| `ds4_cli.c` | `--flint-trace`, `--flint-prefill-trace`, `--flint-generate` `IN OUT` (config fields, parse cases, dispatch lines before the imatrix branch) |
| `tests/ds4_test.c` | the long-context story's token-count threshold (29000 for Flint's tokenizer, 30000 otherwise) |

## Coupling to upstream code

Flint reuses these upstream pieces unchanged. An upstream change to any of
them can break Flint silently: it still compiles, but produces wrong tokens
or wrong streaming behaviour.

- The GGUF reader and tensor types (`gguf_types`, `tensor_data`, the mmap).
- The session and checkpoint containers (`struct ds4_session`, the
  checkpoint token vector, `ds4_session_rewind` semantics, the KV disk store
  payload framing in `ds4_kvstore.c`).
- `ds4_ssd.c` budgets and locks, the Metal expert cache entries
  (`ds4_gpu_stream_expert_cache_*`), the `pread` task runner and
  `ds4_gpu_buffer_address`.
- The tokenizer core (BPE merges, special-token matching) under
  `bpe_tokenize_text_flint`.
- The trampolines in `ds4_metal_internal.h`: `ds4_metal_matmul_f16_buffer`
  copies the n_tok > 8 branch of `ds4_gpu_matmul_f16_tensor_impl`; if
  upstream changes that branch, update the copy.

The model-backed checks in `docs/RECURSIVE_FLINT.md` (trace comparison
against rec, golden check, streamed-equals-resident) catch all of these, so
run them after every merge once they exist.

## Upstream sync procedure

```sh
git fetch upstream
git switch recursive-flint
git merge upstream/main          # merge, never rebase: rerere and history stay usable
make
make ds4_test ds4_agent_test && ./ds4_test --server && ./ds4_agent_test
# model-backed checks per docs/RECURSIVE_FLINT.md once steps 1 and 2 land
git push origin recursive-flint
```

`rerere.enabled` is set in this clone. Recorded conflict resolutions in
contact points are replayed on later merges. Set it in new clones with
`git config rerere.enabled true`.

## Release checks

1. Build with no new warnings (CI enforces this), `./ds4_test --server`,
   `./ds4_agent_test`.
2. The verification gates of `docs/RECURSIVE_FLINT.md`: trace comparison
   against rec, the golden check, streamed output identical to resident.
3. Speed, back to back with rec (never both processes at once), on AC power
   in High Power Mode with no other model process on the GPU.
4. A server smoke test covering `/v1/models`, a chat with reasoning, a tool
   call, a follow-up turn (prefix cache hit), and a restart with
   `--kv-disk-dir` (disk cache hit).

## Regenerating the reference

The reference is rec (`~/.local/bin/rec`) running the INT4 checkpoint it
pulls. If Recursive publishes new weights or a new rec:

1. Pin the new checkpoint digest and shard SHA-256s in
   `flint_int4_to_gguf.py` and in `tests/flint/golden.json`.
2. Re-run `rec serve` greedily on the cases in `golden.json` and rewrite it.
3. Re-dump the traces and logits with `tests/flint/rec_logits.py` and re-run
   the trace comparison.
