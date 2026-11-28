# Inkling 2 Mini (loader, tokenizer, CPU oracle, Metal forward, sessions)

Thinking Machines "Inkling 2 Mini" (GGUF architecture `inkling2_loop`, as written by llama.cpp) is loaded
directly from llama.cpp's GGUF: same tensor names, metadata under `inkling2_loop.*`.

| file | what |
| --- | --- |
| `ds4_inkling2.inc` | config check, tensor binding, tokenizer, the CPU oracle (f32, bit-identical between one batch and token by token) and the `--first-token-test` driver |
| `ds4_inkling2_gpu.inc` | Metal graph: the same 48 layer visits as the oracle, one function for chunked prefill and one-token decoding |
| `ds4_inkling2_session.inc` | `ds4_session_*` for this family (create, sync, eval, rewind, free, logits helpers), greedy CLI loop, session self-test |
| `ds4_inkling2_metal.inc` | Metal wrappers (included at the end of `ds4_metal.m`) |
| `metal/inkling2.metal` | kernels (`kernel_ik_*`); compiled as a library of its own on first use by this family, so other families never read it |

`ds4.c` only has the family hooks (a few lines each, all inert unless `general.architecture == inkling2_loop`).
Supported: `--metal` generation (`--raw-prompt` only: the chat template is not merged), `--first-token-test` on
`--metal` or `--cpu`, `--inspect`. Several sessions per server (`--batched-session N`, up to 8 decoded in one GPU pass), disk KV
snapshots (`--kv-disk-dir`) and contexts of 307,200 tokens per session work (see "Several sessions, disk KV, 262,144 tokens" and "About 300,000 tokens per session" below). Refused with a message:
speculative/MTP decoding, vision, CUDA/ROCm, distributed/TP, SSD streaming.

## Files that load

| file | experts | everything else |
| --- | --- | --- |
| BF16 | BF16 | BF16 dense, F32 small tensors |
| NVFP4 (ggml-org) | layers 0-19 NVFP4 (+ per-expert `.scale`), 20-23 Q8_0 | Q8_0, F32 |
| MXFP4_MOE | MXFP4 | Q8_0, F32 |

The binder checks every tensor's shape and type and rejects tensors it does not know. NVFP4 (GGUF type 40,
64 elements in 36 bytes: four UE4M3 scales, then 32 nibble bytes; byte j of a 16-element sub-block holds
elements j and j+8) has a CPU path and a Metal kernel. The GPU supports F32, F16, BF16, Q8_0, MXFP4 and NVFP4 weights.

## Running the CPU oracle

The CPU path is slow and only meant for parity work. Always use a private lock file and at most 8 threads:

```sh
DS4_LOCK_FILE=/tmp/ds4-inkling.lock DS4_INKLING2_FT_TOKENS=371,151451,149879,151194,149884 \
DS4_INKLING2_FT_POS=4 DS4_INKLING2_FT_OUT=/tmp/top.jsonl \
./ds4 -m Inkling-2-Mini-BF16.gguf --cpu --first-token-test --raw-prompt -p x -t 8
```

| variable | meaning |
| --- | --- |
| `DS4_INKLING2_FT_TOKENS` | token ids (comma list or a file); default is the tokenized prompt |
| `DS4_INKLING2_FT_POS` | token indexes whose next-token logits are produced (default: last) |
| `DS4_INKLING2_FT_OUT` | JSON lines `{"pos":P,"top":[[id,logprob],...]}`; `DS4_INKLING2_FT_TOPK` (default 50) |
| `DS4_INKLING2_FT_LOGITS` | raw f32 logits, 200058 per position |
| `DS4_INKLING2_CHUNK` | tokens per forward call (default 512, at most 4096). One chunk covering the prompt is the batch path, 1 is token-by-token decoding from the cached state; the logits are bit-identical only with `DS4_INKLING2_EXACT=1` |
| `DS4_INKLING2_FT_VISITS` | per layer visit: sums of Q/K/V/R, router logits, `ffn_inp`, `ffn_moe_out`, `ffn_out`, `l_out`, chosen expert ids per token, and router decisions whose 8th/9th margin is under `DS4_INKLING2_FT_TIE` (default 1e-4). llama.cpp names tensors by cache slot (prelude 0-5, first pass 6-17, extra passes 24-47, postlude 18-23), so do ds4's `slot=` fields |
| `DS4_INKLING2_FT_FLIP` | `visit,token,...`: take the 9th expert instead of the 8th there (tests whether a near-tied router pick explains a difference) |
| `DS4_INKLING2_TOK_FILE` | tokenizer only: lines of `0` or `1` (parse special tokens) followed by a JSON string; prints one id array per line |

## State (what the Metal port has to mirror)

48 slots (one per layer visit). Per slot: K and V rows after conv, clip (K) and RMS norm (V), stored as IEEE
half like llama.cpp's cache (attention reads them back, so the rounding is part of the model); local slots
keep a ring of 1024 + chunk rows, the eight global slots (5, 11, 17, 23 and the loop copies 29, 35, 41, 47)
keep every position; plus the last three pre-conv inputs of the 6656 conv channels (f32). Block starts, loop
messages and router results are recomputed per token and never stored.

## Metal

```sh
cd /path/to/ds4   # Metal sources are read from ./metal at start
export DS4_LOCK_FILE=/tmp/ds4-inkling.lock
./ds4 -m Inkling-2-Mini-NVFP4.gguf --metal --raw-prompt -p 'The capital of France is' -n 64 --temp 0 --ctx 4096
```

Memory: the model file is mapped (NVFP4 file: 16.4 GiB). Per context: K/V half rows
(8 global slots x ctx rows, 40 local slots x (1024 + chunk) rows, 2 x 2 KiB per row) plus per-chunk scratch;
`ds4: memory:` and `ds4: inkling2 gpu:` print the estimate and the bytes actually allocated (they agree within 1 %).
Example: ctx 16384, chunk 512: about 0.9 GiB state + 0.4 GiB scratch. `DS4_INKLING2_CHUNK` (default 512, max 4096) sets
the tokens per forward call; it changes the ring size, not the result.

By default (since the speed work) dense matmuls use a matvec kernel for 1-8 tokens and a simdgroup-matrix kernel for more; each is fixed-order, but the two orders differ, so logits depend slightly on the chunk size (forced-pick difference 0.004-0.006 between chunk 1 and 512 on 1038 tokens; free-running router decisions then flip like they do against the CPU). With `DS4_INKLING2_EXACT=1` the original rule holds: the GPU keeps a fixed summation order per kernel (lanes take units l, l+32, ..., then `simd_sum`; attention keys in
blocks of 256 from the window start), so logits do not depend on the chunk size: chunk 512, 100 and 1 give bit-identical
logits (measured up to 9008 tokens, ring wrap included). They are not bit-identical to the CPU oracle: the
router's top-8 sometimes flips on a near-tie (0.05-0.14 % of decisions on long prompts), which is why parity is checked with
the router decisions replayed (below).

| variable | meaning |
| --- | --- |
| `DS4_INKLING2_PICKS_OUT=f` | write every router decision of the run, raw int32 `[48 slots][tokens][8]` (CPU and GPU) |
| `DS4_INKLING2_FT_PICKS=f` | take every router decision from such a file (CPU and GPU); weights still come from this run's probabilities |
| `DS4_INKLING2_FT_PICKLOG=f` | CPU only: list decisions where the forced picks differ from the run's own top 8 |
| `DS4_INKLING2_DECODE_TAIL=k` | first-token test: the last k tokens one at a time (prefill in chunks, then decoding) |
| `DS4_INKLING2_SESSION_TEST=1` | with `--metal --first-token-test` and `DS4_INKLING2_FT_TOKENS`: session API self-test (40 checks: a known-answer check first (a fixed 14-token prompt must give top-1 token 220 at log-prob -0.094 and token 197017 at about -3.7, the llama.cpp values, so a build with wrong kernels cannot pass), error paths, NaN logits, replay, prefix restore. Bit-identity checks are exact under DS4_INKLING2_EXACT=1; the default mode only checks same argmax and total-variation distance <= 0.75, because live decode and replay use different kernels and a router near-tie flip moves the distribution). `tests/inkling2_gpu/selftest.sh` runs it in both modes |
| `DS4_INKLING2_EXACT=1` | diagnostic mode, not for serving: use only the fixed-order kernels (decode matvec, prefill simdgroup matmul and split-key attention are off): logits independent of chunk size and bit-identical between replay and live decode; decode 31 t/s and prefill 157 t/s instead of 59 and 764. Unset, empty or blank, `0`, `false`, `off` and `no` (any case, surrounding blanks ignored) all mean the default kernels, any other value means EXACT (one shared parser, `ds4_inkling2_env.h`) |
| `DS4_INKLING2_ATTN_SCALAR=1` | A/B runs: the old attention kernels for prefill and one-token decode |
| `DS4_INKLING2_TEST_POISON_LOGITS` | test hook: the session self-test sets it to make the read-back logits NaN and checks that the error reaches the caller |
| `DS4_METAL_ENCODER_TIMELINE=f` | (existing) per-dispatch GPU times with kernel names |

Parity method: run the GPU, dump its picks, replay them in the CPU oracle, compare top-20 log-probabilities
(`tests/inkling2_gpu/par.py`; limits 0.01 on a 22-token prompt, 0.02 on 308, 0.06 on 1,038, 0.4 on 9,008).
Free-running logits cannot be compared closely beyond about 10 tokens (llama.cpp differs from itself by the same amount).

### Speed

M3 Ultra, NVFP4 file, no other GPU work running, `tests/inkling2_gpu/speed.py`; llama.cpp (e8d7128, `llama-server -ngl 99 -c 8192 -np 1`) measured in the same session on the same text:

| | decode t/s (400 greedy tokens) | prefill t/s (5,810 tokens) |
| --- | --- | --- |
| ds4 default (fast kernels) | 59.4 (59.0-59.7, 10 runs, one run disturbed at 41.2) | 764.4 (763.8-765.0, 10 runs) |
| llama.cpp | 34.8 (34.7-35.1, 5 runs) | 667.4 (660.5-667.5, 5 runs) |
| ds4 before the kernel work below | 29.0 | 216 |
| `DS4_INKLING2_EXACT=1` | 31.3 (2 runs) | 156.6 (1 run) |

What made the difference (all in fast mode only; EXACT keeps the old kernels):
- Decode is GPU bound (the CPU needs about 1 ms to encode a token, the GPU 17 ms), and a one-token visit used to be ~40 small dispatches of 5-25 us each. One-token paths now batch them: the eight picked NVFP4 experts in one dispatch (gate/up, then down fused with the weighted combine), the six Q8_0 attention projections in one dispatch, the three loop controllers and the three input norms batched, conv + tail update + residual in one kernel, a simdgroup-parallel depth mix: about 960 dispatches per token instead of 2,000.
- Kernels need many loads in flight per thread: single-pass rms, unrolled loops, and a 16-entry nibble table (a 256-entry byte table is slower because of threadgroup bank conflicts).
- Prefill: experts on simdgroup matrices with the tokens of a chunk grouped by expert (any expert type), and flash-style attention on simdgroup matrices (Q and the softmax weights are rounded to half for the matrix products, everything else f32). The K/V caches are zero-initialised because whole 8-key tiles are read.
- `DS4_INKLING2_PROFILE=1` prints CPU encode time versus GPU wait time; `DS4_METAL_ENCODER_TIMELINE=f` plus `tests/inkling2_gpu/tl_agg.py` gives per kernel GPU time; `DS4_INKLING2_ATTN_SCALAR=1` selects the old attention kernel for A/B runs.
- Accuracy (router decisions replayed in the CPU oracle, worst top-20 log-prob difference): decode path (chunk 1) count 0.00014, long_300 0.0007, long_1030 0.0055; prefill (chunk 512) count 0.0003, long_300 0.0008, long_1030 0.010, long_9000 0.043 (exact kernels: 0.0036; limits 0.01 / 0.02 / 0.06 / 0.4). Top-1 matches at every position. Without the replay, some router decisions differ from the CPU at near-ties (8th and 9th scores within about 1e-4).

## Reading disagreements with llama.cpp

Router picks use the top 8 of `sigmoid(logit) + bias`; the 8th and 9th scores are often within 1e-4. llama.cpp's
Metal matrix kernels (used above 8 tokens) round activations to bf16/half, which flips some of those picks;
on short prompts the picks match in every slot, on a 300-token prompt they do not. Compare
`ffn_moe_topk` sums first, and when they differ look at the margin before suspecting a bug.

## Tests

`make test-inkling2` (model free): row dequantisation of every weight type, the pre-tokenizer, the EXACT-switch parser,
the special-token splitter against the old algorithm, and (macOS) `tests/test_inkling2_metal.c`: router with NaN/inf logits,
K/V cache never storing NaN/inf, the split-key decode attention against the plain kernel, Metal start-up without
`metal/inkling2.metal`. Model-backed: `tests/inkling2_gpu/selftest.sh`, `par.py`, `gen_cmp.py`, `speed.py`, `depth.py` (decode speed by
context depth), `tests/inkling2_server/*.py` (server end to end against a running server).

## Chat (template, thinking effort, tools, server ids)

Chat is text rendering plus token parsing; nothing here needs the forward pass. The code is
`ds4_inkling2_chat.inc` (included by `ds4_server.c`: renderer, parser, stream tracker) and a few lines in `ds4.c`
for the CLI (`ds4_encode_chat_prompt`, `ds4_chat_append_message`, `ds4_inkling2_effort_text`). Tests are in
`ds4_inkling2_chat_test.inc` (run by `./ds4_test --server`) and `tests/inkling2_chat_tokens.py`.

### Prompt format

The official `chat_template.jinja` has no generation prompt. A prompt is a list of blocks
`<role token><kind token>text<|end_message|>`; the model starts its own turn with `<|message_model|>`, so a rendered
prompt ends with the last message's `<|end_message|>`. There is no BOS token.

| message | rendered as |
| --- | --- |
| tools | first of all: `<|message_system|>tool_declare<|content_xml|>[{"description":..,"name":..,"parameters":..,"type":"function"},..]<|end_message|>`, keys sorted, compact, no `\u` escaping (a missing description is `""`, missing parameters `{}`) |
| effort | `<|message_system|><|content_text|>Thinking effort level: N<|end_message|>` before the first non-system message (at the end if there are only system messages) |
| system / developer | `<|message_system|><|content_text|>text<|end_message|>`, where it stands |
| user | `<|message_user|><|content_text|>text<|end_message|>`; an array `content` becomes one block per text part |
| assistant | `<|message_model|><|content_thinking|>reasoning<|end_message|>` (only if `reasoning_content` is set), `<|message_model|><|content_text|>text<|end_message|>` (skipped when the text is empty and there are tool calls or reasoning), one `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{..}}<|end_message|>` per tool call (args keys sorted, values compact in their own order, floats as Python prints them), then `<|content_model_end_sampling|>` (id 200006, the EOS token) |
| tool | `<|message_tool|>NAME<|content_text|>result<|end_message|>`; NAME is the message's `name`, else the function name of the assistant tool call with that `tool_call_id`, else empty |

Special-token text inside any message is cut out as a real special token by the tokenizer, because llama.cpp
tokenizes the rendered prompt with special parsing too. A client can therefore forge block markers inside content;
that is the reference behaviour, and the vectors keep it (cases 016-019, 058, 073).

### Thinking effort

The model is always told an effort level and decides itself whether to write a `<|content_thinking|>` block.
`none` renders `0` and the model still thinks a little (vendor note and our llama.cpp checks), so there is no real
"thinking off". ds4 modes and OpenAI `reasoning_effort` map as llama.cpp's template does:

| request | ds4 think mode | prompt says |
| --- | --- | --- |
| `none` | NONE | `0` |
| `minimal` | level 10 | `0.1` |
| `low` | LOW | `0.2` |
| `medium` | MEDIUM | `0.7` |
| omitted, `null`, `high` | HIGH | `0.9` |
| `xhigh`, `max` | level 99, MAX | `0.99` |
| a level string `"1".."100"` (ds4 extension) | level | level/100, at most `0.99` |

A string is trimmed first; any other string is a 400 (llama.cpp: 500 `Unknown reasoning_effort`). A JSON number is
ignored, as in llama.cpp. `max` is not lowered on small contexts. `thinking:{type:disabled}`, `think:false`,
`enable_thinking:false` and the `-chat` model id select `none`; llama.cpp ignores those flags (its template only
reads `reasoning_effort`), so this is a ds4 choice.

### Server

* Model ids: `inkling-2-mini` (effort as requested, default `0.9`), `inkling-2-mini-chat` (effort `none`; it still
  thinks briefly), `inkling-2-mini-reasoner` (effort high unless the request names one). `/v1/models` lists the three.
* `/v1/chat/completions`, `/v1/messages` (Anthropic), `/v1/responses` and `/v1/completions` are supported, stream or
  not; see "API" at the end of this file. `ds4-agent` refuses to start with this family (it is written for the
  think-tag protocols).
* Stop: the EOS token 200006 only (same as llama.cpp). `<|end_message|>` closes a block, it does not stop generation.
* Generated text is parsed like `common/parsers/inkling.cpp`: blocks `<|message_model|>` + `<|content_thinking|>`
  (reasoning_content) or `<|content_text|>` (content) or `NAME<|content_invoke_tool_json|>{"name":..,"args":{..}}`
  (a tool call, named by the JSON, not by the prefix). Several blocks of one kind are concatenated. A turn that is
  cut off inside a block keeps the text so far; a cut-off tool call is dropped. Text that does not start with a block
  is returned as plain content. With `tool_choice: "none"` the tools are still declared and tool calls are dropped.
* Streaming sends `reasoning_content` and `content` deltas as they arrive (holding back half a marker or half a
  UTF-8 character); tool calls stream name first, then the argument text as it is generated ("API" below).
* The tool/thinking live-KV checkpoints of the other families are off for this family: each request re-renders the
  history. A follow-up reuses the live state as far as it equals what was sampled (a client that sends
  `reasoning_content` back gets the whole conversation). A client that drops `reasoning_content` is handled by the
  prefix restore below.
* Unknown `model` ids are accepted and echoed, as for the other families.
* Errors answer 400 with a message: empty `messages`, an unknown `reasoning_effort` string, tool-call arguments that are not a JSON
  object or are nested more than 64 levels, `tools` entries that are not objects.

### Prefix reuse when the client drops `reasoning_content`

The model's own turn is `<|message_model|><|content_thinking|>...<|end_message|><|message_model|><|content_text|>answer`, but the
official template renders a past assistant turn without the thinking block (`<|message_model|><|content_text|>answer`).
A follow-up that omits the reasoning therefore diverges from the live tokens one token after the end of the previous prompt. The GPU
state (conv tails and sliding-window rings) cannot be rolled back, so at the end of every prompt sync the session keeps a copy of
it (the 40 window slots' rings, about 250 MiB; the global slots' rows are never overwritten) with the prompt tokens and the last
logits. A later sync that is not an exact continuation of the live state but starts with the kept prompt restores the copy and
evaluates only the rest. Measured: a 5,909-token follow-up reused 5,838 tokens and took 5.1 s instead of 12.6 s cold, with the same
answer (also after a 2,500-token first answer, which wraps the rings). Limits: one kept state, the latest prompt; a prompt that
differs earlier (an edited or dropped older turn), or two conversations alternating, still replays from zero.

## Running it as a server next to another ds4 server

```sh
cd /path/to/ds4      # Metal sources are read from ./metal at start
DS4_LOCK_FILE=/tmp/ds4-inkling.lock ./ds4-server -m Inkling-2-Mini-NVFP4.gguf --metal \
    --host 127.0.0.1 --port 8001 --ctx 307200 --batched-session 4 \
    --kv-disk-dir /path/to/inkling-kv --kv-disk-space-mb 60000
```

**Suggested default: 4 sessions of 307,200 tokens** (the model card allows 1,048,576, which was not tried). The alternative for many
short conversations is `--ctx 65536 --batched-session 8` with `--kv-disk-space-mb 20000`. The trade-off: 4 x 307,200 gives each of 4
concurrent conversations room for about 300,000 tokens and costs 39 GB of process footprint (10 GB per session) plus the 16.4 GiB
mapped model, while 8 x 65,536 serves 8 conversations of at most 65,536 tokens in 23 GiB of state plus the model. A further concurrent
conversation waits for a free session in either setup. A disk KV file of a deep conversation is large (about 9.7 GB for 290,000
tokens), so give `--kv-disk-space-mb` room for a few of them; decode and prefill slow down with depth (see "About 300,000 tokens
per session").

* **Set `DS4_LOCK_FILE` to a path of its own when another ds4 server runs on the machine.** The default is `/tmp/ds4.lock`:
  an Inkling server on the default lock either refuses to start while the other server runs or, started first, keeps the other
  one from starting. The server prints a loud `WARNING ***` line at start when an Inkling server runs on the default lock.
* **`--kv-disk-dir`: a directory of its own for this server.** Every KV file carries the model id of the server that wrote it
  (Inkling 2 has id 64; the other families have theirs). A server evicts, replaces and counts only files of its own id. Files of
  another model in the same directory are left alone and not counted against `--kv-disk-space-mb`; the server logs a warning
  naming how many there are. That protects them from this server only: a server of an older build still counts and evicts
  everything in its directory, so never point two servers at one directory.
* Use a port of its own (8001 above).
* **Size the state to fit.** One session of 307,200 tokens costs 10.24 GiB (estimate; 10,270 MiB allocated; process footprint
  10 GB for one session, 39 GB for four, 42 GB after requests), so 8 sessions at `--ctx 307200` would need 98.4 GiB in all. The start
  check looks at what is free: it refuses when the sessions plus the mapped model would leave less than 40% of RAM free
  (`DS4_INKLING2_FREE_FLOOR_PCT`, default 40; the fixed budget `DS4_INKLING2_BUDGET_GB`, default half of RAM, still applies and
  the log prints both). On a 256 GB machine with 70% free it printed "98.37 GiB would leave 80.8 GiB free, under the 40% floor
  (102.4 GiB)" for 8 x 307,200; that size starts only when about 78% of RAM is free. 4 x 307,200 (57.4 GiB in all) started at 66% free.
  One session of 262,144 tokens costs 8.9 GiB, so 8 sessions at `--ctx 262144` need 87 GiB plus the model.
  `--ctx 65536 --batched-session 8` is 23 GiB of state plus the 16.4 GiB mapped model.
* **A memory cap tool may not see the model.** The 16.4 GiB NVFP4 file is mapped, not read: a process monitor that counts
  resident memory reports far less than the real footprint. With a cap tool that does count mapped memory, set it to the
  footprint of the configuration: about 48 GB for 8 x 65,536, about 64 GB for 4 x 307,200.
* **It shares the GPU with the other server.** While Inkling prefilled 9,177 tokens, a tiny request to another ds4 server on the
  same GPU took 0.9 s instead of 0.68 s and one of its decode steps took 63-67 ms instead of 22 ms (three probes, not a
  distribution); Inkling itself was not slowed. Long Inkling prefills during the other server's traffic are the thing to avoid.
* A launchd job needs the same environment variable; one job per server.

### launchd job (example)

Save as `~/Library/LaunchAgents/<label>.plist` (or a LaunchDaemon), edit the paths, then
`launchctl bootstrap gui/$(id -u) <file>`. Nothing in the repo installs it.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>example.inkling-server</string>
  <key>WorkingDirectory</key><string>/path/to/ds4</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/ds4/ds4-server</string>
    <string>-m</string><string>/path/to/Inkling-2-Mini-NVFP4.gguf</string>
    <string>--metal</string>
    <string>--host</string><string>127.0.0.1</string>
    <string>--port</string><string>8001</string>
    <string>--ctx</string><string>307200</string>
    <string>--batched-session</string><string>4</string>
    <string>--kv-disk-dir</string><string>/path/to/inkling-kv</string>
    <string>--kv-disk-space-mb</string><string>60000</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <!-- a lock of its own when another ds4 server uses the default /tmp/ds4.lock -->
    <key>DS4_LOCK_FILE</key><string>/tmp/ds4-inkling.lock</string>
    <!-- optional: DS4_INKLING2_FREE_FLOOR_PCT (default 40), DS4_INKLING2_BUDGET_GB, DS4_INKLING2_EXACT=1 -->
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>60</integer>
  <key>StandardOutPath</key><string>/path/to/logs/inkling-server.log</string>
  <key>StandardErrorPath</key><string>/path/to/logs/inkling-server.log</string>
</dict>
</plist>
```

Notes: keep `KeepAlive` with a `ThrottleInterval`: the start check exits with code 1 when memory is short, and launchd
would otherwise restart it every 10 s. If the job starts at boot together with other GPU jobs, the start check sees what they
have already taken. With a memory cap wrapper, put the wrapper in `ProgramArguments` before `ds4-server` and give it 64 GB for this
configuration (see the memory-cap note above).

### Limits

* Without `--batched-session` there is one session: requests queue, and two alternating conversations never reuse the cache (one kept prompt state). With `--batched-session N` every slot keeps its own cache.
* No images or audio.
* Context per session: up to 307,200 tokens tested (see "About 300,000 tokens per session"); one session of 1,048,576 (the model card)
  would need about 33 GiB of state and was not tried.
* Tested context depth: server behaviour to 16,384 tokens (`--ctx 16384`) and the 290k-token needle test below; decode speed measured to about
  59,000 tokens of context in the table (and 65,000 and 81,920 in separate speed-only runs, 290,000 in the needle run). Parity against the CPU oracle:
  up to 9,008 tokens (29,393 for a deeper check).
* Decode slows with depth, see the table below.

### Decode speed by context depth

`tests/inkling2_gpu/depth.py` (CLI, 200 greedy tokens after a prompt of the given length, NVFP4 file, M3 Ultra, no other GPU work running).
One run per cell; llama.cpp figures are from separate runs (same machine and file, client clock, servers alternating), measured at similar
but not identical depths (57 / 6.7k / 13.4k / 32.2k / 65.1k tokens), not re-measured here.

| context (tokens) | ds4 before split-key attention | ds4 now | llama.cpp |
| --- | --- | --- | --- |
| 8 | 60.1 | 61.3 | 34.6 (57 tokens) |
| 6,819 | 44.9 | 55.9 | 33.4 (6.7k) |
| 12,442 | 39.8 | 54.8 | 32.9 (13.4k) |
| 29,393 | 30.0 | 49.9 | 31.1 (32.2k) |
| 59,037 | 20.6 | 43.3 | 29.2 (65.1k) |

Where the slowdown came from: one-token attention ran one threadgroup per head (24 in all), each walking every cached key serially
with three barriers per 256 keys, so a global layer visit cost 370 us at 27k keys of context and, below that, the sliding-window layers (up to
1,024 keys, 36 visits per token) cost 80 us each. Decode attention now splits the keys of one head over many threadgroups
(`kernel_ik_attn_split`: 1,024 keys per split on global slots, 128 on window slots, above 128 keys) and merges the partial
softmaxes (`kernel_ik_attn_reduce`). What is left grows with about 0.1 ms per 1,000 tokens of context, which is about the
memory bandwidth for reading K and V of the eight global slots (measured about 340 GB/s); further gains would need the three query heads that share a K/V head to read it once.
Prefill is unchanged (764 t/s at 5.8k tokens, 600 t/s at 59k).

### Differences from llama.cpp that remain

* A tool schema without the `{type:function,function:{..}}` wrapper (llama.cpp: 500) is declared as given; a request
  that ends on an assistant message with tool calls (llama.cpp: 400) is rendered as is.
* Integers above 2^63 in tool-call arguments are printed as written (llama.cpp wraps them to int64).
* `-0` in arguments is printed `-0.0` (nlohmann reads it as a float); not checked against llama.cpp.
* Images and audio are refused (no vision tower yet).

### Tests and vectors

`tests/test-vectors/inkling2-chat/` holds 77 cases captured from llama.cpp b12802 (e8d7128) serving
`Inkling-2-Mini-NVFP4.gguf`: request, rendered prompt (`/apply-template`), token ids (`/tokenize`, special parsing),
and for 34 tool cases the raw generation with special tokens (`/v1/chat/completions` with `verbose`, `return_tokens`,
temperature 0, so the parsed message belongs to exactly that text) with llama.cpp's parsed message.
`tests/inkling2_chat_fixtures.py` rebuilds the files from a capture directory.

```sh
make ds4_test ds4 && ./ds4_test --server                       # model free: prompt text, parser, streaming, ids
python3 tests/inkling2_chat_tokens.py --model Inkling-2-Mini-NVFP4.gguf   # ds4 token ids vs llama.cpp, needs a GGUF
```

## Several sessions, disk KV, 262,144 tokens

**Sessions.** `ds4-server ... --batched-session N` (N up to 8) creates N independent sessions: each has its own 48-slot K/V, conv
tails and scratch (about 1.4 GiB at ctx 16384); the weights are mapped once. Decode runs one GPU pass for all sessions that are ready
(`ikg_forward_multi`): the dense matvecs, router and the NVFP4 expert kernels see an n-token batch (token index on the grid z for the
expert kernels, matvec kernels with the token count fixed at compile time for 2..8 tokens); K/V store, attention and the conv tails run
per session on that session's own buffers. `DS4_INKLING2_NO_BATCH=1` turns the batched pass off (sessions then take turns; `=0` or empty leave it on).
In `DS4_INKLING2_EXACT=1` mode a batched step is bit-identical to the same sessions decoded one at a time; in default mode the worst
top-5 log-probability difference measured is 0.0001 (self-test: 5 sessions, 30 to 1,300 tokens, ring wrapped, 12 steps).

**Default mode and timing.** In default mode a session's logits do not depend on which neighbours share the step, their order or
their number (2 to 8; bit-identical in the tests), but a step that runs alone uses the one-session kernels and a step that runs
batched uses the batched kernels, and the two round differently (top-20 log-probability difference about 0.0003). Whether a given
decode step ran alone or in a batch depends on what other requests are in flight, so the same request at temperature 0 can sample a
different token where two candidates are nearly tied, and the text then diverges (measured: 2 of 8 answers differed over 200-700
tokens between "alone" and "8 at once"). With `DS4_INKLING2_EXACT=1` the answer depends on the request only, at about half the
batched speed (8 streams: 51-55 s instead of 26-33 s for the same eight requests).
A non-finite logit in one session of a batched step fails only that session (its request ends with an error and the session is
rebuilt by the next request); the other sessions of the step go on.
The server must start a batched prefill at the kept-state frontier (hunk in `server_session_sync`), otherwise its 2,048-token slices
replay a cached prompt from zero.

**Disk KV.** The payload (`iks_save_payload`) is the 13-word header (family tag, --ctx, model-file fingerprint, token count, layout),
the token ids, the f32 logits and, per slot, the conv tails and the live K/V rows (global slots 0..n-1; window slots the last 1,024
positions in position order, so the file does not depend on the chunk size), closed by a 64-bit checksum. A file written with another
--ctx, another model file or another family, a truncated file and a damaged file are refused with a message and leave the session
usable. Size: 205 MiB for 1,303 tokens, 292 MiB for 4,096, 326 MiB for 5,182; save about 50 ms, load about 105 ms for 4,096 tokens.

**Memory at start.** `ik_check_memory_budget` prints the estimate and refuses to start when it does not fit `DS4_INKLING2_BUDGET_GB`
(default: half of the RAM): ctx 262,144 needs 8.87 GiB per session (8 full-context slots = 8 x ctx x K and V = 8.00 GiB, window slots +
snapshot copy 0.48 GiB, scratch 0.39 GiB); 8 sessions 70.95 GiB, plus the 16.4 GiB mapped model.

**Tests.** `tests/inkling2_gpu/selftest.sh` (91 checks, default and EXACT: payload, batch, ctx boundary, cancelled and failed
prefills), `tests/inkling2_server/multi.py` (concurrent equals alone), `kv.py` (restart reuse), `cancel_check.py`, `soak.py`,
`leak.py` (no answer from another conversation after a cancelled or failed prefill), `kvguard.py` and `stagecheck.py` (KV directory
owner guard, staging inside the KV directory), `tests/inkling2_gpu/deep_par.py` (deep CPU-oracle parity), `depth.py` (deep decode table).

## Claude Code against this server

Works for text and tool sessions through `/v1/messages`. What the server does for it:
* An `image`, `document`, `search_result` or `tool_reference` block inside a `tool_result` (Claude Code reading a PNG or PDF, tool search)
  becomes a short text placeholder (`[image omitted: this model cannot read it]`, `[tool loaded: NAME]`) instead of a 400 that would
  stay in the client's history and fail every later turn. An image sent directly in a user message is still refused: there is no vision.
* A prompt over `--ctx` answers `400 prompt is too long: N tokens > M maximum`, the wording Claude Code looks for to compact the
  conversation. Claude Code assumes a large window for an unknown model id: set `CLAUDE_CODE_MAX_CONTEXT_TOKENS` (and the auto-compact
  window) to the `--ctx` of the server so it compacts before the limit.
* `POST /v1/messages/count_tokens` answers `{"input_tokens": N}` (Inkling servers only; other families answer 404 as before).
* Request parsing and prompt rendering are linear in the number of content blocks (20,000 blocks: 0.02 s, was 3 s).

## API: Anthropic Messages, OpenAI Responses, raw completions, logprobs, streamed tool calls

All four endpoints work for this family, streaming or not, and send the same prompt as the chat completions
request for the same conversation (`tests/test-vectors/inkling2-api`: 23 conversations, each as a chat, an Anthropic
and a Responses request; all three render to one prompt, checked model-free in `./ds4_test --server`).
The server keeps no tool or reasoning state between requests for this family: every request is replayed from what
the client sends (the prefix reuse of the previous section still applies).

### `/v1/messages` (Anthropic)

* `system` (string or text blocks) becomes the first system block. `messages` take string or block content. `text`,
  `thinking`, `tool_use` and `tool_result` blocks work; a `thinking` block replays as the model's reasoning block.
  A user message that carries `tool_result` blocks becomes one tool message per result (named after the call that
  has the same `tool_use_id`), followed by the rest of its text as a user message, exactly what the same
  conversation sent to chat completions renders. Tools use `input_schema`; they are declared even for
  `tool_choice: none` (calls are then dropped), like llama.cpp. Forcing a tool (`tool_choice` any/tool) is not
  enforced.

* Thinking control. The template has no thinking switch, only `Thinking effort level: N`. Mapping:
  `thinking.type disabled` = none (0); `thinking.budget_tokens` below 2048 = low (0.2), below 8192 = medium (0.7), below
  32000 = high (0.9), 32000 and over = max (0.99); `enabled` without a budget = high; `output_config.effort` or
  `reasoning_effort` (names as for chat completions) wins over the budget. `max` is not lowered on small contexts.
  llama.cpp ignores `thinking` and answers with 0.9 whatever the request says; the other 40 of 42 prompts in
  the comparison below are identical, the two that differ are exactly this (a budget and a disabled request).

* Answer: `thinking` block (its `signature` is the message id; nothing checks it), `text`, `tool_use` blocks in that order;
  `stop_reason` end_turn / tool_use / max_tokens; `usage` with input and output tokens.

* Stream: `message_start`, then per block `content_block_start`, `thinking_delta` / `text_delta` / `input_json_delta`
  events (a thinking block ends with a `signature_delta`), `content_block_stop`, then `message_delta` and `message_stop`. A tool_use block
  opens with its name as soon as the model has written it, and `partial_json` fragments are the arguments as generated.

### `/v1/responses`

* `input` as a string or items: `message` (user, assistant, system/developer), `reasoning` (merged into the next assistant
  message as its reasoning block), `function_call` (merged into the assistant message before it) and
  `function_call_output`. `instructions` is the system prompt. `reasoning.effort` as `reasoning_effort` does;
  `reasoning.summary` opts in to the reasoning item, as for the other families.

* `previous_response_id` and `conversation` are refused (400), as for the Qwen family: there is no stored response to continue.
  Replay the input items.

* Stream: `response.created`, `output_item.added` / deltas / `.done` for the reasoning item (summary text), the message
  item (`output_text.delta`) and each function call, then `response.completed` (or `.incomplete` for max tokens), every event with a
  `sequence_number`. As for the Qwen family the function_call items come when the turn ends.

* llama.cpp refuses a `reasoning` input item that has no `content` array (HTTP 400 for 4 of the 23 requests); this server accepts `summary` text.

### `/v1/completions`

The prompt is not put into the chat template and no effort block is added: the text goes to the tokenizer as written,
with special tokens parsed (so `<|message_user|>...` can be typed by hand), and the answer is the generated text as the
model wrote it, block markers included. `prompt` may be a string, a list with one string, or token ids (`[1,2,3]` or `[[1,2,3]]`, used as given); a list of several prompts, `echo`, `suffix`, `n` and `best_of` other than the defaults get a 400 that names them (one choice per request); `stream`,
`max_tokens`, `stop`, sampling parameters and `logprobs` work. (The Qwen family wraps the prompt in its chat template here;
llama.cpp does not.) No effort is applied: with an empty history the model continues the text.

### Logprobs

`logprobs: true` with `top_logprobs: 0..20` on chat completions, `logprobs: 0..20` on completions (legacy `tokens`,
`token_logprobs`, `top_logprobs`, `text_offset` lists; llama.cpp answers this endpoint in the chat `content` shape instead). Values are the
model's log-softmax over the whole vocabulary at the position of the sampled token, before temperature, top-k and top-p.
Every sampled token is listed, in order, including reasoning tokens and the block marker tokens (llama.cpp does the same);
the token that ends the turn is not. `token` is the token text (invalid UTF-8 shows as U+FFFD, `bytes` has the real
bytes); entries are capped at -9999. In a stream each token adds one chunk with an empty delta and that token's entry.
`top_logprobs` over 20 is a 400; an explicit `logprobs: false` wins over `top_logprobs`. Against llama.cpp on the same file:
completions, 8 tokens, same tokens, largest difference 0.0020; chat, 12 tokens, largest 0.043 (NVFP4 on both).
Other families still ignore these fields.

### Tool calls in chat streams

The model writes `NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}`. The stream sends the tool call's id and name as soon as
`{"name":"...","args":{` has been generated, then the text of the `args` object in pieces as it appears (the
concatenation is byte for byte the arguments of the non-streaming answer, for every vector in
`tests/test-vectors/inkling2-chat` and every way of cutting the stream, checked in `ds4_test --server`). A call that does not have this
shape (other key order, `args` not an object) is sent whole when the turn ends, like before. A call cut off by `max_tokens` after its
name went out stays partial on the wire (the non-streaming answer drops it; the finish reason is `length`). Anthropic streams the same way
(`tool_use` block, `input_json_delta`); Responses sends function_call items at the end.

### Two items kept as they are

* A user message without a `content` key is accepted as empty text. llama.cpp answers 400 ("All non-assistant messages must contain
  'content'"; `content: null` is accepted there). The shared parser of the Qwen family accepts it. Not changed: the two references differ.

* Special-token text inside message content (`<|message_model|>` typed by a user) is tokenized as the control token. llama.cpp does
  the same (it renders the template and tokenizes with `parse_special`: 20 tokens against 73 literal for the probe in the tests), as does the
  Qwen family. The same text gives 20 prompt tokens here. A deployment that must stop this has to filter requests before this server.

### Checks

```sh
./ds4_test --server                                   # model free: 23 conversations x 3 APIs, budget table, stream splits, fuzz (2 s)
DS4_INKLING2_FUZZ_SECS=240 ./ds4_test --server        # longer fuzzing (also built with -fsanitize=address,undefined)
python3 tests/inkling2_server/api_equiv.py --port 8001   # live server: the three APIs, stream and body, against each other
python3 tests/inkling2_server/logprobs.py --port 8001 --ref llama_lp.json
python3 tests/inkling2_server/sdk_roundtrip.py --port 8001   # anthropic and openai SDK tool round trips (needs both SDKs)
python3 tests/inkling2_api_tokens.py capture --log LLAMA.log --out refs && python3 tests/inkling2_api_tokens.py compare --refs refs --model MODEL.gguf
```

## About 300,000 tokens per session

`--ctx 307200`. A code read found no cap at or below 262,144: the K/V and tensor offsets in the Metal kernels are 64-bit (`ulong`) products,
the row counts are 32-bit but stay under 2^20, the position factor tau(pos) = 1 + 0.1 ln(max(1, pos/8192)) has no ceiling, and the disk KV
payload sizes are 64-bit (a 290,000-token file is 9.7 GB, past 4 GiB). Nothing had to change.

**Memory** (M3 Ultra, 256 GB, next to another busy server). One session at 307,200 allocates 10,270 MiB of tensors (estimate 10.24 GiB) and
the process footprint is 10 GB; four sessions allocate 4 x 10,270 MiB, footprint 39 GB right after start and 42 GB after a few requests;
the mapped model (16.4 GiB) comes on top and is not in either number. Free memory went from 66% to 59% at the start of the four
sessions. Eight sessions are refused by the start check unless about 78% of RAM is free (see "Size the state to fit").

**Boundary.** A prompt of exactly `--ctx` tokens and one of `--ctx` + 1 are refused with HTTP 400 (`context_length_exceeded`, no prefill);
a prompt of ctx - 1 = 307,199 tokens is accepted and one token can be generated (position reaches 307,200). Prefill of the 307,199-token
prompt: 1,023 s (300 tokens/s on average).

**Needle recall at depth** (289,748 prompt tokens of Python standard-library documentation, five notes "the access code for vault NAME
is CODE" at 10, 30, 50, 70 and 90% of the text, then a question for the five codes; greedy, chat endpoint, `reasoning_effort none`,
NVFP4 file, one session). Both ds4 and llama.cpp (e8d7128, `llama-server -ngl 99 -np 1`, same file and prompt) answered all five codes
correctly (`Alpha: TANGERINE-4821`, `Bravo: LANTERN-7305`, `Charlie: MARBLE-1194`, `Delta: COBALT-6672`, `Echo: SADDLE-9058`). The first 10
generated tokens are the same; the top-20 log-probabilities of the chosen tokens differ by less than 0.08 (a single run each, not bit-identical).

| 289,748-token prompt | ds4 | llama.cpp |
| --- | --- | --- |
| prefill | 959 s (302 tokens/s; 600 t/s in the first 10,000 tokens, 250 at 220,000) | 3,081 s (94 tokens/s; 330 t/s in the first 25,000, 100 at 270,000) |
| decode at 290,000 tokens of context | 20.3 tokens/s (441 tokens) | 17.0 tokens/s (391 tokens) |

A second ds4 request with the same prompt reused the kept state (289,748 cached tokens) and answered in 21.8 s. The model thinks aloud
first on this prompt and writes the list after about 330 tokens, so the first generated tokens are the thinking header.
