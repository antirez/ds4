# Qwen terminal and internet tool smoke tests

Tested September 7, 2026 on M3 Ultra, 512 GiB, using the combined Q4_K-imatrix
Qwen model and external Q4_1 PLE. The prior server KV-continuation change does
not itself supply terminal or internet tools.

## Startup defect and fix

Before the fix, `ds4-agent --ple FILE ...` exited with status 2 and
`unknown option: --ple`. Help advertised the option, but the agent parser did
not assign `engine.ple_path`. Published external-PLE Qwen models therefore
could not be started through the supported sidecar option in the native agent.
The agent now forwards `--ple` to the existing engine loader, as CLI/server do.
A parser regression test checks model, PLE and vision arguments together.

## Observed results

| Path | Actual operation | Result |
| --- | --- | --- |
| Native agent, non-interactive | `bash`: working directory and write/read a marker file | Passed; file contents independently verified |
| Native agent, non-interactive | `bash`/`curl`: GitHub repository API | HTTP/2 200; saved response independently parsed |
| Native agent, interactive | `visit_page`: `https://example.com` | Returned Example Domain and its first sentence after Chrome startup approval |
| Native agent, interactive | `google_search`: `ivanfioravanti ds4-metal Qwen3.8` | Returned search content and a result title; no explicit destination URL for the first result |
| Server + local API client | Model requests `bash` for `pwd`, then `curl https://example.com` | Client executed real commands, returned stdout, and model reported the page title |
| Server continuation | Both tool-result turns, reasoning omitted from replay | Both reused live KV via the visible-prefix path |

The native terminal test created `terminal-proof.txt` containing
`DS4_TERMINAL_OK_73921`. Its HTTPS response identified
`ivanfioravanti/ds4-metal`, default branch `main`, and
`pushed_at=2026-09-06T22:47:11Z`. Files were checked separately from the model's
claims. The browser test used the actual native browser tools, not curl.

Native agent settings: Metal, 16K context, temperature 0, thinking disabled,
MTP disabled. The server used 8K context, a 1024-token prefill chunk, temperature
0, low reasoning effort and MTP disabled. All model processes ran serially and
were stopped after testing. `make ds4-agent ds4-server ds4_agent_test` and
`./ds4_agent_test` passed.

The native agent emitted malformed Qwen tool-call text before recovering in
both the terminal and browser tests. These runs establish working access and
successful recovery, not flawless tool generation. Browser startup cannot be
approved by a non-interactive agent; an already-running browser is a separate
case. Third-party clients still need their own tool executor and network access.
The reported X user's exact launch command and client remain unknown, so this
does not establish the cause of that user's failure.

Raw logs, scripts, HTTP evidence and browser transcript are saved locally in
the ignored `OUT/qwen38-real-tools` directory.
