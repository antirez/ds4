# ds4-agent hooks

`ds4-agent` can run optional shell commands immediately before model generation and after a response finishes.

```sh
./ds4-agent \
  --before-response-hook 'cat >> /tmp/ds4-before.jsonl' \
  --after-response-hook 'curl -fsS -X POST -H "Content-Type: application/json" --data-binary @- https://example.test/ds4-hook' \
  --hook-timeout 10
```

The hook receives one JSON object on standard input. Hooks are disabled by default.

## Events

- `before_response`: emitted after the user turn is added and immediately before generation.
- `after_response`: emitted after the assistant turn ends, including accumulated response text, token count, interruption state, and tool-call state.

## Failure behavior

Hook failures are reported on stderr but do not abort the conversation. Commands run as `/bin/sh -c COMMAND` in their own process group. The runner enforces the configured timeout, sends `SIGTERM`, then uses `SIGKILL` after a short grace period.

Treat hook commands as trusted local configuration. Read event data from stdin instead of inserting prompt text into the command string.
