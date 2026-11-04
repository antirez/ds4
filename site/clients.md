# Clients

## Pi

In `.pirc.json`:

```json
{
  "provider": "dwarfstar",
  "dwarfstar": {
    "baseUrl": "http://127.0.0.1:8000",
    "model": "ds4flash"
  }
}
```

## OpenCode

```sh
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_MODEL_NAME=ds4flash
```

## Codex CLI

In `~/.codex/config.json`:

```json
{
  "provider": "openai-compatible",
  "baseUrl": "http://127.0.0.1:8000/v1",
  "model": "ds4flash"
}
```

## Claude Code

Use an OpenAI-compatible proxy configured to `http://127.0.0.1:8000/v1`.

## Any Client

Any OpenAI-compatible client can connect to `http://127.0.0.1:8000/v1`. Supports streaming, tool calls, and vision (with encoder loaded).