# Coding Agent Clients

[README](../README.md) | [Server](SERVER.md)

Start the server first:

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Set the client's context limit no higher than the server's. Output tokens also
consume that context; a client limit does not enlarge the server allocation.
The examples use Flash and localhost. Change the address for a trusted remote
server. The `dsv4-local` strings are placeholders, not server authentication.

## Pi

Add this provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "ds4": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "dsv4-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "deepseek",
        "requiresReasoningContentOnAssistantMessages": true
      },
      "models": [{
        "id": "deepseek-v4-flash",
        "name": "DwarfStar Flash",
        "reasoning": true,
        "thinkingLevelMap": {
          "off": null, "minimal": "low", "low": "low",
          "medium": "medium", "high": "high", "xhigh": "xhigh"
        },
        "input": ["text"],
        "contextWindow": 100000,
        "maxTokens": 16384,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
      }]
    }
  }
}
```

Select `ds4/deepseek-v4-flash` in Pi. This is a text-only client configuration;
image use also requires the appropriate client input declaration and server
vision encoder.

## OpenCode

Merge the provider into `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ds4": {
      "name": "DwarfStar",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "dsv4-local"
      },
      "models": {
        "deepseek-v4-flash": {
          "name": "DwarfStar Flash",
          "limit": {"context": 100000, "output": 16384}
        }
      }
    }
  }
}
```

Select `ds4/deepseek-v4-flash` as the model.

## Codex CLI

Use the Responses API. Add a provider to the Codex configuration:

```toml
[model_providers.ds4]
name = "DwarfStar"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1000000
```

```sh
codex --model deepseek-v4-flash -c model_provider=ds4
```

## Claude Code

Use the Anthropic-compatible endpoint. A shell wrapper can select the local
model for the main agent and its secondary model roles:

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_AUTH_TOKEN="dsv4-local"
export ANTHROPIC_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000
exec claude "$@"
```

Give the wrapper a different name from `claude`. Agent clients can send large
initial prompts, so the first prefill may take time. Disk caching helps reuse
compatible prefixes on later sessions.
