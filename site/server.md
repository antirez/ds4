# Serving

## Start Locally

```sh
./ds4-server --ctx 32768
```

Default address: `http://127.0.0.1:8000`. Use `--host 0.0.0.0` for other interfaces.

## API Endpoints

### Chat Completions

`POST /v1/chat/completions` — OpenAI-compatible:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ds4flash",
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

Supports streaming with `"stream": true`.

### Other Endpoints

| Endpoint | Description |
| --- | --- |
| `POST /v1/completions` | Legacy completions |
| `GET /v1/models` | List loaded models |
| `GET /health` | Server status |

## Multiple Sessions

```sh
./ds4-server --ctx 100000 --batched-session 16
```

Batching starts at 5 ready sessions. Uses `X-Session-Id` header or cookie.

## KV Cache

```sh
./ds4-server --kv-cache-dir ~/.ds4/kvcache
```

## CORS

```sh
./ds4-server --cors
```