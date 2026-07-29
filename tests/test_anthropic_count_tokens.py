#!/usr/bin/env python3
"""Check that Anthropic count_tokens matches /v1/messages input usage."""

import argparse
import json
import sys
import urllib.error
import urllib.request


def post_json(base_url, path, payload, timeout):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("%s returned HTTP %d: %s" %
                           (path, exc.code, detail)) from exc


def require_nonnegative_int(value, field):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError("%s must be a non-negative integer" % field)
    return value


def message_input_tokens(response):
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError("/v1/messages response is missing usage")
    return sum(
        require_nonnegative_int(usage.get(field, 0), "usage.%s" % field)
        for field in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare /v1/messages/count_tokens with /v1/messages usage"
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    payload = {
        "model": args.model,
        "system": "You are a concise assistant.",
        "messages": [
            {
                "role": "user",
                "content": "Explain token counting briefly.",
            }
        ],
        "tools": [
            {
                "name": "lookup",
                "description": "Look something up",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            }
        ],
        "max_tokens": 1,
        "temperature": 0,
    }

    count_response = post_json(
        args.url, "/v1/messages/count_tokens", payload, args.timeout
    )
    count = require_nonnegative_int(
        count_response.get("input_tokens"), "count_tokens.input_tokens"
    )

    message_response = post_json(
        args.url, "/v1/messages", payload, args.timeout
    )
    message_count = message_input_tokens(message_response)
    if count != message_count:
        print(
            "FAIL model=%s count_tokens=%d messages_input_tokens=%d" %
            (args.model, count, message_count),
            file=sys.stderr,
        )
        return 1

    print(
        "PASS model=%s input_tokens=%d" % (args.model, count)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
