#!/usr/bin/env python3
"""Fetch GLM 5.2 OpenRouter continuation/top-logprob vectors.

OpenRouter exposes output-token logprobs and top-logprob slices, not full
vocabulary logits. These fixtures are therefore external continuation checks
plus compact top-logprob spot checks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


MODEL = "z-ai/glm-5.2"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
TOP_LOGPROBS = 20
MAX_COMPLETION_TOKENS = 16
PROVIDER_ORDER = ["parasail/fp8"]
CTX_BY_ID = {
    "short_italian_fact": 16384,
    "short_code_completion": 4096,
    "short_reasoning_plain": 4096,
    "long_memory_archive": 16384,
    "long_code_audit": 16384,
}


def long_memory_prompt() -> str:
    block = (
        "Record {i:03d}: the archive entry says that component alpha keeps a "
        "compressed index, component beta keeps raw observations, and component "
        "gamma reports anomalies only after the checksum phrase appears. "
        "Do not summarize yet; retain the exact final question.\n"
    )
    body = "".join(block.format(i=i) for i in range(72))
    return (
        "You are checking a long technical archive. Read the repeated records "
        "and answer only the final question with one short sentence.\n\n"
        + body
        + "\nFinal question: which component reports anomalies after the checksum phrase appears?"
    )


def long_code_prompt() -> str:
    stanza = (
        "Function f_{i} validates a queue entry, calls normalize_path(), then "
        "appends a compact audit line. The invariant is that strlen() must not "
        "be recomputed when a trusted length returned by snprintf() is already "
        "available. Security note {i}: reject negative sizes before casting.\n"
    )
    body = "".join(stanza.format(i=i) for i in range(68))
    return (
        "Review this generated C-code audit log. After the log, complete the "
        "sentence with the most likely next words.\n\n"
        + body
        + "\nCompletion target: The most important code quality issue is"
    )


PROMPTS = [
    {
        "id": "short_italian_fact",
        "kind": "short",
        "prompt": "Rispondi in italiano con una frase: chi era Ada Lovelace?",
    },
    {
        "id": "short_code_completion",
        "kind": "short",
        "prompt": "Complete the C statement with the next exact token only:\nreturn snprintf(buf, sizeof(buf), \"%d\", value",
    },
    {
        "id": "short_reasoning_plain",
        "kind": "short",
        "prompt": "Answer with only the number: 2048 divided by 128 is",
    },
    {
        "id": "long_memory_archive",
        "kind": "long",
        "prompt": long_memory_prompt(),
    },
    {
        "id": "long_code_audit",
        "kind": "long",
        "prompt": long_code_prompt(),
    },
]


def token_bytes(token: str, value) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    return list(token.encode("utf-8"))


def request_vector(
    api_key: str,
    prompt: str,
    model: str,
    endpoint: str,
    max_completion_tokens: int,
    top_logprobs: int,
    reasoning_effort: str,
    token_limit_field: str,
    provider_order: list[str],
    provider_allow_fallbacks: bool,
    provider_require_parameters: bool,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 1,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "stream": False,
    }
    payload[token_limit_field] = max_completion_tokens
    if reasoning_effort != "omit":
        payload["reasoning"] = {"effort": reasoning_effort}
    if provider_order or provider_require_parameters:
        provider = {"require_parameters": provider_require_parameters}
        if provider_order:
            provider["order"] = provider_order
            provider["allow_fallbacks"] = provider_allow_fallbacks
        payload["provider"] = provider
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "DwarfStar GLM vector checks",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as fp:
        return json.loads(fp.read().decode("utf-8"))


def fetch_vector_with_retry(
    api_key: str,
    prompt: str,
    model: str,
    endpoint: str,
    max_completion_tokens: int,
    top_logprobs: int,
    reasoning_effort: str,
    token_limit_field: str,
    provider_order: list[str],
    provider_allow_fallbacks: bool,
    provider_require_parameters: bool,
) -> dict:
    delay = 1.0
    for attempt in range(6):
        try:
            return request_vector(
                api_key,
                prompt,
                model,
                endpoint,
                max_completion_tokens,
                top_logprobs,
                reasoning_effort,
                token_limit_field,
                provider_order,
                provider_allow_fallbacks,
                provider_require_parameters,
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            last = RuntimeError(f"OpenRouter HTTP {e.code}: {body}")
            if e.code < 500 and e.code != 429:
                raise last from e
        except Exception as e:  # noqa: BLE001 - command-line retry wrapper.
            last = e
        if attempt == 5:
            raise last
        time.sleep(delay)
        delay *= 1.7
    raise AssertionError("unreachable")


def normalize_record(args: argparse.Namespace, prompt_spec: dict, response: dict) -> dict:
    choice = response["choices"][0]
    logprob_items = (choice.get("logprobs") or {}).get("content", []) or []
    if not logprob_items:
        raise RuntimeError(f"{prompt_spec['id']}: response did not include output-token logprobs")
    steps = []
    for step, item in enumerate(logprob_items):
        top = []
        for alt in item.get("top_logprobs", []) or []:
            tok = alt.get("token", "")
            top.append(
                {
                    "token": {
                        "text": tok,
                        "bytes": token_bytes(tok, alt.get("bytes")),
                    },
                    "logprob": alt.get("logprob"),
                }
            )
        tok = item.get("token", "")
        steps.append(
            {
                "step": step,
                "token": {
                    "text": tok,
                    "bytes": token_bytes(tok, item.get("bytes")),
                },
                "logprob": item.get("logprob"),
                "top_logprobs": top,
            }
        )

    request = {
        "model": args.model,
        "temperature": 0,
        "seed": 1,
        "token_limit_field": args.token_limit_field,
        args.token_limit_field: args.max_completion_tokens,
        "logprobs": True,
        "top_logprobs": args.top_logprobs,
        "messages": [{"role": "user", "content": prompt_spec["prompt"]}],
    }
    if args.reasoning_effort != "omit":
        request["reasoning"] = {"effort": args.reasoning_effort}
    if args.provider_order or args.require_parameters:
        provider = {"require_parameters": args.require_parameters}
        if args.provider_order:
            provider["order"] = args.provider_order
            provider["allow_fallbacks"] = args.allow_provider_fallbacks
        request["provider"] = provider

    return {
        "schema": "ds4-openrouter-logprobs-v1",
        "source": "openrouter",
        "model": args.model,
        "endpoint": args.endpoint,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "id": prompt_spec["id"],
        "kind": prompt_spec["kind"],
        "prompt": prompt_spec["prompt"],
        "request": request,
        "provider": response.get("provider"),
        "usage": response.get("usage"),
        "finish_reason": choice.get("finish_reason"),
        "message": choice.get("message", {}),
        "logits_available": False,
        "steps": steps,
    }


def hex_bytes(values: list[int]) -> str:
    return "".join(f"{int(x):02x}" for x in values)


def write_compact_fixture(root: Path, manifest: dict) -> None:
    lines = [
        "# ds4-official-logprob-vectors-v1",
        "# source openrouter z-ai/glm-5.2",
        "# case <id> <ctx> <steps> <prompt-file>",
        "# step <index> <selected-hex> <top-count>",
        "# top <token-hex> <official-logprob>",
        "",
    ]
    if not manifest["prompts"]:
        raise RuntimeError("no OpenRouter vectors were fetched")
    for prompt in manifest["prompts"]:
        vector_id = prompt["id"]
        record = json.loads((root / prompt["official_file"]).read_text(encoding="utf-8"))
        steps = record["steps"][:16]
        if not steps:
            raise RuntimeError(f"{vector_id}: record has no logprob steps")
        prompt_file = root / prompt["prompt_file"]
        lines.append(f"case {vector_id} {CTX_BY_ID[vector_id]} {len(steps)} {prompt_file}")
        for i, step in enumerate(steps):
            top = []
            for alt in step.get("top_logprobs", []):
                lp = float(alt.get("logprob", -9999))
                if lp <= -1000:
                    continue
                token_hex = hex_bytes(alt["token"]["bytes"])
                if token_hex:
                    top.append((token_hex, lp))
            selected_hex = hex_bytes(step["token"]["bytes"])
            if not selected_hex:
                raise RuntimeError(f"{vector_id}: step {i} has an empty selected-token byte string")
            lines.append(f"step {i} {selected_hex} {len(top)}")
            for token_hex, lp in top:
                lines.append(f"top {token_hex} {lp:.9g}")
        lines.append("end")
        lines.append("")
    (root / "official.vec").write_text("\n".join(lines), encoding="ascii")


def write_quality_manifest(root: Path, manifest: dict) -> None:
    lines = ["# id\tprompt_file\tcontinuation_file\tresponse_file"]
    cont_dir = root / "continuations"
    cont_dir.mkdir(parents=True, exist_ok=True)
    if not manifest["prompts"]:
        raise RuntimeError("no OpenRouter vectors were fetched")
    for prompt in manifest["prompts"]:
        record_path = root / prompt["official_file"]
        record = json.loads(record_path.read_text(encoding="utf-8"))
        content = record.get("message", {}).get("content", "")
        cont_path = cont_dir / f"{prompt['id']}.txt"
        cont_path.write_text(content, encoding="utf-8")
        lines.append("\t".join([
            prompt["id"],
            str(root / prompt["prompt_file"]),
            str(cont_path),
            str(record_path),
        ]))
    (root / "manifest.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tests/test-vectors/glm-openrouter", help="output directory")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--max-completion-tokens", type=int, default=MAX_COMPLETION_TOKENS)
    parser.add_argument("--top-logprobs", type=int, default=TOP_LOGPROBS)
    parser.add_argument("--reasoning-effort",
                        choices=("xhigh", "high", "medium", "low", "minimal", "none", "omit"),
                        default="none")
    parser.add_argument("--token-limit-field",
                        choices=("max_tokens", "max_completion_tokens"),
                        default="max_tokens")
    parser.add_argument("--provider-order", default=",".join(PROVIDER_ORDER),
                        help="comma-separated OpenRouter provider slugs")
    parser.add_argument("--allow-provider-fallbacks", action="store_true")
    parser.add_argument("--no-require-parameters", dest="require_parameters",
                        action="store_false",
                        help="allow routing to providers that do not advertise all requested parameters")
    parser.set_defaults(require_parameters=True)
    parser.add_argument("--only", action="append", help="fetch only the named prompt id")
    args = parser.parse_args()

    if args.top_logprobs < 0 or args.top_logprobs > 20:
        raise SystemExit("--top-logprobs must be between 0 and 20")
    if args.max_completion_tokens <= 0:
        raise SystemExit("--max-completion-tokens must be positive")
    args.provider_order = [
        item.strip() for item in args.provider_order.split(",") if item.strip()
    ] if args.provider_order else []

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 2

    root = Path(args.out)
    prompt_dir = root / "prompts"
    official_dir = root / "official"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    official_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.only or [])
    manifest = {
        "schema": "ds4-test-vector-manifest-v1",
        "source": "openrouter",
        "model": args.model,
        "endpoint": args.endpoint,
        "top_logprobs": args.top_logprobs,
        "max_completion_tokens": args.max_completion_tokens,
        "token_limit_field": args.token_limit_field,
        "reasoning_effort": args.reasoning_effort,
        "provider_order": args.provider_order,
        "allow_provider_fallbacks": args.allow_provider_fallbacks,
        "require_parameters": args.require_parameters,
        "prompts": [],
    }

    for spec in PROMPTS:
        if wanted and spec["id"] not in wanted:
            continue
        prompt_path = prompt_dir / f"{spec['id']}.txt"
        prompt_path.write_text(spec["prompt"], encoding="utf-8")

        response = fetch_vector_with_retry(
            api_key,
            spec["prompt"],
            args.model,
            args.endpoint,
            args.max_completion_tokens,
            args.top_logprobs,
            args.reasoning_effort,
            args.token_limit_field,
            args.provider_order,
            args.allow_provider_fallbacks,
            args.require_parameters,
        )
        record = normalize_record(args, spec, response)
        out_path = official_dir / f"{spec['id']}.official.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["prompts"].append(
            {
                "id": spec["id"],
                "kind": spec["kind"],
                "prompt_file": str(prompt_path.relative_to(root)),
                "official_file": str(out_path.relative_to(root)),
                "prompt_chars": len(spec["prompt"]),
                "steps": len(record["steps"]),
            }
        )
        print(f"wrote {out_path} steps={len(record['steps'])}", file=sys.stderr)

    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_quality_manifest(root, manifest)
    if not wanted:
        write_compact_fixture(root, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
