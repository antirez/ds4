#!/usr/bin/env python3
"""Fetch small DeepSeek V4 Flash logprob vectors from the official API.

The API exposes top-logprobs, not full logits.  These vectors are therefore
golden continuation slices: useful for catching tokenizer/template/attention
regressions, but not a replacement for a full internal logit dump.
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


MODEL = "deepseek-v4-flash"
ENDPOINT = "https://api.deepseek.com/chat/completions"
TOP_LOGPROBS = 20
MAX_TOKENS = 4
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


def request_vector(api_key: str, prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as fp:
            return json.loads(fp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"DeepSeek API HTTP {e.code}: {body}") from e


def normalize_record(prompt_spec: dict, response: dict, checkpoint: str) -> dict:
    choice = response["choices"][0]
    logprob_items = choice.get("logprobs", {}).get("content", []) or []
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

    return {
        "schema": "ds4-official-logprobs-v1",
        "source": "deepseek-official-api",
        "model": MODEL,
        "checkpoint": checkpoint,
        "endpoint": ENDPOINT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "id": prompt_spec["id"],
        "kind": prompt_spec["kind"],
        "prompt": prompt_spec["prompt"],
        "request": {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": prompt_spec["prompt"]}],
        },
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
        f"# checkpoint {manifest['checkpoint']}",
        "# case <id> <ctx> <steps> <prompt-file>",
        "# step <index> <selected-hex> <top-count>",
        "# top <token-hex> <official-logprob>",
        "",
    ]
    for prompt in manifest["prompts"]:
        vector_id = prompt["id"]
        record = json.loads((root / prompt["official_file"]).read_text(encoding="utf-8"))
        steps = record["steps"]
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
            lines.append(f"step {i} {hex_bytes(step['token']['bytes'])} {len(top)}")
            for token_hex, lp in top:
                lines.append(f"top {token_hex} {lp:.9g}")
        lines.append("end")
        lines.append("")
    (root / "official.vec").write_text("\n".join(lines), encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="checkpoint label used in the fixture directory, for example 0731",
    )
    parser.add_argument(
        "--out",
        help="output directory (default: tests/test-vectors/flash-CHECKPOINT)",
    )
    parser.add_argument("--only", action="append", help="fetch only the named prompt id")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY is required", file=sys.stderr)
        return 2

    root = Path(args.out or f"tests/test-vectors/flash-{args.checkpoint}")
    prompt_dir = root / "prompts"
    official_dir = root / "official"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    official_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.only or [])
    manifest = {
        "schema": "ds4-test-vector-manifest-v1",
        "source": "deepseek-official-api",
        "model": MODEL,
        "checkpoint": args.checkpoint,
        "endpoint": ENDPOINT,
        "top_logprobs": TOP_LOGPROBS,
        "max_tokens": MAX_TOKENS,
        "prompts": [],
    }

    for spec in PROMPTS:
        if wanted and spec["id"] not in wanted:
            continue
        prompt_path = prompt_dir / f"{spec['id']}.txt"
        prompt_path.write_text(spec["prompt"], encoding="utf-8")

        response = request_vector(api_key, spec["prompt"])
        record = normalize_record(spec, response, args.checkpoint)
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
        print(f"wrote {out_path}")

    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not wanted:
        write_compact_fixture(root, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
