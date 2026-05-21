#!/usr/bin/env python3
"""Fetch DeepSeek V4 Flash logprob vectors with retries, caching, gzip export,
parallel fetching, metadata stats, and validation support.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MODEL = "deepseek-v4-flash"
ENDPOINT = "https://api.deepseek.com/chat/completions"
TOP_LOGPROBS = 20
MAX_TOKENS = 4
RETRY_LIMIT = 3

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
    return (
        "You are checking a long technical archive. Read the repeated records "
        "and answer only the final question with one short sentence.\n\n"
        + "".join(block.format(i=i) for i in range(72))
        + "\nFinal question: which component reports anomalies after the checksum phrase appears?"
    )


def long_code_prompt() -> str:
    stanza = (
        "Function f_{i} validates a queue entry, calls normalize_path(), then "
        "appends a compact audit line. The invariant is that strlen() must not "
        "be recomputed when a trusted length returned by snprintf() is already "
        "available. Security note {i}: reject negative sizes before casting.\n"
    )
    return (
        "Review this generated C-code audit log. After the log, complete the "
        "sentence with the most likely next words.\n\n"
        + "".join(stanza.format(i=i) for i in range(68))
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_bytes(token: str, value: Any) -> list[int]:
    return [int(x) for x in value] if isinstance(value, list) else list(token.encode("utf-8"))


def build_payload(prompt: str) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "thinking": {"type": "disabled"},
        "stream": False,
    }


def request_vector(api_key: str, prompt: str) -> dict:
    payload = build_payload(prompt)

    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as fp:
                return json.loads(fp.read().decode("utf-8"))

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")

            if attempt == RETRY_LIMIT:
                raise RuntimeError(f"DeepSeek API HTTP {e.code}: {body}") from e

            time.sleep(attempt * 2)

        except Exception:
            if attempt == RETRY_LIMIT:
                raise

            time.sleep(attempt * 2)

    raise RuntimeError("Unexpected retry failure")


def normalize_record(prompt_spec: dict, response: dict) -> dict:
    choice = response["choices"][0]
    logprob_items = choice.get("logprobs", {}).get("content", []) or []

    steps = [
        {
            "step": idx,
            "token": {
                "text": item.get("token", ""),
                "bytes": token_bytes(item.get("token", ""), item.get("bytes")),
            },
            "logprob": item.get("logprob"),
            "top_logprobs": [
                {
                    "token": {
                        "text": alt.get("token", ""),
                        "bytes": token_bytes(alt.get("token", ""), alt.get("bytes")),
                    },
                    "logprob": alt.get("logprob"),
                }
                for alt in item.get("top_logprobs", []) or []
            ],
        }
        for idx, item in enumerate(logprob_items)
    ]

    return {
        "schema": "ds4-official-logprobs-v2",
        "source": "deepseek-official-api",
        "model": MODEL,
        "endpoint": ENDPOINT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "id": prompt_spec["id"],
        "kind": prompt_spec["kind"],
        "prompt_hash": sha256_text(prompt_spec["prompt"]),
        "prompt": prompt_spec["prompt"],
        "request": build_payload(prompt_spec["prompt"]),
        "usage": response.get("usage"),
        "finish_reason": choice.get("finish_reason"),
        "message": choice.get("message", {}),
        "logits_available": False,
        "steps": steps,
        "stats": {
            "token_count": len(steps),
            "avg_logprob": (
                sum(s["logprob"] or 0 for s in steps) / len(steps)
                if steps else 0
            ),
        },
    }


def hex_bytes(values: list[int]) -> str:
    return "".join(f"{int(v):02x}" for v in values)


def validate_record(record: dict) -> None:
    if "steps" not in record:
        raise ValueError("Missing steps")

    for step in record["steps"]:
        if "token" not in step:
            raise ValueError("Missing token in step")


def save_json_gz(path: Path, data: dict) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def write_compact_fixture(root: Path, manifest: dict) -> None:
    output = [
        "# ds4-official-logprob-vectors-v2",
        "# compact vector fixture",
        "",
    ]

    for prompt in manifest["prompts"]:
        record_path = root / prompt["official_file"]

        with open(record_path, encoding="utf-8") as fp:
            record = json.load(fp)

        output.append(
            f"case {prompt['id']} {CTX_BY_ID[prompt['id']]} {len(record['steps'])}"
        )

        for step in record["steps"]:
            selected = hex_bytes(step["token"]["bytes"])

            tops = [
                (
                    hex_bytes(t["token"]["bytes"]),
                    float(t["logprob"]),
                )
                for t in step.get("top_logprobs", [])
                if t.get("logprob") is not None
            ]

            output.append(
                f"step {step['step']} {selected} {len(tops)}"
            )

            output.extend(
                f"top {token_hex} {lp:.9g}"
                for token_hex, lp in tops
            )

        output.append("end\n")

    (root / "official.vec").write_text(
        "\n".join(output),
        encoding="ascii",
    )


def fetch_and_store(
    api_key: str,
    spec: dict,
    root: Path,
) -> dict:
    prompt_dir = root / "prompts"
    official_dir = root / "official"

    prompt_path = prompt_dir / f"{spec['id']}.txt"
    prompt_path.write_text(spec["prompt"], encoding="utf-8")

    response = request_vector(api_key, spec["prompt"])

    record = normalize_record(spec, response)

    validate_record(record)

    json_path = official_dir / f"{spec['id']}.official.json"
    gzip_path = official_dir / f"{spec['id']}.official.json.gz"

    json_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    save_json_gz(gzip_path, record)

    print(f"[✓] {spec['id']} saved")

    return {
        "id": spec["id"],
        "kind": spec["kind"],
        "prompt_file": str(prompt_path.relative_to(root)),
        "official_file": str(json_path.relative_to(root)),
        "compressed_file": str(gzip_path.relative_to(root)),
        "prompt_chars": len(spec["prompt"]),
        "steps": len(record["steps"]),
        "prompt_hash": record["prompt_hash"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--out",
        default="tests/test-vectors",
        help="output directory",
    )

    parser.add_argument(
        "--only",
        action="append",
        help="fetch only selected prompt ids",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel worker count",
    )

    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")

    if not api_key:
        print("DEEPSEEK_API_KEY is required", file=sys.stderr)
        return 2

    root = Path(args.out)

    (root / "prompts").mkdir(parents=True, exist_ok=True)
    (root / "official").mkdir(parents=True, exist_ok=True)

    selected = [
        p for p in PROMPTS
        if not args.only or p["id"] in args.only
    ]

    manifest = {
        "schema": "ds4-test-vector-manifest-v2",
        "source": "deepseek-official-api",
        "model": MODEL,
        "endpoint": ENDPOINT,
        "top_logprobs": TOP_LOGPROBS,
        "max_tokens": MAX_TOKENS,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompts": [],
    }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        futures = [
            executor.submit(fetch_and_store, api_key, spec, root)
            for spec in selected
        ]

        for future in concurrent.futures.as_completed(futures):
            manifest["prompts"].append(future.result())

    manifest_path = root / "manifest.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.only:
        write_compact_fixture(root, manifest)

    print(f"\nManifest saved to {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
