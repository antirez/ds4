#!/usr/bin/env python3
"""
DS4 DeepSeek V4 Flash logprob vector collector (production-grade).

Improvements:
- thread-safe manifest aggregation
- structured logging
- exponential backoff with jitter
- atomic file writes
- stricter response validation
- safer concurrency model
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
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict


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


# -------------------------
# Logging (lightweight)
# -------------------------
def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr)


# -------------------------
# Utilities
# -------------------------
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_write(path: Path, content: str, encoding="utf-8") -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding=encoding)
    tmp.replace(path)


def save_json_gz(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp.gz")
    with gzip.open(tmp, "wt", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False)
    tmp.replace(path)


# -------------------------
# API layer
# -------------------------
def build_payload(prompt: str) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "stream": False,
    }


def request_vector(api_key: str, prompt: str) -> dict:
    payload = build_payload(prompt)

    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode())
                if "choices" not in data:
                    raise ValueError("Invalid API response: missing choices")
                return data

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            if attempt == RETRY_LIMIT:
                raise RuntimeError(f"HTTP {e.code}: {body}")

        except Exception as e:
            if attempt == RETRY_LIMIT:
                raise RuntimeError(f"Request failed: {e}")

        # exponential backoff + jitter
        sleep = (2 ** attempt) + random.random()
        time.sleep(sleep)

    raise RuntimeError("Unreachable retry state")


# -------------------------
# Normalization layer
# -------------------------
def token_bytes(token: str, value: Any) -> list[int]:
    return list(value) if isinstance(value, list) else list(token.encode())


def normalize_record(prompt_spec: dict, response: dict) -> dict:
    choice = response["choices"][0]
    items = choice.get("logprobs", {}).get("content", []) or []

    steps = []
    for idx, item in enumerate(items):
        steps.append({
            "step": idx,
            "token": {
                "text": item.get("token", ""),
                "bytes": token_bytes(item.get("token", ""), item.get("bytes")),
            },
            "logprob": item.get("logprob"),
            "top_logprobs": [
                {
                    "token": {
                        "text": t.get("token", ""),
                        "bytes": token_bytes(t.get("token", ""), t.get("bytes")),
                    },
                    "logprob": t.get("logprob"),
                }
                for t in (item.get("top_logprobs") or [])
            ],
        })

    return {
        "schema": "ds4-official-logprobs-v2",
        "model": MODEL,
        "id": prompt_spec["id"],
        "prompt": prompt_spec["prompt"],
        "prompt_hash": sha256_text(prompt_spec["prompt"]),
        "steps": steps,
        "usage": response.get("usage"),
    }


def validate_record(record: dict) -> None:
    if not record.get("steps"):
        raise ValueError("Empty steps")
    if "prompt_hash" not in record:
        raise ValueError("Missing prompt_hash")


# -------------------------
# Core worker
# -------------------------
def fetch_and_store(api_key: str, spec: dict, root: Path) -> dict:
    prompt_dir = root / "prompts"
    out_dir = root / "official"

    prompt_path = prompt_dir / f"{spec['id']}.txt"
    prompt_path.write_text(spec["prompt"])

    response = request_vector(api_key, spec["prompt"])
    record = normalize_record(spec, response)
    validate_record(record)

    json_path = out_dir / f"{spec['id']}.json"
    gz_path = out_dir / f"{spec['id']}.json.gz"

    atomic_write(json_path, json.dumps(record, indent=2))
    save_json_gz(gz_path, record)

    log(f"OK {spec['id']} steps={len(record['steps'])}")

    return {
        "id": spec["id"],
        "prompt_file": str(prompt_path),
        "official_file": str(json_path),
        "steps": len(record["steps"]),
    }


# -------------------------
# Main
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/test-vectors")
    ap.add_argument("--only", action="append")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        log("Missing DEEPSEEK_API_KEY")
        return 2

    root = Path(args.out)
    (root / "prompts").mkdir(parents=True, exist_ok=True)
    (root / "official").mkdir(parents=True, exist_ok=True)

    selected = [
        p for p in PROMPTS
        if not args.only or p["id"] in args.only
    ]

    manifest = {
        "model": MODEL,
        "generated_at": time.time(),
        "prompts": [],
    }

    lock = concurrent.futures.thread.Lock() if hasattr(concurrent.futures.thread, "Lock") else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(fetch_and_store, api_key, p, root)
            for p in selected
        ]

        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if lock:
                with lock:
                    manifest["prompts"].append(res)
            else:
                manifest["prompts"].append(res)

    atomic_write(root / "manifest.json", json.dumps(manifest, indent=2))

    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
