#!/usr/bin/env python3
"""Run fixed vision facts against an OpenAI-compatible endpoint."""

import base64
import json
import mimetypes
import os
import pathlib
import sys
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parent / "vision-fixtures" / "glm53"
ENDPOINT = os.environ.get(
    "DS4_VISION_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("DS4_VISION_MODEL", "z-ai/glm-5.3-flash")
API_KEY = os.environ.get("OPENROUTER_API_KEY")
OFFICIAL_ZAI = os.environ.get("DS4_VISION_OFFICIAL_ZAI") == "1"
SUITE = os.environ.get("DS4_VISION_SUITE", "glm53")
REQUIRED_PROVIDER = os.environ.get("DS4_VISION_REQUIRE_PROVIDER")
REASONING_EFFORT = os.environ.get("DS4_VISION_REASONING_EFFORT", "none")


def contains(answer, requirement):
    if isinstance(requirement, str):
        return requirement in answer
    return any(choice in answer for choice in requirement)


def run_case(case):
    image_path = ROOT / case["image"]
    mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 320,
        "reasoning_effort": REASONING_EFFORT,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": case["prompt"]},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{encoded}"}},
            ],
        }],
    }
    if OFFICIAL_ZAI:
        payload["provider"] = {"order": ["Z.AI"], "allow_fallbacks": False}
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST")
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.load(response)
    answer = result["choices"][0]["message"]["content"]
    provider = result.get("provider")
    normalized = answer.lower()
    missing = [item for item in case["required"] if not contains(normalized, item)]
    forbidden = [item for item in case.get("forbidden", []) if item in normalized]
    if REQUIRED_PROVIDER and provider != REQUIRED_PROVIDER:
        missing.append(f"provider:{REQUIRED_PROVIDER}")
    return answer, missing, forbidden, provider


def main():
    cases = json.loads((ROOT / "cases.json").read_text())
    failures = 0
    for case in cases:
        answer, missing, forbidden, provider = run_case(case)
        passed = not missing and not forbidden
        failures += not passed
        print(json.dumps({
            "name": case["name"],
            "pass": passed,
            "missing": missing,
            "forbidden": forbidden,
            "provider": provider,
            "answer": answer,
        }, ensure_ascii=False))
    print(f"{SUITE} vision quality: {len(cases) - failures}/{len(cases)} passed",
          file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
