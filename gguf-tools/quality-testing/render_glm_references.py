#!/usr/bin/env python3
"""Render GLM 5.3 API fixtures, retaining reasoning before the scored answer."""

import argparse
import csv
import json
from pathlib import Path


def render(prompt, response, effort):
    if response.get("model") != "z-ai/glm-5.3-flash":
        raise ValueError("expected GLM 5.3 Flash response")
    if effort not in ("low", "high", "max"):
        raise ValueError("expected low, high or max reasoning effort")
    message = response["choices"][0]["message"]
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    if not isinstance(reasoning, str):
        raise ValueError("reasoning must be text")
    tokens = response.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    if tokens and not reasoning:
        raise ValueError("provider did not return the reasoning text")
    if not message.get("content"):
        raise ValueError("provider did not return a final answer")
    # Official chat_template.jinja: no separators around roles or think markers.
    return ("[gMASK]<sop><|system|>Reasoning Effort: " + effort.capitalize()
            + "<|user|>" + prompt + "<|assistant|><think>" + reasoning + "</think>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.fixture / "collection.json").read_text())
    directory = args.fixture / "rendered-prompts"
    directory.mkdir(exist_ok=True)
    rows = []
    with (args.fixture / "manifest.tsv").open() as source:
        for row in csv.reader(source, delimiter="\t"):
            if not row or row[0].startswith("#"):
                continue
            case, prompt, continuation, response = row
            data = json.loads(Path(response).read_text())
            if Path(continuation).read_text() != data["choices"][0]["message"]["content"]:
                raise ValueError(f"{case}: continuation differs from saved API response")
            path = directory / (case + ".txt")
            path.write_text(render(Path(prompt).read_text(), data, metadata["reasoning_effort"]))
            rows.append((case, str(path), continuation, response))
    with (args.fixture / "manifest-rendered.tsv").open("w") as dest:
        dest.write("# Use score_official --rendered-prompt; reasoning is part of the prefix.\n")
        csv.writer(dest, delimiter="\t", lineterminator="\n").writerows(rows)


if __name__ == "__main__":
    main()
