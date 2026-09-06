#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tokenizers>=0.22",
# ]
# ///
"""Build deterministic, cache-distinct Qwen acceptance prompts.

Long cases are natural-text rotations of a supplied corpus and are cut by the
official tokenizer, not by bytes or words.  Every emitted long prompt is
round-trip checked to contain exactly its requested number of raw tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


LONG_CASES = {
    "prefill-10k+1": 10_000,
    "prefill-50k+1": 50_000,
    "prefill-100k+1": 100_000,
    "decode-8.5k-500": 8_500,
    "end-to-end-10k+500": 10_000,
    "cold-ple-10k+1": 10_000,
    "mtp-10k+500": 10_000,
}

SHORT_CASES = {
    "decode-code-500": (
        "Continue this implementation with production-quality Python. Include "
        "validation, error handling, tests, and explanatory comments. Do not "
        "stop until the module is complete.\n\n"
        "def merge_sorted_streams(streams):\n"
        "    # Merge lazily while preserving stable source order.\n"
    ),
    "decode-prose-500": (
        "Write a detailed, self-contained account of a coastal town preparing "
        "for an approaching winter storm. Continue for at least twelve "
        "paragraphs, with concrete sensory detail and no headings.\n"
    ),
}

WARMUP_CASE = "warmup-10k+1"
WARMUP_TOKENS = 10_000


def rotated_corpus(corpus: str, sample: int) -> str:
    if not corpus:
        raise ValueError("corpus is empty")
    pivot = (sample * 104_729) % len(corpus)
    return corpus[pivot:] + corpus[:pivot]


def stable_first_tokens(tokenizer: Tokenizer, count: int) -> list[tuple[int, str]]:
    """Find printable token surfaces that remain the first token before LF.

    Persistent servers reuse the live longest-common prefix independently of
    their optional prefix caches.  A different first token for every measured
    request therefore makes ``cached_tokens=0`` an input property instead of a
    server-reset convention.
    """
    result = []
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=False)
    for token_id in range(vocab_size):
        surface = tokenizer.decode([token_id], skip_special_tokens=False)
        if not surface or not surface.isprintable() or surface.isspace():
            continue
        encoded = tokenizer.encode(
            surface + "\n", add_special_tokens=False
        ).ids
        if encoded and encoded[0] == token_id:
            result.append((token_id, surface))
            if len(result) == count:
                return result
    raise ValueError(
        f"tokenizer exposes only {len(result)} stable printable first tokens, "
        f"need {count}"
    )


def exact_prompt(tokenizer: Tokenizer, corpus: str, target: int,
                 case: str, sample: int, first_token: tuple[int, str]
                 ) -> tuple[str, list[int]]:
    first_token_id, first_surface = first_token
    marker = (
        f"{first_surface}\nAcceptance case {case}, independent sample {sample}. "
        "Read the following source carefully and continue only when asked.\n\n"
    )
    source = marker + rotated_corpus(corpus, sample)
    source_ids = tokenizer.encode(source, add_special_tokens=False).ids
    if len(source_ids) < target + 32:
        repeats = (target + 32 + len(source_ids) - 1) // len(source_ids)
        source_ids = tokenizer.encode(
            marker + (rotated_corpus(corpus, sample) * repeats),
            add_special_tokens=False,
        ).ids
    candidate = tokenizer.decode(source_ids[:target], skip_special_tokens=False)
    ids = tokenizer.encode(candidate, add_special_tokens=False).ids
    if len(ids) != target:
        raise ValueError(
            f"{case} sample {sample}: tokenizer round trip produced "
            f"{len(ids)} tokens, expected {target}"
        )
    if not ids or ids[0] != first_token_id:
        raise ValueError(
            f"{case} sample {sample}: first token changed from "
            f"{first_token_id} to {ids[0] if ids else 'empty'}"
        )
    return candidate, ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 3 or args.samples % 2 == 0:
        parser.error("--samples must be an odd number of at least three")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    corpus = args.corpus.read_text()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    case_count = len(LONG_CASES) + len(SHORT_CASES)
    first_tokens = iter(stable_first_tokens(
        tokenizer, case_count * args.samples + 1
    ))

    warmup_first = next(first_tokens)
    warmup_text, warmup_ids = exact_prompt(
        tokenizer, corpus, WARMUP_TOKENS, WARMUP_CASE, 0, warmup_first
    )
    warmup_path = args.out / f"{WARMUP_CASE}-0.txt"
    warmup_path.write_text(warmup_text)
    records.append({
        "case": WARMUP_CASE,
        "sample": 0,
        "path": warmup_path.name,
        "tokens": len(warmup_ids),
        "first_token_id": warmup_ids[0],
        "sha256": hashlib.sha256(warmup_text.encode()).hexdigest(),
        "warmup": True,
    })

    for case, target in LONG_CASES.items():
        for sample in range(args.samples):
            first_token = next(first_tokens)
            text, ids = exact_prompt(
                tokenizer, corpus, target, case, sample, first_token
            )
            path = args.out / f"{case}-{sample}.txt"
            path.write_text(text)
            records.append({
                "case": case,
                "sample": sample,
                "path": path.name,
                "tokens": len(ids),
                "first_token_id": ids[0],
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            })

    for case, base in SHORT_CASES.items():
        for sample in range(args.samples):
            first_token_id, first_surface = next(first_tokens)
            text = (
                f"{first_surface}\nIndependent sample {sample}.\n{base}"
            )
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if not ids or ids[0] != first_token_id:
                raise ValueError(
                    f"{case} sample {sample}: first token changed from "
                    f"{first_token_id} to {ids[0] if ids else 'empty'}"
                )
            path = args.out / f"{case}-{sample}.txt"
            path.write_text(text)
            records.append({
                "case": case,
                "sample": sample,
                "path": path.name,
                "tokens": len(ids),
                "first_token_id": ids[0],
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            })

    manifest = args.out / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(records)} prompts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
