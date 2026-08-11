"""Message-selection rules for chat history compression.

Mirrors planCompressSlices / keepRecentForPass in chat-ui/static/app.js so
tests can assert shrink behavior without a browser or model call.
"""

from __future__ import annotations

import math
from typing import Any

KEEP_RECENT_MESSAGES = 6
FORCE_KEEP_RECENT_MESSAGES = 2
SUMMARY_PREFIX = "[conversation summary]"


def estimate_tokens(text: str | None) -> int:
    """Match app.js estimateTokens: ceil(chars / 4)."""
    return math.ceil(len(text or "") / 4)


def history_token_estimate(messages: list[dict[str, Any]] | None) -> int:
    if not messages:
        return 0
    total = 0
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else ""
        if not isinstance(content, str):
            content = ""
        total += estimate_tokens(content)
    return total


def keep_recent_for_pass(force: bool, pass_number: int) -> int:
    base_keep = FORCE_KEEP_RECENT_MESSAGES if force else KEEP_RECENT_MESSAGES
    return max(0, base_keep - (max(1, pass_number) - 1) * 2)


def plan_compress_slices(
    messages: list[dict[str, Any]] | None,
    keep_recent: int,
    *,
    allow_leftover: bool = True,
    summarize_budget: int | None = None,
) -> dict[str, Any] | None:
    """Return older/recent/chunk/leftover slices, or None if nothing to compress."""
    base = list(messages or [])
    if len(base) < 2:
        return None
    keep = max(0, min(int(keep_recent), len(base) - 1))
    older = base[: len(base) - keep]
    recent = base[len(base) - keep :]
    if not older:
        return None

    chunk = list(older)
    leftover: list[dict[str, Any]] = []
    if allow_leftover and summarize_budget is not None and summarize_budget > 0:
        while len(chunk) > 2 and history_token_estimate(chunk) > summarize_budget:
            half = max(1, math.ceil(len(chunk) / 2))
            leftover = chunk[half:] + leftover
            chunk = chunk[:half]

    return {
        "older": older,
        "recent": recent,
        "chunk": chunk,
        "leftover": leftover,
        "keep": keep,
    }


def apply_summary_message(
    plan: dict[str, Any],
    summary_text: str,
) -> list[dict[str, Any]]:
    """Build the post-compress message list for a planned slice + summary body."""
    return [
        {
            "role": "system",
            "content": f"{SUMMARY_PREFIX}\n{summary_text}",
            "compressed": True,
        },
        *plan["leftover"],
        *plan["recent"],
    ]


def simulate_force_compress(
    messages: list[dict[str, Any]],
    *,
    summary_chars: int = 800,
    max_passes: int = 4,
    hard_token_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Apply force-compress selection with a fixed-size fake summary (no LLM).

    Matches app.js: stop after the first successful shrink unless still at/over
    hard_token_limit (then later passes reduce keep toward 0).
    """
    msgs = list(messages)
    summary_text = "x" * max(0, summary_chars)
    for pass_number in range(1, max_passes + 1):
        keep = keep_recent_for_pass(True, pass_number)
        if len(msgs) <= keep or len(msgs) < 2:
            break
        plan = plan_compress_slices(msgs, keep, allow_leftover=False)
        if plan is None:
            break
        before = history_token_estimate(msgs)
        next_msgs = apply_summary_message(plan, summary_text)
        after = history_token_estimate(next_msgs)
        if after >= before:
            continue
        msgs = next_msgs
        if hard_token_limit is None or after < hard_token_limit:
            break
    return msgs
