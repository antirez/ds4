#!/usr/bin/env python3
"""Resolve a web-search query from the latest user ask plus recent turns.

Asks the same upstream chat API the UI proxies: given X and prior X-1..X-4,
what does the user really mean — that answer is the search query.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PRIOR_TURN_LIMIT = 4
PRIOR_CONTENT_CHARS = 500
REWRITE_TIMEOUT_S = 20
REWRITE_MAX_TOKENS = 48
DEFAULT_MODEL = "deepseek-v4-flash"

INTENT_REWRITE_SYSTEM = (
    "You turn a chat follow-up into a web search query. "
    "Given recent conversation and the latest user message, reply with ONLY a concise "
    "web search query that captures what they really want looked up. "
    "Resolve pronouns and references using prior turns. "
    "No quotes, no explanation, no punctuation fluff — about 3 to 12 words."
)

_QUOTE_RE = re.compile(r'^[\s"\'`“”‘’]+|[\s"\'`“”‘’]+$')
_PREFIX_RE = re.compile(
    r"^(?:search(?:\s+query)?|query|web\s+search)\s*[:=\-–—]\s*",
    re.IGNORECASE,
)


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def select_prior_messages(
    messages: list[dict[str, Any]] | None,
    *,
    limit: int = PRIOR_TURN_LIMIT,
) -> list[dict[str, str]]:
    """Take the last ``limit`` user/assistant turns before the current ask."""
    out: list[dict[str, str]] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        plain = content.strip()
        if not plain:
            continue
        out.append({"role": role, "content": _clip(plain, PRIOR_CONTENT_CHARS)})
    if limit <= 0:
        return []
    return out[-limit:]


def build_intent_rewrite_messages(
    current: str,
    messages: list[dict[str, Any]] | None = None,
    *,
    prior_limit: int = PRIOR_TURN_LIMIT,
) -> list[dict[str, str]]:
    """Build the system/user pair for the intent-rewrite completion call."""
    prior = select_prior_messages(messages, limit=prior_limit)
    lines: list[str] = []
    if prior:
        lines.append("Recent conversation:")
        for msg in prior:
            label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{label}: {msg['content']}")
        lines.append("")
    lines.append(f"Latest user message: {(current or '').strip()}")
    lines.append("")
    lines.append(
        "Reply with only the web search query that captures what they really mean."
    )
    return [
        {"role": "system", "content": INTENT_REWRITE_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def sanitize_rewritten_query(raw: str, *, max_words: int = 12) -> str:
    """Keep a single short search phrase; drop quotes and explanatory fluff."""
    text = (raw or "").strip()
    if not text:
        return ""
    # Prefer the first non-empty line; models sometimes add a second sentence.
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break
    text = _PREFIX_RE.sub("", text)
    text = _QUOTE_RE.sub("", text).strip()
    text = text.rstrip(" .;,:")
    # If the model ignored instructions and wrote a sentence, take a short head.
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text.strip()


def _extract_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = first.get("text")
    return text if isinstance(text, str) else ""


def call_chat_completion(
    api_base: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    timeout_s: float = REWRITE_TIMEOUT_S,
) -> str:
    """Non-streaming completion against the upstream OpenAI-compatible API."""
    base = (api_base or "").rstrip("/")
    if not base:
        raise ValueError("api_base required")
    body = {
        "model": model or DEFAULT_MODEL,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": REWRITE_MAX_TOKENS,
        "think": False,
        "thinking": {"type": "disabled"},
        "messages": messages,
    }
    raw = json.dumps(body).encode("utf-8")
    req = Request(
        f"{base}/v1/chat/completions",
        data=raw,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
    if not isinstance(payload, dict):
        return ""
    return _extract_completion_text(payload)


def rewrite_search_query(
    current: str,
    messages: list[dict[str, Any]] | None = None,
    *,
    api_base: str,
    model: str | None = None,
    timeout_s: float = REWRITE_TIMEOUT_S,
    max_words: int = 12,
    completion_fn: Callable[..., str] | None = None,
) -> str:
    """Ask the upstream model for a resolved search query; empty on failure."""
    plain = (current or "").strip()
    if not plain:
        return ""
    prompt_messages = build_intent_rewrite_messages(plain, messages)
    caller = completion_fn or call_chat_completion
    try:
        raw = caller(
            api_base,
            model=model or DEFAULT_MODEL,
            messages=prompt_messages,
            timeout_s=timeout_s,
        )
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return ""
    except Exception:  # noqa: BLE001 — never block web search on rewrite bugs
        return ""
    return sanitize_rewritten_query(raw, max_words=max_words)
