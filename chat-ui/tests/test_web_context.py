"""Unit tests for multi-result web context assembly and fetching."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

CHAT_UI_DIR = Path(__file__).resolve().parents[1]
if str(CHAT_UI_DIR) not in sys.path:
    sys.path.insert(0, str(CHAT_UI_DIR))

import search_intent as si  # noqa: E402
import web_context as wc  # noqa: E402


class FetchUntilTests(unittest.TestCase):
    def test_skips_failures_and_keeps_fetching(self) -> None:
        hits = [
            wc.SearchHit("a", "https://a.example/1"),
            wc.SearchHit("b", "https://b.example/2"),
            wc.SearchHit("c", "https://c.example/3"),
            wc.SearchHit("d", "https://d.example/4"),
        ]

        def fake_fetch(url: str, max_chars: int = wc.MAX_PAGE_CHARS) -> dict[str, str]:
            if "a.example" in url or "c.example" in url:
                raise wc.URLError("blocked")
            return {"url": url, "title": url, "text": f"body for {url}"}

        with patch.object(wc, "fetch_page_text", side_effect=fake_fetch):
            pages, errors, attempted = wc.fetch_pages_until(hits, max_fetch=2)

        self.assertEqual(len(pages), 2)
        self.assertEqual(attempted, 4)  # tried a,b,c,d until 2 successes
        self.assertEqual({p["url"] for p in pages}, {
            "https://b.example/2",
            "https://d.example/4",
        })
        self.assertGreaterEqual(len(errors), 2)

    def test_prefers_diverse_hosts_first(self) -> None:
        hits = [
            wc.SearchHit("a1", "https://same.example/1"),
            wc.SearchHit("a2", "https://same.example/2"),
            wc.SearchHit("b1", "https://other.example/1"),
        ]
        ordered = wc._ordered_fetch_candidates(hits)
        self.assertEqual(
            [h.url for h in ordered],
            [
                "https://same.example/1",
                "https://other.example/1",
                "https://same.example/2",
            ],
        )


class AssembleContextTests(unittest.TestCase):
    def test_keeps_multiple_page_sections_under_budget(self) -> None:
        hits = [
            wc.SearchHit(f"t{i}", f"https://ex.example/{i}", "snip")
            for i in range(1, 6)
        ]
        pages = [
            {"title": f"Page {i}", "url": f"https://ex.example/{i}", "text": ("word " * 400)}
            for i in range(1, 4)
        ]
        ctx = wc.assemble_context("q", hits, pages, [], max_chars=5000)
        self.assertIn("Search results:", ctx)
        self.assertGreaterEqual(ctx.count("### Page"), 2)
        self.assertLessEqual(len(ctx), 5000)


class IntentRewritePromptTests(unittest.TestCase):
    def test_prompt_includes_prior_turns_and_latest(self) -> None:
        messages = [
            {"role": "user", "content": "Tell me about DeepSeek V4 Flash"},
            {
                "role": "assistant",
                "content": "DeepSeek V4 Flash is a fast open-weight model.",
            },
            {"role": "user", "content": "How big is the context window?"},
            {"role": "assistant", "content": "It targets a long context window."},
        ]
        built = si.build_intent_rewrite_messages(
            "What's the release date?",
            messages,
            prior_limit=4,
        )
        self.assertEqual(built[0]["role"], "system")
        self.assertIn("web search query", built[0]["content"].lower())
        user = built[1]["content"]
        self.assertIn("DeepSeek V4 Flash", user)
        self.assertIn("Latest user message: What's the release date?", user)
        self.assertIn("User:", user)
        self.assertIn("Assistant:", user)

    def test_prior_limit_keeps_last_n_only(self) -> None:
        messages = [
            {"role": "user", "content": f"turn {i}"}
            for i in range(1, 8)
        ]
        prior = si.select_prior_messages(messages, limit=4)
        self.assertEqual([m["content"] for m in prior], [
            "turn 4", "turn 5", "turn 6", "turn 7",
        ])

    def test_sanitize_strips_quotes_and_clamps_words(self) -> None:
        self.assertEqual(
            si.sanitize_rewritten_query('"DeepSeek V4 Flash release date"'),
            "DeepSeek V4 Flash release date",
        )
        self.assertEqual(
            si.sanitize_rewritten_query("Query: DeepSeek V4 Flash release date"),
            "DeepSeek V4 Flash release date",
        )
        long = " ".join(f"w{i}" for i in range(20))
        out = si.sanitize_rewritten_query(long, max_words=12)
        self.assertEqual(len(out.split()), 12)


class DeriveSearchQueryTests(unittest.TestCase):
    """Fallback path (no live LLM): heuristic keyword/topic resolution."""

    def test_follow_up_uses_prior_topic(self) -> None:
        messages = [
            {"role": "user", "content": "Tell me about DeepSeek V4 Flash"},
            {
                "role": "assistant",
                "content": "DeepSeek V4 Flash is a fast open-weight model.",
            },
        ]
        q = wc.derive_search_query("What's the release date?", messages)
        low = q.lower()
        self.assertIn("deepseek", low)
        self.assertTrue("release" in low or "date" in low)
        self.assertLessEqual(len(q.split()), 12)

    def test_self_contained_query_keeps_current_ask(self) -> None:
        messages = [
            {"role": "user", "content": "Earlier we talked about weather in Rome"},
            {"role": "assistant", "content": "Rome is mild this week."},
        ]
        q = wc.derive_search_query("DeepSeek V4 Flash release date", messages)
        low = q.lower()
        self.assertIn("deepseek", low)
        self.assertNotIn("rome", low)
        self.assertNotIn("weather", low)

    def test_multi_sentence_current_keeps_subject(self) -> None:
        q = wc.derive_search_query(
            "We discussed DeepSeek V4 Flash earlier. What's the launch window?",
            [],
        )
        low = q.lower()
        self.assertIn("deepseek", low)
        self.assertTrue("launch" in low or "window" in low)
        # Prefer a compact search phrase over dumping both sentences.
        self.assertLessEqual(len(q.split()), 12)
        self.assertNotIn("we discussed", low)

    def test_llm_rewrite_preferred_when_available(self) -> None:
        messages = [
            {"role": "user", "content": "Tell me about DeepSeek V4 Flash"},
            {
                "role": "assistant",
                "content": "DeepSeek V4 Flash is a fast open-weight model.",
            },
        ]

        def fake_completion(
            api_base: str,
            *,
            model: str,
            messages: list[dict[str, str]],
            timeout_s: float,
        ) -> str:
            self.assertTrue(api_base)
            self.assertEqual(messages[0]["role"], "system")
            return "DeepSeek V4 Flash release date"

        q = wc.derive_search_query(
            "What's the release date?",
            messages,
            api_base="http://127.0.0.1:8000",
            completion_fn=fake_completion,
        )
        self.assertEqual(q, "DeepSeek V4 Flash release date")

    def test_llm_failure_falls_back_to_heuristic(self) -> None:
        messages = [
            {"role": "user", "content": "Tell me about DeepSeek V4 Flash"},
            {
                "role": "assistant",
                "content": "DeepSeek V4 Flash is a fast open-weight model.",
            },
        ]

        def boom(*_args, **_kwargs) -> str:
            raise TimeoutError("upstream slow")

        q = wc.derive_search_query(
            "What's the release date?",
            messages,
            api_base="http://127.0.0.1:8000",
            completion_fn=boom,
        )
        low = q.lower()
        self.assertIn("deepseek", low)
        self.assertTrue("release" in low or "date" in low)


class LiveSearchSmokeTests(unittest.TestCase):
    def test_live_search_returns_multiple_hits(self) -> None:
        try:
            hits = wc.search_duckduckgo("DeepSeek V4 Flash", max_results=5)
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"network/search unavailable: {exc}")
        if len(hits) < 2:
            self.skipTest(f"DDG returned only {len(hits)} hit(s)")
        self.assertGreaterEqual(len(hits), 2)


if __name__ == "__main__":
    unittest.main()
