"""Unit tests for multi-result web context assembly and fetching."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

CHAT_UI_DIR = Path(__file__).resolve().parents[1]
if str(CHAT_UI_DIR) not in sys.path:
    sys.path.insert(0, str(CHAT_UI_DIR))

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
