"""Unit tests for chat list token/size metadata helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

CHAT_UI_DIR = Path(__file__).resolve().parents[1]
if str(CHAT_UI_DIR) not in sys.path:
    sys.path.insert(0, str(CHAT_UI_DIR))

import server  # noqa: E402


class EstimateTokensTests(unittest.TestCase):
    def test_empty_is_zero(self) -> None:
        self.assertEqual(server.estimate_tokens(""), 0)
        self.assertEqual(server.estimate_tokens(None), 0)

    def test_chars_over_four(self) -> None:
        self.assertEqual(server.estimate_tokens("abcd"), 1)
        self.assertEqual(server.estimate_tokens("abcdefgh"), 2)


class MessageContentTextTests(unittest.TestCase):
    def test_string_passthrough(self) -> None:
        self.assertEqual(server.message_content_text("hello"), "hello")

    def test_list_parts_with_text(self) -> None:
        content = [
            {"type": "text", "text": "aa"},
            {"type": "text", "text": "bb"},
            {"type": "image_url", "image_url": {"url": "x"}},
        ]
        self.assertEqual(server.message_content_text(content), "aabb")

    def test_unknown_shape_is_empty(self) -> None:
        self.assertEqual(server.message_content_text({"text": "no"}), "")


class HistoryTokenEstimateTests(unittest.TestCase):
    def test_sums_string_contents(self) -> None:
        messages = [
            {"role": "user", "content": "abcd"},
            {"role": "assistant", "content": "efghijkl"},
        ]
        self.assertEqual(server.history_token_estimate(messages), 3)

    def test_counts_list_part_text(self) -> None:
        messages = [{"role": "user", "content": [{"type": "text", "text": "abcd"}]}]
        self.assertEqual(server.history_token_estimate(messages), 1)

    def test_non_list_is_zero(self) -> None:
        self.assertEqual(server.history_token_estimate(None), 0)


class FormatSizeLabelTests(unittest.TestCase):
    def test_bytes_kb_mb(self) -> None:
        self.assertEqual(server.format_size_label(500), "500 B")
        self.assertEqual(server.format_size_label(1536), "1.5 KB")
        self.assertEqual(server.format_size_label(12 * 1024), "12 KB")
        self.assertEqual(server.format_size_label(2 * 1024 * 1024), "2.0 MB")


class ListChatsMetaTests(unittest.TestCase):
    def test_list_includes_tokens_size_and_updated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chats_dir = Path(tmp)
            body = {
                "id": "chat-a",
                "title": "Sample",
                "created_at": "2026-08-11T10:00:00Z",
                "updated_at": "2026-08-11T12:00:00Z",
                "messages": [
                    {"role": "user", "content": "abcd" * 100},
                    {"role": "assistant", "content": "efgh" * 100},
                ],
            }
            path = chats_dir / "chat-a.json"
            path.write_text(json.dumps(body), encoding="utf-8")
            size_bytes = path.stat().st_size

            rows = server.list_chats(chats_dir)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["id"], "chat-a")
            self.assertEqual(row["updated_at"], "2026-08-11T12:00:00Z")
            self.assertEqual(row["message_count"], 2)
            self.assertEqual(row["size_bytes"], size_bytes)
            self.assertEqual(row["size_label"], server.format_size_label(size_bytes))
            self.assertGreater(row["token_estimate"], 0)
            self.assertEqual(
                row["token_estimate"],
                server.history_token_estimate(body["messages"]),
            )


if __name__ == "__main__":
    unittest.main()
