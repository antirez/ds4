"""Tests for compress message-selection (mirrors chat-ui/static/app.js)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

CHAT_UI_DIR = Path(__file__).resolve().parents[1]
if str(CHAT_UI_DIR) not in sys.path:
    sys.path.insert(0, str(CHAT_UI_DIR))

import compress_history as ch  # noqa: E402


def _msg(role: str, content: str, **extra: object) -> dict:
    row: dict = {"role": role, "content": content}
    row.update(extra)
    return row


def _fat_chat() -> list[dict]:
    """Shape similar to the ingest/summary case: small head, fat web tails."""
    return [
        _msg("system", "[conversation summary]\nold", compressed=True),
        _msg("user", "web-a " + ("A" * 6000)),
        _msg("user", "so?"),
        _msg("user", "continue"),
        _msg("assistant", "ans-1 " + ("B" * 2000)),
        _msg("user", "web-b " + ("C" * 12000)),
        _msg("assistant", "ans-2 " + ("D" * 3000)),
        _msg("user", "web-c " + ("E" * 12000)),
        _msg("user", "web-d " + ("F" * 13000)),
    ]


class EstimateTokensTests(unittest.TestCase):
    def test_ceil_chars_over_four(self) -> None:
        self.assertEqual(ch.estimate_tokens("abcd"), 1)
        self.assertEqual(ch.estimate_tokens("abcde"), 2)
        self.assertEqual(ch.estimate_tokens(""), 0)


class KeepRecentForPassTests(unittest.TestCase):
    def test_force_starts_at_two_then_zero(self) -> None:
        self.assertEqual(ch.keep_recent_for_pass(True, 1), 2)
        self.assertEqual(ch.keep_recent_for_pass(True, 2), 0)
        self.assertEqual(ch.keep_recent_for_pass(True, 3), 0)

    def test_auto_starts_at_six(self) -> None:
        self.assertEqual(ch.keep_recent_for_pass(False, 1), 6)
        self.assertEqual(ch.keep_recent_for_pass(False, 2), 4)


class LegacyKeepPlusOneGateTests(unittest.TestCase):
    def test_old_gate_blocked_summary_plus_six(self) -> None:
        """Document the bug: length == KEEP+1 froze further compress."""
        msgs = _fat_chat()
        # After one legacy pass keep=6 → 1 summary + 6 recent = 7.
        plan = ch.plan_compress_slices(msgs, 6, allow_leftover=False)
        assert plan is not None
        legacy = ch.apply_summary_message(plan, "s" * 500)
        self.assertEqual(len(legacy), 7)
        blocked_by_old_gate = len(legacy) <= ch.KEEP_RECENT_MESSAGES + 1
        self.assertTrue(blocked_by_old_gate)
        # Fixed gate uses length <= keep (not keep+1); keep=2 still works.
        fixed = ch.plan_compress_slices(legacy, 2, allow_leftover=False)
        self.assertIsNotNone(fixed)
        assert fixed is not None
        self.assertEqual(len(fixed["recent"]), 2)
        self.assertGreaterEqual(len(fixed["older"]), 1)


class ForceCompressShrinksTests(unittest.TestCase):
    def test_force_compress_much_smaller_than_source(self) -> None:
        source = _fat_chat()
        before = ch.history_token_estimate(source)
        # ~800 chars summary ≈ model max_tokens=2048 rough upper for dense text.
        out = ch.simulate_force_compress(source, summary_chars=800)
        after = ch.history_token_estimate(out)
        self.assertLess(after, before)
        # Force keep=2 summarizes the fat mid-history; only a short tail remains.
        self.assertLess(after, before * 0.6)
        self.assertTrue(out[0].get("compressed"))
        self.assertTrue(out[0]["content"].startswith(ch.SUMMARY_PREFIX))
        # Force keep=2 on first successful pass → summary + last 2.
        self.assertEqual(len(out), 3)

    def test_force_peels_to_summary_when_still_over_hard_limit(self) -> None:
        source = _fat_chat()
        # Tiny hard limit forces a second pass with keep=0.
        out = ch.simulate_force_compress(
            source, summary_chars=800, hard_token_limit=100
        )
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].get("compressed"))

    def test_reject_growth_when_summary_larger_than_dropped(self) -> None:
        # Tiny older head, huge keep=6 tail — a verbose summary can grow.
        msgs = [
            _msg("user", "hi"),
            _msg("assistant", "ok"),
            *[_msg("user", "x" * 2000) for _ in range(6)],
        ]
        before = ch.history_token_estimate(msgs)
        plan = ch.plan_compress_slices(msgs, 6, allow_leftover=False)
        assert plan is not None
        # Summary larger than the two dropped msgs ("hi"+"ok").
        grown = ch.apply_summary_message(plan, "Z" * 5000)
        self.assertGreaterEqual(ch.history_token_estimate(grown), before)
        # Force path with keep reduction should still land smaller.
        out = ch.simulate_force_compress(msgs, summary_chars=800)
        self.assertLess(ch.history_token_estimate(out), before)

    def test_sidebar_style_ceil_chars_over_four(self) -> None:
        # API/sidebar sums ceil(chars/4) per message (not ceil of the total).
        source = _fat_chat()
        expected = sum(math.ceil(len(m["content"]) / 4) for m in source)
        self.assertEqual(ch.history_token_estimate(source), expected)


if __name__ == "__main__":
    unittest.main()
