"""Unit tests for Mac RAM-safe --ctx policy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import safe_ctx as sc  # noqa: E402


class SafeCtxMathTests(unittest.TestCase):
    def test_128gb_leaves_six_gib_headroom(self) -> None:
        # 128 RAM, 81 model, 6 reserve → 41 GiB for KV → ~1M capped
        ctx = sc.max_safe_ctx(ram_gib=128, model_gib=81, reserve_gib=6)
        self.assertEqual(ctx, 1_000_000)
        planned = sc.planned_gib(81, ctx)
        self.assertLessEqual(planned, 128 - 6 + 1e-9)

    def test_tighter_ram_reduces_ctx(self) -> None:
        ctx = sc.max_safe_ctx(ram_gib=96, model_gib=81, reserve_gib=6)
        # budget 9 GiB → 9/26 * 1M ≈ 346153
        self.assertGreater(ctx, 300_000)
        self.assertLess(ctx, 400_000)
        self.assertLessEqual(sc.planned_gib(81, ctx), 96 - 6 + 1e-6)

    def test_impossible_fit_returns_zero(self) -> None:
        ctx = sc.max_safe_ctx(ram_gib=64, model_gib=81, reserve_gib=6)
        self.assertEqual(ctx, 0)

    def test_batched_sessions_shrink_ctx(self) -> None:
        one = sc.max_safe_ctx(ram_gib=128, model_gib=81, reserve_gib=6, sessions=1)
        two = sc.max_safe_ctx(ram_gib=128, model_gib=81, reserve_gib=6, sessions=2)
        self.assertLess(two, one)


class RamPressureLatchTests(unittest.TestCase):
    def test_trigger_aligns_with_reserve(self) -> None:
        self.assertEqual(sc.RAM_PRESSURE_TRIGGER_GIB, sc.DEFAULT_RESERVE_GIB)
        self.assertGreater(sc.RAM_PRESSURE_CLEAR_GIB, sc.RAM_PRESSURE_TRIGGER_GIB)

    def test_evaluate_transitions(self) -> None:
        self.assertEqual(sc.evaluate_ram_pressure(5.9, False), "trigger")
        self.assertEqual(sc.evaluate_ram_pressure(6.0, False), "trigger")
        self.assertEqual(sc.evaluate_ram_pressure(5.9, True), "hold")
        self.assertEqual(sc.evaluate_ram_pressure(6.0, True), "hold")
        self.assertEqual(sc.evaluate_ram_pressure(6.5, True), "idle")
        self.assertEqual(sc.evaluate_ram_pressure(7.1, True), "clear")
        self.assertEqual(sc.evaluate_ram_pressure(8.0, False), "idle")


if __name__ == "__main__":
    unittest.main()
