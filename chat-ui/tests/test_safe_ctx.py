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


class AvailableRamTests(unittest.TestCase):
    VM_STAT_FIXTURE = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               17436.
Pages active:                           3200370.
Pages inactive:                         3197982.
Pages speculative:                         1192.
Pages throttled:                              0.
Pages wired down:                        467355.
Pages purgeable:                          42348.
File-backed pages:                      5473389.
Anonymous pages:                         926155.
Pages stored in compressor:             3639288.
Pages occupied by compressor:           1424084.
"""

    def test_parse_vm_stat_pages(self) -> None:
        stats = sc._parse_vm_stat_pages(self.VM_STAT_FIXTURE)
        self.assertEqual(stats["Pages wired down"], 467355)
        self.assertEqual(stats["Anonymous pages"], 926155)

    def test_darwin_used_bytes_vm_stat_matches_activity_monitor_formula(self) -> None:
        import unittest.mock as mock

        with mock.patch("subprocess.check_output", return_value=self.VM_STAT_FIXTURE):
            used = sc._darwin_used_bytes_vm_stat(16384)
        # (926155 anon - 42348 purgeable) + 467355 wired + 1424084 compressor
        self.assertEqual(used, 2775246 * 16384)

    def test_parse_meminfo_text(self) -> None:
        raw = "MemTotal:       32000000 kB\nMemAvailable:   12000000 kB\n"
        info = sc._parse_meminfo_text(raw)
        self.assertEqual(info["MemTotal"], 32000000 * 1024)
        self.assertEqual(info["MemAvailable"], 12000000 * 1024)

    def test_available_ram_lower_than_total_shrinks_ctx_vs_ignore_flag(self) -> None:
        # A machine with 128 GiB total but only 96 GiB free should recommend a
        # smaller --ctx than blindly budgeting off total RAM.
        with_free = sc.max_safe_ctx(ram_gib=96, model_gib=81, reserve_gib=6)
        with_total = sc.max_safe_ctx(ram_gib=128, model_gib=81, reserve_gib=6)
        self.assertLess(with_free, with_total)


class ConservativeBudgetTests(unittest.TestCase):
    def test_conservative_caps_below_free_ram(self) -> None:
        # 128 total, 96 free, 81 model → 70% cap = 89.6 < 96
        avail, source, relaxed = sc.effective_budget_gib(
            free_gib=96,
            total_gib=128,
            reserve_gib=6,
            model_gib=81,
            conservative=True,
        )
        self.assertAlmostEqual(avail, 128 * 0.70)
        self.assertIn("70%", source)
        self.assertFalse(relaxed)
        ctx_normal = sc.max_safe_ctx(ram_gib=96, model_gib=81, reserve_gib=6)
        ctx_cons = sc.max_safe_ctx(ram_gib=avail, model_gib=81, reserve_gib=6)
        self.assertLess(ctx_cons, ctx_normal)

    def test_conservative_planned_stays_under_seventy_percent_total(self) -> None:
        avail, _, _ = sc.effective_budget_gib(
            free_gib=96,
            total_gib=128,
            reserve_gib=6,
            model_gib=81,
            conservative=True,
        )
        ctx = sc.max_safe_ctx(ram_gib=avail, model_gib=81, reserve_gib=6)
        planned = sc.planned_gib(81, ctx)
        self.assertLessEqual(planned, 128 * 0.70 + 1e-6)

    def test_conservative_relaxed_when_model_exceeds_cap(self) -> None:
        # 64 GiB machine, 70% = 44.8 — cannot fit an 81 GiB model under cap
        avail, source, relaxed = sc.effective_budget_gib(
            free_gib=50,
            total_gib=64,
            reserve_gib=6,
            model_gib=81,
            conservative=True,
        )
        self.assertEqual(avail, 50)
        self.assertTrue(relaxed)
        self.assertIn("relaxed", source)

    def test_custom_fraction(self) -> None:
        avail, source, relaxed = sc.effective_budget_gib(
            free_gib=100,
            total_gib=128,
            reserve_gib=6,
            model_gib=40,
            conservative=True,
            max_used_fraction=0.60,
        )
        self.assertAlmostEqual(avail, 128 * 0.60)
        self.assertIn("60%", source)
        self.assertFalse(relaxed)


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
