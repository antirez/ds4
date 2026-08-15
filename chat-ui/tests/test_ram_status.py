"""Unit tests for Activity Monitor-aligned Darwin RAM accounting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CHAT_UI_DIR = Path(__file__).resolve().parents[1]
if str(CHAT_UI_DIR) not in sys.path:
    sys.path.insert(0, str(CHAT_UI_DIR))

import server  # noqa: E402

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


class ParseVmStatTests(unittest.TestCase):
    def test_parse_vm_stat_pages(self) -> None:
        stats = server._parse_vm_stat_pages(VM_STAT_FIXTURE)
        self.assertEqual(stats["Pages wired down"], 467355)
        self.assertEqual(stats["Anonymous pages"], 926155)
        self.assertEqual(stats["Pages purgeable"], 42348)
        self.assertEqual(stats["Pages occupied by compressor"], 1424084)

    def test_activity_monitor_used_pages_prefers_anonymous(self) -> None:
        stats = server._parse_vm_stat_pages(VM_STAT_FIXTURE)
        # (926155 - 42348) + 467355 + 1424084
        self.assertEqual(server._activity_monitor_used_pages(stats), 2775246)

    def test_activity_monitor_used_pages_clamps_purgeable(self) -> None:
        stats = {
            "Anonymous pages": 100,
            "Pages purgeable": 250,
            "Pages wired down": 10,
            "Pages occupied by compressor": 5,
        }
        self.assertEqual(server._activity_monitor_used_pages(stats), 15)

    def test_activity_monitor_used_pages_prefers_pages_internal(self) -> None:
        stats = {
            "Pages internal": 900,
            "Anonymous pages": 100,
            "Pages purgeable": 100,
            "Pages wired down": 50,
            "Pages occupied by compressor": 25,
        }
        self.assertEqual(server._activity_monitor_used_pages(stats), 875)

    def test_old_active_formula_overcounts(self) -> None:
        stats = server._parse_vm_stat_pages(VM_STAT_FIXTURE)
        old = (
            stats["Pages active"]
            + stats["Pages wired down"]
            + stats["Pages occupied by compressor"]
        )
        new = server._activity_monitor_used_pages(stats)
        assert new is not None
        self.assertLess(new, old)
        self.assertGreater(old - new, 2_000_000)


if __name__ == "__main__":
    unittest.main()
