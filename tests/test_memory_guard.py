#!/usr/bin/env python3
"""Unit and synthetic-process tests for speed-bench/memory_guard.py."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "speed-bench" / "memory_guard.py"
SPEC = importlib.util.spec_from_file_location("memory_guard", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
memory_guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = memory_guard
SPEC.loader.exec_module(memory_guard)


class MemoryGuardUnitTests(unittest.TestCase):
    def test_parse_kib_lines(self):
        values = memory_guard.parse_kib_lines(
            "MemTotal: 123 kB\nMemAvailable: 45 kB\nignored\n"
        )
        self.assertEqual(values["MemTotal"], 123 * 1024)
        self.assertEqual(values["MemAvailable"], 45 * 1024)

    def test_limit_priority(self):
        process = memory_guard.ProcessMemory(
            process_count=1,
            rss_bytes=20 * memory_guard.MIB,
            pss_bytes=10 * memory_guard.MIB,
        )
        host = memory_guard.HostMemory(
            mem_total_bytes=100 * memory_guard.MIB,
            mem_available_bytes=5 * memory_guard.MIB,
            swap_free_bytes=0,
        )
        self.assertEqual(
            memory_guard.limit_reason(
                process,
                host,
                max_rss_bytes=15 * memory_guard.MIB,
                max_pss_bytes=15 * memory_guard.MIB,
                min_available_bytes=6 * memory_guard.MIB,
            ),
            "host_available",
        )

    def test_parse_nvidia_compute_apps(self):
        value, status = memory_guard.parse_nvidia_compute_apps(
            "101, 256\n202, [N/A]\n303, 99\n", {101}
        )
        self.assertEqual(value, 256 * memory_guard.MIB)
        self.assertEqual(status, "ok")

        value, status = memory_guard.parse_nvidia_compute_apps(
            "101, 256\n202, [N/A]\n", {202}
        )
        self.assertIsNone(value)
        self.assertEqual(status, "not_available")


class MemoryGuardIntegrationTests(unittest.TestCase):
    def run_guard(self, directory: Path, extra: list[str], child: str):
        csv_path = directory / "trace.csv"
        summary_path = directory / "summary.json"
        command = [
            sys.executable,
            str(SCRIPT),
            "--csv",
            str(csv_path),
            "--summary-json",
            str(summary_path),
            "--interval-ms",
            "25",
            "--pss-interval-ms",
            "25",
            "--no-nvidia-smi",
            *extra,
            "--",
            sys.executable,
            "-c",
            child,
        ]
        completed = subprocess.run(command, check=False, timeout=10)
        with csv_path.open(newline="", encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return completed, rows, summary

    def test_success_propagates_child_status_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed, rows, summary = self.run_guard(
                Path(temporary),
                ["--max-rss-mib", "128", "--max-pss-mib", "128"],
                "import time; time.sleep(0.08)",
            )
        self.assertEqual(completed.returncode, 0)
        self.assertGreaterEqual(len(rows), 2)
        self.assertIsNone(summary["watchdog_trigger"])
        self.assertEqual(summary["child_exit_code"], 0)

    def test_rss_limit_terminates_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed, rows, summary = self.run_guard(
                Path(temporary),
                [
                    "--max-rss-mib",
                    "12",
                    "--max-pss-mib",
                    "128",
                    "--grace-seconds",
                    "0.1",
                ],
                (
                    "import subprocess,sys,time; "
                    "subprocess.Popen([sys.executable,'-c',"
                    "'import time; payload=bytearray(32*1024*1024); time.sleep(5)']); "
                    "time.sleep(5)"
                ),
            )
        self.assertEqual(completed.returncode, memory_guard.WATCHDOG_EXIT)
        self.assertEqual(summary["watchdog_trigger"], "group_rss")
        self.assertTrue(any(row["event"].startswith("watchdog_sigterm") for row in rows))

    def test_sigterm_is_forwarded_and_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            csv_path = directory / "trace.csv"
            summary_path = directory / "summary.json"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--csv",
                    str(csv_path),
                    "--summary-json",
                    str(summary_path),
                    "--interval-ms",
                    "25",
                    "--pss-interval-ms",
                    "25",
                    "--no-nvidia-smi",
                    "--",
                    sys.executable,
                    "-c",
                    "import time; time.sleep(5)",
                ]
            )
            for _ in range(100):
                if csv_path.exists() and csv_path.stat().st_size > 0:
                    break
                time.sleep(0.01)
            os.kill(process.pid, signal.SIGTERM)
            self.assertEqual(process.wait(timeout=5), 128 + signal.SIGTERM)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["guard_signal"], "SIGTERM")
        self.assertIsNone(summary["watchdog_trigger"])

    def test_sigkill_escalation_for_uncooperative_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed, rows, summary = self.run_guard(
                Path(temporary),
                [
                    "--max-rss-mib",
                    "12",
                    "--max-pss-mib",
                    "128",
                    "--grace-seconds",
                    "0.1",
                ],
                (
                    "import signal,time; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "payload=bytearray(32*1024*1024); time.sleep(5)"
                ),
            )
        self.assertEqual(completed.returncode, memory_guard.WATCHDOG_EXIT)
        self.assertTrue(summary["sent_sigkill"])
        self.assertTrue(any(row["event"] == "sigkill:group_rss" for row in rows))


if __name__ == "__main__":
    unittest.main()
