#!/usr/bin/env python3
"""Run a command with Linux process-group memory sampling and safety limits.

The guard is designed for unified-memory systems such as NVIDIA DGX Spark,
where host memory pressure is a useful last-resort signal even when per-process
GPU accounting is unavailable. It never starts a model by itself; the command
to supervise must follow ``--`` on the command line.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


MIB = 1024 * 1024
DEFAULT_INTERVAL_MS = 250
DEFAULT_PSS_INTERVAL_MS = 1000
DEFAULT_NVIDIA_INTERVAL_MS = 1000
# Soft stop below the campaign's 110 GB decimal hard cap.
DEFAULT_MAX_GROUP_MIB = 108_000_000_000 // MIB
DEFAULT_MIN_AVAILABLE_MIB = 12 * 1024
WATCHDOG_EXIT = 75


@dataclass
class ProcessMemory:
    process_count: int = 0
    rss_bytes: int = 0
    pss_bytes: int | None = None
    vm_hwm_bytes: int = 0


@dataclass
class HostMemory:
    mem_total_bytes: int
    mem_available_bytes: int
    swap_free_bytes: int


def parse_kib_lines(text: str) -> dict[str, int]:
    """Parse ``Key: N kB`` records and return byte values."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, rest = line.partition(":")
        if not separator:
            continue
        fields = rest.split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        multiplier = 1024 if len(fields) < 2 or fields[1] == "kB" else 1
        values[key] = value * multiplier
    return values


def read_host_memory(path: Path = Path("/proc/meminfo")) -> HostMemory:
    values = parse_kib_lines(path.read_text(encoding="ascii"))
    return HostMemory(
        mem_total_bytes=values["MemTotal"],
        mem_available_bytes=values["MemAvailable"],
        swap_free_bytes=values.get("SwapFree", 0),
    )


def process_group_id(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    try:
        stat = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    closing = stat.rfind(")")
    if closing < 0:
        return None
    fields = stat[closing + 2 :].split()
    if len(fields) < 3:
        return None
    try:
        return int(fields[2])
    except ValueError:
        return None


def process_group_pids(pgid: int, proc_root: Path = Path("/proc")) -> list[int]:
    pids: list[int] = []
    for entry in proc_root.iterdir():
        if entry.name.isdigit() and process_group_id(int(entry.name), proc_root) == pgid:
            pids.append(int(entry.name))
    return sorted(pids)


def read_process_memory(
    pids: list[int],
    *,
    proc_root: Path = Path("/proc"),
) -> ProcessMemory:
    result = ProcessMemory()
    for pid in pids:
        try:
            status_values = parse_kib_lines(
                (proc_root / str(pid) / "status").read_text(encoding="ascii")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        result.process_count += 1
        result.rss_bytes += status_values.get("VmRSS", 0)
        result.vm_hwm_bytes += status_values.get("VmHWM", 0)
    return result


def read_process_pss(
    pids: list[int], proc_root: Path = Path("/proc")
) -> int | None:
    total = 0
    for pid in pids:
        try:
            rollup_values = parse_kib_lines(
                (proc_root / str(pid) / "smaps_rollup").read_text(encoding="ascii")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return None
        total += rollup_values.get("Pss", 0)
    return total


def limit_reason(
    process: ProcessMemory,
    host: HostMemory,
    *,
    max_rss_bytes: int,
    max_pss_bytes: int,
    min_available_bytes: int,
) -> str | None:
    if host.mem_available_bytes <= min_available_bytes:
        return "host_available"
    if process.rss_bytes >= max_rss_bytes:
        return "group_rss"
    if process.pss_bytes is not None and process.pss_bytes >= max_pss_bytes:
        return "group_pss"
    return None


def parse_nvidia_compute_apps(
    output: str, monitored_pids: set[int]
) -> tuple[int | None, str]:
    """Return aggregate NVIDIA-reported bytes and an availability status."""
    total_mib = 0
    matched = False
    unavailable = False
    for line in output.splitlines():
        pid_text, separator, memory_text = line.partition(",")
        if not separator:
            continue
        try:
            pid = int(pid_text.strip())
        except ValueError:
            continue
        if pid not in monitored_pids:
            continue
        matched = True
        memory_text = memory_text.strip()
        try:
            total_mib += int(memory_text)
        except ValueError:
            unavailable = True
    if unavailable:
        return None, "not_available"
    return total_mib * MIB, "ok" if matched else "no_compute_process"


def read_nvidia_memory(pids: list[int]) -> tuple[int | None, str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=0.2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, "unavailable"
    if completed.returncode != 0:
        return None, "error"
    return parse_nvidia_compute_apps(completed.stdout, set(pids))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample and guard a command's Linux process-group memory."
    )
    parser.add_argument("--csv", type=Path, required=True, help="memory trace path")
    parser.add_argument(
        "--summary-json", type=Path, required=True, help="run summary path"
    )
    parser.add_argument(
        "--interval-ms", type=positive_int, default=DEFAULT_INTERVAL_MS
    )
    parser.add_argument(
        "--pss-interval-ms", type=positive_int, default=DEFAULT_PSS_INTERVAL_MS
    )
    parser.add_argument(
        "--nvidia-interval-ms",
        type=positive_int,
        default=DEFAULT_NVIDIA_INTERVAL_MS,
    )
    parser.add_argument(
        "--no-nvidia-smi",
        action="store_true",
        help="disable NVIDIA per-process accounting (CSV status: disabled)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing CSV/JSON artifacts",
    )
    parser.add_argument(
        "--max-rss-mib", type=positive_int, default=DEFAULT_MAX_GROUP_MIB
    )
    parser.add_argument(
        "--max-pss-mib", type=positive_int, default=DEFAULT_MAX_GROUP_MIB
    )
    parser.add_argument(
        "--min-available-mib",
        type=positive_int,
        default=DEFAULT_MIN_AVAILABLE_MIB,
    )
    parser.add_argument("--grace-seconds", type=float, default=10.0)
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="command, preceded by --"
    )
    return parser


def normalize_command(parser: argparse.ArgumentParser, command: list[str]) -> list[str]:
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    return command


def safe_signal_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    if args.grace_seconds < 0:
        raise SystemExit("--grace-seconds must be non-negative")
    if args.pss_interval_ms < args.interval_ms:
        raise SystemExit("--pss-interval-ms must be >= --interval-ms")
    if args.nvidia_interval_ms < args.interval_ms:
        raise SystemExit("--nvidia-interval-ms must be >= --interval-ms")
    if args.csv.resolve() == args.summary_json.resolve():
        raise SystemExit("--csv and --summary-json must be different paths")
    existing = [path for path in (args.csv, args.summary_json) if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise SystemExit("refusing to overwrite existing artifact(s): " + names)

    initial_host = read_host_memory()
    if initial_host.mem_available_bytes <= args.min_available_mib * MIB:
        raise SystemExit(
            "refusing to start: MemAvailable is already at or below "
            f"{args.min_available_mib} MiB"
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    # Fail on output-path errors before a supervised command can allocate memory.
    with args.csv.open("w", encoding="utf-8", newline=""):
        pass

    guard_signal: int | None = None

    def handle_guard_signal(signum: int, _frame: object) -> None:
        nonlocal guard_signal
        guard_signal = signum

    previous_handlers = {
        sig: signal.signal(sig, handle_guard_signal)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }

    started_utc = utc_now()
    started = time.monotonic()
    try:
        child = subprocess.Popen(args.command, start_new_session=True)
    except BaseException:
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)
        raise
    pgid = child.pid
    max_rss = 0
    max_pss = 0
    max_hwm = 0
    min_available: int | None = None
    last_pss: int | None = None
    next_pss_at = started
    last_nvidia: int | None = None
    last_nvidia_status = "disabled" if args.no_nvidia_smi else "not_sampled"
    next_nvidia_at = started
    max_nvidia: int | None = None
    trigger: str | None = None
    termination_started: float | None = None
    sent_sigkill = False
    samples = 0

    fieldnames = [
        "utc",
        "elapsed_ms",
        "root_pid",
        "process_count",
        "group_rss_bytes",
        "group_pss_bytes",
        "group_vm_hwm_bytes",
        "mem_total_bytes",
        "mem_available_bytes",
        "swap_free_bytes",
        "nvidia_process_used_bytes",
        "nvidia_status",
        "event",
    ]

    try:
        with args.csv.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            csv_file.flush()

            while True:
                sample_started = time.monotonic()
                pids = process_group_pids(pgid)
                include_pss = sample_started >= next_pss_at
                process = read_process_memory(pids)
                process.pss_bytes = last_pss
                host = read_host_memory()

                max_rss = max(max_rss, process.rss_bytes)
                if process.pss_bytes is not None:
                    max_pss = max(max_pss, process.pss_bytes)
                max_hwm = max(max_hwm, process.vm_hwm_bytes)
                if last_nvidia is not None:
                    max_nvidia = (
                        last_nvidia
                        if max_nvidia is None
                        else max(max_nvidia, last_nvidia)
                    )
                min_available = (
                    host.mem_available_bytes
                    if min_available is None
                    else min(min_available, host.mem_available_bytes)
                )

                event = ""
                if guard_signal is not None and termination_started is None:
                    event = "guard_signal:" + signal.Signals(guard_signal).name
                    termination_started = sample_started
                    safe_signal_group(pgid, signal.SIGTERM)
                elif trigger is None and guard_signal is None:
                    trigger = limit_reason(
                        process,
                        host,
                        max_rss_bytes=args.max_rss_mib * MIB,
                        max_pss_bytes=args.max_pss_mib * MIB,
                        min_available_bytes=args.min_available_mib * MIB,
                    )
                    if trigger is not None:
                        event = "watchdog_sigterm:" + trigger
                        termination_started = sample_started
                        safe_signal_group(pgid, signal.SIGTERM)
                        print(
                            "memory_guard: limit %s exceeded; sent SIGTERM to pgid %d"
                            % (trigger, pgid),
                            file=sys.stderr,
                            flush=True,
                        )
                if trigger is None and guard_signal is None and include_pss:
                    last_pss = read_process_pss(pids)
                    process.pss_bytes = last_pss
                    next_pss_at = sample_started + args.pss_interval_ms / 1000.0
                    if process.pss_bytes is not None:
                        max_pss = max(max_pss, process.pss_bytes)
                    trigger = limit_reason(
                        process,
                        host,
                        max_rss_bytes=args.max_rss_mib * MIB,
                        max_pss_bytes=args.max_pss_mib * MIB,
                        min_available_bytes=args.min_available_mib * MIB,
                    )
                    if trigger is not None:
                        event = "watchdog_sigterm:" + trigger
                        termination_started = time.monotonic()
                        safe_signal_group(pgid, signal.SIGTERM)
                        print(
                            "memory_guard: limit %s exceeded; sent SIGTERM to pgid %d"
                            % (trigger, pgid),
                            file=sys.stderr,
                            flush=True,
                        )

                if not args.no_nvidia_smi and sample_started >= next_nvidia_at:
                    last_nvidia, last_nvidia_status = read_nvidia_memory(pids)
                    if last_nvidia is not None:
                        max_nvidia = (
                            last_nvidia
                            if max_nvidia is None
                            else max(max_nvidia, last_nvidia)
                        )
                    next_nvidia_at = (
                        sample_started + args.nvidia_interval_ms / 1000.0
                    )
                if (
                    not sent_sigkill
                    and termination_started is not None
                    and sample_started - termination_started >= args.grace_seconds
                    and pids
                ):
                    stop_reason = trigger or "guard_signal"
                    event = "sigkill:" + stop_reason
                    sent_sigkill = True
                    safe_signal_group(pgid, signal.SIGKILL)
                    print(
                        "memory_guard: grace period expired; sent SIGKILL to pgid %d"
                        % pgid,
                        file=sys.stderr,
                        flush=True,
                    )

                writer.writerow(
                    {
                        "utc": utc_now(),
                        "elapsed_ms": round((sample_started - started) * 1000),
                        "root_pid": child.pid,
                        "process_count": process.process_count,
                        "group_rss_bytes": process.rss_bytes,
                        "group_pss_bytes": (
                            "" if process.pss_bytes is None else process.pss_bytes
                        ),
                        "group_vm_hwm_bytes": process.vm_hwm_bytes,
                        "mem_total_bytes": host.mem_total_bytes,
                        "mem_available_bytes": host.mem_available_bytes,
                        "swap_free_bytes": host.swap_free_bytes,
                        "nvidia_process_used_bytes": (
                            "" if last_nvidia is None else last_nvidia
                        ),
                        "nvidia_status": last_nvidia_status,
                        "event": event,
                    }
                )
                csv_file.flush()
                samples += 1

                child_status = child.poll()
                if child_status is not None and not pids:
                    break
                sleep_for = args.interval_ms / 1000.0 - (
                    time.monotonic() - sample_started
                )
                if sleep_for > 0:
                    time.sleep(sleep_for)
    except KeyboardInterrupt:  # Defensive fallback for unusual signal handling.
        guard_signal = guard_signal or signal.SIGINT
        safe_signal_group(pgid, signal.SIGTERM)
        try:
            child.wait(timeout=max(0.0, args.grace_seconds))
        except subprocess.TimeoutExpired:
            sent_sigkill = True
            safe_signal_group(pgid, signal.SIGKILL)

    finally:
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)

    child_status = child.wait()
    finished = time.monotonic()
    if trigger is not None:
        exit_code = WATCHDOG_EXIT
    elif guard_signal is not None:
        exit_code = 128 + guard_signal
    else:
        exit_code = child_status
    summary: dict[str, object] = {
        "command": args.command,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "elapsed_seconds": round(finished - started, 3),
        "sample_interval_ms": args.interval_ms,
        "pss_interval_ms": args.pss_interval_ms,
        "nvidia_interval_ms": args.nvidia_interval_ms,
        "samples": samples,
        "child_exit_code": child_status,
        "exit_code": exit_code,
        "watchdog_trigger": trigger,
        "guard_signal": (
            None if guard_signal is None else signal.Signals(guard_signal).name
        ),
        "sent_sigkill": sent_sigkill,
        "limits": {
            "max_rss_mib": args.max_rss_mib,
            "max_pss_mib": args.max_pss_mib,
            "min_available_mib": args.min_available_mib,
            "grace_seconds": args.grace_seconds,
        },
        "peaks": {
            "group_rss_bytes": max_rss,
            "group_pss_bytes": max_pss,
            "group_vm_hwm_bytes": max_hwm,
            "nvidia_process_used_bytes": max_nvidia,
            "min_mem_available_bytes": min_available,
        },
    }
    write_summary(args.summary_json, summary)
    return exit_code


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.command = normalize_command(parser, args.command)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
