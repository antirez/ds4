#!/usr/bin/env python3
"""Recommend a Mac-safe ds4-server --ctx given RAM headroom.

Policy: planned resident model + context working set should stay at or under
  free_RAM_right_now - reserve_GiB
(default reserve 6 GiB as a buffer for the OS, chat-ui, and other apps growing
their own usage while ds4-server runs). Budgeting off the RAM that's actually
free — rather than total physical RAM — matters because whatever the desktop
already holds (browser tabs, an IDE, etc.) doesn't get freed just because
ds4-server is about to start; pass --ignore-current-usage to fall back to the
old total-RAM policy.

Context scaling uses the Flash figure from the ds4 README: about 26 GiB of
KV/indexer pressure at 1M tokens. Model residency defaults from the GGUF size
with a Flash-q2 mapped/resident factor measured on Metal (~0.83 of file size),
overridable via flags.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

GIB = 1024.0**3
# antirez/ds4 README: ~26 GiB context-side memory at 1M tokens (Flash).
KV_GIB_PER_MTOKEN = 26.0
# Assume the mapped GGUF can be fully resident (Metal unified memory). Safer
# than under-counting; override with --model-gib / --resident-factor if needed.
DEFAULT_RESIDENT_FACTOR = 1.0
DEFAULT_RESERVE_GIB = 6.0
# Conservative launcher: cap planned model+KV at this fraction of total RAM.
CONSERVATIVE_MAX_USED_FRACTION = 0.70
# Chat-ui low-RAM fallback: trigger when free <= reserve, clear latch a bit above.
RAM_PRESSURE_TRIGGER_GIB = DEFAULT_RESERVE_GIB
RAM_PRESSURE_CLEAR_GIB = 7.0
MODEL_MAX_CTX = 1_000_000
DEFAULT_MODEL_NAMES = ("ds4flash.gguf", "ds4flash-q2.gguf", "DeepSeek-V4-Flash.gguf")


def total_ram_bytes() -> int:
    if sys.platform == "darwin":
        import subprocess

        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page > 0:
            return int(pages * page)
    except (ValueError, OSError):
        pass
    raise RuntimeError("cannot determine total RAM on this platform")


def _parse_vm_stat_pages(raw: str) -> dict[str, int]:
    """Parse `vm_stat` text into page counts keyed by field name."""
    stats: dict[str, int] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        digits = "".join(ch for ch in val if ch.isdigit())
        if digits:
            stats[key.strip().strip('"')] = int(digits)
    return stats


def _darwin_used_bytes_host_statistics64(page_size: int) -> int | None:
    """Activity Monitor-style "Memory Used" via mach host_statistics64."""
    import ctypes
    import ctypes.util

    lib_name = ctypes.util.find_library("c")
    if not lib_name:
        return None
    libc = ctypes.CDLL(lib_name, use_errno=True)

    class VmStatistics64(ctypes.Structure):
        _fields_ = [
            ("free_count", ctypes.c_uint32),
            ("active_count", ctypes.c_uint32),
            ("inactive_count", ctypes.c_uint32),
            ("wire_count", ctypes.c_uint32),
            ("zero_fill_count", ctypes.c_uint64),
            ("reactivations", ctypes.c_uint64),
            ("pageins", ctypes.c_uint64),
            ("pageouts", ctypes.c_uint64),
            ("faults", ctypes.c_uint64),
            ("cow_faults", ctypes.c_uint64),
            ("lookups", ctypes.c_uint64),
            ("hits", ctypes.c_uint64),
            ("purges", ctypes.c_uint64),
            ("purgeable_count", ctypes.c_uint32),
            ("speculative_count", ctypes.c_uint32),
            ("decompressions", ctypes.c_uint64),
            ("compressions", ctypes.c_uint64),
            ("swapins", ctypes.c_uint64),
            ("swapouts", ctypes.c_uint64),
            ("compressor_page_count", ctypes.c_uint32),
            ("throttled_count", ctypes.c_uint32),
            ("external_page_count", ctypes.c_uint32),
            ("internal_page_count", ctypes.c_uint32),
            ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
        ]

    HOST_VM_INFO64 = 4
    mach_host_self = libc.mach_host_self
    mach_host_self.restype = ctypes.c_uint
    host_statistics64 = libc.host_statistics64
    host_statistics64.argtypes = [
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    host_statistics64.restype = ctypes.c_int

    info = VmStatistics64()
    count = ctypes.c_uint(ctypes.sizeof(VmStatistics64) // ctypes.sizeof(ctypes.c_int))
    kr = host_statistics64(
        mach_host_self(), HOST_VM_INFO64, ctypes.byref(info), ctypes.byref(count)
    )
    if kr != 0:
        return None
    app_pages = max(0, int(info.internal_page_count) - int(info.purgeable_count))
    used_pages = app_pages + int(info.wire_count) + int(info.compressor_page_count)
    return used_pages * int(page_size)


def _darwin_used_bytes_vm_stat(page_size: int) -> int | None:
    import subprocess

    raw = subprocess.check_output(["vm_stat"], text=True)
    stats = _parse_vm_stat_pages(raw)
    internal = stats.get("Pages internal", stats.get("Anonymous pages"))
    wired = stats.get("Pages wired down")
    compressor = stats.get("Pages occupied by compressor")
    if internal is None or wired is None or compressor is None:
        return None
    purgeable = stats.get("Pages purgeable", 0)
    app_pages = max(0, int(internal) - int(purgeable))
    return (app_pages + int(wired) + int(compressor)) * int(page_size)


def _parse_meminfo_text(raw: str) -> dict[str, int]:
    """Parse `/proc/meminfo` text into byte counts keyed by field name."""
    info: dict[str, int] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        parts = val.split()
        if parts:
            info[key] = int(parts[0]) * 1024
    return info


def available_ram_bytes() -> tuple[int, str]:
    """Currently-free RAM: total minus what other apps already hold.

    Returns (free_bytes, source). Falls back to total RAM ("total-fallback")
    if the platform lookup fails, so a run never hard-crashes over a missing
    tool — it just reverts to the old, more optimistic budget.
    """
    total = total_ram_bytes()
    if sys.platform == "darwin":
        import subprocess

        try:
            page_size = int(
                subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True).strip()
            )
        except (subprocess.CalledProcessError, OSError, ValueError):
            page_size = 16384
        used = _darwin_used_bytes_host_statistics64(page_size)
        source = "host_statistics64"
        if used is None:
            source = "vm_stat"
            try:
                used = _darwin_used_bytes_vm_stat(page_size)
            except (subprocess.CalledProcessError, OSError):
                used = None
        if used is not None:
            return max(0, total - min(used, total)), source
    else:
        try:
            info = _parse_meminfo_text(Path("/proc/meminfo").read_text(encoding="utf-8"))
            if "MemAvailable" in info:
                return info["MemAvailable"], "meminfo"
        except (OSError, ValueError):
            pass
    return total, "total-fallback"


def find_model(path: Path | None, root: Path) -> Path | None:
    if path:
        p = path.expanduser()
        return p if p.is_file() else None
    for name in DEFAULT_MODEL_NAMES:
        cand = root / name
        if cand.is_file():
            return cand
    env = (os.environ.get("DS4_MODEL") or "").strip()
    if env:
        cand = Path(env).expanduser()
        if cand.is_file():
            return cand
    return None


def estimate_model_gib(model: Path | None, model_gib: float | None, factor: float) -> float:
    if model_gib is not None:
        return float(model_gib)
    if model is None:
        return 81.0  # Flash q2 ballpark when file is missing
    return (model.stat().st_size / GIB) * factor


def context_overhead_gib(ctx: int, sessions: int = 1) -> float:
    return KV_GIB_PER_MTOKEN * (max(0, ctx) / 1_000_000.0) * max(1, sessions)


def max_safe_ctx(
    *,
    ram_gib: float,
    model_gib: float,
    reserve_gib: float = DEFAULT_RESERVE_GIB,
    sessions: int = 1,
    hard_cap: int = MODEL_MAX_CTX,
) -> int:
    budget = ram_gib - reserve_gib - model_gib
    if budget <= 0:
        return 0
    per = KV_GIB_PER_MTOKEN * max(1, sessions)
    ctx = int((budget / per) * 1_000_000)
    if ctx < 0:
        return 0
    return min(hard_cap, ctx)


def planned_gib(model_gib: float, ctx: int, sessions: int = 1) -> float:
    return model_gib + context_overhead_gib(ctx, sessions=sessions)


def effective_budget_gib(
    *,
    free_gib: float,
    total_gib: float,
    reserve_gib: float,
    model_gib: float,
    conservative: bool = False,
    max_used_fraction: float = CONSERVATIVE_MAX_USED_FRACTION,
) -> tuple[float, str, bool]:
    """Pick the RAM line passed to max_safe_ctx().

    Normal policy budgets off free RAM. Conservative mode also caps planned
    model+KV at max_used_fraction of total RAM (planned <= available - reserve).
    If the cap cannot fit the model plus reserve, fall back to the free-RAM line.
    """
    if not conservative:
        return free_gib, "free", False

    cap_gib = total_gib * max_used_fraction
    capped = min(free_gib, cap_gib)
    min_for_model = model_gib + reserve_gib
    if capped + 1e-9 < min_for_model:
        pct = int(round(max_used_fraction * 100))
        return (
            free_gib,
            f"free ({pct}% cap relaxed — model needs {model_gib:.1f} GiB)",
            True,
        )
    pct = int(round(max_used_fraction * 100))
    return capped, f"min(free, {pct}% total)", False


def evaluate_ram_pressure(
    free_gib: float,
    latched: bool,
    *,
    trigger_gib: float = RAM_PRESSURE_TRIGGER_GIB,
    clear_gib: float = RAM_PRESSURE_CLEAR_GIB,
) -> str:
    """Latch helper for continuous free-RAM monitoring.

    Returns one of: trigger, hold, clear, idle.
    Trigger when free is at or below trigger_gib while not latched; stay held until
    free rises above clear_gib (hysteresis), then clear.
    """
    if free_gib <= trigger_gib:
        return "hold" if latched else "trigger"
    if latched and free_gib > clear_gib:
        return "clear"
    return "idle"


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None, help="GGUF path")
    parser.add_argument("--model-gib", type=float, default=None, help="Override resident model GiB")
    parser.add_argument(
        "--resident-factor",
        type=float,
        default=DEFAULT_RESIDENT_FACTOR,
        help="File-size → resident factor when --model-gib omitted",
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=DEFAULT_RESERVE_GIB,
        help="RAM to keep free on top of what's already in use (default 6)",
    )
    parser.add_argument(
        "--ignore-current-usage",
        action="store_true",
        help="Budget against total RAM instead of what's free right now",
    )
    parser.add_argument(
        "--conservative",
        action="store_true",
        help=(
            "Cap planned model+KV at "
            + str(int(CONSERVATIVE_MAX_USED_FRACTION * 100))
            + "%% of total RAM (when the model fits)"
        ),
    )
    parser.add_argument(
        "--max-used-fraction",
        type=float,
        default=None,
        help="Override the conservative cap fraction (implies --conservative)",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=1,
        help="Resident KV sessions (--batched-session N)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=None,
        help="If set, only check whether this ctx fits the safe budget",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument(
        "--print-ctx",
        action="store_true",
        help="Print only the recommended integer --ctx",
    )
    args = parser.parse_args(argv)

    conservative = bool(args.conservative or args.max_used_fraction is not None)
    max_used_fraction = (
        args.max_used_fraction
        if args.max_used_fraction is not None
        else CONSERVATIVE_MAX_USED_FRACTION
    )
    if max_used_fraction <= 0 or max_used_fraction > 1:
        parser.error("--max-used-fraction must be between 0 and 1")

    ram_bytes = total_ram_bytes()
    ram_gib = ram_bytes / GIB
    free_gib = ram_gib
    free_source = "total"
    if args.ignore_current_usage:
        available_gib = ram_gib
        available_source = "total (--ignore-current-usage)"
    else:
        available_bytes, free_source = available_ram_bytes()
        free_gib = available_bytes / GIB
        available_gib = free_gib
        available_source = free_source
    model = find_model(args.model, root)
    model_gib = estimate_model_gib(model, args.model_gib, args.resident_factor)
    conservative_relaxed = False
    if conservative and not args.ignore_current_usage:
        available_gib, cap_note, conservative_relaxed = effective_budget_gib(
            free_gib=free_gib,
            total_gib=ram_gib,
            reserve_gib=args.reserve_gib,
            model_gib=model_gib,
            conservative=True,
            max_used_fraction=max_used_fraction,
        )
        available_source = f"{available_source}, {cap_note}"
    elif conservative:
        cap_gib = ram_gib * max_used_fraction
        available_gib = min(available_gib, cap_gib)
        if cap_gib + 1e-9 < model_gib + args.reserve_gib:
            conservative_relaxed = True
        available_source = (
            f"{available_source}, min(total, {int(round(max_used_fraction * 100))}% total)"
        )
    safe = max_safe_ctx(
        ram_gib=available_gib,
        model_gib=model_gib,
        reserve_gib=args.reserve_gib,
        sessions=args.sessions,
    )
    # Round down to a practical multiple of 1024.
    safe_aligned = max(0, (safe // 1024) * 1024)
    ceiling = available_gib - args.reserve_gib

    payload = {
        "platform": platform.system(),
        "ram_gib": round(ram_gib, 2),
        "free_gib": round(free_gib, 2),
        "available_gib": round(available_gib, 2),
        "available_source": available_source,
        "conservative": conservative,
        "conservative_max_used_fraction": round(max_used_fraction, 4) if conservative else None,
        "conservative_relaxed": conservative_relaxed,
        "reserve_gib": args.reserve_gib,
        "ceiling_gib": round(ceiling, 2),
        "model_path": str(model) if model else None,
        "model_gib": round(model_gib, 2),
        "sessions": args.sessions,
        "kv_gib_per_mtoken": KV_GIB_PER_MTOKEN,
        "safe_ctx": safe_aligned,
        "safe_ctx_unaligned": safe,
        "planned_at_safe_ctx_gib": round(
            planned_gib(model_gib, safe_aligned, args.sessions), 2
        ),
    }
    if args.ctx is not None:
        planned = planned_gib(model_gib, args.ctx, args.sessions)
        payload["requested_ctx"] = args.ctx
        payload["planned_at_requested_ctx_gib"] = round(planned, 2)
        payload["fits"] = planned <= ceiling + 1e-9

    if args.print_ctx:
        print(safe_aligned)
        return 0 if safe_aligned > 0 else 1

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"RAM:     {payload['ram_gib']:.2f} GiB total")
        if not args.ignore_current_usage:
            print(f"Free:    {payload['free_gib']:.2f} GiB now  (source: {free_source})")
        print(
            f"Budget:  {payload['available_gib']:.2f} GiB"
            f"  ({payload['available_source']})"
        )
        if conservative:
            pct = int(round(max_used_fraction * 100))
            relaxed = " (cap relaxed)" if conservative_relaxed else ""
            print(f"Mode:    conservative — planned model+KV capped at ~{pct}% of total{relaxed}")
        print(f"Reserve: {payload['reserve_gib']:.2f} GiB  (ceiling {payload['ceiling_gib']:.2f} GiB)")
        print(
            f"Model:   {payload['model_gib']:.2f} GiB"
            + (f"  ({payload['model_path']})" if payload["model_path"] else "  (default estimate)")
        )
        print(
            f"Safe --ctx: {payload['safe_ctx']} tokens"
            f"  (planned ~{payload['planned_at_safe_ctx_gib']:.2f} GiB, "
            f"{args.sessions} session(s))"
        )
        if args.ctx is not None:
            flag = "OK" if payload["fits"] else "TOO HIGH"
            print(
                f"Requested --ctx {args.ctx}: planned ~{payload['planned_at_requested_ctx_gib']:.2f} GiB → {flag}"
            )
        print(
            "\nExample:\n"
            f"  ./ds4-server --metal --host 127.0.0.1 --port 8000 --ctx {payload['safe_ctx']} "
            "--kv-disk-dir ~/.ds4/server-kv --kv-disk-space-mb 8192"
        )
    if args.ctx is not None and not payload.get("fits", True):
        return 2
    return 0 if safe_aligned > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
