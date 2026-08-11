#!/usr/bin/env python3
"""Recommend a Mac-safe ds4-server --ctx given RAM headroom.

Policy: planned resident model + context working set should stay at or under
  total_RAM - reserve_GiB
(default reserve 6 GiB for macOS, chat-ui, browser, and other apps).

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
        help="RAM to keep free (default 6)",
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

    ram_bytes = total_ram_bytes()
    ram_gib = ram_bytes / GIB
    model = find_model(args.model, root)
    model_gib = estimate_model_gib(model, args.model_gib, args.resident_factor)
    safe = max_safe_ctx(
        ram_gib=ram_gib,
        model_gib=model_gib,
        reserve_gib=args.reserve_gib,
        sessions=args.sessions,
    )
    # Round down to a practical multiple of 1024.
    safe_aligned = max(0, (safe // 1024) * 1024)
    ceiling = ram_gib - args.reserve_gib

    payload = {
        "platform": platform.system(),
        "ram_gib": round(ram_gib, 2),
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
        print(f"RAM:     {payload['ram_gib']:.2f} GiB")
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
