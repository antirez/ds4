#!/usr/bin/env python3
"""Run a small steering scale sweep through ds4.

This is intentionally thin: it exercises the same public CLI options users
will use in production and leaves all inference behavior inside ds4.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def read_prompts(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    if not prompts:
        raise SystemExit(f"{path}: no prompts found")
    return prompts


def normalize_argv(argv: list[str]) -> list[str]:
    normalized = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--scales" and index + 1 < len(argv):
            value = argv[index + 1]
            if value.startswith("-") and ("," in value or value.replace(".", "", 1).isdigit()):
                normalized.append(f"--scales={value}")
                index += 2
                continue
        normalized.append(token)
        index += 1
    return normalized


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds4", default="./ds4")
    ap.add_argument("--model", default="ds4flash.gguf")
    ap.add_argument("--direction", required=True,
                    help="flat f32 vector file produced by build_direction.py")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--scales", default="-2,-1,-0.5,0,0.5,1,2")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--nothink", action="store_true")
    args = ap.parse_args(normalize_argv(sys.argv[1:]))

    prompts = read_prompts(Path(args.prompts))
    scales = [float(x) for x in args.scales.split(",") if x.strip()]

    for prompt in prompts:
        print("=" * 80)
        print(f"PROMPT: {prompt}")
        for scale in scales:
            print("-" * 80)
            print(f"FFN scale: {scale:g}")
            cmd = [
                args.ds4,
                "-m", args.model,
                "--ctx", str(args.ctx),
                "-n", str(args.tokens),
                "--temp", "0",
                "--expert-steering-file", args.direction,
                "--expert-steering-scale", str(scale),
                "-p", prompt,
            ]
            if args.nothink:
                cmd.append("--nothink")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
