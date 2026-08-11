#!/usr/bin/env python3
"""Install local Piper TTS (venv + default English voice) under ~/.ds4/piper."""

from __future__ import annotations

import argparse
import os
import subprocess
import urllib.request
import venv
from pathlib import Path

DEFAULT_VOICE = "en_US-lessac-medium"
VOICE_ONNX_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
    f"en/en_US/lessac/medium/{DEFAULT_VOICE}.onnx"
)
VOICE_JSON_URL = VOICE_ONNX_URL + ".json"


def default_root() -> Path:
    override = (os.environ.get("DS4_PIPER_ROOT") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ds4" / "piper"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_venv(venv_dir: Path) -> Path:
    piper_bin = venv_dir / "bin" / "piper"
    if piper_bin.is_file():
        print(f"venv already present: {venv_dir}", flush=True)
        return piper_bin
    print(f"creating venv: {venv_dir}", flush=True)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    pip = venv_dir / "bin" / "pip"
    run([str(pip), "install", "-U", "pip"])
    run([str(pip), "install", "piper-tts>=1.4.0"])
    if not piper_bin.is_file():
        raise SystemExit(f"piper binary missing after install: {piper_bin}")
    return piper_bin


def download(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 1000:
        print(f"already have {dest.name} ({dest.stat().st_size} bytes)", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"wrote {dest} ({dest.stat().st_size} bytes)", flush=True)


def smoke_test(piper_bin: Path, model: Path) -> None:
    out = Path("/tmp/ds4-piper-install-smoke.wav")
    cmd = [
        str(piper_bin),
        "--model",
        str(model),
        "--output_file",
        str(out),
    ]
    print("+", "echo … |", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        input="DwarfStar Piper install smoke test.\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise SystemExit(f"piper smoke failed ({proc.returncode}): {err}")
    if not out.is_file() or out.stat().st_size < 44:
        raise SystemExit("piper smoke produced empty wav")
    print(f"smoke ok: {out} ({out.stat().st_size} bytes)", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Install root (default: ~/.ds4/piper or DS4_PIPER_ROOT)",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip synthesizing a short sample after install",
    )
    args = parser.parse_args(argv)
    root = (args.root or default_root()).expanduser()
    venv_dir = root / "venv"
    voices = root / "voices"
    model = voices / f"{DEFAULT_VOICE}.onnx"
    config = voices / f"{DEFAULT_VOICE}.onnx.json"

    piper_bin = ensure_venv(venv_dir)
    download(VOICE_ONNX_URL, model)
    download(VOICE_JSON_URL, config)
    if not args.skip_smoke:
        smoke_test(piper_bin, model)

    print(
        "\nPiper ready.\n"
        f"  binary: {piper_bin}\n"
        f"  model:  {model}\n"
        "chat-ui auto-prefers this when present (override with DS4_PIPER_BIN / "
        "DS4_PIPER_MODEL). Restart chat-ui after install.\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
