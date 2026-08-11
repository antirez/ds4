"""Local text-to-speech for chat-ui (macOS say, optional Piper)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

MAX_TTS_CHARS = 4_000
SAY_TIMEOUT_SEC = 120
PIPER_TIMEOUT_SEC = 180

# Prefer a natural English voice when present; otherwise the system default.
PREFERRED_SAY_VOICES = ("Samantha", "Daniel", "Karen", "Moira", "Alex", "Fred")


@dataclass(frozen=True)
class TtsTooling:
    say: str | None
    afconvert: str | None
    piper: str | None
    piper_model: str | None

    @property
    def available(self) -> bool:
        if self.piper and self.piper_model:
            return True
        return bool(self.say and self.afconvert)

    @property
    def engine(self) -> str | None:
        if self.piper and self.piper_model:
            return "piper"
        if self.say and self.afconvert:
            return "say"
        return None


class TtsError(RuntimeError):
    """Raised when speech cannot be synthesized."""


def discover_tools() -> TtsTooling:
    piper = shutil.which("piper")
    model = (os.environ.get("DS4_PIPER_MODEL") or "").strip() or None
    if model and not Path(model).is_file():
        model = None
    return TtsTooling(
        say=shutil.which("say"),
        afconvert=shutil.which("afconvert"),
        piper=piper,
        piper_model=model,
    )


def tooling_status(tools: TtsTooling | None = None) -> dict[str, object]:
    t = tools or discover_tools()
    missing: list[str] = []
    if not t.available:
        if not t.say:
            missing.append("say (macOS)")
        if t.say and not t.afconvert:
            missing.append("afconvert (macOS)")
        if t.piper and not t.piper_model:
            missing.append("DS4_PIPER_MODEL (path to .onnx)")
    return {
        "available": t.available,
        "engine": t.engine,
        "say": t.say,
        "afconvert": t.afconvert,
        "piper": t.piper,
        "piper_model": t.piper_model,
        "missing": missing,
        "install_hint": (
            "Built-in on macOS (say + afconvert). Optional neural: install Piper "
            "and set DS4_PIPER_MODEL=/path/to/voice.onnx"
        ),
        "max_chars": MAX_TTS_CHARS,
    }


def prepare_text(text: str) -> str:
    """Keep spoken output short and skip heavy code dumps."""
    if not isinstance(text, str):
        raise TtsError("text must be a string")
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        raise TtsError("text is empty")
    # Drop fenced code blocks; they read poorly aloud.
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    if not cleaned:
        raise TtsError("nothing left to speak after cleanup")
    if len(cleaned) > MAX_TTS_CHARS:
        cleaned = cleaned[: MAX_TTS_CHARS - 1].rstrip() + "…"
    return cleaned


def _run(cmd: list[str], timeout: int, stdin_text: str | None = None) -> None:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=stdin_text,
        )
    except FileNotFoundError as exc:
        raise TtsError(f"missing tool: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TtsError(f"tts timed out: {cmd[0]}") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise TtsError(err or f"{cmd[0]} failed ({proc.returncode})")


def _pick_say_voice(tools: TtsTooling, requested: str | None) -> str | None:
    if requested and requested.strip():
        return requested.strip()
    if not tools.say:
        return None
    try:
        proc = subprocess.run(
            [tools.say, "-v", "?"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    listing = proc.stdout or proc.stderr or ""
    names = {line.split()[0] for line in listing.splitlines() if line.strip()}
    for voice in PREFERRED_SAY_VOICES:
        if voice in names:
            return voice
    return None


def _synthesize_say(text: str, tools: TtsTooling, voice: str | None) -> bytes:
    if not tools.say or not tools.afconvert:
        raise TtsError("macOS say/afconvert not available")
    chosen = _pick_say_voice(tools, voice)
    with tempfile.TemporaryDirectory(prefix="ds4-tts-") as tmp:
        root = Path(tmp)
        aiff_path = root / "speech.aiff"
        wav_path = root / "speech.wav"
        cmd = [tools.say, "-o", str(aiff_path)]
        if chosen:
            cmd.extend(["-v", chosen])
        cmd.append(text)
        _run(cmd, timeout=SAY_TIMEOUT_SEC)
        _run(
            [tools.afconvert, "-f", "WAVE", "-d", "LEI16", str(aiff_path), str(wav_path)],
            timeout=60,
        )
        data = wav_path.read_bytes()
    if len(data) < 44:
        raise TtsError("say produced empty audio")
    return data


def _synthesize_piper(text: str, tools: TtsTooling) -> bytes:
    if not tools.piper or not tools.piper_model:
        raise TtsError("piper not configured")
    with tempfile.TemporaryDirectory(prefix="ds4-tts-") as tmp:
        wav_path = Path(tmp) / "speech.wav"
        cmd = [
            tools.piper,
            "--model",
            tools.piper_model,
            "--output_file",
            str(wav_path),
        ]
        _run(cmd, timeout=PIPER_TIMEOUT_SEC, stdin_text=text)
        data = wav_path.read_bytes()
    if len(data) < 44:
        raise TtsError("piper produced empty audio")
    return data


def synthesize_wav(
    text: str,
    *,
    voice: str | None = None,
    tools: TtsTooling | None = None,
) -> tuple[bytes, str]:
    """Return (wav_bytes, engine_name)."""
    t = tools or discover_tools()
    spoken = prepare_text(text)
    if t.piper and t.piper_model:
        return _synthesize_piper(spoken, t), "piper"
    if t.say and t.afconvert:
        return _synthesize_say(spoken, t, voice), "say"
    raise TtsError(
        "no local TTS engine available; need macOS say+afconvert "
        "or Piper with DS4_PIPER_MODEL"
    )


if __name__ == "__main__":
    import sys

    sample = " ".join(sys.argv[1:]) or "DwarfStar local text to speech is working."
    status = tooling_status()
    print("tts status:", status)
    wav, engine = synthesize_wav(sample)
    out = Path.cwd() / "tts-smoke.wav"
    out.write_bytes(wav)
    print(f"engine={engine} bytes={len(wav)} wrote={out}")
