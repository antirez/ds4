"""Local text-to-speech for chat-ui (Piper preferred, macOS say fallback)."""

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
PREFERRED_SAY_VOICES_IT = ("Alice", "Luca")
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"
DEFAULT_PIPER_VOICE_IT = "it_IT-paola-medium"
PIPER_VOICES = {
    "en": DEFAULT_PIPER_VOICE,
    "it": DEFAULT_PIPER_VOICE_IT,
}


def normalize_tts_lang(value: str | None) -> str:
    raw = (value or "en").strip().lower().replace("_", "-")
    if raw in ("it", "it-it", "italian", "italiano"):
        return "it"
    return "en"

CHAT_UI_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TtsTooling:
    say: str | None
    afconvert: str | None
    piper: str | None
    piper_model: str | None
    piper_config: str | None = None
    piper_model_it: str | None = None
    piper_config_it: str | None = None

    @property
    def available(self) -> bool:
        if self.piper and (self.piper_model or self.piper_model_it):
            return True
        return bool(self.say and self.afconvert)

    @property
    def engine(self) -> str | None:
        if self.piper and (self.piper_model or self.piper_model_it):
            return "piper"
        if self.say and self.afconvert:
            return "say"
        return None

    def piper_for_lang(self, lang: str) -> tuple[str | None, str | None]:
        if lang == "it" and self.piper_model_it:
            return self.piper_model_it, self.piper_config_it
        if self.piper_model:
            return self.piper_model, self.piper_config
        if self.piper_model_it:
            return self.piper_model_it, self.piper_config_it
        return None, None


class TtsError(RuntimeError):
    """Raised when speech cannot be synthesized."""


def piper_root() -> Path:
    override = (os.environ.get("DS4_PIPER_ROOT") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ds4" / "piper"


def _first_existing_file(candidates: list[Path]) -> Path | None:
    for path in candidates:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def _resolve_piper_bin() -> str | None:
    env = (os.environ.get("DS4_PIPER_BIN") or "").strip()
    if env:
        path = Path(env).expanduser()
        if path.is_file():
            return str(path)
    which = shutil.which("piper")
    if which:
        return which
    found = _first_existing_file(
        [
            piper_root() / "venv" / "bin" / "piper",
            CHAT_UI_DIR / ".venv-piper" / "bin" / "piper",
        ]
    )
    return str(found) if found else None


def _piper_config_for(model: Path) -> Path | None:
    config = model.with_suffix(model.suffix + ".json")  # file.onnx.json
    if config.is_file():
        return config
    alt = Path(str(model) + ".json")
    return alt if alt.is_file() else None


def _resolve_piper_model_named(stem: str, env_key: str) -> tuple[str | None, str | None]:
    env = (os.environ.get(env_key) or "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    voices = piper_root() / "voices"
    candidates.append(voices / f"{stem}.onnx")
    if stem.startswith("it_") and voices.is_dir():
        candidates.extend(sorted(voices.glob("it_*.onnx")))
    elif stem.startswith("en_") and voices.is_dir():
        candidates.extend(sorted(voices.glob("en_*.onnx")))
    model = _first_existing_file(candidates)
    if not model:
        return None, None
    config = _piper_config_for(model)
    return str(model), str(config) if config else None


def _resolve_piper_model() -> tuple[str | None, str | None]:
    return _resolve_piper_model_named(DEFAULT_PIPER_VOICE, "DS4_PIPER_MODEL")


def discover_tools() -> TtsTooling:
    model, config = _resolve_piper_model()
    model_it, config_it = _resolve_piper_model_named(
        DEFAULT_PIPER_VOICE_IT, "DS4_PIPER_MODEL_IT"
    )
    return TtsTooling(
        say=shutil.which("say"),
        afconvert=shutil.which("afconvert"),
        piper=_resolve_piper_bin(),
        piper_model=model,
        piper_config=config,
        piper_model_it=model_it,
        piper_config_it=config_it,
    )


def tooling_status(tools: TtsTooling | None = None) -> dict[str, object]:
    t = tools or discover_tools()
    missing: list[str] = []
    if not t.available:
        if not t.piper and not (t.say and t.afconvert):
            missing.append("piper (run: python3 chat-ui/install_piper.py)")
        if t.piper and not t.piper_model and not t.piper_model_it:
            missing.append("piper voice model (.onnx)")
        if not t.say:
            missing.append("say (macOS fallback)")
        if t.say and not t.afconvert:
            missing.append("afconvert (macOS fallback)")
    return {
        "available": t.available,
        "engine": t.engine,
        "prefer": "piper",
        "say": t.say,
        "afconvert": t.afconvert,
        "piper": t.piper,
        "piper_model": t.piper_model,
        "piper_config": t.piper_config,
        "piper_model_it": t.piper_model_it,
        "piper_config_it": t.piper_config_it,
        "piper_root": str(piper_root()),
        "languages": {
            "en": {
                "label": "English",
                "piper": bool(t.piper and t.piper_model),
                "say": bool(t.say and t.afconvert),
            },
            "it": {
                "label": "Italiano",
                "piper": bool(t.piper and t.piper_model_it),
                "say": bool(t.say and t.afconvert),
            },
        },
        "missing": missing,
        "install_hint": (
            "Preferred: python3 chat-ui/install_piper.py "
            "(local neural Piper under ~/.ds4/piper). "
            "Fallback: macOS say + afconvert."
        ),
        "max_chars": MAX_TTS_CHARS,
    }


# Longer tokens first. Keep ASCII "<=" / ">=" as comparisons, not arrows.
_SPEAK_REPLACEMENTS_EN: tuple[tuple[str, str], ...] = (
    (r"<=>|<->|⇔|↔", " back and forth with "),
    (r"=>|->|→|⇒|⟶|➔|➜|➝|➞|➢|➤", " to "),
    (r"<-|←|⇐|⟵", " from "),
    (r"↑|⬆", " up "),
    (r"↓|⬇", " down "),
    (r">=|≥", " greater than or equal to "),
    (r"<=|≤", " less than or equal to "),
    (r"≠|!=", " not equal to "),
    (r"≈|~=", " approximately "),
    (r"±", " plus or minus "),
    (r"×|✕|✖", " times "),
    (r"÷", " divided by "),
    (r"∞", " infinity "),
    (r"°", " degrees "),
    (r"✓|✔|✅", " yes "),
    (r"✗|✘|❌", " no "),
    (r"…", ", "),
    (r"(?<=\w)\.\.\.(?=\s|$)", ", "),
    (r"[•▪▫◦▸►‣⁃]", ", "),
    (r"&", " and "),
    (r"©", " copyright "),
    (r"®", " registered "),
    (r"™", " trademark "),
)
_SPEAK_REPLACEMENTS_IT: tuple[tuple[str, str], ...] = (
    (r"<=>|<->|⇔|↔", " avanti e indietro con "),
    (r"=>|->|→|⇒|⟶|➔|➜|➝|➞|➢|➤", " a "),
    (r"<-|←|⇐|⟵", " da "),
    (r"↑|⬆", " su "),
    (r"↓|⬇", " giù "),
    (r">=|≥", " maggiore o uguale a "),
    (r"<=|≤", " minore o uguale a "),
    (r"≠|!=", " diverso da "),
    (r"≈|~=", " circa "),
    (r"±", " più o meno "),
    (r"×|✕|✖", " per "),
    (r"÷", " diviso "),
    (r"∞", " infinito "),
    (r"°", " gradi "),
    (r"✓|✔|✅", " sì "),
    (r"✗|✘|❌", " no "),
    (r"…", ", "),
    (r"(?<=\w)\.\.\.(?=\s|$)", ", "),
    (r"[•▪▫◦▸►‣⁃]", ", "),
    (r"&", " e "),
    (r"©", " copyright "),
    (r"®", " registrato "),
    (r"™", " marchio "),
)

_MINUS_TOKEN = " <<MINUS>> "


def _compile_replacements(
    rows: tuple[tuple[str, str], ...],
) -> tuple[tuple[re.Pattern[str], str], ...]:
    return tuple((re.compile(pat), repl) for pat, repl in rows)


_COMPILED_EN = _compile_replacements(_SPEAK_REPLACEMENTS_EN)
_COMPILED_IT = _compile_replacements(_SPEAK_REPLACEMENTS_IT)


def _compiled_replacements(lang: str) -> tuple[tuple[re.Pattern[str], str], ...]:
    return _COMPILED_IT if lang == "it" else _COMPILED_EN


def _rewrite_dashes(text: str, lang: str = "en") -> str:
    """Remove dash/hyphen characters so TTS does not say 'dash' / 'trattino'."""
    out = text
    range_word = " a " if lang == "it" else " to "
    minus_word = " meno " if lang == "it" else " minus "
    # Protect numeric negatives before stripping hyphens: -3 -> minus/meno 3
    out = re.sub(r"(?<![\w.])-(?=\d)", _MINUS_TOKEN, out)
    # Unicode dashes and long ASCII rules -> pause
    out = re.sub(r"[—–―‒⁓]+", ", ", out)
    out = re.sub(r"-{2,}", ", ", out)
    # Spaced hyphen separators
    out = re.sub(r"\s+-\s+", ", ", out)
    # Markdown / list markers at line start
    out = re.sub(r"(?m)^[ \t]*-\s+", "", out)
    # Numeric ranges: 3-5 -> 3 to/a 5
    out = re.sub(r"(?<=\d)\s*-\s*(?=\d)", range_word, out)
    # Compound words / titles, including accented Italian letters
    out = re.sub(r"(?<=[A-Za-z0-9À-ÖØ-öø-ÿ])-(?=[A-Za-z0-9À-ÖØ-öø-ÿ])", " ", out)
    # Any leftover hyphen-minus
    out = out.replace("-", ", ")
    out = out.replace(_MINUS_TOKEN, minus_word)
    return out


def speak_friendly(text: str, lang: str = "en") -> str:
    """Rewrite symbols into short phrases that sound natural when spoken."""
    out = text
    for pattern, repl in _compiled_replacements(lang):
        out = pattern.sub(repl, out)
    out = _rewrite_dashes(out, lang=lang)
    # Collapse whitespace introduced by replacements; keep paragraph breaks.
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"(,\s*){2,}", ", ", out)
    return out.strip()


def prepare_text(text: str, lang: str | None = None) -> str:
    """Keep spoken output short and skip heavy code dumps."""
    if not isinstance(text, str):
        raise TtsError("text must be a string")
    chosen_lang = normalize_tts_lang(lang)
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        raise TtsError("text is empty")
    # Drop fenced code blocks; they read poorly aloud.
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = speak_friendly(cleaned, lang=chosen_lang)
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


def _say_voice_names(tools: TtsTooling) -> set[str]:
    if not tools.say:
        return set()
    try:
        proc = subprocess.run(
            [tools.say, "-v", "?"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    listing = proc.stdout or proc.stderr or ""
    return {line.split()[0] for line in listing.splitlines() if line.strip()}


def _pick_say_voice(
    tools: TtsTooling, requested: str | None, lang: str = "en"
) -> str | None:
    if requested and requested.strip():
        return requested.strip()
    names = _say_voice_names(tools)
    preferred = PREFERRED_SAY_VOICES_IT if lang == "it" else PREFERRED_SAY_VOICES
    for voice in preferred:
        if voice in names:
            return voice
    if lang == "it":
        for voice in PREFERRED_SAY_VOICES:
            if voice in names:
                return voice
    return None


def _synthesize_say(
    text: str, tools: TtsTooling, voice: str | None, lang: str = "en"
) -> bytes:
    if not tools.say or not tools.afconvert:
        raise TtsError("macOS say/afconvert not available")
    chosen = _pick_say_voice(tools, voice, lang=lang)
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


def _synthesize_piper(text: str, tools: TtsTooling, lang: str = "en") -> bytes:
    if not tools.piper:
        raise TtsError("piper not configured")
    model, config = tools.piper_for_lang(lang)
    if not model:
        raise TtsError("piper not configured")
    with tempfile.TemporaryDirectory(prefix="ds4-tts-") as tmp:
        wav_path = Path(tmp) / "speech.wav"
        cmd = [
            tools.piper,
            "--model",
            model,
            "--output_file",
            str(wav_path),
        ]
        if config:
            cmd.extend(["--config", config])
        _run(cmd, timeout=PIPER_TIMEOUT_SEC, stdin_text=text)
        data = wav_path.read_bytes()
    if len(data) < 44:
        raise TtsError("piper produced empty audio")
    return data


def synthesize_wav(
    text: str,
    *,
    voice: str | None = None,
    lang: str | None = None,
    tools: TtsTooling | None = None,
) -> tuple[bytes, str]:
    """Return (wav_bytes, engine_name). Prefers Piper when installed."""
    t = tools or discover_tools()
    chosen_lang = normalize_tts_lang(lang or voice)
    spoken = prepare_text(text, lang=chosen_lang)
    model, _config = t.piper_for_lang(chosen_lang)
    if t.piper and model:
        return _synthesize_piper(spoken, t, chosen_lang), "piper"
    if t.say and t.afconvert:
        return _synthesize_say(spoken, t, voice, chosen_lang), "say"
    raise TtsError(
        "no local TTS engine available; run: python3 chat-ui/install_piper.py "
        "(or use macOS say + afconvert)"
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
