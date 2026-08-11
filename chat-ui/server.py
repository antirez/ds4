#!/usr/bin/env python3
"""Local DwarfStar chat UI: static files, chat JSON store, OpenAI API proxy."""

from __future__ import annotations

import argparse
import base64
import json
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# Allow `python3 chat-ui/server.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr import OcrError, extract_attachment, tooling_status  # noqa: E402
from tts import TtsError, synthesize_wav, tooling_status as tts_tooling_status  # noqa: E402
from web_context import build_web_context, derive_search_query  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
try:
    import safe_ctx as _safe_ctx  # noqa: E402
except ImportError:  # pragma: no cover - optional helper
    _safe_ctx = None  # type: ignore

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_API = "http://127.0.0.1:8000"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
MAX_BODY = 32 * 1024 * 1024
CHAT_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
REPO_ROOT = Path(__file__).resolve().parents[1]


def ram_policy_status() -> dict[str, Any]:
    """Mac-safe ctx policy: model + KV <= RAM - 6 GiB."""
    if _safe_ctx is None:
        return {"available": False, "error": "safe_ctx helper missing"}
    try:
        ram_gib = _safe_ctx.total_ram_bytes() / _safe_ctx.GIB
        model = _safe_ctx.find_model(None, REPO_ROOT)
        model_gib = _safe_ctx.estimate_model_gib(model, None, _safe_ctx.DEFAULT_RESIDENT_FACTOR)
        reserve = _safe_ctx.DEFAULT_RESERVE_GIB
        safe = _safe_ctx.max_safe_ctx(ram_gib=ram_gib, model_gib=model_gib, reserve_gib=reserve)
        safe_aligned = max(0, (safe // 1024) * 1024)
        return {
            "available": True,
            "reserve_gib": reserve,
            "ram_gib": round(ram_gib, 2),
            "ceiling_gib": round(ram_gib - reserve, 2),
            "model_gib": round(model_gib, 2),
            "model_path": str(model) if model else None,
            "safe_ctx": safe_aligned,
            "kv_gib_per_mtoken": _safe_ctx.KV_GIB_PER_MTOKEN,
            "policy": "planned_model_plus_ctx <= RAM - 6GiB",
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


def _read_proc_meminfo() -> dict[str, int]:
    """Parse /proc/meminfo into byte counts keyed by field name."""
    info: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        parts = val.split()
        if parts:
            info[key] = int(parts[0]) * 1024
    return info


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


def _activity_monitor_used_pages(stats: dict[str, int]) -> int | None:
    """Activity Monitor Memory Used in pages: app + wired + compressed.

    App Memory ≈ internal (anonymous) pages minus purgeable pages.
    """
    internal = stats.get("Pages internal")
    if internal is None:
        internal = stats.get("Anonymous pages")
    wired = stats.get("Pages wired down")
    compressor = stats.get("Pages occupied by compressor")
    if internal is None or wired is None or compressor is None:
        return None
    purgeable = stats.get("Pages purgeable", 0)
    app_pages = max(0, int(internal) - int(purgeable))
    return app_pages + int(wired) + int(compressor)


def _darwin_memory_used_host_statistics64(page_size: int) -> int | None:
    """Memory Used via mach host_statistics64 (HOST_VM_INFO64)."""
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


def _darwin_memory_used_bytes(page_size: int) -> tuple[int | None, str]:
    """Return (used_bytes, source) matching Activity Monitor Memory Used."""
    used = _darwin_memory_used_host_statistics64(page_size)
    if used is not None:
        return used, "host_statistics64"
    raw = subprocess.check_output(["vm_stat"], text=True)
    pages_used = _activity_monitor_used_pages(_parse_vm_stat_pages(raw))
    if pages_used is None:
        return None, "unknown"
    return pages_used * int(page_size), "vm_stat_activity_monitor"


def system_ram_status() -> dict[str, Any]:
    """Live occupied / total RAM for the UI status bar."""
    try:
        if _safe_ctx is not None:
            total = int(_safe_ctx.total_ram_bytes())
        elif platform.system() == "Darwin":
            total = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            )
        else:
            meminfo = _read_proc_meminfo()
            if "MemTotal" in meminfo:
                total = meminfo["MemTotal"]
            else:
                import os

                pages = int(os.sysconf("SC_PHYS_PAGES"))
                page = int(os.sysconf("SC_PAGE_SIZE"))
                total = pages * page
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"total ram: {exc}"}

    used = None
    source = "unknown"
    try:
        if platform.system() == "Darwin":
            page_size = int(
                subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True).strip()
            )
            used, source = _darwin_memory_used_bytes(page_size)
        else:
            meminfo = _read_proc_meminfo()
            if "MemTotal" in meminfo and "MemAvailable" in meminfo:
                total = meminfo["MemTotal"]
                used = total - meminfo["MemAvailable"]
                source = "meminfo"
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "total_bytes": total,
            "error": str(exc),
        }

    if used is None:
        return {"available": False, "total_bytes": total, "error": "unsupported platform"}

    used = max(0, min(int(used), int(total)))
    free = int(total) - used
    pct = (100.0 * used / total) if total else 0.0
    gib = 1024**3
    # free_bytes = total - used (macOS used ≈ Activity Monitor Memory Used).
    # Chat-ui client treats free < safe_ctx reserve (6 GiB) as pressure.
    return {
        "available": True,
        "total_bytes": int(total),
        "used_bytes": used,
        "free_bytes": free,
        "used_gib": round(used / gib, 2),
        "free_gib": round(free / gib, 2),
        "total_gib": round(total / gib, 2),
        "percent": round(pct, 1),
        "source": source,
        "pressure_trigger_gib": (
            float(_safe_ctx.RAM_PRESSURE_TRIGGER_GIB) if _safe_ctx else 6.0
        ),
        "pressure_clear_gib": (
            float(_safe_ctx.RAM_PRESSURE_CLEAR_GIB) if _safe_ctx else 7.0
        ),
    }


def format_upstream_error(exc: BaseException) -> str:
    """Prefer the upstream JSON/text body over urllib's opaque reason string."""
    if isinstance(exc, HTTPError):
        raw = b""
        try:
            raw = exc.read() or b""
        except Exception:
            raw = b""
        text = raw.decode("utf-8", errors="replace").strip()
        if text:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return text[:2000]
            err = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(err, dict):
                msg = err.get("message") or err.get("code") or text
                code = err.get("code")
                if code and code not in str(msg):
                    return f"{msg} ({code})"
                return str(msg)
            if isinstance(err, str) and err.strip():
                return err.strip()
            return text[:2000]
        reason = getattr(exc, "reason", None) or str(exc)
        return f"HTTP {exc.code}: {reason}"
    return str(exc)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_chats_dir() -> Path:
    return Path.home() / ".ds4" / "chats"


def ensure_chats_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def chat_path(chats_dir: Path, chat_id: str) -> Path:
    if not CHAT_ID_RE.match(chat_id):
        raise ValueError("invalid chat id")
    return chats_dir / f"{chat_id}.json"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    tmp.replace(path)


def estimate_tokens(text: str | None) -> int:
    """Rough local estimate; matches chat-ui/static/app.js estimateTokens (chars/4)."""
    return (len(text or "") + 3) // 4


def history_token_estimate(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
    return total


def format_size_label(size_bytes: int) -> str:
    n = max(0, int(size_bytes))
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        kb = n / 1024
        return f"{kb:.0f} KB" if kb >= 10 else f"{kb:.1f} KB"
    mb = n / (1024 * 1024)
    return f"{mb:.0f} MB" if mb >= 10 else f"{mb:.1f} MB"


def list_chats(chats_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in chats_dir.glob("*.json"):
        try:
            data = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = 0
        messages = data.get("messages") or []
        items.append(
            {
                "id": data.get("id", path.stem),
                "title": data.get("title") or "Untitled",
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "message_count": len(messages) if isinstance(messages, list) else 0,
                "token_estimate": history_token_estimate(messages),
                "size_bytes": size_bytes,
                "size_label": format_size_label(size_bytes),
            }
        )
    items.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
    return items


def new_chat(title: str | None = None) -> dict[str, Any]:
    now = utc_now()
    return {
        "id": str(uuid.uuid4()),
        "title": (title or "New chat").strip() or "New chat",
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "attachments_note": (
            "Text and code files are inlined into the user message. "
            "Images and PDFs are OCR'd locally to text before send; "
            "ds4-server itself is text-only."
        ),
    }


def validate_chat(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("chat must be an object")
    chat_id = data.get("id")
    if not isinstance(chat_id, str) or not CHAT_ID_RE.match(chat_id):
        raise ValueError("chat.id missing or invalid")
    title = data.get("title")
    if title is not None and not isinstance(title, str):
        raise ValueError("chat.title must be a string")
    messages = data.get("messages")
    if messages is None:
        messages = []
    if not isinstance(messages, list):
        raise ValueError("chat.messages must be a list")
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("each message must be an object")
        if msg.get("role") not in ("system", "user", "assistant"):
            raise ValueError("message.role must be system, user, or assistant")
        if not isinstance(msg.get("content"), str):
            raise ValueError("message.content must be a string")
    out = dict(data)
    out["messages"] = messages
    out["title"] = (title or "Untitled").strip() or "Untitled"
    out["updated_at"] = utc_now()
    if not out.get("created_at"):
        out["created_at"] = out["updated_at"]
    return out


def mime_for(path: Path) -> str:
    return {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".svg": "image/svg+xml",
        ".json": "application/json; charset=utf-8",
        ".ico": "image/x-icon",
        ".woff2": "font/woff2",
    }.get(path.suffix.lower(), "application/octet-stream")


def proxy_request(
    api_base: str,
    method: str,
    path_qs: str,
    body: bytes | None,
    headers: dict[str, str],
) -> tuple[int, list[tuple[str, str]], bytes]:
    parsed = urlparse(api_base)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("api base must be http or https")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    target = path_qs if path_qs.startswith("/") else f"/{path_qs}"
    conn_cls = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
    conn = conn_cls(host, port, timeout=600)
    try:
        fwd = {
            "Host": f"{host}:{port}" if parsed.port else host,
            "Accept": headers.get("Accept", "*/*"),
            "Content-Type": headers.get("Content-Type", "application/json"),
            "Connection": "close",
        }
        if body is not None:
            fwd["Content-Length"] = str(len(body))
        conn.request(method, target, body=body, headers=fwd)
        resp = conn.getresponse()
        raw = resp.read()
        out_headers = [
            (k, v)
            for k, v in resp.getheaders()
            if k.lower()
            not in {
                "transfer-encoding",
                "connection",
                "content-length",
                "content-encoding",
            }
        ]
        return resp.status, out_headers, raw
    finally:
        conn.close()


class ChatUIHandler(BaseHTTPRequestHandler):
    server_version = "ds4-chat-ui/1.0"

    @property
    def chats_dir(self) -> Path:
        return self.server.chats_dir  # type: ignore[attr-defined]

    @property
    def api_base(self) -> str:
        return self.server.api_base  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or "0")
        if length < 0 or length > MAX_BODY:
            raise ValueError("body too large")
        return self.rfile.read(length) if length else b""

    def _send(self, status: int, body: bytes, content_type: str, extra: list[tuple[str, str]] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra:
            for key, value in extra:
                self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET(head_only=True)

    def do_GET(self, head_only: bool = False) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "api_base": self.api_base,
                    "chats_dir": str(self.chats_dir),
                    "ocr": tooling_status(),
                    "tts": tts_tooling_status(),
                    "web": {
                        "enabled": True,
                        "provider": "duckduckgo-html",
                        "api_key_required": False,
                    },
                    "ram_policy": ram_policy_status(),
                    "system_ram": system_ram_status(),
                },
            )
            return

        if path == "/api/ram":
            self._send_json(HTTPStatus.OK, system_ram_status())
            return

        if path == "/api/chats":
            self._send_json(HTTPStatus.OK, {"chats": list_chats(self.chats_dir)})
            return

        if path.startswith("/api/chats/"):
            chat_id = path[len("/api/chats/") :]
            try:
                path_on_disk = chat_path(self.chats_dir, chat_id)
            except ValueError:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "invalid chat id")
                return
            if not path_on_disk.is_file():
                self._send_error_json(HTTPStatus.NOT_FOUND, "chat not found")
                return
            try:
                self._send_json(HTTPStatus.OK, read_json(path_on_disk))
            except (OSError, json.JSONDecodeError) as exc:
                self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return

        if path.startswith("/v1/"):
            self._proxy(parsed.path + (("?" + parsed.query) if parsed.query else ""))
            return

        self._serve_static(path, head_only=head_only)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/chats":
            try:
                raw = self._read_body()
                payload = json.loads(raw.decode("utf-8") or "{}")
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            title = payload.get("title") if isinstance(payload, dict) else None
            chat = new_chat(title if isinstance(title, str) else None)
            write_json(chat_path(self.chats_dir, chat["id"]), chat)
            self._send_json(HTTPStatus.CREATED, chat)
            return

        if path == "/api/ocr":
            self._handle_ocr()
            return

        if path == "/api/web-context":
            self._handle_web_context()
            return

        if path == "/api/tts":
            self._handle_tts()
            return

        if path.startswith("/v1/"):
            self._proxy(parsed.path + (("?" + parsed.query) if parsed.query else ""))
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "not found")

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if not path.startswith("/api/chats/"):
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return
        chat_id = path[len("/api/chats/") :]
        try:
            path_on_disk = chat_path(self.chats_dir, chat_id)
            raw = self._read_body()
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, dict):
                payload["id"] = chat_id
            chat = validate_chat(payload)
            write_json(path_on_disk, chat)
            self._send_json(HTTPStatus.OK, chat)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if not path.startswith("/api/chats/"):
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return
        chat_id = path[len("/api/chats/") :]
        try:
            path_on_disk = chat_path(self.chats_dir, chat_id)
        except ValueError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "invalid chat id")
            return
        if path_on_disk.is_file():
            path_on_disk.unlink()
        self._send_json(HTTPStatus.OK, {"deleted": chat_id})

    def _serve_static(self, path: str, head_only: bool = False) -> None:
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        candidate = (STATIC_DIR / rel).resolve()
        try:
            candidate.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._send_error_json(HTTPStatus.FORBIDDEN, "forbidden")
            return
        if not candidate.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return
        data = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_for(candidate))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(data)

    def _handle_ocr(self) -> None:
        try:
            raw = self._read_body()
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if not isinstance(payload, dict):
            self._send_error_json(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
            return
        filename = payload.get("filename")
        data_b64 = payload.get("data_base64")
        if not isinstance(filename, str) or not filename.strip():
            self._send_error_json(HTTPStatus.BAD_REQUEST, "filename required")
            return
        if not isinstance(data_b64, str) or not data_b64.strip():
            self._send_error_json(HTTPStatus.BAD_REQUEST, "data_base64 required")
            return
        try:
            data = base64.b64decode(data_b64, validate=False)
        except Exception as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"invalid base64: {exc}")
            return
        try:
            result = extract_attachment(filename.strip(), data)
            self._send_json(HTTPStatus.OK, result)
        except OcrError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except OSError as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _handle_tts(self) -> None:
        try:
            raw = self._read_body()
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if not isinstance(payload, dict):
            self._send_error_json(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
            return
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            self._send_error_json(HTTPStatus.BAD_REQUEST, "text required")
            return
        voice = payload.get("voice")
        if voice is not None and not isinstance(voice, str):
            self._send_error_json(HTTPStatus.BAD_REQUEST, "voice must be a string")
            return
        try:
            wav, engine = synthesize_wav(text, voice=voice)
            self._send(
                HTTPStatus.OK,
                wav,
                "audio/wav",
                extra=[("X-TTS-Engine", engine)],
            )
        except TtsError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except OSError as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _handle_web_context(self) -> None:
        try:
            raw = self._read_body()
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if not isinstance(payload, dict):
            self._send_error_json(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
            return
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            self._send_error_json(HTTPStatus.BAD_REQUEST, "query required")
            return
        messages = payload.get("messages")
        recent: list[dict[str, str]] | None = None
        if messages is not None:
            if not isinstance(messages, list):
                self._send_error_json(HTTPStatus.BAD_REQUEST, "messages must be a list")
                return
            recent = []
            for item in messages[-12:]:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                content = item.get("content")
                if role not in ("user", "assistant") or not isinstance(content, str):
                    continue
                text = content.strip()
                if not text:
                    continue
                recent.append({"role": role, "content": text[:2000]})
        max_results = payload.get("max_results", 8)
        max_fetch = payload.get("max_fetch", 5)
        fetch_pages = payload.get("fetch_pages", True)
        if not isinstance(max_results, int) or not (1 <= max_results <= 12):
            self._send_error_json(HTTPStatus.BAD_REQUEST, "max_results must be 1..12")
            return
        if not isinstance(max_fetch, int) or not (0 <= max_fetch <= 8):
            self._send_error_json(HTTPStatus.BAD_REQUEST, "max_fetch must be 0..8")
            return
        if not isinstance(fetch_pages, bool):
            self._send_error_json(HTTPStatus.BAD_REQUEST, "fetch_pages must be a boolean")
            return
        model = payload.get("model")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            self._send_error_json(HTTPStatus.BAD_REQUEST, "model must be a non-empty string")
            return
        model_id = model.strip() if isinstance(model, str) else None
        search_query = derive_search_query(
            query.strip(),
            recent,
            api_base=self.api_base,
            model=model_id,
        )
        if not search_query:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "query required")
            return
        try:
            result = build_web_context(
                search_query,
                max_results=max_results,
                max_fetch=max_fetch,
                fetch_pages=fetch_pages,
            )
            self._send_json(HTTPStatus.OK, result)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except RuntimeError as exc:
            self._send_error_json(HTTPStatus.BAD_GATEWAY, str(exc))
        except (OSError, URLError, HTTPError) as exc:
            self._send_error_json(HTTPStatus.BAD_GATEWAY, f"web context failed: {exc}")

    def _proxy(self, path_qs: str) -> None:
        try:
            body = self._read_body() if self.command in ("POST", "PUT", "PATCH") else None
            headers = {k: v for k, v in self.headers.items()}
            compact = body.replace(b" ", b"").lower() if body else b""
            if body and b'"stream":true' in compact:
                self._proxy_stream(path_qs, body, headers)
                return
            status, out_headers, raw = proxy_request(
                self.api_base, self.command, path_qs, body, headers
            )
            self.send_response(status)
            ctype = "application/json; charset=utf-8"
            for key, value in out_headers:
                if key.lower() == "content-type":
                    ctype = value
            if status >= 400 and "json" not in ctype.lower():
                ctype = "application/json; charset=utf-8"
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (OSError, URLError, HTTPError, ValueError) as exc:
            self._send_error_json(HTTPStatus.BAD_GATEWAY, f"upstream error: {format_upstream_error(exc)}")

    def _proxy_stream(self, path_qs: str, body: bytes, headers: dict[str, str]) -> None:
        url = f"{self.api_base.rstrip('/')}{path_qs}"
        req = Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": headers.get("Content-Type", "application/json"),
                "Accept": "text/event-stream",
            },
        )
        try:
            with urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                ctype = resp.headers.get("Content-Type", "text/event-stream")
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except HTTPError as exc:
            if not self.wfile.closed:
                try:
                    detail = format_upstream_error(exc)
                    # Prefer upstream status (e.g. 400 context_length_exceeded).
                    self._send_error_json(exc.code or HTTPStatus.BAD_GATEWAY, detail)
                except OSError:
                    pass
        except (OSError, URLError) as exc:
            if not self.wfile.closed:
                try:
                    self._send_error_json(
                        HTTPStatus.BAD_GATEWAY, f"upstream error: {format_upstream_error(exc)}"
                    )
                except OSError:
                    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DwarfStar local chat UI")
    parser.add_argument("--host", default=DEFAULT_HOST, help="UI bind host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="UI bind port")
    parser.add_argument(
        "--api",
        default=DEFAULT_API,
        help=f"ds4-server base URL (default {DEFAULT_API})",
    )
    parser.add_argument(
        "--chats-dir",
        type=Path,
        default=None,
        help="Conversation store directory (default ~/.ds4/chats)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    chats_dir = ensure_chats_dir(args.chats_dir or default_chats_dir())
    if not STATIC_DIR.is_dir():
        print(f"missing static dir: {STATIC_DIR}", file=sys.stderr)
        return 1

    httpd = ThreadingHTTPServer((args.host, args.port), ChatUIHandler)
    httpd.chats_dir = chats_dir
    httpd.api_base = args.api.rstrip("/")
    print(
        f"DwarfStar chat UI on http://{args.host}:{args.port}\n"
        f"  proxy -> {httpd.api_base}\n"
        f"  chats -> {chats_dir}",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
