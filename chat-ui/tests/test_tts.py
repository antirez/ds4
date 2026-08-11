"""Unit and integration tests for chat-ui local TTS."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

CHAT_UI_DIR = Path(__file__).resolve().parents[1]
if str(CHAT_UI_DIR) not in sys.path:
    sys.path.insert(0, str(CHAT_UI_DIR))

import tts  # noqa: E402
from server import ChatUIHandler, ensure_chats_dir  # noqa: E402


class PrepareTextTests(unittest.TestCase):
    def test_strips_fenced_code(self) -> None:
        out = tts.prepare_text("Hello\n```python\nprint(1)\n```\nworld")
        self.assertIn("Hello", out)
        self.assertIn("world", out)
        self.assertNotIn("print", out)

    def test_truncates_long_input(self) -> None:
        blob = "a" * (tts.MAX_TTS_CHARS + 500)
        out = tts.prepare_text(blob)
        self.assertLessEqual(len(out), tts.MAX_TTS_CHARS)
        self.assertTrue(out.endswith("…"))

    def test_empty_raises(self) -> None:
        with self.assertRaises(tts.TtsError):
            tts.prepare_text("   ")

    def test_code_only_raises(self) -> None:
        with self.assertRaises(tts.TtsError):
            tts.prepare_text("```\nonly code\n```")


class SynthesizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tools = tts.discover_tools()
        if not cls.tools.available:
            raise unittest.SkipTest("no local TTS engine (say/afconvert or Piper)")

    def test_synthesize_wav_header(self) -> None:
        wav, engine = tts.synthesize_wav("DwarfStar TTS unit test.", tools=self.tools)
        self.assertIn(engine, {"say", "piper"})
        self.assertGreater(len(wav), 44)
        self.assertEqual(wav[:4], b"RIFF")
        self.assertEqual(wav[8:12], b"WAVE")

    def test_tooling_status_matches(self) -> None:
        status = tts.tooling_status(self.tools)
        self.assertTrue(status["available"])
        self.assertEqual(status["engine"], self.tools.engine)
        self.assertEqual(status["prefer"], "piper")

    def test_prefers_piper_when_available(self) -> None:
        tools = tts.discover_tools()
        if not (tools.piper and tools.piper_model):
            self.skipTest("piper not installed locally")
        wav, engine = tts.synthesize_wav("Prefer Piper over macOS say.", tools=tools)
        self.assertEqual(engine, "piper")
        self.assertEqual(wav[:4], b"RIFF")


class ApiTtsIntegrationTests(unittest.TestCase):
    """Spin an ephemeral chat-ui server and hit /api/tts."""

    @classmethod
    def setUpClass(cls) -> None:
        tools = tts.discover_tools()
        if not tools.available:
            raise unittest.SkipTest("no local TTS engine for HTTP integration")

        cls._tmpdir = tempfile.TemporaryDirectory(prefix="ds4-chat-ui-test-")
        chats = ensure_chats_dir(Path(cls._tmpdir.name) / "chats")
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), ChatUIHandler)
        cls.httpd.chats_dir = chats
        cls.httpd.api_base = "http://127.0.0.1:9"
        cls.port = cls.httpd.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls._thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls._thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls._tmpdir.cleanup()

    def _get(self, path: str) -> tuple[int, bytes, dict[str, str]]:
        req = urllib.request.Request(self.base + path, method="GET")
        with urllib.request.urlopen(req, timeout=60) as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, resp.read(), headers

    def _post_json(self, path: str, payload: dict) -> tuple[int, bytes, dict[str, str]]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base + path,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                headers = {k.lower(): v for k, v in resp.headers.items()}
                return resp.status, resp.read(), headers
        except urllib.error.HTTPError as exc:
            headers = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
            return exc.code, exc.read(), headers

    def test_health_reports_tts(self) -> None:
        status, raw, _ = self._get("/api/health")
        self.assertEqual(status, 200)
        data = json.loads(raw.decode("utf-8"))
        self.assertIn("tts", data)
        self.assertTrue(data["tts"]["available"])

    def test_tts_returns_wav(self) -> None:
        status, raw, headers = self._post_json(
            "/api/tts", {"text": "Integration test for DwarfStar read aloud."}
        )
        self.assertEqual(status, 200)
        self.assertIn("audio/wav", headers.get("content-type", ""))
        self.assertIn(headers.get("x-tts-engine", ""), {"say", "piper"})
        self.assertEqual(raw[:4], b"RIFF")
        self.assertEqual(raw[8:12], b"WAVE")
        self.assertGreater(len(raw), 44)

    def test_tts_rejects_empty(self) -> None:
        status, raw, _ = self._post_json("/api/tts", {"text": "  "})
        self.assertEqual(status, 400)
        data = json.loads(raw.decode("utf-8"))
        self.assertIn("error", data)


class LiveChatUiProbeTests(unittest.TestCase):
    """Optional probe of a user-started chat-ui on :8787."""

    LIVE = "http://127.0.0.1:8787"

    def test_live_health_or_stale_hint(self) -> None:
        try:
            with urllib.request.urlopen(self.LIVE + "/api/health", timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 — probe only
            self.skipTest(f"live chat-ui not reachable: {exc}")
            return
        if "tts" not in data:
            self.skipTest(
                "live chat-ui on :8787 is stale (no health.tts) — restart chat-ui "
                "so /api/tts is registered, then re-run tests"
            )
        self.assertTrue(data["tts"].get("available"), data["tts"])


if __name__ == "__main__":
    unittest.main()
