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
import auth  # noqa: E402
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

    def test_arrows_sound_conversational(self) -> None:
        out = tts.prepare_text("Heat → steam => motion <- source")
        self.assertNotRegex(out, r"[→⇒←]|=>|->|<-")
        self.assertIn(" to ", out)
        self.assertIn(" from ", out)
        lower = out.lower()
        self.assertNotIn("arrow", lower)

    def test_comparisons_not_confused_with_arrows(self) -> None:
        out = tts.prepare_text("if n <= 3 and m >= 1")
        self.assertIn("less than or equal to", out)
        self.assertIn("greater than or equal to", out)
        self.assertNotIn(" from ", out)

    def test_bullets_and_checks(self) -> None:
        out = tts.prepare_text("• first\n✓ done\n✗ skip")
        self.assertIn("yes", out.lower())
        self.assertIn("no", out.lower())
        self.assertNotRegex(out, r"[•✓✗]")

    def test_dashes_as_pauses(self) -> None:
        out = tts.prepare_text("warm — then cool - finally done -- end")
        self.assertNotIn("-", out)
        self.assertNotRegex(out, r"[—–―]")

    def test_hyphen_compounds_and_ranges(self) -> None:
        out = tts.prepare_text("Ankle-to-Crown Traction, pages 3-5, well-known")
        self.assertNotIn("-", out)
        self.assertIn("Ankle to Crown", out)
        self.assertIn("3 to 5", out)
        self.assertIn("well known", out)
        neg = tts.prepare_text("value is -3 degrees")
        self.assertNotIn("-", neg)
        self.assertIn("minus 3", neg)


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
        auth_path = Path(cls._tmpdir.name) / "auth.yaml"
        auth_path.write_text("username: test\npassword: test\n", encoding="utf-8")
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), ChatUIHandler)
        cls.httpd.chats_root = chats
        cls.httpd.api_base = "http://127.0.0.1:9"
        cls.httpd.user_store = auth.load_user_store(auth_path)
        cls.httpd.sessions = auth.SessionManager()
        cls._cookie = ""
        cls.port = cls.httpd.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls._thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls._thread.start()
        cls._login()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls._tmpdir.cleanup()

    @classmethod
    def _login(cls) -> None:
        status, _, headers = cls._post_json_static(
            "/api/auth/login",
            {"username": "test", "password": "test"},
        )
        assert status == 200
        cls._cookie = headers.get("set-cookie", "")

    @staticmethod
    def _post_json_static(path: str, payload: dict) -> tuple[int, bytes, dict[str, str]]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            ApiTtsIntegrationTests.base + path,
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

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._cookie:
            headers["Cookie"] = self._cookie
        return headers

    def _get(self, path: str) -> tuple[int, bytes, dict[str, str]]:
        req = urllib.request.Request(
            self.base + path,
            method="GET",
            headers=self._auth_headers(),
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, resp.read(), headers

    def _post_json(self, path: str, payload: dict) -> tuple[int, bytes, dict[str, str]]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **self._auth_headers()}
        req = urllib.request.Request(
            self.base + path,
            data=body,
            method="POST",
            headers=headers,
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
