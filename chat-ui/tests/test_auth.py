"""Unit tests for chat-ui YAML credentials and session auth."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

CHAT_UI_DIR = Path(__file__).resolve().parents[1]
if str(CHAT_UI_DIR) not in sys.path:
    sys.path.insert(0, str(CHAT_UI_DIR))

import auth  # noqa: E402
import server  # noqa: E402


class ParseAuthYamlTests(unittest.TestCase):
    def test_parses_plain_keys(self) -> None:
        data = auth.parse_auth_yaml("username: admin\npassword: secret\n")
        self.assertEqual(data["admin"], "secret")

    def test_parses_quoted_values(self) -> None:
        data = auth.parse_auth_yaml('username: "admin"\npassword: \'s3cret\'\n')
        self.assertEqual(data["admin"], "s3cret")

    def test_parses_repeated_pairs(self) -> None:
        text = "username: davide\npassword: one\n\nusername: guest\npassword: two\n"
        data = auth.parse_auth_yaml(text)
        self.assertEqual(data["davide"], "one")
        self.assertEqual(data["guest"], "two")

    def test_parses_users_map(self) -> None:
        text = "users:\n  davide:\n    password: one\n  guest:\n    password: two\n"
        data = auth.parse_auth_yaml(text)
        self.assertEqual(data["davide"], "one")
        self.assertEqual(data["guest"], "two")

    def test_requires_both_keys(self) -> None:
        with self.assertRaises(ValueError):
            auth.parse_auth_yaml("username: only\n")


class UserStoreLoadTests(unittest.TestCase):
    def test_load_user_store_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "auth.yaml"
            path.write_text("username: alice\npassword: wonderland\n", encoding="utf-8")
            store = auth.load_user_store(path)
            self.assertEqual(store.users["alice"], "wonderland")


class VerifyLoginTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = auth.UserStore(users={"admin": "secret"})

    def test_accepts_matching_credentials(self) -> None:
        self.assertTrue(auth.verify_login("admin", "secret", self.store))

    def test_rejects_wrong_password(self) -> None:
        self.assertFalse(auth.verify_login("admin", "wrong", self.store))

    def test_rejects_wrong_username(self) -> None:
        self.assertFalse(auth.verify_login("other", "secret", self.store))


class SessionManagerTests(unittest.TestCase):
    def test_create_and_validate(self) -> None:
        mgr = auth.SessionManager(ttl_seconds=120)
        token = mgr.create("alice")
        self.assertTrue(mgr.valid(token))
        self.assertEqual(mgr.username(token), "alice")

    def test_revoke_invalidates(self) -> None:
        mgr = auth.SessionManager()
        token = mgr.create("alice")
        mgr.revoke(token)
        self.assertFalse(mgr.valid(token))


class AuthHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.auth_path = Path(self.tmp.name) / "auth.yaml"
        self.auth_path.write_text(
            "username: testuser\npassword: testpass\n",
            encoding="utf-8",
        )
        self.chats_root = Path(self.tmp.name) / "chats"
        self.chats_root.mkdir()

        self.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.ChatUIHandler)
        self.httpd.user_store = auth.load_user_store(self.auth_path)
        self.httpd.sessions = auth.SessionManager()
        self.httpd.chats_root = self.chats_root
        self.httpd.api_base = "http://127.0.0.1:8000"
        self.host, self.port = self.httpd.server_address
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        cookie: str | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = HTTPConnection(self.host, self.port, timeout=5)
        headers: dict[str, str] = {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if cookie:
            headers["Cookie"] = cookie
        payload = None if body is None else json.dumps(body).encode("utf-8")
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        conn.close()
        return resp.status, hdrs, raw

    def test_root_serves_login_when_unauthenticated(self) -> None:
        status, _, body = self._request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Sign in", body)

    def test_api_chats_requires_auth(self) -> None:
        status, _, raw = self._request("GET", "/api/chats")
        self.assertEqual(status, 401)
        data = json.loads(raw.decode("utf-8"))
        self.assertEqual(data["error"], "authentication required")

    def test_login_success_sets_cookie_and_unlocks_api(self) -> None:
        status, headers, raw = self._request(
            "POST",
            "/api/auth/login",
            {"username": "testuser", "password": "testpass"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(raw.decode("utf-8")),
            {"ok": True, "username": "testuser"},
        )
        cookie = headers.get("set-cookie", "")
        self.assertIn("ds4_session=", cookie)
        self.assertIn("HttpOnly", cookie)

        status2, _, raw2 = self._request("GET", "/api/chats", cookie=cookie)
        self.assertEqual(status2, 200)
        self.assertIn("chats", json.loads(raw2.decode("utf-8")))

    def test_login_failure(self) -> None:
        status, _, raw = self._request(
            "POST",
            "/api/auth/login",
            {"username": "testuser", "password": "bad"},
        )
        self.assertEqual(status, 401)
        data = json.loads(raw.decode("utf-8"))
        self.assertEqual(data["error"], "invalid username or password")

    def test_logout_clears_session(self) -> None:
        _, headers, _ = self._request(
            "POST",
            "/api/auth/login",
            {"username": "testuser", "password": "testpass"},
        )
        cookie = headers.get("set-cookie", "")
        status, hdrs, _ = self._request("POST", "/api/auth/logout", cookie=cookie)
        self.assertEqual(status, 200)
        self.assertIn("Max-Age=0", hdrs.get("set-cookie", ""))

        status2, _, _ = self._request("GET", "/api/chats", cookie=cookie)
        self.assertEqual(status2, 401)


class MultiUserAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.auth_path = Path(self.tmp.name) / "auth.yaml"
        self.auth_path.write_text(
            "username: alice\npassword: pass-a\nusername: bob\npassword: pass-b\n",
            encoding="utf-8",
        )
        self.chats_root = Path(self.tmp.name) / "chats"
        self.chats_root.mkdir()

        self.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.ChatUIHandler)
        self.httpd.user_store = auth.load_user_store(self.auth_path)
        self.httpd.sessions = auth.SessionManager()
        self.httpd.chats_root = self.chats_root
        self.httpd.api_base = "http://127.0.0.1:8000"
        self.host, self.port = self.httpd.server_address
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        cookie: str | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = HTTPConnection(self.host, self.port, timeout=5)
        headers: dict[str, str] = {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if cookie:
            headers["Cookie"] = cookie
        payload = None if body is None else json.dumps(body).encode("utf-8")
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        conn.close()
        return resp.status, hdrs, raw

    def _login(self, username: str, password: str) -> str:
        status, headers, _ = self._request(
            "POST",
            "/api/auth/login",
            {"username": username, "password": password},
        )
        self.assertEqual(status, 200)
        return headers.get("set-cookie", "")

    def test_session_reports_username(self) -> None:
        cookie = self._login("alice", "pass-a")
        status, _, raw = self._request("GET", "/api/auth/session", cookie=cookie)
        self.assertEqual(status, 200)
        data = json.loads(raw.decode("utf-8"))
        self.assertTrue(data["authenticated"])
        self.assertEqual(data["username"], "alice")

    def test_users_have_isolated_chat_dirs(self) -> None:
        cookie_a = self._login("alice", "pass-a")
        cookie_b = self._login("bob", "pass-b")

        status, _, raw = self._request(
            "POST",
            "/api/chats",
            {"title": "Alice chat"},
            cookie=cookie_a,
        )
        self.assertEqual(status, 201)
        alice_id = json.loads(raw.decode("utf-8"))["id"]

        status, _, raw = self._request(
            "POST",
            "/api/chats",
            {"title": "Bob chat"},
            cookie=cookie_b,
        )
        self.assertEqual(status, 201)
        bob_id = json.loads(raw.decode("utf-8"))["id"]

        status, _, raw = self._request("GET", "/api/chats", cookie=cookie_a)
        alice_rows = json.loads(raw.decode("utf-8"))["chats"]
        self.assertEqual(len(alice_rows), 1)
        self.assertEqual(alice_rows[0]["id"], alice_id)

        status, _, raw = self._request("GET", "/api/chats", cookie=cookie_b)
        bob_rows = json.loads(raw.decode("utf-8"))["chats"]
        self.assertEqual(len(bob_rows), 1)
        self.assertEqual(bob_rows[0]["id"], bob_id)

        status, _, _ = self._request("GET", f"/api/chats/{bob_id}", cookie=cookie_a)
        self.assertEqual(status, 404)

        self.assertTrue((self.chats_root / "alice" / f"{alice_id}.json").is_file())
        self.assertTrue((self.chats_root / "bob" / f"{bob_id}.json").is_file())


class LegacyChatMigrationTests(unittest.TestCase):
    def test_migrate_flat_chats_into_owner_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "chats"
            root.mkdir()
            legacy = root / "abc.json"
            legacy.write_text('{"id":"abc","title":"Old","messages":[]}', encoding="utf-8")
            moved = server.migrate_legacy_chats(root, "davide")
            self.assertEqual(moved, 1)
            self.assertFalse(legacy.exists())
            self.assertTrue((root / "davide" / "abc.json").is_file())

    def test_migration_skips_when_dest_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "chats"
            owner = root / "davide"
            owner.mkdir(parents=True)
            legacy = root / "abc.json"
            legacy.write_text('{"id":"abc"}', encoding="utf-8")
            (owner / "abc.json").write_text('{"id":"abc","kept":true}', encoding="utf-8")
            moved = server.migrate_legacy_chats(root, "davide")
            self.assertEqual(moved, 0)
            self.assertTrue(legacy.is_file())
            self.assertEqual(
                (owner / "abc.json").read_text(encoding="utf-8"),
                '{"id":"abc","kept":true}',
            )


if __name__ == "__main__":
    unittest.main()
