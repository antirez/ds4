"""Tests for chat-ui bind host and startup URL helpers."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

CHAT_UI_DIR = Path(__file__).resolve().parents[1]
if str(CHAT_UI_DIR) not in sys.path:
    sys.path.insert(0, str(CHAT_UI_DIR))

from server import (  # noqa: E402
    DEFAULT_HOST,
    default_bind_host,
    format_startup_banner,
    local_lan_ips,
    parse_args,
)


class DefaultBindHostTests(unittest.TestCase):
    def test_default_is_all_interfaces(self) -> None:
        self.assertEqual(DEFAULT_HOST, "0.0.0.0")

    def test_env_override(self) -> None:
        with patch.dict(os.environ, {"DS4_CHAT_HOST": "192.168.1.5"}, clear=False):
            self.assertEqual(default_bind_host(), "192.168.1.5")

    def test_cli_host_overrides_env(self) -> None:
        with patch.dict(os.environ, {"DS4_CHAT_HOST": "10.0.0.1"}, clear=False):
            args = parse_args(["--host", "127.0.0.1"])
        self.assertEqual(args.host, "127.0.0.1")


class LocalLanIpsTests(unittest.TestCase):
    def test_returns_list_of_dotted_quads(self) -> None:
        with patch("server.socket.socket") as mock_socket_cls, patch(
            "server.platform.system", return_value="Linux"
        ), patch("server.socket.getaddrinfo", return_value=[]):
            mock_sock = mock_socket_cls.return_value.__enter__.return_value
            mock_sock.getsockname.return_value = ("192.168.50.10", 54321)
            ips = local_lan_ips()
        self.assertEqual(ips, ["192.168.50.10"])

    def test_skips_loopback(self) -> None:
        with patch("server.socket.socket") as mock_socket_cls, patch(
            "server.platform.system", return_value="Linux"
        ), patch("server.socket.getaddrinfo", return_value=[]):
            mock_sock = mock_socket_cls.return_value.__enter__.return_value
            mock_sock.getsockname.return_value = ("127.0.0.1", 54321)
            ips = local_lan_ips()
        self.assertEqual(ips, [])


class StartupBannerTests(unittest.TestCase):
    def test_bind_all_includes_local_and_lan(self) -> None:
        with patch("server.local_lan_ips", return_value=["192.168.1.42"]):
            banner = format_startup_banner(
                "0.0.0.0",
                8787,
                "http://127.0.0.1:8000",
                Path("/tmp/chats"),
            )
        self.assertIn("http://127.0.0.1:8787", banner)
        self.assertIn("http://192.168.1.42:8787", banner)
        self.assertIn("localhost", banner.lower())

    def test_localhost_only_single_url(self) -> None:
        banner = format_startup_banner(
            "127.0.0.1",
            8787,
            "http://127.0.0.1:8000",
            Path("/tmp/chats"),
        )
        self.assertIn("http://127.0.0.1:8787", banner)
        self.assertNotIn("lan", banner.lower())


if __name__ == "__main__":
    unittest.main()
