"""YAML-backed credentials and cookie sessions for the chat UI."""

from __future__ import annotations

import os
import re
import secrets
import time
from dataclasses import dataclass
from http.cookies import SimpleCookie
from pathlib import Path

SESSION_COOKIE = "ds4_session"
DEFAULT_SESSION_TTL = 7 * 24 * 3600
USERNAME_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


@dataclass(frozen=True)
class UserStore:
    users: dict[str, str]

    @property
    def usernames(self) -> frozenset[str]:
        return frozenset(self.users)


@dataclass(frozen=True)
class SessionRecord:
    username: str
    expiry: float


def default_auth_path() -> Path:
    override = os.environ.get("DS4_AUTH_FILE", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ds4" / "auth.yaml"


def _strip_yaml_value(val: str) -> str:
    val = val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        return val[1:-1]
    return val


def _parse_kv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if ":" not in stripped:
        return None
    key, _, val = stripped.partition(":")
    key = key.strip()
    if not key:
        return None
    return key, _strip_yaml_value(val)


def parse_auth_yaml(text: str) -> dict[str, str]:
    """Parse username/password pairs from a minimal YAML subset.

    Supported shapes:
      - repeated top-level username/password pairs
      - a ``users:`` map with ``username:`` / ``password:`` children
      - legacy single username/password block
    """
    users: dict[str, str] = {}
    pending_user: str | None = None
    in_users = False
    users_base_indent: int | None = None
    current_map_user: str | None = None

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue

        indent = len(raw) - len(raw.lstrip())
        kv = _parse_kv_line(raw)
        if kv is None:
            continue
        key, val = kv

        if not in_users and key == "users" and not val:
            in_users = True
            users_base_indent = indent
            pending_user = None
            current_map_user = None
            continue

        if in_users and users_base_indent is not None and indent > users_base_indent:
            if indent == users_base_indent + 2 and key not in ("password", "username") and not val:
                current_map_user = key
                continue
            if key == "password" and current_map_user:
                if not val:
                    raise ValueError(f"password required for user {current_map_user!r}")
                users[current_map_user] = val
                current_map_user = None
                continue
            if key == "username" and val:
                pending_user = val
                continue
            if key == "password" and pending_user:
                if not val:
                    raise ValueError(f"password required for user {pending_user!r}")
                users[pending_user] = val
                pending_user = None
                continue
            continue

        if in_users and users_base_indent is not None and indent <= users_base_indent:
            in_users = False
            users_base_indent = None
            current_map_user = None

        if key == "username":
            pending_user = val
            continue
        if key == "password":
            if not pending_user:
                raise ValueError("password without preceding username")
            if not val:
                raise ValueError(f"password required for user {pending_user!r}")
            users[pending_user] = val
            pending_user = None

    if pending_user:
        raise ValueError(f"password missing for user {pending_user!r}")
    if current_map_user:
        raise ValueError(f"password missing for user {current_map_user!r}")
    if not users:
        raise ValueError("auth yaml must define at least one username/password pair")
    return users


def load_user_store(path: Path) -> UserStore:
    if not path.is_file():
        raise FileNotFoundError(f"auth file not found: {path}")
    users = parse_auth_yaml(path.read_text(encoding="utf-8"))
    for username, password in users.items():
        if not USERNAME_RE.match(username):
            raise ValueError(f"invalid username {username!r}")
        if not password:
            raise ValueError(f"password must be non-empty for user {username!r}")
    return UserStore(users=users)


def verify_login(username: str, password: str, store: UserStore) -> bool:
    if not USERNAME_RE.match(username):
        return False
    expected = store.users.get(username)
    if expected is None:
        return False
    return secrets.compare_digest(password, expected)


class SessionManager:
    def __init__(self, ttl_seconds: int = DEFAULT_SESSION_TTL) -> None:
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._sessions: dict[str, SessionRecord] = {}

    def create(self, username: str) -> str:
        if not USERNAME_RE.match(username):
            raise ValueError(f"invalid username {username!r}")
        token = secrets.token_urlsafe(32)
        self._sessions[token] = SessionRecord(
            username=username,
            expiry=time.time() + self.ttl_seconds,
        )
        return token

    def valid(self, token: str | None) -> bool:
        return self.username(token) is not None

    def username(self, token: str | None) -> str | None:
        if not token:
            return None
        record = self._sessions.get(token)
        if record is None:
            return None
        if time.time() > record.expiry:
            self._sessions.pop(token, None)
            return None
        return record.username

    def revoke(self, token: str | None) -> None:
        if token:
            self._sessions.pop(token, None)


def parse_cookie_value(cookie_header: str, name: str) -> str | None:
    jar = SimpleCookie()
    jar.load(cookie_header or "")
    morsel = jar.get(name)
    if morsel is None:
        return None
    value = morsel.value
    return value if value else None


def session_set_cookie(token: str, max_age: int, *, secure: bool = False) -> str:
    parts = [
        f"{SESSION_COOKIE}={token}",
        "HttpOnly",
        "Path=/",
        "SameSite=Lax",
        f"Max-Age={max(0, int(max_age))}",
    ]
    if secure:
        parts.append("Secure")
    return "; ".join(parts)


def session_clear_cookie(*, secure: bool = False) -> str:
    parts = [
        f"{SESSION_COOKIE}=",
        "HttpOnly",
        "Path=/",
        "SameSite=Lax",
        "Max-Age=0",
    ]
    if secure:
        parts.append("Secure")
    return "; ".join(parts)


PUBLIC_STATIC = frozenset({"login.html", "login.js", "styles.css"})
PUBLIC_API_PATHS = frozenset({"/api/auth/login", "/api/auth/session"})


def route_is_public(path: str) -> bool:
    if path in PUBLIC_API_PATHS:
        return True
    rel = path.lstrip("/")
    return rel in PUBLIC_STATIC
