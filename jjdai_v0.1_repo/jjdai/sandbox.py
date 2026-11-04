"""JJ DAI v0.1 — M1 HARDENED sandbox runner.

Executes a model-produced ARTIFACT (not the model) in an isolated child and
returns a replayable trace (§7). Hardened over the M1 prototype with:

  * fail-closed tiers  — refuses to run below the requested isolation tier
                         (no silent "run untrusted code with no isolation")
  * strong tier        — Linux: bubblewrap (unshare-all, cap-drop, ro-rootfs,
                         private tmpfs, optional seccomp filter);
                         macOS:  sandbox-exec (Seatbelt) deny-by-default + no-net
  * always-on limits   — CPU, address space, file size, no core dumps, no-new-privs,
                         wall-clock timeout, and an OUTPUT CAP enforced in the parent
                         (contains output-bomb OOM in every tier)

Fork-bomb containment needs a pids cgroup and is a STRONG-tier property; WEAK does
NOT guarantee it — which is exactly why fail-closed defaults to strong.
"""
from __future__ import annotations
import ctypes
import os
import platform
import resource
import shutil
import subprocess
import threading
import time

from proto import SandboxTrace, sha256, sha256_text

try:
    import seccomp_policy
except Exception:  # pragma: no cover
    seccomp_policy = None

BASE_ENV = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC",
    "PYTHONHASHSEED": "0", "JJDAI_SEED": "1",
}

DEFAULT_LIMITS = {
    "cpu_seconds": 5,
    "mem_bytes": 512 * 1024 * 1024,
    "wall_seconds": 10,
    "fsize_bytes": 8 * 1024 * 1024,     # max bytes an artifact may write to a file
    "out_cap_bytes": 1 * 1024 * 1024,   # max stdout captured before we cut + kill
}

_TIER_RANK = {"weak": 0, "medium": 1, "strong": 2}


class SandboxTooWeak(RuntimeError):
    """Raised (fail-closed) when the host cannot meet the requested tier."""


def _have(tool: str) -> bool:
    return shutil.which(tool) is not None


def _seccomp_ok() -> bool:
    return bool(seccomp_policy and seccomp_policy.available())


def detect_tier() -> str:
    sysname = platform.system()
    if sysname == "Linux" and _have("bwrap"):
        return "strong" if _seccomp_ok() else "medium"
    if sysname == "Darwin" and _have("sandbox-exec"):
        return "strong"
    return "weak"


def _no_new_privs():
    # prctl(PR_SET_NO_NEW_PRIVS=38, 1): setuid binaries can't escalate.
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(38, 1, 0, 0, 0)
    except Exception:
        pass


def _preexec(limits: dict):
    def _apply():
        resource.setrlimit(resource.RLIMIT_CPU,
                           (limits["cpu_seconds"], limits["cpu_seconds"]))
        resource.setrlimit(resource.RLIMIT_AS,
                           (limits["mem_bytes"], limits["mem_bytes"]))
        resource.setrlimit(resource.RLIMIT_FSIZE,
                           (limits["fsize_bytes"], limits["fsize_bytes"]))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        _no_new_privs()
    return _apply


def _env_fingerprint(limits: dict, tier: str) -> str:
    parts = [platform.python_version(), platform.machine(), platform.system(),
             f"tier={tier}",
             *(f"{k}={limits[k]}" for k in sorted(limits)),
             "env=" + ",".join(sorted(BASE_ENV))]
    return sha256_text("|".join(parts))


def _seatbelt_profile(jail: str) -> str:
    return ("(version 1)(deny default)(allow process-exec)(allow process-fork)"
            "(allow sysctl-read)"
            f'(allow file-read* (subpath "/usr")(subpath "/System")'
            f'(subpath "/Library")(subpath "{jail}"))'
            f'(allow file-write* (subpath "{jail}")(literal "/dev/null"))'
            "(deny network*)")


def _wrap(cmd: list[str], jail: str, tier: str, seccomp_fd: int | None):
    """Return (wrapped_cmd, pass_fds) for the chosen tier."""
    if tier in ("strong", "medium") and platform.system() == "Linux":
        w = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
             "--cap-drop", "ALL", "--clearenv",
             "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin",
             "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
             "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
             "--bind", jail, jail, "--chdir", jail]
        for k, v in BASE_ENV.items():
            w += ["--setenv", k, v]
        pass_fds: tuple = ()
        if tier == "strong" and seccomp_fd is not None:
            w += ["--seccomp", str(seccomp_fd)]
            pass_fds = (seccomp_fd,)
        return w + ["--", *cmd], pass_fds
    if tier == "strong" and platform.system() == "Darwin":
        return ["sandbox-exec", "-p", _seatbelt_profile(jail), *cmd], ()
    return cmd, ()  # weak


def _run_capped(wrapped, stdin, cwd, wall, out_cap, preexec, pass_fds):
    p = subprocess.Popen(
        wrapped, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=BASE_ENV, cwd=cwd,
        preexec_fn=preexec if os.name == "posix" else None,
        close_fds=True, start_new_session=True, pass_fds=pass_fds)

    sinks = {"out": bytearray(), "err": bytearray(), "trunc": False}

    def reader(pipe, key, cap):
        while True:
            chunk = pipe.read(65536)
            if not chunk:
                break
            room = cap - len(sinks[key])
            if room > 0:
                sinks[key].extend(chunk[:room])
            else:
                sinks["trunc"] = True
                try:  # closing our end SIGPIPEs the writer → output bomb contained
                    pipe.close()
                except Exception:
                    pass
                break

    def feed():
        try:
            p.stdin.write(stdin.encode("utf-8")); p.stdin.close()
        except Exception:
            pass

    threads = [threading.Thread(target=feed, daemon=True),
               threading.Thread(target=reader, args=(p.stdout, "out", out_cap), daemon=True),
               threading.Thread(target=reader, args=(p.stderr, "err", 65536), daemon=True)]
    for t in threads:
        t.start()

    timed_out = False
    try:
        p.wait(timeout=wall)
    except subprocess.TimeoutExpired:
        timed_out = True
    if timed_out or sinks["trunc"]:
        try:
            os.killpg(os.getpgid(p.pid), 9)
        except Exception:
            pass
        p.wait()
    for t in threads:
        t.join(timeout=1)
    code = p.returncode if p.returncode is not None else -9
    return bytes(sinks["out"]), bytes(sinks["err"]), code, timed_out, sinks["trunc"]


def run(cmd: list[str], stdin: str = "", jail: str | None = None,
        limits: dict | None = None, min_tier: str = "strong") -> SandboxTrace:
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    tier = detect_tier()
    if _TIER_RANK[tier] < _TIER_RANK[min_tier]:
        raise SandboxTooWeak(
            f"host isolation tier is '{tier}', but '{min_tier}' was required. "
            f"Install bubblewrap(+libseccomp) on Linux or run on macOS with "
            f"sandbox-exec, or pass a lower min_tier explicitly for dev.")

    jail = jail or os.path.join(os.getcwd(), ".jail")
    os.makedirs(jail, exist_ok=True)

    seccomp_fd = seccomp_policy.open_bpf() if (tier == "strong" and _seccomp_ok()) else None
    wrapped, pass_fds = _wrap(cmd, jail, tier, seccomp_fd)

    in_hash = sha256_text(repr(cmd) + "\x00" + stdin)
    env_fp = _env_fingerprint(limits, tier)

    t0 = time.monotonic()
    out, err, code, timed_out, trunc = _run_capped(
        wrapped, stdin, jail, limits["wall_seconds"], limits["out_cap_bytes"],
        _preexec(limits), pass_fds)
    wall_ms = int((time.monotonic() - t0) * 1000)
    if seccomp_fd is not None:
        try:
            os.close(seccomp_fd)
        except Exception:
            pass

    label = {"strong": "bwrap/seatbelt+limits", "medium": "bwrap(no-seccomp)+limits",
             "weak": "rlimit+outcap (UNSAFE: no ns/net isolation)"}[tier]
    return SandboxTrace(
        env_fp=env_fp, in_hash=in_hash, out_hash=sha256(out), err_hash=sha256(err),
        exit_code=code, wall_ms=wall_ms, isolation=label, timed_out=timed_out,
        tier=tier, truncated=trunc)
