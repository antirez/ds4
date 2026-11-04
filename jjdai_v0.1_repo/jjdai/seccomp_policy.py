"""JJ DAI v0.1 — M1 seccomp policy (Linux strong tier).

Builds a syscall filter and exports it as a BPF program on an fd, which
bubblewrap consumes via `--seccomp <fd>`. If libseccomp/pyseccomp is not
installed, this module reports unavailable and the sandbox tier caps at
'medium' (namespaces without syscall filtering).

Policy = denylist of escape/attack-surface syscalls (default allow, so the
interpreter keeps working). An allowlist is stricter and is future hardening;
a denylist is the pragmatic first cut that still removes the dangerous edges.
"""
from __future__ import annotations
import os
import tempfile

try:
    import seccomp as _s  # pyseccomp / libseccomp python bindings
    _OK = True
except Exception:  # pragma: no cover
    _s = None
    _OK = False

# Syscalls an artifact runner should never need; blocking them removes network
# egress, module loading, tracing, mounts, and reboot-class escapes.
_DENY = [
    "socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
    "sendto", "recvfrom", "sendmsg", "recvmsg",
    "ptrace", "process_vm_readv", "process_vm_writev",
    "mount", "umount2", "pivot_root", "chroot", "swapon", "swapoff",
    "init_module", "finit_module", "delete_module", "kexec_load", "kexec_file_load",
    "bpf", "perf_event_open", "reboot",
    "clock_settime", "settimeofday", "setns", "unshare", "add_key", "keyctl",
]


def available() -> bool:
    return _OK


def open_bpf() -> int | None:
    """Return an OS fd holding the exported BPF program, or None."""
    if not _OK:
        return None
    try:
        f = _s.SyscallFilter(defaction=_s.ALLOW)
        for name in _DENY:
            try:
                f.add_rule(_s.ERRNO(1), name)  # EPERM
            except Exception:
                pass  # syscall not on this arch; skip
        tmp = tempfile.TemporaryFile()
        f.export_bpf(tmp)
        tmp.flush()
        os.lseek(tmp.fileno(), 0, os.SEEK_SET)
        # dup so the fd survives independently of the TemporaryFile object
        return os.dup(tmp.fileno())
    except Exception:
        return None
