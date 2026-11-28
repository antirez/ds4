"""Exercise disk checkpoints across fresh Metal processes on a real K3 GGUF."""
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile


def prepare_fixtures(state, damaged, legacy):
    with state.open("rb") as payload:
        header = list(struct.unpack("<6I", payload.read(24)))
        magic, version, _, _, tokens, _ = header
        if (magic, version) != (0x3356534B, 2):
            raise ValueError("expected a native K3 payload")
        header[1] = 1
        legacy.write_bytes(struct.pack("<6I", *header) + payload.read(tokens * 4))

    shutil.copyfile(state, damaged)
    with damaged.open("r+b") as payload:
        # Damage the last byte of the recurrent/attention state, keeping the
        # saved digest intact to exercise detection of a same-length bad file.
        payload.seek(-33, os.SEEK_END)
        byte = payload.read(1)[0]
        payload.seek(-33, os.SEEK_END)
        payload.write(bytes([byte ^ 1]))


def run(model, directory):
    exe = Path(__file__).resolve().with_name("test_k3_session_resume")
    state = directory / "state"
    logits = directory / "logits"
    damaged = directory / "damaged"
    legacy = directory / "legacy"

    subprocess.run([exe, "save", model, state, logits], check=True)
    prepare_fixtures(state, damaged, legacy)
    subprocess.run([exe, "restore", model, state, logits, damaged], check=True)
    subprocess.run([exe, "restore", model, legacy, logits], check=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: run_k3_session_resume.py MODEL.gguf")
    with tempfile.TemporaryDirectory(prefix="ds4-k3-resume-") as tmp:
        run(sys.argv[1], Path(tmp))
