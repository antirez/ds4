"""Qwen MTP must honor small prefill buffers and unfused diagnostic mode."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output or Path(tempfile.mkdtemp(prefix="qwen-mtp-limits-"))
    out.mkdir(parents=True, exist_ok=True)
    base = [str(root / "ds4"), "-m", str(args.model.resolve()),
            "--ctx", "256", "--temp", "0",
            "--nothink", "-p", "Count from one to ten.", "-n", "24"]
    for name, chunk, unfused in [("one-row", 1, False),
                                  ("two-rows-depth-three", 2, False),
                                  ("unfused", 128, True)]:
        env = os.environ.copy()
        env.pop("DS4_QWEN4_SPEC_FORCE_ACCEPT", None)
        env["DS4_QWEN4_MTP_DEPTH"] = "3"
        if unfused:
            env["DS4_QWEN4_NO_FUSE"] = "1"
        else:
            env.pop("DS4_QWEN4_NO_FUSE", None)
        outputs = []
        for mode, extra in [("plain", []), ("mtp", ["--mtp"])]:
            result = subprocess.run(base + ["--prefill-chunk", str(chunk)] + extra,
                                    cwd=root, env=env, capture_output=True, timeout=180)
            (out / f"{name}-{mode}.stdout").write_bytes(result.stdout)
            (out / f"{name}-{mode}.stderr").write_bytes(result.stderr)
            assert result.returncode == 0, (name, mode, result.stderr.decode(errors="replace"))
            assert b"failed" not in result.stderr.lower(), (name, mode)
            outputs.append(result.stdout)
        assert outputs[0] == outputs[1], f"{name}: MTP changed the greedy continuation; see {out}"
        print(f"PASS {name}", flush=True)


if __name__ == "__main__":
    main()
