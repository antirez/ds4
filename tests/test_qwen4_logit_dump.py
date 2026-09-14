"""Check Qwen all-row and teacher-forced logit dumps, including chunk boundaries."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output or Path(tempfile.mkdtemp(prefix="qwen-logit-dump-"))
    out.mkdir(parents=True, exist_ok=True)
    out = out.resolve()
    # Token IDs 1..6 are ordinary vocabulary entries. Cross the 128-row buffer.
    tokens = [1, 2, 3, 4, 5, 6] * 23
    lengths = (6, 132, 136)
    ids = lambda values: ",".join(map(str, values))
    rows = []
    for length in lengths:
        rows.append(f"{ids(tokens[:length])}\t{out / f'all-{length}.bin'}\n")
        prefix = length - 2
        rows.append(f"{ids(tokens[:prefix])}|{ids(tokens[prefix:length] + [1])}"
                    f"\t{out / f'target-{length}.bin'}\n")
    manifest = out / "inputs.tsv"
    manifest.write_text("".join(rows))
    env = os.environ.copy()
    env["DS4_QWEN4_GPU"] = "1"
    env["DS4_QWEN4_FT_LIST"] = str(manifest)
    result = subprocess.run(
        [str(root / "ds4"), "--metal", "-m", str(args.model.resolve()),
         "--ctx", "256",
         "--first-token-test", "-p", "hello"], cwd=root, env=env,
        capture_output=True, timeout=300)
    (out / "stdout").write_bytes(result.stdout)
    (out / "stderr").write_bytes(result.stderr)
    assert result.returncode == 0, f"dump failed; see {out}"
    for length in lengths:
        all_rows = np.fromfile(out / f"all-{length}.bin", dtype=np.float32)
        targets = np.fromfile(out / f"target-{length}.bin", dtype=np.float32)
        assert len(all_rows) and len(all_rows) % length == 0
        vocab = len(all_rows) // length
        assert len(targets) == 3 * vocab
        all_rows = all_rows.reshape(length, vocab)
        targets = targets.reshape(3, vocab)
        assert np.isfinite(all_rows).all() and np.isfinite(targets).all()
        # Different batch sizes permit normal rounding, not missing or shifted rows.
        error = float(np.max(np.abs(all_rows[-3:] - targets)))
        assert error < 0.03, (length, error, out)
        np.testing.assert_array_equal(all_rows[-3:].argmax(axis=1), targets.argmax(axis=1))
        print(f"PASS {length} rows: maximum logit difference {error:.6g}", flush=True)


if __name__ == "__main__":
    main()
