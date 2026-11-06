"""Model regression for Qwen steering, MTP, and prompt activation capture."""
import argparse
import array
import importlib.util
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = (args.output or Path(tempfile.mkdtemp(prefix="ds4-steering-"))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("Artifacts:", out, flush=True)
    bank = array.array("f", [0.0]) * (48 * 2560)
    for layer in range(48):
        bank[layer * 2560] = 1.0
    path = out / "directions.f32"
    path.write_bytes(bank.tobytes())
    base = [str(root / "ds4"), "-m", str(args.model.resolve()),
            "--ctx", "512", "--temp", "0", "--nothink", "-p",
            "Write a three-sentence summary of the plot of Hamlet."]

    def run(name, extra, env=None):
        result = subprocess.run(base + extra, cwd=root, env=env,
                                capture_output=True, timeout=180)
        (out / f"{name}.stdout").write_bytes(result.stdout)
        (out / f"{name}.stderr").write_bytes(result.stderr)
        assert result.returncode == 0, (name, result.stderr[-2000:].decode())
        assert b"undersized buffers" not in result.stderr, name
        return result

    for component in ["ffn", "attn"]:
        flags = ["--dir-steering-file", str(path), "--dir-steering-ffn", "0",
                 f"--dir-steering-{component}", "0.1", "-n", "48"]
        plain = run(component, flags)
        mtp = run(component + "-mtp", flags + ["--mtp", "--mtp-exact-sampling", "--mtp-timing"])
        for result in (plain, mtp):
            assert b"Qwen3.8 directional steering enabled:" in result.stderr, result.stderr.decode()
        assert plain.stdout == mtp.stdout, component + " MTP changed greedy output"
        match = re.search(rb"(\d+) verify cycles, (\d+) drafts accepted", mtp.stderr)
        assert match and int(match[1]) > 0 and int(match[2]) > 0, mtp.stderr.decode()

    spec = importlib.util.spec_from_file_location("build_direction", root / "dir-steering/tools/build_direction.py")
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    for component in ["ffn_out", "attn_out"]:
        work = out / component
        work.mkdir(exist_ok=True)
        rows = capture.run_capture(root / "ds4", args.model.resolve(), "Explain a rainbow.",
                                   "", False, 512, component, 48, 2560, work)
        assert len(rows) == 48 and all(len(r) == 2560 for r in rows)
        assert all(math.isfinite(v) for r in rows for v in r)
        assert all(any(v != 0 for v in r) for r in rows)

    prompt_tokens = int(re.search(rb"processing (\d+) input tokens", plain.stderr)[1])
    assert prompt_tokens > 4
    # A one-token prefill tail must be dumped; multi-row speculative decode
    # must not overwrite the prompt activation files. Keep MTP on in both runs.
    for count in [1, 32]:
        work = out / f"dump-{count}"
        work.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.update(DS4_METAL_GRAPH_DUMP_PREFIX=str(work / "dump"),
                   DS4_METAL_GRAPH_DUMP_NAME="ffn_out", DS4_METAL_GRAPH_DUMP_POS="0",
                   DS4_QWEN4_PREFILL_CHUNK=str(prompt_tokens - 1))
        run(f"dump-{count}", ["-n", str(count), "--mtp", "--mtp-exact-sampling"], env)
    for layer in range(48):
        name = f"dump_ffn_out-{layer}_pos0.bin"
        before = (out / "dump-1" / name).read_bytes()
        assert len(before) == 2560 * 4
        assert before == (out / "dump-32" / name).read_bytes(), layer
    print("Qwen steering, MTP, 48-layer captures and prompt-only dumps: OK")


if __name__ == "__main__":
    main()
