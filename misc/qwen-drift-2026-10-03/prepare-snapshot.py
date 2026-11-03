"""Recreate the frozen diagnostic sources without changing the working tree."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tempfile

lane = Path(__file__).resolve().parent
repo = lane.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="head-rebuilt",
                    help="Fresh snapshot directory, relative to this archive or absolute")
args = parser.parse_args()
destination = (lane / args.output).resolve()
manifest = json.loads((lane / "snapshot-manifest.json").read_text())
expected = json.loads((lane / "snapshot-source-sha256.json").read_text())
paths = manifest["paths"]
if set(paths) != set(expected) or any(Path(p).is_absolute() or ".." in Path(p).parts for p in paths):
    raise SystemExit("Invalid snapshot manifest")
if destination.exists():
    raise SystemExit(f"Refusing to overwrite existing snapshot: {destination}")
subprocess.run(["git", "cat-file", "-e", manifest["commit"] + "^{commit}"],
               cwd=repo, check=True)
destination.mkdir(parents=True)
with tempfile.TemporaryDirectory(prefix="qwen-drift-") as temporary:
    archive = Path(temporary) / "sources.tar"
    subprocess.run(["git", "archive", "--format=tar", "--output", str(archive),
                    manifest["commit"], "--", *paths], cwd=repo, check=True)
    subprocess.run(["tar", "-xf", str(archive), "-C", str(destination)], check=True)
subprocess.run(["patch", "--batch", "--forward", "-p1", "-i", str(lane / "diagnostic.patch")],
               cwd=destination, check=True)
for name, digest in expected.items():
    actual = hashlib.sha256((destination / name).read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"Snapshot source mismatch: {name}")
print(f"Verified {len(expected)} frozen source files at {destination}")
print(f"Build with: make -C '{destination}' -j4 ds4")
