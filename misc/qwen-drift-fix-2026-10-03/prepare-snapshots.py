#!/usr/bin/env python3
"""Reconstruct the frozen source snapshots from Git; never build or run them.

Usage from a clone containing both commits:
  python3 prepare-snapshots.py NEW_DIRECTORY --repo /path/to/ds4

The output contains base0aa/ (89 files) and before-fix/ (91 files). The
working tree, historical absolute paths, binaries and build objects are not
inputs. All reconstructed bytes must match source-snapshots-manifest.json.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys


ARCHIVE = Path(__file__).resolve().parent
BASE = "0aaea5a238fb41a35106a551e73c8409dfb751ac"
BEFORE = "575369b19d8754f7eff69b9fdaa9a4c3684b825a"
MANIFEST_SHA256 = "c654d601d546842a9efc176f311d3e185b586e4092334052fcb205883a8f9ac2"
PATCH_SHA256 = "a9b73d2fe19ef4457156900d343ebb025bf1777e04b8d3ffe9c6f4de0e3d74cd"
PATCH_FILES = {"Makefile", "ds4_metal.m", "metal/dense.metal", "metal/qwen4.metal"}
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n\Z")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def relative_path(name):
    path = PurePosixPath(name)
    if (not isinstance(name, str) or not name or path.is_absolute() or
            str(path) != name or "\\" in name or ":" in name or
            any(part in (".", "..", ".git") for part in path.parts)):
        raise ValueError(f"Unsafe source path: {name!r}")
    return name


def git(repo, *args):
    process = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise RuntimeError(f"Git read failed ({' '.join(args)}): "
                           f"{process.stderr.decode('utf-8', errors='replace').strip()}")
    return process.stdout


def verify_files(files, records, label):
    if set(files) != set(records):
        raise ValueError(f"Source inventory differs for {label}")
    for name, record in records.items():
        raw = files[name]
        if len(raw) != record["bytes"] or digest(raw) != record["sha256"]:
            raise ValueError(f"Source bytes differ from manifest: {label}/{name}")


def apply_exact_patch(files, patch):
    """Apply this pinned UTF-8 unified patch with exact line/context checks.

    No fuzzy matching, Git worktree/index mutation, commands or binary patches
    are accepted. The generated patch has LF-terminated inputs and outputs.
    """
    lines = patch.decode("utf-8").splitlines(keepends=True)
    index, changed = 0, set()
    while index < len(lines):
        banner = re.fullmatch(r"diff --git a/(\S+) b/(\S+)\n", lines[index])
        if not banner or banner[1] != banner[2]:
            raise ValueError("Unsupported patch file header")
        name = relative_path(banner[1])
        if name not in PATCH_FILES or name in changed or name not in files:
            raise ValueError(f"Unexpected or duplicate patched source: {name}")
        index += 1
        if lines[index:index + 2] != [f"--- a/{name}\n", f"+++ b/{name}\n"]:
            raise ValueError(f"Unsupported patch metadata: {name}")
        index += 2
        original = files[name].decode("utf-8").splitlines(keepends=True)
        cursor, output, hunks = 0, [], 0
        while index < len(lines) and not lines[index].startswith("diff --git "):
            header = HUNK.fullmatch(lines[index])
            if not header:
                raise ValueError(f"Unsupported patch hunk: {name}")
            old_count = int(header[2]) if header[2] is not None else 1
            new_count = int(header[4]) if header[4] is not None else 1
            old_start = int(header[1]) - (old_count != 0)
            new_start = int(header[3]) - (new_count != 0)
            if not cursor <= old_start <= len(original):
                raise ValueError(f"Invalid old hunk position: {name}")
            output.extend(original[cursor:old_start])
            cursor = old_start
            if len(output) != new_start:
                raise ValueError(f"Invalid new hunk position: {name}")
            index += 1
            consumed, emitted = 0, 0
            while index < len(lines):
                line = lines[index]
                if line.startswith("@@ ") or line.startswith("diff --git "):
                    break
                if not line.endswith("\n") or line[:1] not in (" ", "+", "-"):
                    raise ValueError(f"Unsupported patch content: {name}")
                prefix, body = line[0], line[1:]
                if prefix != "+":
                    if cursor >= len(original) or original[cursor] != body:
                        raise ValueError(f"Patch context mismatch: {name}, line {cursor + 1}")
                    cursor += 1
                    consumed += 1
                if prefix != "-":
                    output.append(body)
                    emitted += 1
                index += 1
            if consumed != old_count or emitted != new_count:
                raise ValueError(f"Patch hunk counts differ: {name}")
            hunks += 1
        output.extend(original[cursor:])
        result = "".join(output).encode("utf-8")
        if not hunks or result == files[name]:
            raise ValueError(f"Patch did not change source: {name}")
        files[name] = result
        changed.add(name)
    if changed != PATCH_FILES:
        raise ValueError("Patch source inventory differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new directory; existing paths are refused")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="clone with both frozen commits")
    args = parser.parse_args()
    destination = args.destination.absolute()
    if os.path.lexists(destination):
        raise ValueError(f"Refusing existing destination: {destination}")

    manifest_raw = (ARCHIVE / "source-snapshots-manifest.json").read_bytes()
    patch = (ARCHIVE / "pending-input.patch").read_bytes()
    if digest(manifest_raw) != MANIFEST_SHA256 or digest(patch) != PATCH_SHA256:
        raise ValueError("Archived manifest or pending-input.patch SHA-256 differs")
    manifest = json.loads(manifest_raw)
    if manifest["base_commit"] != BASE or manifest["working_tree_head"] != BEFORE:
        raise ValueError("Frozen commit identity differs")
    repo = args.repo.resolve()
    for commit in (BASE, BEFORE):
        if git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip() != commit:
            raise ValueError(f"Frozen commit not available: {commit}")

    snapshots = {}
    for label, commit, count in (("base0aa", BASE, 89), ("before-fix", BEFORE, 91)):
        records = manifest["snapshots"][label]["records"]
        if len(records) != count:
            raise ValueError(f"Unexpected source count: {label}")
        files = {relative_path(name): git(repo, "cat-file", "blob", f"{commit}:{name}")
                 for name in records}
        if label == "before-fix":
            apply_exact_patch(files, patch)
        verify_files(files, records, label)
        snapshots[label] = files

    # Reserve the destination exclusively after all in-memory checks pass.
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        for label, files in snapshots.items():
            for name, raw in files.items():
                target = destination / label / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(raw)
            written = {name: (destination / label / name).read_bytes() for name in files}
            verify_files(written, manifest["snapshots"][label]["records"], label)
        report = {"status": "PASS", "base_commit": BASE, "before_commit": BEFORE,
                  "manifest_sha256": MANIFEST_SHA256, "pending_patch_sha256": PATCH_SHA256,
                  "source_counts": {label: len(files) for label, files in snapshots.items()},
                  "patched_files": sorted(PATCH_FILES), "built_or_executed": False}
        (destination / "RECONSTRUCTION.json").write_text(json.dumps(report, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(destination)
        raise
    print(json.dumps({"status": "PASS", "destination": str(destination),
                      "source_counts": report["source_counts"], "built_or_executed": False}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
