#!/usr/bin/env python3
"""Restore a page-aligned V4.1 GGUF from verified standard GGUF split files.

The downloader verifies each input's published SHA-256. This helper preserves
all tensor bytes and metadata, removes split metadata, restores 16 KiB alignment,
and verifies the original whole-file checksum before publishing the output.
Inputs are retained; an interrupted .assembling file can be resumed.
"""
import argparse
import fcntl
import hashlib
import math
import os
from pathlib import Path
import shutil
import struct
import sys

ALIGNMENT = 16384
CHUNK = 16 << 20
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
           7: "?", 10: "Q", 11: "q", 12: "d"}
TYPES = {0: (1, 4), 1: (1, 2), 8: (32, 34), 24: (1, 1), 39: (32, 17)}
SPLIT_KEYS = {"split.no", "split.count", "split.tensors.count"}


def align(n, alignment):
    return (n + alignment - 1) // alignment * alignment


def read(fp, n):
    data = fp.read(n)
    if len(data) != n:
        raise ValueError("Truncated GGUF")
    return data


def scalar(fp, kind):
    return struct.unpack("<" + kind, read(fp, struct.calcsize("<" + kind)))[0]


def string(fp):
    size = scalar(fp, "Q")
    if size > 128 << 20:
        raise ValueError("GGUF string exceeds header limit")
    return read(fp, size).decode("utf-8")


def skip_value(fp, kind, file_size, depth=0):
    if depth > 4:
        raise ValueError("Nested GGUF metadata exceeds limit")
    if kind == 8:
        fp.seek(scalar(fp, "Q"), 1)
    elif kind == 9:
        element, count = scalar(fp, "I"), scalar(fp, "Q")
        if element in SCALARS:
            fp.seek(count * struct.calcsize("<" + SCALARS[element]), 1)
        else:
            if count > file_size:
                raise ValueError("Invalid GGUF array length")
            for _ in range(count):
                skip_value(fp, element, file_size, depth + 1)
    elif kind in SCALARS:
        fp.seek(struct.calcsize("<" + SCALARS[kind]), 1)
    else:
        raise ValueError("Unsupported GGUF metadata type")
    if fp.tell() > min(file_size, 128 << 20):
        raise ValueError("Invalid GGUF metadata extent")


def read_header(fp, file_size):
    if read(fp, 4) != b"GGUF" or scalar(fp, "I") != 3:
        raise ValueError("Expected GGUF v3")
    nt, nk = scalar(fp, "Q"), scalar(fp, "Q")
    if nt > 100000 or nk > 100000:
        raise ValueError("Invalid GGUF directory count")
    metadata, values, kinds, tensors = [], {}, {}, []
    for _ in range(nk):
        start = fp.tell()
        key, kind = string(fp), scalar(fp, "I")
        if key in kinds:
            raise ValueError("Duplicate GGUF metadata key")
        kinds[key] = kind
        value_start = fp.tell()
        skip_value(fp, kind, file_size)
        end = fp.tell()
        if kind in SCALARS or kind == 8:
            fp.seek(value_start)
            values[key] = string(fp) if kind == 8 else scalar(fp, SCALARS[kind])
        fp.seek(start)
        metadata.append((key, read(fp, end - start)))
    alignment = values.get("general.alignment", 32)
    if (not isinstance(alignment, int) or not 1 <= alignment <= ALIGNMENT or
            alignment & (alignment - 1)):
        raise ValueError("Invalid GGUF alignment")
    offset = 0
    for _ in range(nt):
        name, rank = string(fp), scalar(fp, "I")
        if not 1 <= rank <= 4:
            raise ValueError("Invalid tensor rank")
        shape = [scalar(fp, "Q") for _ in range(rank)]
        kind, relative = scalar(fp, "I"), scalar(fp, "Q")
        if kind not in TYPES or any(n == 0 for n in shape):
            raise ValueError("Unsupported MXFP4 model tensor")
        block, size = TYPES[kind]
        if shape[0] % block or relative != offset:
            raise ValueError("Invalid tensor shape or offset")
        nbytes = math.prod(shape) // block * size
        tensors.append((name, shape, kind, relative, nbytes))
        offset += align(nbytes, alignment)
    data_offset = align(fp.tell(), alignment)
    if data_offset + offset != file_size:
        raise ValueError("Incorrect shard file size")
    return metadata, values, kinds, tensors, data_offset


def encoded_string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def merge_plan(headers):
    all_tensors, names = [], set()
    for number, (metadata, values, kinds, tensors, _) in enumerate(headers):
        if (kinds.get("split.no") != 2 or kinds.get("split.count") != 2 or
                kinds.get("split.tensors.count") != 5 or
                values.get("split.no") != number or
                values.get("split.count") != len(headers)):
            raise ValueError("Incorrect shard order/count or split metadata")
        if number and any(key not in SPLIT_KEYS for key, _ in metadata):
            raise ValueError("Unexpected metadata outside first shard")
        for tensor in tensors:
            if tensor[0] in names:
                raise ValueError("Duplicate tensor across shards")
            names.add(tensor[0])
            all_tensors.append(tensor)
    if any(h[1].get("split.tensors.count") != len(all_tensors) for h in headers):
        raise ValueError("Incomplete GGUF shard set")
    first = headers[0]
    if (first[1].get("general.architecture") != "deepseek41" or
            first[2].get("general.alignment") != 4):
        raise ValueError("Expected DeepSeek V4.1 metadata and explicit alignment")
    metadata = []
    for key, raw in first[0]:
        if key in SPLIT_KEYS:
            continue
        if key.startswith("split."):
            raise ValueError("Unexpected split metadata")
        if key == "general.alignment":
            raw = encoded_string(key) + struct.pack("<II", 4, ALIGNMENT)
        metadata.append(raw)
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(all_tensors), len(metadata)))
    header.extend(b"".join(metadata))
    offset = 0
    for name, shape, kind, _, nbytes in all_tensors:
        header.extend(encoded_string(name) + struct.pack("<I", len(shape)))
        header.extend(struct.pack("<" + "Q" * len(shape), *shape))
        header.extend(struct.pack("<IQ", kind, offset))
        offset += align(nbytes, ALIGNMENT)
    header.extend(bytes(align(len(header), ALIGNMENT) - len(header)))
    return bytes(header), len(header) + offset


def verify(path, expected_bytes, expected_sha):
    if path.stat().st_size != expected_bytes:
        raise ValueError("Incorrect file size: " + str(path))
    print("Verifying SHA-256: " + str(path), flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for data in iter(lambda: fp.read(CHUNK), b""):
            digest.update(data)
    if digest.hexdigest() != expected_sha:
        raise ValueError("Checksum mismatch: " + str(path))


def merge(paths, output, expected_bytes, expected_sha):
    pending = Path(str(output) + ".assembling")
    with Path(str(pending) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            verify(output, expected_bytes, expected_sha)
            return
        headers = []
        for path in paths:
            with path.open("rb") as fp:
                headers.append(read_header(fp, os.fstat(fp.fileno()).st_size))
        header, total = merge_plan(headers)
        if total != expected_bytes:
            raise ValueError("Merged layout does not match the original GGUF size")
        completed = pending.stat().st_size if pending.exists() else 0
        if completed > total:
            raise ValueError("Invalid partial assembly; move it aside before retrying")
        if shutil.disk_usage(output.parent).free < total - completed + (1 << 30):
            raise ValueError("Not enough disk space to merge MXFP4; keep the shards and retry")
        print("Merging MXFP4 shards (retaining inputs; restoring 16 KiB alignment)", flush=True)
        with pending.open("r+b" if pending.exists() else "x+b") as dst:
            def append(data):
                # Verify an existing prefix before appending. Never truncate an
                # interrupted or unrelated file to make it fit this assembly.
                overlap = min(len(data), max(0, completed - dst.tell()))
                if overlap and read(dst, overlap) != data[:overlap]:
                    raise ValueError("Partial assembly differs from shards; move it aside before retrying")
                dst.write(data[overlap:])

            append(header)
            for path, h in zip(paths, headers):
                with path.open("rb") as src:
                    for _, _, _, offset, nbytes in h[3]:
                        src.seek(h[4] + offset)
                        left = nbytes
                        while left:
                            data = read(src, min(CHUNK, left))
                            append(data)
                            left -= len(data)
                        append(bytes(align(nbytes, ALIGNMENT) - nbytes))
            dst.flush()
            os.fsync(dst.fileno())
        verify(pending, expected_bytes, expected_sha)
        # Publish without overwriting an output created by another process.
        os.link(pending, output)
        pending.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-bytes", required=True, type=int)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        merge(args.shards, args.output, args.expected_bytes, args.sha256)
    except (OSError, ValueError, OverflowError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
