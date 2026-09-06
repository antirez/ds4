#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface-hub>=1.0",
#   "numpy>=2.0",
# ]
# ///
"""Patch an APFS-cloned Qwen Q4 pack with source-derived Q4_0 experts.

Q4_0 and Q4_K both occupy 144 bytes per 256 values, so every routed tensor
keeps its existing offset and byte extent.  The repack changes only the
payload, tensor-directory qtype words, and cloned manifest; the result is
the standard Q4_0-routed profile.  Packs repacked before the profile was
standardized carry the same provenance under the older
"experimental_source_derived_q4_0_routed" manifest key, which the loader
ignores either way.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent / "gguf-tools"
sys.path.insert(0, os.fspath(TOOLS_DIR))

import qwen4_pack as pack  # noqa: E402

QTYPE_Q4_0 = 2


@dataclass(frozen=True)
class TensorEntry:
    name: str
    shape: tuple[int, ...]
    qtype: int
    qtype_offset: int
    data_offset: int


def parse_directory(path: Path) -> tuple[dict[str, object], dict[str, TensorEntry]]:
    metadata: dict[str, object] = {}
    pending: list[tuple[str, tuple[int, ...], int, int, int]] = []
    with path.open("rb") as fp:
        if pack.read_exact(fp, 4, "GGUF magic") != b"GGUF":
            pack.fail(f"{path}: not a GGUF file")
        version = pack.read_u32(fp, "GGUF version")
        if version != pack.GGUF_VERSION:
            pack.fail(f"{path}: unsupported GGUF version {version}")
        tensor_count = pack.read_u64(fp, "GGUF tensor count")
        metadata_count = pack.read_u64(fp, "GGUF metadata count")
        for _ in range(metadata_count):
            key = pack.read_gguf_string(fp, "GGUF metadata key")
            value_type = pack.read_u32(fp, f"GGUF metadata type for {key}")
            metadata[key] = pack.read_gguf_value(
                fp, value_type, f"GGUF metadata {key}"
            )
        for _ in range(tensor_count):
            name = pack.read_gguf_string(fp, "GGUF tensor name")
            rank = pack.read_u32(fp, f"GGUF rank for {name}")
            shape = tuple(
                pack.read_u64(fp, f"GGUF shape for {name}")
                for _ in range(rank)
            )
            qtype_offset = fp.tell()
            qtype = pack.read_u32(fp, f"GGUF qtype for {name}")
            relative_offset = pack.read_u64(fp, f"GGUF offset for {name}")
            pending.append((name, shape, qtype, qtype_offset, relative_offset))
        alignment = int(metadata.get("general.alignment", pack.GGUF_ALIGNMENT))
        data_start = pack.align(fp.tell(), alignment)
    entries = {
        name: TensorEntry(name, shape, qtype, qtype_offset,
                          data_start + relative_offset)
        for name, shape, qtype, qtype_offset, relative_offset in pending
    }
    if len(entries) != tensor_count:
        pack.fail(f"{path}: duplicate tensor names")
    return metadata, entries


def quantize_q4_0_rows(rows: np.ndarray) -> bytes:
    if rows.ndim != 2 or rows.shape[1] % 32:
        pack.fail(f"invalid Q4_0 row geometry {rows.shape}")
    if rows.dtype == np.dtype("<u2"):
        values = pack.bf16_to_f32(rows)
    else:
        values = np.asarray(rows, dtype="<f4")
    blocks = np.ascontiguousarray(values, dtype="<f4").reshape(-1, 32)
    indices = np.abs(blocks).argmax(axis=1)
    signed_max = blocks[np.arange(blocks.shape[0]), indices]
    delta = signed_max / np.float32(-8.0)
    inverse = np.zeros_like(delta)
    np.divide(np.float32(1.0), delta, out=inverse, where=delta != 0.0)
    codes = np.clip(
        np.trunc(blocks * inverse[:, None] + np.float32(8.5)),
        0, 15,
    ).astype(np.uint8)
    packed = codes[:, :16] | (codes[:, 16:] << np.uint8(4))
    output = np.empty((blocks.shape[0], 18), dtype=np.uint8)
    output[:, :2] = delta.astype("<f2").view(np.uint8).reshape(-1, 2)
    output[:, 2:] = packed
    return output.tobytes()


def write_tensor(fd: int, entry: TensorEntry, source: np.ndarray,
                 executor: concurrent.futures.ThreadPoolExecutor,
                 threads: int, physical_width: int = 0) -> str:
    if entry.qtype != pack.QTYPE_Q4_K or source.ndim != 3:
        pack.fail(f"{entry.name}: expected a three-dimensional Q4_K tensor")
    width = physical_width or source.shape[-1]
    if width < source.shape[-1] or width % 32:
        pack.fail(f"{entry.name}: invalid physical width {width}")
    rows_per_expert = source.shape[1]
    row_bytes = width // 32 * 18
    expected = source.shape[0] * rows_per_expert * row_bytes
    if pack.product(entry.shape) != source.size + (
            width - source.shape[-1]) * source.shape[0] * rows_per_expert:
        pack.fail(f"{entry.name}: source/destination geometry mismatch")
    digest = hashlib.sha256()
    def convert_expert(expert: int) -> bytes:
        encoded = bytearray()
        for row0 in range(0, rows_per_expert, 256):
            chunk = source[expert, row0:row0 + 256]
            if width != source.shape[-1]:
                padded = np.zeros((chunk.shape[0], width), dtype=source.dtype)
                padded[:, :source.shape[-1]] = chunk
                chunk = padded
            encoded += quantize_q4_0_rows(chunk)
        return bytes(encoded)

    written_total = 0
    window = max(threads * 2, 1)
    for start in range(0, source.shape[0], window):
        futures = [
            executor.submit(convert_expert, expert)
            for expert in range(start, min(start + window, source.shape[0]))
        ]
        for relative, future in enumerate(futures):
            expert = start + relative
            raw = future.result()
            offset = entry.data_offset + (
                expert * rows_per_expert
            ) * row_bytes
            if os.pwrite(fd, raw, offset) != len(raw):
                pack.fail(f"{entry.name}: short Q4_0 payload write")
            digest.update(raw)
            written_total += len(raw)
    if written_total != expected:
        pack.fail(
            f"{entry.name}: wrote {written_total} bytes, expected {expected}"
        )
    if os.pwrite(fd, struct.pack("<I", QTYPE_Q4_0), entry.qtype_offset) != 4:
        pack.fail(f"{entry.name}: short qtype write")
    return digest.hexdigest()


def patch_manifest(path: Path, base: Path, digests: dict[str, str]) -> None:
    manifest = json.loads(path.read_text())
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict):
        pack.fail(f"{path}: missing tensor directory")
    for name, digest in digests.items():
        record = tensors.get(name)
        if not isinstance(record, dict) or record.get("qtype") != "Q4_K":
            pack.fail(f"{path}: {name} is not a manifest Q4_K tensor")
        record["qtype"] = "Q4_0"
        record["sha256"] = digest
    quantization = manifest.get("quantization")
    if isinstance(quantization, dict):
        # The converter wrote the recipe for its Q4_K output; the repacked
        # pack's routed experts are source-derived Q4_0, and leaving the
        # stale recipe text behind invites exactly the "is this really
        # Q4_0?" question.  The loader never parses this block (per-tensor
        # records and tensor_manifest_sha256 are authoritative).
        quantization["routed_experts"] = {
            "qtype": "Q4_0",
            "block_size": 32,
            "note": "source-derived Q4_0 repacked in place over the "
                    "original Q4_K byte extents "
                    f"({len(digests)} routed tensors); per-tensor qtype "
                    "records and tensor_manifest_sha256 are authoritative",
        }
    manifest["source_derived_q4_0_routed"] = {
        "tensor_count": len(digests),
        "base_layout": "Q4_K-equal-byte-slots",
    }
    tensor_manifest = json.dumps(
        tensors, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest["tensor_manifest_sha256"] = hashlib.sha256(
        tensor_manifest
    ).hexdigest()
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        pack.fail(f"{path}: missing artifact directory")
    matched = 0
    for artifact in artifacts:
        if isinstance(artifact, dict) and artifact.get("kind") == "base":
            if artifact.get("path") != base.name:
                pack.fail(f"{path}: base artifact name mismatch")
            artifact["bytes"] = base.stat().st_size
            artifact["sha256"] = pack.sha256_file(base)
            matched += 1
    if matched != 1:
        pack.fail(f"{path}: expected exactly one base artifact")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--threads", type=int, default=min(os.cpu_count() or 1, 8),
        help="parallel expert quantization workers (default: up to 8)",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    for path in (args.base, args.manifest):
        if not path.is_file():
            parser.error(f"missing {path}")

    source = pack.SourceDB(args.src)
    pack.validate_source_config(source.config)
    metadata, entries = parse_directory(args.base)
    pack_version = metadata.get("ds4.pack.version")
    profile = next(
        (candidate for candidate in pack.PACK_PROFILES.values()
         if candidate.pack_version == pack_version),
        None,
    )
    if profile is None or profile.name != "q4":
        pack.fail(f"destination is not a Q4 pack: version={pack_version}")
    actions = [
        action for action in pack.build_plan(
            source, profile, include_mtp=False
        )["base"]
        if action.role == "routed_expert"
    ]
    if len(actions) != 96:
        pack.fail(f"expected 96 routed source actions, found {len(actions)}")

    digests: dict[str, str] = {}
    fd = os.open(args.base, os.O_RDWR)
    try:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.threads) as executor:
            for ordinal, action in enumerate(actions, 1):
                values = source.read(action.source)
                if action.kind == "split_gate_up":
                    half = values.shape[1] // 2
                    sources = (values[:, :half, :], values[:, half:, :])
                elif action.kind == "quant":
                    sources = (values,)
                else:
                    pack.fail(f"{action.source}: unexpected action {action.kind}")
                if len(sources) != len(action.specs):
                    pack.fail(f"{action.source}: source/spec count mismatch")
                for spec, tensor_source in zip(
                        action.specs, sources, strict=True):
                    entry = entries.get(spec.name)
                    if entry is None:
                        pack.fail(f"{spec.name}: missing destination tensor")
                    digest = write_tensor(
                        fd, entry, tensor_source, executor, args.threads,
                        action.pad_last_to if len(sources) == 1 else 0,
                    )
                    digests[spec.name] = digest
                    print(
                        f"{ordinal}/{len(actions)} {spec.name} -> Q4_0",
                        flush=True,
                    )
                del values
        os.fsync(fd)
    finally:
        os.close(fd)

    if len(digests) != 144:
        pack.fail(f"expected 144 routed tensors, patched {len(digests)}")
    patch_manifest(args.manifest, args.base, digests)
    print(f"patched {len(digests)} source-derived Q4_0 routed tensors")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"qwen4-q4-0-routed-repack: {error}", file=sys.stderr)
        raise SystemExit(1)
