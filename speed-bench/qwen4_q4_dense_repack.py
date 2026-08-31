#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface-hub>=1.0",
#   "numpy>=2.0",
# ]
# ///
"""Patch a cloned DS4 Qwen base with source-derived Q4_K projections.

The destination base must be an APFS clone of the original pack.  Q4_K
payloads are smaller than their Q8_0 slots, so tensor offsets and the total
file size remain unchanged.  Only the cloned payload pages, GGUF qtype words,
and cloned manifest are modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent / "gguf-tools"
sys.path.insert(0, os.fspath(TOOLS_DIR))

import qwen4_pack as pack  # noqa: E402


@dataclass(frozen=True)
class TensorEntry:
    name: str
    shape: tuple[int, ...]
    qtype: int
    qtype_offset: int
    relative_offset: int
    data_offset: int


def parse_directory(path: Path) -> tuple[dict[str, object], dict[str, TensorEntry]]:
    metadata: dict[str, object] = {}
    pending: list[tuple[str, tuple[int, ...], int, int, int]] = []
    with path.open("rb") as fp:
        if pack.read_exact(fp, 4, "GGUF magic") != b"GGUF":
            pack.fail(f"{path}: not a GGUF file")
        version = pack.read_u32(fp, "GGUF version")
        if version != pack.GGUF_VERSION:
            pack.fail(f"{path}: GGUF version {version} is unsupported")
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
            pending.append(
                (name, shape, qtype, qtype_offset, relative_offset)
            )
        alignment = int(metadata.get("general.alignment", pack.GGUF_ALIGNMENT))
        data_start = pack.align(fp.tell(), alignment)
    entries = {
        name: TensorEntry(
            name=name,
            shape=shape,
            qtype=qtype,
            qtype_offset=qtype_offset,
            relative_offset=relative_offset,
            data_offset=data_start + relative_offset,
        )
        for name, shape, qtype, qtype_offset, relative_offset in pending
    }
    if len(entries) != tensor_count:
        pack.fail(f"{path}: duplicate tensor names")
    return metadata, entries


ROLE_QUANTIZATION_KEYS = {
    "gdn": "gdn_projections",
    "qsa": "qsa_projections",
}


def patch_manifest(path: Path, base: Path, digests: dict[str, str],
                   roles: set[str], pack_version: int,
                   skipped: list[str]) -> None:
    manifest = json.loads(path.read_text())
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict):
        pack.fail(f"{path}: missing tensor directory")
    for name, digest in digests.items():
        record = tensors.get(name)
        if not isinstance(record, dict) or record.get("qtype") != "Q8_0":
            pack.fail(f"{path}: {name} is not a manifest Q8_0 tensor")
        record["qtype"] = "Q4_K"
        record["sha256"] = digest
    quantization = manifest.get("quantization")
    if not isinstance(quantization, dict):
        pack.fail(f"{path}: missing quantization record")
    for role in sorted(roles):
        quantization[ROLE_QUANTIZATION_KEYS[role]] = {"qtype": "Q4_K"}
    manifest["experimental_source_derived_q4_dense"] = {
        "roles": sorted(roles),
        "base_layout": f"v{pack_version}-q8-slots",
        "q8_block_incompatible_tensors": sorted(skipped),
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
        "--roles", default="gdn,qsa",
        help=("comma-separated Q8_0 roles to replace with Q4_K: "
              "gdn,qsa"),
    )
    parser.add_argument("--quants-library", type=Path,
                        default=TOOLS_DIR / "libds4quants.dylib")
    args = parser.parse_args()
    for path in (args.base, args.manifest):
        if not path.is_file():
            parser.error(f"missing {path}")

    roles = {role.strip() for role in args.roles.split(",") if role.strip()}
    unknown_roles = roles - ROLE_QUANTIZATION_KEYS.keys()
    if not roles or unknown_roles:
        parser.error(
            "--roles must contain one or more of "
            f"{','.join(ROLE_QUANTIZATION_KEYS)}; unknown={sorted(unknown_roles)}"
        )

    source = pack.SourceDB(args.src)
    pack.validate_source_config(source.config)
    metadata, entries = parse_directory(args.base)
    pack_version = metadata.get("ds4.pack.version")
    profiles = {
        profile.pack_version: profile for profile in pack.PACK_PROFILES.values()
    }
    profile = profiles.get(pack_version)
    if profile is None:
        pack.fail(f"destination base has unsupported pack version {pack_version}")
    plan = pack.build_plan(source, profile, include_mtp=False)
    quantizer = pack.GGMLQuantizer(args.quants_library)
    actions = [
        action for action in plan["base"]
        if action.kind == "quant" and action.role in roles
    ]
    if not actions:
        pack.fail(f"conversion plan has no projections for roles {sorted(roles)}")

    digests: dict[str, str] = {}
    skipped: list[str] = []
    total_q8 = 0
    total_q4 = 0
    fd = os.open(args.base, os.O_RDWR)
    try:
        for ordinal, action in enumerate(actions, 1):
            if len(action.specs) != 1:
                pack.fail(f"{action.source}: expected one dense output")
            spec = action.specs[0]
            entry = entries.get(spec.name)
            if entry is None or entry.qtype != pack.QTYPE_Q8_0:
                pack.fail(f"{spec.name}: destination is not Q8_0")
            if entry.shape[0] % 256:
                skipped.append(spec.name)
                print(
                    f"skip {action.role} {spec.name}: input width "
                    f"{entry.shape[0]} is not Q4_K block aligned",
                    flush=True,
                )
                continue
            values = source.read(action.source)
            raw = quantizer.encode(values, "Q4_K")
            expected = pack.product(values.shape) // 256 * 144
            if len(raw) != expected or pack.product(entry.shape) != values.size:
                pack.fail(f"{spec.name}: source/destination geometry mismatch")
            written = os.pwrite(fd, raw, entry.data_offset)
            if written != len(raw):
                pack.fail(f"{spec.name}: short Q4_K payload write")
            if os.pwrite(
                    fd, struct.pack("<I", pack.QTYPE_Q4_K),
                    entry.qtype_offset) != 4:
                pack.fail(f"{spec.name}: short qtype write")
            digest = hashlib.sha256(raw).hexdigest()
            digests[spec.name] = digest
            total_q8 += pack.product(values.shape) // 32 * 34
            total_q4 += len(raw)
            print(
                f"{ordinal}/{len(actions)} {action.role} {spec.name} "
                f"{len(raw) / (1 << 20):.2f} MiB",
                flush=True,
            )
        os.fsync(fd)
    finally:
        os.close(fd)

    patch_manifest(
        args.manifest, args.base, digests, roles, pack_version, skipped
    )
    print(
        f"patched {len(digests)} tensors: "
        f"Q8_0={total_q8 / (1 << 30):.3f} GiB "
        f"Q4_K={total_q4 / (1 << 30):.3f} GiB "
        f"saved-logical={((total_q8 - total_q4) / (1 << 30)):.3f} GiB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"qwen4-q4-dense-repack: {error}", file=sys.stderr)
        raise SystemExit(1)
