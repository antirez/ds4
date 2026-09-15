#!/usr/bin/env python3
"""Validate a GLM-5.3 Flash GGUF against its pinned source conversion plan."""

from __future__ import annotations

import argparse
import os
import struct

from glm53_quantize import (
    GGUF_ALIGNMENT,
    GGUF_ARRAY,
    GGUF_BOOL,
    GGUF_FLOAT32,
    GGUF_STRING,
    GGUF_UINT32,
    GGUF_UINT64,
    QTYPE_NAMES,
    SourceDB,
    align,
    build_plan,
    fail,
    iter_native_tensor_bytes,
    read_exact,
    read_gguf_string,
    read_u32,
    read_u64,
    skip_gguf_value,
)


def read_selected_metadata(fp, value_type):
    if value_type == GGUF_STRING:
        return read_gguf_string(fp, "GGUF metadata string")
    if value_type == GGUF_UINT32:
        return read_u32(fp, "GGUF metadata uint32")
    if value_type == GGUF_UINT64:
        return read_u64(fp, "GGUF metadata uint64")
    if value_type == GGUF_FLOAT32:
        return struct.unpack("<f", read_exact(fp, 4, "GGUF metadata float32"))[0]
    if value_type == GGUF_BOOL:
        return bool(read_exact(fp, 1, "GGUF metadata bool")[0])
    skip_gguf_value(fp, value_type)
    return None


def verify_native_payload(fp, data_offset, expected, db):
    try:
        import numpy as np
    except ImportError as error:
        fail(f"NumPy is required for payload verification: {error}")
    verified = 0
    for index, item in enumerate(expected, 1):
        fp.seek(data_offset + item.offset)
        item_bytes = 0
        for source_data in iter_native_tensor_bytes(item, db, np):
            output_data = read_exact(fp, len(source_data), item.name)
            if output_data != source_data:
                fail(f"{item.name}: output payload differs from the pinned source")
            item_bytes += len(source_data)
        if item_bytes != item.nbytes:
            fail(f"{item.name}: verified {item_bytes} bytes, expected {item.nbytes}")
        padding = read_exact(fp, align(item.nbytes) - item.nbytes, f"{item.name} padding")
        if any(padding):
            fail(f"{item.name}: nonzero alignment padding")
        verified += item_bytes
        if index % 100 == 0 or index == len(expected):
            print(f"verified payloads: {index}/{len(expected)} ({verified} bytes)")
    return verified


def validate(path, hf_dir, artifact, source_revision, verify_payload=False):
    db = SourceDB(hf_dir)
    expected = build_plan(db, artifact)

    with open(path, "rb") as fp:
        if read_exact(fp, 4, "GGUF magic") != b"GGUF":
            fail(f"{path}: not a GGUF file")
        version = read_u32(fp, "GGUF version")
        if version != 3:
            fail(f"{path}: expected GGUF v3, got v{version}")
        tensor_count = read_u64(fp, "GGUF tensor count")
        metadata_count = read_u64(fp, "GGUF metadata count")
        if tensor_count != len(expected):
            fail(f"tensor count {tensor_count} != expected {len(expected)}")

        selected = {}
        wanted = {
            "general.architecture",
            "general.alignment",
            "general.source.revision",
            "glm5-next.native_fp8",
        }
        for _ in range(metadata_count):
            key = read_gguf_string(fp, "GGUF metadata key")
            value_type = read_u32(fp, "GGUF metadata type")
            if key in wanted:
                selected[key] = read_selected_metadata(fp, value_type)
            else:
                skip_gguf_value(fp, value_type)

        if selected.get("general.architecture") != "glm5-next":
            fail(f"unexpected architecture: {selected.get('general.architecture')!r}")
        if selected.get("general.alignment") != GGUF_ALIGNMENT:
            fail(f"unexpected alignment: {selected.get('general.alignment')!r}")
        if selected.get("general.source.revision") != source_revision:
            fail(f"unexpected source revision: {selected.get('general.source.revision')!r}")
        if artifact == "fp8" and selected.get("glm5-next.native_fp8") is not True:
            fail("native FP8 artifact is missing its format marker")
        if artifact != "fp8" and selected.get("glm5-next.native_fp8") is not None:
            fail("quantized artifact unexpectedly has the native FP8 format marker")

        seen = set()
        for index, item in enumerate(expected):
            name = read_gguf_string(fp, f"tensor {index} name")
            n_dims = read_u32(fp, f"tensor {index} rank")
            shape = tuple(read_u64(fp, f"tensor {index} dimension") for _ in range(n_dims))
            qtype = read_u32(fp, f"tensor {index} type")
            offset = read_u64(fp, f"tensor {index} offset")
            if name in seen:
                fail(f"duplicate tensor name: {name}")
            seen.add(name)
            if name != item.name:
                fail(f"tensor {index} name {name!r} != expected {item.name!r}")
            if shape != item.shape:
                fail(f"{name}: shape {shape} != expected {item.shape}")
            if qtype != item.qtype:
                fail(
                    f"{name}: type {QTYPE_NAMES.get(qtype, qtype)} != "
                    f"expected {QTYPE_NAMES[item.qtype]}"
                )
            if offset != item.offset:
                fail(f"{name}: offset {offset} != expected {item.offset}")

        data_offset = align(fp.tell())
        expected_size = data_offset
        if expected:
            last = expected[-1]
            expected_size += last.offset + align(last.nbytes)
        actual_size = os.fstat(fp.fileno()).st_size
        if actual_size != expected_size:
            fail(f"file size {actual_size} != expected {expected_size}")

        role_bytes = {}
        type_bytes = {}
        for item in expected:
            role_bytes[item.role] = role_bytes.get(item.role, 0) + item.nbytes
            type_name = QTYPE_NAMES[item.qtype]
            type_bytes[type_name] = type_bytes.get(type_name, 0) + item.nbytes

        verified_bytes = 0
        if verify_payload:
            if artifact != "fp8":
                fail("payload verification currently applies only to the native FP8 artifact")
            verified_bytes = verify_native_payload(fp, data_offset, expected, db)

    db.close()

    print(
        f"validated {path}: {len(expected)} tensors, {actual_size} bytes, "
        f"data offset {data_offset}"
    )
    print("types: " + ", ".join(f"{key}={value}" for key, value in sorted(type_bytes.items())))
    print("roles: " + ", ".join(f"{key}={value}" for key, value in sorted(role_bytes.items())))
    if verify_payload:
        print(f"source payloads matched byte for byte: {verified_bytes} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True, help="official GLM-5.3 Flash snapshot")
    parser.add_argument("--gguf", required=True, help="GGUF to validate")
    parser.add_argument("--artifact", choices=("q4", "q2", "fp8"), required=True)
    parser.add_argument("--verify-payload", action="store_true")
    parser.add_argument(
        "--source-revision",
        default="84c6a6aa9497188e15a635ba793b0f95a79b1033",
    )
    args = parser.parse_args()
    validate(args.gguf, args.hf, args.artifact, args.source_revision, args.verify_payload)


if __name__ == "__main__":
    main()
