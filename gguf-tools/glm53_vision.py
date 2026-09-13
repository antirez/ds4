#!/usr/bin/env python3
"""Create or validate the standalone GLM-5.3 Flash vision encoder GGUF."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import struct
import sys

from glm53_quantize import (
    GGUF_ALIGNMENT,
    GGUF_ARRAY,
    GGUF_FLOAT32,
    GGUF_STRING,
    GGUF_UINT32,
    GGUF_UINT64,
    GGUF_VERSION,
    QTYPE_BF16,
    QTYPE_NAMES,
    SourceDB,
    TensorPlan,
    align,
    fail,
    kv_f32,
    kv_string,
    kv_u32,
    kv_u64,
    pack_string,
    qtype_nbytes,
    read_exact,
    read_gguf_string,
    read_u32,
    read_u64,
    skip_gguf_value,
    tensor_header,
)


DEFAULT_SOURCE_REVISION = "84c6a6aa9497188e15a635ba793b0f95a79b1033"
ARCHITECTURE = "glm5-next-vision"


def kv_f32_array(key, values):
    return (
        pack_string(key)
        + struct.pack("<IIQ", GGUF_ARRAY, GGUF_FLOAT32, len(values))
        + struct.pack(f"<{len(values)}f", *values)
    )


def load_source_config(hf_dir):
    with open(os.path.join(hf_dir, "config.json"), encoding="utf-8") as fp:
        config = json.load(fp)
    with open(os.path.join(hf_dir, "processor_config.json"), encoding="utf-8") as fp:
        processor = json.load(fp)["image_processor"]
    vision = config["vision_config"]

    required = {
        "depth": 24,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_heads": 16,
        "out_hidden_size": 4096,
        "patch_size": 14,
        "projection_intermediate_size": 10240,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    }
    for key, expected in required.items():
        if vision.get(key) != expected:
            fail(f"unexpected vision_config.{key}: {vision.get(key)!r}")
    if processor.get("patch_size") != vision["patch_size"]:
        fail("processor and model patch sizes differ")
    if processor.get("merge_size") != vision["spatial_merge_size"]:
        fail("processor and model merge sizes differ")
    if processor.get("temporal_patch_size") != vision["temporal_patch_size"]:
        fail("processor and model temporal patch sizes differ")
    return config, vision, processor


def build_vision_plan(db):
    plan = []
    offset = 0
    for name in sorted(name for name in db.tensors if name.startswith("model.visual.")):
        info = db.info(name)
        if info["dtype"] != "BF16":
            fail(f"{name}: expected BF16 source, got {info['dtype']}")
        if ".blocks." in name:
            role = "block"
        elif ".patch_embed." in name:
            role = "patch_embedding"
        elif ".merger." in name:
            role = "merger"
        elif ".downsample." in name:
            role = "downsample"
        else:
            role = "normalization"
        item = TensorPlan(
            name=name,
            shape=tuple(reversed(info["shape"])),
            qtype=QTYPE_BF16,
            role=role,
            source=name,
            raw_copy=True,
        )
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
        if item.nbytes != info["nbytes"]:
            fail(f"{name}: planned {item.nbytes} bytes, source has {info['nbytes']}")
        item.offset = offset
        offset += align(item.nbytes)
        plan.append(item)

    if len(plan) != 347:
        fail(f"expected 347 vision tensors, found {len(plan)}")
    return plan


def vision_metadata(hf_dir, source_revision):
    config, vision, processor = load_source_config(hf_dir)
    prefix = "glm5-next-vision"
    records = [
        kv_string("general.architecture", ARCHITECTURE),
        kv_string("general.name", "GLM-5.3-Flash Vision Encoder"),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string("general.source.url", "https://huggingface.co/zai-org/GLM-5.3-Flash"),
        kv_string("general.source.revision", source_revision),
        kv_u32(f"{prefix}.block_count", vision["depth"]),
        kv_u32(f"{prefix}.embedding_length", vision["hidden_size"]),
        kv_u32(f"{prefix}.feed_forward_length", vision["intermediate_size"]),
        kv_u32(f"{prefix}.attention.head_count", vision["num_heads"]),
        kv_u32(f"{prefix}.projection_length", vision["out_hidden_size"]),
        kv_u32(f"{prefix}.projection.feed_forward_length", vision["projection_intermediate_size"]),
        kv_u32(f"{prefix}.patch_size", vision["patch_size"]),
        kv_u32(f"{prefix}.temporal_patch_size", vision["temporal_patch_size"]),
        kv_u32(f"{prefix}.spatial_merge_size", vision["spatial_merge_size"]),
        kv_u32(f"{prefix}.channel_count", vision["in_channels"]),
        kv_f32(f"{prefix}.attention.layer_norm_rms_epsilon", vision["rms_norm_eps"]),
        kv_f32(f"{prefix}.swiglu_limit", vision["swiglu_limit"]),
        kv_u32(f"{prefix}.image_token_id", config["image_token_id"]),
        kv_u32(f"{prefix}.image_start_token_id", config["image_start_token_id"]),
        kv_u32(f"{prefix}.image_end_token_id", config["image_end_token_id"]),
        kv_u32(f"{prefix}.image.min_tokens", processor["min_image_tokens"]),
        kv_u32(f"{prefix}.image.max_tokens", processor["max_image_tokens"]),
        kv_f32_array(f"{prefix}.image.mean", processor["image_mean"]),
        kv_f32_array(f"{prefix}.image.std", processor["image_std"]),
    ]
    return records


def layout(plan, metadata):
    header_bytes = 4 + 4 + 8 + 8
    header_bytes += sum(len(record) for record in metadata)
    header_bytes += sum(len(tensor_header(item)) for item in plan)
    data_offset = align(header_bytes)
    data_bytes = sum(align(item.nbytes) for item in plan)
    return data_offset, data_bytes


def print_summary(plan, metadata):
    data_offset, data_bytes = layout(plan, metadata)
    print(f"tensors: {len(plan)}")
    print(f"metadata_records: {len(metadata)}")
    print(f"metadata_bytes: {data_offset}")
    print(f"tensor_bytes: {sum(item.nbytes for item in plan)}")
    print(f"file_bytes: {data_offset + data_bytes}")
    roles = {}
    for item in plan:
        roles[item.role] = roles.get(item.role, 0) + item.nbytes
    for role, size in sorted(roles.items()):
        print(f"role_bytes: {role} {size}")
    return data_offset, data_bytes


def create_gguf(path, plan, metadata, db, overwrite):
    data_offset, data_bytes = print_summary(plan, metadata)
    required = data_offset + data_bytes + 4 * (1 << 30)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free
    if free < required:
        fail(f"insufficient free space: need output plus reserve {required}, have {free}")
    if os.path.exists(path) and not overwrite:
        fail(f"output exists: {path}; use --overwrite")

    partial = path + ".partial"
    if os.path.exists(partial):
        if not overwrite:
            fail(f"partial output exists: {partial}; use --overwrite")
        os.unlink(partial)

    with open(partial, "wb") as fp:
        fp.write(b"GGUF")
        fp.write(struct.pack("<IQQ", GGUF_VERSION, len(plan), len(metadata)))
        for record in metadata:
            fp.write(record)
        for item in plan:
            fp.write(tensor_header(item))
        if fp.tell() > data_offset:
            fail("GGUF header exceeds its planned data offset")
        fp.write(bytes(data_offset - fp.tell()))

        for index, item in enumerate(plan, 1):
            if fp.tell() != data_offset + item.offset:
                fail(f"{item.name}: output offset mismatch")
            written = 0
            for chunk in db.iter_read(item.source):
                fp.write(chunk)
                written += len(chunk)
            if written != item.nbytes:
                fail(f"{item.name}: copied {written} bytes, expected {item.nbytes}")
            fp.write(bytes(align(item.nbytes) - item.nbytes))
            if index % 25 == 0 or index == len(plan):
                print(f"copied tensors: {index}/{len(plan)}", file=sys.stderr, flush=True)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(partial, path)


def read_metadata_value(fp, value_type):
    if value_type == GGUF_STRING:
        return read_gguf_string(fp, "GGUF metadata string")
    if value_type == GGUF_UINT32:
        return read_u32(fp, "GGUF metadata uint32")
    if value_type == GGUF_UINT64:
        return read_u64(fp, "GGUF metadata uint64")
    if value_type == GGUF_FLOAT32:
        return struct.unpack("<f", read_exact(fp, 4, "GGUF metadata float32"))[0]
    if value_type == GGUF_ARRAY:
        element_type = read_u32(fp, "GGUF array element type")
        count = read_u64(fp, "GGUF array count")
        if element_type == GGUF_FLOAT32 and count <= 1024:
            return list(struct.unpack(f"<{count}f", read_exact(fp, count * 4, "GGUF float array")))
        fail(f"unsupported validation array type/count: {element_type}/{count}")
    skip_gguf_value(fp, value_type)
    return None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        while chunk := fp.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gguf(path, plan, expected_metadata, db, verify_payload, expected_sha256):
    expected_values = {}
    for record in expected_metadata:
        reader = io.BytesIO(record)
        key = read_gguf_string(reader, "expected metadata key")
        value_type = read_u32(reader, "expected metadata type")
        expected_values[key] = read_metadata_value(reader, value_type)

    with open(path, "rb") as fp:
        if read_exact(fp, 4, "GGUF magic") != b"GGUF":
            fail(f"{path}: not a GGUF file")
        version = read_u32(fp, "GGUF version")
        if version != GGUF_VERSION:
            fail(f"expected GGUF v{GGUF_VERSION}, got v{version}")
        tensor_count = read_u64(fp, "GGUF tensor count")
        metadata_count = read_u64(fp, "GGUF metadata count")
        if tensor_count != len(plan):
            fail(f"tensor count {tensor_count} != expected {len(plan)}")
        if metadata_count != len(expected_metadata):
            fail(f"metadata count {metadata_count} != expected {len(expected_metadata)}")

        actual_values = {}
        for _ in range(metadata_count):
            key = read_gguf_string(fp, "GGUF metadata key")
            value_type = read_u32(fp, "GGUF metadata type")
            actual_values[key] = read_metadata_value(fp, value_type)
        if actual_values != expected_values:
            fail("GGUF metadata differs from the pinned conversion metadata")

        for index, item in enumerate(plan):
            name = read_gguf_string(fp, f"tensor {index} name")
            rank = read_u32(fp, f"tensor {index} rank")
            shape = tuple(read_u64(fp, f"tensor {index} dimension") for _ in range(rank))
            qtype = read_u32(fp, f"tensor {index} type")
            offset = read_u64(fp, f"tensor {index} offset")
            if (name, shape, qtype, offset) != (item.name, item.shape, item.qtype, item.offset):
                fail(f"tensor {index} header differs from the conversion plan: {name}")

        data_offset, data_bytes = layout(plan, expected_metadata)
        if fp.tell() > data_offset:
            fail("parsed GGUF header exceeds the planned data offset")
        actual_size = os.fstat(fp.fileno()).st_size
        if actual_size != data_offset + data_bytes:
            fail(f"file size {actual_size} != expected {data_offset + data_bytes}")

        verified = 0
        if verify_payload:
            for index, item in enumerate(plan, 1):
                fp.seek(data_offset + item.offset)
                for source in db.iter_read(item.source):
                    if read_exact(fp, len(source), item.name) != source:
                        fail(f"{item.name}: payload differs from the official source")
                    verified += len(source)
                padding = read_exact(fp, align(item.nbytes) - item.nbytes, f"{item.name} padding")
                if any(padding):
                    fail(f"{item.name}: nonzero alignment padding")
                if index % 25 == 0 or index == len(plan):
                    print(f"verified payloads: {index}/{len(plan)}", file=sys.stderr, flush=True)

    print(
        f"validated {path}: {len(plan)} {QTYPE_NAMES[QTYPE_BF16]} tensors, "
        f"{actual_size} bytes"
    )
    digest = file_sha256(path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        fail(f"SHA-256 {digest} != expected {expected_sha256.lower()}")
    print(f"SHA-256: {digest}")
    if verify_payload:
        print(f"source payloads matched byte for byte: {verified} bytes")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True, help="official GLM-5.3 Flash snapshot")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--out", help="create this vision GGUF")
    output.add_argument("--validate", metavar="GGUF", help="validate an existing vision GGUF")
    parser.add_argument("--source-revision", default=DEFAULT_SOURCE_REVISION)
    parser.add_argument("--verify-payload", action="store_true")
    parser.add_argument("--expected-sha256", help="expected hash when validating")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.out and not args.validate:
        parser.error("one of --out, --validate, or --dry-run is required")
    if args.verify_payload and not args.validate:
        parser.error("--verify-payload requires --validate")
    if args.expected_sha256 and not args.validate:
        parser.error("--expected-sha256 requires --validate")
    return args


def main():
    args = parse_args()
    db = SourceDB(args.hf)
    try:
        plan = build_vision_plan(db)
        metadata = vision_metadata(args.hf, args.source_revision)
        if args.dry_run:
            print_summary(plan, metadata)
        elif args.out:
            create_gguf(args.out, plan, metadata, db, args.overwrite)
            print(f"glm53-vision: wrote {args.out}", file=sys.stderr)
        else:
            validate_gguf(
                args.validate,
                plan,
                metadata,
                db,
                args.verify_payload,
                args.expected_sha256,
            )
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"glm53-vision: error: {error}", file=sys.stderr)
        sys.exit(1)
