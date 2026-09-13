#!/usr/bin/env python3
"""Create or validate the standalone DeepSeek V4 Flash Vision sidecar GGUF."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import struct
import sys

from glm53_quantize import (
    GGUF_ALIGNMENT,
    GGUF_FLOAT32,
    GGUF_STRING,
    GGUF_UINT32,
    GGUF_VERSION,
    QTYPE_BF16,
    QTYPE_F32,
    QTYPE_NAMES,
    SourceDB,
    TensorPlan,
    align,
    fail,
    kv_f32,
    kv_string,
    kv_u32,
    pack_string,
    qtype_nbytes,
    read_exact,
    read_gguf_string,
    read_u32,
    read_u64,
    skip_gguf_value,
    tensor_header,
)


SOURCE_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
SOURCE_REVISION = "e46e16bf6035c6f317eb2ac7458eb0362926d402"
ARCHITECTURE = "deepseek4-vision"
ROUTER_RE = re.compile(r"^(layers\.(\d+)|mtp\.(\d+))\.ffn\.gate\.bias_vl$")
HASH_BIAS_RE = re.compile(r"^layers\.([0-2])\.ffn\.gate\.bias$")


def validate_deepseek4_index(weight_map):
    if len(weight_map) != 72633:
        fail(f"expected 72,633 source tensors, found {len(weight_map)}")
    shards = set(weight_map.values())
    expected_shards = {
        f"model-{index:05d}-of-00048.safetensors"
        for index in range(1, 49)
    }
    if shards != expected_shards:
        fail("source shard inventory differs from the pinned checkpoint")
    selected = {name for name in weight_map if selected_role(name)}
    validate_inventory(selected)
    if "embed.weight" not in weight_map or "head.weight" not in weight_map:
        fail("source language tensor inventory is incomplete")


def validate_deepseek4_fp8_scales(tensors):
    fp8 = {name: info for name, info in tensors.items()
           if info["dtype"] == "F8_E4M3"}
    if len(fp8) != 390:
        fail(f"expected 390 FP8 language tensors, found {len(fp8)}")
    for name, info in fp8.items():
        if not name.endswith(".weight") or len(info["shape"]) != 2:
            fail(f"unexpected FP8 tensor layout: {name}")
        scale_name = name.removesuffix(".weight") + ".scale"
        scale = tensors.get(scale_name)
        expected_shape = [
            (info["shape"][0] + 127) // 128,
            (info["shape"][1] + 127) // 128,
        ]
        if not scale or scale["dtype"] != "F8_E8M0" or \
                scale["shape"] != expected_shape:
            fail(f"{name}: invalid native FP8 scale tensor {scale_name}")


def load_config(hf_dir):
    with open(os.path.join(hf_dir, "config.json"), encoding="utf-8") as fp:
        config = json.load(fp)
    expected = {
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "n_routed_experts": 256,
        "num_hidden_layers": 43,
        "vision_n_layers": 32,
        "vision_dim": 1024,
        "vision_n_heads": 16,
        "vision_inter_dim": 2816,
        "vision_patch_size": 14,
        "vision_rope_theta": 10000.0,
        "vision_downsample_ratio": 3,
        "vision_max_n_token": 384,
        "vision_min_pixels": 147456,
        "vision_max_wh_ratio": 8,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            fail(f"unexpected config {key}: {config.get(key)!r}")
    if config.get("rms_norm_eps") != 1e-20:
        fail(f"unexpected language rms_norm_eps: {config.get('rms_norm_eps')!r}")
    return config


def selected_role(name):
    if name.startswith("vision."):
        return "vision"
    if name.startswith("aligner."):
        return "aligner"
    if name in {"image_start", "image_pad", "image_newline", "image_end"}:
        return "image_embedding"
    if ROUTER_RE.match(name):
        return "visual_router_bias"
    if HASH_BIAS_RE.match(name):
        return "hash_router_bias"
    return None


def validate_inventory(names):
    names = set(names)
    vision = {name for name in names if name.startswith("vision.")}
    aligner = {name for name in names if name.startswith("aligner.")}
    image = {name for name in names if name.startswith("image_")}
    visual_bias = {name for name in names if ROUTER_RE.match(name)}
    hash_bias = {name for name in names if HASH_BIAS_RE.match(name)}
    if len(vision) != 259:
        fail(f"expected 259 ViT tensors, found {len(vision)}")
    if len(aligner) != 4:
        fail(f"expected 4 aligner tensors, found {len(aligner)}")
    if image != {"image_start", "image_pad", "image_newline", "image_end"}:
        fail(f"unexpected image embedding inventory: {sorted(image)}")
    expected_visual = {
        *(f"layers.{layer}.ffn.gate.bias_vl" for layer in range(43)),
        *(f"mtp.{stage}.ffn.gate.bias_vl" for stage in range(3)),
    }
    if visual_bias != expected_visual:
        fail("visual router bias inventory differs from the pinned checkpoint")
    expected_hash = {f"layers.{layer}.ffn.gate.bias" for layer in range(3)}
    if hash_bias != expected_hash:
        fail("hash router bias inventory differs from the pinned checkpoint")
    if len(names) != 316:
        fail(f"expected 316 sidecar tensors, found {len(names)}")


def build_plan(db):
    selected = sorted(name for name in db.tensors if selected_role(name))
    validate_inventory(selected)
    plan = []
    offset = 0
    for name in selected:
        info = db.info(name)
        role = selected_role(name)
        expected_dtype = "BF16" if role in {
            "vision", "aligner", "image_embedding"
        } else "F32"
        if info["dtype"] != expected_dtype:
            fail(f"{name}: expected {expected_dtype}, got {info['dtype']}")
        qtype = QTYPE_BF16 if expected_dtype == "BF16" else QTYPE_F32
        item = TensorPlan(
            name=name,
            shape=tuple(reversed(info["shape"])),
            qtype=qtype,
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
    return plan


def sidecar_metadata(hf_dir, source_revision):
    config = load_config(hf_dir)
    prefix = ARCHITECTURE
    return [
        kv_string("general.architecture", ARCHITECTURE),
        kv_string("general.name", "DeepSeek V4 Flash Vision Encoder"),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string("general.source.url", SOURCE_URL),
        kv_string("general.source.revision", source_revision),
        kv_string(f"{prefix}.checkpoint_variant", "vision-exp"),
        kv_u32(f"{prefix}.block_count", config["vision_n_layers"]),
        kv_u32(f"{prefix}.embedding_length", config["vision_dim"]),
        kv_u32(f"{prefix}.feed_forward_length", config["vision_inter_dim"]),
        kv_u32(f"{prefix}.attention.head_count", config["vision_n_heads"]),
        kv_u32(f"{prefix}.projection_length", config["hidden_size"]),
        kv_u32(f"{prefix}.patch_size", config["vision_patch_size"]),
        kv_f32(f"{prefix}.rope.freq_base", config["vision_rope_theta"]),
        kv_f32(f"{prefix}.attention.layer_norm_rms_epsilon", 1e-6),
        kv_u32(f"{prefix}.downsample_ratio", config["vision_downsample_ratio"]),
        kv_u32(f"{prefix}.image.max_tokens", config["vision_max_n_token"]),
        kv_u32(f"{prefix}.image.min_pixels", config["vision_min_pixels"]),
        kv_u32(f"{prefix}.image.max_width_height_ratio", config["vision_max_wh_ratio"]),
        kv_u32(f"{prefix}.language.block_count", config["num_hidden_layers"]),
        kv_u32(f"{prefix}.language.expert_count", config["n_routed_experts"]),
    ]


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
    if value_type == GGUF_FLOAT32:
        return struct.unpack("<f", read_exact(fp, 4, "GGUF metadata float32"))[0]
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

    digest = file_sha256(path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        fail(f"SHA-256 {digest} != expected {expected_sha256.lower()}")
    print(f"validated {path}: {len(plan)} tensors, {actual_size} bytes")
    print(f"SHA-256: {digest}")
    if verify_payload:
        print(f"source payloads matched byte for byte: {verified} bytes")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True, help="official Vision-Exp snapshot")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--out", help="create this vision GGUF")
    output.add_argument("--validate", metavar="GGUF", help="validate an existing vision GGUF")
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--verify-payload", action="store_true")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.source_revision != SOURCE_REVISION:
        parser.error(f"unsupported source revision: {args.source_revision}")
    if not args.dry_run and not args.out and not args.validate:
        parser.error("one of --out, --validate, or --dry-run is required")
    if args.verify_payload and not args.validate:
        parser.error("--verify-payload requires --validate")
    if args.expected_sha256 and not args.validate:
        parser.error("--expected-sha256 requires --validate")
    return args


def main():
    args = parse_args()
    db = SourceDB(
        args.hf,
        validate_deepseek4_index,
        validate_deepseek4_fp8_scales,
    )
    try:
        plan = build_plan(db)
        metadata = sidecar_metadata(args.hf, args.source_revision)
        if args.dry_run:
            print_summary(plan, metadata)
        elif args.out:
            create_gguf(args.out, plan, metadata, db, args.overwrite)
            print(f"deepseek4-vision: wrote {args.out}", file=sys.stderr)
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
        print(f"deepseek4-vision: error: {error}", file=sys.stderr)
        raise SystemExit(1)
