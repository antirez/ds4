#!/usr/bin/env python3
"""Copy the released DeepSeek V4.1 Flash vision encoder into a separate GGUF.

All 266 tensors remain BF16, byte for byte. Visual router biases are already
part of the language GGUF produced by deepseek41_quantize.py.
"""

import argparse
import json
from pathlib import Path
import sys

from deepseek4_vision import create_gguf, print_summary, validate_gguf
from glm53_quantize import (
    GGUF_ALIGNMENT, QTYPE_BF16, SourceDB, TensorPlan, align,
    kv_f32, kv_string, kv_u32, qtype_nbytes,
)

SOURCE_REVISION = "df42c109f1defefcbfcedbe7d905718a12266e40"
SOURCE_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash"


def source_shapes():
    shapes = {
        "vision.patch_embed.proj.weight": (1024, 588),
        "vision.patch_embed.proj.bias": (1024,),
        "vision.norm.weight": (1024,),
        "aligner.w1.weight": (5120, 9216),
        "aligner.w1.bias": (5120,),
        "aligner.w2.weight": (5120, 5120),
        "aligner.w2.bias": (5120,),
        "image_start": (5120,),
        "image_newline": (5120,),
        "image_end": (5120,),
    }
    for layer in range(32):
        for suffix, shape in {
            "norm1.weight": (1024,), "norm2.weight": (1024,),
            "attn.wqkv.weight": (3072, 1024), "attn.wqkv.bias": (3072,),
            "attn.wo.weight": (1024, 1024), "attn.wo.bias": (1024,),
            "mlp.w1.weight": (5632, 1024), "mlp.w2.weight": (1024, 2816),
        }.items():
            shapes[f"vision.blocks.{layer}.{suffix}"] = shape
    return shapes


def validate_inventory(tensors):
    selected = {name for name in tensors if name.startswith(("vision.", "aligner.", "image_"))}
    expected = set(source_shapes())
    if selected != expected:
        raise ValueError(f"vision inventory mismatch: missing {sorted(expected - selected)}, "
                         f"unexpected {sorted(selected - expected)}")


def build_plan(db):
    validate_inventory(db.tensors)
    plan, offset = [], 0
    for name, shape in sorted(source_shapes().items()):
        info = db.info(name)
        if info["dtype"] != "BF16" or tuple(info["shape"]) != shape:
            raise ValueError(f"{name}: expected BF16 {shape}, found {info['dtype']} {info['shape']}")
        role = "vision" if name.startswith("vision.") else (
            "aligner" if name.startswith("aligner.") else "image_embedding")
        item = TensorPlan(name, tuple(reversed(shape)), QTYPE_BF16, role,
                          source=name, raw_copy=True)
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
        if item.nbytes != info["nbytes"]:
            raise ValueError(f"{name}: source byte size mismatch")
        item.offset = offset
        offset += align(item.nbytes)
        plan.append(item)
    return plan


def sidecar_metadata(hf_dir):
    config = json.loads((Path(hf_dir) / "config.json").read_text())
    if config.get("model_type") != "deepseek_v41" or config.get("image_token_id") != 129264:
        raise ValueError("expected the DeepSeek V4.1 Flash checkpoint")
    for section, expected in {
        "text_config": {"hidden_size": 5120, "num_hidden_layers": 40, "n_routed_experts": 384},
        "vision_config": {"num_hidden_layers": 32, "hidden_size": 1024,
            "num_attention_heads": 16, "intermediate_size": 2816, "patch_size": 14,
            "rope_theta": 10000, "downsample_ratio": 3, "max_image_tokens": 1024,
            "min_pixels": 295936, "max_wh_ratio": None},
    }.items():
        for key, value in expected.items():
            if key not in config[section] or config[section][key] != value:
                raise ValueError(f"unexpected {section}.{key}: {config[section].get(key)!r}")
    prefix = "deepseek4-vision"
    records = [
        kv_string("general.architecture", prefix),
        kv_string("general.name", "DeepSeek V4.1 Flash Vision Encoder"),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string("general.source.url", SOURCE_URL),
        kv_string("general.source.revision", SOURCE_REVISION),
        kv_string(f"{prefix}.checkpoint_variant", "v4.1-flash"),
        kv_f32(f"{prefix}.rope.freq_base", 10000),
        kv_f32(f"{prefix}.attention.layer_norm_rms_epsilon", 1e-6),
    ]
    for key, value in {
        "block_count": 32, "embedding_length": 1024, "feed_forward_length": 2816,
        "attention.head_count": 16, "projection_length": 5120, "patch_size": 14,
        "downsample_ratio": 3, "image.max_tokens": 1024, "image.min_pixels": 295936,
        "image.max_width_height_ratio": 0, "image.token_id": 129264,
        "language.block_count": 40, "language.expert_count": 384,
    }.items():
        records.append(kv_u32(f"{prefix}.{key}", value))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True, help="official V4.1 Flash snapshot")
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--out", help="write this vision GGUF")
    output.add_argument("--validate", metavar="GGUF")
    output.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-payload", action="store_true")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if (args.verify_payload or args.expected_sha256) and not args.validate:
        parser.error("--verify-payload and --expected-sha256 require --validate")
    records = sidecar_metadata(args.hf)
    db = SourceDB(args.hf, validate_inventory, lambda _: None)
    try:
        plan = build_plan(db)
        if args.out:
            create_gguf(args.out, plan, records, db, overwrite=False)
        elif args.validate:
            validate_gguf(args.validate, plan, records, db,
                          args.verify_payload, args.expected_sha256)
        else:
            print_summary(plan, records)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError) as error:
        print(f"deepseek41-vision: {error}", file=sys.stderr)
        raise SystemExit(1)
