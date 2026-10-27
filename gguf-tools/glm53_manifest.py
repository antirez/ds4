#!/usr/bin/env python3
"""Inspect and validate an official GLM-5.3 Flash safetensors snapshot."""

import argparse
import json
import math
import os
import re
import struct
import sys


DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")


def fail(message):
    raise ValueError(message)


def checked_product(values, label):
    result = 1
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            fail(f"{label}: invalid shape dimension {value!r}")
        result *= value
        if result > (1 << 63) - 1:
            fail(f"{label}: tensor element count overflows int64")
    return result


def load_index(path):
    with open(path, "rb") as fp:
        document = json.load(fp)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        fail(f"{path}: missing or empty weight_map")
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not name:
            fail(f"{path}: invalid tensor name")
        if not isinstance(shard, str) or os.path.basename(shard) != shard:
            fail(f"{path}: invalid shard for {name}")
    return document, weight_map


def load_safetensors_header(path):
    file_size = os.path.getsize(path)
    with open(path, "rb") as fp:
        raw_length = fp.read(8)
        if len(raw_length) != 8:
            fail(f"{path}: short safetensors header length")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > file_size - 8:
            fail(f"{path}: header extends beyond file")
        if header_length > (1 << 30):
            fail(f"{path}: unreasonable header length {header_length}")
        raw_header = fp.read(header_length)
    try:
        document = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{path}: invalid safetensors header: {error}")

    payload_size = file_size - 8 - header_length
    tensors = {}
    spans = []
    for name, entry in document.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            fail(f"{path}: bad metadata for {name}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype not in DTYPE_BYTES:
            fail(f"{path}: unsupported dtype {dtype!r} for {name}")
        if not isinstance(shape, list):
            fail(f"{path}: invalid shape for {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
        ):
            fail(f"{path}: invalid data offsets for {name}")
        start, end = offsets
        if start < 0 or end < start or end > payload_size:
            fail(f"{path}: out-of-range data offsets for {name}")
        nbytes = checked_product(shape, name) * DTYPE_BYTES[dtype]
        if end - start != nbytes:
            fail(f"{path}: byte count mismatch for {name}: {end - start} != {nbytes}")
        tensors[name] = {
            "dtype": dtype,
            "shape": shape,
            "offset": 8 + header_length + start,
            "nbytes": nbytes,
        }
        spans.append((start, end, name))

    cursor = 0
    for start, end, name in sorted(spans):
        if start != cursor:
            fail(f"{path}: gap or overlap before {name}: {start} != {cursor}")
        cursor = end
    if cursor != payload_size:
        fail(f"{path}: unclaimed payload bytes: {payload_size - cursor}")
    return tensors


def tensor_scope(name):
    if name.startswith("model.visual."):
        return "vision"
    if name == "lm_head.weight" or name.startswith("model.language_model."):
        return "text"
    return "unknown"


def tensor_role(name):
    if name.endswith(".weight_scale_inv"):
        return "scale"
    if name == "model.language_model.embed_tokens.weight":
        return "embedding"
    if name == "model.language_model.norm.weight":
        return "final_norm"
    if name == "lm_head.weight":
        return "output"
    match = LAYER_RE.match(name)
    if not match:
        return "vision" if name.startswith("model.visual.") else "unknown"
    layer = int(match.group(1))
    tail = match.group(2)
    prefix = "mtp_" if layer == 45 else ""
    expert = EXPERT_RE.match(tail)
    if expert:
        return f"{prefix}routed_{expert.group(2)}"
    if tail.startswith("mlp.shared_experts."):
        return f"{prefix}shared_expert"
    if tail.startswith("mlp.gate."):
        return f"{prefix}router"
    if tail.startswith("mlp."):
        return f"{prefix}dense_ffn"
    if tail.startswith("self_attn.indexer."):
        return f"{prefix}dsa_indexer"
    if tail.startswith("self_attn."):
        if layer < 45 and layer % 4 != 3:
            return "linear_attention"
        return f"{prefix}dsa"
    if tail.startswith("hc_"):
        return "mhc"
    if "norm" in tail:
        return f"{prefix}norm"
    if tail.startswith(("eh_proj.", "shared_head.")):
        return "mtp_head"
    return f"{prefix}other"


def expect_equal(actual, expected, label):
    if actual != expected:
        fail(f"{label}: got {actual!r}, expected {expected!r}")


def validate_glm53_index(weight_map):
    names = set(weight_map)
    scopes = {tensor_scope(name) for name in names}
    if "unknown" in scopes:
        unknown = sorted(name for name in names if tensor_scope(name) == "unknown")
        fail(f"unknown top-level tensors: {unknown[:5]}")

    layers = set()
    hc_layers = set()
    linear_layers = set()
    dsa_layers = set()
    sparse_layers = set()
    expert_ids = {}
    expert_parts = {}
    for name in names:
        match = LAYER_RE.match(name)
        if not match:
            continue
        layer = int(match.group(1))
        tail = match.group(2)
        layers.add(layer)
        if tail.startswith("hc_attn_"):
            hc_layers.add(layer)
        if tail == "self_attn.A_log":
            linear_layers.add(layer)
        if tail == "self_attn.indexer.wk.weight":
            dsa_layers.add(layer)
        expert = EXPERT_RE.match(tail)
        if expert:
            sparse_layers.add(layer)
            expert_ids.setdefault(layer, set()).add(int(expert.group(1)))
            expert_parts.setdefault((layer, int(expert.group(1))), set()).add(expert.group(2))

    expected_linear = {layer for layer in range(45) if layer % 4 != 3}
    expected_dsa = {layer for layer in range(45) if layer % 4 == 3} | {45}
    expected_sparse = set(range(3, 46))
    expect_equal(layers, set(range(46)), "language-model layer set")
    expect_equal(hc_layers, set(range(45)), "mHC layer set")
    expect_equal(linear_layers, expected_linear, "linear-attention layer set")
    expect_equal(dsa_layers, expected_dsa, "DSA layer set")
    expect_equal(sparse_layers, expected_sparse, "sparse FFN layer set")
    for layer in expected_sparse:
        expect_equal(expert_ids.get(layer), set(range(288)), f"layer {layer} expert ids")
        for expert in range(288):
            expect_equal(
                expert_parts.get((layer, expert)),
                {"gate", "up", "down"},
                f"layer {layer} expert {expert} projections",
            )

    required = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
        "model.language_model.layers.45.eh_proj.weight",
        "model.language_model.layers.45.enorm.weight",
        "model.language_model.layers.45.hnorm.weight",
        "model.language_model.layers.45.shared_head.norm.weight",
    }
    missing = sorted(required - names)
    if missing:
        fail(f"missing required tensors: {missing}")


def validate_glm53_full_index(weight_map):
    names = set(weight_map)
    layers = set()
    indexer_layers = set()
    sparse_layers = set()
    expert_ids = {}
    expert_parts = {}
    layer_re = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
    expert_re = re.compile(r"^mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")

    allowed_top = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for name in names:
        match = layer_re.match(name)
        if not match:
            if name not in allowed_top and not name.endswith("_scale_inv"):
                fail(f"unknown full GLM-5.3 tensor: {name}")
            continue
        layer = int(match.group(1))
        tail = match.group(2)
        layers.add(layer)
        if tail == "self_attn.indexer.wk.weight":
            indexer_layers.add(layer)
        expert = expert_re.match(tail)
        if expert:
            sparse_layers.add(layer)
            expert_id = int(expert.group(1))
            expert_ids.setdefault(layer, set()).add(expert_id)
            expert_parts.setdefault((layer, expert_id), set()).add(expert.group(2))

    expect_equal(layers, set(range(79)), "full GLM-5.3 layer set")
    expect_equal(sparse_layers, set(range(3, 79)), "full GLM-5.3 sparse FFN layer set")
    expected_indexers = {0, 1, 2, 78} | set(range(6, 78, 4))
    expect_equal(indexer_layers, expected_indexers, "full GLM-5.3 indexer owner layers")
    for layer in range(3, 79):
        expect_equal(expert_ids.get(layer), set(range(256)), f"layer {layer} expert ids")
        for expert in range(256):
            expect_equal(
                expert_parts.get((layer, expert)),
                {"gate", "up", "down"},
                f"layer {layer} expert {expert} projections",
            )

    required = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.78.eh_proj.weight",
        "model.layers.78.enorm.weight",
        "model.layers.78.hnorm.weight",
        "model.layers.78.shared_head.norm.weight",
    }
    missing = sorted(required - names)
    if missing:
        fail(f"missing required full GLM-5.3 tensors: {missing}")


def validate_fp8_scales(tensors):
    for name, info in tensors.items():
        if info["dtype"] != "F8_E4M3" or not name.endswith(".weight"):
            continue
        if len(info["shape"]) != 2:
            fail(f"{name}: FP8 weight is not two-dimensional")
        scale_name = name + "_scale_inv"
        scale = tensors.get(scale_name)
        if scale is None:
            fail(f"{name}: missing {scale_name}")
        expected_shape = [math.ceil(info["shape"][0] / 128), math.ceil(info["shape"][1] / 128)]
        if scale["dtype"] != "F32" or scale["shape"] != expected_shape:
            fail(
                f"{scale_name}: got {scale['dtype']} {scale['shape']}, "
                f"expected F32 {expected_shape}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_dir", help="downloaded zai-org/GLM-5.3-Flash directory")
    parser.add_argument(
        "--index",
        help="index JSON path; defaults to HF_DIR/model.safetensors.index.json",
    )
    parser.add_argument(
        "--allow-missing-shards",
        action="store_true",
        help="emit index-only rows for shards that are still downloading",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    index_path = args.index or os.path.join(args.hf_dir, "model.safetensors.index.json")
    document, weight_map = load_index(index_path)
    validate_glm53_index(weight_map)

    shard_names = sorted(set(weight_map.values()))
    tensors = {}
    missing_shards = []
    for shard_name in shard_names:
        shard_path = os.path.join(args.hf_dir, shard_name)
        if not os.path.isfile(shard_path):
            missing_shards.append(shard_name)
            continue
        header = load_safetensors_header(shard_path)
        for name, info in header.items():
            assigned = weight_map.get(name)
            if assigned != shard_name:
                fail(f"{shard_name}: {name} is assigned to {assigned!r} by the index")
            if name in tensors:
                fail(f"duplicate tensor in shard headers: {name}")
            tensors[name] = info

    if missing_shards and not args.allow_missing_shards:
        fail(f"missing {len(missing_shards)} shards; first is {missing_shards[0]}")

    present_names = set(tensors)
    expected_present = {name for name, shard in weight_map.items() if shard not in missing_shards}
    if present_names != expected_present:
        missing = sorted(expected_present - present_names)
        extra = sorted(present_names - expected_present)
        fail(f"header/index mismatch: missing={missing[:3]} extra={extra[:3]}")
    validate_fp8_scales(tensors)

    print("scope\trole\tname\tdtype\tshape\tshard\toffset\tnbytes")
    for name in sorted(weight_map):
        info = tensors.get(name)
        dtype = info["dtype"] if info else "-"
        shape = "x".join(str(value) for value in info["shape"]) if info else "-"
        offset = str(info["offset"]) if info else "-"
        nbytes = str(info["nbytes"]) if info else "-"
        print(
            f"{tensor_scope(name)}\t{tensor_role(name)}\t{name}\t{dtype}\t{shape}\t"
            f"{weight_map[name]}\t{offset}\t{nbytes}"
        )

    declared_size = document.get("metadata", {}).get("total_size")
    present_bytes = sum(info["nbytes"] for info in tensors.values())
    print(
        f"glm53-manifest: tensors={len(weight_map)} shards={len(shard_names)} "
        f"present_shards={len(shard_names) - len(missing_shards)} "
        f"present_bytes={present_bytes} declared_bytes={declared_size} "
        f"missing_shards={len(missing_shards)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"glm53-manifest: error: {error}", file=sys.stderr)
        sys.exit(1)
