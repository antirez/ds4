#!/usr/bin/env python3
"""Build DwarfStar GLM-5.3 Flash GGUF files from official FP8."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import dataclasses
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import threading
import time

from glm53_manifest import load_index, load_safetensors_header, validate_fp8_scales, validate_glm53_index


GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

GGUF_UINT32 = 4
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_UINT64 = 10

QTYPE_F32 = 0
QTYPE_F16 = 1
QTYPE_Q8_0 = 8
QTYPE_Q2_K = 10
QTYPE_Q4_K = 12
QTYPE_Q8_K = 15
QTYPE_IQ2_XXS = 16
QTYPE_I8 = 24
QTYPE_BF16 = 30

QTYPE_NAMES = {
    QTYPE_F32: "F32",
    QTYPE_F16: "F16",
    QTYPE_Q8_0: "Q8_0",
    QTYPE_Q2_K: "Q2_K",
    QTYPE_Q4_K: "Q4_K",
    QTYPE_Q8_K: "Q8_K",
    QTYPE_IQ2_XXS: "IQ2_XXS",
    QTYPE_I8: "I8",
    QTYPE_BF16: "BF16",
}

QTYPE_LAYOUT = {
    QTYPE_F32: (1, 4),
    QTYPE_F16: (1, 2),
    QTYPE_Q8_0: (32, 34),
    QTYPE_Q2_K: (256, 84),
    QTYPE_Q4_K: (256, 144),
    QTYPE_Q8_K: (256, 292),
    QTYPE_IQ2_XXS: (256, 66),
    QTYPE_I8: (1, 1),
    QTYPE_BF16: (1, 2),
}

LAYER_PREFIX = "model.language_model.layers"


def fail(message):
    raise ValueError(message)


def align(value, alignment=GGUF_ALIGNMENT):
    return (value + alignment - 1) // alignment * alignment


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def qtype_nbytes(qtype, shape):
    block, block_bytes = QTYPE_LAYOUT[qtype]
    if not shape or shape[0] % block:
        fail(f"shape {shape} is incompatible with {QTYPE_NAMES[qtype]}")
    return shape[0] // block * block_bytes * product(shape[1:])


def regular_qtype(artifact, role, name, source_qtype):
    if artifact not in ("q2", "q4") or source_qtype != QTYPE_BF16:
        return source_qtype
    if role in ("embedding", "output"):
        return QTYPE_Q8_0
    if role == "linear_attention":
        if artifact == "q2" and (
            name.endswith(".kda_q.weight") or
            name.endswith(".kda_k.weight")
        ):
            return QTYPE_Q4_K
        return QTYPE_Q8_0
    return source_qtype


def pack_string(value):
    data = value.encode("utf-8") if isinstance(value, str) else value
    return struct.pack("<Q", len(data)) + data


def kv_string(key, value):
    return pack_string(key) + struct.pack("<I", GGUF_STRING) + pack_string(value)


def kv_u32(key, value):
    return pack_string(key) + struct.pack("<II", GGUF_UINT32, value)


def kv_u64(key, value):
    return pack_string(key) + struct.pack("<IQ", GGUF_UINT64, value)


def kv_f32(key, value):
    return pack_string(key) + struct.pack("<If", GGUF_FLOAT32, value)


def kv_bool(key, value):
    return pack_string(key) + struct.pack("<IB", GGUF_BOOL, bool(value))


def kv_u32_array(key, values):
    return (
        pack_string(key)
        + struct.pack("<IIQ", GGUF_ARRAY, GGUF_UINT32, len(values))
        + struct.pack(f"<{len(values)}I", *values)
    )


def read_exact(fp, length, label):
    data = fp.read(length)
    if len(data) != length:
        fail(f"short read for {label}")
    return data


def read_u32(fp, label):
    return struct.unpack("<I", read_exact(fp, 4, label))[0]


def read_u64(fp, label):
    return struct.unpack("<Q", read_exact(fp, 8, label))[0]


def read_gguf_string(fp, label):
    length = read_u64(fp, f"{label} length")
    if length > 1 << 30:
        fail(f"unreasonable {label} length {length}")
    return read_exact(fp, length, label).decode("utf-8")


def skip_gguf_value(fp, value_type):
    scalar_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if value_type == GGUF_STRING:
        fp.seek(read_u64(fp, "GGUF string length"), os.SEEK_CUR)
        return
    if value_type == GGUF_ARRAY:
        element_type = read_u32(fp, "GGUF array element type")
        count = read_u64(fp, "GGUF array count")
        if count > 1 << 32:
            fail(f"unreasonable GGUF array count {count}")
        if element_type == GGUF_STRING:
            for _ in range(count):
                fp.seek(read_u64(fp, "GGUF array string length"), os.SEEK_CUR)
            return
        size = scalar_sizes.get(element_type)
        if size is None:
            fail(f"unsupported GGUF array element type {element_type}")
        fp.seek(count * size, os.SEEK_CUR)
        return
    size = scalar_sizes.get(value_type)
    if size is None:
        fail(f"unsupported GGUF value type {value_type}")
    fp.seek(size, os.SEEK_CUR)


def load_tokenizer_records(path):
    records = []
    keys = set()
    tokens = None
    with open(path, "rb") as fp:
        if read_exact(fp, 4, "GGUF magic") != b"GGUF":
            fail(f"{path}: not a GGUF file")
        version = read_u32(fp, "GGUF version")
        if version not in (2, 3):
            fail(f"{path}: unsupported GGUF version {version}")
        read_u64(fp, "GGUF tensor count")
        n_kv = read_u64(fp, "GGUF metadata count")
        for _ in range(n_kv):
            start = fp.tell()
            key = read_gguf_string(fp, "GGUF metadata key")
            value_type = read_u32(fp, "GGUF metadata type")
            if key == "tokenizer.ggml.tokens":
                if value_type != GGUF_ARRAY:
                    fail(f"{path}: tokenizer.ggml.tokens is not an array")
                element_type = read_u32(fp, "tokenizer token element type")
                count = read_u64(fp, "tokenizer token count")
                if element_type != GGUF_STRING or count > 1 << 20:
                    fail(f"{path}: invalid tokenizer token array")
                tokens = [read_gguf_string(fp, "tokenizer token") for _ in range(count)]
            else:
                skip_gguf_value(fp, value_type)
            end = fp.tell()
            if key.startswith("tokenizer.") and key != "tokenizer.chat_template":
                fp.seek(start)
                records.append(read_exact(fp, end - start, key))
                keys.add(key)
            fp.seek(end)
    required = {"tokenizer.ggml.model", "tokenizer.ggml.tokens"}
    if not required.issubset(keys):
        fail(f"{path}: tokenizer metadata is incomplete: missing {sorted(required - keys)}")
    if tokens is None:
        fail(f"{path}: tokenizer.ggml.tokens was not decoded")
    return records, tokens


def load_source_tokens(hf_dir, vocab_size):
    path = os.path.join(hf_dir, "tokenizer.json")
    with open(path, "rb") as fp:
        document = json.load(fp)
    vocab = document.get("model", {}).get("vocab")
    added = document.get("added_tokens")
    if not isinstance(vocab, dict) or not isinstance(added, list):
        fail(f"{path}: unsupported tokenizer JSON structure")
    tokens = [None] * vocab_size
    for token, token_id in vocab.items():
        if not isinstance(token, str) or not isinstance(token_id, int):
            fail(f"{path}: invalid base vocabulary entry")
        if token_id < 0 or token_id >= vocab_size or tokens[token_id] is not None:
            fail(f"{path}: duplicate or out-of-range token id {token_id}")
        tokens[token_id] = token
    for entry in added:
        token = entry.get("content") if isinstance(entry, dict) else None
        token_id = entry.get("id") if isinstance(entry, dict) else None
        if not isinstance(token, str) or not isinstance(token_id, int):
            fail(f"{path}: invalid added token entry")
        if token_id < 0 or token_id >= vocab_size or tokens[token_id] is not None:
            fail(f"{path}: duplicate or out-of-range added token id {token_id}")
        tokens[token_id] = token
    for token_id, token in enumerate(tokens):
        if token is None:
            tokens[token_id] = f"[PAD{token_id}]"
    return tokens


def validate_tokenizer_template(hf_dir, template_tokens, vocab_size):
    source_tokens = load_source_tokens(hf_dir, vocab_size)
    if len(template_tokens) != vocab_size:
        fail(f"tokenizer template has {len(template_tokens)} tokens, expected {vocab_size}")
    for token_id, (source, template) in enumerate(zip(source_tokens, template_tokens)):
        if source != template:
            fail(
                f"tokenizer template differs at id {token_id}: "
                f"source={source!r} template={template!r}"
            )


@dataclasses.dataclass
class TensorPlan:
    name: str
    shape: tuple
    qtype: int
    role: str
    source: str | None = None
    row_start: int = 0
    row_count: int | None = None
    expert_layer: int | None = None
    expert_part: str | None = None
    expert_count: int | None = None
    transform: str | None = None
    raw_copy: bool = False
    expert_scale: bool = False
    offset: int = 0
    nbytes: int = 0

    @property
    def is_expert(self):
        return self.expert_layer is not None


class SourceDB:
    def __init__(
        self,
        hf_dir,
        index_validator=validate_glm53_index,
        scale_validator=validate_fp8_scales,
    ):
        self.hf_dir = hf_dir
        index_path = os.path.join(hf_dir, "model.safetensors.index.json")
        document, self.weight_map = load_index(index_path)
        index_validator(self.weight_map)
        self.declared_bytes = document.get("metadata", {}).get("total_size")
        self.tensors = {}
        self._fds = {}
        self._fd_lock = threading.Lock()

        for shard in sorted(set(self.weight_map.values())):
            path = os.path.join(hf_dir, shard)
            if not os.path.isfile(path):
                fail(f"missing source shard {path}")
            for name, info in load_safetensors_header(path).items():
                if self.weight_map.get(name) != shard:
                    fail(f"index assigns {name} to {self.weight_map.get(name)!r}, not {shard}")
                if name in self.tensors:
                    fail(f"duplicate source tensor {name}")
                self.tensors[name] = dict(info, shard=shard)

        if set(self.tensors) != set(self.weight_map):
            missing = sorted(set(self.weight_map) - set(self.tensors))
            fail(f"source headers are incomplete; first missing tensor is {missing[0]}")
        scale_validator(self.tensors)

    def info(self, name):
        try:
            return self.tensors[name]
        except KeyError:
            fail(f"source tensor not found: {name}")

    def _fd(self, shard):
        with self._fd_lock:
            fd = self._fds.get(shard)
            if fd is None:
                fd = os.open(os.path.join(self.hf_dir, shard), os.O_RDONLY)
                self._fds[shard] = fd
            return fd

    def read(self, name):
        info = self.info(name)
        data = os.pread(self._fd(info["shard"]), info["nbytes"], info["offset"])
        if len(data) != info["nbytes"]:
            fail(f"short payload read for {name}")
        return data

    def iter_read(self, name, byte_start=0, byte_count=None, chunk_size=16 << 20):
        info = self.info(name)
        if byte_count is None:
            byte_count = info["nbytes"] - byte_start
        if byte_start < 0 or byte_count < 0 or byte_start + byte_count > info["nbytes"]:
            fail(f"invalid payload range for {name}")
        fd = self._fd(info["shard"])
        offset = info["offset"] + byte_start
        remaining = byte_count
        while remaining:
            length = min(remaining, chunk_size)
            data = os.pread(fd, length, offset)
            if len(data) != length:
                fail(f"short payload read for {name}")
            yield data
            offset += length
            remaining -= length

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


def source_prefix(layer):
    return f"{LAYER_PREFIX}.{layer}"


def add_regular(
    plan,
    db,
    name,
    source,
    qtype,
    role,
    row_start=0,
    row_count=None,
    shape=None,
    transform=None,
):
    info = db.info(source)
    source_shape = info["shape"]
    if row_count is not None:
        if len(source_shape) != 2 or row_start < 0 or row_count <= 0 or row_start + row_count > source_shape[0]:
            fail(f"invalid row slice for {source}")
        source_shape = [row_count, source_shape[1]]
    inferred_shape = tuple(reversed(source_shape))
    shape = tuple(shape) if shape is not None else inferred_shape
    if transform is None and product(shape) != product(inferred_shape):
        fail(f"shape override {shape} changes the element count of {source_shape} for {source}")
    item = TensorPlan(name, shape, qtype, role, source, row_start, row_count, transform=transform)
    item.nbytes = qtype_nbytes(qtype, shape)
    plan.append(item)


def add_experts(plan, db, layer, part, qtype):
    first = f"{source_prefix(layer)}.mlp.experts.0.{part}_proj.weight"
    shape = db.info(first)["shape"]
    if len(shape) != 2:
        fail(f"expert source is not a matrix: {first}")
    for expert in range(288):
        name = f"{source_prefix(layer)}.mlp.experts.{expert}.{part}_proj.weight"
        if db.info(name)["shape"] != shape:
            fail(f"expert shape mismatch: {name}")
    gguf_name = f"blk.{layer}.ffn_{part}_exps.weight"
    item = TensorPlan(
        gguf_name,
        (shape[1], shape[0], 288),
        qtype,
        f"routed_{part}",
        source=f"{source_prefix(layer)}.mlp.experts.{{expert}}.{part}_proj.weight",
        expert_layer=layer,
        expert_part=part,
        expert_count=288,
    )
    item.nbytes = qtype_nbytes(qtype, item.shape)
    plan.append(item)


def add_mhc(plan, db, layer):
    prefix = source_prefix(layer)
    for site in ("attn", "ffn"):
        add_regular(plan, db, f"blk.{layer}.hc_{site}_fn.weight", f"{prefix}.hc_{site}_fn", QTYPE_BF16, "mhc")
        add_regular(plan, db, f"blk.{layer}.hc_{site}_base.weight", f"{prefix}.hc_{site}_base", QTYPE_F32, "mhc")
        add_regular(plan, db, f"blk.{layer}.hc_{site}_scale.weight", f"{prefix}.hc_{site}_scale", QTYPE_F32, "mhc")


def add_linear_attention(plan, db, layer, artifact):
    prefix = f"{source_prefix(layer)}.self_attn"
    mapping = (
        ("q", "q_proj.weight", QTYPE_BF16),
        ("k", "k_proj.weight", QTYPE_BF16),
        ("v", "v_proj.weight", QTYPE_BF16),
        ("q_conv", "q_conv1d.weight", QTYPE_F32),
        ("k_conv", "k_conv1d.weight", QTYPE_F32),
        ("v_conv", "v_conv1d.weight", QTYPE_F32),
        ("f_a", "f_a_proj.weight", QTYPE_BF16),
        ("f_b", "f_b_proj.weight", QTYPE_BF16),
        ("dt_bias", "dt_bias", QTYPE_F32),
        ("a_log", "A_log", QTYPE_F32),
        ("beta", "b_proj.weight", QTYPE_BF16),
        ("g_a", "g_a_proj.weight", QTYPE_BF16),
        ("g_b", "g_b_proj.weight", QTYPE_BF16),
        ("o_norm", "o_norm.weight", QTYPE_F32),
        ("output", "o_proj.weight", QTYPE_BF16),
    )
    for target, source, qtype in mapping:
        name = f"blk.{layer}.kda_{target}.weight"
        add_regular(
            plan,
            db,
            name,
            f"{prefix}.{source}",
            regular_qtype(artifact, "linear_attention", name, qtype),
            "linear_attention",
        )


def add_dsa_attention(plan, db, layer):
    prefix = f"{source_prefix(layer)}.self_attn"
    mapping = (
        ("attn_q_a", "q_a_proj.weight", QTYPE_Q8_0),
        ("attn_q_a_norm", "q_a_layernorm.weight", QTYPE_F32),
        ("attn_q_b", "q_b_proj.weight", QTYPE_Q8_0),
        ("attn_kv_a_mqa", "kv_a_proj_with_mqa.weight", QTYPE_Q8_0),
        ("attn_kv_a_norm", "kv_a_layernorm.weight", QTYPE_F32),
        ("attn_output", "o_proj.weight", QTYPE_Q8_0),
        ("indexer.attn_q_b", "indexer.wq_b.weight", QTYPE_BF16),
        ("indexer.attn_k", "indexer.wk.weight", QTYPE_BF16),
        ("indexer.k_norm", "indexer.k_norm.weight", QTYPE_F32),
        ("indexer.k_norm", "indexer.k_norm.bias", QTYPE_F32),
        ("indexer.proj", "indexer.weights_proj.weight", QTYPE_BF16),
        ("indexer.pool_ape", "indexer.index_kpool_compress_ape", QTYPE_BF16),
        ("indexer.pool_gate", "indexer.index_kpool_compress_gate", QTYPE_BF16),
    )
    for target, source, qtype in mapping:
        suffix = ".bias" if source.endswith(".bias") else ".weight"
        target_name = target + (".bias" if suffix == ".bias" else ".weight")
        add_regular(plan, db, f"blk.{layer}.{target_name}", f"{prefix}.{source}", qtype, "dsa")

    kv_b = f"{prefix}.kv_b_proj.weight"
    if db.info(kv_b)["shape"] != [64 * (256 + 256), 512]:
        fail(f"unexpected kv_b shape for layer {layer}: {db.info(kv_b)['shape']}")
    add_regular(
        plan,
        db,
        f"blk.{layer}.attn_k_b.weight",
        kv_b,
        QTYPE_Q8_0,
        "dsa",
        shape=(256, 512, 64),
        transform="kv_b_k",
    )
    add_regular(
        plan,
        db,
        f"blk.{layer}.attn_v_b.weight",
        kv_b,
        QTYPE_Q8_0,
        "dsa",
        shape=(512, 256, 64),
        transform="kv_b_v",
    )


def add_ffn(plan, db, layer, artifact):
    prefix = f"{source_prefix(layer)}.mlp"
    if layer < 3:
        for part in ("gate", "up", "down"):
            add_regular(plan, db, f"blk.{layer}.ffn_{part}.weight", f"{prefix}.{part}_proj.weight", QTYPE_Q8_0, "dense_ffn")
        return

    add_regular(plan, db, f"blk.{layer}.ffn_gate_inp.weight", f"{prefix}.gate.weight", QTYPE_F32, "router")
    add_regular(
        plan,
        db,
        f"blk.{layer}.exp_probs_b.bias",
        f"{prefix}.gate.e_score_correction_bias",
        QTYPE_F32,
        "router",
    )
    if artifact == "q4":
        routed = {"gate": QTYPE_Q4_K, "up": QTYPE_Q4_K, "down": QTYPE_Q4_K}
    else:
        routed = {"gate": QTYPE_IQ2_XXS, "up": QTYPE_IQ2_XXS, "down": QTYPE_Q2_K}
    for part in ("gate", "up", "down"):
        add_experts(plan, db, layer, part, routed[part])
    for part in ("gate", "up", "down"):
        add_regular(
            plan,
            db,
            f"blk.{layer}.ffn_{part}_shexp.weight",
            f"{prefix}.shared_experts.{part}_proj.weight",
            QTYPE_Q8_0,
            "shared_expert",
        )


SOURCE_QTYPES = {
    "F32": QTYPE_F32,
    "F16": QTYPE_F16,
    "BF16": QTYPE_BF16,
    "F8_E4M3": QTYPE_I8,
}


def native_fp8_plan(db, plan):
    """Replace lossy targets with original source payloads and FP8 scales."""
    native = []
    covered = set()
    for item in plan:
        if item.is_expert:
            expert_count = item.expert_count or 288
            first = item.source.format(expert=0)
            info = db.info(first)
            if info["dtype"] != "F8_E4M3":
                fail(f"native expert source is not FP8 E4M3: {first}")
            weight = dataclasses.replace(item, qtype=QTYPE_I8, raw_copy=True)
            weight.nbytes = qtype_nbytes(weight.qtype, weight.shape)
            native.append(weight)

            scale_info = db.info(first + "_scale_inv")
            if scale_info["dtype"] != "F32":
                fail(f"native FP8 scale is not F32: {first}_scale_inv")
            scale = TensorPlan(
                item.name + "_scale_inv",
                (scale_info["shape"][1], scale_info["shape"][0], expert_count),
                QTYPE_F32,
                item.role + "_scale",
                source=item.source,
                expert_layer=item.expert_layer,
                expert_part=item.expert_part,
                expert_count=expert_count,
                raw_copy=True,
                expert_scale=True,
            )
            scale.nbytes = qtype_nbytes(scale.qtype, scale.shape)
            native.append(scale)
            for expert in range(expert_count):
                source = item.source.format(expert=expert)
                if db.info(source)["dtype"] != "F8_E4M3":
                    fail(f"native expert source is not FP8 E4M3: {source}")
                if db.info(source)["shape"] != info["shape"]:
                    fail(f"native expert shape mismatch: {source}")
                source_scale = db.info(source + "_scale_inv")
                if source_scale["dtype"] != "F32" or source_scale["shape"] != scale_info["shape"]:
                    fail(f"native FP8 scale mismatch: {source}_scale_inv")
                covered.update((source, source + "_scale_inv"))
            continue

        info = db.info(item.source)
        try:
            qtype = SOURCE_QTYPES[info["dtype"]]
        except KeyError:
            fail(f"unsupported native source dtype {info['dtype']} for {item.source}")
        if info["dtype"] == "F8_E4M3" and (item.row_count is not None or item.transform is not None):
            fail(f"native FP8 slicing or transforms are unsupported for {item.source}")
        weight = dataclasses.replace(item, qtype=qtype, raw_copy=True)
        weight.nbytes = qtype_nbytes(weight.qtype, weight.shape)
        native.append(weight)
        covered.add(item.source)
        if info["dtype"] == "F8_E4M3":
            scale_name = item.source + "_scale_inv"
            scale_info = db.info(scale_name)
            if scale_info["dtype"] != "F32":
                fail(f"native FP8 scale is not F32: {scale_name}")
            scale = TensorPlan(
                item.name + "_scale_inv",
                tuple(reversed(scale_info["shape"])),
                QTYPE_F32,
                item.role + "_scale",
                scale_name,
                raw_copy=True,
            )
            scale.nbytes = qtype_nbytes(scale.qtype, scale.shape)
            native.append(scale)
            covered.add(scale_name)

    expected = {name for name in db.tensors if not name.startswith("model.visual.")}
    if covered != expected:
        missing = sorted(expected - covered)
        extra = sorted(covered - expected)
        fail(
            "native FP8 source coverage mismatch: "
            f"missing={missing[:1]} extra={extra[:1]}"
        )
    source_bytes = sum(db.info(name)["nbytes"] for name in expected)
    output_bytes = sum(item.nbytes for item in native)
    if output_bytes != source_bytes:
        fail(f"native FP8 payload size {output_bytes} != source payload size {source_bytes}")
    return native


def build_plan(db, artifact):
    plan = []
    embedding_name = "token_embd.weight"
    add_regular(
        plan,
        db,
        embedding_name,
        "model.language_model.embed_tokens.weight",
        regular_qtype(artifact, "embedding", embedding_name, QTYPE_BF16),
        "embedding",
    )
    for layer in range(46):
        prefix = source_prefix(layer)
        if layer < 45:
            add_mhc(plan, db, layer)
        add_regular(plan, db, f"blk.{layer}.attn_norm.weight", f"{prefix}.input_layernorm.weight", QTYPE_F32, "norm")
        if layer < 45 and layer % 4 != 3:
            add_linear_attention(plan, db, layer, artifact)
        else:
            add_dsa_attention(plan, db, layer)
        add_regular(
            plan,
            db,
            f"blk.{layer}.ffn_norm.weight",
            f"{prefix}.post_attention_layernorm.weight",
            QTYPE_F32,
            "norm",
        )
        add_ffn(plan, db, layer, artifact)
        if layer == 45:
            add_regular(plan, db, "blk.45.nextn.eh_proj.weight", f"{prefix}.eh_proj.weight", QTYPE_BF16, "mtp")
            add_regular(plan, db, "blk.45.nextn.enorm.weight", f"{prefix}.enorm.weight", QTYPE_F32, "mtp")
            add_regular(plan, db, "blk.45.nextn.hnorm.weight", f"{prefix}.hnorm.weight", QTYPE_F32, "mtp")
            add_regular(
                plan,
                db,
                "blk.45.nextn.shared_head_norm.weight",
                f"{prefix}.shared_head.norm.weight",
                QTYPE_F32,
                "mtp",
            )
    add_regular(plan, db, "output_norm.weight", "model.language_model.norm.weight", QTYPE_F32, "output")
    output_name = "output.weight"
    add_regular(
        plan,
        db,
        output_name,
        "lm_head.weight",
        regular_qtype(artifact, "output", output_name, QTYPE_BF16),
        "output",
    )

    if artifact == "fp8":
        plan = native_fp8_plan(db, plan)

    names = [item.name for item in plan]
    if len(names) != len(set(names)):
        fail("duplicate GGUF tensor names in conversion plan")
    offset = 0
    for item in plan:
        item.offset = offset
        offset += align(item.nbytes)
    return plan


def model_metadata(hf_dir, source_revision):
    layer_types = [0 if layer < 45 and layer % 4 != 3 else 1 for layer in range(46)]
    chat_path = os.path.join(hf_dir, "chat_template.jinja")
    with open(chat_path, "rb") as fp:
        chat_template = fp.read()
    records = [
        kv_string("general.architecture", "glm5-next"),
        kv_string("general.name", "GLM-5.3-Flash"),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string("general.source.revision", source_revision),
        kv_u32("glm5-next.block_count", 46),
        kv_u32("glm5-next.trunk_block_count", 45),
        kv_u32("glm5-next.nextn_predict_layers", 1),
        kv_u64("glm5-next.context_length", 1048576),
        kv_u32("glm5-next.embedding_length", 4096),
        kv_u32("glm5-next.vocab_size", 154880),
        kv_u32("glm5-next.feed_forward_length", 12288),
        kv_u32("glm5-next.expert_feed_forward_length", 2048),
        kv_u32("glm5-next.expert_count", 288),
        kv_u32("glm5-next.expert_used_count", 8),
        kv_u32("glm5-next.expert_shared_count", 1),
        kv_u32("glm5-next.leading_dense_block_count", 3),
        kv_f32("glm5-next.expert_weights_scale", 2.5),
        kv_bool("glm5-next.expert_weights_norm", True),
        kv_f32("glm5-next.swiglu_limit", 10.0),
        kv_f32("glm5-next.attention.layer_norm_rms_epsilon", 1.0e-5),
        kv_u32("glm5-next.attention.head_count", 64),
        kv_u32("glm5-next.attention.key_length", 256),
        kv_u32("glm5-next.attention.value_length", 256),
        kv_u32("glm5-next.attention.q_lora_rank", 1536),
        kv_u32("glm5-next.attention.kv_lora_rank", 512),
        kv_u32("glm5-next.attention.rope_dimension_count", 0),
        kv_u32("glm5-next.attention.indexer.head_count", 32),
        kv_u32("glm5-next.attention.indexer.key_length", 128),
        kv_u32("glm5-next.attention.indexer.top_k", 2048),
        kv_u32("glm5-next.attention.indexer.pool_size", 4),
        kv_u32("glm5-next.linear_attention.head_count", 64),
        kv_u32("glm5-next.linear_attention.head_dimension", 128),
        kv_u32("glm5-next.linear_attention.conv_kernel", 4),
        kv_f32("glm5-next.linear_attention.gate_lower_bound", -5.0),
        kv_u32_array("glm5-next.layer_types", layer_types),
        kv_u32("glm5-next.hyper_connection.count", 4),
        kv_u32("glm5-next.hyper_connection.sinkhorn_iterations", 20),
        kv_f32("glm5-next.hyper_connection.epsilon", 1.0e-6),
        kv_string("tokenizer.chat_template", chat_template),
    ]
    return records


class Quantizer:
    def __init__(self, library_path):
        try:
            import numpy as np
        except ImportError as error:
            fail(f"NumPy is required for conversion: {error}")
        self.np = np
        self.lib = ctypes.CDLL(library_path)
        self.lib.ds4q_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ds4q_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.ds4q_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ds4q_quantize_init(QTYPE_IQ2_XXS)
        self.fp8_lut = self._build_fp8_lut()

    def _build_fp8_lut(self):
        np = self.np
        values = np.empty(256, dtype=np.float32)
        for code in range(256):
            absolute = code & 0x7F
            if absolute == 0:
                value = -0.0 if code & 0x80 else 0.0
            elif absolute == 0x7F:
                value = math.nan
            else:
                exponent = (code >> 3) & 0x0F
                mantissa = code & 0x07
                value = math.ldexp(mantissa, -9) if exponent == 0 else math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
                if code & 0x80:
                    value = -value
            values[code] = value
        return values

    def to_f32(self, db, name, row_start=0, row_count=None):
        np = self.np
        info = db.info(name)
        shape = info["shape"]
        raw = db.read(name)
        if info["dtype"] == "F32":
            array = np.frombuffer(raw, dtype="<f4").reshape(shape)
        elif info["dtype"] == "BF16":
            bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            array = bits.view(np.float32).reshape(shape)
        elif info["dtype"] == "F16":
            array = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
        elif info["dtype"] == "F8_E4M3":
            if len(shape) != 2:
                fail(f"unsupported FP8 shape for {name}: {shape}")
            codes = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
            if np.any((codes & 0x7F) == 0x7F):
                fail(f"nonfinite FP8 code in {name}")
            scale_name = name + "_scale_inv"
            scale_info = db.info(scale_name)
            if scale_info["dtype"] != "F32":
                fail(f"{scale_name} is not F32")
            expected_scale_shape = [(dim + 127) // 128 for dim in shape]
            if scale_info["shape"] != expected_scale_shape:
                fail(
                    f"{scale_name} has shape {scale_info['shape']}, "
                    f"expected {expected_scale_shape}"
                )
            scales = np.frombuffer(db.read(scale_name), dtype="<f4").reshape(scale_info["shape"])
            expanded = np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)
            expanded = expanded[: shape[0], : shape[1]]
            array = self.fp8_lut[codes] * expanded
        else:
            fail(f"unsupported source dtype {info['dtype']} for {name}")
        if row_count is not None:
            array = array[row_start : row_start + row_count]
        return np.ascontiguousarray(array, dtype=np.float32)

    def encode(self, array, qtype, imatrix=None):
        np = self.np
        if qtype == QTYPE_F32:
            return np.asarray(array, dtype="<f4").tobytes()
        if qtype == QTYPE_F16:
            return np.asarray(array, dtype="<f2").tobytes()
        if qtype == QTYPE_BF16:
            values = np.asarray(array, dtype="<f4")
            bits = values.view(np.uint32)
            rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
            return (rounded >> 16).astype("<u2").tobytes()
        if array.ndim < 2:
            fail(f"cannot quantize rank-{array.ndim} tensor as {QTYPE_NAMES[qtype]}")
        ncols = array.shape[-1]
        nrows = array.size // ncols
        expected = qtype_nbytes(qtype, (ncols, nrows))
        output = np.empty(expected, dtype=np.uint8)
        array = np.ascontiguousarray(array.reshape(nrows, ncols), dtype=np.float32)
        if qtype == QTYPE_IQ2_XXS and imatrix is None:
            imatrix = np.square(array, dtype=np.float32).sum(axis=0, dtype=np.float32)
        if imatrix is not None:
            imatrix = np.ascontiguousarray(imatrix, dtype=np.float32)
            if imatrix.size != ncols:
                fail(f"imatrix width {imatrix.size} does not match tensor width {ncols}")
            imatrix_ptr = imatrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            imatrix_ptr = None
        written = self.lib.ds4q_quantize_chunk(
            qtype,
            array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data,
            0,
            nrows,
            ncols,
            imatrix_ptr,
        )
        if written != expected:
            fail(f"quantizer wrote {written} bytes, expected {expected}")
        return output.tobytes()


class Imatrix:
    def __init__(self, path, np):
        self.np = np
        self.entries = {}
        if not path:
            return
        with open(path, "rb") as fp:
            n_entries = struct.unpack("<i", read_exact(fp, 4, "imatrix entry count"))[0]
            if n_entries <= 0:
                fail("imatrix has no entries")
            for _ in range(n_entries):
                name_len = struct.unpack("<i", read_exact(fp, 4, "imatrix name length"))[0]
                if name_len <= 0 or name_len > 4096:
                    fail("invalid imatrix name length")
                name = read_exact(fp, name_len, "imatrix name").decode("utf-8")
                read_exact(fp, 4, "imatrix call count")
                value_count = struct.unpack("<i", read_exact(fp, 4, "imatrix value count"))[0]
                if value_count <= 0:
                    fail(f"invalid imatrix value count for {name}")
                values = np.frombuffer(read_exact(fp, value_count * 4, "imatrix values"), dtype="<f4").copy()
                if not np.all(np.isfinite(values)):
                    fail(f"nonfinite imatrix values for {name}")
                self.entries[name] = values

    def expert(self, tensor_name, expert, width, expert_count=288):
        values = self.entries.get(tensor_name)
        if values is None:
            return None
        expected = expert_count * width
        if values.size != expected:
            fail(f"imatrix {tensor_name} has {values.size} values, expected {expected}")
        result = values[expert * width : (expert + 1) * width]
        if not self.np.any(result > 0.0):
            return None
        return result


def tensor_header(item):
    return (
        pack_string(item.name)
        + struct.pack("<I", len(item.shape))
        + struct.pack(f"<{len(item.shape)}Q", *item.shape)
        + struct.pack("<IQ", item.qtype, item.offset)
    )


def print_plan(plan, kv_records, tokenizer_records):
    by_type = {}
    by_role = {}
    for item in plan:
        by_type[item.qtype] = by_type.get(item.qtype, 0) + item.nbytes
        by_role[item.role] = by_role.get(item.role, 0) + item.nbytes
    kv_bytes = sum(map(len, kv_records)) + sum(map(len, tokenizer_records))
    tensor_info_bytes = sum(len(tensor_header(item)) for item in plan)
    data_offset = align(4 + 4 + 8 + 8 + kv_bytes + tensor_info_bytes)
    data_bytes = sum(align(item.nbytes) for item in plan)
    print(f"tensors: {len(plan)}")
    print(f"metadata_records: {len(kv_records) + len(tokenizer_records)}")
    print(f"metadata_bytes: {data_offset}")
    print(f"tensor_bytes: {sum(item.nbytes for item in plan)}")
    print(f"file_bytes: {data_offset + data_bytes}")
    for qtype, nbytes in sorted(by_type.items()):
        print(f"type_bytes: {QTYPE_NAMES[qtype]} {nbytes}")
    for role, nbytes in sorted(by_role.items()):
        print(f"role_bytes: {role} {nbytes}")
    return data_offset, data_bytes


def transform_regular(values, item):
    if item.transform is None:
        return values
    if item.transform not in ("kv_b_k", "kv_b_v"):
        fail(f"unknown transform {item.transform} for {item.name}")
    if values.ndim != 2 or len(item.shape) != 3:
        fail(f"{item.name}: invalid combined kv_b shapes {values.shape} -> {item.shape}")
    n_head = item.shape[2]
    rank = item.shape[1] if item.transform == "kv_b_k" else item.shape[0]
    selected = item.shape[0] if item.transform == "kv_b_k" else item.shape[1]
    if values.shape[1] != rank or values.shape[0] % n_head:
        fail(f"{item.name}: invalid combined kv_b shape {values.shape}")
    combined = values.shape[0] // n_head
    other = combined - selected
    if other <= 0:
        fail(f"{item.name}: invalid combined kv_b dimensions {combined} and {selected}")
    k_dim = selected if item.transform == "kv_b_k" else other
    v_dim = other if item.transform == "kv_b_k" else selected
    per_head = values.reshape(n_head, k_dim + v_dim, rank)
    if item.transform == "kv_b_k":
        return per_head[:, :k_dim, :].transpose(0, 2, 1).copy()
    return per_head[:, k_dim:, :].copy()


def iter_native_tensor_bytes(item, db, np):
    if not item.raw_copy:
        fail(f"{item.name}: tensor is not a native payload")
    if item.is_expert:
        expert_count = item.expert_count or 288
        for expert in range(expert_count):
            source = item.source.format(expert=expert)
            if item.expert_scale:
                source += "_scale_inv"
            yield from db.iter_read(source)
        return

    info = db.info(item.source)
    if item.transform is None:
        row_bytes = info["nbytes"] // info["shape"][0] if item.row_count is not None else 0
        byte_start = item.row_start * row_bytes
        byte_count = item.row_count * row_bytes if item.row_count is not None else info["nbytes"]
        yield from db.iter_read(item.source, byte_start, byte_count)
        return
    if info["dtype"] != "BF16":
        fail(f"{item.name}: raw transform requires BF16 input")
    values = np.frombuffer(db.read(item.source), dtype="<u2").reshape(info["shape"])
    yield transform_regular(values, item).astype("<u2", copy=False).tobytes()


def write_regular(fp, item, db, quantizer):
    if item.raw_copy:
        written = 0
        for data in iter_native_tensor_bytes(item, db, quantizer.np):
            fp.write(data)
            written += len(data)
        if written != item.nbytes:
            fail(f"{item.name}: copied {written} bytes, expected {item.nbytes}")
        return
    values = quantizer.to_f32(db, item.source, item.row_start, item.row_count)
    values = transform_regular(values, item)
    data = quantizer.encode(values, item.qtype)
    if len(data) != item.nbytes:
        fail(f"{item.name}: generated {len(data)} bytes, expected {item.nbytes}")
    fp.write(data)


def write_experts(fp, item, db, quantizer, imatrix, threads):
    expert_count = item.expert_count or 288
    width = item.shape[0]

    def convert(expert):
        source = item.source.format(expert=expert)
        if item.raw_copy:
            if item.expert_scale:
                source += "_scale_inv"
            return db.read(source), False
        values = quantizer.to_f32(db, source)
        weights = imatrix.expert(item.name, expert, width, expert_count)
        return quantizer.encode(values, item.qtype, weights), weights is None

    per_expert = item.nbytes // expert_count
    fallback_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        window = max(threads * 2, 1)
        for start in range(0, expert_count, window):
            futures = [executor.submit(convert, expert) for expert in range(start, min(start + window, expert_count))]
            for future in futures:
                data, fallback = future.result()
                fallback_count += int(fallback)
                if len(data) != per_expert:
                    fail(f"{item.name}: expert generated {len(data)} bytes, expected {per_expert}")
                fp.write(data)
    if imatrix.entries and fallback_count:
        print(
            f"glm53-quantize: {item.name} used weight-based fallback for "
            f"{fallback_count}/{expert_count} unobserved experts",
            file=sys.stderr,
        )


def file_sha256(path):
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        while chunk := fp.read(8 << 20):
            digest.update(chunk)
    return digest.digest()


def conversion_signature(plan, kv_records, tokenizer_records, imatrix_path=None):
    digest = hashlib.sha256()
    for record in kv_records + tokenizer_records:
        digest.update(record)
    for item in plan:
        digest.update(tensor_header(item))
        digest.update((item.source or "").encode())
        digest.update((item.transform or "").encode())
        digest.update(struct.pack("<I", item.expert_count or 0))
        digest.update(bytes((item.raw_copy, item.expert_scale)))
    imatrix_digest = file_sha256(imatrix_path)
    if imatrix_digest is not None:
        digest.update(b"imatrix\0")
        digest.update(imatrix_digest)
    return digest.hexdigest()


def save_resume_state(path, signature, completed):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as fp:
        json.dump({"version": 1, "signature": signature, "completed": completed}, fp)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temporary, path)


def load_resume_state(path, signature, plan):
    with open(path, "r", encoding="utf-8") as fp:
        state = json.load(fp)
    if state.get("version") != 1 or state.get("signature") != signature:
        fail(f"resume state does not match this conversion: {path}")
    completed = state.get("completed")
    if not isinstance(completed, int) or completed < 0 or completed > len(plan):
        fail(f"invalid completed tensor count in {path}")
    return completed


def write_gguf(args, plan, kv_records, tokenizer_records, db):
    library = args.quants_library
    if not library:
        suffix = "dylib" if sys.platform == "darwin" else "so"
        library = os.path.join(os.path.dirname(__file__), f"libds4quants.{suffix}")
    if not os.path.isfile(library):
        fail(f"quantizer library not found: {library}; run make -C gguf-tools")
    quantizer = Quantizer(library)
    imatrix = Imatrix(args.imatrix, quantizer.np)

    data_offset, data_bytes = print_plan(plan, kv_records, tokenizer_records)
    required = data_offset + data_bytes + 32 * (1 << 30)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.out))).free
    if free < required:
        fail(f"insufficient free space: need output plus reserve {required}, have {free}")

    partial = args.out + ".partial"
    resume_path = partial + ".resume.json"
    signature = conversion_signature(plan, kv_records, tokenizer_records, args.imatrix)
    if os.path.exists(args.out) and not args.overwrite:
        fail(f"output exists: {args.out}; use --overwrite")
    if args.overwrite:
        for path in (partial, resume_path):
            if os.path.exists(path):
                os.unlink(path)

    resume = os.path.exists(partial) or os.path.exists(resume_path)
    if resume and not args.resume:
        fail(f"partial conversion exists: use --resume or --overwrite for {partial}")
    if resume and not (os.path.exists(partial) and os.path.exists(resume_path)):
        fail("partial GGUF and resume journal must either both exist or both be absent")

    all_kv = kv_records + tokenizer_records
    completed = 0
    if resume:
        completed = load_resume_state(resume_path, signature, plan)
        expected_size = data_offset
        if completed:
            previous = plan[completed - 1]
            expected_size += previous.offset + align(previous.nbytes)
        with open(partial, "r+b") as fp:
            fp.truncate(expected_size)
        print(f"glm53-quantize: resuming after {completed} tensors", file=sys.stderr)
    else:
        with open(partial, "wb") as fp:
            fp.write(b"GGUF")
            fp.write(struct.pack("<IQQ", GGUF_VERSION, len(plan), len(all_kv)))
            for record in all_kv:
                fp.write(record)
            for item in plan:
                fp.write(tensor_header(item))
            if fp.tell() > data_offset:
                fail("GGUF header exceeds planned data offset")
            fp.write(bytes(data_offset - fp.tell()))
            fp.flush()
            os.fsync(fp.fileno())
        save_resume_state(resume_path, signature, 0)

    with open(partial, "r+b") as fp:
        fp.seek(data_offset)
        if completed:
            previous = plan[completed - 1]
            fp.seek(data_offset + previous.offset + align(previous.nbytes))
        started = time.monotonic()
        for index, item in enumerate(plan[completed:], completed + 1):
            expected_offset = data_offset + item.offset
            if fp.tell() != expected_offset:
                fail(f"output offset mismatch for {item.name}: {fp.tell()} != {expected_offset}")
            tensor_started = time.monotonic()
            if item.is_expert:
                write_experts(fp, item, db, quantizer, imatrix, args.threads)
            else:
                write_regular(fp, item, db, quantizer)
            fp.write(bytes(align(item.nbytes) - item.nbytes))
            fp.flush()
            os.fsync(fp.fileno())
            save_resume_state(resume_path, signature, index)
            elapsed = time.monotonic() - tensor_started
            total_elapsed = time.monotonic() - started
            print(
                f"[{index:4d}/{len(plan):4d}] {item.name} {QTYPE_NAMES[item.qtype]} "
                f"{item.nbytes / (1 << 30):.3f} GiB {elapsed:.1f}s total={total_elapsed / 60:.1f}m",
                file=sys.stderr,
                flush=True,
            )
    os.replace(partial, args.out)
    os.unlink(resume_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True, help="official GLM-5.3 Flash snapshot")
    parser.add_argument("--tokenizer-template", required=True, help="GLM GGUF supplying tokenizer metadata")
    parser.add_argument("--out", help="output GGUF")
    parser.add_argument("--artifact", choices=("q4", "q2", "fp8"), default="q4")
    parser.add_argument("--imatrix", help="legacy DS4/llama.cpp imatrix .dat")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--source-revision", default="84c6a6aa9497188e15a635ba793b0f95a79b1033")
    parser.add_argument("--quants-library", help="path to libds4quants")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume a matching per-tensor partial conversion")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 64:
        parser.error("--threads must be between 1 and 64")
    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run is used")
    if args.artifact == "q4" and args.imatrix:
        print("glm53-quantize: using the imatrix for Q4_K expert scale selection", file=sys.stderr)
    if args.artifact == "fp8" and args.imatrix:
        parser.error("--imatrix does not apply to the lossless FP8 artifact")
    return args


def main():
    args = parse_args()
    db = SourceDB(args.hf)
    try:
        plan = build_plan(db, args.artifact)
        tokenizer_records, template_tokens = load_tokenizer_records(args.tokenizer_template)
        validate_tokenizer_template(args.hf, template_tokens, 154880)
        kv_records = model_metadata(args.hf, args.source_revision)
        if args.artifact == "fp8":
            kv_records.append(kv_bool("glm5-next.native_fp8", True))
        if args.dry_run:
            print_plan(plan, kv_records, tokenizer_records)
        else:
            write_gguf(args, plan, kv_records, tokenizer_records, db)
            print(f"glm53-quantize: wrote {args.out}", file=sys.stderr)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"glm53-quantize: error: {error}", file=sys.stderr)
        sys.exit(1)
