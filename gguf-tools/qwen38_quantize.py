#!/usr/bin/env python3
"""Build DwarfStar Qwen3.8-Flash-Next GGUF files from the official FP8.

Main artifact (``--artifact q2``, the DwarfStar recipe for 64 GB Macs):
routed gate/up experts IQ2_XXS (imatrix-guided when provided), routed down
experts MXFP4 (their 640-wide rows are not QK_K aligned), Q8_0 for every
dense projection, F32 norms/routers. The 51.2B-row PLE n-gram table is NOT
part of the main file: ``--ple-out`` writes it as a separate Q8_0 sidecar
GGUF meant to be mmapped from SSD. The 4B MTP module (mtp.*) and the vision
encoder (model.visual.*) are excluded from the v1 artifact.

Tensor names and metadata keys follow the ds4 loader contract
(``general.architecture = qwen38-next``); see config_validate_qwen38_model
and weights_bind_qwen38_layer in ds4.c.
"""

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

from glm53_manifest import load_index, load_safetensors_header

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
QTYPE_IQ2_XXS = 16
QTYPE_BF16 = 30
QTYPE_MXFP4 = 39

QTYPE_NAMES = {
    QTYPE_F32: "F32",
    QTYPE_F16: "F16",
    QTYPE_Q8_0: "Q8_0",
    QTYPE_Q2_K: "Q2_K",
    QTYPE_Q4_K: "Q4_K",
    QTYPE_IQ2_XXS: "IQ2_XXS",
    QTYPE_BF16: "BF16",
    QTYPE_MXFP4: "MXFP4",
}

QTYPE_LAYOUT = {
    QTYPE_F32: (1, 4),
    QTYPE_F16: (1, 2),
    QTYPE_Q8_0: (32, 34),
    QTYPE_Q2_K: (256, 84),
    QTYPE_Q4_K: (256, 144),
    QTYPE_IQ2_XXS: (256, 66),
    QTYPE_BF16: (1, 2),
    QTYPE_MXFP4: (32, 17),
}

N_LAYER = 48
N_EXPERT = 512
N_EMBD = 2560
N_FF_EXP = 640
N_VOCAB = 248320
GDN_QK_DIM = 16 * 128
GDN_V_DIM = 48 * 128
QSA_Q_DIM = 24 * 256
IDX_Q_DIM = 4 * 128
IDX_K_DIM = 128
PLE_LAYER = 1
PLE_HEADS = 16
PLE_ROW_LEN = 160
PLE_SHARDS = 128

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


def kv_u64_array(key, values):
    return (
        pack_string(key)
        + struct.pack("<IIQ", GGUF_ARRAY, GGUF_UINT64, len(values))
        + struct.pack(f"<{len(values)}Q", *values)
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
    """Copy tokenizer.* records (except chat_template) from a template GGUF."""
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


def qwen38_layer_is_gdn(layer):
    return layer % 4 != 3


def validate_qwen38_index(weight_map):
    """Strict shape-of-the-index check for the text model this plan covers."""
    names = set(weight_map)
    for required in (
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
    ):
        if required not in names:
            fail(f"missing source tensor {required}")
    for layer in range(N_LAYER):
        prefix = f"{LAYER_PREFIX}.{layer}"
        for site in ("attn_hyper_connection", "mlp_hyper_connection"):
            for leaf in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up",
                         "block_inject_weight"):
                if f"{prefix}.{site}.{leaf}.weight" not in names:
                    fail(f"missing {prefix}.{site}.{leaf}.weight")
        if qwen38_layer_is_gdn(layer):
            for leaf in ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight",
                         "in_proj_b.weight", "conv1d.weight", "A_log", "dt_bias",
                         "norm.weight", "out_proj.weight"):
                if f"{prefix}.linear_attn.{leaf}" not in names:
                    fail(f"missing {prefix}.linear_attn.{leaf}")
        else:
            for leaf in ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                         "q_norm.weight", "k_norm.weight", "o_proj.weight",
                         "indexer.index_qk_proj.weight",
                         "indexer.q_layernorm.weight", "indexer.k_layernorm.weight"):
                if f"{prefix}.self_attn.{leaf}" not in names:
                    fail(f"missing {prefix}.self_attn.{leaf}")
        for leaf in ("gate.weight", "shared_expert_gate.weight",
                     "shared_expert.gate_proj.weight", "shared_expert.up_proj.weight",
                     "shared_expert.down_proj.weight"):
            if f"{prefix}.mlp.{leaf}" not in names:
                fail(f"missing {prefix}.mlp.{leaf}")
        for expert in range(N_EXPERT):
            for part in ("gate", "up", "down"):
                if f"{prefix}.mlp.experts.{expert}.{part}_proj.weight" not in names:
                    fail(f"missing expert {expert} {part} in layer {layer}")
    ple = f"{LAYER_PREFIX}.{PLE_LAYER}.ple"
    for leaf in ("conv1d.weight", "key_proj.weight", "value_proj.weight",
                 "norm_conv.weight", "norm_key.weight", "norm_query.weight",
                 "ple_embedding.layer_multipliers",
                 "ple_embedding.ngram_heads_offsets",
                 "ple_embedding.ngram_heads_vocab_sizes",
                 "ple_embedding.ngram_embedding.weight_scale"):
        if f"{ple}.{leaf}" not in names:
            fail(f"missing {ple}.{leaf}")
    for shard in range(PLE_SHARDS):
        if f"{ple}.ple_embedding.ngram_embedding.shard_{shard}.weight" not in names:
            fail(f"missing PLE shard {shard}")


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
    transform: str | None = None
    offset: int = 0
    nbytes: int = 0

    @property
    def is_expert(self):
        return self.expert_layer is not None


class SourceDB:
    def __init__(self, hf_dir):
        self.hf_dir = hf_dir
        index_path = os.path.join(hf_dir, "model.safetensors.index.json")
        document, self.weight_map = load_index(index_path)
        validate_qwen38_index(self.weight_map)
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

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


def source_prefix(layer):
    return f"{LAYER_PREFIX}.{layer}"


def transform_output_shape(db, source, transform):
    info = db.info(source)
    shape = info["shape"]
    if transform == "norm_plus_one":
        # Qwen4ExpTextRMSNorm stores the weight as an offset from 1
        # (Gemma-style); bake the +1 so the engine kernels stay standard.
        return tuple(reversed(shape))
    if transform == "conv_squeeze":
        # [channels, 1, kernel] -> GGUF dims (kernel, 1, channels)
        if len(shape) != 3 or shape[1] != 1:
            fail(f"{source}: unexpected conv weight shape {shape}")
        return (shape[2], 1, shape[0])
    if transform in ("qsa_deinterleave_q", "qsa_deinterleave_gate"):
        # [heads * 2 * head_dim, hidden]: per head [query | gate] halves.
        if len(shape) != 2 or shape[0] != 2 * QSA_Q_DIM or shape[1] != N_EMBD:
            fail(f"{source}: unexpected fused q/gate shape {shape}")
        return (N_EMBD, QSA_Q_DIM)
    if transform == "squeeze":
        return (product(shape),)
    fail(f"unknown transform {transform} for {source}")


def add_regular(plan, db, name, source, qtype, role,
                row_start=0, row_count=None, transform=None):
    info = db.info(source)
    source_shape = info["shape"]
    if transform is not None:
        shape = transform_output_shape(db, source, transform)
    else:
        if row_count is not None:
            if (len(source_shape) != 2 or row_start < 0 or row_count <= 0 or
                    row_start + row_count > source_shape[0]):
                fail(f"invalid row slice for {source}")
            source_shape = [row_count, source_shape[1]]
        shape = tuple(reversed(source_shape))
    item = TensorPlan(name, shape, qtype, role, source, row_start, row_count,
                      transform=transform)
    item.nbytes = qtype_nbytes(qtype, shape)
    plan.append(item)


def add_experts(plan, db, layer, part, qtype):
    first = f"{source_prefix(layer)}.mlp.experts.0.{part}_proj.weight"
    shape = db.info(first)["shape"]
    if len(shape) != 2:
        fail(f"expert source is not a matrix: {first}")
    expected = [N_FF_EXP, N_EMBD] if part in ("gate", "up") else [N_EMBD, N_FF_EXP]
    if shape != expected:
        fail(f"unexpected expert shape for {first}: {shape}, expected {expected}")
    for expert in range(N_EXPERT):
        name = f"{source_prefix(layer)}.mlp.experts.{expert}.{part}_proj.weight"
        if db.info(name)["shape"] != shape:
            fail(f"expert shape mismatch: {name}")
    item = TensorPlan(
        f"blk.{layer}.ffn_{part}_exps.weight",
        (shape[1], shape[0], N_EXPERT),
        qtype,
        f"routed_{part}",
        source=f"{source_prefix(layer)}.mlp.experts.{{expert}}.{part}_proj.weight",
        expert_layer=layer,
        expert_part=part,
    )
    item.nbytes = qtype_nbytes(qtype, item.shape)
    plan.append(item)


def add_hc(plan, db, layer):
    prefix = source_prefix(layer)
    for gguf_site, src_site in (("attn", "attn_hyper_connection"),
                                ("ffn", "mlp_hyper_connection")):
        src = f"{prefix}.{src_site}"
        add_regular(plan, db, f"blk.{layer}.hc_{gguf_site}_down.weight",
                    f"{src}.input_mix_weight_down.weight", QTYPE_Q8_0, "hc")
        add_regular(plan, db, f"blk.{layer}.hc_{gguf_site}_up.weight",
                    f"{src}.input_mix_weight_up.weight", QTYPE_Q8_0, "hc")
        add_regular(plan, db, f"blk.{layer}.hc_{gguf_site}_inject.weight",
                    f"{src}.block_inject_weight.weight", QTYPE_F32, "hc")
        add_regular(plan, db, f"blk.{layer}.hc_{gguf_site}_norm.weight",
                    f"{src}.hc_norm.weight", QTYPE_F32, "hc",
                    transform="norm_plus_one")


def add_gdn(plan, db, layer):
    prefix = f"{source_prefix(layer)}.linear_attn"
    qkv = f"{prefix}.in_proj_qkv.weight"
    qkv_shape = db.info(qkv)["shape"]
    if qkv_shape != [GDN_QK_DIM * 2 + GDN_V_DIM, N_EMBD]:
        fail(f"unexpected GDN qkv shape for layer {layer}: {qkv_shape}")
    add_regular(plan, db, f"blk.{layer}.kda_q.weight", qkv, QTYPE_Q8_0,
                "linear_attention", row_start=0, row_count=GDN_QK_DIM)
    add_regular(plan, db, f"blk.{layer}.kda_k.weight", qkv, QTYPE_Q8_0,
                "linear_attention", row_start=GDN_QK_DIM, row_count=GDN_QK_DIM)
    add_regular(plan, db, f"blk.{layer}.kda_v.weight", qkv, QTYPE_Q8_0,
                "linear_attention", row_start=2 * GDN_QK_DIM, row_count=GDN_V_DIM)
    conv = f"{prefix}.conv1d.weight"
    conv_shape = db.info(conv)["shape"]
    if conv_shape != [GDN_QK_DIM * 2 + GDN_V_DIM, 1, 4]:
        fail(f"unexpected GDN conv shape for layer {layer}: {conv_shape}")
    for target, start, count in (("q_conv", 0, GDN_QK_DIM),
                                 ("k_conv", GDN_QK_DIM, GDN_QK_DIM),
                                 ("v_conv", 2 * GDN_QK_DIM, GDN_V_DIM)):
        item = TensorPlan(
            f"blk.{layer}.kda_{target}.weight", (4, 1, count), QTYPE_F32,
            "linear_attention", source=conv, row_start=start, row_count=count,
            transform="conv_slice",
        )
        item.nbytes = qtype_nbytes(QTYPE_F32, item.shape)
        plan.append(item)
    add_regular(plan, db, f"blk.{layer}.kda_gate.weight",
                f"{prefix}.in_proj_z.weight", QTYPE_Q8_0, "linear_attention")
    add_regular(plan, db, f"blk.{layer}.kda_alpha.weight",
                f"{prefix}.in_proj_a.weight", QTYPE_F32, "linear_attention")
    add_regular(plan, db, f"blk.{layer}.kda_beta.weight",
                f"{prefix}.in_proj_b.weight", QTYPE_F32, "linear_attention")
    add_regular(plan, db, f"blk.{layer}.kda_a_log.weight",
                f"{prefix}.A_log", QTYPE_F32, "linear_attention")
    add_regular(plan, db, f"blk.{layer}.kda_dt_bias.weight",
                f"{prefix}.dt_bias", QTYPE_F32, "linear_attention")
    add_regular(plan, db, f"blk.{layer}.kda_o_norm.weight",
                f"{prefix}.norm.weight", QTYPE_F32, "linear_attention")
    add_regular(plan, db, f"blk.{layer}.kda_output.weight",
                f"{prefix}.out_proj.weight", QTYPE_Q8_0, "linear_attention")


def add_qsa(plan, db, layer):
    prefix = f"{source_prefix(layer)}.self_attn"
    add_regular(plan, db, f"blk.{layer}.attn_q.weight",
                f"{prefix}.q_proj.weight", QTYPE_Q8_0, "qsa",
                transform="qsa_deinterleave_q")
    add_regular(plan, db, f"blk.{layer}.attn_gate.weight",
                f"{prefix}.q_proj.weight", QTYPE_Q8_0, "qsa",
                transform="qsa_deinterleave_gate")
    add_regular(plan, db, f"blk.{layer}.attn_k.weight",
                f"{prefix}.k_proj.weight", QTYPE_Q8_0, "qsa")
    add_regular(plan, db, f"blk.{layer}.attn_v.weight",
                f"{prefix}.v_proj.weight", QTYPE_Q8_0, "qsa")
    add_regular(plan, db, f"blk.{layer}.attn_q_norm.weight",
                f"{prefix}.q_norm.weight", QTYPE_F32, "qsa",
                transform="norm_plus_one")
    add_regular(plan, db, f"blk.{layer}.attn_k_norm.weight",
                f"{prefix}.k_norm.weight", QTYPE_F32, "qsa",
                transform="norm_plus_one")
    add_regular(plan, db, f"blk.{layer}.attn_output.weight",
                f"{prefix}.o_proj.weight", QTYPE_Q8_0, "qsa")

    idx = f"{prefix}.indexer.index_qk_proj.weight"
    idx_shape = db.info(idx)["shape"]
    if idx_shape != [IDX_Q_DIM + IDX_K_DIM, N_EMBD]:
        fail(f"unexpected indexer qk shape for layer {layer}: {idx_shape}")
    add_regular(plan, db, f"blk.{layer}.indexer.q_proj.weight", idx, QTYPE_BF16,
                "indexer", row_start=0, row_count=IDX_Q_DIM)
    add_regular(plan, db, f"blk.{layer}.indexer.k_proj.weight", idx, QTYPE_BF16,
                "indexer", row_start=IDX_Q_DIM, row_count=IDX_K_DIM)
    add_regular(plan, db, f"blk.{layer}.indexer.q_norm.weight",
                f"{prefix}.indexer.q_layernorm.weight", QTYPE_F32, "indexer",
                transform="norm_plus_one")
    add_regular(plan, db, f"blk.{layer}.indexer.k_norm.weight",
                f"{prefix}.indexer.k_layernorm.weight", QTYPE_F32, "indexer",
                transform="norm_plus_one")


def add_ffn(plan, db, layer, artifact):
    prefix = f"{source_prefix(layer)}.mlp"
    add_regular(plan, db, f"blk.{layer}.ffn_gate_inp.weight",
                f"{prefix}.gate.weight", QTYPE_F32, "router")
    add_regular(plan, db, f"blk.{layer}.ffn_gate_inp_shexp.weight",
                f"{prefix}.shared_expert_gate.weight", QTYPE_F32, "router",
                transform="squeeze")
    if artifact == "q4":
        routed = {"gate": QTYPE_Q4_K, "up": QTYPE_Q4_K, "down": QTYPE_MXFP4}
    else:
        routed = {"gate": QTYPE_IQ2_XXS, "up": QTYPE_IQ2_XXS, "down": QTYPE_MXFP4}
    for part in ("gate", "up", "down"):
        add_experts(plan, db, layer, part, routed[part])
    for part in ("gate", "up", "down"):
        add_regular(plan, db, f"blk.{layer}.ffn_{part}_shexp.weight",
                    f"{prefix}.shared_expert.{part}_proj.weight", QTYPE_Q8_0,
                    "shared_expert")


def add_ple_projections(plan, db):
    prefix = f"{source_prefix(PLE_LAYER)}.ple"
    item = TensorPlan(
        f"blk.{PLE_LAYER}.ple_conv1d.weight", None, QTYPE_F32, "ple",
        source=f"{prefix}.conv1d.weight", transform="conv_squeeze",
    )
    item.shape = transform_output_shape(db, item.source, "conv_squeeze")
    item.nbytes = qtype_nbytes(QTYPE_F32, item.shape)
    plan.append(item)
    add_regular(plan, db, f"blk.{PLE_LAYER}.ple_key.weight",
                f"{prefix}.key_proj.weight", QTYPE_Q8_0, "ple")
    add_regular(plan, db, f"blk.{PLE_LAYER}.ple_value.weight",
                f"{prefix}.value_proj.weight", QTYPE_Q8_0, "ple")
    add_regular(plan, db, f"blk.{PLE_LAYER}.ple_norm_conv.weight",
                f"{prefix}.norm_conv.weight", QTYPE_F32, "ple",
                transform="norm_plus_one")
    add_regular(plan, db, f"blk.{PLE_LAYER}.ple_norm_key.weight",
                f"{prefix}.norm_key.weight", QTYPE_F32, "ple",
                transform="norm_plus_one")
    add_regular(plan, db, f"blk.{PLE_LAYER}.ple_norm_query.weight",
                f"{prefix}.norm_query.weight", QTYPE_F32, "ple",
                transform="norm_plus_one")


def build_plan(db, artifact):
    plan = []
    add_regular(plan, db, "token_embd.weight",
                "model.language_model.embed_tokens.weight", QTYPE_Q8_0, "embedding")
    for layer in range(N_LAYER):
        add_hc(plan, db, layer)
        if qwen38_layer_is_gdn(layer):
            add_gdn(plan, db, layer)
        else:
            add_qsa(plan, db, layer)
        add_ffn(plan, db, layer, artifact)
        if layer == PLE_LAYER:
            add_ple_projections(plan, db)
    mixer = "model.language_model.hyper_connection_mixer"
    add_regular(plan, db, "output_hc_down.weight",
                f"{mixer}.input_mix_weight_down.weight", QTYPE_Q8_0, "output")
    add_regular(plan, db, "output_hc_up.weight",
                f"{mixer}.input_mix_weight_up.weight", QTYPE_Q8_0, "output")
    add_regular(plan, db, "output_hc_norm.weight",
                f"{mixer}.hc_norm.weight", QTYPE_F32, "output",
                transform="norm_plus_one")
    add_regular(plan, db, "output.weight", "lm_head.weight", QTYPE_Q8_0, "output")

    names = [item.name for item in plan]
    if len(names) != len(set(names)):
        fail("duplicate GGUF tensor names in conversion plan")
    offset = 0
    for item in plan:
        item.offset = offset
        offset += align(item.nbytes)
    return plan


def read_i64_array(db, np, name, expected_len):
    info = db.info(name)
    if info["dtype"] != "I64" or info["shape"] != [expected_len]:
        fail(f"{name}: expected I64 [{expected_len}], got {info['dtype']} {info['shape']}")
    values = np.frombuffer(db.read(name), dtype="<i8")
    if np.any(values < 0):
        fail(f"{name}: negative value")
    return [int(v) for v in values]


def ple_metadata_values(db, np):
    prefix = f"{source_prefix(PLE_LAYER)}.ple.ple_embedding"
    multipliers = read_i64_array(db, np, f"{prefix}.layer_multipliers", 3)
    offsets = read_i64_array(db, np, f"{prefix}.ngram_heads_offsets", PLE_HEADS)
    vocab_sizes = read_i64_array(db, np, f"{prefix}.ngram_heads_vocab_sizes", PLE_HEADS)
    total = 0
    for head in range(PLE_HEADS):
        if offsets[head] != total:
            fail("ple head offsets are not the running sum of vocab sizes")
        total += vocab_sizes[head]
    return multipliers, offsets, vocab_sizes, total


def model_metadata(hf_dir, source_revision, ple_values):
    layer_types = [0 if qwen38_layer_is_gdn(layer) else 1 for layer in range(N_LAYER)]
    chat_path = os.path.join(hf_dir, "chat_template.jinja")
    with open(chat_path, "rb") as fp:
        chat_template = fp.read()
    multipliers, offsets, vocab_sizes, _total = ple_values
    records = [
        kv_string("general.architecture", "qwen38-next"),
        kv_string("general.name", "Qwen3.8-Flash-Next"),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string("general.source.revision", source_revision),
        kv_u32("qwen38-next.block_count", N_LAYER),
        kv_u32("qwen38-next.trunk_block_count", N_LAYER),
        kv_u32("qwen38-next.nextn_predict_layers", 0),
        kv_u64("qwen38-next.context_length", 262144),
        kv_u32("qwen38-next.embedding_length", N_EMBD),
        kv_u32("qwen38-next.vocab_size", N_VOCAB),
        kv_u32("qwen38-next.expert_feed_forward_length", N_FF_EXP),
        kv_u32("qwen38-next.expert_count", N_EXPERT),
        kv_u32("qwen38-next.expert_used_count", 10),
        kv_u32("qwen38-next.expert_shared_count", 1),
        kv_f32("qwen38-next.expert_weights_scale", 1.0),
        kv_bool("qwen38-next.expert_weights_norm", True),
        kv_f32("qwen38-next.swiglu_limit", 0.0),
        kv_f32("qwen38-next.attention.layer_norm_rms_epsilon", 1.0e-6),
        kv_u32("qwen38-next.attention.head_count", 24),
        kv_u32("qwen38-next.attention.head_count_kv", 2),
        kv_u32("qwen38-next.attention.key_length", 256),
        kv_u32("qwen38-next.attention.value_length", 256),
        kv_u32("qwen38-next.attention.rope_dimension_count", 64),
        kv_f32("qwen38-next.rope.freq_base", 10000000.0),
        kv_u32("qwen38-next.attention.indexer.head_count", 4),
        kv_u32("qwen38-next.attention.indexer.key_length", 128),
        kv_u32("qwen38-next.attention.indexer.top_k", 2048),
        kv_u32("qwen38-next.attention.indexer.pool_size", 4),
        kv_u32("qwen38-next.linear_attention.head_count", 48),
        kv_u32("qwen38-next.linear_attention.qk_head_count", 16),
        kv_u32("qwen38-next.linear_attention.head_dimension", 128),
        kv_u32("qwen38-next.linear_attention.conv_kernel", 4),
        kv_u32("qwen38-next.hyper_connection.count", 4),
        kv_u32("qwen38-next.hyper_connection.low_rank", 320),
        kv_f32("qwen38-next.hyper_connection.epsilon", 1.0e-6),
        kv_u32_array("qwen38-next.layer_types", layer_types),
        kv_u32("qwen38-next.ple.ngram_size", 3),
        kv_u32("qwen38-next.ple.heads_per_ngram", 8),
        kv_u32("qwen38-next.ple.row_length", PLE_ROW_LEN),
        kv_u32("qwen38-next.ple.conv_kernel", 4),
        kv_u32("qwen38-next.ple.layer", PLE_LAYER),
        kv_u32("qwen38-next.ple.eos_token_id", 248044),
        kv_u64_array("qwen38-next.ple.layer_multipliers", multipliers),
        kv_u64_array("qwen38-next.ple.head_offsets", offsets),
        kv_u64_array("qwen38-next.ple.head_vocab_sizes", vocab_sizes),
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
                value = (math.ldexp(mantissa, -9) if exponent == 0
                         else math.ldexp(1.0 + mantissa / 8.0, exponent - 7))
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
            expected_scale_shape = [(dim + 127) // 128 for dim in shape]
            if scale_info["shape"] != expected_scale_shape:
                fail(f"{scale_name} has shape {scale_info['shape']}, "
                     f"expected {expected_scale_shape}")
            raw_scales = db.read(scale_name)
            if scale_info["dtype"] == "F32":
                scales = np.frombuffer(raw_scales, dtype="<f4").reshape(scale_info["shape"])
            elif scale_info["dtype"] == "BF16":
                scale_bits = np.frombuffer(raw_scales, dtype="<u2").astype(np.uint32) << 16
                scales = scale_bits.view(np.float32).reshape(scale_info["shape"])
            else:
                fail(f"{scale_name} has unsupported dtype {scale_info['dtype']}")
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
        if imatrix is not None and qtype in (QTYPE_IQ2_XXS, QTYPE_Q2_K, QTYPE_Q4_K):
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
    """Legacy DS4 imatrix .dat reader (same format as glm53_quantize.py)."""

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
                values = np.frombuffer(
                    read_exact(fp, value_count * 4, "imatrix values"), dtype="<f4").copy()
                if not np.all(np.isfinite(values)):
                    fail(f"nonfinite imatrix values for {name}")
                self.entries[name] = values

    def expert(self, tensor_name, expert, width):
        values = self.entries.get(tensor_name)
        if values is None:
            return None
        expected = N_EXPERT * width
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


def transform_regular(values, item, np):
    if item.transform is None:
        return values
    if item.transform == "norm_plus_one":
        return values + 1.0
    if item.transform == "squeeze":
        return values.reshape(-1)
    if item.transform in ("conv_squeeze", "conv_slice"):
        # source [channels, 1, kernel] -> [channels or slice, kernel]
        if values.ndim != 3 or values.shape[1] != 1:
            fail(f"{item.name}: unexpected conv values shape {values.shape}")
        values = values[:, 0, :]
        if item.transform == "conv_slice":
            values = values[item.row_start : item.row_start + item.row_count]
        return np.ascontiguousarray(values)
    if item.transform in ("qsa_deinterleave_q", "qsa_deinterleave_gate"):
        # [heads * 2 * head_dim, hidden]: rows [h*512 : h*512+256] are the
        # query, rows [h*512+256 : (h+1)*512] the output gate for head h.
        per_head = values.reshape(24, 512, values.shape[1])
        half = per_head[:, :256, :] if item.transform == "qsa_deinterleave_q" \
            else per_head[:, 256:, :]
        return np.ascontiguousarray(half.reshape(QSA_Q_DIM, values.shape[1]))
    fail(f"unknown transform {item.transform} for {item.name}")


def write_regular(fp, item, db, quantizer):
    # conv_slice/deinterleave transforms need the whole source tensor.
    whole = item.transform is not None
    values = quantizer.to_f32(
        db, item.source,
        0 if whole else item.row_start,
        None if whole else item.row_count,
    )
    values = transform_regular(values, item, quantizer.np)
    data = quantizer.encode(values, item.qtype)
    if len(data) != item.nbytes:
        fail(f"{item.name}: generated {len(data)} bytes, expected {item.nbytes}")
    fp.write(data)


def write_experts(fp, item, db, quantizer, imatrix, threads):
    width = item.shape[0]

    def convert(expert):
        source = item.source.format(expert=expert)
        values = quantizer.to_f32(db, source)
        weights = imatrix.expert(item.name, expert, width)
        return quantizer.encode(values, item.qtype, weights), weights is None

    per_expert = item.nbytes // N_EXPERT
    fallback_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        window = max(threads * 2, 1)
        for start in range(0, N_EXPERT, window):
            futures = [executor.submit(convert, expert)
                       for expert in range(start, min(start + window, N_EXPERT))]
            for future in futures:
                data, fallback = future.result()
                fallback_count += int(fallback)
                if len(data) != per_expert:
                    fail(f"{item.name}: expert generated {len(data)} bytes, expected {per_expert}")
                fp.write(data)
    if imatrix.entries and fallback_count:
        print(f"qwen38-quantize: {item.name} used weight-based fallback for "
              f"{fallback_count}/{N_EXPERT} unobserved experts", file=sys.stderr)


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
        digest.update(struct.pack("<qq", item.row_start, item.row_count or -1))
        digest.update(bytes((item.is_expert,)))
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


def load_resume_state(path, signature, total):
    with open(path, "r", encoding="utf-8") as fp:
        state = json.load(fp)
    if state.get("version") != 1 or state.get("signature") != signature:
        fail(f"resume state does not match this conversion: {path}")
    completed = state.get("completed")
    if not isinstance(completed, int) or completed < 0 or completed > total:
        fail(f"invalid completed count in {path}")
    return completed


def open_quantizer(args):
    library = args.quants_library
    if not library:
        suffix = "dylib" if sys.platform == "darwin" else "so"
        library = os.path.join(os.path.dirname(__file__), f"libds4quants.{suffix}")
    if not os.path.isfile(library):
        fail(f"quantizer library not found: {library}; run make -C gguf-tools")
    return Quantizer(library)


def write_gguf(args, plan, kv_records, tokenizer_records, db, quantizer):
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
        completed = load_resume_state(resume_path, signature, len(plan))
        expected_size = data_offset
        if completed:
            previous = plan[completed - 1]
            expected_size += previous.offset + align(previous.nbytes)
        with open(partial, "r+b") as fp:
            fp.truncate(expected_size)
        print(f"qwen38-quantize: resuming after {completed} tensors", file=sys.stderr)
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
                f"{item.nbytes / (1 << 30):.3f} GiB {elapsed:.1f}s "
                f"total={total_elapsed / 60:.1f}m",
                file=sys.stderr,
                flush=True,
            )
    os.replace(partial, args.out)
    os.unlink(resume_path)


def ple_shard_names():
    prefix = f"{source_prefix(PLE_LAYER)}.ple.ple_embedding"
    return [f"{prefix}.ngram_embedding.shard_{shard}.weight" for shard in range(PLE_SHARDS)]


def ple_read_scale(db, quantizer):
    name = f"{source_prefix(PLE_LAYER)}.ple.ple_embedding.ngram_embedding.weight_scale"
    info = db.info(name)
    if product(info["shape"]) != 1:
        fail(f"{name}: expected a scalar, got shape {info['shape']}")
    np = quantizer.np
    raw = db.read(name)
    if info["dtype"] == "F32":
        return float(np.frombuffer(raw, dtype="<f4")[0])
    if info["dtype"] == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return float(bits.view(np.float32)[0])
    fail(f"{name}: unsupported scale dtype {info['dtype']}")


def write_ple_sidecar(args, db, quantizer, ple_values):
    np = quantizer.np
    multipliers, offsets, vocab_sizes, total_rows = ple_values

    shard_names = ple_shard_names()
    shard_rows = []
    dtype = None
    for name in shard_names:
        info = db.info(name)
        if len(info["shape"]) != 2 or info["shape"][1] != PLE_ROW_LEN:
            fail(f"{name}: unexpected shape {info['shape']}")
        if dtype is None:
            dtype = info["dtype"]
        elif info["dtype"] != dtype:
            fail(f"{name}: mixed PLE shard dtypes")
        shard_rows.append(info["shape"][0])
    # The physical table is padded to a multiple of 128 rows
    # (make_ngram_vocab_size_divisible_by); the hash heads only address
    # total_rows of them.
    table_rows = sum(shard_rows)
    padded_rows = (total_rows + 127) // 128 * 128
    if table_rows != padded_rows:
        fail(f"PLE shards cover {table_rows} rows, expected {padded_rows} "
             f"({total_rows} addressable rows padded to 128)")
    scale = ple_read_scale(db, quantizer) if dtype == "F8_E4M3" else None

    qtype = QTYPE_Q8_0
    item = TensorPlan("per_layer_token_embd.weight", (PLE_ROW_LEN, table_rows),
                      qtype, "ple_table")
    item.nbytes = qtype_nbytes(qtype, item.shape)

    kv_records = [
        kv_string("general.architecture", "qwen38-next-ple"),
        kv_string("general.name", "Qwen3.8-Flash-Next PLE table"),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_u32("qwen38-next.ple.row_length", PLE_ROW_LEN),
        kv_u64("qwen38-next.ple.total_rows", total_rows),
        kv_u64("qwen38-next.ple.table_rows", table_rows),
        kv_u64_array("qwen38-next.ple.layer_multipliers", multipliers),
        kv_u64_array("qwen38-next.ple.head_offsets", offsets),
        kv_u64_array("qwen38-next.ple.head_vocab_sizes", vocab_sizes),
    ]
    header = b"GGUF" + struct.pack("<IQQ", GGUF_VERSION, 1, len(kv_records))
    header += b"".join(kv_records)
    header += tensor_header(item)
    data_offset = align(len(header))

    partial = args.ple_out + ".partial"
    resume_path = partial + ".resume.json"
    signature = hashlib.sha256(header + (str(scale) + dtype).encode()).hexdigest()
    if os.path.exists(args.ple_out) and not args.overwrite:
        fail(f"output exists: {args.ple_out}; use --overwrite")
    if args.overwrite:
        for path in (partial, resume_path):
            if os.path.exists(path):
                os.unlink(path)

    resume = os.path.exists(partial) or os.path.exists(resume_path)
    if resume and not args.resume:
        fail(f"partial PLE conversion exists: use --resume or --overwrite for {partial}")

    completed = 0
    row_bytes = qtype_nbytes(qtype, (PLE_ROW_LEN,))
    if resume:
        completed = load_resume_state(resume_path, signature, PLE_SHARDS)
        rows_done = sum(shard_rows[:completed])
        with open(partial, "r+b") as fp:
            fp.truncate(data_offset + rows_done * row_bytes)
        print(f"qwen38-quantize: resuming PLE after {completed} shards", file=sys.stderr)
    else:
        with open(partial, "wb") as fp:
            fp.write(header)
            fp.write(bytes(data_offset - len(header)))
            fp.flush()
            os.fsync(fp.fileno())
        save_resume_state(resume_path, signature, 0)

    started = time.monotonic()
    with open(partial, "r+b") as fp:
        fp.seek(data_offset + sum(shard_rows[:completed]) * row_bytes)
        for index in range(completed, PLE_SHARDS):
            name = shard_names[index]
            info = db.info(name)
            raw = db.read(name)
            if dtype == "F8_E4M3":
                codes = np.frombuffer(raw, dtype=np.uint8).reshape(info["shape"])
                if np.any((codes & 0x7F) == 0x7F):
                    fail(f"nonfinite FP8 code in {name}")
                values = quantizer.fp8_lut[codes] * scale
            elif dtype == "BF16":
                bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
                values = bits.view(np.float32).reshape(info["shape"])
            else:
                fail(f"{name}: unsupported PLE dtype {dtype}")
            values = np.ascontiguousarray(values, dtype=np.float32)
            data = quantizer.encode(values, qtype)
            if len(data) != info["shape"][0] * row_bytes:
                fail(f"{name}: generated {len(data)} bytes")
            fp.write(data)
            fp.flush()
            os.fsync(fp.fileno())
            save_resume_state(resume_path, signature, index + 1)
            total_elapsed = time.monotonic() - started
            print(f"[ple {index + 1:3d}/{PLE_SHARDS}] {info['shape'][0]} rows "
                  f"total={total_elapsed / 60:.1f}m", file=sys.stderr, flush=True)
        end = fp.tell()
        expected_end = data_offset + item.nbytes
        if end != expected_end:
            fail(f"PLE sidecar size mismatch: {end} != {expected_end}")
    os.replace(partial, args.ple_out)
    os.unlink(resume_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True, help="official Qwen3.8-Flash-Next FP8 snapshot")
    parser.add_argument("--tokenizer-template", required=True,
                        help="GGUF supplying tokenizer metadata (metadata-only shard works)")
    parser.add_argument("--out", help="output main GGUF")
    parser.add_argument("--ple-out", help="output PLE sidecar GGUF (Q8_0 table)")
    parser.add_argument("--artifact", choices=("q2", "q4"), default="q2")
    parser.add_argument("--imatrix", help="legacy DS4 imatrix .dat")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--source-revision", default="main")
    parser.add_argument("--quants-library", help="path to libds4quants")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 64:
        parser.error("--threads must be between 1 and 64")
    if not args.dry_run and not (args.out or args.ple_out):
        parser.error("--out and/or --ple-out is required unless --dry-run is used")
    return args


def main():
    args = parse_args()
    db = SourceDB(args.hf)
    try:
        quantizer = open_quantizer(args)
        ple_values = ple_metadata_values(db, quantizer.np)
        plan = build_plan(db, args.artifact)
        tokenizer_records, template_tokens = load_tokenizer_records(args.tokenizer_template)
        validate_tokenizer_template(args.hf, template_tokens, N_VOCAB)
        kv_records = model_metadata(args.hf, args.source_revision, ple_values)
        if args.dry_run:
            print_plan(plan, kv_records, tokenizer_records)
            return
        if args.out:
            write_gguf(args, plan, kv_records, tokenizer_records, db, quantizer)
            print(f"qwen38-quantize: wrote {args.out}", file=sys.stderr)
        if args.ple_out:
            write_ple_sidecar(args, db, quantizer, ple_values)
            print(f"qwen38-quantize: wrote {args.ple_out}", file=sys.stderr)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"qwen38-quantize: error: {error}", file=sys.stderr)
        sys.exit(1)
