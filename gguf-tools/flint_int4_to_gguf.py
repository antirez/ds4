#!/usr/bin/env python3
"""Convert rec's Recursive Flint INT4 checkpoint into the ds4 fork GGUF.

The source is the checkpoint `rec pull` downloads (RecursiveSI/flint-int4:
MLX affine int4 with groups of 64, 8-bit embeddings, BF16 scales and biases,
BF16 norms, routers and attention sinks).  Every weight byte is copied
verbatim, so the GGUF is rec's tensor layout with ds4 names:

  weight tensors   FLINT_U4 (type 143): row-major [out, in] nibbles, low
                   nibble first, two elements per byte (MLX packing); the
                   embedding table is FLINT_U8 (type 144), one byte each
  scales, biases   BF16 [out, in / 64]; w = scale * q + bias per group
  routed experts   one FLINT_EXPERT_PACK (type 145) tensor per layer of
                   shape [256, 1769472]: each expert's gate, up and down
                   matrices with their scales and biases, contiguous, in the
                   order given by flint.expert_pack.layout
  everything else  BF16 as stored

`general.alignment` is 256 so each tensor (and therefore each expert pack)
starts 256-byte aligned.  The GGUF metadata carries the checkpoint digests
from rec's manifest, and the tokenizer (byte-level BPE, 150144 tokens) and
chat template from the checkpoint.

Needs numpy; --check additionally needs mlx (the dequantisation reference).
docs/RECURSIVE_FLINT.md describes the layout and the loader that reads it.

    flint_int4_to_gguf.py --out gguf/flint/flint-int4.gguf --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 256

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)

GGML_BF16 = 30
FLINT_U4 = 143
FLINT_U8 = 144
FLINT_EXPERT_PACK = 145

ARCH = "flint"
N_LAYER = 49
N_EMBD = 2048
N_VOCAB = 150144
N_HEAD = 32
N_HEAD_KV = 4
HEAD_DIM = 128
N_FF_DENSE = 6144
N_FF_EXP = 512
N_FF_SHARED = 1024
N_EXPERT = 256
N_EXPERT_USED = 6
N_DENSE = 2
GROUP = 64
SWA = 513
CTX = 262144
ROPE_THETA = 10000.0
RMS_EPS = 1e-6
GLOBAL_LAYERS = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]
EOG_IDS = [150001, 150003, 150005, 150018]

# Per-expert pack: (name, rows, bytes per row); offsets follow from the order.
PACK_LAYOUT = [
    ("gate.weight", N_FF_EXP, N_EMBD // 2),
    ("gate.scales", N_FF_EXP, N_EMBD // GROUP * 2),
    ("gate.biases", N_FF_EXP, N_EMBD // GROUP * 2),
    ("up.weight", N_FF_EXP, N_EMBD // 2),
    ("up.scales", N_FF_EXP, N_EMBD // GROUP * 2),
    ("up.biases", N_FF_EXP, N_EMBD // GROUP * 2),
    ("down.weight", N_EMBD, N_FF_EXP // 2),
    ("down.scales", N_EMBD, N_FF_EXP // GROUP * 2),
    ("down.biases", N_EMBD, N_FF_EXP // GROUP * 2),
]
PACK_OFFSETS = []
_off = 0
for _name, _rows, _row_bytes in PACK_LAYOUT:
    PACK_OFFSETS.append(_off)
    _off += _rows * _row_bytes
PACK_BYTES = _off
assert PACK_BYTES == 1769472, PACK_BYTES

DEFAULT_MODELS = Path.home() / ".local/share/rec/models"


# --------------------------------------------------------------- safetensors

class Shard:
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "rb")
        (n,) = struct.unpack("<Q", self.f.read(8))
        self.header = json.loads(self.f.read(n))
        self.data_start = 8 + n
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def tensor_bytes(self, name: str) -> memoryview:
        info = self.header[name]
        a, b = info["data_offsets"]
        return memoryview(self.mm)[self.data_start + a : self.data_start + b]


class Checkpoint:
    def __init__(self, src: Path):
        self.src = src
        self.config = json.loads((src / "config.json").read_text())
        self.manifest = json.loads((src / "flint-int4.json").read_text())
        self.shards = [Shard(p) for p in sorted(src.glob("model-*.safetensors"))]
        self.where: dict[str, Shard] = {}
        self.info: dict[str, dict] = {}
        for s in self.shards:
            for k, v in s.header.items():
                if k == "__metadata__":
                    continue
                self.where[k] = s
                self.info[k] = v

    def get(self, name: str) -> memoryview:
        return self.where[name].tensor_bytes(name)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.info[name]["shape"])

    def dtype(self, name: str) -> str:
        return self.info[name]["dtype"]


# --------------------------------------------------------------- gguf writer

class KV:
    def __init__(self):
        self.items: list[tuple[str, int, object]] = []

    def add(self, key, typ, value):
        self.items.append((key, typ, value))

    def u32(self, k, v): self.add(k, T_U32, int(v))
    def f32(self, k, v): self.add(k, T_F32, float(v))
    def bool(self, k, v): self.add(k, T_BOOL, bool(v))
    def str(self, k, v): self.add(k, T_STR, str(v))
    def arr(self, k, typ, vals): self.add(k, T_ARR, (typ, list(vals)))


def _pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _pack_value(typ: int, v) -> bytes:
    if typ == T_U32:
        return struct.pack("<I", v)
    if typ == T_I32:
        return struct.pack("<i", v)
    if typ == T_F32:
        return struct.pack("<f", v)
    if typ == T_BOOL:
        return struct.pack("<B", 1 if v else 0)
    if typ == T_STR:
        return _pack_str(v)
    if typ == T_U64:
        return struct.pack("<Q", v)
    if typ == T_ARR:
        et, vals = v
        out = [struct.pack("<IQ", et, len(vals))]
        if et == T_STR:
            out.extend(_pack_str(x) for x in vals)
        elif et == T_U32:
            out.append(np.asarray(vals, dtype="<u4").tobytes())
        elif et == T_I32:
            out.append(np.asarray(vals, dtype="<i4").tobytes())
        else:
            out.extend(_pack_value(et, x) for x in vals)
        return b"".join(out)
    raise ValueError(typ)


class TensorPlan:
    """A GGUF tensor: ds4 name, type, GGUF dims (ne[0] fastest), byte size and
    a producer that yields its bytes in order."""

    def __init__(self, name, typ, ne, nbytes, producer):
        self.name = name
        self.typ = typ
        self.ne = list(ne)
        self.nbytes = nbytes
        self.producer = producer
        self.offset = 0


def type_nbytes(typ: int, ne: list[int]) -> int:
    n = 1
    for d in ne:
        n *= d
    if typ == GGML_BF16:
        return n * 2
    if typ == FLINT_U4:
        assert ne[0] % 2 == 0
        return n // 2
    if typ in (FLINT_U8, FLINT_EXPERT_PACK):
        return n
    raise ValueError(typ)


def write_gguf(out: Path, kv: KV, tensors: list[TensorPlan], progress=True):
    tmp = out.with_suffix(out.suffix + ".incomplete")
    out.parent.mkdir(parents=True, exist_ok=True)
    pos = 0
    for t in tensors:
        t.offset = pos
        pos += (t.nbytes + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    total = pos
    with open(tmp, "wb") as f:
        hdr = [GGUF_MAGIC, struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(kv.items))]
        for key, typ, val in kv.items:
            hdr.append(_pack_str(key))
            hdr.append(struct.pack("<I", typ))
            hdr.append(_pack_value(typ, val))
        for t in tensors:
            hdr.append(_pack_str(t.name))
            hdr.append(struct.pack("<I", len(t.ne)))
            hdr.append(struct.pack("<%dQ" % len(t.ne), *t.ne))
            hdr.append(struct.pack("<IQ", t.typ, t.offset))
        header = b"".join(hdr)
        f.write(header)
        pad = (-len(header)) % ALIGNMENT
        f.write(b"\0" * pad)
        data_start = f.tell()
        done = 0
        t0 = time.time()
        for i, t in enumerate(tensors):
            assert f.tell() == data_start + t.offset, (t.name, f.tell(), data_start + t.offset)
            n = 0
            for chunk in t.producer():
                f.write(chunk)
                n += len(chunk)
            assert n == t.nbytes, (t.name, n, t.nbytes)
            f.write(b"\0" * ((-n) % ALIGNMENT))
            done += n
            if progress and (i % 50 == 0 or i == len(tensors) - 1):
                dt = time.time() - t0
                rate = done / dt / 1e9 if dt > 0 else 0
                print("  %5d/%d tensors  %6.2f GB  %.2f GB/s  %s" % (i + 1, len(tensors), done / 1e9, rate, t.name),
                      file=sys.stderr)
    os.replace(tmp, out)
    return total


# ----------------------------------------------------------------- the plan

def bf16_tensor(ck: Checkpoint, src: str, dst: str, shape=None) -> TensorPlan:
    assert ck.dtype(src) == "BF16", (src, ck.dtype(src))
    sh = ck.shape(src)
    if shape is not None:
        assert sh == tuple(shape), (src, sh, shape)
    ne = list(reversed(sh))
    mv = ck.get(src)
    return TensorPlan(dst, GGML_BF16, ne, len(mv), lambda mv=mv: (mv,))


def quant_tensor(ck: Checkpoint, src_prefix: str, dst_prefix: str, out_dim: int, in_dim: int,
                 bits: int) -> list[TensorPlan]:
    """weight (U8 nibbles or bytes) + scales + biases of an MLX affine matrix."""
    w, s, b = (src_prefix + ".weight", src_prefix + ".scales", src_prefix + ".biases")
    assert ck.dtype(w) == "U8", w
    per_byte = 8 // bits
    assert ck.shape(w) == (out_dim, in_dim // per_byte), (w, ck.shape(w))
    assert ck.shape(s) == (out_dim, in_dim // GROUP), (s, ck.shape(s))
    assert ck.shape(b) == (out_dim, in_dim // GROUP), (b, ck.shape(b))
    typ = FLINT_U4 if bits == 4 else FLINT_U8
    mv = ck.get(w)
    plans = [TensorPlan(dst_prefix + ".weight", typ, [in_dim, out_dim], len(mv), lambda mv=mv: (mv,))]
    plans.append(bf16_tensor(ck, s, dst_prefix + ".scales"))
    plans.append(bf16_tensor(ck, b, dst_prefix + ".biases"))
    return plans


def expert_pack(ck: Checkpoint, il: int) -> TensorPlan:
    base = "model.layers.%d.mlp.switch_mlp." % il
    parts = []
    for name, rows, row_bytes in PACK_LAYOUT:
        proj, kind = name.split(".")
        src = base + {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}[proj] + "." + kind
        sh = ck.shape(src)
        assert sh[0] == N_EXPERT, (src, sh)
        mv = ck.get(src)
        per_expert = len(mv) // N_EXPERT
        assert per_expert == rows * row_bytes, (src, per_expert, rows * row_bytes)
        parts.append((mv, per_expert))

    def produce(parts=parts):
        for e in range(N_EXPERT):
            for mv, per in parts:
                yield mv[e * per : (e + 1) * per]

    return TensorPlan("blk.%d.ffn_exps.pack" % il, FLINT_EXPERT_PACK, [PACK_BYTES, N_EXPERT],
                      PACK_BYTES * N_EXPERT, produce)


def build_tensors(ck: Checkpoint) -> list[TensorPlan]:
    t: list[TensorPlan] = []
    t += quant_tensor(ck, "model.embed_tokens", "token_embd", N_VOCAB, N_EMBD, bits=8)
    for il in range(N_LAYER):
        p = "model.layers.%d." % il
        blk = "blk.%d." % il
        t.append(bf16_tensor(ck, p + "input_layernorm.weight", blk + "attn_norm.weight", (N_EMBD,)))
        t += quant_tensor(ck, p + "self_attn.q_proj", blk + "attn_q", N_HEAD * HEAD_DIM, N_EMBD, 4)
        t += quant_tensor(ck, p + "self_attn.k_proj", blk + "attn_k", N_HEAD_KV * HEAD_DIM, N_EMBD, 4)
        t += quant_tensor(ck, p + "self_attn.v_proj", blk + "attn_v", N_HEAD_KV * HEAD_DIM, N_EMBD, 4)
        t += quant_tensor(ck, p + "self_attn.o_proj", blk + "attn_output", N_EMBD, N_HEAD * HEAD_DIM, 4)
        t.append(bf16_tensor(ck, p + "self_attn.q_norm.weight", blk + "attn_q_norm.weight", (HEAD_DIM,)))
        t.append(bf16_tensor(ck, p + "self_attn.k_norm.weight", blk + "attn_k_norm.weight", (HEAD_DIM,)))
        t.append(bf16_tensor(ck, p + "self_attn.softmax_offset", blk + "attn_sinks.weight", (N_HEAD,)))
        t.append(bf16_tensor(ck, p + "post_attention_layernorm.weight", blk + "ffn_norm.weight", (N_EMBD,)))
        if il < N_DENSE:
            t += quant_tensor(ck, p + "mlp.gate_proj", blk + "ffn_gate", N_FF_DENSE, N_EMBD, 4)
            t += quant_tensor(ck, p + "mlp.up_proj", blk + "ffn_up", N_FF_DENSE, N_EMBD, 4)
            t += quant_tensor(ck, p + "mlp.down_proj", blk + "ffn_down", N_EMBD, N_FF_DENSE, 4)
        else:
            t.append(bf16_tensor(ck, p + "mlp.gate.weight", blk + "ffn_gate_inp.weight", (N_EXPERT, N_EMBD)))
            t += quant_tensor(ck, p + "mlp.shared_expert.gate_proj", blk + "ffn_gate_shexp", N_FF_SHARED, N_EMBD, 4)
            t += quant_tensor(ck, p + "mlp.shared_expert.up_proj", blk + "ffn_up_shexp", N_FF_SHARED, N_EMBD, 4)
            t += quant_tensor(ck, p + "mlp.shared_expert.down_proj", blk + "ffn_down_shexp", N_EMBD, N_FF_SHARED, 4)
            t.append(expert_pack(ck, il))
    t.append(bf16_tensor(ck, "model.norm.weight", "output_norm.weight", (N_EMBD,)))
    t += quant_tensor(ck, "lm_head", "output", N_VOCAB, N_EMBD, 4)
    names = [x.name for x in t]
    assert len(names) == len(set(names))
    return t


def check_config(cfg: dict):
    exp = {
        "model_type": "flint", "num_hidden_layers": N_LAYER, "hidden_size": N_EMBD,
        "vocab_size": N_VOCAB, "num_attention_heads": N_HEAD, "num_key_value_heads": N_HEAD_KV,
        "head_dim": HEAD_DIM, "intermediate_size": N_FF_DENSE, "moe_intermediate_size": N_FF_EXP,
        "shared_expert_intermediate_size": N_FF_SHARED, "num_experts": N_EXPERT,
        "num_experts_per_tok": N_EXPERT_USED, "mlp_only_layers": list(range(N_DENSE)),
        "sliding_window": SWA, "max_position_embeddings": CTX, "rope_theta": ROPE_THETA,
        "rms_norm_eps": RMS_EPS, "no_rope_layer_indices": GLOBAL_LAYERS,
        "scoring_func": "sigmoid", "norm_topk_prob": True, "router_expert_bias": False,
        "attention_sink": True, "tie_word_embeddings": False,
        "quantization": {"bits": 4, "group_size": GROUP},
    }
    for k, v in exp.items():
        if cfg.get(k) != v:
            sys.exit("config.json %s = %r, expected %r" % (k, cfg.get(k), v))
    mx_exp = cfg.get("mlx_export", {})
    if mx_exp.get("bits") != 4 or mx_exp.get("embedding_bits") != 8 or mx_exp.get("routed_expert_bits") != 4:
        sys.exit("config.json mlx_export is not the INT4 recipe: %r" % mx_exp)
    full = [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
    if full != GLOBAL_LAYERS:
        sys.exit("layer_types full-attention layers %r != %r" % (full, GLOBAL_LAYERS))


def build_tokenizer(kv: KV, src: Path):
    tok = json.loads((src / "tokenizer.json").read_text())
    assert tok["model"]["type"] == "BPE" and not tok["model"].get("byte_fallback")
    assert tok.get("normalizer") is None and tok.get("post_processor") is None
    vocab = tok["model"]["vocab"]
    added = {a["id"]: a for a in tok["added_tokens"]}
    n = max(max(vocab.values()), max(added)) + 1
    assert n == N_VOCAB, n
    tokens = [None] * n
    types = [0] * n
    for s, i in vocab.items():
        tokens[i] = s
        types[i] = 1  # normal
    for i, a in added.items():
        assert tokens[i] is None, (i, tokens[i], a["content"])
        tokens[i] = a["content"]
        types[i] = 3 if a["special"] else 4  # control / user defined
    assert all(x is not None for x in tokens)
    merges = tok["model"]["merges"]
    merges = [m if isinstance(m, str) else " ".join(m) for m in merges]
    kv.str("tokenizer.ggml.model", "gpt2")
    kv.str("tokenizer.ggml.pre", "flint")
    kv.arr("tokenizer.ggml.tokens", T_STR, tokens)
    kv.arr("tokenizer.ggml.token_type", T_I32, types)
    kv.arr("tokenizer.ggml.merges", T_STR, merges)
    kv.u32("tokenizer.ggml.bos_token_id", 150000)
    kv.u32("tokenizer.ggml.eos_token_id", 150001)
    kv.u32("tokenizer.ggml.padding_token_id", 150001)
    kv.bool("tokenizer.ggml.add_bos_token", False)
    kv.bool("tokenizer.ggml.add_eos_token", False)
    kv.str("tokenizer.chat_template", (src / "chat_template.jinja").read_text())
    for name, i in [("<|begin_of_text|>", 150000), ("<|end_of_text|>", 150001), ("<|system|>", 150002),
                    ("<|user|>", 150003), ("<|assistant|>", 150004), ("<|observation|>", 150005),
                    ("<think>", 150006), ("</think>", 150007), ("<tool_call>", 150010),
                    ("</tool_call>", 150011), ("<arg_key>", 150012), ("</arg_key>", 150013),
                    ("<arg_value>", 150014), ("</arg_value>", 150015), ("<tool_response>", 150016),
                    ("</tool_response>", 150017), ("<|developer|>", 150018)]:
        assert tokens[i] == name, (i, tokens[i], name)
    return len(tokens), len(merges)


def build_metadata(ck: Checkpoint) -> KV:
    kv = KV()
    kv.str("general.architecture", ARCH)
    kv.str("general.name", "Recursive Flint INT4")
    kv.str("general.type", "model")
    kv.u32("general.alignment", ALIGNMENT)
    kv.str("general.quantization_version", "flint-int4")  # informational
    kv.u32("%s.block_count" % ARCH, N_LAYER)
    kv.u32("%s.context_length" % ARCH, CTX)
    kv.u32("%s.embedding_length" % ARCH, N_EMBD)
    kv.u32("%s.vocab_size" % ARCH, N_VOCAB)
    kv.u32("%s.feed_forward_length" % ARCH, N_FF_DENSE)
    kv.u32("%s.expert_feed_forward_length" % ARCH, N_FF_EXP)
    kv.u32("%s.expert_shared_feed_forward_length" % ARCH, N_FF_SHARED)
    kv.u32("%s.expert_count" % ARCH, N_EXPERT)
    kv.u32("%s.expert_used_count" % ARCH, N_EXPERT_USED)
    kv.u32("%s.expert_shared_count" % ARCH, 1)
    kv.u32("%s.leading_dense_block_count" % ARCH, N_DENSE)
    kv.u32("%s.attention.head_count" % ARCH, N_HEAD)
    kv.u32("%s.attention.head_count_kv" % ARCH, N_HEAD_KV)
    kv.u32("%s.attention.key_length" % ARCH, HEAD_DIM)
    kv.u32("%s.attention.value_length" % ARCH, HEAD_DIM)
    kv.u32("%s.attention.sliding_window" % ARCH, SWA)
    kv.f32("%s.attention.layer_norm_rms_epsilon" % ARCH, RMS_EPS)
    kv.bool("%s.attention.sink" % ARCH, True)
    kv.u32("%s.rope.dimension_count" % ARCH, HEAD_DIM)
    kv.f32("%s.rope.freq_base" % ARCH, ROPE_THETA)
    kv.arr("%s.full_attention_layers" % ARCH, T_U32, GLOBAL_LAYERS)
    kv.str("%s.router.score_function" % ARCH, "sigmoid")
    kv.bool("%s.router.norm_topk_prob" % ARCH, True)
    kv.arr("%s.eog_token_ids" % ARCH, T_U32, EOG_IDS)
    kv.str("%s.weight_format" % ARCH, "mlx-affine-int4")
    kv.u32("%s.quant.bits" % ARCH, 4)
    kv.u32("%s.quant.group_size" % ARCH, GROUP)
    kv.u32("%s.quant.embedding_bits" % ARCH, 8)
    kv.str("%s.quant.scale_rule" % ARCH, ck.config["mlx_export"].get("scale_rule", ""))
    kv.str("%s.expert_pack.layout" % ARCH, ",".join(n for n, _, _ in PACK_LAYOUT))
    kv.arr("%s.expert_pack.offsets" % ARCH, T_U32, PACK_OFFSETS)
    kv.u32("%s.expert_pack.bytes" % ARCH, PACK_BYTES)
    kv.str("%s.checkpoint.model_id" % ARCH, ck.manifest["model_id"])
    for fname, digest in sorted(ck.manifest["checkpoint_files"].items()):
        kv.str("%s.checkpoint.sha256.%s" % (ARCH, fname), digest)
    return kv


# --------------------------------------------------------------- the check

class GGUFReader:
    def __init__(self, path: Path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0
        assert self._read(4) == GGUF_MAGIC
        version, n_tensors, n_kv = struct.unpack("<IQQ", self._read(20))
        assert version == GGUF_VERSION
        self.kv = {}
        for _ in range(n_kv):
            key = self._str()
            (typ,) = struct.unpack("<I", self._read(4))
            self.kv[key] = self._value(typ)
        self.tensors = {}
        order = []
        for _ in range(n_tensors):
            name = self._str()
            (nd,) = struct.unpack("<I", self._read(4))
            ne = list(struct.unpack("<%dQ" % nd, self._read(8 * nd)))
            typ, off = struct.unpack("<IQ", self._read(12))
            self.tensors[name] = (typ, ne, off)
            order.append(name)
        self.order = order
        align = self.kv.get("general.alignment", 32)
        self.data_start = (self.pos + align - 1) // align * align

    def _read(self, n):
        b = self.mm[self.pos : self.pos + n]
        self.pos += n
        return b

    def _str(self):
        (n,) = struct.unpack("<Q", self._read(8))
        return self._read(n).decode("utf-8")

    def _value(self, typ):
        fmt = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I", T_I32: "<i",
               T_F32: "<f", T_BOOL: "<B", T_U64: "<Q", T_I64: "<q", T_F64: "<d"}
        if typ == T_STR:
            return self._str()
        if typ == T_ARR:
            et, n = struct.unpack("<IQ", self._read(12))
            if et == T_STR:
                return [self._str() for _ in range(n)]
            size = struct.calcsize(fmt[et])
            raw = self._read(size * n)
            return list(struct.unpack("<%d%s" % (n, fmt[et][1]), raw))
        size = struct.calcsize(fmt[typ])
        (v,) = struct.unpack(fmt[typ], self._read(size))
        return bool(v) if typ == T_BOOL else v

    def data(self, name) -> memoryview:
        typ, ne, off = self.tensors[name]
        n = type_nbytes(typ, ne)
        return memoryview(self.mm)[self.data_start + off : self.data_start + off + n]


def bf16_to_f32(raw: memoryview) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return u16.view(np.float32)


def dequant_numpy(w: memoryview, s: memoryview, b: memoryview, out_dim: int, in_dim: int, bits: int) -> np.ndarray:
    scales = bf16_to_f32(s).reshape(out_dim, in_dim // GROUP)
    biases = bf16_to_f32(b).reshape(out_dim, in_dim // GROUP)
    raw = np.frombuffer(w, dtype=np.uint8)
    if bits == 4:
        raw = raw.reshape(out_dim, in_dim // 2)
        q = np.empty((out_dim, in_dim), dtype=np.float32)
        q[:, 0::2] = raw & 0x0F   # low nibble = even element
        q[:, 1::2] = raw >> 4     # high nibble = odd element
    else:
        q = raw.reshape(out_dim, in_dim).astype(np.float32)
    return q.reshape(out_dim, in_dim // GROUP, GROUP) * scales[:, :, None] + biases[:, :, None]


def dequant_mlx(w: memoryview, s: memoryview, b: memoryview, out_dim: int, in_dim: int, bits: int) -> np.ndarray:
    import mlx.core as mx
    wq = mx.array(np.frombuffer(w, dtype="<u4").reshape(out_dim, in_dim * bits // 32))
    sc = mx.array(np.frombuffer(s, dtype="<u2").reshape(out_dim, in_dim // GROUP)).view(mx.bfloat16)
    bi = mx.array(np.frombuffer(b, dtype="<u2").reshape(out_dim, in_dim // GROUP)).view(mx.bfloat16)
    d = mx.dequantize(wq, sc, bi, group_size=GROUP, bits=bits).astype(mx.float32)
    return np.array(d)


def run_check(out: Path, ck: Checkpoint, tensors: list[TensorPlan], full: bool, samples: int):
    print("check: re-reading %s" % out, file=sys.stderr)
    g = GGUFReader(out)
    assert g.order == [t.name for t in tensors], "tensor order differs"
    n_tok = len(g.kv["tokenizer.ggml.tokens"])
    assert n_tok == N_VOCAB, n_tok
    assert g.kv["general.alignment"] == ALIGNMENT
    assert g.kv["%s.expert_pack.bytes" % ARCH] == PACK_BYTES
    # Byte-exact comparison against the producers (every tensor with --check-all,
    # otherwise all small tensors plus a sample of the large ones).
    rng = np.random.default_rng(0)
    big = [t for t in tensors if t.nbytes >= 1 << 20]
    small = [t for t in tensors if t.nbytes < 1 << 20]
    chosen = big if full else [big[i] for i in sorted(rng.choice(len(big), size=min(samples, len(big)), replace=False))]
    nbytes = 0
    t0 = time.time()
    for t in small + chosen:
        typ, ne, off = g.tensors[t.name]
        assert typ == t.typ and ne == t.ne, (t.name, typ, ne)
        assert off % ALIGNMENT == 0, (t.name, off)
        got = g.data(t.name)
        pos = 0
        for chunk in t.producer():
            n = len(chunk)
            if got[pos : pos + n] != chunk:
                sys.exit("check: %s differs at byte %d" % (t.name, pos))
            pos += n
        assert pos == t.nbytes
        nbytes += t.nbytes
    print("check: %d tensors byte-identical (%.2f GB in %.1fs)" % (len(small) + len(chosen), nbytes / 1e9, time.time() - t0),
          file=sys.stderr)

    # Dequantisation: the documented nibble order against mlx on a few matrices.
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        print("check: mlx not installed, skipping the dequantisation reference", file=sys.stderr)
        return
    cases = [("output", N_VOCAB, N_EMBD, 4), ("token_embd", N_VOCAB, N_EMBD, 8),
             ("blk.0.attn_q", N_HEAD * HEAD_DIM, N_EMBD, 4), ("blk.0.ffn_down", N_EMBD, N_FF_DENSE, 4),
             ("blk.7.attn_output", N_EMBD, N_HEAD * HEAD_DIM, 4), ("blk.48.ffn_down_shexp", N_EMBD, N_FF_SHARED, 4)]
    for name, out_dim, in_dim, bits in cases:
        w, s, b = (g.data(name + ".weight"), g.data(name + ".scales"), g.data(name + ".biases"))
        rows = slice(0, min(out_dim, 4096))
        row_n = rows.stop
        wb = w[: row_n * in_dim * bits // 8]
        sb = s[: row_n * in_dim // GROUP * 2]
        bb = b[: row_n * in_dim // GROUP * 2]
        a = dequant_numpy(wb, sb, bb, row_n, in_dim, bits).reshape(row_n, in_dim)
        m = dequant_mlx(wb, sb, bb, row_n, in_dim, bits)
        diff = np.abs(a - m).max()
        scale = np.abs(m).max()
        print("check: dequant %-22s max|numpy-mlx| = %.3g (max|w| = %.3g)" % (name, diff, scale), file=sys.stderr)
        # bf16 products rounded once in mlx; the float32 path differs by rounding only.
        if diff > 1e-2 * max(scale, 1e-6):
            sys.exit("check: dequantisation mismatch on %s: the nibble/group layout is wrong" % name)
    # One expert out of one pack, against the source expert.
    il, e = 30, 123
    pack = g.data("blk.%d.ffn_exps.pack" % il)[e * PACK_BYTES : (e + 1) * PACK_BYTES]
    for (name, rows, row_bytes), off in zip(PACK_LAYOUT, PACK_OFFSETS):
        proj, kind = name.split(".")
        src = "model.layers.%d.mlp.switch_mlp.%s.%s" % (il, {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}[proj], kind)
        mv = ck.get(src)
        per = len(mv) // N_EXPERT
        if pack[off : off + rows * row_bytes] != mv[e * per : (e + 1) * per]:
            sys.exit("check: expert pack layer %d expert %d field %s differs" % (il, e, name))
    gw = pack[PACK_OFFSETS[0] : PACK_OFFSETS[0] + N_FF_EXP * N_EMBD // 2]
    gs = pack[PACK_OFFSETS[1] : PACK_OFFSETS[1] + N_FF_EXP * N_EMBD // GROUP * 2]
    gb = pack[PACK_OFFSETS[2] : PACK_OFFSETS[2] + N_FF_EXP * N_EMBD // GROUP * 2]
    a = dequant_numpy(gw, gs, gb, N_FF_EXP, N_EMBD, 4).reshape(N_FF_EXP, N_EMBD)
    m = dequant_mlx(gw, gs, gb, N_FF_EXP, N_EMBD, 4)
    print("check: dequant expert pack L%d E%d gate  max|numpy-mlx| = %.3g" % (il, e, np.abs(a - m).max()), file=sys.stderr)
    print("check: ok", file=sys.stderr)


# -------------------------------------------------------------------- main

def find_default_src() -> Path | None:
    if not DEFAULT_MODELS.is_dir():
        return None
    cands = [d for d in DEFAULT_MODELS.iterdir() if (d / "flint-int4.json").is_file()]
    if not cands:
        return None
    return max(cands, key=lambda d: d.stat().st_mtime)


def verify_shards(ck: Checkpoint):
    for s in ck.shards:
        want = ck.manifest["checkpoint_files"].get(s.path.name)
        h = hashlib.sha256()
        with open(s.path, "rb") as f:
            while True:
                b = f.read(1 << 24)
                if not b:
                    break
                h.update(b)
        got = h.hexdigest()
        if got != want:
            sys.exit("%s: sha256 %s, manifest says %s" % (s.path.name, got, want))
        print("verified %s" % s.path.name, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=None,
                    help="checkpoint directory (default: newest flint-int4 under ~/.local/share/rec/models)")
    ap.add_argument("--out", type=Path, required=True, help="output GGUF path")
    ap.add_argument("--check", action="store_true",
                    help="re-read the file: byte-exact on all small tensors and a sample of large ones, dequantisation vs mlx")
    ap.add_argument("--check-all", action="store_true", help="like --check but byte-exact on every tensor")
    ap.add_argument("--check-only", action="store_true", help="skip writing; check an existing --out")
    ap.add_argument("--samples", type=int, default=24, help="large tensors sampled by --check")
    ap.add_argument("--verify-shards", action="store_true", help="sha256 the safetensors against rec's manifest first")
    ap.add_argument("--dry-run", action="store_true", help="print the tensor plan and exit")
    args = ap.parse_args()

    src = args.src or find_default_src()
    if src is None or not (src / "flint-int4.json").is_file():
        sys.exit("no flint-int4 checkpoint found; run `rec pull` or pass --src")
    ck = Checkpoint(src)
    check_config(ck.config)
    if ck.manifest.get("weight_format") != "mlx-affine-int4":
        sys.exit("manifest weight_format %r is not mlx-affine-int4" % ck.manifest.get("weight_format"))
    if args.verify_shards:
        verify_shards(ck)

    tensors = build_tensors(ck)
    total = sum(t.nbytes for t in tensors)
    print("source %s: %d tensors in %d shards, %.3f GB of weights" % (src, len(ck.info), len(ck.shards), total / 1e9),
          file=sys.stderr)
    print("plan: %d GGUF tensors" % len(tensors), file=sys.stderr)
    if args.dry_run:
        for t in tensors:
            print("%-32s type %3d ne %-20s %12d bytes" % (t.name, t.typ, t.ne, t.nbytes))
        return
    if not args.check_only:
        kv = build_metadata(ck)
        n_tok, n_merges = build_tokenizer(kv, src)
        print("tokenizer: %d tokens, %d merges" % (n_tok, n_merges), file=sys.stderr)
        t0 = time.time()
        write_gguf(args.out, kv, tensors)
        print("wrote %s (%.3f GB) in %.1fs" % (args.out, args.out.stat().st_size / 1e9, time.time() - t0), file=sys.stderr)
    if args.check or args.check_all or args.check_only:
        run_check(args.out, ck, tensors, full=args.check_all, samples=args.samples)


if __name__ == "__main__":
    main()
