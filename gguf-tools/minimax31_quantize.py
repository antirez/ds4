#!/usr/bin/env python3
"""Build DwarfStar MiniMax-M3.1-Flash GGUF files from the official checkpoint.

The release stores attention, dense and shared-expert projections as MXFP8
(E4M3 values, one E8M0 exponent per 32 inputs) and routed experts as NVFP4
(packed E2M1 pairs, one E4M3 scale per 16 inputs, one F32 global scale per
tensor).  Both are dequantized to F32 here and requantized for ds4.

The GGUF writer, resume journal, quantizer bindings and imatrix reader are
shared with glm53_quantize.py; only the tensor plan, the source formats and
the metadata are MiniMax-specific.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import glm53_quantize as gq
from glm53_quantize import (
    QTYPE_F32,
    QTYPE_IQ2_XXS,
    QTYPE_Q2_K,
    QTYPE_Q4_K,
    QTYPE_Q8_0,
    TensorPlan,
    fail,
    kv_bool,
    kv_f32,
    kv_string,
    kv_u32,
    kv_u64,
    pack_string,
    qtype_nbytes,
)

ARCH = "minimax-m3"
PREFIX = "language_model.model"
N_LAYER = 60
N_DENSE = 3
N_EXPERT = 128
N_EMBD = 6144
N_VOCAB = 200064
N_FF_EXP = 3072

GGUF_INT32 = 5

# Gemma-style RMSNorm multiplies by (1 + w); ds4 norms multiply by w.
GEMMA_NORM = "gemma_norm"

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def read_bytes(hf_dir, name) -> bytes:
    path = os.path.join(hf_dir, name)
    try:
        with open(path, "rb") as fp:
            return fp.read()
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error


def read_json(hf_dir, name) -> dict:
    try:
        document = json.loads(read_bytes(hf_dir, name))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name}: invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name}: expected a JSON object")
    return document


def validate_index(weight_map):
    text = [name for name in weight_map if name.startswith((PREFIX + ".", "language_model.lm_head"))]
    expected = 3 + N_LAYER * (2 + 16) + N_DENSE * 6 + (N_LAYER - N_DENSE) * (2 + 6 + N_EXPERT * 12)
    if len(text) != expected:
        fail(f"text tensor count {len(text)} != expected {expected}; not a MiniMax-M3.1-Flash snapshot?")


def no_scale_check(_tensors):
    pass


class Quantizer(gq.Quantizer):
    """Adds the MXFP8 and NVFP4 source formats to the GLM quantizer."""

    def to_f32(self, db, name, row_start=0, row_count=None):
        np = self.np
        info = db.info(name)
        if name.endswith(".weight_packed"):
            array = self._nvfp4(db, name[: -len(".weight_packed")], info)
        elif info["dtype"] == "F8_E4M3":
            array = self._mxfp8(db, name, info)
        else:
            return super().to_f32(db, name, row_start, row_count)
        if row_count is not None:
            array = array[row_start: row_start + row_count]
        return np.ascontiguousarray(array, dtype=np.float32)

    def _e4m3(self, raw, shape, label):
        codes = self.np.frombuffer(raw, dtype=self.np.uint8).reshape(shape)
        if self.np.any((codes & 0x7F) == 0x7F):
            fail(f"nonfinite FP8 code in {label}")
        return self.fp8_lut[codes]

    def _mxfp8(self, db, name, info):
        np = self.np
        rows, cols = info["shape"]
        values = self._e4m3(db.read(name), (rows, cols), name)
        scale_name = name + "_scale_inv"
        scale_info = db.info(scale_name)
        if scale_info["dtype"] != "U8" or scale_info["shape"] != [rows, cols // 32]:
            fail(f"{scale_name}: expected U8 [{rows}, {cols // 32}], got {scale_info['dtype']} {scale_info['shape']}")
        exps = np.frombuffer(db.read(scale_name), dtype=np.uint8).reshape(rows, cols // 32)
        if np.any(exps == 0xFF):
            fail(f"NaN E8M0 scale in {scale_name}")
        scale = np.ldexp(np.float32(1.0), exps.astype(np.int32) - 127).astype(np.float32)
        return (values.reshape(rows, cols // 32, 32) * scale[:, :, None]).reshape(rows, cols)

    def _nvfp4(self, db, base, info):
        np = self.np
        rows, half = info["shape"]
        cols = half * 2
        packed = np.frombuffer(db.read(base + ".weight_packed"), dtype=np.uint8).reshape(rows, half)
        lut = np.asarray(FP4_VALUES, dtype=np.float32)
        values = np.empty((rows, cols), dtype=np.float32)
        values[:, 0::2] = lut[packed & 0x0F]  # compressed-tensors packs the even element low
        values[:, 1::2] = lut[packed >> 4]
        scale_info = db.info(base + ".weight_scale")
        if scale_info["dtype"] != "F8_E4M3" or scale_info["shape"] != [rows, cols // 16]:
            fail(f"{base}.weight_scale: unexpected {scale_info['dtype']} {scale_info['shape']}")
        scale = self._e4m3(db.read(base + ".weight_scale"), (rows, cols // 16), base)
        global_scale = np.frombuffer(db.read(base + ".weight_global_scale"), dtype="<f4")
        if global_scale.size != 1 or not np.isfinite(global_scale[0]) or global_scale[0] <= 0:
            fail(f"{base}.weight_global_scale is invalid")
        # compressed-tensors stores the global scale as a divisor.
        scale = scale / global_scale[0]
        return (values.reshape(rows, cols // 16, 16) * scale[:, :, None]).reshape(rows, cols)


_base_transform_regular = gq.transform_regular  # main() monkeypatches gq.transform_regular


def transform_regular(values, item):
    if item.transform == GEMMA_NORM:
        return values + 1.0
    return _base_transform_regular(values, item)


def add(plan, db, name, source, qtype, role, transform=None):
    gq.add_regular(plan, db, name, source, qtype, role)
    plan[-1].transform = transform


def add_experts(plan, db, layer, part, source_part, qtype):
    prefix = f"{PREFIX}.layers.{layer}.block_sparse_moe.experts"
    rows, half = db.info(f"{prefix}.0.{source_part}.weight_packed")["shape"]
    for expert in range(N_EXPERT):
        if db.info(f"{prefix}.{expert}.{source_part}.weight_packed")["shape"] != [rows, half]:
            fail(f"expert shape mismatch in layer {layer} {source_part}")
    item = TensorPlan(
        f"blk.{layer}.ffn_{part}_exps.weight",
        (half * 2, rows, N_EXPERT),
        qtype,
        f"routed_{part}",
        source=f"{prefix}.{{expert}}.{source_part}.weight_packed",
        expert_layer=layer,
        expert_part=part,
        expert_count=N_EXPERT,
    )
    item.nbytes = qtype_nbytes(qtype, item.shape)
    plan.append(item)


def build_plan(db, artifact, n_layer=N_LAYER):
    plan = []
    add(plan, db, "token_embd.weight", f"{PREFIX}.embed_tokens.weight", QTYPE_Q8_0, "embedding")
    routed = {
        "q2": {"gate": QTYPE_IQ2_XXS, "up": QTYPE_IQ2_XXS, "down": QTYPE_Q2_K},
        "q4": {"gate": QTYPE_Q4_K, "up": QTYPE_Q4_K, "down": QTYPE_Q4_K},
    }[artifact]
    for layer in range(n_layer):
        p = f"{PREFIX}.layers.{layer}"
        a = f"{p}.self_attn"
        add(plan, db, f"blk.{layer}.attn_norm.weight", f"{p}.input_layernorm.weight", QTYPE_F32, "norm", GEMMA_NORM)
        for target, source in (("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"),
                               ("attn_output", "o_proj"), ("indexer.q_proj", "index_q_proj"),
                               ("indexer.k_proj", "index_k_proj")):
            add(plan, db, f"blk.{layer}.{target}.weight", f"{a}.{source}.weight", QTYPE_Q8_0, "attention")
        for target, source in (("attn_q_norm", "q_norm"), ("attn_k_norm", "k_norm"),
                               ("indexer.q_norm", "index_q_norm"), ("indexer.k_norm", "index_k_norm")):
            add(plan, db, f"blk.{layer}.{target}.weight", f"{a}.{source}.weight", QTYPE_F32, "norm", GEMMA_NORM)
        add(plan, db, f"blk.{layer}.ffn_norm.weight", f"{p}.post_attention_layernorm.weight", QTYPE_F32, "norm", GEMMA_NORM)
        if layer < N_DENSE:
            for part in ("gate", "up", "down"):
                add(plan, db, f"blk.{layer}.ffn_{part}.weight", f"{p}.mlp.{part}_proj.weight", QTYPE_Q8_0, "dense_ffn")
            continue
        m = f"{p}.block_sparse_moe"
        add(plan, db, f"blk.{layer}.ffn_gate_inp.weight", f"{m}.gate.weight", QTYPE_F32, "router")
        add(plan, db, f"blk.{layer}.exp_probs_b.bias", f"{m}.e_score_correction_bias", QTYPE_F32, "router")
        # w1 = gate, w3 = up, w2 = down (Mixtral naming).
        for part, source_part in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
            add_experts(plan, db, layer, part, source_part, routed[part])
        for part in ("gate", "up", "down"):
            add(plan, db, f"blk.{layer}.ffn_{part}_shexp.weight", f"{m}.shared_experts.{part}_proj.weight",
                QTYPE_Q8_0, "shared_expert")
    add(plan, db, "output_norm.weight", f"{PREFIX}.norm.weight", QTYPE_F32, "output", GEMMA_NORM)
    add(plan, db, "output.weight", "language_model.lm_head.weight", QTYPE_Q8_0, "output")

    offset = 0
    for item in plan:
        item.offset = offset
        offset += gq.align(item.nbytes)
    return plan


def kv_string_array(key, values):
    out = pack_string(key) + gq.struct.pack("<IIQ", gq.GGUF_ARRAY, gq.GGUF_STRING, len(values))
    return out + b"".join(pack_string(v) for v in values)


def kv_i32_array(key, values):
    return (pack_string(key) + gq.struct.pack("<IIQ", gq.GGUF_ARRAY, GGUF_INT32, len(values))
            + gq.struct.pack(f"<{len(values)}i", *values))


def tokenizer_records(hf_dir):
    """GPT-2 byte-level BPE metadata straight from tokenizer.json."""
    document = read_json(hf_dir, "tokenizer.json")
    model = document.get("model", {})
    if model.get("type") != "BPE" or not isinstance(model.get("merges"), list):
        fail("tokenizer.json is not a BPE tokenizer")
    tokens = gq.load_source_tokens(hf_dir, N_VOCAB)
    special = {entry["id"] for entry in document["added_tokens"] if entry.get("special")}
    added = {entry["id"] for entry in document["added_tokens"]}
    # llama.cpp token types: 1 normal, 3 control, 4 user-defined, 5 unused.
    types = [3 if i in special else 4 if i in added else 5 if t.startswith("[PAD") else 1
             for i, t in enumerate(tokens)]
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    by_text = {t: i for i, t in enumerate(tokens)}
    return [
        kv_string("tokenizer.ggml.model", "gpt2"),
        kv_string("tokenizer.ggml.pre", "minimax-m2"),
        kv_string_array("tokenizer.ggml.tokens", tokens),
        kv_i32_array("tokenizer.ggml.token_type", types),
        kv_string_array("tokenizer.ggml.merges", merges),
        kv_u32("tokenizer.ggml.bos_token_id", by_text["]~b]"]),
        kv_u32("tokenizer.ggml.eos_token_id", by_text["[e~["]),
        kv_bool("tokenizer.ggml.add_bos_token", False),
    ]


def model_metadata(hf_dir, source_revision, n_layer=N_LAYER):
    config = read_json(hf_dir, "config.json")
    if not isinstance(config.get("text_config"), dict):
        raise ValueError("config.json has no text_config")
    text: dict = config["text_config"]
    sparse: dict = text["sparse_attention_config"]
    checks = {
        "hidden_size": N_EMBD, "num_hidden_layers": N_LAYER, "num_local_experts": N_EXPERT,
        "vocab_size": N_VOCAB, "intermediate_size": N_FF_EXP, "num_attention_heads": 64,
        "num_key_value_heads": 4, "head_dim": 128, "rotary_dim": 64, "num_experts_per_tok": 4,
        "hidden_act": "swigluoai", "scoring_func": "sigmoid", "use_gemma_norm": True,
        "qk_norm_type": "per_head", "attention_output_gate": False,
    }
    for key, value in checks.items():
        if text.get(key) != value:
            fail(f"config {key}={text.get(key)!r}, converter expects {value!r}")
    if text["moe_layer_freq"] != [0] * N_DENSE + [1] * (N_LAYER - N_DENSE):
        fail("unexpected moe_layer_freq")
    if sparse.get("sparse_attention_freq") != [1] * N_LAYER or sparse.get("sparse_score_type") != "max":
        fail("converter expects max-score sparse attention in every layer")
    if sparse["sparse_num_index_heads"] != 4 or sparse["sparse_index_dim"] != 128:
        fail("unexpected sparse indexer shape")
    chat_template = read_bytes(hf_dir, "chat_template.jinja")
    a = ARCH
    return [
        kv_string("general.architecture", a),
        kv_string("general.name", "MiniMax-M3.1-Flash"),
        kv_u32("general.alignment", gq.GGUF_ALIGNMENT),
        kv_string("general.source.revision", source_revision),
        kv_u32(f"{a}.block_count", n_layer),
        kv_u64(f"{a}.context_length", text["max_position_embeddings"]),
        kv_u32(f"{a}.embedding_length", N_EMBD),
        kv_u32(f"{a}.vocab_size", N_VOCAB),
        kv_u32(f"{a}.feed_forward_length", text["dense_intermediate_size"]),
        kv_u32(f"{a}.expert_feed_forward_length", N_FF_EXP),
        kv_u32(f"{a}.expert_count", N_EXPERT),
        kv_u32(f"{a}.expert_used_count", text["num_experts_per_tok"]),
        kv_u32(f"{a}.expert_shared_count", text["n_shared_experts"]),
        kv_u32(f"{a}.leading_dense_block_count", N_DENSE),
        kv_f32(f"{a}.expert_weights_scale", text["routed_scaling_factor"]),
        kv_bool(f"{a}.expert_weights_norm", True),
        kv_f32(f"{a}.swiglu_alpha", text["swiglu_alpha"]),
        kv_f32(f"{a}.swiglu_limit", text["swiglu_limit"]),
        kv_f32(f"{a}.attention.layer_norm_rms_epsilon", text["rms_norm_eps"]),
        kv_u32(f"{a}.attention.head_count", text["num_attention_heads"]),
        kv_u32(f"{a}.attention.head_count_kv", text["num_key_value_heads"]),
        kv_u32(f"{a}.attention.key_length", text["head_dim"]),
        kv_u32(f"{a}.attention.value_length", text["head_dim"]),
        kv_u32(f"{a}.rope.dimension_count", text["rotary_dim"]),
        kv_f32(f"{a}.rope.freq_base", float(text["rope_theta"])),
        kv_u32(f"{a}.attention.indexer.head_count", sparse["sparse_num_index_heads"]),
        kv_u32(f"{a}.attention.indexer.key_length", sparse["sparse_index_dim"]),
        kv_u32(f"{a}.attention.sparse.block_size", sparse["sparse_block_size"]),
        kv_u32(f"{a}.attention.sparse.top_k_blocks", sparse["sparse_topk_blocks"]),
        kv_u32(f"{a}.attention.sparse.init_blocks", sparse["sparse_init_block"]),
        kv_u32(f"{a}.attention.sparse.local_blocks", sparse["sparse_local_block"]),
        kv_string("tokenizer.chat_template", chat_template),
    ]


def main():
    parser = argparse.ArgumentParser(description="Build MiniMax-M3.1-Flash GGUF files for DwarfStar.")
    parser.add_argument("--hf", required=True, help="official MiniMax-M3.1-Flash snapshot")
    parser.add_argument("--out", help="output GGUF")
    parser.add_argument("--artifact", choices=("q2", "q4"), default="q2")
    parser.add_argument("--imatrix", help="legacy DS4/llama.cpp imatrix .dat")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--source-revision", default="main")
    parser.add_argument("--quants-library", help="path to libds4quants")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--layers", type=int, default=N_LAYER,
                        help="keep only the first N layers (validation GGUF, like minimax31_truncate.py)")
    args = parser.parse_args()
    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run is used")
    if not N_DENSE < args.layers <= N_LAYER:
        parser.error(f"--layers must be between {N_DENSE + 1} and {N_LAYER}")

    # The shared writer looks these up at module level.
    gq.Quantizer = Quantizer
    gq.transform_regular = transform_regular

    db = gq.SourceDB(args.hf, index_validator=validate_index, scale_validator=no_scale_check,
                     tensor_prefix="language_model.")
    try:
        plan = build_plan(db, args.artifact, args.layers)
        kv = model_metadata(args.hf, args.source_revision, args.layers)
        tok = tokenizer_records(args.hf)
        if args.dry_run:
            gq.print_plan(plan, kv, tok)
        else:
            gq.write_gguf(args, plan, kv, tok, db)
            print(f"minimax31-quantize: wrote {args.out}", file=sys.stderr)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"minimax31-quantize: error: {error}", file=sys.stderr)
        sys.exit(1)
