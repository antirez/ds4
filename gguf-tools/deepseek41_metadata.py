#!/usr/bin/env python3
"""DeepSeek V4.1 tokenizer and Engram metadata, derived from the released config.

The normalization, prime allocation and seeded hash multipliers follow DeepSeek's
MIT-licensed inference/engram.py. The runtime consumes the resulting arrays, not
a second implementation of Unicode normalization or NumPy's random generator.
"""

import json
import os
import struct

from glm53_quantize import (
    GGUF_ARRAY, GGUF_STRING, GGUF_UINT32, GGUF_UINT64, kv_bool, kv_string,
    kv_f32, kv_u32, kv_u32_array, pack_string,
)

# Apple Silicon VM pages are 16 KiB. The Engram extent must be removable
# without also unmapping bytes of the preceding resident tensor.
GGUF_ALIGNMENT = 16384


def array_record(key, kind, values):
    header = pack_string(key) + struct.pack("<IIQ", GGUF_ARRAY, kind, len(values))
    if kind == GGUF_STRING:
        return header + b"".join(pack_string(value) for value in values)
    fmt = {GGUF_UINT32: "I", GGUF_UINT64: "Q", 5: "i", 6: "f"}[kind]
    return header + struct.pack(f"<{len(values)}{fmt}", *values)


def compressed_token_map(tokenizer):
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(),
        normalizers.Lowercase(), normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel), normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    unique, result = {}, []
    for token_id in range(tokenizer.get_vocab_size(with_added_tokens=True)):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = tokenizer.id_to_token(token_id)
        else:
            key = normalizer.normalize_str(text) or text
        result.append(unique.setdefault(key, len(unique)))
    return result, len(unique)


def engram_layout(config, tokenizer):
    import numpy as np
    from sympy import nextprime

    token_map, vocab_size = compressed_token_map(tokenizer)
    if vocab_size != config["engram_compressed_vocab_size"]:
        raise ValueError(f"Engram vocabulary mismatch: {vocab_size}")
    if len(token_map) != config["vocab_size"]:
        raise ValueError("Engram requires every released tokenizer ID")
    layers = config["engram_layer_ids"]
    if len(layers) != 2 or config["engram_max_ngram_size"] != 4 or \
            config["engram_n_heads"] != 8 or config["engram_head_dim"] != 256:
        raise ValueError("unsupported Engram shape")
    seen, primes, multipliers = set(), [], []
    bound = (np.iinfo(np.int64).max // vocab_size) // 2
    for layer in layers:
        row = []
        for _ in range(config["engram_max_ngram_size"] - 1):
            current = config["engram_vocab_size"] - 1
            for _ in range(config["engram_n_heads"]):
                current = int(nextprime(current))
                while current in seen:
                    current = int(nextprime(current))
                seen.add(current)
                row.append(current)
        primes.append(row)
        rng = np.random.default_rng(10007 * layer)
        multipliers.append((rng.integers(0, bound, size=4, dtype=np.int64) * 2 + 1).tolist())
    if [sum(row) for row in primes] != config["engram_num_embeddings"]:
        raise ValueError("Engram table size differs from hash bucket layout")
    return dict(token_map=token_map, compressed_vocab_size=vocab_size,
                pad_id=token_map[config["engram_pad_token_id"]],
                rows=config["engram_num_embeddings"], layers=layers,
                primes=primes, multipliers=multipliers)


def metadata(hf_dir, revision):
    from tokenizers import Tokenizer

    with open(os.path.join(hf_dir, "config.json"), "rb") as fp:
        config = json.load(fp)
    if config["model_type"] != "deepseek_v41":
        raise ValueError("not a DeepSeek V4.1 source checkpoint")
    text = config["text_config"]
    tokenizer = Tokenizer.from_file(os.path.join(hf_dir, "tokenizer.json"))
    layout = engram_layout(text, tokenizer)
    with open(os.path.join(hf_dir, "tokenizer.json"), "rb") as fp:
        tok = json.load(fp)
    if tok["model"]["type"] != "BPE":
        raise ValueError("unsupported tokenizer")
    # This tokenizer lists special tokens in both the BPE vocabulary and
    # added_tokens. Its parsed ID table resolves that intentional overlap.
    tokens = [tokenizer.id_to_token(i) for i in range(text["vocab_size"])]
    if any(token is None for token in tokens):
        raise ValueError("tokenizer has missing IDs")
    special = {item["id"] for item in tok["added_tokens"] if item["special"]}
    merges = [" ".join(pair) if isinstance(pair, list) else pair for pair in tok["model"]["merges"]]
    records = [
        kv_string("general.architecture", "deepseek41"),
        kv_string("general.name", "DeepSeek V4.1 Flash"),
        kv_string("general.source.url", "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash"),
        kv_string("general.source.revision", revision),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string("deepseek41.config", json.dumps(config, sort_keys=True, separators=(",", ":"))),
        kv_string("deepseek41.engram.encoding", "e4m3_e8m0_32_row264"),
        kv_u32_array("deepseek41.engram.layer_ids", layout["layers"]),
        kv_u32_array("deepseek41.engram.rows", layout["rows"]),
        kv_u32("deepseek41.engram.compressed_vocab_size", layout["compressed_vocab_size"]),
        kv_u32("deepseek41.engram.pad_id", layout["pad_id"]),
        kv_u32_array("deepseek41.engram.token_map", layout["token_map"]),
        kv_u32_array("deepseek41.engram.primes", sum(layout["primes"], [])),
        array_record("deepseek41.engram.multipliers", GGUF_UINT64, sum(layout["multipliers"], [])),
        kv_string("tokenizer.ggml.model", "gpt2"),
        kv_string("tokenizer.ggml.pre", "joyai-llm"),
        array_record("tokenizer.ggml.tokens", GGUF_STRING, tokens),
        array_record("tokenizer.ggml.token_type", 5, [3 if i in special else 1 for i in range(len(tokens))]),
        array_record("tokenizer.ggml.merges", GGUF_STRING, merges),
        kv_u32("tokenizer.ggml.bos_token_id", config["bos_token_id"]),
        kv_u32("tokenizer.ggml.eos_token_id", config["eos_token_id"]),
        kv_bool("tokenizer.ggml.add_bos_token", True),
        kv_bool("tokenizer.ggml.add_eos_token", False),
    ]
    for key in (
        "vocab_size", "hidden_size", "moe_intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "qk_rope_head_dim",
        "q_lora_rank", "o_lora_rank", "o_groups", "n_routed_experts", "n_shared_experts",
        "num_experts_per_tok", "max_position_embeddings", "sliding_window",
        "index_n_heads", "index_head_dim", "index_topk", "candidate_source_layer_id",
        "candidate_topk_blocks", "candidate_block_size", "hc_mult", "hc_sinkhorn_iters",
        "rope_theta", "compress_rope_theta",
    ):
        records.append(kv_u32(f"deepseek41.{key}", text[key]))
    for key in ("rms_norm_eps", "hc_eps", "swiglu_limit", "routed_scaling_factor"):
        records.append(kv_f32(f"deepseek41.{key}", text[key]))
    for key in ("scoring_func", "hidden_act", "topk_method"):
        records.append(kv_string(f"deepseek41.{key}", text[key]))
    records.append(kv_bool("deepseek41.norm_topk_prob", text["norm_topk_prob"]))
    records.append(kv_u32_array("deepseek41.compress_ratios", text["compress_ratios"][:text["num_hidden_layers"]]))
    for key in ("kv_source_layer_ids", "index_source_layer_ids"):
        records.append(kv_u32_array(f"deepseek41.{key}", text[key]))
    for key in ("factor", "beta_fast", "beta_slow", "original_max_position_embeddings"):
        records.append(kv_f32(f"deepseek41.rope_scaling.{key}", text["rope_scaling"][key]))
    return config, records
