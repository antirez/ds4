#!/usr/bin/env python3.12
"""Generate a GGUF template for deepseek4-quantize from safetensors metadata.

The template contains only GGUF header + metadata + tensor info descriptors.
The deepseek4-quantize tool will regenerate all tensor data from safetensors.

Usage:
    python3.12 gen_gguf_template.py --hf ../DeepSeek-V4-Flash-4Expert --out template.gguf
"""

import json, struct, sys, os, argparse
from collections import OrderedDict

# ── GGUF constants ──
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGUF tensor types (dtype for tensors)
GGUF_TENSOR_F32 = 0
GGUF_TENSOR_F16 = 1
GGUF_TENSOR_Q4_0 = 2
GGUF_TENSOR_Q4_1 = 3
GGUF_TENSOR_Q8_0 = 8
GGUF_TENSOR_Q8_1 = 9
GGUF_TENSOR_Q2_K = 10
GGUF_TENSOR_Q3_K = 11
GGUF_TENSOR_Q4_K = 12
GGUF_TENSOR_Q5_K = 13
GGUF_TENSOR_Q6_K = 14
GGUF_TENSOR_Q8_K = 15
GGUF_TENSOR_IQ2_XXS = 16
GGUF_TENSOR_IQ2_XS = 17
GGUF_TENSOR_IQ3_XXS = 18
GGUF_TENSOR_IQ1_S = 19
GGUF_TENSOR_IQ4_NL = 20
GGUF_TENSOR_IQ3_S = 21
GGUF_TENSOR_IQ2_S = 22
GGUF_TENSOR_IQ4_XS = 23
GGUF_TENSOR_I8 = 24
GGUF_TENSOR_I16 = 25
GGUF_TENSOR_I32 = 26
GGUF_TENSOR_I64 = 27
GGUF_TENSOR_F64 = 28
GGUF_TENSOR_IQ1_M = 29
GGUF_TENSOR_BF16 = 30

def write_u8(f, v): f.write(struct.pack('<B', v))
def write_u16(f, v): f.write(struct.pack('<H', v))
def write_u32(f, v): f.write(struct.pack('<I', v))
def write_i32(f, v): f.write(struct.pack('<i', v))
def write_u64(f, v): f.write(struct.pack('<Q', v))
def write_f32(f, v): f.write(struct.pack('<f', v))
def write_f64(f, v): f.write(struct.pack('<d', v))
def write_bool(f, v): f.write(struct.pack('<?', v))
def write_string(f, s):
    b = s.encode()
    write_u64(f, len(b))
    f.write(b)

def write_gguf_value(f, typecode, value):
    """Write a GGUF metadata value of the given type."""
    if typecode == GGUF_TYPE_UINT32:
        write_u32(f, value)
    elif typecode == GGUF_TYPE_UINT64:
        write_u64(f, value)
    elif typecode == GGUF_TYPE_FLOAT32:
        write_f32(f, value)
    elif typecode == GGUF_TYPE_FLOAT64:
        write_f64(f, value)
    elif typecode == GGUF_TYPE_BOOL:
        write_bool(f, value)
    elif typecode == GGUF_TYPE_STRING:
        write_string(f, value)
    elif typecode == GGUF_TYPE_UINT8:
        write_u8(f, value)
    elif typecode == GGUF_TYPE_INT32:
        write_i32(f, value)
    elif typecode == GGUF_TYPE_ARRAY:
        elem_type, arr = value
        write_u32(f, elem_type)
        write_u64(f, len(arr))
        for v in arr:
            write_gguf_value(f, elem_type, v)
    else:
        raise ValueError(f"Unknown GGUF type: {typecode}")

# ── Layer mapping: GGUF name suffix -> HF safetensors name suffix ──
LAYER_MAP = OrderedDict([
    ("hc_attn_base.weight",             "hc_attn_base"),
    ("hc_attn_fn.weight",               "hc_attn_fn"),
    ("hc_attn_scale.weight",            "hc_attn_scale"),
    ("hc_ffn_base.weight",              "hc_ffn_base"),
    ("hc_ffn_fn.weight",                "hc_ffn_fn"),
    ("hc_ffn_scale.weight",             "hc_ffn_scale"),
    ("attn_sinks.weight",               "attn.attn_sink"),
    ("attn_q_a.weight",                "attn.wq_a.weight"),
    ("attn_q_b.weight",                "attn.wq_b.weight"),
    ("attn_q_a_norm.weight",           "attn.q_norm.weight"),
    ("attn_kv.weight",                 "attn.wkv.weight"),
    ("attn_kv_a_norm.weight",          "attn.kv_norm.weight"),
    ("attn_output_a.weight",           "attn.wo_a.weight"),
    ("attn_output_b.weight",           "attn.wo_b.weight"),
    ("attn_compressor_ape.weight",     "attn.compressor.ape"),
    ("attn_compressor_kv.weight",      "attn.compressor.wkv.weight"),
    ("attn_compressor_gate.weight",    "attn.compressor.wgate.weight"),
    ("attn_compressor_norm.weight",    "attn.compressor.norm.weight"),
    ("indexer.attn_q_b.weight",        "attn.indexer.wq_b.weight"),
    ("indexer.proj.weight",            "attn.indexer.weights_proj.weight"),
    ("indexer_compressor_ape.weight",  "attn.indexer.compressor.ape"),
    ("indexer_compressor_kv.weight",   "attn.indexer.compressor.wkv.weight"),
    ("indexer_compressor_gate.weight", "attn.indexer.compressor.wgate.weight"),
    ("indexer_compressor_norm.weight", "attn.indexer.compressor.norm.weight"),
    ("attn_norm.weight",               "attn_norm.weight"),
    ("ffn_norm.weight",                "ffn_norm.weight"),
    ("ffn_gate_shexp.weight",          "ffn.shared_experts.w1.weight"),
    ("ffn_up_shexp.weight",            "ffn.shared_experts.w3.weight"),
    ("ffn_down_shexp.weight",          "ffn.shared_experts.w2.weight"),
    ("ffn_gate_inp.weight",            "ffn.gate.weight"),
    ("exp_probs_b.bias",               "ffn.gate.bias"),
    ("ffn_gate_tid2eid.weight",        "ffn.gate.tid2eid"),
])

# HF tensor name to scale companion
HF_SCALE_SUFFIXES = {
    ".weight": ".scale",
    ".bias": ".scale",  # actually no, but some have scales
}

def hf_dtype_to_gguf(hf_dtype, tensor_name=""):
    """Map HF safetensors dtype to GGUF template tensor type.
    
    Simple rule: everything -> F32 (the quant policy handles 2D+ tensors via 
    --experts/--attention/--dense flags), except I64 -> I32 (tid2eid routing table).
    
    F32 is safe because:
    - 1D tensors (norms/scales/bias): policy does NOT apply, F32 preserved
    - 2D tensors: policy applies and overrides F32 to the correct quant type
    """
    if hf_dtype == "I64":
        return GGUF_TENSOR_I32
    return GGUF_TENSOR_F32

def main():
    parser = argparse.ArgumentParser(description="Generate GGUF template for deepseek4-quantize")
    parser.add_argument("--hf", required=True, help="HuggingFace model directory")
    parser.add_argument("--out", required=True, help="Output GGUF template path")
    parser.add_argument("--n-experts", type=int, default=256, help="Number of routed experts")
    parser.add_argument("--n-layers", type=int, default=43, help="Number of layers")
    parser.add_argument("--n-expert-used", type=int, default=4, help="Experts used per token")
    args = parser.parse_args()

    # ── 1. Read safetensors index ──
    index_path = os.path.join(args.hf, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found")
        sys.exit(1)
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    print(f"Found {len(weight_map)} tensors in index")

    # Determine shapes by reading all safetensors headers
    hf_shapes = {}  # hf_name -> (shape, dtype)
    st_files = sorted(set(weight_map.values()))
    print(f"Reading {len(st_files)} safetensors files for tensor metadata...")
    for st_file in st_files:
        st_path = os.path.join(args.hf, st_file)
        with open(st_path, "rb") as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_size).decode())
        for name in header:
            hf_shapes[name] = (tuple(header[name]["shape"]), header[name]["dtype"])

    print(f"Parsed {len(hf_shapes)} tensor shapes. Writing template...")

    # ── 2. Build metadata KVs ──
    metadata = []

    def add_u32(key, val):
        metadata.append((key, GGUF_TYPE_UINT32, val))

    def add_u64(key, val):
        metadata.append((key, GGUF_TYPE_UINT64, val))

    def add_f32(key, val):
        metadata.append((key, GGUF_TYPE_FLOAT32, val))

    def add_bool(key, val):
        metadata.append((key, GGUF_TYPE_BOOL, val))

    def add_str(key, val):
        metadata.append((key, GGUF_TYPE_STRING, val))

    def add_f64(key, val):
        metadata.append((key, GGUF_TYPE_FLOAT64, val))

    def add_arr(key, elem_type, val_list):
        metadata.append((key, GGUF_TYPE_ARRAY, (elem_type, val_list)))

    add_str("general.architecture", "deepseek4")
    add_str("general.name", "DeepSeek V4 Flash 4Expert")
    add_u32("general.alignment", 32)

    # Pre-tokenizer
    add_str("tokenizer.ggml.pre", "joyai-llm")

    # Model config
    add_u32("deepseek4.block_count", args.n_layers)
    add_u64("deepseek4.context_length", 65536)
    add_u32("deepseek4.embedding_length", 4096)
    add_u32("deepseek4.vocab_size", 129280)
    add_u32("deepseek4.attention.head_count", 64)
    add_u32("deepseek4.attention.head_count_kv", 1)
    add_u32("deepseek4.attention.key_length", 512)
    add_u32("deepseek4.attention.value_length", 512)
    add_u32("deepseek4.rope.dimension_count", 64)
    add_u32("deepseek4.attention.q_lora_rank", 1024)
    add_u32("deepseek4.attention.output_lora_rank", 1024)
    add_u32("deepseek4.attention.output_group_count", 8)
    add_u32("deepseek4.expert_count", args.n_experts)
    add_u32("deepseek4.expert_used_count", args.n_expert_used)
    add_u32("deepseek4.expert_feed_forward_length", 2048)
    add_u32("deepseek4.expert_shared_count", 1)
    add_u32("deepseek4.hash_layer_count", 3)
    add_u32("deepseek4.attention.sliding_window", 128)
    add_u32("deepseek4.attention.indexer.head_count", 64)
    add_u32("deepseek4.attention.indexer.key_length", 128)
    add_u32("deepseek4.attention.indexer.top_k", 512)
    add_u32("deepseek4.hyper_connection.count", 4)
    add_u32("deepseek4.hyper_connection.sinkhorn_iterations", 20)

    add_f32("deepseek4.attention.layer_norm_rms_epsilon", 1.0e-6)
    add_f32("deepseek4.hyper_connection.epsilon", 1.0e-6)
    add_f32("deepseek4.expert_weights_scale", 1.5)
    add_bool("deepseek4.expert_weights_norm", True)
    add_f64("deepseek4.rope.freq_base", 10000.0)
    add_f64("deepseek4.rope.scaling.factor", 16.0)
    add_f32("deepseek4.rope.scaling.yarn_beta_fast", 32.0)
    add_f32("deepseek4.rope.scaling.yarn_beta_slow", 1.0)
    add_f64("deepseek4.attention.compress_rope_freq_base", 160000.0)
    add_u64("deepseek4.rope.scaling.original_context_length", 65536)

    # Compress ratios per layer (required array)
    compress_ratios = []
    for il in range(args.n_layers):
        if il < 2:
            compress_ratios.append(0)
        elif il % 2 == 0:
            compress_ratios.append(4)
        else:
            compress_ratios.append(128)
    add_arr("deepseek4.attention.compress_ratios", GGUF_TYPE_UINT32, compress_ratios)

    # SwiGLU clamp exponents (required float array per layer)
    swiglu_clamp = [10.0] * args.n_layers
    add_arr("deepseek4.swiglu_clamp_exp", GGUF_TYPE_FLOAT32, swiglu_clamp)

    # ── Tokenizer data ──
    with open(os.path.join(args.hf, "tokenizer_config.json")) as f:
        tok_config = json.load(f)
    with open(os.path.join(args.hf, "tokenizer.json")) as f:
        tok_data = json.load(f)

    model_type = tok_data.get("model", {}).get("type", "gpt2").lower()
    add_str("tokenizer.ggml.model", model_type)

    # Build vocabulary: base vocab (0..127999) + added tokens (128000+)
    vocab_dict = tok_data.get("model", {}).get("vocab", {})
    merges = tok_data.get("model", {}).get("merges", [])
    added_tokens = tok_data.get("added_tokens", [])

    # Sort by ID
    base_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    added_vocab = sorted(added_tokens, key=lambda x: x["id"])

    # Combine: base vocab (IDs 0..127999), then added tokens (IDs 128000+)
    all_tokens = [t for t, _ in base_vocab]
    # Check if added tokens fill IDs 128000..129279
    for at in added_vocab:
        tid = at["id"]
        # Extend tokens list if needed
        while len(all_tokens) <= tid:
            all_tokens.append("")
        all_tokens[tid] = at["content"]

    add_arr("tokenizer.ggml.tokens", GGUF_TYPE_STRING, all_tokens)

    # Scores: 0 for special tokens, default -inf for others
    # BPE tokens don't always have scores. Use 0.0 as default.
    scores = [0.0] * len(all_tokens)
    add_arr("tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores)

    # Token types: 3=CONTROL for special, 1=UNKNOWN for unk, 0=NORMAL for others
    token_types = [0] * len(all_tokens)
    token_types[0] = 3   # BOS = control
    token_types[1] = 3   # EOS = control
    token_types[2] = 3   # PAD = control
    add_arr("tokenizer.ggml.token_type", GGUF_TYPE_UINT32, token_types)

    # Merges (BPE)
    add_arr("tokenizer.ggml.merges", GGUF_TYPE_STRING, merges)

    # Special token IDs
    add_u32("tokenizer.ggml.bos_token_id", 0)
    add_u32("tokenizer.ggml.eos_token_id", 1)
    add_bool("tokenizer.ggml.add_bos_token", False)
    add_bool("tokenizer.ggml.add_eos_token", False)

    # Chat template
    chat_template = tok_config.get("chat_template", "")
    if chat_template:
        add_str("tokenizer.chat_template", chat_template)

    # ── 3. Build tensor list ──
    tensor_infos = []  # (name, rank, dims_list, gguf_type)

    # Top-level tensors
    top_tensors = {
        "token_embd.weight": "embed.weight",
        "output.weight": "head.weight",
        "output_norm.weight": "norm.weight",
        "output_hc_base.weight": "hc_head_base",
        "output_hc_fn.weight": "hc_head_fn",
        "output_hc_scale.weight": "hc_head_scale",
    }
    for gguf_name, hf_name in top_tensors.items():
        if hf_name in hf_shapes:
            shape, dtype = hf_shapes[hf_name]
            dims = list(reversed(shape))  # GGUF uses reversed dims
            dtype = hf_dtype_to_gguf(hf_shapes[hf_name][1]) if hf_name in hf_shapes else GGUF_TENSOR_F16
            tensor_infos.append((gguf_name, len(dims), dims, dtype))

    # Per-layer tensors
    for layer in range(args.n_layers):
        for gguf_suffix, hf_suffix in LAYER_MAP.items():
            hf_name = f"layers.{layer}.{hf_suffix}"
            if hf_name not in hf_shapes:
                # Some tensors only exist in certain layers
                if gguf_suffix == "ffn_gate_tid2eid.weight" and layer >= 3:
                    continue
                if "compressor" in gguf_suffix and layer >= 3:
                    continue
                if "indexer" in gguf_suffix:
                    continue  # indexer tensors have different naming
                continue
            shape, dtype = hf_shapes[hf_name]
            dims = list(reversed(shape))
            tensor_infos.append((f"blk.{layer}.{gguf_suffix}", len(dims), dims,
                                hf_dtype_to_gguf(hf_shapes[hf_name][1])))

    # Add scale tensors (for quantized weights with .scale companions)
    extra_tensors = []
    for name, rank, dims, dtype in tensor_infos:
        # Check if there's a .scale companion in HF
        # GGUF naming for scales: weight_name is the base, scale is embedded
        pass  # The quantizer handles scale merging

    # Expert tensors (routed experts)
    # GGUF shapes from ds4.c tensor_expect_routed_expert:
    #   gate/up: [DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT] = [4096, 2048, 256]
    #   down:    [DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT] = [2048, 4096, 256]
    for layer in range(args.n_layers):
        for expert_type, hf_pattern, dims in [
            ("ffn_gate_exps.weight", "w1", [4096, 2048, args.n_experts]),
            ("ffn_up_exps.weight",   "w3", [4096, 2048, args.n_experts]),
            ("ffn_down_exps.weight", "w2", [2048, 4096, args.n_experts]),
        ]:
            tensor_infos.append((f"blk.{layer}.{expert_type}", len(dims), dims,
                                GGUF_TENSOR_F16))  # policy will quantize

    print(f"Total tensor descriptors: {len(tensor_infos)}")

    # ── 4. Write GGUF file ──
    with open(args.out, "wb") as f:
        # Magic + Version
        f.write(GGUF_MAGIC)
        write_u32(f, GGUF_VERSION)
        write_u64(f, len(tensor_infos))  # n_tensors
        write_u64(f, len(metadata))      # n_kv

        # Write metadata KVs
        for key, typecode, val in metadata:
            write_string(f, key)
            write_u32(f, typecode)
            write_gguf_value(f, typecode, val)

        # Write tensor infos
        for name, rank, dims, dtype in tensor_infos:
            write_string(f, name)
            write_u32(f, rank)
            for d in dims:
                write_u64(f, d)
            write_u32(f, dtype)
            write_u64(f, 0)  # offset (placeholder, quantizer will rewrite)

        # Pad to alignment
        pos = f.tell()
        aligned = ((pos + 31) // 32) * 32
        f.write(b'\0' * (aligned - pos))

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"\nTemplate written: {args.out} ({size_mb:.1f} MB)")
    print(f"\nNow run:")
    print(f"  ./gguf-tools/deepseek4-quantize \\")
    print(f"    --hf {args.hf} \\")
    print(f"    --template {args.out} \\")
    print(f"    --out ds4flash-4expert.gguf \\")
    print(f"    --experts q4_k \\")
    print(f"    --attention-proj q8_0 \\")
    print(f"    --shared q8_0 \\")
    print(f"    --output q8_0 \\")
    print(f"    --embedding f16 \\")
    print(f"    --dense f16 \\")
    print(f"    --n-experts {args.n_experts} \\")
    print(f"    --overwrite")

if __name__ == "__main__":
    main()
