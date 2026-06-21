#!/usr/bin/env python3
"""Build a GLM-5.2 template GGUF for ds4.

The template supplies:
  - general.architecture = "glm_moe_dsa"
  - deepseek4.* dimensional metadata (reused so ds4's existing reader works)
  - glm.* GLM-specific keys (new, read only for GLM-variant)
  - tokenizer.ggml.{tokens,token_type,merges,scores} from the GLM tokenizer
  - tensor metadata (name + shape + template dtype) for every GLM tensor

Tensor bytes are NOT written; glm-quantize regenerates from HF safetensors.
"""
import json
import struct
import sys
import os
import numpy as np

try:
    import gguf
except ImportError:
    print("pip install gguf", file=sys.stderr)
    sys.exit(1)

CONFIG_PATH = os.environ.get("GLM_CONFIG", "config.json")
TOKENIZER_PATH = os.environ.get("GLM_TOKENIZER", "tokenizer.json")
TOKENIZER_CFG_PATH = os.environ.get("GLM_TOKENIZER_CFG", "tokenizer_config.json")
OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "glm-template.gguf"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_tokenizer():
    with open(TOKENIZER_PATH) as f:
        return json.load(f)


def load_tokenizer_config():
    with open(TOKENIZER_CFG_PATH) as f:
        return json.load(f)


def main():
    cfg = load_config()
    tok = load_tokenizer()
    tok_cfg = load_tokenizer_config()

    n_layers = cfg["num_hidden_layers"]  # 78 (excl. MTP)
    n_mtp = cfg.get("num_nextn_predict_layers", 1)  # 1
    hidden = cfg["hidden_size"]  # 6144
    n_heads = cfg["num_attention_heads"]  # 64
    n_kv_heads = cfg["num_key_value_heads"]  # 64 (GLM uses MQA→64 via kv_a_proj)
    q_lora = cfg["q_lora_rank"]  # 2048
    kv_lora = cfg["kv_lora_rank"]  # 512
    qk_nope = cfg["qk_nope_head_dim"]  # 192
    qk_rope = cfg["qk_rope_head_dim"]  # 64
    qk_head = cfg["qk_head_dim"]  # 256
    v_head = cfg["v_head_dim"]  # 256
    n_experts = cfg["n_routed_experts"]  # 256
    n_experts_used = cfg["num_experts_per_tok"]  # 8
    n_shared = cfg["n_shared_experts"]  # 1
    moe_inter = cfg["moe_intermediate_size"]  # 2048
    dense_inter = cfg["intermediate_size"]  # 12288
    first_k_dense = cfg["first_k_dense_replace"]  # 3
    rope_theta = cfg.get("rope_parameters", {}).get("rope_theta", 8000000.0)
    rms_eps = cfg["rms_norm_eps"]  # 1e-5
    routed_scale = cfg["routed_scaling_factor"]  # 2.5
    vocab_size = cfg["vocab_size"]  # 154880
    indexer_types = cfg["indexer_types"]
    index_topk = cfg["index_topk"]  # 2048
    index_n_heads = cfg["index_n_heads"]  # 32
    index_head_dim = cfg["index_head_dim"]  # 128
    index_topk_freq = cfg["index_topk_freq"]  # 4

    q_dim = n_heads * qk_head  # 64*256 = 16384 (q_b_proj out)
    kv_a_out = kv_lora + qk_rope  # 512+64 = 576 (kv_a_proj out)
    kv_b_out = n_heads * (qk_nope + v_head)  # 64*448 = 28672 (kv_b_proj out)
    o_in = n_heads * v_head  # 64*256 = 16384 (o_proj in)

    # ds4 tensor type codes (matching gguf.GGMLQuantizationType / ds4's TENSOR_
    F32 = gguf.GGMLQuantizationType.F32
    F16 = gguf.GGMLQuantizationType.F16
    Q8_0 = gguf.GGMLQuantizationType.Q8_0
    Q2_K = gguf.GGMLQuantizationType.Q2_K
    IQ2_XXS = gguf.GGMLQuantizationType.IQ2_XXS

    writer = gguf.GGUFWriter(path=OUT_PATH, arch="glm_moe_dsa")

    # --- General metadata ---
    writer.add_string("general.name", "GLM-5.2")
    writer.add_uint32("general.file_type", gguf.GGMLQuantizationType.Q8_0)

    # --- Reused deepseek4.* dimensional keys ---
    writer.add_uint32("deepseek4.block_count", n_layers)  # 78
    writer.add_uint32("deepseek4.embedding_length", hidden)  # 6144
    writer.add_uint32("deepseek4.vocab_size", vocab_size)  # 154880
    writer.add_uint32("deepseek4.attention.head_count", n_heads)  # 64
    writer.add_uint32("deepseek4.attention.head_count_kv", n_kv_heads)  # 64
    writer.add_uint32("deepseek4.attention.key_length", qk_head)  # 256
    writer.add_uint32("deepseek4.attention.value_length", v_head)  # 256
    writer.add_uint32("deepseek4.rope.dimension_count", qk_rope)  # 64
    writer.add_uint32("deepseek4.attention.q_lora_rank", q_lora)  # 2048
    writer.add_uint32("deepseek4.attention.output_lora_rank", 0)  # N/A for GLM
    writer.add_uint32("deepseek4.attention.output_group_count", 1)  # N/A
    writer.add_float32("deepseek4.attention.compress_rope_freq_base", 1e6)
    writer.add_uint32("deepseek4.expert_count", n_experts)  # 256
    writer.add_uint32("deepseek4.expert_used_count", n_experts_used)  # 8
    writer.add_uint32("deepseek4.expert_shared_count", n_shared)  # 1
    writer.add_uint32("deepseek4.expert_feed_forward_length", moe_inter)  # 2048
    writer.add_uint32("deepseek4.hash_layer_count", first_k_dense)  # reuse: 3 (GLM dense 0-2 have no hash-routing but we use same bound)
    writer.add_uint32("deepseek4.attention.sliding_window", 0)  # GLM has no SWA
    writer.add_uint32("deepseek4.attention.indexer.head_count", index_n_heads)  # 32
    writer.add_uint32("deepseek4.attention.indexer.key_length", index_head_dim)  # 128
    writer.add_uint32("deepseek4.attention.indexer.top_k", index_topk)  # 2048
    writer.add_uint32("deepseek4.hyper_connection.count", 0)  # no HC
    writer.add_uint32("deepseek4.hyper_connection.sinkhorn_iterations", 0)
    writer.add_float32("deepseek4.rope.freq_base", rope_theta)  # 8e6
    writer.add_float32("deepseek4.rope.scaling.factor", 1.0)
    writer.add_float32("deepseek4.rope.scaling.yarn_beta_fast", 32.0)
    writer.add_float32("deepseek4.rope.scaling.yarn_beta_slow", 1.0)
    writer.add_uint64("deepseek4.rope.scaling.original_context_length", 4096)
    writer.add_float32("deepseek4.expert_weights_scale", routed_scale)  # 2.5
    writer.add_float32("deepseek4.attention.layer_norm_rms_epsilon", rms_eps)
    writer.add_float32("deepseek4.hyper_connection.epsilon", 1e-6)
    writer.add_bool("deepseek4.expert_weights_norm", True)

    # --- GLM-specific keys ---
    writer.add_uint32("glm.block_count", n_layers)
    writer.add_uint32("glm.embedding_length", hidden)
    writer.add_uint32("glm.vocab_size", vocab_size)
    writer.add_uint32("glm.attention.head_count", n_heads)
    writer.add_uint32("glm.attention.q_lora_rank", q_lora)
    writer.add_uint32("glm.attention.kv_lora_rank", kv_lora)
    writer.add_uint32("glm.attention.qk_nope_head_dim", qk_nope)  # 192
    writer.add_uint32("glm.attention.qk_rope_head_dim", qk_rope)  # 64
    writer.add_uint32("glm.attention.qk_head_dim", qk_head)  # 256
    writer.add_uint32("glm.attention.v_head_dim", v_head)  # 256
    writer.add_uint32("glm.expert_count", n_experts)
    writer.add_uint32("glm.expert_used_count", n_experts_used)
    writer.add_uint32("glm.expert_shared_count", n_shared)
    writer.add_uint32("glm.expert_feed_forward_length", moe_inter)
    writer.add_uint32("glm.first_k_dense_replace", first_k_dense)  # 3
    writer.add_uint32("glm.dense_ff_length", dense_inter)  # 12288
    writer.add_uint32("glm.n_mtp_layers", n_mtp)  # 1
    writer.add_float32("glm.routed_scaling_factor", routed_scale)  # 2.5
    writer.add_uint32("glm.router_scoring", 1)  # 1 = sigmoid_noaux_tc
    writer.add_uint32("glm.index_topk", index_topk)
    writer.add_uint32("glm.index_n_heads", index_n_heads)
    writer.add_uint32("glm.index_head_dim", index_head_dim)
    writer.add_uint32("glm.index_topk_freq", index_topk_freq)
    # indexer_types as a string array
    writer.add_array("glm.indexer_types", indexer_types)

    # --- Tokenizer ---
    tok_model = tok["model"]
    tokens = [tok for tok, _ in sorted(tok_model["vocab"].items(), key=lambda x: x[1])]
    # Extend with added tokens
    added = tok.get("added_tokens", [])
    max_base_id = max(tok_model["vocab"].values()) if tok_model["vocab"] else -1
    # Build the full ID-ordered token list
    # Base vocab is 0..max_base_id (154819), added tokens 154820..154855
    all_tokens = list(tokens)
    # Add added tokens sorted by id
    for at in sorted(added, key=lambda a: a["id"]):
        if at["id"] >= len(all_tokens):
            # Pad any gap
            while len(all_tokens) < at["id"]:
                all_tokens.append("")
            all_tokens.append(at["content"])
    # Build token_type: 1=normal, 3=control (unused, all normal for BPE)
    token_types = [1] * len(all_tokens)
    for at in sorted(added, key=lambda a: a["id"]):
        if at["id"] < len(token_types):
            token_types[at["id"]] = 1  # all treated as normal for ds4 BPE path

    writer.add_string("tokenizer.ggml.model", "gpt2")  # BPE
    writer.add_array("tokenizer.ggml.tokens", all_tokens)
    writer.add_array("tokenizer.ggml.token_type", token_types)
    if "merges" in tok_model:
        merges = [" ".join(m) if isinstance(m, list) else m for m in tok_model["merges"]]
        writer.add_array("tokenizer.ggml.merges", merges)

    # BOS/EOS/pad from tokenizer_config
    eos_token = tok_cfg.get("eos_token", "")
    pad_token = tok_cfg.get("pad_token", "")
    # GLM eos is a dict with id; use first eos id from config
    if isinstance(eos_token, dict):
        writer.add_uint32("tokenizer.ggml.eos_token_id", eos_token.get("id", 154820))
    elif isinstance(eos_token, str):
        # Find in vocab
        eos_id = tok_model["vocab"].get(eos_token, 154820)
        writer.add_uint32("tokenizer.ggml.eos_token_id", eos_id)
    else:
        writer.add_uint32("tokenizer.ggml.eos_token_id", 154820)
    if isinstance(pad_token, dict):
        writer.add_uint32("tokenizer.ggml.pad_token_id", pad_token.get("id", 154820))
    elif isinstance(pad_token, str):
        pad_id = tok_model["vocab"].get(pad_token, 154820)
        writer.add_uint32("tokenizer.ggml.pad_token_id", pad_id)
    else:
        writer.add_uint32("tokenizer.ggml.pad_token_id", 154820)

    # --- Tensor metadata ---
    # For each layer, emit the GLM tensor set with template types.
    # The quantizer will override these per the quant policy.
    #
    # GLM tensor names (ds4 GGUF naming, GLM-specific additions):
    #   Common attention: attn_norm, attn_q_a, attn_q_a_norm, attn_q_b,
    #                     attn_kv, attn_kv_a_norm, attn_kv_b, attn_output
    #   Indexer (full layers only): indexer_attn_q_b, indexer_proj,
    #     indexer_compressor_ape, indexer_compressor_kv,
    #     indexer_compressor_gate, indexer_compressor_norm
    #   Dense MLP (layers 0..first_k_dense-1): ffn_gate, ffn_up, ffn_down
    #   MoE (layers first_k_dense..n_layers-1): ffn_gate_inp, exp_probs_b.bias,
    #     ffn_gate_exps, ffn_up_exps, ffn_down_exps,
    #     ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp
    #   Top-level: token_embd, output_norm, output
    #   MTP (layer n_layers): same as MoE + mtp_eh_proj, mtp_enorm, mtp_hnorm,
    #     mtp_shared_head_norm

    def add_tensor(name, shape, dtype):
        # Header-only: add tensor info with no data.  shape is in HF/logical
        # order (rows, cols); the writer stores it as ne[0]=last, ne[1]=prior,
        # matching ds4's tensor_expect_layout(d0=cols, d1=rows) convention.
        shape = list(shape)
        nelems = 1
        for d in shape:
            nelems *= d
        if dtype in (F32, F16):
            unit = {F32: 4, F16: 2}[dtype]
            tensor_dtype = np.float32 if dtype == F32 else np.float16
            writer.add_tensor_info(name, shape, tensor_dtype=tensor_dtype, raw_dtype=dtype, tensor_nbytes=nelems * unit)
        else:
            # For quantized, ncols = the rightmost (inner-most row-major) dim.
            ncols = shape[-1] if len(shape) >= 1 else 1
            block_elems, block_bytes = gguf.GGML_QUANT_SIZES[dtype]
            n_blocks = (ncols + block_elems - 1) // block_elems
            nrows = 1
            for d in shape[:-1]:
                nrows *= d
            nbytes = n_blocks * block_bytes * nrows
            writer.add_tensor_info(name, shape, tensor_dtype=np.float32, raw_dtype=dtype, tensor_nbytes=nbytes)

    # Top-level tensors
    add_tensor("token_embd.weight", [vocab_size, hidden], F16)
    add_tensor("output_norm.weight", [hidden], F32)
    add_tensor("output.weight", [vocab_size, hidden], Q8_0)

    # Emit only the 78 transformer layers.  MTP (layer 78) is a separate
    # model loaded via --mtp as mtp.0.* and is incompatible with --ssd-streaming.

    for il in range(n_layers):
        is_dense = il < first_k_dense

        # Layer norms (shared by all layers)
        add_tensor(f"blk.{il}.attn_norm.weight", [hidden], F32)
        add_tensor(f"blk.{il}.ffn_norm.weight", [hidden], F32)

        # Attention: standard MLA
        add_tensor(f"blk.{il}.attn_q_a.weight", [q_lora, hidden], Q8_0)
        add_tensor(f"blk.{il}.attn_q_a_norm.weight", [q_lora], F32)
        add_tensor(f"blk.{il}.attn_q_b.weight", [q_dim, q_lora], Q8_0)
        add_tensor(f"blk.{il}.attn_kv.weight", [kv_a_out, hidden], Q8_0)
        add_tensor(f"blk.{il}.attn_kv_a_norm.weight", [kv_lora], F32)
        add_tensor(f"blk.{il}.attn_kv_b.weight", [kv_b_out, kv_lora], Q8_0)
        add_tensor(f"blk.{il}.attn_output.weight", [hidden, o_in], Q8_0)

        # Indexer (DSA IndexShare) omitted for the first port — pure MLA is
        # correct at short/medium contexts; IndexShare math differs structurally
        # (GLM has no DeepSeek-style compressor) and is a follow-up.

        # FFN
        if is_dense:
            add_tensor(f"blk.{il}.ffn_gate.weight", [dense_inter, hidden], Q8_0)
            add_tensor(f"blk.{il}.ffn_up.weight", [dense_inter, hidden], Q8_0)
            add_tensor(f"blk.{il}.ffn_down.weight", [hidden, dense_inter], Q8_0)
        else:
            # MoE: router + routed experts + shared expert
            add_tensor(f"blk.{il}.ffn_gate_inp.weight", [n_experts, hidden], F16)
            add_tensor(f"blk.{il}.exp_probs_b.bias", [n_experts], F32)
            add_tensor(f"blk.{il}.ffn_gate_exps.weight", [n_experts, moe_inter, hidden], IQ2_XXS)
            add_tensor(f"blk.{il}.ffn_up_exps.weight", [n_experts, moe_inter, hidden], IQ2_XXS)
            add_tensor(f"blk.{il}.ffn_down_exps.weight", [n_experts, hidden, moe_inter], Q2_K)
            add_tensor(f"blk.{il}.ffn_gate_shexp.weight", [moe_inter, hidden], Q8_0)
            add_tensor(f"blk.{il}.ffn_up_shexp.weight", [moe_inter, hidden], Q8_0)
            add_tensor(f"blk.{il}.ffn_down_shexp.weight", [hidden, moe_inter], Q8_0)

    # Header-only write: no tensor data, just metadata + tensor info.
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    print(f"Wrote {OUT_PATH}")
    # Print tensor count
    print(f"Tensors: {sum(len(s) for s in writer.tensors)}")


if __name__ == "__main__":
    main()
