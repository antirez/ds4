#!/usr/bin/env python3
"""NumPy reference forward for MiniMax-M3.1-Flash on the official checkpoint.

There is no Transformers modeling file for this release; this mirrors vLLM's
minimax_m3 model (nvidia/model.py, common/indexer.py, common/sparse_attention.py)
and is the golden source for the ds4 graph:

  x = embed(tok)
  per layer:  h = attn(gemma_rms(x)); x += h;  x += ffn(gemma_rms(x))
  attn:       q/k/v/index_q/index_k projections, per-head gemma RMSNorm on all
              five, partial NeoX RoPE (first 64 of 128 dims) on q, k, index_q,
              index_k.  Index head h scores each 128-token block of index keys
              by max(q_h . k) over the block's causal tokens; KV head h attends
              only to the top 16 blocks, with the query's own block forced in.
  ffn:        layers 0-2 dense SwiGLU-OAI; others sigmoid-routed top-4 of 128
              experts (bias for choice only, renormalized, x2.0) + shared expert.
  SwiGLU-OAI: g = min(g, 7); u = clip(u, -7, 7); g * sigmoid(1.702 g) * (u + 1)

Runs on the CPU one token at a time; meant for a few dozen tokens.  Routed
experts are dequantized on every use, never cached: one f32 expert is 226 MB,
so caching every expert a run touches would not fit in memory.
--layers N stops after N layers and still applies the final norm and lm_head,
matching a minimax31_truncate.py --layers N GGUF.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_quantize as gq  # noqa: E402
import minimax31_quantize as mq  # noqa: E402

P = mq.PREFIX


class Model:
    def __init__(self, hf_dir, block_size=None, n_layers=60):
        self.db = gq.SourceDB(hf_dir, index_validator=mq.validate_index,
                              scale_validator=mq.no_scale_check, tensor_prefix="language_model.")
        self.q = mq.Quantizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libds4quants.dylib"))
        cfg = mq.read_json(hf_dir, "config.json")["text_config"]
        sp = cfg["sparse_attention_config"]
        self.eps = cfg["rms_norm_eps"]
        self.n_head, self.n_kv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.n_rot, self.theta = cfg["rotary_dim"], float(cfg["rope_theta"])
        self.block = block_size or sp["sparse_block_size"]
        self.topk = sp["sparse_topk_blocks"]
        self.local = sp["sparse_local_block"]
        self.init = sp["sparse_init_block"]
        self.alpha, self.limit = cfg["swiglu_alpha"], cfg["swiglu_limit"]
        self.scale = cfg["routed_scaling_factor"]
        self.n_used = cfg["num_experts_per_tok"]
        self.n_layers = n_layers
        self.cache = {}

    def w(self, name):
        if name not in self.cache:
            self.cache[name] = self.q.to_f32(self.db, name)
        return self.cache[name]

    def norm_w(self, name):
        return self.w(name) + 1.0  # gemma norm

    def rms(self, x, weight):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps) * weight

    def rope(self, x, pos):  # x [..., hd]; NeoX pairs (i, i + n_rot/2) over the first n_rot dims
        half = self.n_rot // 2
        # float32 inverse frequencies and angle, as vLLM's cos/sin cache
        freq = (1.0 / self.theta ** (np.arange(0, self.n_rot, 2, dtype=np.float32) / self.n_rot)).astype(np.float32)
        ang = (np.float32(pos) * freq).astype(np.float64)
        c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        x = x.copy()
        a, b = x[..., :half].copy(), x[..., half:self.n_rot].copy()
        x[..., :half] = a * c - b * s
        x[..., half:self.n_rot] = a * s + b * c
        return x

    def act(self, g, u):
        g = np.minimum(g, self.limit)
        u = np.clip(u, -self.limit, self.limit)
        return g / (1.0 + np.exp(-self.alpha * g)) * (u + 1.0)

    def mlp(self, x, prefix):
        g = self.w(f"{prefix}.gate_proj.weight") @ x
        u = self.w(f"{prefix}.up_proj.weight") @ x
        return self.w(f"{prefix}.down_proj.weight") @ self.act(g, u)

    def expert(self, layer, e):
        b = f"{P}.layers.{layer}.block_sparse_moe.experts.{e}"
        return tuple(self.q.to_f32(self.db, f"{b}.{p}.weight_packed") for p in ("w1", "w3", "w2"))

    def moe(self, x, layer):
        m = f"{P}.layers.{layer}.block_sparse_moe"
        logits = self.w(f"{m}.gate.weight") @ x
        scores = 1.0 / (1.0 + np.exp(-logits))
        choice = np.argsort(-(scores + self.w(f"{m}.e_score_correction_bias")), kind="stable")[: self.n_used]
        weights = scores[choice] / scores[choice].sum() * self.scale
        out = self.mlp(x, f"{m}.shared_experts")
        for e, wt in zip(choice, weights, strict=True):
            w1, w3, w2 = self.expert(layer, int(e))
            out = out + wt * (w2 @ self.act(w1 @ x, w3 @ x))
        return out, choice

    def select(self, iq, ik_cache, pos):
        """Top blocks per index head for the query at pos (block ids, ascending)."""
        n_blocks = pos // self.block + 1
        scores = iq @ ik_cache[: pos + 1].T  # [Hi, pos+1]
        pad = n_blocks * self.block - (pos + 1)
        scores = np.pad(scores, ((0, 0), (0, pad)), constant_values=-np.inf)
        block_max = scores.reshape(scores.shape[0], n_blocks, self.block).max(axis=2)
        forced = np.zeros(n_blocks, dtype=bool)
        forced[: self.init] = True
        forced[max(0, n_blocks - self.local):] = True
        out = []
        for h in range(block_max.shape[0]):
            s = np.where(forced, np.inf, block_max[h])
            out.append(np.sort(np.argsort(-s, kind="stable")[: self.topk]))
        return out

    def attention(self, x, layer, pos, kv):
        a = f"{P}.layers.{layer}.self_attn"
        H, Hkv, D = self.n_head, self.n_kv, self.hd
        q = self.rms((self.w(f"{a}.q_proj.weight") @ x).reshape(H, D), self.norm_w(f"{a}.q_norm.weight"))
        k = self.rms((self.w(f"{a}.k_proj.weight") @ x).reshape(Hkv, D), self.norm_w(f"{a}.k_norm.weight"))
        v = (self.w(f"{a}.v_proj.weight") @ x).reshape(Hkv, D)
        iq = self.rms((self.w(f"{a}.index_q_proj.weight") @ x).reshape(Hkv, -1), self.norm_w(f"{a}.index_q_norm.weight"))
        ik = self.rms(self.w(f"{a}.index_k_proj.weight") @ x, self.norm_w(f"{a}.index_k_norm.weight"))
        q, k, iq, ik = self.rope(q, pos), self.rope(k, pos), self.rope(iq, pos), self.rope(ik, pos)
        kv["k"][pos], kv["v"][pos], kv["ik"][pos] = k, v, ik
        blocks = self.select(iq, kv["ik"], pos)
        out = np.empty((H, D), dtype=np.float32)
        group = H // Hkv
        for h in range(Hkv):
            idx = np.concatenate([np.arange(b * self.block, min((b + 1) * self.block, pos + 1)) for b in blocks[h]])
            keys, vals = kv["k"][idx, h], kv["v"][idx, h]
            s = q[h * group:(h + 1) * group] @ keys.T / np.sqrt(D)
            p = np.exp(s - s.max(axis=1, keepdims=True))
            out[h * group:(h + 1) * group] = (p / p.sum(axis=1, keepdims=True)) @ vals
        return self.w(f"{a}.o_proj.weight") @ out.reshape(-1)

    def forward(self, tokens, n_ctx):
        kv = [dict(k=np.zeros((n_ctx, self.n_kv, self.hd), np.float32),
                   v=np.zeros((n_ctx, self.n_kv, self.hd), np.float32),
                   ik=np.zeros((n_ctx, self.hd), np.float32)) for _ in range(self.n_layers)]
        embed = self.w(f"{P}.embed_tokens.weight")
        for pos, tok in enumerate(tokens):
            x = embed[tok].astype(np.float32)
            for layer in range(self.n_layers):
                p = f"{P}.layers.{layer}"
                x = x + self.attention(self.rms(x, self.norm_w(f"{p}.input_layernorm.weight")), layer, pos, kv[layer])
                h = self.rms(x, self.norm_w(f"{p}.post_attention_layernorm.weight"))
                x = x + (self.mlp(h, f"{p}.mlp") if layer < 3 else self.moe(h, layer)[0])
            yield self.w("language_model.lm_head.weight") @ self.rms(x, self.norm_w(f"{P}.norm.weight"))


def main():
    ap = argparse.ArgumentParser(description="MiniMax-M3.1-Flash NumPy reference forward")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--tokens", required=True, help="comma-separated prompt token ids")
    ap.add_argument("--generate", type=int, default=4, help="greedy tokens after the prompt")
    ap.add_argument("--block-size", type=int, help="override the sparse block size to exercise selection")
    ap.add_argument("--out", help="write per-position top-20 logits as JSON")
    ap.add_argument("--layers", type=int, default=60, help="run only the first N layers (truncated-GGUF checks)")
    args = ap.parse_args()
    if args.layers < 1 or args.layers > 60:
        ap.error("--layers must be between 1 and 60")
    tokens = [int(t) for t in args.tokens.split(",")]
    model = Model(args.hf, args.block_size, args.layers)
    n_prompt = len(tokens)
    rows = []
    stream = model.forward(tokens, n_prompt + args.generate)
    for i in range(n_prompt + args.generate):
        logits = next(stream)
        top = np.argsort(-logits)[:20]
        rows.append({"pos": i, "top": [[int(t), float(logits[t])] for t in top]})
        print(f"pos {i}: top5 {[int(t) for t in top[:5]]}", file=sys.stderr, flush=True)
        if i + 1 >= len(tokens) and len(tokens) < n_prompt + args.generate:
            tokens.append(int(top[0]))
    result = json.dumps({"tokens": tokens, "n_prompt": n_prompt, "rows": rows})
    if not args.out:
        print(result)
        return
    try:
        with open(args.out, "w") as fp:
            fp.write(result)
    except OSError as error:
        sys.exit(f"minimax31-reference: cannot write {args.out}: {error}")


if __name__ == "__main__":
    main()
