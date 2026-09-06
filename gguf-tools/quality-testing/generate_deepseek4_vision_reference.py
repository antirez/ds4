#!/usr/bin/env python3
"""Generate stage fixtures with the pinned official DeepSeek vision graph."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open


SOURCE_REVISION = "e46e16bf6035c6f317eb2ac7458eb0362926d402"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return parser.parse_args()


def load_tensor_map(hf_dir):
    with open(os.path.join(hf_dir, "model.safetensors.index.json"), encoding="utf-8") as fp:
        return json.load(fp)["weight_map"]


def load_module_state(hf_dir, weight_map, prefix):
    names = sorted(name for name in weight_map if name.startswith(prefix))
    shards = {}
    state = {}
    try:
        for name in names:
            shard_name = weight_map[name]
            shard = shards.get(shard_name)
            if shard is None:
                shard = safe_open(
                    os.path.join(hf_dir, shard_name),
                    framework="pt",
                    device="cpu",
                )
                shards[shard_name] = shard
            state[name.removeprefix(prefix)] = shard.get_tensor(name)
    finally:
        for shard in shards.values():
            shard.__exit__(None, None, None)
    return state


def dump_tensor(path, tensor):
    np.asarray(tensor.detach().float().cpu().numpy(), dtype="<f4").tofile(path)


def main():
    args = parse_args()
    hf_dir = os.path.abspath(args.hf)
    with open(os.path.join(hf_dir, "config.json"), encoding="utf-8") as fp:
        config = json.load(fp)
    if config.get("_commit_hash") not in (None, SOURCE_REVISION):
        raise SystemExit("config revision does not match the pinned checkpoint")

    inference_dir = os.path.join(hf_dir, "inference")
    sys.path.insert(0, inference_dir)
    from image_processor import build_image_block, load_image
    from vision import Aligner, ViT, apply_rotary, get_vision_cos_sin

    config["dim"] = config["hidden_size"]
    model_args = SimpleNamespace(**config)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(args.device)
    vit = ViT(model_args)
    aligner = Aligner(model_args)
    weight_map = load_tensor_map(hf_dir)
    vit.load_state_dict(load_module_state(hf_dir, weight_map, "vision."), strict=True)
    aligner.load_state_dict(load_module_state(hf_dir, weight_map, "aligner."), strict=True)
    vit.eval()
    aligner.eval()

    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(
        {"url": os.path.abspath(args.image)}, model_args)
    patches = patches.to(args.device)
    types, perm = build_image_block(n_llm_h, n_llm_w, 0)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        hidden = vit.patch_embed(patches)
        dump_tensor(str(prefix) + ".patch_embed.f32", hidden)
        cos, sin = get_vision_cos_sin(
            n_vit_h, n_vit_w, vit.rope_dim, vit.rope_theta)
        for layer, block in enumerate(vit.blocks):
            if layer == 0:
                norm1 = block.norm1(hidden)
                qkv = block.attn.wqkv(norm1)
                q, k, v = (
                    tensor.view(hidden.size(0), block.attn.n_heads,
                                block.attn.head_dim)
                    for tensor in qkv.chunk(3, dim=-1)
                )
                q = apply_rotary(q, cos, sin)
                k = apply_rotary(k, cos, sin)
                attention = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
                attention = attention.transpose(0, 1).reshape(hidden.size(0), -1)
                attention_residual = hidden + block.attn.wo(attention)
                norm2 = block.norm2(attention_residual)
                mlp_w1 = block.mlp.w1(norm2)
                gate, up = mlp_w1.chunk(2, dim=-1)
                mlp_mid = torch.nn.functional.silu(gate) * up
                hidden = attention_residual + block.mlp.w2(mlp_mid)
                for name, tensor in (
                    ("block0_norm1", norm1),
                    ("block0_qkv", qkv),
                    ("block0_q", q),
                    ("block0_k", k),
                    ("block0_v", v),
                    ("block0_attention", attention),
                    ("block0_attn_residual", attention_residual),
                    ("block0_norm2", norm2),
                    ("block0_mlp_w1", mlp_w1),
                    ("block0_mlp_mid", mlp_mid),
                ):
                    dump_tensor(str(prefix) + f".{name}.f32", tensor)
            else:
                hidden = block(hidden, cos, sin)
            if layer in (0, 15, 31):
                dump_tensor(str(prefix) + f".block{layer}.f32", hidden)
        hidden = vit.norm(hidden)
        dump_tensor(str(prefix) + ".vision_norm.f32", hidden)
        aligned = aligner(hidden, n_vit_h, n_vit_w)
        dump_tensor(str(prefix) + ".aligned.f32", aligned)

    np.asarray(patches.float().cpu().numpy(), dtype="<f4").tofile(
        str(prefix) + ".patches.f32")
    np.asarray(types.cpu().numpy(), dtype="<i8").tofile(
        str(prefix) + ".types.i64")
    np.asarray(perm.cpu().numpy(), dtype="<i8").tofile(
        str(prefix) + ".perm.i64")
    metadata = {
        "source_revision": SOURCE_REVISION,
        "image": os.path.basename(args.image),
        "n_vit_h": n_vit_h,
        "n_vit_w": n_vit_w,
        "n_llm_h": n_llm_h,
        "n_llm_w": n_llm_w,
        "patch_count": int(patches.shape[0]),
        "aligned_count": int(aligned.shape[0]),
        "block_token_count_at_start_0": int(types.numel()),
    }
    with open(str(prefix) + ".json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, sort_keys=True)
        fp.write("\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
