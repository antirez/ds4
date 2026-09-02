#!/usr/bin/env python3
"""Compare DS4's Qwen3.8 vision tower against the HF implementation.

1. Runs the HF Qwen4ExpVisionModel (weights from shard 1 of the checkpoint,
   fp32 on CPU) on IMAGE through the Qwen2-VL image processor.
2. Runs tests/test_qwen4_vision with the mmproj GGUF on the same image.
3. Reports per-token cosine similarity and max abs difference.

Usage: python tests/qwen4_vision_ref.py --snapshot DIR --mmproj FILE --image FILE [--max-tokens N]
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np


def hf_embeddings(snapshot, image_path, min_tokens, max_tokens):
    import torch
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil as Qwen2VLImageProcessor
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

    cfg = AutoConfig.from_pretrained(snapshot).vision_config
    cfg._attn_implementation = "eager"
    model = Qwen4ExpVisionModel(cfg).float().eval()
    weight_map = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    shards = sorted({shard for key, shard in weight_map.items() if key.startswith("model.visual.")})
    state = {}
    for shard in shards:
        with safe_open(os.path.join(snapshot, shard), "pt") as f:
            for key in f.keys():
                if key.startswith("model.visual."):
                    state[key[len("model.visual."):]] = f.get_tensor(key).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if "inv_freq" not in m]
    if missing or unexpected:
        sys.exit(f"vision weights mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    ip = Qwen2VLImageProcessor(min_pixels=min_tokens * 32 * 32, max_pixels=max_tokens * 32 * 32,
                               patch_size=16, merge_size=2, temporal_patch_size=2,
                               image_mean=[0.5] * 3, image_std=[0.5] * 3)
    img = Image.open(image_path).convert("RGB")
    out = ip(images=img, return_tensors="pt")
    with torch.no_grad():
        res = model(out["pixel_values"].float(), grid_thw=out["image_grid_thw"])
    emb = res.pooler_output if hasattr(res, "pooler_output") else res[1]
    if isinstance(emb, (list, tuple)):
        emb = emb[0]
    thw = out["image_grid_thw"][0].tolist()
    return emb.numpy(), (thw[1] // 2, thw[2] // 2)


def ds4_embeddings(binary, mmproj, image_path, out_path, min_tokens, max_tokens):
    subprocess.run([binary, mmproj, image_path, out_path, str(min_tokens), str(max_tokens)], check=True)
    with open(out_path, "rb") as f:
        n, dim, gh, gw = struct.unpack("<4I", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(n, dim)
    return data, (gh, gw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--binary", default=os.path.join(os.path.dirname(__file__), "test_qwen4_vision"))
    ap.add_argument("--min-tokens", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default=None, help="DS4 embedding dump path (default: a temp file)")
    ap.add_argument("--min-cos", type=float, default=0.99)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(tempfile.mkdtemp(prefix="qwen4-vision-"), "embeddings.bin")

    ds4, ds4_grid = ds4_embeddings(args.binary, args.mmproj, args.image, args.out, args.min_tokens, args.max_tokens)
    ref, ref_grid = hf_embeddings(args.snapshot, args.image, args.min_tokens, args.max_tokens)
    print(f"ds4 grid {ds4_grid} tokens {ds4.shape}; hf grid {ref_grid} tokens {ref.shape}")
    if ds4.shape != ref.shape:
        sys.exit("token layout mismatch")
    cos = np.sum(ds4 * ref, axis=1) / (np.linalg.norm(ds4, axis=1) * np.linalg.norm(ref, axis=1) + 1e-12)
    diff = np.abs(ds4 - ref)
    rel = diff.max() / (np.abs(ref).max() + 1e-12)
    print(f"cosine min {cos.min():.6f} mean {cos.mean():.6f}; max|d| {diff.max():.4f} "
          f"(ref max|x| {np.abs(ref).max():.3f}, rel {rel:.4f}); mean|d| {diff.mean():.5f}")
    if not np.isfinite(ds4).all() or cos.min() < args.min_cos:
        sys.exit("FAIL")
    print("ok")


if __name__ == "__main__":
    main()
