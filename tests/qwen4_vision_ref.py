#!/usr/bin/env python3
"""Compare Metal vision execution with HF using identical GGUF weights.

Also report GGUF-versus-original HF quantization quality separately. The default
0.99 threshold gates implementation parity; quality retains the same threshold
but does not fail execution unless --require-quality is supplied.
Requires numpy, torch, torchvision, pillow, safetensors, transformers with
qwen4_exp support, and gguf. Repeat --image to evaluate a suite in one process.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hf_model(snapshot, mmproj=None):
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

    cfg = AutoConfig.from_pretrained(snapshot).vision_config
    cfg._attn_implementation = "eager"
    model = Qwen4ExpVisionModel(cfg).float().eval()
    if mmproj is not None:
        state = gguf_state(model, mmproj)
    else:
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
    return model


def hf_embeddings(snapshot, image_path, min_tokens, max_tokens, model=None):
    import torch
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil as Qwen2VLImageProcessor

    if model is None:
        model = load_hf_model(snapshot)
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


def gguf_state(model, path):
    """Load every GGUF tensor into the HF tower, including split temporal Conv3D."""
    import torch
    from gguf import GGUFReader, dequantize

    reader = GGUFReader(path)
    tensors = {t.name: t for t in reader.tensors}
    used = set()

    def tensor(name):
        used.add(name)
        t = tensors[name]
        shape = tuple(reversed(t.shape.tolist()))
        return torch.from_numpy(dequantize(t.data, t.tensor_type).reshape(shape).copy()).float()

    state = {}
    for key in model.state_dict():
        if key == "patch_embed.proj.weight":
            state[key] = torch.stack([tensor("v.patch_embd.weight"),
                                      tensor("v.patch_embd.weight.1")], dim=2)
            continue
        if key.startswith("blocks."):
            _, layer, rest = key.split(".", 2)
            for old, new in [("attn.proj.", "attn_out."), ("attn.qkv.", "attn_qkv."),
                             ("mlp.linear_fc1.", "ffn_up."), ("mlp.linear_fc2.", "ffn_down."),
                             ("norm1.", "ln1."), ("norm2.", "ln2.")]:
                if rest.startswith(old):
                    name = f"v.blk.{layer}." + rest.replace(old, new, 1)
                    break
            else:
                raise ValueError(f"unmapped HF vision tensor: {key}")
        elif key.startswith("merger."):
            name = key.replace("merger.linear_fc1.", "mm.0.").replace(
                "merger.linear_fc2.", "mm.2.").replace("merger.norm.", "v.post_ln.")
        elif key == "patch_embed.proj.bias":
            name = "v.patch_embd.bias"
        elif key == "pos_embed.weight":
            name = "v.position_embd.weight"
        else:
            raise ValueError(f"unmapped HF vision tensor: {key}")
        state[key] = tensor(name)
    if used != set(tensors):
        raise ValueError(f"unused GGUF tensors: {sorted(set(tensors) - used)}")
    return state


def compare(actual, reference, actual_grid, reference_grid, threshold):
    if actual.shape != reference.shape or actual_grid != reference_grid:
        return {"pass": False, "error": "token layout mismatch"}
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return {"pass": False, "error": "non-finite embeddings"}
    # Use float64 for the metric so numerical noise does not yield cosine > 1.
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    diff = np.abs(a - b)
    return {"pass": bool(cos.min() >= threshold), "min_cos": float(cos.min()),
            "mean_cos": float(cos.mean()), "worst_token": int(cos.argmin()),
            "max_abs": float(diff.max()), "mean_abs": float(diff.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--image", required=True, action="append")
    ap.add_argument("--binary", default=os.path.join(os.path.dirname(__file__), "test_qwen4_vision"))
    ap.add_argument("--min-tokens", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", help="DS4 embedding dump path (single image only)")
    ap.add_argument("--min-cos", type=float, default=0.99)
    ap.add_argument("--require-quality", action="store_true",
                    help="also fail if GGUF-versus-original quality is below --min-cos")
    ap.add_argument("--json-report")
    args = ap.parse_args()
    if args.out and len(args.image) != 1:
        ap.error("--out requires exactly one image")
    if not 0 < args.min_cos <= 1 or not 0 < args.min_tokens <= args.max_tokens:
        ap.error("invalid cosine threshold or token limits")
    original_model = load_hf_model(args.snapshot)
    gguf_model = load_hf_model(args.snapshot, args.mmproj)
    results = []
    with tempfile.TemporaryDirectory(prefix="qwen4-vision-") as directory:
        for i, image_path in enumerate(args.image):
            out = args.out or os.path.join(directory, f"{i}.bin")
            actual, grid = ds4_embeddings(args.binary, args.mmproj, image_path, out,
                                          args.min_tokens, args.max_tokens)
            quant, quant_grid = hf_embeddings(args.snapshot, image_path, args.min_tokens,
                                              args.max_tokens, gguf_model)
            original, original_grid = hf_embeddings(args.snapshot, image_path, args.min_tokens,
                                                    args.max_tokens, original_model)
            result = {"image": image_path, "image_sha256": sha256_file(image_path),
                      "grid": grid, "shape": list(actual.shape),
                      "implementation": compare(actual, quant, grid, quant_grid, args.min_cos),
                      "quantization_quality": compare(quant, original, quant_grid, original_grid, args.min_cos),
                      "metal_vs_original": compare(actual, original, grid, original_grid, args.min_cos)}
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {"snapshot": args.snapshot, "mmproj": args.mmproj,
              "mmproj_sha256": sha256_file(args.mmproj),
              "versions": {name: importlib.metadata.version(name)
                           for name in ("numpy", "torch", "transformers", "gguf", "pillow")},
              "min_cos": args.min_cos, "min_tokens": args.min_tokens,
              "max_tokens": args.max_tokens, "require_quality": args.require_quality,
              "results": results}
    if args.json_report:
        with open(args.json_report, "w") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
    if any(not r["implementation"]["pass"] or
           (args.require_quality and not r["quantization_quality"]["pass"]) for r in results):
        sys.exit("FAIL")
    print("ok: implementation parity; quantization quality reported separately")


if __name__ == "__main__":
    main()
