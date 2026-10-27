#!/usr/bin/env python3
"""First-token source oracle with one layer and selected experts in CPU memory.

No full model is constructed or mmaped. The pinned release's Blocks and CPU
kernel translations do the arithmetic; Engram rows and head rows use pread.
This is a bounded diagnostic, never a production inference backend.
"""

import argparse
import dataclasses
import gc
import json
from pathlib import Path
import resource
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I8": torch.float4_e2m1fn_x2,
}


def tensor_bytes(data, dtype, shape):
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(dtype).reshape(shape)


def first_token(ref, args, db, tokenizer, token, trace):
    """Use unchanged released Blocks; only parameter storage is made lazy."""
    original_init = ref.Expert.__init__
    original_forward = ref.Expert.forward
    original_engram_init = ref.ParallelEngramEmbedding.__init__

    def expert_init(self, dim, inter_dim, dtype=None, swiglu_limit=0):
        # Unselected experts must not allocate any weight storage.
        if dtype == torch.float4_e2m1fn_x2:
            with torch.device("meta"):
                original_init(self, dim, inter_dim, dtype, swiglu_limit)
        else:
            original_init(self, dim, inter_dim, dtype, swiglu_limit)

    def expert_forward(self, x, weights=None):
        if self.w1.weight.is_meta:
            load(self, self.source_prefix, lazy=False)
        return original_forward(self, x, weights)

    def engram_init(self, *a, **kw):
        with torch.device("meta"):
            original_engram_init(self, *a, **kw)

    def rows(name, start, count):
        info = db.info(name)
        cols = info["shape"][1]
        dtype = DTYPES[info["dtype"]]
        size = torch.empty((), dtype=dtype).element_size()
        data = b"".join(db.iter_read(name, start * cols * size, count * cols * size))
        return tensor_bytes(data, dtype, (count, cols))

    def engram_forward(self, ids):
        name = self.source_prefix + ".weight"
        flat = ids.flatten().tolist()
        codes = torch.cat([rows(name, i, 1) for i in flat]).float().reshape(-1, 8, 32)
        scales = torch.cat([rows(self.source_prefix + ".scale", i, 1) for i in flat]).float()
        return (codes * scales[..., None]).flatten(-2).to(torch.bfloat16).reshape(*ids.shape, 256)

    def load(module, prefix, lazy=True):
        for name, parameter in list(module.named_parameters()):
            source = prefix + "." + name
            if lazy and (".ffn.experts." in source or ".engram.embed." in source):
                continue
            info = db.info(source)
            values = tensor_bytes(db.read(source), DTYPES[info["dtype"]], info["shape"])
            if info["dtype"] == "F8_E4M3" and parameter.dtype != values.dtype:
                # Released wo_a is loaded as BF16 in the reference. FP8 * a
                # power of two has at most three significand bits.
                scale = tensor_bytes(
                    db.read(source.removesuffix(".weight") + ".scale"),
                    torch.float8_e8m0fnu,
                    db.info(source.removesuffix(".weight") + ".scale")["shape"],
                )
                values = values.float() * scale.float().repeat_interleave(32, 0).repeat_interleave(
                    32, 1
                )
            values = values.to(parameter.dtype)
            parent, _, key = name.rpartition(".")
            setattr(
                module.get_submodule(parent) if parent else module,
                key,
                torch.nn.Parameter(values, requires_grad=False),
            )
        for name, child in module.named_modules():
            if isinstance(child, ref.Expert) and child.w1.weight.is_meta:
                child.source_prefix = prefix + ("." + name if name else "")
            if isinstance(child, ref.ParallelEngramEmbedding):
                child.source_prefix = prefix + "." + name
            if isinstance(child, ref.Linear) and child.scale is not None:
                child.weight.scale = child.scale

    ref.Expert.__init__ = expert_init
    ref.Expert.forward = expert_forward
    ref.ParallelEngramEmbedding.__init__ = engram_init
    ref.ParallelEngramEmbedding.forward = engram_forward
    ref.world_size, ref.rank = 1, 0
    ref.default_dtype = torch.float8_e4m3fn
    ref.shared_attn = ref.SharedAttentionRuntime()
    torch.set_default_dtype(torch.bfloat16)
    layout = ref.EngramLayout.from_args(args)
    hashes = ref.NgramHashState(args, layout, tokenizer)(torch.tensor([[token]]), 0)
    h = rows("embed.weight", token, 1).reshape(1, 1, -1).unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
    pre = ref.make_identity_pre_mix(h, args.hc_mult)
    with torch.inference_mode():
        for layer in range(args.n_layers):
            block = ref.Block(layer, args, layout).eval()
            load(block, f"layers.{layer}")
            if block.engram is not None:
                h = block.engram(h, hashes[:, :, block.engram.layer_hash_index, :])
            h, pre = block(h, 0, pre, None)
            trace[f"layer.{layer}"] = h.float().numpy().copy()
            trace[f"pre.{layer}"] = pre.float().numpy().copy()
            print(
                f"source reference layer {layer} complete; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1<<30):.2f} GiB",
                flush=True,
            )
            if layer + 1 == args.n_layers:
                h = block.hc_pre(h, pre)
            del block
            gc.collect()
        norm = ref.RMSNorm(args.dim, args.norm_eps)
        load(norm, "norm")
        h = norm(h)[:, -1].float()
        output = []
        for start in range(0, args.vocab_size, 32):
            count = min(32, args.vocab_size - start)
            output.append(F.linear(h, rows("head.weight", start, count).float()))
        return torch.cat(output, -1).numpy().copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--token-id", type=int, default=67, help="Single source token; 0 is BOS, 67 is raw a")
    a = p.parse_args()
    sys.path[:0] = [str(a.hf / "inference"), str(ROOT / "gguf-tools")]
    import deepseek41_native_reference

    sys.modules["kernel"] = deepseek41_native_reference
    image = types.ModuleType("image_processor")
    for n in ("IMAGE", "IMAGE_END", "IMAGE_NEW_LINE", "IMAGE_START"):
        setattr(image, n, 0)
    sys.modules["image_processor"] = image
    vision = types.ModuleType("vision")
    vision.Aligner = vision.ViT = None
    sys.modules["vision"] = vision
    import model as ref
    from tokenizers import Tokenizer
    from deepseek41_quantize import SourceDB, validate_scales

    config = json.loads((a.hf / "inference/config.json").read_text())
    config.update(
        max_batch_size=1,
        max_seq_len=16,
        temperature=0,
        n_mtp_layers=0,
        dspark_block_size=0,
        vision_n_layers=0,
    )
    args = ref.ModelArgs(**config)
    tok = Tokenizer.from_file(str(a.hf / "tokenizer.json"))

    class TokenizerWrapper:
        backend_tokenizer = tok

        def __len__(self):
            return tok.get_vocab_size()

    ids = [a.token_id]
    if not 0 <= a.token_id < args.vocab_size:
        p.error("--token-id is outside the source vocabulary")
    torch.set_num_threads(4)
    db = SourceDB(str(a.hf), index_validator=lambda _: None, scale_validator=validate_scales)
    started = time.monotonic()
    try:
        trace = {}
        y = first_token(ref, args, db, TokenizerWrapper(), ids[0], trace)
        assert y.shape == (1, 129280) and np.all(np.isfinite(y))
        y.astype("<f4").tofile(a.out)
        np.savez(str(a.out) + ".trace.npz", **trace)
        Path(str(a.out) + ".json").write_text(
            json.dumps(
                {
                    "token_ids": ids,
                    "argmax": int(y.argmax()),
                    "seconds": time.monotonic() - started,
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "args": dataclasses.asdict(args),
                },
                indent=2,
            )
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
