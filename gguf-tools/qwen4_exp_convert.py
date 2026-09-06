#!/usr/bin/env python3
"""Convert a Qwen3.8-Flash-Next (qwen4_exp) HF checkpoint into a DS4 GGUF.

Wraps llama.cpp's conversion package, so the file keeps the upstream
`qwen4exp` schema, and adds the MTP block as the trailing blk.<n_layer>
(nextn.* tensors plus `qwen4exp.nextn_predict_layers`); --no-mtp leaves it
out for stock llama.cpp.  Norms, conv kernels, ssm_a, dt biases and the
routers stay F32; the other tensor types follow the options below.

Needs a llama.cpp master checkout (--llama-cpp or $LLAMA_CPP) and a Python
with torch, safetensors and transformers.  The output is written to
<out>.incomplete and renamed when complete.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="HF checkpoint directory")
    ap.add_argument("--out", required=True, help="output GGUF path")
    ap.add_argument("--outtype", default="q8_0", choices=["q8_0", "f32"], help="type of the dense projections")
    ap.add_argument("--experts", default=None, choices=["q8_0", "mxfp4", "q4_k", "f32"],
                    help="routed expert type (default: follows --outtype)")
    ap.add_argument("--experts-down", default=None, choices=["q8_0", "mxfp4", "f32"],
                    help="down_exps type when --experts needs 256-wide rows (default: q8_0, or mxfp4 with --experts mxfp4)")
    ap.add_argument("--ngram", default=None, choices=["q8_0", "mxfp4", "q4_0", "f32"],
                    help="PLE n-gram table type (default: follows --outtype)")
    ap.add_argument("--hc-type", default=None, choices=["f16", "f32", "q8_0"],
                    help="hyper-connection down/up/inject type (default: f16, or f32 with --outtype f32)")
    ap.add_argument("--indexer", default=None, choices=["bf16", "f16", "q8_0", "f32"],
                    help="QSA indexer q/k projection type (default: bf16 as released, f32 with --outtype f32)")
    ap.add_argument("--no-mtp", action="store_true", help="drop the MTP block (stock llama.cpp compatible)")
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP", os.path.expanduser("~/repo/llama.cpp")),
                    help="llama.cpp checkout with conversion/qwen4exp.py (default: $LLAMA_CPP or ~/repo/llama.cpp)")
    ap.add_argument("--dry-run", action="store_true", help="plan the tensor types without writing")
    ap.add_argument("--verbose", action="store_true", help="pass --verbose to the llama.cpp converter")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.llama_cpp, "conversion", "qwen4exp.py")):
        sys.exit(f"{args.llama_cpp}/conversion/qwen4exp.py not found: point --llama-cpp at a llama.cpp master checkout")
    sys.path.insert(0, os.path.join(args.llama_cpp, "gguf-py"))
    sys.path.insert(0, args.llama_cpp)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    import torch
    import gguf
    from conversion.qwen4exp import Qwen4ExpTextModel

    Q = gguf.GGMLQuantizationType
    T = gguf.MODEL_TENSOR
    type_of = {"q8_0": Q.Q8_0, "mxfp4": Q.MXFP4, "q4_k": Q.Q4_K, "q4_0": Q.Q4_0,
               "f16": Q.F16, "bf16": Q.BF16, "f32": Q.F32}
    all_f32 = args.outtype == "f32"
    experts_t = type_of[args.experts or args.outtype]
    experts_down_t = type_of[args.experts_down or ("mxfp4" if args.experts == "mxfp4" else args.outtype)]
    ngram_t = type_of[args.ngram or args.outtype]
    hc_t = type_of[args.hc_type or ("f32" if all_f32 else "f16")]
    indexer_t = type_of[args.indexer or ("f32" if all_f32 else "bf16")]
    keep_mtp = not args.no_mtp

    # the upstream arch table lacks the nextn tensors; the mixin renames mtp.* onto them
    arch_list = gguf.MODEL_TENSORS[gguf.MODEL_ARCH.QWEN4EXP]
    for key in (T.NEXTN_EH_PROJ, T.NEXTN_ENORM, T.NEXTN_HNORM):
        if key not in arch_list:
            arch_list.append(key)

    class DS4Qwen4ExpModel(Qwen4ExpTextModel):
        model_arch = gguf.MODEL_ARCH.QWEN4EXP
        no_mtp = not keep_mtp
        supports_mtp_export = False
        _MIXER = "mtp.hyper_connection_mixer."

        @classmethod
        def filter_tensors(cls, item):
            name, gen = item
            if name.startswith("model." + cls._MIXER):
                name = name.replace("model.", "", 1)
            if name.startswith(cls._MIXER):
                if cls.no_mtp:
                    return None
                assert cls._original_block_count is not None
                return f"model.layers.{cls._original_block_count}.mtp_mixer.{name[len(cls._MIXER):]}", gen
            return super().filter_tensors((name, gen))

        def index_tensors(self, remote_hf_model_id=None):
            tensors = super().index_tensors(remote_hf_model_id=remote_hf_model_id)
            emb = tensors.pop("mtp.fc_embedding.weight", None)
            hid = tensors.pop("mtp.fc_hidden.weight", None)
            if keep_mtp and (emb is None) != (hid is None):
                raise SystemExit("mtp.fc_embedding.weight and mtp.fc_hidden.weight must both be present")
            if keep_mtp and emb is not None:
                assert self._original_block_count is not None
                # fc_embedding first: eh_proj multiplies concat(e, h)
                tensors[f"model.layers.{self._original_block_count}.eh_proj.weight"] = \
                    lambda: torch.cat([emb(), hid()], dim=1)
            return tensors

        def modify_tensors(self, data_torch, name, bid):
            if ".mtp_mixer." in name:
                assert bid is not None
                part = name.rsplit(".mtp_mixer.", 1)[1]
                out = {"hc_norm.weight": ("hc_head_norm", data_torch + 1),
                       "input_mix_weight_down.weight": ("hc_head_down", data_torch),
                       "input_mix_weight_up.weight": ("hc_head_up", data_torch)}[part]
                return [(f"blk.{bid}.nextn.{out[0]}.weight", out[1])]
            return super().modify_tensors(data_torch, name, bid)

        def set_gguf_parameters(self):
            # the MTP block keeps its QSA indexer, so it gets the same ratio as the trunk
            writer = self.gguf_writer
            orig = writer.add_attention_compress_ratios
            ratio = self.hparams["indexer_compress_ratio"]

            def padded(values):
                orig(list(values) + [ratio] * (self.block_count - len(values)))
            writer.add_attention_compress_ratios = padded
            try:
                super().set_gguf_parameters()
            finally:
                writer.add_attention_compress_ratios = orig

        def tensor_force_quant(self, name, new_name, bid, n_dims):
            if n_dims <= 1 or new_name.endswith(("conv1d.weight", "ssm_a", "_norm.weight")):
                return Q.F32
            if new_name == "per_layer_token_embd.weight":
                return ngram_t
            for key in (T.FFN_GATE_EXP, T.FFN_UP_EXP):
                if self.match_model_tensor_name(new_name, key, bid):
                    return experts_t
            if self.match_model_tensor_name(new_name, T.FFN_DOWN_EXP, bid):
                # down_exps rows are moe_intermediate_size wide (640): no 256-block types
                if args.experts_down or (experts_t == Q.Q4_K and self.hparams["moe_intermediate_size"] % 256 != 0):
                    return experts_down_t
                return experts_t
            if ".nextn.hc_head_" in new_name or "hc_attn_" in new_name or "hc_ffn_" in new_name \
                    or new_name.startswith("output_hc_"):
                return hc_t
            if new_name.endswith((".indexer.q_proj.weight", ".indexer.k_proj.weight")):
                return indexer_t
            if self.match_model_tensor_name(new_name, T.FFN_GATE_INP, bid):
                return Q.F32
            if all_f32:
                return Q.F32
            return super().tensor_force_quant(name, new_name, bid, n_dims)

    ftype = gguf.LlamaFileType.ALL_F32 if all_f32 else gguf.LlamaFileType.MOSTLY_Q8_0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".incomplete")

    model = DS4Qwen4ExpModel(Path(args.src), ftype, tmp, use_temp_file=False, dry_run=args.dry_run)
    model.write()
    if args.dry_run:
        return
    os.replace(tmp, out)
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
