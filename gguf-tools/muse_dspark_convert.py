#!/usr/bin/env python3
"""Convert Meta's Muse Spark Open DSpark safetensors checkpoint to GGUF v3.

Metal draft matmuls can use F16 matrices or preserve the checkpoint's BF16
matrices. CPU-side lookup tensors stay BF16. The d2t vocabulary map is stored
as GGML I32 because every token id fits in signed 32 bits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path


GGUF_VERSION = 3
ALIGNMENT = 32
GGML_I32 = 26
GGML_F16 = 1
GGML_BF16 = 30
GGML_F32 = 0
GGUF_UINT32 = 4
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_FLOAT32 = 6


def pack_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def kv_u32(key: str, value: int) -> bytes:
    return pack_string(key) + struct.pack("<II", GGUF_UINT32, value)


def kv_f32(key: str, value: float) -> bytes:
    return pack_string(key) + struct.pack("<If", GGUF_FLOAT32, value)


def kv_string(key: str, value: str) -> bytes:
    return pack_string(key) + struct.pack("<I", GGUF_STRING) + pack_string(value)


def kv_u32_array(key: str, values: list[int]) -> bytes:
    return (pack_string(key) + struct.pack("<IIQ", GGUF_ARRAY, GGUF_UINT32, len(values))
            + struct.pack(f"<{len(values)}I", *values))


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def read_safetensors_header(fp):
    raw = fp.read(8)
    if len(raw) != 8:
        raise ValueError("source is too small to be a safetensors file")
    header_len = struct.unpack("<Q", raw)[0]
    if header_len == 0 or header_len > 128 * 1024 * 1024:
        raise ValueError(f"invalid safetensors header length {header_len}")
    header = json.loads(fp.read(header_len))
    data_start = 8 + header_len
    return header, data_start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="model.safetensors from Inferact/Muse-Spark-Open-DSpark")
    ap.add_argument("output", type=Path, help="output GGUF path")
    ap.add_argument("--force", action="store_true", help="replace an existing output after successful conversion")
    ap.add_argument("--bf16-matmuls", action="store_true",
                    help="preserve BF16 projection matrices instead of converting them to F16")
    args = ap.parse_args()

    if not args.source.is_file():
        ap.error(f"source does not exist: {args.source}")
    if args.output.exists() and not args.force:
        ap.error(f"output exists (pass --force to replace): {args.output}")
    tmp = args.output.with_name(args.output.name + ".tmp")
    if tmp.exists():
        ap.error(f"temporary output exists; inspect or remove it first: {tmp}")

    with args.source.open("rb") as src:
        header, data_start = read_safetensors_header(src)
        tensors = []
        ranges = []
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = info.get("dtype")
            shape = info.get("shape")
            offsets = info.get("data_offsets")
            if dtype not in ("BF16", "I64") or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"unsupported or malformed tensor {name}: {info}")
            count = 1
            for dim in shape:
                if not isinstance(dim, int) or dim < 0:
                    raise ValueError(f"invalid shape for {name}: {shape}")
                count *= dim
            start, end = offsets
            width = 2 if dtype == "BF16" else 8
            if end - start != count * width or start < 0 or end < start:
                raise ValueError(f"invalid byte range for {name}")
            metal_matmul = (name in ("fc.weight", "lm_head.weight") or
                            name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight",
                                           "o_gate_proj.weight", "o_proj.weight",
                                           "gate_proj.weight", "up_proj.weight", "down_proj.weight",
                                           "markov_w2.weight")))
            qtype = (GGML_F16 if dtype == "BF16" and metal_matmul and not args.bf16_matmuls else
                     GGML_BF16 if dtype == "BF16" else GGML_I32)
            nbytes = count * (2 if qtype in (GGML_F16, GGML_BF16) else 4)
            tensor = {"name": name, "shape": list(reversed(shape)), "qtype": qtype,
                      "src": data_start + start, "src_bytes": end - start,
                      "nbytes": nbytes, "dtype": dtype, "payload": None}
            tensors.append(tensor)
            ranges.append((start, end, name))

        ranges.sort()
        previous = 0
        for start, end, name in ranges:
            if start < previous:
                raise ValueError(f"overlapping safetensors payload at {name}")
            previous = end
        if previous != args.source.stat().st_size - data_start:
            raise ValueError("safetensors payload length does not match its header")
        if not any(t["name"] == "d2t" and t["dtype"] == "I64" for t in tensors):
            raise ValueError("checkpoint is missing its I64 d2t vocabulary map")

        def read_bf16_values(name: str) -> list[float]:
            info = header.get(name)
            if not info or info.get("dtype") != "BF16":
                raise ValueError(f"checkpoint is missing BF16 tensor {name}")
            start, end = info["data_offsets"]
            src.seek(data_start + start)
            raw = src.read(end - start)
            return [struct.unpack("<f", struct.pack("<I", struct.unpack_from("<H", raw, i)[0] << 16))[0]
                    for i in range(0, len(raw), 2)]

        def add_f32(name: str, values: list[float], shape: list[int]) -> None:
            payload = struct.pack(f"<{len(values)}f", *values)
            tensors.append({"name": name, "shape": list(reversed(shape)), "qtype": GGML_F32,
                            "src": 0, "src_bytes": 0, "nbytes": len(payload),
                            "dtype": "F32_SYNTH", "payload": payload})

        # Runtime kernels consume explicit affine gains and residual blend
        # coefficients. The checkpoint stores centered norm weights and one
        # learned residual gate, so materialize their inference forms here.
        for name in ("hidden_norm.weight", "norm.weight"):
            add_f32(name + ".runtime_scale", [1.0 + x for x in read_bf16_values(name)], [8192])
        add_f32("qk_norm_ones", [1.0] * 64, [64])
        temperature = 0.3
        for layer in range(5):
            for norm_name in ("self_attn.pre_qkv_proj_norm.weight", "pre_ffn_norm.weight", "post_ffn_norm.weight"):
                name = f"layers.{layer}.{norm_name}"
                add_f32(name.removesuffix(".weight") + ".runtime_scale",
                        [1.0 + x for x in read_bf16_values(name)], [8192])
            for gate_name in ("add_attn.gate", "add_ffn.gate"):
                name = f"layers.{layer}.{gate_name}"
                beta, alpha = [], []
                for x in read_bf16_values(name):
                    z = x / temperature
                    b = 1.0 / (1.0 + math.exp(-z))
                    a = max(1.0 / (1.0 + math.exp(z)) * (1.0 + b), 1e-3) ** 0.5
                    beta.append(b)
                    alpha.append(a)
                prefix = name.removesuffix(".gate")
                add_f32(prefix + ".resid_alpha", alpha, [8192])
                add_f32(prefix + ".resid_beta", beta, [8192])

        metadata = [
            kv_string("general.architecture", "muse-spark-dspark"),
            kv_string("general.name", "Muse Spark Open DSpark drafter"),
            kv_u32("general.alignment", ALIGNMENT),
            kv_u32("muse-spark-dspark.block_size", 7),
            kv_u32("muse-spark-dspark.markov_rank", 512),
            kv_u32("muse-spark-dspark.mask_token_id", 200018),
            kv_u32("muse-spark-dspark.vocab_size", 202048),
            kv_u32("muse-spark-dspark.embedding_length", 8192),
            kv_u32("muse-spark-dspark.context_length", 1048576),
            kv_u32("muse-spark-dspark.sliding_window", 2048),
            kv_f32("muse-spark-dspark.rms_norm_eps", 1e-5),
            kv_f32("muse-spark-dspark.rms_post_norm_eps", 1e-8),
            kv_f32("muse-spark-dspark.rope_theta", 500000.0),
            kv_f32("muse-spark-dspark.qk_scale_factor", 5.4730064863875),
            kv_f32("muse-spark-dspark.residual_gate_temperature", 0.3),
            kv_f32("muse-spark-dspark.output_multiplier", 0.1767766952966369),
            kv_f32("muse-spark-dspark.output_soft_cap_temp", 20.0),
            kv_u32_array("muse-spark-dspark.target_layer_ids", [13, 29, 45, 57, 61]),
        ]
        offset = 0
        for tensor in tensors:
            tensor["offset"] = offset
            offset += align(tensor["nbytes"])

        directory = bytearray()
        for tensor in tensors:
            shape = tensor["shape"]
            directory += pack_string(tensor["name"])
            directory += struct.pack("<I", len(shape))
            directory += struct.pack(f"<{len(shape)}Q", *shape)
            directory += struct.pack("<IQ", tensor["qtype"], tensor["offset"])
        header_bytes = (b"GGUF" + struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(metadata))
                        + b"".join(metadata) + directory)
        data_offset = align(len(header_bytes))
        args.output.parent.mkdir(parents=True, exist_ok=True)

        with tmp.open("xb") as out:
            out.write(header_bytes)
            out.write(b"\0" * (data_offset - len(header_bytes)))
            for index, tensor in enumerate(tensors, 1):
                expected = data_offset + tensor["offset"]
                if out.tell() != expected:
                    raise ValueError(f"internal offset mismatch at {tensor['name']}")
                if tensor["dtype"] == "F32_SYNTH":
                    out.write(tensor["payload"])
                elif tensor["dtype"] == "BF16":
                    remaining = tensor["src_bytes"]
                    pos = tensor["src"]
                    while remaining:
                        chunk = os.pread(src.fileno(), min(16 * 1024 * 1024, remaining), pos)
                        if not chunk:
                            raise ValueError(f"short read at {tensor['name']}")
                        if tensor["qtype"] == GGML_F16:
                            import numpy as np
                            bf16 = np.frombuffer(chunk, dtype="<u2")
                            f32 = (bf16.astype(np.uint32) << 16).view(np.float32)
                            if not np.isfinite(f32).all() or np.any(np.abs(f32) > 65504.0):
                                raise ValueError(f"{tensor['name']} cannot be represented as F16")
                            out.write(f32.astype(np.float16).tobytes())
                        else:
                            out.write(chunk)
                        pos += len(chunk)
                        remaining -= len(chunk)
                else:
                    raw = os.pread(src.fileno(), tensor["src_bytes"], tensor["src"])
                    values = struct.unpack(f"<{len(raw) // 8}q", raw)
                    if any(value < 0 or value >= 202048 for value in values):
                        raise ValueError("d2t contains an out-of-range token id")
                    out.write(struct.pack(f"<{len(values)}i", *values))
                padding = align(tensor["nbytes"]) - tensor["nbytes"]
                if padding:
                    out.write(b"\0" * padding)
                if index % 10 == 0 or index == len(tensors):
                    print(f"converted {index}/{len(tensors)} tensors", file=sys.stderr)
            out.flush()
            os.fsync(out.fileno())
    os.replace(tmp, args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"muse_dspark_convert: {exc}", file=sys.stderr)
        raise SystemExit(1)
