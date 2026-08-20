#!/usr/bin/env python3
"""Pack EigenLabs/Qwen3.8-27B-MTP-bf16 (or any 15-tensor official head)
into a ds4 sidecar GGUF with mtp.0.* names.

Matrices become F16. RMS weights become F32 with HF (1+w) baked in, matching
ollama_nvfp4_to_gguf.py. fc [5120, 10240] splits into e_proj / h_proj.
"""

import json
import os
import struct
import sys

import numpy as np

GGUF_MAGIC = b"GGUF"
T_F32, T_F16 = 0, 1
ALIGN = 32

# HF/safetensors name -> (gguf name, kind)
# kind: f16 | f32_plus | fc_split
PLAN = [
    ("pre_fc_norm_embedding.weight", "mtp.0.enorm.weight", "f32_plus"),
    ("pre_fc_norm_hidden.weight", "mtp.0.hnorm.weight", "f32_plus"),
    ("norm.weight", "mtp.0.norm.weight", "f32_plus"),
    ("layers.0.input_layernorm.weight", "mtp.0.attn_norm.weight", "f32_plus"),
    ("layers.0.post_attention_layernorm.weight", "mtp.0.ffn_norm.weight", "f32_plus"),
    ("layers.0.self_attn.q_norm.weight", "mtp.0.attn_q_norm.weight", "f32_plus"),
    ("layers.0.self_attn.k_norm.weight", "mtp.0.attn_k_norm.weight", "f32_plus"),
    ("layers.0.self_attn.q_proj.weight", "mtp.0.attn_q.weight", "f16"),
    ("layers.0.self_attn.k_proj.weight", "mtp.0.attn_k.weight", "f16"),
    ("layers.0.self_attn.v_proj.weight", "mtp.0.attn_v.weight", "f16"),
    ("layers.0.self_attn.o_proj.weight", "mtp.0.attn_output.weight", "f16"),
    ("layers.0.mlp.gate_proj.weight", "mtp.0.ffn_gate.weight", "f16"),
    ("layers.0.mlp.up_proj.weight", "mtp.0.ffn_up.weight", "f16"),
    ("layers.0.mlp.down_proj.weight", "mtp.0.ffn_down.weight", "f16"),
    ("fc.weight", "mtp.0.fc.weight", "fc_split"),
]


def bf16_to_f32(buf):
    u16 = np.frombuffer(buf, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32).copy()


def f32_to_f16_bytes(x):
    return np.asarray(x, np.float32).astype(np.float16).tobytes()


def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data = f.read()
    hdr.pop("__metadata__", None)
    out = {}
    for name, info in hdr.items():
        a, b = info["data_offsets"]
        out[name] = (info["dtype"], info["shape"], data[a:b])
    return out


def write_string(out, s):
    e = s.encode("utf-8")
    out.write(struct.pack("<Q", len(e)))
    out.write(e)


def main():
    if len(sys.argv) != 3:
        print("usage: mtp_head_to_gguf.py HEAD.safetensors OUT.gguf")
        sys.exit(2)
    src, dest = sys.argv[1], sys.argv[2]
    st = read_st(src)
    tensors = []
    for hf, gg, kind in PLAN:
        if hf not in st:
            raise SystemExit(f"missing {hf}")
        dt, shape, raw = st[hf]
        if dt not in ("BF16", "F16", "F32"):
            raise SystemExit(f"{hf} dtype {dt}")
        if dt == "BF16":
            x = bf16_to_f32(raw)
        elif dt == "F16":
            x = np.frombuffer(raw, np.float16).astype(np.float32)
        else:
            x = np.frombuffer(raw, np.float32).copy()
        x = x.reshape(shape)

        if kind == "f32_plus":
            x = x + np.float32(1)
            tensors.append({"name": gg, "dims": [int(d) for d in x.shape],
                            "type": T_F32, "blob": np.ascontiguousarray(x).tobytes()})
        elif kind == "f16":
            # PyTorch [out, in]; GGUF dims [in, out]; bytes stay [out, in].
            if x.ndim != 2:
                raise SystemExit(f"{hf} expected 2D, got {x.shape}")
            dims = [int(x.shape[1]), int(x.shape[0])]
            tensors.append({"name": gg, "dims": dims, "type": T_F16,
                            "blob": f32_to_f16_bytes(np.ascontiguousarray(x))})
        elif kind == "fc_split":
            if x.shape != (5120, 10240):
                raise SystemExit(f"fc unexpected {x.shape}")
            e, h = x[:, :5120], x[:, 5120:]
            for name, half in (("mtp.0.e_proj.weight", e), ("mtp.0.h_proj.weight", h)):
                tensors.append({"name": name, "dims": [5120, 5120], "type": T_F16,
                                "blob": f32_to_f16_bytes(np.ascontiguousarray(half))})
        print(f"  {gg:36s} {kind:8s} {list(shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    out = open(dest, "wb")
    out.write(GGUF_MAGIC)
    out.write(struct.pack("<I", 3))
    out.write(struct.pack("<Q", len(tensors)))
    out.write(struct.pack("<Q", 2))
    write_string(out, "general.architecture")
    out.write(struct.pack("<I", 8))  # STRING
    write_string(out, "qwen3")
    write_string(out, "general.name")
    out.write(struct.pack("<I", 8))
    write_string(out, "qwen38-mtp-head-f16")

    off = 0
    offsets = []
    for t in tensors:
        offsets.append(off)
        off += len(t["blob"])
        off = (off + ALIGN - 1) // ALIGN * ALIGN
    for t, o in zip(tensors, offsets):
        write_string(out, t["name"])
        out.write(struct.pack("<I", len(t["dims"])))
        for d in t["dims"]:
            out.write(struct.pack("<Q", d))
        out.write(struct.pack("<I", t["type"]))
        out.write(struct.pack("<Q", o))
    pos = out.tell()
    out.write(b"\0" * ((ALIGN - pos % ALIGN) % ALIGN))
    body = out.tell()
    for t, o in zip(tensors, offsets):
        out.seek(body + o)
        out.write(t["blob"])
    out.seek(0, 2)
    print(f"wrote {dest} {out.tell()/1e6:.1f} MB  tensors={len(tensors)}")
    out.close()


if __name__ == "__main__":
    main()
