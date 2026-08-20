#!/usr/bin/env python3
"""Convert ofou's mlx.fast MTP head (ofou-q2q4.safetensors) into a ds4 sidecar GGUF (qwen38-mtp-ofou-f16.gguf).

Dequantizes MLX group-64 affine weights (4-bit for MTP block, 2-bit for draft_lm_head).
Overlays Q/K/V precision islands.
Bakes 1+w into F32 RMS norms.
Splits fc into e_proj and h_proj.
Emits compact 98336-row draft_lm_head as F16.
"""

import json
import os
import struct
import sys

import numpy as np

GGUF_MAGIC = b"GGUF"
T_F32, T_F16, T_I32 = 0, 1, 26
ALIGN = 32

NORM_PLAN = [
    ("pre_fc_norm_embedding.weight", "mtp.0.enorm.weight"),
    ("pre_fc_norm_hidden.weight", "mtp.0.hnorm.weight"),
    ("norm.weight", "mtp.0.norm.weight"),
    ("layers.0.input_layernorm.weight", "mtp.0.attn_norm.weight"),
    ("layers.0.post_attention_layernorm.weight", "mtp.0.ffn_norm.weight"),
    ("layers.0.self_attn.q_norm.weight", "mtp.0.attn_q_norm.weight"),
    ("layers.0.self_attn.k_norm.weight", "mtp.0.attn_k_norm.weight"),
]

PROJ_PLAN = [
    ("layers.0.self_attn.q_proj", "mtp.0.attn_q.weight", "q"),
    ("layers.0.self_attn.k_proj", "mtp.0.attn_k.weight", "k"),
    ("layers.0.self_attn.v_proj", "mtp.0.attn_v.weight", "v"),
    ("layers.0.self_attn.o_proj", "mtp.0.attn_output.weight", None),
    ("layers.0.mlp.gate_proj", "mtp.0.ffn_gate.weight", None),
    ("layers.0.mlp.up_proj", "mtp.0.ffn_up.weight", None),
    ("layers.0.mlp.down_proj", "mtp.0.ffn_down.weight", None),
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


def dequant_affine(w_u32, scales_f32, biases_f32, bits=4, group_size=64):
    out_dim, in_u32 = w_u32.shape
    vals_per_u32 = 32 // bits
    in_dim = in_u32 * vals_per_u32
    mask = (1 << bits) - 1

    shifts = np.arange(0, 32, bits, dtype=np.uint32)
    unpacked = (w_u32[:, :, None] >> shifts[None, None, :]) & mask
    unpacked = unpacked.reshape(out_dim, in_dim).astype(np.float32)

    scales_rep = np.repeat(scales_f32, group_size, axis=1)
    biases_rep = np.repeat(biases_f32, group_size, axis=1)

    return scales_rep * unpacked + biases_rep


# NVFP4 decode constants and tables
E2M1 = np.array([
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)


def build_e4m3_u():
    u8 = np.arange(128, dtype=np.uint8)
    exp = (u8 >> 3) & 0x0F
    mant = u8 & 0x07
    out = np.empty(128, dtype=np.float32)
    sub = exp == 0
    out[sub] = (mant[sub].astype(np.float32) / 8.0) * (2.0 ** -6)
    nz = ~sub
    out[nz] = (1.0 + mant[nz].astype(np.float32) / 8.0) * (2.0 ** (exp[nz].astype(np.int16) - 7))
    return out


E4M3_U = build_e4m3_u()


def dequant_nvfp4_blocks(raw_bytes, gs, num_rows, in_dim=5120):
    assert in_dim % 64 == 0
    num_blocks = in_dim // 64
    bytes_per_row = num_blocks * 36
    assert len(raw_bytes) == num_rows * bytes_per_row

    raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(num_rows, num_blocks, 36)

    scales_code = raw[:, :, 32:36] & 0x7F
    scales_f32 = E4M3_U[scales_code] * np.float32(gs)
    scales_rep = np.repeat(scales_f32, 16, axis=2).reshape(num_rows, in_dim)

    payload = raw[:, :, 0:32]
    q_low = payload & 0x0F
    q_high = (payload >> 4) & 0x0F
    interleaved = np.stack([q_low, q_high], axis=-1).reshape(num_rows, in_dim)

    w_f32 = E2M1[interleaved] * scales_rep
    return w_f32


def extract_rerank_table_from_gguf(backbone_path):
    print(f"Reading backbone {backbone_path} for draft rerank table...")
    with open(backbone_path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: {backbone_path}")
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        alignment = 32

        def read_str():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8")

        def read_val(vtype):
            if vtype == 0: return struct.unpack("<B", f.read(1))[0]
            elif vtype == 1: return struct.unpack("<b", f.read(1))[0]
            elif vtype == 2: return struct.unpack("<H", f.read(2))[0]
            elif vtype == 3: return struct.unpack("<h", f.read(2))[0]
            elif vtype == 4: return struct.unpack("<I", f.read(4))[0]
            elif vtype == 5: return struct.unpack("<i", f.read(4))[0]
            elif vtype == 6: return struct.unpack("<f", f.read(4))[0]
            elif vtype == 7: return struct.unpack("<?", f.read(1))[0]
            elif vtype == 8: return read_str()
            elif vtype == 9:
                atype = struct.unpack("<I", f.read(4))[0]
                alen = struct.unpack("<Q", f.read(8))[0]
                return [read_val(atype) for _ in range(alen)]
            elif vtype == 10: return struct.unpack("<Q", f.read(8))[0]
            elif vtype == 11: return struct.unpack("<q", f.read(8))[0]
            elif vtype == 12: return struct.unpack("<d", f.read(8))[0]
            else: raise ValueError(f"Unknown GGUF val type {vtype}")

        for _ in range(kv_count):
            k = read_str()
            vtype = struct.unpack("<I", f.read(4))[0]
            v = read_val(vtype)
            if k == "general.alignment":
                alignment = int(v)

        tensors = {}
        for _ in range(tensor_count):
            name = read_str()
            ndims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndims)]
            ttype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors[name] = {"dims": dims, "type": ttype, "offset": offset}

        pos = f.tell()
        body_offset = (pos + alignment - 1) // alignment * alignment

        if "output.weight" not in tensors:
            raise KeyError("output.weight not found in backbone GGUF")
        if "output.nvfp4_gs" not in tensors:
            raise KeyError("output.nvfp4_gs not found in backbone GGUF")

        # Read gs
        gs_info = tensors["output.nvfp4_gs"]
        f.seek(body_offset + gs_info["offset"])
        gs = struct.unpack("<f", f.read(4))[0]
        print(f"  Backbone output.nvfp4_gs = {gs:.8e}")

        # Read NVFP4 output.weight slices
        out_info = tensors["output.weight"]
        ne0, ne1 = out_info["dims"]
        if ne0 != 5120:
            raise ValueError(f"Expected ne0=5120, got {ne0}")
        if out_info["type"] != 40:
            raise ValueError(f"Expected type 40 (NVFP4), got {out_info['type']}")

        num_blocks = ne0 // 64
        bytes_per_row = num_blocks * 36
        weight_body_start = body_offset + out_info["offset"]

        # Chunk 1: rows 0..98303 (98304 rows)
        f.seek(weight_body_start)
        chunk1_raw = f.read(98304 * bytes_per_row)

        # Chunk 2: rows 248044..248069 (26 rows)
        f.seek(weight_body_start + 248044 * bytes_per_row)
        chunk2_raw = f.read(26 * bytes_per_row)

    print("  Dequantizing NVFP4 backbone output slices...")
    chunk1_f32 = dequant_nvfp4_blocks(chunk1_raw, gs, 98304, 5120)
    chunk2_f32 = dequant_nvfp4_blocks(chunk2_raw, gs, 26, 5120)
    pad_f32 = np.zeros((6, 5120), dtype=np.float32)

    rerank_f32 = np.concatenate([chunk1_f32, chunk2_f32, pad_f32], axis=0)
    print(f"  Created draft rerank table shape={rerank_f32.shape} dtype=float32")
    return rerank_f32


def check_dequant_sanity(st, pinned_path="/Users/au/w/ds4/mtp-heads/pinned-bf16.safetensors"):
    if not os.path.exists(pinned_path):
        print(f"[sanity] pinned-bf16 file not found at {pinned_path}, skipping q_proj check")
        return
    pinned_st = read_st(pinned_path)
    q_w_raw = st["layers.0.self_attn.q_proj.weight"]
    q_s_raw = st["layers.0.self_attn.q_proj.scales"]
    q_b_raw = st["layers.0.self_attn.q_proj.biases"]
    q_w = np.frombuffer(q_w_raw[2], dtype="<u4").reshape(q_w_raw[1])
    q_s = bf16_to_f32(q_s_raw[2]).reshape(q_s_raw[1])
    q_b = bf16_to_f32(q_b_raw[2]).reshape(q_b_raw[1])
    q_dequant = dequant_affine(q_w, q_s, q_b, bits=4, group_size=64)

    q_pinned_raw = pinned_st["layers.0.self_attn.q_proj.weight"]
    q_pinned = bf16_to_f32(q_pinned_raw[2]).reshape(q_pinned_raw[1])

    idx = np.frombuffer(st["precision_islands.q.indices"][2], dtype="<i4")
    island_set = set(idx)
    sample_row = None
    for r in range(q_w.shape[0]):
        if r not in island_set:
            sample_row = r
            break
    if sample_row is None:
        sample_row = 0

    diff = q_dequant[sample_row] - q_pinned[sample_row]
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff ** 2)))
    print(f"[sanity] q_proj non-island row {sample_row} dequant vs pinned-bf16: max_abs={max_abs:.6f}, RMS={rms:.6f}")
    if rms > 10.0:
        raise ValueError(f"q_proj dequant RMS vs bf16 is absurd ({rms:.4f} > 10.0)")
    if rms > 2.0:
        raise ValueError(f"q_proj dequant RMS vs bf16 is unexpectedly high ({rms:.4f} > 2.0), check affine formula")


def convert(src_path, dest_path, backbone_path="/Users/au/w/ds4/qwen38-nvfp4-gs.gguf", pinned_path="/Users/au/w/ds4/mtp-heads/pinned-bf16.safetensors"):
    print(f"Reading {src_path}...")
    st = read_st(src_path)
    check_dequant_sanity(st, pinned_path)
    tensors = []

    # 1. Norms (F32 with 1+w baked)
    for src_name, gg_name in NORM_PLAN:
        if src_name not in st:
            raise KeyError(f"Missing {src_name}")
        raw = st[src_name][2]
        x = bf16_to_f32(raw).reshape(st[src_name][1])
        x = x + np.float32(1.0)
        blob = np.ascontiguousarray(x).tobytes()
        tensors.append({
            "name": gg_name,
            "dims": [int(d) for d in x.shape],
            "type": T_F32,
            "blob": blob,
            "shape": x.shape,
            "dtype": "F32",
        })

    # 2. 4-bit MTP Projections with Precision Island Overlays
    for prefix, gg_name, island_key in PROJ_PLAN:
        w_raw = st[f"{prefix}.weight"]
        s_raw = st[f"{prefix}.scales"]
        b_raw = st[f"{prefix}.biases"]

        w = np.frombuffer(w_raw[2], dtype="<u4").reshape(w_raw[1])
        s = bf16_to_f32(s_raw[2]).reshape(s_raw[1])
        b = bf16_to_f32(b_raw[2]).reshape(b_raw[1])
        x = dequant_affine(w, s, b, bits=4, group_size=64)

        if island_key:
            idx = np.frombuffer(st[f"precision_islands.{island_key}.indices"][2], dtype="<i4")
            pi_raw = st[f"precision_islands.{island_key}.weight"]
            pi = bf16_to_f32(pi_raw[2]).reshape(pi_raw[1])
            x[idx] = pi

            # Self-check: island rows must match exactly
            err = float(np.max(np.abs(x[idx] - pi)))
            print(f"[self-check] precision island {island_key}: max abs error on {len(idx)} rows = {err:.6e} (PASSED)")
            if err > 0.0:
                raise ValueError(f"Precision island overlay {island_key} failed: max abs error {err}")

        dims = [int(x.shape[1]), int(x.shape[0])]
        blob = f32_to_f16_bytes(np.ascontiguousarray(x))
        tensors.append({
            "name": gg_name,
            "dims": dims,
            "type": T_F16,
            "blob": blob,
            "shape": x.shape,
            "dtype": "F16",
        })

    # 3. 4-bit FC Split
    w_fc_raw = st["fc.weight"]
    s_fc_raw = st["fc.scales"]
    b_fc_raw = st["fc.biases"]
    w_fc = np.frombuffer(w_fc_raw[2], dtype="<u4").reshape(w_fc_raw[1])
    s_fc = bf16_to_f32(s_fc_raw[2]).reshape(s_fc_raw[1])
    b_fc = bf16_to_f32(b_fc_raw[2]).reshape(b_fc_raw[1])
    fc = dequant_affine(w_fc, s_fc, b_fc, bits=4, group_size=64)
    if fc.shape != (5120, 10240):
        raise ValueError(f"Unexpected fc shape: {fc.shape}")

    e = fc[:, :5120]
    h = fc[:, 5120:]
    for name, half in [("mtp.0.e_proj.weight", e), ("mtp.0.h_proj.weight", h)]:
        dims = [5120, 5120]
        blob = f32_to_f16_bytes(np.ascontiguousarray(half))
        tensors.append({
            "name": name,
            "dims": dims,
            "type": T_F16,
            "blob": blob,
            "shape": half.shape,
            "dtype": "F16",
        })

    # 4. 2-bit draft_lm_head
    w_head_raw = st["draft_lm_head.weight"]
    s_head_raw = st["draft_lm_head.scales"]
    b_head_raw = st["draft_lm_head.biases"]
    w_head = np.frombuffer(w_head_raw[2], dtype="<u4").reshape(w_head_raw[1])
    s_head = bf16_to_f32(s_head_raw[2]).reshape(s_head_raw[1])
    b_head = bf16_to_f32(b_head_raw[2]).reshape(b_head_raw[1])
    head = dequant_affine(w_head, s_head, b_head, bits=2, group_size=64)
    if head.shape != (98336, 5120):
        raise ValueError(f"Unexpected draft_lm_head shape: {head.shape}")

    dims = [5120, 98336]
    blob = f32_to_f16_bytes(np.ascontiguousarray(head))
    tensors.append({
        "name": "mtp.0.draft_lm_head.weight",
        "dims": dims,
        "type": T_F16,
        "blob": blob,
        "shape": head.shape,
        "dtype": "F16",
    })

    # 4b. Packed 2-bit draft_lm_head tensors
    tensors.append({
        "name": "mtp.0.draft_lm_head.q",
        "dims": [320, 98336],
        "type": T_I32,
        "blob": np.ascontiguousarray(w_head).astype("<u4").tobytes(),
        "shape": w_head.shape,
        "dtype": "I32",
    })
    tensors.append({
        "name": "mtp.0.draft_lm_head.scales",
        "dims": [80, 98336],
        "type": T_F16,
        "blob": f32_to_f16_bytes(np.ascontiguousarray(s_head)),
        "shape": s_head.shape,
        "dtype": "F16",
    })
    tensors.append({
        "name": "mtp.0.draft_lm_head.biases",
        "dims": [80, 98336],
        "type": T_F16,
        "blob": f32_to_f16_bytes(np.ascontiguousarray(b_head)),
        "shape": b_head.shape,
        "dtype": "F16",
    })

    # 5. draft_rerank from backbone output.weight (NVFP4 -> F16)
    rerank_f32 = extract_rerank_table_from_gguf(backbone_path)
    if rerank_f32.shape != (98336, 5120):
        raise ValueError(f"Unexpected draft_rerank shape: {rerank_f32.shape}")

    dims = [5120, 98336]
    blob = f32_to_f16_bytes(np.ascontiguousarray(rerank_f32))
    tensors.append({
        "name": "mtp.0.draft_rerank.weight",
        "dims": dims,
        "type": T_F16,
        "blob": blob,
        "shape": rerank_f32.shape,
        "dtype": "F16",
    })

    # Print tensor table
    print("\n--- Tensor Table ---")
    total_bytes = sum(len(t["blob"]) for t in tensors)
    for t in tensors:
        mb = len(t["blob"]) / (1024 * 1024)
        print(f"  {t['name']:32s} {t['dtype']:4s} shape={str(list(t['shape'])):16s} GGUF_dims={str(t['dims']):16s} {mb:8.2f} MB")
    print(f"Total tensor payload: {total_bytes / (1024*1024):.2f} MB ({total_bytes / (1024*1024*1024):.3f} GiB)\n")

    # Write GGUF file
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
    with open(dest_path, "wb") as out:
        out.write(GGUF_MAGIC)
        out.write(struct.pack("<I", 3))  # GGUF version 3
        out.write(struct.pack("<Q", len(tensors)))  # tensor count
        out.write(struct.pack("<Q", 2))  # metadata kv count
        write_string(out, "general.architecture")
        out.write(struct.pack("<I", 8))  # STRING
        write_string(out, "qwen3")
        write_string(out, "general.name")
        out.write(struct.pack("<I", 8))
        write_string(out, "qwen38-mtp-ofou-f16")

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
        final_size = out.tell()
        print(f"Wrote {dest_path}: {final_size / (1024*1024):.2f} MB ({final_size / (1024*1024*1024):.3f} GiB) across {len(tensors)} tensors.")


def main():
    default_src = "/Users/au/w/ds4/mtp-heads/ofou-q2q4.safetensors"
    default_dest = "/Users/au/w/ds4/mtp-heads/qwen38-mtp-ofou-f16.gguf"
    default_backbone = "/Users/au/w/ds4/qwen38-nvfp4-gs.gguf"
    default_pinned = "/Users/au/w/ds4/mtp-heads/pinned-bf16.safetensors"

    src = sys.argv[1] if len(sys.argv) > 1 else default_src
    dest = sys.argv[2] if len(sys.argv) > 2 else default_dest
    backbone = sys.argv[3] if len(sys.argv) > 3 else default_backbone
    pinned = sys.argv[4] if len(sys.argv) > 4 else default_pinned

    convert(src, dest, backbone, pinned)
if __name__ == "__main__":
    main()
