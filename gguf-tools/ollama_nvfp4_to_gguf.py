#!/usr/bin/env python3
"""Pack ollama qwen3.8:27b-mlx NVFP4 blobs into a DS4 GGUF (type 40).

Uses the official Q4_K_M GGUF only as metadata/tokenizer template.
Weights come from ~/.ollama/models blobs (safetensors: U32 qs + U8 e4m3
scales + F32 global_scale). NVFP4 is repacked to ggml block_nvfp4
(36B / 64 elems) with global_scale folded into the e4m3 scales.
"""

import json
import os
import struct
import sys

import numpy as np

GGUF_MAGIC = b"GGUF"
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)
_FIXED = {
    U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
    U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
    U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8),
}
T_F32, T_F16, T_BF16, T_NVFP4 = 0, 1, 30, 40
ALIGN = 32

# E2M1 codes 0..15 (same table mlx / ggml kvalues_mxfp4 uses).
E2M1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    np.float32,
)


def e4m3_to_f32(u8):
    u8 = np.asarray(u8, dtype=np.uint8)
    sign = np.where(u8 & 0x80, np.float32(-1.0), np.float32(1.0))
    exp = (u8 >> 3) & 0x0F
    mant = u8 & 0x07
    out = np.empty(u8.shape, np.float32)
    sub = exp == 0
    out[sub] = sign[sub] * (mant[sub].astype(np.float32) / 8.0) * np.float32(2.0 ** -6)
    nz = ~sub
    out[nz] = sign[nz] * (1.0 + mant[nz].astype(np.float32) / 8.0) * (2.0 ** (exp[nz].astype(np.int16) - 7))
    out[u8 == 0x7F] = np.nan
    out[u8 == 0xFF] = np.nan
    return out


def f32_to_e4m3(x):
    x = np.asarray(x, dtype=np.float32)
    sign = np.where(np.signbit(x), np.uint8(0x80), np.uint8(0))
    ax = np.abs(x)
    out = np.zeros(x.shape, np.uint8)
    finite = np.isfinite(ax) & (ax > 0)
    # brute 7-bit magnitude: pick closest e4m3 code
    codes = np.arange(1, 0x7F, dtype=np.uint8)
    vals = e4m3_to_f32(codes)
    # vectorized nearest
    ax_f = ax.reshape(-1)
    mask = finite.reshape(-1)
    if mask.any():
        d = np.abs(vals[None, :] - ax_f[mask, None])
        pick = codes[d.argmin(axis=1)]
        flat = out.reshape(-1)
        flat[mask] = pick
        out = flat.reshape(x.shape)
    out |= sign
    out[~np.isfinite(x)] = 0x7F
    out[x == 0] = 0
    return out


class Reader:
    def __init__(self, buf):
        self.b = buf
        self.o = 0

    def raw(self, n):
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def fixed(self, t):
        fmt, n = _FIXED[t]
        return struct.unpack(fmt, self.raw(n))[0]

    def string(self):
        n = self.fixed(U64)
        return self.raw(n).decode("utf-8", "replace")

    def value(self, t):
        if t == STRING:
            return self.string()
        if t == ARRAY:
            et = self.fixed(U32)
            n = self.fixed(U64)
            return (et, [self.value(et) for _ in range(n)])
        return self.fixed(t)


def write_string(out, s):
    e = s.encode("utf-8")
    out.write(struct.pack("<Q", len(e)))
    out.write(e)


def write_value(out, t, v):
    if t == STRING:
        write_string(out, v)
        return
    if t == ARRAY:
        et, items = v
        out.write(struct.pack("<I", et))
        out.write(struct.pack("<Q", len(items)))
        for it in items:
            write_value(out, et, it)
        return
    fmt, _ = _FIXED[t]
    out.write(struct.pack(fmt, v))


def parse_gguf_header(path):
    with open(path, "rb") as f:
        head = f.read(32 * 1024 * 1024)
    r = Reader(head)
    if r.raw(4) != GGUF_MAGIC:
        sys.exit("template is not GGUF")
    version = r.fixed(U32)
    n_tensors = r.fixed(U64)
    n_kv = r.fixed(U64)
    kv = []
    for _ in range(n_kv):
        k = r.string()
        t = r.fixed(U32)
        kv.append((k, t, r.value(t)))
    return version, kv


def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data = f.read()
    hdr.pop("__metadata__", None)
    tensors = {}
    for name, info in hdr.items():
        a, b = info["data_offsets"]
        tensors[name] = (info["dtype"], info["shape"], data[a:b])
    return tensors


def blob_path(digest, blobdir):
    h = digest.split(":")[-1]
    p = os.path.join(blobdir, "sha256-" + h)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def load_manifest(manifest):
    d = json.load(open(manifest))
    blobdir = os.path.expanduser("~/.ollama/models/blobs")
    by_name = {}
    for L in d["layers"]:
        n = L.get("name")
        if n:
            by_name[n] = blob_path(L["digest"], blobdir)
    return by_name


def nvfp4_to_blocks(qs_u32, scales_u8, global_scale, out_rows, in_dim):
    """qs_u32: [out, in/8]; scales U8 [out, in/16] kept as UE4M3.
    global_scale is NOT folded: e4m3 cannot represent ~1e-4*20 without 2x snap.
    Caller stores gs as a sidecar F32 tensor."""
    del global_scale
    nblk = in_dim // 64
    qs = np.frombuffer(qs_u32, dtype=np.uint32).reshape(out_rows, in_dim // 8)
    sc = np.frombuffer(scales_u8, dtype=np.uint8).reshape(out_rows, in_dim // 16) & np.uint8(0x7F)
    out = np.empty((out_rows, nblk, 36), np.uint8)
    for b in range(nblk):
        chunk = qs[:, b * 8:(b + 1) * 8].astype(np.uint32)
        raw = chunk.view(np.uint8).reshape(out_rows, 32)
        out[:, b, 0:32] = raw
        out[:, b, 32:36] = sc[:, b * 4:(b + 1) * 4]
    return out.tobytes()


def bf16_to_f32(buf):
    u16 = np.frombuffer(buf, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_f16_bytes(x):
    return np.asarray(x, np.float32).astype(np.float16).tobytes()


K_HEADS, V_HEADS, HEAD_K, HEAD_V = 16, 48, 128, 128
V_PER_K = V_HEADS // K_HEADS


def tiled_v_perm(num_k, num_v_per_k, head_dim):
    # HF grouped [G0_v0..vr-1, G1_v0..] -> tiled [G0_v0, G1_v0, ...]
    idx = np.arange(num_k * num_v_per_k * head_dim)
    return idx.reshape(num_k, num_v_per_k, head_dim).transpose(1, 0, 2).reshape(-1)


def apply_gdn_nvfp4_reorder(src, qs_u32, scales_u8, out_rows, in_dim):
    """Match llama.cpp Qwen NVFP4 V-head tiled reorder."""
    qs = np.frombuffer(qs_u32, dtype=np.uint32).reshape(out_rows, in_dim // 8).copy()
    sc = np.frombuffer(scales_u8, dtype=np.uint8).reshape(out_rows, in_dim // 16).copy()
    row_v = tiled_v_perm(K_HEADS, V_PER_K, HEAD_V)
    row_a = tiled_v_perm(K_HEADS, V_PER_K, 1)
    if src.endswith(".linear_attn.in_proj_qkv.weight"):
        q_dim = HEAD_K * K_HEADS
        k_dim = HEAD_K * K_HEADS
        v = qs[q_dim + k_dim:]
        vs = sc[q_dim + k_dim:]
        qs[q_dim + k_dim:] = v[row_v]
        sc[q_dim + k_dim:] = vs[row_v]
    elif src.endswith(".linear_attn.in_proj_z.weight"):
        qs = qs[row_v]
        sc = sc[row_v]
    elif src.endswith((".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight")):
        qs = qs[row_a]
        sc = sc[row_a]
    elif src.endswith(".linear_attn.out_proj.weight"):
        col = tiled_v_perm(K_HEADS, V_PER_K, HEAD_V)
        codes = np.empty((out_rows, in_dim), np.uint8)
        raw = qs.view(np.uint8).reshape(out_rows, in_dim // 2)
        codes[:, 0::2] = raw & 0x0F
        codes[:, 1::2] = raw >> 4
        codes = codes[:, col]
        packed = np.empty_like(raw)
        packed[:] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        qs = packed.view(np.uint32).reshape(out_rows, in_dim // 8)
        group = (col.reshape(-1, 16)[:, 0] // 16).astype(np.int64)
        sc = sc[:, group]
    return qs.tobytes(), sc.tobytes()


def apply_gdn_f16_reorder(src, x):
    if src.endswith((".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight")):
        row_a = tiled_v_perm(K_HEADS, V_PER_K, 1)
        if x.ndim == 2 and x.shape[0] == V_HEADS:
            return x[row_a]
    return x


# llama.cpp convert stores HF (1+w) RMS. ds4 multiplies stored w only.
# ollama/HF keep zero-centered. Bake here. ssm_norm is already unshifted.
BAKE_ONE_PLUS = (
    "attn_norm.weight",
    "ffn_norm.weight",
    "post_attention_norm.weight",
    "output_norm.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "enorm.weight",
    "hnorm.weight",
    "mtp.0.norm.weight",
)


# ollama HF name -> (gguf name, kind)
# kind: nvfp4 | f32 | f16 | conv
def layer_map(il, linear):
    p = f"model.language_model.layers.{il}"
    g = f"blk.{il}"
    m = [
        (f"{p}.input_layernorm.weight", f"{g}.attn_norm.weight", "f32"),
        (f"{p}.post_attention_layernorm.weight", f"{g}.ffn_norm.weight", "f32"),
        (f"{p}.post_attention_layernorm.weight", f"{g}.post_attention_norm.weight", "f32"),
        (f"{p}.mlp.gate_proj.weight", f"{g}.ffn_gate.weight", "nvfp4"),
        (f"{p}.mlp.up_proj.weight", f"{g}.ffn_up.weight", "nvfp4"),
        (f"{p}.mlp.down_proj.weight", f"{g}.ffn_down.weight", "nvfp4"),
    ]
    if linear:
        m += [
            (f"{p}.linear_attn.in_proj_qkv.weight", f"{g}.attn_qkv.weight", "nvfp4"),
            (f"{p}.linear_attn.in_proj_z.weight", f"{g}.attn_gate.weight", "nvfp4"),
            (f"{p}.linear_attn.in_proj_a.weight", f"{g}.ssm_alpha.weight", "f16"),
            (f"{p}.linear_attn.in_proj_b.weight", f"{g}.ssm_beta.weight", "f16"),
            (f"{p}.linear_attn.out_proj.weight", f"{g}.ssm_out.weight", "nvfp4"),
            (f"{p}.linear_attn.conv1d.weight", f"{g}.ssm_conv1d.weight", "conv"),
            (f"{p}.linear_attn.A_log", f"{g}.ssm_a", "f32"),
            (f"{p}.linear_attn.dt_bias", f"{g}.ssm_dt.bias", "f32"),
            (f"{p}.linear_attn.norm.weight", f"{g}.ssm_norm.weight", "f32"),
        ]
    else:
        m += [
            (f"{p}.self_attn.q_proj.weight", f"{g}.attn_q.weight", "nvfp4"),
            (f"{p}.self_attn.k_proj.weight", f"{g}.attn_k.weight", "nvfp4"),
            (f"{p}.self_attn.v_proj.weight", f"{g}.attn_v.weight", "nvfp4"),
            (f"{p}.self_attn.o_proj.weight", f"{g}.attn_output.weight", "nvfp4"),
            (f"{p}.self_attn.q_norm.weight", f"{g}.attn_q_norm.weight", "f32"),
            (f"{p}.self_attn.k_norm.weight", f"{g}.attn_k_norm.weight", "f32"),
        ]
    return m


def convert_one(kind, st, src_name=""):
    """Return (gguf_type, dims_in_out, bytes). dims = [in, out] GGUF convention."""
    # pick primary tensor
    items = [(k, v) for k, v in st.items()]
    by = {k.split(".")[-1]: (k, v) for k, v in items}
    # keys may be full names
    def find(suffix):
        hits = []
        for k, v in st.items():
            if k == suffix or k.endswith("." + suffix) or k.endswith(suffix):
                hits.append((k, v))
        # Prefer the shortest key so ".scale" wins over ".global_scale"
        # and ".weight" wins over ".weight.scale".
        if not hits:
            return None, None
        hits.sort(key=lambda kv: len(kv[0]))
        return hits[0]

    if kind == "nvfp4":
        wk, winfo = find("weight")
        if not wk:
            wk, winfo = items[0]
        dt, shape, raw = winfo
        # weight is U32 [out, in/8]
        out_r, in8 = shape
        in_dim = in8 * 8
        sk, sinfo = find("scale")
        if sk and sk.endswith("global_scale"):
            sinfo = None
            for k, v in st.items():
                if k.endswith(".scale") and not k.endswith("global_scale"):
                    sinfo = v
                    break
        gk, ginfo = find("global_scale")
        gs = struct.unpack("<f", ginfo[2])[0] if ginfo else 1.0
        if src_name:
            raw, sinfo_bytes = apply_gdn_nvfp4_reorder(src_name, raw, sinfo[2], out_r, in_dim)
            sinfo = (sinfo[0], sinfo[1], sinfo_bytes)
        blob = nvfp4_to_blocks(raw, sinfo[2], gs, out_r, in_dim)
        return T_NVFP4, [in_dim, out_r], blob

    # single tensor
    k, (dt, shape, raw) = items[0]
    if kind == "conv":
        # ollama [C,1,K] bf16. Kernel wants [C][K]; GGUF dims [K, C].
        # llama.cpp tiles the V-channel block the same way as in_proj_qkv V.
        x = bf16_to_f32(raw).reshape(shape)
        if x.ndim == 3:
            x = x.reshape(x.shape[0], x.shape[-1])  # [C, K]
        if x.shape[0] == 10240:
            row_v = tiled_v_perm(K_HEADS, V_PER_K, HEAD_V)
            perm = np.concatenate([np.arange(4096), 4096 + row_v])
            x = x[perm]
        x = np.ascontiguousarray(x)
        return T_F32, [int(x.shape[1]), int(x.shape[0])], x.astype(np.float32).tobytes()
    if kind == "f32":
        if dt == "BF16":
            x = bf16_to_f32(raw)
        elif dt == "F32":
            x = np.frombuffer(raw, np.float32)
        else:
            raise SystemExit(f"bad f32 src {dt} {k}")
        dims = list(shape) if shape else [x.size]
        if len(dims) == 2 and dims[0] < dims[1] and dims[0] in (48, 128, 256):
            x = x.reshape(dims).T
            dims = [dims[1], dims[0]]
        else:
            x = x.reshape(dims)
        # llama.cpp stores A = -exp(A_log), V-tiled like alpha/beta.
        if src_name.endswith(".linear_attn.A_log"):
            x = -np.exp(x.astype(np.float32))
            x = x.reshape(-1)[tiled_v_perm(K_HEADS, V_PER_K, 1)]
            dims = [int(x.size)]
        elif src_name.endswith(".linear_attn.dt_bias"):
            x = x.reshape(-1)[tiled_v_perm(K_HEADS, V_PER_K, 1)]
            dims = [int(x.size)]
        return T_F32, dims, np.ascontiguousarray(x, np.float32).tobytes()
    if kind == "f16":
        x = bf16_to_f32(raw).reshape(shape)
        x = apply_gdn_f16_reorder(src_name, x)
        if x.ndim == 2:
            # PyTorch [out, in]; GGUF dims [in, out]; bytes stay [out, in].
            dims = [int(x.shape[1]), int(x.shape[0])]
        else:
            dims = [int(d) for d in x.shape]
        return T_F16, dims, f32_to_f16_bytes(np.ascontiguousarray(x))
    raise SystemExit(kind)


def main():
    template = sys.argv[1]
    dest = sys.argv[2]
    manifest = os.path.expanduser(
        "~/.ollama/models/manifests/registry.ollama.ai/library/qwen3.8/27b-mlx"
    )
    blobs = load_manifest(manifest)
    linear = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
              24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42,
              44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 58, 60, 61, 62}

    plan = []
    plan.append(("model.language_model.embed_tokens.weight", "token_embd.weight", "f16"))
    plan.append(("model.language_model.norm.weight", "output_norm.weight", "f32"))
    plan.append(("lm_head.weight", "output.weight", "nvfp4"))
    for il in range(64):
        plan.extend(layer_map(il, il in linear))
    plan += [
        ("mtp.pre_fc_norm_embedding.weight", "mtp.0.enorm.weight", "f32"),
        ("mtp.pre_fc_norm_hidden.weight", "mtp.0.hnorm.weight", "f32"),
        ("mtp.norm.weight", "mtp.0.norm.weight", "f32"),
        ("mtp.layers.0.input_layernorm.weight", "mtp.0.attn_norm.weight", "f32"),
        ("mtp.layers.0.post_attention_layernorm.weight", "mtp.0.ffn_norm.weight", "f32"),
        ("mtp.layers.0.self_attn.q_proj.weight", "mtp.0.attn_q.weight", "nvfp4"),
        ("mtp.layers.0.self_attn.k_proj.weight", "mtp.0.attn_k.weight", "nvfp4"),
        ("mtp.layers.0.self_attn.v_proj.weight", "mtp.0.attn_v.weight", "nvfp4"),
        ("mtp.layers.0.self_attn.o_proj.weight", "mtp.0.attn_output.weight", "nvfp4"),
        ("mtp.layers.0.self_attn.q_norm.weight", "mtp.0.attn_q_norm.weight", "f32"),
        ("mtp.layers.0.self_attn.k_norm.weight", "mtp.0.attn_k_norm.weight", "f32"),
        ("mtp.layers.0.mlp.gate_proj.weight", "mtp.0.ffn_gate.weight", "nvfp4"),
        ("mtp.layers.0.mlp.up_proj.weight", "mtp.0.ffn_up.weight", "nvfp4"),
        ("mtp.layers.0.mlp.down_proj.weight", "mtp.0.ffn_down.weight", "nvfp4"),
        ("mtp.fc.weight", "mtp.0.fc.weight", "nvfp4_fc"),
    ]


    tensors = []
    for src, dst, kind in plan:
        if src not in blobs:
            print("MISSING", src)
            continue
        st = read_st(blobs[src])
        if kind == "nvfp4_fc":
            # ollama mtp.fc is NVFP4 [out=5120, in=10240] = [e | h] on the
            # input axis. ds4 wants two [5120,5120] matvecs.
            wk, winfo = None, None
            for k, v in st.items():
                if k.endswith(".weight") and not k.endswith(".weight.scale"):
                    wk, winfo = k, v
                    break
            dt, shape, raw = winfo
            out_r, in8 = shape
            in_dim = in8 * 8
            if in_dim != 10240 or out_r != 5120:
                raise SystemExit(f"mtp.fc unexpected {shape}")
            qs = np.frombuffer(raw, np.uint32).reshape(out_r, in8)
            sc = None
            gs = 1.0
            for k, v in st.items():
                if k.endswith("global_scale"):
                    gs = struct.unpack("<f", v[2])[0]
                elif k.endswith(".scale") and not k.endswith("global_scale"):
                    sc = np.frombuffer(v[2], np.uint8).reshape(out_r, in_dim // 16)
            half8, halfsc = in8 // 2, (in_dim // 16) // 2
            for name, qs_h, sc_h in (
                ("mtp.0.e_proj.weight", qs[:, :half8], sc[:, :halfsc]),
                ("mtp.0.h_proj.weight", qs[:, half8:], sc[:, halfsc:]),
            ):
                blob = nvfp4_to_blocks(qs_h.tobytes(), sc_h.tobytes(), gs, out_r, 5120)
                tensors.append({"name": name, "dims": [5120, 5120], "type": T_NVFP4, "blob": blob})
                tensors.append({"name": name[:-7] + ".nvfp4_gs", "dims": [1], "type": T_F32,
                                "blob": struct.pack("<f", gs)})
                print(f"  {name:42s} type={T_NVFP4} [5120, 5120] {len(blob)/1e6:.2f}MB")
            continue
        ttype, dims, blob = convert_one(kind, st, src)
        if any(dst.endswith(s) for s in BAKE_ONE_PLUS):
            x = np.frombuffer(blob, np.float32).copy()
            x += np.float32(1)
            blob = x.tobytes()

        tensors.append({"name": dst, "dims": dims, "type": ttype, "blob": blob})
        print(f"  {dst:42s} type={ttype} {dims} {len(blob)/1e6:.2f}MB")
        if kind == "nvfp4":
            gs = 1.0
            for k, v in st.items():
                if k.endswith("global_scale"):
                    gs = struct.unpack("<f", v[2])[0]
                    break
            gs_name = dst[:-7] + ".nvfp4_gs" if dst.endswith(".weight") else dst + ".nvfp4_gs"
            tensors.append({"name": gs_name, "dims": [1], "type": T_F32,
                            "blob": struct.pack("<f", gs)})


    version, kv = parse_gguf_header(template)
    # drop alignment override; keep tokenizer kv
    out = open(dest, "wb")
    out.write(GGUF_MAGIC)
    out.write(struct.pack("<I", version))
    out.write(struct.pack("<Q", len(tensors)))
    out.write(struct.pack("<Q", len(kv)))
    for k, t, v in kv:
        write_string(out, k)
        out.write(struct.pack("<I", t))
        write_value(out, t, v)

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
    print(f"wrote {dest} {out.tell()/1e9:.2f}G  tensors={len(tensors)}")
    out.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: ollama_nvfp4_to_gguf.py TEMPLATE.gguf OUT.gguf")
        sys.exit(2)
    main()
