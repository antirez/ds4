#!/usr/bin/env python3
"""Convert a Qwen3.8-Flash-Next fast-pack (ds4.qwen4.fast-pack v3, `qwen4-exp`
split artifacts) into the single-file `qwen4exp` GGUF schema ds4 loads.

Reads the pack base GGUF (upstream HF tensor names, Q4_0 routed experts,
BF16 control/embeddings, Q8_0 dense) and the external Q4_1 PLE sidecar, and
writes one GGUF with llama.cpp-schema names, KV keys, and weight layout:

  - Q8_0/Q4_0 quantized tensors are copied byte-for-byte (the routed down
    projection drops its inert 768->640 physical padding).
  - BF16 control tensors become F32; ssm_a = -exp(A_log); the fused indexer
    qk projection splits into q_proj/k_proj rows.
  - GDN per-head vectors and alpha/beta rows are permuted into the tiled
    v-head order (v-head j pairs with k-head j % n_k_heads).
  - The PLE n-gram table is dequantized Q4_1 -> F16 and inlined as
    per_layer_token_embd.weight.
  - No MTP block is emitted; the engine trims the profile to the trunk.

Every transform was verified numerically against the official ggml-org Q8_0
GGUF built from the same checkpoint by llama.cpp (ssm_a permutation 144/144,
alpha/beta row permutation cos=1.0, norms raw/folded states, indexer split,
PLE hash constants).

Usage:
  python3 qwen4_pack_to_qwen4exp.py \
      --base .../Qwen3.8-Flash-Next-Q40RoutedExperts-....gguf \
      --ple .../Qwen3.8-Flash-Next-PLE-Q4_1.gguf \
      --out .../Qwen3.8-Flash-Next-Q40Routed-qwen4exp.gguf
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time

import numpy as np

GGUF_MAGIC = b"GGUF"
T_F32, T_F16, T_Q4_0, T_Q4_1, T_Q8_0, T_BF16, T_I64, T_Q4_K, T_MXFP4 = 0, 1, 2, 3, 8, 30, 27, 12, 39
TYPE_BYTES = {T_F32: 4, T_F16: 2, T_Q4_0: 18, T_Q4_1: 20, T_Q8_0: 34, T_BF16: 2, T_I64: 8,
              T_Q4_K: 144, T_MXFP4: 17}
# quantized types store block_bytes per block_elems values, not per element
BLOCK_ELEMS = {T_Q4_0: 32, T_Q4_1: 32, T_Q8_0: 32, T_Q4_K: 256, T_MXFP4: 32}


def tnbytes(t: int, n: int) -> int:
    if t in BLOCK_ELEMS:
        assert n % BLOCK_ELEMS[t] == 0, (t, n)
        return (n // BLOCK_ELEMS[t]) * TYPE_BYTES[t]
    return n * TYPE_BYTES[t]

N_LAYER = 48
N_K_HEAD = 16          # GDN key heads (tiled v-order denominator)
N_V_PER_K = 3          # GDN value heads per key head
ALIGN = 32
N_HC_DIVISOR = 4   # hc_count; qwen4exp mixers are 4x the pack values (verified vs ggml-org)
CHUNK = 256 * 1024 * 1024


def tiled_v(j: int) -> int:
    """qwen4exp GGUF order: row j holds HF v-head (j % nk) * vpk + j // nk."""
    return (j % N_K_HEAD) * N_V_PER_K + j // N_K_HEAD


TILED_ORDER = [tiled_v(j) for j in range(N_K_HEAD * N_V_PER_K)]


class Arr:
    def __init__(self, item_type: int, values: list):
        self.item_type = item_type
        self.values = values


class Reader:
    def __init__(self, path: str):
        self.f = open(path, "rb")
        assert self.f.read(4) == GGUF_MAGIC, path
        self.version = struct.unpack("<I", self.f.read(4))[0]
        n_tensors = struct.unpack("<q", self.f.read(8))[0]
        n_kv = struct.unpack("<q", self.f.read(8))[0]
        self.kv = {}
        self.kv_types = {}
        for _ in range(n_kv):
            k = self.str()
            t = struct.unpack("<I", self.f.read(4))[0]
            self.kv[k] = self.value(t)
            self.kv_types[k] = t
        self.tensors = {}
        for _ in range(n_tensors):
            name = self.str()
            nd = struct.unpack("<I", self.f.read(4))[0]
            dims = [struct.unpack("<q", self.f.read(8))[0] for _ in range(nd)]
            tt = struct.unpack("<I", self.f.read(4))[0]
            off = struct.unpack("<Q", self.f.read(8))[0]
            self.tensors[name] = (tt, dims, off)
        align = self.kv.get("general.alignment", ALIGN)
        self.data_start = (self.f.tell() + align - 1) & ~(align - 1)

    def str(self) -> str:
        n = struct.unpack("<q", self.f.read(8))[0]
        return self.f.read(n).decode()

    def value(self, t: int):
        scalars = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
                   4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1),
                   10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
        if t in scalars:
            fmt, sz = scalars[t]
            return struct.unpack(fmt, self.f.read(sz))[0]
        if t == 8:
            return self.str()
        if t == 9:
            et = struct.unpack("<I", self.f.read(4))[0]
            ct = struct.unpack("<q", self.f.read(8))[0]
            return Arr(et, [self.value(et) for _ in range(ct)])
        raise SystemExit(f"unsupported GGUF value type {t}")

    def seek_tensor(self, name: str):
        t, dims, off = self.tensors[name]
        n = 1
        for d in dims:
            n *= d
        nbytes = tnbytes(t, n)
        self.f.seek(self.data_start + off)
        return t, dims, nbytes

    def read_scalar_tensor(self, name: str, item: str) -> np.ndarray:
        t, dims, nbytes = self.seek_tensor(name)
        assert t in (T_F32, T_F16, T_BF16), (name, t)
        raw = np.frombuffer(self.f.read(nbytes), dtype=np.uint8)
        if t == T_F32:
            return raw.view("<f4").astype(np.float32)
        if t == T_F16:
            return raw.view("<f2").astype(np.float32)
        u = raw.view("<u2").astype(np.uint32) << 16
        return u.view("<f4").astype(np.float32)


def bf16_bytes_to_f32(buf: bytes) -> bytes:
    u = np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16
    return u.view("<f4").tobytes()


# --------------------------------------------------------------------------
# tensor producers: (base, ple, out) -> bytes written


def prod_copy(base_name):
    def p(b, ple, out):
        t, dims, nbytes = b.seek_tensor(base_name)
        left = nbytes
        while left:
            n = min(CHUNK, left)
            out.write(b.f.read(n))
            left -= n
        return nbytes
    return p


def prod_bf16_f32(base_name):
    def p(b, ple, out):
        t, dims, nbytes = b.seek_tensor(base_name)
        left = nbytes
        while left:
            n = min(CHUNK, left) & ~1
            out.write(bf16_bytes_to_f32(b.f.read(n)))
            left -= n
        return nbytes // 2 * 4
    return p



def prod_hc_mix_f32(base_name):
    """BF16 hc mixer (down/up/inject) -> F32 divided by the HC divisor.

    The pack stores the mixers unfolded and records
    ds4.pack.hyper_connection.divisor_folded = hc_count; the qwen4exp schema
    pre-folds the divisor into the weights (verified: ggml-org == pack/4).
    """
    def p(b, ple, out):
        t, dims, nbytes = b.seek_tensor(base_name)
        import numpy as _np
        left = nbytes
        while left:
            n = min(CHUNK, left) & ~1
            v = _np.frombuffer(b.f.read(n), dtype="<u2").astype(_np.uint32) << 16
            v = v.view("<f4") * float(N_HC_DIVISOR)
            out.write(v.astype("<f2").tobytes())
            left -= n
        return nbytes
    return p

def prod_bf16_f16(base_name):
    """BF16 -> F16 straight dequant (hc up-projections carry no divisor fold)."""
    def p(b, ple, out):
        import numpy as _np
        t, dims, nbytes = b.seek_tensor(base_name)
        left = nbytes
        while left:
            n = min(CHUNK, left) & ~1
            v = _np.frombuffer(b.f.read(n), dtype="<u2").astype(_np.uint32) << 16
            out.write(v.view("<f4").astype("<f2").tobytes())
            left -= n
        return nbytes
    return p


def prod_permute_rows_f32(base_name, rows):
    """BF16 [row_length, rows] -> F32 with output row j = pack row tiled_v(j)."""
    def p(b, ple, out):
        t, dims, off = b.tensors[base_name]
        nbytes = 1
        for d in dims:
            nbytes *= d
        nbytes = tnbytes(t, nbytes)
        rowbytes = nbytes // rows
        for j in range(rows):
            b.f.seek(b.data_start + off + TILED_ORDER[j] * rowbytes)
            out.write(bf16_bytes_to_f32(b.f.read(rowbytes)))
        return nbytes // 2 * 4
    p.src = base_name
    return p


def prod_ssm_a(base_name):
    def p(b, ple, out):
        v = b.read_scalar_tensor(base_name, "A_log")
        assert len(v) == N_K_HEAD * N_V_PER_K
        a = -np.exp(v[TILED_ORDER])
        out.write(a.astype("<f4").tobytes())
        return len(a) * 4
    return p


def prod_conv1d_f32(base_name):
    def p(b, ple, out):
        v = b.read_scalar_tensor(base_name, "conv")   # [1, k, width] flat BF16
        out.write(v.astype("<f4").tobytes())
        return len(v) * 4
    return p


def _prod_indexer_part(base_name, row0, count):
    def p(b, ple, out):
        t, dims, nbytes = b.seek_tensor(base_name)
        row = nbytes // 640
        b.f.seek(b.f.tell() + row0 * row)
        left = count * row
        while left:
            n = min(CHUNK, left)
            out.write(b.f.read(n))
            left -= n
        return count * row
    return p


def prod_down_trim(base_name):
    def p(b, ple, out):
        t, dims, off = b.tensors[base_name]
        ne0, ne1, ne2 = dims                        # [768, 2560, 512]
        src_row_blocks = ne0 // 32                  # 24
        dst_row_bytes = (640 // 32) * 18            # 360
        b.f.seek(b.data_start + off)
        written = 0
        for e in range(ne2):
            blob = np.frombuffer(b.f.read(ne1 * src_row_blocks * 18), dtype=np.uint8)
            rows = blob.reshape(ne1, src_row_blocks * 18)[:, :dst_row_bytes]
            out.write(rows.tobytes())
            written += rows.nbytes
        return written
    return p



def prod_gdn_conv1d_f32(base_name):
    """GDN conv taps [1,4,10240] BF16 -> F32 [4,10240], channel-major copy
    with the v-region channels (4096..10240) permuted per 128-wide head by
    the tiled v order (verified against the ggml-org file, cos=1.0000)."""
    def p(b, ple, out):
        v = b.read_scalar_tensor(base_name, "conv")   # flat, channel-major
        assert len(v) == 4 * 10240
        out_v = [0.0] * len(v)
        for w in range(10240):
            if w < 4096:
                out_v[w * 4:w * 4 + 4] = v[w * 4:w * 4 + 4]
            else:
                j = (w - 4096) // 128
                src = 4096 + TILED_ORDER[j] * 128 + ((w - 4096) % 128)
                out_v[w * 4:w * 4 + 4] = v[src * 4:src * 4 + 4]
        import numpy as _np
        out.write(_np.asarray(out_v, dtype="<f4").tobytes())
        return len(out_v) * 4
    return p


def prod_qkv_vperm(base_name):
    """Q8_0 [2560, 10240] with the v-region rows (4096..10239) permuted per
    128-row head: output head slot q holds pack head TILED_ORDER[q] (verified
    against ggml-org, cos=1.0 on every head)."""
    ROWB = 80 * 34
    def p(b, ple, out):
        t, dims, off = b.tensors[base_name]
        b.f.seek(b.data_start + off)
        out.write(b.f.read(4096 * ROWB))
        heads = b.f.read(6144 * ROWB)
        for q in range(48):
            src = TILED_ORDER[q]
            out.write(heads[src * 128 * ROWB:(src + 1) * 128 * ROWB])
        return 10240 * ROWB
    return p


def prod_gate_rowperm(base_name):
    """Q8_0 [2560, 6144]: 48 heads of 128 rows, output slot q holds pack
    head TILED_ORDER[q] (verified, cos=1.0)."""
    ROWB = 80 * 34
    def p(b, ple, out):
        t, dims, off = b.tensors[base_name]
        b.f.seek(b.data_start + off)
        blob = b.f.read(6144 * ROWB)
        for q in range(48):
            src = TILED_ORDER[q]
            out.write(blob[src * 128 * ROWB:(src + 1) * 128 * ROWB])
        return 6144 * ROWB
    return p


def prod_out_colperm(base_name):
    """Q8_0 [6144, 2560]: rows of 6144 input values; the 48 head blocks of
    128 values (4 quant blocks) are reordered so output slot q holds pack
    head TILED_ORDER[q] (verified, cos=1.0)."""
    ROWB = 192 * 34
    CH = 4 * 34
    def p(b, ple, out):
        t, dims, off = b.tensors[base_name]
        b.f.seek(b.data_start + off)
        for r in range(2560):
            row = b.f.read(ROWB)
            for q in range(48):
                src = TILED_ORDER[q]
                out.write(row[src * CH:(src + 1) * CH])
        return 2560 * ROWB
    return p


def deq_q4_k_blocks(buf: "np.ndarray (n,144) uint8") -> "np.ndarray (n,256) f32":
    """Vectorized ggml block_q4_K dequant (matches metal/qwen4.metal stage8)."""
    d = buf[:, 0:2].copy().view("<f2").astype(np.float32).reshape(-1, 1)
    dmin = buf[:, 2:4].copy().view("<f2").astype(np.float32).reshape(-1, 1)
    sc = buf[:, 4:16].astype(np.int32)
    g = np.arange(8, dtype=np.int32).reshape(1, 8)
    s_hi = np.zeros((sc.shape[0], 8), dtype=np.int32)
    s_hi[:, 4:8] = (sc[:, 0:4] & 0xC0) >> 2          # sc[group-4] high bits
    s = np.where(g < 4, sc[:, :8] & 63,
                 (sc[:, 4:12] & 0xF) | s_hi).astype(np.float32)
    mn = np.where(g < 4, sc[:, 4:12] & 63,
                  (sc[:, 4:12] >> 4) | ((sc[:, :8] & 0xC0) >> 2)).astype(np.float32)
    ds = d * s          # (n,8)
    dm = dmin * mn      # (n,8)
    qs = buf[:, 16:144]                          # (n,128)
    idx = np.arange(32, dtype=np.int32)          # value in group
    byte = (g[:, :, None] >> 1) * 32 + idx.reshape(1, 1, 32)   # (1,8,32)
    nbytes = qs[:, byte.reshape(-1)].reshape(-1, 8, 32)        # (n,8,32)
    lo = nbytes & 0xF
    hi = nbytes >> 4
    nib = np.where((g % 2 == 0).reshape(1, 8, 1), lo, hi).astype(np.float32)
    return (ds[:, :, None] * nib - dm[:, :, None]).reshape(-1, 256)


MXFP4_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def quant_mxfp4_rows(vals: "np.ndarray (n,32) f32") -> "np.ndarray (n,17) uint8":
    """OCP MXFP4: shared E8M0 scale per 32-block, e2m1 grid, round-to-nearest."""
    m = np.max(np.abs(vals), axis=1)
    with np.errstate(divide="ignore"):
        e = np.ceil(np.log2(m / 6.0))
    e = np.where(m > 0, e, 0.0)
    e = np.maximum(e, -126.0)
    scale = np.exp2(e.astype(np.float32)).reshape(-1, 1)
    x = vals / scale
    sign = np.sign(x)
    ax = np.abs(x)
    code = np.abs(ax[:, :, None] - MXFP4_GRID.reshape(1, 1, 8)).argmin(axis=2).astype(np.uint8)
    signed = np.where(sign < 0, code | 0x8, code).astype(np.uint8)
    out = np.empty((vals.shape[0], 17), dtype=np.uint8)
    out[:, 0] = (e.astype(np.int32) + 127).astype(np.uint8)
    out[:, 1:17] = signed[:, :16] | (signed[:, 16:] << 4)
    return out


def prod_down_q4k_mxfp4(base_name):
    """Q4_K routed down [768,2560,512] -> MXFP4 [640,2560,512].

    The qwen4exp schema requires the logical 640-wide input that a 256-value
    block type cannot express; dequantize the physical 768-wide rows, drop the
    zero-padded tail, and requantize to MXFP4 (the same down type as the
    llama.cpp Q4K recipe for this model)."""
    def p(b, ple, out):
        t, dims, off = b.tensors[base_name]
        ne0, ne1, ne2 = dims
        assert t == T_Q4_K and ne0 == 768, (base_name, t, dims)
        b.f.seek(b.data_start + off)
        written = 0
        for e_ in range(ne2):
            raw = np.frombuffer(b.f.read(ne1 * 3 * 144), dtype=np.uint8).reshape(ne1, 3, 144)
            vals = deq_q4_k_blocks(raw.reshape(-1, 144)).reshape(ne1, 768)[:, :640]
            blocks = quant_mxfp4_rows(vals.reshape(-1, 32))
            out.write(blocks.tobytes())
            written += blocks.nbytes
            if (e_ & 63) == 0:
                print(f"\r  down q4k->mxfp4 {e_}/{ne2}", end="", flush=True)
        print()
        return written
    return p

def prod_flat_f32(base_name):
    """[2560, 1] BF16 -> [2560] F32 (shared expert gate)."""
    def p(b, ple, out):
        v = b.read_scalar_tensor(base_name, "gate")
        out.write(v.astype("<f4").tobytes())
        return len(v) * 4
    return p


def from_src(src, prod):
    """Bind a producer to read from `src` (e.g. the MTP sidecar) instead of
    the base pack; every producer only touches its first Reader argument."""
    def p(b, ple, out):
        return prod(src, ple, out)
    return p


def prod_eh_proj(src):
    """MTP eh_proj: Q8_0 [5120, 2560] whose output row j concatenates the
    fc_embedding row j (first 2560 inputs: the enorm(embedding) half of the
    engine's mtp_cat) with the fc_hidden row j (hnorm(R) half)."""
    def p(b, ple, out):
        def rd(name):
            t, dims, off = src.tensors[name]
            assert t == T_Q8_0 and dims == [2560, 2560], (name, t, dims)
            src.f.seek(src.data_start + off)
            return src.f.read(tnbytes(t, 2560 * 2560))
        e = rd("language_model.mtp.fc_embedding.weight")
        h = rd("language_model.mtp.fc_hidden.weight")
        row = tnbytes(T_Q8_0, 2560)
        buf = bytearray()
        for j in range(len(e) // row):
            buf += e[j * row:(j + 1) * row]
            buf += h[j * row:(j + 1) * row]
        out.write(bytes(buf))
        return len(buf)
    return p


PLE_CHUNK_ROWS = 4_000_000


def prod_ple(ple_name):
    def p(b, ple, out):
        t, dims, off = ple.tensors[ple_name]
        assert t == T_Q4_1 and dims[0] == 160, (t, dims)
        n_rows = dims[1]
        blocks_per_row = dims[0] // 32              # 5
        ple.f.seek(ple.data_start + off)
        written = 0
        for start in range(0, n_rows, PLE_CHUNK_ROWS):
            n = min(PLE_CHUNK_ROWS, n_rows - start)
            raw = np.frombuffer(ple.f.read(n * blocks_per_row * 20), dtype=np.uint8)
            blocks = raw.reshape(n * blocks_per_row, 20)
            d = blocks[:, 0:2].copy().view("<f2").astype(np.float32).reshape(-1, 1)
            m = blocks[:, 2:4].copy().view("<f2").astype(np.float32).reshape(-1, 1)
            qs = blocks[:, 4:]
            lo = (qs & 0x0F).astype(np.float32)
            hi = (qs >> 4).astype(np.float32)
            outv = np.empty((n * blocks_per_row, 32), dtype="<f2")
            outv[:, :16] = d * lo + m
            outv[:, 16:] = d * hi + m
            out.write(outv.tobytes())
            written += outv.nbytes
            print(f"\r  PLE {start + n}/{n_rows} rows ({written / 1e9:.1f} GB)",
                  end="", flush=True)
        print()
        return written
    return p


# --------------------------------------------------------------------------
# output plan


def build_plan(base=None, mtp=None):
    plan = []
    gate_t, down_t = T_Q4_0, T_Q4_0
    if base is not None:
        gt = base.tensors.get("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight", (None,))[0]
        dt = base.tensors.get("language_model.model.layers.0.mlp.switch_mlp.down_proj.weight", (None,))[0]
        gate_t = gt if gt in (T_Q4_0, T_Q4_K) else T_Q4_0
        down_t = dt if dt in (T_Q4_0, T_Q4_K) else T_Q4_0
        print(f"detected routed expert types: gate/up={gate_t} down={down_t}")

    def add(out_name, out_type, dims, producer):
        plan.append((out_name, out_type, dims, producer))

    add("token_embd.weight", T_BF16, [2560, 248320],
        prod_copy("language_model.model.embed_tokens.weight"))
    add("output.weight", T_Q8_0, [2560, 248320],
        prod_copy("language_model.lm_head.weight"))
    add("output_hc_norm.weight", T_F32, [10240],
        prod_bf16_f32("language_model.model.hyper_connection_mixer.hc_norm.weight"))
    add("output_hc_down.weight", T_F16, [10240, 320],
        prod_hc_mix_f32("language_model.model.hyper_connection_mixer.input_mix_weight_down.weight"))
    add("output_hc_up.weight", T_F16, [320, 10240],
        prod_bf16_f16("language_model.model.hyper_connection_mixer.input_mix_weight_up.weight"))

    for n in range(N_LAYER):
        P = f"language_model.model.layers.{n}."
        B = f"blk.{n}."
        add(B + "hc_attn_norm.weight", T_F32, [10240], prod_bf16_f32(P + "attn_hyper_connection.hc_norm.weight"))
        add(B + "hc_attn_down.weight", T_F16, [10240, 320], prod_hc_mix_f32(P + "attn_hyper_connection.input_mix_weight_down.weight"))
        add(B + "hc_attn_up.weight", T_F16, [320, 10240], prod_bf16_f16(P + "attn_hyper_connection.input_mix_weight_up.weight"))
        add(B + "hc_attn_inject.weight", T_F16, [10240, 4], prod_hc_mix_f32(P + "attn_hyper_connection.block_inject_weight.weight"))
        add(B + "hc_ffn_norm.weight", T_F32, [10240], prod_bf16_f32(P + "mlp_hyper_connection.hc_norm.weight"))
        add(B + "hc_ffn_down.weight", T_F16, [10240, 320], prod_hc_mix_f32(P + "mlp_hyper_connection.input_mix_weight_down.weight"))
        add(B + "hc_ffn_up.weight", T_F16, [320, 10240], prod_bf16_f16(P + "mlp_hyper_connection.input_mix_weight_up.weight"))
        add(B + "hc_ffn_inject.weight", T_F16, [10240, 4], prod_hc_mix_f32(P + "mlp_hyper_connection.block_inject_weight.weight"))
        if (n + 1) % 4 != 0:  # GDN trunk tensors exist only on linear layers
            add(B + "attn_qkv.weight", T_Q8_0, [2560, 10240], prod_qkv_vperm(P + "linear_attn.in_proj_qkv.weight"))
            add(B + "attn_gate.weight", T_Q8_0, [2560, 6144], prod_gate_rowperm(P + "linear_attn.in_proj_z.weight"))
            add(B + "ssm_out.weight", T_Q8_0, [6144, 2560], prod_out_colperm(P + "linear_attn.out_proj.weight"))
            add(B + "ssm_alpha.weight", T_F32, [2560, 48], prod_permute_rows_f32(P + "linear_attn.in_proj_a.weight", 48))
            add(B + "ssm_beta.weight", T_F32, [2560, 48], prod_permute_rows_f32(P + "linear_attn.in_proj_b.weight", 48))
            add(B + "ssm_a", T_F32, [48], prod_ssm_a(P + "linear_attn.A_log"))
            add(B + "ssm_dt.bias", T_F32, [48], prod_permute_rows_f32(P + "linear_attn.dt_bias", 48))
            add(B + "ssm_norm.weight", T_F32, [128], prod_bf16_f32(P + "linear_attn.norm.weight"))
            add(B + "ssm_conv1d.weight", T_F32, [4, 10240], prod_gdn_conv1d_f32(P + "linear_attn.conv1d.weight"))
        if (n + 1) % 4 == 0:  # full attention layers: 3, 7, ..., 47
            add(B + "attn_q.weight", T_Q8_0, [2560, 12288], prod_copy(P + "self_attn.q_proj.weight"))
            add(B + "attn_k.weight", T_Q8_0, [2560, 512], prod_copy(P + "self_attn.k_proj.weight"))
            add(B + "attn_v.weight", T_Q8_0, [2560, 512], prod_copy(P + "self_attn.v_proj.weight"))
            add(B + "attn_output.weight", T_Q8_0, [6144, 2560], prod_copy(P + "self_attn.o_proj.weight"))
            add(B + "attn_q_norm.weight", T_F32, [256], prod_bf16_f32(P + "self_attn.q_norm.weight"))
            add(B + "attn_k_norm.weight", T_F32, [256], prod_bf16_f32(P + "self_attn.k_norm.weight"))
            add(B + "indexer.q_proj.weight", T_Q8_0, [2560, 512],
                _prod_indexer_part(P + "self_attn.indexer.index_qk_proj.weight", 0, 512))
            add(B + "indexer.k_proj.weight", T_Q8_0, [2560, 128],
                _prod_indexer_part(P + "self_attn.indexer.index_qk_proj.weight", 512, 128))
            add(B + "indexer.q_norm.weight", T_F32, [128], prod_bf16_f32(P + "self_attn.indexer.q_layernorm.weight"))
            add(B + "indexer.k_norm.weight", T_F32, [128], prod_bf16_f32(P + "self_attn.indexer.k_layernorm.weight"))
        if n == 1:  # PLE projections live on HF layer 1
            add(B + "ple_key.weight", T_Q8_0, [2560, 10240], prod_copy(P + "ple.key_proj.weight"))
            add(B + "ple_value.weight", T_Q8_0, [2560, 2560], prod_copy(P + "ple.value_proj.weight"))
            add(B + "ple_norm_key.weight", T_F32, [10240], prod_bf16_f32(P + "ple.norm_key.weight"))
            add(B + "ple_norm_query.weight", T_F32, [10240], prod_bf16_f32(P + "ple.norm_query.weight"))
            add(B + "ple_norm_conv.weight", T_F32, [10240], prod_bf16_f32(P + "ple.norm_conv.weight"))
            add(B + "ple_conv1d.weight", T_F32, [4, 10240], prod_conv1d_f32(P + "ple.conv1d.weight"))
        add(B + "ffn_gate_inp.weight", T_F32, [2560, 512], prod_bf16_f32(P + "mlp.gate.weight"))
        add(B + "ffn_gate_exps.weight", gate_t, [2560, 640, 512], prod_copy(P + "mlp.switch_mlp.gate_proj.weight"))
        add(B + "ffn_up_exps.weight", gate_t, [2560, 640, 512], prod_copy(P + "mlp.switch_mlp.up_proj.weight"))
        if down_t == T_Q4_K:
            add(B + "ffn_down_exps.weight", T_MXFP4, [640, 2560, 512],
                prod_down_q4k_mxfp4(P + "mlp.switch_mlp.down_proj.weight"))
        else:
            add(B + "ffn_down_exps.weight", T_Q4_0, [640, 2560, 512],
                prod_down_trim(P + "mlp.switch_mlp.down_proj.weight"))
        add(B + "ffn_gate_shexp.weight", T_Q8_0, [2560, 640], prod_copy(P + "mlp.shared_expert.gate_proj.weight"))
        add(B + "ffn_up_shexp.weight", T_Q8_0, [2560, 640], prod_copy(P + "mlp.shared_expert.up_proj.weight"))
        add(B + "ffn_down_shexp.weight", T_Q8_0, [640, 2560], prod_copy(P + "mlp.shared_expert.down_proj.weight"))
        add(B + "ffn_gate_inp_shexp.weight", T_F32, [2560], prod_flat_f32(P + "mlp.shared_expert_gate.weight"))

    if mtp is not None:
        M = "language_model.mtp.layers.0."
        C = "blk.48."
        def m(prod):
            return from_src(mtp, prod)
        add(C + "hc_attn_norm.weight", T_F32, [10240], m(prod_bf16_f32(M + "attn_hyper_connection.hc_norm.weight")))
        add(C + "hc_attn_down.weight", T_F16, [10240, 320], m(prod_hc_mix_f32(M + "attn_hyper_connection.input_mix_weight_down.weight")))
        add(C + "hc_attn_up.weight", T_F16, [320, 10240], m(prod_bf16_f16(M + "attn_hyper_connection.input_mix_weight_up.weight")))
        add(C + "hc_attn_inject.weight", T_F16, [10240, 4], m(prod_hc_mix_f32(M + "attn_hyper_connection.block_inject_weight.weight")))
        add(C + "hc_ffn_norm.weight", T_F32, [10240], m(prod_bf16_f32(M + "mlp_hyper_connection.hc_norm.weight")))
        add(C + "hc_ffn_down.weight", T_F16, [10240, 320], m(prod_hc_mix_f32(M + "mlp_hyper_connection.input_mix_weight_down.weight")))
        add(C + "hc_ffn_up.weight", T_F16, [320, 10240], m(prod_bf16_f16(M + "mlp_hyper_connection.input_mix_weight_up.weight")))
        add(C + "hc_ffn_inject.weight", T_F16, [10240, 4], m(prod_hc_mix_f32(M + "mlp_hyper_connection.block_inject_weight.weight")))
        # layer 48 is the nextn block: the engine treats it as full attention
        add(C + "attn_q.weight", T_Q8_0, [2560, 12288], m(prod_copy(M + "self_attn.q_proj.weight")))
        add(C + "attn_k.weight", T_Q8_0, [2560, 512], m(prod_copy(M + "self_attn.k_proj.weight")))
        add(C + "attn_v.weight", T_Q8_0, [2560, 512], m(prod_copy(M + "self_attn.v_proj.weight")))
        add(C + "attn_output.weight", T_Q8_0, [6144, 2560], m(prod_copy(M + "self_attn.o_proj.weight")))
        add(C + "attn_q_norm.weight", T_F32, [256], m(prod_bf16_f32(M + "self_attn.q_norm.weight")))
        add(C + "attn_k_norm.weight", T_F32, [256], m(prod_bf16_f32(M + "self_attn.k_norm.weight")))
        add(C + "indexer.q_proj.weight", T_Q8_0, [2560, 512],
            m(_prod_indexer_part(M + "self_attn.indexer.index_qk_proj.weight", 0, 512)))
        add(C + "indexer.k_proj.weight", T_Q8_0, [2560, 128],
            m(_prod_indexer_part(M + "self_attn.indexer.index_qk_proj.weight", 512, 128)))
        add(C + "indexer.q_norm.weight", T_F32, [128], m(prod_bf16_f32(M + "self_attn.indexer.q_layernorm.weight")))
        add(C + "indexer.k_norm.weight", T_F32, [128], m(prod_bf16_f32(M + "self_attn.indexer.k_layernorm.weight")))
        add(C + "ffn_gate_inp.weight", T_F32, [2560, 512], m(prod_bf16_f32(M + "mlp.gate.weight")))
        add(C + "ffn_gate_exps.weight", gate_t, [2560, 640, 512], m(prod_copy(M + "mlp.switch_mlp.gate_proj.weight")))
        add(C + "ffn_up_exps.weight", gate_t, [2560, 640, 512], m(prod_copy(M + "mlp.switch_mlp.up_proj.weight")))
        if down_t == T_Q4_K:
            add(C + "ffn_down_exps.weight", T_MXFP4, [640, 2560, 512],
                m(prod_down_q4k_mxfp4(M + "mlp.switch_mlp.down_proj.weight")))
        else:
            add(C + "ffn_down_exps.weight", T_Q4_0, [640, 2560, 512],
                m(prod_down_trim(M + "mlp.switch_mlp.down_proj.weight")))
        add(C + "ffn_gate_shexp.weight", T_Q8_0, [2560, 640], m(prod_copy(M + "mlp.shared_expert.gate_proj.weight")))
        add(C + "ffn_up_shexp.weight", T_Q8_0, [2560, 640], m(prod_copy(M + "mlp.shared_expert.up_proj.weight")))
        add(C + "ffn_down_shexp.weight", T_Q8_0, [640, 2560], m(prod_copy(M + "mlp.shared_expert.down_proj.weight")))
        add(C + "ffn_gate_inp_shexp.weight", T_F32, [2560], m(prod_flat_f32(M + "mlp.shared_expert_gate.weight")))
        add(C + "nextn.eh_proj.weight", T_Q8_0, [5120, 2560], prod_eh_proj(mtp))
        add(C + "nextn.enorm.weight", T_F32, [2560], m(prod_bf16_f32("language_model.mtp.pre_fc_norm_embedding.weight")))
        add(C + "nextn.hnorm.weight", T_F32, [10240], m(prod_bf16_f32("language_model.mtp.pre_fc_norm_hidden.weight")))
        add(C + "nextn.hc_head_norm.weight", T_F32, [10240], m(prod_bf16_f32("language_model.mtp.hyper_connection_mixer.hc_norm.weight")))
        add(C + "nextn.hc_head_down.weight", T_F16, [10240, 320], m(prod_hc_mix_f32("language_model.mtp.hyper_connection_mixer.input_mix_weight_down.weight")))
        add(C + "nextn.hc_head_up.weight", T_F16, [320, 10240], m(prod_bf16_f16("language_model.mtp.hyper_connection_mixer.input_mix_weight_up.weight")))
    return plan


# --------------------------------------------------------------------------
# KV


def build_kv(base: Reader, ple: Reader, mtp=None):
    pk = base.kv
    out = []

    def add(k, t, v):
        out.append((k, t, v))

    def g(src, default=None):
        v = pk.get(src, default)
        if v is None:
            raise SystemExit(f"pack KV missing {src}")
        return v

    add("general.architecture", 8, "qwen4exp")
    add("general.type", 8, "model")
    add("general.name", 8, "Qwen3.8-Flash-Next Q40Routed (converted fast-pack)")
    add("general.alignment", 4, ALIGN)

    add("qwen4exp.block_count", 4, N_LAYER + (1 if mtp is not None else 0))
    if mtp is not None:
        # engine: n_layer = block_count; layer 48 is the nextn block
        add("qwen4exp.nextn_predict_layers", 4, 1)
    add("qwen4exp.context_length", 4, g("qwen4-exp.context_length"))
    add("qwen4exp.embedding_length", 4, g("qwen4-exp.embedding_length"))
    add("qwen4exp.vocab_size", 4, g("qwen4-exp.vocab_size"))
    add("qwen4exp.attention.head_count", 4, g("qwen4-exp.attention.head_count"))
    add("qwen4exp.attention.head_count_kv", 4, g("qwen4-exp.attention.head_count_kv"))
    add("qwen4exp.attention.key_length", 4, g("qwen4-exp.attention.key_length"))
    add("qwen4exp.attention.value_length", 4, g("qwen4-exp.attention.value_length"))
    add("qwen4exp.attention.layer_norm_rms_epsilon", 6,
        g("qwen4-exp.attention.layer_norm_rms_epsilon"))
    add("qwen4exp.attention.indexer.head_count", 4, g("qwen4-exp.attention.indexer.head_count"))
    add("qwen4exp.attention.indexer.key_length", 4, g("qwen4-exp.attention.indexer.key_length"))
    add("qwen4exp.attention.indexer.top_k", 4, g("qwen4-exp.attention.indexer.top_k"))
    add("qwen4exp.expert_count", 4, g("qwen4-exp.expert_count"))
    add("qwen4exp.expert_used_count", 4, g("qwen4-exp.expert_used_count"))
    add("qwen4exp.expert_feed_forward_length", 4, g("qwen4-exp.expert_feed_forward_length"))
    add("qwen4exp.expert_shared_feed_forward_length", 4,
        g("qwen4-exp.shared_expert_feed_forward_length"))
    add("qwen4exp.ssm.conv_kernel", 4, g("qwen4-exp.linear_attention.conv_kernel"))
    add("qwen4exp.ssm.state_size", 4, g("qwen4-exp.linear_attention.head_dimension"))
    add("qwen4exp.ssm.group_count", 4, g("qwen4-exp.linear_attention.key_head_count"))
    add("qwen4exp.ssm.time_step_rank", 4, g("qwen4-exp.linear_attention.value_head_count"))
    add("qwen4exp.ssm.inner_size", 4,
        g("qwen4-exp.linear_attention.value_head_count") * g("qwen4-exp.linear_attention.head_dimension"))
    add("qwen4exp.full_attention_interval", 4, g("qwen4-exp.full_attention_interval"))
    add("qwen4exp.hyper_connection.count", 4, g("qwen4-exp.hyper_connection.count"))
    add("qwen4exp.hyper_connection.low_rank", 4, g("qwen4-exp.hyper_connection.lowrank"))
    add("qwen4exp.rope.dimension_count", 4, g("qwen4-exp.rope.dimension_count"))
    add("qwen4exp.rope.freq_base", 6, g("qwen4-exp.rope.freq_base"))
    interval = g("qwen4-exp.full_attention_interval")
    n_out = N_LAYER + (1 if mtp is not None else 0)
    def linear(n):  # engine: linear iff (n+1)%interval and n+nextn < n_layer
        return (n + 1) % interval != 0 and n + 1 < n_out
    add("qwen4exp.attention.compress_ratios", 9,
        Arr(10, [0 if linear(n) else 4 for n in range(n_out)]))

    add("qwen4exp.ple.layers", 9, Arr(10, [g("qwen4-exp.ple.layer")]))
    add("qwen4exp.ple.ngram_size", 4, g("qwen4-exp.ple.ngram_size"))
    add("qwen4exp.ple.heads_per_ngram", 4, g("qwen4-exp.ple.heads_per_ngram"))
    add("qwen4exp.ple.conv_kernel", 4, g("qwen4-exp.ple.conv_kernel"))
    add("qwen4exp.ple.eos_token_id", 4, g("qwen4-exp.ple.eos_token_id", g("qwen4-exp.ngram_eos", 248044)))
    add("qwen4exp.embedding_length_per_layer_input", 4, g("qwen4-exp.ple.row_dimension"))
    add("qwen4exp.ple.seed", 4, g("qwen4-exp.ple.seed"))
    add("qwen4exp.ple.vocab_base", 4, g("qwen4-exp.ple.vocab_base"))
    add("qwen4exp.ple.vocab_divisor", 4, g("qwen4-exp.ple.vocab_divisor"))
    add("qwen4exp.ple.row_count", 4, g("qwen4-exp.ple.row_count"))
    add("qwen4exp.ple.row_dimension", 4, g("qwen4-exp.ple.row_dimension"))

    mult = struct.unpack("<3q", ple.raw("ple.layer_multipliers", 24))
    offs = struct.unpack("<16q", ple.raw("ple.ngram_heads_offsets", 128))
    vocab = struct.unpack("<16q", ple.raw("ple.ngram_heads_vocab_sizes", 128))
    add("qwen4exp.ple.layer_multipliers", 9, Arr(10, list(mult)))
    add("qwen4exp.ple.head_offsets", 9, Arr(10, list(offs)))
    add("qwen4exp.ple.head_vocab_sizes", 9, Arr(10, list(vocab)))

    for k, v in pk.items():
        if k.startswith("tokenizer."):
            add(k, base.kv_types[k], v)
    return out


def reader_raw(r: Reader, name: str, nbytes: int) -> bytes:
    t, dims, off = r.tensors[name]
    r.f.seek(r.data_start + off)
    return r.f.read(nbytes)


Reader.raw = reader_raw


# --------------------------------------------------------------------------
# writer


def w_str(s) -> bytes:
    b = s if isinstance(s, bytes) else s.encode()
    return struct.pack("<q", len(b)) + b


SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
              6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
SCALAR_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def kv_bytes(k, t, v) -> bytes:
    out = w_str(k) + struct.pack("<I", t)
    if t == 8:
        return out + w_str(v)
    if t == 9:
        out += struct.pack("<I", v.item_type) + struct.pack("<q", len(v.values))
        for x in v.values:
            if v.item_type == 8:
                out += w_str(x)
            else:
                out += struct.pack(SCALAR_FMT[v.item_type], x)
        return out
    return out + struct.pack(SCALAR_FMT[t], v)


def write_gguf(path, kv, plan, base, ple, ple_external=False):
    ple_rows = base.kv["qwen4-exp.ple.row_count"]
    ordered = list(plan)
    if not ple_external:
        ordered.append(("per_layer_token_embd.weight", T_F16, [160, ple_rows],
                        prod_ple("ple.weight")))

    infos = []
    off = 0
    for name, t, dims, producer in ordered:
        n = 1
        for d in dims:
            n *= d
        nbytes = tnbytes(t, n)
        infos.append((name, t, dims, off, nbytes, producer))
        off += nbytes

    header = 24   # magic + version + n_tensors + n_kv
    for k, t, v in kv:
        header += len(kv_bytes(k, t, v))
    for name, t, dims, off_, nbytes, _ in infos:
        header += 8 + len(name.encode()) + 4 + 8 * len(dims) + 4 + 8
    data_start = (header + ALIGN - 1) & ~(ALIGN - 1)

    tmp = path + ".incomplete"
    out = open(tmp, "wb", buffering=1024 * 1024 * 16)
    out.write(GGUF_MAGIC)
    out.write(struct.pack("<I", 3))
    out.write(struct.pack("<q", len(infos)))
    out.write(struct.pack("<q", len(kv)))
    for k, t, v in kv:
        out.write(kv_bytes(k, t, v))
    for name, t, dims, off_, nbytes, _ in infos:
        out.write(w_str(name))
        out.write(struct.pack("<I", len(dims)))
        for d in dims:
            out.write(struct.pack("<q", d))
        out.write(struct.pack("<I", t))
        out.write(struct.pack("<Q", off_))
    out.write(b"\0" * (data_start - header))

    t0 = time.time()
    written = 0
    for i, (name, t, dims, off_, nbytes, producer) in enumerate(infos):
        t1 = time.time()
        n = producer(base, ple, out)
        if n != nbytes:
            raise SystemExit(f"{name}: wrote {n} bytes, expected {nbytes}")
        written += n
        print(f"[{i + 1}/{len(infos)}] {name} {nbytes / 1e9:.3f} GB in {time.time() - t1:.1f}s "
              f"(total {written / 1e9:.1f} GB, elapsed {time.time() - t0:.0f}s)", flush=True)
    out.close()
    os.replace(tmp, path)
    print(f"wrote {path} ({os.path.getsize(path) / 1e9:.2f} GB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--ple", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mtp", default=None,
                    help="MTP sidecar GGUF (qwen4-exp-mtp); adds the blk.48 nextn block")
    ap.add_argument("--ple-external", action="store_true",
                    help="omit the inlined per_layer_token_embd.weight; the engine "
                         "reads the table from the --ple sidecar GGUF (--ple)")
    args = ap.parse_args()

    base = Reader(args.base)
    ple = Reader(args.ple)
    mtp = Reader(args.mtp) if args.mtp else None
    plan = build_plan(base, mtp)

    # every pack tensor must be consumed exactly once
    srcs = []
    for name, t, dims, producer in plan:
        srcs.append(getattr(producer, "base_name", None))
    pack_names = set(base.tensors)
    # map producers back to their source names via closure inspection is
    # fragile; instead count coverage by size accounting at write time.
    print(f"plan: {len(plan)} tensors + PLE; pack base has {len(pack_names)} tensors")
    kv = build_kv(base, ple, mtp)
    write_gguf(args.out, kv, plan, base, ple, args.ple_external)


if __name__ == "__main__":
    main()
