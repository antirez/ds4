#!/usr/bin/env python3
"""Requantize a GGUF's 2D weight tensors to DS4 Q4_64A (type 36).

Q4_64A block: 36 bytes per 64 weights -- qs[32] packed nibbles (even index in
the low nibble), bf16 scale at offset 32, bf16 bias at offset 34.  Value is
q * scale + bias, matching gguf-tools/quants.c ds4q_quantize_q4_64a and
metal/dense.metal ds4_dense_block_q4_64a.
"""

import argparse
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

T_F32, T_F16, T_Q8_0, T_Q4_K, T_Q5_K, T_Q6_K = 0, 1, 8, 12, 13, 14
T_Q4_64A = 36

BLOCK = {T_F32: (1, 4), T_F16: (1, 2), T_Q8_0: (32, 34),
         T_Q4_K: (256, 144), T_Q5_K: (256, 176), T_Q6_K: (256, 210),
         T_Q4_64A: (64, 36)}

NAME = {T_F32: "f32", T_F16: "f16", T_Q8_0: "q8_0", T_Q4_K: "q4_K",
        T_Q5_K: "q5_K", T_Q6_K: "q6_K", T_Q4_64A: "q4_64a"}


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


def parse(path):
    with open(path, "rb") as f:
        head = f.read(64 * 1024 * 1024)
    r = Reader(head)
    if r.raw(4) != GGUF_MAGIC:
        sys.exit("not a GGUF file")
    version = r.fixed(U32)
    n_tensors = r.fixed(U64)
    n_kv = r.fixed(U64)
    kv = []
    for _ in range(n_kv):
        k = r.string()
        t = r.fixed(U32)
        kv.append((k, t, r.value(t)))
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.fixed(U32)
        dims = [r.fixed(U64) for _ in range(nd)]
        ttype = r.fixed(U32)
        off = r.fixed(U64)
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
    align = 32
    for k, _, v in kv:
        if k == "general.alignment":
            align = int(v)
    data_start = (r.o + align - 1) // align * align
    return version, kv, tensors, align, data_start


def nbytes(ttype, nelem):
    blk, size = BLOCK[ttype]
    return nelem // blk * size


def _k_scales(sc_raw, nb):
    sc = np.empty((nb, 8), np.float32)
    mn = np.empty((nb, 8), np.float32)
    for j in range(8):
        if j < 4:
            s = sc_raw[:, j] & 63
            m = sc_raw[:, j + 4] & 63
        else:
            s = (sc_raw[:, j + 4] & 0xF) | ((sc_raw[:, j - 4] >> 6) << 4)
            m = (sc_raw[:, j + 4] >> 4) | ((sc_raw[:, j] >> 6) << 4)
        sc[:, j] = s
        mn[:, j] = m
    return sc, mn


def deq_q4_k(raw, nelem):
    nb = nelem // 256
    b = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 144)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    sc, mn = _k_scales(b[:, 4:16].astype(np.uint16), nb)
    qs = b[:, 16:144]
    out = np.empty((nb, 256), np.float32)
    for p in range(4):
        chunk = qs[:, p * 32:(p + 1) * 32]
        lo = (chunk & 0xF).astype(np.float32)
        hi = (chunk >> 4).astype(np.float32)
        j0, j1 = 2 * p, 2 * p + 1
        out[:, p * 64:p * 64 + 32] = (d * sc[:, j0:j0 + 1]) * lo - dmin * mn[:, j0:j0 + 1]
        out[:, p * 64 + 32:p * 64 + 64] = (d * sc[:, j1:j1 + 1]) * hi - dmin * mn[:, j1:j1 + 1]
    return out.reshape(-1)


def deq_q5_k(raw, nelem):
    nb = nelem // 256
    b = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 176)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    sc, mn = _k_scales(b[:, 4:16].astype(np.uint16), nb)
    qh = b[:, 16:48]
    qs = b[:, 48:176]
    out = np.empty((nb, 256), np.float32)
    for p in range(4):
        chunk = qs[:, p * 32:(p + 1) * 32]
        j0, j1 = 2 * p, 2 * p + 1
        b0 = ((qh >> (2 * p + 0)) & 1).astype(np.float32) * 16.0
        b1 = ((qh >> (2 * p + 1)) & 1).astype(np.float32) * 16.0
        lo = (chunk & 0xF).astype(np.float32) + b0
        hi = (chunk >> 4).astype(np.float32) + b1
        out[:, p * 64:p * 64 + 32] = (d * sc[:, j0:j0 + 1]) * lo - dmin * mn[:, j0:j0 + 1]
        out[:, p * 64 + 32:p * 64 + 64] = (d * sc[:, j1:j1 + 1]) * hi - dmin * mn[:, j1:j1 + 1]
    return out.reshape(-1)


def deq_q6_k(raw, nelem):
    nb = nelem // 256
    b = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 210)
    ql = b[:, 0:128]
    qh = b[:, 128:192]
    sc = b[:, 192:208].copy().view(np.int8).astype(np.float32)
    d = b[:, 208:210].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    out = np.empty((nb, 256), np.float32)
    is_ = np.arange(32) // 16
    for n in range(2):
        qlc = ql[:, n * 64:(n + 1) * 64]
        qhc = qh[:, n * 32:(n + 1) * 32]
        scc = sc[:, n * 8:(n + 1) * 8]
        base = n * 128
        q1 = ((qlc[:, 0:32] & 0xF).astype(np.int16) | (((qhc >> 0) & 3).astype(np.int16) << 4)) - 32
        q2 = ((qlc[:, 32:64] & 0xF).astype(np.int16) | (((qhc >> 2) & 3).astype(np.int16) << 4)) - 32
        q3 = ((qlc[:, 0:32] >> 4).astype(np.int16) | (((qhc >> 4) & 3).astype(np.int16) << 4)) - 32
        q4 = ((qlc[:, 32:64] >> 4).astype(np.int16) | (((qhc >> 6) & 3).astype(np.int16) << 4)) - 32
        out[:, base + 0:base + 32] = d * scc[:, is_ + 0] * q1
        out[:, base + 32:base + 64] = d * scc[:, is_ + 2] * q2
        out[:, base + 64:base + 96] = d * scc[:, is_ + 4] * q3
        out[:, base + 96:base + 128] = d * scc[:, is_ + 6] * q4
    return out.reshape(-1)


def deq_q8_0(raw, nelem):
    nb = nelem // 32
    b = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 34)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    q = b[:, 2:34].copy().view(np.int8).astype(np.float32)
    return (d * q).reshape(-1)


DEQ = {T_Q4_K: deq_q4_k, T_Q5_K: deq_q5_k, T_Q6_K: deq_q6_k, T_Q8_0: deq_q8_0}


def quant_q4_64a(x):
    """x: float32 [n], n % 64 == 0 -> uint8 [n/64, 36]."""
    g = np.ascontiguousarray(x.reshape(-1, 64), dtype=np.float32)
    mn = g.min(axis=1, keepdims=True)
    mx = g.max(axis=1, keepdims=True)
    scale = ((mx - mn) / 15.0).astype(np.float32)
    bias = mn.astype(np.float32)
    # The block stores bf16 by truncating the f32 bit pattern, so quantize
    # against the values the reader will actually reconstruct.
    s_b = (scale.view(np.uint32) >> 16).astype(np.uint16)
    b_b = (bias.view(np.uint32) >> 16).astype(np.uint16)
    s_eff = (s_b.astype(np.uint32) << 16).view(np.float32)
    b_eff = (b_b.astype(np.uint32) << 16).view(np.float32)
    zero = (s_eff == 0.0)
    safe = np.where(zero, np.float32(1.0), s_eff)
    q = np.rint((g - b_eff) / safe)
    q = np.clip(q, 0, 15)
    q = np.where(zero, 0, q).astype(np.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    out = np.empty((g.shape[0], 36), np.uint8)
    out[:, 0:32] = packed
    out[:, 32:34] = s_b.copy().view(np.uint8).reshape(-1, 2)
    out[:, 34:36] = b_b.copy().view(np.uint8).reshape(-1, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep", default="",
                    help="comma-separated substrings of tensor names to leave untouched")
    ap.add_argument("--from-types", default="q4_K,q5_K,q6_K,q8_0",
                    help="comma-separated source type names eligible for conversion")
    args = ap.parse_args()

    version, kv, tensors, align, data_start = parse(args.src)
    print(f"gguf v{version} tensors={len(tensors)} kv={len(kv)} align={align} data@{data_start}")

    counts = {}
    for t in tensors:
        counts[t["type"]] = counts.get(t["type"], 0) + 1
    print("input types:", {NAME.get(k, k): v for k, v in sorted(counts.items())})

    keep = [s for s in args.keep.split(",") if s]
    want = {s.strip() for s in args.from_types.split(",") if s.strip()}
    plan = []
    for t in tensors:
        nelem = 1
        for d in t["dims"]:
            nelem *= d
        convert = (t["type"] in DEQ and len(t["dims"]) >= 2
                   and t["dims"][0] % 64 == 0
                   and NAME.get(t["type"]) in want
                   and not any(k in t["name"] for k in keep))
        plan.append((t, nelem, convert))

    old_bytes = sum(nbytes(t["type"], n) for t, n, _ in plan)
    new_bytes = sum(nbytes(T_Q4_64A, n) if c else nbytes(t["type"], n)
                    for t, n, c in plan)
    n_conv = sum(1 for _, _, c in plan if c)
    print(f"convert {n_conv}/{len(plan)} tensors  {old_bytes/1e9:.2f}G -> {new_bytes/1e9:.2f}G")
    if args.list or not args.dst:
        seen = set()
        for t, n, c in plan:
            key = (t["type"], c, len(t["dims"]), t["name"].split(".")[-2] if "." in t["name"] else t["name"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  {'CONV' if c else 'keep'} {NAME.get(t['type'], t['type']):7s} {t['dims']} {t['name']}")
        return

    src = open(args.src, "rb")
    out = open(args.dst, "wb")
    out.write(GGUF_MAGIC)
    out.write(struct.pack("<I", version))
    out.write(struct.pack("<Q", len(tensors)))
    out.write(struct.pack("<Q", len(kv)))
    for k, t, v in kv:
        write_string(out, k)
        out.write(struct.pack("<I", t))
        write_value(out, t, v)

    new_off = 0
    offsets = []
    for t, n, c in plan:
        offsets.append(new_off)
        new_off += nbytes(T_Q4_64A if c else t["type"], n)
        new_off = (new_off + align - 1) // align * align
    for (t, n, c), off in zip(plan, offsets):
        write_string(out, t["name"])
        out.write(struct.pack("<I", len(t["dims"])))
        for d in t["dims"]:
            out.write(struct.pack("<Q", d))
        out.write(struct.pack("<I", T_Q4_64A if c else t["type"]))
        out.write(struct.pack("<Q", off))

    pos = out.tell()
    out.write(b"\0" * ((align - pos % align) % align))
    body_start = out.tell()

    for i, ((t, n, c), off) in enumerate(zip(plan, offsets)):
        src.seek(data_start + t["offset"])
        raw = src.read(nbytes(t["type"], n))
        if c:
            blob = quant_q4_64a(DEQ[t["type"]](raw, n)).tobytes()
        else:
            blob = raw
        out.seek(body_start + off)
        out.write(blob)
        if i % 100 == 0 or i + 1 == len(plan):
            print(f"  [{i+1}/{len(plan)}] {t['name']}", flush=True)
    out.seek(0, 2)
    end = out.tell()
    out.close()
    src.close()
    print(f"wrote {args.dst}  {end/1e9:.2f}G")


if __name__ == "__main__":
    main()
