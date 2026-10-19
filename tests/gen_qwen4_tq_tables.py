#!/usr/bin/env python3
"""Generate the ds4 Qwen3.8-Flash-Next TurboQuant tables and golden vectors.

Reference: mlx-vlm/mlx_vlm/turboquant.py (_rht_sign_vector, _codebook,
_TurboQuantMSECodec.quantize, _pack_lowbit).  Run from the repo root:
  python3 tests/gen_qwen4_tq_tables.py          # writes ds4_qwen4_tq_tables.h
Prints sha256 pins for every table (f32 little-endian bytes) and one golden
quantized row per bit width (packed words + f16 norm) for tests/test_qwen4_tq.c,
including a ready-to-paste `C golden` line for that test's pin table.
Codebooks of widths MIN_BITS..MAX_BITS are concatenated into flat cb[]/mid[]
arrays indexed by DS4_QWEN4_TQ_TAB_BASE(bits).
"""
import hashlib
import math

import numpy as np

DIM = 256
SEED = 0  # mlx-vlm DEFAULT_TURBOQUANT_SEED
EPS = np.float32(1e-6)
OUT = "ds4_qwen4_tq_tables.h"
# Supported widths.  2 is the floor: 1-bit leaves two centroids and is
# functionally useless for KV; oMLX documents 2 as its lowest shipped depth.
MIN_BITS, MAX_BITS = 2, 8
# Width b starts at tab_base(b) in the flat codebook/midpoint arrays, because
# sum_{i=MIN..b-1} 2^i == 2^b - 2^MIN.  mid[] has one entry fewer than cb[],
# so it shares the base and leaves that last slot unused.
def tab_base(bits: int) -> int:
    return (1 << bits) - (1 << MIN_BITS)


ENTRIES = sum(1 << b for b in range(MIN_BITS, MAX_BITS + 1))


def rht_sign_vector(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + dim * 7919)
    return rng.choice([-1.0, 1.0], size=dim).astype(np.float32)


def beta_pdf(grid: np.ndarray, dim: int) -> np.ndarray:
    log_coeff = math.lgamma(dim / 2) - 0.5 * math.log(math.pi) - math.lgamma((dim - 1) / 2)
    log_pdf = log_coeff + ((dim - 3) / 2) * np.log(np.clip(1.0 - grid**2, 1e-30, None))
    pdf = np.exp(log_pdf - np.max(log_pdf))
    s = pdf.sum()
    return pdf / s if s != 0 else np.full_like(grid, 1.0 / len(grid))


def codebook(dim: int, nbits: int) -> np.ndarray:
    levels = 1 << nbits
    grid = np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, 32768, dtype=np.float32)
    weights = beta_pdf(grid, dim)
    cdf = np.cumsum(weights)
    quantiles = (np.arange(levels, dtype=np.float32) + 0.5) / levels
    centroids = np.interp(quantiles, cdf, grid).astype(np.float32)
    for _ in range(100):
        boundaries = np.empty(levels + 1, dtype=np.float32)
        boundaries[0], boundaries[-1] = -1.0, 1.0
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        new = centroids.copy()
        for i in range(levels):
            if i == levels - 1:
                mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
            else:
                mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
            bw = weights[mask]
            if bw.size == 0:
                continue
            tw = bw.sum()
            if tw > 0:
                new[i] = np.sum(bw * grid[mask]) / tw
        if np.max(np.abs(new - centroids)) < 1e-6:
            centroids = new
            break
        centroids = new
    return centroids.astype(np.float32)


def _wht(y: np.ndarray) -> np.ndarray:
    """Canonical f32 stage butterfly: stage s pairs (i, i+2^s) -> (a+b, a-b),
    ascending s; one exact 1/sqrt(D) scale at the end (1/16 for D=256)."""
    n = len(y)
    stride = 1
    while stride < n:
        for base in range(0, n, 2 * stride):
            a = y[base:base + stride].copy()
            b = y[base + stride:base + 2 * stride].copy()
            y[base:base + stride] = a + b
            y[base + stride:base + 2 * stride] = a - b
        stride *= 2
    return (y * np.float32(1.0 / math.isqrt(n))).astype(np.float32)


def rht(x: np.ndarray, sgn: np.ndarray) -> np.ndarray:
    """Forward RHT: hadamard(sgn*x)/sqrt(D) — signs BEFORE the transform."""
    return _wht((x.astype(np.float32) * sgn).astype(np.float32).copy())


def rht_inverse(y: np.ndarray, sgn: np.ndarray) -> np.ndarray:
    """Inverse RHT: sgn * hadamard(y)/sqrt(D) — signs AFTER the transform."""
    return (_wht(y.astype(np.float32).copy()) * sgn).astype(np.float32)


def pack_lowbit(idx: np.ndarray, nbits: int) -> np.ndarray:
    pw = (len(idx) * nbits + 31) // 32
    packed = np.zeros(pw, dtype=np.uint32)
    for i, v in enumerate(idx):
        off = i * nbits
        w, s = off // 32, off % 32
        packed[w] |= np.uint32(v) << np.uint32(s)
        spill = s + nbits - 32
        if spill > 0:
            packed[w + 1] |= np.uint32(v) >> np.uint32(nbits - spill)
    return packed


def unpack_lowbit(packed: np.ndarray, nbits: int, length: int) -> np.ndarray:
    mask = (1 << nbits) - 1
    out = np.zeros(length, dtype=np.uint32)
    for i in range(length):
        off = i * nbits
        w, s = off // 32, off % 32
        v = int(packed[w]) >> s
        spill = s + nbits - 32
        if spill > 0:
            v |= int(packed[w + 1]) << (nbits - spill)
        out[i] = v & mask
    return out


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def c_lit(v) -> str:
    """f32 literal: .9g drops the decimal point on exact zeros ('0' -> '0f'
    is an octal constant), and the padding slots between codebooks are 0."""
    s = f"{float(v):.9g}"
    if not any(c in s for c in ".eE"):
        s += ".0"
    return s + "f"


def main() -> None:
    signs = rht_sign_vector(DIM, SEED)
    words = np.zeros(DIM // 32, dtype=np.uint32)
    for dd in range(DIM):
        if signs[dd] > 0:
            words[dd // 32] |= np.uint32(1 << (dd % 32))

    # butterfly vs matmul cross-check (Sylvester Hadamard)
    H = np.array([[1.0]])
    while len(H) < DIM:
        H = np.block([[H, H], [H, -H]])
    xc = np.random.default_rng(7).standard_normal(DIM).astype(np.float32)
    assert np.max(np.abs(rht(xc, signs).astype(np.float64)
                         - (H @ (xc.astype(np.float64) * signs.astype(np.float64))) / 16.0)) < 2e-5

    tables = {}
    for nb in range(MIN_BITS, MAX_BITS + 1):
        cb = codebook(DIM, nb)
        tables[nb] = (cb, ((cb[:-1] + cb[1:]) / 2).astype(np.float32))

    def flat(which: int) -> np.ndarray:
        out = np.zeros(ENTRIES, dtype=np.float32)
        for nb in range(MIN_BITS, MAX_BITS + 1):
            arr = tables[nb][which]
            out[tab_base(nb):tab_base(nb) + arr.size] = arr
        return out

    cb_all, mid_all = flat(0), flat(1)

    print("/* sha256 pins (f32 LE bytes) */")
    print(f"signs_f32      {sha(signs)}")
    print(f"codebooks_f32  {sha(cb_all)}   /* widths {MIN_BITS}..{MAX_BITS}, concatenated */")
    print(f"midpoints_f32  {sha(mid_all)}")
    for nb, (cb, mid) in tables.items():
        print(f"codebook{nb}_f32   {sha(cb)}")
        print(f"midpoints{nb}_f32  {sha(mid)}")

    # golden row: deterministic pseudo-random in [-4, 4)
    dd = np.arange(DIM, dtype=np.uint32)
    x = ((((dd * np.uint32(2654435761)) % np.uint32(2001)).astype(np.float32)) - 1000.0) / 250.0
    for nb, (cb, mid) in tables.items():
        ss = np.float32(0.0)
        for v in x:
            ss = np.float32(ss + np.float32(v * v))  # ascending f32, bit-exact with the C reference
        norm = np.float32(math.sqrt(float(ss)))
        unit = np.array([np.float32(v / np.maximum(norm, EPS)) for v in x], dtype=np.float32)
        rot = rht(unit, signs)
        idx = np.searchsorted(mid, rot, side="left").astype(np.uint32)
        assert unpack_lowbit(pack_lowbit(idx, nb), nb, DIM).tolist() == idx.tolist()
        packed = pack_lowbit(idx, nb)
        nh = np.float16(norm)
        est = rht_inverse(cb[idx], signs).astype(np.float32) * np.float32(nh)
        cos = float(np.dot(est.astype(np.float64), x.astype(np.float64))
                    / (np.linalg.norm(est) * np.linalg.norm(x)))
        print(f"\n/* golden row, bits={nb}: x[d] = (((d*2654435761) % 2001) - 1000) / 250 */")
        print(f"norm_f32       {float(norm):.9g}")
        print(f"norm_f16_bits  0x{nh.view(np.uint16):04x}")
        print(f"packed_sha256  {sha(packed)}")
        print("packed[0..3]   " + " ".join(f"0x{w:08x}" for w in packed[:4]))
        print(f"roundtrip_cos  {cos:.6f}")
        print(f"C golden       {{ {nb}, 0x{nh.view(np.uint16):04x}u, "
              + ", ".join(f"0x{w:08x}u" for w in packed[:4]) + f", {cos:.6f} }},")

    with open(OUT, "w") as fp:
        fp.write("/* Generated by tests/gen_qwen4_tq_tables.py - do not edit.\n")
        fp.write(" * sha256 pins (f32 LE bytes) are in the generator stdout; see the plan. */\n")
        fp.write("#ifndef DS4_QWEN4_TQ_TABLES_H\n#define DS4_QWEN4_TQ_TABLES_H\n\n")
        fp.write("#ifdef DS4_QWEN4_TQ_TABLES_DEFINITION\n")
        fp.write("const ds4_qwen4_tq_tables ds4_qwen4_tq_table = {\n")
        fp.write("    { " + " ".join(f"0x{w:08x}u," for w in words) + " },\n")
        for arr in (cb_all, mid_all):
            fp.write("    {\n")
            for i in range(0, len(arr), 8):
                fp.write("        " + ", ".join(c_lit(v) for v in arr[i:i + 8]) + ",\n")
            fp.write("    },\n")
        fp.write("};\n")
        fp.write("#endif\n\n#endif\n")
    print(f"\nwrote {OUT}")
    print("header sha256:", hashlib.sha256(open(OUT, 'rb').read()).hexdigest())


if __name__ == "__main__":
    main()
