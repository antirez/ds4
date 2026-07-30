#!/usr/bin/env python3
"""MXFP6 attention round-trip splice (quality pre-check, no engine changes).

Takes the production gguf, and for every MXFP8 (type 38) tensor in the
selected attention/shared set: decode -> quantize through MXFP6 (group-32
E8M0 scale, element E2M3 or E3M2, RNE) -> dequantize -> re-encode MXFP8.
The tensor keeps type 38 and byte size, so the splice is an IN-PLACE patch
of a copy: header, offsets, and every unselected byte are untouched and the
unmodified engine runs the result.

Usage:
  splice_mxfp6_attn.py SRC.gguf --list
  splice_mxfp6_attn.py SRC.gguf OUT.gguf --fmt e2m3|e3m2 [--set attn|attn+shexp]

Precedents: temp/splice_mxfp8_head.py (MXFP8 33B block layout, scale rule),
temp/p0/splice_attn_fp8.py (attention tensor set).
"""
import argparse
import hashlib
import os
import shutil
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxfp6_lib import roundtrip_mxfp8_container, self_test

ALIGN = 32
T_MXFP8 = 38
# type -> (block elems, block bytes); from the ds4 converter/splice precedents
BLK = {0: (1, 4), 1: (1, 2), 8: (32, 34), 10: (256, 84), 16: (256, 66),
       26: (1, 4), 30: (1, 2), 38: (32, 33), 39: (32, 17), 40: (32, 17)}

CHUNK_GROUPS = 4 << 20          # 4Mi groups * 33 B = 132 MB raw per chunk


def parse_gguf(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", "not a gguf"
    ver, = struct.unpack("<I", f.read(4))
    nt, = struct.unpack("<Q", f.read(8))
    nkv, = struct.unpack("<Q", f.read(8))

    def rs():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n)

    def rv(t):
        if t == 8:
            return rs()
        if t == 9:
            et, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            return [rv(et) for _ in range(n)]
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        fm = {0: "<b", 1: "<B", 2: "<h", 3: "<H", 4: "<i", 5: "<I", 6: "<f",
              7: "<?", 10: "<q", 11: "<Q", 12: "<d"}
        return struct.unpack(fm[t], f.read(sz[t]))[0]

    for _ in range(nkv):
        rs()
        t, = struct.unpack("<I", f.read(4))
        rv(t)
    tens = []
    for _ in range(nt):
        nb, = struct.unpack("<Q", f.read(8))
        name = f.read(nb).decode()
        nd, = struct.unpack("<I", f.read(4))
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        typ, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        tens.append((name, typ, dims, off))
    data_start = (f.tell() + ALIGN - 1) // ALIGN * ALIGN
    f.close()
    return tens, data_start


def tensor_bytes(typ, dims):
    be, bb = BLK[typ]
    ne = 1
    for d in dims:
        ne *= d
    assert ne % be == 0, (typ, dims)
    return ne // be * bb


def is_selected(name, which):
    """The per-token-read attention weight set (what PULSAR_ATTN_MXFP4 targeted).
    Excludes: output.weight (LM head - already has its own 1.87% verdict),
    token_embd (gathered, ~0 traffic), dspark.* / mtp.* (drafter: affects
    acceptance rate, not output quality - and would contaminate the frontier
    compare's attribution), experts (2-bit already)."""
    if name.startswith(("dspark.", "mtp.")):
        return False
    base = name.split(".")[-2] if "." in name else name
    attn = base.startswith("attn_") or ".attn_" in name
    shexp = "_shexp" in name
    if which == "attn":
        return attn
    if which == "attn+shexp":
        return attn or shexp
    raise ValueError(which)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--fmt", choices=["e2m3", "e3m2"])
    ap.add_argument("--set", dest="which", default="attn+shexp",
                    choices=["attn", "attn+shexp"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-sha", action="store_true")
    args = ap.parse_args()

    tens, data_start = parse_gguf(args.src)

    sel = [(n, t, d, o) for (n, t, d, o) in tens
           if t == T_MXFP8 and is_selected(n, args.which)]
    other_mx = [(n, t, d, o) for (n, t, d, o) in tens
                if t == T_MXFP8 and not is_selected(n, args.which)]

    if args.list:
        by_kind = {}
        for n, t, d, o in tens:
            by_kind.setdefault(t, [0, 0])
            by_kind[t][0] += 1
            by_kind[t][1] += tensor_bytes(t, d)
        print("== type census ==")
        for t, (c, b) in sorted(by_kind.items()):
            print(f"  type {t:3d}: {c:4d} tensors, {b/1e9:8.3f} GB")
        print(f"== MXFP8 selected ({args.which}): {len(sel)} tensors, "
              f"{sum(tensor_bytes(t, d) for _, t, d, _ in sel)/1e9:.3f} GB ==")
        seen = set()
        for n, t, d, o in sel:
            base = n.replace(n.split(".")[1], "L", 1) if n.split(".")[0] == "blk" else n
            if base not in seen:
                seen.add(base)
                print(f"  {base:40s} dims={d}")
        print(f"== MXFP8 NOT selected: {len(other_mx)} ==")
        for n, t, d, o in other_mx:
            print(f"  {n:40s} dims={d} {tensor_bytes(t, d)/1e9:.3f} GB")
        return

    assert args.out and args.fmt, "need OUT and --fmt (or --list)"
    print("library self-test...", flush=True)
    self_test()

    total = sum(tensor_bytes(t, d) for _, t, d, _ in sel)
    print(f"splicing {len(sel)} tensors ({total/1e9:.2f} GB) fmt={args.fmt} "
          f"set={args.which}", flush=True)

    if not os.path.exists(args.out):
        print(f"copying {args.src} -> {args.out} ...", flush=True)
        t0 = time.time()
        shutil.copyfile(args.src, args.out)
        print(f"  copied in {time.time()-t0:.0f}s", flush=True)
    else:
        print(f"OUT exists, patching in place: {args.out}", flush=True)

    src_sz, out_sz = os.path.getsize(args.src), os.path.getsize(args.out)
    assert src_sz == out_sz, (src_sz, out_sz)

    worst = (0.0, None)
    t0 = time.time()
    agg_se = 0.0
    agg_sref = 0.0
    agg_n = 0
    with open(args.src, "rb") as fin, open(args.out, "r+b") as fout:
        for i, (name, typ, dims, off) in enumerate(sel):
            nbytes = tensor_bytes(typ, dims)
            ngroups = nbytes // 33
            fin.seek(data_start + off)
            fout.seek(data_start + off)
            t_se = t_sref = 0.0
            t_max = 0.0
            done = 0
            while done < ngroups:
                g = min(CHUNK_GROUPS, ngroups - done)
                raw = np.frombuffer(fin.read(g * 33), dtype=np.uint8).reshape(g, 33)
                out, st = roundtrip_mxfp8_container(raw, args.fmt)
                fout.write(out.tobytes())
                t_se += st["rms_err"] ** 2 * g * 32
                t_sref += st["rms_ref"] ** 2 * g * 32
                t_max = max(t_max, st["max_abs_err"])
                done += g
            rel = (t_se / t_sref) ** 0.5 if t_sref else 0.0
            agg_se += t_se
            agg_sref += t_sref
            agg_n += ngroups * 32
            if rel > worst[0]:
                worst = (rel, name)
            print(f"  [{i+1}/{len(sel)}] {name:44s} {nbytes/1e6:9.1f} MB "
                  f"relRMS {rel*100:6.3f}% maxerr {t_max:.4g}", flush=True)
    rel_all = (agg_se / agg_sref) ** 0.5
    print(f"DONE in {time.time()-t0:.0f}s: weight-space relative RMS error "
          f"(all spliced) = {rel_all*100:.3f}%, worst tensor {worst[1]} "
          f"= {worst[0]*100:.3f}%", flush=True)

    if not args.no_sha:
        print("sha256...", flush=True)
        h = hashlib.sha256()
        with open(args.out, "rb") as f:
            while True:
                b = f.read(64 << 20)
                if not b:
                    break
                h.update(b)
        print(f"sha256({args.out}) = {h.hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
