#!/usr/bin/env python3
"""Copy the first N layers of a MiniMax-M3.1-Flash GGUF into a validation GGUF.

The result keeps the embeddings, the output head and every blk.<i>.* tensor
with i < N, with minimax-m3.block_count set to N.  It exists to compare the
ds4 graph against minimax31_reference.py --layers N on machines that cannot
hold all 60 layers; it is not a usable model.
"""

import argparse
import os
import struct
import sys

import glm53_quantize as gq


def read_header(fp):
    if gq.read_exact(fp, 4, "magic") != b"GGUF":
        gq.fail("not a GGUF file")
    version = gq.read_u32(fp, "version")
    n_tensors = gq.read_u64(fp, "tensor count")
    n_kv = gq.read_u64(fp, "metadata count")
    records = []
    for _ in range(n_kv):
        start = fp.tell()
        key = gq.read_gguf_string(fp, "key")
        value_type = gq.read_u32(fp, "type")
        gq.skip_gguf_value(fp, value_type)
        end = fp.tell()
        fp.seek(start)
        records.append((key, value_type, gq.read_exact(fp, end - start, key)))
    tensors = []
    for _ in range(n_tensors):
        name = gq.read_gguf_string(fp, "tensor name")
        ndim = gq.read_u32(fp, "ndim")
        shape = struct.unpack(f"<{ndim}Q", gq.read_exact(fp, 8 * ndim, "shape"))
        qtype, offset = struct.unpack("<IQ", gq.read_exact(fp, 12, "type/offset"))
        tensors.append((name, shape, qtype, offset))
    alignment = gq.GGUF_ALIGNMENT
    for key, _, raw in records:
        if key == "general.alignment":
            alignment = struct.unpack("<I", raw[-4:])[0]
    data_offset = gq.align(fp.tell(), alignment)
    return version, records, tensors, data_offset, alignment


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, required=True)
    args = ap.parse_args()
    if args.layers < 4 or args.layers > 60:
        ap.error("--layers must be between 4 and 60")
    if os.path.exists(args.out):
        gq.fail(f"output exists: {args.out}")
    try:
        truncate(args)
    except OSError as error:
        raise ValueError(f"{args.src} -> {args.out}: {error}") from error
    os.replace(args.out + ".partial", args.out)
    print(f"minimax31-truncate: {args.layers} layers -> {args.out}", file=sys.stderr)


def truncate(args):
    with open(args.src, "rb") as src:
        version, records, tensors, data_offset, alignment = read_header(src)
        keep = [t for t in tensors
                if not t[0].startswith("blk.") or int(t[0].split(".")[1]) < args.layers]
        out_records = [gq.kv_u32("minimax-m3.block_count", args.layers) if key == "minimax-m3.block_count" else raw
                       for key, _, raw in records]
        sizes = [gq.qtype_nbytes(qtype, list(shape)) for _, shape, qtype, _ in keep]
        offsets, offset = [], 0
        for nbytes in sizes:
            offsets.append(offset)
            offset += gq.align(nbytes, alignment)
        header = b"GGUF" + struct.pack("<IQQ", version, len(keep), len(out_records)) + b"".join(out_records)
        for (name, shape, qtype, _), off in zip(keep, offsets, strict=True):
            header += (gq.pack_string(name) + struct.pack("<I", len(shape)) +
                       struct.pack(f"<{len(shape)}Q", *shape) + struct.pack("<IQ", qtype, off))
        out_data = gq.align(len(header), alignment)
        with open(args.out + ".partial", "wb") as dst:
            dst.write(header + bytes(out_data - len(header)))
            for (name, _, _, src_off), nbytes in zip(keep, sizes, strict=True):
                src.seek(data_offset + src_off)
                remaining = nbytes
                while remaining:
                    chunk = src.read(min(remaining, 64 << 20))
                    if not chunk:
                        gq.fail(f"short read in {name}")
                    dst.write(chunk)
                    remaining -= len(chunk)
                dst.write(bytes(gq.align(nbytes, alignment) - nbytes))


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(f"minimax31-truncate: error: {error}", file=sys.stderr)
        sys.exit(1)
