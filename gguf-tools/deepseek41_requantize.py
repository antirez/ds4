#!/usr/bin/env python3
"""Experimentally requantize V4.1 Q8_0 attention using an APFS copy-on-write clone.

This adds quantization error to already quantized weights. Prefer the original
safetensors converter for production artifacts. Only the five dense attention
projections change; expert and Engram bytes retain their original offsets.
There is deliberately no full-copy fallback when filesystem cloning fails.
"""

import argparse
import ctypes
import dataclasses
import hashlib
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

from deepseek41_quantize import (ATTENTION_SUFFIXES, QUANTIZATION, AttentionImatrix,
                                NativeQuantizer, attention_importance,
                                quantization_recipe)
from glm53_quantize import (TensorPlan, QTYPE_Q8_0, QTYPE_Q4_K, align, kv_string,
                            qtype_nbytes, read_exact, read_u32, read_u64,
                            read_gguf_string, skip_gguf_value, tensor_header)


@dataclasses.dataclass
class Header:
    records: dict
    metadata: dict
    tensors: list
    data_start: int
    alignment: int
    size: int
    raw: bytes


def read_header(fp):
    """Read and bounds-check the directory, never map/read model payloads."""
    fp.seek(0)
    size = os.fstat(fp.fileno()).st_size
    if read_exact(fp, 4, "magic") != b"GGUF" or read_u32(fp, "version") != 3:
        raise ValueError("expected GGUF v3")
    count, nmeta = read_u64(fp, "tensor count"), read_u64(fp, "metadata count")
    if not 1 <= count <= 10000 or not 1 <= nmeta <= 10000:
        raise ValueError("unsupported GGUF directory size")
    records, metadata = {}, {}
    for _ in range(nmeta):
        start = fp.tell()
        key = read_gguf_string(fp, "metadata key")
        kind = read_u32(fp, "metadata type")
        if key in records:
            raise ValueError(f"duplicate metadata {key}")
        if kind == 8 and not key.startswith("tokenizer."):
            metadata[key] = read_gguf_string(fp, key)
        elif kind == 4:
            metadata[key] = read_u32(fp, key)
        else:
            skip_gguf_value(fp, kind)
        end = fp.tell()
        if end > min(size, 64 << 20):
            raise ValueError("GGUF header exceeds 64 MiB or file size")
        fp.seek(start)
        records[key] = read_exact(fp, end - start, key)
    alignment = metadata.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment < 32 or alignment > 1 << 20 or alignment & (alignment - 1):
        raise ValueError("invalid GGUF alignment")
    tensors, names = [], set()
    for _ in range(count):
        name = read_gguf_string(fp, "tensor name")
        rank = read_u32(fp, "rank")
        if rank < 1 or rank > 4 or name in names:
            raise ValueError(f"invalid tensor directory entry {name}")
        names.add(name)
        shape = tuple(read_u64(fp, "dimension") for _ in range(rank))
        kind, offset = read_u32(fp, "tensor type"), read_u64(fp, "tensor offset")
        if not all(shape) or offset % alignment:
            raise ValueError(f"invalid tensor dimensions/offset {name}")
        try:
            nbytes = qtype_nbytes(kind, shape)
        except KeyError as error:
            raise ValueError(f"unsupported tensor type {kind}") from error
        tensors.append(TensorPlan(name, shape, kind, "", offset=offset, nbytes=nbytes))
        if fp.tell() > min(size, 64 << 20):
            raise ValueError("GGUF header exceeds 64 MiB or file size")
    data_start = align(fp.tell(), alignment)
    previous = 0
    for item in sorted(tensors, key=lambda t: t.offset):
        if item.offset < previous or data_start + item.offset + item.nbytes > size:
            raise ValueError(f"overlapping/out-of-file tensor {item.name}")
        previous = item.offset + item.nbytes
    fp.seek(0)
    raw = read_exact(fp, data_start, "GGUF header")
    return Header(records, metadata, tensors, data_start, alignment, size, raw)


def plan_requantization(header, attention_path=None):
    if header.metadata.get("general.architecture") != "deepseek41":
        raise ValueError("expected deepseek41 architecture")
    layers = header.metadata.get("deepseek41.num_hidden_layers")
    if not isinstance(layers, int) or not 1 <= layers <= 128:
        raise ValueError("invalid V4.1 layer count")
    recipe = header.metadata.get("deepseek41.quantization")
    quant = next((key for key, value in QUANTIZATION.items() if value == recipe), None)
    if quant is None:
        raise ValueError("expected an original Q8_0 attention recipe")
    by_name = {item.name: item for item in header.tensors}
    selected = {}
    for layer in range(layers):
        for suffix in ATTENTION_SUFFIXES:
            name = f"blk.{layer}.{suffix}"
            item = by_name.get(name)
            if item is None or item.qtype != QTYPE_Q8_0 or len(item.shape) != 2 or item.shape[0] % 256:
                raise ValueError(f"expected a Q8_0 matrix with 256-aligned rows: {name}")
            selected[name] = dataclasses.replace(item, qtype=QTYPE_Q4_K,
                                                 nbytes=qtype_nbytes(QTYPE_Q4_K, item.shape))
    records = dict(header.records)
    updates = {
        "deepseek41.quantization": quantization_recipe(quant, "q4_k"),
        "deepseek41.attention_calibration": "imatrix" if attention_path else "uncalibrated",
        "deepseek41.attention_source": "requantized Q8_0; original offsets retained",
        "deepseek41.attention_source_header_sha256": hashlib.sha256(header.raw).hexdigest(),
    }
    if attention_path:
        updates["deepseek41.attention_imatrix.file"] = os.path.basename(attention_path)
    else:
        records.pop("deepseek41.attention_imatrix.file", None)
    for key, value in updates.items():
        records[key] = kv_string(key, value)
    raw = b"GGUF" + struct.pack("<IQQ", 3, len(header.tensors), len(records))
    raw += b"".join(records.values())
    raw += b"".join(tensor_header(selected.get(item.name, item)) for item in header.tensors)
    # The GGUF reader derives data_start from the aligned directory length.
    # Moving that boundary would silently reinterpret every unchanged offset.
    if align(len(raw), header.alignment) != header.data_start:
        raise ValueError("insufficient original header padding for provenance; refusing to move payloads")
    return selected, raw + bytes(header.data_start - len(raw))


def clone_file(source, target):
    if sys.platform != "darwin":
        raise OSError("this mode requires macOS clonefile; no full-copy fallback")
    libc = ctypes.CDLL(None, use_errno=True)
    clone = libc.clonefile
    clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    clone.restype = ctypes.c_int
    if clone(os.fsencode(source), os.fsencode(target), 1) != 0:  # CLONE_NOFOLLOW
        error = ctypes.get_errno()
        raise OSError(error, "clonefile failed; no full-copy fallback", str(target))


def encoded_chunks(source, header, original, target, quantizer, importance):
    width, rows = original.shape
    blocks = width // 32
    dtype = quantizer.np.dtype([("d", "<f2"), ("q", "i1", (32,))])
    source.seek(header.data_start + original.offset)
    for row in range(0, rows, 128):
        count = min(128, rows - row)
        packed = read_exact(source, count * blocks * 34, original.name)
        values = quantizer.np.frombuffer(packed, dtype=dtype).reshape(count, blocks)
        values = values["d"].astype("float32")[:, :, None] * values["q"].astype("float32")
        values = values.reshape(count, width)
        if not quantizer.np.all(quantizer.np.isfinite(values)):
            raise ValueError(f"nonfinite Q8_0 weight in {original.name}")
        data = quantizer.encode(values, target.qtype, importance)
        expected = count * qtype_nbytes(target.qtype, (width,))
        if len(data) != expected:
            raise ValueError(f"incorrect encoded size for {original.name}")
        yield data


def requantize(args):
    source_path, output_path = Path(args.source_gguf).resolve(), Path(args.out).absolute()
    if output_path.exists() or output_path.is_symlink():
        raise ValueError(f"refusing to overwrite {output_path}")
    with source_path.open("rb") as source:
        identity = os.fstat(source.fileno())
        header = read_header(source)
        selected, replacement = plan_requantization(header, args.attention_imatrix)
        imatrix = None
        if args.attention_imatrix:
            import numpy as np
            imatrix = AttentionImatrix(args.attention_imatrix, np)
        importance = {name: attention_importance(imatrix, item) for name, item in selected.items()}
        changed_bytes = sum(item.nbytes for item in selected.values())
        print(f"Q4 attention: {len(selected)} tensors, {changed_bytes} bytes; logical file size remains {header.size}")
        print("Only changed extents consume new space; this is a lossy Q8_0 requantization.")
        if args.dry_run:
            return
        quantizer = NativeQuantizer(args.quants_library)
        reserve = 2 << 30
        required = header.data_start + sum(align(item.nbytes, 1 << 20) for item in selected.values()) + reserve
        if shutil.disk_usage(output_path.parent).free < required:
            raise ValueError(f"insufficient space for changed extents plus 2 GiB reserve ({required} bytes)")
        with tempfile.TemporaryDirectory(prefix=".ds41-q4-", dir=output_path.parent) as tmp:
            partial = Path(tmp) / "model.gguf"
            clone_file(source_path, partial)
            with partial.open("r+b") as target:
                if read_header(target).raw != header.raw or os.fstat(target.fileno()).st_size != header.size:
                    raise ValueError("source changed while cloning")
                for original in header.tensors:
                    item = selected.get(original.name)
                    if item is None:
                        continue
                    if shutil.disk_usage(output_path.parent).free < reserve:
                        raise ValueError("free space fell below the 2 GiB reserve")
                    target.seek(header.data_start + item.offset)
                    for data in encoded_chunks(source, header, original, item, quantizer, importance[item.name]):
                        if target.write(data) != len(data):
                            raise OSError("short output write")
                    if target.tell() != header.data_start + item.offset + item.nbytes:
                        raise ValueError(f"incorrect output extent for {item.name}")
                    print(f"converted {item.name}", flush=True)
                target.seek(0)
                target.write(replacement)
                target.flush()
                os.fsync(target.fileno())
                if read_header(target).raw != replacement:
                    raise ValueError("written header failed verification")
            after = os.stat(source_path)
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (identity.st_dev, identity.st_ino, identity.st_size, identity.st_mtime_ns):
                raise ValueError("source changed during conversion")
            # link() publishes without replacing a destination created by another
            # process after our initial check; TemporaryDirectory drops its alias.
            os.link(partial, output_path)
    print(f"wrote {output_path}")


def check_requantization(args):
    """Verify the directory and all changed weights against the source Q8 file.

    Untouched payloads are preserved by clonefile; this bounded audit deliberately
    does not reread hundreds of GiB of expert/Engram data.
    """
    quantizer = NativeQuantizer(args.quants_library)
    imatrix = AttentionImatrix(args.attention_imatrix, quantizer.np) if args.attention_imatrix else None
    with open(args.source_gguf, "rb") as source, open(args.out, "rb") as target:
        header, actual = read_header(source), read_header(target)
        selected, replacement = plan_requantization(header, args.attention_imatrix)
        if actual.raw != replacement or actual.size != header.size:
            raise ValueError("requantized header/file size does not match source plan")
        for original in header.tensors:
            item = selected.get(original.name)
            if item is None:
                continue
            importance = attention_importance(imatrix, item)
            target.seek(header.data_start + item.offset)
            for expected in encoded_chunks(source, header, original, item, quantizer, importance):
                if read_exact(target, len(expected), item.name) != expected:
                    raise ValueError(f"requantized payload differs for {item.name}")
    print(f"PASS: original offsets/provenance and all {len(selected)} Q4 attention payloads")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gguf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--attention-imatrix")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true", help="audit an existing output against source Q8 attention")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument("--quants-library", default=str(Path(__file__).with_name(f"libds4quants.{suffix}")))
    args = parser.parse_args()
    if args.check and args.dry_run:
        parser.error("--check and --dry-run are mutually exclusive")
    try:
        if args.check:
            check_requantization(args)
        else:
            requantize(args)
    except (OSError, ValueError) as error:
        sys.exit(f"deepseek41-requantize: {error}")


if __name__ == "__main__":
    main()
