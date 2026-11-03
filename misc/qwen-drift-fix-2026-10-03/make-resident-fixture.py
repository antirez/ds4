#!/usr/bin/env python3
"""Plan a bounded resident Qwen fixture; write gigabytes only with --write.

Keep the production 48-layer shapes and alias each layer's tensor data to the
corresponding tensor of original layer n % 4. Preserve tokenizer, embedding,
output, and all copied tensor payloads. Replace PLE's hashed table with one
zero BF16 row. This synthetic fixture tests runtime arithmetic, not quality.
No NumPy, model inference, Metal calls, or source-file modifications.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct

HEADER_LIMIT = 64 << 20
COPY_CHUNK = 16 << 20
NGRAM = "per_layer_token_embd.weight"
LAYOUTS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
           8: (32, 34), 10: (256, 84), 12: (256, 144),
           16: (256, 66), 27: (1, 8), 30: (1, 2), 39: (32, 17)}
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
           6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def align(n, multiple):
    return (n + multiple - 1) // multiple * multiple


def string_bytes(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def record(key, kind, value):
    if kind == 8:
        body = string_bytes(value)
    elif kind == 9:
        subtype, items = value
        if subtype not in SCALARS:
            raise ValueError("only scalar override arrays are supported")
        body = struct.pack("<IQ", subtype, len(items))
        body += struct.pack("<" + SCALARS[subtype] * len(items), *items)
    else:
        body = struct.pack("<" + SCALARS[kind], value)
    return string_bytes(key) + struct.pack("<I", kind) + body


@dataclasses.dataclass(frozen=True)
class Tensor:
    name: str
    kind: int
    dims: tuple
    offset: int
    size: int


def read_header(path):
    with path.open("rb") as fp:
        identity = os.fstat(fp.fileno())
        limit = min(identity.st_size, HEADER_LIMIT)

        def take(n):
            if n < 0 or n > limit - fp.tell():
                raise ValueError("truncated or oversized GGUF header")
            data = fp.read(n)
            if len(data) != n:
                raise ValueError("truncated GGUF header")
            return data

        def unpack(fmt):
            return struct.unpack("<" + fmt, take(struct.calcsize("<" + fmt)))[0]

        def text_value():
            return take(unpack("Q")).decode("utf-8")

        def value(kind, retain):
            if kind == 8:
                return text_value()
            if kind in SCALARS:
                return unpack(SCALARS[kind])
            if kind != 9:
                raise ValueError("unsupported GGUF metadata type")
            subtype, count = unpack("I"), unpack("Q")
            if subtype == 8:
                if count > (limit - fp.tell()) // 8:
                    raise ValueError("oversized string array")
                for _ in range(count):
                    text_value()
                return None
            if subtype not in SCALARS:
                raise ValueError("unsupported GGUF array type")
            raw = take(count * struct.calcsize("<" + SCALARS[subtype]))
            return (subtype, list(struct.unpack("<" + SCALARS[subtype] * count, raw))) if retain else None

        if take(4) != b"GGUF" or unpack("I") != 3:
            raise ValueError("expected GGUF v3")
        nt, nk = unpack("Q"), unpack("Q")
        if not 0 < nt < 100000 or not 0 < nk < 10000:
            raise ValueError("invalid directory count")
        metadata, records = {}, {}
        for _ in range(nk):
            start = fp.tell()
            key = text_value()
            if key in records:
                raise ValueError("duplicate metadata key")
            metadata[key] = value(unpack("I"), key.startswith("qwen4exp."))
            end = fp.tell()
            fp.seek(start)
            records[key] = take(end - start)
        tensors = {}
        for _ in range(nt):
            name, rank = text_value(), unpack("I")
            if name in tensors or not 1 <= rank <= 4:
                raise ValueError("invalid tensor name or rank")
            dims = tuple(unpack("Q") for _ in range(rank))
            kind, offset = unpack("I"), unpack("Q")
            if kind not in LAYOUTS or any(d <= 0 for d in dims):
                raise ValueError("unsupported tensor: " + name)
            block, width = LAYOUTS[kind]
            if dims[0] % block:
                raise ValueError("unaligned tensor: " + name)
            tensors[name] = Tensor(name, kind, dims, offset, math.prod(dims) // block * width)
        alignment = metadata.get("general.alignment", 32)
        if not isinstance(alignment, int) or alignment < 32 or alignment > 1 << 20 or alignment & (alignment - 1):
            raise ValueError("invalid source alignment")
        data_start = align(fp.tell(), alignment)
        for t in tensors.values():
            if t.offset % alignment or data_start + t.offset + t.size > identity.st_size:
                raise ValueError("invalid source tensor range: " + t.name)
        return metadata, records, tensors, data_start, identity


def make_plan(source, output):
    meta, records, tensors, source_data, identity = read_header(source)
    if meta.get("general.architecture") != "qwen4exp" or meta.get("qwen4exp.embedding_length") != 2560:
        raise ValueError("expected production-width qwen4exp")
    if meta.get("qwen4exp.block_count") not in (48, 49):
        raise ValueError("expected 48 trunk layers, optionally with MTP")
    if NGRAM not in tensors or tensors[NGRAM].kind != 30 or tensors[NGRAM].dims[0] != 160:
        raise ValueError("expected native BF16 PLE with row width160")
    ngram_size = meta.get("qwen4exp.ple.ngram_size")
    heads_per_ngram = meta.get("qwen4exp.ple.heads_per_ngram")
    if not isinstance(ngram_size, int) or not isinstance(heads_per_ngram, int):
        raise ValueError("missing PLE head geometry")
    # Runtime macro DS4_N_PLE_HEADS = (N_PLE_NGRAM - 1) * HEADS_PER_NGRAM.
    # Only bi-grams and tri-grams have embeddings: the unigram is not a head.
    n_heads = (ngram_size - 1) * heads_per_ngram
    if not 0 < n_heads <= 64:
        raise ValueError("invalid PLE head geometry")
    for key in ("qwen4exp.ple.head_offsets", "qwen4exp.ple.head_vocab_sizes"):
        if not isinstance(meta.get(key), tuple) or len(meta[key][1]) != n_heads:
            raise ValueError("source PLE head array has wrong length: " + key)
    alignment = max(16384, int(os.sysconf("SC_PAGE_SIZE")))
    if alignment & (alignment - 1):
        raise ValueError("unsupported system page size")
    overrides = {
        "general.alignment": (4, alignment),
        "general.name": (8, "Qwen resident arithmetic fixture: 48 layers, four aliased real blocks, zero PLE"),
        "qwen4exp.block_count": (4, 48),
        "qwen4exp.nextn_predict_layers": (4, 0),
        "qwen4exp.ple.head_offsets": (9, (10, [0] * n_heads)),
        "qwen4exp.ple.head_vocab_sizes": (9, (10, [1] * n_heads)),
        "qwen4exp.ple.row_count": (10, 1),
    }
    for key, (kind, val) in overrides.items():
        records[key] = record(key, kind, val)
    # Retain required leaf names from all trunk layers; no extra PLE layer
    # names, no nextn block, no image-projector tensors.
    aliases = []
    for t in tensors.values():
        match = re.fullmatch(r"blk\.(\d+)\.(.+)", t.name)
        if match:
            layer, leaf = int(match[1]), match[2]
            if layer >= 48:
                continue
            original = "blk." + str(layer % 4) + "." + leaf
            if original not in tensors:
                raise ValueError("missing prototype tensor: " + original)
            src = tensors[original]
            if (t.kind, t.dims, t.size) != (src.kind, src.dims, src.size):
                raise ValueError("prototype geometry differs: " + t.name)
            aliases.append((t.name, original))
        elif t.name != NGRAM:
            if t.name not in ("token_embd.weight", "output.weight", "output_hc_norm.weight",
                              "output_hc_down.weight", "output_hc_up.weight"):
                raise ValueError("unexpected global tensor: " + t.name)
            aliases.append((t.name, t.name))
    payloads, offsets, cursor = [], {}, 0
    for original in dict.fromkeys(original for _, original in aliases):
        t = tensors[original]
        cursor = align(cursor, alignment)
        offsets[original] = cursor
        payloads.append({"source_tensor": original, "source_absolute_offset": source_data + t.offset,
                         "output_relative_offset": cursor, "bytes": t.size,
                         "kind": t.kind, "dims": list(t.dims)})
        cursor += t.size
    cursor = align(cursor, alignment)
    ngram_offset = cursor
    directory = bytearray()
    alias_manifest = []
    for name, original in aliases:
        t = tensors[original]
        directory += string_bytes(name) + struct.pack("<I", len(t.dims))
        directory += struct.pack("<" + "Q" * len(t.dims), *t.dims)
        directory += struct.pack("<IQ", t.kind, offsets[original])
        alias_manifest.append({"tensor": name, "source_tensor": original,
                               "output_relative_offset": offsets[original]})
    directory += string_bytes(NGRAM) + struct.pack("<IQQIQ", 2, 160, 1, 30, ngram_offset)
    header = b"GGUF" + struct.pack("<IQQ", 3, len(aliases) + 1, len(records))
    header += b"".join(records.values()) + directory
    target_data = align(len(header), alignment)
    header += bytes(target_data - len(header))
    manifest = {
        "fixture_kind": "synthetic repeated-real-weight resident arithmetic fixture",
        "quality_model": False,
        "source": str(source.resolve()), "output": str(output.resolve()),
        "source_bytes": identity.st_size, "source_mtime_ns": identity.st_mtime_ns,
        "source_device": identity.st_dev, "source_inode": identity.st_ino,
        "source_data_start": source_data, "output_data_start": target_data,
        "header_sha256": hashlib.sha256(header).hexdigest(), "alignment": alignment,
        "output_file_bytes": target_data + ngram_offset + 320,
        "resident_mapping_bytes": target_data + ngram_offset,
        "copied_payload_bytes": sum(p["bytes"] for p in payloads),
        "native_ngram_bytes": 320, "layers": 48, "unique_original_layers": [0, 1, 2, 3],
        "ple_heads": n_heads,
        "nextn_predict_layers": 0,
        "metadata_overrides": {key: {"kind": kind, "value": val} for key, (kind, val) in overrides.items()},
        "payloads": payloads, "aliases": alias_manifest,
        "status": "planned; no GGUF output written",
        "limitations": ["Real layers repeat modulo four; this is not the original full model.",
                        "PLE hash ranges select one zero row; predictor is absent.",
                        "Use only for same-fixture base/fix arithmetic comparisons.",
                        "GGUF tensor aliases are supported by ds4 but some external readers reject overlaps.",
                        "No guarantee of M4/Q2 numerical parity from M1/Q4 fixture results."],
    }
    return header, manifest


def source_unchanged(source, manifest):
    s = source.stat()
    return (s.st_size, s.st_mtime_ns, s.st_dev, s.st_ino) == (
        manifest["source_bytes"], manifest["source_mtime_ns"],
        manifest["source_device"], manifest["source_inode"])


def write_fixture(source, output, header, manifest):
    partial = output.with_name(output.name + ".incomplete")
    if output.exists() or partial.exists():
        raise ValueError("output already exists; choose a new output path")
    if source.resolve() == output.resolve() or not source_unchanged(source, manifest):
        raise ValueError("source identity changed or output equals source")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < manifest["output_file_bytes"] + (256 << 20):
        raise ValueError("insufficient free disk space for fixture plus 256MiB reserve")
    with source.open("rb") as src, partial.open("xb") as dst:
        dst.write(header)
        for p in manifest["payloads"]:
            dst.seek(manifest["output_data_start"] + p["output_relative_offset"])
            src.seek(p["source_absolute_offset"])
            remaining, digest = p["bytes"], hashlib.sha256()
            while remaining:
                raw = src.read(min(COPY_CHUNK, remaining))
                if not raw:
                    raise ValueError("truncated source tensor")
                dst.write(raw)
                digest.update(raw)
                remaining -= len(raw)
            p["copied_payload_sha256"] = digest.hexdigest()
            print("copied", p["source_tensor"], p["bytes"], flush=True)
        dst.seek(manifest["resident_mapping_bytes"])
        dst.write(bytes(320))
        dst.flush()
        os.fsync(dst.fileno())
    if not source_unchanged(source, manifest) or partial.stat().st_size != manifest["output_file_bytes"]:
        raise ValueError("source changed or output size incorrect; incomplete file retained")
    # Atomic no-clobber publication: another writer must not be overwritten
    # after the initial existence check. Both names are in the same directory.
    os.link(partial, output)
    partial.unlink()
    manifest["status"] = "written; payload hashes recorded; not yet loaded or GPU-tested"


def repair_header(source, output, header, manifest, manifest_path):
    """Repair a metadata-only mistake without copying the tensor payloads."""
    previous = json.loads(manifest_path.read_text())
    for key in ("source", "output", "source_bytes", "source_mtime_ns", "source_device",
                "source_inode", "output_data_start", "output_file_bytes",
                "resident_mapping_bytes", "copied_payload_bytes", "alignment", "aliases"):
        if previous[key] != manifest[key]:
            raise ValueError("repair would change data layout or source identity: " + key)
    if len(header) != previous["output_data_start"] or not source_unchanged(source, previous):
        raise ValueError("header repair would move tensor data or source changed")
    old_payloads = {p["source_tensor"]: p for p in previous["payloads"]}
    for p in manifest["payloads"]:
        old = old_payloads[p["source_tensor"]]
        if any(old[k] != p[k] for k in p):
            raise ValueError("repair would change tensor layout: " + p["source_tensor"])
        if "copied_payload_sha256" not in old:
            raise ValueError("repair requires a written payload manifest")
        p["copied_payload_sha256"] = old["copied_payload_sha256"]
    if not output.is_file() or output.stat().st_size != previous["output_file_bytes"]:
        raise ValueError("fixture missing or size differs from written manifest")
    with output.open("r+b") as fp:
        old_header = fp.read(previous["output_data_start"])
        if hashlib.sha256(old_header).hexdigest() != previous["header_sha256"]:
            raise ValueError("fixture header changed since written manifest")
        fp.seek(0)
        fp.write(header)
        fp.flush()
        os.fsync(fp.fileno())
        fp.seek(0)
        if hashlib.sha256(fp.read(len(header))).hexdigest() != manifest["header_sha256"]:
            raise ValueError("repaired header verification failed")
    manifest["previous_header_sha256"] = previous["header_sha256"]
    manifest["status"] = "header repaired; tensor data unchanged; not yet loaded or GPU-tested"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    mutation = parser.add_mutually_exclusive_group()
    mutation.add_argument("--write", action="store_true", help="explicitly copy gigabytes into output")
    mutation.add_argument("--repair-header", action="store_true", help="repair existing header without moving tensors")
    args = parser.parse_args()
    header, manifest = make_plan(args.source, args.output)
    if args.manifest.resolve() in (args.source.resolve(), args.output.resolve()):
        raise ValueError("manifest must not overwrite source or output")
    if args.write:
        write_fixture(args.source, args.output, header, manifest)
    elif args.repair_header:
        repair_header(args.source, args.output, header, manifest, args.manifest)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "resident_mapping_bytes",
                                                   "output_file_bytes", "copied_payload_bytes")}, indent=2))


if __name__ == "__main__":
    main()
