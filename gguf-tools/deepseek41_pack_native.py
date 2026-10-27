#!/usr/bin/env python3
"""Losslessly pack native V4.1 dense matrices in an APFS clone.

The original file is read-only. Tensor offsets stay fixed, so expert and Engram
payloads remain shared clone extents. Only matrices supported by the native
BF16 matvec are eligible; every converted value must round-trip bit for bit.
This requires macOS clonefile and the converter's existing NumPy dependency.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import time

import numpy as np

from glm53_quantize import align, read_u32, read_u64, read_gguf_string, skip_gguf_value
from glm53_validate_gguf import read_selected_metadata


DENSE = re.compile(r"(?:output|blk\.\d+\.(?:attn_q_a|attn_q_b|attn_kv|attn_output_a|attn_output_b|ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp))\.weight\Z")


def layout(fp):
    if fp.read(4) != b"GGUF" or read_u32(fp, "version") != 3:
        raise ValueError("expected GGUF v3")
    count, meta_count = read_u64(fp, "tensors"), read_u64(fp, "metadata")
    metadata, records = {}, []
    for _ in range(meta_count):
        start = fp.tell()
        name, kind = read_gguf_string(fp, "key"), read_u32(fp, "type")
        if name in {"general.alignment", "deepseek41.native_weights", "general.source.revision", "deepseek41.quantization"}:
            metadata[name] = read_selected_metadata(fp, kind)
        else:
            skip_gguf_value(fp, kind)
        end = fp.tell()
        fp.seek(start)
        records.append((name, fp.read(end - start)))
    tensor_start = fp.tell()
    tensors = []
    for _ in range(count):
        name, rank = read_gguf_string(fp, "tensor"), read_u32(fp, "rank")
        shape = [read_u64(fp, "dimension") for _ in range(rank)]
        type_pos = fp.tell() - tensor_start
        kind, offset = read_u32(fp, "tensor type"), read_u64(fp, "offset")
        tensors.append(dict(name=name, shape=shape, type=kind, offset=offset, type_pos=type_pos))
    end = fp.tell()
    fp.seek(tensor_start)
    tensor_header = bytearray(fp.read(end - tensor_start))
    if metadata.get("deepseek41.native_weights") is not True:
        raise ValueError("only an original-precision native artifact is eligible")
    return metadata, records, tensors, tensor_header, align(end, metadata["general.alignment"])


def identity(st):
    return dict(size=st.st_size, inode=st.st_ino, mtime_ns=st.st_mtime_ns)


def pack(source, output, report):
    started = time.monotonic()
    source, output, report = source.resolve(), output.resolve(), report.resolve()
    partial = output.with_name(output.name + ".partial")
    if len({source, output, partial, report}) != 4 or output.exists() or partial.exists() or report.exists():
        raise ValueError("source, output, partial file and report must be distinct; outputs must not exist")
    before = identity(source.stat())
    with source.open("rb") as original:
        metadata, records, tensors, tensor_header, data = layout(original)
        candidates = [t for t in tensors if DENSE.fullmatch(t['name']) and t['type'] == 0]
        if not candidates:
            raise ValueError("no native F32 dense matrices to pack")
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        clone = lib.clonefile
        clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clone.restype = ctypes.c_int
        if clone(os.fsencode(source), os.fsencode(partial), 0):
            raise OSError(ctypes.get_errno(), "clonefile failed; no full-file copy is attempted")
        converted, retained = [], []
        try:
            with partial.open("r+b") as target:
                for tensor in candidates:
                    count = math.prod(tensor['shape'])
                    start = data + tensor['offset']
                    if len(tensor['shape']) != 2 or tensor['shape'][0] < 32 or tensor['shape'][0] % 4:
                        raise ValueError("unsupported native matrix dimensions")
                    if start > before['size'] or count * 4 > before['size'] - start:
                        raise ValueError("tensor exceeds source file")
                    # First inspect all bits. No rounding or floating conversion.
                    exact = True
                    original.seek(start)
                    for off in range(0, count, 4 * 1024 * 1024):
                        n = min(count - off, 4 * 1024 * 1024)
                        raw = original.read(n * 4)
                        if len(raw) != n * 4:
                            raise ValueError("short source read")
                        bits = np.frombuffer(raw, dtype='<u4')
                        if np.any(bits & 0xffff) or np.any((bits & 0x7f800000) == 0x7f800000):
                            exact = False
                            break
                    if not exact:
                        retained.append(tensor['name'])
                        continue
                    source_hash, packed_hash = hashlib.sha256(), hashlib.sha256()
                    original.seek(start)
                    for off in range(0, count, 4 * 1024 * 1024):
                        n = min(count - off, 4 * 1024 * 1024)
                        raw = original.read(n * 4)
                        if len(raw) != n * 4:
                            raise ValueError("short source read")
                        bits = np.frombuffer(raw, dtype='<u4')
                        packed = (bits >> 16).astype('<u2').tobytes()
                        target.seek(start + off * 2)
                        if target.write(packed) != len(packed):
                            raise ValueError("short packed write")
                        target.flush()
                        target.seek(start + off * 2)
                        readback = target.read(len(packed))
                        restored = (np.frombuffer(readback, dtype='<u2').astype('<u4') << 16).tobytes()
                        if restored != raw:
                            raise ValueError(tensor['name'] + ": bitwise round-trip failed")
                        source_hash.update(raw)
                        packed_hash.update(readback)
                    struct.pack_into('<I', tensor_header, tensor['type_pos'], 30)
                    converted.append(dict(name=tensor['name'], values=count, source_bytes=count * 4,
                        packed_bytes=count * 2, original_f32_sha256=source_hash.hexdigest(),
                        packed_bf16_sha256=packed_hash.hexdigest(), roundtrip_bit_identical=True))
                    print(f"Verified {tensor['name']}: {count} exact values", flush=True)
                if not converted:
                    raise ValueError("no matrices have an exact BF16 representation")
                header = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(records)))
                for name, record in records:
                    if name == 'deepseek41.quantization':
                        key = name.encode()
                        value = b'released MXFP4 experts and Engram; lossless BF16/F32 dense weights'
                        record = struct.pack('<Q', len(key)) + key + struct.pack('<IQ', 8, len(value)) + value
                    header.extend(record)
                header.extend(tensor_header)
                if align(len(header), metadata['general.alignment']) != data:
                    raise ValueError("new metadata would move tensor data")
                header.extend(b'\0' * (data - len(header)))
                target.seek(0)
                if target.write(header) != len(header):
                    raise ValueError("short header write")
                target.flush()
                os.fsync(target.fileno())
                target.seek(0)
                _, _, actual, _, actual_data = layout(target)
                expected_types = {t['name']: (30 if t['name'] in {c['name'] for c in converted} else t['type']) for t in tensors}
                if actual_data != data or any((a['name'], a['shape'], a['offset'], a['type']) !=
                    (b['name'], b['shape'], b['offset'], expected_types[b['name']]) for a, b in zip(actual, tensors)):
                    raise ValueError("packed tensor directory readback differs")
            if identity(source.stat()) != before:
                raise ValueError("source file identity changed during packing")
            os.link(partial, output)  # fails if another writer created output
            partial.unlink()
        except BaseException:
            if partial.exists():
                partial.unlink()
            raise
    result = dict(source=str(source), output=str(output), source_identity=before,
        output_identity=identity(output.stat()), source_revision=metadata.get('general.source.revision'),
        converted=converted, retained_f32=retained, converted_matrices=len(converted),
        exact_values=sum(t['values'] for t in converted),
        resident_weight_bytes_saved=sum(t['source_bytes'] - t['packed_bytes'] for t in converted),
        all_converted_values_bit_identical=True, original_file_unchanged=True,
        unmodified_payloads='APFS clone extents; only listed dense payload ranges and header were written',
        disk_layout='Original offsets and logical file size retained; this is not a smaller-file claim',
        elapsed_seconds=time.monotonic() - started)
    with report.open('x') as fp:
        fp.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in {'converted', 'retained_f32'}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    pack(args.source, args.output, args.report)
