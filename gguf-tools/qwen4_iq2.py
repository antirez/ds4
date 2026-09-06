#!/usr/bin/env python3
"""Build calibrated IQ2_XXS trunk gate/up tensors, then insert into a Qwen GGUF.

The two stages allow quantization beside the original BF16 checkpoint and
assembly beside an existing qwen4exp model. All other tensors are byte-copied.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time

from qwen4_pack import (SourceDB, QwenGGUFImatrix, GGMLQuantizer,
                        Q2_IMATRIX_SHA256, encode_weighted_experts)
from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str, tnbytes


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def quantize(args):
    args.experts_dir.mkdir(parents=True, exist_ok=True)
    db = SourceDB(args.source)
    imatrix = QwenGGUFImatrix(args.imatrix, Q2_IMATRIX_SHA256)
    quant = GGMLQuantizer(args.library)
    quant.lib.ds4q_quantize_init(16)  # Initialize shared lookup tables before workers.
    manifest = args.experts_dir / 'experts.json'
    identity = {'source': str(args.source.resolve()), 'imatrix': imatrix.provenance(),
                'library_sha256': digest(args.library), 'format': 'IQ2_XXS'}
    report = json.loads(manifest.read_text()) if manifest.exists() else {**identity, 'tensors': {}}
    for key, value in identity.items():
        if report[key] != value:
            raise SystemExit(f'Resume provenance mismatch: {key}')
    for layer in range(48):
        source = f'model.language_model.layers.{layer}.mlp.experts.gate_up_proj'
        info = db.tensors[source]
        if info['dtype'] != 'BF16' or info['shape'] != (512, 1280, 2560):
            raise SystemExit(f'Unexpected source layout: {source}: {info}')
        names = [f'blk.{layer}.ffn_{part}_exps.weight' for part in ('gate', 'up')]
        if all(n in report['tensors'] and (args.experts_dir / n).is_file() and
               digest(args.experts_dir / n) == report['tensors'][n]['sha256'] for n in names):
            print(f'Layer {layer}: verified existing tensors', flush=True)
            continue
        values = db.read(source)
        source_hash = hashlib.sha256(values).hexdigest()
        for part, name in enumerate(names):
            start = time.monotonic()
            raw = encode_weighted_experts(values[:, part * 640:(part + 1) * 640, :],
                                         'IQ2_XXS', imatrix, name, quant, args.threads)
            if len(raw) != 512 * 640 * 10 * 66:
                raise SystemExit(f'Invalid output size for {name}')
            path = args.experts_dir / name
            tmp = path.with_suffix('.incomplete')
            tmp.write_bytes(raw)
            tmp.replace(path)
            report['tensors'][name] = {'sha256': hashlib.sha256(raw).hexdigest(),
                'bytes': len(raw), 'source_tensor': source, 'source_sha256': source_hash,
                'seconds': time.monotonic() - start}
            save(manifest, report)
            print(f'{name}: {len(raw)} bytes, {time.monotonic() - start:.1f}s', flush=True)
        del values
    imatrix.close()


def assemble(args):
    if args.out.exists() or Path(str(args.out) + '.incomplete').exists():
        raise SystemExit('Output already exists; choose a new path')
    manifest = json.loads((args.experts_dir / 'experts.json').read_text())
    expected = {f'blk.{i}.ffn_{p}_exps.weight' for i in range(48) for p in ('gate', 'up')}
    if set(manifest['tensors']) != expected:
        raise SystemExit('Expected all 96 calibrated trunk tensors')
    base = Reader(str(args.template))
    if base.kv.get('general.architecture') != 'qwen4exp' or base.kv.get('qwen4exp.block_count') != 49:
        raise SystemExit('Template must be a combined 49-layer qwen4exp GGUF')
    if 'per_layer_token_embd.weight' in base.tensors:
        raise SystemExit('Template must use an external PLE sidecar')
    alignment = base.kv.get('general.alignment', 32)
    align = lambda n: (n + alignment - 1) // alignment * alignment
    metadata = {k: (base.kv_types[k], v) for k, v in base.kv.items()}
    metadata['general.name'] = (8, 'Qwen3.8 Flash Next IQ2_XXS imatrix trunk, MXFP4 down, MTP')
    metadata['general.file_type'] = (4, 19)  # LLAMA_FTYPE_MOSTLY_IQ2_XXS
    metadata['ds4.iq2.imatrix_sha256'] = (8, manifest['imatrix']['sha256'])
    plan = []
    offset = 0
    for name, (kind, dims, old_offset) in base.tensors.items():
        replace = name in expected
        if replace and (kind != 12 or dims != [2560, 640, 512]):
            raise SystemExit(f'Unexpected template expert layout: {name}')
        size = math.prod(dims) // 256 * 66 if replace else tnbytes(kind, math.prod(dims))
        if replace:
            path = args.experts_dir / name
            if path.stat().st_size != size or digest(path) != manifest['tensors'][name]['sha256']:
                raise SystemExit(f'Invalid calibrated payload: {name}')
        plan.append((name, 16 if replace else kind, dims, offset, size, old_offset))
        offset = align(offset + size)
    prefix = b'GGUF' + struct.pack('<IQQ', 3, len(plan), len(metadata))
    prefix += b''.join(kv_bytes(k, t, v) for k, (t, v) in metadata.items())
    for name, kind, dims, pos, size, old in plan:
        prefix += w_str(name) + struct.pack('<I', len(dims))
        prefix += struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', kind, pos)
    data_start = align(len(prefix))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(args.out) + '.incomplete')
    records = {}
    with tmp.open('wb') as out:
        out.write(prefix)
        out.write(bytes(data_start - len(prefix)))
        for name, kind, dims, pos, size, old in plan:
            out.seek(data_start + pos)
            src = (args.experts_dir / name).open('rb') if name in expected else base.f
            if src is base.f:
                src.seek(base.data_start + old)
            h = hashlib.sha256()
            left = size
            while left:
                chunk = src.read(min(left, 8 << 20))
                if not chunk:
                    raise SystemExit(f'Short tensor read: {name}')
                out.write(chunk)
                h.update(chunk)
                left -= len(chunk)
            if src is not base.f:
                src.close()
            records[name] = {'type': kind, 'bytes': size, 'sha256': h.hexdigest(),
                             'replaced': name in expected}
        out.truncate(data_start + offset)
    # Read back every payload before publishing the completed artifact.
    check = Reader(str(tmp))
    for name, rec in records.items():
        check.f.seek(check.data_start + check.tensors[name][2])
        h = hashlib.sha256()
        left = rec['bytes']
        while left:
            chunk = check.f.read(min(left, 8 << 20))
            if not chunk:
                raise SystemExit(f'Short verification read: {name}')
            h.update(chunk)
            left -= len(chunk)
        if h.hexdigest() != rec['sha256']:
            raise SystemExit(f'Written tensor mismatch: {name}')
    check.f.close()
    base.f.close()
    tmp.replace(args.out)
    save(Path(str(args.out) + '.json'), {'template': str(args.template.resolve()),
        'quantization': manifest, 'tensors': records, 'bytes': args.out.stat().st_size,
        'sha256': digest(args.out)})
    print(f'Verified {args.out}: {args.out.stat().st_size} bytes', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='stage', required=True)
    q = sub.add_parser('quantize')
    q.add_argument('--source', type=Path, required=True)
    q.add_argument('--imatrix', type=Path, required=True)
    q.add_argument('--library', type=Path, required=True)
    q.add_argument('--threads', type=int, default=8)
    a = sub.add_parser('assemble')
    a.add_argument('--template', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    for p in (q, a):
        p.add_argument('--experts-dir', type=Path, required=True)
    args = ap.parse_args()
    (quantize if args.stage == 'quantize' else assemble)(args)


if __name__ == '__main__':
    main()
