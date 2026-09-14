#!/usr/bin/env python3
"""Repack Qwen with unchanged main/MTP tensors and original BF16 n-grams."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct

from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str

NGRAM = 'per_layer_token_embd.weight'
SOURCE_PREFIX = 'model.language_model.layers.1.ple.ple_embedding.'
BLOCKS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
          8: (32, 34), 10: (256, 84), 12: (256, 144), 16: (256, 66),
          27: (1, 8), 30: (1, 2), 39: (32, 17)}


def size_of(kind, dims):
    block, size = BLOCKS[kind]
    if not dims or any(d <= 0 for d in dims) or dims[0] % block:
        raise ValueError('Invalid tensor dimensions or block alignment')
    return math.prod(dims) // block * size


def source_layout(source, model):
    if source.is_file():
        native = Reader(str(source))
        try:
            if native.kv.get('general.architecture') != 'qwen4exp':
                raise ValueError('Expected a verified native n-gram Qwen GGUF')
            ngram_geometry(native)
            for key in ('row_dimension', 'row_count', 'layer_multipliers', 'head_offsets', 'head_vocab_sizes'):
                a = native.kv['qwen4exp.ple.'+key]
                b = model.kv.get('qwen4exp.ple.'+key, a)
                if getattr(a, 'values', a) != getattr(b, 'values', b):
                    raise ValueError('Native GGUF n-gram geometry differs from main model')
            kind, dims, offset = native.tensors[NGRAM]
            if kind != 30 or dims != [model.kv['qwen4exp.ple.row_dimension'],
                                      native.kv['qwen4exp.ple.row_count']]:
                raise ValueError('Expected original BF16 n-grams')
            size = size_of(kind, dims)
            if native.data_start + offset + size > source.stat().st_size:
                raise ValueError('Truncated native n-gram GGUF')
            return [(NGRAM, source, native.data_start+offset, size)], dims
        finally:
            native.f.close()
    index = json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    names = [n for n in index if n.startswith(SOURCE_PREFIX)]
    entries = {}
    for filename in sorted({index[n] for n in names}):
        path = source / filename
        if Path(filename).name != filename:
            raise ValueError('Source shard must be a filename')
        with path.open('rb') as fp:
            raw = fp.read(8)
            if len(raw) != 8:
                raise ValueError('Truncated safetensors header')
            header_size, = struct.unpack('<Q', raw)
            if header_size > min(path.stat().st_size - 8, 64 << 20):
                raise ValueError('Invalid safetensors header size')
            header = json.loads(fp.read(header_size))
        for name in names:
            if index[name] != filename:
                continue
            tensor = header[name]
            start, end = tensor['data_offsets']
            if not 0 <= start < end <= path.stat().st_size - 8 - header_size:
                raise ValueError('Invalid source tensor offsets')
            entries[name] = (path, 8 + header_size + start, end - start, tensor)
    shards = []
    marker = SOURCE_PREFIX + 'ngram_embedding.shard_'
    numbered = sorted((int(n[len(marker):].split('.')[0]), n)
                      for n in entries if n.startswith(marker) and n.endswith('.weight'))
    if not numbered or [i for i, n in numbered] != list(range(len(numbered))):
        raise ValueError('Missing or duplicate n-gram shards')
    width = model.kv['qwen4exp.ple.row_dimension']
    rows = 0
    for _, name in numbered:
        path, offset, size, tensor = entries[name]
        shape = tensor['shape']
        if tensor['dtype'] != 'BF16' or len(shape) != 2 or shape[0] <= 0 or shape[1] != width:
            raise ValueError('Expected original BF16 n-gram rows')
        if size != math.prod(shape) * 2:
            raise ValueError('Incorrect BF16 source payload size')
        rows += shape[0]
        shards.append((name, path, offset, size))
    if rows < ngram_geometry(model) or rows != model.kv.get('qwen4exp.ple.row_count', rows):
        raise ValueError('N-gram row count does not match main model')
    for source_key, gguf_key in [('layer_multipliers', 'layer_multipliers'),
                                  ('ngram_heads_offsets', 'head_offsets'),
                                  ('ngram_heads_vocab_sizes', 'head_vocab_sizes')]:
        path, offset, size, tensor = entries[SOURCE_PREFIX + source_key]
        expected = model.kv['qwen4exp.ple.' + gguf_key].values
        if tensor['dtype'] != 'I64' or tensor['shape'] != [len(expected)] or size != 8 * len(expected):
            raise ValueError('Invalid source n-gram hash metadata')
        with path.open('rb') as fp:
            fp.seek(offset)
            actual = struct.unpack('<' + 'q' * len(expected), fp.read(size))
        if list(actual) != expected:
            raise ValueError('Source n-gram hashes do not match main model')
    return shards, [width, rows]


def ngram_geometry(model):
    """Accept both upstream GGUF metadata and the older DS4 pack metadata."""
    width = model.kv.get('qwen4exp.embedding_length_per_layer_input',
                         model.kv.get('qwen4exp.ple.row_dimension'))
    offsets = model.kv['qwen4exp.ple.head_offsets'].values
    sizes = model.kv['qwen4exp.ple.head_vocab_sizes'].values
    if not width or width < 1 or width > 160 or not offsets or len(offsets) != len(sizes):
        raise ValueError('Invalid n-gram geometry')
    if any(o < 0 or s <= 0 for o, s in zip(offsets, sizes)):
        raise ValueError('Invalid n-gram hash ranges')
    rows = max(o+s for o, s in zip(offsets, sizes))
    if rows > 0xffffffff:
        raise ValueError('N-gram row count exceeds runtime limits')
    name = 'qwen4exp.ple.row_dimension'
    if model.kv.get(name, width) != width:
        raise ValueError('Inconsistent n-gram row dimension')
    if name not in model.kv:
        model.kv[name], model.kv_types[name] = width, 4
    if not rows <= model.kv.get('qwen4exp.ple.row_count', rows) <= 0xffffffff:
        raise ValueError('Invalid n-gram row count')
    return rows


def copy_hash(src, dst, size):
    digest = hashlib.sha256()
    while size:
        data = src.read(min(size, 16 << 20))
        if not data:
            raise ValueError('Truncated tensor payload')
        if dst is not None:
            dst.write(data)
        digest.update(data)
        size -= len(data)
    return digest.hexdigest()


def repack(template, source, output, revision):
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Source revision must be an immutable 40-character commit SHA')
    pending = Path(str(output) + '.incomplete')
    if output.exists() or pending.exists():
        raise ValueError('Output already exists; choose a new path or inspect the incomplete file')
    model = Reader(str(template))
    try:
        if model.kv.get('general.architecture') != 'qwen4exp':
            raise ValueError('Expected an active qwen4exp GGUF')
        ngram_geometry(model)
        shards, dims = source_layout(source, model)
        if 'qwen4exp.ple.row_count' not in model.kv:
            model.kv['qwen4exp.ple.row_count'] = dims[1]
            model.kv_types['qwen4exp.ple.row_count'] = 10
        if source.is_file():
            native = Reader(str(source))
            try:
                if native.kv.get('ds4.qwen4.ngram.source_revision') != revision:
                    raise ValueError('Native n-gram source revision does not match')
            finally:
                native.f.close()
        metadata = {k: (model.kv_types[k], v) for k, v in model.kv.items()}
        metadata['ds4.qwen4.ngram.source_revision'] = (8, revision)
        metadata['ds4.qwen4.ngram.source_repository'] = (8, 'Qwen/Qwen3.8-Flash-Next')
        alignment = model.kv.get('general.alignment', 32)
        if alignment < 1 or alignment & (alignment - 1):
            raise ValueError('Invalid GGUF alignment')
        align = lambda n, a=alignment: (n + a - 1) // a * a
        plan, offset = [], 0
        for name, (kind, shape, old_offset) in model.tensors.items():
            if name == NGRAM:
                continue
            size = size_of(kind, shape)
            if model.data_start + old_offset + size > template.stat().st_size:
                raise ValueError('Main tensor exceeds input file')
            plan.append((name, kind, shape, offset, size, old_offset))
            offset = align(offset + size)

        prefix = b'GGUF' + struct.pack('<IQQ', 3, len(plan)+1, len(metadata))
        prefix += b''.join(kv_bytes(k, t, v) for k, (t, v) in metadata.items())

        def header(ngram_offset):
            data = bytearray(prefix)
            for name, kind, shape, pos in [(n, k, s, o) for n, k, s, o, _, _ in plan] + [(NGRAM, 30, dims, ngram_offset)]:
                data += w_str(name) + struct.pack('<I', len(shape))
                data += struct.pack('<'+'Q'*len(shape), *shape) + struct.pack('<IQ', kind, pos)
            return data

        data_start = align(len(header(0)))
        table_start = align(data_start + offset, max(65536, alignment))
        total = table_start + math.prod(dims) * 2
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free < total + (16 << 30):
            raise ValueError('Not enough disk space with a 16 GiB reserve')
        records = []
        with pending.open('xb') as dst:
            dst.write(header(table_start - data_start))
            for name, kind, shape, pos, size, old in plan:
                dst.seek(data_start + pos)
                model.f.seek(model.data_start + old)
                digest = copy_hash(model.f, dst, size)
                records.append(dict(name=name, offset=data_start+pos, bytes=size, sha256=digest))
            dst.seek(table_start)
            for name, path, pos, size in shards:
                print('Copying original BF16', name, flush=True)
                offset = dst.tell()
                with path.open('rb') as src:
                    src.seek(pos)
                    digest = copy_hash(src, dst, size)
                records.append(dict(name=name, offset=offset, bytes=size, sha256=digest))
            if dst.tell() != total:
                raise ValueError('Incorrect assembled size')
            dst.flush()
            os.fsync(dst.fileno())
        print('Verifying every output tensor', flush=True)
        with pending.open('rb') as check:
            for rec in records:
                check.seek(rec['offset'])
                if copy_hash(check, None, rec['bytes']) != rec['sha256']:
                    raise ValueError('Output payload verification failed: ' + rec['name'])
        reread = Reader(str(pending))
        try:
            if reread.tensors[NGRAM] != (30, dims, table_start-data_start):
                raise ValueError('Incorrect output n-gram header')
        finally:
            reread.f.close()
        with pending.open('rb') as fp:
            checksum = copy_hash(fp, None, total)
        report = dict(source_revision=revision, template=str(template), bytes=total,
                      sha256=checksum, tensors=records)
        Path(str(output)+'.json').write_text(json.dumps(report, indent=2)+'\n')
        pending.rename(output)
        print('Verified', output, total, checksum, flush=True)
        return report
    finally:
        model.f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True,
                        help='original HF checkpoint directory or a verified native n-gram GGUF')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args()
    try:
        repack(args.model, args.source, args.output, args.source_revision)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error)+'\n')


if __name__ == '__main__':
    main()
