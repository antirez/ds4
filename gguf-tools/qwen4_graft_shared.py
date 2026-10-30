#!/usr/bin/env python3
"""Graft a pruned checkpoint's healed shared experts into a qwen4exp GGUF.

azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound routes 5 experts per
token instead of 10 and compensates with a retrained ("healed") shared expert
that is 1280 wide instead of 640.  Its router and routed experts are the
original ones, so a stock qwen4exp GGUF only needs, for every trunk layer:

  blk.N.ffn_gate_shexp / ffn_up_shexp / ffn_down_shexp   healed, FP8 -> Q8_0
  blk.N.ffn_gate_inp_shexp                               healed, BF16 -> F32

plus expert_used_count and expert_shared_feed_forward_length from the
checkpoint's config.  Every other tensor, the MTP block (which keeps the stock
shared expert) and the BF16 n-gram table are copied byte for byte; the table
stays page-aligned after the resident weights.

Usage:
  python3 gguf-tools/qwen4_graft_shared.py \\
    --model gguf/Qwen3.8-Flash-Next-Q4.gguf \\
    --source ~/.cache/huggingface/hub/models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound/snapshots/<rev> \\
    --output gguf/Qwen3.8-Flash-Next-Q4-top5-healed.gguf

Needs numpy.  The output is written to <output>.incomplete, every tensor is
re-read and checked against its hash, and only then renamed into place.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct

import numpy as np

from qwen4_native_ngrams import NGRAM, copy_hash, size_of
from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str

F32, Q8_0 = 0, 8
HF_LAYER = 'model.language_model.layers.{}.mlp.'


def fp8_e4m3fn_table():
    """All 256 E4M3FN values (bias 7, no infinities, S.1111.111 is NaN)."""
    out = np.empty(256, np.float32)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        exp, man = (b >> 3) & 0xF, b & 0x7
        if exp == 0xF and man == 0x7:
            out[b] = np.nan
        elif exp == 0:
            out[b] = sign * (man / 8.0) * 2.0 ** -6
        else:
            out[b] = sign * (1.0 + man / 8.0) * 2.0 ** (exp - 7)
    return out


FP8 = fp8_e4m3fn_table()


class Safetensors:
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / 'model.safetensors.index.json').read_text())['weight_map']
        self.headers = {}

    def raw(self, name):
        shard = self.root / self.index[name]
        if shard not in self.headers:
            with shard.open('rb') as f:
                n, = struct.unpack('<Q', f.read(8))
                self.headers[shard] = (8 + n, json.loads(f.read(n)))
        base, header = self.headers[shard]
        entry = header[name]
        start, end = entry['data_offsets']
        with shard.open('rb') as f:
            f.seek(base + start)
            data = f.read(end - start)
        if len(data) != end - start:
            raise ValueError('Truncated tensor ' + name)
        return entry['dtype'], entry['shape'], data

    def f32(self, name):
        dtype, shape, data = self.raw(name)
        if dtype == 'BF16':
            arr = (np.frombuffer(data, np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dtype == 'F32':
            arr = np.frombuffer(data, np.float32)
        else:
            raise ValueError(f'{name}: unexpected dtype {dtype}')
        return arr.reshape(shape)

    def fp8_block(self, name):
        """Blockwise FP8 weight times weight_scale_inv, as float32 [out, in]."""
        dtype, shape, data = self.raw(name)
        if dtype != 'F8_E4M3':
            raise ValueError(f'{name}: expected F8_E4M3, found {dtype}')
        w = FP8[np.frombuffer(data, np.uint8)].reshape(shape)
        scale = self.f32(name + '_scale_inv')
        rows, cols = shape
        br, bc = math.ceil(rows / scale.shape[0]), math.ceil(cols / scale.shape[1])
        full = np.repeat(np.repeat(scale, br, axis=0), bc, axis=1)[:rows, :cols]
        out = w * full
        if not np.isfinite(out).all():
            raise ValueError(name + ': non-finite weights')
        return out


def quantize_q8_0(w):
    """ggml quantize_row_q8_0_ref over rows of w (blocks of 32 along the row)."""
    rows, cols = w.shape
    if cols % 32:
        raise ValueError('Q8_0 rows must be a multiple of 32')
    blocks = w.reshape(rows * cols // 32, 32).astype(np.float32)
    amax = np.abs(blocks).max(axis=1)
    d = amax / 127.0
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d != 0)
    v = blocks * inv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
    out = np.empty(len(blocks), dtype=[('d', '<f2'), ('q', 'i1', 32)])
    out['d'] = d.astype(np.float16)
    out['q'] = q
    return out.tobytes()


def dequantize_q8_0(data, rows, cols):
    blk = np.frombuffer(data, dtype=[('d', '<f2'), ('q', 'i1', 32)])
    return (blk['q'].astype(np.float32) * blk['d'].astype(np.float32)[:, None]).reshape(rows, cols)


def graft(template, source, output):
    pending = Path(str(output) + '.incomplete')
    if output.exists() or pending.exists():
        raise ValueError('Output already exists; choose a new path or inspect the incomplete file')
    config = json.loads((source / 'config.json').read_text())
    text = config.get('text_config', config)
    used = int(text['num_experts_per_tok'])
    shared_ff = int(text['shared_expert_intermediate_size'])
    n_trunk = int(text['num_hidden_layers'])
    st = Safetensors(source)
    model = Reader(str(template))
    try:
        if model.kv.get('general.architecture') != 'qwen4exp':
            raise ValueError('Expected an active qwen4exp GGUF')
        if NGRAM not in model.tensors:
            raise ValueError('Template lacks the BF16 n-gram table')
        n_embd = model.kv['qwen4exp.embedding_length']
        metadata = {k: (model.kv_types[k], v) for k, v in model.kv.items()}
        metadata['qwen4exp.expert_used_count'] = (4, used)
        metadata['qwen4exp.expert_shared_feed_forward_length'] = (4, shared_ff)
        metadata['ds4.qwen4.shared_expert.source_repository'] = (
            8, 'azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound')
        metadata['ds4.qwen4.shared_expert.source_revision'] = (8, source.name)

        replace = {}
        for il in range(n_trunk):
            hf = HF_LAYER.format(il)
            for part, gname, shape in (('gate_proj', 'ffn_gate_shexp', (n_embd, shared_ff)),
                                       ('up_proj', 'ffn_up_shexp', (n_embd, shared_ff)),
                                       ('down_proj', 'ffn_down_shexp', (shared_ff, n_embd))):
                replace[f'blk.{il}.{gname}.weight'] = (Q8_0, shape, (hf + f'shared_expert.{part}.weight',))
            replace[f'blk.{il}.ffn_gate_inp_shexp.weight'] = (F32, None, (hf + 'shared_expert_gate.weight',))
        missing = [n for n in replace if n not in model.tensors]
        if missing:
            raise ValueError('Template lacks ' + ', '.join(missing[:4]))

        alignment = model.kv.get('general.alignment', 32)
        align = lambda n, a=alignment: (n + a - 1) // a * a
        plan, offset = [], 0
        for name, (kind, shape, old_offset) in model.tensors.items():
            if name == NGRAM:
                continue
            if name in replace:
                new_kind, new_shape, _ = replace[name]
                kind, shape = new_kind, (new_shape or shape)
            size = size_of(kind, shape)
            plan.append((name, kind, tuple(shape), offset, size, old_offset))
            offset = align(offset + size)
        ng_kind, ng_shape, ng_old = model.tensors[NGRAM]
        ng_size = size_of(ng_kind, ng_shape)

        prefix = b'GGUF' + struct.pack('<IQQ', 3, len(plan) + 1, len(metadata))
        prefix += b''.join(kv_bytes(k, t, v) for k, (t, v) in metadata.items())

        def header(ngram_offset):
            data = bytearray(prefix)
            for name, kind, shape, pos in [(n, k, s, o) for n, k, s, o, _, _ in plan] + \
                                          [(NGRAM, ng_kind, tuple(ng_shape), ngram_offset)]:
                data += w_str(name) + struct.pack('<I', len(shape))
                data += struct.pack('<' + 'Q' * len(shape), *shape) + struct.pack('<IQ', kind, pos)
            return data

        data_start = align(len(header(0)))
        table_start = align(data_start + offset, max(65536, alignment))
        total = table_start + ng_size
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free < total + (16 << 30):
            raise ValueError('Not enough disk space with a 16 GiB reserve')

        records, worst = [], 0.0
        import hashlib
        with pending.open('xb') as dst:
            dst.write(header(table_start - data_start))
            for name, kind, shape, pos, size, old in plan:
                dst.seek(data_start + pos)
                if name in replace:
                    (hf_name,) = replace[name][2]
                    if kind == Q8_0:
                        w = st.fp8_block(hf_name)
                        if w.shape != (shape[1], shape[0]):
                            raise ValueError(f'{hf_name}: shape {w.shape} does not fit {shape}')
                        payload = quantize_q8_0(w)
                        err = np.abs(dequantize_q8_0(payload, *w.shape) - w).max() / max(np.abs(w).max(), 1e-12)
                        worst = max(worst, float(err))
                    else:
                        w = st.f32(hf_name).reshape(-1)
                        if w.size != math.prod(shape):
                            raise ValueError(f'{hf_name}: {w.size} values for {shape}')
                        payload = w.astype('<f4').tobytes()
                    if len(payload) != size:
                        raise ValueError(f'{name}: {len(payload)} bytes, expected {size}')
                    dst.write(payload)
                    digest = hashlib.sha256(payload).hexdigest()
                else:
                    model.f.seek(model.data_start + old)
                    digest = copy_hash(model.f, dst, size)
                records.append(dict(name=name, offset=data_start + pos, bytes=size, sha256=digest))
            dst.seek(table_start)
            print('Copying the BF16 n-gram table', flush=True)
            model.f.seek(model.data_start + ng_old)
            digest = copy_hash(model.f, dst, ng_size)
            records.append(dict(name=NGRAM, offset=table_start, bytes=ng_size, sha256=digest))
            if dst.tell() != total:
                raise ValueError('Incorrect assembled size')
            dst.flush()
            os.fsync(dst.fileno())
        print(f'Q8_0 worst relative error on the healed shared experts: {worst:.4f}', flush=True)
        print('Verifying every output tensor', flush=True)
        with pending.open('rb') as check:
            for rec in records:
                check.seek(rec['offset'])
                if copy_hash(check, None, rec['bytes']) != rec['sha256']:
                    raise ValueError('Output payload verification failed: ' + rec['name'])
        reread = Reader(str(pending))
        try:
            if reread.kv['qwen4exp.expert_used_count'] != used or \
               reread.tensors[NGRAM][2] != table_start - data_start:
                raise ValueError('Output header did not read back as written')
        finally:
            reread.f.close()
        report = dict(template=str(template), source=str(source), experts_used=used,
                      shared_ff=shared_ff, bytes=total, q8_0_worst_rel_error=worst, tensors=records)
        Path(str(output) + '.json').write_text(json.dumps(report, indent=2) + '\n')
        pending.rename(output)
        print('Verified', output, total, flush=True)
        return report
    finally:
        model.f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True, help='stock qwen4exp GGUF with BF16 n-grams')
    parser.add_argument('--source', type=Path, required=True, help='pruned checkpoint snapshot directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        graft(args.model, args.source.resolve(), args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
