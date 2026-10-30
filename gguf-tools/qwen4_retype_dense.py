#!/usr/bin/env python3
"""Shrink the dense weights a Qwen3.8 decode step reads, in a copy of a qwen4exp GGUF.

On a DGX Spark, decode is limited by memory bandwidth, and the dense weights
dominate what each verify pass reads.  Two retypes keep quality close to the
input file:

  --hc-q8          the F16 hyper-connection projections (hc_*_down/up/inject,
                   output_hc_*, nextn.hc_head_*) become Q8_0, halving them.
  --head-gptq DIR  output.weight becomes Q4_0, converted exactly from the
                   AutoRound-tuned int4 lm_head of a GPTQ checkpoint
                   (symmetric, group 128, no act-order), e.g.
                   azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound.
                   Each 128-wide GPTQ group becomes four Q4_0 blocks that share
                   its fp16 scale, so no value changes.

Everything else, including the BF16 n-gram table, is copied byte for byte.
The output is written to <output>.incomplete, re-read against its hashes, and
renamed into place.  Needs numpy.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct

import numpy as np

from qwen4_graft_shared import Safetensors, quantize_q8_0
from qwen4_native_ngrams import NGRAM, copy_hash, size_of
from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str

F16, Q4_0, Q8_0 = 1, 2, 8
HC = re.compile(r'(blk\.\d+\.(hc_(attn|ffn)_(down|up|inject)|nextn\.hc_head_(down|up))|output_hc_(down|up))\.weight$')


def read_f16(model, offset, shape):
    model.f.seek(model.data_start + offset)
    n = math.prod(shape)
    return np.frombuffer(model.f.read(2 * n), '<f2').astype(np.float32).reshape(shape[::-1])


def gptq_head_q4_0(st, rows, cols):
    """lm_head GPTQ int4 (sym, g128) as Q4_0 rows [rows=vocab, cols=embd]."""
    _, qshape, qraw = st.raw('lm_head.qweight')
    _, sshape, sraw = st.raw('lm_head.scales')
    _, zshape, zraw = st.raw('lm_head.qzeros')
    if qshape != [cols // 8, rows] or sshape[1] != rows:
        raise ValueError(f'lm_head shapes {qshape} {sshape} do not fit [{rows}, {cols}]')
    group = cols // sshape[0]
    if group % 32:
        raise ValueError('GPTQ group size must be a multiple of 32')
    zeros = np.frombuffer(zraw, '<u4').reshape(zshape)
    zq = np.stack([(zeros >> (4 * k)) & 0xF for k in range(8)], axis=-1).reshape(zshape[0], -1)
    # symmetric int4 has zero point 8; GPTQ v1 packing stores it as 7 (zero - 1)
    if len(np.unique(zq)) != 1 or int(zq.flat[0]) not in (7, 8):
        raise ValueError('lm_head is not symmetric GPTQ with zero point 8')
    q = np.frombuffer(qraw, '<u4').reshape(cols // 8, rows)
    # unpack along the input dimension: int32 row i holds inputs 8i..8i+7, low nibble first
    q = np.stack([(q >> (4 * k)) & 0xF for k in range(8)], axis=1).reshape(cols, rows).T.astype(np.uint8)
    scales = np.frombuffer(sraw, '<f2').reshape(sshape).T            # [rows, groups]
    blocks = q.reshape(rows, cols // 32, 32)
    d = np.repeat(scales, group // 32, axis=1)                       # [rows, cols/32]
    out = np.empty((rows, cols // 32), dtype=[('d', '<f2'), ('qs', 'u1', 16)])
    out['d'] = d
    out['qs'] = blocks[:, :, :16] | (blocks[:, :, 16:] << 4)
    return out.tobytes(), (blocks.astype(np.float32) - 8) * d.astype(np.float32)[:, :, None]


def retype(template, output, hc_q8, head_src):
    pending = Path(str(output) + '.incomplete')
    if output.exists() or pending.exists():
        raise ValueError('Output already exists; choose a new path or inspect the incomplete file')
    model = Reader(str(template))
    st = Safetensors(head_src) if head_src else None
    try:
        if model.kv.get('general.architecture') != 'qwen4exp':
            raise ValueError('Expected an active qwen4exp GGUF')
        metadata = {k: (model.kv_types[k], v) for k, v in model.kv.items()}
        if hc_q8:
            metadata['ds4.qwen4.retype.hc'] = (8, 'F16->Q8_0')
        if head_src:
            metadata['ds4.qwen4.retype.output'] = (8, 'GPTQ int4 g128 -> Q4_0 from ' + head_src.name)

        change = {}
        for name, (kind, shape, _) in model.tensors.items():
            if hc_q8 and kind == F16 and HC.match(name):
                if shape[0] % 32:
                    raise ValueError(f'{name}: rows of {shape[0]} do not fit Q8_0 blocks')
                change[name] = Q8_0
            if head_src and name == 'output.weight':
                change[name] = Q4_0
        if not change:
            raise ValueError('Nothing to change: pass --hc-q8 and/or --head-gptq')

        alignment = model.kv.get('general.alignment', 32)
        align = lambda n, a=alignment: (n + a - 1) // a * a
        plan, offset = [], 0
        for name, (kind, shape, old) in model.tensors.items():
            if name == NGRAM:
                continue
            kind = change.get(name, kind)
            size = size_of(kind, shape)
            plan.append((name, kind, tuple(shape), offset, size, old))
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
        if shutil.disk_usage(output.parent).free < total + (16 << 30):
            raise ValueError('Not enough disk space with a 16 GiB reserve')

        records, worst_hc, head_note = [], 0.0, None
        with pending.open('xb') as dst:
            dst.write(header(table_start - data_start))
            for name, kind, shape, pos, size, old in plan:
                dst.seek(data_start + pos)
                if name in change and kind == Q8_0:
                    w = read_f16(model, old, shape)
                    payload = quantize_q8_0(w)
                    blk = np.frombuffer(payload, dtype=[('d', '<f2'), ('q', 'i1', 32)])
                    back = (blk['q'].astype(np.float32) * blk['d'].astype(np.float32)[:, None]).reshape(w.shape)
                    worst_hc = max(worst_hc, float(np.abs(back - w).max() / max(np.abs(w).max(), 1e-12)))
                elif name in change and kind == Q4_0:
                    payload, deq = gptq_head_q4_0(st, shape[1], shape[0])
                    # compare against the input head (Q8_0) on a sample of rows
                    old_kind = model.tensors[name][0]
                    if old_kind == Q8_0:
                        rb = shape[0] // 32 * 34
                        model.f.seek(model.data_start + old)
                        ref = np.frombuffer(model.f.read(rb * 4096), dtype=[('d', '<f2'), ('q', 'i1', 32)])
                        ref = (ref['q'].astype(np.float32) * ref['d'].astype(np.float32)[:, None]).reshape(4096, -1)
                        cur = deq[:4096].reshape(4096, -1)
                        cos = (ref * cur).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(cur, axis=1) + 1e-12)
                        head_note = f'int4 head vs input head, first 4096 rows: mean cosine {cos.mean():.4f}, min {cos.min():.4f}'
                    del deq
                else:
                    model.f.seek(model.data_start + old)
                    digest = copy_hash(model.f, dst, size)
                    records.append(dict(name=name, offset=data_start + pos, bytes=size, sha256=digest))
                    continue
                if len(payload) != size:
                    raise ValueError(f'{name}: {len(payload)} bytes, expected {size}')
                dst.write(payload)
                records.append(dict(name=name, offset=data_start + pos, bytes=size,
                                    sha256=hashlib.sha256(payload).hexdigest()))
            dst.seek(table_start)
            print('Copying the BF16 n-gram table', flush=True)
            model.f.seek(model.data_start + ng_old)
            records.append(dict(name=NGRAM, offset=table_start, bytes=ng_size,
                                sha256=copy_hash(model.f, dst, ng_size)))
            if dst.tell() != total:
                raise ValueError('Incorrect assembled size')
            dst.flush()
            os.fsync(dst.fileno())
        print(f'Retyped {len(change)} tensors', flush=True)
        if hc_q8:
            print(f'Q8_0 worst relative error on hc projections: {worst_hc:.4f}', flush=True)
        if head_note:
            print(head_note, flush=True)
        print('Verifying every output tensor', flush=True)
        with pending.open('rb') as check:
            for rec in records:
                check.seek(rec['offset'])
                if copy_hash(check, None, rec['bytes']) != rec['sha256']:
                    raise ValueError('Output payload verification failed: ' + rec['name'])
        report = dict(template=str(template), retyped=sorted(change), bytes=total,
                      hc_q8_worst_rel_error=worst_hc if hc_q8 else None, head=head_note, tensors=records)
        Path(str(output) + '.json').write_text(json.dumps(report, indent=2) + '\n')
        pending.rename(output)
        print('Verified', output, total, flush=True)
    finally:
        model.f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hc-q8', action='store_true', help='F16 hyper-connection projections to Q8_0')
    parser.add_argument('--head-gptq', type=Path, help='GPTQ checkpoint whose int4 lm_head becomes output.weight')
    args = parser.parse_args()
    try:
        retype(args.model, args.output, args.hc_q8, args.head_gptq.resolve() if args.head_gptq else None)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
