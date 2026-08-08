#!/usr/bin/env python3
"""Prune the stock 0731 IQ2XXS GGUF to REAP25 (256 experts layers 0-2, 192 in 3-42)
using the B-exact keep map. ds4-GGUF variant: u64 key/name lengths, no tensor size
field, 32-byte data alignment."""
import json, os, struct, time
import numpy as np
from gguf import GGUFReader

SRC = '/Users/ljubomir/NVME_4TB_SSD_GRAUGEAR_Users_ljubomir/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
OUT = '/Users/ljubomir/NVME_4TB_SSD_GRAUGEAR_Users_ljubomir/ds4/gguf/DeepSeek-V4-Flash-0731-REAP25-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf'
KEEP_F = '/Users/ljubomir/NVME_4TB_SSD_GRAUGEAR_Users_ljubomir/ds4/work/keep_map_bexact.json'
ALIGN = 32
PRUNE = ('ffn_gate_inp.weight', 'ffn_gate_exps.weight', 'ffn_up_exps.weight',
         'ffn_down_exps.weight', 'exp_probs_b.bias')

KEEP = {int(k): sorted(int(i) for i in v) for k, v in json.load(open(KEEP_F))['keep_map'].items()}

# sizes from GGUFReader (authoritative data-view byte counts)
reader = GGUFReader(SRC)
SIZES = {t.name: t.data.nbytes for t in reader.tensors}
print(f'reader: {len(reader.tensors)} tensors', flush=True)

def walk(f, t):
    if t == 8:
        ln = struct.unpack('<Q', f.read(8))[0]; f.seek(ln, 1)
    elif t == 9:
        et, cnt = struct.unpack('<IQ', f.read(12))
        for _ in range(cnt): walk(f, et)
    elif t in (12, 10, 11): f.seek(8, 1)
    elif t in (2, 3): f.seek(2, 1)
    elif t in (0, 1, 7): f.seek(1, 1)
    elif t in (4, 5, 6): f.seek(4, 1)
    elif t == 13: f.seek(2, 1)
    else: raise ValueError(f'unknown kv type {t}')

def ser(t, v):
    if t == 7: return struct.pack('<B', 1 if v else 0)
    if t == 8:
        b = v.encode(); return struct.pack('<Q', len(b)) + b
    if t == 9:
        out = struct.pack('<IQ', 5, len(v))
        for x in v: out += struct.pack('<i', int(x))
        return out
    raise ValueError(t)

t0 = time.time()
f = open(SRC, 'rb')
assert f.read(4) == b'GGUF'
version, n_tensors, n_kv = struct.unpack('<IQQ', f.read(20))
assert version == 3, version
print(f'source: v{version}, {n_tensors} tensors, {n_kv} kv', flush=True)

kv_ranges = []
for _ in range(n_kv):
    s = f.tell()
    ln = struct.unpack('<Q', f.read(8))[0]; f.seek(ln, 1)      # u64 key len
    t = struct.unpack('<I', f.read(4))[0]
    walk(f, t)
    kv_ranges.append((s, f.tell()))

tensors = []  # (name, dims, type, old_offset, old_size)
for _ in range(n_tensors):
    ln = struct.unpack('<Q', f.read(8))[0]                     # u64 name len
    name = f.read(ln).decode()
    nd = struct.unpack('<I', f.read(4))[0]
    dims = struct.unpack(f'<{nd}Q', f.read(8*nd))
    typ = struct.unpack('<I', f.read(4))[0]
    off = struct.unpack('<Q', f.read(8))[0]                    # no size field
    size = SIZES[name]
    tensors.append((name, dims, typ, off, size))
SRC_DATA_START = (f.tell() + 31) & ~31                     # reader aligns data start to 32
f.close()
print(f'parsed {len(tensors)} tensor infos, src data_start={SRC_DATA_START}', flush=True)

# ---- plan ----
plan = []
for name, dims, typ, old_off, old_size in tensors:
    parts = name.split('.')
    L = int(parts[1]) if (len(parts) == 4 and parts[0] == 'blk') else -1
    leaf = (parts[2] + '.' + parts[3]) if len(parts) == 4 else ''
    if L >= 3 and leaf in PRUNE:
        assert dims[-1] == 256 and old_size % 256 == 0, (name, dims, old_size)
        keep = KEEP[L]
        assert len(keep) == 192
        new_dims = dims[:-1] + (192,)
        new_size = old_size // 256 * 192
        plan.append((name, new_dims, typ, old_off, old_size, new_size, keep))
        print(f'  prune {name}: {list(dims)} -> {list(new_dims)}  {old_size} -> {new_size} B', flush=True)
    else:
        plan.append((name, dims, typ, old_off, old_size, old_size, None))

def align(x):
    return x  # ds4 GGUF: strictly sequential offsets, no padding

new_off = 0
out_plan = []
for name, dims, typ, old_off, old_size, new_size, keep in plan:
    out_plan.append((name, dims, typ, old_off, old_size, new_size, keep, new_off))
    new_off = new_off + new_size
print(f'output data: {new_off/1e9:.1f} GB (source {sum(s for _,_,_,_,s,_,_ in plan)/1e9:.1f} GB)', flush=True)

# ---- write ----
reap_kv = [
    (b'reap.enabled', 7, ser(7, True)),
    (b'reap.layout', 8, ser(8, 'ds4-compact-v1')),
    (b'reap.layer.expert_count', 9, ser(9, [256]*43)),
    (b'reap.layer.keep_count', 9, ser(9, [256]*3 + [192]*40)),
]
g = open(OUT, 'wb')
g.write(b'GGUF')
g.write(struct.pack('<IQQ', version, len(out_plan), n_kv + len(reap_kv)))
f = open(SRC, 'rb')
for s, e in kv_ranges:
    f.seek(s); g.write(f.read(e - s))
for key, t, val in reap_kv:
    g.write(struct.pack('<Q', len(key))); g.write(key)
    g.write(struct.pack('<I', t)); g.write(val)
for name, dims, typ, old_off, old_size, new_size, keep, off in out_plan:
    b = name.encode()
    g.write(struct.pack('<Q', len(b))); g.write(b)
    g.write(struct.pack('<I', len(dims)))
    g.write(struct.pack(f'<{len(dims)}Q', *dims))
    g.write(struct.pack('<I', typ))
    g.write(struct.pack('<Q', off))
OUT_DATA_START = (g.tell() + 31) & ~31
if g.tell() != OUT_DATA_START:
    g.write(b'\0' * (OUT_DATA_START - g.tell()))           # pad to 32-byte boundary
f.close()

src = np.memmap(SRC, dtype=np.uint8, mode='r')
CHUNK = 64 * 1024 * 1024
for name, dims, typ, old_off, old_size, new_size, keep, off in out_plan:
    sbase = SRC_DATA_START + old_off
    if keep is None:
        left = old_size; pos = 0
        while left > 0:
            n = min(CHUNK, left)
            g.seek(OUT_DATA_START + off + pos)
            g.write(src[sbase + pos:sbase + pos + n].tobytes())
            pos += n; left -= n
    else:
        per = old_size // 256
        dst = OUT_DATA_START + off
        for k in keep:
            s = sbase + k * per
            g.seek(dst)
            for p in range(0, per, CHUNK):
                n = min(CHUNK, per - p)
                g.write(src[s + p:s + p + n].tobytes())
            dst += per
g.close()
print(f'written {os.path.getsize(OUT)/1e9:.1f} GB (data {new_off/1e9:.1f} GB) in {time.time()-t0:.0f}s', flush=True)
