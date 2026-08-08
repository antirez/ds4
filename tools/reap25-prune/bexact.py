#!/usr/bin/env python3
"""B-exact: recover pipenetwork's REAP25 keep map by matching REAP25 MLX expert
weights against the stock 0731 IQ2XXS GGUF, per layer, with 4-part agreement
(gate/up/down expert weights + router rows).

Usage: bexact.py [layer]   (omit layer = all 3..42)
"""
import json, os, sys, time
import numpy as np

REAP_DIR = '/Users/ljubomir/NVME_4TB_SSD_GRAUGEAR_Users_ljubomir/DeepSeek-V4-Flash-MLX-REAP25'
GGUF = '/Users/ljubomir/NVME_4TB_SSD_GRAUGEAR_Users_ljubomir/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
WORK = '/Users/ljubomir/NVME_4TB_SSD_GRAUGEAR_Users_ljubomir/ds4/work'
QK = 256

import mlx.core as mx
from gguf import GGUFReader

tabs = np.load(f'{WORK}/iq2_tables.npz')
GRID = tabs['iq2xxs_grid']      # [256, 8]
KSIGNS = tabs['ksigns_iq2xs']   # [128]
KMASK = tabs['kmask_iq2xs']     # [8]

parts = {'gate': ('gate_proj', 'ffn_gate_exps.weight'),
         'up':   ('up_proj',   'ffn_up_exps.weight'),
         'down': ('down_proj', 'ffn_down_exps.weight')}
POOL_WIN = (8, 256)   # pooled dims: [rows/8, cols/256] -> 4096 for all expert tensors

def f16_to_f32(u16):
    return u16.reshape(-1).astype('<u2').view('<f2').astype(np.float32)

def dequant_iq2xxs(blk):
    """blk: [nb, 66] uint8 (fp16 d + 64B qs) -> f32 [nb, 256]"""
    nb = blk.shape[0]
    d = f16_to_f32(blk[:, 0:2].copy().view(np.uint16))          # [nb]
    qs = blk[:, 2:66]                                            # [nb, 64]
    out = np.empty((nb, 256), dtype=np.float32)
    for ib in range(8):
        w32 = qs[:, 4*ib+4:4*ib+8].copy().view('<u4').reshape(-1)  # [nb] aux32[1]
        db = d * (0.5 + (w32 >> 28).astype(np.float32)) * 0.25
        for l in range(4):
            g = qs[:, 4*ib + l]                                    # [nb] grid idx
            grid8 = GRID[g]                                        # [nb, 8]
            sidx = (w32 >> (7*l)) & 127
            flip = ((KSIGNS[sidx][:, None] & KMASK[None, :]) != 0)
            out[:, ib*32 + l*8: ib*32 + (l+1)*8] = db[:, None] * grid8 * np.where(flip, -1.0, 1.0)
    return out

def dequant_q2k(blk):
    """blk: [nb, 84] uint8 (scales16, qs64, dmin, d) -> f32 [nb, 256]"""
    nb = blk.shape[0]
    sc = blk[:, 0:16]
    qs = blk[:, 16:80]
    dmin = f16_to_f32(blk[:, 82:84].copy().view(np.uint16))
    d = f16_to_f32(blk[:, 80:82].copy().view(np.uint16))
    out = np.empty((nb, 256), dtype=np.float32)
    for n in range(2):
        qb = qs[:, n*32:(n+1)*32]
        for j in range(4):
            shift = 2*j
            s0 = sc[:, 8*n + 2*j]; s1 = sc[:, 8*n + 2*j + 1]
            dl0 = d * (s0 & 15).astype(np.float32); ml0 = dmin * (s0 >> 4).astype(np.float32)
            dl1 = d * (s1 & 15).astype(np.float32); ml1 = dmin * (s1 >> 4).astype(np.float32)
            v0 = dl0[:, None] * ((qb[:, 0:16] >> shift) & 3).astype(np.float32) - ml0[:, None]
            v1 = dl1[:, None] * ((qb[:, 16:32] >> shift) & 3).astype(np.float32) - ml1[:, None]
            col = n*128 + j*32
            out[:, col:col+16] = v0
            out[:, col+16:col+32] = v1
    return out

def pool_expert(mat, win):
    """mat: [rows, cols] -> pooled [(rows/rb)*(cols/cb)] via (rb, cb) window means."""
    r, c = mat.shape
    rb, cb = win
    m = mat.reshape(r // rb, rb, c // cb, cb)
    return m.mean(axis=(1, 3)).reshape(-1)

# ---------------- GGUF side ----------------
reader = GGUFReader(GGUF)
tensors = {t.name: t for t in reader.tensors}
print('GGUF loaded, tensors:', len(tensors), flush=True)

def gguf_expert_pooled(L, part):
    """pooled per-expert vectors [256, 4096] from the stock GGUF.
    reader data: uint8 [n_exp, ne1, blocks_per_col*blk_size]"""
    mlx_name, gguf_name = parts[part]
    t = tensors[f'blk.{L}.{gguf_name}']
    ne0, ne1, n_exp = (int(x) for x in t.shape)
    raw = np.array(t.data, dtype=np.uint8)          # [n_exp, ne1, bpc*blk]
    is_xxs = t.tensor_type.name == 'IQ2_XXS'
    blk_size = 66 if is_xxs else 84
    bpc = raw.shape[2] // blk_size                  # blocks per col = ne0//256
    pooled = np.empty((n_exp, 4096), dtype=np.float32)
    for e in range(n_exp):
        blk = raw[e].reshape(-1, blk_size)
        if is_xxs:
            v = dequant_iq2xxs(blk).reshape(ne1, ne0)
        else:
            v = dequant_q2k(blk).reshape(ne1, ne0)
        pooled[e] = pool_expert(v, POOL_WIN)
    return pooled

def gguf_router(L):
    t = tensors[f'blk.{L}.ffn_gate_inp.weight']
    # F16 typed view [256 experts, 4096]
    return np.asarray(t.data).astype(np.float32)

# ---------------- REAP25 (MLX) side ----------------
idx = json.load(open(f'{REAP_DIR}/model.safetensors.index.json'))
wm = idx['weight_map']
shard_cache = {}

def reap25_tensor(name):
    shard = wm[name]
    if shard not in shard_cache:
        import safetensors.torch
        shard_cache[shard] = safetensors.torch.safe_open(f'{REAP_DIR}/{shard}', framework='pt')
    return shard_cache[shard]

def reap25_expert_pooled(L, part):
    """pooled per-expert vectors [192, 4096] from REAP25 MLX."""
    proj, _ = parts[part]
    base = f'layers.{L}.ffn.experts.{proj}'
    w = reap25_tensor(base + '.weight').get_tensor(base + '.weight').numpy()          # U32 packed
    sc = reap25_tensor(base + '.scales').get_tensor(base + '.scales').float().numpy() # BF16 -> f32
    bi = reap25_tensor(base + '.biases').get_tensor(base + '.biases').float().numpy() # BF16 -> f32
    n_exp = w.shape[0]
    qw = mx.array(w); qsc = mx.array(sc); qbi = mx.array(bi)
    dec = mx.dequantize(qw, qsc, qbi, group_size=64, bits=4).astype(mx.float32)
    arr = np.asarray(dec)                                  # [E, out, in]
    pooled = np.empty((n_exp, 4096), dtype=np.float32)
    for e in range(n_exp):
        m = arr[e]                                         # [out, in]  (MLX Linear layout)
        pooled[e] = pool_expert(m, POOL_WIN)
    return pooled

def reap25_router(L):
    base = f'layers.{L}.ffn.gate'
    st = reap25_tensor(base + '.weight')
    w = st.get_tensor(base + '.weight').float().numpy()  # BF16 [192, 4096]
    return w

# ---------------- matching ----------------
def match_part(A, B):
    """A: [192, D], B: [256, D] -> best idx per row + margin."""
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    cos = An @ Bn.T                                        # [192, 256]
    best = cos.argsort(axis=1)[:, ::-1]
    top1 = best[:, 0]; top2 = best[:, 1]
    margin = cos[np.arange(cos.shape[0]), top1] - cos[np.arange(cos.shape[0]), top2]
    return top1, margin, cos

def run_layer(L):
    votes = {p: None for p in ('gate', 'up', 'down', 'router')}
    margins = {}
    for p in ('gate', 'up', 'down'):
        A = reap25_expert_pooled(L, p)
        B = gguf_expert_pooled(L, p)
        votes[p], margins[p], cos = match_part(A, B)
    A = reap25_router(L); B = gguf_router(L)
    votes['router'], margins['router'], _ = match_part(A, B)
    # agreement
    agree = {p: set() for p in votes}
    full = 0
    for j in range(192):
        ids = {votes[p][j] for p in votes}
        for p in votes:
            agree[p].add(votes[p][j])
        if len(ids) == 1:
            full += 1
    uniq = {p: len(agree[p]) for p in votes}
    return votes, margins, full, uniq

if __name__ == '__main__':
    layers = [int(a) for a in sys.argv[1:]] or list(range(3, 43))
    all_votes = {}
    report = []
    t0 = time.time()
    for L in layers:
        tl = time.time()
        votes, margins, full, uniq = run_layer(L)
        ok = full == 192 and all(u == 192 for u in uniq.values())
        all_votes[L] = {p: [int(v) for v in votes[p]] for p in votes}
        report.append((L, full, uniq, {p: round(float(margins[p].min()), 4) for p in margins}, ok))
        print(f'L{L}: 4-part full-agree {full}/192 | unique ids {uniq} | min margins '
              f'{ {p: round(float(margins[p].min()),3) for p in margins} } | {"OK" if ok else "PROBLEM"} '
              f'({time.time()-tl:.1f}s)', flush=True)
    print(f'total {time.time()-t0:.0f}s')
    json.dump({'layers': all_votes, 'report': report},
              open(f'{WORK}/bexact_votes.json', 'w'), indent=1)
