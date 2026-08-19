#!/usr/bin/env python3
"""Extract IQ2_XXS dequant tables from llama.cpp ggml-common.h into npz."""
import re, numpy as np

SRC = '/Users/ljubomir/llama.cpp/worktrees/upstream-master/ggml/src/ggml-common.h'
OUT = '/Users/ljubomir/NVME_4TB_SSD_GRAUGEAR_Users_ljubomir/ds4/work/iq2_tables.npz'
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)

text = open(SRC).read()

def parse_table(name):
    m = re.search(r'GGML_TABLE_BEGIN\((\w+), %s, (\d+)\)\n(.*?)\nGGML_TABLE_END' % re.escape(name), text, re.S)
    assert m, name
    dtype, n = m.group(1), int(m.group(2))
    nums = [int(x, 0) for x in re.findall(r'0x[0-9a-fA-F]+|\d+', m.group(3))]
    assert len(nums) == n, (name, len(nums), n)
    arr = np.array(nums, dtype=np.uint64 if dtype.startswith('uint64') else np.uint8)
    return arr

kmask = parse_table('kmask_iq2xs')          # [8] uint8
ksigns = parse_table('ksigns_iq2xs')        # [128] uint8
grid = parse_table('iq2xxs_grid')           # [256] uint64 -> bytes LE -> [256,8]
grid8 = np.frombuffer(grid.astype('<u8').tobytes(), dtype=np.uint8).reshape(256, 8)

np.savez(OUT, kmask_iq2xs=kmask, ksigns_iq2xs=ksigns, iq2xxs_grid=grid8)
print('kmask:', kmask.tolist())
print('grid[0]:', grid8[0].tolist(), 'grid[1]:', grid8[1].tolist())
print('saved', OUT)
