#!/usr/bin/env python3
"""Check the complete converter/engine contract using sparse GGUF fixtures.

Supply a JSON map of SafeTensors headers (tensor name to dtype/shape/offsets),
or a source directory with all shards. No weight payloads are read or allocated.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-tools"))
from deepseek41_metadata import GGUF_ALIGNMENT, metadata
from deepseek41_quantize import SourceDB, build_plan, validate_scales
from glm53_quantize import (align, kv_f32, kv_u32_array, pack_string,
                            tensor_header)


def check(hf_dir, headers, checker, fixture=None):
    if headers:
        tensors = json.loads(Path(headers).read_text())
        db = types.SimpleNamespace(tensors=tensors, info=tensors.__getitem__)
    else:
        db = SourceDB(hf_dir, index_validator=lambda _: None, scale_validator=validate_scales)
    validate_scales(db.tensors)
    config, records = metadata(hf_dir, "layout-test")
    plan = build_plan(db, config)
    if fixture:
        header = b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(records))
        header += b"".join(records) + b"".join(tensor_header(item) for item in plan)
        header += bytes(align(len(header), GGUF_ALIGNMENT) - len(header))
        with open(fixture, "xb") as fp:
            fp.write(header)
            fp.truncate(len(header) + plan[-1].offset + plan[-1].nbytes)
        print("Created sparse zero-weight fixture:", fixture)
        return

    def run(items, kv, succeeds):
        header = b"GGUF" + struct.pack("<IQQ", 3, len(items), len(kv))
        header += b"".join(kv) + b"".join(tensor_header(item) for item in items)
        header += bytes(align(len(header), GGUF_ALIGNMENT) - len(header))
        length = len(header) + items[-1].offset + items[-1].nbytes
        with tempfile.NamedTemporaryFile(suffix=".gguf") as fp:
            fp.write(header)
            fp.truncate(length)
            fp.flush()
            result = subprocess.run([checker, fp.name], capture_output=True, text=True)
        if (result.returncode == 0) != succeeds:
            raise AssertionError(result.stdout + result.stderr)
        if succeeds:
            parameters = sum((256 * t.shape[1] if t.role == "engram_disk" else math.prod(t.shape))
                             for t in items)
            for text in ["layers: 40", "train context: 1048576",
                         "attention: heads=64 kv_heads=1 head_dim=512 swa=128",
                         "indexer: heads=32 head_dim=128 top_k=512",
                         "experts: count=384 used=6 groups=0 groups_used=0",
                         f"logical parameters: {parameters / 1e9:.2f} B"]:
                assert text in result.stdout, (text, result.stdout)
        print((result.stdout or result.stderr).strip())

    run(plan, records, True)
    # Equal element count but transposed Engram projection must not be accepted.
    bad = copy.deepcopy(plan)
    item = next(t for t in bad if t.name == "blk.1.engram_kv.weight")
    item.shape = tuple(reversed(item.shape))
    run(bad, records, False)
    for key, replacement in (
        ("deepseek41.rms_norm_eps", kv_f32("deepseek41.rms_norm_eps", 1e-6)),
        ("deepseek41.kv_source_layer_ids", kv_u32_array("deepseek41.kv_source_layer_ids", [2, 8, 15, 20])),
        ("deepseek41.compress_ratios", kv_u32_array("deepseek41.compress_ratios", [0] * 40)),
    ):
        bad_records = [replacement if entry.startswith(pack_string(key)) else entry for entry in records]
        run(plan, bad_records, False)
    bad = [t for t in plan if t.name != "blk.20.indexer.attn_k.weight"]
    run(bad, records, False)
    print("V4.1 complete manifest and invalid-layout rejection: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--source-headers")
    parser.add_argument("--checker", default=str(ROOT / "tests/test_deepseek41_gguf"))
    parser.add_argument("--fixture", help="create a sparse zero-weight GGUF instead of running layout checks")
    args = parser.parse_args()
    check(args.hf_dir, args.source_headers, args.checker, args.fixture)
