#!/usr/bin/env python3
"""Small conversion fixtures; optionally check Engram against DeepSeek's code."""

import argparse
import contextlib
import ctypes
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-tools"))
from deepseek41_quantize import (NativeQuantizer, validate_scales, write_engram,
                                write_gguf, scale_name, QUANTIZATION, build_plan)
from deepseek41_metadata import GGUF_ALIGNMENT, engram_layout, metadata
import deepseek41_validate_gguf as artifact_audit
from glm53_quantize import (
    TensorPlan, QTYPE_F32, QTYPE_I8, QTYPE_IQ2_XXS, QTYPE_Q2_K, QTYPE_Q4_K,
    align, kv_string, kv_u32, load_tokenizer_records, print_plan, qtype_nbytes,
)


class MemoryDB:
    def __init__(self, weights, scales, name="layers.1.test.weight", dtype="F8_E4M3"):
        self.name = name
        self.arrays = {name: weights, scale_name(name): scales}
        self.tensors = {
            key: dict(shape=list(value.shape), dtype=dtype if key == name else "F8_E8M0")
            for key, value in self.arrays.items()
        }

    def info(self, name):
        return self.tensors[name]

    def read(self, name):
        return self.arrays[name].tobytes()

    def iter_read(self, name, byte_start=0, byte_count=None):
        data = self.read(name)
        yield data[byte_start:None if byte_count is None else byte_start + byte_count]


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        suffix = "dylib" if sys.platform == "darwin" else "so"
        cls.q = NativeQuantizer(str(ROOT / "gguf-tools" / f"libds4quants.{suffix}"))

    def test_fp8_blocks(self):
        codes = (np.arange(64 * 96, dtype=np.uint32) % 256).astype(np.uint8).reshape(64, 96)
        codes[(codes & 127) == 127] = 0
        scales = np.array([[126, 127, 128], [123, 129, 130]], dtype=np.uint8)
        db = MemoryDB(codes, scales)
        validate_scales(db.tensors)
        actual = self.q.to_f32(db, db.name)
        expected = np.zeros_like(actual)
        for r in range(64):
            for c in range(96):
                b = int(codes[r, c])
                e, m = (b >> 3) & 15, b & 7
                value = (m / 512) if e == 0 else (1 + m / 8) * 2 ** (e - 7)
                value *= (-1 if b & 128 else 1) * 2. ** (int(scales[r // 32, c // 32]) - 127)
                expected[r, c] = value
        np.testing.assert_array_equal(actual, expected)
        scales[0, 0] = 255
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.q.to_f32(db, db.name)

    def test_fp4_order_and_quants(self):
        codes = np.arange(256, dtype=np.uint8).reshape(2, 128)
        scales = np.arange(120, 136, dtype=np.uint8).reshape(2, 8)
        db = MemoryDB(codes, scales, dtype="I8")
        validate_scales(db.tensors)
        actual = self.q.to_f32(db, db.name)
        expected = np.empty((2, 256), np.float32)
        magnitude = (0, .5, 1, 1.5, 2, 3, 4, 6)
        for r in range(2):
            for c in range(256):
                b = (int(codes[r, c // 2]) >> (4 * (c % 2))) & 15
                expected[r, c] = magnitude[b & 7] * (-1 if b & 8 else 1) * 2. ** (int(scales[r, c // 32]) - 127)
        np.testing.assert_array_equal(actual, expected)
        for qt in (QTYPE_IQ2_XXS, QTYPE_Q2_K, QTYPE_Q4_K):
            self.assertEqual(self.q.encode(actual, qt), self.q.encode(expected, qt))
            weights = np.linspace(.01, 2, 256, dtype=np.float32)
            self.assertEqual(self.q.encode(actual, qt, weights), self.q.encode(expected, qt, weights))

    def test_engram_pack(self):
        rng = np.random.default_rng(17)
        weights = rng.integers(0, 255, (16385, 256), dtype=np.uint8)
        weights[(weights & 127) == 127] = 0
        scales = rng.integers(110, 145, (16385, 8), dtype=np.uint8)
        db = MemoryDB(weights, scales, name="layers.1.engram.embed.weight")
        validate_scales(db.tensors)
        item = TensorPlan("blk.1.engram_embd.weight", (264, 16385), QTYPE_I8, "engram_disk", source=db.name)
        fp = io.BytesIO()
        write_engram(fp, item, db, np)
        packed = np.frombuffer(fp.getvalue(), np.uint8).reshape(16385, 264)
        np.testing.assert_array_equal(packed[:, :256], weights)
        np.testing.assert_array_equal(packed[:, 256:], scales)
        artifact_audit.check_payload(fp, 0, item, db, self.q, None)
        fp.seek(16384 * 264 + 256)
        fp.write(bytes([int(scales[16384, 0]) ^ 1]))
        with self.assertRaisesRegex(ValueError, "differs from source"):
            artifact_audit.check_payload(fp, 0, item, db, self.q, None)
        with self.assertRaisesRegex(ValueError, "never be materialized"):
            self.q.to_f32(db, db.name)
        scales[-1, -1] = 255
        with self.assertRaisesRegex(ValueError, "nonfinite Engram"):
            write_engram(io.BytesIO(), item, db, np)

    def test_resume(self):
        class DB:
            tensors = {f"test.{i}": dict(shape=[2, 32], dtype="F32") for i in range(3)}
            def info(self, name):
                return self.tensors[name]
            def read(self, name):
                return (np.arange(64, dtype=np.float32) + int(name[-1])).tobytes()
            def close(self):
                pass
        db = DB()
        plan = []
        for name in db.tensors:
            item = TensorPlan(name, (32, 2), QTYPE_F32, "test", source=name)
            item.nbytes = qtype_nbytes(item.qtype, item.shape)
            item.offset = sum(align(t.nbytes, GGUF_ALIGNMENT) for t in plan)
            plan.append(item)
        records = [kv_string("general.architecture", "deepseek41"),
                   kv_u32("general.alignment", GGUF_ALIGNMENT),
                   kv_string("general.source.revision", "0" * 40),
                   kv_string("deepseek41.quantization", QUANTIZATION["q2"]),
                   kv_string("deepseek41.calibration", "weight-energy bootstrap")]
        original = NativeQuantizer.to_f32
        def interrupted(q, source, name):
            if name == "test.1":
                raise OSError("interrupted conversion")
            return original(q, source, name)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = types.SimpleNamespace(out=str(Path(tmp) / "model.gguf"), imatrix=None,
                                         quants_library=self.q.lib._name, threads=2, resume=False)
            partial = Path(args.out + ".partial")
            journal = Path(args.out + ".partial.json")
            with mock.patch.object(NativeQuantizer, "to_f32", interrupted):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    write_gguf(args, plan, records, db)
            self.assertEqual(json.loads(journal.read_text())["completed"], 1)
            with self.assertRaisesRegex(ValueError, "require --resume"):
                write_gguf(args, plan, records, db)
            args.resume = True
            with self.assertRaisesRegex(ValueError, "does not match"):
                write_gguf(args, plan, records + [kv_string("changed", "recipe")], db)
            data_start, size = print_plan(plan, records, [], GGUF_ALIGNMENT)
            with partial.open("ab") as fp:
                fp.write(b"unfinished tensor payload")
            write_gguf(args, plan, records, db)
            self.assertFalse(partial.exists())
            self.assertFalse(journal.exists())
            result = Path(args.out).read_bytes()
            self.assertEqual(len(result), data_start + size)
            self.assertEqual(data_start % GGUF_ALIGNMENT, 0)
            self.assertEqual(result[data_start:], b"".join(
                db.read(t.source) + bytes(align(t.nbytes, GGUF_ALIGNMENT) - t.nbytes) for t in plan))
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                write_gguf(args, plan, records, db)
            (Path(tmp) / "config.json").write_text("{}")
            audit_args = types.SimpleNamespace(hf=tmp, gguf=args.out, source_revision="0" * 40,
                payload=True, imatrix=None, quant="q2", quants_library=self.q.lib._name)
            with mock.patch.object(artifact_audit, "SourceDB", return_value=db), \
                 mock.patch.object(artifact_audit, "build_plan", return_value=plan):
                artifact_audit.validate(audit_args)
                audit_args.quant = "q4"
                with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                    artifact_audit.validate(audit_args)
                audit_args.quant = "q2"
                with open(args.out, "r+b") as fp:
                    fp.seek(data_start)
                    fp.write(b"\x01")
                with self.assertRaisesRegex(ValueError, "payload differs"):
                    artifact_audit.validate(audit_args)
                with open(args.out, "r+b") as fp:
                    fp.truncate(len(result) - 1)
                with self.assertRaisesRegex(ValueError, "file size"):
                    artifact_audit.validate(audit_args)

    def test_invalid_scales_and_float(self):
        with self.assertRaisesRegex(ValueError, "unknown quantization recipe"):
            build_plan(None, {}, "invalid")
        codes = np.zeros((2, 128), np.uint8)
        db = MemoryDB(codes, np.zeros((2, 7), np.uint8), dtype="I8")
        with self.assertRaisesRegex(ValueError, "expected E8M0 scales"):
            validate_scales(db.tensors)
        with self.assertRaisesRegex(ValueError, "overflow"):
            self.q.encode(np.array([[65536]], np.float32), 1)


def official_engram(reference_dir, library_path):
    import __future__
    import torch
    from tokenizers import Tokenizer

    # Postponed annotations allow the released module to run on Python 3.9 too;
    # no arithmetic or model code is changed.
    path = Path(reference_dir) / "inference/engram.py"
    module = types.ModuleType("deepseek_official_engram")
    sys.modules[module.__name__] = module
    exec(compile(path.read_text(), str(path), "exec", flags=__future__.annotations.compiler_flag), module.__dict__)
    config = json.loads((Path(reference_dir) / "config.json").read_text())["text_config"]
    tokenizer = Tokenizer.from_file(str(Path(reference_dir) / "tokenizer.json"))
    own = engram_layout(config, tokenizer)
    _, records = metadata(reference_dir, "0" * 40)
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(records)) + b"".join(records))
        fp.flush()
        _, encoded_tokens = load_tokenizer_records(fp.name)
    assert encoded_tokens == [tokenizer.id_to_token(i) for i in range(config["vocab_size"])]
    args = types.SimpleNamespace(**config)
    args.engram_pad_id = config["engram_pad_token_id"]
    args.max_batch_size, args.max_seq_len = 1, 1024
    reference = module.EngramLayout.from_args(args)
    class Wrapper:
        backend_tokenizer = tokenizer
        def __len__(self):
            return tokenizer.get_vocab_size()
    hashes = module.NgramHashState(args, reference, Wrapper())
    assert hashes.token_map.tolist() == own["token_map"]
    assert hashes.multipliers.tolist() == own["multipliers"]
    assert hashes.primes.flatten(1).tolist() == own["primes"]
    class Layout(ctypes.Structure):
        _fields_ = [("token_map", ctypes.POINTER(ctypes.c_uint32)),
                    ("vocab_size", ctypes.c_uint32), ("compressed_vocab_size", ctypes.c_uint32),
                    ("pad_id", ctypes.c_uint32), ("rows", ctypes.c_uint32 * 2),
                    ("multipliers", (ctypes.c_uint64 * 4) * 2),
                    ("primes", (ctypes.c_uint32 * 24) * 2)]
    class History(ctypes.Structure):
        _fields_ = [("tail", ctypes.c_int32 * 3)]
    lib = ctypes.CDLL(library_path)
    lib.ds4_engram_layout_valid.argtypes = [ctypes.POINTER(Layout)]
    lib.ds4_engram_layout_valid.restype = ctypes.c_bool
    lib.ds4_engram_history_reset.argtypes = [ctypes.POINTER(History)]
    lib.ds4_engram_hash.argtypes = [ctypes.POINTER(Layout), ctypes.POINTER(History),
                                   ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint8),
                                   ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32)]
    lib.ds4_engram_hash.restype = ctypes.c_bool
    token_map = (ctypes.c_uint32 * len(own["token_map"]))(*own["token_map"])
    layout = Layout(token_map, len(token_map), own["compressed_vocab_size"], own["pad_id"])
    layout.rows[:] = own["rows"]
    for layer in range(2):
        layout.multipliers[layer][:] = own["multipliers"][layer]
        layout.primes[layer][:] = own["primes"][layer]
    assert lib.ds4_engram_layout_valid(ctypes.byref(layout))
    rng = np.random.default_rng(984)
    tokens = rng.integers(0, len(token_map), 1024, dtype=np.int32)
    mask = rng.integers(0, 2, 1024, dtype=np.uint8)
    for masked in (False, True):
        m = torch.from_numpy(mask.astype(bool))[None] if masked else None
        expected = hashes(torch.from_numpy(tokens.astype(np.int64))[None], 0, m).numpy()[0]
        for chunk in (1, 2, 3, 4, 17, 127, 128, 129, 511, 1024):
            history = History()
            lib.ds4_engram_history_reset(ctypes.byref(history))
            actual = np.empty((1024, 2, 24), np.uint32)
            for start in range(0, len(tokens), chunk):
                n = min(chunk, len(tokens) - start)
                assert lib.ds4_engram_hash(ctypes.byref(layout), ctypes.byref(history),
                                          tokens[start:].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                          mask[start:].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)) if masked else None,
                                          n, actual[start:].ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)))
            np.testing.assert_array_equal(actual, expected)
    print("Released tokenizer map, primes, multipliers and C hashes: exact")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference")
    parser.add_argument("--engram-library")
    args, extra = parser.parse_known_args()
    if args.reference:
        if not args.engram_library:
            parser.error("--reference requires --engram-library")
        official_engram(args.reference, args.engram_library)
    unittest.main(argv=[sys.argv[0], *extra])
