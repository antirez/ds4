#!/usr/bin/env python3
"""Lossless V4.1 DSpark conversion, including interrupted output recovery."""

import contextlib
import io
import json
import struct
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-tools"))
from deepseek41_dspark import (LosslessWeights, build_plan, write_gguf,
                               verify_gguf, ALIGNMENT)
from glm53_quantize import SourceDB, TensorPlan, QTYPE_F16, QTYPE_F32, QTYPE_MXFP4, kv_string, kv_u32, qtype_nbytes


class MemoryDB:
    def __init__(self):
        self.tensors, self.data = {}, {}

    def add(self, name, array, dtype):
        self.tensors[name] = dict(shape=list(array.shape), dtype=dtype, shard="fixture.safetensors")
        self.data[name] = array.tobytes()

    def info(self, name):
        return self.tensors[name]

    def read(self, name):
        return self.data[name]


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.q, self.db = LosslessWeights(), MemoryDB()

    def test_draft_only_source_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = json.dumps({
                "mtp.0.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                "layers.39.weight": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
            }).encode()
            root.joinpath("draft.safetensors").write_bytes(
                struct.pack("<Q", len(header)) + header + struct.pack("<ff", 3., 4.))
            index = {"weight_map": {"mtp.0.weight": "draft.safetensors",
                                   "layers.39.weight": "draft.safetensors",
                                   "layers.0.weight": "not-downloaded.safetensors"}}
            root.joinpath("model.safetensors.index.json").write_text(json.dumps(index))
            validator = mock.Mock()
            db = SourceDB(tmp, index_validator=validator, scale_validator=lambda _: None,
                          tensor_prefix="mtp.")
            try:
                validator.assert_called_once_with(index["weight_map"])
                self.assertEqual(set(db.tensors), {"mtp.0.weight"})
                self.assertEqual(db.read("mtp.0.weight"), struct.pack("<f", 3.))
            finally:
                db.close()
            with self.assertRaisesRegex(ValueError, "missing source shard"):
                SourceDB(tmp, index_validator=lambda _: None, scale_validator=lambda _: None)
            index["weight_map"]["mtp.1.weight"] = "draft.safetensors"
            root.joinpath("model.safetensors.index.json").write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                SourceDB(tmp, index_validator=lambda _: None, scale_validator=lambda _: None,
                         tensor_prefix="mtp.")

    def test_native_fp4_all_codes_scales_and_layout(self):
        name = "mtp.0.ffn.experts.0.w1.weight"
        codes = np.arange(256, dtype=np.uint8).reshape(2, 128)
        scales = np.arange(119, 135, dtype=np.uint8).reshape(2, 8)
        self.db.add(name, codes, "I8")
        self.db.add(name.removesuffix(".weight") + ".scale", scales, "F8_E8M0")
        payload = self.q.expert_bytes(self.db, name)
        self.assertEqual(len(payload), qtype_nbytes(QTYPE_MXFP4, (256, 2)))
        packed = np.frombuffer(payload, np.uint8).reshape(2, 8, 17)
        lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
        expected = self.q.to_f32(self.db, name)
        for r in range(2):
            for block in range(8):
                self.assertEqual(packed[r, block, 0], scales[r, block])
                for j in range(32):
                    source = int(codes[r, block * 16 + j // 2]) >> (4 * (j % 2)) & 15
                    actual = int(packed[r, block, 1 + j % 16]) >> (4 * (j // 16)) & 15
                    self.assertEqual(source, actual)
                    value = np.ldexp(lut[actual], int(scales[r, block]) - 127)
                    self.assertEqual(value, expected[r, block * 32 + j])
        scales[0, 0] = 255
        self.db.add(name.removesuffix(".weight") + ".scale", scales, "F8_E8M0")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.q.expert_bytes(self.db, name)

    def test_fp8_partial_blocks_lossless_type_selection(self):
        name = "mtp.0.main_proj.weight"
        codes = np.full((35, 67), 56, np.uint8)
        scales = np.array([[120, 127, 130], [125, 128, 131]], np.uint8)
        self.db.add(name, codes, "F8_E4M3")
        self.db.add("mtp.0.main_proj.scale", scales, "F8_E8M0")
        self.assertEqual(self.q.dense_type(self.db, name), QTYPE_F16)
        expected = np.empty(codes.shape, np.float32)
        for r in range(35):
            for col in range(67):
                expected[r, col] = 2. ** (int(scales[r // 32, col // 32]) - 127)
        np.testing.assert_array_equal(self.q.to_f32(self.db, name), expected)
        scales[0, 0] = 90
        self.db.add("mtp.0.main_proj.scale", scales, "F8_E8M0")
        self.assertEqual(self.q.dense_type(self.db, name), QTYPE_F32)
        item = TensorPlan("test", (67, 35), QTYPE_F16, "projection", source=name)
        with self.assertRaisesRegex(ValueError, "change source"):
            self.q.dense_bytes(self.db, item)
        scales[0, 0] = 160
        self.db.add("mtp.0.main_proj.scale", scales, "F8_E8M0")
        self.assertEqual(self.q.dense_type(self.db, name), QTYPE_F32)

    def test_bf16_small_values_are_not_dropped(self):
        name = "mtp.2.markov_head.embed.weight"
        values = np.array([[1., 2. ** -40, -2. ** -40, 2. ** 30]], np.float32)
        self.db.add(name, (values.view(np.uint32) >> 16).astype("<u2"), "BF16")
        self.assertEqual(self.q.dense_type(self.db, name), QTYPE_F32)
        item = TensorPlan("test", (4, 1), QTYPE_F32, "markov", source=name)
        self.assertEqual(self.q.dense_bytes(self.db, item), values.tobytes())

    def test_invalid_family(self):
        with self.assertRaisesRegex(ValueError, "V4.1 checkpoint"):
            build_plan(self.db, {"model_type": "deepseek_v4"}, self.q)

    def test_atomic_resume_and_complete_payload_audit(self):
        plan = []
        for i in range(3):
            name = f"mtp.{i}.norm.weight"
            self.db.add(name, np.arange(32, dtype=np.float32) + i, "F32")
            plan.append(TensorPlan(name, (32,), QTYPE_F32, "norm", source=name,
                                   offset=i * ALIGNMENT, nbytes=128))
        records = [kv_string("general.architecture", "deepseek41-dspark"), kv_u32("general.alignment", ALIGNMENT)]
        original = self.q.dense_bytes
        def interrupt(db, item):
            if item == plan[1]:
                raise OSError("interrupted")
            return original(db, item)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            self.db.hf_dir = tmp
            Path(tmp, "fixture.safetensors").write_bytes(b"stable source fixture")
            args = types.SimpleNamespace(out=str(Path(tmp, "draft.gguf")), resume=False)
            with mock.patch.object(self.q, "dense_bytes", interrupt):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    write_gguf(args, self.db, self.q, plan, records)
            self.assertFalse(Path(args.out).exists())
            self.assertEqual(json.loads(Path(args.out + ".partial.json").read_text())["completed"], 1)
            args.resume = True
            with self.assertRaisesRegex(ValueError, "does not match"):
                write_gguf(args, self.db, self.q, plan, records + [kv_u32("changed", 1)])
            write_gguf(args, self.db, self.q, plan, records)
            verify_gguf(args.out, self.db, self.q, plan, records)
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                write_gguf(args, self.db, self.q, plan, records)
            with open(args.out, "r+b") as fp:
                fp.seek(-ALIGNMENT, 2)
                fp.write(b"bad!")
            with self.assertRaisesRegex(ValueError, "payload differs"):
                verify_gguf(args.out, self.db, self.q, plan, records)
            with open(args.out, "r+b") as fp:
                fp.truncate(1)
            with self.assertRaises(ValueError):
                verify_gguf(args.out, self.db, self.q, plan, records)


if __name__ == "__main__":
    unittest.main()
