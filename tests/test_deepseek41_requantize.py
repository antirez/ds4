#!/usr/bin/env python3
"""Tiny real clonefile fixtures: source preservation, payloads, and rollback."""

import contextlib
import dataclasses
import io
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
import deepseek41_requantize as rq
from deepseek41_quantize import (ATTENTION_SUFFIXES, NativeQuantizer,
                                attention_importance)
from glm53_quantize import (TensorPlan, Imatrix, QTYPE_Q8_0, QTYPE_Q4_K,
                            QTYPE_I8, align, kv_string, kv_u32, tensor_header)


class RequantizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        suffix = "dylib" if sys.platform == "darwin" else "so"
        cls.q = NativeQuantizer(str(ROOT / "gguf-tools" / f"libds4quants.{suffix}"))

    def fixture(self, path):
        items, payloads = [], {}
        for index, suffix in enumerate(ATTENTION_SUFFIXES):
            width, rows = 256 + (index % 2) * 256, 129 if index == 0 else index + 2
            values = np.sin(np.arange(width * rows, dtype=np.float32)).reshape(rows, width)
            data = self.q.encode(values, QTYPE_Q8_0)
            name = "blk.0." + suffix
            offset = sum(align(t.nbytes, 16384) for t in items)
            items.append(TensorPlan(name, (width, rows), QTYPE_Q8_0, "attention", offset=offset, nbytes=len(data)))
            payloads[name] = data
        data = bytes(range(256)) * 5
        items.append(TensorPlan("blk.0.engram_embd.weight", (256, 5), QTYPE_I8, "engram_disk",
                                offset=sum(align(t.nbytes, 16384) for t in items), nbytes=len(data)))
        payloads[items[-1].name] = data
        records = [kv_string("general.architecture", "deepseek41"), kv_u32("general.alignment", 16384),
                   kv_u32("deepseek41.num_hidden_layers", 1),
                   kv_string("deepseek41.quantization", rq.QUANTIZATION["q2"]),
                   kv_string("deepseek41.calibration", "imatrix"),
                   kv_string("quantize.imatrix.file", "original-experts.dat")]
        header = b"GGUF" + struct.pack("<IQQ", 3, len(items), len(records))
        header += b"".join(records) + b"".join(tensor_header(item) for item in items)
        header += bytes(align(len(header), 16384) - len(header))
        with path.open("wb") as fp:
            fp.write(header)
            for item in items:
                fp.write(payloads[item.name])
                fp.write(bytes([0xA5]) * (align(item.nbytes, 16384) - item.nbytes))
        return items, payloads

    def args(self, source, output, **kw):
        return types.SimpleNamespace(source_gguf=str(source), out=str(output), attention_imatrix=None,
                                     dry_run=False, quants_library=self.q.lib._name, **kw)

    def write_imatrix(self, path, items):
        with path.open("wb") as fp:
            fp.write(struct.pack("<i", len(items)))
            for item in items:
                name = item.name.encode()
                weights = np.linspace(.01, 2., item.shape[0], dtype=np.float32)
                fp.write(struct.pack("<i", len(name)) + name + struct.pack("<ii", 1, weights.size) + weights.tobytes())

    @unittest.skipUnless(sys.platform == "darwin", "native copy-on-write requires macOS")
    def test_clone_preserves_every_unchanged_byte(self):
        for calibrated in (False, True):
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                source, output = Path(tmp) / "source.gguf", Path(tmp) / "out.gguf"
                items, payloads = self.fixture(source)
                before = source.read_bytes()
                args = self.args(source, output)
                if calibrated:
                    calibration = Path(tmp) / "attention.dat"
                    self.write_imatrix(calibration, items[:-1])
                    args.attention_imatrix = str(calibration)
                rq.requantize(args)
                self.assertNotEqual(source.stat().st_ino, output.stat().st_ino)
                self.assertEqual(source.read_bytes(), before)
                after = output.read_bytes()
                self.assertEqual(len(before), len(after))
                with source.open("rb") as fp:
                    h = rq.read_header(fp)
                with output.open("rb") as fp:
                    new = rq.read_header(fp)
                self.assertEqual(h.data_start, new.data_start)
                self.assertEqual(new.metadata["deepseek41.calibration"], "imatrix")
                self.assertEqual(new.metadata["quantize.imatrix.file"], "original-experts.dat")
                self.assertEqual(new.metadata["deepseek41.attention_calibration"], "imatrix" if calibrated else "uncalibrated")
                expected_file = bytearray(before)
                expected_file[:h.data_start] = new.raw
                for old, item in zip(items[:-1], new.tensors[:-1]):
                    self.assertEqual(item.qtype, QTYPE_Q4_K)
                    self.assertEqual(old.offset, item.offset)
                    # Independent scalar GGML Q8_0 decode, including signed codes.
                    values = []
                    packed = payloads[old.name]
                    for offset in range(0, len(packed), 34):
                        scale, *codes = struct.unpack_from("<e32b", packed, offset)
                        values.extend(scale * code for code in codes)
                    values = np.array(values, dtype=np.float32).reshape(old.shape[1], old.shape[0])
                    importance = np.linspace(.01, 2., old.shape[0], dtype=np.float32) if calibrated else None
                    expected = self.q.encode(values, QTYPE_Q4_K, importance)
                    start = h.data_start + item.offset
                    expected_file[start:start + len(expected)] = expected
                self.assertEqual(after, bytes(expected_file))
                rq.check_requantization(args)
                with output.open("r+b") as fp:
                    fp.seek(h.data_start + new.tensors[0].offset)
                    byte = fp.read(1)
                    fp.seek(-1, 1)
                    fp.write(bytes([byte[0] ^ 1]))
                with self.assertRaisesRegex(ValueError, "payload differs"):
                    rq.check_requantization(args)
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    rq.requantize(args)

    def test_rejected_inputs_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            source, output = Path(tmp) / "source.gguf", Path(tmp) / "out.gguf"
            self.fixture(source)
            original = source.read_bytes()
            args = self.args(source, output)
            args.dry_run = True
            with mock.patch.object(rq, "clone_file", side_effect=AssertionError("clone in dry-run")):
                rq.requantize(args)
            self.assertFalse(output.exists())
            args.dry_run = False
            for patch in (mock.patch.object(rq, "clone_file", side_effect=OSError("unsupported clone")),
                          mock.patch.object(rq.shutil, "disk_usage", return_value=types.SimpleNamespace(free=0))):
                with patch, self.assertRaises((OSError, ValueError)):
                    rq.requantize(args)
                self.assertFalse(output.exists())
                self.assertFalse(list(Path(tmp).glob(".ds41-q4-*")))
                self.assertEqual(source.read_bytes(), original)
            with source.open("rb") as fp:
                h = rq.read_header(fp)
            bad = dataclasses.replace(h, tensors=[dataclasses.replace(t, qtype=QTYPE_Q4_K) for t in h.tensors])
            with self.assertRaisesRegex(ValueError, "expected a Q8_0"):
                rq.plan_requantization(bad)
            bad = dataclasses.replace(h, data_start=32, alignment=32)
            with self.assertRaisesRegex(ValueError, "header padding"):
                rq.plan_requantization(bad)
            if sys.platform == "darwin":
                with mock.patch.object(rq, "encoded_chunks", side_effect=ValueError("injected failure")):
                    with self.assertRaisesRegex(ValueError, "injected failure"):
                        rq.requantize(args)
                self.assertFalse(output.exists())
                self.assertFalse(list(Path(tmp).glob(".ds41-q4-*")))
                self.assertEqual(source.read_bytes(), original)

    def test_attention_imatrix_contract(self):
        item = TensorPlan("blk.0.attn_output_a.weight", (256, 8), QTYPE_Q4_K, "attention")
        imatrix = Imatrix(None, np)
        for values in (None, np.ones(255), np.zeros(256), -np.ones(256), np.full(256, np.nan)):
            imatrix.entries = {} if values is None else {item.name: values}
            with self.assertRaises(ValueError):
                attention_importance(imatrix, item)
        valid = np.ones(256, dtype=np.float32)
        imatrix.entries[item.name] = valid
        self.assertIs(attention_importance(imatrix, item), valid)
        self.assertIsNone(attention_importance(None, item))

    def test_dry_run_validates_calibration_without_quantizer(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            source, output = Path(tmp) / "source.gguf", Path(tmp) / "out.gguf"
            items, _ = self.fixture(source)
            args = self.args(source, output)
            args.dry_run = True
            path = Path(tmp) / "dense.dat"
            args.attention_imatrix = str(path)
            with mock.patch.object(rq, "NativeQuantizer", side_effect=AssertionError("library loaded in dry-run")):
                with self.assertRaises(FileNotFoundError):
                    rq.requantize(args)
                self.write_imatrix(path, items[:-1])
                valid = path.read_bytes()
                rq.requantize(args)
                self.write_imatrix(path, items[:1])
                with self.assertRaisesRegex(ValueError, "missing attention"):
                    rq.requantize(args)
                self.write_imatrix(path, [items[0], items[0]])
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    rq.requantize(args)
                for value in (-1., float("nan"), float("inf")):
                    malformed = bytearray(valid)
                    first_value = 4 + 4 + len(items[0].name.encode()) + 8
                    struct.pack_into("<f", malformed, first_value, value)
                    path.write_bytes(malformed)
                    with self.assertRaisesRegex(ValueError, "invalid attention"):
                        rq.requantize(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
