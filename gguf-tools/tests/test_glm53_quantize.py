#!/usr/bin/env python3

import math
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glm53_quantize import (
    QTYPE_BF16,
    QTYPE_F32,
    QTYPE_I8,
    QTYPE_IQ2_XXS,
    QTYPE_Q2_K,
    QTYPE_Q4_K,
    QTYPE_Q8_0,
    Imatrix,
    Quantizer,
    TensorPlan,
    conversion_signature,
    iter_native_tensor_bytes,
    native_fp8_plan,
    regular_qtype,
    transform_regular,
)
import glm53_full_quantize as full_glm


class FakeSourceDB:
    def __init__(self, codes, scales):
        self.codes = codes
        self.scales = scales

    def info(self, name):
        if name.endswith("_scale_inv"):
            return {"dtype": "F32", "shape": list(self.scales.shape)}
        return {"dtype": "F8_E4M3", "shape": list(self.codes.shape)}

    def read(self, name):
        if name.endswith("_scale_inv"):
            return self.scales.astype("<f4").tobytes()
        return self.codes.tobytes()


class FakeNativeDB:
    def __init__(self):
        self.codes = np.arange(256 * 128, dtype=np.uint32).astype(np.uint8).tobytes()
        self.scales = np.array([1.0, 2.0], dtype="<f4").tobytes()
        self.tensors = {
            "weight": {"dtype": "F8_E4M3", "shape": [256, 128], "nbytes": 256 * 128},
            "weight_scale_inv": {"dtype": "F32", "shape": [2, 1], "nbytes": 8},
        }

    def info(self, name):
        return self.tensors[name]

    def iter_read(self, name, byte_start=0, byte_count=None, chunk_size=16 << 20):
        data = self.scales if name.endswith("_scale_inv") else self.codes
        if byte_count is None:
            byte_count = len(data) - byte_start
        yield data[byte_start : byte_start + byte_count]


class FakeFullFFNDB:
    def info(self, name):
        if ".experts." in name or ".shared_experts." in name:
            shape = [6144, 2048] if ".down_proj." in name else [2048, 6144]
        elif name.endswith(".gate.weight"):
            shape = [256, 6144]
        elif name.endswith(".gate.e_score_correction_bias"):
            shape = [256]
        else:
            raise KeyError(name)
        return {"shape": shape}


def bare_quantizer():
    quantizer = Quantizer.__new__(Quantizer)
    quantizer.np = np
    quantizer.fp8_lut = quantizer._build_fp8_lut()
    return quantizer


class GLM53QuantizeTests(unittest.TestCase):
    def test_conversion_signature_includes_imatrix_contents(self):
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(b"first calibration")
            fp.flush()
            first = conversion_signature([], [], [], fp.name)
            fp.seek(0)
            fp.write(b"other calibration")
            fp.truncate()
            fp.flush()
            second = conversion_signature([], [], [], fp.name)
        self.assertNotEqual(first, second)

    def test_imatrix_unobserved_expert_uses_fallback(self):
        values = np.zeros(288 * 2, dtype="<f4")
        values[2:4] = (1.0, 2.0)
        name = b"blk.3.ffn_gate_exps.weight"
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(struct.pack("<i", 1))
            fp.write(struct.pack("<i", len(name)))
            fp.write(name)
            fp.write(struct.pack("<i", 1))
            fp.write(struct.pack("<i", len(values)))
            fp.write(values.tobytes())
            fp.flush()
            matrix = Imatrix(fp.name, np)
        self.assertIsNone(matrix.expert(name.decode(), 0, 2))
        np.testing.assert_array_equal(
            matrix.expert(name.decode(), 1, 2),
            np.array([1.0, 2.0], dtype=np.float32),
        )

    def test_imatrix_accepts_full_glm_expert_count(self):
        values = np.zeros(256 * 2, dtype="<f4")
        values[-2:] = (3.0, 4.0)
        name = b"blk.77.ffn_up_exps.weight"
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(struct.pack("<i", 1))
            fp.write(struct.pack("<i", len(name)))
            fp.write(name)
            fp.write(struct.pack("<i", 1))
            fp.write(struct.pack("<i", len(values)))
            fp.write(values.tobytes())
            fp.flush()
            matrix = Imatrix(fp.name, np)
        np.testing.assert_array_equal(
            matrix.expert(name.decode(), 255, 2, 256),
            np.array([3.0, 4.0], dtype=np.float32),
        )

    def test_q2_common_tensor_recipe(self):
        self.assertEqual(
            regular_qtype("q2", "embedding", "token_embd.weight", QTYPE_BF16),
            QTYPE_Q8_0,
        )
        self.assertEqual(
            regular_qtype("q2", "linear_attention", "blk.0.kda_q.weight", QTYPE_BF16),
            QTYPE_Q4_K,
        )
        self.assertEqual(
            regular_qtype("q2", "linear_attention", "blk.0.kda_v.weight", QTYPE_BF16),
            QTYPE_Q8_0,
        )
        self.assertEqual(
            regular_qtype("q4", "linear_attention", "blk.0.kda_q.weight", QTYPE_BF16),
            QTYPE_Q8_0,
        )
        self.assertEqual(
            regular_qtype("q4", "embedding", "token_embd.weight", QTYPE_BF16),
            QTYPE_Q8_0,
        )

    def test_full_glm_provisional_uses_q2k_experts(self):
        default = []
        provisional = []
        full_glm.add_ffn(default, FakeFullFFNDB(), 3)
        full_glm.add_ffn(provisional, FakeFullFFNDB(), 3, provisional_q2k=True)
        self.assertEqual(
            {item.qtype for item in default if item.is_expert},
            {QTYPE_IQ2_XXS},
        )
        self.assertEqual(
            {item.qtype for item in provisional if item.is_expert},
            {QTYPE_Q2_K},
        )

    def test_full_glm_metadata_keys_are_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "chat_template.jinja"), "wb") as fp:
                fp.write(b"template")
            records = full_glm.model_metadata(directory, "revision")
        keys = []
        for record in records:
            length = struct.unpack("<Q", record[:8])[0]
            keys.append(record[8 : 8 + length].decode())
        self.assertEqual(len(keys), len(set(keys)))

    def test_fp8_e4m3_edge_values(self):
        lut = bare_quantizer().fp8_lut
        self.assertEqual(lut[0x00], 0.0)
        self.assertEqual(lut[0x01], 2.0**-9)
        self.assertEqual(lut[0x07], 7.0 * 2.0**-9)
        self.assertEqual(lut[0x08], 2.0**-6)
        self.assertEqual(lut[0x7E], 448.0)
        self.assertEqual(lut[0x81], -(2.0**-9))
        self.assertTrue(math.isnan(float(lut[0x7F])))
        self.assertTrue(math.isnan(float(lut[0xFF])))

    def test_fp8_block_scale_reconstruction(self):
        codes = np.zeros((256, 256), dtype=np.uint8)
        codes[0, 0] = 0x38
        codes[0, 128] = 0x38
        codes[128, 0] = 0xB8
        codes[128, 128] = 0x38
        scales = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        values = bare_quantizer().to_f32(FakeSourceDB(codes, scales), "weight")
        self.assertEqual(values[0, 0], 1.0)
        self.assertEqual(values[0, 128], 2.0)
        self.assertEqual(values[128, 0], -3.0)
        self.assertEqual(values[128, 128], 4.0)

    def test_fp8_partial_edge_blocks(self):
        codes = np.zeros((129, 257), dtype=np.uint8)
        codes[0, 0] = 0x38
        codes[0, 256] = 0x38
        codes[128, 0] = 0xB8
        codes[128, 256] = 0x38
        scales = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        values = bare_quantizer().to_f32(FakeSourceDB(codes, scales), "weight")
        self.assertEqual(values.shape, codes.shape)
        self.assertEqual(values[0, 0], 1.0)
        self.assertEqual(values[0, 256], 3.0)
        self.assertEqual(values[128, 0], -4.0)
        self.assertEqual(values[128, 256], 6.0)

    def test_fp8_nonfinite_code_is_rejected(self):
        codes = np.zeros((128, 128), dtype=np.uint8)
        codes[3, 7] = 0x7F
        scales = np.ones((1, 1), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "nonfinite FP8 code"):
            bare_quantizer().to_f32(FakeSourceDB(codes, scales), "weight")

    def test_native_fp8_plan_preserves_codes_and_scales(self):
        source = TensorPlan("target.weight", (128, 256), QTYPE_BF16, "dense", "weight")
        source.nbytes = 128 * 256 * 2
        db = FakeNativeDB()
        plan = native_fp8_plan(db, [source])
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0].name, "target.weight")
        self.assertEqual(plan[0].qtype, QTYPE_I8)
        self.assertEqual(plan[0].nbytes, 256 * 128)
        self.assertTrue(plan[0].raw_copy)
        self.assertEqual(plan[1].name, "target.weight_scale_inv")
        self.assertEqual(plan[1].qtype, QTYPE_F32)
        self.assertEqual(plan[1].shape, (1, 2))
        self.assertEqual(plan[1].nbytes, 8)
        self.assertTrue(plan[1].raw_copy)
        self.assertEqual(b"".join(iter_native_tensor_bytes(plan[0], db, np)), db.codes)
        self.assertEqual(b"".join(iter_native_tensor_bytes(plan[1], db, np)), db.scales)

    def test_bf16_round_to_nearest_even(self):
        values = np.array([1.0, -2.5, np.float32(1.00390625)], dtype=np.float32)
        encoded = bare_quantizer().encode(values, QTYPE_BF16)
        bits = np.frombuffer(encoded, dtype="<u2").astype(np.uint32) << 16
        decoded = bits.view(np.float32)
        np.testing.assert_array_equal(decoded[:2], values[:2])
        self.assertEqual(decoded[2], 1.0)

    def test_combined_kv_is_split_per_head_and_k_is_transposed(self):
        values = np.arange(64 * 512 * 512, dtype=np.float32).reshape(64 * 512, 512)
        source = values.reshape(64, 512, 512)
        k_item = TensorPlan("k", (256, 512, 64), 8, "dsa", transform="kv_b_k")
        v_item = TensorPlan("v", (512, 256, 64), 8, "dsa", transform="kv_b_v")
        k = transform_regular(values, k_item)
        v = transform_regular(values, v_item)
        self.assertEqual(k.shape, (64, 512, 256))
        self.assertEqual(v.shape, (64, 256, 512))
        self.assertEqual(k[7, 31, 19], source[7, 19, 31])
        self.assertEqual(v[7, 19, 31], source[7, 256 + 19, 31])

    def test_full_glm_combined_kv_dimensions(self):
        values = np.arange(4 * (3 + 5) * 7, dtype=np.float32).reshape(4 * 8, 7)
        source = values.reshape(4, 8, 7)
        k_item = TensorPlan("k", (3, 7, 4), 8, "dsa", transform="kv_b_k")
        v_item = TensorPlan("v", (7, 5, 4), 8, "dsa", transform="kv_b_v")
        k = transform_regular(values, k_item)
        v = transform_regular(values, v_item)
        self.assertEqual(k.shape, (4, 7, 3))
        self.assertEqual(v.shape, (4, 5, 7))
        self.assertEqual(k[2, 6, 1], source[2, 1, 6])
        self.assertEqual(v[2, 4, 6], source[2, 3 + 4, 6])


if __name__ == "__main__":
    unittest.main()
