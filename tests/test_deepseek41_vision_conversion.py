#!/usr/bin/env python3
"""Vision conversion inventory and malformed-source rejection (no model needed)."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-tools"))
from deepseek41_vision import build_plan, source_shapes
from glm53_quantize import QTYPE_BF16, align


class Source:
    def __init__(self):
        self.tensors = {name: {"dtype": "BF16", "shape": list(shape),
                               "nbytes": math.prod(shape) * 2}
                        for name, shape in source_shapes().items()}

    def info(self, name):
        return self.tensors[name]


class VisionConversion(unittest.TestCase):
    def test_lossless_plan(self):
        source = Source()
        source.tensors["layers.0.ffn.gate.bias_vl"] = {"dtype": "F32"}
        plan = build_plan(source)
        self.assertEqual(len(plan), 266)
        self.assertEqual(sum(item.nbytes for item in plan), 970536960)
        end = 0
        for item in plan:
            self.assertEqual(item.qtype, QTYPE_BF16)
            self.assertTrue(item.raw_copy)
            self.assertEqual(item.source, item.name)
            self.assertEqual(item.shape, tuple(reversed(source.info(item.name)["shape"])))
            self.assertEqual(item.offset, end)
            end += align(item.nbytes)

    def test_reject_missing_extra_and_wrong_tensors(self):
        source = Source()
        del source.tensors["image_start"]
        with self.assertRaisesRegex(ValueError, "inventory"):
            build_plan(source)
        source = Source()
        source.tensors["image_pad"] = copy.deepcopy(source.tensors["image_end"])
        with self.assertRaisesRegex(ValueError, "inventory"):
            build_plan(source)
        for field, value in (("dtype", "F32"), ("shape", [4096]), ("nbytes", 0)):
            source = Source()
            source.tensors["image_end"][field] = value
            with self.assertRaises(ValueError):
                build_plan(source)


if __name__ == "__main__":
    unittest.main()
