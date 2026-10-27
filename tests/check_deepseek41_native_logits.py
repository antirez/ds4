#!/usr/bin/env python3
"""Check independent first-token logits and report full-depth rounding sensitivity.

Later-token metrics are diagnostics, not a claim of cross-backend bit equality.
Native FP8/FP4 activation boundaries can amplify a single FP32 rounding bit.
The C runner separately asserts prefill/decode and fresh snapshot continuation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("fixture", type=Path)
p.add_argument("control", type=Path)
a = p.parse_args()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def logits(path):
    x = np.fromfile(path, "<f4").reshape(-1, 64)
    if len(x) < 520 or not np.all(np.isfinite(x)):
        raise ValueError(f"{path}: need at least 520 finite logit rows")
    return x


identity = digest(a.fixture / "model.gguf")
assert identity == digest(a.control / "model.gguf"), "control weights changed"
reference = logits(a.fixture / "reference.f32")
metal = logits(a.fixture / "metal.f32")
stream = logits(a.fixture / "metal-stream.f32")
control = logits(a.control / "reference.f32")
assert reference.shape == metal.shape == stream.shape == control.shape
np.testing.assert_array_equal(metal.view("u4"), stream.view("u4"))
# First token exercises all 40 layers, both Engrams, both cache precisions,
# grouped projections and routed/shared experts against an independent graph.
np.testing.assert_allclose(metal[0], reference[0], rtol=0, atol=1e-6)
report = {
    "fixture_sha256": identity,
    "tokens": len(reference),
    "resident_stream_bitwise_equal": True,
    "first_token_max_abs": float(abs(metal[0] - reference[0]).max()),
}
for name, values in (("metal", metal), ("cpu_double_accumulation_control", control)):
    relative = np.linalg.norm(values - reference, axis=1) / np.linalg.norm(reference, axis=1)
    cosine = (
        np.sum(values * reference, axis=1)
        / np.linalg.norm(values, axis=1)
        / np.linalg.norm(reference, axis=1)
    )
    report[name] = {
        "relative_l2_max": float(relative.max()),
        "relative_l2_mean": float(relative.mean()),
        "cosine_min": float(cosine.min()),
        "cosine_mean": float(cosine.mean()),
        "top1_equal": int((values.argmax(1) == reference.argmax(1)).sum()),
    }
print(json.dumps(report, indent=2))
