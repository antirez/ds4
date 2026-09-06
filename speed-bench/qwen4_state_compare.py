#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0"]
# ///
"""Compare complete Qwen DS4 session payloads from two prefill schedules."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


HEADER_FIELDS = 13
LAYERS = 48
VOCAB = 248_320
KV_ROW = 2 * 256
INDEX_DIM = 128
INDEX_RATIO = 4
GDN_CONV = (2 * 16 + 48) * 128 * 4
GDN_STATE = 48 * 128 * 128
PLE_STATE = 4 * 2560 * 9
HC_STATE = 4 * 2560
MTP_ENABLED = 1


class Payload:
    def __init__(self, path: Path):
        self.path = path
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        if self.data.size < HEADER_FIELDS * 4:
            raise ValueError(f"{path}: truncated header")
        self.header = struct.unpack_from("<13I", self.data, 0)
        self.offset = HEADER_FIELDS * 4

    def take(self, dtype: str, count: int, label: str) -> np.ndarray:
        itemsize = np.dtype(dtype).itemsize
        size = count * itemsize
        if count < 0 or self.offset + size > self.data.size:
            raise ValueError(f"{self.path}: truncated {label}")
        result = np.ndarray(
            (count,), dtype=dtype, buffer=self.data, offset=self.offset
        )
        self.offset += size
        return result


def compare_exact(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape or not np.array_equal(left, right):
        unequal = np.flatnonzero(left != right)
        first = int(unequal[0]) if unequal.size else -1
        raise ValueError(
            f"{label}: exact mismatch at {first}: "
            f"{left[first] if first >= 0 else '?'} != "
            f"{right[first] if first >= 0 else '?'}"
        )


def compare_float(left: np.ndarray, right: np.ndarray, label: str,
                  atol: float, rtol: float) -> None:
    close = np.isclose(left, right, atol=atol, rtol=rtol, equal_nan=False)
    if not bool(np.all(close)):
        first = int(np.flatnonzero(~close)[0])
        delta = abs(float(left[first]) - float(right[first]))
        raise ValueError(
            f"{label}: mismatch at {first}: {left[first]:.9g} != "
            f"{right[first]:.9g} (abs {delta:.3g})"
        )
    delta = float(np.max(np.abs(left - right), initial=0.0))
    print(f"{label}: PASS values={left.size} max_abs={delta:.3g}")


def compare_qsa(left: Payload, right: Payload, live: int,
                prefix: str) -> None:
    pooled = live // INDEX_RATIO
    for name, count in (
        ("key", live * KV_ROW),
        ("value", live * KV_ROW),
        ("raw-index", live * INDEX_DIM),
        ("pooled-index", pooled * INDEX_DIM),
    ):
        compare_exact(
            left.take("<u2", count, f"{prefix} {name}"),
            right.take("<u2", count, f"{prefix} {name}"),
            f"{prefix} {name}",
        )


def compare(left_path: Path, right_path: Path,
            atol: float, rtol: float) -> None:
    left = Payload(left_path)
    right = Payload(right_path)
    # Field 3 is the scratch/work cap selected by the run. It is deliberately
    # not durable model state and is expected to differ for 2K versus 8K.
    for index, (a, b) in enumerate(zip(left.header, right.header)):
        if index != 3 and a != b:
            raise ValueError(f"header field {index}: {a} != {b}")
    if left.header[3] not in (2048, 4096, 8192) or right.header[3] not in (
        2048, 4096, 8192
    ):
        raise ValueError("payload contains an unsupported Qwen work cap")
    live = left.header[4]
    mtp_live = left.header[6]
    tokens = left.header[7]
    flags = left.header[12]
    if live != tokens:
        raise ValueError("Qwen payload cache frontier differs from token count")

    compare_exact(
        left.take("<u4", tokens, "tokens"),
        right.take("<u4", tokens, "tokens"),
        "tokens",
    )
    compare_float(
        left.take("<f4", VOCAB, "logits"),
        right.take("<f4", VOCAB, "logits"),
        "logits", atol, rtol,
    )
    for layer in range(LAYERS):
        prefix = f"layer {layer}"
        if (layer + 1) % 4 == 0:
            compare_qsa(left, right, live, prefix)
        else:
            compare_float(
                left.take("<f4", GDN_CONV, f"{prefix} convolution"),
                right.take("<f4", GDN_CONV, f"{prefix} convolution"),
                f"{prefix} convolution", atol, rtol,
            )
            compare_float(
                left.take("<f4", GDN_STATE, f"{prefix} recurrence"),
                right.take("<f4", GDN_STATE, f"{prefix} recurrence"),
                f"{prefix} recurrence", atol, rtol,
            )
    compare_float(
        left.take("<f4", PLE_STATE, "PLE state"),
        right.take("<f4", PLE_STATE, "PLE state"),
        "PLE state", atol, rtol,
    )
    if flags & MTP_ENABLED:
        compare_float(
            left.take("<f4", HC_STATE, "MTP state"),
            right.take("<f4", HC_STATE, "MTP state"),
            "MTP state", atol, rtol,
        )
        compare_float(
            left.take("<f4", HC_STATE, "MTP trunk"),
            right.take("<f4", HC_STATE, "MTP trunk"),
            "MTP trunk", atol, rtol,
        )
        compare_qsa(left, right, mtp_live, "MTP QSA")
    if left.offset != left.data.size or right.offset != right.data.size:
        raise ValueError(
            f"trailing bytes: left={left.data.size-left.offset}, "
            f"right={right.data.size-right.offset}"
        )
    print(
        f"Qwen state parity PASS tokens={tokens} "
        f"chunks={left.header[3]}/{right.header[3]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-4)
    args = parser.parse_args()
    compare(args.left, args.right, args.atol, args.rtol)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"qwen4-state-compare: {error}")
        raise SystemExit(2)
