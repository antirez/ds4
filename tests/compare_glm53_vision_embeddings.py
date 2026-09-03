#!/usr/bin/env python3
"""Compare a GPU vision embedding with an accepted reference."""

import array
import math
import pathlib
import sys


def read_f32(path):
    values = array.array("f")
    values.frombytes(pathlib.Path(path).read_bytes())
    return values


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} REFERENCE.f32 CANDIDATE.f32", file=sys.stderr)
        return 2
    reference = read_f32(sys.argv[1])
    candidate = read_f32(sys.argv[2])
    if not reference or len(reference) != len(candidate):
        print("vision embedding lengths differ or are empty", file=sys.stderr)
        return 1

    dot = ref_sq = candidate_sq = sum_abs = sum_sq = max_abs = 0.0
    nonfinite = 0
    for expected, actual in zip(reference, candidate):
        if not math.isfinite(actual):
            nonfinite += 1
            continue
        delta = abs(expected - actual)
        max_abs = max(max_abs, delta)
        sum_abs += delta
        sum_sq += delta * delta
        dot += expected * actual
        ref_sq += expected * expected
        candidate_sq += actual * actual

    count = len(reference)
    cosine = dot / math.sqrt(ref_sq * candidate_sq)
    mean_abs = sum_abs / count
    rms = math.sqrt(sum_sq / count)
    print(f"values={count} nonfinite={nonfinite} max_abs={max_abs:.9g} "
          f"mean_abs={mean_abs:.9g} rms={rms:.9g} cosine={cosine:.12f}")
    if nonfinite or max_abs > 0.06 or mean_abs > 0.001 or cosine < 0.995:
        print("vision embedding comparison failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
