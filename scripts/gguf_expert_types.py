#!/usr/bin/env python3
"""Report the ggml type of each routed-expert tensor in a GGUF prefix.

ds41_stream_table() in ds4.c accepts one exact combination -- gate and up
IQ2_XXS, down Q2_K -- so extending it for GSQ-RCO needs the real combination,
not an inference from aggregate type counts. Prints every ffn_*_exps tensor it
can reach plus a summary of which types appear in which role.

Takes the path as an argument so nothing is imported from the download dir.
"""
import struct
import sys
from collections import defaultdict

GGML_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 39: "MXFP4",
}


class R:
    def __init__(self, b):
        self.b, self.o = b, 0

    def take(self, n):
        if self.o + n > len(self.b):
            raise EOFError(f"need {n} at {self.o}, have {len(self.b)}")
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def s(self):
        return self.take(self.u64()).decode("utf-8", "replace")

    def val(self, t):
        # Consume every element, array tails included: sampling desyncs
        # everything that follows.
        if t in (0, 1, 7):
            return self.take(1)[0]
        if t in (2, 3):
            return struct.unpack("<H", self.take(2))[0]
        if t in (4, 5):
            return self.u32()
        if t == 6:
            return struct.unpack("<f", self.take(4))[0]
        if t == 8:
            return self.s()
        if t == 9:
            et = self.u32()
            n = self.u64()
            return [self.val(et) for _ in range(n)]
        if t in (10, 11):
            return self.u64()
        if t == 12:
            return struct.unpack("<d", self.take(8))[0]
        raise ValueError(f"unknown metadata type {t}")


def main(path):
    r = R(open(path, "rb").read())
    assert r.take(4) == b"GGUF", "not a GGUF file"
    r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()

    for _ in range(n_kv):
        k = r.s()
        r.val(r.u32())

    roles = defaultdict(set)
    shown = 0
    truncated = False
    seen = 0
    try:
        for _ in range(n_tensors):
            name = r.s()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            t = r.u32()
            r.u64()
            seen += 1
            if "_exps" not in name:
                continue
            tn = GGML_TYPE.get(t, f"type{t}")
            role = name.split(".")[-2] if "." in name else name
            roles[role].add(tn)
            if shown < 200:
                print(f"   {tn:9} {name}  dims={dims}")
                shown += 1
    except EOFError:
        truncated = True

    print(f"\n   read {seen} of {n_tensors} tensor entries"
          f"{' (prefix truncated)' if truncated else ''}")
    print("\n   role -> types seen in this shard")
    for role in sorted(roles):
        print(f"      {role:16} {', '.join(sorted(roles[role]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
