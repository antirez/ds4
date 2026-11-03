#!/usr/bin/env python3
"""Dump the deepseek41.* metadata of a GGUF prefix.

model_unmap_engram() in ds4.c dies unless deepseek41.engram.encoding is exactly
"e4m3_e8m0_32_row264", and requires the engram tensors to be type 24 (I8) with
dim[0] == 264. GSQ-RCO advertises an IQ4_NL engram file, so whether those
agree decides whether a resident configuration is even reachable.

Takes the path as an argument so nothing is imported from the download dir.
"""
import struct
import sys


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

    kv = {}
    for _ in range(n_kv):
        k = r.s()
        kv[k] = r.val(r.u32())

    print("   deepseek41.* metadata")
    for k in sorted(kv):
        if not k.startswith("deepseek41."):
            continue
        v = kv[k]
        if isinstance(v, list):
            v = f"[{len(v)} items]"
        print(f"      {k} = {v}")

    want = "e4m3_e8m0_32_row264"
    got = kv.get("deepseek41.engram.encoding")
    print(f"\n   ds4 requires engram.encoding == {want!r}")
    print(f"   this file has                  {got!r}")
    print(f"   -> {'MATCH' if got == want else 'MISMATCH: model_unmap_engram would die'}")

    # Engram tensors ds4 looks for by name, and the type/shape it demands.
    print("\n   engram tensors in this shard (ds4 wants type 24 = I8, dim[0] == 264)")
    found = 0
    try:
        for _ in range(n_tensors):
            name = r.s()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            t = r.u32()
            r.u64()
            if "engram" in name:
                print(f"      {name}  type={t}  dims={dims}")
                found += 1
    except EOFError:
        pass
    if not found:
        print("      (none in this shard)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
