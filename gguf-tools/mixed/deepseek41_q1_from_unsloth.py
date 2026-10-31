#!/usr/bin/env python3
"""Build a ds4-native DeepSeek-V4.1-Flash-1017 Q1 GGUF from Unsloth UD-IQ1_S.

The Unsloth UD-IQ1_S split GGUFs cannot be loaded by ds4 directly: ds4 has no
split-GGUF loader and its V4.1 Engram tables use a native disk-row encoding.
But the routed-expert and dense tensors are layout-compatible: identical
names and dims.  This tool transplants the routed experts (and, with
--dense-from-donor, chosen dense roles) from the Unsloth shards into a local
ds4-native 1017 GGUF (the calibrated Q2), which supplies metadata, every other
tensor, and the Engram tail.  Copied tensors are byte-identical.  With
--requant-iq1s, Unsloth's IQ2_XXS experts are dequantized and requantized to
IQ1_S by ggml's own quantizer and imatrix; that is lossy on the layers
Unsloth chose to protect, and is the price of fitting 128 GiB.

Layout strategy (clone-patch): the output starts as an APFS clone (cp -c)
of the base file, so the 189 GiB Engram tail shares blocks with the base.
Resident tensors are repacked from the tensor-data start in base order; the
space freed by smaller tensors becomes a dead gap before the page-aligned
Engram tables.  ds4 maps only up to the page-aligned resident end
(model_unmap_engram in ds4.c), so the gap costs no RAM.  With all options
(see download_model.sh ds41f-q1) resident weights drop from about 152 GiB
(Q2) to about 106 GiB, which fits a 128 GiB Mac without --ssd-streaming.

Run with --dry-run first.  Use --force only to overwrite an existing output.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_DEFAULT_ALIGNMENT = 32

GGUF_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                     10: 8, 11: 8, 12: 8}
GGUF_VALUE_STRING = 8
GGUF_VALUE_ARRAY = 9

# GGML quant type -> (block elements, bytes per block, display name).
GGML_QUANT_SIZES = {
    0: (1, 4, "F32"), 1: (1, 2, "F16"), 2: (32, 18, "Q4_0"),
    8: (32, 34, "Q8_0"), 10: (256, 84, "Q2_K"), 12: (256, 144, "Q4_K"),
    13: (256, 176, "Q5_K"), 14: (256, 210, "Q6_K"), 16: (256, 66, "IQ2_XXS"),
    19: (256, 50, "IQ1_S"), 20: (32, 18, "IQ4_NL"), 24: (1, 1, "I8"),
    26: (1, 4, "I32"), 29: (256, 56, "IQ1_M"), 30: (1, 2, "BF16"),
    39: (32, 17, "MXFP4"),
}

ENGRAM_TENSORS = ("blk.1.engram_embd.weight", "blk.14.engram_embd.weight")
EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight",
                   "ffn_down_exps.weight")
DONOR_EXPERT_TYPES = (16, 19)  # IQ2_XXS, IQ1_S
DONOR_DENSE_TYPES = (8, 13, 14)  # Q8_0, Q5_K, Q6_K
CHUNK = 8 * 1024 * 1024
EXPERTS_PER_TASK = 16


class Die(Exception):
    pass


def align(x: int, n: int) -> int:
    return ((x + n - 1) // n) * n


def tensor_bytes(dims: list[int], qtype: int) -> int:
    if qtype not in GGML_QUANT_SIZES:
        raise Die(f"unsupported GGML type {qtype}")
    elems = 1
    for d in dims:
        elems *= d
    blk_elems, blk_bytes, _ = GGML_QUANT_SIZES[qtype]
    if elems % blk_elems:
        raise Die(f"element count {elems} not divisible by block {blk_elems}")
    return elems // blk_elems * blk_bytes


def type_name(qtype: int) -> str:
    return GGML_QUANT_SIZES.get(qtype, (0, 0, f"T{qtype}"))[2]


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    qtype: int
    rel_offset: int
    type_pos: int = 0
    offset_pos: int = 0
    new_type: int = -1
    new_abs: int = -1
    src_path: Path | None = None
    src_abs: int = -1
    requant: bool = False

    @property
    def nbytes(self) -> int:
        return tensor_bytes(self.dims, self.qtype)


@dataclass
class GgufHeader:
    path: Path
    tensors: list[TensorInfo]
    data_pos: int
    alignment: int = GGUF_DEFAULT_ALIGNMENT
    kv: dict[str, object] = field(default_factory=dict)

    def by_name(self) -> dict[str, TensorInfo]:
        return {t.name: t for t in self.tensors}


class Reader:
    def __init__(self, buf: bytes):
        self.b = buf
        self.p = 0

    def u(self, fmt: str) -> int:
        v = struct.unpack_from(fmt, self.b, self.p)[0]
        self.p += struct.calcsize(fmt)
        return v

    def s(self) -> str:
        n = self.u("<Q")
        v = self.b[self.p:self.p + n].decode("utf8")
        self.p += n
        return v

    def skip_value(self, vtype: int) -> None:
        if vtype == GGUF_VALUE_STRING:
            self.s()
        elif vtype == GGUF_VALUE_ARRAY:
            etype = self.u("<I")
            for _ in range(self.u("<Q")):
                self.skip_value(etype)
        else:
            self.p += GGUF_SCALAR_SIZES[vtype]


def parse_header(path: Path, read_bytes: int = 64 * 1024 * 1024) -> GgufHeader:
    with open(path, "rb") as f:
        buf = f.read(read_bytes)
    r = Reader(buf)
    if r.u("<I") != GGUF_MAGIC:
        raise Die(f"{path}: not a GGUF file")
    if r.u("<I") != 3:
        raise Die(f"{path}: only GGUF v3 is supported")
    n_tensors = r.u("<Q")
    n_kv = r.u("<Q")
    kv: dict[str, object] = {}
    for _ in range(n_kv):
        key = r.s()
        vtype = r.u("<I")
        if vtype == 4:
            kv[key] = r.u("<I")
        elif vtype == GGUF_VALUE_STRING:
            kv[key] = r.s()
        else:
            r.skip_value(vtype)
    tensors: list[TensorInfo] = []
    for _ in range(n_tensors):
        name = r.s()
        ndim = r.u("<I")
        dims = [r.u("<Q") for _ in range(ndim)]
        type_pos = r.p
        qtype = r.u("<I")
        offset_pos = r.p
        rel = r.u("<Q")
        tensors.append(TensorInfo(name, dims, qtype, rel, type_pos, offset_pos))
    alignment = kv.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
    if not isinstance(alignment, int) or alignment <= 0:
        alignment = GGUF_DEFAULT_ALIGNMENT
    data_pos = align(r.p, alignment)
    return GgufHeader(path, tensors, data_pos, alignment, kv)


def copy_range(src_fd: int, src_off: int, dst_fd: int, dst_off: int,
               nbytes: int) -> None:
    done = 0
    while done < nbytes:
        chunk = os.pread(src_fd, min(CHUNK, nbytes - done), src_off + done)
        if not chunk:
            raise Die("short read while copying tensor data")
        os.pwrite(dst_fd, chunk, dst_off + done)
        done += len(chunk)


def sha256_range(fd: int, off: int, nbytes: int) -> str:
    h = hashlib.sha256()
    done = 0
    while done < nbytes:
        chunk = os.pread(fd, min(CHUNK, nbytes - done), off + done)
        if not chunk:
            raise Die("short read while hashing")
        h.update(chunk)
        done += len(chunk)
    return h.hexdigest()


def is_expert(name: str) -> bool:
    return name.endswith(EXPERT_SUFFIXES)


def tensor_role(name: str) -> str:
    return name.split(".", 2)[2] if name.startswith("blk.") else name


@dataclass
class Plan:
    resident: list[TensorInfo]
    resident_end: int
    resident_bytes: int
    engram_start: int
    missing: list[str]


def build_plan(base: GgufHeader,
               donors: dict[str, tuple[Path, TensorInfo, int]],
               requant_iq1s: bool = False,
               dense_suffixes: tuple[str, ...] = ()) -> Plan:
    """Assign new offsets/types. donors: name -> (path, info, data_pos)."""
    base_by_name = base.by_name()
    missing: list[str] = []
    for t in base.tensors:
        t.new_type = t.qtype
        if tensor_role(t.name) in dense_suffixes:
            d = donors.get(t.name)
            if d is None:
                raise Die(f"no donor for dense tensor {t.name}")
            dpath, dinfo, dpos = d
            if dinfo.dims != t.dims:
                raise Die(f"dims mismatch for {t.name}: base {t.dims} "
                          f"donor {dinfo.dims}")
            if dinfo.qtype not in DONOR_DENSE_TYPES:
                raise Die(f"donor type {type_name(dinfo.qtype)} for {t.name} "
                          f"is not Q8_0/Q5_K/Q6_K")
            t.new_type = dinfo.qtype
            t.src_path = dpath
            t.src_abs = dpos + dinfo.rel_offset
        elif is_expert(t.name):
            d = donors.get(t.name)
            if d is None:
                missing.append(t.name)
                continue
            dpath, dinfo, dpos = d
            if dinfo.dims != t.dims:
                raise Die(f"dims mismatch for {t.name}: base {t.dims} "
                          f"donor {dinfo.dims}")
            if dinfo.qtype not in DONOR_EXPERT_TYPES:
                raise Die(f"donor type {type_name(dinfo.qtype)} for {t.name} "
                          f"is not IQ1_S/IQ2_XXS")
            t.new_type = dinfo.qtype
            t.src_path = dpath
            t.src_abs = dpos + dinfo.rel_offset
            if requant_iq1s and dinfo.qtype == 16:
                t.new_type = 19
                t.requant = True
    extra = [n for n in donors if n not in base_by_name]
    if extra:
        raise Die(f"donor tensors not in base: {extra[:4]}")

    engram = [t for t in base.tensors if t.name in ENGRAM_TENSORS]
    if len(engram) != len(ENGRAM_TENSORS):
        raise Die("base file is missing its V4.1 Engram tensors")
    engram_start = min(base.data_pos + t.rel_offset for t in engram)

    cursor = base.data_pos
    resident = sorted((t for t in base.tensors if t.name not in ENGRAM_TENSORS),
                      key=lambda t: t.rel_offset)
    for t in resident:
        cursor = align(cursor, base.alignment)
        t.new_abs = cursor
        cursor += tensor_bytes(t.dims, t.new_type)
    for t in engram:
        t.new_abs = base.data_pos + t.rel_offset
    if cursor > engram_start:
        raise Die(f"resident tensors end at {cursor}, past Engram start "
                  f"{engram_start}; cannot clone-patch")
    return Plan(resident, cursor, cursor - base.data_pos, engram_start,
                missing)


_ggml = None


def _requant_task(task: tuple[str, str, str, int, int, int, int, int, int,
                              str, int, int]) -> int:
    """Dequantize IQ2_XXS experts [e0, e1) and requantize them to IQ1_S with
    ggml's own quantizer and the per-expert importance vector."""
    global _ggml
    import ctypes
    import numpy as np
    (lib, out, src, src_abs, dst_abs, n, rows, e0, e1,
     imx, sum2_abs, counts_abs) = task
    if _ggml is None:
        _ggml = ctypes.CDLL(lib)
        _ggml.dequantize_row_iq2_xxs.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        _ggml.ggml_quantize_chunk.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        _ggml.ggml_quantize_chunk.restype = ctypes.c_size_t
    src_eb = rows * (n // 256 * 66)
    dst_eb = rows * (n // 256 * 50)
    sfd = os.open(src, os.O_RDONLY)
    ofd = os.open(out, os.O_RDWR)
    ifd = os.open(imx, os.O_RDONLY)
    try:
        vals = np.empty(rows * n, np.float32)
        dst = np.empty(dst_eb, np.uint8)
        for e in range(e0, e1):
            raw = os.pread(sfd, src_eb, src_abs + e * src_eb)
            if len(raw) != src_eb:
                raise Die("short read of donor expert")
            _ggml.dequantize_row_iq2_xxs(raw, vals.ctypes.data, rows * n)
            count = np.frombuffer(os.pread(ifd, 4, counts_abs + 4 * e),
                                  np.float32)[0]
            if count > 0:
                w = np.frombuffer(os.pread(ifd, 4 * n, sum2_abs + 4 * n * e),
                                  np.float32) / count
            else:
                w = np.ones(n, np.float32)
            w = np.ascontiguousarray(w, np.float32)
            got = _ggml.ggml_quantize_chunk(19, vals.ctypes.data,
                                            dst.ctypes.data, 0, rows, n,
                                            w.ctypes.data)
            if got != dst_eb:
                raise Die(f"IQ1_S quantizer wrote {got}, expected {dst_eb}")
            os.pwrite(ofd, dst.tobytes(), dst_abs + e * dst_eb)
    finally:
        os.close(sfd)
        os.close(ofd)
        os.close(ifd)
    return e1 - e0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a ds4-native V4.1-Flash-1017 Q1 GGUF from "
                    "Unsloth UD-IQ1_S expert tensors")
    ap.add_argument("--base", required=True, type=Path,
                    help="ds4-native 1017 GGUF (e.g. the calibrated Q2)")
    ap.add_argument("--donor", required=True, type=Path, action="append",
                    help="Unsloth UD-IQ1_S shard GGUF (repeat per shard)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and print the layout plan only")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output file")
    ap.add_argument("--no-clone", action="store_true",
                    help="full copy instead of APFS clone (needs free disk "
                         "for the whole base file)")
    ap.add_argument("--verify-hashes", type=int, default=3,
                    help="spot-check this many expert tensors by SHA-256")
    ap.add_argument("--requant-iq1s", action="store_true",
                    help="requantize donor IQ2_XXS experts to IQ1_S (needs "
                         "--imatrix); lossy on Unsloth's protected layers")
    ap.add_argument("--imatrix", type=Path,
                    help="Unsloth imatrix GGUF (imatrix_unsloth.gguf_file)")
    ap.add_argument("--ggml-lib", type=Path,
                    default=Path("/opt/homebrew/lib/libggml-base.dylib"),
                    help="libggml-base providing the IQ1_S quantizer")
    ap.add_argument("--dense-from-donor", action="append", default=[],
                    metavar="ROLE",
                    help="take donor dense tensors with this role (the name "
                         "after blk.N.; repeat), e.g. attn_q_b.weight")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    try:
        base = parse_header(args.base)
        dense_suffixes = tuple(args.dense_from_donor)
        if args.requant_iq1s and not args.imatrix:
            raise Die("--requant-iq1s needs --imatrix")
        donors: dict[str, tuple[Path, TensorInfo, int]] = {}
        dense_donors: dict[str, tuple[Path, TensorInfo, int]] = {}
        for dpath in args.donor:
            hdr = parse_header(dpath, read_bytes=8 * 1024 * 1024)
            for t in hdr.tensors:
                if is_expert(t.name):
                    if t.name in donors:
                        raise Die(f"duplicate donor tensor {t.name}")
                    donors[t.name] = (dpath, t, hdr.data_pos)
                elif tensor_role(t.name) in dense_suffixes:
                    dense_donors[t.name] = (dpath, t, hdr.data_pos)
        plan = build_plan(base, {**donors, **dense_donors},
                          args.requant_iq1s, dense_suffixes)
        imx_tensors: dict[str, TensorInfo] = {}
        imx_pos = 0
        if args.requant_iq1s:
            imx = parse_header(args.imatrix, read_bytes=8 * 1024 * 1024)
            imx_tensors, imx_pos = imx.by_name(), imx.data_pos
            for t in plan.resident:
                if t.requant and (t.name + ".in_sum2" not in imx_tensors or
                                  t.name + ".counts" not in imx_tensors):
                    raise Die(f"imatrix has no entry for {t.name}")

        experts = [t for t in base.tensors if is_expert(t.name)]
        by_type: dict[str, int] = {}
        for t in experts:
            by_type[type_name(t.new_type)] = \
                by_type.get(type_name(t.new_type), 0) + 1
        gib = 1024 ** 3
        print(f"base:    {args.base} ({len(base.tensors)} tensors, "
              f"data_pos={base.data_pos})")
        print(f"donors:  {len(donors)} expert tensors from "
              f"{len(args.donor)} shard(s); expert mix {by_type}")
        print(f"plan:    resident {plan.resident_bytes / gib:.2f} GiB, "
              f"dead gap {(plan.engram_start - plan.resident_end) / gib:.2f} GiB "
              f"before Engram at {plan.engram_start}")
        if plan.missing:
            print(f"missing: {len(plan.missing)} expert tensors have no "
                  f"donor (first: {plan.missing[0]})")
            if not args.dry_run:
                raise Die("incomplete donor shards; refusing to build")
        if args.dry_run:
            print("dry-run: no output written")
            return 0

        if args.out.exists() and not args.force:
            raise Die(f"{args.out} exists; pass --force to overwrite")
        if args.out.resolve() == args.base.resolve():
            raise Die("output must not be the base file")
        if args.out.exists():
            args.out.unlink()
        if args.no_clone:
            subprocess.run(["cp", str(args.base), str(args.out)], check=True)
        else:
            subprocess.run(["cp", "-c", str(args.base), str(args.out)],
                           check=True)

        out_fd = os.open(args.out, os.O_RDWR)
        src_fds: dict[Path, int] = {}
        try:
            def fd_for(p: Path) -> int:
                if p not in src_fds:
                    src_fds[p] = os.open(p, os.O_RDONLY)
                return src_fds[p]

            moved = skipped = 0
            for t in plan.resident:
                if t.requant:
                    continue
                if t.src_path is None:
                    src_fd, src_abs = fd_for(args.base), \
                        base.data_pos + t.rel_offset
                    if src_abs == t.new_abs:
                        skipped += 1
                        continue
                else:
                    src_fd, src_abs = fd_for(t.src_path), t.src_abs
                copy_range(src_fd, src_abs, out_fd, t.new_abs,
                           tensor_bytes(t.dims, t.new_type))
                moved += 1
            for t in base.tensors:
                if t.new_type != t.qtype:
                    os.pwrite(out_fd, struct.pack("<I", t.new_type),
                              t.type_pos)
                new_rel = t.new_abs - base.data_pos
                if new_rel != t.rel_offset:
                    os.pwrite(out_fd, struct.pack("<Q", new_rel),
                              t.offset_pos)
            print(f"copied {moved} tensors, {skipped} already in place")

            tasks = []
            for t in plan.resident:
                if not t.requant:
                    continue
                assert t.src_path is not None
                n, rows, ne = t.dims
                s2 = imx_tensors[t.name + ".in_sum2"]
                cn = imx_tensors[t.name + ".counts"]
                if s2.dims != [n, ne] or cn.dims != [1, ne]:
                    raise Die(f"imatrix shape mismatch for {t.name}")
                for e0 in range(0, ne, EXPERTS_PER_TASK):
                    tasks.append((str(args.ggml_lib), str(args.out),
                                  str(t.src_path), t.src_abs, t.new_abs,
                                  n, rows, e0, min(ne, e0 + EXPERTS_PER_TASK),
                                  str(args.imatrix), imx_pos + s2.rel_offset,
                                  imx_pos + cn.rel_offset))
            if tasks:
                import multiprocessing as mp
                n_req = sum(1 for t in plan.resident if t.requant)
                print(f"requantizing {n_req} IQ2_XXS tensors to IQ1_S "
                      f"({len(tasks)} tasks, {args.jobs} jobs)", flush=True)
                done = 0
                with mp.get_context("spawn").Pool(args.jobs) as pool:
                    for k in pool.imap_unordered(_requant_task, tasks):
                        done += k
                        if done % (EXPERTS_PER_TASK * 48) == 0:
                            print(f"  {done} experts requantized", flush=True)
                print(f"requantized {done} experts")
            os.fsync(out_fd)

            checks = [t for t in plan.resident if t.src_path is not None
                      and not t.requant][:args.verify_hashes]
            for t in checks:
                assert t.src_path is not None
                n = tensor_bytes(t.dims, t.new_type)
                got = sha256_range(out_fd, t.new_abs, n)
                want = sha256_range(fd_for(t.src_path), t.src_abs, n)
                if got != want:
                    raise Die(f"hash mismatch after copy: {t.name}")
                print(f"verified {t.name} sha256={got[:16]}...")
        finally:
            os.close(out_fd)
            for fd in src_fds.values():
                os.close(fd)

        out_hdr = parse_header(args.out)
        if len(out_hdr.tensors) != len(base.tensors):
            raise Die("output tensor count changed")
        print(f"done: {args.out} ({args.out.stat().st_size / gib:.2f} GiB on "
              f"disk incl. shared clone blocks)")
        return 0
    except Die as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
