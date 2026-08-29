#!/usr/bin/env python3
"""Pack the expert weights of a GGUF into one flat file for faster SSD streaming.

In the GGUF, one expert's gate, up and down matrices live in three separate
tensors, about a gigabyte apart on disk. So loading one expert during SSD
streaming costs three scattered reads. This tool copies the bytes of every
expert into a second file (model.gguf.dsp) where the three slices of each
expert sit next to each other:

    expert 0: gate | up | down
    expert 1: gate | up | down
    ...

The engine then loads an expert with one contiguous read. The bytes are copied
exactly, nothing is quantized or transformed, and the engine finds the .dsp
file automatically when it sits next to the model.

By default, after every byte has been verified, the tool releases the expert
space inside the GGUF so you do not store 137 GB twice. The GGUF file stays
and keeps working with this engine, it just shrinks on disk. This is fully
reversible with --restore as long as the .dsp file exists. If you want to keep
the GGUF complete (for example to keep using it with other software), pass
--keep-gguf-experts.

Typical use:

    python3 misc/pack_experts.py model.gguf              # pack, verify, release
    python3 misc/pack_experts.py model.gguf --keep-gguf-experts
    python3 misc/pack_experts.py model.gguf --restore    # put the bytes back
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import platform
import re
import struct
import sys
from pathlib import Path


HEADER_SIZE = 32
ENTRY = struct.Struct("<6Q")
ALIGNMENT = 4096
N_EXPERT = 256


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def parse_layers(spec: str) -> list[int]:
    layers: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo_text, hi_text = item.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                raise ValueError(f"descending layer range: {item}")
            layers.update(range(lo, hi + 1))
        else:
            layers.add(int(item))
    if not layers:
        raise ValueError("no layers selected")
    return sorted(layers)


def load_parser(script_dir: Path):
    parser_path = (
        script_dir.parent
        / "gguf-tools"
        / "mixed"
        / "splice_mixed_expert_layers_gguf.py"
    )
    spec = importlib.util.spec_from_file_location("ds4_gguf_parser", parser_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load GGUF parser at {parser_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # DeepSeek V4 Flash MXFP4: 32 values in one 17-byte block.
    module.GGML_QUANT_SIZES[39] = (32, 17, "MXFP4")
    return module


def discover_layers(by_name: dict) -> list[int]:
    layers = []
    pattern = re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$")
    for name in by_name:
        m = pattern.match(name)
        if m:
            layers.append(int(m.group(1)))
    return sorted(layers)


def release_range(fd: int, offset: int, length: int) -> bool:
    """Release the disk space of a byte range, so reads there return zeros.

    Only the 4096-aligned interior of the range is released. Returns False if
    the file system cannot do this, and the file is left untouched then.
    """
    start = align_up(offset, ALIGNMENT)
    end = align_down(offset + length, ALIGNMENT)
    if end <= start:
        return True
    if platform.system() == "Darwin":
        import fcntl as fcntl_mod

        F_PUNCHHOLE = 99
        # struct fpunchhole: fp_flags u32, reserved u32, fp_offset i64,
        # fp_length i64. Passed through Python's fcntl, which hands the
        # kernel a pointer to a copy of these bytes.
        arg = struct.pack("<IIqq", 0, 0, start, end - start)
        try:
            fcntl_mod.fcntl(fd, F_PUNCHHOLE, arg)
            return True
        except OSError:
            return False
    if platform.system() == "Linux":
        libc = ctypes.CDLL(None, use_errno=True)
        FALLOC_FL_KEEP_SIZE = 1
        FALLOC_FL_PUNCH_HOLE = 2
        libc.fallocate.restype = ctypes.c_int
        libc.fallocate.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
        ]
        return libc.fallocate(
            fd, FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE, start, end - start
        ) == 0
    return False


def disk_usage_gib(path: Path) -> float:
    st = os.stat(path)
    return st.st_blocks * 512 / (1 << 30)


def read_exact(fd: int, length: int, offset: int) -> bytes:
    data = os.pread(fd, length, offset)
    if len(data) != length:
        raise EOFError(f"short read of {length} bytes at offset {offset}")
    return data


def write_exact(fd: int, data: bytes, offset: int) -> None:
    written = 0
    while written < len(data):
        n = os.pwrite(fd, data[written:], offset + written)
        if n <= 0:
            raise OSError("short write")
        written += n


def read_index(dsp_fd: int) -> list[tuple[int, ...]]:
    header = read_exact(dsp_fd, HEADER_SIZE, 0)
    if header[:4] != b"DSP1":
        raise ValueError("this is not a packed expert file (bad header)")
    version, index_off, count = struct.unpack_from("<IQQ", header, 4)
    if version != 1 or count == 0:
        raise ValueError("unsupported packed expert file")
    blob = read_exact(dsp_fd, count * ENTRY.size, index_off)
    return [ENTRY.unpack_from(blob, i * ENTRY.size) for i in range(count)]


def verify_against_gguf(gguf_fd: int, dsp_fd: int,
                        entries: list[tuple[int, ...]]) -> int:
    """Compare every expert byte between the GGUF and the packed file."""
    checked = 0
    for n, (gate_off, up_off, down_off, gate_len, down_len, packed_off) in \
            enumerate(entries):
        slices = (
            (gate_off, packed_off, gate_len),
            (up_off, packed_off + gate_len, gate_len),
            (down_off, packed_off + 2 * gate_len, down_len),
        )
        for gguf_off, dsp_off, length in slices:
            a = read_exact(gguf_fd, length, gguf_off)
            b = read_exact(dsp_fd, length, dsp_off)
            if a != b:
                raise ValueError(
                    f"verification failed at expert {n}, GGUF offset {gguf_off}"
                )
            checked += length
        if (n + 1) % 1024 == 0:
            print(f"verified {n + 1}/{len(entries)} experts", flush=True)
    return checked


def restore(source: Path, dsp_path: Path) -> int:
    if not dsp_path.exists():
        print(f"cannot restore: {dsp_path} does not exist")
        return 1
    gguf_fd = os.open(source, os.O_RDWR)
    dsp_fd = os.open(dsp_path, os.O_RDONLY)
    try:
        entries = read_index(dsp_fd)
        print(f"restoring {len(entries)} experts from {dsp_path.name} "
              f"into {source.name}")
        for n, (gate_off, up_off, down_off, gate_len, down_len, packed_off) in \
                enumerate(entries):
            slices = (
                (gate_off, packed_off, gate_len),
                (up_off, packed_off + gate_len, gate_len),
                (down_off, packed_off + 2 * gate_len, down_len),
            )
            for gguf_off, dsp_off, length in slices:
                write_exact(gguf_fd, read_exact(dsp_fd, length, dsp_off),
                            gguf_off)
            if (n + 1) % 1024 == 0:
                print(f"restored {n + 1}/{len(entries)} experts", flush=True)
        os.fsync(gguf_fd)
        print("verifying...")
        verify_against_gguf(gguf_fd, dsp_fd, entries)
    finally:
        os.close(gguf_fd)
        os.close(dsp_fd)
    print(f"done. {source.name} is complete again "
          f"({disk_usage_gib(source):.1f} GiB on disk). "
          f"you can delete {dsp_path.name} if you want.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="the GGUF model file")
    parser.add_argument(
        "output", type=Path, nargs="?",
        help="where to write the packed expert file "
             "(default: <model>.dsp next to the model, which the engine "
             "finds automatically)",
    )
    parser.add_argument(
        "--layers",
        help="only pack these layers, e.g. 0-7,20 (default: every layer "
             "that has expert tensors)",
    )
    parser.add_argument(
        "--keep-gguf-experts", action="store_true",
        help="keep the expert bytes inside the GGUF too, so the GGUF stays "
             "usable with other software (costs the disk space twice)",
    )
    parser.add_argument(
        "--restore", action="store_true",
        help="copy the expert bytes from the .dsp file back into the GGUF, "
             "undoing the space release",
    )
    parser.add_argument("--force", action="store_true", help="replace output")
    args = parser.parse_args()

    source = args.source.resolve()
    output = (args.output.resolve() if args.output
              else source.with_name(source.name + ".dsp"))

    if args.restore:
        return restore(source, output)

    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists (pass --force to replace it)")

    gguf_parser = load_parser(Path(__file__).resolve().parent)
    info = gguf_parser.parse_gguf(source)
    by_name = info.tensor_by_name

    layers = (parse_layers(args.layers) if args.layers
              else discover_layers(by_name))
    if not layers:
        raise ValueError("no expert tensors found in this GGUF")

    tensor_sets = []
    for layer in layers:
        names = {
            kind: f"blk.{layer}.ffn_{kind}_exps.weight"
            for kind in ("gate", "up", "down")
        }
        tensors = {kind: by_name.get(name) for kind, name in names.items()}
        missing = [names[kind] for kind, tensor in tensors.items() if tensor is None]
        if missing:
            raise ValueError(f"missing expert tensors: {', '.join(missing)}")
        sizes = {kind: tensor.n_bytes // N_EXPERT for kind, tensor in tensors.items()}
        if any(tensor.n_bytes % N_EXPERT for tensor in tensors.values()):
            raise ValueError(f"layer {layer}: tensor is not divisible by {N_EXPERT}")
        if sizes["gate"] != sizes["up"]:
            raise ValueError(f"layer {layer}: gate/up expert sizes differ")
        tensor_sets.append((layer, tensors, sizes))

    # Refuse to pack a GGUF whose expert bytes were already moved out. Real
    # expert weights are never all zeros. The sample sits in the aligned
    # middle of the tensor, because releasing space leaves the unaligned
    # first and last few bytes of a tensor in place.
    source_fd = os.open(source, os.O_RDONLY)
    zero_layers = 0
    for layer, tensors, sizes in tensor_sets:
        tensor = tensors["gate"]
        mid = align_down(tensor.data_offset + tensor.n_bytes // 2, ALIGNMENT)
        sample = read_exact(source_fd, min(65536, sizes["gate"]), mid)
        if not any(sample):
            zero_layers += 1
    if zero_layers == len(tensor_sets):
        os.close(source_fd)
        print("the expert bytes in this GGUF read as all zeros. it looks like")
        print("they were already moved into a packed expert file. refusing to")
        print("pack. if you have the .dsp file and want the bytes back in the")
        print(f"GGUF, run: python3 {sys.argv[0]} --restore {source}")
        return 1
    if zero_layers:
        print(f"warning: {zero_layers} of {len(tensor_sets)} sampled layers "
              f"read as zeros, this GGUF may be damaged")

    # The packed file is a full second copy of the expert bytes, so make
    # sure the disk can hold it before writing anything.
    need = sum(t.n_bytes for _, tensors, _ in tensor_sets
               for t in tensors.values())
    vfs = os.statvfs(output.parent if output.parent.exists() else source.parent)
    free = vfs.f_bavail * vfs.f_frsize
    if free < need + (2 << 30):
        os.close(source_fd)
        print(f"not enough disk space: packing needs "
              f"{need / (1 << 30):.1f} GiB free and this disk has "
              f"{free / (1 << 30):.1f} GiB. after packing you can release "
              f"the same amount inside the GGUF, but during packing both "
              f"copies exist.")
        return 1

    count = len(tensor_sets) * N_EXPERT
    index_offset = HEADER_SIZE
    data_offset = align_up(index_offset + count * ENTRY.size, ALIGNMENT)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    entries: list[bytes] = []
    output_fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
    try:
        os.ftruncate(output_fd, data_offset)
        cursor = data_offset
        completed = 0
        for layer, tensors, sizes in tensor_sets:
            for expert in range(N_EXPERT):
                gate_off = tensors["gate"].data_offset + expert * sizes["gate"]
                up_off = tensors["up"].data_offset + expert * sizes["up"]
                down_off = tensors["down"].data_offset + expert * sizes["down"]
                packed_off = cursor
                for offset, length in (
                    (gate_off, sizes["gate"]),
                    (up_off, sizes["up"]),
                    (down_off, sizes["down"]),
                ):
                    write_exact(output_fd,
                                read_exact(source_fd, length, offset), cursor)
                    cursor += length
                entries.append(
                    ENTRY.pack(
                        gate_off,
                        up_off,
                        down_off,
                        sizes["gate"],
                        sizes["down"],
                        packed_off,
                    )
                )
                completed += 1
                if completed % 1024 == 0:
                    gib = (cursor - data_offset) / (1 << 30)
                    print(
                        f"packed {completed}/{count} experts ({gib:.2f} GiB)",
                        flush=True,
                    )

        header = bytearray(HEADER_SIZE)
        header[:4] = b"DSP1"
        struct.pack_into("<IQQ", header, 4, 1, index_offset, count)
        os.pwrite(output_fd, header, 0)
        index_blob = b"".join(entries)
        if os.pwrite(output_fd, index_blob, index_offset) != len(index_blob):
            raise OSError("short index write")
        os.fsync(output_fd)

        print("verifying every byte...")
        parsed = [ENTRY.unpack(e) for e in entries]
        checked = verify_against_gguf(source_fd, output_fd, parsed)
        print(f"verified: all {checked / (1 << 30):.2f} GiB match the GGUF")
    except BaseException:
        os.close(output_fd)
        output_fd = -1
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(source_fd)
        if output_fd >= 0:
            os.close(output_fd)

    os.replace(temporary, output)
    print(f"wrote {output} ({output.stat().st_size / (1 << 30):.2f} GiB)")

    if args.keep_gguf_experts:
        print("kept the expert bytes inside the GGUF too (--keep-gguf-experts)")
        return 0

    print("releasing the expert space inside the GGUF...")
    gguf_rw = os.open(source, os.O_RDWR)
    try:
        ok = True
        for layer, tensors, sizes in tensor_sets:
            for tensor in tensors.values():
                if not release_range(gguf_rw, tensor.data_offset, tensor.n_bytes):
                    ok = False
                    break
            if not ok:
                break
        os.fsync(gguf_rw)
    finally:
        os.close(gguf_rw)
    if not ok:
        print("this file system cannot release space inside a file. the GGUF")
        print("is unchanged, you now simply have both copies.")
        return 0
    print(f"done. the GGUF now uses {disk_usage_gib(source):.1f} GiB on disk")
    print(f"(its reported size stays the same, the expert region reads as")
    print(f"zeros). the engine finds {output.name} automatically when it sits")
    print(f"next to the model. to undo this:")
    print(f"    python3 {sys.argv[0]} --restore {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
