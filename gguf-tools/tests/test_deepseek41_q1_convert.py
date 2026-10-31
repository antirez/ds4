import importlib.util
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "mixed" / \
    "deepseek41_q1_from_unsloth.py"
_spec = importlib.util.spec_from_file_location("deepseek41_q1_from_unsloth",
                                               SCRIPT)
assert _spec is not None and _spec.loader is not None
conv = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = conv
_spec.loader.exec_module(conv)

PAGE = 16384


def gguf_string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path: Path,
               tensors: list[tuple[str, list[int], int, bytes]],
               engram_pad: bool) -> None:
    header = struct.pack("<IIQQ", conv.GGUF_MAGIC, 3, len(tensors), 1)
    header += gguf_string("general.architecture")
    header += struct.pack("<I", 8) + gguf_string("deepseek41")
    data_pos = conv.align(
        len(header) + sum(len(gguf_string(n)) + 4 + 8 * len(d) + 4 + 8
                          for n, d, _t, _p in tensors), 32)
    body = bytearray()
    infos = bytearray()
    cursor = 0
    for name, dims, qtype, payload in tensors:
        if name in conv.ENGRAM_TENSORS and engram_pad:
            pad = conv.align(data_pos + cursor, PAGE) - (data_pos + cursor)
            body += b"\0" * pad
            cursor += pad
        cursor = conv.align(cursor, 32)
        body += b"\0" * (cursor - len(body))
        infos += gguf_string(name)
        infos += struct.pack("<I", len(dims))
        infos += b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<IQ", qtype, cursor)
        body += payload
        cursor += len(payload)
    out = header + bytes(infos)
    out += b"\0" * (data_pos - len(out)) + bytes(body)
    path.write_bytes(out)


class Q1ConvertTest(unittest.TestCase):
    def test_clone_patch_transplants_experts_and_keeps_engram(self):
        gate_dims, down_dims = [256, 2], [256, 2]
        emb = (b"\x01" * (256 * 2))
        base_gate = b"\x02" * conv.tensor_bytes(gate_dims, 16)
        base_down = b"\x03" * conv.tensor_bytes(down_dims, 10)
        engram_a = b"\x04" * (264 * 4)
        engram_b = b"\x05" * (264 * 4)
        donor_gate = b"\x12" * conv.tensor_bytes(gate_dims, 19)
        donor_down = b"\x13" * conv.tensor_bytes(down_dims, 19)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            base, donor, out = td / "base.gguf", td / "donor.gguf", td / "o.gguf"
            write_gguf(base, [
                ("token_embd.weight", [256], 1, emb),
                ("blk.0.ffn_gate_exps.weight", gate_dims, 16, base_gate),
                ("blk.0.ffn_down_exps.weight", down_dims, 10, base_down),
                ("blk.1.engram_embd.weight", [264, 4], 24, engram_a),
                ("blk.14.engram_embd.weight", [264, 4], 24, engram_b),
            ], engram_pad=True)
            write_gguf(donor, [
                ("blk.0.ffn_gate_exps.weight", gate_dims, 19, donor_gate),
                ("blk.0.ffn_down_exps.weight", down_dims, 19, donor_down),
            ], engram_pad=False)

            r = subprocess.run(
                [sys.executable, str(SCRIPT), "--base", str(base),
                 "--donor", str(donor), "--out", str(out)],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)

            base_hdr = conv.parse_header(base)
            out_hdr = conv.parse_header(out)
            raw = out.read_bytes()
            by_name = out_hdr.by_name()

            def payload(hdr, name):
                t = hdr.by_name()[name]
                start = hdr.data_pos + t.rel_offset
                return raw[start:start + t.nbytes] if hdr is out_hdr else None

            self.assertEqual(by_name["blk.0.ffn_gate_exps.weight"].qtype, 19)
            self.assertEqual(by_name["blk.0.ffn_down_exps.weight"].qtype, 19)
            self.assertEqual(payload(out_hdr, "blk.0.ffn_gate_exps.weight"),
                             donor_gate)
            self.assertEqual(payload(out_hdr, "blk.0.ffn_down_exps.weight"),
                             donor_down)
            self.assertEqual(payload(out_hdr, "token_embd.weight"), emb)
            base_by = base_hdr.by_name()
            for name, want in (("blk.1.engram_embd.weight", engram_a),
                               ("blk.14.engram_embd.weight", engram_b)):
                self.assertEqual(by_name[name].rel_offset,
                                 base_by[name].rel_offset)
                self.assertEqual(payload(out_hdr, name), want)
            resident_end = max(
                out_hdr.data_pos + t.rel_offset + t.nbytes
                for t in out_hdr.tensors if t.name not in conv.ENGRAM_TENSORS)
            engram_start = out_hdr.data_pos + \
                by_name["blk.1.engram_embd.weight"].rel_offset
            self.assertLessEqual(resident_end, engram_start)
            self.assertEqual(out.stat().st_size, base.stat().st_size)


if __name__ == "__main__":
    unittest.main()
