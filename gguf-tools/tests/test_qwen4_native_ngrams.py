"""Lossless Qwen packaging: source BF16 bytes and calibrated tensors survive."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen4_native_ngrams as pack
from qwen4_pack_to_qwen4exp import Arr, Reader, kv_bytes, w_str


class NativeNgrams(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root/'model.gguf'
        self.output = self.root/'out.gguf'
        self.source = self.root/'source'
        self.source.mkdir()
        self.aux = {'layer_multipliers': [11, 13, 17], 'head_offsets': [0, 2],
                    'head_vocab_sizes': [2, 4]}
        self.metadata = {
            'general.architecture': (8, 'qwen4exp'),
            'general.alignment': (4, 32),
            'qwen4exp.ple.row_dimension': (4, 160),
            'qwen4exp.ple.row_count': (10, 6),
            **{'qwen4exp.ple.'+k: (9, Arr(10, v)) for k, v in self.aux.items()}}
        self.payload = bytes(range(66))
        self.write_model()
        # Deliberately put shard 1 before shard 0 in the index and file.
        self.raw = bytes(range(256))*7 + bytes(range(128))
        self.assertEqual(len(self.raw), 6*160*2)
        self.source_data = {
            'ngram_embedding.shard_1.weight': ('BF16', [3, 160], self.raw[960:]),
            'ngram_embedding.shard_0.weight': ('BF16', [3, 160], self.raw[:960]),
            'layer_multipliers': ('I64', [3], struct.pack('<3q', *self.aux['layer_multipliers'])),
            'ngram_heads_offsets': ('I64', [2], struct.pack('<2q', *self.aux['head_offsets'])),
            'ngram_heads_vocab_sizes': ('I64', [2], struct.pack('<2q', *self.aux['head_vocab_sizes']))}
        self.write_source()

    def write_model(self):
        header = b'GGUF' + struct.pack('<IQQ', 3, 1, len(self.metadata))
        header += b''.join(kv_bytes(k, t, v) for k, (t, v) in self.metadata.items())
        header += w_str('blk.0.ffn_gate_exps.weight') + struct.pack('<I3QIQ', 3, 256, 1, 1, 16, 0)
        self.model.write_bytes(header + bytes((-len(header)) % 32) + self.payload)

    def write_source(self):
        header, data = {}, b''
        for name, (kind, shape, raw) in self.source_data.items():
            header[pack.SOURCE_PREFIX+name] = dict(dtype=kind, shape=shape,
                data_offsets=[len(data), len(data)+len(raw)])
            data += raw
        blob = json.dumps(header).encode()
        (self.source/'shard.safetensors').write_bytes(struct.pack('<Q', len(blob))+blob+data)
        (self.source/'model.safetensors.index.json').write_text(json.dumps(
            {'weight_map': {name: 'shard.safetensors' for name in header}}))

    def run_pack(self):
        return pack.repack(self.model, self.source, self.output, 'a'*40)

    def test_payloads_and_layout(self):
        report = self.run_pack()
        reader = Reader(str(self.output))
        self.addCleanup(reader.f.close)
        kind, dims, offset = reader.tensors[pack.NGRAM]
        self.assertEqual((kind, dims), (30, [160, 6]))
        self.assertEqual((reader.data_start+offset) % 65536, 0)
        reader.f.seek(reader.data_start+offset)
        self.assertEqual(reader.f.read(), self.raw)
        kind, dims, offset = reader.tensors['blk.0.ffn_gate_exps.weight']
        self.assertEqual((kind, dims), (16, [256, 1, 1]))
        reader.f.seek(reader.data_start+offset)
        self.assertEqual(reader.f.read(len(self.payload)), self.payload)
        self.assertEqual(report['sha256'], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertFalse(Path(str(self.output)+'.incomplete').exists())
        with self.assertRaises(ValueError): self.run_pack()

    def test_reject_wrong_source_precision(self):
        _, shape, raw = self.source_data['ngram_embedding.shard_0.weight']
        self.source_data['ngram_embedding.shard_0.weight'] = ('F16', shape, raw)
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'original BF16'): self.run_pack()
        self.assertFalse(self.output.exists())

    def test_upstream_geometry_metadata(self):
        del self.metadata['qwen4exp.ple.row_dimension']
        del self.metadata['qwen4exp.ple.row_count']
        self.metadata['qwen4exp.embedding_length_per_layer_input'] = (4, 160)
        self.write_model()
        self.run_pack()
        reader = Reader(str(self.output))
        self.addCleanup(reader.f.close)
        self.assertEqual(reader.tensors[pack.NGRAM][1], [160, 6])

    def test_source_padding_is_retained(self):
        # The released table has more rows than its hash ranges address.
        self.raw += bytes(320)
        self.source_data['ngram_embedding.shard_1.weight'] = ('BF16', [4, 160], self.raw[960:])
        self.write_source()
        for explicit in (True, False):
            if explicit:
                self.metadata['qwen4exp.ple.row_count'] = (10, 7)
            else:
                del self.metadata['qwen4exp.ple.row_count']
            self.write_model()
            self.output = self.root/f'padded-{explicit}.gguf'
            self.run_pack()
            reader = Reader(str(self.output))
            with reader.f:
                self.assertEqual(reader.tensors[pack.NGRAM][1], [160, 7])
                reader.f.seek(reader.data_start+reader.tensors[pack.NGRAM][2])
                self.assertEqual(reader.f.read(), self.raw)

    def test_reuse_verified_native_gguf(self):
        self.run_pack()
        second = self.root/'second.gguf'
        pack.repack(self.model, self.output, second, 'a'*40)
        self.assertEqual(self.output.read_bytes(), second.read_bytes())
        with self.assertRaisesRegex(ValueError, 'revision'):
            pack.repack(self.model, self.output, self.root/'wrong.gguf', 'b'*40)

    def test_reject_wrong_hash_geometry(self):
        self.source_data['layer_multipliers'] = ('I64', [3], struct.pack('<3q', 11, 13, 19))
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'hashes'): self.run_pack()

    def test_reject_missing_shard(self):
        del self.source_data['ngram_embedding.shard_0.weight']
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'Missing'): self.run_pack()

    def test_reject_wrong_row_count(self):
        self.metadata['qwen4exp.ple.row_count'] = (10, 7)
        self.write_model()
        with self.assertRaisesRegex(ValueError, 'row count'): self.run_pack()

    def test_reject_truncated_source(self):
        path = self.source/'shard.safetensors'
        with path.open('r+b') as fp:
            fp.truncate(path.stat().st_size-1)
        with self.assertRaisesRegex(ValueError, 'offsets'): self.run_pack()


if __name__ == '__main__':
    unittest.main()
