"""Host-only negative controls for the complete frontier dump gate."""

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_frontier_logits as gate


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def fixture(frontier=2048, previous=0, values=None, **changes):
    values = [1.0, -2.0, 3.0, 0.0] if values is None else values
    values = [f32(value) for value in values]
    best = max(range(len(values)), key=values.__getitem__)
    document = {'source': 'ds4-bench', 'model': '/models/flash', 'backend': 'rocm',
                'quality': False, 'quant_bits': 2, 'prompt_tokens': frontier,
                'frontier_tokens': frontier, 'prefill_tokens': frontier - previous,
                'ctx': 8192, 'vocab': len(values), 'argmax_id': best,
                'argmax_logit': '__ARGMAX__', 'logits': '__LOGITS__'}
    document.update(changes)
    return (json.dumps(document).replace('"__ARGMAX__"', format(values[best], '.9g'))
            .replace('"__LOGITS__"', '[' + ','.join(format(value, '.9g') for value in values) + ']') + '\n').encode()


class FrontierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.old, self.new = self.root / 'old', self.root / 'new'
        self.old.mkdir()
        self.new.mkdir()
        self.expected = gate.validate_expectations([2048, 4096], 8192, '/models/flash', 'rocm', False, 2, 4)
        for directory in (self.old, self.new):
            (directory / 'frontier_002048.logits.json').write_bytes(fixture())
            (directory / 'frontier_004096.logits.json').write_bytes(fixture(4096, 2048))

    def compare(self):
        return gate.compare_directories(self.old, self.new, [2048, 4096], self.expected)

    def assert_bad(self, raw):
        path = self.new / 'frontier_002048.logits.json'
        path.write_bytes(raw)
        report = self.compare()
        self.assertEqual(report['status'], 'invalid')
        self.assertFalse(report['strict_float32_identity'])
        self.assertFalse(report['frontiers'][0]['complete_finite_validation'])
        self.assertTrue(report['frontiers'][1]['complete_finite_validation'])
        self.assertTrue(report['frontiers'][0]['errors'])

    def test_complete_identity_and_full_vector_hashes(self):
        result = self.compare()
        self.assertEqual(result['status'], 'identical')
        self.assertTrue(result['complete_finite_coverage'])
        self.assertEqual(len(result['frontiers']), 2)
        for row in result['frontiers']:
            self.assertEqual(row['values'], 4)
            self.assertEqual(row['changed_count'], 0)
            self.assertEqual(row['baseline']['logits_float32_sha256'], row['candidate']['logits_float32_sha256'])

    def test_non_argmax_drift_cannot_hide_behind_same_argmax(self):
        (self.new / 'frontier_002048.logits.json').write_bytes(fixture(values=[1, -1, 3, 0]))
        (self.new / 'frontier_004096.logits.json').write_bytes(fixture(4096, 2048, values=[1, -2, 3, 1]))
        result = self.compare()
        self.assertEqual(result['status'], 'drift')
        self.assertTrue(result['complete_finite_coverage'])
        self.assertEqual([row['changed_count'] for row in result['frontiers']], [1, 1])
        self.assertEqual(result['frontiers'][0]['first_change']['index'], 1)
        self.assertEqual(result['frontiers'][0]['max_abs'], 1)

    def test_negative_zero_survives_integer_form_json_parsing(self):
        (self.new / 'frontier_002048.logits.json').write_bytes(fixture(values=[1, -2, 3, -0.0]))
        row = self.compare()['frontiers'][0]
        self.assertFalse(row['strict_float32_identity'])
        self.assertEqual(row['changed_count'], 1)
        self.assertEqual(row['max_abs'], 0)
        self.assertEqual(row['first_change']['baseline_bits'], '00000000')
        self.assertEqual(row['first_change']['candidate_bits'], '80000000')

    def test_writer_nine_digit_numbers_recover_original_float32_bits(self):
        literals = [('1.00000012', 0x3f800001), ('-0', 0x80000000),
                    ('1.40129846e-45', 0x00000001), ('3.40282347e+38', 0x7f7fffff),
                    ('-3.40282347e+38', 0xff7fffff), ('0.100000001', 0x3dcccccd)]
        for literal, bits in literals:
            word, _ = gate.binary32(gate.JSONNumber(literal), 'test')
            self.assertEqual(struct.unpack('<I', word)[0], bits)
        for literal in ('0.1', '1.0000001192092896', '1.0', '1e0', '1e-999', '1e999', '3.5e38'):
            with self.subTest(literal=literal), self.assertRaises(gate.InvalidDump):
                gate.binary32(gate.JSONNumber(literal), 'test')

    def test_null_bool_string_nonfinite_and_overflow_logits_are_rejected(self):
        good = fixture()
        for token in ('null', 'true', 'false', '"1"', 'NaN', 'Infinity', '-Infinity', '1e999', '3.5e38', '1e-999'):
            with self.subTest(token=token):
                self.assert_bad(good.replace(b'[1,-2,3,0]', ('[1,-2,3,' + token + ']').encode()))

    def test_numeric_metadata_types_and_counts_are_strict(self):
        numeric = ('quant_bits', 'prompt_tokens', 'frontier_tokens', 'prefill_tokens', 'ctx', 'vocab', 'argmax_id')
        for key in numeric:
            for value in (None, True, False, '2', 2.0, -1, 2**31):
                with self.subTest(key=key, value=value):
                    self.assert_bad(fixture(**{key: value}))
        for changes in ({'quant_bits': 4}, {'prompt_tokens': 2047}, {'frontier_tokens': 4096},
                        {'prefill_tokens': 0}, {'ctx': 4096}, {'vocab': 3}, {'argmax_id': 4}):
            self.assert_bad(fixture(**changes))
        for value in (None, 0, 1, 'false', True):
            self.assert_bad(fixture(quality=value))

    def test_source_model_backend_metadata_must_match_explicit_expectations(self):
        for key in ('source', 'model', 'backend'):
            for value in (None, False, 0, '', 'wrong'):
                with self.subTest(key=key, value=value):
                    self.assert_bad(fixture(**{key: value}))

    def test_full_lengths_argmax_value_and_tie_break_validation(self):
        self.assert_bad(fixture().replace(b'[1,-2,3,0]', b'[1,-2,3]'))
        self.assert_bad(fixture().replace(b'[1,-2,3,0]', b'[1,-2,3,0,4]'))
        self.assert_bad(fixture(argmax_id=0, argmax_logit=1))
        self.assert_bad(fixture(argmax_logit=2))
        self.assert_bad(fixture(argmax_logit=None))
        self.assert_bad(fixture(argmax_logit=True))
        self.assert_bad(fixture(values=[3, 3, 0, 0], argmax_id=1))
        (self.new / 'frontier_002048.logits.json').write_bytes(fixture(values=[3, 3, 0, 0]))
        self.assertTrue(self.compare()['frontiers'][0]['complete_finite_validation'])
        # Equality as a Python number must not hide signed-zero argmax corruption.
        self.assert_bad(fixture(values=[-0.0, -1, -2, -3], argmax_logit=0))

    def test_duplicate_missing_extra_truncated_and_malformed_json_rejected(self):
        good = fixture()
        malformed = [b'', good[:-1], good[:-3], good + b'{}\n', b'[]\n',
                     good.replace(b'{', b'{"source":"ds4-bench",', 1),
                     good.replace(b'"source": "ds4-bench", ', b''),
                     good.replace(b'{', b'{"extra":1,', 1),
                     good.replace(b'"logits": [1,-2,3,0]', b'"logits": null'),
                     good.replace(b'"logits": [1,-2,3,0]', b'"logits": {"a":1,"a":2}'),
                     good.replace(b'flash', b'\xff')]
        for raw in malformed:
            with self.subTest(raw=raw[:60]):
                self.assert_bad(raw)

    def test_first_frontier_is_from_zero_and_second_is_an_increment(self):
        self.assert_bad(fixture(prefill_tokens=1024))
        (self.new / 'frontier_002048.logits.json').write_bytes(fixture())
        (self.new / 'frontier_004096.logits.json').write_bytes(fixture(4096, 0))
        result = self.compare()
        self.assertEqual(result['status'], 'invalid')
        self.assertIn('prefill_tokens', result['frontiers'][1]['errors'][0])

    def test_exact_directory_coverage_and_symlink_rejection(self):
        extra = self.new / 'extra.log'
        extra.write_text('unselected artifact')
        self.assertEqual(self.compare()['status'], 'invalid')
        extra.unlink()
        path = self.new / 'frontier_002048.logits.json'
        path.unlink()
        self.assertEqual(self.compare()['status'], 'invalid')
        path.symlink_to(self.old / path.name)
        self.assertEqual(self.compare()['status'], 'invalid')

    def test_expected_frontiers_and_context_require_explicit_sane_contract(self):
        for frontiers in ([], [0], [-1], [True], [4096, 2048], [2048, 2048]):
            with self.assertRaises(gate.InvalidDump):
                gate.validate_expectations(frontiers, 8192, '/models/flash', 'rocm', False, 2, 4)
            with self.assertRaises(gate.InvalidDump):
                gate.compare_directories(self.old, self.new, frontiers, self.expected)
        for ctx in (4095, 4096, True, 2**31):
            with self.assertRaises(gate.InvalidDump):
                gate.validate_expectations([2048, 4096], ctx, '/models/flash', 'rocm', False, 2, 4)

    def test_cli_exit_status_and_exclusive_output_for_identity_drift_invalid(self):
        command = [sys.executable, str(Path(gate.__file__)), str(self.old), str(self.new),
                   '--frontiers', '2048', '4096', '--ctx', '8192', '--model', '/models/flash',
                   '--backend', 'rocm', '--quality', 'false', '--quant-bits', '2', '--vocab', '4']
        output = self.root / 'identity.json'
        process = subprocess.run(command + ['--output', str(output)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        original = output.read_bytes()
        self.assertEqual(json.loads(original)['status'], 'identical')
        self.assertEqual(subprocess.run(command + ['--output', str(output)], capture_output=True).returncode, 1)
        self.assertEqual(output.read_bytes(), original)
        (self.new / 'frontier_002048.logits.json').write_bytes(fixture(values=[1, -1, 3, 0]))
        output = self.root / 'drift.json'
        self.assertEqual(subprocess.run(command + ['--output', str(output)], capture_output=True).returncode, 1)
        self.assertEqual(json.loads(output.read_bytes())['status'], 'drift')
        (self.new / 'frontier_002048.logits.json').write_bytes(fixture(vocab=True))
        output = self.root / 'invalid.json'
        self.assertEqual(subprocess.run(command + ['--output', str(output)], capture_output=True).returncode, 1)
        self.assertEqual(json.loads(output.read_bytes())['status'], 'invalid')


if __name__ == '__main__':
    unittest.main()
