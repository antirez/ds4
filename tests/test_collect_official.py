#!/usr/bin/env python3
"""Collector requests must record the distribution actually requested."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "collector", ROOT / "gguf-tools/quality-testing/collect_official.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectorTests(unittest.TestCase):
    def test_temperature_and_record(self):
        for temperature in (None, "1"):
            with self.subTest(temperature=temperature), tempfile.TemporaryDirectory() as td:
                prompt = Path(td) / "input.jsonl"
                prompt.write_text('{"prompt": "Why?"}\n')
                out = Path(td) / "output"
                argv = ["collect", "--out", str(out), "--prompts", str(prompt),
                        "--count", "1", "--api-key-env", "TEST_API_KEY"]
                if temperature:
                    argv += ["--temperature", temperature]
                response = {"model": "test-model", "choices": [{
                    "message": {"content": "Because."},
                    "logprobs": {"content": [{"token": "Because", "logprob": -0.5}]}}]}
                with patch("sys.argv", argv), patch.dict("os.environ", {"TEST_API_KEY": "test"}), \
                     patch.object(collector.urllib.request, "urlopen",
                                  return_value=io.BytesIO(json.dumps(response).encode())) as request, \
                     patch.object(collector.time, "sleep"):
                    self.assertEqual(collector.main(), 0)
                sent = json.loads(request.call_args.args[0].data)
                recorded = json.loads((out / "collection.json").read_text())
                expected = float(temperature or 0)
                self.assertEqual(sent["temperature"], expected)
                self.assertEqual(recorded["temperature"], expected)
                self.assertTrue(sent["logprobs"])
                self.assertEqual(json.loads((out / "responses/case_000.json").read_text()), response)

    def test_invalid_temperature(self):
        for value in ("-0.1", "2.1", "nan", "inf", "-inf"):
            with self.subTest(value=value), patch("sys.argv", ["collect", "--temperature=" + value]), \
                 patch.object(collector.urllib.request, "urlopen") as request:
                with self.assertRaisesRegex(SystemExit, "--temperature must"):
                    collector.main()
                request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
