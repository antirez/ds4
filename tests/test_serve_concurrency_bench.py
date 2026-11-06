import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "serve_bench", Path(__file__).resolve().parents[1] / "speed-bench/serve_concurrency_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def event(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


class ServingBenchmarkTest(unittest.TestCase):
    text = event({"choices": [{"delta": {"content": "several tokens"}}]})
    finish = event({"choices": [{"delta": {}, "finish_reason": "length"}]})
    usage = event({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 3}})
    done = b"data: [DONE]\n\n"

    def read(self, stream):
        with patch.object(bench.urllib.request, "urlopen", return_value=io.BytesIO(stream)):
            return bench.post_stream("http://localhost/v1/chat/completions", {"max_tokens": 3}, 1)

    def test_counts_tokens_not_chunks(self):
        result = self.read(self.text + self.finish + self.usage + self.done)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output_tokens, 3)
        self.assertEqual(result.prompt_tokens, 4)

    def test_rejects_incomplete_or_failed_streams(self):
        for stream in [
            self.text, self.text + self.finish + self.usage,
            self.text + self.usage + self.done,
            self.text + self.finish + self.done,
            self.text + event({"error": {"message": "decode failed"}}) + self.done,
            self.text + event({"choices": [{"finish_reason": "error"}]}) + self.usage + self.done,
            self.text + self.finish + self.usage + b"data: {broken\n\n" + self.done,
            self.text + self.finish + event({"usage": {"completion_tokens": 4}}) + self.done,
            self.finish + self.usage + self.done,
        ]:
            with self.subTest(stream=stream):
                result = self.read(stream)
                self.assertFalse(result.ok)
                self.assertTrue(result.error)


if __name__ == "__main__":
    unittest.main()
