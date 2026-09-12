#!/usr/bin/env python3
"""Offline downloader checks with small, genuinely hashed artifact fixtures."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
Q2 = "DeepSeek-V4.1-Flash-Q2.gguf"
Q4 = "DeepSeek-V4.1-Flash-Q4.gguf"
PART1, PART2 = Q4 + ".part1", Q4 + ".part2"
VISION = "DeepSeek-V4.1-Flash-Vision.gguf"
ARTIFACTS = {
    Q2: (365713686528, "1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42"),
    Q4: (518596067328, "a5e2e2c3ada4b2e98d9f9e4b50f6d9c2a12c2c96f5da165c07e13aff9264984e"),
    PART1: (480000000000, "6442b1f9224079662c02003c0ef9ef6be6e2aff509510f681dab9e6cc41df246"),
    PART2: (38596067328, "7c3e10646c918eeaffbc39305a75ec96117450262c61454ff194cef00d7617f0"),
    VISION: (970555552, "cc283f032b3e8b8d78aeb5fccaa14e97b859b0c53aae3cd6bffa690ddf0e9e15"),
}


def payload(name):
    if name == Q4:
        return payload(PART1) + payload(PART2)
    return (name + "\n").encode()


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / "source tree"
        self.root.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.out = self.root / "gguf files"
        self.script = self.root / "download_model.sh"
        text = (ROOT / "download_model.sh").read_text()
        for name, (size, sha) in ARTIFACTS.items():
            data = payload(name)
            self.assertIn("expected_bytes=" + str(size), text)
            self.assertIn("expected_sha=" + sha, text)
            text = text.replace("expected_bytes=" + str(size), "expected_bytes=" + str(len(data)))
            text = text.replace("expected_sha=" + sha, "expected_sha=" + hashlib.sha256(data).hexdigest())
        self.script.write_text(text)
        hf = self.bin / "hf"
        hf.write_text("""#!/usr/bin/env python3
import os
from pathlib import Path
import sys
args = sys.argv[1:]
assert args[0] == 'download'
assert args[1] == 'antirez/deepseek-v4.1-flash-gguf'
if os.environ.get('FAIL_DOWNLOAD'):
    sys.exit(7)
out = Path(args[args.index('--local-dir') + 1])
out.mkdir(parents=True, exist_ok=True)
(out / args[2]).write_bytes((args[2] + '\\n').encode())
""")
        hf.chmod(0o755)
        self.env = dict(os.environ, HOME=str(self.root / "home"), HF_TOKEN="",
                        PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        DS4_GGUF_DIR=str(self.out))

    def run_download(self, target, ok=True):
        result = subprocess.run(["sh", str(self.script), target], env=self.env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_quants_vision_and_existing(self):
        self.assertIn("Verifying SHA-256", self.run_download("ds41f-q2"))
        link = self.root / "ds4flash.gguf"
        self.assertEqual(link.resolve(), self.out / Q2)
        self.assertIn("Already downloaded", self.run_download("ds41f-q2"))
        self.assertIn("Verifying SHA-256", self.run_download("ds41f-q4"))
        self.assertEqual(link.resolve(), self.out / Q4)
        self.assertEqual((self.out / Q4).read_bytes(), payload(Q4))
        for name in (PART1, PART2, Q4 + ".assembling"):
            self.assertFalse((self.out / name).exists())
        self.assertIn("Already downloaded", self.run_download("ds41f-q4"))
        self.assertIn("Verifying SHA-256", self.run_download("ds41f-vision"))
        self.assertEqual(link.resolve(), self.out / Q4)

    def test_truncated_and_corrupt_artifacts(self):
        self.out.mkdir()
        for target, name in (("ds41f-q2", Q2), ("ds41f-q4", Q4), ("ds41f-vision", VISION)):
            with self.subTest(target=target):
                path = self.out / name
                path.write_bytes(b"short")
                self.assertIn("Incorrect file size", self.run_download(target, ok=False))
                self.assertFalse((self.root / "ds4flash.gguf").exists())
                path.write_bytes(b"x" * len(payload(name)))
                self.assertIn("Checksum mismatch", self.run_download(target, ok=False))
                self.assertFalse((self.root / "ds4flash.gguf").exists())

    def test_failure_does_not_replace_link(self):
        self.run_download("ds41f-q2")
        self.env["FAIL_DOWNLOAD"] = "1"
        self.run_download("ds41f-q4", ok=False)
        self.run_download("ds41f-vision", ok=False)
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)

    def test_partial_requires_explicit_cleanup(self):
        self.out.mkdir()
        (self.out / (Q2 + ".part")).write_bytes(b"partial")
        self.assertIn("cannot resume", self.run_download("ds41f-q2", ok=False))
        self.assertFalse((self.root / "ds4flash.gguf").exists())

    def test_q4_resumes_interrupted_assembly(self):
        self.run_download("ds41f-q2")
        self.env["FAIL_DOWNLOAD"] = "1"
        pending = self.out / (Q4 + ".assembling")
        for tail in (b"", payload(PART2)[:7], payload(PART2)):
            with self.subTest(tail=len(tail)):
                pending.write_bytes(payload(PART1) + tail)
                (self.out / PART2).write_bytes(payload(PART2))
                self.run_download("ds41f-q4")
                self.assertEqual((self.out / Q4).read_bytes(), payload(Q4))
                self.assertFalse(pending.exists())
                self.assertFalse((self.out / PART2).exists())
                (self.out / Q4).unlink()

    def test_q4_rejects_corrupt_parts_and_assembly(self):
        self.run_download("ds41f-q2")
        link = self.root / "ds4flash.gguf"
        for name in (PART1, PART2):
            with self.subTest(part=name):
                for other in (PART1, PART2):
                    (self.out / other).write_bytes(payload(other))
                (self.out / name).write_bytes(b"x" * len(payload(name)))
                self.assertIn("Checksum mismatch", self.run_download("ds41f-q4", ok=False))
                self.assertEqual(link.resolve(), self.out / Q2)
                self.assertFalse((self.out / Q4).exists())
        (self.out / PART1).unlink()
        (self.out / PART2).write_bytes(payload(PART2))
        pending = self.out / (Q4 + ".assembling")
        pending.write_bytes(b"x" * len(payload(PART1)))
        self.env["FAIL_DOWNLOAD"] = "1"
        self.assertIn("Checksum mismatch", self.run_download("ds41f-q4", ok=False))
        self.assertEqual(link.resolve(), self.out / Q2)
        self.assertFalse((self.out / Q4).exists())
        self.assertTrue(pending.exists())

    def test_q4_no_space_preserves_resumable_parts(self):
        self.run_download("ds41f-q2")
        for name in (PART1, PART2):
            (self.out / name).write_bytes(payload(name))
        (self.bin / "sitecustomize.py").write_text(
            "import shutil\nfrom collections import namedtuple\n"
            "shutil.disk_usage = lambda _: namedtuple('usage', 'total used free')(1, 1, 0)\n")
        self.env["PYTHONPATH"] = str(self.bin)
        self.assertIn("Not enough disk space", self.run_download("ds41f-q4", ok=False))
        self.assertEqual((self.out / (Q4 + ".assembling")).read_bytes(), payload(PART1))
        self.assertEqual((self.out / PART2).read_bytes(), payload(PART2))
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)
        del self.env["PYTHONPATH"]
        self.env["FAIL_DOWNLOAD"] = "1"
        self.run_download("ds41f-q4")
        self.assertEqual((self.out / Q4).read_bytes(), payload(Q4))

    def test_help_and_invalid_target(self):
        help_text = self.run_download("--help")
        self.assertIn("ds41f-q2", help_text)
        self.assertIn("ds41f-q4", help_text)
        self.assertIn("ds41f-vision", help_text)
        self.assertIn("Unknown model", self.run_download("nonexistent", ok=False))
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
