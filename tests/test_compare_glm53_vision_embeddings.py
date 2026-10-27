"""The quality gate must reject corrupt references as well as outputs."""

import array
import pathlib
import subprocess
import sys
import tempfile
import unittest


class VisionComparisonTest(unittest.TestCase):
    def compare(self, reference, candidate):
        with tempfile.TemporaryDirectory() as directory:
            paths = [pathlib.Path(directory) / name for name in ('ref', 'out')]
            for path, values in zip(paths, (reference, candidate)):
                path.write_bytes(array.array('f', values).tobytes())
            return subprocess.run(
                [sys.executable,
                 str(pathlib.Path(__file__).with_name('compare_glm53_vision_embeddings.py')),
                 *map(str, paths)], capture_output=True, text=True).returncode

    def test_equal(self):
        self.assertEqual(self.compare([1, 2, 3], [1, 2, 3]), 0)

    def test_invalid(self):
        for bad in ([float('nan'), 2, 3], [float('inf'), 2, 3], [0, 0, 0], []):
            with self.subTest(values=bad):
                self.assertNotEqual(self.compare(bad, [1, 2, 3]), 0)
                self.assertNotEqual(self.compare([1, 2, 3], bad), 0)

    def test_different(self):
        self.assertNotEqual(self.compare([1, 2, 3], [-1, -2, -3]), 0)


if __name__ == '__main__':
    unittest.main()
