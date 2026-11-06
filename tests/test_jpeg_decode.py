"""CPU JPEG decoder checks against Pillow/libjpeg. Run with:
uv run --with numpy --with pillow python -m unittest discover -s tests -p test_jpeg_decode.py
"""
import ctypes
import io
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

import numpy as np
from PIL import Image


class DecodedImage(ctypes.Structure):
    _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
                ("rgb", ctypes.POINTER(ctypes.c_uint8)),
                ("fingerprint", ctypes.c_uint8 * 32)]


class JpegDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="ds4-jpeg-test-")
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(__file__).resolve().parents[1]
        lib = Path(cls.temp.name) / "image.so"
        subprocess.run(shlex.split(os.environ.get("CC", "cc")) +
                       ["-shared", "-fPIC", "-O2", str(root / "ds4_image.c"),
                        "-lm", "-o", str(lib)], check=True)
        cls.lib = ctypes.CDLL(str(lib))
        cls.lib.ds4_image_decode_memory.argtypes = [ctypes.POINTER(DecodedImage),
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        cls.lib.ds4_image_decode_memory.restype = ctypes.c_int
        cls.lib.ds4_image_free.argtypes = [ctypes.POINTER(DecodedImage)]
        cls.lib.ds4_image_free.restype = None

    def check_decode(self, encoded):
        expected = np.array(Image.open(io.BytesIO(encoded)).convert("RGB"))
        image = DecodedImage()
        error = ctypes.create_string_buffer(256)
        data = ctypes.create_string_buffer(encoded)
        try:
            self.assertEqual(self.lib.ds4_image_decode_memory(
                ctypes.byref(image), data, len(encoded), error, len(error)), 1,
                error.value.decode())
            actual = np.ctypeslib.as_array(image.rgb,
                shape=(image.height * image.width * 3,)).reshape(image.height, image.width, 3)
            np.testing.assert_array_equal(actual, expected)
        finally:
            self.lib.ds4_image_free(ctypes.byref(image))

    def test_earth_regression(self):
        self.check_decode((Path(__file__).parent / "vision-fixtures/glm53/earth.jpg").read_bytes())

    def test_sampling_progression_and_edges(self):
        for width, height in [(1, 1), (3, 5), (7, 9), (16, 16), (31, 27), (64, 49)]:
            y, x = np.indices((height, width))
            pixels = np.stack([(x * 19 + y * 3) % 256, (y * 31 + x * 7) % 256,
                               ((x // 3 + y // 2) % 2) * 255], axis=-1).astype(np.uint8)
            for progressive in [False, True]:
                for sampling in [0, 1, 2]:
                    with self.subTest(size=(width, height), progressive=progressive, sampling=sampling):
                        buf = io.BytesIO()
                        Image.fromarray(pixels).save(buf, format="JPEG", quality=91,
                            subsampling=sampling, progressive=progressive)
                        self.check_decode(buf.getvalue())

    def test_grayscale(self):
        for progressive in [False, True]:
            with self.subTest(progressive=progressive):
                buf = io.BytesIO()
                Image.fromarray(np.arange(31 * 27, dtype=np.uint8).reshape(27, 31)).save(
                    buf, format="JPEG", progressive=progressive)
                self.check_decode(buf.getvalue())


if __name__ == "__main__":
    unittest.main()
