# Qwen3.8 vision implementation parity and quantization quality

September 7, 2026, Apple M3 Ultra with 512 GiB. The downloaded Q8 encoder was
compared with HF twice: using identical dequantized GGUF weights for execution
parity, and using the original checkpoint for quantization quality. Both retain
the 0.99 minimum per-token cosine threshold. See the
[reproduction command](../docs/QWEN38_FLASH_NEXT.md) and
[metrics, hashes and package versions](qwen38-vision-parity-results.json).

All six images now pass implementation parity and the suite exits zero.
Before the decoder fix, the Earth JPEG failed at 0.957435 minimum cosine. A
diagnostic lossless PNG created from Pillow's decoded pixels passed with the
same resize, isolating JPEG decoding as the cause. Iris now uses centered
chroma interpolation and rounded color conversion, following the algorithms
described in libjpeg-turbo's [upsampling](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/jdsample.c)
and [color conversion](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/jdcolor.c).
Earth's decoded RGB pixels now match Pillow exactly, and its implementation
parity is 0.999999998. The results retain the original failure for comparison.

The two synthetic fixtures require no resize. Diagram, OCR, and screenshot
fixtures cover rounding dimensions to patch multiples and reducing images to
the 1024-token limit. All five pass implementation parity, with minimum cosine
at least 0.9995718.

Five of six images also pass quantization quality, but the failing image is
different: ORBIT scores 0.982068 for HF Q8 versus HF original, while its
same-weight implementation parity is above 0.9999998. This preserves the
previously identified Q8 degradation as a quality failure rather than a Metal
execution failure. Explicit `--require-quality` on ORBIT exits 1 as expected.

Four model-free metric tests pass, covering layout mismatch, non-finite inputs,
a bad individual token hidden by a high mean, and cosine versus absolute error.
The harness loads every GGUF tensor and checks the HF state mapping, including
reassembly of the two temporal patch convolutions. Decoder regression tests
check 39 JPEG cases with exact Pillow pixel agreement: baseline and progressive
scans, 4:4:4/4:2:2/4:2:0 sampling, grayscale, tiny images and odd dimensions.
Those tests also exposed premature progressive scan termination when Huffman
lookahead reached a marker with valid bits still buffered. The decoder now
consumes those bits. The existing DeepSeek image test and builds of the CLI,
server and Qwen vision test pass.

No cosine threshold was relaxed. JPEG pixels change for all models using the
shared Iris decoder; exact agreement is established for these fixtures, not
every JPEG encoding. Functional image answer quality remains covered by the separate
[API smoke results](qwen38-vision-quality.md); neither this small numerical suite
nor those smoke tests establishes general vision accuracy.
