# Qwen3.8 vision parity fixtures

Two synthetic 640 x 480 images created locally for the CLI and encoder checks:

- `orbit.png`: ORBIT 4729 above a red square on white.
- `maple.png`: MAPLE 8153 above a blue circle on white.

Their dimensions require no resizing at the default 64–1024 token limits.
The ORBIT fixture exposed Q8-versus-original checkpoint drift below the 0.99
minimum per-token cosine threshold, while the same-weight Metal/HF comparison
passed. Keep this fixture when changing quantized reference handling.

The documented suite also uses the existing GLM fixtures for OCR, diagrams,
photography and resize coverage. These are numerical comparisons; the metric
does not grade generated image descriptions.
