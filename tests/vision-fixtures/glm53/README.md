# GLM 5.3 vision fixtures

These small images test facts that a plausible generic description cannot
cover: exact OCR, diagram order, spatial relations, screenshot diagnostics,
photograph recognition, and rejection of an unrelated premise. The expected
facts live in `cases.json`.

Run a local `ds4-server` with the GLM 5.3 Flash text GGUF and vision sidecar,
then use:

```sh
python3 tests/run_glm53_vision_quality.py
```

To compare with the official Z.AI provider through OpenRouter:

```sh
DS4_VISION_ENDPOINT=https://openrouter.ai/api/v1/chat/completions \
DS4_VISION_OFFICIAL_ZAI=1 OPENROUTER_API_KEY=... \
python3 tests/run_glm53_vision_quality.py
```

For `ds4-agent`, copy only the PNG/JPEG files to a temporary directory before
testing. Keeping vector sources or expected-answer files beside the images lets
an agent bypass visual inspection with its text tools.

`earth.jpg` is NASA's public-domain Apollo 17 image AS17-148-22727, downloaded
from Wikimedia Commons:
https://commons.wikimedia.org/wiki/File:Earth_apollo17.jpg
