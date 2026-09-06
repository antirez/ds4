#!/usr/bin/env python3
"""Model-backed regression for full GLM attention (not the Flash checkpoint).

Run on an idle GPU host, with external memory monitoring for SSD streaming:
  python3 tests/test_glm_rope_prefill.py --model /path/to/GLM-5.3-Q2.gguf
The default uses bounded SSD streaming, never full residency.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--bin', default='./ds4')
    parser.add_argument('--cache', default='16GB')
    args = parser.parse_args()
    base = [args.bin, '-m', args.model, '--ssd-streaming',
            '--ssd-streaming-cache-experts', args.cache,
            '--ctx', '2048', '--nothink', '--temp', '0', '--seed', '1']
    env = os.environ.copy()
    env.pop('DS4_GLM_DISABLE_FLASH_PREFILL', None)
    notes = ('Alpha validates incoming records. Beta stores the records. '
             'Gamma detects and reports anomalies. Neither Alpha nor Beta '
             'reports anomalies.\n')
    prompt = notes * 12 + '\nWhich component reports anomalies? Reply only with its name.'
    with tempfile.TemporaryDirectory(prefix='ds4-glm-rope-') as tmp:
        directory = Path(tmp)
        logits = []
        for disabled in (False, True):
            run_env = env.copy()
            if disabled:
                run_env['DS4_GLM_DISABLE_FLASH_PREFILL'] = '1'
            path = directory / ('reference.json' if disabled else 'default.json')
            subprocess.run(base + ['-p', prompt, '--dump-logits', str(path)],
                           env=run_env, check=True, timeout=1200)
            result = json.loads(path.read_text())
            if result['prompt_tokens'] < 256:
                raise AssertionError('prompt did not exercise batched prefill')
            logits.append(result['logits'])
        if not logits[0] or len(logits[0]) != len(logits[1]):
            raise AssertionError('missing or differently sized logits')
        if not all(math.isfinite(x) for row in logits for x in row):
            raise AssertionError('non-finite logits')
        error = max(abs(a - b) for a, b in zip(*logits))
        if error > 1e-5:
            raise AssertionError(f'default prefill differs from RoPE control: {error}')
        result = subprocess.run(base + ['-p', prompt, '--tokens', '16'],
                                env=env, check=True, text=True,
                                stdout=subprocess.PIPE, timeout=1200)
        answer = result.stdout.strip().rstrip('.').strip()
        if answer != 'Gamma':
            raise AssertionError(f'incorrect answer: {answer!r}')
        print(f'GLM RoPE prefill: PASS, {len(logits[0])} logits, max error {error}')


if __name__ == '__main__':
    main()
