#!/usr/bin/env python3
"""Validate complete ds4-bench frontier dumps and require exact finite float32 bits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys


FIELDS = {'source', 'model', 'backend', 'quality', 'quant_bits', 'prompt_tokens',
          'frontier_tokens', 'prefill_tokens', 'ctx', 'vocab', 'argmax_id',
          'argmax_logit', 'logits'}
INT_MAX = 2**31 - 1


class InvalidDump(ValueError):
    pass


class JSONNumber(str):
    """Keep decimal spelling, including integer-form -0, until field validation."""


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidDump(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def reject_constant(value):
    raise InvalidDump(f'non-finite JSON constant: {value}')


def integer(value, name, minimum=0):
    if type(value) is not JSONNumber or not re.fullmatch(r'0|[1-9][0-9]*', value):
        raise InvalidDump(f'{name}: expected canonical JSON integer, not bool/null/string/float')
    number = int(value)
    if not minimum <= number <= INT_MAX:
        raise InvalidDump(f'{name}: integer out of range')
    return number


def binary32(value, name):
    if type(value) is not JSONNumber:
        raise InvalidDump(f'{name}: expected a JSON number, not bool/null/string')
    try:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise InvalidDump(f'{name}: non-finite number')
        raw = struct.pack('<f', parsed)
        rounded = struct.unpack('<f', raw)[0]
        if not math.isfinite(rounded):
            raise InvalidDump(f'{name}: non-finite binary32 value')
    except (OverflowError, ValueError) as exc:
        raise InvalidDump(f'{name}: invalid finite binary32 number: {exc}') from exc
    # The C writer promotes float to double and prints %.9g. Re-formatting the
    # recovered binary32 verifies the source spelling and preserves signed zero.
    if format(rounded, '.9g') != value:
        raise InvalidDump(f'{name}: not the writer\'s canonical %.9g binary32 representation')
    return raw, rounded


def validate_expectations(frontiers, ctx, model, backend, quality, quant_bits, vocab):
    if (not frontiers or any(type(n) is not int or not 0 < n <= INT_MAX for n in frontiers)
            or frontiers != sorted(set(frontiers))):
        raise InvalidDump('expected frontiers must be unique, positive, strictly increasing integers')
    if type(ctx) is not int or not frontiers[-1] < ctx <= INT_MAX:
        raise InvalidDump('expected ctx must exceed the final frontier and fit a signed C int')
    if type(model) is not str or not model or '\0' in model:
        raise InvalidDump('expected model must be a nonempty path string without NUL')
    if backend not in ('rocm', 'cuda', 'metal', 'cpu') or type(backend) is not str:
        raise InvalidDump('unsupported expected backend')
    if type(quality) is not bool or type(quant_bits) is not int or quant_bits not in (0, 2, 4):
        raise InvalidDump('expected quality must be boolean and quant_bits must be 0, 2, or 4')
    if type(vocab) is not int or not 0 < vocab <= INT_MAX:
        raise InvalidDump('expected vocab must be a positive signed C int')
    return {'source': 'ds4-bench', 'model': model, 'backend': backend,
            'quality': quality, 'quant_bits': quant_bits, 'ctx': ctx, 'vocab': vocab}


def validate_dump(path, expected, frontier, previous):
    if path.is_symlink() or not path.is_file():
        raise InvalidDump('dump must be an existing regular file, not a symlink')
    raw = path.read_bytes()
    if not raw or not raw.endswith(b'\n'):
        raise InvalidDump('dump is empty or missing its final newline')
    try:
        document = json.loads(raw.decode('utf-8'), object_pairs_hook=unique_object,
                              parse_int=JSONNumber, parse_float=JSONNumber,
                              parse_constant=reject_constant)
    except (ValueError, UnicodeError) as exc:
        raise InvalidDump(f'invalid JSON: {exc}') from exc
    if type(document) is not dict or set(document) != FIELDS:
        raise InvalidDump('dump must have exactly the known ds4-bench metadata and logits fields')
    for key in ('source', 'model', 'backend'):
        if type(document[key]) is not str or document[key] != expected[key]:
            raise InvalidDump(f'{key}: does not match explicit expected metadata')
    if type(document['quality']) is not bool or document['quality'] != expected['quality']:
        raise InvalidDump('quality: does not match explicit expected boolean')
    metadata = {key: document[key] for key in ('source', 'model', 'backend', 'quality')}
    counts = {key: integer(document[key], key) for key in
              ('quant_bits', 'prompt_tokens', 'frontier_tokens', 'prefill_tokens', 'ctx', 'vocab', 'argmax_id')}
    required_counts = {key: expected[key] for key in ('quant_bits', 'ctx', 'vocab')}
    required_counts.update(prompt_tokens=frontier, frontier_tokens=frontier,
                           prefill_tokens=frontier - previous)
    for key, value in required_counts.items():
        if counts[key] != value:
            raise InvalidDump(f'{key}: got {counts[key]}, expected {value}')
    if counts['argmax_id'] >= expected['vocab']:
        raise InvalidDump('argmax_id: outside vocabulary')
    if type(document['logits']) is not list or len(document['logits']) != expected['vocab']:
        raise InvalidDump('logits: vector length must exactly match expected vocabulary')
    words = bytearray()
    values = []
    for index, value in enumerate(document['logits']):
        word, decoded = binary32(value, f'logits[{index}]')
        words.extend(word)
        values.append(decoded)
    argmax_word, argmax_value = binary32(document['argmax_logit'], 'argmax_logit')
    argmax = counts['argmax_id']
    if argmax_word != words[4 * argmax:4 * argmax + 4]:
        raise InvalidDump('argmax_logit: float32 bits do not match logits[argmax_id]')
    # Both current CPU argmax implementations retain the lowest ID on ties.
    if argmax != max(range(len(values)), key=values.__getitem__):
        raise InvalidDump('argmax_id: not the first maximum of the complete logit vector')
    metadata.update(counts)
    metadata['argmax_logit'] = argmax_value
    return {'metadata': metadata, 'file_sha256': hashlib.sha256(raw).hexdigest(),
            'logits_float32_sha256': hashlib.sha256(words).hexdigest()}, bytes(words), values


def compare_directories(baseline, candidate, frontiers, expected):
    try:
        validated = validate_expectations(frontiers, expected['ctx'], expected['model'],
                                           expected['backend'], expected['quality'],
                                           expected['quant_bits'], expected['vocab'])
    except (KeyError, TypeError) as exc:
        raise InvalidDump(f'incomplete expected metadata: {exc}') from exc
    if expected != validated:
        raise InvalidDump('expected metadata does not match the explicit source contract')
    names = {f'frontier_{frontier:06d}.logits.json' for frontier in frontiers}
    errors = []
    for label, directory in (('baseline', baseline), ('candidate', candidate)):
        try:
            present = {path.name for path in directory.iterdir()}
            if present != names:
                errors.append(f'{label} file coverage: missing={sorted(names - present)}, extra={sorted(present - names)}')
        except OSError as exc:
            errors.append(f'{label} directory: {exc}')
    results = []
    previous = 0
    for frontier in frontiers:
        filename = f'frontier_{frontier:06d}.logits.json'
        row = {'frontier': frontier, 'filename': filename, 'errors': [],
               'complete_finite_validation': False, 'strict_float32_identity': False}
        data = []
        for label, directory in (('baseline', baseline), ('candidate', candidate)):
            try:
                info, words, values = validate_dump(directory / filename, expected, frontier, previous)
                row[label] = info
                data.append((words, values))
            except (OSError, InvalidDump, OverflowError, RecursionError) as exc:
                row['errors'].append(f'{label}: {exc}')
        if len(data) == 2:
            old, new = data
            count = 0
            first = None
            max_abs = 0.0
            for index, (x, y) in enumerate(zip(old[1], new[1])):
                offset = 4 * index
                xb, yb = old[0][offset:offset + 4], new[0][offset:offset + 4]
                if xb != yb:
                    count += 1
                    max_abs = max(max_abs, abs(x - y))
                    if first is None:
                        first = {'index': index, 'baseline': x, 'candidate': y,
                                 'baseline_bits': f'{int.from_bytes(xb, "little"):08x}',
                                 'candidate_bits': f'{int.from_bytes(yb, "little"):08x}'}
            row.update(complete_finite_validation=True, strict_float32_identity=count == 0,
                       values=expected['vocab'], changed_count=count, first_change=first, max_abs=max_abs)
        results.append(row)
        previous = frontier
    complete = not errors and all(row['complete_finite_validation'] for row in results)
    identical = complete and all(row['strict_float32_identity'] for row in results)
    return {'format_version': 1, 'status': 'identical' if identical else ('drift' if complete else 'invalid'),
            'strict_float32_identity': identical, 'complete_finite_coverage': complete,
            'scope': 'selected prefill frontiers only; no model quality or decode correctness verdict',
            'expected_frontiers': frontiers, 'expected_metadata': expected,
            'errors': errors, 'frontiers': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--frontiers', type=int, nargs='+', required=True)
    parser.add_argument('--ctx', type=int, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--backend', choices=('rocm', 'cuda', 'metal', 'cpu'), required=True)
    parser.add_argument('--quality', choices=('true', 'false'), required=True)
    parser.add_argument('--quant-bits', type=int, choices=(0, 2, 4), required=True)
    parser.add_argument('--vocab', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True, help='exclusively create a new JSON report')
    args = parser.parse_args(argv)
    try:
        expected = validate_expectations(args.frontiers, args.ctx, args.model, args.backend,
                                         args.quality == 'true', args.quant_bits, args.vocab)
        report = compare_directories(args.baseline, args.candidate, args.frontiers, expected)
        with args.output.open('x', encoding='utf-8') as output:
            output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(f'{report["status"].upper()}: {len(report["frontiers"])} expected frontiers; {args.output}', file=sys.stderr)
        return 0 if report['strict_float32_identity'] else 1
    except (OSError, ValueError, OverflowError, RecursionError) as exc:
        print(f'INVALID: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
