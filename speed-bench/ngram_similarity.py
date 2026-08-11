#!/usr/bin/env python3
"""Workload self-similarity probe for speculative-decoding triage.

For each input file, runs an online n-gram predictor (orders 1..3) over the
stream and reports its hit rate: at each position the predictor proposes the
most frequent continuation of the current context seen so far, then learns the
actual one. High hit rates flag self-similar streams (code, structured output)
where draft acceptance is likely to be high; low rates flag prose-like streams.

This is a workload characterization aid, NOT a bound on learned-draft
acceptance (a trained draft can beat stream self-similarity). Units are words
(whitespace-split) by default, or bytes with --bytes; real tokenizer streams
can be fed later via files with one token id per line and --ids.

Standard library only, like the other speed-bench tools.
"""
import argparse
import sys
from collections import defaultdict, Counter


def stream_units(path, mode):
    data = open(path, 'rb').read()
    if mode == 'bytes':
        return list(data)
    if mode == 'ids':
        return [int(x) for x in data.split()]
    return data.decode('utf-8', 'replace').split()


def hit_rate(units, order, warmup):
    model = defaultdict(Counter)
    hits = 0
    scored = 0
    ctx = tuple()
    for i, u in enumerate(units):
        if len(ctx) == order:
            c = model.get(ctx)
            if i >= warmup and c:
                pred = max(c.items(), key=lambda kv: kv[1])[0]
                hits += pred == u
                scored += 1
            model[ctx][u] += 1
        ctx = (ctx + (u,))[-order:] if order else tuple()
    return hits / scored if scored else 0.0, scored


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('files', nargs='+')
    ap.add_argument('--bytes', action='store_true', help='byte units instead of words')
    ap.add_argument('--ids', action='store_true', help='inputs are one token id per line')
    ap.add_argument('--orders', default='1,2,3')
    ap.add_argument('--warmup', type=int, default=256,
                    help='positions before scoring starts (default 256)')
    args = ap.parse_args()
    mode = 'bytes' if args.bytes else 'ids' if args.ids else 'words'
    orders = [int(x) for x in args.orders.split(',')]
    print(f'unit={mode} warmup={args.warmup}')
    header = f'{"file":40} {"units":>9} ' + ' '.join(f'{"o"+str(o):>7}' for o in orders)
    print(header)
    for path in args.files:
        units = stream_units(path, mode)
        cells = []
        for o in orders:
            r, n = hit_rate(units, o, args.warmup)
            cells.append(f'{r*100:6.1f}%')
        name = path if len(path) <= 40 else '...' + path[-37:]
        print(f'{name:40} {len(units):9} ' + ' '.join(cells))


if __name__ == '__main__':
    main()
