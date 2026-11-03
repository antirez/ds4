#!/usr/bin/env python3
"""Map each DS4_TENSOR_IQ2_XXS line in ds4.c to its enclosing function.

Classifying the 58 sites by hand is error-prone and the point is to decide
which ones the deepseek41 SSD-streaming path actually reaches, so derive the
enclosing function mechanically: scan for lines that start at column 0 and
look like a definition, and remember the most recent one above each hit.
"""
import re
import sys

DEF = re.compile(r'^(?:static\s+|DS4_[A-Z_]+\s+)*[A-Za-z_][\w \t*]*\b(\w+)\s*\(')


def main(path):
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    cur = '(file scope)'
    out = []
    for i, ln in enumerate(lines, 1):
        if ln and ln[0] not in ' \t#/*}' and '(' in ln and not ln.rstrip().endswith(';'):
            m = DEF.match(ln)
            if m:
                cur = m.group(1)
        if 'DS4_TENSOR_IQ2_XXS' in ln:
            out.append((i, cur, ln.strip()))

    by_fn = {}
    for i, fn, _ in out:
        by_fn.setdefault(fn, []).append(i)

    print(f"{len(out)} sites in {len(by_fn)} functions\n")
    for fn, ls in sorted(by_fn.items(), key=lambda kv: kv[1][0]):
        print(f"  {fn}")
        print(f"      lines {', '.join(str(x) for x in ls)}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
