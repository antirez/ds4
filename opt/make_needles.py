#!/usr/bin/env python3
"""Build long-context needle documents for Kolibri-1 retrieval gates.

Two documents from promessi_sposi.txt:
  needle-30k.txt: ~28K tokens with a code at ~50% and one at ~95%, question at end
  needle-2k.txt:  ~1.8K tokens with one code at ~90%
Success = the greedy answer quotes both codes exactly.
"""
import sys, os

SRC = "speed-bench/promessi_sposi.txt"
OUT = "opt/runs"
NEEDLE_1 = " Merk dir: der geheime Zugangscode fuer das Mailaender Archiv lautet ZQ-7741-KATZE. "
NEEDLE_2 = " Notiere ausserdem: der zweite Zugangscode fuer das Berliner Archiv lautet LT-9082-VOGEL. "
QUESTION = ("\n\nFrage: Nenne die beiden geheimen Zugangscode, die im obigen Text erwaehnt werden. "
            "Antworte nur mit den beiden Codes in der Reihenfolge ihres Auftretens, getrennt durch ein Komma.")

def build(path, target_chars, needle_at):
    text = open(SRC, encoding="utf-8").read()
    text = " ".join(text.split())
    if len(text) < target_chars:
        sys.exit(f"source too short: {len(text)} < {target_chars}")
    cut = text[:target_chars]
    pos = int(target_chars * needle_at)
    doc = cut[:pos] + NEEDLE_1 + cut[pos:]
    if needle_at < 0.9:
        pos2 = int(len(doc) * 0.95)
        doc = doc[:pos2] + NEEDLE_2 + doc[pos2:]
    return doc + QUESTION

def main():
    os.makedirs(OUT, exist_ok=True)
    # ~28K tokens at ~4.5 chars/token for Italian
    open(f"{OUT}/needle-30k.txt", "w", encoding="utf-8").write(build(None, 126000, 0.5))
    open(f"{OUT}/needle-2k.txt", "w", encoding="utf-8").write(build(None, 8200, 0.9))
    for f in ("needle-30k.txt", "needle-2k.txt"):
        n = os.path.getsize(f"{OUT}/{f}")
        print(f"{f}: {n} chars (~{n//4} tokens)")

if __name__ == "__main__":
    main()
