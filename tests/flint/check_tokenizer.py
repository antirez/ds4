#!/usr/bin/env python3
"""Compare ds4's Flint tokenizer with HF `tokenizers` on the checkpoint's
tokenizer.json, which is what rec uses.

For every case the text goes through `./ds4 -m GGUF --dump-tokens --raw -p`
and through tokenizers' `encode(text, add_special_tokens=False)`; the id
lists must be identical.  Cases cover the three pre-tokenizer splits (digit
groups, CJK runs, the GPT-4o word pattern with its case rules and
contractions), whitespace and newline handling, punctuation runs, control
characters, and special tokens embedded in text.  --fuzz N adds N random
strings drawn from a mixed alphabet.

    tests/flint/check_tokenizer.py [--model GGUF] [--src CHECKPOINT_DIR] [--fuzz 200]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_MODELS = Path.home() / ".local/share/rec/models"

CASES = [
    "Hello world",
    "hello world",
    "HelloWorld fooBar FOOBar FOO Foo fOO",
    "ABC abc AbC aBC",
    "I'm sure you're right, they'll've done it; we'd've, it's, don't, can't, WON'T, I'M, YOU'RE",
    "don'tcha 'tis 'twas rock'n'roll",
    "The quick brown fox jumps over the lazy dog.",
    "1 12 123 1234 12345 123456 1234567",
    "a1b22c333d4444",
    "pi is 3.14159 and e is 2.71828; 1,000,000 items at $19.99 each = 19990000",
    "version 2.0.1-beta+build.4567",
    "  leading spaces and trailing spaces   ",
    "tabs\tand\ttabs\t\tdouble",
    "line one\nline two\n\nline four\r\nwindows\r\n\r\n",
    "trailing newline\n",
    "\n\nleading newlines",
    "spaces before newline   \n   and after",
    "mixed \t \n \t whitespace",
    "x = (a + b) * c / d - e ** 2 % 3 << 4 >> 5 & 6 | 7 ^ 8 ~9 !10",
    "def f(x):\n    return {'a': [1, 2, 3], \"b\": (4, 5)}\n",
    "if (x->y != NULL && *p == '\\0') { printf(\"%d\\n\", i++); }",
    ">>> import os; os.path.join('a', 'b')  # comment",
    "URL: https://example.com/path?q=1&r=2#frag and email: a.b-c_d@example.co.uk",
    "C'est l'été à Paris: déjà vu, naïve, façade, Ærøskøbing, Straße",
    "ÉCOLE École école ÉcOLE",
    "Ελληνικά ΕΛΛΗΝΙΚΆ ελληνικά Ωμέγα",
    "Русский текст: Привет, мир! ПРИВЕТ",
    "日本語のテキストです。漢字とひらがなとカタカナ。",
    "中文测试：你好，世界！１２３ 全角数字 ４５６",
    "한국어 텍스트입니다",
    "العربية والعبرية עברית",
    "Deutsch: Ärger, Öl, Übung, groß; dansk: æøå ÆØÅ",
    "emoji 😀 🎉 👍🏽 and symbols ∑ ∫ ≠ → ← ♥ © ® ™",
    "combining: e\u0301 a\u0308 n\u0303 \u0301alone",
    "zero\u200bwidth\u200cjoiners\u200d and nbsp\u00a0here",
    "control \x01\x02 chars \x1f and \x7f del",
    "\x1c\x1d\x1e\x1f information separators",
    "ideographic space\u3000and em space\u2003and thin space\u2009",
    "U+0085 next line\u0085here and \u2028 line sep \u2029 para sep",
    "ALLCAPS WORDS AND MiXeD CaSe and camelCaseWord and PascalCaseWord",
    "x2 y3 z44 abc123def 123abc abc123",
    "<|user|>hello<|assistant|>",
    "<|system|>You are helpful.<|user|>Hi<|assistant|><think>thinking</think>answer<|end_of_text|>",
    "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
    "<|observation|>result<|assistant|>",
    "<|developer|>dev message",
    "text with <|not_a_token|> and <think>>",
    "",
    " ",
    "\n",
    "'",
    "'s",
    "a'",
    "'S 'T 'RE 'VE 'M 'LL 'D",
    "  's  'll  ",
    "ǅ ǈ ǋ titlecase letters ǅungla",
    "ʰʲ modifier letters ªº ordinal",
    "ⅷ Ⅷ ① ⑴ ½ ¼ number-like",
    "٠١٢٣٤٥٦٧٨٩ arabic-indic digits ०१२ devanagari",
    "math 𝐀𝐁𝐂 𝑎𝑏𝑐 and 🅰🅱",
    "repeated!!!!!! punctuation?????? ...... ------ ======",
    "a  b   c    d     e",
    "a\n b\n  c\n   d",
    "end with space ",
    "end with two spaces  ",
    "end with tab\t",
    "\r\n",
    "\r",
    "a\rb",
]


def find_default_src() -> Path | None:
    if not DEFAULT_MODELS.is_dir():
        return None
    cands = [d for d in DEFAULT_MODELS.iterdir() if (d / "tokenizer.json").is_file() and (d / "flint-int4.json").is_file()]
    return max(cands, key=lambda d: d.stat().st_mtime) if cands else None


def ds4_tokens(ds4: Path, model: Path, text: str) -> list[int]:
    r = subprocess.run([str(ds4), "-m", str(model), "--dump-tokens", "--raw", "-p", text],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("ds4 failed on %r:\n%s" % (text, r.stderr))
    first = r.stdout.split("\n", 1)[0]
    return json.loads(first)


def fuzz_strings(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    alphabet = (
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        + list(" " * 12 + "\n\n\t\r")
        + list(".,;:!?'\"()[]{}<>-_=+*/\\|@#$%^&~`")
        + list("éèêàçñüößÉÀÇÑÜ")
        + list("日本語漢字ひらがなカタカナ中文你好")
        + list("αβγΩΔ абвгдЖ")
        + ["😀", "🎉", "\u0301", "\u00a0", "\u200b", "\x01", "'s", "'ll", "'RE", "<|user|>", "<think>", "123", "4567"]
    )
    out = []
    for _ in range(n):
        k = rng.randint(1, 40)
        out.append("".join(rng.choice(alphabet) for _ in range(k)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path(os.environ.get("FLINT_MODEL", ROOT.parent / "ds4/gguf/flint/flint-int4.gguf")))
    ap.add_argument("--src", type=Path, default=None, help="checkpoint dir with tokenizer.json")
    ap.add_argument("--ds4", type=Path, default=ROOT / "ds4")
    ap.add_argument("--fuzz", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    src = args.src or find_default_src()
    if src is None:
        sys.exit("no checkpoint with tokenizer.json found; pass --src")
    tok = Tokenizer.from_file(str(src / "tokenizer.json"))

    cases = CASES + fuzz_strings(args.fuzz, args.seed)
    bad = 0
    for text in cases:
        want = tok.encode(text, add_special_tokens=False).ids
        got = ds4_tokens(args.ds4, args.model, text)
        if got != want:
            bad += 1
            print("MISMATCH %r" % text)
            print("  ds4 : %s" % got)
            print("  hf  : %s" % want)
            print("  ds4 : %s" % [tok.id_to_token(i) for i in got])
            print("  hf  : %s" % [tok.id_to_token(i) for i in want])
        elif args.verbose:
            print("ok %r -> %d tokens" % (text, len(got)))
    print("%d cases, %d mismatches" % (len(cases), bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
