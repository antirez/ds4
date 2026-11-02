#!/usr/bin/env python3
"""Write a byte-level BPE vocab in the text format tests/test_tokenizer reads.

    tokenizer_vocab.py tokenizer.json OUT
    tokenizer_vocab.py encoder.json vocab.bpe OUT      (GPT-2 layout)

The format is the token count, one token per line in id order, then one
"left right" merge per line in rank order.  Added tokens from tokenizer.json
are included at their ids.
"""
import json
import sys


def from_tokenizer_json(path):
    data = json.load(open(path, encoding="utf-8"))
    model = data["model"]
    vocab = dict(model["vocab"])
    for added in data.get("added_tokens", []):
        vocab[added["content"]] = added["id"]
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    return vocab, merges


def from_gpt2(encoder, bpe):
    vocab = json.load(open(encoder, encoding="utf-8"))
    lines = open(bpe, encoding="utf-8").read().split("\n")
    merges = [line for line in lines[1:] if line and not line.startswith("#version")]
    return vocab, merges


def main():
    if len(sys.argv) == 3:
        vocab, merges = from_tokenizer_json(sys.argv[1])
    elif len(sys.argv) == 4:
        vocab, merges = from_gpt2(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)
    n = max(vocab.values()) + 1
    tokens = [""] * n
    for text, token_id in vocab.items():
        if "\n" in text:
            sys.exit(f"token {token_id} contains a newline")
        tokens[token_id] = text
    with open(sys.argv[-1], "w", encoding="utf-8", newline="\n") as out:
        out.write(f"{n}\n")
        out.writelines(t + "\n" for t in tokens)
        out.writelines(m + "\n" for m in merges)


if __name__ == "__main__":
    main()
