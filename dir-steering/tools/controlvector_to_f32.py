#!/usr/bin/env python3
"""Convert a `controlvector` GGUF of projection directions to a ds4 steering file.

Some ablation tools ship their per-layer directions as a GGUF with
`general.architecture = controlvector` and one F32 tensor `direction.<layer>`
of length n_embd per steered layer. ds4 reads a flat little-endian F32 matrix
of n_layer x n_embd instead, one row per layer, zero rows for layers that are
not steered.

Only convert files whose vectors are directions to project out (ds4 computes
y -= scale * d * dot(d, y)). Additive control vectors, the llama.cpp
`--control-vector` kind, mean something else and are not usable this way.

Usage:
    python3 controlvector_to_f32.py IN.gguf OUT.f32 --layers 40 [--offset 0] [--normalize]

`direction.N` lands in ds4 layer N + offset. The default keeps the file's own
numbering, where `direction.N` is layer N counted from 0.

GLP files (`glp.*` keys, weightless/captain-vector) also name the site the
vector was calibrated at and its strength. The converter checks what ds4 can
apply (projection mode, one direction per layer, one alpha) and prints the
matching flags; the scale does not carry over to a different site.
"""
import argparse
import math
import struct
import sys
from array import array

SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Reader:
    def __init__(self, data):
        self.data, self.off = data, 0

    def take(self, fmt):
        value = struct.unpack_from("<" + fmt, self.data, self.off)[0]
        self.off += struct.calcsize("<" + fmt)
        return value

    def string(self):
        n = self.take("Q")
        value = self.data[self.off:self.off + n].decode("utf-8")
        self.off += n
        return value

    def value(self, kind):
        if kind == 8:
            return self.string()
        if kind == 9:
            item, n = self.take("I"), self.take("Q")
            return [self.value(item) for _ in range(n)]
        return self.take(SCALAR[kind])


def read_controlvector(path):
    data = open(path, "rb").read()
    if data[:4] != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF file")
    r = Reader(data)
    r.off = 4
    version, n_tensors, n_kv = r.take("I"), r.take("Q"), r.take("Q")
    if version < 2:
        raise SystemExit(f"{path}: GGUF version {version} is not supported")
    meta = {}
    for _ in range(n_kv):
        key = r.string()
        meta[key] = r.value(r.take("I"))
    if meta.get("general.architecture") != "controlvector":
        raise SystemExit(f"{path}: general.architecture is {meta.get('general.architecture')!r}, "
                         "expected 'controlvector'")
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.take("Q") for _ in range(r.take("I"))]
        kind, offset = r.take("I"), r.take("Q")
        infos.append((name, dims, kind, offset))
    align = meta.get("general.alignment", 32)
    base = (r.off + align - 1) // align * align
    vectors = {}
    for name, dims, kind, offset in infos:
        if not name.startswith("direction."):
            continue
        if kind != 0 or len(dims) != 1:
            raise SystemExit(f"{path}: {name} must be a 1-D F32 tensor")
        vec = array("f")
        vec.frombytes(data[base + offset:base + offset + 4 * dims[0]])
        if sys.byteorder != "little":
            vec.byteswap()
        index = name.split(".", 1)[1]
        if not index.isdigit():
            raise SystemExit(f"{path}: {name} is a second direction for a layer; ds4 holds one")
        vectors[int(index)] = vec
    if not vectors:
        raise SystemExit(f"{path}: no direction.<layer> tensors")
    return meta, vectors


# GLP hook point -> ds4 flags, given alpha. On V4.1 the residual site is the HC
# fold after each layer, projected through the mean of the copies.
GLP_FLAGS = {
    "ffn_out_pre_residual": "--dir-steering-ffn {a}",
    "attn_out_pre_residual": "--dir-steering-attn {a} --dir-steering-ffn 0",
    "residual_stream_post_layer": "--dir-steering-resid {a} --dir-steering-ffn 0",
    # spec_version 3: the fold plus its companion attention writer, same alpha
    "hc_fold_pre_engram": "--dir-steering-resid {a} --dir-steering-attn {a} --dir-steering-ffn 0",
}


def check_glp(meta, keys):
    """Refuse what ds4 cannot apply; return the suggested flags."""
    mode = meta.get("glp.mode", "add")
    if mode != "project":
        raise SystemExit(f"glp.mode is {mode!r}: ds4 steering only projects directions out")
    version = meta.get("glp.spec_version", 1)
    if version not in (1, 2, 3):
        raise SystemExit(f"glp.spec_version {version} is not known to this converter")
    if meta.get("glp.rank", 1) != 1:
        raise SystemExit(f"glp.rank is {meta.get('glp.rank')}: ds4 holds one direction per layer")
    for key in ("glp.dir_scales", "glp.layer_scales"):
        if key in meta:
            raise SystemExit(f"{key} is set: ds4 applies one scale to every layer")
    hook = meta.get("glp.hook_point", "residual_stream_post_layer")
    if hook == "hc_fold_pre_engram" and version < 3:
        raise SystemExit("hc_fold_pre_engram needs glp.spec_version 3")
    if hook not in GLP_FLAGS:
        raise SystemExit(f"glp.hook_point {hook!r} has no ds4 site")
    ids = meta.get("glp.layer_ids_zero_based")
    if ids is not None:
        listed = sorted(int(x) for x in str(ids).split(",") if x.strip())
        if listed != sorted(keys):
            raise SystemExit(f"glp.layer_ids_zero_based {listed} does not match the "
                             f"direction tensors {sorted(keys)}")
    alpha = meta.get("glp.alpha_default", 1.0)
    return hook, GLP_FLAGS[hook].format(a=f"{alpha:g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--layers", type=int, required=True,
                    help="model layers in the ds4 file (DeepSeek V4.1 Flash: 40)")
    ap.add_argument("--offset", type=int, default=0, help="ds4 layer = N + offset for direction.N")
    ap.add_argument("--normalize", action="store_true", help="rescale every non-zero row to unit length")
    args = ap.parse_args()

    meta, vectors = read_controlvector(args.input)
    glp = check_glp(meta, vectors.keys()) if any(k.startswith("glp.") for k in meta) else None
    widths = {len(v) for v in vectors.values()}
    if len(widths) != 1:
        raise SystemExit(f"directions have different widths: {sorted(widths)}")
    width = widths.pop()
    rows = [None] * args.layers
    for key, vec in sorted(vectors.items()):
        layer = key + args.offset
        if not 0 <= layer < args.layers:
            raise SystemExit(f"direction.{key} maps to layer {layer}, outside 0..{args.layers - 1}")
        norm = math.sqrt(sum(x * x for x in vec))
        if args.normalize and norm > 0:
            vec = array("f", (x / norm for x in vec))
            norm = 1.0
        if norm > 0 and abs(norm - 1.0) > 1e-3:
            print(f"warning: direction.{key} has norm {norm:.4f}; ds4 expects unit directions "
                  "(--normalize)", file=sys.stderr)
        rows[layer] = vec
    out = array("f")
    for row in rows:
        out.extend(row if row is not None else array("f", bytes(4 * width)))
    if sys.byteorder != "little":
        out.byteswap()
    with open(args.output, "wb") as f:
        out.tofile(f)
    steered = [i for i, row in enumerate(rows) if row is not None]
    print(f"wrote {args.output}: {args.layers} x {width} F32, layers {steered[0]}..{steered[-1]} "
          f"({len(steered)} steered), model hint {meta.get('controlvector.model_hint', '?')!r}")
    if glp:
        if args.offset:
            print(f"warning: --offset {args.offset} moves a GLP file off its own layer ids",
                  file=sys.stderr)
        print(f"GLP hook {glp[0]}: --dir-steering-file {args.output} {glp[1]}")


if __name__ == "__main__":
    main()
