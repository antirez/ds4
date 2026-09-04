#!/usr/bin/env python3
"""Package a ds4 raw steering vector as a GLP file (GGUF Layer Projection).

The raw format build_direction.py writes is a headerless blob of
n_layers * n_embd float32: it carries no operation, no hook point, no layer map
and no base-model pin. That is fine on the machine that produced it and unsafe
to hand to anyone else, because every one of those is silently wrong when it is
wrong:

  operation   llama.cpp has shipped control vectors since 2024 with the same
              tensor convention -- direction.<N>, fp32, 1-D -- but ADDS them:
              h <- h + v. ds4 PROJECTS: y -= scale * v * dot(v, y). A file
              loads into either runtime without an error and is wrong in one of
              them. An additive apply of a projective direction pushes every
              token along the direction instead of removing it.

  hook point  ds4 steers the block writers (ffn_out = moe + shared, or the
              attention output) before the residual fold. llama.cpp's
              build_cvec() and the weightless vLLM overlay steer the post-layer
              residual. A writer holds only what that block computed this
              layer; the folded residual holds the accumulated sum, so
              projecting it also removes what upstream layers contributed.
              Steering a writer prevents refusal being added, steering the
              residual deletes refusal already there -- different
              interventions, and the dose does not transfer.

  layer map   direction.N applies at layer N. A one-layer shift does not fail,
              it degrades -- adjacent layers' refusal directions have cosine
              similarity 0.555-0.979 -- so it survives a smoke test.

  base model  a direction is tied to the exact checkpoint it was derived from.
              Applying one to a different revision is undefined.

GLP is standard GGUF v3 with llama.cpp's tensor convention unchanged, plus a
"glp.*" block that states all four. ds4 reads it back with ds4_glp.c and
refuses rather than misapplies. Spec:
https://github.com/msuiche/weightless/blob/main/spec/GLP.md

No third-party dependencies: this writes GGUF with struct, so it runs wherever
ds4 builds.

Usage:
  f32_to_glp.py IN.f32 OUT.gguf --n-embd 4096 \\
      --base-model DeepSeek-V4-Flash-0731 --base-org deepseek-ai \\
      [--meta IN.json] [--hook ffn_out_pre_residual] [--alpha 1.0] \\
      [--layers 10-38] [--base-revision COMMIT] [--repo-url URL] \\
      [--method ...] [--contrast ...]

--meta reads build_direction.py's sidecar JSON, which supplies the shape and
maps its "component" field to the hook point, so the two cannot disagree.
"""

import argparse
import datetime
import hashlib
import json
import os
import struct
import sys

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8

GGML_TYPE_F32 = 0

GLP_SPEC_VERSION = 1

# build_direction.py names ds4's steering sites "ffn_out" and "attn_out".
# The GLP hook names are explicit that these are the block writers, before the
# residual / hyper-connection fold -- which is what makes them different from
# residual_stream_post_layer rather than a synonym for it.
COMPONENT_TO_HOOK = {
    "ffn_out": "ffn_out_pre_residual",
    "attn_out": "attn_out_pre_residual",
}
VALID_HOOKS = set(COMPONENT_TO_HOOK.values())


def die(msg):
    sys.stderr.write("f32_to_glp: %s\n" % msg)
    sys.exit(1)


def parse_layers(spec, n_layers):
    """"10-38", "10,11,12", or "10-38,41" -> sorted list of layer ids."""
    out = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                die("--layers range %s is inverted" % part)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    for layer in sorted(out):
        if layer < 0 or layer >= n_layers:
            die("--layers names layer %d, outside 0..%d" % (layer, n_layers - 1))
    return sorted(out)


# ---------------------------------------------------------------------------
# GGUF writer
# ---------------------------------------------------------------------------

def w_string(buf, s):
    b = s.encode("utf-8")
    buf.append(struct.pack("<Q", len(b)))
    buf.append(b)


def kv(buf, key, vtype, payload):
    w_string(buf, key)
    buf.append(struct.pack("<I", vtype))
    buf.append(payload)


def kv_str(buf, key, value):
    w_string(buf, key)
    buf.append(struct.pack("<I", GGUF_TYPE_STRING))
    b = value.encode("utf-8")
    buf.append(struct.pack("<Q", len(b)))
    buf.append(b)


def write_gguf(path, meta_strs, meta_u32, meta_f32, meta_bool, tensors):
    """tensors: list of (name, list-of-float). All 1-D F32."""
    kv_count = len(meta_strs) + len(meta_u32) + len(meta_f32) + len(meta_bool)

    head = [GGUF_MAGIC, struct.pack("<I", GGUF_VERSION),
            struct.pack("<Q", len(tensors)), struct.pack("<Q", kv_count)]
    for k, v in meta_strs:
        kv_str(head, k, v)
    for k, v in meta_u32:
        kv(head, k, GGUF_TYPE_UINT32, struct.pack("<I", v))
    for k, v in meta_f32:
        kv(head, k, GGUF_TYPE_FLOAT32, struct.pack("<f", v))
    for k, v in meta_bool:
        kv(head, k, GGUF_TYPE_BOOL, struct.pack("<B", 1 if v else 0))

    # Tensor directory. Offsets are relative to the aligned start of the data
    # section, and each tensor is padded up to the alignment.
    offset = 0
    for name, values in tensors:
        w_string(head, name)
        head.append(struct.pack("<I", 1))              # 1-D
        head.append(struct.pack("<Q", len(values)))    # n_embd
        head.append(struct.pack("<I", GGML_TYPE_F32))
        head.append(struct.pack("<Q", offset))
        nbytes = len(values) * 4
        offset += (nbytes + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT

    blob = b"".join(head)
    pad = (GGUF_ALIGNMENT - (len(blob) % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
    body = [blob, b"\0" * pad]
    for _, values in tensors:
        raw = struct.pack("<%df" % len(values), *values)
        body.append(raw)
        body.append(b"\0" * ((GGUF_ALIGNMENT - (len(raw) % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT))

    with open(path, "wb") as fp:
        fp.write(b"".join(body))


# ---------------------------------------------------------------------------
# GGUF reader, for the round-trip assertion only
# ---------------------------------------------------------------------------

def read_back(path):
    """Return (kv_dict_of_strings, {layer: [floats]}). Deliberately minimal."""
    with open(path, "rb") as fp:
        data = fp.read()
    pos = 0

    def u32():
        nonlocal pos
        v = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        return v

    def u64():
        nonlocal pos
        v = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        return v

    def s():
        nonlocal pos
        n = u64()
        v = data[pos:pos + n].decode("utf-8")
        pos += n
        return v

    assert data[0:4] == GGUF_MAGIC, "not GGUF"
    pos = 4
    assert u32() == GGUF_VERSION, "not GGUF v3"
    n_tensors = u64()
    n_kv = u64()

    strings, alignment = {}, GGUF_ALIGNMENT
    scalar = {GGUF_TYPE_UINT32: 4, GGUF_TYPE_FLOAT32: 4, GGUF_TYPE_BOOL: 1}
    for _ in range(n_kv):
        key = s()
        vtype = u32()
        if vtype == GGUF_TYPE_STRING:
            strings[key] = s()
        elif vtype in scalar:
            raw = data[pos:pos + scalar[vtype]]
            pos += scalar[vtype]
            if key == "general.alignment":
                alignment = struct.unpack("<I", raw)[0]
            strings[key] = raw
        else:
            raise AssertionError("unexpected kv type %d for %s" % (vtype, key))

    infos = []
    for _ in range(n_tensors):
        name = s()
        ndim = u32()
        assert ndim == 1, "%s is %d-D, GLP directions are 1-D" % (name, ndim)
        n = u64()
        ttype = u32()
        assert ttype == GGML_TYPE_F32, "%s is ggml type %d, not F32" % (name, ttype)
        infos.append((name, n, u64()))

    data_start = (pos + alignment - 1) // alignment * alignment
    out = {}
    for name, n, rel in infos:
        assert name.startswith("direction."), "unexpected tensor %s" % name
        start = data_start + rel
        out[int(name.split(".", 1)[1])] = list(
            struct.unpack_from("<%df" % n, data, start))
    return strings, out


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="raw .f32 blob (n_layers * n_embd float32)")
    ap.add_argument("out", help="output .gguf")
    ap.add_argument("--meta", help="build_direction.py sidecar JSON")
    ap.add_argument("--n-embd", type=int, help="hidden width (from --meta if given)")
    ap.add_argument("--hook", help="glp.hook_point; default from --meta component")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="glp.alpha_default: the scale this vector was "
                         "calibrated at. ds4 uses it as the default "
                         "--dir-steering-ffn when the hook matches.")
    ap.add_argument("--layers", help="layer ids to emit; default: every "
                                     "non-zero row")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--base-org", default="")
    ap.add_argument("--base-revision", default="",
                    help="the base checkpoint's commit. A direction is tied to "
                         "the revision it was derived from, so pin it.")
    ap.add_argument("--repo-url", default="")
    ap.add_argument("--method", default="paired_difference_of_means")
    ap.add_argument("--contrast", default="")
    args = ap.parse_args()

    n_embd, component = args.n_embd, None
    n_layers_meta = None
    if args.meta:
        with open(args.meta, encoding="utf-8") as fp:
            meta = json.load(fp)
        shape = meta.get("shape")
        if shape and len(shape) == 2:
            n_layers_meta, meta_embd = int(shape[0]), int(shape[1])
            if n_embd and n_embd != meta_embd:
                die("--n-embd %d disagrees with %s shape %r"
                    % (n_embd, args.meta, shape))
            n_embd = meta_embd
        component = meta.get("component")
        if not args.contrast:
            good, bad = meta.get("good_file"), meta.get("bad_file")
            if good and bad:
                args.contrast = "%s_vs_%s" % (os.path.basename(good),
                                              os.path.basename(bad))
    if not n_embd:
        die("--n-embd is required unless --meta carries the shape")

    hook = args.hook or COMPONENT_TO_HOOK.get(component or "")
    if not hook:
        die("no hook point: pass --hook (%s), or --meta with a component field. "
            "A vector that does not say where it was calibrated cannot be "
            "applied safely anywhere else -- the dose does not transfer between "
            "sites, and applying it at the wrong one degrades silently."
            % ", ".join(sorted(VALID_HOOKS)))
    if hook not in VALID_HOOKS:
        die("--hook %r is not a site ds4 applies at (%s). "
            "residual_stream_post_layer is llama.cpp's and the vLLM overlay's "
            "site; ds4 steers the block writers."
            % (hook, ", ".join(sorted(VALID_HOOKS))))

    raw = open(args.src, "rb").read()
    if len(raw) % (n_embd * 4):
        die("%s is %d bytes, not a multiple of n_embd*4 = %d"
            % (args.src, len(raw), n_embd * 4))
    n_layers = len(raw) // (n_embd * 4)
    if n_layers_meta and n_layers != n_layers_meta:
        die("%s holds %d layers, %s declares %d"
            % (args.src, n_layers, args.meta, n_layers_meta))
    rows = [list(struct.unpack_from("<%df" % n_embd, raw, i * n_embd * 4))
            for i in range(n_layers)]

    if args.layers:
        layers = parse_layers(args.layers, n_layers)
    else:
        layers = [i for i, r in enumerate(rows) if any(v != 0.0 for v in r)]
    if not layers:
        die("%s has no non-zero rows" % args.src)

    # direction.0 cannot be expressed: direction.N applies at layer N with no
    # offset, so the container has no slot for layer 0. Say so rather than
    # emitting direction.0 and letting a reader reject the file later.
    if layers[0] == 0:
        die("layer 0 cannot be expressed in this container (direction.N "
            "applies at layer N, and direction.0 is invalid). Drop layer 0 "
            "with --layers, or re-derive without it.")

    # Unit-normalise. Projection is quadratic in the norm, so an off-unit
    # direction folds a hidden strength into the data that the caller's scale
    # then multiplies again -- and alpha is meant to be the only strength knob.
    tensors, renormed = [], 0
    for layer in layers:
        row = rows[layer]
        norm = sum(v * v for v in row) ** 0.5
        if norm <= 1e-12:
            die("layer %d is all zero; it would steer nothing. Omit it from "
                "--layers rather than shipping a dead slot." % layer)
        if abs(norm - 1.0) > 1e-6:
            row = [v / norm for v in row]
            renormed += 1
        tensors.append(("direction.%d" % layer, row))

    # The hash covers tensor bytes only, not metadata: glp.created makes the
    # file non-reproducible, so this is what lets two people confirm they hold
    # the same direction regardless of when it was packaged.
    h = hashlib.sha256()
    for _, row in tensors:
        h.update(struct.pack("<%df" % len(row), *row))

    meta_strs = [
        ("general.architecture", "controlvector"),
        ("general.type", "controlvector"),
        ("general.name", os.path.basename(args.out)),
        # The one field a reader may not ignore.
        ("glp.mode", "project"),
        ("glp.hook_point", hook),
        ("glp.method", args.method),
        # Redundant with the tensor names on purpose: readable in a gguf_dump,
        # and a cross-check that catches an exporter writing the wrong ids.
        ("glp.layer_ids_zero_based", ",".join(str(x) for x in layers)),
        ("glp.content_sha256", h.hexdigest()),
        ("glp.created", datetime.date.today().isoformat()),
        ("general.base_model.0.name", args.base_model),
    ]
    if args.contrast:
        meta_strs.append(("glp.contrast", args.contrast))
    if args.base_org:
        meta_strs.append(("general.base_model.0.organization", args.base_org))
    if args.base_revision:
        meta_strs.append(("general.base_model.0.version", args.base_revision))
    if args.repo_url:
        meta_strs.append(("general.base_model.0.repo_url", args.repo_url))

    meta_u32 = [
        ("glp.spec_version", GLP_SPEC_VERSION),
        ("glp.rank", 1),
        ("general.base_model.count", 1),
    ]
    meta_f32 = [("glp.alpha_default", args.alpha)]
    meta_bool = [("glp.orthonormal", True)]

    write_gguf(args.out, meta_strs, meta_u32, meta_f32, meta_bool, tensors)

    # Round-trip, because the failures this container prevents are the ones
    # that do not raise: assert the layer ids landed verbatim and the values
    # survived exactly.
    strings, back = read_back(args.out)
    assert strings["glp.mode"] == "project", "mode did not survive"
    assert strings["glp.hook_point"] == hook, "hook_point did not survive"
    assert sorted(back) == layers, (
        "layer ids changed in the round trip: wrote %r, read %r"
        % (layers, sorted(back)))
    for name, row in tensors:
        got = back[int(name.split(".", 1)[1])]
        assert len(got) == len(row), "%s changed length" % name
        for a, b in zip(row, got):
            assert struct.pack("<f", a) == struct.pack("<f", b), \
                "%s changed value" % name

    print("wrote %s" % args.out)
    print("  mode           project")
    print("  hook_point     %s" % hook)
    print("  alpha_default  %g" % args.alpha)
    print("  coverage       %d directions, n_embd=%d, layers %d..%d"
          % (len(layers), n_embd, layers[0], layers[-1]))
    print("  base model     %s%s" % (args.base_model,
                                     " @ " + args.base_revision
                                     if args.base_revision else ""))
    print("  content sha256 %s" % h.hexdigest())
    print("  size           %d bytes" % os.path.getsize(args.out))
    if renormed:
        print("  note           %d of %d rows were rescaled to unit length"
              % (renormed, len(layers)))
    if not args.base_revision:
        print("  warning        no --base-revision. A direction is tied to the "
              "exact\n                 checkpoint it was derived from; without "
              "a commit pin a\n                 consumer cannot tell whether "
              "it holds the right one.")


if __name__ == "__main__":
    main()
