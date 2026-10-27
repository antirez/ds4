#!/usr/bin/env python3
"""Independent NumPy checks of real-weight V4.1 DSpark Metal/CUDA dumps.

Run test_deepseek41_dspark first. Source tensors are decoded from safetensors,
not from the GGUF being checked. Only individual projections/experts are loaded.
The reference uses DwarfStar's expanded-weight activation policy, retaining the
model's explicit BF16 residual and FP8 KV boundaries.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-tools"))
from deepseek41_dspark import LosslessWeights
from glm53_quantize import SourceDB


def bf16(x):
    bits = np.asarray(x, dtype=np.float32).copy().view(np.uint32)
    bits += 0x7fff + ((bits >> 16) & 1)
    return (bits & np.uint32(0xffff0000)).view(np.float32)


def norm(x, weight):
    return bf16(x * (1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-20)) * weight)


def rope(x, start, inverse=False):
    result = x.copy()
    angles = np.arange(start, start + len(x), dtype=np.float64)[:, None] * (
        10000.0 ** (-np.arange(0, 64, 2, dtype=np.float64) / 64))
    angles = angles.reshape((len(x),) + (1,) * (x.ndim - 2) + (32,))
    c, s = np.cos(angles), np.sin(angles) * (-1 if inverse else 1)
    re, im = x[..., -64::2].copy(), x[..., -63::2].copy()
    result[..., -64::2] = bf16(re * c - im * s)
    result[..., -63::2] = bf16(re * s + im * c)
    return result


def fp8(x):
    blocks = bf16(x).reshape(-1, 32)
    scale = np.exp2(np.ceil(np.log2(np.maximum(abs(blocks).max(axis=1), 1e-4) / 448)))
    v = abs(blocks) / scale[:, None]
    step = np.exp2(np.maximum(np.floor(np.log2(np.maximum(v, 2 ** -9))) - 3, -9))
    q = np.minimum(np.rint(v / step) * step, 448)
    return bf16(np.copysign(q, blocks) * scale[:, None]).reshape(x.shape)


def mixes(x, fn, scale, base):
    flat = x.reshape(len(x), -1)
    z = (flat @ fn.T) / np.sqrt(np.mean(flat * flat, axis=-1, keepdims=True) + 1e-20)
    pre = 1 / (1 + np.exp(-(z[:, :4] * scale[0] + base[:4]))) + 1e-6
    post = 2 / (1 + np.exp(-(z[:, 4:8] * scale[1] + base[4:8])))
    comb = (z[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4)
    comb = np.exp(comb - comb.max(axis=-1, keepdims=True))
    comb = comb / comb.sum(axis=-1, keepdims=True) + 1e-6
    comb /= comb.sum(axis=-2, keepdims=True) + 1e-6
    for _ in range(19):
        comb /= comb.sum(axis=-1, keepdims=True) + 1e-6
        comb /= comb.sum(axis=-2, keepdims=True) + 1e-6
    return pre, post, comb


def collapse(x, pre):
    return bf16(np.sum(x * pre[..., None], axis=1))


def expand(block, residual, post, comb):
    return bf16(post[..., None] * block[:, None, :] + np.einsum("shk,shd->skd", comb, residual))


def swiglu(gate, up):
    gate, up = np.minimum(gate, 10), np.clip(up, -10, 10)
    return gate / (1 + np.exp(-gate)) * up


def q8_mmq(x):
    """CUDA MXFP4 MMQ: 32 values, FP32 scale, round away from zero."""
    blocks = np.asarray(x, np.float32).reshape(-1, 32)
    maximum = abs(blocks).max(axis=-1, keepdims=True)
    inverse = np.divide(127.0, maximum, out=np.ones_like(maximum), where=maximum != 0)
    scaled = blocks * inverse
    quantized = np.copysign(np.floor(abs(scaled) + 0.5), scaled)
    return (quantized / inverse).reshape(x.shape)


def q8_k(x):
    """ROCm routed experts: 256 values, FP32 scale, round to nearest even."""
    blocks = np.asarray(x, np.float32).reshape(-1, 256)
    maximum = abs(blocks).max(axis=-1, keepdims=True)
    inverse = np.divide(127.0, maximum, out=np.ones_like(maximum), where=maximum != 0)
    return (np.rint(blocks * inverse) / inverse).reshape(x.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("dumps", type=Path)
    ap.add_argument("--expert-activation", choices=("f32", "q8-mmq", "q8-k"), default="f32",
                    help="activation arithmetic used by the backend's native MXFP4 experts")
    args = ap.parse_args()
    db = SourceDB(str(args.source), index_validator=lambda _: None, scale_validator=lambda _: None,
                  tensor_prefix="mtp.")
    decoder = LosslessWeights()
    failures = []

    def weight(name):
        return decoder.to_f32(db, name)

    def load(name, shape):
        return np.fromfile(args.dumps / (name + ".f32"), dtype=np.float32).reshape(shape)

    def check(name, reference, tolerance=0.001):
        actual = load(name, reference.shape)
        relative = np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-30)
        cosine = np.sum(actual * reference) / max(np.linalg.norm(actual) * np.linalg.norm(reference), 1e-30)
        print(f"{name}: relative={relative:.7g} cosine={cosine:.9f} max={abs(actual-reference).max():.7g}", flush=True)
        if not np.isfinite(actual).all() or relative > tolerance:
            failures.append(name)
        return actual

    captures = []
    for stage in range(3):
        i = np.arange(137 * 4 * 5120, dtype=np.int64)
        x = ((i * 17 + stage * 29) % 257 - 128).astype(np.float32).reshape(137, 4, 5120) / 64
        captures.append(bf16(x.mean(axis=1))[-128:])
    main_x = norm(bf16(np.concatenate(captures, axis=-1) @ weight("mtp.0.main_proj.weight").T),
                  weight("mtp.0.main_norm.weight"))
    # Isolate stages so ordinary BF16 boundary changes do not compound.
    main_x = check("main_x", main_x, 0.003)
    residual = load("embedded", (5, 4, 5120))
    pre = np.tile([1, 0, 0, 0], (5, 1)).astype(np.float32)
    for stage in range(3):
        p = f"mtp.{stage}."
        def w(name):
            return weight(p + name)
        def linear(x, name, rounded=True):
            y = x @ w(name).T
            return bf16(y) if rounded else y
        def hc(x, site):
            return mixes(x, w(f"hc_{site}_fn"), w(f"hc_{site}_scale"), w(f"hc_{site}_base"))
        attn_pre, attn_post, attn_comb = hc(residual, "attn")
        x = norm(collapse(residual, pre), w("attn_norm.weight"))
        qr = norm(linear(x, "attn.wq_a.weight"), w("attn.q_norm.weight"))
        q = rope(linear(qr, "attn.wq_b.weight").reshape(5, 64, 512), 137)
        q = check(f"stage{stage}_q", q)
        kv = fp8(rope(norm(linear(x, "attn.wkv.weight"), w("attn.kv_norm.weight")), 137))
        kv = check(f"stage{stage}_kv", kv, 0.04)
        past = fp8(rope(norm(linear(main_x, "attn.wkv.weight"), w("attn.kv_norm.weight")), 9))
        window = np.concatenate([np.roll(past, 9, axis=0), kv])
        window = check(f"window{stage}", window, 0.04)
        scores = np.einsum("shd,td->sht", q, window) / np.sqrt(512)
        sink = w("attn.attn_sink")[None, :, None]
        mx = np.maximum(scores.max(axis=-1, keepdims=True), sink)
        probs = np.exp(scores - mx)
        probs /= probs.sum(axis=-1, keepdims=True) + np.exp(sink - mx)
        heads = rope(bf16(np.einsum("sht,td->shd", probs, window)), 137, True)
        heads = check(f"stage{stage}_heads", heads, 0.008)
        low = bf16(np.einsum("sgd,god->sgo", heads.reshape(5, 8, 4096),
                            w("attn.wo_a.weight").reshape(8, 1024, 4096))).reshape(5, 8192)
        block = linear(low, "attn.wo_b.weight")
        after = check(f"stage{stage}_after_attn", expand(block, residual, attn_post, attn_comb))
        pre, post, comb = hc(after, "ffn")
        x = check(f"stage{stage}_norm", norm(collapse(after, attn_pre), w("ffn_norm.weight")))
        route = check(f"stage{stage}_route_logits", linear(x, "ffn.gate.weight", False), 0.0001)
        probs = np.sqrt(np.logaddexp(route, 0))
        selected = np.argsort(-(probs + w("ffn.gate.bias")), axis=-1)[:, :3]
        rw = np.take_along_axis(probs, selected, axis=-1)
        rw = rw / rw.sum(axis=-1, keepdims=True) * 1.5
        mid = bf16(swiglu(linear(x, "ffn.shared_experts.w1.weight"), linear(x, "ffn.shared_experts.w3.weight")))
        shared = check(f"stage{stage}_shared", linear(mid, "ffn.shared_experts.w2.weight"))
        routed = np.zeros((5, 5120), np.float32)
        for token in range(5):
            for j, expert in enumerate(selected[token]):
                e = f"ffn.experts.{expert}."
                if args.expert_activation != "f32":
                    quantize = q8_mmq if args.expert_activation == "q8-mmq" else q8_k
                    qx = quantize(x[token])
                    mid = swiglu(linear(qx, e + "w1.weight", False),
                                 linear(qx, e + "w3.weight", False)) * rw[token, j]
                    routed[token] += linear(quantize(mid), e + "w2.weight", False)
                else:
                    mid = swiglu(linear(x[token], e + "w1.weight", False),
                                 linear(x[token], e + "w3.weight", False))
                    routed[token] += linear(mid, e + "w2.weight", False) * rw[token, j]
        routed = check(f"stage{stage}_routed", routed, 0.008)
        residual = check(f"stage{stage}_residual", expand(bf16(shared + routed), after, post, comb))
        # Production split stores pre/post/comb consecutively.
        split = load(f"stage{stage}_ffn_split", (5, 24))
        pre = split[:, :4]
    hidden = check("hidden", collapse(residual, pre), 0.003)
    check("norm", norm(hidden, weight("mtp.2.norm.weight")), 0.003)
    logits = load("logits", (5, 129280))
    previous = 100
    embeddings, head = weight("mtp.2.markov_head.embed.weight"), weight("mtp.2.markov_head.head.weight")
    conf_weight = weight("mtp.2.confidence_head.proj.weight")
    for row in range(5):
        embed = embeddings[previous]
        bias = check(f"markov{row}", head @ embed, 0.00001)
        check(f"confidence{row}", conf_weight @ np.concatenate([hidden[row], embed]), 0.00001)
        logits[row] += bias
        previous = int(logits[row].argmax())
    check("biased_logits", logits, 0.000001)
    if failures:
        raise SystemExit("FAIL: " + ", ".join(failures))
    print("Independent real-weight draft graph reference: PASS")


if __name__ == "__main__":
    main()
