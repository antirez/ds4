"""CPU translations of the pinned release's TileLang kernels, for bounded reference models."""

import torch
import torch.nn.functional as F


def act_quant(x, block_size=32, scale_fmt="ue8m0", scale_dtype=None, inplace=False):
    v = x.float().reshape(*x.shape[:-1], -1, block_size)
    scale = torch.exp2(torch.ceil(torch.log2(v.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)))
    q = (v / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * scale).reshape(x.shape).to(x.dtype))
        return x
    return q.reshape(x.shape), scale.squeeze(-1)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    v = x.float().reshape(*x.shape[:-1], -1, block_size)
    a = v.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        scale = (a.clamp_min(6 * 2**-9) / 6).to(scale_dtype).float()
    else:
        scale = torch.exp2(torch.ceil(torch.log2(a.clamp_min(6 * 2**-126) / 6)))
    u = v / scale
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    # argmin's first occurrence gives ties to the even E2M1 significand.
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    idx = (u.abs()[..., None] - lut[order]).abs().argmin(-1)
    out = (torch.copysign(lut[order[idx]], u) * scale).reshape(x.shape).to(x.dtype)
    if inplace:
        x.copy_(out)
        return x
    raise ValueError("reference only needs in-place FP4 activation rounding")


def _block_scaled_linear(x, scales, weights, weight_scales, block_size):
    # kernel.py clears C_local after each scale block. One long BLAS dot
    # changes the FP32 rounding before the final BF16 conversion.
    out = torch.zeros((*x.shape[:-1], weights.shape[0]), dtype=torch.float32)
    for k in range(x.shape[-1] // block_size):
        start, end = k * block_size, (k + 1) * block_size
        block = F.linear(x[..., start:end], weights[:, start:end])
        out += block * scales[..., k, None] * weight_scales[:, k]
    return out.to(torch.bfloat16)


def fp8_gemm(x, s, w, ws, scale_dtype=None, block_size=32):
    scales = ws.float().repeat_interleave(block_size, 0)[: w.shape[0]]
    return _block_scaled_linear(x.float(), s.float(), w.float(), scales, block_size)


def fp4_gemm(x, s, w, ws, scale_dtype=None, act_block_size=32):
    bits = w.view(torch.uint8)
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32
    )
    weights = torch.stack((lut[(bits & 15).long()], lut[(bits >> 4).long()]), -1).flatten(-2)
    scales = s.float().repeat_interleave(act_block_size // 32, -1)
    return _block_scaled_linear(x.float(), scales, weights, ws.float(), 32)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    h = hc_mult
    pre = torch.sigmoid(mixes[..., :h] * hc_scale[0] + hc_base[:h]) + eps
    post = 2 * torch.sigmoid(mixes[..., h : 2 * h] * hc_scale[1] + hc_base[h : 2 * h])
    comb = (mixes[..., 2 * h :] * hc_scale[2] + hc_base[2 * h :]).unflatten(-1, (h, h))
    comb = torch.exp(comb - comb.amax(-1, keepdim=True))
    comb = comb / comb.sum(-1, keepdim=True) + eps
    comb /= comb.sum(-2, keepdim=True) + eps
    for _ in range(sinkhorn_iters - 1):
        comb /= comb.sum(-1, keepdim=True) + eps
        comb /= comb.sum(-2, keepdim=True) + eps
    return pre, post, comb


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    # Follow kernel.py's 64-key online-softmax blocks and BF16 probability cast.
    out = torch.zeros_like(q, dtype=torch.float32)
    maxval = torch.full(q.shape[:-1], -1e30, dtype=torch.float32)
    total = torch.zeros_like(maxval)
    for start in range(0, topk_idxs.shape[-1], 64):
        ids = topk_idxs[..., start : start + 64].long()
        keys = kv[torch.arange(kv.shape[0])[:, None, None], ids.clamp_min(0)]
        scores = torch.einsum("bshd,bskd->bshk", q.float(), keys.float()) * softmax_scale
        scores.masked_fill_((ids < 0).unsqueeze(-2), -torch.inf)
        newmax = torch.maximum(maxval, scores.amax(-1))
        correction = torch.exp(maxval - newmax)
        probs = torch.exp(scores - newmax[..., None])
        total = total * correction + probs.sum(-1)
        out = out * correction[..., None] + torch.einsum(
            "bshk,bskd->bshd", probs.to(torch.bfloat16).float(), keys.float()
        )
        maxval = newmax
    return (out / (total + torch.exp(attn_sink - maxval))[..., None]).to(q.dtype)
