"""Build a bounded full-depth fixture from the unchanged pinned reference model.

Torch is a test oracle only. Production does not import or require it.
"""

import argparse, dataclasses, hashlib, json, pathlib, struct, sys, types
import numpy as np
import torch
from tokenizers import Tokenizer, models

p = argparse.ArgumentParser()
p.add_argument("--reference-dir", type=pathlib.Path, required=True)
p.add_argument("--out", type=pathlib.Path, required=True)
p.add_argument("--tokens", type=int, default=520)
p.add_argument("--double-accumulation", action="store_true")
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[1]
for name, digest in {
    "model.py": "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d",
    "engram.py": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    "kernel.py": "e3083573a0a050f391ef0eafcd601aa18f9da71329141fe95dfcecaeaa9cc17e",
}.items():
    if hashlib.sha256((a.reference_dir / "inference" / name).read_bytes()).hexdigest() != digest:
        raise ValueError(f"{name}: expected uncensored 1017 pinned reference")
if (
    hashlib.sha256((a.reference_dir / "config.json").read_bytes()).hexdigest()
    != "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
):
    raise ValueError("config.json: expected pinned reference")
sys.path[:0] = [str(a.reference_dir / "inference"), str(root / "gguf-tools")]
import deepseek41_native_reference

sys.modules["kernel"] = deepseek41_native_reference
im = types.ModuleType("image_processor")
for n in ["IMAGE", "IMAGE_END", "IMAGE_NEW_LINE", "IMAGE_START"]:
    setattr(im, n, 0)
sys.modules["image_processor"] = im
vi = types.ModuleType("vision")
vi.Aligner = vi.ViT = None
sys.modules["vision"] = vi
import model as ref
import deepseek41_quantize as conv
from deepseek41_metadata import array_record, engram_layout, GGUF_ALIGNMENT
from glm53_quantize import (
    kv_bool,
    kv_string,
    kv_u32,
    tensor_header,
    align,
    QTYPE_F32,
    QTYPE_MXFP4,
    GGUF_STRING,
    GGUF_UINT32,
    GGUF_UINT64,
)

config = json.loads((a.reference_dir / "config.json").read_text())
c = config["text_config"]
c.update(
    hidden_size=256,
    moe_intermediate_size=256,
    vocab_size=64,
    num_attention_heads=64,
    q_lora_rank=64,
    o_lora_rank=32,
    o_groups=8,
    n_routed_experts=8,
    index_n_heads=32,
    engram_vocab_size=17,
    engram_compressed_vocab_size=64,
)
tok = Tokenizer(models.WordLevel({f"token{i}": i for i in range(64)}))


class Wrap:
    backend_tokenizer = tok

    def __len__(self):
        return 64


args = ref.ModelArgs(
    max_batch_size=1,
    max_seq_len=max(256, a.tokens + 1),
    temperature=0,
    dim=256,
    moe_inter_dim=256,
    vocab_size=64,
    n_layers=40,
    n_heads=64,
    n_routed_experts=8,
    n_activated_experts=6,
    q_lora_rank=64,
    head_dim=512,
    rope_head_dim=64,
    o_groups=8,
    o_lora_rank=32,
    window_size=128,
    compress_ratios=tuple(c["compress_ratios"][:40]),
    kv_source_layers=tuple(c["kv_source_layer_ids"]),
    index_source_layers=tuple(c["index_source_layer_ids"]),
    index_n_heads=32,
    index_head_dim=128,
    index_topk=512,
    compress_rope_theta=160000,
    original_seq_len=65536,
    rope_factor=16,
    route_scale=1.5,
    swiglu_limit=10,
    candidate_source_layer=20,
    candidate_topk_blocks=2048,
    candidate_block_size=8,
    engram_layer_ids=(1, 14),
    engram_vocab_size=17,
    engram_max_ngram_size=4,
    engram_n_heads=8,
    engram_head_dim=256,
    engram_compressed_vocab_size=64,
)
layout = ref.EngramLayout.from_args(args)
rows = [sum(sum(x) for x in layer) for layer in layout.primes]
args.engram_num_embeddings = tuple(rows)
c["engram_num_embeddings"] = rows
layout = engram_layout(c, tok)
torch.set_num_threads(4)
torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(1017)
m = ref.Transformer(args, Wrap()).eval()
# Initialize finite representable release-format weights; no zero graph or trivial routes.
with torch.no_grad():
    for name, param in m.named_parameters():
        if param.dtype == torch.float4_e2m1fn_x2:
            param.view(torch.uint8).copy_(torch.randint(0, 256, param.shape, dtype=torch.uint8))
        elif param.dtype == torch.float8_e8m0fnu:
            param.view(torch.uint8).fill_(117 if ".experts." in name else 118)
        elif name.endswith("norm.weight") or name.endswith(("q_weight", "k_weight")):
            param.copy_(1 + torch.randn(param.shape, dtype=torch.float32) * 0.02)
        elif name.endswith("_scale"):
            param.fill_(0.1)
        elif name.endswith("_base"):
            param.copy_(torch.randn(param.shape, dtype=torch.float32) * 0.2)
        elif param.ndim == 1:
            param.copy_(torch.randn(param.shape, dtype=torch.float32) * 0.1)
        elif param.dtype == torch.float8_e4m3fn:
            param.copy_((torch.randn(param.shape, dtype=torch.float32) * 2).to(param.dtype))
        else:
            param.copy_(
                torch.randn(param.shape, dtype=torch.float32)
                * (0.2 if name == "embed.weight" else 0.02)
            )
    # The published head is BF16 even though the reference keeps it in F32.
    m.head.weight.copy_(m.head.weight.to(torch.bfloat16).float())
state = dict(m.named_parameters())
for i in range(40):
    name = f"layers.{i}.ffn.gate.bias_vl"
    state[name] = state[name.removesuffix("_vl")].detach().clone()


class DB:
    def __init__(self):
        self.tensors = {}
        for n, t in state.items():
            dtype = {
                torch.float32: "F32",
                torch.bfloat16: "BF16",
                torch.float8_e4m3fn: "F8_E4M3",
                torch.float8_e8m0fnu: "F8_E8M0",
                torch.float4_e2m1fn_x2: "I8",
            }[t.dtype]
            self.tensors[n] = {"dtype": dtype, "shape": list(t.shape)}

    def info(self, n):
        return self.tensors[n]

    def read(self, n):
        return state[n].detach().contiguous().view(torch.uint8).numpy().tobytes()

    def iter_read(self, name, byte_start=0, byte_count=None):
        raw = self.read(name)
        yield raw[byte_start : None if byte_count is None else byte_start + byte_count]


db = DB()
plan = conv.build_plan(db, config, "native")
quant = conv.NativeQuantizer(str(root / "gguf-tools/libds4quants.dylib"))
records = [
    kv_string("general.architecture", "deepseek41"),
    kv_string("deepseek41.engram.encoding", "e4m3_e8m0_32_row264"),
    kv_u32("general.alignment", GGUF_ALIGNMENT),
    kv_bool("deepseek41.native_weights", True),
    array_record(
        "deepseek41.native.fp8_inputs",
        GGUF_STRING,
        [
            it.name
            for it in plan
            if it.qtype == QTYPE_F32
            and db.info(it.source)["dtype"] == "F8_E4M3"
            and ".wo_a." not in it.source
        ],
    ),
]
for key, typ, v in [
    ("token_map", GGUF_UINT32, layout["token_map"]),
    ("rows", GGUF_UINT32, rows),
    ("primes", GGUF_UINT32, sum(layout["primes"], [])),
    ("multipliers", GGUF_UINT64, sum(layout["multipliers"], [])),
]:
    records.append(array_record("deepseek41.engram." + key, typ, v))
records += [
    kv_u32("deepseek41.engram.compressed_vocab_size", 64),
    kv_u32("deepseek41.engram.pad_id", 2),
]
out = a.out
out.mkdir(parents=True, exist_ok=True)
header = (
    b"GGUF"
    + struct.pack("<IQQ", 3, len(plan), len(records))
    + b"".join(records)
    + b"".join(tensor_header(it) for it in plan)
)
start = align(len(header), GGUF_ALIGNMENT)
with (out / "model.gguf").open("wb") as f:
    f.write(header)
    f.write(bytes(start - len(header)))
    for it in plan:
        assert f.tell() == start + it.offset, it.name
        if it.role == "engram_disk":
            raw = np.frombuffer(db.read(it.source), np.uint8).reshape(-1, 256)
            sc = np.frombuffer(db.read(conv.scale_name(it.source)), np.uint8).reshape(-1, 8)
            data = np.concatenate([raw, sc], axis=1).tobytes()
        elif it.qtype == QTYPE_MXFP4:
            data = b"".join(
                quant.to_mxfp4(db, it.source.format(expert=i)) for i in range(it.expert_count)
            )
        else:
            data = quant.to_f32(db, it.source).astype("<f4").tobytes()
        assert len(data) == it.nbytes, (it.name, len(data), it.nbytes)
        f.write(data)
        f.write(bytes(align(len(data), GGUF_ALIGNMENT) - len(data)))
print("fixture written", flush=True)
# This control changes accumulation only, after the exact same artifact is written.
# It measures quantization-boundary sensitivity; it is not another weight recipe.
if a.double_accumulation:

    def double_linear(x, weight, bias=None):
        return torch.nn.functional.linear_original(
            x.double(), weight.double(), None if bias is None else bias.double()
        ).to(x.dtype)

    torch.nn.functional.linear_original = torch.nn.functional.linear
    torch.nn.functional.linear = double_linear
tokens = [(i * 17 + 3) % 64 for i in range(a.tokens)]
np.array(tokens, dtype="<i4").tofile(out / "tokens.i32")
logits = []
with torch.inference_mode():
    for pos, token in enumerate(tokens):
        _, y, _ = m(torch.tensor([[token]]), pos)
        logits.append(y.float().numpy())
        if pos % 16 == 0:
            print("reference token", pos, flush=True)
np.concatenate(logits).astype("<f4").tofile(out / "reference.f32")
# Validate the lazy source oracle against the unchanged, fully loaded model
# before it is trusted with checkpoint-sized tables and expert banks.
from deepseek41_native_checkpoint_reference import first_token

bounded = first_token(ref, args, db, Wrap(), tokens[0], {})
np.testing.assert_allclose(bounded, logits[0], rtol=0, atol=1e-6)
bounded.astype("<f4").tofile(out / "bounded-source-first-token.f32")
print("bounded source oracle matches original reference", flush=True)
(out / "args.json").write_text(json.dumps(dataclasses.asdict(args), indent=2))
print("reference complete", flush=True)
