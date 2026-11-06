#!/usr/bin/env python3
"""Extract DeepSeek V4.1 Flash DSpark without further weight quantization.

Routed experts retain their native MXFP4 codes and scales. Dense weights are
expanded to F16 only when every value is exactly representable, otherwise F32.
The target model supplies the token embedding and output projection at runtime.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time

from deepseek41_quantize import NativeQuantizer, scale_name
from glm53_quantize import (
    SourceDB, TensorPlan, QTYPE_F16, QTYPE_F32, QTYPE_MXFP4, align,
    conversion_signature, kv_string, kv_u32, kv_u32_array, load_resume_state,
    print_plan, qtype_nbytes, save_resume_state, tensor_header,
    read_exact,
)

ALIGNMENT = 16384


class LosslessWeights(NativeQuantizer):
    def __init__(self):
        # No C quantizer or imatrix is needed: this converter changes layouts,
        # never weight values.
        import numpy as np
        self.np = np
        self.fp8_lut = self._build_fp8_lut()

    def dense_type(self, db, name):
        values = self.to_f32(db, name)
        with self.np.errstate(over="ignore", under="ignore"):
            restored = values.astype(self.np.float16).astype(self.np.float32)
        return QTYPE_F16 if self.np.array_equal(values, restored) else QTYPE_F32

    def dense_bytes(self, db, item):
        values = self.to_f32(db, item.source)
        if item.qtype == QTYPE_F16:
            with self.np.errstate(over="ignore", under="ignore"):
                restored = values.astype(self.np.float16).astype(self.np.float32)
            if not self.np.array_equal(values, restored):
                raise ValueError(f"{item.name}: F16 would change source values")
        return self.encode(values, item.qtype)

    def expert_bytes(self, db, name):
        np = self.np
        info, sinfo = db.info(name), db.info(scale_name(name))
        rows, packed_cols = info["shape"]
        if (info["dtype"] != "I8" or packed_cols % 16 or
                sinfo["dtype"] != "F8_E8M0" or
                sinfo["shape"] != [rows, packed_cols // 16]):
            raise ValueError(f"{name}: invalid native FP4 tensor or scales")
        codes = np.frombuffer(db.read(name), dtype=np.uint8).reshape(-1, 16)
        scales = np.frombuffer(db.read(scale_name(name)), dtype=np.uint8)
        if np.any(scales == 255):
            raise ValueError(f"{name}: nonfinite MXFP4 scale")
        # Source nibbles are consecutive. GGUF puts values 0..15 in the low
        # nibbles, 16..31 in the high nibbles, after the unchanged E8M0 byte.
        unpacked = np.empty((codes.shape[0], 32), dtype=np.uint8)
        unpacked[:, 0::2], unpacked[:, 1::2] = codes & 15, codes >> 4
        packed = np.empty((codes.shape[0], 17), dtype=np.uint8)
        packed[:, 0] = scales
        packed[:, 1:] = unpacked[:, :16] | (unpacked[:, 16:] << 4)
        if not (np.array_equal(packed[:, 1:] & 15, unpacked[:, :16]) and
                np.array_equal(packed[:, 1:] >> 4, unpacked[:, 16:])):
            raise ValueError(f"{name}: MXFP4 repacking changed a code")
        return packed.tobytes()


def build_plan(db, config, weights):
    if config.get("model_type") != "deepseek_v41":
        raise ValueError("expected a DeepSeek V4.1 checkpoint")
    c = config["text_config"]
    if config["quantization_config"]["weight_block_size"] != [32, 32]:
        raise ValueError("expected native 32x32 FP8 blocks")
    dim, inter, hc = c["hidden_size"], c["moe_intermediate_size"], c["hc_mult"]
    heads, hd = c["num_attention_heads"], c["head_dim"]
    qrank, orank, groups = c["q_lora_rank"], c["o_lora_rank"], c["o_groups"]
    experts = c["dspark_n_routed_experts"]
    stages, vocab, rank = c["num_nextn_predict_layers"], c["vocab_size"], c["dspark_markov_rank"]
    targets = c["dspark_target_layer_ids"]
    if (stages != 3 or len(targets) != 3 or len(set(targets)) != 3 or
            any(not 0 <= layer < c["num_hidden_layers"] for layer in targets) or
            not 0 < c["dspark_num_experts_per_tok"] <= experts or
            not 0 <= c["dspark_noise_token_id"] < vocab or
            not 0 < c["dspark_block_size"] <= c["sliding_window"]):
        raise ValueError("invalid DSpark dimensions or target layers")
    plan, consumed = [], set()

    def claim(name, shape, dtype=None):
        info = db.info(name)
        if info["shape"] != list(shape) or (dtype and info["dtype"] != dtype):
            raise ValueError(f"{name}: unexpected {info['dtype']} {info['shape']}")
        consumed.add(name)
        if info["dtype"] in ("I8", "F8_E4M3"):
            expected = ([shape[0], shape[1] // 16] if info["dtype"] == "I8"
                        else [(n + 31) // 32 for n in shape])
            scale = db.info(scale_name(name))
            if scale["dtype"] != "F8_E8M0" or scale["shape"] != expected:
                raise ValueError(f"{name}: invalid native scale shape")
            consumed.add(scale_name(name))

    def regular(dst, src, shape, role):
        claim(src, shape)
        # Keep norms, routers, HC and confidence in F32 for the runtime's
        # FP32 reductions. Matrix F16 is selected only after a lossless scan.
        qt = weights.dense_type(db, src) if role in ("projection", "markov") else QTYPE_F32
        plan.append(TensorPlan(dst, tuple(reversed(shape)), qt, role, source=src))

    for stage in range(stages):
        prefix = f"mtp.{stage}"
        for site in ("attn", "ffn"):
            for part, shape in (("fn", (hc * (hc + 2), hc * dim)),
                                ("base", (hc * (hc + 2),)), ("scale", (3,))):
                regular(f"{prefix}.hc_{site}_{part}.weight", f"{prefix}.hc_{site}_{part}", shape, "hc")
            regular(f"{prefix}.{site}_norm.weight", f"{prefix}.{site}_norm.weight", (dim,), "norm")
        for dst, src, shape, role in (
            ("attn_sinks.weight", "attn_sink", (heads,), "norm"),
            ("attn_q_a.weight", "wq_a.weight", (qrank, dim), "projection"),
            ("attn_q_b.weight", "wq_b.weight", (heads * hd, qrank), "projection"),
            ("attn_q_a_norm.weight", "q_norm.weight", (qrank,), "norm"),
            ("attn_kv.weight", "wkv.weight", (hd, dim), "projection"),
            ("attn_kv_a_norm.weight", "kv_norm.weight", (hd,), "norm"),
            ("attn_output_a.weight", "wo_a.weight", (groups * orank, heads * hd // groups), "projection"),
            ("attn_output_b.weight", "wo_b.weight", (dim, groups * orank), "projection"),
        ):
            regular(f"{prefix}.{dst}", f"{prefix}.attn.{src}", shape, role)
        regular(f"{prefix}.ffn_gate_inp.weight", f"{prefix}.ffn.gate.weight", (experts, dim), "router")
        for suffix, dst in (("bias", "exp_probs_b.bias"), ("bias_vl", "exp_probs_b_vl.bias")):
            regular(f"{prefix}.{dst}", f"{prefix}.ffn.gate.{suffix}", (experts,), "router")
        for part, src, shape in (("gate", "w1", (inter, dim)),
                                 ("up", "w3", (inter, dim)), ("down", "w2", (dim, inter))):
            regular(f"{prefix}.ffn_{part}_shexp.weight", f"{prefix}.ffn.shared_experts.{src}.weight", shape, "projection")
            pattern = f"{prefix}.ffn.experts.{{expert}}.{src}.weight"
            for expert in range(experts):
                claim(pattern.format(expert=expert), (shape[0], shape[1] // 2), "I8")
            plan.append(TensorPlan(f"{prefix}.ffn_{part}_exps.weight", (*reversed(shape), experts),
                                   QTYPE_MXFP4, "experts", source=pattern, expert_layer=stage,
                                   expert_part=part, expert_count=experts))
        if stage == 0:
            regular(f"{prefix}.main_proj.weight", f"{prefix}.main_proj.weight", (dim, dim * len(targets)), "projection")
            regular(f"{prefix}.main_norm.weight", f"{prefix}.main_norm.weight", (dim,), "norm")
        if stage == stages - 1:
            regular(f"{prefix}.norm.weight", f"{prefix}.norm.weight", (dim,), "norm")
            for src, dst in (("embed", "markov_w1"), ("head", "markov_w2")):
                regular(f"{prefix}.markov_head.{dst}.weight", f"{prefix}.markov_head.{src}.weight", (vocab, rank), "markov")
            regular(f"{prefix}.confidence_head.proj.weight", f"{prefix}.confidence_head.proj.weight", (1, dim + rank), "confidence")
    unknown = {name for name in db.tensors if name.startswith("mtp.")} - consumed
    if unknown:
        raise ValueError(f"unclaimed DSpark tensors: {sorted(unknown)[:10]}")
    offset = 0
    for item in plan:
        item.offset, item.nbytes = offset, qtype_nbytes(item.qtype, item.shape)
        offset += align(item.nbytes, ALIGNMENT)
    return plan


def metadata(config, revision):
    c = config["text_config"]
    records = [
        kv_string("general.architecture", "deepseek41-dspark"),
        kv_string("general.name", "DeepSeek V4.1 Flash DSpark"),
        kv_string("general.source.url", "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash"),
        kv_string("general.source.revision", revision),
        kv_u32("general.alignment", ALIGNMENT),
        kv_string("dspark.target_architecture", "deepseek41"),
        kv_string("dspark.source_config", json.dumps(config, sort_keys=True, separators=(",", ":"))),
        kv_string("dspark.weight_encoding", "native MXFP4; lossless F16/F32 expansion"),
        kv_u32_array("dspark.target_layer_ids", c["dspark_target_layer_ids"]),
    ]
    for key in ("block_size", "markov_rank", "noise_token_id", "n_routed_experts", "num_experts_per_tok"):
        records.append(kv_u32(f"dspark.{key}", c[f"dspark_{key}"]))
    return records


def source_fingerprint(db):
    shards = sorted({info["shard"] for name, info in db.tensors.items() if name.startswith("mtp.")})
    result = []
    for shard in shards:
        h = hashlib.sha256()
        with open(os.path.join(db.hf_dir, shard), "rb") as fp:
            for chunk in iter(lambda: fp.read(8 << 20), b""):
                h.update(chunk)
        result.append((shard, h.hexdigest()))
    return result


def write_gguf(args, db, weights, plan, records):
    data_start, data_bytes = print_plan(plan, records, [], ALIGNMENT)
    if os.path.exists(args.out):
        raise ValueError(f"refusing to overwrite {args.out}")
    partial, journal = args.out + ".partial", args.out + ".partial.json"
    fingerprint = source_fingerprint(db)
    signature = hashlib.sha256((conversion_signature(plan, records, [], None) +
                                json.dumps(fingerprint)).encode()).hexdigest()
    completed = 0
    if os.path.exists(partial) or os.path.exists(journal):
        if not args.resume or not (os.path.exists(partial) and os.path.exists(journal)):
            raise ValueError("partial file and journal require --resume")
        completed = load_resume_state(journal, signature, plan)
    end = data_start + (plan[completed - 1].offset + align(plan[completed - 1].nbytes, ALIGNMENT) if completed else 0)
    if shutil.disk_usage(os.path.dirname(os.path.abspath(args.out))).free < data_start + data_bytes - end + (32 << 30):
        raise ValueError("insufficient disk space for output plus 32 GiB reserve")
    header = b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(records))
    header += b"".join(records) + b"".join(tensor_header(item) for item in plan)
    header += bytes(data_start - len(header))
    if os.path.exists(partial):
        with open(partial, "rb") as fp:
            if fp.read(data_start) != header or os.fstat(fp.fileno()).st_size < end:
                raise ValueError("partial GGUF is truncated or has a different header")
    else:
        with open(partial, "xb") as fp:
            fp.write(header)
            fp.flush()
            os.fsync(fp.fileno())
        save_resume_state(journal, signature, 0)
    with open(partial, "r+b") as fp:
        fp.truncate(end)
        fp.seek(end)
        for i in range(completed, len(plan)):
            item, start = plan[i], time.monotonic()
            if fp.tell() != data_start + item.offset:
                raise ValueError(f"incorrect offset for {item.name}")
            if item.is_expert:
                for expert in range(item.expert_count):
                    fp.write(weights.expert_bytes(db, item.source.format(expert=expert)))
            else:
                fp.write(weights.dense_bytes(db, item))
            if fp.tell() != data_start + item.offset + item.nbytes:
                raise ValueError(f"incorrect payload size for {item.name}")
            fp.write(bytes(align(item.nbytes, ALIGNMENT) - item.nbytes))
            fp.flush()
            os.fsync(fp.fileno())
            save_resume_state(journal, signature, i + 1)
            print(f"[{i + 1}/{len(plan)}] {item.name}: {time.monotonic() - start:.1f}s", flush=True)
    # Refuse publication if an in-place source edit occurred during conversion.
    if source_fingerprint(db) != fingerprint:
        raise ValueError("source shards changed during conversion")
    os.rename(partial, args.out)
    os.unlink(journal)


def verify_gguf(path, db, weights, plan, records):
    """Check the complete layout and every payload, not a sample of experts."""
    with open(path, "rb") as fp:
        if read_exact(fp, 24, "header") != b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(records)):
            raise ValueError("GGUF header differs from source plan")
        encoded = b"".join(records) + b"".join(tensor_header(t) for t in plan)
        if read_exact(fp, len(encoded), "metadata and tensors") != encoded:
            raise ValueError("GGUF metadata or tensor layout differs from source plan")
        start = align(fp.tell(), ALIGNMENT)
        size = start + plan[-1].offset + align(plan[-1].nbytes, ALIGNMENT)
        if os.fstat(fp.fileno()).st_size != size:
            raise ValueError("incorrect GGUF file size")
        for item in plan:
            fp.seek(start + item.offset)
            if item.is_expert:
                for expert in range(item.expert_count):
                    expected = weights.expert_bytes(db, item.source.format(expert=expert))
                    if read_exact(fp, len(expected), item.name) != expected:
                        raise ValueError(f"{item.name}: expert {expert} payload differs from source")
            else:
                expected = weights.dense_bytes(db, item)
                if read_exact(fp, len(expected), item.name) != expected:
                    raise ValueError(f"{item.name}: dense payload differs from source")
        print(f"PASS: all {len(plan)} tensor layouts and payloads match the source", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="audit an existing --out file without writing it")
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.source_revision):
        parser.error("source revision must be a full commit hash")
    with open(os.path.join(args.hf, "config.json"), "rb") as fp:
        config = json.load(fp)
    db = SourceDB(args.hf, index_validator=lambda _: None, scale_validator=lambda _: None,
                  tensor_prefix="mtp.")
    try:
        weights = LosslessWeights()
        plan = build_plan(db, config, weights)
        records = metadata(config, args.source_revision)
        if args.verify:
            verify_gguf(args.out, db, weights, plan, records)
        elif args.dry_run:
            print_plan(plan, records, [], ALIGNMENT)
        else:
            write_gguf(args, db, weights, plan, records)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(f"deepseek41-dspark: {error}")
