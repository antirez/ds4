#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface-hub>=1.0",
#   "numpy>=2.0",
# ]
# ///
"""Build the versioned DS4 Qwen3.8-Flash-Next fast pack.

The converter consumes the official BF16 safetensors checkpoint.  It emits
one deterministic trunk GGUF, an SSD-backed Q4_1 PLE GGUF, and optional
vision/MTP GGUF sidecars.  Unsloth/GGUF input is deliberately not a supported
source format: DS4 records and validates the official source revision and the
exact GGML block geometry used by its native kernels.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import dataclasses
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
from huggingface_hub import get_token, hf_hub_download

from glm53_quantize import (
    GGUF_ALIGNMENT,
    GGUF_VERSION,
    align,
    kv_f32,
    kv_string,
    kv_u32,
    kv_u64,
    load_tokenizer_records,
    pack_string,
)


ARCH = "qwen4-exp"
PACK_VERSION = 3
CONVERSION_STATE_VERSION = 4
MANIFEST_NAME = "qwen3.8-flash-next-q4.manifest.json"
STATE_NAME = "qwen3.8-flash-next-q4.conversion-state.json"
BASE_NAME = (
    "Qwen3.8-Flash-Next-Q4KExperts-BF16Emb-BF16Control-"
    "Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf"
)
PLE_NAME = "Qwen3.8-Flash-Next-PLE-Q4_1.gguf"
VISION_NAME = "qwen3.8-flash-next-q4-vision.gguf"
MTP_NAME = "qwen3.8-flash-next-q4-mtp.gguf"

# Selected MTP routed-expert importance mode (None or "weight-energy").
MTP_IMATRIX_MODE: str | None = None
Q2_PACK_VERSION = 4
Q2_CONVERSION_STATE_VERSION = 1
Q2_MANIFEST_NAME = "qwen3.8-flash-next-q2.manifest.json"
Q2_STATE_NAME = "qwen3.8-flash-next-q2.conversion-state.json"
Q2_BASE_NAME = (
    "Qwen3.8-Flash-Next-IQ2XXSGateUp-Q2KDown-BF16Emb-BF16Control-"
    "Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf"
)
Q2_PLE_NAME = "Qwen3.8-Flash-Next-Q2-PLE-Q4_1.gguf"
Q2_VISION_NAME = "qwen3.8-flash-next-q2-vision.gguf"
Q2_MTP_NAME = "qwen3.8-flash-next-q2-mtp.gguf"
Q2_IMATRIX_REPOSITORY = "unsloth/Qwen3.8-Flash-Next-GGUF"
Q2_IMATRIX_REVISION = "c8b5954a88c2775c546b92593eda40ea041d3176"
Q2_IMATRIX_SHA256 = (
    "a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154"
)
LEGACY_ARTIFACT_NAMES = (
    "qwen3.8-flash-next-q4.gguf",
    "qwen3.8-flash-next-q4-ple.safetensors",
    "qwen3.8-flash-next-q4-00001-of-00004.gguf",
    "qwen3.8-flash-next-q4-00002-of-00004.gguf",
    "qwen3.8-flash-next-q4-00003-of-00004.gguf",
    "qwen3.8-flash-next-q4-00004-of-00004.gguf",
)
FREE_SPACE_RESERVE_BYTES = 32 * (1 << 30)
NGRAM_MARK = ".ple.ple_embedding.ngram_embedding.shard_"
PLE_AUX_SUFFIXES = (
    ".ple.ple_embedding.layer_multipliers",
    ".ple.ple_embedding.ngram_heads_offsets",
    ".ple.ple_embedding.ngram_heads_vocab_sizes",
)
PLE_WEIGHT_NAME = "ple.weight"
PLE_AUX_NAMES = (
    "ple.layer_multipliers",
    "ple.ngram_heads_offsets",
    "ple.ngram_heads_vocab_sizes",
)
ARTIFACT_TENSOR_COUNTS = {
    "base": 1211,
    "vision": 333,
    "mtp": 32,
    "ple": 4,
}

QTYPE_F32 = 0
QTYPE_F16 = 1
QTYPE_Q4_1 = 3
QTYPE_Q8_0 = 8
QTYPE_Q2_K = 10
QTYPE_Q4_K = 12
QTYPE_IQ2_XXS = 16
QTYPE_I32 = 26
QTYPE_I64 = 27
QTYPE_BF16 = 30

DTYPE_TO_QTYPE = {
    "F32": QTYPE_F32,
    "F16": QTYPE_F16,
    "Q4_1": QTYPE_Q4_1,
    "Q8_0": QTYPE_Q8_0,
    "Q2_K": QTYPE_Q2_K,
    "Q4_K": QTYPE_Q4_K,
    "IQ2_XXS": QTYPE_IQ2_XXS,
    "I32": QTYPE_I32,
    "U32": QTYPE_I32,
    "I64": QTYPE_I64,
    "BF16": QTYPE_BF16,
}
DTYPE_TO_NUMPY = {
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I32": np.dtype("<i4"),
    "U32": np.dtype("<u4"),
    "I64": np.dtype("<i8"),
    "BF16": np.dtype("<u2"),
}
DTYPE_BYTES = {key: value.itemsize for key, value in DTYPE_TO_NUMPY.items()}
QTYPE_LAYOUT = {
    "Q4_1": (32, 20),
    "Q8_0": (32, 34),
    "Q2_K": (256, 84),
    "Q4_K": (256, 144),
    "IQ2_XXS": (256, 66),
}


@dataclasses.dataclass(frozen=True)
class PackProfile:
    name: str
    pack_version: int
    state_version: int
    manifest_name: str
    state_name: str
    base_name: str
    ple_name: str
    vision_name: str
    mtp_name: str
    gate_up_qtype: str
    down_qtype: str
    required_ds4_version: str
    pack_identity: str


Q4_PROFILE = PackProfile(
    name="q4",
    pack_version=PACK_VERSION,
    state_version=CONVERSION_STATE_VERSION,
    manifest_name=MANIFEST_NAME,
    state_name=STATE_NAME,
    base_name=BASE_NAME,
    ple_name=PLE_NAME,
    vision_name=VISION_NAME,
    mtp_name=MTP_NAME,
    gate_up_qtype="Q4_K",
    down_qtype="Q4_K",
    required_ds4_version="qwen3.8-flash-next-q4",
    pack_identity="ds4-qwen3.8-flash-next-q4-v3",
)
Q2_PROFILE = PackProfile(
    name="q2",
    pack_version=Q2_PACK_VERSION,
    state_version=Q2_CONVERSION_STATE_VERSION,
    manifest_name=Q2_MANIFEST_NAME,
    state_name=Q2_STATE_NAME,
    base_name=Q2_BASE_NAME,
    ple_name=Q2_PLE_NAME,
    vision_name=Q2_VISION_NAME,
    mtp_name=Q2_MTP_NAME,
    gate_up_qtype="IQ2_XXS",
    down_qtype="Q2_K",
    required_ds4_version="qwen3.8-flash-next-q2",
    pack_identity="ds4-qwen3.8-flash-next-q2-v4",
)
PACK_PROFILES = {profile.name: profile for profile in (Q4_PROFILE, Q2_PROFILE)}

CONFIG = {
    "layers": 48,
    "hidden": 2560,
    "vocab": 248320,
    "context": 262144,
    "attn_heads": 24,
    "kv_heads": 2,
    "head_dim": 256,
    "rope_dim": 64,
    "rope_theta": 10_000_000.0,
    "full_attention_interval": 4,
    "linear_key_heads": 16,
    "linear_value_heads": 48,
    "linear_head_dim": 128,
    "linear_conv": 4,
    "experts": 512,
    "experts_used": 10,
    "expert_ff": 640,
    "shared_ff": 640,
    "hc_count": 4,
    "hc_lowrank": 320,
    "qsa_heads": 4,
    "qsa_kv_heads": 1,
    "qsa_dim": 128,
    "qsa_topk": 2048,
    "qsa_ratio": 4,
    # Checkpoint config uses one-indexed layer ids; the runtime stores index 1.
    "ple_layer_ids": (2,),
    "ple_layer": 1,
    "ple_dim": 2560,
    "ple_row_dim": 160,
    "ple_rows": 320_001_536,
    "ple_conv": 4,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ngram_vocab_base": 20_000_000,
    "ngram_vocab_divisor": 128,
    "ngram_seed": 1234,
    "split_ngram_parts": 128,
    "output_gate_type": "sigmoid",
    # The model-config EOS is the PLE history-reset token.  Generation uses
    # the tokenizer's <|im_end|> token instead.
    "ngram_eos": 248044,
    "generation_eos": 248046,
    "image_token": 248056,
    "vision_start_token": 248053,
    "vision_end_token": 248054,
    "rms_eps": 1.0e-6,
    "routed_down_physical_input": 768,
    "vision_fc2_physical_input": 4320,
}

NORM_FOLD_SUFFIXES = (
    "hc_norm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "q_layernorm.weight",
    "k_layernorm.weight",
    "ple.norm_key.weight",
    "ple.norm_query.weight",
    "ple.norm_conv.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)


def fail(message: str):
    raise ValueError(message)


def pack_profile(value: str | PackProfile | None = None) -> PackProfile:
    if isinstance(value, PackProfile):
        return value
    try:
        return PACK_PROFILES[value or "q4"]
    except KeyError:
        fail(f"unknown Qwen pack profile {value!r}")


def profile_from_args(args) -> PackProfile:
    return pack_profile(getattr(args, "profile", "q4"))


def product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def sha256_file(path: Path, offset: int = 0, length: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        fp.seek(offset)
        remaining = length
        while remaining is None or remaining:
            amount = 16 << 20 if remaining is None else min(16 << 20, remaining)
            data = fp.read(amount)
            if not data:
                if remaining not in (None, 0):
                    fail(f"short read while hashing {path}")
                break
            digest.update(data)
            if remaining is not None:
                remaining -= len(data)
    return digest.hexdigest()


def decode_safetensors_header(raw_length: bytes, raw_header: bytes,
                              source: str) -> tuple[dict, int]:
    if len(raw_length) != 8:
        fail(f"{source}: truncated safetensors header")
    length = struct.unpack("<Q", raw_length)[0]
    if length == 0 or length > 1 << 30:
        fail(f"{source}: invalid safetensors header size {length}")
    if len(raw_header) != length:
        fail(f"{source}: truncated safetensors JSON header")
    header = json.loads(raw_header)
    header.pop("__metadata__", None)
    return header, 8 + length


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as fp:
        raw_length = fp.read(8)
        if len(raw_length) != 8:
            fail(f"{path}: truncated safetensors header")
        length = struct.unpack("<Q", raw_length)[0]
        if length == 0 or length > 1 << 30:
            fail(f"{path}: invalid safetensors header size {length}")
        raw_header = fp.read(length)
    return decode_safetensors_header(raw_length, raw_header, os.fspath(path))


def validate_hub_filename(filename: str) -> str:
    path = Path(filename)
    if not filename or path.is_absolute() or ".." in path.parts:
        fail(f"unsafe Hub checkpoint filename: {filename!r}")
    return filename


def hub_file_url(repo_id: str, revision: str, filename: str) -> str:
    if not repo_id or not revision:
        fail("remote source requires a repository and immutable revision")
    filename = validate_hub_filename(filename)
    return (
        "https://huggingface.co/"
        + urllib.parse.quote(repo_id, safe="/")
        + "/resolve/"
        + urllib.parse.quote(revision, safe="")
        + "/"
        + urllib.parse.quote(filename, safe="/")
    )


def read_hub_range(repo_id: str, revision: str, filename: str,
                   begin: int, end: int, token: str | None) -> tuple[bytes, int]:
    headers = {"Range": f"bytes={begin}-{end}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = hub_file_url(repo_id, revision, filename)
    last_error = None
    for attempt in range(6):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    fail(
                        f"Hub range request for {filename} returned HTTP "
                        f"{response.status}, expected 206"
                    )
                content_range = response.headers.get("Content-Range", "")
                try:
                    total = int(content_range.rsplit("/", 1)[1])
                except (IndexError, ValueError):
                    fail(
                        f"Hub range response for {filename} has no total size"
                    )
                expected = end - begin + 1
                data = response.read(expected + 1)
            if len(data) != expected:
                raise OSError(
                    f"short Hub range read: got {len(data)}, expected {expected}"
                )
            return data, total
        except urllib.error.HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504):
                raise
            last_error = error
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                OSError) as error:
            last_error = error
        if attempt + 1 < 6:
            time.sleep(min(2 ** attempt, 16))
    fail(f"Hub range request for {filename} failed after retries: {last_error}")


def read_hub_safetensors_header(repo_id: str, revision: str, filename: str,
                                token: str | None) -> tuple[dict, int, int]:
    probe_size = 1 << 20
    probe, total = read_hub_range(
        repo_id, revision, filename, 0, probe_size - 1, token
    )
    raw_length = probe[:8]
    length = struct.unpack("<Q", raw_length)[0]
    if length == 0 or length > 1 << 30:
        fail(f"{filename}: invalid remote safetensors header size {length}")
    if 8 + length <= len(probe):
        raw_header = probe[8:8 + length]
    else:
        raw_header, confirmed_total = read_hub_range(
            repo_id, revision, filename, 8, 7 + length, token
        )
        if confirmed_total != total:
            fail(f"Hub object size changed while reading {filename}")
    header, data_offset = decode_safetensors_header(
        raw_length, raw_header, f"{repo_id}@{revision}/{filename}"
    )
    return header, data_offset, total


class SourceDB:
    def __init__(self, root: Path, remote_repo: str | None = None,
                 revision: str | None = None, token: str | None = None):
        self.root = root
        self.remote_repo = remote_repo
        self.revision = revision
        self.token = token
        self.active_paths: dict[str, Path] = {}
        config_path = root / "config.json"
        if not config_path.is_file():
            fail(f"missing {config_path}")
        self.config = json.loads(config_path.read_text())
        if self.config.get("quantization_config") or self.config.get("quantization"):
            fail("the DS4 pack requires the official dense checkpoint; direct quantized/Unsloth input is unsupported")
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            document = json.loads(index_path.read_text())
            self.weight_map = document.get("weight_map")
            if not isinstance(self.weight_map, dict) or not self.weight_map:
                fail(f"{index_path}: missing weight_map")
        elif (root / "model.safetensors").is_file():
            header, _ = read_safetensors_header(root / "model.safetensors")
            self.weight_map = {name: "model.safetensors" for name in header}
        else:
            fail(f"{root}: no safetensors checkpoint found")

        self.tensors: dict[str, dict] = {}
        self.shards: dict[str, dict] = {}
        source_filenames = sorted(set(self.weight_map.values()))
        for ordinal, filename in enumerate(source_filenames, 1):
            validate_hub_filename(filename)
            path = root / filename
            if path.is_file():
                header, data_offset = read_safetensors_header(path)
                file_size = path.stat().st_size
            elif remote_repo:
                if not revision:
                    fail("--remote-repo requires --source-revision")
                if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(source_filenames):
                    print(
                        f"discovering remote source headers "
                        f"{ordinal}/{len(source_filenames)}",
                        flush=True,
                    )
                header, data_offset, file_size = read_hub_safetensors_header(
                    remote_repo, revision, filename, token
                )
            else:
                fail(
                    f"missing source shard {path}; use --remote-repo to "
                    "stream official shards through bounded staging"
                )
            self.shards[filename] = {
                "path": path if path.is_file() else None,
                "size": file_size,
                "data_offset": data_offset,
            }
            for name, info in header.items():
                if self.weight_map.get(name) != filename:
                    fail(f"source index assigns {name} to the wrong shard")
                if name in self.tensors:
                    fail(f"duplicate source tensor {name}")
                begin, end = info["data_offsets"]
                dtype = info["dtype"]
                shape = tuple(int(value) for value in info["shape"])
                if dtype not in DTYPE_TO_NUMPY:
                    fail(f"{name}: unsupported source dtype {dtype}")
                if end - begin != product(shape) * DTYPE_BYTES[dtype]:
                    fail(f"{name}: source payload size does not match dtype/shape")
                self.tensors[name] = {
                    "filename": filename,
                    "path": path if path.is_file() else None,
                    "offset": data_offset + begin,
                    "nbytes": end - begin,
                    "dtype": dtype,
                    "shape": shape,
                }
        if set(self.tensors) != set(self.weight_map):
            missing = sorted(set(self.weight_map) - set(self.tensors))
            fail(f"source checkpoint is incomplete; first missing tensor: {missing[0]}")

    def activate_shard(self, filename: str, path: Path):
        if filename not in self.shards or not path.is_file():
            fail(f"cannot activate unavailable source shard {filename}")
        self.active_paths[filename] = path

    def deactivate_shard(self, filename: str):
        self.active_paths.pop(filename, None)

    def read_metadata_file(self, filename: str,
                           max_bytes: int = 16 << 20) -> bytes:
        validate_hub_filename(filename)
        path = self.root / filename
        if path.is_file():
            data = path.read_bytes()
        elif self.remote_repo and self.revision:
            _, total = read_hub_range(
                self.remote_repo, self.revision, filename, 0, 0, self.token
            )
            if total <= 0 or total > max_bytes:
                fail(
                    f"remote metadata {filename} is {total} bytes, outside "
                    f"the allowed range"
                )
            data, confirmed = read_hub_range(
                self.remote_repo, self.revision, filename,
                0, total - 1, self.token
            )
            if confirmed != total:
                fail(f"Hub object size changed while reading {filename}")
        else:
            fail(f"missing source metadata {path}")
        if not data or len(data) > max_bytes:
            fail(f"source metadata {filename} is empty or too large")
        return data

    def read(self, name: str) -> np.ndarray:
        try:
            info = self.tensors[name]
        except KeyError:
            fail(f"source tensor not found: {name}")
        path = info["path"] or self.active_paths.get(info["filename"])
        if not path:
            fail(
                f"source shard {info['filename']} is not materialized for "
                f"tensor {name}"
            )
        with path.open("rb") as fp:
            fp.seek(info["offset"])
            raw = fp.read(info["nbytes"])
        if len(raw) != info["nbytes"]:
            fail(f"short payload read for {name}")
        return np.frombuffer(raw, dtype=DTYPE_TO_NUMPY[info["dtype"]]).reshape(info["shape"])

    @contextmanager
    def materialize(self, filename: str, staging_root: Path):
        shard = self.shards.get(filename)
        if shard is None:
            fail(f"unknown source shard {filename}")
        if shard["path"]:
            self.activate_shard(filename, shard["path"])
            try:
                yield shard["path"]
            finally:
                self.deactivate_shard(filename)
            return
        if not self.remote_repo or not self.revision:
            fail(f"source shard {filename} is unavailable")
        staging_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix="shard-", dir=staging_root))
        try:
            downloaded = Path(hf_hub_download(
                repo_id=self.remote_repo,
                filename=filename,
                revision=self.revision,
                token=self.token,
                cache_dir=temporary / "cache",
            ))
            if downloaded.stat().st_size != shard["size"]:
                fail(
                    f"downloaded {filename} has {downloaded.stat().st_size} "
                    f"bytes, expected {shard['size']}"
                )
            self.activate_shard(filename, downloaded)
            try:
                yield downloaded
            finally:
                self.deactivate_shard(filename)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


def text_config(document: dict) -> dict:
    candidate = document.get("text_config")
    return candidate if isinstance(candidate, dict) else document


def config_value(config: dict, *names, default=None):
    for name in names:
        if name in config:
            return config[name]
    return default


def validate_source_config(document: dict):
    config = text_config(document)
    root_type = document.get("model_type")
    model_type = config.get("model_type", root_type)
    if model_type not in ("qwen4_exp", "qwen4_exp_text"):
        fail(f"source model_type is {model_type!r}, expected qwen4_exp")
    checks = {
        "num_hidden_layers": CONFIG["layers"],
        "hidden_size": CONFIG["hidden"],
        "vocab_size": CONFIG["vocab"],
        "max_position_embeddings": CONFIG["context"],
        "num_attention_heads": CONFIG["attn_heads"],
        "num_key_value_heads": CONFIG["kv_heads"],
        "head_dim": CONFIG["head_dim"],
        "full_attention_interval": CONFIG["full_attention_interval"],
        "hc_count": CONFIG["hc_count"],
        "hc_lowrank": CONFIG["hc_lowrank"],
        "linear_num_key_heads": CONFIG["linear_key_heads"],
        "linear_num_value_heads": CONFIG["linear_value_heads"],
        "linear_key_head_dim": CONFIG["linear_head_dim"],
        "linear_value_head_dim": CONFIG["linear_head_dim"],
        "linear_conv_kernel_dim": CONFIG["linear_conv"],
        "indexer_n_heads": CONFIG["qsa_heads"],
        "indexer_kv_heads": CONFIG["qsa_kv_heads"],
        "indexer_head_dim": CONFIG["qsa_dim"],
        "indexer_budget": CONFIG["qsa_topk"],
        "indexer_compress_ratio": CONFIG["qsa_ratio"],
        "num_experts": CONFIG["experts"],
        "num_experts_per_tok": CONFIG["experts_used"],
        "moe_intermediate_size": CONFIG["expert_ff"],
        "shared_expert_intermediate_size": CONFIG["shared_ff"],
        "ple_embed_dim": CONFIG["ple_dim"],
        "ple_conv_kernel_size": CONFIG["ple_conv"],
        "ngram_size": CONFIG["ngram_size"],
        "heads_per_ngram": CONFIG["heads_per_ngram"],
        "ngram_vocab_size_base": CONFIG["ngram_vocab_base"],
        "make_ngram_vocab_size_divisible_by": CONFIG["ngram_vocab_divisor"],
        "split_ngram_parts": CONFIG["split_ngram_parts"],
        "output_gate_type": CONFIG["output_gate_type"],
        "eos_token_id": CONFIG["ngram_eos"],
    }
    aliases = {
        "num_experts": ("num_experts", "num_local_experts"),
        "num_experts_per_tok": ("num_experts_per_tok", "num_experts_per_token"),
    }
    for name, expected in checks.items():
        actual = config_value(config, *aliases.get(name, (name,)))
        if actual != expected:
            fail(f"source config {name}={actual!r}, expected {expected}")
    if tuple(config.get("ple_layer_ids", ())) != CONFIG["ple_layer_ids"]:
        fail(
            f"source config ple_layer_ids={config.get('ple_layer_ids')!r}, "
            f"expected {list(CONFIG['ple_layer_ids'])}"
        )
    if config.get("seed", CONFIG["ngram_seed"]) != CONFIG["ngram_seed"]:
        fail(
            f"source config seed={config.get('seed')!r}, "
            f"expected {CONFIG['ngram_seed']}"
        )
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != CONFIG["layers"]:
        fail("source config has missing or invalid layer_types")
    for layer, layer_type in enumerate(layer_types):
        expected = "full_attention" if (layer + 1) % CONFIG["full_attention_interval"] == 0 else "linear_attention"
        if layer_type not in (expected, "qwen_sparse_attention" if expected == "full_attention" else expected):
            fail(
                f"source layer {layer} type is {layer_type!r}, expected {expected!r}"
            )
    rope = config.get("rope_parameters", config)
    rope_dim = config_value(config, "partial_rotary_factor")
    if rope_dim is not None:
        rope_dim = round(float(rope_dim) * CONFIG["head_dim"])
    else:
        rope_dim = config_value(config, "rotary_dim", "rope_dim", default=CONFIG["rope_dim"])
    if rope_dim != CONFIG["rope_dim"]:
        fail(f"source partial RoPE dimension is {rope_dim}, expected {CONFIG['rope_dim']}")
    theta = config_value(rope, "rope_theta", default=config.get("rope_theta"))
    if theta is not None and float(theta) != CONFIG["rope_theta"]:
        fail(f"source RoPE theta is {theta}, expected {CONFIG['rope_theta']}")


def renamed(name: str) -> str:
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model."):]
    if name.startswith("mtp."):
        return "language_model.mtp." + name[len("mtp."):]
    if name == "lm_head.weight":
        return "language_model.lm_head.weight"
    return name


def needs_norm_fold(name: str) -> bool:
    return name.endswith(NORM_FOLD_SUFFIXES)


def needs_hc_divisor_fold(name: str) -> bool:
    return name.endswith((
        ".input_mix_weight_down.weight",
        ".block_inject_weight.weight",
    ))


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype="<f4").view("<u4")
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2")


def bf16_to_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype="<u2").astype("<u4") << 16).view("<f4")


GGUF_VALUE_UINT8 = 0
GGUF_VALUE_INT8 = 1
GGUF_VALUE_UINT16 = 2
GGUF_VALUE_INT16 = 3
GGUF_VALUE_UINT32 = 4
GGUF_VALUE_INT32 = 5
GGUF_VALUE_FLOAT32 = 6
GGUF_VALUE_BOOL = 7
GGUF_VALUE_STRING = 8
GGUF_VALUE_ARRAY = 9
GGUF_VALUE_UINT64 = 10
GGUF_VALUE_INT64 = 11
GGUF_VALUE_FLOAT64 = 12
GGUF_SCALAR_FORMATS = {
    GGUF_VALUE_UINT8: "<B",
    GGUF_VALUE_INT8: "<b",
    GGUF_VALUE_UINT16: "<H",
    GGUF_VALUE_INT16: "<h",
    GGUF_VALUE_UINT32: "<I",
    GGUF_VALUE_INT32: "<i",
    GGUF_VALUE_FLOAT32: "<f",
    GGUF_VALUE_BOOL: "<?",
    GGUF_VALUE_UINT64: "<Q",
    GGUF_VALUE_INT64: "<q",
    GGUF_VALUE_FLOAT64: "<d",
}


def read_exact(fp, length: int, label: str) -> bytes:
    data = fp.read(length)
    if len(data) != length:
        fail(f"short read for {label}")
    return data


def read_u32(fp, label: str) -> int:
    return struct.unpack("<I", read_exact(fp, 4, label))[0]


def read_u64(fp, label: str) -> int:
    return struct.unpack("<Q", read_exact(fp, 8, label))[0]


def read_gguf_string(fp, label: str) -> str:
    length = read_u64(fp, f"{label} length")
    if length > 1 << 30:
        fail(f"unreasonable {label} length {length}")
    try:
        return read_exact(fp, length, label).decode("utf-8")
    except UnicodeDecodeError as error:
        fail(f"{label} is not valid UTF-8: {error}")


def read_gguf_value(fp, value_type: int, label: str):
    if value_type == GGUF_VALUE_STRING:
        return read_gguf_string(fp, label)
    if value_type == GGUF_VALUE_ARRAY:
        element_type = read_u32(fp, f"{label} array type")
        count = read_u64(fp, f"{label} array count")
        if count > 1 << 32:
            fail(f"unreasonable {label} array count {count}")
        if element_type == GGUF_VALUE_STRING:
            return [
                read_gguf_string(fp, f"{label} array item")
                for _ in range(count)
            ]
        fmt = GGUF_SCALAR_FORMATS.get(element_type)
        if fmt is None:
            fail(f"unsupported {label} GGUF array type {element_type}")
        size = struct.calcsize(fmt)
        raw = read_exact(fp, count * size, label)
        return list(struct.unpack(f"<{count}{fmt[-1]}", raw))
    fmt = GGUF_SCALAR_FORMATS.get(value_type)
    if fmt is None:
        fail(f"unsupported {label} GGUF value type {value_type}")
    return struct.unpack(fmt, read_exact(fp, struct.calcsize(fmt), label))[0]


@dataclasses.dataclass(frozen=True)
class ImatrixTensor:
    shape: tuple[int, ...]
    qtype: int
    offset: int
    nbytes: int


class QwenGGUFImatrix:
    """Validated, mmap-backed Qwen expert importance statistics.

    llama.cpp's GGUF imatrix stores raw input-square sums and a separate
    count for every routed expert.  Keeping the 553 MB file mapped avoids a
    second resident copy while source shards are being quantized.
    """

    def __init__(self, path: Path, expected_sha256: str | None = None,
                 repository: str = Q2_IMATRIX_REPOSITORY,
                 revision: str = Q2_IMATRIX_REVISION):
        self.path = Path(path)
        if not self.path.is_file():
            fail(f"Q2 imatrix not found: {self.path}")
        self.sha256 = sha256_file(self.path)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            fail(
                f"Q2 imatrix checksum mismatch: expected {expected_sha256}, "
                f"got {self.sha256}"
            )
        self.repository = repository
        self.revision = revision
        self.metadata: dict[str, object] = {}
        self.tensors: dict[str, ImatrixTensor] = {}
        self.entries: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.fallback_entries: list[dict[str, int | str]] = []
        self._mapping = np.memmap(self.path, dtype=np.uint8, mode="r")
        self._parse()
        self._validate_qwen_entries()

    def _parse(self):
        file_size = self.path.stat().st_size
        with self.path.open("rb") as fp:
            if read_exact(fp, 4, "imatrix GGUF magic") != b"GGUF":
                fail(f"{self.path}: Q2 imatrix is not a GGUF file")
            version = read_u32(fp, "imatrix GGUF version")
            if version != GGUF_VERSION:
                fail(
                    f"{self.path}: Q2 imatrix uses GGUF v{version}, "
                    f"expected v{GGUF_VERSION}"
                )
            tensor_count = read_u64(fp, "imatrix tensor count")
            metadata_count = read_u64(fp, "imatrix metadata count")
            if tensor_count == 0 or tensor_count > 1 << 20:
                fail(f"{self.path}: invalid imatrix tensor count {tensor_count}")
            if metadata_count == 0 or metadata_count > 1 << 20:
                fail(
                    f"{self.path}: invalid imatrix metadata count "
                    f"{metadata_count}"
                )
            for _ in range(metadata_count):
                key = read_gguf_string(fp, "imatrix metadata key")
                if key in self.metadata:
                    fail(f"{self.path}: duplicate imatrix metadata key {key}")
                value_type = read_u32(fp, f"imatrix metadata type for {key}")
                self.metadata[key] = read_gguf_value(
                    fp, value_type, f"imatrix metadata {key}"
                )
            directories = []
            for _ in range(tensor_count):
                name = read_gguf_string(fp, "imatrix tensor name")
                if name in self.tensors or any(row[0] == name for row in directories):
                    fail(f"{self.path}: duplicate imatrix tensor {name}")
                dimensions = read_u32(fp, f"imatrix dimensions for {name}")
                if dimensions == 0 or dimensions > 4:
                    fail(
                        f"{self.path}: invalid imatrix rank {dimensions} "
                        f"for {name}"
                    )
                shape = tuple(
                    read_u64(fp, f"imatrix shape for {name}")
                    for _ in range(dimensions)
                )
                if any(value == 0 or value > 1 << 32 for value in shape):
                    fail(f"{self.path}: invalid imatrix shape {shape} for {name}")
                qtype = read_u32(fp, f"imatrix qtype for {name}")
                offset = read_u64(fp, f"imatrix offset for {name}")
                directories.append((name, shape, qtype, offset))
            data_offset = align(fp.tell())
        for name, shape, qtype, offset in directories:
            if qtype != QTYPE_F32:
                fail(f"{self.path}: imatrix tensor {name} is not F32")
            nbytes = product(shape) * 4
            absolute = data_offset + offset
            if absolute < data_offset or absolute + nbytes > file_size:
                fail(f"{self.path}: imatrix tensor {name} is out of bounds")
            self.tensors[name] = ImatrixTensor(
                shape=shape, qtype=qtype, offset=absolute, nbytes=nbytes
            )
        if self.metadata.get("general.type") != "imatrix":
            fail(f"{self.path}: GGUF general.type is not imatrix")
        datasets = self.metadata.get("imatrix.datasets")
        if (not isinstance(datasets, list) or not datasets or
                any(not isinstance(item, str) or not item for item in datasets)):
            fail(f"{self.path}: missing or invalid imatrix.datasets metadata")
        for key in ("imatrix.chunk_count", "imatrix.chunk_size"):
            value = self.metadata.get(key)
            if not isinstance(value, int) or value <= 0:
                fail(f"{self.path}: missing or invalid {key} metadata")

    def _array(self, tensor: ImatrixTensor) -> np.ndarray:
        return np.frombuffer(
            self._mapping, dtype="<f4", count=product(tensor.shape),
            offset=tensor.offset,
        )

    def _validate_qwen_entries(self):
        for layer in range(CONFIG["layers"]):
            for part, width in (
                    ("gate", CONFIG["hidden"]),
                    ("up", CONFIG["hidden"]),
                    ("down", CONFIG["expert_ff"])):
                base = f"blk.{layer}.ffn_{part}_exps.weight"
                sums_name = base + ".in_sum2"
                counts_name = base + ".counts"
                sums_tensor = self.tensors.get(sums_name)
                counts_tensor = self.tensors.get(counts_name)
                if sums_tensor is None or counts_tensor is None:
                    missing = sums_name if sums_tensor is None else counts_name
                    fail(f"{self.path}: required Q2 imatrix tensor is missing: {missing}")
                expected_sums = (width, CONFIG["experts"])
                expected_counts = (1, CONFIG["experts"])
                if sums_tensor.shape != expected_sums:
                    fail(
                        f"{self.path}: {sums_name} has shape "
                        f"{sums_tensor.shape}, expected {expected_sums}"
                    )
                if counts_tensor.shape != expected_counts:
                    fail(
                        f"{self.path}: {counts_name} has shape "
                        f"{counts_tensor.shape}, expected {expected_counts}"
                    )
                sums = self._array(sums_tensor).reshape(CONFIG["experts"], width)
                counts = self._array(counts_tensor)
                if not np.all(np.isfinite(sums)) or np.any(sums < 0.0):
                    fail(f"{self.path}: {sums_name} has nonfinite/negative statistics")
                if (not np.all(np.isfinite(counts)) or
                        np.any(counts < 0.0) or
                        not np.all(counts == np.rint(counts))):
                    fail(f"{self.path}: {counts_name} has invalid counts")
                self.entries[base] = (sums, counts)
                self.fallback_entries.extend(
                    {
                        "tensor": base,
                        "expert": int(expert),
                        "part": part,
                    }
                    for expert in np.flatnonzero(counts == 0.0)
                )
        if self.fallback_entries:
            gate_up = sum(
                entry["part"] in ("gate", "up")
                for entry in self.fallback_entries
            )
            down = self.fallback_count - gate_up
            print(
                "qwen4-pack: Q2 imatrix has "
                f"{self.fallback_count} zero-count expert entries "
                f"(gate/up={gate_up}, down={down}); using per-expert "
                "weight-energy fallback",
                file=sys.stderr,
                flush=True,
            )

    @property
    def fallback_count(self) -> int:
        return len(self.fallback_entries)

    def expert_weights(self, name: str, expert: int,
                       values: np.ndarray) -> np.ndarray:
        try:
            sums, counts = self.entries[name]
        except KeyError:
            fail(f"required Q2 imatrix entry is missing: {name}")
        if expert < 0 or expert >= CONFIG["experts"]:
            fail(f"invalid Q2 routed expert index {expert}")
        count = counts[expert]
        if count > 0.0:
            return np.ascontiguousarray(sums[expert] / count, dtype="<f4")
        source = bf16_to_f32(values)
        weights = np.square(source, dtype=np.float32).sum(
            axis=0, dtype=np.float32
        )
        if not np.all(np.isfinite(weights)):
            fail(f"{name} expert {expert}: nonfinite weight-energy fallback")
        return np.ascontiguousarray(weights, dtype="<f4")

    def provenance(self) -> dict:
        return {
            "format": "gguf-imatrix-v3",
            "repository": self.repository,
            "revision": self.revision,
            "sha256": self.sha256,
            "datasets": self.metadata["imatrix.datasets"],
            "chunk_count": self.metadata["imatrix.chunk_count"],
            "chunk_size": self.metadata["imatrix.chunk_size"],
            "zero_count_fallback": {
                "method": "per-expert-input-column-weight-energy",
                "count": self.fallback_count,
                "gate_up_count": sum(
                    entry["part"] in ("gate", "up")
                    for entry in self.fallback_entries
                ),
                "down_count": sum(
                    entry["part"] == "down"
                    for entry in self.fallback_entries
                ),
                "entries": self.fallback_entries,
            },
        }

    def close(self):
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mmap = getattr(mapping, "_mmap", None)
            if mmap is not None:
                mmap.close()
            self._mapping = None


class GGMLQuantizer:
    """Thin binding to the standard GGML block packers shipped with DS4."""

    def __init__(self, library_path: Path | str):
        library_path = Path(library_path)
        if not library_path.is_file():
            fail(
                f"quantizer library not found: {library_path}; "
                "run make -C gguf-tools quants-shared"
            )
        self.library_path = library_path.resolve()
        self.lib = ctypes.CDLL(os.fspath(self.library_path))
        self.lib.ds4q_can_quantize.argtypes = [ctypes.c_int]
        self.lib.ds4q_can_quantize.restype = ctypes.c_bool
        self.lib.ds4q_block_size.argtypes = [ctypes.c_int]
        self.lib.ds4q_block_size.restype = ctypes.c_int64
        self.lib.ds4q_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.lib.ds4q_row_size.restype = ctypes.c_size_t
        self.lib.ds4q_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ds4q_quantize_init.restype = None
        self.lib.ds4q_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.ds4q_quantize_chunk.restype = ctypes.c_size_t

    def encode(self, values: np.ndarray, qtype: str,
               imatrix: np.ndarray | None = None) -> bytes:
        if qtype not in QTYPE_LAYOUT:
            fail(f"unsupported GGML quantization target {qtype}")
        source = np.asarray(values)
        if source.ndim == 0:
            fail("GGML block quantization requires at least one dimension")
        qtype_id = DTYPE_TO_QTYPE[qtype]
        block_size = self.lib.ds4q_block_size(qtype_id)
        if (not self.lib.ds4q_can_quantize(qtype_id) or block_size <= 0 or
                source.shape[-1] <= 0 or source.shape[-1] % block_size):
            fail(
                f"{qtype} width {source.shape[-1]} is not divisible by "
                f"GGML block size {block_size}"
            )
        if source.dtype == np.dtype("<u2"):
            source = bf16_to_f32(source)
        else:
            source = np.asarray(source, dtype="<f4")
        if not np.all(np.isfinite(source)):
            fail("GGML block quantization input contains NaN or infinity")
        source = np.ascontiguousarray(source, dtype="<f4")
        nrows = product(source.shape[:-1])
        ncols = source.shape[-1]
        if qtype == "IQ2_XXS" and imatrix is None:
            fail("IQ2_XXS quantization requires an importance matrix")
        if imatrix is not None:
            imatrix = np.ascontiguousarray(imatrix, dtype="<f4")
            if imatrix.ndim != 1 or imatrix.size != ncols:
                fail(
                    f"{qtype} imatrix width {imatrix.size} does not match "
                    f"tensor width {ncols}"
                )
            if not np.all(np.isfinite(imatrix)) or np.any(imatrix < 0.0):
                fail(f"{qtype} imatrix contains nonfinite/negative weights")
            imatrix_ptr = imatrix.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            )
        else:
            imatrix_ptr = None
        row_size = self.lib.ds4q_row_size(qtype_id, ncols)
        if row_size <= 0:
            fail(f"invalid {qtype} row geometry {ncols}")
        output = np.empty(nrows * row_size, dtype=np.uint8)
        self.lib.ds4q_quantize_init(qtype_id)
        written = self.lib.ds4q_quantize_chunk(
            qtype_id,
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.c_void_p),
            0,
            nrows,
            ncols,
            imatrix_ptr,
        )
        if written != output.nbytes:
            fail(
                f"native {qtype} quantization wrote {written} bytes, "
                f"expected {output.nbytes}"
            )
        return output.tobytes()


def default_quantizer_library() -> Path:
    suffix = "dylib" if sys.platform == "darwin" else "so"
    return Path(__file__).with_name(f"libds4quants.{suffix}")


@dataclasses.dataclass
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]  # physical row-major/safetensors order
    role: str
    source: str
    logical_shape: tuple[int, ...] | None = None
    padding: dict | None = None
    offset: int = 0

    @property
    def gguf_shape(self):
        return tuple(reversed(self.shape))

    @property
    def qtype(self):
        return DTYPE_TO_QTYPE[self.dtype]

    @property
    def nbytes(self):
        layout = QTYPE_LAYOUT.get(self.dtype)
        if layout:
            block_size, block_bytes = layout
            if not self.shape or self.shape[-1] % block_size:
                fail(
                    f"{self.name}: physical width is incompatible with "
                    f"{self.dtype}"
                )
            return (
                product(self.shape[:-1]) *
                (self.shape[-1] // block_size) * block_bytes
            )
        return product(self.shape) * DTYPE_BYTES[self.dtype]


@dataclasses.dataclass
class Action:
    source: str
    kind: str
    role: str
    specs: list[TensorSpec]
    qtype: str | None = None
    scale: float = 1.0
    pad_last_to: int = 0
    imatrix_names: tuple[str, ...] = ()
    mtp_fallback: bool = False


def quant_specs(base: str, shape: tuple[int, ...], role: str,
                source: str, qtype: str,
                logical_shape: tuple[int, ...] | None = None) -> list[TensorSpec]:
    if qtype not in QTYPE_LAYOUT:
        fail(f"unsupported GGML tensor type {qtype}")
    padding = None
    if logical_shape is not None and tuple(logical_shape) != tuple(shape):
        if logical_shape[:-1] != shape[:-1] or logical_shape[-1] > shape[-1]:
            fail(f"{base}: unsupported physical padding geometry")
        padding = {
            "axis": -1,
            "logical": logical_shape[-1],
            "physical": shape[-1],
            "fill": 0,
        }
    return [
        TensorSpec(
            base, qtype, shape, role, source,
            logical_shape=logical_shape or shape,
            padding=padding,
        ),
    ]


def layer_of(name: str) -> int | None:
    marker = "language_model.model.layers."
    if not name.startswith(marker):
        return None
    tail = name[len(marker):]
    digits = tail.split(".", 1)[0]
    return int(digits) if digits.isdigit() else None


def artifact_for(name: str) -> str:
    if name.startswith("model.visual."):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    if NGRAM_MARK in name:
        return "ple"
    return "base"


def base_locality_group(name: str) -> int:
    """Keep the former four-shard payload locality inside one base GGUF."""
    target = renamed(name)
    layer = layer_of(target)
    if layer is not None:
        return min(layer // 12, 3)
    if target.startswith("language_model.lm_head"):
        return 3
    return 0


def is_bf16_control(target: str) -> bool:
    return (
        target.endswith("embed_tokens.weight") or
        target.endswith(".mlp.gate.weight") or
        target.endswith(".mlp.shared_expert_gate.weight") or
        "hyper_connection" in target or
        target.endswith(".linear_attn.in_proj_a.weight") or
        target.endswith(".linear_attn.in_proj_b.weight")
    )


def q8_role(target: str) -> str:
    if target.endswith("lm_head.weight"):
        return "output"
    if ".mlp.shared_expert." in target:
        return "shared_expert"
    if ".linear_attn." in target:
        return "gdn"
    if ".self_attn." in target:
        return "qsa"
    if ".ple." in target:
        return "ple_projection"
    if target.startswith("model.visual."):
        return "vision_projection"
    return "dense"


def make_action(db: SourceDB, source: str,
                profile: str | PackProfile = "q4") -> Action | None:
    profile = pack_profile(profile)
    info = db.tensors[source]
    shape = info["shape"]
    dtype = info["dtype"]
    target = renamed(source)
    scale = 1.0 / CONFIG["hc_count"] if needs_hc_divisor_fold(target) else 1.0
    if NGRAM_MARK in source:
        return None
    if source.endswith(PLE_AUX_SUFFIXES):
        return None
    # This is a lookup table, not a dense projection. Keep the small visual
    # position grid in BF16 because the patch kernel gathers and interpolates
    # four rows directly.
    if target == "model.visual.pos_embed.weight":
        if dtype != "BF16" or shape != (2304, 1152):
            fail(f"{source}: invalid Qwen visual position embedding")
        return Action(
            source,
            "copy",
            "vision_embedding",
            [TensorSpec(target, dtype, shape, "vision_embedding", source)],
        )
    # Q8_0 has 32-value blocks. The official visual FC2 consumes 4304 features,
    # so append 16 zero-valued columns and advertise the physical width.
    if (target.startswith("model.visual.blocks.") and
            target.endswith(".mlp.linear_fc2.weight")):
        if dtype != "BF16" or shape != (1152, 4304):
            fail(f"{source}: invalid Qwen visual FC2 geometry")
        padded_shape = (1152, CONFIG["vision_fc2_physical_input"])
        return Action(
            source,
            "quant",
            "vision_projection",
            quant_specs(
                target, padded_shape, "vision_projection", source, "Q8_0",
                logical_shape=shape,
            ),
            "Q8_0",
            pad_last_to=CONFIG["vision_fc2_physical_input"],
        )
    if target.endswith(".mlp.experts.gate_up_proj"):
        expected = (
            CONFIG["experts"], 2 * CONFIG["expert_ff"], CONFIG["hidden"]
        )
        if dtype != "BF16" or shape != expected:
            fail(f"{source}: invalid routed gate/up tensor")
        half_shape = (shape[0], shape[1] // 2, shape[2])
        prefix = target[:-len("experts.gate_up_proj")] + "switch_mlp."
        specs = quant_specs(prefix + "gate_proj.weight", half_shape,
                            "routed_expert", source,
                            profile.gate_up_qtype)
        specs += quant_specs(prefix + "up_proj.weight", half_shape,
                             "routed_expert", source,
                             profile.gate_up_qtype)
        imatrix_names = ()
        mtp_fallback = False
        if profile.name == "q2":
            layer = layer_of(renamed(source))
            if layer is None:
                if ".mtp." in target and MTP_IMATRIX_MODE == "weight-energy":
                    mtp_fallback = True
                else:
                    fail(f"{source}: Q2 routed gate/up has no imatrix layer")
            else:
                imatrix_names = (
                    f"blk.{layer}.ffn_gate_exps.weight",
                    f"blk.{layer}.ffn_up_exps.weight",
                )
        return Action(
            source, "split_gate_up", "routed_expert", specs,
            profile.gate_up_qtype,
            imatrix_names=imatrix_names,
            mtp_fallback=mtp_fallback,
        )
    if target.endswith(".mlp.experts.down_proj"):
        expected = (
            CONFIG["experts"], CONFIG["hidden"], CONFIG["expert_ff"]
        )
        if dtype != "BF16" or shape != expected:
            fail(f"{source}: invalid routed down tensor")
        base = target[:-len("experts.down_proj")] + "switch_mlp.down_proj.weight"
        physical_shape = shape[:-1] + (CONFIG["routed_down_physical_input"],)
        kind = "quant"
        imatrix_names = ()
        mtp_fallback = False
        if profile.name == "q2":
            layer = layer_of(renamed(source))
            if layer is None:
                if ".mtp." in target and MTP_IMATRIX_MODE == "weight-energy":
                    mtp_fallback = True
                    kind = "quant_experts"
                else:
                    fail(f"{source}: Q2 routed down has no imatrix layer")
            else:
                kind = "quant_experts"
                imatrix_names = (f"blk.{layer}.ffn_down_exps.weight",)
        return Action(
            source, kind, "routed_expert",
            quant_specs(
                base, physical_shape, "routed_expert", source,
                profile.down_qtype,
                logical_shape=shape,
            ),
            profile.down_qtype,
            pad_last_to=CONFIG["routed_down_physical_input"],
            imatrix_names=imatrix_names,
            mtp_fallback=mtp_fallback,
        )

    if dtype == "BF16" and len(shape) == 2 and not is_bf16_control(target):
        role = q8_role(target)
        if shape[-1] % QTYPE_LAYOUT["Q8_0"][0]:
            fail(
                f"{source}: Q8_0 projection width {shape[-1]} requires "
                "an explicit physical-padding rule"
            )
        return Action(
            source, "quant", role,
            quant_specs(target, shape, role, source, "Q8_0"),
            "Q8_0", scale=scale,
        )
    if dtype not in DTYPE_TO_QTYPE:
        fail(f"{source}: unsupported pass-through dtype {dtype}")
    role = "embedding" if target.endswith("embed_tokens.weight") else "control"
    kind = "conv" if target.endswith("conv1d.weight") and len(shape) == 3 else "copy"
    if kind == "conv":
        shape = (shape[0], shape[2], shape[1])
    if needs_norm_fold(target):
        if dtype != "BF16":
            fail(f"{source}: folded RMSNorm is not BF16")
        kind = "norm_fold"
    spec = TensorSpec(target, dtype, shape, role, source, logical_shape=shape)
    return Action(source, kind, role, [spec], scale=scale)


def weight_energy_weights(source_values: np.ndarray) -> np.ndarray:
    source = bf16_to_f32(source_values)
    weights = np.square(source, dtype=np.float32).sum(
        axis=0, dtype=np.float32
    )
    if not np.all(np.isfinite(weights)):
        fail('MTP expert: nonfinite weight-energy fallback')
    return np.ascontiguousarray(weights, dtype='<f4')


def encode_fallback_experts(values: np.ndarray, qtype: str,
                            quantizer: GGMLQuantizer,
                            threads: int,
                            source_name: str,
                            pad_last_to: int = 0) -> bytes:
    # Deterministic per-expert weight-energy importance for MTP experts.
    # The calibrated GGUF imatrix covers only the 48 base layers; the MTP
    # routed experts use the same fallback selection the imatrix applies
    # to its zero-count experts, recorded as such in the manifest.
    if (values.ndim != 3 or
            values.shape[0] != CONFIG['experts'] or threads < 1):
        fail(f'{source_name}: invalid fallback expert geometry')

    def convert(expert: int) -> bytes:
        source_values = np.ascontiguousarray(values[expert])
        weights = weight_energy_weights(source_values)
        expert_values = source_values
        if pad_last_to:
            if pad_last_to < source_values.shape[-1]:
                fail(f'{source_name}: physical padding shrinks the source')
            padding = pad_last_to - source_values.shape[-1]
            expert_values = np.pad(
                source_values, ((0, 0), (0, padding)), mode='constant'
            )
            weights = np.pad(weights, (0, padding), mode='constant')
        return quantizer.encode(expert_values, qtype, weights)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        blocks = list(pool.map(convert, range(values.shape[0])))
    return b''.join(blocks)


def encode_weighted_experts(values: np.ndarray, qtype: str,
                            imatrix: QwenGGUFImatrix,
                            imatrix_name: str,
                            quantizer: GGMLQuantizer,
                            threads: int,
                            pad_last_to: int = 0) -> bytes:
    if (values.ndim != 3 or
            values.shape[0] != CONFIG["experts"] or threads < 1):
        fail(f"{imatrix_name}: invalid weighted expert conversion geometry")

    def convert(expert: int) -> bytes:
        source_values = np.ascontiguousarray(values[expert])
        weights = imatrix.expert_weights(
            imatrix_name, expert, source_values
        )
        expert_values = source_values
        if pad_last_to:
            if pad_last_to < source_values.shape[-1]:
                fail(f"{imatrix_name}: physical padding shrinks the source")
            padding = pad_last_to - source_values.shape[-1]
            expert_values = np.pad(
                source_values, ((0, 0), (0, padding)), mode="constant"
            )
            weights = np.pad(weights, (0, padding), mode="constant")
        return quantizer.encode(expert_values, qtype, weights)

    encoded = bytearray()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=threads) as executor:
        window = max(threads * 2, 1)
        for start in range(0, values.shape[0], window):
            futures = [
                executor.submit(convert, expert)
                for expert in range(
                    start, min(start + window, values.shape[0])
                )
            ]
            for future in futures:
                encoded += future.result()
    return bytes(encoded)


def encode_action(action: Action, db: SourceDB,
                  quantizer: GGMLQuantizer,
                  imatrix: QwenGGUFImatrix | None = None,
                  threads: int = 1) -> list[bytes]:
    value = db.read(action.source)
    if action.scale != 1.0:
        if db.tensors[action.source]["dtype"] != "BF16":
            fail(f"{action.source}: hyper-connection divisor fold requires BF16")
        value = f32_to_bf16(
            bf16_to_f32(value) * np.float32(action.scale)
        )
    if action.kind == "copy":
        return [np.ascontiguousarray(value).tobytes()]
    if action.kind == "conv":
        return [np.ascontiguousarray(np.swapaxes(value, 1, 2)).tobytes()]
    if action.kind == "norm_fold":
        return [f32_to_bf16(bf16_to_f32(value) + 1.0).tobytes()]
    if action.kind == "quant":
        if action.pad_last_to:
            if action.pad_last_to < value.shape[-1]:
                fail(f"{action.source}: physical padding shrinks the source")
            value = np.pad(
                value,
                [(0, 0)] * (value.ndim - 1) +
                [(0, action.pad_last_to - value.shape[-1])],
                mode="constant",
            )
        return [quantizer.encode(value, action.qtype)]
    if action.kind == "quant_experts":
        if action.mtp_fallback:
            return [encode_fallback_experts(
                value, action.qtype, quantizer, threads,
                action.source, action.pad_last_to,
            )]
        if imatrix is None or len(action.imatrix_names) != 1:
            fail(
                f"{action.source}: weighted Q2 expert conversion requires "
                "the validated Qwen GGUF imatrix"
            )
        return [encode_weighted_experts(
            value, action.qtype, imatrix, action.imatrix_names[0],
            quantizer, threads, action.pad_last_to,
        )]
    if action.kind == "split_gate_up":
        half = value.shape[1] // 2
        result = []
        for part_index, part in enumerate((value[:, :half], value[:, half:])):
            if action.qtype != "IQ2_XXS":
                result.append(quantizer.encode(
                    np.ascontiguousarray(part), action.qtype
                ))
                continue
            if action.mtp_fallback:
                result.append(encode_fallback_experts(
                    part, action.qtype, quantizer, threads, action.source,
                ))
                continue
            if imatrix is None or len(action.imatrix_names) != 2:
                fail(
                    f"{action.source}: Q2 routed gate/up conversion requires "
                    "the validated Qwen GGUF imatrix"
                )
            imatrix_name = action.imatrix_names[part_index]
            result.append(encode_weighted_experts(
                part, action.qtype, imatrix, imatrix_name,
                quantizer, threads,
            ))
        return result
    fail(f"unknown conversion action {action.kind}")


def tensor_header(spec: TensorSpec) -> bytes:
    return (
        pack_string(spec.name)
        + struct.pack("<I", len(spec.gguf_shape))
        + struct.pack(f"<{len(spec.gguf_shape)}Q", *spec.gguf_shape)
        + struct.pack("<IQ", spec.qtype, spec.offset)
    )


def tensor_record(spec: TensorSpec, artifact: str, sha256: str) -> dict:
    record = {
        "artifact": artifact,
        "source": spec.source,
        "qtype": spec.dtype,
        "logical_shape": list(spec.logical_shape or spec.shape),
        "physical_shape": list(spec.shape),
        "role": spec.role,
        "sha256": sha256,
    }
    if spec.padding is not None:
        record["padding"] = spec.padding
    return record


def common_metadata(pack_id: str, source_revision: str,
                    artifact: str,
                    profile: str | PackProfile = "q4") -> list[bytes]:
    profile = pack_profile(profile)
    c = CONFIG
    records = [
        kv_string("general.architecture", ARCH if artifact == "base" else f"{ARCH}-{artifact}"),
        kv_string(
            "general.name",
            f"Qwen3.8-Flash-Next DS4 {profile.name.upper()} {artifact}",
        ),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string("general.source.revision", source_revision),
        kv_u32("ds4.pack.version", profile.pack_version),
        kv_string("ds4.pack.id", pack_id),
        kv_string("ds4.pack.artifact", artifact),
        kv_string("ds4.pack.manifest.file", profile.manifest_name),
        kv_u32("ds4.pack.shard.count", 1),
        kv_u32("ds4.pack.shard.index", 0),
        kv_string("ds4.pack.quant.format", "ggml-block"),
        kv_string(
            "ds4.pack.quant.routed",
            "Q4_K" if profile.name == "q4" else "mixed-q2",
        ),
        kv_string("ds4.pack.quant.embedding", "BF16"),
        kv_string("ds4.pack.quant.control", "source"),
        kv_string("ds4.pack.quant.dense", "Q8_0"),
        kv_string("ds4.pack.quant.gdn", "Q8_0"),
        kv_string("ds4.pack.quant.qsa", "Q8_0"),
        kv_string("ds4.pack.quant.shared", "Q8_0"),
        kv_string("ds4.pack.quant.output", "Q8_0"),
        kv_string("ds4.pack.quant.ple_projection", "Q8_0"),
        kv_string("ds4.pack.quant.vision_projection", "Q8_0"),
        kv_string("ds4.pack.quant.ple", "Q4_1"),
        kv_u32("ds4.pack.padding.routed_down.logical_input", c["expert_ff"]),
        kv_u32(
            "ds4.pack.padding.routed_down.physical_input",
            c["routed_down_physical_input"],
        ),
        kv_u32("ds4.pack.padding.vision_fc2.logical_input", 4304),
        kv_u32(
            "ds4.pack.padding.vision_fc2.physical_input",
            c["vision_fc2_physical_input"],
        ),
        kv_u32("ds4.pack.hyper_connection.divisor_folded", c["hc_count"]),
        kv_u32("qwen4-exp.block_count", c["layers"]),
        kv_u64("qwen4-exp.context_length", c["context"]),
        kv_u32("qwen4-exp.embedding_length", c["hidden"]),
        kv_u32("qwen4-exp.vocab_size", c["vocab"]),
        kv_f32("qwen4-exp.attention.layer_norm_rms_epsilon", c["rms_eps"]),
        kv_u32("qwen4-exp.attention.head_count", c["attn_heads"]),
        kv_u32("qwen4-exp.attention.head_count_kv", c["kv_heads"]),
        kv_u32("qwen4-exp.attention.key_length", c["head_dim"]),
        kv_u32("qwen4-exp.attention.value_length", c["head_dim"]),
        kv_u32("qwen4-exp.rope.dimension_count", c["rope_dim"]),
        kv_f32("qwen4-exp.rope.freq_base", c["rope_theta"]),
        kv_u32("qwen4-exp.full_attention_interval", c["full_attention_interval"]),
        kv_u32("qwen4-exp.linear_attention.key_head_count", c["linear_key_heads"]),
        kv_u32("qwen4-exp.linear_attention.value_head_count", c["linear_value_heads"]),
        kv_u32("qwen4-exp.linear_attention.head_dimension", c["linear_head_dim"]),
        kv_u32("qwen4-exp.linear_attention.conv_kernel", c["linear_conv"]),
        kv_u32("qwen4-exp.expert_count", c["experts"]),
        kv_u32("qwen4-exp.expert_used_count", c["experts_used"]),
        kv_u32("qwen4-exp.expert_feed_forward_length", c["expert_ff"]),
        kv_u32("qwen4-exp.shared_expert_feed_forward_length", c["shared_ff"]),
        kv_u32("qwen4-exp.hyper_connection.count", c["hc_count"]),
        kv_u32("qwen4-exp.hyper_connection.lowrank", c["hc_lowrank"]),
        kv_u32("qwen4-exp.attention.indexer.head_count", c["qsa_heads"]),
        kv_u32("qwen4-exp.attention.indexer.head_count_kv", c["qsa_kv_heads"]),
        kv_u32("qwen4-exp.attention.indexer.key_length", c["qsa_dim"]),
        kv_u32("qwen4-exp.attention.indexer.top_k", c["qsa_topk"]),
        kv_u32("qwen4-exp.attention.indexer.pool_size", c["qsa_ratio"]),
        kv_u32("qwen4-exp.ple.layer", c["ple_layer"]),
        kv_u32("qwen4-exp.ple.embedding_length", c["ple_dim"]),
        kv_u64("qwen4-exp.ple.row_count", c["ple_rows"]),
        kv_u32("qwen4-exp.ple.row_dimension", c["ple_row_dim"]),
        kv_u32("qwen4-exp.ple.conv_kernel", c["ple_conv"]),
        kv_u32("qwen4-exp.ple.ngram_size", c["ngram_size"]),
        kv_u32("qwen4-exp.ple.heads_per_ngram", c["heads_per_ngram"]),
        kv_u64("qwen4-exp.ple.vocab_base", c["ngram_vocab_base"]),
        kv_u64("qwen4-exp.ple.vocab_divisor", c["ngram_vocab_divisor"]),
        kv_u64("qwen4-exp.ple.seed", c["ngram_seed"]),
        kv_u32("qwen4-exp.nextn_predict_layers", 1),
        kv_u32("qwen4-exp.vision.image_token_id", c["image_token"]),
        kv_u32("qwen4-exp.vision.start_token_id", c["vision_start_token"]),
        kv_u32("qwen4-exp.vision.end_token_id", c["vision_end_token"]),
    ]
    if profile.name == "q2":
        routed_index = next(
            index for index, record in enumerate(records)
            if record.startswith(pack_string("ds4.pack.quant.routed"))
        ) + 1
        records[routed_index:routed_index] = [
            kv_string("ds4.pack.quant.routed_gate_up", "IQ2_XXS"),
            kv_string("ds4.pack.quant.routed_down", "Q2_K"),
        ]
    return records


def write_gguf(path: Path, actions: list[Action], metadata: list[bytes],
               tensor_records: dict, quantizer: GGMLQuantizer,
               imatrix: QwenGGUFImatrix | None = None,
               threads: int = 1):
    specs = [spec for action in actions for spec in action.specs]
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        fail(f"{path.name}: duplicate target tensor name")
    offset = 0
    for spec in specs:
        spec.offset = offset
        offset += align(spec.nbytes)
    header_bytes = 4 + 4 + 8 + 8 + sum(map(len, metadata)) + sum(
        len(tensor_header(spec)) for spec in specs
    )
    data_offset = align(header_bytes)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as fp:
        fp.write(b"GGUF")
        fp.write(struct.pack("<IQQ", GGUF_VERSION, len(specs), len(metadata)))
        for record in metadata:
            fp.write(record)
        for spec in specs:
            fp.write(tensor_header(spec))
        fp.write(b"\0" * (data_offset - fp.tell()))
        for action in actions:
            payloads = encode_action(
                action, SOURCE, quantizer, imatrix, threads
            )
            if len(payloads) != len(action.specs):
                fail(f"{action.source}: internal output-count mismatch")
            for spec, raw in zip(action.specs, payloads):
                if len(raw) != spec.nbytes:
                    fail(f"{spec.name}: encoded {len(raw)} bytes, expected {spec.nbytes}")
                expected = data_offset + spec.offset
                if fp.tell() != expected:
                    fail(f"{spec.name}: internal GGUF offset mismatch")
                fp.write(raw)
                padding = align(len(raw)) - len(raw)
                if padding:
                    fp.write(b"\0" * padding)
                tensor_records[spec.name] = tensor_record(
                    spec, path.name, hashlib.sha256(raw).hexdigest()
                )
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temporary, path)


def gguf_layout(actions: list[Action], metadata: list[bytes],
                label: str = "GGUF"):
    specs = [spec for action in actions for spec in action.specs]
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        fail(f"{label}: duplicate target tensor name")
    offsets = []
    payload_bytes = 0
    for spec in specs:
        offsets.append(payload_bytes)
        payload_bytes += align(spec.nbytes)
    tensor_directory_bytes = sum(
        len(pack_string(spec.name)) + 4 + 8 * len(spec.gguf_shape) + 4 + 8
        for spec in specs
    )
    header_bytes = (
        4 + 4 + 8 + 8 + sum(map(len, metadata)) + tensor_directory_bytes
    )
    data_offset = align(header_bytes)
    return specs, offsets, data_offset, data_offset + payload_bytes


class RandomGGUFWriter:
    """Preallocated GGUF writer for source-shard-ordered conversion.

    Tensor payloads are written to their deterministic final offsets, allowing
    one official checkpoint shard to be downloaded, converted into any of the
    DS4 artifacts, fsynced, and removed before the next shard is fetched.
    """

    def __init__(self, path: Path, actions: list[Action], metadata: list[bytes],
                 resume: bool = False):
        self.path = path
        self.actions = actions
        (self.specs, offsets, self.data_offset,
         self.final_size) = gguf_layout(actions, metadata, path.name)
        for spec, offset in zip(self.specs, offsets):
            spec.offset = offset
        prefix = bytearray()
        prefix += b"GGUF"
        prefix += struct.pack(
            "<IQQ", GGUF_VERSION, len(self.specs), len(metadata)
        )
        for record in metadata:
            prefix += record
        for spec in self.specs:
            prefix += tensor_header(spec)
        prefix += b"\0" * (self.data_offset - len(prefix))
        if resume:
            if not path.is_file() or path.stat().st_size != self.final_size:
                fail(
                    f"cannot resume {path.name}: preallocated size does not "
                    f"match the conversion state"
                )
            with path.open("rb") as check:
                if check.read(len(prefix)) != prefix:
                    fail(
                        f"cannot resume {path.name}: GGUF header does not "
                        f"match the current conversion plan"
                    )
        else:
            with path.open("wb") as fp:
                fp.write(prefix)
                fp.truncate(self.final_size)
                fp.flush()
                os.fsync(fp.fileno())
        self.fp = path.open("r+b")

    def write_action(self, action: Action, db: SourceDB,
                     quantizer: GGMLQuantizer, tensor_records: dict,
                     imatrix: QwenGGUFImatrix | None = None,
                     threads: int = 1):
        payloads = encode_action(
            action, db, quantizer, imatrix, threads
        )
        if len(payloads) != len(action.specs):
            fail(f"{action.source}: internal output-count mismatch")
        digests = {}
        for spec, raw in zip(action.specs, payloads):
            if len(raw) != spec.nbytes:
                fail(
                    f"{spec.name}: encoded {len(raw)} bytes, "
                    f"expected {spec.nbytes}"
                )
            self.fp.seek(self.data_offset + spec.offset)
            self.fp.write(raw)
            digest = hashlib.sha256(raw).hexdigest()
            tensor_records[spec.name] = tensor_record(
                spec, self.path.name, digest
            )
            digests[spec.name] = digest
        return digests

    def sync(self):
        self.fp.flush()
        os.fsync(self.fp.fileno())

    def close(self):
        if not self.fp.closed:
            self.fp.close()


def save_conversion_state(path: Path, state: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n"
    with temporary.open("w") as fp:
        fp.write(encoded)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temporary, path)


@dataclasses.dataclass(frozen=True)
class PleGGUFLayout:
    prefix: bytes
    data_offset: int
    specs: tuple[TensorSpec, ...]
    row_bytes: int
    final_size: int


def ple_tensor_specs(rows: int, dim: int = 160) -> list[TensorSpec]:
    if rows <= 0 or dim <= 0 or dim % QTYPE_LAYOUT["Q4_1"][0]:
        fail("PLE Q4_1 geometry must have positive rows and 32-wide blocks")
    return [
        TensorSpec(
            PLE_WEIGHT_NAME, "Q4_1", (rows, dim), "ple",
            "official PLE n-gram embedding shards",
            logical_shape=(rows, dim),
        ),
        TensorSpec(
            PLE_AUX_NAMES[0], "I64", (3,), "ple_geometry",
            PLE_AUX_SUFFIXES[0], logical_shape=(3,),
        ),
        TensorSpec(
            PLE_AUX_NAMES[1], "I64", (16,), "ple_geometry",
            PLE_AUX_SUFFIXES[1], logical_shape=(16,),
        ),
        TensorSpec(
            PLE_AUX_NAMES[2], "I64", (16,), "ple_geometry",
            PLE_AUX_SUFFIXES[2], logical_shape=(16,),
        ),
    ]


def ple_gguf_layout(rows: int, pack_id: str, source_revision: str,
                    dim: int = 160,
                    profile: str | PackProfile = "q4") -> PleGGUFLayout:
    profile = pack_profile(profile)
    specs = ple_tensor_specs(rows, dim)
    if len(specs) != ARTIFACT_TENSOR_COUNTS["ple"]:
        fail(
            f"PLE GGUF has {len(specs)} tensors, expected "
            f"{ARTIFACT_TENSOR_COUNTS['ple']}"
        )
    actions = [
        Action(spec.source, "copy", spec.role, [spec]) for spec in specs
    ]
    metadata = common_metadata(pack_id, source_revision, "ple", profile)
    laid_out, offsets, data_offset, final_size = gguf_layout(
        actions, metadata, profile.ple_name
    )
    for spec, offset in zip(laid_out, offsets):
        spec.offset = offset
    prefix = bytearray(b"GGUF")
    prefix += struct.pack(
        "<IQQ", GGUF_VERSION, len(laid_out), len(metadata)
    )
    for record in metadata:
        prefix += record
    for spec in laid_out:
        prefix += tensor_header(spec)
    prefix += b"\0" * (data_offset - len(prefix))
    return PleGGUFLayout(
        prefix=bytes(prefix),
        data_offset=data_offset,
        specs=tuple(laid_out),
        row_bytes=laid_out[0].nbytes // rows,
        final_size=final_size,
    )


class PleGGUFWriter:
    """Resumable random-access writer for the SSD-backed PLE GGUF."""

    def __init__(self, path: Path, rows: int, pack_id: str,
                 source_revision: str, quantizer: GGMLQuantizer,
                 dim: int = 160, resume: bool = False,
                 profile: str | PackProfile = "q4"):
        self.path = path
        self.rows = rows
        self.dim = dim
        self.quantizer = quantizer
        self.profile = pack_profile(profile)
        layout = ple_gguf_layout(
            rows, pack_id, source_revision, dim, self.profile
        )
        self.data_offset = layout.data_offset
        self.specs = layout.specs
        self.specs_by_name = {spec.name: spec for spec in self.specs}
        self.weight_spec = self.specs_by_name[PLE_WEIGHT_NAME]
        self.row_bytes = layout.row_bytes
        prefix = layout.prefix
        if resume:
            if not path.is_file() or path.stat().st_size != layout.final_size:
                fail(
                    f"cannot resume {path.name}: PLE GGUF size does not "
                    "match the conversion state"
                )
            with path.open("rb") as check:
                if check.read(len(prefix)) != prefix:
                    fail(
                        f"cannot resume {path.name}: PLE GGUF header does "
                        "not match the current conversion plan"
                    )
        else:
            with path.open("wb") as fp:
                fp.write(prefix)
                fp.truncate(layout.final_size)
                fp.flush()
                os.fsync(fp.fileno())
        self.fp = path.open("r+b")

    def write_rows(self, row: int, value: np.ndarray):
        if value.ndim != 2:
            fail("PLE source shard must be rank 2")
        count = value.shape[0]
        if (row < 0 or count <= 0 or value.shape[1] != self.dim or
                row + count > self.rows):
            fail("PLE source shard has incompatible geometry")
        raw = self.quantizer.encode(value, "Q4_1")
        expected = count * self.row_bytes
        if len(raw) != expected:
            fail(
                f"PLE Q4_1 encoder wrote {len(raw)} bytes, expected "
                f"{expected}"
            )
        self.fp.seek(
            self.data_offset + self.weight_spec.offset + row * self.row_bytes
        )
        self.fp.write(raw)
        return hashlib.sha256(raw).hexdigest()

    def write_aux(self, target: str, value: np.ndarray):
        if target not in PLE_AUX_NAMES:
            fail(f"unknown PLE auxiliary tensor {target}")
        spec = self.specs_by_name[target]
        value = np.asarray(value, dtype="<i8")
        if value.shape != spec.shape:
            fail(f"{target}: incompatible PLE auxiliary geometry")
        raw = np.ascontiguousarray(value).tobytes()
        if len(raw) != spec.nbytes:
            fail(f"{target}: incompatible PLE auxiliary payload")
        self.fp.seek(self.data_offset + spec.offset)
        self.fp.write(raw)
        return hashlib.sha256(raw).hexdigest()

    def sync(self):
        self.fp.flush()
        os.fsync(self.fp.fileno())

    def close(self):
        if not self.fp.closed:
            self.fp.close()


def _authenticated_digest(value, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(char not in "0123456789abcdef" for char in value)):
        fail(f"{label}: invalid payload checksum state")
    return value


def _verify_payload_range(path: Path, offset: int, nbytes: int,
                          expected: str, label: str):
    expected = _authenticated_digest(expected, label)
    actual = sha256_file(path, offset, nbytes)
    if actual != expected:
        fail(
            f"{label}: payload checksum mismatch in {path.name}: "
            f"expected {expected}, got {actual}"
        )


def verify_shard_payload_auth(source_shard: str, auth: dict,
                              work: list[tuple[str, Action]],
                              ple_sources: list[str],
                              aux_sources: list[str], db: SourceDB,
                              writers: dict[str, RandomGGUFWriter],
                              ple_writer: PleGGUFWriter,
                              ple_rows: dict[str, int],
                              aux_by_name: dict[str, np.ndarray],
                              tensor_records: dict) -> set[str]:
    """Authenticate every artifact range attributed to one source shard."""
    label = f"conversion state for {source_shard}"
    if not isinstance(auth, dict) or set(auth) != {
            "tensors", "ple_ranges", "ple_aux"}:
        fail(f"{label}: invalid payload authentication directory")
    tensor_auth = auth["tensors"]
    ple_auth = auth["ple_ranges"]
    aux_auth = auth["ple_aux"]
    if not all(isinstance(value, dict) for value in (
            tensor_auth, ple_auth, aux_auth)):
        fail(f"{label}: invalid payload authentication entries")

    expected_tensors = {
        spec.name: (key, spec)
        for key, action in work
        for spec in action.specs
    }
    if set(tensor_auth) != set(expected_tensors):
        fail(f"{label}: tensor payload authentication set does not match plan")
    for name, (key, spec) in expected_tensors.items():
        digest = _authenticated_digest(
            tensor_auth[name], f"{label} tensor {name}"
        )
        writer = writers[key]
        _verify_payload_range(
            writer.path, writer.data_offset + spec.offset, spec.nbytes,
            digest, f"{label} tensor {name}",
        )
        expected_record = tensor_record(spec, writer.path.name, digest)
        if tensor_records.get(name) != expected_record:
            fail(f"{label} tensor {name}: manifest record does not match payload")

    if set(ple_auth) != set(ple_sources):
        fail(f"{label}: PLE row authentication set does not match plan")
    for name in ple_sources:
        rows = db.tensors[name]["shape"][0]
        row = ple_rows[name]
        entry = ple_auth[name]
        expected_fields = {
            "row": row,
            "rows": rows,
            "bytes": rows * ple_writer.row_bytes,
        }
        if (not isinstance(entry, dict) or
                any(entry.get(key) != value
                    for key, value in expected_fields.items()) or
                set(entry) != {*expected_fields, "sha256"}):
            fail(f"{label} PLE range {name}: invalid payload geometry state")
        _verify_payload_range(
            ple_writer.path,
            ple_writer.data_offset + ple_writer.weight_spec.offset +
            row * ple_writer.row_bytes,
            expected_fields["bytes"], entry["sha256"],
            f"{label} PLE range {name}",
        )

    expected_aux = {ple_aux_target(name): name for name in aux_sources}
    if set(aux_auth) != set(expected_aux):
        fail(f"{label}: PLE auxiliary authentication set does not match plan")
    for target, source in expected_aux.items():
        entry = aux_auth[target]
        spec = ple_writer.specs_by_name[target]
        expected_digest = hashlib.sha256(
            np.ascontiguousarray(aux_by_name[source], dtype="<i8").tobytes()
        ).hexdigest()
        expected_fields = {"source": source, "bytes": spec.nbytes}
        if (not isinstance(entry, dict) or
                any(entry.get(key) != value
                    for key, value in expected_fields.items()) or
                set(entry) != {*expected_fields, "sha256"} or
                entry.get("sha256") != expected_digest):
            fail(f"{label} PLE auxiliary {target}: invalid payload state")
        _verify_payload_range(
            ple_writer.path, ple_writer.data_offset + spec.offset,
            spec.nbytes, entry["sha256"],
            f"{label} PLE auxiliary {target}",
        )
    return set(expected_tensors)


def ple_source_layout(db: SourceDB):
    expected_aux = {
        PLE_AUX_SUFFIXES[0]: np.asarray(
            [23703573157769, 20109073645365, 8052911324071], dtype="<i8"
        ),
        PLE_AUX_SUFFIXES[2]: np.asarray(
            [
                20000003, 20000023, 20000033, 20000047,
                20000059, 20000063, 20000069, 20000077,
                20000081, 20000093, 20000107, 20000147,
                20000153, 20000159, 20000161, 20000171,
            ],
            dtype="<i8",
        ),
    }
    expected_aux[PLE_AUX_SUFFIXES[1]] = np.concatenate((
        np.asarray([0], dtype="<i8"),
        np.cumsum(expected_aux[PLE_AUX_SUFFIXES[2]][:-1], dtype="<i8"),
    ))
    aux_by_name = {}
    for suffix, expected in expected_aux.items():
        matches = [name for name in db.tensors if name.endswith(suffix)]
        if len(matches) != 1:
            fail(f"source checkpoint must contain exactly one {suffix} tensor")
        info = db.tensors[matches[0]]
        if info["dtype"] != "I64" or info["shape"] != expected.shape:
            fail(f"{matches[0]} has incompatible PLE hash geometry")
        aux_by_name[matches[0]] = expected

    names = sorted(
        (name for name in db.tensors if NGRAM_MARK in name),
        key=lambda name: int(name.rsplit("_", 1)[1].split(".", 1)[0]),
    )
    if not names:
        fail("official checkpoint contains no PLE n-gram shards")
    expected = list(range(len(names)))
    actual = [int(name.rsplit("_", 1)[1].split(".", 1)[0]) for name in names]
    if actual != expected:
        fail("PLE n-gram shard indices are incomplete or unordered")
    row_offsets = {}
    rows = 0
    for name in names:
        info = db.tensors[name]
        if (info["dtype"] != "BF16" or len(info["shape"]) != 2 or
                info["shape"][1] != CONFIG["ple_row_dim"]):
            fail(f"{name}: expected BF16 rank-2 PLE shard width 160")
        row_offsets[name] = rows
        rows += info["shape"][0]
    if rows != CONFIG["ple_rows"]:
        fail(f"merged PLE has {rows} rows, expected {CONFIG['ple_rows']}")
    return names, row_offsets, rows, aux_by_name


def validate_ple_aux_tensor(db: SourceDB, name: str, expected: np.ndarray):
    actual = db.read(name)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        fail(f"{name} does not match the native DS4 PLE hash geometry")


def ple_aux_target(source: str) -> str:
    for suffix, target in zip(PLE_AUX_SUFFIXES, PLE_AUX_NAMES):
        if source.endswith(suffix):
            return target
    fail(f"unknown PLE auxiliary source tensor {source}")


def create_ple(db: SourceDB, destination: Path, tensor_records: dict,
               pack_id: str, source_revision: str,
               quantizer: GGMLQuantizer,
               profile: str | PackProfile = "q4"):
    names, row_offsets, rows, aux_by_name = ple_source_layout(db)
    writer = PleGGUFWriter(
        destination, rows, pack_id, source_revision, quantizer,
        profile=profile,
    )
    for name, expected in aux_by_name.items():
        validate_ple_aux_tensor(db, name, expected)
        writer.write_aux(ple_aux_target(name), expected)
    for index, name in enumerate(names):
        value = db.read(name)
        writer.write_rows(row_offsets[name], value)
        print(
            f"  PLE {index + 1}/{len(names)} "
            f"rows={row_offsets[name] + value.shape[0]}",
            flush=True,
        )
    writer.sync()
    writer.close()
    record_ple_tensors(destination, writer, tensor_records)


def build_plan(db: SourceDB,
               profile: str | PackProfile = "q4",
               include_mtp: bool = True) -> dict[str, list[Action]]:
    profile = pack_profile(profile)
    plan = {"base": [], "vision": [], "mtp": []}
    base_groups: list[list[Action]] = [[] for _ in range(4)]
    for source in sorted(db.tensors):
        artifact = artifact_for(source)
        if artifact == "ple":
            continue
        if artifact == "mtp" and not include_mtp:
            continue
        action = make_action(db, source, profile)
        if action is not None:
            if artifact == "base":
                base_groups[base_locality_group(source)].append(action)
            else:
                plan[artifact].append(action)
    plan["base"] = [
        action for group in base_groups for action in group
    ]
    layers = {
        layer_of(spec.name)
        for action in plan["base"] for spec in action.specs
        if layer_of(spec.name) is not None
    }
    expected = set(range(CONFIG["layers"]))
    if layers != expected:
        fail(
            f"base GGUF layer coverage is {sorted(layers)}, "
            f"expected {sorted(expected)}"
        )
    return plan


def validate_artifact_tensor_counts(plan: dict[str, list[Action]],
                                    artifacts: Iterable[str]):
    for artifact in artifacts:
        if artifact not in ("base", "vision", "mtp"):
            fail(f"unknown Qwen artifact {artifact}")
        actual = sum(len(action.specs) for action in plan[artifact])
        expected = ARTIFACT_TENSOR_COUNTS[artifact]
        if actual != expected:
            fail(
                f"{artifact} GGUF has {actual} tensors, expected {expected}"
            )


def artifact_record(path: Path, kind: str) -> dict:
    return {
        "kind": kind,
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def refuse_legacy_artifacts(output_root: Path,
                            profile: str | PackProfile = "q4"):
    profile = pack_profile(profile)
    if profile.name != "q4":
        return
    legacy = [output_root / name for name in LEGACY_ARTIFACT_NAMES]
    found = [path for path in legacy if path.exists() or path.is_symlink()]
    if found:
        fail(
            f"legacy Qwen pack artifact is not accepted by pack v{profile.pack_version}: "
            f"{found[0]}; convert into a new output directory"
        )


def record_ple_tensors(destination: Path, writer: PleGGUFWriter,
                       tensor_records: dict):
    for spec in writer.specs:
        digest = sha256_file(
            destination, writer.data_offset + spec.offset, spec.nbytes
        )
        tensor_records[spec.name] = tensor_record(
            spec, destination.name, digest
        )


def write_pack_manifest(args, pack_id: str, tensor_records: dict,
                        artifacts: list[dict],
                        imatrix: QwenGGUFImatrix | None = None):
    profile = profile_from_args(args)
    tensor_manifest = json.dumps(
        tensor_records, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest = {
        "schema": "ds4.qwen4.fast-pack",
        "version": profile.pack_version,
        "required_ds4_version": profile.required_ds4_version,
        "architecture": ARCH,
        "pack_id": pack_id,
        "source": {
            "repository": args.remote_repo or "Qwen/Qwen3.8-Flash-Next",
            "revision": args.source_revision,
        },
        "geometry": CONFIG,
        "quantization": {
            "format": "ggml-block",
            "routed_experts": {
                "qtype": "Q4_K",
                "block_size": QTYPE_LAYOUT["Q4_K"][0],
            },
            "embeddings": {"qtype": "BF16"},
            "control": {"qtype": "source", "allowed": ["BF16", "F32"]},
            "dense_projections": {
                "qtype": "Q8_0",
                "block_size": QTYPE_LAYOUT["Q8_0"][0],
            },
            "gdn_projections": {"qtype": "Q8_0"},
            "qsa_projections": {"qtype": "Q8_0"},
            "shared_expert_projections": {"qtype": "Q8_0"},
            "output": {"qtype": "Q8_0"},
            "ple_projections": {"qtype": "Q8_0"},
            "vision_projections": {"qtype": "Q8_0"},
            "ple_embedding": {
                "qtype": "Q4_1",
                "block_size": QTYPE_LAYOUT["Q4_1"][0],
                "cpu_mapped": True,
            },
            "physical_padding": {
                "routed_down_input": {
                    "logical": CONFIG["expert_ff"],
                    "physical": CONFIG["routed_down_physical_input"],
                    "fill": 0,
                },
                "vision_fc2_input": {
                    "logical": 4304,
                    "physical": CONFIG["vision_fc2_physical_input"],
                    "fill": 0,
                },
            },
        },
        "transforms": {
            "zero_centered_rms_norm": "add-one-bf16",
            "hyper_connection_divisor": {
                "folded_into": [
                    "input_mix_weight_down",
                    "block_inject_weight",
                ],
                "value": CONFIG["hc_count"],
            },
        },
        "artifacts": artifacts,
        "tensor_manifest_sha256": hashlib.sha256(tensor_manifest).hexdigest(),
        "tensors": tensor_records,
    }
    if profile.name == "q2":
        if imatrix is None:
            fail("Q2 pack manifest requires validated imatrix provenance")
        manifest["quantization"]["routed_experts"] = {
            "gate_up": {
                "qtype": "IQ2_XXS",
                "block_size": QTYPE_LAYOUT["IQ2_XXS"][0],
                "block_bytes": QTYPE_LAYOUT["IQ2_XXS"][1],
            },
            "down": {
                "qtype": "Q2_K",
                "block_size": QTYPE_LAYOUT["Q2_K"][0],
                "block_bytes": QTYPE_LAYOUT["Q2_K"][1],
            },
            "imatrix": imatrix.provenance(),
        }
    manifest_path = args.out / profile.manifest_name
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary.open("w") as fp:
        fp.write(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temporary, manifest_path)
    print(
        f"wrote {manifest_path.name}: pack_id={pack_id} "
        f"tensors={len(tensor_records)} artifacts={len(artifacts)}"
    )


def existing_storage_root(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            fail(f"cannot resolve storage volume for {path}")
        candidate = parent
    return candidate if candidate.is_dir() else candidate.parent


def require_conversion_space(output_root: Path, staging_root: Path,
                             artifact_bytes: int, stage_bytes: int,
                             reserve_bytes: int = FREE_SPACE_RESERVE_BYTES):
    output_volume = existing_storage_root(output_root)
    staging_volume = existing_storage_root(staging_root)
    same_volume = os.stat(output_volume).st_dev == os.stat(staging_volume).st_dev
    output_required = artifact_bytes + reserve_bytes
    if same_volume:
        output_required += stage_bytes
    output_free = shutil.disk_usage(output_volume).free
    if output_free < output_required:
        fail(
            "insufficient free space for Qwen conversion on the output "
            f"volume: need {output_required} bytes for selected artifacts, "
            "largest staged shard, and reserve; "
            f"have {output_free}"
        )
    if not same_volume and stage_bytes:
        staging_required = stage_bytes + reserve_bytes
        staging_free = shutil.disk_usage(staging_volume).free
        if staging_free < staging_required:
            fail(
                "insufficient free space for Qwen source staging: need "
                f"{staging_required} bytes for the largest staged shard and "
                f"reserve; have {staging_free}"
            )


def create_streamed_pack(args, db: SourceDB, plan: dict[str, list[Action]],
                         quantizer: GGMLQuantizer, pack_id: str,
                         tokenizer: list[bytes], chat_template: bytes,
                         imatrix: QwenGGUFImatrix | None = None):
    profile = profile_from_args(args)
    if profile.name == "q2" and imatrix is None:
        fail("Q2 pack conversion requires a validated GGUF imatrix")
    refuse_legacy_artifacts(args.out, profile)
    selected = ["base"]
    if not args.no_vision:
        if not plan["vision"]:
            fail("vision sidecar requested but the checkpoint has no model.visual tensors")
        selected.append("vision")
    if not args.no_mtp:
        if not plan["mtp"]:
            fail("MTP sidecar requested but the checkpoint has no mtp tensors")
        selected.append("mtp")

    filenames = {
        "base": profile.base_name,
        "vision": profile.vision_name,
        "mtp": profile.mtp_name,
    }
    metadata = {}
    for key in selected:
        if key == "base":
            records = common_metadata(
                pack_id, args.source_revision, "base", profile
            )
            records += tokenizer
            records.append(
                kv_string("tokenizer.chat_template", chat_template)
            )
        else:
            records = common_metadata(
                pack_id, args.source_revision, key, profile
            )
        metadata[key] = records

    source_directory = {
        name: {
            "filename": info["filename"],
            "offset": info["offset"],
            "nbytes": info["nbytes"],
            "dtype": info["dtype"],
            "shape": info["shape"],
        }
        for name, info in sorted(db.tensors.items())
    }
    source_identity = hashlib.sha256(
        json.dumps(source_directory, sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()
    identity = {
        "state_version": profile.state_version,
        "pack_id": pack_id,
        "source_revision": args.source_revision,
        "source_index_sha256": source_identity,
        "tokenizer_template_sha256": sha256_file(args.tokenizer_template),
        "chat_template_sha256": hashlib.sha256(chat_template).hexdigest(),
        "remote_repo": args.remote_repo,
        "selected_artifacts": selected,
    }
    if profile.name == "q2":
        identity["profile"] = profile.name
        identity["pack_version"] = profile.pack_version
        identity["imatrix"] = imatrix.provenance()
    state_path = args.out / profile.state_name
    manifest_path = args.out / profile.manifest_name
    if manifest_path.exists():
        fail(
            f"{args.out} already contains a finalized Qwen pack; choose a "
            "new --out directory so the existing pack remains recoverable"
        )
    validate_artifact_tensor_counts(plan, selected)
    if args.fresh and state_path.exists():
        state_path.unlink()
    resume = state_path.is_file()
    partial_artifacts = [
        args.out / filenames[key] for key in selected
    ] + [args.out / profile.ple_name]
    if not resume and not args.fresh:
        existing = [path for path in partial_artifacts if path.exists()]
        if existing:
            fail(
                f"partial Qwen artifact exists without a conversion state: "
                f"{existing[0]}; pass --fresh to replace partial artifacts"
            )
    if not resume and args.fresh:
        for path in partial_artifacts:
            if path.exists() or path.is_symlink():
                path.unlink()
    if resume:
        state = json.loads(state_path.read_text())
        if state.get("identity") != identity:
            stored_identity = state.get("identity")
            if isinstance(stored_identity, dict):
                identity_keys = sorted(
                    set(stored_identity).union(identity)
                )
                changed = [
                    key for key in identity_keys
                    if stored_identity.get(key) != identity.get(key)
                ]
                detail = f" (changed identity fields: {', '.join(changed)})"
                tokenizer_only = changed == ["tokenizer_template_sha256"]
            else:
                detail = " (stored identity is malformed)"
                tokenizer_only = False
            if tokenizer_only:
                print(
                    "qwen4-pack: tokenizer template file changed; "
                    "requiring the regenerated GGUF header to match the "
                    "partial pack exactly"
                )
            else:
                fail(
                    f"{state_path} belongs to a different conversion{detail}; "
                    "pass --fresh to replace the partial artifacts"
                )
        completed_entries = state.get("completed_shards", [])
        if (not isinstance(completed_entries, list) or
                any(not isinstance(name, str) for name in completed_entries) or
                len(completed_entries) != len(set(completed_entries))):
            fail(f"{state_path}: invalid completed source shard state")
        completed = set(completed_entries)
        tensor_records = state.get("tensor_records", {})
        if not isinstance(tensor_records, dict):
            fail(f"{state_path}: invalid tensor record state")
        payload_auth = state.get("payload_auth")
        if not isinstance(payload_auth, dict):
            fail(
                f"{state_path}: completed payload authentication is missing; "
                "pass --fresh to replace this pre-authentication state"
            )
        if set(payload_auth) != completed:
            fail(
                f"{state_path}: payload authentication shards do not match "
                "completed source shards"
            )
        print(
            f"resuming conversion after {len(completed)} source shards",
            flush=True,
        )
    else:
        completed = set()
        tensor_records = {}
        payload_auth = {}
        state = {
            "identity": identity,
            "completed_shards": [],
            "tensor_records": tensor_records,
            "payload_auth": payload_auth,
        }

    writers = {}
    ple_writer = None
    try:
        ple_names, ple_rows, total_ple_rows, aux_by_name = ple_source_layout(db)
        ple_path = args.out / profile.ple_name

        work_by_shard: dict[str, list[tuple[str, Action]]] = {}
        for key in selected:
            for action in plan[key]:
                source_shard = db.tensors[action.source]["filename"]
                work_by_shard.setdefault(source_shard, []).append((key, action))
        ple_by_shard: dict[str, list[str]] = {}
        for name in ple_names:
            ple_by_shard.setdefault(
                db.tensors[name]["filename"], []
            ).append(name)
        aux_by_shard: dict[str, list[str]] = {}
        for name in aux_by_name:
            aux_by_shard.setdefault(
                db.tensors[name]["filename"], []
            ).append(name)
        required_shards = sorted(
            set(work_by_shard) | set(ple_by_shard) | set(aux_by_shard)
        )
        unknown_completed = completed - set(required_shards)
        if unknown_completed:
            fail(
                f"{state_path}: completed unknown source shard "
                f"{sorted(unknown_completed)[0]}"
            )

        staging_root = args.staging_dir or (
            args.out /
            (".qwen4-source-stage" if profile.name == "q4" else
             ".qwen2-source-stage")
        )
        artifact_sizes = {
            args.out / filenames[key]: gguf_layout(
                plan[key], metadata[key], filenames[key]
            )[3]
            for key in selected
        }
        artifact_sizes[ple_path] = ple_gguf_layout(
            total_ple_rows, pack_id, args.source_revision,
            profile=profile,
        ).final_size
        if resume:
            artifact_bytes = 0
            for path, final_size in artifact_sizes.items():
                stat = path.stat()
                allocated = min(final_size, stat.st_blocks * 512)
                artifact_bytes += final_size - allocated
        else:
            artifact_bytes = sum(artifact_sizes.values())
        stage_bytes = max(
            (
                db.shards[name]["size"]
                for name in required_shards
                if name not in completed and not db.shards[name]["path"]
            ),
            default=0,
        )
        require_conversion_space(
            args.out, staging_root, artifact_bytes, stage_bytes
        )
        print(
            f"space admission: remaining-artifacts={artifact_bytes / 1e9:.3f} GB "
            f"largest-stage={stage_bytes / 1e9:.3f} GB "
            f"reserve={FREE_SPACE_RESERVE_BYTES / (1 << 30):.0f} GiB",
            flush=True,
        )

        for key in selected:
            writers[key] = RandomGGUFWriter(
                args.out / filenames[key], plan[key], metadata[key], resume
            )
        ple_writer = PleGGUFWriter(
            ple_path, total_ple_rows, pack_id, args.source_revision,
            quantizer, resume=resume, profile=profile
        )
        authenticated_tensors = set()
        for source_shard in sorted(completed):
            authenticated_tensors.update(verify_shard_payload_auth(
                source_shard, payload_auth[source_shard],
                work_by_shard.get(source_shard, []),
                ple_by_shard.get(source_shard, []),
                aux_by_shard.get(source_shard, []), db, writers, ple_writer,
                ple_rows, aux_by_name, tensor_records,
            ))
        if set(tensor_records) != authenticated_tensors:
            fail(
                f"{state_path}: tensor records are not exactly the "
                "authenticated completed payloads"
            )
        if completed:
            print(
                f"authenticated {len(completed)} completed source shards",
                flush=True,
            )
        print(
            f"streaming {len(required_shards)} source shards through "
            f"{staging_root}; largest={max(db.shards[name]['size'] for name in required_shards) / 1e9:.3f} GB",
            flush=True,
        )
        for ordinal, source_shard in enumerate(required_shards, 1):
            if source_shard in completed:
                continue
            print(
                f"source {ordinal}/{len(required_shards)} {source_shard} "
                f"({db.shards[source_shard]['size'] / 1e9:.3f} GB)",
                flush=True,
            )
            with db.materialize(source_shard, staging_root):
                shard_auth = {
                    "tensors": {},
                    "ple_ranges": {},
                    "ple_aux": {},
                }
                for name in aux_by_shard.get(source_shard, []):
                    validate_ple_aux_tensor(db, name, aux_by_name[name])
                    target = ple_aux_target(name)
                    digest = ple_writer.write_aux(
                        target, aux_by_name[name]
                    )
                    spec = ple_writer.specs_by_name[target]
                    shard_auth["ple_aux"][target] = {
                        "source": name,
                        "bytes": spec.nbytes,
                        "sha256": digest,
                    }
                for key, action in work_by_shard.get(source_shard, []):
                    digests = writers[key].write_action(
                        action, db, quantizer, tensor_records, imatrix,
                        getattr(args, "threads", 1),
                    )
                    overlap = set(shard_auth["tensors"]) & set(digests)
                    if overlap:
                        fail(
                            f"{source_shard}: duplicate authenticated tensor "
                            f"{sorted(overlap)[0]}"
                        )
                    shard_auth["tensors"].update(digests)
                for name in ple_by_shard.get(source_shard, []):
                    value = db.read(name)
                    digest = ple_writer.write_rows(ple_rows[name], value)
                    rows = value.shape[0]
                    shard_auth["ple_ranges"][name] = {
                        "row": ple_rows[name],
                        "rows": rows,
                        "bytes": rows * ple_writer.row_bytes,
                        "sha256": digest,
                    }
                    print(
                        f"  PLE part {name.rsplit('_', 1)[1].split('.', 1)[0]} "
                        f"rows={ple_rows[name] + value.shape[0]}",
                        flush=True,
                    )
                    del value
                for writer in writers.values():
                    writer.sync()
                ple_writer.sync()
                verify_shard_payload_auth(
                    source_shard, shard_auth,
                    work_by_shard.get(source_shard, []),
                    ple_by_shard.get(source_shard, []),
                    aux_by_shard.get(source_shard, []), db, writers,
                    ple_writer, ple_rows, aux_by_name, tensor_records,
                )
                completed.add(source_shard)
                payload_auth[source_shard] = shard_auth
                state["completed_shards"] = sorted(completed)
                state["tensor_records"] = tensor_records
                state["payload_auth"] = payload_auth
                save_conversion_state(state_path, state)

        if completed != set(required_shards) or set(payload_auth) != completed:
            fail("streamed conversion payload authentication is incomplete")
        expected_tensors = {
            spec.name
            for key in selected
            for action in plan[key]
            for spec in action.specs
        }
        if set(tensor_records) != expected_tensors:
            missing = sorted(expected_tensors - set(tensor_records))
            extra = sorted(set(tensor_records) - expected_tensors)
            fail(
                "streamed conversion tensor directory mismatch: "
                f"missing={missing[:1]} extra={extra[:1]}"
            )
        for writer in writers.values():
            writer.sync()
            writer.close()
        ple_writer.sync()
        ple_writer.close()

        record_ple_tensors(ple_path, ple_writer, tensor_records)
        artifacts = [artifact_record(args.out / profile.base_name, "base")]
        artifacts.append(artifact_record(ple_path, "ple"))
        if "vision" in selected:
            artifacts.append(
                artifact_record(args.out / profile.vision_name, "vision")
            )
        if "mtp" in selected:
            artifacts.append(
                artifact_record(args.out / profile.mtp_name, "mtp")
            )
        write_pack_manifest(
            args, pack_id, tensor_records, artifacts, imatrix
        )
        state_path.unlink()
        try:
            staging_root.rmdir()
        except OSError:
            pass
    finally:
        for writer in writers.values():
            writer.close()
        if ple_writer is not None:
            ple_writer.close()


def rebuild_vision_artifact(args, db: SourceDB,
                            plan: dict[str, list[Action]],
                            quantizer: GGMLQuantizer,
                            pack_id: str,
                            imatrix: QwenGGUFImatrix | None = None):
    """Atomically rebuild the small vision sidecar in a finalized pack.

    The official visual graph lives in one source shard.  Keeping this repair
    path separate from the 128-shard PLE stream makes format fixes resumable in
    practice without rewriting a valid 100 GB base/PLE pack.  The existing
    manifest is the authority for every untouched artifact and tensor record.
    """
    profile = profile_from_args(args)
    manifest_path = args.out / profile.manifest_name
    if not manifest_path.is_file():
        fail(f"--rebuild-vision requires an existing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "ds4.qwen4.fast-pack" or
            manifest.get("version") != profile.pack_version or
            manifest.get("pack_id") != pack_id or
            manifest.get("source", {}).get("revision") !=
            args.source_revision):
        fail("existing Qwen pack manifest does not match this conversion")
    tensors = manifest.get("tensors")
    artifacts = manifest.get("artifacts")
    if not isinstance(tensors, dict) or not isinstance(artifacts, list):
        fail("existing Qwen pack manifest has no tensor/artifact directory")
    vision_entries = [
        entry for entry in artifacts
        if isinstance(entry, dict) and entry.get("kind") == "vision"
    ]
    if (len(vision_entries) != 1 or
            vision_entries[0].get("path") != profile.vision_name):
        fail("existing Qwen pack manifest has no unique vision sidecar")

    actions = plan["vision"]
    if not actions:
        fail("official checkpoint has no model.visual tensors")
    validate_artifact_tensor_counts(plan, ["vision"])
    temporary = args.out / (profile.vision_name + ".rebuild")
    if temporary.exists():
        temporary.unlink()
    metadata = common_metadata(
        pack_id, args.source_revision, "vision", profile
    )
    writer = RandomGGUFWriter(temporary, actions, metadata, resume=False)
    new_records = {}
    staging_root = args.staging_dir or (args.out / ".qwen4-vision-stage")
    try:
        work_by_shard: dict[str, list[Action]] = {}
        for action in actions:
            shard = db.tensors[action.source]["filename"]
            work_by_shard.setdefault(shard, []).append(action)
        for ordinal, source_shard in enumerate(sorted(work_by_shard), 1):
            print(
                f"vision source {ordinal}/{len(work_by_shard)} "
                f"{source_shard} "
                f"({db.shards[source_shard]['size'] / 1e9:.3f} GB)",
                flush=True,
            )
            with db.materialize(source_shard, staging_root):
                for action in work_by_shard[source_shard]:
                    writer.write_action(action, db, quantizer, new_records)
                writer.sync()
        expected = {
            spec.name for action in actions for spec in action.specs
        }
        if set(new_records) != expected:
            fail("rebuilt vision tensor directory is incomplete")
        writer.close()
        for record in new_records.values():
            record["artifact"] = profile.vision_name
        final_path = args.out / profile.vision_name
        os.replace(temporary, final_path)

        merged_records = {
            name: record for name, record in tensors.items()
            if record.get("artifact") != profile.vision_name
        }
        merged_records.update(new_records)
        merged_artifacts = []
        for entry in artifacts:
            if entry.get("kind") == "vision":
                merged_artifacts.append(artifact_record(final_path, "vision"))
            else:
                merged_artifacts.append(entry)
        write_pack_manifest(
            args, pack_id, merged_records, merged_artifacts, imatrix
        )
    finally:
        writer.close()
        if temporary.exists():
            temporary.unlink()
        try:
            staging_root.rmdir()
        except OSError:
            pass


def rebuild_mtp_artifact(args, db: SourceDB,
                        plan: dict[str, list[Action]],
                        quantizer: GGMLQuantizer,
                        pack_id: str,
                        imatrix: QwenGGUFImatrix | None = None):
    # Atomically add the MTP sidecar to a finalized pack.  The existing
    # manifest is the authority for every untouched artifact and tensor
    # record; the pack must not already carry an MTP sidecar.
    profile = profile_from_args(args)
    manifest_path = args.out / profile.manifest_name
    if not manifest_path.is_file():
        fail(f'--rebuild-mtp requires an existing {manifest_path}')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != 'ds4.qwen4.fast-pack' or
            manifest.get('version') != profile.pack_version or
            manifest.get('pack_id') != pack_id or
            manifest.get('source', {}).get('revision') !=
            args.source_revision):
        fail('existing Qwen pack manifest does not match this conversion')
    tensors = manifest.get('tensors')
    artifacts = manifest.get('artifacts')
    if not isinstance(tensors, dict) or not isinstance(artifacts, list):
        fail('existing Qwen pack manifest has no tensor/artifact directory')
    mtp_entries = [
        entry for entry in artifacts
        if isinstance(entry, dict) and entry.get('kind') == 'mtp'
    ]
    if mtp_entries:
        fail('--rebuild-mtp requires a pack without an existing MTP sidecar')
    actions = plan['mtp']
    if not actions:
        fail('official checkpoint has no MTP tensors')
    validate_artifact_tensor_counts(plan, ['mtp'])
    temporary = args.out / (profile.mtp_name + '.rebuild')
    if temporary.exists():
        temporary.unlink()
    metadata = common_metadata(
        pack_id, args.source_revision, 'mtp', profile
    )
    writer = RandomGGUFWriter(temporary, actions, metadata, resume=False)
    new_records = {}
    staging_root = args.staging_dir or (args.out / '.qwen4-mtp-stage')
    try:
        work_by_shard: dict[str, list[Action]] = {}
        for action in actions:
            shard = db.tensors[action.source]['filename']
            work_by_shard.setdefault(shard, []).append(action)
        for ordinal, source_shard in enumerate(sorted(work_by_shard), 1):
            print(
                f'mtp source {ordinal}/{len(work_by_shard)} '
                f'{source_shard} '
                f"({db.shards[source_shard]['size'] / 1e9:.3f} GB)",
                flush=True,
            )
            with db.materialize(source_shard, staging_root):
                for action in work_by_shard[source_shard]:
                    writer.write_action(
                        action, db, quantizer, new_records, imatrix,
                        getattr(args, 'threads', 1),
                    )
                writer.sync()
        expected = {
            spec.name for action in actions for spec in action.specs
        }
        if set(new_records) != expected:
            fail('rebuilt MTP tensor directory is incomplete')
        writer.close()
        for record in new_records.values():
            record['artifact'] = profile.mtp_name
        final_path = args.out / profile.mtp_name
        os.replace(temporary, final_path)
        merged_records = dict(tensors)
        overlap = set(merged_records) & set(new_records)
        if overlap:
            fail(f'--rebuild-mtp tensor collision: {sorted(overlap)[0]}')
        merged_records.update(new_records)
        merged_artifacts = list(artifacts)
        merged_artifacts.append(artifact_record(final_path, 'mtp'))
        write_pack_manifest(
            args, pack_id, merged_records, merged_artifacts, imatrix
        )
    finally:
        writer.close()
        if temporary.exists():
            temporary.unlink()
        try:
            staging_root.rmdir()
        except OSError:
            pass


def plan_summary(plan: dict[str, list[Action]], db: SourceDB,
                 profile: str | PackProfile = "q4"):
    profile = pack_profile(profile)
    print(
        "DS4 Qwen3.8-Flash-Next pack plan" if profile.name == "q4" else
        "DS4 Qwen3.8-Flash-Next Q2 pack plan"
    )
    print(f"source tensors: {len(db.tensors)}")
    type_bytes: dict[str, int] = {}
    for name in ("base", "vision", "mtp"):
        actions = plan[name]
        specs = [spec for action in actions for spec in action.specs]
        for spec in specs:
            type_bytes[spec.dtype] = type_bytes.get(spec.dtype, 0) + spec.nbytes
        print(f"{name}: actions={len(actions)} tensors={len(specs)} "
              f"payload={sum(spec.nbytes for spec in specs) / 1e9:.3f} GB")
    ngram = [name for name in db.tensors if NGRAM_MARK in name]
    ple_rows = sum(db.tensors[name]["shape"][0] for name in ngram)
    print(f"ple: source_shards={len(ngram)} rows={ple_rows}")
    if profile.name == "q2":
        print(
            "q2_recipe: gate_up=IQ2_XXS down=Q2_K dense=Q8_0 "
            "embedding_control=BF16 ple=Q4_1"
        )
        for qtype, nbytes in sorted(type_bytes.items()):
            print(f"type_bytes: {qtype} {nbytes}")
        ple_bytes = ple_rows * (
            CONFIG["ple_row_dim"] // QTYPE_LAYOUT["Q4_1"][0]
        ) * QTYPE_LAYOUT["Q4_1"][1]
        base_bytes = sum(
            spec.nbytes for action in plan["base"] for spec in action.specs
        )
        sidecar_bytes = sum(
            spec.nbytes
            for name in ("vision", "mtp")
            for action in plan[name]
            for spec in action.specs
        )
        print(f"projected_base_bytes: {base_bytes}")
        print(f"projected_ple_bytes: {ple_bytes}")
        print(f"projected_optional_sidecar_bytes: {sidecar_bytes}")
        print(
            "projected_all_payload_bytes: "
            f"{base_bytes + ple_bytes + sidecar_bytes}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=tuple(PACK_PROFILES), default="q4",
        help="pack recipe: q4 keeps pack v3; q2 emits mixed-IQ2/Q2 pack v4",
    )
    parser.add_argument("--src", required=True, type=Path,
                        help="official checkpoint or local metadata directory")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-revision", required=True,
                        help="immutable official Hub commit SHA")
    parser.add_argument("--tokenizer-template", default=None, type=Path,
                        help="GGUF with the exact 248320-token Qwen tokenizer")
    parser.add_argument("--quants-library", type=Path,
                        help="path to libds4quants (defaults beside this script)")
    parser.add_argument(
        "--imatrix", type=Path,
        help="Q2 GGUF imatrix (required by --profile q2 except --dry-run)",
    )
    parser.add_argument(
        "--imatrix-revision", default=Q2_IMATRIX_REVISION,
        help="immutable source revision recorded for the Q2 imatrix",
    )
    parser.add_argument(
        "--threads", type=int,
        default=min(8, os.cpu_count() or 1),
        help="ordered parallel IQ2 expert encodes (default: min(8, CPU count))",
    )
    parser.add_argument(
        "--remote-repo",
        help=(
            "Hub repository used for missing source shards and metadata; "
            "one shard is staged at a time"
        ),
    )
    parser.add_argument(
        "--staging-dir", type=Path,
        help="temporary source-shard directory (defaults under --out)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="replace any resumable partial conversion in --out",
    )
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument(
        "--mtp-imatrix",
        choices=("weight-energy",),
        default=None,
        help=(
            "select the MTP routed-expert importance source for --profile "
            "q2; weight-energy uses the deterministic per-expert "
            "input-column weight-energy fallback for every MTP expert"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rebuild-mtp",
        action="store_true",
        help=(
            "atomically add an MTP sidecar to an existing finalized pack "
            "that was converted without one"
        ),
    )
    parser.add_argument(
        "--rebuild-vision",
        action="store_true",
        help=(
            "atomically rebuild only the vision sidecar and merge it into an "
            "existing finalized pack"
        ),
    )
    args = parser.parse_args(argv)
    if args.threads < 1 or args.threads > 64:
        parser.error("--threads must be between 1 and 64")
    if args.profile == "q4" and args.imatrix is not None:
        parser.error("--imatrix applies only to --profile q2")
    if args.profile == "q2" and not args.dry_run and args.imatrix is None:
        parser.error("--profile q2 requires --imatrix")
    if args.profile == "q2" and not args.no_mtp:
        if args.mtp_imatrix != "weight-energy":
            parser.error(
                "--profile q2 with an MTP sidecar requires "
                "--mtp-imatrix weight-energy (no calibrated MTP expert "
                "imatrix exists; the deterministic per-expert weight-energy "
                "fallback is the only supported selection)"
            )
    if (args.tokenizer_template is None and not args.dry_run and
            not args.rebuild_vision and not args.rebuild_mtp):
        parser.error("--tokenizer-template is required")
    return args


SOURCE: SourceDB


def main(argv=None):
    global SOURCE
    args = parse_args(argv)
    profile = profile_from_args(args)
    if (len(args.source_revision) != 40 or
            any(ch not in "0123456789abcdefABCDEF" for ch in args.source_revision)):
        fail("--source-revision must be a 40-character immutable commit SHA")
    if (profile.name == "q2" and
            (len(args.imatrix_revision) != 40 or
             any(ch not in "0123456789abcdefABCDEF"
                 for ch in args.imatrix_revision))):
        fail("--imatrix-revision must be a 40-character immutable commit SHA")
    global MTP_IMATRIX_MODE
    MTP_IMATRIX_MODE = args.mtp_imatrix
    imatrix = None
    try:
        if profile.name == "q2" and args.imatrix is not None:
            imatrix = QwenGGUFImatrix(
                args.imatrix,
                expected_sha256=Q2_IMATRIX_SHA256,
                repository=Q2_IMATRIX_REPOSITORY,
                revision=args.imatrix_revision,
            )
        token = get_token() if args.remote_repo else None
        SOURCE = SourceDB(
            args.src, args.remote_repo, args.source_revision, token
        )
        validate_source_config(SOURCE.config)
        plan = build_plan(
            SOURCE, profile,
            include_mtp=(profile.name == "q4" or not args.no_mtp),
        )
        validate_artifact_tensor_counts(
            plan,
            ["base"] + [name for name in ("vision", "mtp") if plan[name]],
        )
        plan_summary(plan, SOURCE, profile)
        if args.dry_run:
            return 0

        args.out.mkdir(parents=True, exist_ok=True)
        refuse_legacy_artifacts(args.out, profile)
        quantizer = GGMLQuantizer(
            args.quants_library or default_quantizer_library()
        )
        identity = profile.pack_identity + "\0" + args.source_revision
        if profile.name == "q2":
            identity += (
                "\0" + imatrix.sha256 + "\0" + args.imatrix_revision
            )
        pack_id = hashlib.sha256(identity.encode()).hexdigest()[:32]
        if args.rebuild_vision:
            if args.no_vision or args.fresh:
                fail("--rebuild-vision is incompatible with --no-vision/--fresh")
            rebuild_vision_artifact(
                args, SOURCE, plan, quantizer, pack_id, imatrix
            )
            return 0
        if args.rebuild_mtp:
            if args.no_mtp:
                fail("--rebuild-mtp is incompatible with --no-mtp")
            rebuild_mtp_artifact(
                args, SOURCE, plan, quantizer, pack_id, imatrix
            )
            return 0
        tokenizer, tokens = load_tokenizer_records(args.tokenizer_template)
        if len(tokens) != CONFIG["vocab"]:
            fail(
                f"tokenizer template has {len(tokens)} tokens, "
                f"expected {CONFIG['vocab']}"
            )
        chat_template = SOURCE.read_metadata_file("chat_template.jinja")
        if not chat_template.strip():
            fail("official chat template is empty")
        create_streamed_pack(
            args, SOURCE, plan, quantizer, pack_id, tokenizer,
            chat_template, imatrix,
        )
        return 0
    finally:
        if imatrix is not None:
            imatrix.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"qwen4-pack: {error}", file=sys.stderr)
        sys.exit(1)
