#!/usr/bin/env python3
"""DS4 helpers for DeepSpec target-cache interoperability."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import tempfile
import textwrap
from pathlib import Path

INDEX_RECORD_STRUCT = struct.Struct("<QIIQQQQQ")
TARGET_CACHE_VERSION = 2
EXPECTED_HIDDEN_DTYPE = "bfloat16"
EXPECTED_TOKEN_DTYPE = "int32"
EXPECTED_MASK_DTYPE = "uint8"
DEFAULT_TARGET_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_CHAT_TEMPLATE = "deepseek_v4_rendered"
DEFAULT_DSPARK_BLOCK_SIZE = 5
DEFAULT_DSPARK_MTP_LAYERS = 3
DEFAULT_TARGET_LAYER_IDS = [40, 41, 42]
DEFAULT_MASK_TOKEN_ID = 128799


class CacheValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CacheValidationError(message)


def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except OSError as exc:
        raise CacheValidationError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CacheValidationError(f"invalid JSON in {path}: {exc}") from exc
    _require(isinstance(data, dict), f"{path} is not a JSON object")
    return data


def _required_int(manifest: dict, key: str) -> int:
    value = manifest.get(key)
    _require(isinstance(value, int) and value >= 0, f"manifest.{key} must be a non-negative integer")
    return value


def _validate_manifest(manifest: dict,
                       expected_target_model: str | None,
                       expected_chat_template: str | None) -> tuple[int, list[int], int, list[dict]]:
    _require(manifest.get("version") == TARGET_CACHE_VERSION,
             f"manifest.version must be {TARGET_CACHE_VERSION}")
    if "format" in manifest:
        _require(manifest["format"] == "deepspec-target-cache",
                 "manifest.format must be deepspec-target-cache")
    _require(manifest.get("hidden_dtype") == EXPECTED_HIDDEN_DTYPE,
             f"manifest.hidden_dtype must be {EXPECTED_HIDDEN_DTYPE}")
    _require(manifest.get("token_dtype") == EXPECTED_TOKEN_DTYPE,
             f"manifest.token_dtype must be {EXPECTED_TOKEN_DTYPE}")
    _require(manifest.get("mask_dtype") == EXPECTED_MASK_DTYPE,
             f"manifest.mask_dtype must be {EXPECTED_MASK_DTYPE}")
    _require(manifest.get("index_record_size") == INDEX_RECORD_STRUCT.size,
             f"manifest.index_record_size must be {INDEX_RECORD_STRUCT.size}")

    hidden_size = _required_int(manifest, "hidden_size")
    _require(hidden_size > 0, "manifest.hidden_size must be positive")
    num_samples = _required_int(manifest, "num_samples")
    num_shards = _required_int(manifest, "num_shards")

    layers = manifest.get("target_layer_ids")
    _require(isinstance(layers, list) and len(layers) > 0,
             "manifest.target_layer_ids must be a non-empty list")
    _require(all(isinstance(layer, int) and layer >= 0 for layer in layers),
             "manifest.target_layer_ids must contain non-negative integers")
    _require(len(set(layers)) == len(layers), "manifest.target_layer_ids must not contain duplicates")
    _require(layers == sorted(layers), "manifest.target_layer_ids must be sorted in capture order")

    target_hidden_layers = manifest.get("target_hidden_layers")
    if target_hidden_layers is not None:
        _require(target_hidden_layers == len(layers),
                 "manifest.target_hidden_layers must match target_layer_ids length")

    if expected_target_model is not None:
        _require(manifest.get("target_model_name_or_path") == expected_target_model,
                 "manifest.target_model_name_or_path does not match expected target model")

    if expected_chat_template is not None:
        convention = manifest.get("input_convention")
        _require(isinstance(convention, dict), "manifest.input_convention must be an object")
        _require(convention.get("chat_template") == expected_chat_template,
                 "manifest.input_convention.chat_template does not match expected template")

    shards = manifest.get("shards")
    _require(isinstance(shards, list), "manifest.shards must be a list")
    _require(len(shards) == num_shards, "manifest.num_shards must match shards length")
    if num_samples > 0:
        _require(num_shards > 0, "manifest with samples must contain at least one shard")
    return hidden_size, layers, num_samples, shards


def _load_shard_map(cache_dir: Path, shards: list[dict]) -> dict[int, Path]:
    shard_map: dict[int, Path] = {}
    for entry in shards:
        _require(isinstance(entry, dict), "manifest.shards entries must be objects")
        shard_id = entry.get("shard_id")
        file_name = entry.get("file_name")
        _require(isinstance(shard_id, int) and shard_id >= 0, "shard_id must be a non-negative integer")
        _require(isinstance(file_name, str) and file_name, "shard file_name must be a non-empty string")
        _require(shard_id not in shard_map, f"duplicate shard_id {shard_id}")
        path = cache_dir / file_name
        _require(path.is_file(), f"missing shard file {path}")
        shard_map[shard_id] = path
    return shard_map


def _intervals_for_record(seq_len: int,
                          hidden_size: int,
                          num_layers: int,
                          offsets: tuple[int, int, int, int, int]) -> list[tuple[str, int, int]]:
    input_ids_offset, attention_mask_offset, loss_mask_offset, target_hidden_offset, target_last_offset = offsets
    target_hidden_bytes = seq_len * num_layers * hidden_size * 2
    target_last_bytes = seq_len * hidden_size * 2
    return [
        ("input_ids", input_ids_offset, seq_len * 4),
        ("attention_mask", attention_mask_offset, seq_len),
        ("loss_mask", loss_mask_offset, seq_len),
        ("target_hidden_states", target_hidden_offset, target_hidden_bytes),
        ("target_last_hidden_states", target_last_offset, target_last_bytes),
    ]


def _validate_record(cache_dir: Path,
                     record_index: int,
                     record: tuple[int, int, int, int, int, int, int, int],
                     shard_map: dict[int, Path],
                     hidden_size: int,
                     num_layers: int) -> None:
    sample_id, shard_id, seq_len, input_ids_offset, attention_mask_offset, loss_mask_offset, target_hidden_offset, target_last_offset = record
    _require(sample_id == record_index,
             f"record {record_index} sample_id is {sample_id}, expected {record_index}")
    _require(seq_len > 0, f"record {record_index} seq_len must be positive")
    _require(shard_id in shard_map, f"record {record_index} references unknown shard_id {shard_id}")
    shard = shard_map[shard_id]
    shard_size = shard.stat().st_size
    intervals = _intervals_for_record(seq_len,
                                      hidden_size,
                                      num_layers,
                                      (input_ids_offset,
                                       attention_mask_offset,
                                       loss_mask_offset,
                                       target_hidden_offset,
                                       target_last_offset))
    sorted_intervals = sorted(intervals, key=lambda item: item[1])
    for name, offset, size in sorted_intervals:
        _require(offset >= 0, f"record {record_index} {name} offset must be non-negative")
        _require(size > 0, f"record {record_index} {name} size must be positive")
        _require(offset + size <= shard_size,
                 f"record {record_index} {name} extends beyond shard {shard.relative_to(cache_dir)}")
    for (_, prev_offset, prev_size), (name, offset, _) in zip(sorted_intervals, sorted_intervals[1:]):
        _require(prev_offset + prev_size <= offset,
                 f"record {record_index} {name} overlaps previous tensor payload")


def validate_target_cache(cache_dir: Path,
                          expected_target_model: str | None = None,
                          expected_chat_template: str | None = None) -> dict:
    cache_dir = cache_dir.resolve()
    _require(cache_dir.is_dir(), f"cache directory does not exist: {cache_dir}")
    manifest = _read_json(cache_dir / "manifest.json")
    hidden_size, layers, num_samples, shards = _validate_manifest(manifest,
                                                                 expected_target_model,
                                                                 expected_chat_template)
    shard_map = _load_shard_map(cache_dir, shards)
    index_path = cache_dir / "samples.idx"
    _require(index_path.is_file(), f"missing index file {index_path}")
    index_size = index_path.stat().st_size
    _require(index_size == num_samples * INDEX_RECORD_STRUCT.size,
             "samples.idx size must equal num_samples * index_record_size")
    with index_path.open("rb") as fp:
        for record_index in range(num_samples):
            raw = fp.read(INDEX_RECORD_STRUCT.size)
            _require(len(raw) == INDEX_RECORD_STRUCT.size,
                     f"short samples.idx record {record_index}")
            _validate_record(cache_dir,
                             record_index,
                             INDEX_RECORD_STRUCT.unpack(raw),
                             shard_map,
                             hidden_size,
                             len(layers))
    return {
        "cache_dir": str(cache_dir),
        "num_samples": num_samples,
        "num_shards": len(shards),
        "hidden_size": hidden_size,
        "target_layer_ids": layers,
        "index_record_size": INDEX_RECORD_STRUCT.size,
    }

def render_nonseq_config(target_cache_path: str | None = None,
                         target_model_name_or_path: str = DEFAULT_TARGET_MODEL,
                         chat_template: str = DEFAULT_CHAT_TEMPLATE,
                         target_layer_ids: list[int] | None = None,
                         max_train_steps: int | None = None,
                         global_batch_size: int = 512,
                         local_batch_size: int = 1) -> str:
    """Return a DeepSpec config for a DeepSeek-V4 non-Markov DSpark pilot."""
    if target_layer_ids is None:
        target_layer_ids = DEFAULT_TARGET_LAYER_IDS
    _require(len(target_layer_ids) > 0, "target_layer_ids must not be empty")
    return textwrap.dedent(f"""\
        # Generated by ds4_deepspec.py for DS4 DeepSpec training.
        import os

        try:
            from deepspec.trainer import DeepSeekV4DSparkTrainer
        except ImportError as exc:
            raise RuntimeError(
                "DS4 DeepSeek-V4 DSpark training needs a DeepSpec checkout/fork "
                "that provides DeepSeekV4DSparkTrainer; upstream DeepSpec main "
                "currently ships Qwen3/Gemma trainers only."
            ) from exc

        BASE_TB_DIR = os.path.expanduser("~/tensorboard")
        BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")

        seed = 42
        project_name = "deepspec"
        exp_name = "dspark_block5_deepseek_v4_flash_nonseq"

        model = dict(
            target_model_name_or_path={target_model_name_or_path!r},
            block_size={DEFAULT_DSPARK_BLOCK_SIZE},
            num_draft_layers={len(target_layer_ids)},
            target_layer_ids={target_layer_ids!r},
            mask_token_id={DEFAULT_MASK_TOKEN_ID},
            num_anchors=512,
            markov_rank=0,
            markov_head_type="vanilla",
            confidence_head_alpha=0.0,
            confidence_head_with_markov=False,
        )

        train = dict(
            trainer_cls=DeepSeekV4DSparkTrainer,
            lr=6.0e-4,
            warmup_ratio=0.04,
            weight_decay=0.0,
            precision="bf16",
            local_batch_size={local_batch_size},
            global_batch_size={global_batch_size},
            num_train_epochs=10,
            max_train_steps={max_train_steps!r},
            max_grad_norm=1.0,
            sharding_strategy="no_shard",
            torch_compile=False,
            loss_decay_gamma=None,
            ce_loss_alpha=1.0,
            l1_loss_alpha=0.0,
        )

        logging = dict(
            logging_steps=10,
            checkpointing_steps=3000,
        )

        data = dict(
            target_cache_path={target_cache_path!r},
            chat_template={chat_template!r},
            max_length=4096,
            num_workers=4,
        )

        def finalize_cfg(cfg):
            logging_cfg = dict(cfg["logging"])
            project = str(cfg["project_name"])
            exp = str(cfg["exp_name"])
            logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project, exp)
            logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project, exp)
            cfg["logging"] = logging_cfg
            return cfg
        """)


def _target_cache_config_defaults(target_cache_path: str,
                                  target_model_name_or_path: str | None,
                                  chat_template: str | None) -> tuple[str, str, list[int]]:
    cache_dir = Path(target_cache_path)
    manifest = _read_json(cache_dir / "manifest.json")

    manifest_target = manifest.get("target_model_name_or_path")
    if target_model_name_or_path is None:
        _require(isinstance(manifest_target, str) and manifest_target,
                 "manifest.target_model_name_or_path is required to emit a config without --target-model")
        target_model_name_or_path = manifest_target
    elif manifest_target is not None:
        _require(isinstance(manifest_target, str) and manifest_target,
                 "manifest.target_model_name_or_path must be a non-empty string when present")
        _require(manifest_target == target_model_name_or_path,
                 "manifest.target_model_name_or_path does not match expected target model")

    convention = manifest.get("input_convention")
    manifest_template = None
    if convention is not None:
        _require(isinstance(convention, dict), "manifest.input_convention must be an object")
        manifest_template = convention.get("chat_template")
        if manifest_template is not None:
            _require(isinstance(manifest_template, str) and manifest_template,
                     "manifest.input_convention.chat_template must be a non-empty string when present")
    if chat_template is None:
        _require(isinstance(manifest_template, str) and manifest_template,
                 "manifest.input_convention.chat_template is required to emit a config without --chat-template")
        chat_template = manifest_template
    elif manifest_template is not None:
        _require(manifest_template == chat_template,
                 "manifest.input_convention.chat_template does not match expected template")

    _, target_layer_ids, _, _ = _validate_manifest(manifest, None, None)
    return target_model_name_or_path, chat_template, target_layer_ids


def write_nonseq_config(path: Path,
                        target_cache_path: str | None = None,
                        target_model_name_or_path: str | None = None,
                        chat_template: str | None = None,
                        max_train_steps: int | None = None,
                        global_batch_size: int = 512,
                        local_batch_size: int = 1,
                        overwrite: bool = False) -> dict:
    if path.exists() and not overwrite:
        raise CacheValidationError(f"refusing to overwrite existing config: {path}")
    _require(target_cache_path is not None and target_cache_path != "",
             "--target-cache is required with --emit-nonseq-config")
    if max_train_steps is not None:
        _require(max_train_steps > 0, "--max-train-steps must be positive")
    _require(global_batch_size > 0, "--global-batch-size must be positive")
    _require(local_batch_size > 0, "--local-batch-size must be positive")
    target_model_name_or_path, chat_template, target_layer_ids = _target_cache_config_defaults(
        target_cache_path,
        target_model_name_or_path,
        chat_template)
    config = render_nonseq_config(target_cache_path,
                                  target_model_name_or_path,
                                  chat_template,
                                  target_layer_ids,
                                  max_train_steps,
                                  global_batch_size,
                                  local_batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config, encoding="utf-8")
    return {
        "config": str(path),
        "target_model_name_or_path": target_model_name_or_path,
        "chat_template": chat_template,
        "target_cache_path": target_cache_path,
        "markov_rank": 0,
    }


def _write_self_test_cache(cache_dir: Path,
                           target_model_name_or_path: str = DEFAULT_TARGET_MODEL,
                           chat_template: str = DEFAULT_CHAT_TEMPLATE,
                           include_optional_config: bool = True) -> None:
    hidden_size = 4
    layers = [1, 2, 3]
    seq_len = 2
    shard = cache_dir / "shard-00000.bin"
    index = cache_dir / "samples.idx"
    manifest = cache_dir / "manifest.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    offsets: list[int] = []
    payloads = [
        struct.pack("<ii", 11, 12),
        b"\x01\x01",
        b"\x01\x01",
        b"\x00\x3f" * (seq_len * len(layers) * hidden_size),
        b"\x00\x3f" * (seq_len * hidden_size),
    ]
    pos = 0
    with shard.open("wb") as fp:
        for payload in payloads:
            offsets.append(pos)
            fp.write(payload)
            pos += len(payload)
    with index.open("wb") as fp:
        fp.write(INDEX_RECORD_STRUCT.pack(0, 0, seq_len, *offsets))
    manifest_data = {
        "version": TARGET_CACHE_VERSION,
        "format": "deepspec-target-cache",
        "producer": "ds4-self-test",
        "num_samples": 1,
        "num_shards": 1,
        "target_layer_ids": layers,
        "hidden_size": hidden_size,
        "target_hidden_layers": len(layers),
        "hidden_dtype": EXPECTED_HIDDEN_DTYPE,
        "token_dtype": EXPECTED_TOKEN_DTYPE,
        "mask_dtype": EXPECTED_MASK_DTYPE,
        "index_record_size": INDEX_RECORD_STRUCT.size,
        "shards": [{"file_name": shard.name, "shard_id": 0}],
    }
    if include_optional_config:
        manifest_data["target_model_name_or_path"] = target_model_name_or_path
        manifest_data["input_convention"] = {"chat_template": chat_template}
    manifest.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")


def self_test() -> dict:
    with tempfile.TemporaryDirectory(prefix="ds4-deepspec-cache-") as tmp:
        cache_dir = Path(tmp) / "cache"
        config_path = Path(tmp) / "dspark_v4_nonseq.py"
        self_test_target_model = "local/self-test-target"
        self_test_chat_template = "self_test_template"
        _write_self_test_cache(cache_dir,
                               target_model_name_or_path=self_test_target_model,
                               chat_template=self_test_chat_template)
        cache_result = validate_target_cache(cache_dir,
                                             expected_target_model=self_test_target_model,
                                             expected_chat_template=self_test_chat_template)
        config_result = write_nonseq_config(config_path,
                                            target_cache_path=str(cache_dir),
                                            max_train_steps=1)
        config_text = config_path.read_text(encoding="utf-8")
        compile(config_text, str(config_path), "exec")
        _require(f"target_model_name_or_path={self_test_target_model!r}" in config_text,
                 "emitted config must inherit target model from cache manifest")
        _require(f"chat_template={self_test_chat_template!r}" in config_text,
                 "emitted config must inherit chat template from cache manifest")
        _require("block_size=5" in config_text, "emitted config must use DeepSeek-V4 DSpark block_size=5")
        _require("num_draft_layers=3" in config_text, "emitted config must use the three DSpark MTP layers")
        _require("target_layer_ids=[1, 2, 3]" in config_text,
                 "emitted config must inherit target layers from cache manifest")
        optional_cache_dir = Path(tmp) / "optional-cache"
        optional_config_path = Path(tmp) / "optional_nonseq.py"
        explicit_target_model = "explicit/target"
        explicit_chat_template = "explicit_template"
        _write_self_test_cache(optional_cache_dir, include_optional_config=False)
        optional_config = write_nonseq_config(optional_config_path,
                                              target_cache_path=str(optional_cache_dir),
                                              target_model_name_or_path=explicit_target_model,
                                              chat_template=explicit_chat_template,
                                              max_train_steps=1)
        optional_text = optional_config_path.read_text(encoding="utf-8")
        compile(optional_text, str(optional_config_path), "exec")
        _require(optional_config["target_model_name_or_path"] == explicit_target_model,
                 "explicit target model must be accepted when optional manifest target is absent")
        _require(optional_config["chat_template"] == explicit_chat_template,
                 "explicit chat template must be accepted when optional manifest template is absent")
        _require(f"target_model_name_or_path={explicit_target_model!r}" in optional_text,
                 "explicit target model must be emitted when optional manifest target is absent")
        _require(f"chat_template={explicit_chat_template!r}" in optional_text,
                 "explicit chat template must be emitted when optional manifest template is absent")
        cache_result["nonseq_config"] = config_result
        return cache_result


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DS4 DeepSpec target-cache artifacts.")
    parser.add_argument("cache_dir", nargs="?", help="Directory containing manifest.json, samples.idx, and shard files.")
    parser.add_argument("--target-model", help="Expected manifest target_model_name_or_path, or emitted config target model.")
    parser.add_argument("--chat-template", help="Expected manifest input_convention.chat_template, or emitted config chat template.")
    parser.add_argument("--self-test", action="store_true", help="Run the built-in synthetic cache/config compatibility smoke.")
    parser.add_argument("--emit-nonseq-config", metavar="FILE", help="Write a DeepSeek-V4 non-Markov DSpark DeepSpec config.")
    parser.add_argument("--target-cache", help="target_cache_path value for --emit-nonseq-config.")
    parser.add_argument("--max-train-steps", type=int, help="Optional train.max_train_steps value for the emitted config.")
    parser.add_argument("--global-batch-size", type=int, default=512, help="Emitted train.global_batch_size. Default: 512.")
    parser.add_argument("--local-batch-size", type=int, default=1, help="Emitted train.local_batch_size. Default: 1.")
    parser.add_argument("--overwrite", action="store_true", help="Allow --emit-nonseq-config to replace FILE.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    try:
        if args.emit_nonseq_config:
            result = write_nonseq_config(Path(args.emit_nonseq_config),
                                         target_cache_path=args.target_cache,
                                         target_model_name_or_path=args.target_model,
                                         chat_template=args.chat_template,
                                         max_train_steps=args.max_train_steps,
                                         global_batch_size=args.global_batch_size,
                                         local_batch_size=args.local_batch_size,
                                         overwrite=args.overwrite)
        elif args.self_test:
            result = self_test()
        else:
            _require(args.cache_dir is not None, "cache_dir is required unless --self-test or --emit-nonseq-config is used")
            result = validate_target_cache(Path(args.cache_dir),
                                           expected_target_model=args.target_model,
                                           expected_chat_template=args.chat_template)
    except CacheValidationError as exc:
        print(f"ds4-deepspec: {exc}", file=sys.stderr)
        return 1
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
