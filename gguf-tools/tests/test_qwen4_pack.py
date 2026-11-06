#!/usr/bin/env python3

import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen4_pack import (
    Action,
    ARTIFACT_TENSOR_COUNTS,
    BASE_NAME,
    CONFIG,
    GGMLQuantizer,
    MANIFEST_NAME,
    PACK_VERSION,
    Q2_BASE_NAME,
    Q2_MANIFEST_NAME,
    Q2_MTP_NAME,
    Q2_PACK_VERSION,
    Q2_PLE_NAME,
    Q2_PROFILE,
    Q2_STATE_NAME,
    Q2_VISION_NAME,
    PLE_AUX_NAMES,
    PLE_NAME,
    PleGGUFWriter,
    RandomGGUFWriter,
    QwenGGUFImatrix,
    TensorSpec,
    artifact_for,
    base_locality_group,
    bf16_to_f32,
    build_plan,
    common_metadata,
    create_streamed_pack,
    default_quantizer_library,
    encode_action,
    encode_weighted_experts,
    f32_to_bf16,
    layer_of,
    make_action,
    parse_args,
    ple_gguf_layout,
    quant_specs,
    record_ple_tensors,
    refuse_legacy_artifacts,
    require_conversion_space,
    tensor_record,
    validate_artifact_tensor_counts,
    validate_source_config,
    verify_shard_payload_auth,
    write_pack_manifest,
)


class FakeDB:
    def __init__(self, tensors, values=None):
        self.tensors = tensors
        self.values = values or {}

    def read(self, name):
        return self.values[name]


def info(dtype, shape):
    return {"dtype": dtype, "shape": tuple(shape)}


def metadata_value(records, key):
    encoded_key = struct.pack("<Q", len(key)) + key.encode()
    record = next(row for row in records if row.startswith(encoded_key))
    value_type = struct.unpack_from("<I", record, len(encoded_key))[0]
    return value_type, record[len(encoded_key) + 4:]


def write_test_imatrix(path, layers=2, experts=3, width=256,
                        missing=None, invalid_negative=None):
    def packed_string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    metadata = [
        packed_string("general.type") + struct.pack("<I", 8) +
        packed_string("imatrix"),
        packed_string("imatrix.datasets") + struct.pack("<IIQ", 9, 8, 1) +
        packed_string("synthetic-test"),
        packed_string("imatrix.chunk_count") + struct.pack("<II", 4, 2),
        packed_string("imatrix.chunk_size") + struct.pack("<II", 4, 128),
    ]
    tensors = []
    for layer in range(layers):
        for part in ("gate", "up", "down"):
            base = f"blk.{layer}.ffn_{part}_exps.weight"
            sums_name = base + ".in_sum2"
            counts_name = base + ".counts"
            sums = np.arange(
                1, experts * width + 1, dtype="<f4"
            ).reshape(experts, width)
            counts = np.arange(1, experts + 1, dtype="<f4")
            if layer == 0 and part in ("gate", "down"):
                counts[1] = 0.0
            if invalid_negative == sums_name:
                sums[0, 0] = -1.0
            if sums_name != missing:
                tensors.append((sums_name, (width, experts), sums.tobytes()))
            if counts_name != missing:
                tensors.append((counts_name, (1, experts), counts.tobytes()))

    offsets = []
    offset = 0
    for _, _, raw in tensors:
        offsets.append(offset)
        offset = (offset + len(raw) + 31) // 32 * 32
    directory = bytearray()
    for (name, shape, _), tensor_offset in zip(tensors, offsets):
        directory += packed_string(name)
        directory += struct.pack("<I", len(shape))
        directory += struct.pack(f"<{len(shape)}Q", *shape)
        directory += struct.pack("<IQ", 0, tensor_offset)
    prefix = bytearray(b"GGUF")
    prefix += struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for record in metadata:
        prefix += record
    prefix += directory
    data_offset = (len(prefix) + 31) // 32 * 32
    prefix += bytes(data_offset - len(prefix))
    with path.open("wb") as fp:
        fp.write(prefix)
        for (_, _, raw), tensor_offset in zip(tensors, offsets):
            self_offset = fp.tell() - data_offset
            if self_offset < tensor_offset:
                fp.write(bytes(tensor_offset - self_offset))
            fp.write(raw)


def reference_q4_1(row):
    row = np.asarray(row, dtype=np.float32)
    minimum = np.float32(np.min(row))
    maximum = np.float32(np.max(row))
    scale = np.float32((maximum - minimum) / np.float32(15.0))
    inverse = np.float32(1.0 / scale) if scale else np.float32(0.0)
    quants = []
    for index in range(16):
        low = int(np.float32((row[index] - minimum) * inverse) + np.float32(0.5))
        high = int(np.float32((row[index + 16] - minimum) * inverse) + np.float32(0.5))
        quants.append(min(15, low) | (min(15, high) << 4))
    return struct.pack("<ee", float(scale), float(minimum)) + bytes(quants)


def reference_q8_0(row):
    row = np.asarray(row, dtype=np.float32)
    maximum = np.float32(np.max(np.abs(row)))
    scale = np.float32(maximum / np.float32(127.0))
    inverse = np.float32(1.0 / scale) if scale else np.float32(0.0)

    def round_away(value):
        return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)

    quants = np.asarray(
        [round_away(float(np.float32(value * inverse))) for value in row],
        dtype=np.int8,
    )
    return struct.pack("<e", float(scale)) + quants.tobytes()


class Qwen4PackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.quantizer = GGMLQuantizer(default_quantizer_library())

    def test_v3_public_artifact_names_are_exact(self):
        self.assertEqual(PACK_VERSION, 3)
        self.assertEqual(
            BASE_NAME,
            "Qwen3.8-Flash-Next-Q4KExperts-BF16Emb-BF16Control-"
            "Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf",
        )
        self.assertEqual(PLE_NAME, "Qwen3.8-Flash-Next-PLE-Q4_1.gguf")
        self.assertEqual(MANIFEST_NAME, "qwen3.8-flash-next-q4.manifest.json")

    def test_q2_v4_has_distinct_artifact_and_state_identity(self):
        self.assertEqual(Q2_PACK_VERSION, 4)
        self.assertEqual(
            Q2_BASE_NAME,
            "Qwen3.8-Flash-Next-IQ2XXSGateUp-Q2KDown-BF16Emb-"
            "BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf",
        )
        self.assertEqual(
            Q2_MANIFEST_NAME, "qwen3.8-flash-next-q2.manifest.json"
        )
        self.assertEqual(
            Q2_STATE_NAME, "qwen3.8-flash-next-q2.conversion-state.json"
        )
        self.assertNotEqual(Q2_PLE_NAME, PLE_NAME)
        self.assertNotEqual(Q2_VISION_NAME, "qwen3.8-flash-next-q4-vision.gguf")
        self.assertNotEqual(Q2_MTP_NAME, "qwen3.8-flash-next-q4-mtp.gguf")

    def test_q2_v4_metadata_declares_mixed_routed_qtypes(self):
        records = common_metadata("pack", "a" * 40, "base", Q2_PROFILE)
        for key, expected in (
            ("ds4.pack.quant.routed", "mixed-q2"),
            ("ds4.pack.quant.routed_gate_up", "IQ2_XXS"),
            ("ds4.pack.quant.routed_down", "Q2_K"),
        ):
            value_type, raw = metadata_value(records, key)
            self.assertEqual(value_type, 8)
            length = struct.unpack_from("<Q", raw)[0]
            self.assertEqual(raw[8:8 + length].decode(), expected)
        value_type, raw = metadata_value(records, "ds4.pack.version")
        self.assertEqual(value_type, 4)
        self.assertEqual(struct.unpack("<I", raw)[0], 4)

    def test_q2_cli_fails_closed_without_no_mtp(self):
        required = [
            "--profile", "q2",
            "--src", "/source",
            "--out", "/output",
            "--source-revision", "a" * 40,
            "--tokenizer-template", "/tokenizer.gguf",
            "--dry-run",
        ]
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit) as error:
                parse_args(required)
        self.assertEqual(error.exception.code, 2)
        parsed = parse_args(required + ["--no-mtp"])
        self.assertEqual(parsed.profile, "q2")
        self.assertTrue(parsed.no_mtp)

    def test_v3_artifact_tensor_count_contract_is_exact(self):
        self.assertEqual(
            ARTIFACT_TENSOR_COUNTS,
            {"base": 1211, "vision": 333, "mtp": 32, "ple": 4},
        )
        spec = TensorSpec("tensor", "BF16", (1,), "control", "source")
        plan = {
            artifact: [Action(
                "source", "copy", "control", [spec] * count
            )]
            for artifact, count in ARTIFACT_TENSOR_COUNTS.items()
            if artifact != "ple"
        }
        validate_artifact_tensor_counts(plan, ("base", "vision", "mtp"))
        plan["mtp"][0].specs.pop()
        with self.assertRaisesRegex(
                ValueError, "mtp GGUF has 31 tensors, expected 32"):
            validate_artifact_tensor_counts(plan, ("mtp",))

    def test_v3_metadata_declares_standard_qtypes_and_padding(self):
        records = common_metadata("pack", "a" * 40, "base")

        value_type, raw = metadata_value(records, "ds4.pack.version")
        self.assertEqual(value_type, 4)  # GGUF UINT32
        self.assertEqual(struct.unpack("<I", raw)[0], 3)

        for key, expected in (
            ("ds4.pack.quant.routed", "Q4_K"),
            ("ds4.pack.quant.embedding", "BF16"),
            ("ds4.pack.quant.dense", "Q8_0"),
            ("ds4.pack.quant.ple", "Q4_1"),
        ):
            value_type, raw = metadata_value(records, key)
            self.assertEqual(value_type, 8)  # GGUF STRING
            length = struct.unpack_from("<Q", raw)[0]
            self.assertEqual(raw[8:8 + length].decode(), expected)

        for key, expected in (
            ("ds4.pack.padding.routed_down.logical_input", 640),
            ("ds4.pack.padding.routed_down.physical_input", 768),
            ("ds4.pack.padding.vision_fc2.logical_input", 4304),
            ("ds4.pack.padding.vision_fc2.physical_input", 4320),
        ):
            value_type, raw = metadata_value(records, key)
            self.assertEqual(value_type, 4)
            self.assertEqual(struct.unpack("<I", raw)[0], expected)

    def test_manifest_records_qtypes_physical_shapes_padding_and_checksums(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            args = SimpleNamespace(
                out=out,
                remote_repo=None,
                source_revision="a" * 40,
            )
            spec = quant_specs(
                "down.weight", (1, 768), "routed_expert", "source",
                "Q4_K", logical_shape=(1, 640),
            )[0]
            tensors = {
                spec.name: tensor_record(spec, BASE_NAME, "d" * 64),
            }
            artifacts = [
                {
                    "kind": "base",
                    "path": BASE_NAME,
                    "bytes": 1,
                    "sha256": "b" * 64,
                },
                {
                    "kind": "ple",
                    "path": PLE_NAME,
                    "bytes": 1,
                    "sha256": "c" * 64,
                },
            ]
            write_pack_manifest(args, "pack", tensors, artifacts)
            manifest = json.loads((out / MANIFEST_NAME).read_text())
            self.assertEqual(manifest["version"], 3)
            self.assertEqual(manifest["quantization"]["format"], "ggml-block")
            self.assertEqual(
                manifest["quantization"]["routed_experts"]["qtype"],
                "Q4_K",
            )
            self.assertEqual(
                manifest["quantization"]["ple_embedding"]["qtype"],
                "Q4_1",
            )
            record = manifest["tensors"]["down.weight"]
            self.assertEqual(record["qtype"], "Q4_K")
            self.assertEqual(record["logical_shape"], [1, 640])
            self.assertEqual(record["physical_shape"], [1, 768])
            self.assertEqual(
                record["padding"],
                {"axis": -1, "fill": 0, "logical": 640, "physical": 768},
            )
            self.assertEqual(record["sha256"], "d" * 64)
            self.assertEqual(manifest["artifacts"], artifacts)

    def test_q2_manifest_records_imatrix_and_mixed_routed_recipe(self):
        class FakeImatrix:
            def provenance(self):
                return {
                    "format": "gguf-imatrix-v3",
                    "sha256": "a" * 64,
                    "revision": "b" * 40,
                    "zero_count_fallback": {
                        "method": "per-expert-input-column-weight-energy",
                        "count": 24,
                    },
                }

        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            args = SimpleNamespace(
                out=out,
                profile="q2",
                remote_repo="Qwen/Qwen3.8-Flash-Next",
                source_revision="a" * 40,
            )
            write_pack_manifest(
                args, "pack", {}, [], FakeImatrix()
            )
            manifest = json.loads((out / Q2_MANIFEST_NAME).read_text())
            self.assertEqual(manifest["version"], 4)
            routed = manifest["quantization"]["routed_experts"]
            self.assertEqual(routed["gate_up"]["qtype"], "IQ2_XXS")
            self.assertEqual(routed["down"]["qtype"], "Q2_K")
            self.assertEqual(
                routed["imatrix"]["zero_count_fallback"]["count"], 24
            )

    def test_legacy_artifacts_are_refused_even_for_a_fresh_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            legacy = out / "qwen3.8-flash-next-q4-00001-of-00004.gguf"
            legacy.write_bytes(b"old")
            with self.assertRaisesRegex(ValueError, "legacy Qwen pack artifact"):
                refuse_legacy_artifacts(out)

    def test_streamed_conversion_refuses_finalized_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            template = out / "tokenizer.gguf"
            template.write_bytes(b"tokenizer")
            (out / MANIFEST_NAME).write_text("{}\n")
            args = SimpleNamespace(
                out=out,
                no_vision=True,
                no_mtp=True,
                source_revision="a" * 40,
                tokenizer_template=template,
                remote_repo=None,
                fresh=False,
            )
            db = FakeDB({
                "source.a": {
                    "filename": "model-00001-of-00001.safetensors",
                    "offset": 0,
                    "nbytes": 2,
                    "dtype": "BF16",
                    "shape": (1,),
                },
            })
            plan = {"base": [], "vision": [], "mtp": []}
            with self.assertRaisesRegex(ValueError, "finalized Qwen pack"):
                create_streamed_pack(
                    args, db, plan, self.quantizer, "pack", [],
                    b"chat template",
                )

    def test_conversion_space_admission_counts_artifacts_stage_and_reserve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch(
                "qwen4_pack.shutil.disk_usage",
                return_value=SimpleNamespace(free=99),
            ):
                with self.assertRaisesRegex(ValueError, "insufficient free space"):
                    require_conversion_space(
                        root, root / "stage", 80, 10, reserve_bytes=10
                    )
            with mock.patch(
                "qwen4_pack.shutil.disk_usage",
                return_value=SimpleNamespace(free=100),
            ):
                require_conversion_space(
                    root, root / "stage", 80, 10, reserve_bytes=10
                )

    def test_random_gguf_writer_is_resumable_and_offset_stable(self):
        values = {
            "source.a": np.arange(8, dtype="<u2").reshape(2, 4),
            "source.b": np.arange(4, dtype="<i4"),
        }
        db = FakeDB(
            {
                "source.a": info("BF16", (2, 4)),
                "source.b": info("I32", (4,)),
            },
            values,
        )
        actions = [
            Action(
                "source.a", "copy", "control",
                [TensorSpec(
                    "target.a", "BF16", (2, 4), "control", "source.a"
                )],
            ),
            Action(
                "source.b", "copy", "control",
                [TensorSpec(
                    "target.b", "I32", (4,), "control", "source.b"
                )],
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.gguf"
            records = {}
            writer = RandomGGUFWriter(path, actions, [], resume=False)
            writer.write_action(actions[1], db, self.quantizer, records)
            writer.sync()
            writer.close()

            resumed = RandomGGUFWriter(path, actions, [], resume=True)
            resumed.write_action(actions[0], db, self.quantizer, records)
            resumed.sync()
            data_offset = resumed.data_offset
            offsets = [action.specs[0].offset for action in actions]
            resumed.close()

            with path.open("rb") as fp:
                self.assertEqual(fp.read(4), b"GGUF")
                self.assertEqual(
                    struct.unpack("<IQQ", fp.read(20)), (3, 2, 0)
                )
                fp.seek(data_offset + offsets[0])
                self.assertEqual(
                    fp.read(values["source.a"].nbytes),
                    values["source.a"].tobytes(),
                )
                fp.seek(data_offset + offsets[1])
                self.assertEqual(
                    fp.read(values["source.b"].nbytes),
                    values["source.b"].tobytes(),
                )
            self.assertEqual(records["target.a"]["qtype"], "BF16")
            self.assertEqual(records["target.a"]["physical_shape"], [2, 4])

            with path.open("r+b") as fp:
                fp.write(b"BAD!")
            with self.assertRaisesRegex(ValueError, "header does not match"):
                RandomGGUFWriter(path, actions, [], resume=True)

    def test_completed_tensor_payload_corruption_is_not_resigned(self):
        source_shard = "model-00001-of-00001.safetensors"
        value = np.arange(8, dtype="<u2").reshape(2, 4)
        db = FakeDB({"source.a": info("BF16", value.shape)}, {"source.a": value})
        action = Action(
            "source.a", "copy", "control",
            [TensorSpec("target.a", "BF16", value.shape, "control", "source.a")],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "base.gguf"
            records = {}
            writer = RandomGGUFWriter(path, [action], [], resume=False)
            digests = writer.write_action(action, db, self.quantizer, records)
            writer.sync()
            auth = {
                "tensors": digests,
                "ple_ranges": {},
                "ple_aux": {},
            }
            verify_shard_payload_auth(
                source_shard, auth, [("base", action)], [], [], db,
                {"base": writer}, None, {}, {}, records,
            )

            inconsistent = json.loads(json.dumps(records))
            inconsistent["target.a"]["role"] = "tampered"
            with self.assertRaisesRegex(ValueError, "manifest record"):
                verify_shard_payload_auth(
                    source_shard, auth, [("base", action)], [], [], db,
                    {"base": writer}, None, {}, {}, inconsistent,
                )

            with path.open("r+b") as fp:
                fp.seek(writer.data_offset + action.specs[0].offset)
                fp.write(b"\xff")
                fp.flush()
                os.fsync(fp.fileno())
            with self.assertRaisesRegex(ValueError, "payload checksum mismatch"):
                verify_shard_payload_auth(
                    source_shard, auth, [("base", action)], [], [], db,
                    {"base": writer}, None, {}, {}, records,
                )
            writer.close()

    def test_ple_is_one_q4_1_tensor_plus_three_i64_aux_tensors(self):
        layout = ple_gguf_layout(2, "pack", "a" * 40, dim=32)
        self.assertEqual(layout.prefix[:4], b"GGUF")
        self.assertEqual(
            struct.unpack_from("<IQQ", layout.prefix, 4)[:2], (3, 4)
        )
        self.assertEqual(
            [spec.name for spec in layout.specs],
            ["ple.weight", *PLE_AUX_NAMES],
        )
        self.assertEqual(
            [spec.dtype for spec in layout.specs],
            ["Q4_1", "I64", "I64", "I64"],
        )
        self.assertEqual(layout.specs[0].gguf_shape, (32, 2))
        self.assertEqual(layout.row_bytes, 20)

    def test_ple_gguf_writer_resumes_and_preserves_rows_and_aux_tensors(self):
        values = f32_to_bf16(
            np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(2, 32)
        )
        aux = {
            PLE_AUX_NAMES[0]: np.asarray([11, 12, 13], dtype="<i8"),
            PLE_AUX_NAMES[1]: np.arange(16, dtype="<i8"),
            PLE_AUX_NAMES[2]: np.arange(20, 36, dtype="<i8"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / PLE_NAME
            writer = PleGGUFWriter(
                path, rows=2, pack_id="pack", source_revision="a" * 40,
                quantizer=self.quantizer, dim=32,
            )
            writer.write_rows(0, values[:1])
            for name, value in aux.items():
                writer.write_aux(name, value)
            writer.sync()
            writer.close()

            resumed = PleGGUFWriter(
                path, rows=2, pack_id="pack", source_revision="a" * 40,
                quantizer=self.quantizer, dim=32, resume=True,
            )
            resumed.write_rows(1, values[1:])
            resumed.sync()
            specs = resumed.specs_by_name
            data_offset = resumed.data_offset
            resumed.close()

            layout = ple_gguf_layout(2, "pack", "a" * 40, dim=32)
            self.assertEqual(path.stat().st_size, layout.final_size)
            with path.open("rb") as fp:
                weight = specs["ple.weight"]
                fp.seek(data_offset + weight.offset)
                self.assertEqual(
                    fp.read(weight.nbytes),
                    self.quantizer.encode(values, "Q4_1"),
                )
                for name, expected in aux.items():
                    spec = specs[name]
                    fp.seek(data_offset + spec.offset)
                    self.assertEqual(fp.read(spec.nbytes), expected.tobytes())

            records = {}
            record_ple_tensors(path, resumed, records)
            self.assertEqual(set(records), {"ple.weight", *PLE_AUX_NAMES})
            self.assertEqual(records["ple.weight"]["qtype"], "Q4_1")
            self.assertEqual(records["ple.weight"]["physical_shape"], [2, 32])
            self.assertEqual(len(records["ple.weight"]["sha256"]), 64)

            with self.assertRaisesRegex(ValueError, "header does not match"):
                PleGGUFWriter(
                    path, rows=2, pack_id="other",
                    source_revision="a" * 40,
                    quantizer=self.quantizer, dim=32, resume=True,
                )

    def test_completed_ple_row_and_aux_corruption_is_not_resigned(self):
        source_shard = "model-00001-of-00001.safetensors"
        row_source = "root.ple.ple_embedding.ngram_embedding.shard_0.weight"
        aux_source = "root.ple.ple_embedding.layer_multipliers"
        values = f32_to_bf16(
            np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(1, 32)
        )
        aux = np.asarray([11, 12, 13], dtype="<i8")
        db = FakeDB({
            row_source: info("BF16", values.shape),
            aux_source: info("I64", aux.shape),
        })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / PLE_NAME
            writer = PleGGUFWriter(
                path, rows=1, pack_id="pack", source_revision="a" * 40,
                quantizer=self.quantizer, dim=32,
            )
            row_digest = writer.write_rows(0, values)
            aux_digest = writer.write_aux(PLE_AUX_NAMES[0], aux)
            writer.sync()
            auth = {
                "tensors": {},
                "ple_ranges": {
                    row_source: {
                        "row": 0,
                        "rows": 1,
                        "bytes": writer.row_bytes,
                        "sha256": row_digest,
                    },
                },
                "ple_aux": {
                    PLE_AUX_NAMES[0]: {
                        "source": aux_source,
                        "bytes": aux.nbytes,
                        "sha256": aux_digest,
                    },
                },
            }
            verify_shard_payload_auth(
                source_shard, auth, [], [row_source], [aux_source], db, {},
                writer, {row_source: 0}, {aux_source: aux}, {},
            )

            with path.open("r+b") as fp:
                fp.seek(writer.data_offset + writer.weight_spec.offset)
                fp.write(b"\xff")
                fp.flush()
                os.fsync(fp.fileno())
            with self.assertRaisesRegex(ValueError, "payload checksum mismatch"):
                verify_shard_payload_auth(
                    source_shard, auth, [], [row_source], [aux_source], db,
                    {}, writer, {row_source: 0}, {aux_source: aux}, {},
                )

            writer.write_rows(0, values)
            writer.sync()
            aux_spec = writer.specs_by_name[PLE_AUX_NAMES[0]]
            with path.open("r+b") as fp:
                fp.seek(writer.data_offset + aux_spec.offset)
                fp.write(b"\xff")
                fp.flush()
                os.fsync(fp.fileno())
            with self.assertRaisesRegex(ValueError, "payload checksum mismatch"):
                verify_shard_payload_auth(
                    source_shard, auth, [], [row_source], [aux_source], db,
                    {}, writer, {row_source: 0}, {aux_source: aux}, {},
                )
            writer.close()

    def test_routed_gate_up_uses_two_standard_q4_k_tensors(self):
        name = "model.language_model.layers.7.mlp.experts.gate_up_proj"
        action = make_action(
            FakeDB({name: info("BF16", (512, 1280, 2560))}), name
        )
        self.assertEqual(action.kind, "split_gate_up")
        self.assertEqual(action.qtype, "Q4_K")
        self.assertEqual(len(action.specs), 2)
        self.assertEqual(
            [spec.name.rsplit(".", 2)[-2] for spec in action.specs],
            ["gate_proj", "up_proj"],
        )
        self.assertTrue(all(spec.dtype == "Q4_K" for spec in action.specs))
        self.assertTrue(
            all(spec.shape == (512, 640, 2560) for spec in action.specs)
        )

    def test_q2_routed_plan_uses_iq2_gate_up_and_weighted_q2_k_down(self):
        gate_up = "model.language_model.layers.7.mlp.experts.gate_up_proj"
        down = "model.language_model.layers.7.mlp.experts.down_proj"
        db = FakeDB({
            gate_up: info("BF16", (512, 1280, 2560)),
            down: info("BF16", (512, 2560, 640)),
        })
        gate_action = make_action(db, gate_up, Q2_PROFILE)
        self.assertEqual(gate_action.qtype, "IQ2_XXS")
        self.assertEqual(
            [spec.dtype for spec in gate_action.specs],
            ["IQ2_XXS", "IQ2_XXS"],
        )
        self.assertEqual(
            gate_action.imatrix_names,
            (
                "blk.7.ffn_gate_exps.weight",
                "blk.7.ffn_up_exps.weight",
            ),
        )
        down_action = make_action(db, down, Q2_PROFILE)
        self.assertEqual((down_action.kind, down_action.qtype),
                         ("quant_experts", "Q2_K"))
        self.assertEqual(
            down_action.imatrix_names,
            ("blk.7.ffn_down_exps.weight",),
        )
        self.assertEqual(down_action.specs[0].shape, (512, 2560, 768))

    def test_routed_down_is_zero_padded_from_640_to_768(self):
        name = "model.language_model.layers.7.mlp.experts.down_proj"
        production = make_action(
            FakeDB({name: info("BF16", (512, 2560, 640))}), name
        )
        spec = production.specs[0]
        self.assertEqual(production.qtype, "Q4_K")
        self.assertEqual(spec.dtype, "Q4_K")
        self.assertEqual(spec.logical_shape, (512, 2560, 640))
        self.assertEqual(spec.shape, (512, 2560, 768))
        self.assertEqual(
            spec.padding,
            {"axis": -1, "logical": 640, "physical": 768, "fill": 0},
        )

        values = f32_to_bf16(
            np.linspace(-1.0, 1.0, 640, dtype=np.float32).reshape(1, 1, 640)
        )
        db = FakeDB({name: info("BF16", values.shape)}, {name: values})
        action = Action(
            name,
            "quant",
            "routed_expert",
            quant_specs(
                "down.weight", (1, 1, 768), "routed_expert", name,
                "Q4_K", logical_shape=(1, 1, 640),
            ),
            "Q4_K",
            pad_last_to=768,
        )
        encoded = encode_action(action, db, self.quantizer)
        expected = self.quantizer.encode(
            np.pad(values, ((0, 0), (0, 0), (0, 128))), "Q4_K"
        )
        self.assertEqual(encoded, [expected])

    def test_projection_families_use_q8_0_while_control_stays_source_type(self):
        names_and_roles = {
            "model.language_model.layers.3.self_attn.q_proj.weight": "qsa",
            "model.language_model.layers.2.linear_attn.out_proj.weight": "gdn",
            "model.language_model.layers.1.mlp.shared_expert.up_proj.weight":
                "shared_expert",
            "model.language_model.layers.1.ple.proj.weight": "ple_projection",
            "lm_head.weight": "output",
        }
        tensors = {
            name: info("BF16", (32, 32)) for name in names_and_roles
        }
        db = FakeDB(tensors)
        for name, role in names_and_roles.items():
            action = make_action(db, name)
            self.assertEqual((action.kind, action.qtype, action.role),
                             ("quant", "Q8_0", role))
            self.assertEqual(action.specs[0].dtype, "Q8_0")

        controls = {
            "model.language_model.embed_tokens.weight": (248320, 2560),
            "model.language_model.layers.0.mlp.gate.weight": (512, 2560),
            "model.language_model.layers.0.linear_attn.in_proj_a.weight":
                (32, 32),
            "model.language_model.layers.0.attn_hyper_connection."
            "input_mix_weight_down.weight": (32, 32),
        }
        control_db = FakeDB({
            name: info("BF16", shape) for name, shape in controls.items()
        })
        for name in controls:
            action = make_action(control_db, name)
            self.assertEqual(action.kind, "copy")
            self.assertIsNone(action.qtype)
            self.assertEqual(action.specs[0].dtype, "BF16")

    def test_optional_mtp_uses_the_same_qtype_rules(self):
        projection = "mtp.layers.0.self_attn.q_proj.weight"
        embedding = "mtp.embed_tokens.weight"
        db = FakeDB({
            projection: info("BF16", (32, 32)),
            embedding: info("BF16", (248320, 2560)),
        })
        self.assertEqual(artifact_for(projection), "mtp")
        self.assertEqual(make_action(db, projection).qtype, "Q8_0")
        self.assertEqual(make_action(db, embedding).specs[0].dtype, "BF16")

    def test_visual_fc2_is_zero_padded_to_q8_0_width_4320(self):
        name = "model.visual.blocks.0.mlp.linear_fc2.weight"
        values = f32_to_bf16(
            np.linspace(-1.0, 1.0, 2 * 4304, dtype=np.float32).reshape(2, 4304)
        )
        db = FakeDB(
            {name: info("BF16", (1152, 4304))},
            {name: values},
        )
        action = make_action(db, name)
        spec = action.specs[0]
        self.assertEqual((action.qtype, action.pad_last_to), ("Q8_0", 4320))
        self.assertEqual(spec.logical_shape, (1152, 4304))
        self.assertEqual(spec.shape, (1152, 4320))
        self.assertEqual(
            encode_action(action, db, self.quantizer),
            [self.quantizer.encode(
                np.pad(values, ((0, 0), (0, 16))), "Q8_0"
            )],
        )

    def test_visual_position_embedding_remains_bf16(self):
        name = "model.visual.pos_embed.weight"
        action = make_action(
            FakeDB({name: info("BF16", (2304, 1152))}), name
        )
        self.assertEqual(action.kind, "copy")
        self.assertEqual(action.role, "vision_embedding")
        self.assertEqual(action.specs[0].dtype, "BF16")

    def test_hyper_connection_divisor_is_folded_in_bf16(self):
        name = (
            "model.language_model.layers.0.attn_hyper_connection."
            "input_mix_weight_down.weight"
        )
        value = f32_to_bf16(
            np.linspace(-2.0, 2.0, 64, dtype=np.float32).reshape(1, 64)
        )
        db = FakeDB({name: info("BF16", value.shape)}, {name: value})
        action = make_action(db, name)
        self.assertEqual(action.scale, 0.25)
        self.assertEqual(action.specs[0].dtype, "BF16")
        expected = f32_to_bf16(
            bf16_to_f32(value) * np.float32(0.25)
        ).tobytes()
        self.assertEqual(encode_action(action, db, self.quantizer), [expected])

    def test_standard_q4_1_matches_reference_block_bytes(self):
        row = np.linspace(-3.0, 2.0, 32, dtype=np.float32)
        encoded = self.quantizer.encode(row[None, :], "Q4_1")
        self.assertEqual(len(encoded), 20)
        self.assertEqual(encoded, reference_q4_1(row))

    def test_standard_q8_0_matches_reference_block_bytes(self):
        row = np.linspace(-2.0, 3.0, 32, dtype=np.float32)
        encoded = self.quantizer.encode(row[None, :], "Q8_0")
        self.assertEqual(len(encoded), 34)
        self.assertEqual(encoded, reference_q8_0(row))

    def test_standard_q4_k_matches_canonical_apple_bf16_vector(self):
        # A BF16-rounded adversarial block that distinguishes the upstream
        # HAVE_BUGGY_APPLE_LINKER accumulation order from reassociated builds.
        source = bytes.fromhex(
            "59bd553ed9bebc3d9ebe723d8ebe7d3d15be8b3e89be0cbd3ebfa2beecbe4a3e"
            "933ec6bee53cdfbdac3dc2bd58be053f353ec4beb43db93eb23e85bd6abe4a3f"
            "aebed03ed93d763e883e6cbe20bfa03a913d64beecbe1f3ef9beb13cab3dd63d"
            "3a3ea3bc8f3c6ebf0a3ffd3a9dbe9e3e15bfc5be863e403e4abcbdbedabc313e"
            "143f5e3cbcbee43e94bd093bda3d7abb3d3ec03ed03ed43ee7be723fffbee3be"
            "f5be073e123e0ebf133e363ec0becebdbf3b293f35bf34bd8cbe38beb4beefbe"
            "063f98b9dd3df0be88be20bf093f773d91bc733fccbd03bfd63eb3be77bfb63b"
            "1abfc3bd783ea03e183f043e76be9dbe763eca3e123e5bbefc3d9e3e393e01bf"
            "18bf8e3e063e973e8a3c343d343ef53d2abffdbd14be64bd25bf563e02bea53e"
            "153f64bec4bd7e3d91be023f0e3ed33e083f423f2b3e433f0dbe1abf6bbebbbc"
            "46be78be653f90bed23e043f40be553ed4bd6d3ea03e433e753ed83c043f1d3e"
            "13bfe73d0e3dac3e6f3db3bea7bef73e30bfaf3eb9becc3d1d3f923e773d53be"
            "3b3ed83cbabe483f7dbe8a3d39bd823edbbefb3efbbd8dbccd3d47bf513e1e3e"
            "6dbd483fdcbd89be983e593eecbe953e4abe1abff33ef53d3dbeff3e62be6dbe"
            "81bde23e0b3d82be2abf9bbe423e23beb23e2bbeb83e8fbd11be23bf87bedf3c"
            "3abe9dbe64bee3be1dbea23d58bee23e993e953e3e3afb3cf9be2f3e8abe653e"
        )
        expected = bytes.fromhex(
            "1a180324f2f1f6bfaeb9ecbfb0c2149467d9a3c8c4783498a67a45b74094a3a9"
            "ba939706f89665cc3953c8bb9b5795bfcc87934a6636c78688fa7a4ab25f0282"
            "3278a8a1d8986355a6bc906694a59342514af84ab7c759986096a6969079c68a"
            "1d8576a8743c48cb0daf398fd6a1755789f8945f0558c77ae37ce68778005999"
            "675f66357aa963fad5d29c9926bc55c5"
        )
        row = np.frombuffer(source, dtype="<u2").reshape(1, 256)
        encoded = self.quantizer.encode(row, "Q4_K")
        self.assertEqual(len(encoded), 144)
        self.assertEqual(encoded, expected)
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            "d0cfb11a385c38c5b64f576f81c02fcbf616eecf2553a83b232500a29877546b",
        )

    def test_q2_weighted_expert_encoding_is_ordered_across_thread_counts(self):
        class FakeImatrix:
            def expert_weights(self, name, expert, values):
                del name
                return np.linspace(
                    1.0 + expert / 1024.0, 2.0, values.shape[-1],
                    dtype=np.float32,
                )

        source = f32_to_bf16(np.linspace(
            -2.0, 2.0, 512 * 256, dtype=np.float32
        ).reshape(512, 1, 256))
        serial = encode_weighted_experts(
            source, "IQ2_XXS", FakeImatrix(), "blk.0.ffn_gate_exps.weight",
            self.quantizer, 1,
        )
        parallel = encode_weighted_experts(
            source, "IQ2_XXS", FakeImatrix(), "blk.0.ffn_gate_exps.weight",
            self.quantizer, 4,
        )
        self.assertEqual(serial, parallel)
        self.assertEqual(len(serial), 512 * 66)

        down = f32_to_bf16(np.linspace(
            -1.0, 1.0, 512 * 640, dtype=np.float32
        ).reshape(512, 1, 640))
        padded = encode_weighted_experts(
            down, "Q2_K", FakeImatrix(), "blk.0.ffn_down_exps.weight",
            self.quantizer, 4, pad_last_to=768,
        )
        self.assertEqual(len(padded), 512 * 3 * 84)

    def test_iq2_quantizer_fails_closed_without_imatrix(self):
        with self.assertRaisesRegex(ValueError, "requires an importance matrix"):
            self.quantizer.encode(
                np.zeros((1, 256), dtype=np.float32), "IQ2_XXS"
            )

    def test_qwen_gguf_imatrix_validates_normalizes_and_falls_back(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
                CONFIG,
                {"layers": 2, "experts": 3, "hidden": 256,
                 "expert_ff": 256}):
            path = Path(temporary) / "imatrix.gguf"
            write_test_imatrix(path)
            imatrix = QwenGGUFImatrix(path)
            try:
                self.assertEqual(len(imatrix.entries), 6)
                self.assertEqual(imatrix.fallback_count, 2)
                values = f32_to_bf16(np.ones((2, 256), dtype=np.float32))
                positive = imatrix.expert_weights(
                    "blk.0.ffn_gate_exps.weight", 0, values
                )
                expected = np.arange(1, 257, dtype=np.float32)
                np.testing.assert_array_equal(positive, expected)
                fallback = imatrix.expert_weights(
                    "blk.0.ffn_gate_exps.weight", 1, values
                )
                np.testing.assert_array_equal(
                    fallback, np.full(256, 2.0, dtype=np.float32)
                )
                provenance = imatrix.provenance()["zero_count_fallback"]
                self.assertEqual(provenance["count"], 2)
                self.assertEqual(provenance["gate_up_count"], 1)
                self.assertEqual(provenance["down_count"], 1)
            finally:
                imatrix.close()

    def test_qwen_gguf_imatrix_rejects_missing_or_negative_entries(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
                CONFIG,
                {"layers": 2, "experts": 3, "hidden": 256,
                 "expert_ff": 256}):
            root = Path(temporary)
            missing = root / "missing.gguf"
            write_test_imatrix(
                missing,
                missing="blk.1.ffn_up_exps.weight.counts",
            )
            with self.assertRaisesRegex(ValueError, "required Q2 imatrix"):
                QwenGGUFImatrix(missing)

            negative = root / "negative.gguf"
            write_test_imatrix(
                negative,
                invalid_negative="blk.0.ffn_gate_exps.weight.in_sum2",
            )
            with self.assertRaisesRegex(ValueError, "nonfinite/negative"):
                QwenGGUFImatrix(negative)

    def test_block_quantization_rejects_invalid_width_and_nonfinite_input(self):
        with self.assertRaisesRegex(ValueError, "block size"):
            self.quantizer.encode(np.zeros((1, 31), dtype=np.float32), "Q8_0")
        row = np.zeros((1, 32), dtype=np.float32)
        row[0, 5] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            self.quantizer.encode(row, "Q4_1")

    def test_single_base_contains_every_layer_and_preserves_locality(self):
        names = [
            "model.language_model.embed_tokens.weight",
            "model.language_model.lm_head.weight",
        ]
        names += [
            f"model.language_model.layers.{layer}.input_layernorm.weight"
            for layer in range(CONFIG["layers"])
        ]
        tensors = {name: info("BF16", (CONFIG["hidden"],)) for name in names}
        plan = build_plan(FakeDB(tensors))
        self.assertEqual(set(plan), {"base", "vision", "mtp"})
        self.assertEqual(
            [action.source for action in plan["base"]],
            sorted(sorted(names), key=base_locality_group),
        )
        layers = {
            layer_of(spec.name)
            for action in plan["base"]
            for spec in action.specs
            if layer_of(spec.name) is not None
        }
        self.assertEqual(layers, set(range(CONFIG["layers"])))

    def test_sidecars_are_classified_before_single_base(self):
        self.assertEqual(
            artifact_for("model.visual.blocks.0.attn.qkv.weight"), "vision"
        )
        self.assertEqual(
            artifact_for("mtp.layers.0.self_attn.q_proj.weight"), "mtp"
        )
        self.assertEqual(
            artifact_for(
                "model.language_model.layers.1.ple.ple_embedding."
                "ngram_embedding.shard_0.weight"
            ),
            "ple",
        )
        self.assertEqual(
            artifact_for("model.language_model.layers.47.input_layernorm.weight"),
            "base",
        )

    def test_official_geometry_is_required(self):
        config = {
            "model_type": "qwen4_exp",
            "num_hidden_layers": 48,
            "hidden_size": 2560,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "full_attention_interval": 4,
            "hc_count": 4,
            "hc_lowrank": 320,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000.0,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "ple_layer_ids": [2],
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ngram_vocab_size_base": 20_000_000,
            "make_ngram_vocab_size_divisible_by": 128,
            "split_ngram_parts": 128,
            "output_gate_type": "sigmoid",
            "eos_token_id": 248044,
            "layer_types": [
                "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
                for layer in range(48)
            ],
        }
        validate_source_config(config)
        config["partial_rotary_factor"] = 1.0
        with self.assertRaisesRegex(ValueError, "partial RoPE"):
            validate_source_config(config)


if __name__ == "__main__":
    unittest.main()
