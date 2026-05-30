#!/usr/bin/env python3
"""Stress structured-output decoding across ds4 and llama.cpp servers.

The OpenAI-compatible surfaces standardize JSON Schema structured outputs and
JSON mode. llguidance itself supports a wider set of grammar tags: JSON Schema,
JSON object, regex, Lark, and the internal guidance grammar-list wire format.

This script keeps those layers explicit:

* ds4 is exercised through /v1/chat/completions and /v1/responses with the
  json_schema/json_object request shapes and ds4's llguidance extension types.
* llama.cpp is exercised with the same OpenAI-compatible JSON cases and, for
  broader llguidance grammar-family cases, with llama.cpp's top-level grammar
  request extension.
* Unsupported target/API/case combinations are reported as SKIP by default. Use
  --strict-skips to make them fail the run, or --force-extensions to send
  experimental non-OpenAI response_format types to targets that do not expose
  them by default.

Examples:
  python3 stress-test-cli.py

  python3 stress-test-cli.py --start never \
      --ds4-base-url http://127.0.0.1:8000/v1 \
      --llama-base-url http://127.0.0.1:8080/v1

  python3 stress-test-cli.py --targets llama --families regex,lark,llguidance \
      --llama-hf-model unsloth/Qwen3.5-9B-GGUF:Q8_0
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_DS4_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_LLAMA_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LLAMA_HF_MODEL = "unsloth/Qwen3.5-9B-GGUF:Q8_0"


class ValidationError(Exception):
    pass


class UnsupportedCase(Exception):
    pass


Validator = Callable[[str], str]


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    prompt: str
    validator: Validator
    schema: dict[str, Any] | None = None
    data: str = ""
    llama_grammar: str | None = None
    oracle_sample: str | None = None
    max_tokens: int = 192


@dataclass
class Target:
    name: str
    base_url: str
    model: str
    command: list[str] | None
    cwd: Path
    supports_response_format_extensions: bool
    supports_grammar_extension: bool
    process: subprocess.Popen[str] | None = None
    log_path: Path | None = None
    started_by_us: bool = False


@dataclass
class Counts:
    passed: int = 0
    failed: int = 0
    skipped: int = 0


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def type_matches(value: Any, typ: str) -> bool:
    if typ == "object":
        return isinstance(value, dict)
    if typ == "array":
        return isinstance(value, list)
    if typ == "string":
        return isinstance(value, str)
    if typ == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if typ == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if typ == "boolean":
        return isinstance(value, bool)
    if typ == "null":
        return value is None
    return True


def _validate_format(value: str, fmt: str, path: str) -> None:
    if fmt == "date":
        try:
            _dt.date.fromisoformat(value)
        except ValueError as exc:
            raise ValidationError(f"{path}: expected RFC3339 date, got {value!r}") from exc
    elif fmt == "time":
        try:
            _dt.time.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError(f"{path}: expected RFC3339 time, got {value!r}") from exc
    elif fmt == "date-time":
        try:
            _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError(f"{path}: expected RFC3339 date-time, got {value!r}") from exc
    elif fmt == "email":
        if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value) is None:
            raise ValidationError(f"{path}: expected email, got {value!r}")
    elif fmt == "uuid":
        if re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
            r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}",
            value,
        ) is None:
            raise ValidationError(f"{path}: expected uuid, got {value!r}")


def validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    if "allOf" in schema:
        for option in schema["allOf"]:
            validate_schema(value, option, path)

    if "anyOf" in schema:
        errors: list[str] = []
        for option in schema["anyOf"]:
            try:
                validate_schema(value, option, path)
                return
            except ValidationError as exc:
                errors.append(str(exc))
        raise ValidationError(f"{path}: did not match anyOf: {'; '.join(errors)}")

    if "oneOf" in schema:
        matches = 0
        errors: list[str] = []
        for option in schema["oneOf"]:
            try:
                validate_schema(value, option, path)
                matches += 1
            except ValidationError as exc:
                errors.append(str(exc))
        if matches != 1:
            raise ValidationError(f"{path}: expected exactly one oneOf match, got {matches}: {'; '.join(errors)}")
        return

    if "const" in schema and value != schema["const"]:
        raise ValidationError(f"{path}: expected const {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(f"{path}: expected one of {schema['enum']!r}, got {value!r}")

    typ = schema.get("type")
    if isinstance(typ, list):
        if not any(type_matches(value, t) for t in typ):
            raise ValidationError(f"{path}: wrong type {type(value).__name__}, expected {typ!r}")
    elif isinstance(typ, str) and not type_matches(value, typ):
        raise ValidationError(f"{path}: wrong type {type(value).__name__}, expected {typ!r}")

    if typ == "object" or "properties" in schema:
        if not isinstance(value, dict):
            raise ValidationError(f"{path}: expected object")
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ValidationError(f"{path}: missing required property {key!r}")
        min_props = schema.get("minProperties")
        max_props = schema.get("maxProperties")
        if min_props is not None and len(value) < min_props:
            raise ValidationError(f"{path}: expected at least {min_props} properties")
        if max_props is not None and len(value) > max_props:
            raise ValidationError(f"{path}: expected at most {max_props} properties")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(props))
            if extra:
                raise ValidationError(f"{path}: extra properties {extra!r}")
        for key, sub in props.items():
            if key in value:
                validate_schema(value[key], sub, f"{path}.{key}")

    if typ == "array" or "items" in schema or "prefixItems" in schema:
        if not isinstance(value, list):
            raise ValidationError(f"{path}: expected array")
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < min_items:
            raise ValidationError(f"{path}: expected at least {min_items} items")
        if max_items is not None and len(value) > max_items:
            raise ValidationError(f"{path}: expected at most {max_items} items")
        prefix_items = schema.get("prefixItems")
        if isinstance(prefix_items, list):
            for i, sub in enumerate(prefix_items):
                if i < len(value):
                    validate_schema(value[i], sub, f"{path}[{i}]")
        items = schema.get("items")
        if isinstance(items, dict):
            start = len(prefix_items) if isinstance(prefix_items, list) else 0
            for i, item in enumerate(value[start:], start):
                validate_schema(item, items, f"{path}[{i}]")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValidationError(f"{path}: string shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValidationError(f"{path}: string longer than maxLength {schema['maxLength']}")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise ValidationError(f"{path}: {value!r} does not match {schema['pattern']!r}")
        if "format" in schema:
            _validate_format(value, schema["format"], path)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValidationError(f"{path}: {value!r} is below minimum {schema['minimum']!r}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValidationError(f"{path}: {value!r} is above maximum {schema['maximum']!r}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ValidationError(f"{path}: {value!r} is not above exclusiveMinimum {schema['exclusiveMinimum']!r}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            raise ValidationError(f"{path}: {value!r} is not below exclusiveMaximum {schema['exclusiveMaximum']!r}")
        if "multipleOf" in schema:
            q = value / schema["multipleOf"]
            if not math.isclose(q, round(q), rel_tol=0.0, abs_tol=1e-9):
                raise ValidationError(f"{path}: {value!r} is not a multiple of {schema['multipleOf']!r}")


def parse_json_strict(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"output is not JSON: {text!r}") from exc


def json_schema_validator(schema: dict[str, Any]) -> Validator:
    def _validate(text: str) -> str:
        value = parse_json_strict(text)
        validate_schema(value, schema)
        return compact_json(value)

    return _validate


def json_object_validator(text: str) -> str:
    value = parse_json_strict(text)
    if not isinstance(value, dict):
        raise ValidationError(f"output is not a JSON object: {value!r}")
    return compact_json(value)


def regex_validator(pattern: str) -> Validator:
    rx = re.compile(pattern)

    def _validate(text: str) -> str:
        value = text.strip()
        if rx.fullmatch(value) is None:
            raise ValidationError(f"{value!r} does not match /{pattern}/")
        return value

    return _validate


def choice_validator(choices: set[str]) -> Validator:
    def _validate(text: str) -> str:
        value = text.strip()
        if value not in choices:
            raise ValidationError(f"{value!r} is not one of {sorted(choices)!r}")
        return value

    return _validate


def permutation_validator(chars: str) -> Validator:
    expected = sorted(chars)

    def _validate(text: str) -> str:
        value = text.strip()
        if sorted(value) != expected or len(value) != len(chars):
            raise ValidationError(f"{value!r} is not a permutation of {chars!r}")
        return value

    return _validate


def substring_chunk_validator(prefix: str, words: list[str]) -> Validator:
    allowed = {""}
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            allowed.add(" ".join(words[i:j]))

    def _validate(text: str) -> str:
        value = text.strip()
        if not value.startswith(prefix):
            raise ValidationError(f"{value!r} does not start with {prefix!r}")
        tail = value[len(prefix):]
        if tail not in allowed:
            raise ValidationError(f"{tail!r} is not an allowed contiguous word substring")
        return value

    return _validate


def make_cases() -> list[Case]:
    calendar_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 80},
            "date": {"type": "string", "format": "date"},
            "participants": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": 5,
            },
        },
        "required": ["name", "date", "participants"],
        "additionalProperties": False,
    }
    status_schema = {
        "type": "object",
        "properties": {
            "status": {"const": "ok"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
            "retry_count": {"type": "integer", "minimum": 0, "maximum": 5},
            "active": {"type": "boolean"},
        },
        "required": ["status", "priority", "retry_count", "active"],
        "additionalProperties": False,
    }
    ticket_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": "TCK-[0-9]{3}"},
            "owner": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
            "priority": {"oneOf": [{"const": "low"}, {"const": "medium"}, {"const": "high"}]},
            "contact": {"type": "string", "format": "email"},
        },
        "required": ["id", "owner", "priority", "contact"],
        "additionalProperties": False,
    }
    reading_schema = {
        "type": "object",
        "properties": {
            "reading": {
                "type": "array",
                "prefixItems": [
                    {"const": "temperature_c"},
                    {"type": "number", "minimum": -40, "exclusiveMaximum": 80, "multipleOf": 0.5},
                ],
                "minItems": 2,
                "maxItems": 2,
            },
            "tags": {
                "type": "array",
                "items": {"type": "string", "pattern": "[a-z]{3,8}"},
                "minItems": 2,
                "maxItems": 3,
            },
        },
        "required": ["reading", "tags"],
        "additionalProperties": False,
    }

    inline_json_schema = {
        "type": "object",
        "properties": {
            "kind": {"const": "metric"},
            "value": {"type": "integer", "minimum": 1, "maximum": 9},
        },
        "required": ["kind", "value"],
        "additionalProperties": False,
    }
    inline_json_lark = f"""%llguidance {{}}
start: %json {compact_json(inline_json_schema)}
"""
    choice_lark = """%llguidance {}
start: "red" | "green" | "blue"
"""
    regex_lark = """%llguidance {}
start: /INV-[0-9]{4}/
"""
    regex_ext_lark = """%llguidance {}
start: "chunk:" %regex { "substring_words": "alpha beta gamma delta" }
"""
    parametric_lark = """%llguidance {}
start: perm::0x0
perm::_: ""                   %if is_ones([0:3])
       | "a" perm::set_bit(0) %if bit_clear(0)
       | "b" perm::set_bit(1) %if bit_clear(1)
       | "c" perm::set_bit(2) %if bit_clear(2)
"""
    guidance_lark = """%llguidance {}
start: "YES" | "NO"
"""
    guidance_wire = compact_json({"grammars": [{"lark_grammar": guidance_lark}]})

    return [
        Case(
            name="json_schema_calendar",
            family="json_schema",
            prompt="Return one lunch calendar event for Alice and Bob on 2026-06-01. Return only JSON.",
            schema=calendar_schema,
            validator=json_schema_validator(calendar_schema),
            data=compact_json(calendar_schema),
            oracle_sample='{"name":"Lunch","date":"2026-06-01","participants":["Alice","Bob"]}',
        ),
        Case(
            name="json_schema_status",
            family="json_schema",
            prompt="Return a compact health-check object with status ok. Return only JSON.",
            schema=status_schema,
            validator=json_schema_validator(status_schema),
            data=compact_json(status_schema),
            oracle_sample='{"status":"ok","priority":"medium","retry_count":2,"active":true}',
        ),
        Case(
            name="json_schema_anyof_oneof_format",
            family="json_schema",
            prompt=(
                "Return one support ticket with an id like TCK-123, an owner or null, "
                "one priority, and a contact email. Return only JSON."
            ),
            schema=ticket_schema,
            validator=json_schema_validator(ticket_schema),
            data=compact_json(ticket_schema),
            oracle_sample='{"id":"TCK-123","owner":null,"priority":"high","contact":"ops@example.com"}',
        ),
        Case(
            name="json_schema_tuple_numeric",
            family="json_schema",
            prompt="Return one sensor reading tuple and two short lowercase tags. Return only JSON.",
            schema=reading_schema,
            validator=json_schema_validator(reading_schema),
            data=compact_json(reading_schema),
            oracle_sample='{"reading":["temperature_c",21.5],"tags":["lab","green"]}',
        ),
        Case(
            name="json_object_mode",
            family="json_object",
            prompt="Return a JSON object with a tiny task description and whether it is done.",
            validator=json_object_validator,
            data="",
            oracle_sample='{"task":"check","done":false}',
        ),
        Case(
            name="regex_invoice_id",
            family="regex",
            prompt="Return exactly one invoice id in the form INV-0427. No quotes, no prose.",
            validator=regex_validator(r"INV-[0-9]{4}"),
            data=r"INV-[0-9]{4}",
            llama_grammar=regex_lark,
            oracle_sample="INV-0427",
            max_tokens=32,
        ),
        Case(
            name="lark_choice",
            family="lark",
            prompt="Return exactly one lowercase color: red, green, or blue. No quotes, no prose.",
            validator=choice_validator({"red", "green", "blue"}),
            data=choice_lark,
            llama_grammar=choice_lark,
            oracle_sample="green",
            max_tokens=16,
        ),
        Case(
            name="lark_inline_json",
            family="lark",
            prompt='Return a compact JSON object with kind "metric" and a small integer value.',
            validator=json_schema_validator(inline_json_schema),
            data=inline_json_lark,
            llama_grammar=inline_json_lark,
            oracle_sample='{"kind":"metric","value":7}',
            max_tokens=64,
        ),
        Case(
            name="lark_regex_ext_substring",
            family="lark",
            prompt="Return exactly chunk:beta gamma. No quotes, no prose.",
            validator=substring_chunk_validator("chunk:", ["alpha", "beta", "gamma", "delta"]),
            data=regex_ext_lark,
            llama_grammar=regex_ext_lark,
            oracle_sample="chunk:beta gamma",
            max_tokens=32,
        ),
        Case(
            name="lark_parametric_permutation",
            family="lark",
            prompt="Return exactly one permutation of the letters a, b, and c. No separators, no prose.",
            validator=permutation_validator("abc"),
            data=parametric_lark,
            llama_grammar=parametric_lark,
            oracle_sample="cab",
            max_tokens=16,
        ),
        Case(
            name="llguidance_internal_wire",
            family="llguidance",
            prompt="Return exactly YES or NO in uppercase. No punctuation, no prose.",
            validator=choice_validator({"YES", "NO"}),
            data=guidance_wire,
            llama_grammar=guidance_lark,
            oracle_sample="YES",
            max_tokens=16,
        ),
    ]


def post_json(url: str, payload: dict[str, Any], timeout: float, api_key: str | None = None) -> dict[str, Any]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {raw[:1600]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc)) from exc
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON response: {raw[:1600]}") from exc
    if isinstance(body, dict) and body.get("error"):
        raise RuntimeError(f"API error: {body['error']!r}")
    return body


def get_status(url: str, timeout: float) -> tuple[int | None, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(512).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(512).decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return None, str(exc)


def extract_chat_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"missing choices in chat response: {body!r}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
        if parts:
            return "".join(parts)
    raise RuntimeError(f"missing text content in chat response: {body!r}")


def extract_responses_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    parts: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for part in item.get("content", []):
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    if parts:
        return "".join(parts)
    raise RuntimeError(f"missing output text in responses response: {body!r}")


def response_format_for_case(case: Case, target: Target, api: str, args: argparse.Namespace) -> dict[str, Any]:
    if case.family == "json_object":
        fmt: dict[str, Any] = {"type": "json_object"}
        if args.json_object_schema:
            fmt["schema"] = {"type": "object"}
        return fmt

    if case.family == "json_schema":
        if not case.schema:
            raise RuntimeError(f"{case.name}: missing schema")
        if api == "chat":
            if target.name == "llama" and args.llama_chat_schema_style == "flat":
                return {
                    "type": "json_schema",
                    "name": case.name,
                    "strict": True,
                    "schema": case.schema,
                }
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": case.name,
                    "strict": True,
                    "schema": case.schema,
                },
            }
        return {
            "type": "json_schema",
            "name": case.name,
            "strict": True,
            "schema": case.schema,
        }

    if not args.force_extensions and not target.supports_response_format_extensions:
        raise UnsupportedCase(f"{target.name} does not expose {case.family!r} through OpenAI response_format")

    if case.family == "regex":
        return {"type": "regex", "regex": case.data}
    if case.family == "lark":
        return {"type": "lark", "grammar": case.data}
    if case.family == "llguidance":
        return {"type": "llguidance", "grammar": case.data}
    raise UnsupportedCase(f"unknown structured-output family {case.family!r}")


def add_llama_common_payload_fields(payload: dict[str, Any], target: Target, args: argparse.Namespace) -> None:
    if target.name != "llama":
        return
    if args.llama_disable_thinking:
        payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    if args.seed is not None:
        payload["seed"] = args.seed


def chat_payload(target: Target, case: Case, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": target.model,
        "messages": [{"role": "user", "content": case.prompt}],
        "max_tokens": case.max_tokens,
        "temperature": 0,
    }
    if (case.family in {"json_schema", "json_object"} or
            args.force_extensions or
            target.supports_response_format_extensions):
        payload["response_format"] = response_format_for_case(case, target, "chat", args)
    elif target.supports_grammar_extension and case.llama_grammar:
        payload["grammar"] = case.llama_grammar
    else:
        raise UnsupportedCase(f"{target.name}/chat cannot carry {case.family!r}")
    if args.seed is not None:
        payload["seed"] = args.seed
    add_llama_common_payload_fields(payload, target, args)
    return payload


def responses_payload(target: Target, case: Case, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": target.model,
        "input": case.prompt,
        "max_output_tokens": case.max_tokens,
        "temperature": 0,
    }
    if (case.family in {"json_schema", "json_object"} or
            args.force_extensions or
            target.supports_response_format_extensions):
        payload["text"] = {"format": response_format_for_case(case, target, "responses", args)}
    elif target.supports_grammar_extension and case.llama_grammar:
        payload["grammar"] = case.llama_grammar
    else:
        raise UnsupportedCase(f"{target.name}/responses cannot carry {case.family!r}")
    if args.seed is not None:
        payload["seed"] = args.seed
    add_llama_common_payload_fields(payload, target, args)
    return payload


def check_case(target: Target, api: str, case: Case, args: argparse.Namespace) -> str:
    if api == "chat":
        payload = chat_payload(target, case, args)
        body = post_json(f"{target.base_url}/chat/completions", payload, args.timeout, args.api_key)
        text = extract_chat_text(body)
    elif api == "responses":
        payload = responses_payload(target, case, args)
        body = post_json(f"{target.base_url}/responses", payload, args.timeout, args.api_key)
        text = extract_responses_text(body)
    else:
        raise RuntimeError(f"unknown api {api!r}")
    return case.validator(text)


def split_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def flatten_extra_args(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        out.extend(shlex.split(value))
    return out


def base_root(base_url: str) -> str:
    if base_url.endswith("/v1"):
        return base_url[:-3]
    return base_url.rstrip("/")


def port_from_url(base_url: str, default: int) -> int:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.port:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    if parsed.scheme == "http":
        return 80
    return default


def target_is_ready(target: Target, timeout: float = 2.0) -> bool:
    for url in (f"{base_root(target.base_url)}/health", f"{target.base_url}/models"):
        status, _body = get_status(url, timeout)
        if status == 200:
            return True
    return False


def wait_ready(target: Target, startup_timeout: float) -> None:
    deadline = time.time() + startup_timeout
    last = ""
    while time.time() < deadline:
        if target.process and target.process.poll() is not None:
            log_hint = f" log={target.log_path}" if target.log_path else ""
            raise RuntimeError(f"{target.name} server exited with code {target.process.returncode}.{log_hint}")
        for url in (f"{base_root(target.base_url)}/health", f"{target.base_url}/models"):
            status, body = get_status(url, 2.0)
            last = f"{url}: {status} {body[:200]}"
            if status == 200:
                return
        time.sleep(1.0)
    log_hint = f" log={target.log_path}" if target.log_path else ""
    raise RuntimeError(f"{target.name} did not become ready within {startup_timeout:.0f}s ({last}).{log_hint}")


def start_target_if_needed(target: Target, args: argparse.Namespace) -> None:
    already_ready = target_is_ready(target)
    if args.start == "never":
        if not already_ready:
            raise RuntimeError(f"{target.name} is not reachable at {target.base_url} and --start=never was set")
        return
    if already_ready and args.start != "always":
        return
    if not target.command:
        raise RuntimeError(f"{target.name} has no launch command")

    log_file = tempfile.NamedTemporaryFile(
        prefix=f"stress-{target.name}-",
        suffix=".log",
        mode="w",
        encoding="utf-8",
        delete=False,
    )
    target.log_path = Path(log_file.name)
    target.process = subprocess.Popen(
        target.command,
        cwd=str(target.cwd),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    target.started_by_us = True
    wait_ready(target, args.startup_timeout)


def stop_target(target: Target) -> None:
    if not target.process or target.process.poll() is not None:
        return
    target.process.terminate()
    try:
        target.process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        target.process.kill()
        target.process.wait(timeout=15)


def build_ds4_command(args: argparse.Namespace) -> list[str]:
    if args.ds4_cmd:
        return shlex.split(args.ds4_cmd.format(port=port_from_url(args.ds4_base_url, 8000)))
    return [
        args.ds4_binary,
        "--model",
        args.ds4_model_path,
        "--ctx",
        str(args.server_ctx),
        "--tokens",
        str(args.server_default_tokens),
        "--port",
        str(port_from_url(args.ds4_base_url, 8000)),
        *flatten_extra_args(args.ds4_extra_arg),
    ]


def build_llama_command(args: argparse.Namespace) -> list[str]:
    if args.llama_cmd:
        return shlex.split(args.llama_cmd.format(port=port_from_url(args.llama_base_url, 8080)))
    cmd = [
        args.llama_binary,
        "-hf",
        args.llama_hf_model,
        "-c",
        str(args.server_ctx),
        "--port",
        str(port_from_url(args.llama_base_url, 8080)),
        "--jinja",
    ]
    if args.llama_ngl is not None:
        cmd.extend(["-ngl", str(args.llama_ngl)])
    cmd.extend(flatten_extra_args(args.llama_extra_arg))
    return cmd


def selected_cases(args: argparse.Namespace) -> list[Case]:
    cases = make_cases()
    families = set(split_csv(args.families)) if args.families != "all" else set()
    names = set(args.case or [])
    out = [
        c for c in cases
        if (not families or c.family in families) and (not names or c.name in names)
    ]
    missing = names - {c.name for c in cases}
    if missing:
        raise SystemExit(f"unknown case(s): {', '.join(sorted(missing))}")
    known_families = {c.family for c in cases}
    unknown_families = families - known_families
    if unknown_families:
        raise SystemExit(f"unknown family/families: {', '.join(sorted(unknown_families))}")
    return out


def run_llguidance_oracle(cases: list[Case], args: argparse.Namespace) -> Counts:
    counts = Counts()
    if args.oracle == "never":
        return counts
    try:
        import llguidance  # type: ignore
    except ModuleNotFoundError:
        msg = "SKIP oracle/llguidance: python package is not importable"
        if args.oracle == "require":
            print(msg, file=sys.stderr)
            counts.failed += 1
        elif args.verbose:
            print(msg)
            counts.skipped += 1
        return counts

    try:
        tok = llguidance.LLTokenizer("byte")
    except Exception as exc:
        msg = f"SKIP oracle/llguidance: failed to create byte tokenizer: {exc}"
        if args.oracle == "require":
            print(msg, file=sys.stderr)
            counts.failed += 1
        elif args.verbose:
            print(msg)
            counts.skipped += 1
        return counts

    for case in cases:
        if case.family == "json_schema":
            grammar = llguidance.LLMatcher.grammar_from_json_schema(case.schema)
        elif case.family == "json_object":
            grammar = llguidance.LLMatcher.grammar_from_json_schema({"type": "object"})
        elif case.family == "regex":
            grammar = llguidance.LLMatcher.grammar_from_regex(case.data)
        elif case.family == "lark":
            grammar = llguidance.LLMatcher.grammar_from_lark(case.data)
        elif case.family == "llguidance":
            grammar = llguidance.grammar_from("llguidance", case.data)
        else:
            counts.skipped += 1
            continue

        label = f"oracle/{case.family}/{case.name}"
        try:
            err = llguidance.LLMatcher.validate_grammar(grammar, tok)
            if err:
                raise RuntimeError(err)
            if case.oracle_sample is not None:
                matcher = llguidance.LLMatcher(tok, grammar)
                for token in tok.tokenize_str(case.oracle_sample):
                    bias = matcher.compute_logit_bias()
                    if token >= len(bias) or bias[token] != 200:
                        raise RuntimeError(f"sample token {token} is not allowed")
                    if not matcher.consume_token(token):
                        raise RuntimeError(f"sample token {token} was rejected")
                if not matcher.is_accepting():
                    raise RuntimeError("sample did not leave matcher in accepting state")
            print(f"PASS {label}")
            counts.passed += 1
        except Exception as exc:
            print(f"FAIL {label}: {exc}", file=sys.stderr)
            counts.failed += 1
            if args.fail_fast:
                raise
    return counts


def run_target(target: Target, cases: list[Case], args: argparse.Namespace) -> Counts:
    counts = Counts()
    start_target_if_needed(target, args)
    apis = split_csv(args.apis)
    for repeat in range(args.repeat):
        for api in apis:
            for case in cases:
                label = f"{target.name}/{api}/{case.family}/{case.name}"
                if args.repeat > 1:
                    label = f"{label}#{repeat + 1}"
                t0 = time.time()
                try:
                    value = check_case(target, api, case, args)
                    elapsed = time.time() - t0
                    if args.verbose:
                        print(f"PASS {label} {elapsed:.2f}s {value}")
                    else:
                        print(f"PASS {label} {elapsed:.2f}s")
                    counts.passed += 1
                except UnsupportedCase as exc:
                    elapsed = time.time() - t0
                    msg = f"SKIP {label} {elapsed:.2f}s: {exc}"
                    if args.strict_skips:
                        print(msg, file=sys.stderr)
                        counts.failed += 1
                        if args.fail_fast:
                            raise
                    else:
                        print(msg)
                        counts.skipped += 1
                except Exception as exc:
                    counts.failed += 1
                    print(f"FAIL {label}: {exc}", file=sys.stderr)
                    if args.fail_fast:
                        raise
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", default="ds4,llama", help="Comma-separated targets: ds4,llama")
    p.add_argument("--apis", default="chat,responses", help="Comma-separated APIs: chat,responses")
    p.add_argument("--families", default="all", help="Comma-separated families or all")
    p.add_argument("--case", action="append", help="Run only this case name; may repeat")
    p.add_argument("--list-cases", action="store_true", help="Print selected cases and exit")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--startup-timeout", type=float, default=900.0)
    p.add_argument("--start", choices=["missing", "never", "always"], default="missing")
    p.add_argument("--stop-started", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict-skips", action="store_true", help="Treat unsupported matrix entries as failures")
    p.add_argument("--force-extensions", action="store_true", help="Send regex/lark/llguidance as experimental response_format types")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--api-key", help="Optional bearer token for non-local OpenAI-compatible servers")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--json-object-schema", action="store_true", help="Attach {'type':'object'} to json_object mode")
    p.add_argument("--oracle", choices=["auto", "never", "require"], default="auto", help="Local Python llguidance grammar validation")

    p.add_argument("--server-ctx", type=int, default=8192)
    p.add_argument("--server-default-tokens", type=int, default=384)

    p.add_argument("--ds4-base-url", default=DEFAULT_DS4_BASE_URL)
    p.add_argument("--ds4-model", default="ds4")
    p.add_argument("--ds4-binary", default="./ds4-server")
    p.add_argument("--ds4-model-path", default="ds4flash.gguf")
    p.add_argument("--ds4-cmd", help="Override ds4 launch command; {port} is expanded")
    p.add_argument("--ds4-extra-arg", action="append", help="Extra ds4-server args; may repeat")

    p.add_argument("--llama-base-url", default=DEFAULT_LLAMA_BASE_URL)
    p.add_argument("--llama-model", default=DEFAULT_LLAMA_HF_MODEL)
    p.add_argument("--llama-binary", default="llama-server")
    p.add_argument("--llama-hf-model", default=DEFAULT_LLAMA_HF_MODEL)
    p.add_argument("--llama-cmd", help="Override llama launch command; {port} is expanded")
    p.add_argument("--llama-extra-arg", action="append", help="Extra llama-server args; may repeat")
    p.add_argument("--llama-ngl", type=int, default=999, help="llama.cpp GPU layers; set -1 to omit")
    p.add_argument(
        "--llama-chat-schema-style",
        choices=["openai", "flat"],
        default="flat",
        help="json_schema shape for llama.cpp chat response_format",
    )
    p.add_argument("--llama-disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.llama_ngl is not None and args.llama_ngl < 0:
        args.llama_ngl = None

    repo = Path(__file__).resolve().parent
    targets_requested = split_csv(args.targets)
    unknown_targets = set(targets_requested) - {"ds4", "llama"}
    if unknown_targets:
        raise SystemExit(f"unknown target(s): {', '.join(sorted(unknown_targets))}")

    cases = selected_cases(args)
    if args.list_cases:
        for case in cases:
            print(f"{case.family}\t{case.name}")
        return 0

    total = Counts()

    oracle_counts = run_llguidance_oracle(cases, args)
    total.passed += oracle_counts.passed
    total.failed += oracle_counts.failed
    total.skipped += oracle_counts.skipped

    target_map: dict[str, Target] = {
        "ds4": Target(
            name="ds4",
            base_url=args.ds4_base_url.rstrip("/"),
            model=args.ds4_model,
            command=build_ds4_command(args),
            cwd=repo,
            supports_response_format_extensions=True,
            supports_grammar_extension=False,
        ),
        "llama": Target(
            name="llama",
            base_url=args.llama_base_url.rstrip("/"),
            model=args.llama_model,
            command=build_llama_command(args),
            cwd=repo,
            supports_response_format_extensions=False,
            supports_grammar_extension=True,
        ),
    }

    try:
        for name in targets_requested:
            target = target_map[name]
            counts = run_target(target, cases, args)
            total.passed += counts.passed
            total.failed += counts.failed
            total.skipped += counts.skipped
            if args.stop_started and target.started_by_us:
                stop_target(target)
    finally:
        for target in target_map.values():
            if args.stop_started and target.started_by_us:
                stop_target(target)

    print(f"SUMMARY pass={total.passed} fail={total.failed} skip={total.skipped}")
    return 1 if total.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
