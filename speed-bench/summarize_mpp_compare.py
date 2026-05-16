#!/usr/bin/env python3
"""Summarize DS4 Metal Tensor comparator logs.

This parses stderr/stdout from runs with DS4_METAL_MPP_COMPARE_ROUTE set. The
comparator reports local projection deltas between the legacy path and the
candidate Tensor path; this helper turns those raw lines into persistent
Markdown/JSON summaries for prefill optimization notes.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


COMPARE_RE = re.compile(
    r"Metal Tensor compare route=(?P<route>\w+) module=(?P<module>.*?) "
    r"shape=(?P<dim0>\d+)x(?P<dim1>\d+)x(?P<dim2>\d+) "
    r"max_abs=(?P<max_abs>[0-9.eE+-]+) rms=(?P<rms>[0-9.eE+-]+) "
    r"nonfinite=(?P<nonfinite>\d+) max_index=(?P<max_index>\d+)"
)
DELTA_RE = re.compile(
    r"Metal Tensor compare route=(?P<route>\w+) module=(?P<module>.*?) "
    r"largest deltas:(?P<deltas>.*)"
)
DELTA_ITEM_RE = re.compile(
    r"idx=(?P<idx>\d+) ref=(?P<ref>[0-9.eE+-]+) "
    r"cand=(?P<cand>[0-9.eE+-]+) abs=(?P<abs>[0-9.eE+-]+)"
)
BREACH_RE = re.compile(
    r"Metal Tensor compare route=(?P<route>\w+) module=(?P<module>.*?) "
    r"exceeded target max_abs<=0.001 rms<=0.0001"
)
LIMIT_RE = re.compile(
    r"Metal Tensor compare reached DS4_METAL_MPP_COMPARE_MAX=(?P<max>\d+) "
    r"without a target breach"
)
LAYER_RE = re.compile(r"layer=(?P<layer>\d+)")


@dataclass
class DeltaItem:
    idx: int
    ref: float
    cand: float
    abs_delta: float


@dataclass
class CompareItem:
    source: Path
    route: str
    module: str
    dim0: int
    dim1: int
    dim2: int
    max_abs: float
    rms: float
    nonfinite: int
    max_index: int
    deltas: list[DeltaItem] = field(default_factory=list)

    @property
    def layer(self) -> int | None:
        match = LAYER_RE.search(self.module)
        return int(match.group("layer")) if match else None

    @property
    def shape(self) -> str:
        return f"{self.dim0}x{self.dim1}x{self.dim2}"


@dataclass
class CompareSummary:
    items: list[CompareItem] = field(default_factory=list)
    breaches: list[dict[str, Any]] = field(default_factory=list)
    limit_hits: list[dict[str, Any]] = field(default_factory=list)


def parse_log(path: Path) -> CompareSummary:
    summary = CompareSummary()
    pending: dict[tuple[str, str], CompareItem] = {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if match := COMPARE_RE.search(line):
            item = CompareItem(
                source=path,
                route=match.group("route"),
                module=match.group("module"),
                dim0=int(match.group("dim0")),
                dim1=int(match.group("dim1")),
                dim2=int(match.group("dim2")),
                max_abs=float(match.group("max_abs")),
                rms=float(match.group("rms")),
                nonfinite=int(match.group("nonfinite")),
                max_index=int(match.group("max_index")),
            )
            summary.items.append(item)
            pending[(item.route, item.module)] = item
        if match := DELTA_RE.search(line):
            key = (match.group("route"), match.group("module"))
            item = pending.get(key)
            if item is not None:
                item.deltas = [
                    DeltaItem(
                        idx=int(delta.group("idx")),
                        ref=float(delta.group("ref")),
                        cand=float(delta.group("cand")),
                        abs_delta=float(delta.group("abs")),
                    )
                    for delta in DELTA_ITEM_RE.finditer(match.group("deltas"))
                ]
        if match := BREACH_RE.search(line):
            summary.breaches.append(
                {
                    "source": str(path),
                    "route": match.group("route"),
                    "module": match.group("module"),
                }
            )
        if match := LIMIT_RE.search(line):
            summary.limit_hits.append(
                {
                    "source": str(path),
                    "max": int(match.group("max")),
                }
            )
    return summary


def merge_summaries(summaries: list[CompareSummary]) -> CompareSummary:
    merged = CompareSummary()
    for summary in summaries:
        merged.items.extend(summary.items)
        merged.breaches.extend(summary.breaches)
        merged.limit_hits.extend(summary.limit_hits)
    return merged


def pct(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


def item_to_json(item: CompareItem) -> dict[str, Any]:
    return {
        "source": str(item.source),
        "route": item.route,
        "module": item.module,
        "layer": item.layer,
        "shape": item.shape,
        "max_abs": item.max_abs,
        "rms": item.rms,
        "nonfinite": item.nonfinite,
        "max_index": item.max_index,
        "largest_deltas": [
            {
                "idx": delta.idx,
                "ref": delta.ref,
                "cand": delta.cand,
                "abs": delta.abs_delta,
            }
            for delta in item.deltas
        ],
    }


def as_json(summary: CompareSummary, *, max_abs_target: float, rms_target: float) -> dict[str, Any]:
    route_counts = Counter(item.route for item in summary.items)
    layer_counts = Counter(item.layer for item in summary.items if item.layer is not None)
    route_worst: dict[str, dict[str, Any]] = {}
    for route in sorted(route_counts):
        route_items = [item for item in summary.items if item.route == route]
        route_worst[route] = {
            "count": len(route_items),
            "worst_max_abs": item_to_json(max(route_items, key=lambda item: item.max_abs)),
            "worst_rms": item_to_json(max(route_items, key=lambda item: item.rms)),
        }
    threshold_breaches = [
        item
        for item in summary.items
        if item.nonfinite or item.max_abs > max_abs_target or item.rms > rms_target
    ]
    return {
        "targets": {
            "max_abs": max_abs_target,
            "rms": rms_target,
        },
        "count": len(summary.items),
        "route_counts": dict(route_counts),
        "layer_counts": {str(layer): count for layer, count in sorted(layer_counts.items())},
        "breaches": summary.breaches,
        "limit_hits": summary.limit_hits,
        "threshold_breaches": [item_to_json(item) for item in threshold_breaches],
        "top_max_abs": [
            item_to_json(item)
            for item in sorted(summary.items, key=lambda item: item.max_abs, reverse=True)
        ],
        "top_rms": [
            item_to_json(item)
            for item in sorted(summary.items, key=lambda item: item.rms, reverse=True)
        ],
        "route_worst": route_worst,
    }


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|")


def render_item_row(item: CompareItem) -> str:
    return (
        "| "
        f"`{markdown_escape(item.route)}` | "
        f"`{markdown_escape(item.module)}` | "
        f"{item.layer if item.layer is not None else 'n/a'} | "
        f"`{item.shape}` | "
        f"{item.max_abs:.6g} | "
        f"{item.rms:.6g} | "
        f"{item.nonfinite} | "
        f"{item.max_index} |"
    )


def render_markdown(
    summary: CompareSummary,
    *,
    max_abs_target: float,
    rms_target: float,
    top: int,
) -> str:
    route_counts = Counter(item.route for item in summary.items)
    layer_counts = Counter(item.layer for item in summary.items if item.layer is not None)
    threshold_breaches = [
        item
        for item in summary.items
        if item.nonfinite or item.max_abs > max_abs_target or item.rms > rms_target
    ]

    blocks: list[str] = [
        "# DS4 Metal Tensor Comparator Summary",
        "",
        f"Parsed comparisons: `{len(summary.items)}`",
        f"Targets: max abs `<= {max_abs_target:.6g}`, RMS `<= {rms_target:.6g}`",
        "",
    ]
    if route_counts:
        blocks.append(
            "Routes: "
            + ", ".join(f"`{route}`={count}" for route, count in route_counts.most_common())
        )
        blocks.append("")
    if layer_counts:
        blocks.append(
            "Layers with comparisons: "
            + ", ".join(f"`{layer}`={count}" for layer, count in sorted(layer_counts.items()))
        )
        blocks.append("")

    if threshold_breaches:
        blocks.extend(
            [
                "## Target Breaches",
                "",
                "| Route | Module | Layer | Shape | Max abs | RMS | Nonfinite | Max index |",
                "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in sorted(threshold_breaches, key=lambda item: item.max_abs, reverse=True):
            blocks.append(render_item_row(item))
        blocks.append("")
    else:
        blocks.extend(["## Target Breaches", "", "None.", ""])

    if summary.breaches:
        blocks.extend(["Comparator breach lines:", ""])
        for breach in summary.breaches:
            blocks.append(
                f"- `{markdown_escape(breach['route'])}` "
                f"`{markdown_escape(breach['module'])}` in `{markdown_escape(breach['source'])}`"
            )
        blocks.append("")
    if summary.limit_hits:
        blocks.extend(["Comparator limit lines:", ""])
        for hit in summary.limit_hits:
            blocks.append(
                f"- reached `DS4_METAL_MPP_COMPARE_MAX={hit['max']}` without breach "
                f"in `{markdown_escape(hit['source'])}`"
            )
        blocks.append("")

    blocks.extend(
        [
            "## Worst Max Abs",
            "",
            "| Route | Module | Layer | Shape | Max abs | RMS | Nonfinite | Max index |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in sorted(summary.items, key=lambda item: item.max_abs, reverse=True)[:top]:
        blocks.append(render_item_row(item))
    blocks.append("")

    blocks.extend(
        [
            "## Worst RMS",
            "",
            "| Route | Module | Layer | Shape | Max abs | RMS | Nonfinite | Max index |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in sorted(summary.items, key=lambda item: item.rms, reverse=True)[:top]:
        blocks.append(render_item_row(item))
    blocks.append("")

    blocks.extend(
        [
            "## Route Summary",
            "",
            "| Route | Count | Share | Worst max abs | Worst max abs module | Worst RMS | Worst RMS module |",
            "| --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for route, count in route_counts.most_common():
        route_items = [item for item in summary.items if item.route == route]
        max_abs_item = max(route_items, key=lambda item: item.max_abs)
        rms_item = max(route_items, key=lambda item: item.rms)
        blocks.append(
            "| "
            f"`{markdown_escape(route)}` | "
            f"{count} | "
            f"{pct(count, len(summary.items)):.1f}% | "
            f"{max_abs_item.max_abs:.6g} | "
            f"`{markdown_escape(max_abs_item.module)}` | "
            f"{rms_item.rms:.6g} | "
            f"`{markdown_escape(rms_item.module)}` |"
        )
    blocks.append("")

    top_delta_items = [item for item in sorted(summary.items, key=lambda item: item.max_abs, reverse=True) if item.deltas]
    if top_delta_items:
        blocks.extend(["## Largest Delta Details", ""])
        for item in top_delta_items[: min(top, 5)]:
            blocks.append(
                f"### `{markdown_escape(item.route)}` `{markdown_escape(item.module)}`"
            )
            blocks.append("")
            blocks.append("| Idx | Ref | Cand | Abs |")
            blocks.append("| ---: | ---: | ---: | ---: |")
            for delta in item.deltas:
                blocks.append(
                    f"| {delta.idx} | {delta.ref:.6g} | {delta.cand:.6g} | {delta.abs_delta:.6g} |"
                )
            blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="comparator log/stderr files")
    parser.add_argument("--top", type=int, default=20, help="number of rows to show in top tables")
    parser.add_argument(
        "--max-abs-target",
        type=float,
        default=1.0e-3,
        help="local comparator max-abs target",
    )
    parser.add_argument(
        "--rms-target",
        type=float,
        default=1.0e-4,
        help="local comparator RMS target",
    )
    parser.add_argument("--output", type=Path, help="write Markdown summary here")
    parser.add_argument("--json-output", type=Path, help="write JSON summary here")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top < 1:
        raise SystemExit("--top must be >= 1")
    summaries = [parse_log(path) for path in args.logs]
    summary = merge_summaries(summaries)
    markdown = render_markdown(
        summary,
        max_abs_target=args.max_abs_target,
        rms_target=args.rms_target,
        top=args.top,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                as_json(
                    summary,
                    max_abs_target=args.max_abs_target,
                    rms_target=args.rms_target,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
