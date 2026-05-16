#!/usr/bin/env python3
"""Summarize DS4 Metal stage-profile logs.

This parses stderr/stdout from runs with profiling envs such as
DS4_METAL_LAYER_PROFILE=1, DS4_METAL_MOE_STAGE_PROFILE=1, and
DS4_METAL_Q8_PREFILL_PROFILE=1. The output is intentionally simple Markdown so
local optimization notes can be pasted into the experiment log.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LAYER_STAGE_RE = re.compile(
    r"metal layer stage part=(?P<part>\w+) layer=(?P<layer>\d+) "
    r"pos=(?P<pos>\d+) tokens=(?P<tokens>\d+) "
    r"(?P<stage>[a-z_]+)=(?P<ms>[0-9.]+) ms"
)
MOE_STAGE_RE = re.compile(
    r"Metal routed MoE stage layer=(?P<layer>\d+) tokens=(?P<tokens>\d+) "
    r"pairs=(?P<pairs>\d+) experts=(?P<experts>\d+) .*? "
    r"path=(?P<path>\w+) mpp=(?P<mpp>[0-9/]+) tile=(?P<tile>[0-9/]+) "
    r"mid=(?P<mid>\w+) (?P<stage>[a-z_]+)=(?P<ms>[0-9.]+) ms"
)
Q8_STAGE_RE = re.compile(
    r"Metal Q8_0 prefill profile layer=(?P<layer>\d+) pos=(?P<pos>\d+) "
    r"(?P<route>[a-z0-9_]+) in=(?P<input>\d+) out=(?P<output>\d+) "
    r"tok=(?P<tokens>\d+) (?P<ms>[0-9.]+) ms"
)
ATTN_OUTPUT_RE = re.compile(
    r"Metal attention output stage tokens=(?P<tokens>\d+) "
    r"(?P<stage>[a-z_]+)=(?P<ms>[0-9.]+) ms"
)
FLASH_ATTN_RE = re.compile(
    r"Metal FlashAttention prefill stage mode=(?P<mode>\w+) "
    r"tokens=(?P<tokens>\d+) comp=(?P<comp>\d+) keys=(?P<keys>\d+) "
    r"heads=(?P<heads>\d+) dim=(?P<dim>\d+) window=(?P<window>\d+) "
    r"ratio=(?P<ratio>\d+) (?P<stage>[a-z_]+)=(?P<ms>[0-9.]+) ms"
)
THROUGHPUT_RE = re.compile(
    r"prefill: (?P<prefill>[0-9.]+) t/s, generation: (?P<generation>[0-9.]+) t/s"
)


@dataclass
class StageSummary:
    total_ms: float = 0.0
    count: int = 0

    def add(self, ms: float) -> None:
        self.total_ms += ms
        self.count += 1

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0


@dataclass
class ProfileSummary:
    path: Path
    events: int = 0
    stages: dict[str, StageSummary] = field(default_factory=lambda: defaultdict(StageSummary))
    layers: dict[int, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    moe_paths: Counter[str] = field(default_factory=Counter)
    moe_mpp: Counter[str] = field(default_factory=Counter)
    moe_mpp_stages: dict[str, dict[str, StageSummary]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(StageSummary))
    )
    q8_shapes: dict[str, StageSummary] = field(default_factory=lambda: defaultdict(StageSummary))
    flash_shapes: dict[str, StageSummary] = field(default_factory=lambda: defaultdict(StageSummary))
    throughput: list[dict[str, float]] = field(default_factory=list)

    def add(self, key: str, layer: int | None, ms: float) -> None:
        self.events += 1
        self.stages[key].add(ms)
        if layer is not None:
            self.layers[layer][key] += ms


def parse_profile(path: Path) -> ProfileSummary:
    summary = ProfileSummary(path=path)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if match := LAYER_STAGE_RE.search(line):
            key = f"{match.group('part')}.{match.group('stage')}"
            summary.add(key, int(match.group("layer")), float(match.group("ms")))
            continue
        if match := MOE_STAGE_RE.search(line):
            key = f"moe_stage.{match.group('stage')}"
            summary.add(key, int(match.group("layer")), float(match.group("ms")))
            summary.moe_paths[match.group("path")] += 1
            mpp_mask = match.group("mpp")
            summary.moe_mpp[mpp_mask] += 1
            summary.moe_mpp_stages[mpp_mask][match.group("stage")].add(float(match.group("ms")))
            continue
        if match := Q8_STAGE_RE.search(line):
            key = f"q8.{match.group('route')}"
            ms = float(match.group("ms"))
            summary.add(key, int(match.group("layer")), ms)
            shape = (
                f"{match.group('route')} in={match.group('input')} "
                f"out={match.group('output')} tok={match.group('tokens')}"
            )
            summary.q8_shapes[shape].add(ms)
            continue
        if match := ATTN_OUTPUT_RE.search(line):
            key = f"attn_output.{match.group('stage')}"
            summary.add(key, None, float(match.group("ms")))
            continue
        if match := FLASH_ATTN_RE.search(line):
            key = f"flash_attn.{match.group('mode')}.{match.group('stage')}"
            ms = float(match.group("ms"))
            summary.add(key, None, ms)
            shape = (
                f"{match.group('mode')} tokens={match.group('tokens')} "
                f"comp={match.group('comp')} keys={match.group('keys')} "
                f"heads={match.group('heads')} dim={match.group('dim')} "
                f"window={match.group('window')} ratio={match.group('ratio')}"
            )
            summary.flash_shapes[shape].add(ms)
            continue
        if match := THROUGHPUT_RE.search(line):
            summary.throughput.append(
                {
                    "prefill_tps": float(match.group("prefill")),
                    "generation_tps": float(match.group("generation")),
                }
            )
    return summary


def pct(part: float, total: float) -> float:
    return 100.0 * part / total if total else 0.0


def as_json(summary: ProfileSummary) -> dict[str, Any]:
    total_ms = sum(stage.total_ms for stage in summary.stages.values())
    return {
        "path": str(summary.path),
        "events": summary.events,
        "total_ms": total_ms,
        "throughput": summary.throughput,
        "moe_paths": dict(summary.moe_paths),
        "moe_mpp": dict(summary.moe_mpp),
        "moe_mpp_stages": {
            mask: {
                stage_name: {
                    "total_ms": stage.total_ms,
                    "count": stage.count,
                    "avg_ms": stage.avg_ms,
                    "share_pct": pct(stage.total_ms, total_ms),
                }
                for stage_name, stage in sorted(
                    stages.items(),
                    key=lambda item: item[1].total_ms,
                    reverse=True,
                )
            }
            for mask, stages in sorted(summary.moe_mpp_stages.items())
        },
        "q8_shapes": {
            key: {
                "total_ms": shape.total_ms,
                "count": shape.count,
                "avg_ms": shape.avg_ms,
                "share_pct": pct(shape.total_ms, total_ms),
            }
            for key, shape in sorted(
                summary.q8_shapes.items(),
                key=lambda item: item[1].total_ms,
                reverse=True,
            )
        },
        "flash_shapes": {
            key: {
                "total_ms": shape.total_ms,
                "count": shape.count,
                "avg_ms": shape.avg_ms,
                "share_pct": pct(shape.total_ms, total_ms),
            }
            for key, shape in sorted(
                summary.flash_shapes.items(),
                key=lambda item: item[1].total_ms,
                reverse=True,
            )
        },
        "stages": {
            key: {
                "total_ms": stage.total_ms,
                "count": stage.count,
                "avg_ms": stage.avg_ms,
                "share_pct": pct(stage.total_ms, total_ms),
            }
            for key, stage in sorted(
                summary.stages.items(),
                key=lambda item: item[1].total_ms,
                reverse=True,
            )
        },
        "layers": {
            str(layer): {
                "total_ms": sum(counter.values()),
                "stages": dict(counter.most_common()),
            }
            for layer, counter in sorted(summary.layers.items())
        },
    }


def render_markdown(summaries: list[ProfileSummary], top: int) -> str:
    blocks: list[str] = [
        "# DS4 Metal Stage Profile Summary",
        "",
        "Note: some profile lines are nested views of the same work, such as",
        "`ffn.routed_moe` and `moe_stage.*`, or `attn.output_proj` and",
        "`attn_output.*`. Treat percentages as ranking aids, not exclusive",
        "wall-time shares.",
        "",
    ]
    for summary in summaries:
        total_ms = sum(stage.total_ms for stage in summary.stages.values())
        blocks.append(f"## {summary.path}")
        blocks.append("")
        if summary.throughput:
            last = summary.throughput[-1]
            blocks.append(
                "Throughput: "
                f"prefill `{last['prefill_tps']:.2f} t/s`, "
                f"generation `{last['generation_tps']:.2f} t/s`"
            )
            blocks.append("")
        blocks.append(f"Parsed events: `{summary.events}`, parsed stage total: `{total_ms:.3f} ms`")
        if summary.moe_paths:
            path_counts = ", ".join(f"`{name}`={count}" for name, count in summary.moe_paths.most_common())
            blocks.append(f"MoE paths: {path_counts}")
        if summary.moe_mpp:
            mpp_counts = ", ".join(f"`{name}`={count}" for name, count in summary.moe_mpp.most_common())
            blocks.append(f"MoE mpp masks: {mpp_counts}")
        blocks.append("")
        if summary.moe_mpp_stages:
            blocks.append("| MoE mpp mask | top stages | total ms | share |")
            blocks.append("| --- | --- | ---: | ---: |")
            mask_totals = [
                (sum(stage.total_ms for stage in stages.values()), mask, stages)
                for mask, stages in summary.moe_mpp_stages.items()
            ]
            for mask_total, mask, stages in sorted(mask_totals, reverse=True):
                top_stages = ", ".join(
                    f"`{name}`={stage.total_ms:.1f}"
                    for name, stage in sorted(
                        stages.items(),
                        key=lambda item: item[1].total_ms,
                        reverse=True,
                    )[:5]
                )
                blocks.append(
                    f"| `{mask}` | {top_stages} | {mask_total:.3f} | "
                    f"{pct(mask_total, total_ms):.1f}% |"
                )
            blocks.append("")
        blocks.append("| Stage | total ms | events | avg ms | share |")
        blocks.append("| --- | ---: | ---: | ---: | ---: |")
        for key, stage in sorted(
            summary.stages.items(),
            key=lambda item: item[1].total_ms,
            reverse=True,
        )[:top]:
            blocks.append(
                f"| `{key}` | {stage.total_ms:.3f} | {stage.count} | "
                f"{stage.avg_ms:.3f} | {pct(stage.total_ms, total_ms):.1f}% |"
            )
        blocks.append("")
        if summary.q8_shapes:
            blocks.append("| Q8 shape | total ms | events | avg ms | share |")
            blocks.append("| --- | ---: | ---: | ---: | ---: |")
            for key, shape in sorted(
                summary.q8_shapes.items(),
                key=lambda item: item[1].total_ms,
                reverse=True,
            )[:top]:
                blocks.append(
                    f"| `{key}` | {shape.total_ms:.3f} | {shape.count} | "
                    f"{shape.avg_ms:.3f} | {pct(shape.total_ms, total_ms):.1f}% |"
                )
            blocks.append("")
        if summary.flash_shapes:
            blocks.append("| FlashAttention shape | total ms | events | avg ms | share |")
            blocks.append("| --- | ---: | ---: | ---: | ---: |")
            for key, shape in sorted(
                summary.flash_shapes.items(),
                key=lambda item: item[1].total_ms,
                reverse=True,
            )[:top]:
                blocks.append(
                    f"| `{key}` | {shape.total_ms:.3f} | {shape.count} | "
                    f"{shape.avg_ms:.3f} | {pct(shape.total_ms, total_ms):.1f}% |"
                )
            blocks.append("")
        blocks.append("| Layer | total ms | top stages |")
        blocks.append("| ---: | ---: | --- |")
        layer_totals = [
            (sum(counter.values()), layer, counter)
            for layer, counter in summary.layers.items()
        ]
        for layer_total, layer, counter in sorted(layer_totals, reverse=True)[:top]:
            top_stages = ", ".join(f"`{name}`={value:.1f}" for name, value in counter.most_common(4))
            blocks.append(f"| {layer} | {layer_total:.3f} | {top_stages} |")
        blocks.append("")
    return "\n".join(blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="profile log/stderr files to summarize")
    parser.add_argument("--top", type=int, default=18, help="number of stages/layers to print")
    parser.add_argument("--output", type=Path, help="write Markdown summary to this file")
    parser.add_argument(
        "--json",
        "--json-output",
        dest="json",
        type=Path,
        help="write machine-readable summary JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = [parse_profile(path) for path in args.logs]
    markdown = render_markdown(summaries, args.top)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(markdown)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([as_json(summary) for summary in summaries], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
