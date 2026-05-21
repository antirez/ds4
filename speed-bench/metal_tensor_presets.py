"""Named Metal Tensor prefill candidate environment presets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidatePreset:
    label: str
    env: dict[str, str]
    description: str


CANDIDATE_PRESETS: dict[str, CandidatePreset] = {
    "mpp-fast": CandidatePreset(
        label="mpp-fast",
        env={"DS4_METAL_MPP_FAST": "1"},
        description="All-routed-MoE fast Tensor profile.",
    ),
    "mpp-fast-skip-down26-29-30": CandidatePreset(
        label="mpp-fast-skip-down26-29-30",
        env={
            "DS4_METAL_MPP_FAST": "1",
            "DS4_METAL_MPP_MOE_DOWN_FILTER": "layer=0-25,layer=27-28,layer=31-42",
        },
        description="Best current prefill-first default-off candidate.",
    ),
    "mpp-fast-skip-down26-29-30-mid-f32": CandidatePreset(
        label="mpp-fast-skip-down26-29-30-mid-f32",
        env={
            "DS4_METAL_MPP_FAST": "1",
            "DS4_METAL_MPP_MOE_DOWN_FILTER": "layer=0-25,layer=27-28,layer=31-42",
            "DS4_METAL_MOE_MID_F32": "1",
        },
        description="Best current balanced default-off candidate for flatter generation timing.",
    ),
    "mpp-fast-continuation-chunks": CandidatePreset(
        label="mpp-fast-continuation-chunks",
        env={
            "DS4_METAL_MPP_FAST": "1",
            "DS4_METAL_MPP_MOE_GATE_FILTER": "layer=15-42,pos=512,pos=1024,pos=2048,pos=4096",
            "DS4_METAL_MPP_MOE_UP_FILTER": "layer=15-42,pos=512,pos=1024,pos=2048,pos=4096",
            "DS4_METAL_MPP_MOE_DOWN_FILTER": "layer=12-42,pos=512,pos=1024,pos=2048,pos=4096",
        },
        description="Fast routed-MoE only for continuation prefill chunks; needs extra chunked drift coverage.",
    ),
    "experimental-moe-matmul": CandidatePreset(
        label="experimental-moe-matmul",
        env={"DS4_METAL_EXPERIMENTAL_MOE_MATMUL": "1"},
        description="Experimental all-layer routed-MoE matmul route.",
    ),
}


def preset_help() -> str:
    return "\n".join(
        f"  {name}: {preset.description}"
        for name, preset in sorted(CANDIDATE_PRESETS.items())
    )
