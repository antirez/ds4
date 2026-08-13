#!/usr/bin/env python3
"""No-model source-order guard for the persistent Metal support mmap view."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def between(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[begin:finish]


def main() -> None:
    engine = (ROOT / "ds4.c").read_text()
    metal = (ROOT / "ds4_metal.m").read_text()
    header = (ROOT / "ds4_gpu.h").read_text()

    close = between(engine, "void ds4_engine_close(ds4_engine *e)", "static bool ds4_dspark_stats_enabled")
    assert "ds4_gpu_release_persistent_support_map();" in close
    assert close.index("ds4_gpu_release_persistent_support_map();") < close.index(
        "if (e->mtp_model.map) model_close(&e->mtp_model);"
    )

    release = between(
        metal,
        "void ds4_gpu_release_persistent_support_map(void)",
        "static id<MTLBuffer> ds4_gpu_wrap_model_range",
    )
    assert "ds4_gpu_synchronize()" in release
    assert release.index("ds4_gpu_synchronize()") < release.index("g_support_model_buffer = nil;")
    assert "newBufferWithBytesNoCopy" in metal
    assert "ds4_gpu_register_persistent_support_map" in header
    assert "ds4_gpu_release_persistent_support_map" in header

    bind = engine.index("dspark_weights_bind_optional(&e->dspark_weights")
    detect = engine.index(
        "support_model_detect(&e->mtp_model, &e->support_stages, &dspark);",
        engine.index("if (opt->mtp_path && opt->mtp_path[0] &&", bind),
    )
    startup_policy = engine.index(
        "!support_kind_allowed_for_startup(\n"
        "                    e->dspark, e->ssd_streaming, e->support_kind)",
        detect,
    )
    legacy_bind = engine.index(
        "if (e->support_kind == DS4_SUPPORT_MTP_LEGACY)", startup_policy
    )
    readiness = engine.index(
        "!dspark_weights_ready_for_enabled_runtime(\n"
        "                        &e->dspark_weights)",
        bind,
    )
    persistent_register = engine.index(
        "ds4_gpu_register_persistent_support_map(", readiness
    )
    assert detect < startup_policy < legacy_bind
    assert bind < readiness < persistent_register
    print("test_persistent_support_lifecycle: PASS")


if __name__ == "__main__":
    main()
