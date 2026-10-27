#!/usr/bin/env python3
"""Compare V4.1 resize/layout planning with the released image_processor.py.

Build ds4_image.c as a shared library and pass it with --library. This test
uses only the reference's pure resize functions; PyTorch is not required.
"""
import argparse
import ast
import ctypes as C
import json
import math
from pathlib import Path
from types import SimpleNamespace


class Image(C.Structure):
    _fields_ = [("width", C.c_uint32), ("height", C.c_uint32),
                ("rgb", C.POINTER(C.c_uint8)), ("fingerprint", C.c_uint8 * 32)]


class Patches(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in (
        "content_width", "content_height", "padded_width", "padded_height",
        "grid_width", "grid_height", "llm_grid_width", "llm_grid_height", "patch_count")]
    _fields_ += [("patches", C.POINTER(C.c_float))]


class Layout(C.Structure):
    _fields_ = [("token_count", C.c_uint32), ("image_count", C.c_uint32),
                ("types", C.POINTER(C.c_uint8)), ("perm", C.POINTER(C.c_uint32))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--reference", required=True, help="official checkpoint directory")
    args = parser.parse_args()
    source = Path(args.reference) / "inference/image_processor.py"
    names = {"num_image_tokens", "llm_grid", "solve_resize_ratio", "safe_resize", "plan_image_grid"}
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    namespace = {"math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    config = json.loads((Path(args.reference) / "config.json").read_text())["vision_config"]
    params = SimpleNamespace(vision_patch_size=config["patch_size"],
        vision_downsample_ratio=config["downsample_ratio"],
        vision_max_n_token=config["max_image_tokens"], vision_min_pixels=config["min_pixels"],
        vision_max_wh_ratio=config["max_wh_ratio"])
    lib = C.CDLL(args.library)
    lib.ds4_image_preprocess_deepseek41.argtypes = [C.POINTER(Patches), C.POINTER(Image), C.c_char_p, C.c_size_t]
    lib.ds4_deepseek4_image_patches_free.argtypes = [C.POINTER(Patches)]
    lib.ds4_deepseek41_image_layout_build.argtypes = [C.POINTER(Layout), C.c_uint32, C.c_uint32, C.c_char_p, C.c_size_t]
    lib.ds4_deepseek4_image_layout_free.argtypes = [C.POINTER(Layout)]
    cases = [(1, 1), (17, 9), (9, 17), (543, 545), (544, 544), (777, 391),
             (1920, 1080), (1080, 1920), (2048, 2048), (4096, 64), (64, 4096)]
    for width, height in cases:
        rgb = (C.c_uint8 * (width * height * 3))()
        image = Image(width, height, rgb)
        patches, layout, err = Patches(), Layout(), C.create_string_buffer(256)
        try:
            assert lib.ds4_image_preprocess_deepseek41(C.byref(patches), C.byref(image), err, len(err)), err.value
            expected = namespace["plan_image_grid"](width, height, params)
            actual = (patches.llm_grid_height, patches.llm_grid_width,
                      patches.padded_height, patches.padded_width)
            assert actual == expected, (width, height, actual, expected)
            h, w = actual[:2]
            assert lib.ds4_deepseek41_image_layout_build(C.byref(layout), h, w, err, len(err)), err.value
            assert layout.token_count == h * (w + 1) + 2 <= 1024
            assert layout.image_count == h * w
            assert list(layout.types[:layout.token_count]) == [0] + ([2] * w + [3]) * h + [4]
            assert list(layout.perm[:layout.image_count]) == list(range(h * w))
            assert patches.patch_count == patches.grid_width * patches.grid_height
            print(f"{width}x{height}: resize {actual[3]}x{actual[2]}, {layout.token_count} tokens: exact")
        finally:
            lib.ds4_deepseek4_image_patches_free(C.byref(patches))
            lib.ds4_deepseek4_image_layout_free(C.byref(layout))
    for h, w in [(0, 1), (1, 0), (32, 32), (0xffffffff, 0xffffffff)]:
        layout, err = Layout(), C.create_string_buffer(256)
        assert not lib.ds4_deepseek41_image_layout_build(C.byref(layout), h, w, err, len(err))
        assert not layout.types and not layout.perm
    print("V4.1 official resize plans, row-major layout, budget and invalid grids: PASS")


if __name__ == "__main__":
    main()
