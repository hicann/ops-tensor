#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Compare the NPU MX quant output (Y + Y_scale) with the golden result.

Y dtype follows A dtype: fp8_e4m3 or fp8_e5m2.
Y_scale is always fp8_e8m0.
"""

import argparse

import ml_dtypes
import numpy as np


FP8_E4M3FN = ml_dtypes.float8_e4m3fn
FP8_E5M2 = ml_dtypes.float8_e5m2
FP8_E8M0 = ml_dtypes.float8_e8m0fnu

_DTYPE_MAP = {"fp8_e4m3": FP8_E4M3FN, "fp8_e5m2": FP8_E5M2}


def load_y_as_fp32(path, dtype):
    raw = np.fromfile(path, dtype=np.uint8)
    return raw.view(_DTYPE_MAP[dtype]).astype(np.float32)


def load_y_scale_as_fp32(path):
    raw = np.fromfile(path, dtype=np.uint8)
    return raw.view(FP8_E8M0).astype(np.float32)


def compare(name, golden_path, actual_path, dtype, is_scale=False):
    if is_scale:
        golden = load_y_scale_as_fp32(golden_path)
        actual = load_y_scale_as_fp32(actual_path)
        atol = 1.0
        rtol = 1e-3
    else:
        golden = load_y_as_fp32(golden_path, dtype)
        actual = load_y_as_fp32(actual_path, dtype)
        atol = 1.0
        rtol = 1e-3

    n = min(golden.size, actual.size)
    if n == 0:
        raise ValueError(f"{name}: empty output")
    golden = golden[:n]
    actual = actual[:n]

    close = np.isclose(actual, golden, rtol=rtol, atol=atol)
    error_count = int(np.count_nonzero(~close))
    error_ratio = error_count / n
    max_abs_error = float(np.max(np.abs(golden - actual)))

    print(f"[verify] {name}: dtype={dtype}, elements={n}")
    print(f"  max abs diff: {max_abs_error:.6e}")
    print(f"  error count (atol>{atol}): {error_count}/{n}, ratio={error_ratio:.6f}")

    if not np.all(close):
        index = int(np.flatnonzero(~close)[0])
        raise ValueError(
            f"{name} mismatch at {index}: expected {golden[index]}, "
            f"got {actual[index]}, abs_error={abs(float(golden[index]) - float(actual[index]))}"
        )
    print(f"[PASS] {name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify NPU output (Y + Y_scale) against golden data."
    )
    parser.add_argument("golden_y", help="Path to golden Y binary")
    parser.add_argument("actual_y", help="Path to NPU Y binary")
    parser.add_argument("golden_y_scale", help="Path to golden Y_scale binary")
    parser.add_argument("actual_y_scale", help="Path to NPU Y_scale binary")
    parser.add_argument(
        "--dtype",
        required=True,
        choices=("fp8_e4m3", "fp8_e5m2"),
        help="Y output dtype (follows A dtype)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        compare("y", args.golden_y, args.actual_y, args.dtype, is_scale=False)
        compare(
            "y_scale",
            args.golden_y_scale,
            args.actual_y_scale,
            args.dtype,
            is_scale=True,
        )
    except (OSError, ValueError) as error:
        print(error)
        return 1
    print("[PASS] NPU results are consistent with golden.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
