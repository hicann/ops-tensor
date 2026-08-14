#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Compare BF16 with the HiFloat8 policy or compare Int32 exactly."""

import argparse
import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import numpy as np
import torch


POINT_ERROR_TOL = 1e-1
RATIO_POINT_ERROR_TOL = 1e-3
ERROR_RATIO_TOL = 1e-3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("actual")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    args = parser.parse_args()

    dtype = args.dtype.lower()
    is_int32 = dtype in ("int32", "int32_t")
    if not is_int32 and dtype not in ("bfloat16", "bfloat16_t", "bf16"):
        print(f"unsupported output dtype: {args.dtype}")
        return 1

    file_dtype = np.int32 if is_int32 else np.uint16
    golden = np.fromfile(args.golden, dtype=file_dtype)
    actual = np.fromfile(args.actual, dtype=file_dtype)
    if golden.shape != actual.shape:
        print(f"shape mismatch: golden={golden.shape}, actual={actual.shape}")
        return 1

    expected_size = args.batch * args.m * args.n
    if golden.size != expected_size:
        print(f"element count mismatch: expected={expected_size}, actual={golden.size}")
        return 1

    if is_int32:
        mismatch_count = int(np.count_nonzero(golden != actual))
        max_abs_diff = (
            int(np.max(np.abs(actual.astype(np.int64) - golden.astype(np.int64))))
            if expected_size
            else 0
        )
        print(f"max abs diff: {max_abs_diff}")
        print(f"mismatch count: {mismatch_count}/{expected_size}")
        if mismatch_count != 0:
            return 1
        print(f"PASS: verified {expected_size} Int32 elements")
        return 0

    shape = (args.batch, args.m, args.n)
    golden_tensor = torch.from_numpy(golden).view(torch.bfloat16).reshape(shape).float()
    actual_tensor = torch.from_numpy(actual).view(torch.bfloat16).reshape(shape).float()
    abs_diff = torch.abs(actual_tensor - golden_tensor)
    finite_mask = (
        torch.isfinite(golden_tensor)
        & torch.isfinite(actual_tensor)
        & torch.isfinite(abs_diff)
    )
    abs_golden = torch.abs(golden_tensor)
    rel_diff = torch.where(
        abs_golden > 0,
        abs_diff / abs_golden,
        torch.where(
            abs_diff == 0,
            torch.zeros_like(abs_diff),
            torch.full_like(abs_diff, float("inf")),
        ),
    )
    point_error_count = int(((rel_diff > POINT_ERROR_TOL) | ~finite_mask).sum().item())
    ratio_error_count = int(
        ((abs_diff > RATIO_POINT_ERROR_TOL) | ~finite_mask).sum().item()
    )
    error_ratio = ratio_error_count / expected_size if expected_size else 0.0

    print(f"max abs diff: {abs_diff.max().item() if expected_size else 0.0}")
    print(f"point error count(>{POINT_ERROR_TOL}): {point_error_count}/{expected_size}")
    print(
        f"ratio error count(>{RATIO_POINT_ERROR_TOL}): {ratio_error_count}/{expected_size}, "
        f"error ratio: {error_ratio:.6f}"
    )
    if point_error_count != 0 or error_ratio > ERROR_RATIO_TOL:
        return 1
    print(f"PASS: verified {expected_size} BF16 elements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
