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

"""Compare the NPU FP16 output with the generated NumPy golden result."""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("actual")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=0.08)
    args = parser.parse_args()

    golden = np.fromfile(args.golden, dtype=np.float16)
    actual = np.fromfile(args.actual, dtype=np.float16)
    if golden.shape != actual.shape:
        raise ValueError(f"output size mismatch: {actual.size} != {golden.size}")

    close = np.isclose(actual.astype(np.float32), golden.astype(np.float32), rtol=args.rtol, atol=args.atol)
    if not np.all(close):
        index = int(np.flatnonzero(~close)[0])
        error = abs(float(actual[index]) - float(golden[index]))
        raise ValueError(
            f"mismatch at {index}: expected {golden[index]}, got {actual[index]}, abs_error={error}")
    max_error = float(np.max(np.abs(actual.astype(np.float32) - golden.astype(np.float32))))
    print(f"[PASS] {actual.size} FP16 outputs, max_abs_error={max_error}")


if __name__ == "__main__":
    main()
