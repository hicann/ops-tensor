#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

"""Compare QGMM MX NPU output with the generated NumPy golden result."""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Verify QGMM MX output")
    parser.add_argument("golden")
    parser.add_argument("actual")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()
    golden = np.fromfile(args.golden, dtype=np.float32)
    actual = np.fromfile(args.actual, dtype=np.float32)
    if golden.shape != actual.shape:
        raise ValueError(
            f"output size mismatch: actual={actual.size}, golden={golden.size}"
        )
    close = np.isclose(actual, golden, rtol=args.rtol, atol=args.atol, equal_nan=False)
    if not np.all(close):
        mismatch = np.flatnonzero(~close)
        index = int(mismatch[0])
        max_error = float(np.max(np.abs(actual - golden)))
        raise ValueError(
            f"mismatch at {index}: expected={golden[index]}, actual={actual[index]}, "
            f"mismatches={mismatch.size}/{actual.size}, max_abs_error={max_error}"
        )
    max_error = float(np.max(np.abs(actual - golden))) if actual.size else 0.0
    print(f"[PASS] {actual.size} FP32 outputs, max_abs_error={max_error}")


if __name__ == "__main__":
    main()
