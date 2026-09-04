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

"""Generate deterministic inputs for the executable public QBMM kernel API examples."""

import argparse
import os

import ml_dtypes
import numpy as np


MX_GROUP_SIZE = 32
MX_K_ALIGN = 64
MX_SCALE_VALUES = np.array([0.5, 1.0, 2.0], dtype=np.float32)
MIX_X1_SCALE_VALUES = np.array([0.5, 1.0, 2.0, 0.25], dtype=np.float32)
MIX_X2_SCALE_VALUES = np.array([2.0, 0.5, 0.25, 1.0, 4.0], dtype=np.float32)


def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def generate(args):
    if min(args.m, args.k, args.n) <= 0:
        raise ValueError("M, K, and N must be positive")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(args.output_dir), "output"), exist_ok=True)
    rng = np.random.default_rng(20260904)

    if args.variant in ("mx_without_batch", "mx_streamk"):
        if args.k % MX_K_ALIGN != 0:
            raise ValueError("MX variants require K to be a multiple of 64")
        a_fp32 = rng.choice(
            np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32), size=(args.m, args.k)
        )
        b_fp32 = rng.choice(
            np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32), size=(args.k, args.n)
        )
        a = a_fp32.astype(ml_dtypes.float8_e4m3fn)
        b_dtype = (
            ml_dtypes.float8_e5m2
            if args.variant == "mx_streamk"
            else ml_dtypes.float8_e4m3fn
        )
        b = b_fp32.astype(b_dtype)
        scale_k = align(args.k, MX_K_ALIGN) // MX_GROUP_SIZE
        scale_a_indices = (
            np.arange(args.m)[:, None] + np.arange(scale_k)[None, :]
        ) % MX_SCALE_VALUES.size
        scale_b_indices = (
            2 * np.arange(scale_k)[:, None] + np.arange(args.n)[None, :] + 1
        ) % MX_SCALE_VALUES.size
        scale_a = MX_SCALE_VALUES[scale_a_indices].astype(ml_dtypes.float8_e8m0fnu)
        scale_b = MX_SCALE_VALUES[scale_b_indices].astype(ml_dtypes.float8_e8m0fnu)
        scale_a_expanded = np.repeat(scale_a.astype(np.float32), MX_GROUP_SIZE, axis=1)[
            :, : args.k
        ]
        scale_b_expanded = np.repeat(scale_b.astype(np.float32), MX_GROUP_SIZE, axis=0)[
            : args.k, :
        ]
        golden = np.matmul(
            a.astype(np.float32) * scale_a_expanded,
            b.astype(np.float32) * scale_b_expanded,
        ).astype(np.float16)
        scale_b = (
            scale_b.reshape(scale_k // 2, 2, args.n).transpose(0, 2, 1).reshape(-1)
        )
    else:
        a = rng.integers(-4, 5, size=(args.m, args.k), dtype=np.int8)
        b = rng.integers(-4, 5, size=(args.k, args.n), dtype=np.int8)
        if args.variant == "pertensor_streamk":
            scale_a = None
            scale_b = np.array([0.5], dtype=np.float32)
            golden = (
                np.matmul(a.astype(np.int32), b.astype(np.int32)).astype(np.float32)
                * scale_b[0]
            ).astype(np.float16)
        else:
            scale_a = MIX_X1_SCALE_VALUES[np.arange(args.m) % MIX_X1_SCALE_VALUES.size]
            scale_b = MIX_X2_SCALE_VALUES[np.arange(args.n) % MIX_X2_SCALE_VALUES.size]
            golden = (
                np.matmul(a.astype(np.int32), b.astype(np.int32)).astype(np.float32)
                * scale_a.reshape(args.m, 1)
                * scale_b.reshape(1, args.n)
            ).astype(np.float16)

    a.tofile(os.path.join(args.output_dir, "input_a.bin"))
    b.tofile(os.path.join(args.output_dir, "input_b.bin"))
    if scale_a is not None:
        scale_a.tofile(os.path.join(args.output_dir, "scale_a.bin"))
    scale_b.tofile(os.path.join(args.output_dir, "scale_b.bin"))
    golden.tofile(os.path.join(args.output_dir, "golden_c.bin"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        required=True,
        choices=(
            "mix",
            "mix_without_batch",
            "mx_without_batch",
            "mx_streamk",
            "pertensor_streamk",
        ),
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
