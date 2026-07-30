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

"""Generate deterministic MX inputs and a NumPy golden result."""

import argparse
import os

import ml_dtypes
import numpy as np


FP8_E4M3FN = ml_dtypes.float8_e4m3fn
FP8_E8M0 = ml_dtypes.float8_e8m0fnu
FP4_E2M1_TO_FP32 = np.arange(16, dtype=np.int8).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
GROUP_SIZE = 32
NZ_K0 = 32
NZ_N0 = 16


def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def pack_fp4(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.shape[-1] % 2 != 0:
        raise ValueError("FP4 packing axis must have an even length")
    low = np.bitwise_and(codes[..., 0::2], 0x0F)
    high = np.bitwise_and(codes[..., 1::2], 0x0F)
    return np.bitwise_or(low, np.left_shift(high, 4)).astype(np.uint8)


def unpack_fp4(packed):
    packed = np.asarray(packed, dtype=np.uint8)
    codes = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    codes[..., 0::2] = np.bitwise_and(packed, 0x0F)
    codes[..., 1::2] = np.right_shift(packed, 4)
    return codes


def format_weight(codes, layout):
    if layout == "nd":
        # The ND B tensor is logical (N, K), so the two FP4 values in one byte
        # are adjacent on K for each N row.
        return pack_fp4(codes.T).reshape(-1)

    k, n = codes.shape
    k_aligned = align(k, NZ_K0)
    n_aligned = align(n, NZ_N0)
    padded = np.zeros((k_aligned, n_aligned), dtype=np.uint8)
    padded[:k, :n] = codes
    nz = padded.reshape(k_aligned // NZ_K0, NZ_K0, n_aligned // NZ_N0, NZ_N0)
    nz = nz.transpose(0, 2, 3, 1)
    return pack_fp4(nz).reshape(-1)


def decode_weight(packed, layout, k, n):
    if layout == "nd":
        packed_nd = np.asarray(packed, dtype=np.uint8).reshape(n, (k + 1) // 2)
        return unpack_fp4(packed_nd)[..., :k].T

    k_aligned = align(k, NZ_K0)
    n_aligned = align(n, NZ_N0)
    packed_nz = np.asarray(packed, dtype=np.uint8).reshape(
        k_aligned // NZ_K0, n_aligned // NZ_N0, NZ_N0, NZ_K0 // 2
    )
    nz_codes = unpack_fp4(packed_nz)
    return nz_codes.transpose(0, 3, 1, 2).reshape(k_aligned, n_aligned)[:k, :n]


def make_scale(shape):
    """Create valid E8M0 values with ml_dtypes."""
    exponents = np.indices(shape, dtype=np.int64).sum(axis=0) % 4 - 1
    values = np.exp2(exponents.astype(np.float32))
    return values.astype(FP8_E8M0)


def generate(args):
    if min(args.m, args.n, args.k) <= 0 or args.k % 8 != 0:
        raise ValueError("M/N must be positive and K must be a multiple of 8")
    if args.layout == "nz" and args.n % 8 != 0:
        raise ValueError("NZ N must be a multiple of 8")
    if args.bias not in (0, args.n):
        raise ValueError("bias must be 0 or N")

    os.makedirs(args.output_dir, exist_ok=True)
    scale_k = align(args.k, 64) // GROUP_SIZE
    rng = np.random.default_rng(20260727)
    x1 = rng.choice(np.array([0.5, 1.0, -1.0, 2.0], dtype=np.float32), size=(args.m, args.k))
    x1 = x1.astype(FP8_E4M3FN)

    # Use every 4-bit code, including zero and negative values. The hardware
    # input is the code, not a host float4 object.
    x2_codes = rng.integers(0, 16, size=(args.k, args.n), dtype=np.uint8)

    scale_a = make_scale((args.m, scale_k))
    scale_b = make_scale((scale_k, args.n))
    scale_a_broadcast = np.repeat(scale_a.astype(np.float32), GROUP_SIZE, axis=1)[:, :args.k]
    scale_b_broadcast = np.repeat(scale_b.astype(np.float32), GROUP_SIZE, axis=0)[:args.k, :]

    packed_weight = format_weight(x2_codes, args.layout)
    x2 = FP4_E2M1_TO_FP32[decode_weight(packed_weight, args.layout, args.k, args.n)]
    x1_dequant = x1.astype(np.float32) * scale_a_broadcast
    x2_dequant = x2 * scale_b_broadcast

    bias = np.zeros(args.n, dtype=np.float16)
    if args.bias:
        bias = np.where(np.arange(args.n) % 2 == 0, 0.5, -1.0).astype(np.float16)
    golden = np.matmul(x1_dequant, x2_dequant)
    golden = (golden + bias.astype(np.float32)).astype(bias.dtype)

    x1.view(np.uint8).tofile(os.path.join(args.output_dir, "input_a.bin"))
    format_weight(x2_codes, args.layout).tofile(os.path.join(args.output_dir, "input_b.bin"))
    scale_a.view(np.uint8).tofile(os.path.join(args.output_dir, "scale_a.bin"))
    # ScaleBDN has a column-major GM stride: raw bytes are N-major while the
    # logical tensor consumed by the kernel is (scale_k, N).
    scale_b.T.view(np.uint8).tofile(os.path.join(args.output_dir, "scale_b.bin"))
    bias.tofile(os.path.join(args.output_dir, "bias.bin"))
    np.zeros((args.m, args.n), dtype=bias.dtype).tofile(os.path.join(args.output_dir, "initial_c.bin"))
    golden.tofile(os.path.join(args.output_dir, "golden_c.bin"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--bias", type=int, default=0)
    parser.add_argument("--layout", choices=("nd", "nz"), required=True)
    parser.add_argument("--output-dir", required=True)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
