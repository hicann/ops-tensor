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

"""Generate deterministic MX inputs and compute golden: matmul -> gelu -> dynamic MX quant.

A is MX-quantized FP8 (fp8_e4m3 / fp8_e5m2). B is MX-quantized FP8 (fp8_e4m3 only), NZ layout.
C dtype is float (L0C float32 accumulator, DualDst to UB).
Output Y dtype follows A dtype; Y_scale is fp8_e8m0.
"""

import argparse
import os

import ml_dtypes
import numpy as np
from ml_dtypes import bfloat16


FP8_E4M3FN = ml_dtypes.float8_e4m3fn
FP8_E5M2 = ml_dtypes.float8_e5m2
FP8_E8M0 = ml_dtypes.float8_e8m0fnu

GROUP_SIZE = 32
MXFP_DIVISOR_SIZE = 64
MXFP_MULTI_BASE_SIZE = 2

_DTYPE_MAP = {
    "fp8_e4m3": FP8_E4M3FN,
    "fp8_e5m2": FP8_E5M2,
}

_EMAX_MAP = {
    "fp8_e4m3": 8,
    "fp8_e5m2": 15,
}

_DTYPE_MAX = {
    "fp8_e4m3": 448.0,
    "fp8_e5m2": 57344.0,
}

_C0_SIZE = 32


def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


# ---------------------------------------------------------------------------
# Scale helper
# ---------------------------------------------------------------------------


def make_scale(shape):
    exponents = np.indices(shape, dtype=np.int64).sum(axis=0) % 4 - 1
    values = np.exp2(exponents.astype(np.float32))
    return values.astype(FP8_E8M0)


# ---------------------------------------------------------------------------
# Quantized data generation for FP8
# ---------------------------------------------------------------------------


def generate_quantized_data(shape, dtype, rng):
    np_dtype = _DTYPE_MAP[dtype]
    values = rng.choice(np.array([0.5, 1.0, -1.0, 2.0], dtype=np.float32), size=shape)
    return values.astype(np_dtype), values.astype(np.float32)


def _transpose_quantized(raw, dtype, rows, cols):
    return raw.reshape(rows, cols).T.copy()


# ---------------------------------------------------------------------------
# Scale formatting (matmul input scales, with C0=2 interleave)
# ---------------------------------------------------------------------------


def format_scale_a(scale_a, m):
    return scale_a


def format_scale_a_trans(scale_a, m):
    scale_k = scale_a.shape[1]
    scale_k_aligned = align(scale_k, 2)
    sa = np.zeros((m, scale_k_aligned), dtype=FP8_E8M0)
    sa[:m, :scale_k] = scale_a
    return sa.T.reshape(scale_k_aligned // 2, 2, m).transpose(0, 2, 1).reshape(-1)


def format_scale_b(scale_b, n):
    scale_k = scale_b.shape[0]
    scale_k_aligned = align(scale_k, 2)
    sb = np.zeros((scale_k_aligned, n), dtype=FP8_E8M0)
    sb[:scale_k, :] = scale_b
    return sb.reshape(scale_k_aligned // 2, 2, n).transpose(0, 2, 1).reshape(-1)


def format_scale_b_trans(scale_b, n):
    return scale_b.T.copy()


# ---------------------------------------------------------------------------
# NZ conversion (weight B is always NZ layout)
# ---------------------------------------------------------------------------


def _convert_to_nz(data_nd, dtype, k, n, trans_b):
    mat = data_nd.view(np.uint8).reshape(k, n)
    if not trans_b:
        num_k_tiles = align(k, 16) // 16
        num_n_tiles = align(n, _C0_SIZE) // _C0_SIZE
        out = np.zeros(num_n_tiles * num_k_tiles * 16 * _C0_SIZE, dtype=np.uint8)
        idx = 0
        for nt in range(num_n_tiles):
            for kt in range(num_k_tiles):
                for r in range(16):
                    for c in range(_C0_SIZE):
                        out[idx] = mat[kt * 16 + r, nt * _C0_SIZE + c]
                        idx += 1
    else:
        num_k_tiles = align(k, _C0_SIZE) // _C0_SIZE
        num_n_tiles = align(n, 16) // 16
        out = np.zeros(num_k_tiles * num_n_tiles * _C0_SIZE * 16, dtype=np.uint8)
        idx = 0
        for kt in range(num_k_tiles):
            for nt in range(num_n_tiles):
                for c in range(16):
                    for r in range(_C0_SIZE):
                        out[idx] = mat[kt * _C0_SIZE + r, nt * 16 + c]
                        idx += 1
    return out


# ---------------------------------------------------------------------------
# Gelu + OCP dynamic MX quantization (golden computation)
# ---------------------------------------------------------------------------


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def ocp_mx_quantize(data, emax, dtype_max):
    """OCP MX quantization: per-32-element block, output float32 quantized + e8m0 scale."""
    orig_shape = data.shape
    n_elements = data.size
    padded = align(n_elements, GROUP_SIZE)
    flat = np.zeros(padded, dtype=np.float32)
    flat[:n_elements] = data.flatten()
    n_blocks = padded // GROUP_SIZE
    blocks = flat.reshape(n_blocks, GROUP_SIZE)
    abs_max = np.max(np.abs(blocks), axis=1)
    abs_max = np.maximum(abs_max, 1e-30)
    shared_exp = np.floor(np.log2(abs_max)).astype(np.int32) - emax
    shared_exp = np.maximum(shared_exp, -127)
    mx_scale = np.exp2(shared_exp.astype(np.float32))
    quantized = blocks / mx_scale[:, None]
    quantized = np.clip(quantized, -dtype_max, dtype_max)
    out = quantized.flatten()[:n_elements].reshape(orig_shape)
    scale_out = mx_scale.astype(FP8_E8M0)
    return out, scale_out


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def _write_bin(path, arr):
    if hasattr(arr, "view") and arr.dtype != np.uint8:
        arr.view(np.uint8).tofile(path)
    else:
        arr.tofile(path)


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


def generate(args):
    if min(args.m, args.n, args.k) <= 0 or args.k % 8 != 0:
        raise ValueError("M/N must be positive and K must be a multiple of 8")
    if args.bias not in (0, args.n):
        raise ValueError("bias must be 0 or N")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(args.output_dir), "output"), exist_ok=True)

    scale_k = align(args.k, MXFP_DIVISOR_SIZE) // GROUP_SIZE
    scale_n = align(args.n, MXFP_DIVISOR_SIZE) // GROUP_SIZE
    rng = np.random.default_rng(20260727)

    a_raw, a_fp32 = generate_quantized_data((args.m, args.k), args.a_dtype, rng)
    b_raw, b_fp32 = generate_quantized_data((args.k, args.n), args.b_dtype, rng)
    scale_a = make_scale((args.m, scale_k))
    scale_b = make_scale((scale_k, args.n))

    sa_b = np.repeat(scale_a.astype(np.float32), GROUP_SIZE, axis=1)[:, : args.k]
    sb_b = np.repeat(scale_b.astype(np.float32), GROUP_SIZE, axis=0)[: args.k, :]
    matmul_result = np.matmul(a_fp32 * sa_b, b_fp32 * sb_b)

    bias_np = np.zeros(args.n, dtype=np.float32)
    if args.bias:
        bias_np = np.where(np.arange(args.n) % 2 == 0, 0.5, -1.0).astype(np.float32)
    matmul_with_bias = matmul_result + bias_np

    # C dtype = float (L0C float32 accumulator, DualDst fixpipe to UB as float)
    # Gelu (tanh) reads float from UB, computes in float, outputs bf16 internally
    gelu_result = gelu_tanh(matmul_with_bias).astype(bfloat16)

    # Dynamic OCP MX quantization -> Y (out dtype = a_dtype) + Y_scale (e8m0)
    emax = _EMAX_MAP[args.a_dtype]
    dtype_max = _DTYPE_MAX[args.a_dtype]
    quantized_fp32, y_scale_flat = ocp_mx_quantize(
        gelu_result.astype(np.float32), emax, dtype_max
    )

    golden_y = quantized_fp32.astype(_DTYPE_MAP[args.a_dtype])
    y_scale_per_row = y_scale_flat.reshape(args.m, -1)
    golden_y_scale = np.zeros((args.m, scale_n), dtype=FP8_E8M0)
    golden_y_scale[:, : y_scale_per_row.shape[1]] = y_scale_per_row

    d = args.output_dir
    a_write = (
        _transpose_quantized(a_raw, args.a_dtype, args.m, args.k)
        if args.trans_a
        else a_raw
    )
    _write_bin(os.path.join(d, "input_a.bin"), a_write)
    _write_bin(
        os.path.join(d, "input_b.bin"),
        _convert_to_nz(b_raw, args.b_dtype, args.k, args.n, args.trans_b),
    )

    sa_fmt = (
        format_scale_a_trans(scale_a, args.m)
        if args.trans_a
        else format_scale_a(scale_a, args.m)
    )
    sb_fmt = (
        format_scale_b_trans(scale_b, args.n)
        if args.trans_b
        else format_scale_b(scale_b, args.n)
    )
    _write_bin(os.path.join(d, "scale_a.bin"), sa_fmt)
    _write_bin(os.path.join(d, "scale_b.bin"), sb_fmt)

    bias_out = bias_np if args.bias else np.zeros(1, dtype=np.float32)
    bias_out.tofile(os.path.join(d, "bias.bin"))

    golden_y.tofile(os.path.join(d, "golden_y.bin"))
    golden_y_scale.view(np.uint8).tofile(os.path.join(d, "golden_y_scale.bin"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate quant_matmul_activation_quant input and golden data."
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--bias", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--a-dtype", required=True, choices=("fp8_e4m3", "fp8_e5m2"))
    parser.add_argument("--b-dtype", required=True, choices=("fp8_e4m3",))
    parser.add_argument("--trans-a", action="store_true", default=False)
    parser.add_argument("--trans-b", action="store_true", default=False)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
