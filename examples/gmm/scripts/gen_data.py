#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

"""Generate deterministic QGMM MX inputs and a NumPy golden result."""

import argparse
import os

import numpy as np


MX_SCALE_ONE = np.uint8(0x7F)
TYPE_INFO = {
    "mxfp8_e4m3": (
        np.array([0.5, -0.5, 1.0, -1.0], np.float32),
        np.array([0x30, 0xB0, 0x38, 0xB8], np.uint8),
        32,
    ),
    "mxfp8_e5m2": (
        np.array([0.5, -0.5, 1.0, -1.0], np.float32),
        np.array([0x38, 0xB8, 0x3C, 0xBC], np.uint8),
        32,
    ),
    "mxfp4_e2m1": (
        np.array([0.5, -0.5, 1.0, -1.0], np.float32),
        np.array([0x1, 0x9, 0x2, 0xA], np.uint8),
        64,
    ),
    "mxfp4_e1m2": (
        np.array([0.5, -0.5, 1.0, -1.0], np.float32),
        np.array([0x2, 0xA, 0x4, 0xC], np.uint8),
        64,
    ),
}


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _pack_fp4(values):
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if flat.size % 2:
        flat = np.pad(flat, (0, 1))
    return flat[0::2] | (flat[1::2] << 4)


def _encode(values, dtype):
    return (
        _pack_fp4(values)
        if dtype.startswith("mxfp4")
        else np.asarray(values, np.uint8).reshape(-1)
    )


def _format_weight(codes, weight_format, k, n, c0):
    if weight_format == "nd":
        return codes.reshape(-1)
    if weight_format == "dn":
        return codes.T.reshape(-1)
    if weight_format == "nz":
        padded = np.zeros((_align_up(k, 16), _align_up(n, c0)), np.uint8)
        padded[:k, :n] = codes
        return (
            padded.reshape(-1, 16, padded.shape[1] // c0, c0)
            .transpose(2, 0, 1, 3)
            .reshape(-1)
        )
    padded = np.zeros((_align_up(n, 16), _align_up(k, c0)), np.uint8)
    padded[:n, :k] = codes.T
    return (
        padded.reshape(-1, 16, padded.shape[1] // c0, c0)
        .transpose(2, 0, 1, 3)
        .reshape(-1)
    )


def _make_group_list(group_num, split_size, list_type):
    lengths = np.full(group_num, split_size, np.int64)
    if list_type == "offset":
        return np.cumsum(lengths, dtype=np.int64)
    if list_type == "sparse":
        return np.column_stack((np.arange(group_num, dtype=np.int64), lengths)).reshape(
            -1
        )
    return lengths


def _write_weights(output_dir, encoded, scales, multi_tensor):
    if multi_tensor:
        for index, (weight, scale) in enumerate(zip(encoded, scales)):
            weight.tofile(os.path.join(output_dir, f"input_b_{index}.bin"))
            scale.tofile(os.path.join(output_dir, f"scale_b_{index}.bin"))
        return
    np.concatenate(encoded).tofile(os.path.join(output_dir, "input_b.bin"))
    np.concatenate(scales).tofile(os.path.join(output_dir, "scale_b.bin"))


def generate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    values, codes, c0 = TYPE_INFO[args.dtype]
    rng = np.random.default_rng(args.seed)
    a_indices = rng.integers(0, values.size, size=(args.e, args.m, args.k))
    b_indices = rng.integers(0, values.size, size=(args.e, args.k, args.n))
    a_values = values[a_indices]
    b_values = values[b_indices]

    if args.trans_a:
        input_a = _encode(codes[a_indices].transpose(0, 2, 1), args.dtype)
        golden = sum(a_values[index] @ b_values[index] for index in range(args.e))
    else:
        input_a = _encode(codes[a_indices], args.dtype)
        golden = np.concatenate(
            [a_values[index] @ b_values[index] for index in range(args.e)], axis=0
        )

    encoded_b = [
        _encode(
            _format_weight(
                codes[b_indices[index]], args.weight_format, args.k, args.n, c0
            ),
            args.dtype,
        )
        for index in range(args.e)
    ]
    scale_k = _align_up(args.k, 64) // 32
    scale_factor = 2 if args.trans_a else 1
    scale_a = np.full(args.e * args.m * scale_k * scale_factor, MX_SCALE_ONE, np.uint8)
    scale_b = [
        np.full(args.n * scale_k * scale_factor, MX_SCALE_ONE, np.uint8)
        for _ in range(args.e)
    ]
    bias = np.full((args.e, args.n), 0.25 if args.with_bias else 0.0, np.float32)
    if args.with_bias:
        golden += bias[0] if args.trans_a else np.repeat(bias, args.m, axis=0)

    input_a.tofile(os.path.join(args.output_dir, "input_a.bin"))
    scale_a.tofile(os.path.join(args.output_dir, "scale_a.bin"))
    bias.tofile(os.path.join(args.output_dir, "bias.bin"))
    _make_group_list(
        args.e, args.k if args.trans_a else args.m, args.group_list_type
    ).tofile(os.path.join(args.output_dir, "group_list.bin"))
    _write_weights(args.output_dir, encoded_b, scale_b, args.multi_tensor)
    golden.astype(np.float32).tofile(os.path.join(args.output_dir, "golden_c.bin"))


def main():
    parser = argparse.ArgumentParser(description="Generate QGMM MX example data")
    parser.add_argument("--e", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", choices=tuple(TYPE_INFO), required=True)
    parser.add_argument(
        "--weight-format", choices=("nd", "dn", "nz", "zn"), required=True
    )
    parser.add_argument(
        "--group-list-type", choices=("length", "offset", "sparse"), required=True
    )
    parser.add_argument("--trans-a", action="store_true")
    parser.add_argument("--multi-tensor", action="store_true")
    parser.add_argument("--with-bias", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
