#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------

"""Generate test data for transpose quantized batch matmul MX.

Generates binary .bin files compatible with the tqbmm_mx example executable.
Supports fp8_e4m3 and fp4_e2m1 (packed) data types with E8M0 scale factors.
"""

import os
import argparse
import numpy as np


def get_scale_k_len(k):
    return ((k + 63) // 64) * 2


def pack_fp4(values):
    """Pack pairs of uint8 FP4 nibble values into bytes, low nibble first.

    Matches TTK pack_4bits and the hardware MMAD expectation (element i -> low
    nibble, element i+1 -> high nibble).
    """
    if len(values) % 2 != 0:
        values = np.append(values, 0)
    result = np.zeros(len(values) // 2, dtype=np.uint8)
    for i in range(0, len(values), 2):
        result[i // 2] = (values[i] & 0xF) | ((values[i + 1] & 0xF) << 4)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("Batch", type=int)
    parser.add_argument(
        "--dtype", type=str, default="fp8_e4m3", choices=["fp8_e4m3", "fp4_e2m1"]
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to write .bin files"
    )
    args = parser.parse_args()

    m, n, k, batch = args.M, args.N, args.K, args.Batch
    scale_k = get_scale_k_len(k)
    out_dir = (
        args.output_dir
        if args.output_dir
        else os.path.dirname(os.path.abspath(__file__))
    )
    os.makedirs(out_dir, exist_ok=True)

    if args.dtype == "fp4_e2m1":
        a_data = np.random.randint(0, 16, size=m * batch * k, dtype=np.uint8)
        b_data = np.random.randint(0, 16, size=k * batch * n, dtype=np.uint8)
        a_bytes = pack_fp4(a_data)
        b_bytes = pack_fp4(b_data)
    else:
        a_data = np.random.randint(0, 256, size=m * batch * k, dtype=np.uint8)
        b_data = np.random.randint(0, 256, size=k * batch * n, dtype=np.uint8)
        a_bytes = a_data
        b_bytes = b_data

    scale_a = np.full(m * batch * scale_k, 0x7F, dtype=np.uint8)
    scale_b = np.full(batch * scale_k * n, 0x7F, dtype=np.uint8)
    initial_c = np.zeros(m * batch * n * 2, dtype=np.uint8)

    a_bytes.tofile(os.path.join(out_dir, "input_a.bin"))
    b_bytes.tofile(os.path.join(out_dir, "input_b.bin"))
    scale_a.tofile(os.path.join(out_dir, "scale_a.bin"))
    scale_b.tofile(os.path.join(out_dir, "scale_b.bin"))
    initial_c.tofile(os.path.join(out_dir, "initial_c.bin"))

    print(f"Generated: M={m}, N={n}, K={k}, batch={batch}, dtype={args.dtype}")


if __name__ == "__main__":
    main()
