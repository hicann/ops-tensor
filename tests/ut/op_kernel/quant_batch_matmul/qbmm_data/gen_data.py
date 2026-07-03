# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

#!/usr/bin/env python3

import argparse
import os
import numpy as np

def gen_qbmm_data(m, n, k, output_dir='./'):
    os.makedirs(output_dir, exist_ok=True)

    # Generate random int8 data for A (M, K) and B (K, N)
    a = np.random.randint(-128, 127, size=(m, k), dtype=np.int8)
    b = np.random.randint(-128, 127, size=(k, n), dtype=np.int8)

    # Generate a single float pertokenScale value (activation per-token scale, 1.0f)
    pertoken_scale = np.array([1.0], dtype=np.float32)

    # Generate a single uint64 scale value (weight scale, 1.0 in float, stored as uint64)
    # 0x3F80000000000000 = 1.0f in IEEE 754 float, zero-extended to uint64
    scale = np.array([0x3F80000000000000], dtype=np.uint64)

    # Compute golden output using float32: C = A * B * pertokenScale * scale
    # A8W8 matmul: int8 * int8 -> int32 accumulation, then dequantize with scales
    a_fp32 = a.astype(np.float32)
    b_fp32 = b.astype(np.float32)
    c_fp32 = np.matmul(a_fp32, b_fp32)
    # Apply pertokenScale and scale (both 1.0 means no change)
    pertoken_scale_float = pertoken_scale[0]
    scale_float = 1.0
    c_fp32 = c_fp32 * pertoken_scale_float * scale_float
    # Cast to half (float16)
    golden = c_fp32.astype(np.float16)

    a.tofile(os.path.join(output_dir, 'input_a.bin'))
    b.tofile(os.path.join(output_dir, 'input_b.bin'))
    pertoken_scale.tofile(os.path.join(output_dir, 'pertoken_scale.bin'))
    scale.tofile(os.path.join(output_dir, 'scale.bin'))
    golden.tofile(os.path.join(output_dir, 'golden_c.bin'))

    return a, b, pertoken_scale, scale, golden


def main():
    parser = argparse.ArgumentParser(description='Generate QBMM A8W8 test data')
    parser.add_argument('--m', type=int, required=True, help='M dimension')
    parser.add_argument('--n', type=int, required=True, help='N dimension')
    parser.add_argument('--k', type=int, required=True, help='K dimension')
    parser.add_argument('--output_dir', type=str, default='./')

    args = parser.parse_args()

    gen_qbmm_data(args.m, args.n, args.k, args.output_dir)

if __name__ == '__main__':
    main()
