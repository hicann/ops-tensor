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

def gen_matmul_data(m, n, k, dtype='float16', output_dir='./'):
    os.makedirs(output_dir, exist_ok=True)

    dtype_map = {
        'float16': np.float16,
        'float32': np.float32,
        'bfloat16': np.float32
    }

    np_dtype = dtype_map[dtype]

    a = np.random.randn(m, k).astype(np_dtype)
    b = np.random.randn(k, n).astype(np_dtype)

    if dtype == 'float16':
        golden = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
    else:
        golden = np.matmul(a, b)

    a.tofile(os.path.join(output_dir, 'input_a.bin'))
    b.tofile(os.path.join(output_dir, 'input_b.bin'))
    golden.tofile(os.path.join(output_dir, 'golden_c.bin'))

    return a, b, golden


def main():
    parser = argparse.ArgumentParser(description='Generate Blaze MatMul test data')
    parser.add_argument('--m', type=int, required=True, help='M dimension')
    parser.add_argument('--n', type=int, required=True, help='N dimension')
    parser.add_argument('--k', type=int, required=True, help='K dimension')
    parser.add_argument('--dtype', type=str, default='float16',
                       choices=['float16', 'float32', 'bfloat16'])
    parser.add_argument('--output_dir', type=str, default='./')

    args = parser.parse_args()

    gen_matmul_data(args.m, args.n, args.k, args.dtype, args.output_dir)

if __name__ == '__main__':
    main()