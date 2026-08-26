#!/bin/bash
# TQBMM MX Transpose Quant Batch MatMul example run script

set -e

M=${1:-128}
K=${2:-512}
N=${3:-256}
BATCH=${4:-1}
BIAS=${5:-0}
A_DTYPE=${6:-fp8_e4m3}
B_DTYPE=${7:-fp8_e4m3}
C_DTYPE=${8:-bfloat16}
TRANS_A=${9:-false}
TRANS_B=${10:-false}
FORMAT=${11:-"(ND,ND)"}
BASE_M=${12:-128}
BASE_N=${13:-256}
BASE_K=${14:-64}
K_L1=${15:-64}
SCALE_K_L1=${16:-64}
L1_BUFFERS=${17:-2}
DB_L0C=${18:-1}
A_FULL_LOAD=${19:-false}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p input output

GROUP_SIZE=32
MXFP_DIVISOR_SIZE=64
SCALE_K=$(( (K + MXFP_DIVISOR_SIZE - 1) / MXFP_DIVISOR_SIZE * (MXFP_DIVISOR_SIZE / GROUP_SIZE) ))

# Generate input data
python3 -c "
import numpy as np
import os

m, k, n, batch = $M, $K, $N, $BATCH
scale_k = $SCALE_K
a_dtype = '$A_DTYPE'
b_dtype = '$B_DTYPE'
c_dtype = '$C_DTYPE'
bias = $BIAS

if a_dtype == 'fp4_e2m1':
    a_size = (m * batch * k + 1) // 2
else:
    a_size = m * batch * k
if b_dtype == 'fp4_e2m1':
    b_size = (k * batch * n + 1) // 2
else:
    b_size = k * batch * n

np.zeros(a_size, dtype=np.uint8).tofile('input/input_a.bin')
np.zeros(b_size, dtype=np.uint8).tofile('input/input_b.bin')
np.full(m * batch * scale_k, 0x7f, dtype=np.uint8).tofile('input/scale_a.bin')
np.full(batch * scale_k * n, 0x7f, dtype=np.uint8).tofile('input/scale_b.bin')
c_elem_size = 4 if c_dtype == 'float32' else 2
np.zeros(m * batch * n * c_elem_size, dtype=np.uint8).tofile('input/initial_c.bin')
if bias > 0:
    np.zeros(bias * 4, dtype=np.float32).tofile('input/bias.bin')
print('Input data generated.')
"

./tqbmm_mx "$M" "$K" "$N" "$BATCH" "$BIAS" "$A_DTYPE" "$B_DTYPE" "$C_DTYPE" \
    "$TRANS_A" "$TRANS_B" "$FORMAT" "$BASE_M" "$BASE_N" "$BASE_K" "$K_L1" "$SCALE_K_L1" \
    "$L1_BUFFERS" "$DB_L0C" "$A_FULL_LOAD"

echo "Done. Output: output/npu_out.bin"
