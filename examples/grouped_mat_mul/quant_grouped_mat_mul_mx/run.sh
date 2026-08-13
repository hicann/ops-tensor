#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OP_DIR=$(dirname "${SCRIPT_DIR}")
EXAMPLES_DIR=$(dirname "${OP_DIR}")
BUILD_DIR="${SCRIPT_DIR}/build"
TARGET=quant_grouped_mat_mul_mx
CASE_FILE=""
SKIP_BUILD=false
BUILD_ONLY=false

log_info() { echo "[INFO] $*"; }
log_success() { echo "[SUCCESS] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

usage() {
    cat <<EOF
Usage: bash run.sh [--case=<csv>] [--skip-build] [--build-only]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --case=*) CASE_FILE="${1#*=}"; shift ;;
        --case) CASE_FILE="${2:?--case needs a value}"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --build-only) BUILD_ONLY=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH is not set; source the CANN set_env.sh first}"

if [[ "${SKIP_BUILD}" != true ]]; then
    cmake -S "${EXAMPLES_DIR}" -B "${BUILD_DIR}" -DASCEND_NPU_ARCH=dav-3510
    cmake --build "${BUILD_DIR}" --target "${TARGET}" -j
fi

if [[ "${BUILD_ONLY}" == true ]]; then
    exit 0
fi

EXECUTABLE="${BUILD_DIR}/grouped_mat_mul/quant_grouped_mat_mul_mx/quant_grouped_mat_mul_mx"
if [[ ! -x "${EXECUTABLE}" ]]; then
    log_error "Executable not found: ${EXECUTABLE}"
    exit 1
fi

if [[ -z "${CASE_FILE}" ]]; then
    CASE_FILE="${SCRIPT_DIR}/quant_grouped_mat_mul_mx.csv"
fi
RESULT_FILE="${SCRIPT_DIR}/quant_grouped_mat_mul_mx_result.csv"
python3 "${SCRIPT_DIR}/parse_csv.py" "${EXECUTABLE}" "${CASE_FILE}" "${RESULT_FILE}"

log_success "QGMM MX example passed"
