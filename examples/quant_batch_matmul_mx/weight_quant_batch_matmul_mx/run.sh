#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
BUILD_DIR="${SCRIPT_DIR}/build"
CASE_FILE="${SCRIPT_DIR}/weight_quant_batch_matmul_mx.csv"
BUILD_ONLY=false
SKIP_BUILD=false

usage() {
    echo "Usage: bash run.sh [--case <csv>] [--skip-build] [--build-only]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)
            if [[ -z "${2:-}" ]]; then
                usage >&2
                exit 1
            fi
            CASE_FILE="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set. Source the CANN environment first." >&2
    exit 1
fi
if [[ ! -f "${CASE_FILE}" ]]; then
    echo "CSV case file not found: ${CASE_FILE}" >&2
    exit 1
fi

if [[ "${SKIP_BUILD}" != true ]]; then
    rm -rf "${BUILD_DIR}"
    cmake -S "${EXAMPLES_DIR}" -B "${BUILD_DIR}" -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}"
    cmake --build "${BUILD_DIR}" --target weight_quant_batch_matmul_mx -j"$(nproc)"
fi

if [[ "${BUILD_ONLY}" == true ]]; then
    exit 0
fi

EXECUTABLE="${BUILD_DIR}/quant_batch_matmul_mx/weight_quant_batch_matmul_mx/weight_quant_batch_matmul_mx"
if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Executable not found: ${EXECUTABLE}" >&2
    exit 1
fi
python3 "${SCRIPT_DIR}/parse_csv.py" "${EXECUTABLE}" "${CASE_FILE}" "${CASE_FILE%.csv}_result.csv"
