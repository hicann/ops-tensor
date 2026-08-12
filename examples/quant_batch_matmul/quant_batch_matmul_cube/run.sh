#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -euo pipefail

# ── Key Variables ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIO_NAME="$(basename "${SCRIPT_DIR}")"
OP_DIR="$(dirname "${SCRIPT_DIR}")"
EXAMPLES_DIR="$(dirname "${OP_DIR}")"
EXAMPLES_COMMON_DIR="${EXAMPLES_DIR}/common"
REPO_ROOT="$(dirname "${EXAMPLES_DIR}")"
BUILD_DIR="${SCRIPT_DIR}/build"
CASE_FILE="${SCRIPT_DIR}/quant_batch_matmul_cube.csv"
TARGET="quant_batch_matmul_cube"
SKIP_BUILD=false
BUILD_ONLY=false

# ── Color Definitions ─────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: bash run.sh [OPTIONS]

Options:
  --case=<path>      CSV file containing test cases for batch execution.
  --skip-build       Skip CMake build stage.
  --build-only       Only build, do not execute or verify.
  -h, --help         Show this help.
EOF
}

# ── Argument Parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --case=*)
            CASE_FILE="${1#*=}"
            [[ -z "${CASE_FILE}" ]] && { log_error "--case needs a value"; exit 1; }
            shift
            ;;
        --case)
            [[ -z "${2:-}" ]] && { log_error "--case needs a value"; exit 1; }
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
            log_error "unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# ── Pre-flight Checks ────────────────────────────────────────────────────────
preflight() {
    local missing=0

    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        log_error "ASCEND_HOME_PATH is not set. Please source CANN set_env.sh first."
        missing=1
    elif [[ ! -d "${ASCEND_HOME_PATH}" ]]; then
        log_error "ASCEND_HOME_PATH directory does not exist: ${ASCEND_HOME_PATH}"
        missing=1
    else
        log_info "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
    fi

    if ! command -v bisheng &>/dev/null; then
        log_error "bisheng compiler not found in PATH"
        missing=1
    else
        log_info "bisheng: $(bisheng --version 2>&1 | head -1)"
    fi

    if ! command -v g++ &>/dev/null; then
        log_error "g++ compiler not found in PATH"
        missing=1
    else
        log_info "g++: $(g++ --version 2>&1 | head -1)"
    fi

    if ! command -v python3 &>/dev/null; then
        log_error "python3 not found in PATH"
        missing=1
    else
        log_info "python3: $(python3 --version 2>&1)"
    fi

    if ! command -v cmake &>/dev/null; then
        log_error "cmake not found in PATH"
        missing=1
    else
        log_info "cmake: $(cmake --version | head -1)"
    fi

    if [[ ${missing} -ne 0 ]]; then
        log_error "Pre-flight checks failed. Please fix the above issues and retry."
        exit 1
    fi

    log_success "All pre-flight checks passed"
}

# ── Build ─────────────────────────────────────────────────────────────────────
source "${EXAMPLES_COMMON_DIR}/submodule_utils.sh"

do_build() {
    log_info "Building ${SCENARIO_NAME} ..."

    if ! ensure_tensor_api_submodule "${REPO_ROOT}"; then
        return 1
    fi

    rm -rf "${BUILD_DIR}"
    cmake -B "${BUILD_DIR}" -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" "${EXAMPLES_DIR}"
    if ! cmake --build "${BUILD_DIR}" --target "${TARGET}" -j"$(nproc)"; then
        log_error "Build FAILED for ${TARGET}"
        return 1
    fi

    log_success "Build succeeded for ${TARGET}"
}

# ── Run CSV Cases ────────────────────────────────────────────────────────────
run_csv_cases() {
    local csv_file="$1"
    local executable="${BUILD_DIR}/quant_batch_matmul/${SCENARIO_NAME}/${TARGET}"
    local result_file="${csv_file%.csv}_result.csv"

    if [[ ! -f "${csv_file}" ]]; then
        log_error "CSV file not found: ${csv_file}"
        return 1
    fi
    if [[ ! -x "${executable}" ]]; then
        log_error "Executable not found: ${executable}"
        return 1
    fi

    log_info "Running test cases from ${csv_file}"
    if python3 "${SCRIPT_DIR}/parse_csv.py" "${executable}" "${csv_file}" "${result_file}"; then
        log_success "All test cases completed. Results: ${result_file}"
        return 0
    fi

    log_error "Some test cases failed. Check ${result_file} for details."
    return 1
}

# ── Cleanup ──────────────────────────────────────────────────────────────────
cleanup_data() {
    log_info "Cleaning up generated data ..."
    rm -rf "${SCRIPT_DIR}/data"
    log_success "Cleanup completed"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    echo "=========================================="
    echo "  quant_batch_matmul_cube One-Click Runner"
    echo "  Scenario: ${SCENARIO_NAME}"
    echo "=========================================="

    preflight

    if [[ "${SKIP_BUILD}" != true ]]; then
        do_build
    else
        log_info "Skipping build (--skip-build)"
    fi

    if [[ "${BUILD_ONLY}" == true ]]; then
        log_info "Build-only mode, skipping execution and verification"
        return 0
    fi

    local csv_result=0
    run_csv_cases "${CASE_FILE}" || csv_result=$?
    cleanup_data
    return "${csv_result}"
}

main "$@"
