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
OP_SCRIPTS_DIR="${OP_DIR}/scripts"
EXAMPLES_DIR="$(dirname "${OP_DIR}")"
EXAMPLES_COMMON_DIR="${EXAMPLES_DIR}/common"
REPO_ROOT="$(dirname "${EXAMPLES_DIR}")"
BUILD_DIR="${SCRIPT_DIR}/build"
DATA_ROOT="${SCRIPT_DIR}/data"

# ── Color Definitions ─────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Built-in Defaults (per example) ───────────────────────────────────────────
declare -A EXAMPLE_M EXAMPLE_K EXAMPLE_N EXAMPLE_BIAS EXAMPLE_LAYOUT
declare -A EXAMPLE_BASE_M EXAMPLE_BASE_N EXAMPLE_BASE_K
declare -A EXAMPLE_TILE_K_L1 EXAMPLE_SCALE_K_L1 EXAMPLE_K_BUB EXAMPLE_N_BUB
declare -A EXAMPLE_L1_BUFFERS EXAMPLE_BLOCK_NUM

EXAMPLE_M[weight_quant_batch_matmul_mx]=32
EXAMPLE_K[weight_quant_batch_matmul_mx]=128
EXAMPLE_N[weight_quant_batch_matmul_mx]=40
EXAMPLE_BIAS[weight_quant_batch_matmul_mx]=0
EXAMPLE_LAYOUT[weight_quant_batch_matmul_mx]=ND
EXAMPLE_BASE_M[weight_quant_batch_matmul_mx]=32
EXAMPLE_BASE_N[weight_quant_batch_matmul_mx]=32
EXAMPLE_BASE_K[weight_quant_batch_matmul_mx]=64
EXAMPLE_TILE_K_L1[weight_quant_batch_matmul_mx]=64
EXAMPLE_SCALE_K_L1[weight_quant_batch_matmul_mx]=64
EXAMPLE_K_BUB[weight_quant_batch_matmul_mx]=64
EXAMPLE_N_BUB[weight_quant_batch_matmul_mx]=32
EXAMPLE_L1_BUFFERS[weight_quant_batch_matmul_mx]=2
EXAMPLE_BLOCK_NUM[weight_quant_batch_matmul_mx]=1

# ── CLI Defaults ──────────────────────────────────────────────────────────────
TARGET=""
CASE_FILE=""
SKIP_BUILD=false
BUILD_ONLY=false
M=""
K=""
N=""
BIAS=""
LAYOUT=""
BASE_M=""
BASE_N=""
BASE_K=""
TILE_K_L1=""
SCALE_K_L1=""
K_BUB=""
N_BUB=""
L1_BUFFERS=""
BLOCK_NUM=""

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: bash run.sh [OPTIONS]

Options:
  --target=<name>    Specify a single example to run.
  --case=<path>      CSV file containing test cases for batch execution.
  --skip-build       Skip CMake build stage.
  --build-only       Only build, do not execute or verify.
  -h, --help         Show this help.
EOF
}

# ── Argument Parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target=*)
            TARGET="${1#*=}"
            [[ -z "${TARGET}" ]] && { log_error "--target needs a value"; exit 1; }
            shift
            ;;
        --target)
            [[ -z "${2:-}" ]] && { log_error "--target needs a value"; exit 1; }
            TARGET="$2"
            shift 2
            ;;
        --case=*)
            CASE_FILE="${1#*=}"
            [[ -z "${CASE_FILE}" ]] && { log_error "--case needs a value"; exit 1; }
            shift
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
        -*)
            log_error "unknown option: $1"
            usage
            exit 1
            ;;
        *)
            if [[ -z "$M" ]]; then
                M="$1"
            elif [[ -z "$K" ]]; then
                K="$1"
            elif [[ -z "$N" ]]; then
                N="$1"
            elif [[ -z "$BIAS" ]]; then
                BIAS="$1"
            elif [[ -z "$LAYOUT" ]]; then
                LAYOUT="$1"
            elif [[ -z "$BASE_M" ]]; then
                BASE_M="$1"
            elif [[ -z "$BASE_N" ]]; then
                BASE_N="$1"
            elif [[ -z "$BASE_K" ]]; then
                BASE_K="$1"
            elif [[ -z "$TILE_K_L1" ]]; then
                TILE_K_L1="$1"
            elif [[ -z "$SCALE_K_L1" ]]; then
                SCALE_K_L1="$1"
            elif [[ -z "$K_BUB" ]]; then
                K_BUB="$1"
            elif [[ -z "$N_BUB" ]]; then
                N_BUB="$1"
            elif [[ -z "$L1_BUFFERS" ]]; then
                L1_BUFFERS="$1"
            elif [[ -z "$BLOCK_NUM" ]]; then
                BLOCK_NUM="$1"
            else
                log_error "unexpected argument: $1"
                usage
                exit 1
            fi
            shift
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

    if [[ $missing -ne 0 ]]; then
        log_error "Pre-flight checks failed. Please fix the above issues and retry."
        exit 1
    fi

    log_success "All pre-flight checks passed"
}

# ── Example Discovery ────────────────────────────────────────────────────────
discover_examples() {
    local examples
    examples=$(grep -E '^\s*ops_example_add_executable\(' "${SCRIPT_DIR}/CMakeLists.txt" 2>/dev/null | \
               sed -E 's/.*ops_example_add_executable\(([a-zA-Z0-9_]+).*/\1/')
    if [[ -z "$examples" ]]; then
        log_error "No examples registered in ${SCRIPT_DIR}/CMakeLists.txt"
        exit 1
    fi
    echo "$examples"
}

# ── Resolve Parameters for an Example ─────────────────────────────────────────
resolve_params() {
    local example="$1"

    if [[ -z "${EXAMPLE_M[$example]+x}" ]]; then
        log_error "No built-in defaults for example: ${example}"
        exit 1
    fi

    M_RESOLVED="${M:-${EXAMPLE_M[$example]}}"
    K_RESOLVED="${K:-${EXAMPLE_K[$example]}}"
    N_RESOLVED="${N:-${EXAMPLE_N[$example]}}"
    BIAS_RESOLVED="${BIAS:-${EXAMPLE_BIAS[$example]}}"
    LAYOUT_RESOLVED="${LAYOUT:-${EXAMPLE_LAYOUT[$example]}}"
    BASE_M_RESOLVED="${BASE_M:-${EXAMPLE_BASE_M[$example]}}"
    BASE_N_RESOLVED="${BASE_N:-${EXAMPLE_BASE_N[$example]}}"
    BASE_K_RESOLVED="${BASE_K:-${EXAMPLE_BASE_K[$example]}}"
    TILE_K_L1_RESOLVED="${TILE_K_L1:-${EXAMPLE_TILE_K_L1[$example]}}"
    SCALE_K_L1_RESOLVED="${SCALE_K_L1:-${EXAMPLE_SCALE_K_L1[$example]}}"
    K_BUB_RESOLVED="${K_BUB:-${EXAMPLE_K_BUB[$example]}}"
    N_BUB_RESOLVED="${N_BUB:-${EXAMPLE_N_BUB[$example]}}"
    L1_BUFFERS_RESOLVED="${L1_BUFFERS:-${EXAMPLE_L1_BUFFERS[$example]}}"
    BLOCK_NUM_RESOLVED="${BLOCK_NUM:-${EXAMPLE_BLOCK_NUM[$example]}}"
}

# ── Build ─────────────────────────────────────────────────────────────────────
source "${EXAMPLES_COMMON_DIR}/submodule_utils.sh"

do_build() {
    local target="${EXAMPLE:-${TARGET:-weight_quant_batch_matmul_mx}}"
    log_info "Building ${SCENARIO_NAME} ..."

    if ! ensure_tensor_api_submodule "${REPO_ROOT}"; then
        return 1
    fi

    rm -rf "${BUILD_DIR}"

    cmake -B "${BUILD_DIR}" \
        -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
        "${EXAMPLES_DIR}"

    if ! cmake --build "${BUILD_DIR}" --target "${target}" -j"$(nproc)"; then
        log_error "Build FAILED for ${target}"
        return 1
    fi

    log_success "Build succeeded for ${target}"
}

# ── Generate Test Data ───────────────────────────────────────────────────────
do_gen_data() {
    CASE_DATA_DIR="${DATA_ROOT}/${EXAMPLE}"
    log_info "Generating test data for ${EXAMPLE} (M=${M_RESOLVED} K=${K_RESOLVED} N=${N_RESOLVED} layout=${LAYOUT_RESOLVED} bias=${BIAS_RESOLVED}) ..."

    python3 "${OP_SCRIPTS_DIR}/gen_data.py" \
        --m "${M_RESOLVED}" --k "${K_RESOLVED}" --n "${N_RESOLVED}" \
        --bias "${BIAS_RESOLVED}" --layout "${LAYOUT_RESOLVED}" \
        --output-dir "${CASE_DATA_DIR}"

    log_success "Test data generated"
}

# ── Execute ──────────────────────────────────────────────────────────────────
do_run() {
    log_info "Running ${EXAMPLE} ..."

    local exec_path="${BUILD_DIR}/quant_batch_matmul_mx/${SCENARIO_NAME}/${EXAMPLE}"
    if [[ ! -x "${exec_path}" ]]; then
        log_error "Executable not found: ${exec_path}"
        return 1
    fi

    if ! "${exec_path}" \
        "${M_RESOLVED}" "${K_RESOLVED}" "${N_RESOLVED}" "${BIAS_RESOLVED}" "${LAYOUT_RESOLVED}" \
        "${BASE_M_RESOLVED}" "${BASE_N_RESOLVED}" "${BASE_K_RESOLVED}" \
        "${TILE_K_L1_RESOLVED}" "${SCALE_K_L1_RESOLVED}" "${K_BUB_RESOLVED}" "${N_BUB_RESOLVED}" \
        "${L1_BUFFERS_RESOLVED}" "${BLOCK_NUM_RESOLVED}" "${CASE_DATA_DIR}"; then
        log_error "Execution FAILED for ${EXAMPLE}"
        return 1
    fi

    log_success "Execution completed for ${EXAMPLE}"
}

# ── Verify ───────────────────────────────────────────────────────────────────
do_verify() {
    log_info "Verifying results for ${EXAMPLE} ..."

    if ! python3 "${OP_SCRIPTS_DIR}/verify_result.py" \
        "${CASE_DATA_DIR}/golden_c.bin" "${CASE_DATA_DIR}/npu_out.bin"; then
        log_error "Verification FAILED for ${EXAMPLE}"
        return 1
    fi

    log_success "Verification passed for ${EXAMPLE}"
}

# ── Cleanup ──────────────────────────────────────────────────────────────────
cleanup_data() {
    log_info "Cleaning up generated data ..."
    rm -rf "${DATA_ROOT}"
    log_success "Cleanup completed"
}

# ── Run CSV Cases ────────────────────────────────────────────────────────────
run_csv_cases() {
    local csv_file="$1"

    if [[ ! -f "$csv_file" ]]; then
        log_error "CSV file not found: $csv_file"
        return 1
    fi

    local target="${TARGET:-weight_quant_batch_matmul_mx}"
    local executable="${BUILD_DIR}/quant_batch_matmul_mx/${SCENARIO_NAME}/${target}"

    if [[ ! -x "$executable" ]]; then
        log_error "Executable not found: $executable"
        return 1
    fi

    local result_file="${csv_file%.csv}_result.csv"

    log_info "Running test cases from $csv_file"
    if python3 "${SCRIPT_DIR}/parse_csv.py" "$executable" "$csv_file" "$result_file"; then
        log_success "All test cases completed. Results: $result_file"
        return 0
    fi

    log_error "Some test cases failed. Check $result_file for details."
    return 1
}

# ── Run Single Example ───────────────────────────────────────────────────────
run_example() {
    local example="$1"
    EXAMPLE="$example"

    resolve_params "$example"

    echo ""
    echo "========================================================================"
    echo "  Example: ${EXAMPLE}"
    echo "  Shape:   ${M_RESOLVED} x ${K_RESOLVED} x ${N_RESOLVED}"
    echo "  layout:  ${LAYOUT_RESOLVED}   bias: ${BIAS_RESOLVED}   blocks: ${BLOCK_NUM_RESOLVED}"
    echo "  base:    ${BASE_M_RESOLVED} x ${BASE_N_RESOLVED} x ${BASE_K_RESOLVED}"
    echo "========================================================================"

    local exec_path="${BUILD_DIR}/quant_batch_matmul_mx/${SCENARIO_NAME}/${EXAMPLE}"
    if [[ "$SKIP_BUILD" != true ]]; then
        if ! do_build; then
            log_error "build examples:${SCENARIO_NAME} failed"
            return 1
        fi
    else
        log_info "Skipping build (--skip-build)"
        if [[ ! -x "${exec_path}" ]]; then
            log_error "Executable not found: ${exec_path}. Remove --skip-build or build first."
            return 1
        fi
    fi

    if [[ "$BUILD_ONLY" == true ]]; then
        log_info "Build-only mode, skipping execution and verification"
        return 0
    fi

    if ! do_gen_data; then
        cleanup_data
        return 1
    fi

    local run_result=0
    if ! do_run; then
        run_result=1
    elif ! do_verify; then
        run_result=1
    fi

    cleanup_data
    return $run_result
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    echo "=========================================="
    echo "  weight_quant_batch_matmul_mx Runner"
    echo "  Scenario: ${SCENARIO_NAME}"
    echo "=========================================="

    preflight

    if [[ -n "$CASE_FILE" ]]; then
        if [[ "$SKIP_BUILD" != true ]]; then
            do_build
        fi
        if [[ "$BUILD_ONLY" == true ]]; then
            return 0
        fi

        local csv_result=0
        if ! run_csv_cases "$CASE_FILE"; then
            csv_result=1
        fi
        cleanup_data
        return $csv_result
    fi

    if [[ -n "$TARGET" ]]; then
        local registered
        registered=$(discover_examples)
        if ! echo "$registered" | grep -qx "$TARGET"; then
            log_error "Target '${TARGET}' not registered in CMakeLists.txt"
            log_info "Available examples:"
            echo "$registered" | while read -r value; do echo "  - $value"; done
            exit 1
        fi
        EXAMPLES="$TARGET"
        MODE="single"
    else
        EXAMPLES="$(discover_examples)"
        MODE="multi"
    fi

    log_info "Mode: ${MODE}"
    log_info "Examples: $(echo "$EXAMPLES" | tr '\n' ' ')"

    local all_passed=true
    local results=""

    for example in $EXAMPLES; do
        set +e
        run_example "$example"
        local rc=$?
        set -e

        if [[ $rc -eq 0 ]]; then
            if [[ "$MODE" == "single" ]]; then
                echo ""
                log_success "[PASS] ${example}"
            fi
            results="${results}${example}: PASS\n"
        else
            if [[ "$MODE" == "single" ]]; then
                echo ""
                log_error "[FAIL] ${example}"
            fi
            results="${results}${example}: FAIL\n"
            all_passed=false
        fi
    done

    if [[ "$MODE" == "multi" ]]; then
        echo ""
        echo "=========================================="
        echo "  Summary"
        echo "=========================================="
        echo -e "$results" | grep -v '^$'
        echo "=========================================="

        if [[ "$all_passed" == true ]]; then
            log_success "All examples PASSED"
        else
            log_error "Some examples FAILED"
        fi
    fi

    if [[ "$all_passed" != true ]]; then
        exit 1
    fi
}

main "$@"
