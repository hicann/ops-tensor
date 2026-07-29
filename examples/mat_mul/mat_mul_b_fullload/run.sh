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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIO_NAME="$(basename "${SCRIPT_DIR}")"
OP_DIR="$(dirname "${SCRIPT_DIR}")"
OP_SCRIPTS_DIR="${OP_DIR}/scripts"
EXAMPLES_DIR="$(dirname "${OP_DIR}")"
BUILD_DIR="${SCRIPT_DIR}/build"
WORK_DIR="${SCRIPT_DIR}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

declare -A EXAMPLE_M EXAMPLE_K EXAMPLE_N EXAMPLE_TA EXAMPLE_TB EXAMPLE_DTYPE EXAMPLE_HF32 EXAMPLE_BIAS EXAMPLE_FORMAT

EXAMPLE_M[mat_mul_b_fullload]=256
EXAMPLE_K[mat_mul_b_fullload]=512
EXAMPLE_N[mat_mul_b_fullload]=32
EXAMPLE_TA[mat_mul_b_fullload]=false
EXAMPLE_TB[mat_mul_b_fullload]=false
EXAMPLE_DTYPE[mat_mul_b_fullload]=float16
EXAMPLE_HF32[mat_mul_b_fullload]=false
EXAMPLE_BIAS[mat_mul_b_fullload]=0
EXAMPLE_FORMAT[mat_mul_b_fullload]="(ND,ND)"

TARGET=""
CASE_FILE=""
SKIP_BUILD=false
BUILD_ONLY=false
M=""
K=""
N=""
TRANS_A=""
TRANS_B=""
DTYPE=""
IS_HF32=""
BIAS=""
FORMAT=""

usage() {
    cat <<EOF
Usage: bash run.sh [OPTIONS]

Options:
  --target=<name>    Specify a single example to run (e.g. mat_mul_b_fullload).
  --case=<path>      CSV file containing test cases for batch execution.
  --skip-build       Skip CMake build stage.
  --build-only       Only build, do not execute or verify.
  -h, --help         Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target=*) TARGET="${1#*=}"; shift ;;
        --target) [[ -z "${2:-}" ]] && { log_error "--target needs a value"; exit 1; }; TARGET="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --build-only) BUILD_ONLY=true; shift ;;
        --case=*) CASE_FILE="${1#*=}"; shift ;;
        --case) [[ -z "${2:-}" ]] && { log_error "--case needs a value"; exit 1; }; CASE_FILE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) log_error "unknown option: $1"; usage; exit 1 ;;
        *)
            if [[ -z "$M" ]]; then M="$1"
            elif [[ -z "$K" ]]; then K="$1"
            elif [[ -z "$N" ]]; then N="$1"
            elif [[ -z "$TRANS_A" ]]; then TRANS_A="$1"
            elif [[ -z "$TRANS_B" ]]; then TRANS_B="$1"
            elif [[ -z "$DTYPE" ]]; then DTYPE="$1"
            elif [[ -z "$IS_HF32" ]]; then IS_HF32="$1"
            elif [[ -z "$BIAS" ]]; then BIAS="$1"
            elif [[ -z "$FORMAT" ]]; then FORMAT="$1"
            else log_error "unexpected argument: $1"; usage; exit 1; fi
            shift ;;
    esac
done

preflight() {
    local missing=0
    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        log_error "ASCEND_HOME_PATH is not set."; missing=1
    elif [[ ! -d "${ASCEND_HOME_PATH}" ]]; then
        log_error "ASCEND_HOME_PATH does not exist: ${ASCEND_HOME_PATH}"; missing=1
    else log_info "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"; fi
    if ! command -v bisheng &>/dev/null; then log_error "bisheng not found"; missing=1
    else log_info "bisheng: $(bisheng --version 2>&1 | head -1)"; fi
    if ! command -v g++ &>/dev/null; then log_error "g++ not found"; missing=1
    else log_info "g++: $(g++ --version 2>&1 | head -1)"; fi
    if ! command -v python3 &>/dev/null; then log_error "python3 not found"; missing=1
    else log_info "python3: $(python3 --version 2>&1)"; fi
    if ! command -v cmake &>/dev/null; then log_error "cmake not found"; missing=1
    else log_info "cmake: $(cmake --version | head -1)"; fi
    if [[ $missing -ne 0 ]]; then log_error "Pre-flight checks failed."; exit 1; fi
    log_success "All pre-flight checks passed"
}

discover_examples() {
    local examples
    examples=$(grep -E '^\s*ops_example_add_executable\(' "${SCRIPT_DIR}/CMakeLists.txt" 2>/dev/null | \
               sed -E 's/.*ops_example_add_executable\(([a-zA-Z0-9_]+).*/\1/')
    if [[ -z "$examples" ]]; then log_error "No examples registered"; exit 1; fi
    echo "$examples"
}

resolve_params() {
    local example="$1"
    if [[ -z "${EXAMPLE_M[$example]+x}" ]]; then log_error "No defaults for: ${example}"; exit 1; fi
    M_RESOLVED="${M:-${EXAMPLE_M[$example]}}"
    K_RESOLVED="${K:-${EXAMPLE_K[$example]}}"
    N_RESOLVED="${N:-${EXAMPLE_N[$example]}}"
    TA_RESOLVED="${TRANS_A:-${EXAMPLE_TA[$example]}}"
    TB_RESOLVED="${TRANS_B:-${EXAMPLE_TB[$example]}}"
    DTYPE_RESOLVED="${DTYPE:-${EXAMPLE_DTYPE[$example]}}"
    HF32_RESOLVED="${IS_HF32:-${EXAMPLE_HF32[$example]}}"
    BIAS_RESOLVED="${BIAS:-${EXAMPLE_BIAS[$example]}}"
    FORMAT_RESOLVED="${FORMAT:-${EXAMPLE_FORMAT[$example]}}"
}

do_build() {
    local target="${EXAMPLE:-${TARGET:-mat_mul_b_fullload}}"
    log_info "Building ${SCENARIO_NAME} ..."
    rm -rf "${BUILD_DIR}"
    cmake -B "${BUILD_DIR}" -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" "${EXAMPLES_DIR}"
    if ! cmake --build "${BUILD_DIR}" --target "${target}" -j"$(nproc)"; then
        log_error "Build FAILED for ${target}"; return 1; fi
    log_success "Build succeeded for ${target}"
}

do_gen_data() {
    log_info "Generating test data ..."
    (cd "${OP_SCRIPTS_DIR}" && python3 gen_data.py \
        "${M_RESOLVED}" "${K_RESOLVED}" "${N_RESOLVED}" \
        "${TA_RESOLVED}" "${TB_RESOLVED}" "${DTYPE_RESOLVED}" \
        "${BIAS_RESOLVED}" "${FORMAT_RESOLVED}")
    log_success "Test data generated"
}

do_run() {
    log_info "Running ${EXAMPLE} ..."
    local exec_path="${BUILD_DIR}/mat_mul/${SCENARIO_NAME}/${EXAMPLE}"
    if [[ ! -x "${exec_path}" ]]; then log_error "Executable not found: ${exec_path}"; return 1; fi
    if ! (cd "${OP_SCRIPTS_DIR}" && "${exec_path}" \
        "${M_RESOLVED}" "${K_RESOLVED}" "${N_RESOLVED}" \
        "${TA_RESOLVED}" "${TB_RESOLVED}" "${DTYPE_RESOLVED}" \
        "${HF32_RESOLVED}" "${BIAS_RESOLVED}" "${FORMAT_RESOLVED}"); then
        log_error "Execution FAILED"; return 1; fi
    log_success "Execution completed"; return 0
}

do_verify() {
    log_info "Verifying results ..."
    local hf32_flag=""
    if [[ "$HF32_RESOLVED" == "true" ]]; then hf32_flag="--hf32"; fi
    if ! (cd "${OP_SCRIPTS_DIR}" && python3 "${OP_SCRIPTS_DIR}/verify_result.py" \
        "${M_RESOLVED}" "${N_RESOLVED}" "${DTYPE_RESOLVED}" --bias "${BIAS_RESOLVED}" ${hf32_flag}); then
        log_error "Verification FAILED"; return 1; fi
    log_success "Verification passed"; return 0
}

cleanup_data() {
    rm -rf "${OP_SCRIPTS_DIR}/input" "${OP_SCRIPTS_DIR}/output"
}

run_csv_cases() {
    local csv_file="$1"
    if [[ ! -f "$csv_file" ]]; then log_error "CSV not found: $csv_file"; return 1; fi
    local target="${TARGET:-mat_mul_b_fullload}"
    local executable="${BUILD_DIR}/mat_mul/${SCENARIO_NAME}/${target}"
    if [[ ! -x "$executable" ]]; then log_error "Executable not found: $executable"; return 1; fi
    local result_file="${csv_file%.csv}_result.csv"
    python3 "${SCRIPT_DIR}/parse_csv.py" "$executable" "$csv_file" "$result_file"
    if [[ $? -eq 0 ]]; then log_success "All cases completed. Results: $result_file"; return 0
    else log_error "Some cases failed."; return 1; fi
}

run_example() {
    local example="$1"
    EXAMPLE="$example"
    resolve_params "$example"
    echo ""
    echo "========================================================================"
    echo "  Example: ${EXAMPLE}"
    echo "  Shape:   ${M_RESOLVED} x ${K_RESOLVED} x ${N_RESOLVED}"
    echo "  transA:  ${TA_RESOLVED}   transB:  ${TB_RESOLVED}"
    echo "  dtype:   ${DTYPE_RESOLVED}   isHf32: ${HF32_RESOLVED}"
    echo "  bias:    ${BIAS_RESOLVED}   format: ${FORMAT_RESOLVED}"
    echo "========================================================================"
    do_gen_data
    local exec_path="./build/mat_mul/${SCENARIO_NAME}/${EXAMPLE}"
    if [[ "$SKIP_BUILD" != true ]]; then do_build
    else
        if [[ ! -x "${exec_path}" ]]; then log_error "Executable not found: ${exec_path}"; return 1; fi
    fi
    if [[ "$BUILD_ONLY" == true ]]; then return 0; fi
    do_run; local run_result=$?
    if [[ $run_result -eq 0 ]]; then do_verify; run_result=$?; fi
    cleanup_data
    return $run_result
}

main() {
    echo "=========================================="
    echo "  mat_mul_b_fullload One-Click Runner"
    echo "  Scenario: ${SCENARIO_NAME}"
    echo "=========================================="
    preflight
    if [[ -n "$CASE_FILE" ]]; then
        if [[ "$SKIP_BUILD" != true ]]; then do_build; fi
        if [[ "$BUILD_ONLY" == true ]]; then return 0; fi
        run_csv_cases "$CASE_FILE"; local csv_result=$?
        cleanup_data; return $csv_result
    fi
    if [[ -n "$TARGET" ]]; then
        local registered; registered=$(discover_examples)
        if ! echo "$registered" | grep -qx "$TARGET"; then
            log_error "Target '${TARGET}' not registered"; exit 1; fi
        EXAMPLES="$TARGET"; MODE="single"
    else
        EXAMPLES="$(discover_examples)"; MODE="multi"
    fi
    log_info "Mode: ${MODE}"
    local all_passed=true; local results=""
    for example in $EXAMPLES; do
        set +e; run_example "$example"; local rc=$?; set -e
        if [[ $rc -eq 0 ]]; then
            [[ "$MODE" == "single" ]] && { echo ""; log_success "[PASS] ${example}"; }
            results="${results}${example}: PASS\n"
        else
            [[ "$MODE" == "single" ]] && { echo ""; log_error "[FAIL] ${example}"; }
            results="${results}${example}: FAIL\n"; all_passed=false
        fi
    done
    if [[ "$MODE" == "multi" ]]; then
        echo ""; echo "=========================================="
        echo "  Summary"; echo "=========================================="
        echo -e "$results" | grep -v '^$'; echo "=========================================="
        if [[ "$all_passed" == true ]]; then log_success "All PASSED"
        else log_error "Some FAILED"; fi
    fi
    [[ "$all_passed" != true ]] && exit 1
}

main "$@"
