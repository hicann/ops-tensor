#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
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
WORK_DIR="${SCRIPT_DIR}"

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

# ── Built-in Defaults ─────────────────────────────────────────────────────────
declare -A EXAMPLE_M EXAMPLE_K EXAMPLE_N EXAMPLE_BIAS EXAMPLE_ADTYPE EXAMPLE_BDTYPE \
          EXAMPLE_CDTYPE EXAMPLE_TA EXAMPLE_TB \
          EXAMPLE_BM EXAMPLE_BN EXAMPLE_BK \
          EXAMPLE_KL1 EXAMPLE_SKL1 EXAMPLE_L1BUF EXAMPLE_DBL0C EXAMPLE_AFULLLOAD EXAMPLE_FORMAT

EXAMPLE_M[quant_batch_matmul_mx]=64
EXAMPLE_K[quant_batch_matmul_mx]=128
EXAMPLE_N[quant_batch_matmul_mx]=128
EXAMPLE_BIAS[quant_batch_matmul_mx]=0
EXAMPLE_ADTYPE[quant_batch_matmul_mx]=fp8_e4m3
EXAMPLE_BDTYPE[quant_batch_matmul_mx]=fp8_e4m3
EXAMPLE_CDTYPE[quant_batch_matmul_mx]=float16
EXAMPLE_TA[quant_batch_matmul_mx]=false
EXAMPLE_TB[quant_batch_matmul_mx]=false
EXAMPLE_BM[quant_batch_matmul_mx]=64
EXAMPLE_BN[quant_batch_matmul_mx]=128
EXAMPLE_BK[quant_batch_matmul_mx]=64
EXAMPLE_KL1[quant_batch_matmul_mx]=64
EXAMPLE_SKL1[quant_batch_matmul_mx]=64
EXAMPLE_L1BUF[quant_batch_matmul_mx]=2
EXAMPLE_DBL0C[quant_batch_matmul_mx]=1
EXAMPLE_AFULLLOAD[quant_batch_matmul_mx]=false
EXAMPLE_FORMAT[quant_batch_matmul_mx]="(ND,ND)"

# ── CLI Defaults ──────────────────────────────────────────────────────────────
TARGET=""
CASE_FILE=""
SKIP_BUILD=false
BUILD_ONLY=false
M="" K="" N="" BIAS="" ADTYPE="" BDTYPE="" CDTYPE="" TA="" TB="" FORMAT=""
BM="" BN="" BK="" KL1="" SKL1="" L1BUF="" DBL0C="" AFULLLOAD=""

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
        --target=*) TARGET="${1#*=}"; shift ;;
        --target)   [[ -z "${2:-}" ]] && { log_error "--target needs a value"; exit 1; }; TARGET="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --build-only) BUILD_ONLY=true; shift ;;
        --case=*)   CASE_FILE="${1#*=}"; shift ;;
        --case)     [[ -z "${2:-}" ]] && { log_error "--case needs a value"; exit 1; }; CASE_FILE="$2"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        -*)         log_error "unknown option: $1"; usage; exit 1 ;;
        *)
            if [[ -z "$M" ]]; then M="$1"
            elif [[ -z "$K" ]]; then K="$1"
            elif [[ -z "$N" ]]; then N="$1"
            elif [[ -z "$BIAS" ]]; then BIAS="$1"
            elif [[ -z "$ADTYPE" ]]; then ADTYPE="$1"
            elif [[ -z "$BDTYPE" ]]; then BDTYPE="$1"
            elif [[ -z "$CDTYPE" ]]; then CDTYPE="$1"
            elif [[ -z "$TA" ]]; then TA="$1"
            elif [[ -z "$TB" ]]; then TB="$1"
            elif [[ -z "$FORMAT" ]]; then FORMAT="$1"
            elif [[ -z "$BM" ]]; then BM="$1"
            elif [[ -z "$BN" ]]; then BN="$1"
            elif [[ -z "$BK" ]]; then BK="$1"
            elif [[ -z "$KL1" ]]; then KL1="$1"
            elif [[ -z "$SKL1" ]]; then SKL1="$1"
            elif [[ -z "$L1BUF" ]]; then L1BUF="$1"
            elif [[ -z "$DBL0C" ]]; then DBL0C="$1"
            elif [[ -z "$AFULLLOAD" ]]; then AFULLLOAD="$1"
            else log_error "unexpected argument: $1"; usage; exit 1
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
        log_error "bisheng compiler not found in PATH"; missing=1
    else
        log_info "bisheng: $(bisheng --version 2>&1 | head -1)"
    fi

    if ! command -v g++ &>/dev/null; then
        log_error "g++ compiler not found in PATH"; missing=1
    else
        log_info "g++: $(g++ --version 2>&1 | head -1)"
    fi

    if ! command -v python3 &>/dev/null; then
        log_error "python3 not found in PATH"; missing=1
    else
        log_info "python3: $(python3 --version 2>&1)"
    fi

    if ! command -v cmake &>/dev/null; then
        log_error "cmake not found in PATH"; missing=1
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

# ── Resolve Parameters ───────────────────────────────────────────────────────
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
    ADTYPE_RESOLVED="${ADTYPE:-${EXAMPLE_ADTYPE[$example]}}"
    BDTYPE_RESOLVED="${BDTYPE:-${EXAMPLE_BDTYPE[$example]}}"
    CDTYPE_RESOLVED="${CDTYPE:-${EXAMPLE_CDTYPE[$example]}}"
    TA_RESOLVED="${TA:-${EXAMPLE_TA[$example]}}"
    TB_RESOLVED="${TB:-${EXAMPLE_TB[$example]}}"
    FORMAT_RESOLVED="${FORMAT:-${EXAMPLE_FORMAT[$example]}}"
    BM_RESOLVED="${BM:-${EXAMPLE_BM[$example]}}"
    BN_RESOLVED="${BN:-${EXAMPLE_BN[$example]}}"
    BK_RESOLVED="${BK:-${EXAMPLE_BK[$example]}}"
    KL1_RESOLVED="${KL1:-${EXAMPLE_KL1[$example]}}"
    SKL1_RESOLVED="${SKL1:-${EXAMPLE_SKL1[$example]}}"
    L1BUF_RESOLVED="${L1BUF:-${EXAMPLE_L1BUF[$example]}}"
    DBL0C_RESOLVED="${DBL0C:-${EXAMPLE_DBL0C[$example]}}"
    AFULLLOAD_RESOLVED="${AFULLLOAD:-${EXAMPLE_AFULLLOAD[$example]}}"
    FORMAT_RESOLVED="${FORMAT:-${EXAMPLE_FORMAT[$example]}}"
}

# ── Build ─────────────────────────────────────────────────────────────────────
source "${EXAMPLES_COMMON_DIR}/submodule_utils.sh"

do_build() {
    local target="${EXAMPLE:-${TARGET:-quant_batch_matmul_mx}}"

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
    log_info "Generating test data for ${EXAMPLE} (M=${M_RESOLVED} K=${K_RESOLVED} N=${N_RESOLVED}"
    log_info "  a_dtype=${ADTYPE_RESOLVED} b_dtype=${BDTYPE_RESOLVED} c_dtype=${CDTYPE_RESOLVED}"
    log_info "  transA=${TA_RESOLVED} transB=${TB_RESOLVED}) ..."

    local trans_a_flag=""
    local trans_b_flag=""
    [[ "$TA_RESOLVED" == "true" ]] && trans_a_flag="--trans-a"
    [[ "$TB_RESOLVED" == "true" ]] && trans_b_flag="--trans-b"

    (cd "${OP_SCRIPTS_DIR}" && python3 gen_data_mx.py \
        --m "${M_RESOLVED}" --k "${K_RESOLVED}" --n "${N_RESOLVED}" \
        --bias "${BIAS_RESOLVED}" \
        --a-dtype "${ADTYPE_RESOLVED}" --b-dtype "${BDTYPE_RESOLVED}" \
        --c-dtype "${CDTYPE_RESOLVED}" \
        --format "${FORMAT_RESOLVED}" \
        ${trans_a_flag} ${trans_b_flag} \
        --output-dir "${OP_SCRIPTS_DIR}/input")

    log_success "Test data generated"
}

# ── Execute ──────────────────────────────────────────────────────────────────
do_run() {
    log_info "Running ${EXAMPLE} ..."

    local exec_path="${BUILD_DIR}/quant_batch_matmul/${SCENARIO_NAME}/${EXAMPLE}"

    if [[ ! -x "${exec_path}" ]]; then
        log_error "Executable not found: ${exec_path}"
        return 1
    fi

    if ! (cd "${OP_SCRIPTS_DIR}" && "${exec_path}" \
        "${M_RESOLVED}" "${K_RESOLVED}" "${N_RESOLVED}" "${BIAS_RESOLVED}" \
        "${ADTYPE_RESOLVED}" "${BDTYPE_RESOLVED}" "${CDTYPE_RESOLVED}" \
        "${TA_RESOLVED}" "${TB_RESOLVED}" "${FORMAT_RESOLVED}" \
        "${BM_RESOLVED}" "${BN_RESOLVED}" "${BK_RESOLVED}" \
        "${KL1_RESOLVED}" "${SKL1_RESOLVED}" "${L1BUF_RESOLVED}" "${DBL0C_RESOLVED}" \
        "${AFULLLOAD_RESOLVED}"); then
        log_error "Execution FAILED for ${EXAMPLE}"
        return 1
    fi
    log_success "Execution completed for ${EXAMPLE}"
    return 0
}

# ── Verify ───────────────────────────────────────────────────────────────────
do_verify() {
    log_info "Verifying results for ${EXAMPLE} ..."

    if ! (cd "${OP_SCRIPTS_DIR}" && python3 verify_result_mx.py \
        ./input/golden_c.bin ./output/npu_out.bin \
        --dtype "${CDTYPE_RESOLVED}"); then
        log_error "Verification FAILED for ${EXAMPLE}"
        return 1
    fi
    log_success "Verification passed for ${EXAMPLE}"
    return 0
}

# ── Cleanup ──────────────────────────────────────────────────────────────────
cleanup_data() {
    log_info "Cleaning up generated data ..."
    rm -rf "${OP_SCRIPTS_DIR}/input"
    rm -rf "${OP_SCRIPTS_DIR}/output"
    log_success "Cleanup completed"
}

# ── Run CSV Cases ────────────────────────────────────────────────────────────
run_csv_cases() {
    local csv_file="$1"
    if [[ ! -f "$csv_file" ]]; then
        log_error "CSV file not found: $csv_file"
        return 1
    fi

    local target="${TARGET:-quant_batch_matmul_mx}"
    local executable="${BUILD_DIR}/quant_batch_matmul/${SCENARIO_NAME}/${target}"

    if [[ ! -x "$executable" ]]; then
        log_error "Executable not found: $executable"
        return 1
    fi

    local result_file="${csv_file%.csv}_result.csv"
    log_info "Running test cases from $csv_file"
    python3 "${SCRIPT_DIR}/parse_csv.py" "$executable" "$csv_file" "$result_file"
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        log_success "All test cases completed. Results: $result_file"
        return 0
    else
        log_error "Some test cases failed. Check $result_file for details."
        return 1
    fi
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
    echo "  A/B:     ${ADTYPE_RESOLVED} x ${BDTYPE_RESOLVED} -> ${CDTYPE_RESOLVED}"
    echo "  transA:  ${TA_RESOLVED}   transB: ${TB_RESOLVED}   format: ${FORMAT_RESOLVED}"
    echo "  Bias:    ${BIAS_RESOLVED}"
    echo "  Tile:    [${BM_RESOLVED}, ${BN_RESOLVED}, ${BK_RESOLVED}]"
    echo "  kL1/scaleKL1: ${KL1_RESOLVED} / ${SKL1_RESOLVED}  l1Buffers=${L1BUF_RESOLVED}"
    echo "  dbL0C=${DBL0C_RESOLVED}  aFullLoad=${AFULLLOAD_RESOLVED}  format=${FORMAT_RESOLVED}"
    echo "========================================================================"

    local exec_path="${BUILD_DIR}/quant_batch_matmul/${SCENARIO_NAME}/${EXAMPLE}"

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

    do_gen_data

    do_run
    local run_result=$?

    if [[ $run_result -eq 0 ]]; then
        do_verify
        run_result=$?
    fi

    cleanup_data
    return $run_result
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    echo "=========================================="
    echo "  quant_batch_matmul_mx One-Click Runner"
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
        run_csv_cases "$CASE_FILE"
        local csv_result=$?
        cleanup_data
        return $csv_result
    fi

    if [[ -n "$TARGET" ]]; then
        local registered
        registered=$(discover_examples)
        if ! echo "$registered" | grep -qx "$TARGET"; then
            log_error "Target '${TARGET}' not registered in CMakeLists.txt"
            log_info "Available examples:"
            echo "$registered" | while read -r v; do echo "  - $v"; done
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
                echo ""; log_success "[PASS] ${example}"
            fi
            results="${results}${example}: PASS\n"
        else
            if [[ "$MODE" == "single" ]]; then
                echo ""; log_error "[FAIL] ${example}"
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
