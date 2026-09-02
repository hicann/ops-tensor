#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

##############################################################################
# ops-tensor Examples — Unified Runner
# See --help for full option list.
##############################################################################

set -euo pipefail

# ── Key Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(dirname "${EXAMPLES_DIR}")"
BUILD_DIR="${EXAMPLES_DIR}/build"
RUN_CASE_PY="${SCRIPT_DIR}/run_case.py"

# ── Color Definitions ────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
# stderr：discover_examples 在 $(...) 中调用，stdout 会被吞掉，错误必须走 stderr 才可见
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── CLI State ────────────────────────────────────────────────────────────────
OPS_NAME=""
TARGET=""
CASE_FILE=""
TI_ARG=""
SKIP_BUILD=false
BUILD_ONLY=false
FORCE_SUBMODULE=false

# ── Usage ────────────────────────────────────────────────────────────────────
usage() {
    cat <<'EOF'
Usage: bash examples/common/run.sh [OPTIONS]

Compile and run ops-tensor example examples.

Options:
  --ops=<names>       Operator directory name(s), comma-separated.
                      Multiple: --ops=mat_mul,gmm (runs all examples under each).
                      Single or omitted: --target may be specified.
  --target=<names>    Example name(s), comma-separated.
                      Multiple: --target=mat_mul_basic,mat_mul_streamk
                      (requires single --ops, --ti not allowed).
  --case=<path>       CSV file with test cases (equals form only).
                      If omitted, auto-discovers {target}.csv in the example dir.
  --ti=<N>            Run only test case at index N (0-based).
  --ti=<N-M>          Run test cases from index N to M (inclusive).
                      Only allowed with single --ops and single --target.
  --skip-build       Skip the CMake build stage.
  --build-only       Only build; do not run or verify.
  --force-submodule  Force update the tensor_api submodule to the pinned commit
                     recorded in the superproject (instead of skipping when
                     already initialized). Applied once before running, also
                     effective with --skip-build / --build-only.
  -h, --help          Show this help message and exit.

Discovery rules:
  --ops=A --target=X  Run single example: examples/{A}/{X}/
  --ops=A --target=X,Y  Run examples/{A}/{X}/ and examples/{A}/{Y}/
  --ops=A,B           Run all examples under examples/{A}/ and examples/{B}/
  --ops=A             Run all examples under examples/{A}/
  (none)              Run all examples across all operators

  Directories named 'common/', 'scripts/', and 'build/' are always skipped.

Constraints:
  --ops multiple      --target not allowed, --ti not allowed
  --target multiple   --ops must be single, --ti not allowed
  --ti                Requires single --ops and single --target

Examples:
  bash examples/common/run.sh --ops=mat_mul
  bash examples/common/run.sh --ops=mat_mul,gmm
  bash examples/common/run.sh --ops=mat_mul --target=mat_mul_basic
  bash examples/common/run.sh --ops=mat_mul --target=mat_mul_basic,mat_mul_streamk
  bash examples/common/run.sh --ops=mat_mul --target=mat_mul_basic --ti=0-5
  bash examples/common/run.sh --skip-build --ops=mat_mul --target=mat_mul_basic
EOF
}

# ── Argument Parsing ─────────────────────────────────────────────────────────
# Strict: only --key=value forms. No positional args, no --case path (space).
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ops=*)
            OPS_NAME="${1#*=}"
            shift
            ;;
        --target=*)
            TARGET="${1#*=}"
            shift
            ;;
        --case=*)
            CASE_FILE="${1#*=}"
            shift
            ;;
        --ti=*)
            TI_ARG="${1#*=}"
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
        --force-submodule)
            FORCE_SUBMODULE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            echo ""
            usage
            exit 1
            ;;
        *)
            log_error "Positional arguments are not supported: '$1'"
            echo ""
            usage
            exit 1
            ;;
    esac
done

# ── Multi-value Validation ──────────────────────────────────────────────────
OPS_MULTI=false
TARGET_MULTI=false

if [[ -n "$OPS_NAME" && "$OPS_NAME" == *","* ]]; then
    OPS_MULTI=true
fi
if [[ -n "$TARGET" && "$TARGET" == *","* ]]; then
    TARGET_MULTI=true
fi

if [[ -n "$CASE_FILE" ]]; then
    if [[ -z "$OPS_NAME" || -z "$TARGET" || "$OPS_MULTI" == true || "$TARGET_MULTI" == true ]]; then
        log_error "--case requires single --ops and single --target"
        exit 1
    fi
fi

if [[ "$OPS_MULTI" == true ]]; then
    if [[ -n "$TARGET" ]]; then
        log_error "--target is not allowed when --ops has multiple values"
        exit 1
    fi
    if [[ -n "$TI_ARG" ]]; then
        log_error "--ti is not allowed when --ops has multiple values"
        exit 1
    fi
fi

if [[ "$TARGET_MULTI" == true ]]; then
    if [[ "$OPS_MULTI" == true ]]; then
        log_error "--ops must be single when --target has multiple values"
        exit 1
    fi
    if [[ -z "$OPS_NAME" ]]; then
        log_error "--ops is required when --target is specified"
        exit 1
    fi
    if [[ -n "$TI_ARG" ]]; then
        log_error "--ti is not allowed when --target has multiple values"
        exit 1
    fi
fi

# ── Pre-flight Checks ───────────────────────────────────────────────────────
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
# Outputs lines of "ops_name/target_name" pairs.
# Skips: common/, scripts/, and non-directory entries.
discover_examples() {
    local ops_filter="$1"
    local target_filter="$2"

    local -a ops_list=()
    local _op
    if [[ -n "$ops_filter" ]]; then
        IFS=',' read -ra ops_list <<< "$ops_filter"
        for _op in "${ops_list[@]}"; do
            _op="${_op## }"
            _op="${_op%% }"
            if [[ ! -d "${EXAMPLES_DIR}/${_op}" ]]; then
                log_error "Operator directory not found: ${EXAMPLES_DIR}/${_op}"
                exit 1
            fi
        done
    else
        local op_dir op_name
        for op_dir in "${EXAMPLES_DIR}"/*/; do
            [[ -d "$op_dir" ]] || continue
            op_name="$(basename "$op_dir")"
            [[ "$op_name" == "common" || "$op_name" == "scripts" || "$op_name" == "build" ]] && continue
            ops_list+=("$op_name")
        done
    fi

    local -a targets_list=()
    if [[ -n "$target_filter" ]]; then
        IFS=',' read -ra targets_list <<< "$target_filter"
    fi

    local _t
    for _op in "${ops_list[@]}"; do
        if [[ ${#targets_list[@]} -eq 0 ]]; then
            _discover_op_examples "$_op"
        else
            for _t in "${targets_list[@]}"; do
                _t="${_t## }"
                _t="${_t%% }"
                if [[ ! -d "${EXAMPLES_DIR}/${_op}/${_t}" ]]; then
                    log_error "Example directory not found: ${EXAMPLES_DIR}/${_op}/${_t}"
                    exit 1
                fi
                echo "${_op}/${_t}"
            done
        fi
    done
}

_discover_op_examples() {
    local op_name="$1"
    local op_dir="${EXAMPLES_DIR}/${op_name}"
    local example_dir example_name
    for example_dir in "${op_dir}"/*/; do
        [[ -d "$example_dir" ]] || continue
        example_name="$(basename "$example_dir")"
        [[ "$example_name" == "common" || "$example_name" == "scripts" || "$example_name" == "build" ]] && continue
        echo "${op_name}/${example_name}"
    done
}

# ── Build ────────────────────────────────────────────────────────────────────
source "${REPO_ROOT}/scripts/submodule_utils.sh"

configure_build() {
    if ! ensure_tensor_api_submodule "${REPO_ROOT}"; then
        log_error "Failed to initialize tensor_api submodule"
        return 1
    fi

    log_info "Configuring CMake (build dir: ${BUILD_DIR})..."
    cmake -B "${BUILD_DIR}" \
        -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
        "${EXAMPLES_DIR}"
}

build_target() {
    local target="$1"

    log_info "Building target: ${target}"
    if ! cmake --build "${BUILD_DIR}" --target "${target}" -j"$(nproc)"; then
        log_error "Build FAILED for target: ${target}"
        return 1
    fi

    log_success "Build succeeded for target: ${target}"
}

# ── Run Single example ─────────────────────────────────────────────────────
# Arguments: ops_name example_name
# Returns: 0 on PASS, 1 on FAIL
run_example() {
    local ops_name="$1"
    local example_name="$2"

    local example_dir="${EXAMPLES_DIR}/${ops_name}/${example_name}"
    local op_scripts_dir="${EXAMPLES_DIR}/${ops_name}/scripts"
    local exec_path="${BUILD_DIR}/${ops_name}/${example_name}"
    local conf_path="${example_dir}/${example_name}.conf"

    # ── Resolve CSV ──────────────────────────────────────────────────────
    local csv_file=""
    if [[ -n "$CASE_FILE" ]]; then
        csv_file="$CASE_FILE"
    else
        csv_file="${example_dir}/${example_name}.csv"
    fi

    if [[ ! -f "$csv_file" ]]; then
        log_error "CSV file not found: ${csv_file}"
        return 1
    fi

    # ── Build (unless skipped) ───────────────────────────────────────────
    if [[ "$SKIP_BUILD" != true ]]; then
        if ! build_target "$example_name"; then
            return 1
        fi
    fi

    if [[ "$BUILD_ONLY" == true ]]; then
        return 0
    fi

    # ── Validate executable ──────────────────────────────────────────────
    if [[ ! -x "$exec_path" ]]; then
        log_error "Executable not found: ${exec_path}"
        log_error "Build first or remove --skip-build."
        return 1
    fi

    # ── Validate .conf ───────────────────────────────────────────────────
    if [[ ! -f "$conf_path" ]]; then
        log_error "Config file not found: ${conf_path}"
        return 1
    fi

    # ── Prepare result path ──────────────────────────────────────────────
    local result_file="${csv_file%.csv}_result.csv"

    # ── Run via run_case.py ──────────────────────────────────────────────
    local run_args=("$exec_path" "$csv_file" "$result_file" "$conf_path")
    if [[ -n "$TI_ARG" ]]; then
        run_args+=("--ti=${TI_ARG}")
    fi

    local run_rc=0
    python3 "${RUN_CASE_PY}" "${run_args[@]}" || run_rc=$?

    # ── Cleanup ──────────────────────────────────────────────────────────
    cleanup_data "$op_scripts_dir"

    return $run_rc
}

# ── Cleanup ──────────────────────────────────────────────────────────────────
cleanup_data() {
    local op_scripts_dir="$1"
    if [[ -d "${op_scripts_dir}/input" ]]; then
        rm -rf "${op_scripts_dir}/input"
    fi
    if [[ -d "${op_scripts_dir}/output" ]]; then
        rm -rf "${op_scripts_dir}/output"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
main() {
    echo "=========================================="
    echo "  ops-tensor Examples Unified Runner"
    echo "=========================================="

    preflight

    # ── Force submodule sync (once, independent of --skip-build) ────────
    if [[ "$FORCE_SUBMODULE" == true ]]; then
        if ! ensure_tensor_api_submodule "${REPO_ROOT}" true; then
            log_error "Failed to force-update tensor_api submodule"
            exit 1
        fi
        FORCE_SUBMODULE=false
    fi

    # ── Discover examples ───────────────────────────────────────────────
    local examples
    examples="$(discover_examples "$OPS_NAME" "$TARGET")"

    if [[ -z "$examples" ]]; then
        log_error "No examples found (ops=${OPS_NAME:-<all>}, target=${TARGET:-<all>})"
        exit 1
    fi

    local example_count
    example_count="$(echo "$examples" | wc -l)"
    log_info "Discovered ${example_count} example(s)"

    # ── Configure once for all examples ─────────────────────────────────
    if [[ "$SKIP_BUILD" == true ]]; then
        log_info "Skipping build (--skip-build)"
    else
        if ! configure_build; then
            log_error "CMake configuration failed"
            exit 1
        fi
    fi
    if [[ "$BUILD_ONLY" == true ]]; then
        log_info "Build-only mode: no execution after build"
    fi

    # ── Run each example ────────────────────────────────────────────────
    local total_pass=0
    local total_fail=0
    local results=""

    while IFS= read -r example_path; do
        local ops_name
        ops_name="$(echo "$example_path" | cut -d'/' -f1)"
        local example_name
        example_name="$(echo "$example_path" | cut -d'/' -f2)"

        log_info "Running ${ops_name}/${example_name}"

        set +e
        run_example "$ops_name" "$example_name"
        local rc=$?
        set -e

        if [[ $rc -eq 0 ]]; then
            total_pass=$((total_pass + 1))
            results="${results}  ${ops_name}/${example_name}: PASS\n"
            log_success "[PASS] ${ops_name}/${example_name}"
        else
            total_fail=$((total_fail + 1))
            results="${results}  ${ops_name}/${example_name}: FAIL\n"
            log_error "[FAIL] ${ops_name}/${example_name}"
        fi
    done <<< "$examples"

    # ── Summary ──────────────────────────────────────────────────────────
    echo ""
    echo "=========================================="
    echo "  Summary"
    echo "=========================================="
    echo -e "$results" | grep -v '^$'
    echo "------------------------------------------"
    log_info "Total: ${example_count}  |  PASS: ${total_pass}  |  FAIL: ${total_fail}"
    echo "=========================================="

    if [[ $total_fail -gt 0 ]]; then
        log_error "Some examples FAILED"
        exit 1
    fi

    log_success "All examples PASSED"
}

main "$@"
