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

##############################################################################
# ops-tensor Examples — Shared Submodule Utilities
#
# Sourced by each example's run.sh to initialize the tensor_api submodule
# before build. All examples share this single implementation.
#
# Usage:
#   source "${EXAMPLES_COMMON_DIR}/submodule_utils.sh"
#   ensure_tensor_api_submodule "${REPO_ROOT}"
#
# Caller contract:
#   - Must define log_info / log_success / log_error functions before sourcing
#   - Pass repo root (absolute path) as $1 to ensure_tensor_api_submodule
##############################################################################

# 初始化 tensor_api submodule（使用 superproject 记录的 pinned commit）
# examples/CMakeLists.txt 的 include 路径只指向 submodule，不会回退到 CANN 环境
ensure_tensor_api_submodule() {
    local repo_root="${1:-.}"

    log_info "Initializing tensor_api submodule (at pinned commit)..."

    if ! command -v git &> /dev/null; then
        log_error "git is not installed, cannot initialize tensor_api submodule"
        return 1
    fi

    local _orig_dir="$(pwd)"
    cd "${repo_root}"

    if ! git rev-parse --is-inside-work-tree &> /dev/null; then
        log_error "Not a git repository: ${repo_root}"
        log_error "Cannot initialize tensor_api submodule. Please ensure ops-tensor is a git clone."
        cd "${_orig_dir}"
        return 1
    fi

    # --init / --recursive: 初始化 submodule 并递归处理嵌套 submodule
    if ! git submodule update --init --recursive include/tensor_api; then
        log_error "Failed to initialize tensor_api submodule"
        log_error "Manual recovery:"
        log_error "  cd ${repo_root} && git submodule update --init --recursive include/tensor_api"
        cd "${_orig_dir}"
        return 1
    fi

    cd "${_orig_dir}"
    log_success "tensor_api submodule initialized successfully"
}
