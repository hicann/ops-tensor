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
    local _submod_dir="${repo_root}/include/tensor_api"
    local _repo_url="https://gitcode.com/cann/asc-devkit.git"
    local _repo_branch="feature/tensor_api_from_9.0.0"

    if [ -d "$_submod_dir" ] && [ -n "$(ls -A "$_submod_dir" 2>/dev/null)" ]; then
        log_info "tensor_api already exists, skip init"
        return 0
    fi

    log_info "Initializing tensor_api submodule..."

    local _fail_msg=""
    if ! command -v git &> /dev/null; then
        _fail_msg="git is not installed"
    elif ! (cd "${repo_root}" && git rev-parse --is-inside-work-tree &> /dev/null); then
        _fail_msg="not a git repository: ${repo_root}"
    else
        if ! (cd "${repo_root}" && git submodule update --init --recursive include/tensor_api 2>&1); then
            _fail_msg="git submodule update failed"
        elif [ -z "$(ls -A "${_submod_dir}" 2>/dev/null)" ]; then
            _fail_msg="tensor_api directory is empty after init"
        fi
    fi

    if [ -n "$_fail_msg" ]; then
        log_error "${_fail_msg}"
        log_error "Manually download and place at: ${_submod_dir}"
        log_error "Repo: ${_repo_url} (branch: ${_repo_branch})"
        return 1
    fi

    log_success "tensor_api submodule initialized"
}
