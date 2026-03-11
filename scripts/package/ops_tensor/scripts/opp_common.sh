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
# ops-tensor Common Functions Library
# ----------------------------------------------------------------------------

# ==================== Log Directory Initialization ====================
init_log_dir() {
  if [ "$(id -u)" != "0" ]; then
    _LOG_PATH="${HOME}/var/log/ascend_seclog"
  else
    _LOG_PATH="/var/log/ascend_seclog"
  fi
  _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"

  if [ ! -d "${_LOG_PATH}" ]; then
    mkdir -p "${_LOG_PATH}" 2>/dev/null || true
  fi
}

# ==================== Log Functions ====================
getdate() {
  local _cur_date=$(date +"%Y-%m-%d %H:%M:%S")
  echo "${_cur_date}"
}

logandprint() {
  local is_error_level=$(echo "$1" | grep -E 'ERROR|WARN|INFO')
  if [ "${is_quiet}" != "y" ] || [ "${is_error_level}" != "" ]; then
    echo "[OpsTensor] [$(getdate)] ""$1"
  fi
  if [ -d "${_LOG_PATH}" ]; then
    echo "[OpsTensor] [$(getdate)] ""$1" >>"${_INSTALL_LOG_FILE}"
  fi
}

# Initialize log directory when sourced
init_log_dir
