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
# ops-tensor Uninstall Script
# ----------------------------------------------------------------------------

CURR_PATH=$(dirname "$(readlink -f "$0")")
OPP_PLATFORM_DIR="ops_tensor"

# Source common functions
OPP_COMMON_FILE="${CURR_PATH}/opp_common.sh"
if [ -f "${OPP_COMMON_FILE}" ]; then
  . "${OPP_COMMON_FILE}"
fi

# ==================== Argument Parsing ====================
IS_QUIET=n
QUIET_PARAMETER=""
RET_STATUS=""

if [ "$#" != "0" ]; then
  if [ "$1" = "--quiet" ] && [ "$#" = "1" ]; then
      IS_QUIET=y
      QUIET_PARAMETER="--quiet"
  else
      logandprint "Please use correct parameters, only support input nothing or only --quiet parameter."
      exit 1
  fi
fi

# Set install path from script location
INSTALLED_PATH="$(cd "${CURR_PATH}/../../../.."; pwd)"

logandprint "[INFO]: Starting ops_tensor uninstallation..."
logandprint "[INFO]: Target path: ${INSTALLED_PATH}"

# Check for install script
INSTALL_SHELL="${CURR_PATH}/install.sh"
if [ -f "${INSTALL_SHELL}" ]; then
  logandprint "[INFO]: Using install script for uninstallation..."
  PARENT_INSTALLED_PATH="$(cd "${INSTALLED_PATH}/.."; pwd)"

  sh "${INSTALL_SHELL}" "--aa" "--aa" "--uninstall" "--install-path=${INSTALLED_PATH}" "${QUIET_PARAMETER}"
  RET_STATUS="$?"

  if [ "${RET_STATUS}" != "0" ]; then
      logandprint "[ERROR]: Uninstallation failed with status ${RET_STATUS}"
      exit 1
  fi

  # Clean empty parent directory
  if [ -d "${PARENT_INSTALLED_PATH}" ]; then
      SUBDIRS_PARAM_INSTALL=$(ls "${PARENT_INSTALLED_PATH}" 2>/dev/null)
      if [ "${SUBDIRS_PARAM_INSTALL}" = "" ]; then
          rm -rf "${PARENT_INSTALLED_PATH}"
      fi
  fi
else
  logandprint "[ERROR]: install.sh not found at ${INSTALL_SHELL}"
  exit 1
fi

logandprint "[INFO]: Uninstallation completed successfully."
exit 0
