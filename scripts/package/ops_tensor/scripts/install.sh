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
# ops-tensor Installation Script (Built-in Package)
# ----------------------------------------------------------------------------

# ==================== Error Codes ====================
PARAM_INVALID="0x0002"
FILE_NOT_EXIST="0x0080"
PERM_DENIED="0x0093"
PERM_DENIED_DES="Permission denied."

# ==================== Global Variables ====================
OPP_PLATFORM_DIR="ops_tensor"
OPP_PLATFORM_UPPER=$(echo "${OPP_PLATFORM_DIR}" | tr '[:lower:]' '[:upper:]')
CURR_OPERATE_USER="$(id -nu 2>/dev/null)"
CURR_OPERATE_GROUP="$(id -ng 2>/dev/null)"

# Defaults for general user
if [ "$(id -u)" != "0" ]; then
  DEFAULT_INSTALL_PATH="${HOME}/Ascend"
else
  IS_FOR_ALL="y"
  DEFAULT_INSTALL_PATH="/usr/local/Ascend"
fi

# Run package's files info
CURR_PATH=$(dirname "$(readlink -f "$0")")
RUN_PKG_INFO_FILE="${CURR_PATH}/../scene.info"
VERSION_INFO_FILE="${CURR_PATH}/../version.info"
COMMON_INC_FILE="${CURR_PATH}/common_func.inc"
COMMON_FUNC_V2_PATH="${CURR_PATH}/common_func_v2.inc"
VERSION_CFG_PATH="${CURR_PATH}/version_cfg.inc"
COMMON_PARSER_FILE="${CURR_PATH}/install_common_parser.sh"
INSTALL_SHELL_FILE="${CURR_PATH}/opp_install.sh"
UNINSTALL_SHELL_FILE="${CURR_PATH}/opp_uninstall.sh"
OPP_COMMON_FILE="${CURR_PATH}/opp_common.sh"

# Source common functions if available
if [ -f "${OPP_COMMON_FILE}" ]; then
  . "${OPP_COMMON_FILE}"
fi
if [ -f "${COMMON_INC_FILE}" ]; then
  . "${COMMON_INC_FILE}"
fi
if [ -f "${COMMON_FUNC_V2_PATH}" ]; then
  . "${COMMON_FUNC_V2_PATH}"
fi
if [ -f "${VERSION_CFG_PATH}" ]; then
  . "${VERSION_CFG_PATH}"
fi

# Get architecture info from scene.info if available
if [ -f "${RUN_PKG_INFO_FILE}" ]; then
  ARCH_INFO=$(grep -e "arch" "$RUN_PKG_INFO_FILE" 2>/dev/null | cut --only-delimited -d"=" -f2-)
else
  ARCH_INFO=$(uname -m)
fi

# Installation info
ASCEND_INSTALL_INFO="ascend_install.info"
TARGET_INSTALL_PATH="${DEFAULT_INSTALL_PATH}"
TARGET_USERNAME="${CURR_OPERATE_USER}"
TARGET_USERGROUP="${CURR_OPERATE_GROUP}"
TARGET_VERSION_DIR=""
TARGET_SHARED_INFO_DIR=""

# Keys for ascend_install.info
KEY_INSTALLED_UNAME="USERNAME"
KEY_INSTALLED_UGROUP="USERGROUP"
KEY_INSTALLED_TYPE="${OPP_PLATFORM_UPPER}_INSTALL_TYPE"
KEY_INSTALLED_PATH="${OPP_PLATFORM_UPPER}_INSTALL_PATH_VAL"
KEY_INSTALLED_VERSION="${OPP_PLATFORM_UPPER}_VERSION"
KEY_INSTALLED_FEATURE="${OPP_PLATFORM_UPPER}_INSTALL_FEATURE"
KEY_INSTALLED_CHIP="${OPP_PLATFORM_UPPER}_INSTALL_CHIP"

# Keys for run package
KEY_RUNPKG_VERSION="Version"

# Command line options
CMD_LIST="$*"
IS_UNINSTALL=n
IS_INSTALL=n
IS_UPGRADE=n
IS_QUIET=n
IS_INPUT_PATH=n
IS_CHECK=n
IS_PRE_CHECK=n
IN_INSTALL_TYPE=""
IN_INSTALL_PATH=""
IS_DOCKER_INSTALL=n
IS_SETENV=n
IS_JIT=n
DOCKER_ROOT=""
CONFLICT_CMD_NUMS=0
IN_FEATURE="All"

# ==================== Log Helper Functions ====================
startlog() {
  logandprint "[INFO]: Start Time: $(getdate)"
}

exitlog() {
  logandprint "[INFO]: End Time: $(getdate)"
}

# ==================== Version Functions ====================
get_package_version() {
  local var_name="$1"
  local version_file="$2"

  if [ -f "${version_file}" ]; then
      local ver=$(grep "^${KEY_RUNPKG_VERSION}=" "${version_file}" 2>/dev/null | cut -d"=" -f2-)
      if [ -n "${ver}" ]; then
          eval "${var_name}=\"${ver}\""
          return
      fi
  fi
  eval "${var_name}=\"9.0.0\""
}

get_installed_info() {
  local key="$1"
  local res=""
  if [ -f "${INSTALL_INFO_FILE}" ]; then
      chmod 644 "${INSTALL_INFO_FILE}" >/dev/null 2>&1
      res=$(cat "${INSTALL_INFO_FILE}" 2>/dev/null | grep "${key}" | awk -F = '{print $2}')
  fi
  echo "${res}"
}

# ==================== Path Check Functions ====================
check_install_path_valid() {
  local path="$1"
  # 黑名单设置，不允许//，...这样的路径
  if echo "${path}" | grep -Eq '/{2,}|\.{3,}'; then
      return 1
  fi
  # 白名单设置，只允许常见字符
  if echo "${path}" | grep -Eq '^~?[a-zA-Z0-9./_-]*$'; then
      return 0
  else
      return 1
  fi
}

judgment_path() {
  check_install_path_valid "${1}"
  if [ $? -ne 0 ]; then
      logandprint "[ERROR]: The install path ${1} is invalid, only characters in [a-z,A-Z,0-9,-,_,/,.] are supported!"
      exitlog
      exit 1
  fi
}

check_install_path() {
  TARGET_INSTALL_PATH="$1"

  # Empty path check
  if [ "x${TARGET_INSTALL_PATH}" = "x" ]; then
      logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --install-path not support empty path."
      exitlog
      exit 1
  fi

  # Space check
  if echo "x${TARGET_INSTALL_PATH}" | grep -q " "; then
      logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --install-path cannot contain space character."
      exitlog
      exit 1
  fi

  # Delete last "/"
  local temp_path="${TARGET_INSTALL_PATH}"
  temp_path=$(echo "${temp_path%/}")
  if [ x"${temp_path}" = "x" ]; then
      temp_path="/"
  fi

  # Convert relative path to absolute path
  local prefix=$(echo "${temp_path}" | cut -d"/" -f1 | cut -d"~" -f1)
  if [ "x${prefix}" = "x" ]; then
      TARGET_INSTALL_PATH="${temp_path}"
  else
      prefix=$(echo "${RUN_PATH}" | cut -d"/" -f1 | cut -d"~" -f1)
      if [ x"${prefix}" = "x" ]; then
          TARGET_INSTALL_PATH="${RUN_PATH}/${temp_path}"
      else
          logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES: Run package path is invalid: $RUN_PATH"
          exitlog
          exit 1
      fi
  fi

  # Convert '~' to home path
  local home=$(echo "${TARGET_INSTALL_PATH}" | cut -d"~" -f1)
  if [ "x${home}" = "x" ]; then
      local temp_path_value=$(echo "${TARGET_INSTALL_PATH}" | cut -d"~" -f2)
      if [ "$(id -u)" -eq 0 ]; then
          TARGET_INSTALL_PATH="/root${temp_path_value}"
      else
          local home_path=$(eval echo "${HOME}")
          home_path=$(echo "${home_path}%/")
          TARGET_INSTALL_PATH="${home_path}${temp_path_value}"
      fi
  fi
}

get_run_path() {
  RUN_PATH=$(echo "$2" | cut -d"-" -f3-)
  if [ x"${RUN_PATH}" = x"" ]; then
      RUN_PATH=$(pwd)
  else
      RUN_PATH=$(echo "${RUN_PATH%/}")
      if [ "x${RUN_PATH}" = "x" ]; then
          RUN_PATH=$(pwd)
      fi
  fi
}

check_docker_path() {
  local docker_path="$1"
  if [ "${docker_path}" != "/"* ]; then
      logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --docker-root must be absolute path starting with /."
      exitlog
      exit 1
  fi
  if [ ! -d "${docker_path}" ]; then
      logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The directory:${docker_path} not exist."
      exitlog
      exit 1
  fi
}

# ==================== Version Directory Check ====================
is_version_dirpath() {
  local path="$1"
  local dirname=$(basename "$path")
  # Check if directory name looks like a version (e.g., cann, 9.0.0, etc.)
  if [[ "$dirname" =~ ^[0-9]+\.[0-9]+ ]] || [ "$dirname" = "cann" ]; then
      return 0
  fi
  return 1
}

# ==================== Argument Parsing ====================
get_opts() {
  # Skip first two parameters (run package and directory)
  local i=0
  while true; do
      if [ "x$1" = "x" ]; then
          break
      fi
      if [ "$(expr substr "$1" 1 2)" = "--" ]; then
          i=$(expr $i + 1)
      fi
      if [ $i -gt 2 ]; then
          break
      fi
      shift 1
  done

  if [ "$*" = "" ]; then
      logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:\
 only support one type: full/upgrade/uninstall, operation execute failed!\
 Please use [--help] to see the usage."
      exitlog
      exit 1
  fi

  while true; do
      case "$1" in
          --full)
              IN_INSTALL_TYPE=$(echo "${1}" | awk -F"--" '{print $2}')
              IS_INSTALL="y"
              CONFLICT_CMD_NUMS=$(expr "$CONFLICT_CMD_NUMS" + 1)
              shift
              ;;
          --upgrade)
              IS_UPGRADE="y"
              CONFLICT_CMD_NUMS=$(expr "$CONFLICT_CMD_NUMS" + 1)
              shift
              ;;
          --uninstall)
              IS_UNINSTALL="y"
              CONFLICT_CMD_NUMS=$(expr "$CONFLICT_CMD_NUMS" + 1)
              shift
              ;;
          --install-path=*)
              IS_INPUT_PATH="y"
              IN_INSTALL_PATH=$(echo "${1}" | cut -d"=" -f2-)
              judgment_path "${IN_INSTALL_PATH}"
              check_install_path "${IN_INSTALL_PATH}"
              shift
              ;;
          --quiet)
              IS_QUIET="y"
              shift
              ;;
          --install-for-all)
              IS_FOR_ALL="y"
              shift
              ;;
          -*)
              logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Unsupported parameter [$1],\
 operation execute failed. Please use [--help] to see the usage."
              exitlog
              exit 1
              ;;
          *)
              break
              ;;
      esac
  done
}

check_opts() {
  if [ "${CONFLICT_CMD_NUMS}" != 1 ]; then
      logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:\
 only support one type: full/upgrade/uninstall, operation execute failed!\
 Please use [--help] to see the usage."
      exitlog
      exit 1
  fi
}

# ==================== Environment Initialization ====================
init_env() {
  init_log_dir

  if is_version_dirpath "$TARGET_INSTALL_PATH"; then
      PKG_VERSION_DIR="$(basename "$TARGET_INSTALL_PATH")"
      TARGET_INSTALL_PATH="$(dirname "$TARGET_INSTALL_PATH")"
  else
      PKG_VERSION_DIR="cann"
  fi
  TARGET_VERSION_DIR="$TARGET_INSTALL_PATH/$PKG_VERSION_DIR"

  TARGET_SHARED_INFO_DIR="${TARGET_VERSION_DIR}/share/info"
  UNINSTALL_SHELL_FILE="${TARGET_SHARED_INFO_DIR}/${OPP_PLATFORM_DIR}/script/opp_uninstall.sh"
  INSTALL_INFO_FILE="${TARGET_SHARED_INFO_DIR}/${OPP_PLATFORM_DIR}/${ASCEND_INSTALL_INFO}"

  logandprint "[INFO]: Execute the ops_tensor run package."
  logandprint "[INFO]: OperationLogFile path: ${_INSTALL_LOG_FILE}."
  logandprint "[INFO]: Input params: $CMD_LIST"

  get_package_version "RUN_PKG_VERSION" "$VERSION_INFO_FILE"
  local installed_version=$(get_installed_info "${KEY_INSTALLED_VERSION}")
  if [ "${installed_version}" = "" ]; then
      logandprint "[INFO]: Version of installing ops_tensor module is ${RUN_PKG_VERSION}."
  else
      logandprint "[INFO]: Existed ops_tensor module version is ${installed_version}, the new version is ${RUN_PKG_VERSION}."
  fi
}

check_pre_install() {
  local installed_user=$(get_installed_info "${KEY_INSTALLED_UNAME}")
  local installed_group=$(get_installed_info "${KEY_INSTALLED_UGROUP}")

  if [ "${installed_user}" != "" ] || [ "${installed_group}" != "" ]; then
      if [ "${installed_user}" != "${TARGET_USERNAME}" ] || [ "${installed_group}" != "${TARGET_USERGROUP}" ]; then
          logandprint "[ERROR]: The user and group are not same with last installation, do not support overwriting installation!"
          exitlog
          exit 1
      fi
  fi

  if [ "${IS_UPGRADE}" = "y" ]; then
      if [ ! -e "${INSTALL_INFO_FILE}" ]; then
          logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The directory:${TARGET_INSTALL_PATH} has no ops_tensor installed, upgrade failed."
          exitlog
          exit 1
      fi
      IN_INSTALL_TYPE=$(get_installed_info "${KEY_INSTALLED_TYPE}")
  fi
}

mkdir_install_path() {
  local base_dir=$(dirname "${TARGET_INSTALL_PATH}")
  if [ ! -d "${base_dir}" ]; then
      logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The directory:${base_dir} not exist, please create this directory."
      exitlog
      exit 1
  fi

  if [ -d "${TARGET_INSTALL_PATH}" ]; then
      test -w "${TARGET_INSTALL_PATH}" >>/dev/null 2>&1
      if [ "$?" -ne 0 ]; then
          logandprint "[ERROR]: ERR_NO:${PERM_DENIED};ERR_DES:${PERM_DENIED_DES}. User ${TARGET_USERNAME} cannot access ${TARGET_INSTALL_PATH}."
          exit 1
      fi
  else
      test -w "${base_dir}" >>/dev/null 2>&1
      if [ "$?" -ne 0 ]; then
          logandprint "[ERROR]: ERR_NO:${PERM_DENIED};ERR_DES:${PERM_DENIED_DES}. User ${TARGET_USERNAME} cannot access ${base_dir}."
          exit 1
      else
          mkdir -p "${TARGET_INSTALL_PATH}"
          if [ "$(id -u)" -eq 0 ] && [ "${IS_FOR_ALL}" = "y" ]; then
              chmod 755 "${TARGET_INSTALL_PATH}"
          else
              chmod 750 "${TARGET_INSTALL_PATH}"
              chown "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${TARGET_INSTALL_PATH}" 2>/dev/null || true
          fi
      fi
  fi
}

# ==================== Clean Before Reinstall ====================
clean_before_reinstall() {
  local installed_path=$(get_installed_info "${KEY_INSTALLED_PATH}")
  local existed_files=$(find "${TARGET_SHARED_INFO_DIR}/${OPP_PLATFORM_DIR}" -type f -print 2>/dev/null)

  if [ -z "${existed_files}" ]; then
      logandprint "[INFO]: Directory is empty, directly install ops_tensor module."
      return 0
  fi

  if [ "${IS_QUIET}" = "y" ]; then
      logandprint "[WARNING]: Directory has files existed or ops_tensor installed, continuing installation."
  else
      if [ ! -f "${INSTALL_INFO_FILE}" ]; then
          logandprint "[INFO]: Directory has files existed, do you want to continue? [y/n]"
      else
          logandprint "[INFO]: ops_tensor package has been installed at $(get_installed_info "${KEY_INSTALLED_PATH}"), version $(get_installed_info "${KEY_INSTALLED_VERSION}"), this package version is ${RUN_PKG_VERSION}, do you want to continue? [y/n]"
      fi
      while true; do
          read yn
          if [ "$yn" = "n" ]; then
              logandprint "[INFO]: Exit installation."
              exitlog
              exit 0
          elif [ "$yn" = "y" ]; then
              break
          else
              echo "[WARNING]: Input error, please input y or n!"
          fi
      done
  fi

  if [ "${installed_path}" = "${TARGET_VERSION_DIR}" ]; then
      logandprint "[INFO]: Clean the installed ops_tensor module before install."
      if [ -f "${UNINSTALL_SHELL_FILE}" ]; then
          bash "${UNINSTALL_SHELL_FILE}" "${TARGET_VERSION_DIR}" "upgrade" "${IS_QUIET}" "${IN_FEATURE}" "${IS_DOCKER_INSTALL}" "${DOCKER_ROOT}" "$PKG_VERSION_DIR"
          if [ "$?" != "0" ]; then
              logandprint "[ERROR]: Failed to clean installed directory."
              return 1
          fi
      fi
  fi
  return 0
}

# ==================== Architecture Check ====================
check_arch() {
  local architecture=$(uname -m)
  if [ -n "${ARCH_INFO}" ] && [ "${architecture}" != "${ARCH_INFO}" ]; then
      logandprint "[WARNING]: The architecture of run package (${ARCH_INFO}) is different from current system (${architecture})."
  fi
}

# ==================== Install Functions ====================
install_package() {
  if [ "${IS_INSTALL}" = "n" ] && [ "${IS_UPGRADE}" = "n" ]; then
      return
  fi

  logandprint "[INFO]: ============================================"
  logandprint "[INFO]: Starting ops_tensor installation..."
  logandprint "[INFO]: Version: ${RUN_PKG_VERSION}"
  logandprint "[INFO]: Install path: ${TARGET_VERSION_DIR}"
  logandprint "[INFO]: Install type: ${IN_INSTALL_TYPE}"
  logandprint "[INFO]: ============================================"

  check_arch

  # Clean before reinstall
  clean_before_reinstall
  if [ "$?" != "0" ]; then
      logandprint "[ERROR]: Installation failed during cleanup."
      exitlog
      exit 1
  fi

  # Use opp_install.sh to install (通过 COMMON_PARSER_FILE 完成安装)
  if [ -f "${INSTALL_SHELL_FILE}" ]; then
      bash "${INSTALL_SHELL_FILE}" "${TARGET_INSTALL_PATH}" "${TARGET_USERNAME}" "${TARGET_USERGROUP}" \
          "${IN_FEATURE}" "${IN_INSTALL_TYPE}" "${IS_FOR_ALL}" "${IS_SETENV}" \
          "${IS_DOCKER_INSTALL}" "${DOCKER_ROOT}" "${PKG_VERSION_DIR}"
      if [ "$?" != "0" ]; then
          logandprint "[ERROR]: Installation failed."
          exitlog
          exit 1
      fi
  else
      logandprint "[ERROR]: opp_install.sh not found at ${INSTALL_SHELL_FILE}"
      exitlog
      exit 1
  fi

  logandprint "[INFO]: ============================================"
  logandprint "[INFO]: Installation completed successfully!"
  logandprint "[INFO]: Install path: ${TARGET_VERSION_DIR}/${OPP_PLATFORM_DIR}"
  logandprint "[INFO]:"
  logandprint "[INFO]: To use ops-tensor, you may need to set environment:"
  logandprint "[INFO]:   source ${TARGET_VERSION_DIR}/set_env.bash"
  logandprint "[INFO]: ============================================"
}

# ==================== Uninstall Functions ====================
uninstall_package() {
  if [ "${IS_UNINSTALL}" = "n" ]; then
      return
  fi

  logandprint "[INFO]: Starting uninstallation..."

  # Check if ops_tensor is installed by looking for install info file
  if [ ! -f "${INSTALL_INFO_FILE}" ]; then
      logandprint "[INFO]: ops_tensor is not installed (no install info found at ${INSTALL_INFO_FILE})"
      exitlog
      exit 0
  fi

  # Use opp_uninstall.sh to uninstall (通过 COMMON_PARSER_FILE 完成卸载)
  if [ -f "${UNINSTALL_SHELL_FILE}" ]; then
      bash "${UNINSTALL_SHELL_FILE}" "${TARGET_INSTALL_PATH}" "uninstall" "${IS_QUIET}" "${IN_FEATURE}" \
          "${IS_DOCKER_INSTALL}" "${DOCKER_ROOT}" "${PKG_VERSION_DIR}"
      if [ "$?" != "0" ]; then
          logandprint "[ERROR]: Uninstallation failed."
          exitlog
          exit 1
      fi
  else
      logandprint "[ERROR]: opp_uninstall.sh not found at ${UNINSTALL_SHELL_FILE}"
      exitlog
      exit 1
  fi

  logandprint "[INFO]: Uninstallation completed successfully."
}

# ==================== Main Function ====================
main() {
  get_run_path "$@"
  startlog
  get_opts "$@"
  check_opts
  init_env
  check_pre_install
  mkdir_install_path
  install_package
  uninstall_package
  exitlog
}

main "$@"
exit 0
