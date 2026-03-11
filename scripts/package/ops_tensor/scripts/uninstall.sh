#!/bin/bash
##############################################################################
# ops-tensor Uninstallation Script
#
# 功能: 卸载 ops-tensor 库
##############################################################################

# ==================== 错误码定义 ====================
OPERATE_FAILED="0x0001"
PARAM_INVALID="0x0002"
FILE_NOT_EXIST="0x0080"
FILE_NOT_EXIST_DES="File not found."

# ==================== 全局变量 ====================
OP_PLATFORM_DIR="ops_tensor"
OP_PLATFORM_UPPER=$(echo "${OP_PLATFORM_DIR}" | tr '[:lower:]' '[:upper:]')

# 默认安装路径
DEFAULT_INSTALL_PATH="/usr/local/Ascend"

# 脚本相关路径
CURR_PATH=$(dirname $(readlink -f $0))
VERSION_INFO_FILE="${CURR_PATH}/../version.info"

# 安装相关变量
ASCEND_INSTALL_INFO="ascend_install.info"
TARGET_INSTALL_PATH="${DEFAULT_INSTALL_PATH}"
TARGET_VERSION_DIR=""
TARGET_SHARED_INFO_DIR=""
PKG_VERSION_DIR="cann"
RUN_FROM_PACKAGE="false"

# ascend_install.info 中的 key
KEY_INSTALLED_UNAME="USERNAME"
KEY_INSTALLED_UGROUP="USERGROUP"
KEY_INSTALLED_PATH="${OP_PLATFORM_UPPER}_INSTALL_PATH"
KEY_INSTALLED_VERSION="${OP_PLATFORM_UPPER}_VERSION"

# 命令行参数
CMD_LIST="$*"
IS_QUIET=n
IS_UPGRADE=n

# 日志
COMM_LOGFILE="/tmp/ops_tensor_uninstall_$(date +%Y%m%d_%H%M%S).log"

# ==================== 日志函数 ====================

getdate() {
    date "+%Y-%m-%d %H:%M:%S"
}

logandprint() {
    local level="$1"
    local msg="$2"
    local log_msg="[${OP_PLATFORM_DIR}] [$(getdate)] [${level}]: ${msg}"
    echo "${log_msg}" | tee -a "${COMM_LOGFILE}"
}

log_info() {
    logandprint "INFO" "$1"
}

log_warning() {
    logandprint "WARNING" "$1"
}

log_error() {
    logandprint "ERROR" "$1"
}

startlog() {
    log_info "Start Time: $(getdate)"
    log_info "Input params: ${CMD_LIST}"
}

exitlog() {
    log_info "End Time: $(getdate)"
}

# ==================== 安装信息管理 ====================

get_installed_info() {
    local key="$1"
    local res=""
    if [ -f "${INSTALL_INFO_FILE}" ]; then
        chmod 644 "${INSTALL_INFO_FILE}" >/dev/null 2>&1
        res=$(grep "^${key}=" "${INSTALL_INFO_FILE}" | awk -F = '{print $2}')
    fi
    echo "${res}"
}

# ==================== 主卸载函数 ====================

uninstall_package() {
    log_info "========== Uninstalling ${OP_PLATFORM_DIR} =========="

    # 如果未指定安装路径，尝试从 ascend_install.info 读取
    if [ -z "${TARGET_VERSION_DIR}" ] && [ -f "${INSTALL_INFO_FILE}" ]; then
        local saved_path=$(get_installed_info "${KEY_INSTALLED_PATH}")
        if [ -n "${saved_path}" ]; then
            TARGET_VERSION_DIR="${saved_path}"
            log_info "Read install path from info file: ${TARGET_VERSION_DIR}"
        fi
    fi

    # 如果仍然没有路径，使用默认路径
    if [ -z "${TARGET_VERSION_DIR}" ]; then
        TARGET_VERSION_DIR="${TARGET_INSTALL_PATH}/${PKG_VERSION_DIR}"
        log_warning "Install path not specified, using default: ${TARGET_VERSION_DIR}"
    fi

    TARGET_SHARED_INFO_DIR="${TARGET_VERSION_DIR}/share/info"
    INSTALL_INFO_FILE="${TARGET_SHARED_INFO_DIR}/${OP_PLATFORM_DIR}/${ASCEND_INSTALL_INFO}"

    log_info "Install path: ${TARGET_VERSION_DIR}"

    # 检查安装信息文件
    if [ ! -f "${INSTALL_INFO_FILE}" ]; then
        log_warning "Installation info file not found: ${INSTALL_INFO_FILE}"
        log_warning "Package may not be installed."

        # 尝试删除文件（清理模式）
        local lib_file="${TARGET_VERSION_DIR}/lib/libops_tensor.so"
        local header_file="${TARGET_VERSION_DIR}/include/ops_tensor.h"
        local info_dir="${TARGET_SHARED_INFO_DIR}/${OP_PLATFORM_DIR}"

        [ -f "${lib_file}" ] && rm -f "${lib_file}" && log_info "Removed: ${lib_file}"
        [ -f "${header_file}" ] && rm -f "${header_file}" && log_info "Removed: ${header_file}"
        [ -d "${info_dir}" ] && rm -rf "${info_dir}" && log_info "Removed: ${info_dir}"

        # 清理空目录
        clean_empty_dirs

        log_info "========== Uninstall completed =========="
        return 0
    fi

    # 读取安装信息
    local installed_path=$(get_installed_info "${KEY_INSTALLED_PATH}")
    local installed_version=$(get_installed_info "${KEY_INSTALLED_VERSION}")
    local installed_user=$(get_installed_info "${KEY_INSTALLED_UNAME}")

    log_info "Installed path: ${installed_path}"
    log_info "Installed version: ${installed_version}"

    # 权限检查
    if [ "$(id -u)" != "0" ]; then
        if [ "${CURR_USER}" != "${installed_user}" ]; then
            log_error "Permission denied. Package was installed by ${installed_user}, current user is ${CURR_USER}"
            log_error "Please run uninstall as root or as the installing user."
            exitlog
            exit 1
        fi
    fi

    # 确认卸载
    if [ "${IS_QUIET}" = "n" ] && [ "${IS_UPGRADE}" = "n" ] && [ "${RUN_FROM_PACKAGE}" = "false" ]; then
        log_info "Do you want to uninstall ${OP_PLATFORM_DIR}? [y/n]"
        while true; do
            read yn
            if [ "$yn" = "n" ]; then
                log_info "Uninstall cancelled."
                exitlog
                exit 0
            elif [ "$yn" = "y" ]; then
                break
            else
                echo "Please input y or n."
            fi
        done
    fi

    # 删除文件
    local lib_file="${TARGET_VERSION_DIR}/lib/libops_tensor.so"
    local header_file="${TARGET_VERSION_DIR}/include/ops_tensor.h"
    local info_dir="${TARGET_SHARED_INFO_DIR}/${OP_PLATFORM_DIR}"

    if [ -f "${lib_file}" ]; then
        rm -f "${lib_file}"
        log_info "Removed: ${lib_file}"
    else
        log_warning "Library not found: ${lib_file}"
    fi

    if [ -f "${header_file}" ]; then
        rm -f "${header_file}"
        log_info "Removed: ${header_file}"
    else
        log_warning "Header not found: ${header_file}"
    fi

    # 删除脚本和配置
    if [ -d "${info_dir}" ]; then
        rm -rf "${info_dir}"
        log_info "Removed: ${info_dir}"
    fi

    # 删除安装信息文件
    if [ -f "${INSTALL_INFO_FILE}" ]; then
        rm -f "${INSTALL_INFO_FILE}"
        log_info "Removed: ${INSTALL_INFO_FILE}"
    fi

    # 清理空目录
    clean_empty_dirs

    log_info "========== Uninstall completed successfully =========="
    return 0
}

# 清理空目录
clean_empty_dirs() {
    # 清理 lib 目录
    if [ -d "${TARGET_VERSION_DIR}/lib" ] && [ -z "$(ls -A "${TARGET_VERSION_DIR}/lib")" ]; then
        rm -rf "${TARGET_VERSION_DIR}/lib"
        log_info "Removed empty directory: ${TARGET_VERSION_DIR}/lib"
    fi

    # 清理 include 目录
    if [ -d "${TARGET_VERSION_DIR}/include" ] && [ -z "$(ls -A "${TARGET_VERSION_DIR}/include")" ]; then
        rm -rf "${TARGET_VERSION_DIR}/include"
        log_info "Removed empty directory: ${TARGET_VERSION_DIR}/include"
    fi

    # 清理 share/info 目录
    if [ -d "${TARGET_SHARED_INFO_DIR}" ] && [ -z "$(ls -A "${TARGET_SHARED_INFO_DIR}")" ]; then
        rm -rf "${TARGET_SHARED_INFO_DIR}"
        log_info "Removed empty directory: ${TARGET_SHARED_INFO_DIR}"
    fi

    # 清理版本目录
    if [ -d "${TARGET_VERSION_DIR}" ] && [ -z "$(ls -A "${TARGET_VERSION_DIR}")" ]; then
        rm -rf "${TARGET_VERSION_DIR}"
        log_info "Removed empty directory: ${TARGET_VERSION_DIR}"
    fi
}

# ==================== 参数解析 ====================

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  --install-path=PATH    Installation directory (if not specified, will read from install info)
  --quiet                Quiet mode, no confirmation prompts
  --upgrade              Upgrade mode (no confirmation)
  --help                 Show this help message

Examples:
  $0                                    # Uninstall from recorded path
  $0 --install-path=/usr/local/Ascend   # Uninstall from specific path
  $0 --quiet                            # Uninstall without confirmation

Environment Variables:
  TARGET_ENV        Installation directory

EOF
}

parse_params() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --install-path=*)
                local path="${1#*=}"
                TARGET_VERSION_DIR="${path}"
                shift
                ;;
            --quiet)
                IS_QUIET=y
                shift
                ;;
            --upgrade)
                IS_UPGRADE=y
                IS_QUIET=y
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # 支持 TARGET_ENV 环境变量
    if [ -n "${TARGET_ENV}" ]; then
        if [[ "${TARGET_ENV}" == */cann ]]; then
            TARGET_VERSION_DIR="${TARGET_ENV}"
        else
            TARGET_VERSION_DIR="${TARGET_ENV}/cann"
        fi
    fi
}

# ==================== 主函数 ====================

main() {
    startlog

    # 解析参数
    parse_params "$@"

    # 执行卸载
    uninstall_package
    local ret=$?

    exitlog

    if [ ${ret} -eq 0 ]; then
        log_info "Uninstall completed successfully!"
        exit 0
    else
        log_error "Uninstall failed with error code: ${ret}"
        exit ${ret}
    fi
}

# 运行主函数
main "$@"
