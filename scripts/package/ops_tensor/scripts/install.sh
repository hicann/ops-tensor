#!/bin/bash
##############################################################################
# ops-tensor Installation Script
#
# 功能: 安装、卸载、升级 ops-tensor 库
# 支持: 安装、卸载、升级、静默安装
##############################################################################

# ==================== 错误码定义 ====================
OPERATE_FAILED="0x0001"
PARAM_INVALID="0x0002"
FILE_NOT_EXIST="0x0080"
FILE_NOT_EXIST_DES="File not found."
PERM_DENIED="0x0093"
PERM_DENIED_DES="Permission denied."
VERSION_CONFLICT="0x0094"
VERSION_CONFLICT_DES="Version conflict."

# ==================== 全局变量 ====================
OP_PLATFORM_DIR="ops_tensor"
OP_PLATFORM_UPPER=$(echo "${OP_PLATFORM_DIR}" | tr '[:lower:]' '[:upper:]')

# 当前操作用户
CURR_OPERATE_USER="$(id -nu 2>/dev/null)"
CURR_OPERATE_GROUP="$(id -ng 2>/dev/null)"

# 默认安装路径
if [ "$(id -u)" != "0" ]; then
    IS_FOR_ALL="n"
    DEFAULT_INSTALL_PATH="${HOME}/Ascend"
else
    IS_FOR_ALL="y"
    DEFAULT_INSTALL_PATH="/usr/local/Ascend"
fi

# 脚本相关路径
CURR_PATH=$(dirname $(readlink -f $0))
VERSION_INFO_FILE="${CURR_PATH}/../version.info"
SCENE_INFO_FILE="${CURR_PATH}/../scene.info"

# 安装相关变量
ASCEND_INSTALL_INFO="ascend_install.info"
TARGET_INSTALL_PATH="${DEFAULT_INSTALL_PATH}"
TARGET_USERNAME="${CURR_OPERATE_USER}"
TARGET_USERGROUP="${CURR_OPERATE_GROUP}"
TARGET_VERSION_DIR=""
TARGET_SHARED_INFO_DIR=""
PKG_VERSION_DIR="cann"

# ascend_install.info 中的 key
KEY_INSTALLED_UNAME="USERNAME"
KEY_INSTALLED_UGROUP="USERGROUP"
KEY_INSTALLED_PATH="${OP_PLATFORM_UPPER}_INSTALL_PATH"
KEY_INSTALLED_VERSION="${OP_PLATFORM_UPPER}_VERSION"

# 命令行参数
CMD_LIST="$*"
IS_UNINSTALL=n
IS_INSTALL=n
IS_UPGRADE=n
IS_QUIET=n
IS_CHECK=n
IS_SETENV=n
IN_INSTALL_TYPE="all"
CONFLICT_CMD_NUMS=0

# 日志
COMM_LOGFILE="/tmp/ops_tensor_install_$(date +%Y%m%d_%H%M%S).log"

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

# 从 ascend_install.info 获取已安装信息
get_installed_info() {
    local key="$1"
    local res=""
    if [ -f "${INSTALL_INFO_FILE}" ]; then
        chmod 644 "${INSTALL_INFO_FILE}" >/dev/null 2>&1
        res=$(grep "^${key}=" "${INSTALL_INFO_FILE}" | awk -F = '{print $2}')
    fi
    echo "${res}"
}

# 写入安装信息
set_installed_info() {
    local key="$1"
    local value="$2"

    if [ ! -f "${INSTALL_INFO_FILE}" ]; then
        touch "${INSTALL_INFO_FILE}"
        chmod 644 "${INSTALL_INFO_FILE}"
    fi

    # 如果 key 已存在，删除旧值
    if grep -q "^${key}=" "${INSTALL_INFO_FILE}"; then
        sed -i "/^${key}=/d" "${INSTALL_INFO_FILE}"
    fi

    # 添加新值
    echo "${key}=${value}" >> "${INSTALL_INFO_FILE}"
}

# 从 version.info 获取版本号
get_package_version() {
    local var_name="$1"
    local version_file="$2"
    if [ -f "${version_file}" ]; then
        local version=$(grep "^Version=" "${version_file}" | awk -F = '{print $2}')
        eval "${var_name}='${version}'"
    else
        eval "${var_name}=''"
    fi
}

# ==================== 路径和权限检查 ====================

# 检查路径有效性
check_install_path_valid() {
    local path="$1"
    # 只允许字母、数字、-、_、/
    if echo "${path}" | grep -vqE '^[a-zA-Z0-9/_-]+$'; then
        log_error "Install path contains invalid characters. Only [a-z,A-Z,0-9,-,_,/] are allowed."
        return 1
    fi
    return 0
}

# 创建安装目录
mkdir_install_path() {
    local base_dir=$(dirname ${TARGET_INSTALL_PATH})

    if [ ! -d ${base_dir} ]; then
        log_error "Base directory does not exist: ${base_dir}"
        exitlog
        exit 1
    fi

    if [ -d "${TARGET_INSTALL_PATH}" ]; then
        test -w ${TARGET_INSTALL_PATH} >>/dev/null 2>&1
        if [ "$?" -ne 0 ]; then
            log_error "ERR_NO:${PERM_DENIED};ERR_DES:${PERM_DENIED_DES}. No write permission for ${TARGET_INSTALL_PATH}"
            exit 1
        fi
    else
        test -w ${base_dir} >>/dev/null 2>&1
        if [ "$?" -ne 0 ]; then
            log_error "ERR_NO:${PERM_DENIED};ERR_DES:${PERM_DENIED_DES}. No write permission for ${base_dir}"
            exit 1
        else
            mkdir -p "${TARGET_INSTALL_PATH}"
            chmod 750 "${TARGET_INSTALL_PATH}"
            log_info "Created install directory: ${TARGET_INSTALL_PATH}"
        fi
    fi
}

# ==================== 版本检查和清理 ====================

# 重新安装前清理
clean_before_reinstall() {
    local installed_path=$(get_installed_info "${KEY_INSTALLED_PATH}")
    local existed_files=$(find "${TARGET_SHARED_INFO_DIR}/${OP_PLATFORM_DIR}" -type f -print 2>/dev/null)

    if [ -z "${existed_files}" ]; then
        log_info "Directory is empty, proceed with installation."
        return 0
    fi

    if [ "${IS_QUIET}" = "y" ]; then
        log_warning "Directory has existing files. Continuing installation in quiet mode."
    else
        local installed_version=$(get_installed_info "${KEY_INSTALLED_VERSION}")
        if [ -n "${installed_version}" ]; then
            log_info "Package already installed at ${installed_path}, version: ${installed_version}"
            log_info "New package version: ${RUN_PKG_VERSION}"
        fi
        log_info "Do you want to continue? [y/n]"
        while true; do
            read yn
            if [ "$yn" = "n" ]; then
                log_info "Installation cancelled."
                exitlog
                exit 0
            elif [ "$yn" = "y" ]; then
                break
            else
                echo "Please input y or n."
            fi
        done
    fi

    if [ "${installed_path}" = "${TARGET_VERSION_DIR}" ]; then
        log_info "Cleaning previously installed module before reinstall."
        if [ ! -f "${UNINSTALL_SHELL_FILE}" ]; then
            log_error "Uninstall script not found: ${UNINSTALL_SHELL_FILE}"
            return 1
        fi
        bash "${UNINSTALL_SHELL_FILE}" "${TARGET_VERSION_DIR}" "upgrade" "${IS_QUIET}"
        if [ "$?" != 0 ]; then
            log_error "Failed to clean previous installation."
            return 1
        fi
    fi
    return 0
}

# 检查版本兼容性
check_version_conflict() {
    local installed_version=$(get_installed_info "${KEY_INSTALLED_VERSION}")
    if [ -n "${installed_version}" ] && [ -n "${RUN_PKG_VERSION}" ]; then
        if [ "${installed_version}" != "${RUN_PKG_VERSION}" ]; then
            if [ "${IS_UPGRADE}" = "n" ]; then
                log_error "ERR_NO:${VERSION_CONFLICT};ERR_DES:${VERSION_CONFLICT_DES}"
                log_error "Installed version: ${installed_version}, Package version: ${RUN_PKG_VERSION}"
                log_error "Please use --upgrade to upgrade or --uninstall to remove old version first."
                exitlog
                exit 1
            fi
        fi
    fi
}

# ==================== 文件操作 ====================

# 创建目录
comm_create_dir() {
    local dir_path="$1"
    local perm="$2"
    local owner="$3"
    local is_for_all="$4"

    if [ ! -d "${dir_path}" ]; then
        mkdir -p "${dir_path}"
        chmod "${perm}" "${dir_path}"
        if [ "${is_for_all}" = "y" ]; then
            chown "root:root" "${dir_path}"
        else
            chown "${owner}" "${dir_path}"
        fi
        log_info "Created directory: ${dir_path}"
    fi
}

# 拷贝文件
comm_copy_file() {
    local src="$1"
    local dst="$2"
    local perm="$3"
    local owner="$4"
    local is_for_all="$5"

    if [ -f "${src}" ]; then
        cp -f "${src}" "${dst}"
        chmod "${perm}" "${dst}"
        if [ "${is_for_all}" = "y" ]; then
            chown "root:root" "${dst}"
        else
            chown "${owner}" "${dst}"
        fi
        log_info "Installed: ${dst}"
    else
        log_error "Source file not found: ${src}"
        return 1
    fi
}

# ==================== 主要功能函数 ====================

# 初始化环境变量
init_env() {
    startlog

    # 处理版本目录
    if is_version_dirpath "$TARGET_INSTALL_PATH"; then
        PKG_VERSION_DIR="$(basename "$TARGET_INSTALL_PATH")"
        TARGET_INSTALL_PATH="$(dirname "$TARGET_INSTALL_PATH")"
    fi

    TARGET_VERSION_DIR="$TARGET_INSTALL_PATH/$PKG_VERSION_DIR"
    TARGET_SHARED_INFO_DIR="${TARGET_VERSION_DIR}/share/info"
    INSTALL_INFO_FILE="${TARGET_SHARED_INFO_DIR}/${OP_PLATFORM_DIR}/${ASCEND_INSTALL_INFO}"
    UNINSTALL_SHELL_FILE="${TARGET_SHARED_INFO_DIR}/${OP_PLATFORM_DIR}/scripts/uninstall.sh"

    log_info "Install path: ${TARGET_INSTALL_PATH}"
    log_info "Version dir: ${TARGET_VERSION_DIR}"
    log_info "Info dir: ${TARGET_SHARED_INFO_DIR}"

    # 获取包版本
    get_package_version "RUN_PKG_VERSION" "${VERSION_INFO_FILE}"
    local installed_version=$(get_installed_info "${KEY_INSTALLED_VERSION}")

    if [ -n "${installed_version}" ]; then
        log_info "Installed version: ${installed_version}"
    fi

    if [ -n "${RUN_PKG_VERSION}" ]; then
        log_info "Package version: ${RUN_PKG_VERSION}"
    fi
}

# 检查是否为版本目录路径
is_version_dirpath() {
    local path="$1"
    local dirname=$(basename "$path")
    # 常见的版本目录名称
    for ver_dir in "cann" "CANN" "ascend" "ASCEND" "toolkit" "TOOLKIT"; do
        if [ "${dirname}" = "${ver_dir}" ]; then
            return 0
        fi
    done
    return 1
}

# 安装前检查
check_pre_install() {
    local installed_user=$(get_installed_info "${KEY_INSTALLED_UNAME}")
    local installed_group=$(get_installed_info "${KEY_INSTALLED_UGROUP}")

    if [ -n "${installed_user}" ] || [ -n "${installed_group}" ]; then
        if [ "${installed_user}" != "${TARGET_USERNAME}" ] || [ "${installed_group}" != "${TARGET_USERGROUP}" ]; then
            log_error "User/Group mismatch with previous installation."
            log_error "Installed as: ${installed_user}:${installed_group}"
            log_error "Current user: ${TARGET_USERNAME}:${TARGET_USERGROUP}"
            exitlog
            exit 1
        fi
    fi

    if [ "${IS_UPGRADE}" = "y" ]; then
        if [ ! -f "${INSTALL_INFO_FILE}" ]; then
            log_error "ERR_NO:${FILE_NOT_EXIST};ERR_DES:${FILE_NOT_EXIST_DES}"
            log_error "Package not installed at ${TARGET_INSTALL_PATH}, upgrade failed."
            exitlog
            exit 1
        fi
    fi
}

# 安装包
install_package() {
    if [ "${IS_INSTALL}" = "n" ] && [ "${IS_UPGRADE}" = "n" ]; then
        return
    fi

    log_info "========== Installing ${OP_PLATFORM_DIR} =========="

    # 创建目录结构
    mkdir_install_path

    local lib_dir="${TARGET_VERSION_DIR}/lib"
    local include_dir="${TARGET_VERSION_DIR}/include"
    local info_dir="${TARGET_SHARED_INFO_DIR}/${OP_PLATFORM_DIR}"
    local script_dir="${info_dir}/scripts"

    # 创建目录
    comm_create_dir "${lib_dir}" "755" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"
    comm_create_dir "${include_dir}" "755" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"
    comm_create_dir "${info_dir}" "555" "root:root" "y"
    comm_create_dir "${script_dir}" "555" "root:root" "y"

    # 获取包根目录
    PKG_ROOT="${CURR_PATH}/../.."

    # 安装库文件
    if [ -f "${PKG_ROOT}/lib/libops_tensor.so" ]; then
        comm_copy_file "${PKG_ROOT}/lib/libops_tensor.so" \
                      "${lib_dir}/libops_tensor.so" \
                      "550" \
                      "${TARGET_USERNAME}:${TARGET_USERGROUP}" \
                      "${IS_FOR_ALL}"
    else
        log_error "Library file not found: ${PKG_ROOT}/lib/libops_tensor.so"
        return 1
    fi

    # 安装头文件
    if [ -f "${PKG_ROOT}/include/ops_tensor.h" ]; then
        comm_copy_file "${PKG_ROOT}/include/ops_tensor.h" \
                      "${include_dir}/ops_tensor.h" \
                      "440" \
                      "${TARGET_USERNAME}:${TARGET_USERGROUP}" \
                      "${IS_FOR_ALL}"
    else
        log_error "Header file not found: ${PKG_ROOT}/include/ops_tensor.h"
        return 1
    fi

    # 安装版本信息
    if [ -f "${VERSION_INFO_FILE}" ]; then
        comm_copy_file "${VERSION_INFO_FILE}" \
                      "${info_dir}/version.info" \
                      "440" \
                      "root:root" \
                      "y"
    fi

    # 安装脚本
    comm_copy_file "${CURR_PATH}/install.sh" \
                  "${script_dir}/install.sh" \
                  "555" \
                  "root:root" \
                  "y"

    comm_copy_file "${CURR_PATH}/uninstall.sh" \
                  "${script_dir}/uninstall.sh" \
                  "555" \
                  "root:root" \
                  "y"

    # 记录安装信息
    set_installed_info "${KEY_INSTALLED_UNAME}" "${TARGET_USERNAME}"
    set_installed_info "${KEY_INSTALLED_UGROUP}" "${TARGET_USERGROUP}"
    set_installed_info "${KEY_INSTALLED_PATH}" "${TARGET_VERSION_DIR}"
    set_installed_info "${KEY_INSTALLED_VERSION}" "${RUN_PKG_VERSION}"

    log_info "========== Installation completed =========="
    return 0
}

# 卸载包
uninstall_package() {
    if [ "${IS_UNINSTALL}" = "n" ]; then
        return
    fi

    log_info "========== Uninstalling ${OP_PLATFORM_DIR} =========="

    if [ ! -f "${INSTALL_INFO_FILE}" ]; then
        log_warning "Installation info file not found: ${INSTALL_INFO_FILE}"
        log_warning "Package may not be installed or installation info is corrupted."

        # 尝试删除文件
        local lib_file="${TARGET_VERSION_DIR}/lib/libops_tensor.so"
        local header_file="${TARGET_VERSION_DIR}/include/ops_tensor.h"
        local info_dir="${TARGET_SHARED_INFO_DIR}/${OP_PLATFORM_DIR}"

        [ -f "${lib_file}" ] && rm -f "${lib_file}"
        [ -f "${header_file}" ] && rm -f "${header_file}"
        [ -d "${info_dir}" ] && rm -rf "${info_dir}"

        # 尝试删除空目录
        if [ -d "${TARGET_VERSION_DIR}" ]; then
            if [ -z "$(ls -A "${TARGET_VERSION_DIR}")" ]; then
                rm -rf "${TARGET_VERSION_DIR}"
                log_info "Removed empty directory: ${TARGET_VERSION_DIR}"
            fi
        fi

        log_info "========== Uninstall completed =========="
        return 0
    fi

    # 读取安装信息
    local installed_path=$(get_installed_info "${KEY_INSTALLED_PATH}")
    local installed_version=$(get_installed_info "${KEY_INSTALLED_VERSION}")

    log_info "Installed path: ${installed_path}"
    log_info "Installed version: ${installed_version}"

    # 删除文件
    local lib_file="${TARGET_VERSION_DIR}/lib/libops_tensor.so"
    local header_file="${TARGET_VERSION_DIR}/include/ops_tensor.h"
    local info_dir="${TARGET_SHARED_INFO_DIR}/${OP_PLATFORM_DIR}"

    if [ -f "${lib_file}" ]; then
        rm -f "${lib_file}"
        log_info "Removed: ${lib_file}"
    fi

    if [ -f "${header_file}" ]; then
        rm -f "${header_file}"
        log_info "Removed: ${header_file}"
    fi

    if [ -d "${info_dir}" ]; then
        rm -rf "${info_dir}"
        log_info "Removed: ${info_dir}"
    fi

    # 删除安装信息文件
    rm -f "${INSTALL_INFO_FILE}"

    # 尝试删除空目录
    if [ -d "${TARGET_VERSION_DIR}/lib" ] && [ -z "$(ls -A "${TARGET_VERSION_DIR}/lib")" ]; then
        rm -rf "${TARGET_VERSION_DIR}/lib"
    fi

    if [ -d "${TARGET_VERSION_DIR}/include" ] && [ -z "$(ls -A "${TARGET_VERSION_DIR}/include")" ]; then
        rm -rf "${TARGET_VERSION_DIR}/include"
    fi

    if [ -d "${TARGET_VERSION_DIR}" ] && [ -z "$(ls -A "${TARGET_VERSION_DIR}")" ]; then
        rm -rf "${TARGET_VERSION_DIR}"
        log_info "Removed empty directory: ${TARGET_VERSION_DIR}"
    fi

    if [ -d "${TARGET_SHARED_INFO_DIR}" ] && [ -z "$(ls -A "${TARGET_SHARED_INFO_DIR}")" ]; then
        rm -rf "${TARGET_SHARED_INFO_DIR}"
    fi

    log_info "========== Uninstall completed =========="
    return 0
}

# ==================== 参数解析 ====================

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  --install              Install the package (default)
  --uninstall            Uninstall the package
  --upgrade              Upgrade the package
  --install-path=PATH    Installation directory (default: ${DEFAULT_INSTALL_PATH})
  --quiet                Quiet mode, no interactive prompts
  --help                 Show this help message

Examples:
  $0 --install                           # Install to default path
  $0 --install --install-path=/opt/ascend  # Install to custom path
  $0 --uninstall                         # Uninstall
  $0 --upgrade                           # Upgrade existing installation

Environment Variables:
  TARGET_ENV        Installation directory (overrides --install-path)

EOF
}

parse_params() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --install)
                IS_INSTALL=y
                shift
                ;;
            --uninstall)
                IS_UNINSTALL=y
                shift
                ;;
            --upgrade)
                IS_UPGRADE=y
                IS_INSTALL=y
                shift
                ;;
            --install-path=*)
                TARGET_INSTALL_PATH="${1#*=}"
                shift
                ;;
            --quiet)
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

    # 默认操作为安装
    if [ "${IS_INSTALL}" = "n" ] && [ "${IS_UNINSTALL}" = "n" ]; then
        IS_INSTALL=y
    fi

    # 检查冲突选项
    if [ "${IS_INSTALL}" = "y" ] && [ "${IS_UNINSTALL}" = "y" ]; then
        log_error "Cannot specify both --install and --uninstall"
        exit 1
    fi

    # 支持 TARGET_ENV 环境变量
    if [ -n "${TARGET_ENV}" ]; then
        TARGET_INSTALL_PATH="${TARGET_ENV}"
    fi
}

# ==================== 主函数 ====================

main() {
    # 解析参数
    parse_params "$@"

    # 初始化环境
    init_env

    # 安装前检查
    if [ "${IS_INSTALL}" = "y" ]; then
        check_pre_install
        if [ "${IS_UPGRADE}" = "n" ]; then
            clean_before_reinstall
        fi
    fi

    # 执行操作
    local ret=0
    if [ "${IS_INSTALL}" = "y" ]; then
        install_package
        ret=$?
    elif [ "${IS_UNINSTALL}" = "y" ]; then
        uninstall_package
        ret=$?
    fi

    exitlog

    if [ ${ret} -eq 0 ]; then
        log_info "Operation completed successfully!"
        exit 0
    else
        log_error "Operation failed with error code: ${ret}"
        exit ${ret}
    fi
}

# 运行主函数
main "$@"
