#!/bin/bash

##############################################################################
# ops-tensor Build Script
#
# 使用方法:
#   ./build.sh [OPTIONS]
#
# 选项:
#   --ops=OP_LIST    指定要编译的算子列表 (逗号分隔)
#   --run            编译后执行测试
#   --pkg            编译并打包成 .run 文件
#   --soc=SOC        指定目标 SoC 型号 (当前仅支持: Ascend950, 支持小写输入)
#   -j[N]            编译线程数，默认为 8，例如: -j16
#   -h, --help       显示帮助信息
#
# 行为说明:
#   无参数                编译所有算子，不执行测试
#   --ops=add             只编译 add 算子，不执行测试
#   --ops=add,sub         编译 add 和 sub 算子，不执行测试
#   --run                 编译所有算子，并执行所有算子的测试
#   --ops=add --run       编译 add 算子，并执行 add 算子的测试
#   --ops=add,sub --run   编译 add,sub 算子，并执行这些算子的测试
#   --pkg                 编译所有算子并打包成 .run 文件 (默认 SoC: Ascend950)
#   --ops=add --pkg       编译 add 算子并打包成 .run 文件
#   --soc=ascend950 --pkg  为 Ascend950 芯片打包 (支持小写)
#   -j16                  使用 16 个线程编译
#
# 示例:
#   ./build.sh                        # 编译所有算子 (默认 8 线程)
#   ./build.sh --ops=add              # 只编译 add 算子
#   ./build.sh -j16                   # 使用 16 线程编译所有算子
#   ./build.sh --run                  # 编译所有算子并执行测试
#   ./build.sh --ops=add --run        # 编译 add 算子并执行测试
#   ./build.sh --pkg                  # 编译所有算子并打包 (默认: Ascend950)
#   ./build.sh --ops=add --pkg        # 编译 add 算子并打包
#   ./build.sh --soc=ascend950 --pkg  # 为 Ascend950 打包 (支持小写输入)
#
# 支持的 SoC 型号:
#   Ascend950 (默认)
##############################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
BUILD_OPERATORS="all"
RUN_TESTS=false
ENABLE_PACKAGE=false
SOC_NAME="Ascend950"  # 默认 SoC 型号 (仅支持 Ascend950)
BUILD_DIR="build"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THREAD_NUM=8  # 默认编译线程数
CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# 设置 _ASCEND_INSTALL_PATH（优先级：ASCEND_INSTALL_PATH > ASCEND_HOME_PATH > 默认值）
if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
fi

# SoC 名称标准化函数（首字母大写，其余小写）
normalize_soc_name() {
    local soc="$1"
    # 转换为小写，然后首字母大写
    echo "${soc}" | sed 's/.*/\L&/; s/^./\U&/'
}

# SoC 名称映射到完整的 SOC_VERSION（CANN ASC 编译器要求的格式）
get_soc_version() {
    local soc_name="$1"
    local soc_lower=$(echo "$soc_name" | tr '[:upper:]' '[:lower:]')

    # 映射表：友好名称 -> CANN 期望的完整版本标识
    case "$soc_lower" in
        "ascend950")
            echo "ascend950dt_9595"
            ;;
        "ascend910b"|"ascend910_b")
            echo "ascend910b3"
            ;;
        "ascend910_93")
            echo "ascend910_93"
            ;;
        "ascend910")
            echo "ascend910"
            ;;
        "ascend310p"|"ascend310_p")
            echo "ascend310p"
            ;;
        "ascend310b"|"ascend310_b")
            echo "ascend310b"
            ;;
        *)
            # 如果没有映射，直接使用小写名称
            echo "$soc_lower"
            ;;
    esac
}

# 检查 ASCEND 环境变量
check_ascend_env() {
    log_info "Checking ASCEND environment..."

    if [ -z "${ASCEND_HOME_PATH}" ]; then
        log_error "ASCEND_HOME_PATH environment variable is not set!"
        echo ""
        echo "Please source the CANN environment script:"
        echo "  source /usr/local/Ascend/cann/set_env.sh"
        echo ""
        echo "Then run build.sh again."
        exit 1
    fi

    if [ ! -d "${ASCEND_HOME_PATH}" ]; then
        log_error "ASCEND_HOME_PATH directory does not exist: ${ASCEND_HOME_PATH}"
        exit 1
    fi

    log_success "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
}

# 日志函数（统一接口）
log() {
    local level=$1
    local msg=$2
    local color=""
    local prefix=""

    case $level in
        info)
            color="${BLUE}"
            prefix="[INFO]"
            ;;
        success)
            color="${GREEN}"
            prefix="[SUCCESS]"
            ;;
        warning)
            color="${YELLOW}"
            prefix="[WARNING]"
            ;;
        error)
            color="${RED}"
            prefix="[ERROR]"
            ;;
        verbose)
            if [ "$VERBOSE" != true ]; then
                return
            fi
            color="${BLUE}"
            prefix="[VERBOSE]"
            ;;
        *)
            color="${NC}"
            prefix="[LOG]"
            ;;
    esac

    echo -e "${color}${prefix}${NC} ${msg}"
}

# 兼容性别名
log_info() { log info "$1"; }
log_success() { log success "$1"; }
log_warning() { log warning "$1"; }
log_error() { log error "$1"; }
log_verbose() { log verbose "$1"; }

# 显示帮助信息
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --ops=OP_LIST    Specify operators to build (comma-separated)
  --run            Run tests after build
  --pkg            Build package (.run file)
  --soc=SOC        Target SoC model (default: Ascend950, case-insensitive)
  -j[N]            Number of compile threads (default: 8), e.g., -j16
  -h, --help       Show this help message

Supported SoC models:
  Ascend950    (dav-3101, default)
  Ascend910B   (dav-2201)
  Ascend910_93 (dav-2201)
  Ascend910    (dav-2101)
  Ascend310P   (dav-2101)

Examples:
  $(basename "$0")                          # Build all operators (default 8 threads)
  $(basename "$0") --ops=add                # Build only 'add' operator
  $(basename "$0") --ops=add,sub            # Build 'add' and 'sub' operators
  $(basename "$0") -j16                     # Build with 16 threads
  $(basename "$0") --run                    # Build all and run tests
  $(basename "$0") --ops=add --run          # Build 'add' and run tests
  $(basename "$0") --pkg                    # Build package (default SoC: Ascend950)
  $(basename "$0") --ops=add --pkg          # Build 'add' and package
  $(basename "$0") --soc=Ascend950 --pkg    # Build package for Ascend950
  $(basename "$0") --soc=ascend950 --pkg    # Build package (lowercase also works)

EOF
}

# 解析逗号分隔的算子列表
parse_operators() {
    local ops_str=$1
    if [ "$ops_str" = "all" ]; then
        echo "all"
    else
        IFS=',' read -ra OPS <<< "$ops_str"
        printf '%s\n' "${OPS[@]}"
    fi
}

# 获取所有可用的算子
get_available_operators() {
    find "$SCRIPT_DIR/src" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" 2>/dev/null | sort
}

# 验证指定的算子是否存在
validate_operators() {
    local ops=($1)
    local available_ops=($(get_available_operators))
    local invalid_ops=()

    for op in "${ops[@]}"; do
        local found=false
        for avail_op in "${available_ops[@]}"; do
            if [ "$op" = "$avail_op" ]; then
                found=true
                break
            fi
        done
        if [ "$found" = false ]; then
            invalid_ops+=("$op")
        fi
    done

    if [ ${#invalid_ops[@]} -gt 0 ]; then
        log_error "Invalid operators: ${invalid_ops[*]}"
        log_info "Available operators: ${available_ops[*]}"
        exit 1
    fi
}

# 清理构建目录
clean_build() {
    log_info "Cleaning build directory..."
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        log_success "Build directory cleaned"
    else
        log_warning "Build directory does not exist, nothing to clean"
    fi
}

# 构建项目
build_project() {
    log_info "Starting build..."

    # 创建构建目录
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # 配置CMake
    log_info "Configuring CMake..."
    local cmake_args=()

    if [ "$BUILD_OPERATORS" != "all" ]; then
        cmake_args+=(-DENABLED_OPERATORS="${BUILD_OPERATORS}")
        log_info "Enabled operators: ${BUILD_OPERATORS}"
    else
        log_info "All operators enabled"
    fi

    # 如果需要运行测试，启用测试编译
    if [ "$RUN_TESTS" = true ]; then
        cmake_args+=(-DBUILD_TESTING=ON)
        log_info "Test compilation: ENABLED"
    else
        cmake_args+=(-DBUILD_TESTING=OFF)
        log_info "Test compilation: DISABLED"
    fi

    # 如果需要打包，启用打包
    if [ "$ENABLE_PACKAGE" = true ]; then
        cmake_args+=(-DENABLE_PACKAGE=ON)
        log_info "Package generation: ENABLED"
    fi

    # 传递 SoC 型号和 ASCEND 路径
    # SOC_VERSION: ASC 编译器期望简化格式（如 ascend950，不是 ascend950dt_9595）
    # ASCEND_SOC: init_env.cmake 用于映射 NPU 架构
    SOC_NAME_LOWER=$(echo "${SOC_NAME}" | tr '[:upper:]' '[:lower:]')
    cmake_args+=(-DSOC_VERSION="${SOC_NAME_LOWER}")
    cmake_args+=(-DASCEND_SOC="${SOC_NAME}")
    cmake_args+=(-DASCEND_HOME_PATH="${ASCEND_HOME_PATH}")
    log_info "Target SoC: ${SOC_NAME}"
    log_info "  -> SOC_VERSION: ${SOC_NAME_LOWER} (for ASC compiler)"
    log_info "  -> ASCEND_SOC: ${SOC_NAME} (for NPU arch mapping)"
    log_info "ASCEND path: ${ASCEND_HOME_PATH}"

    cmake "${cmake_args[@]}" ..

    # 编译
    log_info "Compiling with ${THREAD_NUM} threads..."
    cmake --build . -j${THREAD_NUM}

    if [ $? -eq 0 ]; then
        log_success "Build succeeded"
    else
        log_error "Build failed"
        exit 1
    fi

    cd "$SCRIPT_DIR"
}

# 打包
build_package() {
    log_info "Building package..."

    cd "$BUILD_DIR"

    # 运行 ctest package target
    log_info "Running CPack..."
    cpack

    if [ $? -eq 0 ]; then
        log_success "Package created successfully"
        # 查找生成的 .run 文件
        local run_file=$(ls *.run 2>/dev/null | head -n 1)
        if [ -n "$run_file" ]; then
            log_success "Package file: $BUILD_DIR/$run_file"
        fi
    else
        log_error "Package creation failed"
        exit 1
    fi

    cd "$SCRIPT_DIR"
}

# 运行测试
run_tests() {
    log_info "Running tests with ctest..."

    cd "$BUILD_DIR"

    # 运行测试（all_ops_test 会自动运行所有已注册的测试）
    ctest --verbose

    local test_result=$?

    cd "$SCRIPT_DIR"

    if [ $test_result -ne 0 ]; then
        log_error "Some tests failed"
        exit 1
    else
        log_success "All tests passed"
    fi
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --ops=*)
                # 提取等号后的值
                BUILD_OPERATORS="${1#*=}"
                shift
                ;;
            --run)
                RUN_TESTS=true
                shift
                ;;
            --pkg)
                ENABLE_PACKAGE=true
                shift
                ;;
            --soc=*)
                # 提取 SoC 型号并标准化（支持小写输入）
                SOC_INPUT="${1#*=}"
                SOC_NAME=$(normalize_soc_name "${SOC_INPUT}")
                shift
                ;;
            -j*)
                # 提取线程数
                if [[ "$1" == "-j" ]]; then
                    # -j N 的形式
                    THREAD_NUM="$2"
                    shift 2
                else
                    # -jN 的形式
                    THREAD_NUM="${1#-j}"
                    shift
                fi
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo ""
                show_help
                exit 1
                ;;
        esac
    done
}

# 主函数
main() {
    echo "=========================================="
    echo "    ops-tensor Build Script"
    echo "=========================================="
    echo ""

    parse_arguments "$@"

    # 检查并调整线程数
    if [ "$THREAD_NUM" -gt "$CORE_NUMS" ]; then
        log_warning "Thread num $THREAD_NUM over core num $CORE_NUMS, adjust to $CORE_NUMS"
        THREAD_NUM=$CORE_NUMS
    fi

    # ASCEND 环境检查（强制要求）
    check_ascend_env

    # 验证算子
    if [ "$BUILD_OPERATORS" != "all" ]; then
        IFS=',' read -ra OPS <<< "$BUILD_OPERATORS"
        validate_operators "${OPS[*]}"
    fi

    # 显示执行计划
    echo ""
    log_info "========== Build Plan =========="
    if [ "$BUILD_OPERATORS" = "all" ]; then
        log_info "Build operators: all"
    else
        log_info "Build operators: ${BUILD_OPERATORS}"
    fi

    if [ "$RUN_TESTS" = true ]; then
        log_success "Run tests: YES"
        if [ "$BUILD_OPERATORS" != "all" ]; then
            log_info "Test operators: ${BUILD_OPERATORS}"
        else
            log_info "Test operators: all built operators"
        fi
    else
        log_info "Run tests: NO"
    fi

    if [ "$ENABLE_PACKAGE" = true ]; then
        log_success "Build package: YES"
        log_info "Target SoC: ${SOC_NAME}"
    else
        log_info "Build package: NO"
    fi
    echo "=================================="
    echo ""

    # 构建项目
    build_project

    # 运行测试（如果指定）
    if [ "$RUN_TESTS" = true ]; then
        run_tests
    fi

    # 打包（如果指定）
    if [ "$ENABLE_PACKAGE" = true ]; then
        build_package
    fi

    log_success "All operations completed!"
}

# 运行主函数
main "$@"
