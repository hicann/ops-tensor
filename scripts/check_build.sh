#!/bin/bash

##############################################################################
# check_build.sh
#
# 功能:
#   1. 检查编译产物是否正确生成
#   2. 验证库文件和可执行文件
#   3. 检查符号导出
#   4. 验证安装结果
#
# 使用方法:
#   ./check_build.sh [OPTIONS]
#
# 选项:
#   -b, --build-dir DIR       构建目录 (默认: build)
#   -v, --verbose             详细输出
#   -h, --help                显示帮助
##############################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 默认参数
BUILD_DIR="build"
VERBOSE=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# 显示帮助
show_help() {
    grep '^#' "$0" | grep -v '!/bin/bash' | sed 's/^# //' | sed 's/^#//'
}

# 检查文件是否存在
check_file() {
    local file=$1
    local desc=$2

    if [ -e "$file" ]; then
        log_success "$desc 存在: $file"
        if [ "$VERBOSE" = true ]; then
            ls -lh "$file"
        fi
        return 0
    else
        log_error "$desc 不存在: $file"
        return 1
    fi
}

# 检查目录是否存在
check_dir() {
    local dir=$1
    local desc=$2

    if [ -d "$dir" ]; then
        log_success "$desc 存在: $dir"
        return 0
    else
        log_error "$desc 不存在: $dir"
        return 1
    fi
}

# 检查动态库
check_shared_library() {
    local lib=$1
    local libname=$(basename "$lib")

    log_info "检查动态库: $libname"

    if [ ! -f "$lib" ]; then
        log_error "动态库文件不存在"
        return 1
    fi

    # 检查文件权限
    if [ -x "$lib" ]; then
        log_success "动态库可执行权限正确"
    else
        log_warning "动态库缺少可执行权限"
    fi

    # 使用 nm/objdump 检查符号 (如果可用)
    if command -v nm &> /dev/null; then
        log_info "检查符号表..."
        local symbol_count=$(nm -D "$lib" 2>/dev/null | wc -l)
        if [ $symbol_count -gt 0 ]; then
            log_success "动态库包含 $symbol_count 个符号"
            if [ "$VERBOSE" = true ]; then
                nm -D "$lib" | head -20
                echo "..."
            fi
        else
            log_warning "动态库没有导出符号"
        fi
    fi

    # 检查依赖库 (Linux)
    if command -v ldd &> /dev/null; then
        log_info "检查依赖库..."
        ldd "$lib" 2>/dev/null | head -10
    fi

    return 0
}

# 检查可执行文件
check_executable() {
    local exe=$1
    local exename=$(basename "$exe")

    log_info "检查可执行文件: $exename"

    if [ ! -f "$exe" ]; then
        log_error "可执行文件不存在"
        return 1
    fi

    if [ -x "$exe" ]; then
        log_success "可执行文件权限正确"
    else
        log_error "可执行文件不可执行"
        return 1
    fi

    return 0
}

# 主检查函数
main() {
    cd "$SCRIPT_DIR"

    echo "=========================================="
    echo "    构建产物检查脚本"
    echo "=========================================="
    echo ""

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -b|--build-dir)
                BUILD_DIR="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done

    local total_checks=0
    local passed_checks=0
    local failed_checks=0

    # 1. 检查构建目录结构
    echo "---------- 构建目录结构检查 ----------"
    check_dir "$BUILD_DIR" "构建目录" && ((passed_checks++)) || ((failed_checks++))
    ((total_checks++))

    check_dir "$BUILD_DIR/lib" "库目录" && ((passed_checks++)) || ((failed_checks++))
    ((total_checks++))

    check_dir "$BUILD_DIR/bin" "可执行文件目录" && ((passed_checks++)) || ((failed_checks++))
    ((total_checks++))

    echo ""

    # 2. 检查动态库
    echo "---------- 动态库检查 ----------"
    local lib_file="$BUILD_DIR/lib/libops_tensor.so"
    if [ -f "$lib_file" ]; then
        check_shared_library "$lib_file" && ((passed_checks++)) || ((failed_checks++))
    else
        # 检查 Windows 版本
        lib_file="$BUILD_DIR/lib/ops_tensor.dll"
        if [ -f "$lib_file" ]; then
            check_shared_library "$lib_file" && ((passed_checks++)) || ((failed_checks++))
        else
            log_error "找不到动态库文件"
            ((failed_checks++))
        fi
    fi
    ((total_checks++))

    echo ""

    # 3. 检查可执行文件
    echo "---------- 可执行文件检查 ----------"
    for test_exe in "$BUILD_DIR/bin"/*_test; do
        if [ -f "$test_exe" ]; then
            check_executable "$test_exe" && ((passed_checks++)) || ((failed_checks++))
            ((total_checks++))
        fi
    done

    echo ""

    # 4. 检查静态库
    echo "---------- 静态库检查 ----------"
    for static_lib in "$BUILD_DIR/lib"/*.a; do
        if [ -f "$static_lib" ]; then
            check_file "$static_lib" "静态库 $(basename $static_lib)" && ((passed_checks++)) || ((failed_checks++))
            ((total_checks++))
        fi
    done

    echo ""

    # 5. 检查头文件
    echo "---------- 头文件检查 ----------"
    if [ -d "$BUILD_DIR/include" ]; then
        local header_count=$(find "$BUILD_DIR/include" -name "*.h" -o -name "*.hpp" 2>/dev/null | wc -l)
        log_success "找到 $header_count 个头文件"
        ((passed_checks++))
    else
        log_warning "生成的头文件目录不存在 (可选)"
    fi
    ((total_checks++))

    echo ""

    # 总结
    echo "=========================================="
    echo "       检查结果总结"
    echo "=========================================="
    log_info "总检查项: $total_checks"
    log_success "通过: $passed_checks"
    if [ $failed_checks -gt 0 ]; then
        log_error "失败: $failed_checks"
    fi
    echo "=========================================="

    if [ $failed_checks -gt 0 ]; then
        exit 1
    fi
}

main "$@"
