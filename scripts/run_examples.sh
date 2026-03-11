#!/bin/bash

##############################################################################
# run_examples.sh
#
# 功能:
#   1. 指定执行某个/某些或全部算子的示例代码
#   2. 统计执行结果
#   3. 生成性能报告
#
# 使用方法:
#   ./run_examples.sh [OPTIONS]
#
# 选项:
#   -o, --operators OP1,OP2    指定要运行的算子 (默认: all)
#   -r, --repeat N             重复执行次数 (默认: 1)
#   -p, --performance          启用性能统计
#   -v, --verbose              详细输出
#   -h, --help                 显示帮助
##############################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 默认参数
OPERATORS="all"
REPEAT=1
PERFORMANCE=false
VERBOSE=false
BUILD_DIR="build"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助
show_help() {
    grep '^#' "$0" | grep -v '!/bin/bash' | sed 's/^# //' | sed 's/^#//'
}

# 获取所有可用的算子
get_available_operators() {
    find "$SCRIPT_DIR/tests" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" 2>/dev/null | sort
}

# 验证算子
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
        log_error "无效的算子: ${invalid_ops[*]}"
        log_info "可用的算子: ${available_ops[*]}"
        exit 1
    fi
}

# 运行单个算子示例
run_operator_example() {
    local op=$1
    local repeat=$2

    local test_exe="$BUILD_DIR/bin/${op}_test"

    if [ ! -f "$test_exe" ]; then
        log_warning "算子 $op 的测试文件不存在: $test_exe"
        return 1
    fi

    log_info "运行 $op 算子示例..."

    local total_time=0
    local success_count=0
    local fail_count=0

    for ((i=1; i<=repeat; i++)); do
        if [ "$VERBOSE" = true ]; then
            log_info "  第 $i 次执行..."
        fi

        local start_time=$(date +%s%N)

        if "$test_exe"; then
            success_count=$((success_count + 1))
            if [ "$PERFORMANCE" = true ]; then
                local end_time=$(date +%s%N)
                local elapsed=$((end_time - start_time))
                total_time=$((total_time + elapsed))
                log_info "  耗时: $((elapsed / 1000000))ms"
            fi
        else
            fail_count=$((fail_count + 1))
            log_error "  第 $i 次执行失败"
        fi
    done

    echo ""
    log_info "$op 算子执行结果:"
    log_info "  成功: $success_count"
    if [ $fail_count -gt 0 ]; then
        log_error "  失败: $fail_count"
    fi

    if [ "$PERFORMANCE" = true ] && [ $success_count -gt 0 ]; then
        local avg_time=$((total_time / success_count / 1000000))
        log_info "  平均耗时: ${avg_time}ms"
    fi
    echo ""

    if [ $fail_count -gt 0 ]; then
        return 1
    fi
    return 0
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -o|--operators)
                OPERATORS="$2"
                shift 2
                ;;
            -r|--repeat)
                REPEAT="$2"
                shift 2
                ;;
            -p|--performance)
                PERFORMANCE=true
                shift
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
}

# 主函数
main() {
    cd "$SCRIPT_DIR"

    echo "=========================================="
    echo "    ops-tensor 示例运行脚本"
    echo "=========================================="
    echo ""

    parse_arguments "$@"

    # 检查构建目录
    if [ ! -d "$BUILD_DIR" ]; then
        log_error "构建目录不存在，请先运行 build.sh 进行编译"
        exit 1
    fi

    # 验证算子
    if [ "$OPERATORS" != "all" ]; then
        IFS=',' read -ra OPS <<< "$OPERATORS"
        validate_operators "${OPS[*]}"
    fi

    # 确定要运行的算子
    local run_ops=()
    if [ "$OPERATORS" = "all" ]; then
        run_ops=($(get_available_operators))
    else
        IFS=',' read -ra OPS <<< "$OPERATORS"
        run_ops=("${OPS[@]}")
    fi

    # 统计变量
    local total_ops=${#run_ops[@]}
    local total_success=0
    local total_fail=0
    local failed_ops=()

    # 运行每个算子
    for op in "${run_ops[@]}"; do
        if run_operator_example "$op" "$REPEAT"; then
            total_success=$((total_success + 1))
        else
            total_fail=$((total_fail + 1))
            failed_ops+=("$op")
        fi
    done

    # 输出总结
    echo ""
    echo "=========================================="
    echo "       总体执行结果"
    echo "=========================================="
    log_info "总算子数: $total_ops"
    log_success "成功: $total_success"
    if [ $total_fail -gt 0 ]; then
        log_error "失败: $total_fail"
        log_error "失败的算子: ${failed_ops[*]}"
    fi
    echo "=========================================="

    if [ $total_fail -gt 0 ]; then
        exit 1
    fi
}

main "$@"
