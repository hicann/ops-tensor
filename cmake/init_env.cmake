##############################################################################
# 环境初始化模块
# 功能：检查环境、设置 NPU 架构、发现算子
##############################################################################

macro(init_env)
    # 1. 检查 ASCEND 环境
    _check_ascend()

    # 2. 设置 NPU 架构
    _setup_npu_arch()

    # 3. 发现算子
    _discover_operators()
endmacro()

##############################################################################
# 内部宏
##############################################################################

# 1. 检查 ASCEND 路径
macro(_check_ascend)
    if(NOT EXISTS "${ASCEND_HOME_PATH}")
        message(FATAL_ERROR
            "✗ ASCEND_HOME_PATH not found: ${ASCEND_HOME_PATH}\n"
            "  Fix: source /usr/local/Ascend/cann/set_env.sh"
        )
    endif()
    message(STATUS "✓ ASCEND: ${ASCEND_HOME_PATH}")
endmacro()

# 2. 设置 NPU 架构（SoC 映射）
macro(_setup_npu_arch)
    # 映射表（注意：dav-3101 已改名为 dav-3510）
    # 注意：当前版本仅支持 Ascend950，其他 SoC 型号暂不支持
    #       映射表保留是为了未来扩展和代码兼容性
    set(SOC_TO_NPU_ARCH_MAP
        "ascend950"     "dav-3510"
        "Ascend950"     "dav-3510"
        "ascend910b"    "dav-2201"
        "Ascend910B"    "dav-2201"
        "ascend910_93"  "dav-2201"
        "Ascend910_93"  "dav-2201"
        "ascend910"     "dav-2101"
        "Ascend910"     "dav-2101"
        "ascend310p"    "dav-2101"
        "Ascend310P"    "dav-2101"
    )

    # 确定架构（优先级：直接指定 > SOC_VERSION > ASCEND_SOC > 默认）
    if(DEFINED ASCEND_NPU_ARCH AND NOT "${ASCEND_NPU_ARCH}" STREQUAL "")
        message(STATUS "✓ NPU: ${ASCEND_NPU_ARCH} (user)")
    elseif(DEFINED SOC_VERSION AND NOT "${SOC_VERSION}" STREQUAL "")
        list(FIND SOC_TO_NPU_ARCH_MAP "${SOC_VERSION}" IDX)
        if(IDX EQUAL -1)
            message(WARNING "Unknown SOC_VERSION: ${SOC_VERSION}, using default")
            set(ASCEND_NPU_ARCH "dav-3510")
        else()
            math(EXPR IDX "${IDX} + 1")
            list(GET SOC_TO_NPU_ARCH_MAP ${IDX} ASCEND_NPU_ARCH)
        endif()
        message(STATUS "✓ NPU: ${ASCEND_NPU_ARCH} (${SOC_VERSION})")
    elseif(DEFINED ASCEND_SOC AND NOT "${ASCEND_SOC}" STREQUAL "")
        list(FIND SOC_TO_NPU_ARCH_MAP "${ASCEND_SOC}" IDX)
        if(IDX EQUAL -1)
            message(FATAL_ERROR "✗ Unknown SoC: ${ASCEND_SOC}")
        endif()
        math(EXPR IDX "${IDX} + 1")
        list(GET SOC_TO_NPU_ARCH_MAP ${IDX} ASCEND_NPU_ARCH)
        message(STATUS "✓ NPU: ${ASCEND_NPU_ARCH} (${ASCEND_SOC})")
    else()
        set(ASCEND_NPU_ARCH "dav-3510")
        message(STATUS "✓ NPU: ${ASCEND_NPU_ARCH} (default)")
    endif()
endmacro()

# 3. 发现算子（Header-only模式，算子在include/blaze/中）
macro(_discover_operators)
    # Header-only模式不需要扫描src目录
    # Blaze算子在 include/blaze/gemm/ 中
    set(BUILD_OPERATORS "matmul")
    message(STATUS "✓ Build: ${BUILD_OPERATORS}")
endmacro()
