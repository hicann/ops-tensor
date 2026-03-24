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
        "Ascend950"     "dav-3510"
        "Ascend910B"    "dav-2201"
        "Ascend910_93"  "dav-2201"
        "Ascend910"     "dav-2101"
        "Ascend310P"    "dav-2101"
    )

    # 确定架构（优先级：直接指定 > SoC映射 > 默认）
    if(DEFINED ASCEND_NPU_ARCH AND NOT "${ASCEND_NPU_ARCH}" STREQUAL "")
        message(STATUS "✓ NPU: ${ASCEND_NPU_ARCH} (user)")
    elseif(DEFINED ASCEND_SOC)
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

# 3. 发现算子
macro(_discover_operators)
    # 扫描 src 目录
    file(GLOB OPS RELATIVE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/*)
    foreach(OP ${OPS})
        if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/src/${OP})
            list(APPEND OPERATOR_DIRS ${OP})
        endif()
    endforeach()
    list(SORT OPERATOR_DIRS)

    # 解析用户指定（不强制清空，检查是否已定义）
    if(NOT DEFINED ENABLED_OPERATORS OR "${ENABLED_OPERATORS}" STREQUAL "")
        set(BUILD_OPERATORS ${OPERATOR_DIRS})
        message(STATUS "✓ Build: all (${OPERATOR_DIRS})")
    else()
        string(REPLACE "," ";" OP_LIST "${ENABLED_OPERATORS}")
        foreach(OP ${OP_LIST})
            list(FIND OPERATOR_DIRS ${OP} IDX)
            if(IDX EQUAL -1)
                message(FATAL_ERROR "✗ Unknown operator: ${OP}")
            endif()
            list(APPEND BUILD_OPERATORS ${OP})
        endforeach()
        message(STATUS "✓ Build: ${BUILD_OPERATORS}")
    endif()
endmacro()
