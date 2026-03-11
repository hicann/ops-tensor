##############################################################################
# func.cmake
#
# 公共函数集合
#
# 提供统一的算子注册和测试注册接口
##############################################################################

##############################################################################
# 函数: register_operator
#
# 功能: 注册算子源文件到 ops_tensor 动态库
#
# 参数:
#   NAME        - 算子名称 (必需)
#   ARCH_DIR    - 架构特定目录名 (可选，默认: arch35)
#
# 使用示例:
#   register_operator(NAME add ARCH_DIR arch35)
#
# 说明:
#   - 自动收集当前目录下所有 .cpp 文件
#   - 自动收集 ARCH_DIR 目录下所有 .cpp 文件并设置为 ASC 语言
##############################################################################
function(register_operator)
    # 解析参数
    cmake_parse_arguments(ARG
        ""                              # 选项
        "NAME;ARCH_DIR"                 # 单值参数
        ""                              # 多值参数
        ${ARGN}
    )

    # 必需参数检查
    if(NOT ARG_NAME)
        message(FATAL_ERROR "register_operator: NAME parameter is required")
    endif()

    # 设置默认架构目录
    if(NOT ARG_ARCH_DIR)
        set(ARG_ARCH_DIR "arch35")
    endif()

    # 算子名称转大写
    string(TOUPPER ${ARG_NAME} OP_UPPER)

    # 算子源文件路径
    set(OP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

    # 收集当前目录下的所有 .cpp 文件（非递归）
    file(GLOB CPP_SOURCES "${OP_SOURCE_DIR}/*.cpp")

    # 收集架构目录下的所有 .cpp 文件（非递归）
    file(GLOB ARCH_SOURCES "${OP_SOURCE_DIR}/${ARG_ARCH_DIR}/*.cpp")

    # 合并所有源文件
    set(ALL_SOURCES ${CPP_SOURCES} ${ARCH_SOURCES})

    message(STATUS "  Registered operator: ${ARG_NAME}")
    if(CPP_SOURCES)
        message(STATUS "    C++ Sources: ${CPP_SOURCES}")
    endif()
    if(ARCH_SOURCES)
        message(STATUS "    ASC Sources: ${ARCH_SOURCES}")
    endif()

    # 一次性将所有源文件添加到 ops_tensor 库
    if(ALL_SOURCES)
        target_sources(${OPS_TENSOR} PRIVATE ${ALL_SOURCES})
    endif()

    # 设置架构目录下源文件的语言属性为 ASC
    if(ARCH_SOURCES)
        message(STATUS "    Set ASC language for: ${ARCH_SOURCES}")
    endif()

    # 添加算子特定的包含目录
    target_include_directories(${OPS_TENSOR} PRIVATE
        ${OP_SOURCE_DIR}
        ${OP_SOURCE_DIR}/${ARG_ARCH_DIR}
    )

    # 添加算子特定的编译定义
    target_compile_definitions(${OPS_TENSOR} PRIVATE
        ENABLE_OPERATOR_${OP_UPPER}=1
    )
endfunction()

##############################################################################
# 函数: register_operator_test
#
# 功能: 注册算子测试到 CTest
#
# 参数:
#   OP_NAME   - 算子名称 (必需)
##############################################################################
function(register_operator_test OP_NAME)
    if(NOT BUILD_TESTING)
        return()
    endif()

    # 测试可执行文件
    add_executable(${OP_NAME}_test tests/${OP_NAME}_test.cpp)

    # 链接库
    target_link_libraries(${OP_NAME}_test
        PRIVATE
        ${OPS_TENSOR}
        ${ASCEND_HOME_PATH}/lib64/libascendcl.so
        ${ASCEND_HOME_PATH}/lib64/libacl_rt.so
    )

    # 包含目录
    target_include_directories(${OP_NAME}_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_HOME_PATH}/include
    )

    # 编译定义
    string(TOUPPER ${OP_NAME} OP_UPPER)
    target_compile_definitions(${OP_NAME}_test PRIVATE ENABLE_OPERATOR_${OP_UPPER}=1)

    # 添加到CTest
    add_test(NAME ${OP_NAME}_test COMMAND ${OP_NAME}_test)
    set_tests_properties(${OP_NAME}_test PROPERTIES
        LABELS "operator;${OP_NAME}"
        TIMEOUT 300
    )

    message(STATUS "  Configured test for: ${OP_NAME}")
endfunction()
