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

##############################################################################
# 版本和依赖管理函数
##############################################################################

##############################################################################
# 宏: replace_cur_major_minor_ver
#
# 功能: 替换依赖版本变量中的 CUR_MAJOR_MINOR_VER 占位符
##############################################################################
macro(replace_cur_major_minor_ver)
    string(REPLACE CUR_MAJOR_MINOR_VER "${CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_VERSION_MAJOR_MINOR}" depend "${depend}")
endmacro()

##############################################################################
# 函数: set_package
#
# 功能: 设置包名和版本号
#
# 参数:
#   name        - 包名称 (必需)
#   VERSION     - 版本号 (必需)
##############################################################################
function(set_package name)
    cmake_parse_arguments(VERSION "" "VERSION" "" ${ARGN})
    set(VERSION "${VERSION_VERSION}")
    if(NOT name)
        message(FATAL_ERROR "The name parameter is not set in set_package.")
    endif()
    if(NOT VERSION)
        message(FATAL_ERROR "The VERSION parameter is not set in set_package(${name}).")
    endif()
    string(REGEX MATCH "^([0-9]+\\.[0-9]+)" VERSION_MAJOR_MINOR "${VERSION}")
    list(APPEND CANN_VERSION_PACKAGES "${name}")
    set(CANN_VERSION_PACKAGES "${CANN_VERSION_PACKAGES}" PARENT_SCOPE)
    set(CANN_VERSION_CURRENT_PACKAGE "${name}" PARENT_SCOPE)
    set(CANN_VERSION_${name}_VERSION "${VERSION}" PARENT_SCOPE)
    set(CANN_VERSION_${name}_VERSION_MAJOR_MINOR "${VERSION_MAJOR_MINOR}" PARENT_SCOPE)
    set(CANN_VERSION_${name}_BUILD_DEPS PARENT_SCOPE)
    set(CANN_VERSION_${name}_RUN_DEPS PARENT_SCOPE)
endfunction()

##############################################################################
# 函数: set_build_dependencies
#
# 功能: 设置构建依赖
#
# 参数:
#   pkg_name    - 依赖包名称 (必需)
#   depend      - 依赖版本信息 (必需)
##############################################################################
function(set_build_dependencies pkg_name depend)
    if(NOT CANN_VERSION_CURRENT_PACKAGE)
        message(FATAL_ERROR "The set_package must be invoked first.")
    endif()
    if(NOT pkg_name)
        message(FATAL_ERROR "The pkg_name parameter is not set in set_build_dependencies.")
    endif()
    if(NOT depend)
        message(FATAL_ERROR "The depend parameter is not set in set_build_dependencies.")
    endif()
    replace_cur_major_minor_ver()
    list(APPEND CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_BUILD_DEPS "${pkg_name}" "${depend}")
    set(CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_BUILD_DEPS "${CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_BUILD_DEPS}" PARENT_SCOPE)
endfunction()

##############################################################################
# 函数: set_run_dependencies
#
# 功能: 设置运行时依赖
#
# 参数:
#   pkg_name    - 依赖包名称 (必需)
#   depend      - 依赖版本信息 (必需)
##############################################################################
function(set_run_dependencies pkg_name depend)
    if(NOT CANN_VERSION_CURRENT_PACKAGE)
        message(FATAL_ERROR "The set_package must be invoked first.")
    endif()
    if(NOT pkg_name)
        message(FATAL_ERROR "The pkg_name parameter is not set in set_run_dependencies.")
    endif()
    if(NOT depend)
        message(FATAL_ERROR "The depend parameter is not set in set_run_dependencies.")
    endif()
    replace_cur_major_minor_ver()
    list(APPEND CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_RUN_DEPS "${pkg_name}" "${depend}")
    set(CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_RUN_DEPS "${CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_RUN_DEPS}" PARENT_SCOPE)
endfunction()

##############################################################################
# 函数: check_pkg_build_deps
#
# 功能: 检查构建依赖是否满足
#
# 参数:
#   pkg_name    - 包名称 (必需)
##############################################################################
function(check_pkg_build_deps pkg_name)
    execute_process(
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_build_dependencies.py "$ENV{ASCEND_HOME_PATH}" ${CANN_VERSION_${pkg_name}_BUILD_DEPS}
        RESULT_VARIABLE result
    )
    if(result)
        message(FATAL_ERROR "Check ${pkg_name} build dependencies failed!")
    endif()
endfunction()

##############################################################################
# 函数: add_version_info_targets
#
# 功能: 添加版本信息生成目标
#
# 说明:
#   - 为每个已注册的包生成 version.{pkg_name}.info 文件
#   - 目标名格式为：version_{pkg_name}_info
##############################################################################
function(add_version_info_targets)
    foreach(pkg_name ${CANN_VERSION_PACKAGES})
        add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/version.${pkg_name}.info
            COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_version_info.py --output ${CMAKE_BINARY_DIR}/version.${pkg_name}.info
                    "${CANN_VERSION_${pkg_name}_VERSION}" ${CANN_VERSION_${pkg_name}_RUN_DEPS}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/version.cmake ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_version_info.py
            VERBATIM
        )
        add_custom_target(version_${pkg_name}_info ALL DEPENDS ${CMAKE_BINARY_DIR}/version.${pkg_name}.info)
    endforeach()
endfunction()
