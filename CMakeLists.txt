# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

##############################################################################
# ops-tensor CMake 配置
##############################################################################

cmake_minimum_required(VERSION 3.16)

##############################################################################
# 一、基础配置
##############################################################################

# 项目声明
project(ops_tensor)

# 包含变量和版本配置
include(cmake/func.cmake)
include(cmake/variables.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/version.cmake)

# 设置项目变量
set(OPS_TENSOR ops_tensor)
set(PROJECT_VERSION "1.0.0")

# 查找 ASC 编译器
find_package(ASC REQUIRED)
project(${OPS_TENSOR} VERSION ${PROJECT_VERSION} LANGUAGES ASC CXX)

##############################################################################
# 二、环境初始化（封装）
##############################################################################

include(cmake/init_env.cmake)
init_env()  # 发现算子

##############################################################################
# 三、核心构建流程
##############################################################################

# 3.1 编译测试（可选）
option(BUILD_TESTING "Build tests" OFF)

# 3.2 编译框架层（lib/ 目录）
add_subdirectory(lib)

# 3.3 创建 ops_tensor 动态库
add_library(${OPS_TENSOR} SHARED)

# 链接框架层 OBJECT 库和 CANN 依赖库
target_link_libraries(${OPS_TENSOR} PRIVATE ${OPS_TENSOR}_framework tiling_api ascendcl)

# 添加 CANN 库搜索路径（C++ 链接器需要显式指定）
target_link_directories(${OPS_TENSOR} PRIVATE ${ASCEND_HOME_PATH}/lib64)

# 设置库的版本属性
set_target_properties(${OPS_TENSOR} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "cann_ops_tensor"  # 生成文件名：libcann_ops_tensor.so
)

# 设置公共包含目录
target_include_directories(${OPS_TENSOR} PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/lib
    ${ASCEND_HOME_PATH}/include               # acl.h 等核心头文件
    ${ASCEND_HOME_PATH}/pkg_inc/op_common/    # ASCEND 公共操作
    ${ASCEND_HOME_PATH}/pkg_inc/base/         # ASCEND 基础类型
    ${ASCEND_HOME_PATH}/pkg_inc/              # ASCEND 头文件
)

target_compile_options(${OPS_TENSOR} PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${ASCEND_NPU_ARCH}>
)

# 3.4 编译算子层（src/ 目录）
# 注意：必须在创建 ${OPS_TENSOR} 之后，否则 register_operator() 无法添加源文件
add_subdirectory(src)

# 3.5 设置源文件语言属性
get_target_property(LIB_TARGET_SOURCE ${OPS_TENSOR} SOURCES)
set_source_files_properties(${LIB_TARGET_SOURCE} PROPERTIES LANGUAGE ASC)

# 3.6 编译测试
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# 3.7 打包配置
option(ENABLE_PACKAGE "Enable package generation" OFF)
if(ENABLE_PACKAGE)
    # 设置第三方库路径
    if(NOT CANN_3RD_LIB_PATH)
        set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/third_party CACHE STRING "CANN third party lib path")
    endif()
    # 检查构建依赖
    check_pkg_build_deps(ops_tensor)
    # 生成版本信息目标
    add_version_info_targets()
    # 打包
    include(cmake/package.cmake)
    pack_built_in()
endif()

##############################################################################
# 四、安装规则
##############################################################################

# 安装库文件到 ops_tensor/lib64/
install(TARGETS ${OPS_TENSOR}
    LIBRARY DESTINATION ${OPS_TENSOR_LIB_INSTALL_DIR}
    RUNTIME DESTINATION bin
)

# 安装公共头文件到 ops_tensor/include/
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${OPS_TENSOR_INC_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

##############################################################################
# 五、构建摘要
##############################################################################

message(STATUS "")
message(STATUS "==========================================")
message(STATUS "  Build Summary")
message(STATUS "==========================================")
message(STATUS "  Version   : ${PROJECT_VERSION}")
message(STATUS "  NPU       : ${ASCEND_NPU_ARCH}")
message(STATUS "  Operators : ${BUILD_OPERATORS}")
message(STATUS "  Tests     : ${BUILD_TESTING}")
message(STATUS "==========================================")
