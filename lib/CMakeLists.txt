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
# lib/CMakeLists.txt
#
# 功能:
#   框架层库 - 自动收集所有源文件
#   包含 core/、elementwise/、utils/ 等所有框架层代码
##############################################################################

# 收集所有框架层源文件（递归查找 lib/ 目录下所有 .cpp 文件）
file(GLOB_RECURSE LIB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# 移除测试文件（如果有）
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*test.*\\.cpp")

message(STATUS "==========================================")
message(STATUS "Framework layer sources:")
foreach(SRC ${LIB_SOURCES})
    message(STATUS "  ${SRC}")
endforeach()
message(STATUS "==========================================")

# 创建框架层库
add_library(${OPS_TENSOR}_framework OBJECT
    ${LIB_SOURCES}
)

# 设置源文件语言为 ASC（强制使用 ASC 编译器）
set_source_files_properties(${LIB_SOURCES} PROPERTIES LANGUAGE ASC)

# 设置公共包含目录
target_include_directories(${OPS_TENSOR}_framework PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lib
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/pkg_inc/op_common/
    ${ASCEND_HOME_PATH}/pkg_inc/base/
    ${ASCEND_HOME_PATH}/pkg_inc/
)

# 设置编译选项
target_compile_options(${OPS_TENSOR}_framework PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${ASCEND_NPU_ARCH}>
)

message(STATUS "Framework library '${OPS_TENSOR}_framework' configured successfully")
