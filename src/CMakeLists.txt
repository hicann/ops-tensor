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
# src/CMakeLists.txt
#
# 功能:
#   算子层 - 注册各算子到 ${OPS_TENSOR} 库
##############################################################################

# 引入算子注册函数
include(${CMAKE_SOURCE_DIR}/cmake/func.cmake)

##############################################################################
# 注册各算子子目录
##############################################################################

message(STATUS "==========================================")
message(STATUS "Registering operators...")
message(STATUS "==========================================")

foreach(OP ${BUILD_OPERATORS})
    if(EXISTS ${CMAKE_SOURCE_DIR}/src/${OP}/CMakeLists.txt)
        add_subdirectory(${OP})
        message(STATUS "Added operator directory: ${OP}")
    else()
        message(WARNING "Operator ${OP} missing CMakeLists.txt, skipping")
    endif()
endforeach()

message(STATUS "==========================================")
message(STATUS "Operator registration completed")
message(STATUS "==========================================")
message(STATUS "")

message(STATUS "Operator layer configured successfully")
