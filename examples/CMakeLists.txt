# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the software repository for the full text of the License.
# ----------------------------------------------------------------------------

##############################################################################
# ops-tensor Examples — Shared CMake Configuration (L1)
#
# This is a standalone CMake project that builds all example executables.
# It uses the Blaze header-only GEMM framework and links only ACL runtime libs.
##############################################################################

cmake_minimum_required(VERSION 3.16)
project(ops_tensor_examples LANGUAGES CXX)

##############################################################################
# 1. ASC Compiler Discovery
##############################################################################

find_package(ASC REQUIRED)
enable_language(ASC)

##############################################################################
# 2. NPU Architecture Configuration (default: dav-3510 for Ascend950)
##############################################################################

if(NOT DEFINED ASCEND_NPU_ARCH)
    set(ASCEND_NPU_ARCH "dav-3510")
endif()
add_compile_options($<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${ASCEND_NPU_ARCH}>)

##############################################################################
# 3. Path Configuration
##############################################################################

set(OPS_TENSOR_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(BLAZE_INCLUDE_DIR ${OPS_TENSOR_ROOT}/include)
set(EXAMPLES_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

##############################################################################
# 4. Public Include Directories
##############################################################################

include_directories(
    ${BLAZE_INCLUDE_DIR}           # Blaze header-only framework
    ${BLAZE_INCLUDE_DIR}/blaze     # blaze/gemm/kernel/ etc.
    ${BLAZE_INCLUDE_DIR}/tensor_api/include  # tensor_api/tensor.h etc.
    ${BLAZE_INCLUDE_DIR}/tensor_api  # impl/tensor_api/algorithm/ etc.
    ${EXAMPLES_COMMON_DIR}         # data_utils.h etc.
    ${ASCEND_HOME_PATH}/include    # ACL core headers
    ${ASCEND_HOME_PATH}/include/op_common/
    ${ASCEND_HOME_PATH}/pkg_inc/op_common/
    ${ASCEND_HOME_PATH}/pkg_inc/base/
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/x86_64-linux/asc/include/tiling  # platform_ascendc.h
    ${ASCEND_HOME_PATH}/asc/impl/c_api  # instr_impl/ header resolution
    ${ASCEND_HOME_PATH}/asc/impl/utils  # impl/utils header resolution
)

##############################################################################
# 5. Public Link Directories and Libraries
##############################################################################

link_directories(${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/x86_64-linux/lib64)
link_libraries(ascendcl ascendc_runtime runtime mmpa c_sec dl)

##############################################################################
# 6. Core Macro: Register an Example Executable
#
# Usage:
#   ops_example_add_executable(target_name source1.cpp [source2.cpp ...])
#
# This macro:
#   - Marks source files as ASC language (compiled by bisheng)
#   - Creates an executable target
#   - Links platform + tiling_api for GetCoreNumAic()
#   - Installs to bin/ directory
##############################################################################

macro(ops_example_add_executable NAME)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE ASC)
    add_executable(${NAME} ${ARGN})
    target_link_libraries(${NAME} PRIVATE platform tiling_api)
    install(TARGETS ${NAME} DESTINATION bin)
endmacro()

##############################################################################
# 7. Load Operator Subdirectories
#
# Add new operators here: add_subdirectory(new_op)
##############################################################################

add_subdirectory(mat_mul)
add_subdirectory(transpose_batch_mat_mul)
add_subdirectory(quant_batch_matmul_mx)
