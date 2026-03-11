# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# ops-tensor 安装路径定义

# 包名称
set(PKG_NAME "ops_tensor")

# 默认 KERNEL_ARCH（简化处理 ascend950）
set(KERNEL_ARCH "ascend950")

# 架构检测
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ARCH x86_64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
    set(ARCH aarch64)
else()
    message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    set(ARCH x86_64)
endif()

# Built-in 算子包安装路径 (简化版: lib64 和 include)
set(PATH_NAME "ops_tensor")

# 头文件安装到 ops_tensor/include/
set(OPS_TENSOR_INC_INSTALL_DIR           ops_tensor/include)
# 库文件安装到 ops_tensor/lib64/
set(OPS_TENSOR_LIB_INSTALL_DIR           ops_tensor/lib64)
# 版本信息
set(VERSION_INFO_INSTALL_DIR        ops_tensor)

# 输出路径
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build_out)

# 打包相关路径
set(SCRIPT_PACKAGE_DIR ${CMAKE_SOURCE_DIR}/scripts/package)
set(COMMON_SCRIPT_DIR ${SCRIPT_PACKAGE_DIR}/common)
set(OPS_TENSOR_SCRIPT_DIR ${SCRIPT_PACKAGE_DIR}/ops_tensor/scripts)

message(STATUS "Package: ${PKG_NAME}")
message(STATUS "Architecture: ${ARCH}")
message(STATUS "Kernel Arch: ${KERNEL_ARCH}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
