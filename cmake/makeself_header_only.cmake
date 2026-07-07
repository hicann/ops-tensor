# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# makeself_header_only.cmake - Header-only library 打包脚本

set(MAKESELF_EXE ${CPACK_MAKESELF_PATH}/makeself.sh)
set(MAKESELF_HEADER_EXE ${CPACK_MAKESELF_PATH}/makeself-header.sh)

if(NOT EXISTS ${MAKESELF_EXE})
    message(FATAL_ERROR "makeself.sh not found: ${MAKESELF_EXE}")
endif()

# 创建临时安装目录
set(STAGING_DIR "${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/header_only_staging")
file(REMOVE_RECURSE "${STAGING_DIR}")
file(MAKE_DIRECTORY "${STAGING_DIR}")

# 安装到临时目录
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${CPACK_CMAKE_BINARY_DIR}" --prefix "${STAGING_DIR}"
    RESULT_VARIABLE INSTALL_RESULT
)

if(NOT INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR "Installation failed: ${INSTALL_RESULT}")
endif()

# 创建安装脚本（兼容不同目录结构）
set(INSTALL_SH "${STAGING_DIR}/install.sh")
file(WRITE ${INSTALL_SH} "#!/bin/bash
set -e

if [ -n \"\${ASCEND_INSTALL_PATH}\" ]; then
    INSTALL_BASE=\"\${ASCEND_INSTALL_PATH}\"
else
    INSTALL_BASE=\"/usr/local/Ascend/cann\"
fi

TARGET_DIR=\"\${INSTALL_BASE}/ops_tensor\"echo \"Installing ops-tensor header-only library to \${TARGET_DIR}\"mkdir -p \"\${TARGET_DIR}\"if [ -d \"ops_tensor/include\" ]; then
    cp -r ops_tensor/include \"\${TARGET_DIR}/\"elif [ -d \"include\" ]; then
    mkdir -p \"\${TARGET_DIR}/include\"    cp -r include/* \"\${TARGET_DIR}/include/\"fi

echo \"Installation completed.\"
exit 0
")

# makeself 打包
set(PACKAGE_NAME "${CPACK_PACKAGE_FILE_NAME}.run")

message(STATUS "Creating package: ${PACKAGE_NAME}")

execute_process(
    COMMAND bash ${MAKESELF_EXE}
        --header ${MAKESELF_HEADER_EXE}
        --follow
        --nocompress
        ${STAGING_DIR}
        ${PACKAGE_NAME}
        "ops-tensor header-only library"
        install.sh
    WORKING_DIRECTORY ${CPACK_CMAKE_BINARY_DIR}
    RESULT_VARIABLE MAKESELF_RESULT
    ERROR_VARIABLE MAKESELF_ERROR
    OUTPUT_VARIABLE MAKESELF_OUTPUT
)

if(NOT MAKESELF_RESULT EQUAL 0)
    message(FATAL_ERROR "makeself failed: ${MAKESELF_ERROR}")
endif()

# 移动到输出目录
execute_process(COMMAND mkdir -p ${CPACK_PACKAGE_DIRECTORY})
execute_process(COMMAND mv ${CPACK_CMAKE_BINARY_DIR}/${PACKAGE_NAME} ${CPACK_PACKAGE_DIRECTORY}/)

if(EXISTS ${CPACK_PACKAGE_DIRECTORY}/${PACKAGE_NAME})
    message(STATUS "Package created: ${CPACK_PACKAGE_DIRECTORY}/${PACKAGE_NAME}")
else()
    message(FATAL_ERROR "Package not found: ${CPACK_PACKAGE_DIRECTORY}/${PACKAGE_NAME}")
endif()