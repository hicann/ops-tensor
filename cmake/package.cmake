# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# ops-tensor 打包函数 (Built-in 包)

# 下载 makeself 工具
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/makeself-fetch.cmake)

function(pack_built_in)
    message(STATUS "============================================")
    message(STATUS "Packing built-in ops_tensor package...")
    message(STATUS "  SOC: ${SOC_VERSION}")
    message(STATUS "  Architecture: ${ARCH}")
    message(STATUS "============================================")

    # 1. 安装脚本文件 (与 ops-math 保持一致)
    set(script_prefix ${CMAKE_SOURCE_DIR}/scripts/package/ops_tensor/scripts)
    install(DIRECTORY ${script_prefix}/
        DESTINATION share/info/ops_tensor/script
        FILE_PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
        DIRECTORY_PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
        REGEX "(setenv|prereq_check)\.(bash|fish|csh)" EXCLUDE
    )

    # 2. 通用脚本文件 (与 ops-math 保持一致)
    set(SCRIPTS_FILES
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/check_version_required.awk
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func.inc
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.sh
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.csh
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.fish
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_compatiable.inc
    )
    install(FILES ${SCRIPTS_FILES}
        DESTINATION share/info/ops_tensor/script
        OPTIONAL
    )

    # 3. 打包相关脚本
    set(COMMON_FILES
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/install_common_parser.sh
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func_v2.inc
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_installer.inc
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/script_operator.inc
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_cfg.inc
    )
    set(PACKAGE_FILES
        ${COMMON_FILES}
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/multi_version.inc
    )
    set(LATEST_MANGER_FILES
        ${COMMON_FILES}
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func.inc
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_compatiable.inc
        ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/check_version_required.awk
    )
    set(CONF_FILES
        ${CMAKE_SOURCE_DIR}/scripts/package/common/cfg/path.cfg
    )

    # 4. 安装版本信息
    install(FILES ${CMAKE_BINARY_DIR}/version.ops_tensor.info
        DESTINATION share/info/ops_tensor
        RENAME version.info
    )

    # 5. 安装配置文件和脚本
    install(FILES ${CONF_FILES}
        DESTINATION ops_tensor/conf
        OPTIONAL
    )
    install(FILES ${PACKAGE_FILES}
        DESTINATION share/info/ops_tensor/script
        OPTIONAL
    )
    install(FILES ${LATEST_MANGER_FILES}
        DESTINATION latest_manager
        OPTIONAL
    )

    # 6. 安装 latest_manager 脚本
    if(EXISTS ${CMAKE_SOURCE_DIR}/scripts/package/latest_manager/scripts/)
        install(DIRECTORY ${CMAKE_SOURCE_DIR}/scripts/package/latest_manager/scripts/
            DESTINATION latest_manager
            OPTIONAL
        )
    endif()

    # 7. CPack 配置
    set(CPACK_PACKAGE_NAME "cann-${SOC_VERSION}-ops-tensor")
    set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}_linux-${ARCH}")
    set(CPACK_INSTALL_PREFIX "/")

    set(CPACK_CMAKE_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
    set(CPACK_CMAKE_BINARY_DIR "${CMAKE_BINARY_DIR}")
    set(CPACK_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
    set(CPACK_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    set(CPACK_MAKESELF_PATH "${MAKESELF_PATH}")
    set(CPACK_SOC "${SOC_VERSION}")
    set(CPACK_ARCH "${ARCH}")
    set(CPACK_SET_DESTDIR ON)

    set(CPACK_GENERATOR External)
    set(CPACK_EXTERNAL_PACKAGE_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/makeself_built_in.cmake")
    set(CPACK_EXTERNAL_ENABLE_STAGING true)
    set(CPACK_PACKAGE_DIRECTORY "${CMAKE_INSTALL_PREFIX}")

    message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
    include(CPack)
endfunction()