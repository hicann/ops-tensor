# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it
# under the terms and conditions of the CANN Open Software License Agreement
# Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in
# compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 查找 opbase 依赖（提供 atvoss 模板支持）
if(EXISTS "${PROJECT_SOURCE_DIR}/../opbase")
  # 获取绝对路径，确保跨平台兼容
  get_filename_component(OPBASE_SOURCE_PATH
                         ${CMAKE_CURRENT_SOURCE_DIR}/../opbase REALPATH)
  message(STATUS "Find opbase source dir: ${OPBASE_SOURCE_PATH}")
  message(STATUS "opbase path exists: ${EXISTS}")
elseif(EXISTS "${CANN_3RD_LIB_PATH}/opbase")
  # 转换为绝对路径
  get_filename_component(OPBASE_SOURCE_PATH
                         ${CANN_3RD_LIB_PATH}/opbase ABSOLUTE)
  message(STATUS "Find opbase source dir: ${OPBASE_SOURCE_PATH}")
else()
  if(EXISTS "${PROJECT_SOURCE_DIR}/build/_deps/opbase-subbuild")
    file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/build/_deps/opbase-subbuild)
  endif()
  include(FetchContent)

  FetchContent_Declare(
    opbase
    GIT_REPOSITORY https://gitcode.com/cann/opbase.git
    GIT_TAG 929902e577bb809a377a6ab17cb7b00a4dfddc1a
    GIT_PROGRESS TRUE
    SOURCE_DIR ${CANN_3RD_LIB_PATH}/opbase)

  FetchContent_Populate(opbase)

  set(OPBASE_SOURCE_PATH ${CANN_3RD_LIB_PATH}/opbase)

  if(EXISTS ${OPBASE_SOURCE_PATH}/include)
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/include)
  endif()
  if(EXISTS ${OPBASE_SOURCE_PATH}/aicpu_common)
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/aicpu_common)
  endif()
endif()
