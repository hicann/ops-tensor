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
# tests/CMakeLists.txt - Header-only Library Tests
##############################################################################

if(BUILD_TESTING)
    # 收集测试源文件（tests/ut/op_kernel/目录）
    file(GLOB_RECURSE TEST_SOURCES 
        ${CMAKE_CURRENT_SOURCE_DIR}/ut/*.cpp
    )

    # 创建测试可执行文件
    add_executable(all_ops_test
        ${TEST_SOURCES}
    )

    # 设置包含目录
    target_include_directories(all_ops_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_HOME_PATH}/include
    )

    # 链接库
    target_link_libraries(all_ops_test PRIVATE
        ${ASCEND_HOME_PATH}/lib64/libascendcl.so
        ${ASCEND_HOME_PATH}/lib64/libacl_rt.so
    )

    # 添加到CTest
    add_test(NAME all_ops_test COMMAND all_ops_test)
    
    message(STATUS "✓ Test executable: all_ops_test")
endif()
