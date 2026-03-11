##############################################################################
# tests/CMakeLists.txt
#
# 自动扫描所有算子测试并生成 all_tests.cpp
##############################################################################

# 只在 BUILD_TESTING 开启时执行
if(BUILD_TESTING)

    # 只扫描已启用算子的测试头文件（相对于 src 目录）
    set(TEST_HEADERS "")
    foreach(OP ${BUILD_OPERATORS})
        set(test_header "${OP}/tests/${OP}_test.h")
        # 检查文件是否存在
        if(EXISTS "${CMAKE_SOURCE_DIR}/src/${test_header}")
            list(APPEND TEST_HEADERS ${test_header})
        endif()
    endforeach()

    message(STATUS "找到测试头文件: ${TEST_HEADERS}")

    # 生成 include 语句列表（不带 src/ 前缀）
    set(TEST_INCLUDES "")
    foreach(test_header ${TEST_HEADERS})
        set(TEST_INCLUDES "${TEST_INCLUDES}#include \"${test_header}\"\n")
    endforeach()

    # 配置生成 all_tests.cpp
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/all_tests.cpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/all_tests.cpp
        @ONLY
    )

    # 收集测试源文件
    set(TEST_SOURCES "")
    foreach(OP ${BUILD_OPERATORS})
        set(test_cpp "${CMAKE_SOURCE_DIR}/src/${OP}/tests/${OP}_test.cpp")
        if(EXISTS ${test_cpp})
            list(APPEND TEST_SOURCES ${test_cpp})
        endif()
    endforeach()

    # 创建测试可执行文件
    add_executable(all_ops_test
        ${CMAKE_CURRENT_BINARY_DIR}/all_tests.cpp
        ${CMAKE_SOURCE_DIR}/src/common/tests/test_common.cpp
        ${TEST_SOURCES}
    )

    # 设置包含目录
    target_include_directories(all_ops_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/common/tests
        ${ASCEND_HOME_PATH}/include
    )

    # 链接库
    target_link_libraries(all_ops_test PRIVATE
        ${OPS_TENSOR}
        ${ASCEND_HOME_PATH}/lib64/libascendcl.so
        ${ASCEND_HOME_PATH}/lib64/libacl_rt.so
    )

    # 添加到 CTest
    add_test(NAME all_ops_test COMMAND all_ops_test)

    message(STATUS "已配置测试目标: all_ops_test")

endif()
