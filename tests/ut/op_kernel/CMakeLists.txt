cmake_minimum_required(VERSION 3.16)

# 设置三方库路径（fetch_cann_cmake 使用此变量）
if(NOT OPS_TENSOR_ROOT)
    set(OPS_TENSOR_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()
if(NOT CANN_3RD_LIB_PATH)
    set(CANN_3RD_LIB_PATH ${OPS_TENSOR_ROOT}/third_party CACHE STRING "CANN third party lib path")
endif()

# 获取 cann-cmake 模块集合（提供 add_cann_third_party 等宏）
include(${OPS_TENSOR_ROOT}/cmake/third_party/fetch_cann_cmake.cmake)

project(ops_tensor_kernel_ut LANGUAGES CXX)

set(SOC_VERSION "ascend950")
set(TIKICPULIB_TARGET "ascend950pr_9599")
set(UT_EXE_NAME "ops_tensor_kernel_ut")

if(NOT ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    else()
        message(FATAL_ERROR "ASCEND_HOME_PATH not set. Source CANN set_env.sh first.")
    endif()
endif()

set(SYSTEM_PREFIX "x86_64-linux")
set(UT_KERNEL_INC ${CMAKE_CURRENT_SOURCE_DIR})

# GTest: 通过 cann-cmake 统一管理三方库
add_cann_third_party(gtest)

list(APPEND CMAKE_PREFIX_PATH
    "${ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake"
    "${ASCEND_HOME_PATH}/compiler/ascendc_kernel_cmake"
)

find_package(tikicpulib REQUIRED)

set(OP_TEST_OBJECTS "")

include_directories(${GTEST_INCLUDE})
set(KERNEL_UT_GTEST_DEP third_party_gtest)

# Support OP_KERNEL_OPS variable for operator filtering
if(OP_KERNEL_OPS)
    # Specified operator list: build only the requested operators
    string(REPLACE "," ";" OP_LIST ${OP_KERNEL_OPS})
    foreach(OP_NAME ${OP_LIST})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${OP_NAME}/CMakeLists.txt)
            add_subdirectory(${OP_NAME})
            message(STATUS "  [opkernel] Added operator: ${OP_NAME}")
        else()
            message(WARNING "  [opkernel] Operator '${OP_NAME}' not found under ${CMAKE_CURRENT_SOURCE_DIR}, skipping")
        endif()
    endforeach()
else()
    # No operator list specified: auto-discover all operators with CMakeLists.txt
    file(GLOB OP_SUBDIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
    foreach(OP_DIR ${OP_SUBDIRS})
        if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${OP_DIR} AND
           EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${OP_DIR}/CMakeLists.txt)
            add_subdirectory(${OP_DIR})
            message(STATUS "  [opkernel] Added operator: ${OP_DIR}")
        endif()
    endforeach()
endif()

add_executable(${UT_EXE_NAME}_${SOC_VERSION}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op_kernel_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/k3_pvwrap.cpp
    ${OP_TEST_OBJECTS}
)

target_compile_options(${UT_EXE_NAME}_${SOC_VERSION} PRIVATE
    -D_GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(${UT_EXE_NAME}_${SOC_VERSION} PRIVATE
    GTest::gtest
    GTest::gtest_main
    tikicpulib::${TIKICPULIB_TARGET}
    pthread
)

target_include_directories(${UT_EXE_NAME}_${SOC_VERSION} PRIVATE
    ${OPS_TENSOR_ROOT}/include
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include
    ${ASCEND_HOME_PATH}/include
    ${UT_KERNEL_INC}
)

target_link_directories(${UT_EXE_NAME}_${SOC_VERSION} PRIVATE
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/lib64
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/simulator/dav_3510/lib
)

message(STATUS "ops_tensor Kernel UT configured: ${SOC_VERSION}")
message(STATUS "ops_tensor Kernel UT configuration completed")