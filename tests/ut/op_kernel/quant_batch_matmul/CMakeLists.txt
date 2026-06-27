cmake_minimum_required(VERSION 3.16)

set(OP_NAME "quant_batch_matmul_v3")

set(OP_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_quant_batch_matmul.cpp)

add_library(opkernel_${OP_NAME} OBJECT ${OP_TEST_SRC})

target_compile_options(opkernel_${OP_NAME} PRIVATE
    -g
    -DUT_SOC_VERSION="${SOC_VERSION}"
    -DKERNELUT=1
    -DBLAZE_UT=1
    -DASCENDC_CPU_DEBUG=1
    -DUT_KERNEL_SRC_DIR="${UT_KERNEL_INC}"
    -fPIC
    -std=c++17
)

target_include_directories(opkernel_${OP_NAME} PRIVATE
    ${OPS_TENSOR_ROOT}/include
    ${OPS_TENSOR_ROOT}/include/tensor_api/include
    ${OPS_TENSOR_ROOT}/include/tensor_api
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/impl
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/basic_api
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/interface
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/tiling
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/adv_api
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/c_api
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/simt_api
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/utils
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/impl/basic_api
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/impl/adv_api
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/impl/c_api
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/impl/simt_api
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/impl/utils
    ${ASCEND_HOME_PATH}/compiler/ascendc/include/highlevel_api
    ${ASCEND_HOME_PATH}/include/base
    ${ASCEND_HOME_PATH}/include/register
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/tools/lib/include
    ${ASCEND_HOME_PATH}/tools/lib/include/basic_api/utils
    ${UT_KERNEL_INC}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(opkernel_${OP_NAME} PRIVATE
    tikicpulib::${TIKICPULIB_TARGET}
)

# gtest 由 ExternalProject 构建，需确保头文件/库文件在 OBJECT library 编译前就位
if(KERNEL_UT_GTEST_DEP AND TARGET ${KERNEL_UT_GTEST_DEP})
    add_dependencies(opkernel_${OP_NAME} ${KERNEL_UT_GTEST_DEP})
endif()

list(APPEND OP_TEST_OBJECTS "$<TARGET_OBJECTS:opkernel_${OP_NAME}>")
set(OP_TEST_OBJECTS ${OP_TEST_OBJECTS} PARENT_SCOPE)

message(STATUS "  [quant_batch_matmul_v3] Kernel UT test object configured")
