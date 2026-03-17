##############################################################################
# src/CMakeLists.txt
#
# 功能:
#   直接创建 ops_tensor 动态库
#   各算子通过 register_operator() 函数注册源文件
#   lib/ 目录包含框架层 C++ 代码
##############################################################################

# 引入算子注册函数
include(${CMAKE_SOURCE_DIR}/cmake/func.cmake)

##############################################################################
# 收集框架层源文件 (lib/ 目录)
##############################################################################

set(LIB_SOURCES
    ${CMAKE_SOURCE_DIR}/lib/core/handle.cpp
    ${CMAKE_SOURCE_DIR}/lib/core/tensor_descriptor.cpp
    ${CMAKE_SOURCE_DIR}/lib/core/operation_descriptor.cpp
    ${CMAKE_SOURCE_DIR}/lib/core/plan_preference.cpp
    ${CMAKE_SOURCE_DIR}/lib/core/plan.cpp
    ${CMAKE_SOURCE_DIR}/lib/elementwise/elementwise_binary.cpp
    ${CMAKE_SOURCE_DIR}/lib/elementwise/elementwise.cpp
    ${CMAKE_SOURCE_DIR}/lib/utils/utils.cpp
    ${CMAKE_SOURCE_DIR}/lib/utils/validation.cpp
)

##############################################################################
# 创建 ops_tensor 动态库
##############################################################################

# 创建一个占位源文件（用于满足 CMake 的最低要求）
file(WRITE ${CMAKE_BINARY_DIR}/ops_tensor_stub.cpp "// Stub file for ops_tensor library")

# 创建动态库目标（使用占位源文件和框架层源文件）
add_library(${OPS_TENSOR} SHARED
    ${CMAKE_BINARY_DIR}/ops_tensor_stub.cpp
    ${LIB_SOURCES}
)

# 设置库的版本属性
set_target_properties(${OPS_TENSOR} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "cann_ops_tensor"  # 生成文件名：libcann_ops_tensor.so
)

# 设置公共包含目录
target_include_directories(${OPS_TENSOR} PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/lib
    ${ASCEND_HOME_PATH}/include               # acl.h 等核心头文件
    ${ASCEND_HOME_PATH}/pkg_inc/op_common/    # ASCEND 公共操作（kernel_operator.h 等）
    ${ASCEND_HOME_PATH}/pkg_inc/base/         # ASCEND 基础类型
    ${ASCEND_HOME_PATH}/pkg_inc/              # ASCEND 头文件
)

# 设置框架层 C++ 源文件的编译选项
set_source_files_properties(
    ${LIB_SOURCES}
    PROPERTIES LANGUAGE CXX
)

target_compile_options(${OPS_TENSOR} PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${ASCEND_NPU_ARCH}>
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17 -fPIC>
)

# 链接 CANN tiling 平台库 (提供 GetCoreNumAiv 等函数)
target_link_libraries(${OPS_TENSOR} PRIVATE
    tiling_api
    ascendcl
)

##############################################################################
# 添加各算子子目录
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

# 获取所有源文件
get_target_property(LIB_TARGET_SOURCE ${OPS_TENSOR} SOURCES)

# 分离 ASC 源文件和 C++ 源文件
set(ASC_SOURCES "")
set(CPP_SOURCES "")

foreach(SRC ${LIB_TARGET_SOURCE})
    get_filename_component(SRC_EXT ${SRC} EXT)
    if(SRC_EXT STREQUAL ".cpp" AND NOT SRC MATCHES "ops_tensor_stub.cpp")
        list(APPEND CPP_SOURCES ${SRC})
    elseif(SRC_EXT STREQUAL ".cpp" AND SRC MATCHES "ops_tensor_stub.cpp")
        # Stub file, skip
    else()
        list(APPEND ASC_SOURCES ${SRC})
    endif()
endforeach()

# 设置 ASC 源文件的属性
if(ASC_SOURCES)
    set_source_files_properties(
        ${ASC_SOURCES}
        PROPERTIES LANGUAGE ASC
    )
    message(STATUS "ASC sources: ${ASC_SOURCES}")
endif()

# 设置 C++ 源文件的属性
if(CPP_SOURCES)
    set_source_files_properties(
        ${CPP_SOURCES}
        PROPERTIES LANGUAGE CXX
    )
    message(STATUS "C++ sources (framework): ${CPP_SOURCES}")
endif()

message(STATUS "==========================================")
message(STATUS "Operator registration completed")
message(STATUS "==========================================")
message(STATUS "")

##############################################################################
# 安装规则
##############################################################################

# 安装库文件到 ops_tensor/lib64/
install(TARGETS ${OPS_TENSOR}
    LIBRARY DESTINATION ${OPS_TENSOR_LIB_INSTALL_DIR}
    RUNTIME DESTINATION bin
)

# 安装公共头文件到 ops_tensor/include/
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${OPS_TENSOR_INC_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

message(STATUS "Operator library 'ops_tensor' configured successfully")
