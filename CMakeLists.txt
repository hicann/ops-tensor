##############################################################################
# ops-tensor CMake 配置
##############################################################################

cmake_minimum_required(VERSION 3.16)

##############################################################################
# 一、基础配置
##############################################################################

# 项目声明
project(ops_tensor)

# 设置项目变量
set(OPS_TENSOR ops_tensor)
set(PROJECT_VERSION "1.0.0")

# 查找 ASC 编译器
find_package(ASC REQUIRED)
project(${OPS_TENSOR} VERSION ${PROJECT_VERSION} LANGUAGES ASC CXX)

##############################################################################
# 二、环境初始化（封装）
##############################################################################

include(cmake/init_env.cmake)
init_env()  # 发现算子

##############################################################################
# 三、核心构建流程
##############################################################################

# 3.1 编译测试（可选）
option(BUILD_TESTING "Build tests" OFF)
if(BUILD_TESTING)
    enable_testing()
    # 测试在 src/CMakeLists.txt 中配置
endif()

# 3.2 编译算子
add_subdirectory(src)

# 3.3 编译测试
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# 3.4 打包（可选）
option(ENABLE_PACKAGE "Enable package generation" OFF)
if(ENABLE_PACKAGE)
    include(cmake/package.cmake)
endif()

##############################################################################
# 四、构建摘要
##############################################################################

message(STATUS "")
message(STATUS "==========================================")
message(STATUS "  Build Summary")
message(STATUS "==========================================")
message(STATUS "  Version   : ${PROJECT_VERSION}")
message(STATUS "  NPU       : ${ASCEND_NPU_ARCH}")
message(STATUS "  Operators : ${BUILD_OPERATORS}")
message(STATUS "  Tests     : ${BUILD_TESTING}")
message(STATUS "==========================================")
