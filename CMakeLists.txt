##############################################################################
# ops-tensor CMake 配置
##############################################################################

cmake_minimum_required(VERSION 3.16)

##############################################################################
# 一、基础配置
##############################################################################

# 项目声明
project(ops_tensor)

# 包含变量和版本配置
include(cmake/func.cmake)
include(cmake/variables.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/version.cmake)

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

# 3.2 编译算子
add_subdirectory(src)

# 3.3 编译测试
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# 3.4 打包配置
option(ENABLE_PACKAGE "Enable package generation" OFF)
if(ENABLE_PACKAGE)
    # 设置第三方库路径
    if(NOT CANN_3RD_LIB_PATH)
        set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/third_party CACHE STRING "CANN third party lib path")
    endif()
    # 检查构建依赖
    check_pkg_build_deps(ops_tensor)
    # 生成版本信息目标
    add_version_info_targets()
    # 打包
    include(cmake/package.cmake)
    pack_built_in()
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
