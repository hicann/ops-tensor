##############################################################################
# package.cmake
#
# CPack 配置 - 为 ops-tensor 生成 run 包
##############################################################################

# 检测架构
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ARCH x86_64)
    message(STATUS "Detected architecture: x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
    set(ARCH aarch64)
    message(STATUS "Detected architecture: aarch64")
else()
    message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    set(ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

# 设置芯片类型（从命令行获取，默认 Ascend910B）
if(NOT DEFINED ASCEND_COMPUTE_UNIT)
    set(ASCEND_COMPUTE_UNIT "Ascend910B")
    message(STATUS "ASCEND_COMPUTE_UNIT not set, using default: Ascend910B")
endif()

set(CPACK_SOC ${ASCEND_COMPUTE_UNIT})
message(STATUS "Package target SoC: ${CPACK_SOC}")

##############################################################################
# 安装规则
##############################################################################

# 安装库文件
install(TARGETS ops_tensor
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# 安装头文件
install(FILES ${CMAKE_SOURCE_DIR}/include/cann_ops_tensor.h
    DESTINATION include
)

# 安装版本信息
install(FILES ${CMAKE_SOURCE_DIR}/version.info
    DESTINATION share/info/ops_tensor
)

# 安装脚本文件
set(SCRIPT_DIR ${CMAKE_SOURCE_DIR}/scripts/package/ops_tensor/scripts)

install(DIRECTORY ${SCRIPT_DIR}/
    DESTINATION share/info/ops_tensor/scripts
    FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
    DIRECTORY_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)

##############################################################################
# CPack 配置
##############################################################################

# 从 CMAKE 传递的 ASCEND_SOC 获取 SoC 型号，如果没有指定则使用默认值
if(NOT DEFINED ASCEND_SOC)
    set(ASCEND_SOC "Ascend950")
    message(STATUS "ASCEND_SOC not set, using default: Ascend950")
endif()

set(CPACK_SOC_NAME ${ASCEND_SOC})
message(STATUS "Package target SoC: ${CPACK_SOC_NAME}")

# 使用 makeself 生成 .run 包
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_INSTALL_PREFIX "/")

set(CPACK_CMAKE_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
set(CPACK_CMAKE_BINARY_DIR "${CMAKE_BINARY_DIR}")
set(CPACK_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CPACK_SET_DESTDIR ON)
set(CPACK_GENERATOR External)
set(CPACK_EXTERNAL_PACKAGE_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/run_package.cmake")
set(CPACK_EXTERNAL_ENABLE_STAGING true)
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_INSTALL_PREFIX}")

set(CPACK_ARCH ${ARCH})

message(STATUS "CPACK_PACKAGE_NAME = ${CPACK_PACKAGE_NAME}")
message(STATUS "CPACK_ARCH = ${CPACK_ARCH}")

include(CPack)
