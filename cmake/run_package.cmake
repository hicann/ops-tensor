##############################################################################
# run_package.cmake
#
# 自定义 makeself 打包脚本 - 为 ops-tensor 生成 .run 包
##############################################################################

# 创建临时安装目录 (staging)
set(STAGING_DIR "${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/makeself_staging")
file(REMOVE_RECURSE "${STAGING_DIR}")
file(MAKE_DIRECTORY "${STAGING_DIR}")

message(STATUS "")
message(STATUS "==========================================")
message(STATUS "Packaging ops-tensor")
message(STATUS "==========================================")
message(STATUS "Staging directory: ${STAGING_DIR}")
message(STATUS "Target architecture: ${CPACK_ARCH}")
message(STATUS "Target SoC: ${CPACK_SOC_NAME}")
message(STATUS "")

# 执行 install 到临时目录
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${CPACK_CMAKE_BINARY_DIR}" --prefix "${STAGING_DIR}"
    RESULT_VARIABLE INSTALL_RESULT
    OUTPUT_VARIABLE INSTALL_OUTPUT
    ERROR_VARIABLE INSTALL_ERROR
)

if(NOT INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR "Installation to staging directory failed:\n${INSTALL_ERROR}")
endif()

message(STATUS "Files installed to staging directory")

# 设置环境变量供脚本使用
set(ENV{TARGET_ENV} "/usr/local/Ascend")

# 生成包名 (格式: cann-{soc}-ops-tensor-{version}_{os}_{arch}.run)
# 示例: cann-Ascend950-ops-tensor-1.0.0_linux-x86_64.run
set(PACKAGE_NAME "cann-${CPACK_SOC_NAME}-ops-tensor-${PROJECT_VERSION}_${CMAKE_SYSTEM_NAME}-${CPACK_ARCH}.run")
set(PACKAGE_COMMENT "${CPACK_SOC_NAME}_OPS_TENSOR_RUN_PACKAGE")

message(STATUS "Package name: ${PACKAGE_NAME}")
message(STATUS "Package comment: ${PACKAGE_COMMENT}")

# 检查 makeself 是否可用
find_program(MAKESELF_EXE makeself)

if(NOT MAKESELF_EXE)
    message(FATAL_ERROR "makeself not found. Please install makeself to create run packages.")
endif()

message(STATUS "Package name: ${PACKAGE_NAME}")
message(STATUS "Running makeself...")

# 构建 makeself 命令
set(MAKESELF_ARGS
    --nocomp
    --nox11
    "${STAGING_DIR}"
    "${PACKAGE_NAME}"
    "${PACKAGE_COMMENT}"
    "share/info/ops_tensor/scripts/install.sh"
)

# 执行 makeself 打包
execute_process(
    COMMAND ${MAKESELF_EXE} ${MAKESELF_ARGS}
    WORKING_DIRECTORY ${STAGING_DIR}
    RESULT_VARIABLE EXEC_RESULT
    OUTPUT_VARIABLE EXEC_OUTPUT
    ERROR_VARIABLE EXEC_ERROR
)

if(NOT EXEC_RESULT EQUAL 0)
    message(FATAL_ERROR "makeself packaging failed:\n${EXEC_ERROR}")
endif()

# 移动生成的 .run 文件到输出目录
file(MAKE_DIRECTORY ${CPACK_PACKAGE_DIRECTORY})

set(PACKAGE_OUTPUT_PATH "${CPACK_PACKAGE_DIRECTORY}/${PACKAGE_NAME}")

execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy
        "${STAGING_DIR}/${PACKAGE_NAME}"
        "${PACKAGE_OUTPUT_PATH}"
    RESULT_VARIABLE COPY_RESULT
)

if(COPY_RESULT EQUAL 0)
    message(STATUS "")
    message(STATUS "==========================================")
    message(STATUS "Package created successfully!")
    message(STATUS "==========================================")
    message(STATUS "Output: ${PACKAGE_OUTPUT_PATH}")
    message(STATUS "")
    message(STATUS "To install:")
    message(STATUS "  ${PACKAGE_OUTPUT_PATH}")
    message(STATUS "")
    message(STATUS "To uninstall:")
    message(STATUS "  ${PACKAGE_OUTPUT_PATH} --uninstall")
    message(STATUS "==========================================")
else()
    message(FATAL_ERROR "Failed to copy package to output directory")
endif()

# 清理 staging 目录
file(REMOVE_RECURSE "${STAGING_DIR}")
