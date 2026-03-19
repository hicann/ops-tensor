##############################################################################
# src/CMakeLists.txt
#
# 功能:
#   算子层 - 注册各算子到 ${OPS_TENSOR} 库
##############################################################################

# 引入算子注册函数
include(${CMAKE_SOURCE_DIR}/cmake/func.cmake)

##############################################################################
# 注册各算子子目录
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

message(STATUS "==========================================")
message(STATUS "Operator registration completed")
message(STATUS "==========================================")
message(STATUS "")

message(STATUS "Operator layer configured successfully")
