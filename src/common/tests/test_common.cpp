/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You should refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_common.cpp
 * @brief 公共测试框架实现
 */

#include "test_common.h"

namespace OpsTensorTest {

// 全局测试统计定义
TestStats g_global_stats;

int ACLManager::init(aclrtStream& stream) {
    // 初始化 ACL
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclInit failed, ret=" << ret << std::endl;
        return -1;
    }

    // 设置设备
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed, ret=" << ret << std::endl;
        aclFinalize();
        return -1;
    }

    // 创建 stream
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtCreateStream failed, ret=" << ret << std::endl;
        aclrtResetDevice(0);
        aclFinalize();
        return -1;
    }

    return 0;
}

void ACLManager::finalize(aclrtStream& stream) {
    if (stream) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(0);
    aclFinalize();
}

} // namespace OpsTensorTest
