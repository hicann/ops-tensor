/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_qbmm_mix_common.h
 * \brief Shared includes + AIC/AIV cross-core sync helper for the QBMM MIX kernels
 *        (kernel_qbmm_mix.h multi-batch and kernel_qbmm_mix_without_batch.h single-batch).
 */
#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

// AIC(cube) <-> AIV(vector) cross-core notify/wait handshake shared by the MIX kernels.
// Non-template base so the four methods and flag constants are found by ordinary unqualified
// lookup in the derived kernel classes (a dependent/template base would require this-> or
// Base:: qualification at every call site).
struct QbmmMixSyncHelper {
    static constexpr uint16_t AIC_SYNC_AIV_MODE = 4;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 0;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 1;
    static constexpr uint16_t FLAG_ID_MAX = 16;

    __aicore__ inline void NotifyVector()
    {
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
    }
    __aicore__ inline void WaitForVector()
    {
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
    }
    __aicore__ inline void NotifyCube()
    {
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG);
    }
    __aicore__ inline void WaitForCube()
    {
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG);
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
