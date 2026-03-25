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
 * \file load_data_routing.h
 * \brief
 */

#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_LOAD_DATA_ROUTING_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_LOAD_DATA_ROUTING_H

#include "impl/experimental/tensor_api/arch/cube_datamove/load_data/npu_arch_2201/load_data_l12l0a.h"
#include "impl/experimental/tensor_api/arch/cube_datamove/load_data/npu_arch_2201/load_data_l12l0b.h"

#include "impl/experimental/tensor_api/arch/cube_datamove/load_data/npu_arch_3510/load_data_l12l0a.h"
#include "impl/experimental/tensor_api/arch/cube_datamove/load_data/npu_arch_3510/load_data_l12l0b.h"

namespace AscendC {
namespace Te {

class LoadDataIgnore {
public:
    template <const LoadDataTrait& trait, typename ...Args>
    __aicore__ inline void Run(const Args&... args) {}
};

template <Hardware dstPos, Hardware srcPos, uint32_t Version>
struct LoadDataTensor2Tensor {
    using type = LoadDataIgnore;
};

template <Hardware dstPos, Hardware srcPos, uint32_t Version>
struct LoadDataTensor2TensorNoCoord {
    using type = LoadDataIgnore;
};

template <>
struct LoadDataTensor2Tensor<Hardware::L0A, Hardware::L1, ArchVersion::V2201>
{
    using type = LoadDataL12L0A2201;
};

template <>
struct LoadDataTensor2Tensor<Hardware::L0B, Hardware::L1, ArchVersion::V2201>
{
    using type = LoadDataL12L0B2201;
};

template <>
struct LoadDataTensor2TensorNoCoord<Hardware::L0A, Hardware::L1, ArchVersion::V3510>
{
    using type = LoadDataL12L0A3510;
};

template <>
struct LoadDataTensor2TensorNoCoord<Hardware::L0B, Hardware::L1, ArchVersion::V3510>
{
    using type = LoadDataL12L0B3510;
};

template <>
struct LoadDataTensor2Tensor<Hardware::L0A, Hardware::L1, ArchVersion::V3510>
{
    using type = LoadDataL12L0AWithCoord3510;
};

template <>
struct LoadDataTensor2Tensor<Hardware::L0B, Hardware::L1, ArchVersion::V3510>
{
    using type = LoadDataL12L0BWithCoord3510;
};
} // namespace Te
} // namespace AscendC
#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_LOAD_DATA_ROUTING_H