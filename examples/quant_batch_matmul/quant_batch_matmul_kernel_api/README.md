# QuantBatchMatmul Kernel API Example

本样例把 QBMM Kernel API 文档中的五种组件组装放入真实 `__global__` Kernel entry，覆盖：

- `KernelMmadWithScaleMix`
- `KernelMmadWithScaleMixWithoutBatch`
- `KernelMmadWithScaleMxWithoutBatch`
- `KernelQbmmMultiBlockStreamK`
- `KernelQbmmPertensorMultiBlockStreamK`

样例包含 host 侧 ACL 初始化、GM 分配、Kernel 启动、结果回读、CSV 数据生成和 NumPy golden 校验，
并由仓库 `examples/CMakeLists.txt` 注册编译。

## 执行

```bash
bash examples/common/run.sh \
    --ops=quant_batch_matmul \
    --target=quant_batch_matmul_kernel_api
```

只执行单个场景，例如 MX StreamK：

```bash
bash examples/common/run.sh \
    --ops=quant_batch_matmul \
    --target=quant_batch_matmul_kernel_api \
    --ti=3
```

用例定义见 `quant_batch_matmul_kernel_api.csv`。运行器依次生成 `input/*.bin`、启动 NPU Kernel，
并使用 `verify_result_mx.py` 对 `output/npu_out.bin` 做 FP16 golden 校验。
