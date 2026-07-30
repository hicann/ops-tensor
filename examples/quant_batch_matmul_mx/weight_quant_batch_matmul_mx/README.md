# weight_quant_batch_matmul_mx 样例

本样例通过一组精简的 CSV 用例运行 Blaze Weight-only MX Kernel。每个用例会生成 FP8 E4M3 激活值、
打包的 FP4 E2M1 权重、E8M0 MX Scale、FP16 Bias，以及由 CPU 计算得到的 FP16 Golden 结果。

## 在 NPU 上编译运行

```bash
source /path/to/cann/set_env.sh
bash run.sh
```

数据生成和 Golden 校验脚本依赖 `numpy` 和 `ml_dtypes`（`float8_e4m3fn`、`float8_e8m0fnu` 和
`float4_e2m1fn`），脚本位于 `../scripts/`。请在已安装这两个 Python 包的环境中运行。

仅编译、不启动 Device Kernel：

```bash
bash run.sh --build-only
```

使用其他用例文件，或复用已有构建结果：

```bash
bash run.sh --case weight_quant_batch_matmul_mx.csv --skip-build
```

默认 CSV 用例数量较少，但覆盖了无 Bias 的 Weight ND 场景和带 FP16 Bias 的 Weight NZ 场景。两个用例均使用
`M=32, N=40, K=128`、64 元素的 K Tile、两个 L1 Buffer，以及非对齐的 N 尾块。可执行文件要求运行在
Ascend950 NPU 上；CPU 侧验证由 Kernel UT 提供。
