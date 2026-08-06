# quant_batch_matmul_cube 样例

本样例直接组装并调用 `include/blaze/gemm/kernel/kernel_qbmm_cube.h`，演示 HiFloat8 Quant Batch Matmul
的 Cube/Fixpipe 路径。A/B 为 `hifloat8_t`，L0C 为 FP32，输出为 BF16，可选 Bias 为所有 Batch 共享的
`float[N]`。

数据生成和校验逻辑参考 `ops-samples` 的 `quant_matmul_hifp8/scripts`：

- TT：`x1quantmode=pertensor`、`x2quantmode=pertensor`，A/B scale 均为 FP32 标量；
- TC：`x1quantmode=default`、`x2quantmode=perchannel`，B scale 文件为 FP32 `[N]`，Host 侧转换为
  Fixpipe 使用的 `uint64_t[N]`；
- 也支持 `x1quantmode=default`、`x2quantmode=pertensor`。

`gen_data.py` 支持生成 HiFloat8/BF16 和 Int8/Int32 数据。HiFloat8 Golden 按 Fixpipe 精度清除 scale
的低 13 位并保存为 BF16；Int8 Matmul 使用 Int32 累加和 Int32 Golden。`verify_result.py` 对 BF16
使用参考样例的误差标准，对 Int32 进行逐元素精确比较。

Kernel 的 `AType/BType/CType/BiasType/X2ScaleType` 均为模板参数；Host 根据 CSV dtype 选择对应的
预编译实例，buffer 大小也按实例类型计算。

## CSV 参数

CSV 每行包含：

```text
batch,M,K,N,AType,BType,CType,bias,biasType,transA,transB,x1quantmode,x2quantmode,x2ScaleType,
baseM,baseN,baseK,kL1
```

其中：

- `bias` 为 `0` 或 `N`；
- `transA/transB` 支持 `true/false` 或 `1/0`；
- 当前 HiFloat8 样例支持 `AType/BType=hifloat8_t`、`CType=bfloat16_t`、`biasType=float`；
- Int8 样例支持 `AType/BType=int8_t`、`CType=int32_t`、`biasType=int32_t`，量化模式为
  `default/default`，`x2ScaleType=float` 仅作为未使用的模板类型占位；
- TT 的 `x2ScaleType=float`，TC per-channel 的 `x2ScaleType=uint64_t`；TC 的 scale 文件仍为 FP32，
  Host 将其截断并编码为传给 Kernel 的设备侧 `uint64_t` scale；
- `baseM <= 256`、`baseK <= 128`、`baseN <= 256`，最大基本块为 `256 * 128 * 256`；
- `l1BufferNum` 固定为 `2`，`blockNum` 固定为 `32`，不在 CSV 中配置。

CSV 还包含用于输出和数据目录命名的 `casename` 列。当前样例仅使用 ND 输入，默认用例覆盖 HiFloat8
TT/TC，以及 Int8 的普通 ND、transA+transB 和 transB+batch 场景。

## 在 NPU 上编译运行

```bash
source /path/to/cann/set_env.sh
bash run.sh
```

仅编译、不启动 Device Kernel：

```bash
bash run.sh --build-only
```

使用其他用例文件，或复用已有构建结果：

```bash
bash run.sh --case quant_batch_matmul_cube.csv --skip-build
```

单独生成默认 TC Batch 用例对应的输入：

```bash
python3 gen_data.py \
  --batch 2 --m 128 --k 256 --n 128 --bias 0 \
  --a-type hifloat8_t --b-type hifloat8_t --c-type bfloat16_t --bias-type float \
  --trans-a false --trans-b false \
  --x1-quant-mode default --x2-quant-mode perchannel \
  --x2-scale-type uint64_t \
  --output-dir data/tc_nd_batch
```

对应的可执行文件调用为：

```bash
./build/quant_batch_matmul_cube/quant_batch_matmul_cube \
  2 128 256 128 hifloat8_t hifloat8_t bfloat16_t 0 float \
  false false default perchannel uint64_t \
  256 256 128 128 data/tc_nd_batch
```
