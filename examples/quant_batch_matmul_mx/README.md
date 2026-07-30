# Quant Batch MatMul MX 样例

本目录包含 MXA8W4 量化 Batch MatMul 数据流样例。每个子目录都是独立的模板实现，维护自己的 CMake
目标和 CSV 用例；数据生成和结果校验脚本统一存放在 `scripts/` 目录中。

当前实现：

- `weight_quant_batch_matmul_mx`：基于 Blaze Tensor API 的实现，覆盖 Weight ND 和 Weight NZ 用例。

在实现目录中运行样例：

```bash
cd examples/quant_batch_matmul_mx/weight_quant_batch_matmul_mx
bash run.sh --build-only
```
