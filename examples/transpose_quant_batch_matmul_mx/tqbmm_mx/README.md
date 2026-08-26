# TransposeQuantBatchMatMul MX Example

TQBMM MX 量化 Transpose Batch MatMul 示例，使用 `GemmUniversal` (kernel_tqbmm_mx.h) 和 `BlockMmad` (block_mmad_qbmm_mx.h)。

## 运行

```bash
./run.sh <m> <k> <n> <batch> <bias> <a_dtype> <b_dtype> <c_dtype> <transA> <transB> <format> <base_m> <base_n> <base_k> <tile_k_l1> <scale_k_l1> <l1_buffers> <db_l0c> <a_full_load>
```

示例：
```bash
./run.sh 128 512 256 2 0 fp8_e4m3 fp8_e4m3 bfloat16 false false "(ND,ND)" 128 256 64 64 64 2 1 false
```

## 参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| m, k, n | 矩阵维度 | 128, 512, 256 |
| batch | batch 维度 | 1, 2, 4 |
| bias | bias 元素数（0 表示无 bias） | 0 |
| a_dtype | A 矩阵数据类型 | fp8_e4m3, fp4_e2m1 |
| b_dtype | B 矩阵数据类型 | fp8_e4m3, fp4_e2m1 |
| c_dtype | C 矩阵数据类型 | float16, bfloat16, float32 |
| transA | A 是否转置 | true/false |
| transB | B 是否转置 | true/false |
| format | 数据格式 | (ND,ND), (ND,NZ) |
| base_m | M 轴 tile 大小 | 128 |
| base_n | N 轴 tile 大小 | 256 |
| base_k | K 轴 tile 大小 | 64 |
| tile_k_l1 | L1 中 K 轴 tile 大小 | 64 |
| scale_k_l1 | L1 中 scale K 轴 tile 大小 | 64 |
| l1_buffers | L1 buffer 数量 | 2 |
| db_l0c | L0C 双缓冲开关 | 1 |
| a_full_load | A 矩阵全加载 | false |

## 布局说明

- A（x1）物理存储 `[M, B, K]`（permX1，对应 `transA` 语义）；数据文件按 M-B-K 顺序排列。
- C（y）物理输出 `[M, B, N]`（Batch 内嵌于 M 平面：B 步长 = N，M 步长 = B*N）。
- x1Scale 物理布局 `[M, B, scaleKLen]`，x2Scale 物理布局 `[B, scaleKLen, N]`，
  `scaleKLen = ceil(k/64)*2`（e8m0 指数，值域 [2^-126, 2^127]）。
- MxFP4 输入按 `fp4x2` 打包（每字节 2 个 fp4）；数据生成脚本
  `scripts/gen_data.py` 负责生成与上述布局一致的 `.bin` 文件。
- 详细布局/步长约定见 `docs/API/gemm/kernel/kernel_tqbmm_mx.md` 的"布局约定"小节。
