# 项目文档

## 目录说明

```
docs/
├── zh/                        # 中文文档
│   ├── context/              # 公共文档（环境、build 参数等）
│   │   └── build.md          # build.sh 参数说明
│   ├── develop/              # 开发文档
│   │   └── test_writing_guide.md  # 测试编写指南
│   └── README.md             # 本文件
└── README.md                 # 英文文档索引
```

## 文档列表

| 文档 | 说明 |
|------|------|
| [build 参数说明](zh/context/build.md) | 介绍 build.sh 脚本的功能和参数含义 |
| [算子开发指南](zh/develop/operator_development_guide.md) | 介绍如何从零开发一个新算子 |
| [测试编写指南](zh/develop/test_writing_guide.md) | 介绍如何快速编写算子测试文件 |

## 文档使用说明

- **build 参数说明**：详细介绍 build.sh 的所有参数、使用示例和注意事项
- **算子开发指南**：介绍算子开发的完整流程，包括目录结构、文件说明、开发步骤等
- **测试编写指南**：详细介绍 ops-tensor 测试框架的使用方法和最佳实践
- 更多文档正在持续完善中...
