# 项目文档

## 目录说明

关键目录结构如下：

```
docs/
├── zh/                                # 中文文档目录
│   ├── context/                       # 公共文档目录
│   │   ├── build.md                   # build.sh 参数说明
│   │   ├── dir_structure.md           # 项目目录结构
│   │   └── quick_install.md           # 环境部署指南
│   ├── debug/                         # 调试调优文档目录
│   │   └── op_debug_prof.md           # 算子调试调优
│   ├── develop/                       # 开发文档目录
│   │   ├── operator_development_guide.md  # 算子开发指南
│   │   └── test_writing_guide.md      # 测试编写指南
│   ├── invocation/                    # 算子调用文档目录
│   │   └── quick_op_invocation.md     # 算子调用指南
│   ├── op_list.md                     # 算子列表
│   └── README.md                      # 中文文档索引
└── README.md                          # 文档总入口
```

## 文档列表

| 文档 | 说明 |
| :--- | :--- |
| [算子列表](zh/op_list.md) | 介绍项目包含的所有算子清单 |
| [环境部署](zh/context/quick_install.md) | 介绍项目的基础环境搭建 |
| [build 参数说明](zh/context/build.md) | 介绍 build.sh 脚本功能和参数含义 |
| [算子开发指南](zh/develop/operator_development_guide.md) | 介绍如何从零开发一个新算子 |
| [测试编写指南](zh/develop/test_writing_guide.md) | 介绍如何快速编写算子测试文件 |
| [算子调用指南](zh/invocation/quick_op_invocation.md) | 介绍算子编译、安装和调用方法 |
| [算子调试调优](zh/debug/op_debug_prof.md) | 介绍常见的算子调试、调优方法 |

## 附录

| 文档 | 说明 |
| :--- | :--- |
| [项目目录](zh/context/dir_structure.md) | 介绍项目完整的目录结构和各目录/文件的作用 |
