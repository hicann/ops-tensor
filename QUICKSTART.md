# 算子开发快速入门：基于ops-tensor仓

本指南旨在帮助你快速上手基于CANN和`ops-tensor`算子仓的使用，最简化地完成环境安装、编译部署及算子运行。

## 一、环境安装

### 1. 有环境场景：Docker安装

Docker安装环境以Atlas A2产品（910B）为例。

**前提条件**：
* **Docker环境**：宿主机已安装Docker引擎（版本1.11.2及以上）。
* **驱动与固件**：宿主机已安装昇腾NPU的[驱动与固件](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.RC3.alpha002&driver=1.0.26.alpha)Ascend HDK 24.1.0版本以上。安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)》。
    > **注意**：使用`npu-smi info`查看对应的驱动与固件版本。

#### 下载镜像

拉取已预集成CANN软件包及`ops-tensor`所需依赖的镜像。

* **操作步骤**：
    1. 以root用户登录宿主机。
    2. 执行拉取命令（请根据你的宿主机架构选择）：
        * ARM架构：
        ```bash
        docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
        ```
        * X86架构：
        ```bash
        docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
        ```
        > **注意**：正常网速下，镜像下载时间约为5-10分钟。

#### Docker运行

请根据以下命令运行docker：

```bash
docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops bash
```

#### 检查环境

进入容器后，验证环境和驱动是否正常。

* **检查NPU设备**：
    ```bash
    npu-smi info
    ```
* **检查CANN安装**：
    ```bash
    cat /usr/local/Ascend/ascend-toolkit/latest/opp/version.info
    ```

### 2. 无环境场景：WebIDE开发（建设中）

对于无环境的用户，提供WebIDE开发方式，目前本方式正在建设中。

## 二、编译部署

### 1. 拉取ops-tensor仓库代码

```bash
git clone https://gitcode.com/cann/ops-tensor.git
cd ops-tensor
```

### 2. 编译ops-tensor算子包

```bash
bash build.sh --pkg
```

若提示如下信息，说明编译成功。

```bash
Self-extractable archive "cann-ops-tensor-1.0.0-linux-*.run" successfully created.
Build package success: build/cann-ops-tensor-1.0.0-linux-*.run
```

编译成功后，run包存放于项目根目录的build目录下。

### 3. 安装ops-tensor算子包

```bash
./build/cann-ops-tensor-*linux*.run
```

ops-tensor安装在`${ASCEND_HOME_PATH}/cann`路径中，`${ASCEND_HOME_PATH}`表示CANN软件安装目录。

### 4. 配置环境变量

```bash
source ${ASCEND_HOME_PATH}/cann/set_env.bash
```

### 5. 运行测试

```bash
bash build.sh --run
```

## 三、更多帮助

- [CANN 开发文档](https://www.hiascend.com/document)
- [Ascend C API 参考](https://hiascend.com/document/redirect/CannCommunityAscendCApi)
