# AET-CNN

**AET-CNN 是一个使用 AET 语言编写的图片分类训练网络软件。**

本项目主要用于验证 **AET 编译器** 在人工智能计算、异构编程、面向对象编程以及高性能计算方面的能力。

AET-CNN 实现了完整的 CNN 训练流程，并通过 CIFAR 数据集训练测试 AET 语言在实际 AI 工作负载中的表现。

在作者的测试环境中，AET-CNN 的 CIFAR 训练速度相比 Darknet-Alex 快约 **30%~40%**。

------

# 项目介绍

AET-CNN 不只是一个 CNN 网络实现，更是 **AET 编译器的实际应用案例**。

通过 AET-CNN，可以展示 AET 的核心能力：

- 异构计算支持
- GPU 加速
- 面向对象编程
- C 语言兼容
- 编译器自动优化
- AI 工作负载支持

AET 的目标是：

> 使用一次编写的代码，在不同计算核心上运行。

让开发者同时获得：

- C 语言的执行效率
- 面向对象的软件设计能力
- 异构计算的高性能

------

# 性能测试

## CIFAR 图片分类训练测试

测试环境：

- GPU：NVIDIA GTX 1650
- CUDA 后端
- 数据集：CIFAR
- 网络模型：AlexNet 类 CNN 网络

测试结果：

| 项目         | 训练速度     |
| ------------ | ------------ |
| Darknet-Alex | 基准         |
| AET-CNN      | 快约 30%~40% |

测试结果表明：

AET 编译器生成的 GPU 程序能够在实际深度学习训练任务中达到较高性能。

> 注：性能会受到 GPU 型号、CUDA 版本、驱动以及系统环境影响。

------

# 什么是 AET？

AET 全称：

**Active Expandable Translator**

是一种面向异构计算的新型编程语言和编译器系统。

AET 的设计目标：

> 编写一次，多芯运行。

传统异构计算开发通常需要开发者针对不同硬件分别编写代码，例如：

- CPU 程序
- GPU CUDA 程序
- 不同加速器程序

AET 希望通过编译器技术减少这种差异，让开发者使用统一语言描述计算任务。

------

# AET 语言特点

## 1. 完全兼容 C

AET 保留 C 语言生态，兼容已有 C 代码。

例如：

```c
int main()
{
    printf("Hello AET\n");
    return 0;
}
```

原有 C 程序可以逐步迁移到 AET。

------

## 2. 面向对象支持

AET 在保持 C 兼容性的基础上，引入面向对象编程能力。

例如：dfdf

```c
class$ Network
{
    public$ void$ train()
};

impl$ Network
{
    void$ train(){
        ...
    }
}
```

可以使用类、对象等方式组织大型软件系统。

------

## 3. 异构计算能力

AET 面向多核心计算环境设计。

同一份源代码可以面向不同计算资源：

- CPU
- GPU
- 其他异构计算设备

由编译器负责代码转换和优化。

------

# AET-CNN 架构

整体结构：

```
             AET-CNN
                |
        AET 编程语言
                |
          AET 编译器
                |
      CPU / GPU / 异构硬件
```

AET-CNN 的网络结构、训练逻辑以及计算操作均使用 AET 编写。

通过 AET 编译器生成目标平台代码，并利用底层硬件能力执行。

------

# 为什么开发 AET-CNN？

人工智能的发展带来了大量计算需求。

目前常见方式：

- Python + 深度学习框架
- CUDA 手写高性能计算

两者之间存在一定矛盾：

高级语言：

- 开发效率高
- 但性能依赖底层框架

底层语言：

- 性能高
- 但开发复杂

AET 尝试探索：

> 能否通过新的语言和编译器，让程序员同时拥有高级语言开发效率和底层硬件性能？

AET-CNN 是这个方向的一个实践验证。

------

# 编译运行

## 环境要求

- Linux
- NVIDIA GPU
- CUDA 环境
- AET 编译器

## 获取代码

```bash
git clone https://github.com/onlineaet/aet-cnn.git

cd aet-cnn
```

------

## 编译

```bash
编译前，请修改文件makefile中的内容。
原内容：
GCC_AET_INSTALL_DIR  :=/home/xx/aetgcc/install
把/home/xx/aetgcc/install改为你的aet安装路径
make -j$(nproc)
```

------

## 运行训练

```bash
运行classifier_cifar.sh前，请改classifier_cifar.sh中的cifar数据路径。
把/home/xxx/aet-cnn-data改为下载的文件aet-cnn-data.tar.xz的解压路径。
export LD_LIBRARY_PATH=你的aet安装路径/lib64:$LD_LIBRARY_PATH
./classifier_cifar.sh
```

程序将启动 CIFAR 图片分类训练。

------

# AET 与 AET-CNN 的关系

```
              应用层
              AET-CNN
                 ↓
              AET语言
                 ↓
             AET编译器
                 ↓
          CPU / GPU / 多芯计算
```

AET-CNN 是 AET 编译器的一个展示项目。

真正的核心是：

**AET 编译器和 AET 语言体系。**

------

# 后续计划

未来方向：

- 更多神经网络算子支持
- GPU 后端优化
- 更多硬件平台支持
- 更大规模 AI 模型支持
- 编译器优化增强
- 分布式计算支持

------

# 关于 AET

AET 是一个探索下一代异构计算编程方式的编译器项目。

目标：

> 让异构计算编程更加简单，同时保持接近原生硬件的性能。

AET 尝试结合：

- C 语言兼容性
- 面向对象设计
- 编译器优化能力
- 异构计算能力

------

# 作者

AET Compiler & AET-CNN

onlineaet@163.com

一个探索未来异构编程语言方向的开源项目。