- RepVGG（[Making VGG-style ConvNets Great Again](https://arxiv.org/pdf/2101.03697.pdf)）是最近提出的一个增强版 VGG 模型
- 通过融入残差分支和使用参数重排策略，使得模型的精度和速度都有大幅提升

- 本项目参考官方开源项目 [【DingXiaoH/RepVGG】](https://github.com/DingXiaoH/RepVGG)

  ## RepVGG

  - RepVGG 是一个分类网路，该网络是在 VGG 网络的基础上进行改进，主要的改进点包括：
    - （1）在 VGG 网络的 Block 块中加入了 Identity 和残差分支，相当于把 ResNet 网络中的精华应用到VGG网络中
    - （2）模型推理阶段，通过 Op 融合策略将所有的网络层都转换为 Conv3*3，便于模型的部署与加速
  - 其模型结构示意图如下

![img](https://ai-studio-static-online.cdn.bcebos.com/70dbfea52b4d431a9a8408dd6e0e54e8aae433caadbd409ab806bf63e37d1840)