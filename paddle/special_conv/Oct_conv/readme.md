## OctConv   Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks With Octave Convolution  2019ICCV

图片通常包含低频信息和高频信息

OctConv网络的核心思想是提出了OctConv模块。该模块对图片包含的低频信息和高频信息分别处理并结合起来计算更新。在处理低频信息时将低频信息用低维度的张量来表示

![octconv实现细节](E:\桌面\github\paddle\spacial_conv\Oct_conv\octconv实现细节.png)

![img](https://ai-studio-static-online.cdn.bcebos.com/1d9d70f419724ab289d0e2c11d37172cc53389eef13a4706b0e288ce2583d445)

论文原文：[Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049)

参考代码：[PyTorch的实现](https://github.com/d-li14/octconv.pytorch)

