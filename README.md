# CV_project
## CV领域一些项目
### 一、paddle 文件夹是使用paddle框架  paddle2.0 或  paddle 1.8.4
####  1.attention 是注意力机制模块的学习  包括 non_local ,GCNet 和 BAT
（1）.non_local 使用最常用的 EmbeddedGaussian non_lock 模块
（2）.使用resnet18 加一个残差块  和加一个non_local模块做对比实验 再cifar数据集上对比  再visuadl上进行可视化 
####  2.InterpretDL 是飞浆的可解释性算法库
TODO 算法库安装失败 所以没有完成  只有 分类数据集的两种划分方法  包括标注分类数据集划分 和 飞浆API分类数据集划分
### 3.json_to_dataset 为labelme标注 多边形框转化为掩膜和可视化图像的通用程序
labelme==3.16.7 其它版本易出错
### 4.Network 为一些前沿的骨干网络 或者 感兴趣网络的复现
（1)repvgg 借鉴残差模块的思想  deploy时将重组为直连的3×3卷积  加快测试速度 减少了参数数量  主要看参数融合部分 
### 5.special conv 特殊的卷积构造思想
（1）Oct_conv 
该模块对图片包含的低频信息和高频信息分别处理并结合起来计算更新。在处理低频信息时将低频信息用低维度的张量来表示
（2）Dcn_conv 可变形卷积  飞浆有专门的API
### 一、pytorch 文件夹是使用pytorch框架  pytorch1.20 
#### 1.EffientDet  复现EffientDet网络 包括数据读取 网络构造 训练
effientnet  从三个角度使用NAs搜索网络  输入图像分辨率（更大的输入图像分辨率）  网络宽度（groups） 网络深度(更多的blocks堆叠) 
