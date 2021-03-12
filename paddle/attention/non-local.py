# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 15:32:35 2021

@author: 12722
框架  paddle2.0 
数据集 cifar10
实现non-local 算法  
"""
# In[] 依赖项导入  数据读取  参数设置
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10
import warnings 
warnings.filterwarnings("ignore", category=Warning) # 过滤报警信息

BATCH_SIZE = 32
PIC_SIZE = 96
EPOCH_NUM = 10
CLASS_DIM = 10
PLACE = paddle.CUDAPlace(0)

# 数据集处理流程
transform  =  T.Compose(
    [T.Resize(PIC_SIZE),
     T.Transpose(),
     T.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]),
     ])

train_dataset = Cifar10(mode='train',transform=transform)
train_loader = DataLoader(train_dataset,places=PLACE,batch_size=BATCH_SIZE,shuffle=True,drop_last=True,
                          use_shared_memory=False,num_workers=0)
val_dataset = Cifar10(mode='test', transform=transform)
valid_loader = DataLoader(val_dataset, places=PLACE, shuffle=False, batch_size=BATCH_SIZE, 
                          drop_last=True, num_workers=0, use_shared_memory=False)


# In[]  Embedded Gaussian 实现 Non-local 模块
class EmbeddedGaussian(nn.Layer):
    def __init__(self,shape):
        # shape [N,C,H,W]
        super(EmbeddedGaussian,self).__init__()
        input_dim = shape[1]
        
        # 进行一个两倍的通道缩减  最后再升上来  减少参数量 
        self.theta = nn.Conv2D(input_dim, input_dim//2, kernel_size=1)
        self.phi = nn.Conv2D(in_channels=input_dim, out_channels=input_dim//2, kernel_size=1)
        self.g = nn.Conv2D(in_channels=input_dim, out_channels=input_dim//2, kernel_size=1)
        
        self.conv = nn.Conv2D(in_channels=input_dim//2, out_channels=input_dim, kernel_size=1)
        # 模块最后的1×1卷积后面加了一个 BN 层，这个 BN 层的放大系数（也就是权重参数）全部初始化为 0，
        # 以确保此模块的初始状态为恒等映射，使得其可以被插入到使用预训练权重的模型中去
        self.bn = nn.BatchNorm2D(input_dim, weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0)))
    def forward(self,x):
        shape = x.shape
        
        # 铺平操作   N * C//2 * (H*W)
        theta = paddle.flatten(self.theta(x),start_axis=2,stop_axis=-1)
        phi = paddle.flatten(self.phi(x), start_axis=2, stop_axis=-1)
        g = paddle.flatten(self.g(x), start_axis=2, stop_axis=-1)
        
        # theta 和phi  矩阵相乘 （C//2 * (H*W) ** ((H*W)*c//2) = ((h*w)*(h*w))
        non_local = paddle.matmul(x=theta, y=phi,transpose_y=True)
        # 经过softmax
        non_local = nn.functional.softmax(non_local)
        
        non_local = paddle.matmul(non_local, g) #(h*w)*c//2
        
        non_local = paddle.reshape(non_local,[shape[0],shape[1]//2,shape[2],shape[3]])
        non_local = self.bn(self.conv(non_local))
        
        return non_local + x
# 测试
n1 = EmbeddedGaussian([16,16,8,8])
x = paddle.to_tensor(np.random.uniform(-1,1,[16,16,8,8]).astype('float32'))
y = n1(x)
print(y.shape)

# In[] 运行效果对比
"""
在 ResNet18 模型结构上加上一个残差块作为基线版本，
后面的 Non-local 模块就替换这个残差块。
这样能确认效果的提升来自 Non-Local 结构，而非增加的参数。
"""
class Residual(nn.Layer):
    def __init__(self, num_channels, num_filters, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.use_1x1conv = use_1x1conv
        model = [
            nn.Conv2D(num_channels, num_filters, 3, stride=stride, padding=1),
            nn.BatchNorm2D(num_filters),
            nn.ReLU(),
            nn.Conv2D(num_filters, num_filters, 3, stride=1, padding=1),
            nn.BatchNorm2D(num_filters),
        ]
        self.model = nn.Sequential(*model)
        if use_1x1conv:
            model_1x1 = [nn.Conv2D(num_channels, num_filters, 1, stride=stride)]
            self.model_1x1 = nn.Sequential(*model_1x1)
    def forward(self, X):
        Y = self.model(X)
        if self.use_1x1conv:
            X = self.model_1x1(X)
        return paddle.nn.functional.relu(X + Y)
class ResnetBlock(nn.Layer):
    def __init__(self, num_channels, num_filters, num_residuals, first_block=False):
        super(ResnetBlock, self).__init__()
        model = []
        for i in range(num_residuals):
            if i == 0:
                if not first_block:
                    model += [Residual(num_channels, num_filters, use_1x1conv=True, stride=2)]
                else:
                    model += [Residual(num_channels, num_filters)]
            else:
                model += [Residual(num_filters, num_filters)]
        self.model = nn.Sequential(*model)
    def forward(self, X):
        return self.model(X)
class ResNet(nn.Layer):
    def __init__(self,num_classes = 10):
        super(ResNet,self).__init__()
        model = [
            nn.Conv2D(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
            ]
        model += [
            ResnetBlock(64, 64, 2,first_block=True),
            ResnetBlock(64, 128, 2),
            ResnetBlock(128, 256, 2+1), # 加入一个残差块 
            ResnetBlock(256, 512, 2)
            ]
        model += [
            nn.AdaptiveAvgPool2D(output_size=1),
            nn.Flatten(start_axis=1, stop_axis=-1),
            nn.Linear(512, num_classes),
            ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        y = self.model(x)
        return y
    
# 模型定义
model = paddle.Model(ResNet(num_classes=CLASS_DIM)) 
# 设置训练模型所需的optimizer, loss, metric
model.prepare(
    paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters()),
    paddle.nn.CrossEntropyLoss(),
    paddle.metric.Accuracy(topk=(1,5))
    )
# 启动训练、评估
model.fit(
    train_data=train_loader,eval_data=valid_loader,
    epochs=EPOCH_NUM,log_freq=500,
    callbacks=paddle.callbacks.VisualDL(log_dir='./log/BLResNet18+1')
    )       
    
# In[] 将加入的残差块换成 Embedded Gaussian non_local
class Residual(nn.Layer):
    def __init__(self, num_channels, num_filters, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.use_1x1conv = use_1x1conv
        model = [
            nn.Conv2D(num_channels, num_filters, 3, stride=stride, padding=1),
            nn.BatchNorm2D(num_filters),
            nn.ReLU(),
            nn.Conv2D(num_filters, num_filters, 3, stride=1, padding=1),
            nn.BatchNorm2D(num_filters),
        ]
        self.model = nn.Sequential(*model)
        if use_1x1conv:
            model_1x1 = [nn.Conv2D(num_channels, num_filters, 1, stride=stride)]
            self.model_1x1 = nn.Sequential(*model_1x1)
    def forward(self, X):
        Y = self.model(X)
        if self.use_1x1conv:
            X = self.model_1x1(X)
        return paddle.nn.functional.relu(X + Y)
class ResnetBlock(nn.Layer):
    def __init__(self, num_channels, num_filters, num_residuals, first_block=False):
        super(ResnetBlock, self).__init__()
        model = []
        for i in range(num_residuals):
            if i == 0:
                if not first_block:
                    model += [Residual(num_channels, num_filters, use_1x1conv=True, stride=2)]
                else:
                    model += [Residual(num_channels, num_filters)]
            else:
                model += [Residual(num_filters, num_filters)]
        self.model = nn.Sequential(*model)
    def forward(self, X):
        return self.model(X)
class ResNetNonLocal(nn.Layer):
    def __init__(self, num_classes=10):
        super(ResNetNonLocal, self).__init__()
        model = [
            nn.Conv2D(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        ]

        model += [
            ResnetBlock(64, 64, 2, first_block=True),
            ResnetBlock(64, 128, 2),
            ResnetBlock(128, 256, 2),
            EmbeddedGaussian([BATCH_SIZE,256,14,14]),
            ResnetBlock(256, 512, 2)
            ]
        model += [
            nn.AdaptiveAvgPool2D(output_size=1),
            nn.Flatten(start_axis=1, stop_axis=-1),
            nn.Linear(512, num_classes),
        ]
        self.model = nn.Sequential(*model)
    def forward(self, inputs):
        y = self.model(inputs)
        return y
# 模型定义
model = paddle.Model(ResNetNonLocal(num_classes=CLASS_DIM))
# 设置训练模型所需的optimizer, loss, metric
model.prepare(
    paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters()),
    paddle.nn.CrossEntropyLoss(),
    paddle.metric.Accuracy(topk=(1, 5)))
# 启动训练、评估
model.fit(train_loader, valid_loader, epochs=EPOCH_NUM, log_freq=500, 
    callbacks=paddle.callbacks.VisualDL(log_dir='./log/EmbeddedGaussion'))

            
                                
                                 
        
        



















