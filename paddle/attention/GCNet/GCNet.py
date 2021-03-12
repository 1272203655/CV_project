# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:00:08 2021

@author: 12722

GCNet 
paddle2.0
数据集使用cifar10

代码参考github https://github.com/xvjiarui/GCNet  pytorch源码
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
# In[] 网络结构


            
            
           
           
       
        

























