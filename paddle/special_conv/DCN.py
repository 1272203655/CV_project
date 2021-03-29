# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 20:25:49 2021

@author: 12722
"""

import paddle 
import paddle.nn.functional as F
from paddle.vision.transforms import ToTensor

from paddle.vision.ops import DeformConv2D

print(paddle.__version__)

# In[] 数据准备
transform = ToTensor()
cifar10_train = paddle.vision.datasets.Cifar10(mode='train',
                                               transform=transform)
cifar10_test = paddle.vision.datasets.Cifar10(mode='test',
                                              transform=transform)

# 构建训练集数据加载器
train_loader = paddle.io.DataLoader(cifar10_train, batch_size=64, shuffle=True)

# 构建测试集数据加载器
test_loader = paddle.io.DataLoader(cifar10_test, batch_size=64, shuffle=False)

# In[] 规则卷积
class MyNet(paddle.nn.Layer):
    def __init__(self,num_classes=1):
        super(MyNet,self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding = 1)
        self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(3,3),  stride=2, padding = 0)
        self.conv3 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3,3), stride=2, padding = 0)
        self.conv4 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3,3), stride=2, padding = 1)
        self.flatten = paddle.nn.Flatten()
        self.linear1 = paddle.nn.Linear(in_features=1024, out_features=64)
        self.linear2 = paddle.nn.Linear(in_features=64, out_features=num_classes)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
cnn1 = MyNet(num_classes=1)
model1 = paddle.Model(cnn1)
model1.summary((64,3,32,32))
# In[] 
from paddle.metric import Accuracy
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model1.parameters())
# 配置模型
model1.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )

# 训练模型
model1.fit(train_data=train_loader,
        eval_data=test_loader,
        epochs=2,
        verbose=1
        )  
# In[] DCNv1
class DCNv1(paddle.nn.Layer):
    def __init__(self,num_classes = 1):
        super(DCNv1,self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding = 1)
        # self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(3,3),  stride=2, padding = 0)
        # self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv3 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3,3), stride=2, padding = 0)

        self.offsets = paddle.nn.Conv2D(64, 18, kernel_size=3, stride=2, padding=1)                                 
        self.conv4 = DeformConv2D(in_channels=64, out_channels=64, kernel_size=3,stride=2,padding=1)
        self.flatten = paddle.nn.Flatten()

        self.linear1 = paddle.nn.Linear(in_features=1024, out_features=64)
        self.linear2 = paddle.nn.Linear(in_features=64, out_features=num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        
        offsets = self.offsets(x)
        x = self.conv4(x, offsets)
        x = F.relu(x)
        
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
# 可视化模型

cnn2 = DCNv1()

model2 = paddle.Model(cnn2)

model2.summary((64, 3, 32, 32))

# In[] DCNv2
class dcn2(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(dcn2, self).__init__()

        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding = 1)
        self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(3,3),  stride=2, padding = 0)
        self.conv3 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3,3), stride=2, padding = 0)
        self.offsets = paddle.nn.Conv2D(64, 18, kernel_size=3, stride=2, padding=1)
        self.mask = paddle.nn.Conv2D(64, 9, kernel_size=3, stride=2, padding=1)
        self.conv4 = DeformConv2D(in_channels=64, out_channels=64, kernel_size=(3,3), stride=2, padding = 1)
        self.flatten = paddle.nn.Flatten()
        self.linear1 = paddle.nn.Linear(in_features=1024, out_features=64)
        self.linear2 = paddle.nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        offsets = self.offsets(x)
        masks = self.mask(x)
        x = self.conv4(x, offsets, masks)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
cnn3 = dcn2()
model3 = paddle.Model(cnn3)
model3.summary((64, 3, 32, 32))