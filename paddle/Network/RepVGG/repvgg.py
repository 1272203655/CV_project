# -*- coding: utf-8 -*-
"""
使用paddle2.0复现 RepVGG网络模型
通过融入残差分支 和 参数重排策略  使得模型的精度和速度都很大提升
# TODO 未完成  参数融合部分不太理解 
"""

import paddle
import paddle.nn as nn 
import numpy as np

# Conv+BN
class ConvBN(nn.Layer):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,groups=1):
        super(ConvBN,self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size,stride,padding,groups=groups,bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)
    def forward(self, x):
        y = self.bn(self.conv(x))
        return y

# 构建RepVGGBlock模块
# RepVGG除了最后的池化层和分类层之外，都是清一色RepVGGBlock堆叠，十分简单
class RepVGGBlock(nn.Layer):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,
                 padding_mode='zeros',deploy=False):
        super(RepVGGBlock,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.deploy = deploy    # deploy是推理部署的意思
        
        assert kernel_size == 3  # 图像padding=1后经过 3x3 卷积之后图像大小不变
        assert padding == 1 
        
        padding_11 = padding - kernel_size//2

        self.relu  = nn.ReLU()
        if self.deploy:   # 定义推理模型时，基本block就是一个简单的 conv2D
            self.rbr_reparam = nn.Conv2D(in_channels, out_channels, kernel_size,stride,
                                         padding,dilation,groups,padding_mode,bias_attr=True)
        else:
            self.rbr_indentity = nn.BatchNorm2D(
                num_features=in_channels) if out_channels==in_channels and stride == 1 else None
            self.rbr_dense = ConvBN(in_channels, out_channels, kernel_size, stride, padding, groups)
            self.rbr_1x1 = ConvBN(in_channels, out_channels, 1, stride, padding_11, groups)
            # print('RepVGG Block, identity = ', self.rbr_indentity)   
            # 这句话就是判断这个block没有identity，没有的话返回None，具体看下图输出
            # 定义训练模型时，基本block是 identity、1x1 conv_bn、3x3 conv_bn 组合

    def forward(self, x):
        # 判断模型状态，若处于评估预测状态，则启用重排后的模型进行前向计算
        if hasattr(self, 'rbr_reparam'):
            return self.relu(self.rbr_reparam(x))
            # 推理阶段, conv2D 后 ReLU

        # 若处于训练状态下
        if self.rbr_indentity is None:
            id_out = 0
        else:
            # 如果 BN 层存在，则计算 BN
            id_out = self.rbr_indentity(x)
        # relu(3X3 ConvBN + 1X1 ConvBN + bn or 0)    
        return self.relu(self.rbr_dense(x)+self.rbr_1x1(x)+ id_out)

    def get_equivalent_kerbel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense) # 卷积核两个参数 W 和 b 提出来
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_indentity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, \
                        bias3x3 + bias1x1 + biasid
        
    def _pad_1x1_to_3x3_tensor(self,kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1,1,1,1])
        
    def _fuse_bn_tensor(self,branch):
        if branch is None:
            return 0,0

        if isinstance(branch, ConvBN):
            kernel = branch.conv.weight  # conv权重
            running_mean = branch.bn._mean  # BN mean
            running_var = branch.bn._variance    # BN var
            gamma = branch.bn.weight             # BN γ 
            beta = branch.bn.bias                # BN β
            eps = branch.bn._epsilon             # 防止分母为0
            # 当branch是3x3、1x1时候，返回以上数据，为后面做融合
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self,'id_tensor'):
                input_dim = self.in_channels // self.groups                                       # 通道分组，单个GPU不用考虑，详情去搜索分组卷积
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)    # 定义新的3x3卷积核，参数为0，这里用到DepthWise，详情去搜索MobileNetV1
                                                                                                 
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1                                      # 将卷积核对角线部分赋予1
                self.id_tensor = paddle.to_tensor(kernel_value)

            kernel = self.id_tensor               # conv权重       
            running_mean = branch._mean           # BN mean
            running_var = branch._variance        # BN var
            gamma = branch.weight                 # BN γ
            beta = branch.bias                    # BN β
            eps = branch._epsilon                 # 防止分母为0
            # 当branch是 identity，也即只有BN时候返回以上数据
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        # 提取W、b，不管你是 3x3 1x1 identity都要提取
        return kernel * t , beta - running_mean * gamma/std   #具体看论文公式 推导

        # 上述公式没有提到 conv 1x1、conv 3x3 的 bias
        # 官方代码里面不考虑训练模型的conv的bias，所以上面去掉b 
    def repvgg_convert(self):
        kernel,bias = self.get_equivalent_kerbel_bias()
        return kernel.numpy(), bias.numpy()

class RepVGG(nn.Layer):
    def __init__(self,num_blocks,num_classes=100,width_multiplier=None, override_groups_map=None, deploy=False):
        super(RepVGG, self).__init__()  
        assert len(width_multiplier)==4  # 瘦身因子，减小网络的宽度，就是输出通道乘以权重变小还是变大
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict() # 这部分是分组卷积，单个GPU不用考虑
        assert 0 not in self.override_groups_map

        self.in_planes = min(64,int(64*width_multiplier[0]))
        self.stage0 = RepVGGBlock(in_channels=3,out_channels=self.in_planes,
                        kernel_size=3,stride=2,padding=1,deploy=self.deploy)
        self.cur_layer_idx = 1 
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2D(output_size=1)  # 全局池化，变成 Nx1x1（CxHxW），类似 flatten
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self,planes,num_blocks,stride):
        strides = [stride] + [1] * (num_blocks-1)    #print[2,1,1,1,1,.....]
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1) # 分组卷积
            blocks.append(RepVGGBlock(
                in_channels=self.in_planes,out_channels=planes,kernel_size=3,
                stride=stride, padding=1, groups=cur_groups, deploy=self.deploy
            ))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)
    def forward(self,x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = paddle.flatten(out,start_axis=1)
        out = self.linear(out)
        return out

# 实例化
optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

def create_RepVGG_A0(deploy=False,num_classes=10):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)
def create_RepVGG_A1(deploy=False,num_classes=10):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_A2(deploy=False,num_classes=10):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)


def create_RepVGG_B0(deploy=False,num_classes=10):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1(deploy=False,num_classes=10):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1g2(deploy=False,num_classes=10):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B1g4(deploy=False,num_classes=10):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)

def create_RepVGG_B2(deploy=False,num_classes=10):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B2g2(deploy=False,num_classes=10):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B2g4(deploy=False,num_classes=10):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B3(deploy=False,num_classes=10):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B3g2(deploy=False,num_classes=10):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B3g4(deploy=False,num_classes=10):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)


    






            
            
            
        
            
        
