# -*- coding: utf-8 -*-
"""基于octconv的ResNet定义文件"""
import paddle.nn as nn
from octconv import Conv_BN_ACT , Conv_BN
import paddle

__all__ = ['OctResNet', 'oct_resnet26', 'oct_resnet50', 'oct_resnet101', 'oct_resnet152', 'oct_resnet200']

class Bottleneck(nn.Layer):
    expansion = 4
    def __init__(self,inplanes,planes,stride=1,downsample=None,groups=1,
                 base_width = 64,alpha_in=0.5,alpha_out=0.5,norm_layer=None,output=False):
        super(Bottleneck,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes*(base_width/64.)) * groups
        
        self.conv1 = Conv_BN_ACT(inplanes,width,kernel_size=1,alpha_in=alpha_in,alpha_out=alpha_out,norm_layer=norm_layer)
        self.conv2 = Conv_BN_ACT(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, norm_layer=norm_layer,
                                 alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5)
        self.conv3 = Conv_BN(width, planes * self.expansion, kernel_size=1, norm_layer=norm_layer,
                             alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
    def forward(self,x):
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None
        
        x_h,x_l = self.conv1(x)
        x_h,x_l = self.conv2((x_h,x_l))
        x_h,x_l = self.conv3((x_h,x_l))
        
        if self.downsample is not None:
            identity_h,identity_l = self.downsample(x)
        
        x_h += identity_h
        x_l = x_l + identity_l if identity_l is not None else None
        
        x_h = self.relu(x_h)
        x_l = self.relu(x_l) if x_l is not None else None
        return x_h,x_l

class OctResNet(nn.Layer):
    def __init__(self,block,layers,num_classes=10,zero_init_residual=False,
                 groups=1,width_per_groups = 64 ,norm_layer=None):
        super(OctResNet,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_groups
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=7,stride=2,
                               padding = 3,bias_attr=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu  = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3,stride=2,padding=1)
        
        self.layer1 = self._make_layer(block,64,layers[0],norm_layer=norm_layer,alpha_in=0)
        self.layer2 = self._make_layer(block, 128, layers[1],stride=2,norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, alpha_out=0, output=True)
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                m.weight=m.create_parameter(m.weight.shape, default_initializer=nn.initializer.KaimingNormal())
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                m.weight.set_value(paddle.ones(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.sublayers():
                if isinstance(m, Bottleneck):
                    if m.conv3.bn_h:
                        m.conv3.bn_h.weight.set_value(paddle.zeros(m.conv3.bn_h.weight.shape))
                    elif m.conv3.bn_l:
                        m.conv3.bn_l.weight.set_value(paddle.zeros(m.conv3.bn_l.weight.shape))
        
    def _make_layer(self,block,planes,blocks,stride=1,alpha_in=0.5,alpha_out=0.5,norm_layer=None,output=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        downsample = None
        if stride!=1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                Conv_BN(self.inplanes,planes*block.expansion,kernel_size=1,stride=stride,alpha_in=alpha_in,
                        alpha_out=alpha_out)
                )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample, self.groups,self.base_width,
                            alpha_in,alpha_out,norm_layer,output))
        self.inplanes = planes * block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5, output=output))
        return nn.Sequential(*layers)
    def forward(self, x):
        x =self.maxpool( self.relu(self.bn1(self.conv1(x))))
        x_h,x_l = self.layer1(x)
        x_h, x_l = self.layer2((x_h,x_l))
        x_h, x_l = self.layer3((x_h,x_l))
        x_h, x_l = self.layer4((x_h,x_l))
        x = self.avgpool(x_h)
        x = paddle.reshape(x, [x.shape[0],-1])
        x = self.fc(x)
        
        return x
def oct_resnet26(pretrained=False, **kwargs):
    model = OctResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model


def oct_resnet50(pretrained=False, **kwargs):
    model = OctResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def oct_resnet101(pretrained=False, **kwargs):
    model = OctResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def oct_resnet152(pretrained=False, **kwargs):
    model = OctResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def oct_resnet200(pretrained=False, **kwargs):
    model = OctResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

if __name__ == '__main__':
    network = oct_resnet50(num_classes=10, zero_init_residual=True)
    img = paddle.zeros([1, 3, 224, 224])
    outs = network(img)
    print(outs.shape)
        
            
        
        
        
        
        
