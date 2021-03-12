# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:24:31 2021

@author: 12722
non_local 和BAT模型定义文件
"""

import paddle
from paddle import nn
from paddle.nn import functional as F
import math
import numpy as np
from context_block import ContextBlock

def get_nonlocal_block(block_type):
    block_dict = {'nl': NonLocal, 'bat': BATBlock, 'gc': ContextBlock}
    if block_type in block_dict:
        return block_dict[block_type]
    else:
        raise ValueError("UNKOWN NONLOCAL BLOCK TYPE:", block_type)

class NonLocalModule(nn.Layer):
    def __init__(self,in_channels,**kwargs):
        super(NonLocalModule, self).__init__()
    def init_modules(self):
        for m in self.sublayers():
            if len(m.sublayers()) > 0:
                continue
            if isinstance(m, nn.Conv2D):
                m.weight=m.create_parameter(m.weight.shape, default_initializer=nn.initializer.KaimingNormal())
                if len(list(m.parameters())) > 1:
                    m.bias.set_value(paddle.zeros(m.bias.shape))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.zeros(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))
            elif isinstance(m, nn.GroupNorm):
                m.weight.set_value(paddle.zeros(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))
            elif len(list(m.parameters())) > 0:
                raise ValueError("UNKOWN NONLOCAL LAYER TYPE:", name, m)
class NonLocal(NonLocalModule):
    def __init__(self, inplanes, use_scale=True,**kwargs):
        planes = inplanes // 2
        self.use_scale = use_scale
        super(NonLocal, self).__init__(inplanes)
        self.t = nn.Conv2D(inplanes, planes, kernel_size=1,
                           stride=1, bias_attr=True)
        self.p = nn.Conv2D(inplanes, planes, kernel_size=1,
                           stride=1, bias_attr=True)
        self.g = nn.Conv2D(inplanes, planes, kernel_size=1,
                           stride=1, bias_attr=True)
        self.softmax = nn.Softmax(axis=2)
        self.z = nn.Conv2D(planes, inplanes, kernel_size=1,
                           stride=1, bias_attr=True)
        self.bn = nn.BatchNorm2D(inplanes)
    def forward(self, x):
        residual = x
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.shape
        
        t = paddle.transpose(paddle.reshape(t, (b,c,-1)), (0,2,1))
        p = paddle.reshape(p,(b,c,-1))
        g = paddle.transpose(paddle.reshape(g, (b, c, -1)), (0, 2, 1))

        att = paddle.bmm(t, p) # b h*w H*w
        if self.use_scale:
            att = paddle.divide(att, paddle.to_tensor(c**0.5))
        
        att = self.softmax(att) 
        x = paddle.bmm(att, g)  # b h*w c
        x = paddle.transpose(x, (0, 2, 1))
        x = paddle.reshape(x, (b, c, h, w))
        
        x = self.z(x)
        x = self.bn(x) + residual
        return x
    
        

        
        
        
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                