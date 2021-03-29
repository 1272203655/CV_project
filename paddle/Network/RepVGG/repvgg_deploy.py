"""
test 部分 
训练模型转 推理模型  精华部分

多分支结构适合训练，去拟合数据，防止梯度爆炸梯度消失，
但是多分支结构有很多不足，比如占内存，每一个分支就要占据一部分内存，比如推理速度变慢，mac成本高，
而对于传统VGG来说，虽然精度不是很高，但是其推理速度十分优秀，原因是直筒式结构，单路一路到底的卷积，
尤其是 3x3 卷积，得益于现有计算库比如CuDNN，其计算密度比其他卷积高不少，所以基于此作者提出了结构重参数化，将1x1，identity分支参数融合到3x3，
使得RepVGG模型在推理阶段是一路的3x3卷积到底，在速度与精度部分实现了SOTA
"""

import paddle.nn as nn
from repvgg import create_RepVGG_A0
import numpy as np 
import paddle
from paddle.vision.datasets import Cifar10
import paddle.vision.transforms  as T 



repvgg_a0 = create_RepVGG_A0(deploy=False,num_classes=10)

# deploy 时将卷积权重转给直连的3×3卷积 bias 置0
def repvgg_model_convert(model,bulid_func):
    converted_weights = {}  # 将训练模型各层 W 和 bias 存入字典
    for name,module in model.named_sublayers():
        if hasattr(module,'repvgg_convert'):
            kernel, bias = module.repvgg_convert()
            converted_weights[name + '.rbr_reparam.weight'] = kernel
            converted_weights[name + '.rbr_reparam.bias'] = bias
        elif isinstance(module,nn.Linear):
            converted_weights[name + '.weight'] = module.weight.numpy()
            converted_weights[name + '.bias'] = module.bias.numpy()
    
    deploy_model = bulid_func
    for name,param in deploy_model.named_parameters():
        print('deploy param: ', name, np.mean(converted_weights[name]))
        param.data = paddle.to_tensor(converted_weights[name])
    
    return deploy_model

deploy_model = repvgg_model_convert(repvgg_a0,create_RepVGG_A0(deploy=True,num_classes=10))
deploy_model_hapi = paddle.Model(deploy_model)
deploy_model_hapi.summary((1,3,224,224))   # 参数总量 68.5M





transforms = T.Compose([
    T.Resize((224,224)),
    T.Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'),
    T.ToTensor()
])
val_dataset = Cifar10(mode='test',transform=transforms)