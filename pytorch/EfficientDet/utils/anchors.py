import itertools
import numpy as np
import torch
import torch.nn as nn

"""
跟yolo产生锚框  思想类似
在P3-p7每个特征层的每个像素点 产生9个锚框 
"""
class Anchors(nn.Module):
    def __init__(self,anchor_scale=4.,pyramid_levels=[3,4,5,6,7]):
        super().__init__()
        self.anchor_scale = anchor_scale
        self.pyramid_levels = pyramid_levels
        # strides步长为[8, 16, 32, 64, 128]， 特征点的间距
        self.stride = [2**x for x in self.pyramid_levels]
        self.scales = np.array([2**0,2**(1.0/3.0),2**(2.0/3.0)])
        self.ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    def forward(self,image):
        image_shape = image.shape[2:]
        boxes_all = []
        for stride in self.stride:
            boxes_level = []
            for scale,ratio in itertools.product(self.scales,self.ratios):  # 9
                """itertools.product 跟循环嵌套的for循环一致
                    product(A,B) == ((x,y) for x in A for y in B)
                """
                if image_shape[1] % stride !=0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride *scale
                anchor_size_x_2 = base_anchor_size * ratio[0]/2.0
                anchor_size_y_2 = base_anchor_size * ratio[1]/2.0
                x = np.arange(stride/2,image_shape[1],stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                
                xv,yv = np.meshgrid(x,y)
                
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)
                
                # y1,x1,y2,x2 vstack按行排列起来
                boxes = np.vstack((yv - anchor_size_y_2,xv-anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                
                boxes = np.swapaxes(boxes, 0, 1)  # np.asary 维度互换
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level,axis=1)
            boxes_all.append(boxes_level.reshape([-1,4]))
        anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = torch.from_numpy(anchor_boxes).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0) #在0维 增加维度
        
        return anchor_boxes
        
        
                
                
                
