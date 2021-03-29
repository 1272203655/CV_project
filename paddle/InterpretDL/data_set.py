import os
import numpy as np
import paddle 
from paddle.io import Dataset
from paddle.vision.datasets import DatasetFolder


class WoodDataset(Dataset):
    def __init__(self, mode='train',transforms=None):
        super(WoodDataset,self).__init__()
        self.mode = mode
        self.transforms = transforms 
        train_image_dir = r'E:\桌面\github\paddle\InterpretDL\wood_defect\trainImageSet'
        eval_image_dir = r'E:\桌面\github\paddle\InterpretDL\wood_defect\evalImageSet'
        train_data_folder = DatasetFolder(train_image_dir,transforms)
        eval_data_folder =DatasetFolder(eval_image_dir,transforms)
        if mode == 'train':
            self.data = train_data_folder
        elif mode == 'eval':
            self.data = eval_data_folder
    def __getitem__(self,index):
        data = np.array(self.data[index][0]).astype('float32')
        label = np.array([self.data[index][1]]).astype('int64')
        return data,label
    def __len__(self):
        return len(self.data)

train_dataset = WoodDataset('train',None)
for data,label in train_dataset:
    print(data.shape,label)
    break





