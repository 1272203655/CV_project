"""
分类数据集的划分 
适用于paddle2.0的dataset构建
根据文件夹名称 确定类别
"""

import os
import shutil
import random
train_ratio = 0.8

train_dir = r'paddle\InterpretDL\wood_defect\trainImageSet'
eval_dir = r'paddle\InterpretDL\wood_defect\evalImageSet'
paths = os.listdir(r'paddle\InterpretDL\wood_defect')

if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(eval_dir):
    os.mkdir(eval_dir)

for path in paths:
    imgs_dir = os.path.join(r'paddle\InterpretDL\wood_defect', path)

    target_train_dir = os.path.join(train_dir,path)
    target_eval_dir = os.path.join(eval_dir,path)

    if not os.path.exists(target_train_dir):
        os.mkdir(target_train_dir)
    if not os.path.exists(target_eval_dir):
        os.mkdir(target_eval_dir)
    
    for file in os.listdir(imgs_dir):
        if random.uniform(0,1) <= train_ratio:
            shutil.copyfile(os.path.join(r'paddle\InterpretDL\wood_defect',path,file),os.path.join(target_train_dir,file))
        else:
            shutil.copyfile(os.path.join(r'paddle\InterpretDL\wood_defect',path,file),os.path.join(target_eval_dir,file))




