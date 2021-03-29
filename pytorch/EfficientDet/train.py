"""
训练主函数
"""

import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.efficientdet import EfficientDetBackbone
from nets.efficientdet_training import FocalLoss
from utils.dataloader import EfficientdetDataset, efficientdet_dataset_collate

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def get_classes(class_path):
    with open(class_path) as f:
        class_name = f.readlines()
    class_name = [c.strip() for c in class_name]
    return class_name
def fit_one_epoch(net,focal_loss,epoch,epoch_size,epoch_size_eval,gen,genval,Epoch,cuda):
    total_r_loss = 0
    total_c_loss = 0
    total_loss = 0
    val_loss = 0

    net.train()
    with tqdm(total = epoch_size,desc=f'Epoch {epoch+1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration,batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images,targets = batch[0],batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            optimizer.zero_grad()
            _,regression,classification,anchors = net(images)
            loss,c_loss,r_loss = focal_loss(classification,regression,anchors,targets,cuda=cuda)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_r_loss += r_loss.item()
            total_c_loss += c_loss.item()

            pbar.set_postfix(
                **{'class_loss':total_c_loss/(iteration+1),
                'regression_loss':total_r_loss/(iteration+1),
                'lr':get_lr(optimizer)}
            )
            pbar.update(1)
    net.eval()
    print('start Val')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                _, regression, classification, anchors = net(images_val)
                loss, c_loss, r_loss = focal_loss(classification, regression, anchors, targets_val, cuda=cuda)
                val_loss += loss.item()
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update()
    print("Finish val")
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    return val_loss/(epoch_size_val+1)

if __name__ == "__main__":
    phi = 0  #控制efficientDet版本
    # 根据phi的值选择输入图片的大小
    input_size = [512, 640, 768, 896, 1024, 1280, 1408, 1536]
    input_shape = [input_size[phi],input_size[phi]]
    Cuda =True
    # 获取种类名
    classes_path = 'classes.txt'
    class_name = get_classes(classes_path)
    num_classes = len(class_name)

    model = EfficientDetBackbone(num_classes,phi)
    model_path = 'efficientdet-d0.pth'
    print("load weight into state dice ...")

    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()
    efficient_loss = FocalLoss()
    #   获得图片路径和标签
    annotations_path = 'train.txt'
    val_split = 0.1
    with open(annotations_path) as f:
        lines = f.readlines()
    np.random.seed(1)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    if True:
        lr = 1e-3
        Batch_size = 8
        Init_epch = 0
        Freeze_Epoch = 50

        optimizer = optim.Adam(net.params,lr,weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=2,verbose=True)  
        #当网络的评价指标不在提升的时候，可以通过降低网络的学习率来提高网络性能学习率下降策略
        train_dataset = EfficientdetDataset(lines[:num_train],(input_shape[0],input_shape[1]),is_train=True)
        gen = DataLoader(train_dataset,Batch_size,shuffle=True,num_workers=4,pin_memory=True,
                drop_last=True,collate_fn=efficientdet_dataset_collate)
        
        val_dataset = EfficientdetDataset(lines[num_train:], (input_shape[0], input_shape[1]), is_train=False)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=efficientdet_dataset_collate)
        
        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        #   冻结一定部分训练
        for param in model.backbone_net.parameters():
            param.requires_grad = False
        for epoch in range(Init_epch,Freeze_Epoch):
            val_loss = fit_one_epoch(net,efficient_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step(val_loss)
    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        lr = 1e-4
        Batch_size = 4
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100

        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        train_dataset = EfficientdetDataset(lines[:num_train], (input_shape[0], input_shape[1]), is_train=True)
        val_dataset = EfficientdetDataset(lines[num_train:], (input_shape[0], input_shape[1]), is_train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=efficientdet_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=efficientdet_dataset_collate)

                        
        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.backbone_net.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            val_loss = fit_one_epoch(net,efficient_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)
            lr_scheduler.step(val_loss)
        








