# -*- coding: utf-8 -*-
from config import config_parameters
from paddle.io import DataLoader
from paddle.vision.datasets import Cifar10
import paddle.vision.transforms as T
from nlnet import nlnet50
import paddle
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='eval')
parser.add_argument('--net', default='no',type='str')
parser.add_argument('--num_classes', default=config_parameters['class_dim'], type=int,
                    help='number of classes for classification')
args = parser.parse_args()

transform  =  T.Compose(
    [T.Resize(224),
     T.Transpose(),
     T.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]),
     ])
PLACE=  paddle.CUDAPlace(0)
val_dataset = Cifar10(mode='test', transform=transform)
valid_loader = DataLoader(val_dataset, places=PLACE, shuffle=False, batch_size=args.batch_size, 
                          drop_last=True, num_workers=0, use_shared_memory=False)

if args.net == 'nl':
    model = nlnet50(num_classes=args.num_classes, nltype='nl')
if args.net == 'bat':
    model = nlnet50(num_classes=args.num_classes, nltype='bat')
if args.net == 'gc':
    model = nlnet50(num_classes=args.num_classes, nltype='gc')
elif args.net == 'resnet':
    model = paddle.vision.models.resnet50(num_classes=args.num_classes)
    
weights = paddle.load(args.net+'.pdparams')
model.set_state_dict(weights)
model.prepare(loss=paddle.nn.CrossEntropyLoss(),
            metrics=paddle.metric.Accuracy())
result = model.evaluate(valid_loader,verbose=1)
print(result)
    