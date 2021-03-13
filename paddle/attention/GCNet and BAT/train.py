# -*- coding: utf-8 -*-
import paddle
from paddle.io import DataLoader
from paddle.vision.datasets import Cifar10
from config import config_parameters
import paddle.vision.transforms as T
from nlnet import nlnet50
import argparse
import warnings
warnings.filterwarnings("ignore")

# 编写用户友好的命令行接口  程序定义它需要的参数，然后 argparse 将弄清如何从 sys.argv 解析出那些参数。 argparse 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。
parser = argparse.ArgumentParser(description='resnet Training')

parser.add_argument('--net', default = 'no',type=str,help = 'the arch to use')
parser.add_argument('--num_classes', default=config_parameters['class_dim'], type=int,
                    help='number of classes for classification')
parser.add_argument('--batch_size', default=config_parameters['batch_size'],  type=int,
                    help='batch_size')
parser.add_argument('--lr', default=config_parameters['lr'], type=float)
parser.add_argument('--epochs', type = int, default = config_parameters['epochs'])
parser.add_argument('--weights', default='no',  type=str,
                    help='the path for pretrained model')
parser.add_argument('--pretrained', default=False,  type=bool,
                    help='whether to load pretrained weights')
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--warmup', type = int, default = 5)
args = parser.parse_args()

transform  =  T.Compose(
    [T.Resize(224),
     T.Transpose(),
     T.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]),
     ])
PLACE=  paddle.CUDAPlace(0)
train_dataset = Cifar10(mode='train',transform=transform)
train_loader = DataLoader(train_dataset,places=PLACE,batch_size=args.batch_size,shuffle=True,drop_last=True,
                          use_shared_memory=False,num_workers=0)
val_dataset = Cifar10(mode='test', transform=transform)
valid_loader = DataLoader(val_dataset, places=PLACE, shuffle=False, batch_size=args.batch_size, 
                          drop_last=True, num_workers=0, use_shared_memory=False)

if args.net == 'nl':
    model = nlnet50(num_classes=args.num_classes,nltype = 'nl')
if args.net == 'bat':
    model = nlnet50(num_classes=args.num_classes, nltype='bat')
if args.net == 'gc':
    model = nlnet50(num_classes=args.num_classes, nltype='gc')
elif args.net == 'resnet':
    model = paddle.vision.models.resnet50(num_classes=args.num_classes)
    
if args.pretrained:
    weights = paddle.load(args.weights)
    model.set_state_dict(weights)
    print('loading pretrained models')

class SaveBestModel(paddle.callbacks.Callback):
    def __init__(self,target=0.5,path='./best_model',verbose=0):
        self.target = target
        self.epoch = None
        self.path = path
    def on_epoch_end(self, epoch,logs=None):
        self.epoch = epoch
    def on_eval_end(self,logs=None):
        if logs.get('acc') > self.target:
            self.target = logs.get('acc')
            self.model.save(self.path)
            print('best acc is {} at epoch {}'.format(self.target, self.epoch))
            
callback_visualdl = paddle.callbacks.VisualDL(log_dir=args.net)
callback_savebestmodel = SaveBestModel(target=0.5, path=args.net)
callbacks = [callback_visualdl, callback_savebestmodel]

base_lr = args.lr
wamup_steps = args.warmup
epochs = args.epochs

def make_optimizer(parameters=None):
    momentum = 0.9
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=base_lr, T_max=epochs,verbose=False)
    
    learning_rate = paddle.optimizer.lr.LinearWarmup(
        learning_rate=learning_rate,
        warmup_steps=wamup_steps,
        start_lr=base_lr / 5.,
        end_lr=base_lr,
        verbose=False)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=momentum,
        parameters=parameters)
    return optimizer

optimizer = make_optimizer(model.parameters())
model = paddle.Model(model)
model.prepare(optimizer,
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy()) 
model.fit(train_loader,
          valid_loader,
          epochs=epochs,
          callbacks=callbacks,
          verbose=1)

    
    


