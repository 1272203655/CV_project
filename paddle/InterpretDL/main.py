import paddle
import warnings
import paddle.vision.transforms as T 
warnings.filterwarnings("ignore")
from data_set import WoodDataset
from paddle.vision.models  import resnet18

train_transforms = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
eval_transforms = T.Compose([
    T.Resize(256), 
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
])

train_dataset = WoodDataset(mode='train',transforms=train_transforms)
eval_dataset = WoodDataset(mode='eval',transforms=eval_transforms)

train_loader = paddle.io.DataLoader(train_dataset,places=paddle.CUDAPlace(0),batch_size=8,shuffle=True,drop_last=True)
eval_loader = paddle.io.DataLoader(eval_dataset,places=paddle.CUDAPlace(0),batch_size=8,shuffle=True,drop_last=True)


model = resnet18(pretrained=False, num_classes=4)
optimizer = paddle.optimizer.Momentum(learning_rate=0.001,momentum=0.9,parameters=model.parameters())


import interpretdl as it 
fe = it.ForgettingEventsInterpreter(model, True)   # 初始化对训练数据解释的算法

epochs = 10
stats, (count_forgotten, forgotten) = fe.interpret(
    train_loader,
    optimizer,
    batch_size=8,
    epochs=epochs,
    save_path='assets')