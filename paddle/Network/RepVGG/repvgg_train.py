from repvgg import create_RepVGG_A0
import paddle.vision.transforms as T 
from paddle.vision.datasets import Cifar10
import paddle 

# 训练部分  使用高层API 

repvgg_a0=create_RepVGG_A0(num_classes=10)

transforms = T.Compose([
    T.Resize((224,224)),
    T.Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'),
    T.ToTensor()
])

train_dataset = Cifar10(mode='train',transform=transforms)
val_dataset = Cifar10(mode='test',transform=transforms)

model = paddle.Model(repvgg_a0)
# model.summary((1,3,224,224))

model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001,parameters=model.parameters()),
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy())

vdl_callback = paddle.callbacks.VisualDL(log_dir='log') # 训练可视化

model.fit(
    train_data=train_dataset, 
    eval_data=val_dataset, 
    batch_size=64, 
    epochs=10, 
    save_dir='save_models', 
    verbose=1, 
    callbacks=vdl_callback # 训练可视化
)