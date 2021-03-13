# non_local   GCNet  and BAT模型

## 本项目使用cifar10数据集 进行训练和验证 

## 文件结构

1.config,py              配置文件

2.context_block.py            GCNet模块定义文件

3.nlnet.py          网络定义文件

4.non_local.py        non_local和BAT模型定义文件

5.train.py   训练

6.eval.py  验证

启动训练   在终端输入  python  train.py --net "nl" or " gc" or "bat"  

non-local论文原文：[Non-local Neural Networks](https://arxiv.org/abs/1711.07971)

GCNet原文：[Global Context Networks](https://arxiv.org/abs/2012.13375)

BAT论文原文：[Non-Local Neural Networks with Grouped Bilinear Attentional Transforms](http://openaccess.thecvf.com/content_CVPR_2020/html/Chi_Non-Local_Neural_Networks_With_Grouped_Bilinear_Attentional_Transforms_CVPR_2020_paper.html)

参考实现： https://github.com/BA-Transform/BAT-Image-Classification

https://github.com/xvjiarui/GCNet

https://aistudio.baidu.com/aistudio/projectdetail/1589847?channelType=0&channel=0

​					

​					 