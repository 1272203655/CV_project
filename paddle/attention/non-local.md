## Non-local 实现空间注意力的原理

普通的滤波都是3×3的卷积核，然后在整个图片上进行移动，处理的是3×3局部的信息。Non-Local Means操作则是结合了一个比较大的搜索范围，并进行加权。

全连接层虽然连接了相邻层的全部神经元，但只对单个神经元进行了权重学习，并未学习神经元之间的联系。

![img](https://ai-studio-static-online.cdn.bcebos.com/b4b59550764945cbbd7a372b62d1b376b851ae92947b4ecc82beaaf319b88f78)

- x是输入信号，cv中使用的一般是feature map
- i 代表的是输出位置，如空间、时间或者时空的索引，他的响应应该对j进行枚举然后计算得到的
- f 函数式计算i和j的相似度
- g 函数计算feature map在j位置的表示
- 最终的y是通过响应因子C(x) 进行标准化处理以后得到的

## Non-local 结构的实现



![f30547b985b84e169e3fcbfc6710e538bddfe2fc1d884c3493d951fd4d14bf78_cropped](E:\桌面\github\paddle\attention\f30547b985b84e169e3fcbfc6710e538bddfe2fc1d884c3493d951fd4d14bf78_cropped.png)

如上图所示，先将输入的特征图降维（降到1维）后逐次嵌入（embed）到 theta、phi 和 g 三个向量中。然后，将向量 theta 和向量 phi 的转置相乘，做 softmax 激活后再与向量 g 相乘。最后，再将这个 Non-local 操作包裹一层，这通过一个1×1的卷积核和一个跨层连接实现，以方便嵌入此注意力模块到现有系统中

## embedded Gaussian为例

![img](https://img-blog.csdnimg.cn/20200105163010813.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)