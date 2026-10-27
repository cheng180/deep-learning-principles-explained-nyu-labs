<center>姓名：程鹏炜  学号：24130061003</center>

| 姓名和学号？         | 程鹏炜，24130061003                                          |
| -------------------- | ------------------------------------------------------------ |
| 本实验属于哪门课程？ | 中国海洋大学25秋《软件工程原理与实践》                       |
| 实验名称？           | 实验2：深度学习基础                                          |
| 博客链接：           | https://blog.csdn.net/cpw123_/article/details/168344448?sharetype=blogdetail&sharerId=168344448&sharerefer=PC&sharesource=cpw123_&spm=1011.2480.3001.8118 |

## 一、实验内容

## 代码练习和思考

### **1.手写数字识别任务**-**数据处理**：

加载下载数据集，train=true下载训练集然后对数据做transform操作基本上就是改变一些维度让或者裁剪预处理等等，这里就是简单的处理让他变成tensor按照对应维度的归一化然后还有规定batch数量。

![71a29bf2d5e1a7ea6c8651a12ef4c7e4](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\71a29bf2d5e1a7ea6c8651a12ef4c7e4.png)

![2efa29d085d04b5581f6d66e46bdabd4](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\2efa29d085d04b5581f6d66e46bdabd4.png)

划分训练和测试集，基本上都是同样的操作注意shuffle与否，基本上都需要去打乱数据顺序防止出现学习到数据排列规律。

![48a1ea5d3d7277e7b25c8b9f0273cd67](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\48a1ea5d3d7277e7b25c8b9f0273cd67.png)

展示训练集，然后再去看测试集能发现同样的数据是长得不一样的。比如这个7

![15d345a1954a856e136d6482aa58d1b2](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\15d345a1954a856e136d6482aa58d1b2.png)

为什么使用logsoftmax可能更适配吧不清楚为什么不直接使用softmax

### **2.模型构建**：

![00bc23aedece2f3fdd89ae2c18aeaca6](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\00bc23aedece2f3fdd89ae2c18aeaca6.png)

可以看到在普通的神经网络上和之前没什么区别，主要是cnn里面我们发现每一层传输的过程中特征通道维数是变化的而且是根据卷积计算的过程中kernel滤波器去做卷积运算和padding带来的变化，以及池化的时候做max还是avg都带来对应维度的变化。

整理了一点：

卷积层（无填充 padding=0，步长 stride=1）：
输出边长 = 输入边长 - kernel_size + 1
如果有填充和步长，通用公式是：
输出边长 = (输入边长 + 2×padding - kernel_size) / stride + 1，取整数部分
最大池化层：
输出边长 = 输入边长 / kernel_size（一般 stride 等于 kernel_size）

### **3.训练测试：**

![e215e73d00791ac12ba8e9dca561b8f6](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\e215e73d00791ac12ba8e9dca561b8f6.png)

值得注意的是grad每次归零然后就是前向传播反向传播即可。更新优化器更新参数打印日志。

发现报错了，忘记写get parameter函数了：

![d5f819be91c88b36aaff0d5c60e4bc45](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\d5f819be91c88b36aaff0d5c60e4bc45.png)

## ![d3f8eb77149121191fde9d6d08669b17](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\d3f8eb77149121191fde9d6d08669b17.png)

结果准确率85左右发现不对劲啊cnn应该没这么低然后就发现test用的model是fnn的不是cnn的

![1776b7547b5e02056e0d74aa0506bf0c](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\1776b7547b5e02056e0d74aa0506bf0c.png)

![3805354b601ad93da25e1d1027c27a71](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\3805354b601ad93da25e1d1027c27a71.png)

![c74d4d04d0928b0cb05df2b9e1954074](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\c74d4d04d0928b0cb05df2b9e1954074.png)

修改之后就对了95%

打乱之后的照片样子---像素之间乱窜最后准确率cnn大幅度下降了

![a9b1386cbc06ad9089ef7187bc856c80](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\a9b1386cbc06ad9089ef7187bc856c80.png)

![a5dc1f4d147568ed3963eae1094dc201](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\a5dc1f4d147568ed3963eae1094dc201.png)

### **4.cifar10数据集的实验-数据准备与处理**：

## ![8c33b81acfee65ccbd2c8bdf0552a718](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\8c33b81acfee65ccbd2c8bdf0552a718.png)

数据集下载非常慢而且临时存储在临时磁盘上面下次连接的时候就没了数据集。连接谷歌云硬盘本地下载传上去然后每次只要解压到notebook的对应位置即可。

### **5.model**

## ![3a26b5d65377bc54f27c957a145a10ec](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\3a26b5d65377bc54f27c957a145a10ec.png)

这里要加一个flatten才行也可以像实验里面用view做变换但是感觉flatten不用脑子哈哈。

![a83d8d139813f6eb80fb3cbd1c7cd84d](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\a83d8d139813f6eb80fb3cbd1c7cd84d.png)

这里不知道是不是版本问题应该改成直接调用next而不是调用iter的next方法。

### **6.mathematic**

先看看数据长什么样再训练按理来说这个步骤应该放在data部分的但是不改了，**大家也可以按照data-model-mathematic-experiment-conclusion的架构组织文章**尤其是这种深度学习类型的就这几个重点。

训练之后发现准确率只有41%非常奇怪找不到问题

![78765c9b83b25de4be8f79631eb4b216](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\78765c9b83b25de4be8f79631eb4b216.png)

然后打印发现数据集少了几个batch估计下载的时候出了问题，所以我后面用本地加云盘方式重新传了一边数据：

![78cda4e7dc66aebdeba41cf3a12a7f43](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\78cda4e7dc66aebdeba41cf3a12a7f43.png)

![abeebcb541ab8ca5e6f5e0e160c9bd11](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\abeebcb541ab8ca5e6f5e0e160c9bd11.png)

![a3e63bd3761f47352ef5d18a32a31996](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\a3e63bd3761f47352ef5d18a32a31996.png)

结果发现准确率还是只有50%。最后检查一圈发现是lr学习率少打一个0，设的太大了

![594ce71d8a9d6720250246423b4afe8e](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\594ce71d8a9d6720250246423b4afe8e.png)

### **7.VCG**-model：

前面数据集都一样我们直接看一下model部分有什么不同：

![3fd56ec30f8c0686276bc5a097d775b2](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\3fd56ec30f8c0686276bc5a097d775b2.png)

首先多出来了padding 填充，做卷积的时候当kernel大小取得不合适或者必须这么取的时候一定会出现一个kernel里面并不是我所需要的一个矩阵而是比如kernel 3*3 然后实际的数据大小不够刚刚好做几个卷积就需要我们填充

![image-20261017171700007](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261017171700007.png)

![image-20261017171804325](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261017171804325.png)

简单看一下几种类型。

然后除了padding就是池化了，之前都是无脑maxpooling在输出的里面用一个kernel去去这里面最大值作为结果的位置中的值

![image-20261017171942541](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261017171942541.png)

然后vcg里面加入了avgpooling不再是简单的取最大值虽然最大值一定程度上能反应这一区域的特征，而实际上cnn就是一个关注局部特征的方法，但是很多时候关注局部特征会出现盲人摸象的情况，所以一直都在追求泛化上面越走越远。

![image-20261017172146370](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261017172146370.png)

总结来说池化就是采样，采样好理解吧，取出样本来观察！哈哈。

![image-20261017172257380](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261017172257380.png)

最后这个为什么训练这么慢是什么情况是不是gpu没用啊？

PyTorch版本: 2.11.0+cpu CUDA是否可用: False PyTorch CUDA版本: None 当前设备变量: cpu 当前PyTorch没有检测到可用的CUDA GPU
我是傻逼没开gpu。点击更改运行时换成gpu。

![image-20261017173818113](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261017173818113.png)

只有七十八哈哈总是差点。

## 二、问题总结与体会

## I.dataloader 里面shuffle取不同的值的区别

shuffle是打乱图片顺序的意思，就是洗牌，如果不shuffle，原始数据集的图片可能是按照顺序放的比如dog一百张然后cat100张，那如果划分batch的时候这一类完全是dog或者完全是cat，很有可能模型就学到了这汇总顺序然后导致泛化很差，训练集上表现的非常高的准确率到测试集上就拉跨了。。。

## II.transform里，取了不同的值有什么区别

```python
    datasets.MNIST('./data',train=True,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])),
```

那这个举例子：transform里面对训练数据的预处理，首先转化成张量，可以理解为向量或者矩阵，然后需要归一化，比如像素值rgb是0-255要归一化到0-1.然后是标准化，这个就是和正态分布有关系了
$$

x_{\text{new}}=\frac{x-\mu}{\sigma}
$$
这样归一化要知道像素值的平均值和标准差才能这样标准化。这些值是发布数据集的嘉豪给出来的

注意compose的使用：之前没了解过是个串联变换的管道，进行连续的变化，总结一些常见预处理变化以及顺序的重要性

```python
from torchvision import transforms

# 训练时的变换管道
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),  # 1. 随机裁剪缩放
    transforms.RandomHorizontalFlip(p=0.5),                # 2. 随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # 3. 颜色抖动
    transforms.ToTensor(),                                  # 4. 转张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        # 5. 标准化
                        std=[0.229, 0.224, 0.225])
])

# 测试时的变换管道（确定性的）
test_transform = transforms.Compose([
    transforms.Resize(256),           # 1. 短边缩到256
    transforms.CenterCrop(224),        # 2. 中心裁剪224x224
    transforms.ToTensor(),             # 3. 转张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 使用
img = Image.open('cat.jpg')
img_train = train_transform(img)  # 每次调用结果不同（随机增强）
img_test = test_transform(img)    # 每次调用结果相同

```

比如肯定要先totensor再normalize等等。

## III.epoch和batch的区别

epoch是训练了所有的全部的数据之后就叫一个epoch

batch就是一次迭代参数所要训练的样本或者数据数量，然后可以理解为batch_size越小同样的数据量一个epoch要迭代参数，更新梯度的次数就越多，这有时候是好事情有时候也不是好事情，因为一次迭代的样本少了可能进入一些死胡同。所以batch_size的大小需要实际去动态调整对应的超参数。

引申出来的iteration就是数据量÷batch*epoch就是跑完训练所需要的总迭代次数。

## IV.1x1的卷积和 FC 有什么区别？主要起什么作⽤？

![image-20261017190419521](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261017190419521.png)

fc就是fully connected layer。就是所有的神经元对应相乘，两个矩阵相乘，一个权重矩阵一个参数矩阵，然后cnn卷积如果是1*1的。其实形容的就是kernel是1 * 1的一个矩阵，运算过程相当于fully connected

先讲一下卷积的通道数变化

![image-20261017210324626](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261017210324626.png)

一开始问的ai但是ai输出的话真的是模棱两可

![image-20261017210359975](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261017210359975.png)

实际上kernel怎么做运算我们都知道矩阵对应位置相×相加但是为什么通道数量会在cnn进行的过程中发生变化？谁决定的，那么这里就可以从这个图片中看出来首先kernel的通道数必须和input的一致然后卷积之后相加就得到一个输出通道，然后用多个不同的kernel就会得到大量不同的输出channel然后就知道output cahnnel实际上等于kernel的数量，然后为什么要设计这么多的kernel和通道，实际上kernel是为了提取局部特征，设计不同的kernel的0 1 位就能把对应位置的特征提取出来，那通道的个数就是我想要提取不同特征的数量，实际上和机器学习中的特征工程有点类似。

知道这个之后我们就可以介绍为什么1*1的卷积核比fc好了

![image-20261017210912282](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261017210912282.png)

看这个图最终的输出结果是6 * 6 * filters。最终的输出通道数取决于我使用了几个单位kernel。那么这就可以用来降维升维获得不同的特征。这是他和fc的主要区别。

1.主要的作用是降维，当输入为6x6x32时，1x1卷积的形式是1x1x32，当只有一个1x1卷积核的时候，此时输出为6x6x1。此时便可以体会到1x1卷积的实质作用：降维。当1x1卷积核的个数小于输入channels数量时，即降维[3]。

![image-20261017211558202](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261017211558202.png)

2.降维的同时我不同channel设计不同kernel带来的非线性的增加，一个filter对应卷积后得到一个feature map，不同的filter(不同的weight和bias)，卷积以后得到不同的feature map，提取不同的特征。

3.channel变换，后面接不同维度的kernel让线性组合传下去。

## V.residual leanring 为什么能够提升准确率？

### 1.先考虑一下残差是什么。

![582f6d23d45e59f7b49303fb3bdd16e1](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\582f6d23d45e59f7b49303fb3bdd16e1.png)

残差就是这样把和目标输出的结果减去x作为参数去经过神经网络去更新。

### 2.残差的作用1：

之前提到过的梯度消失的问题残差就能很好的解决，反向传播的时候这个x就变成了+1，有效的抑制了梯度消失的问题，那么梯度消失如果减缓了，那可以训练更多的轮次和更精度的学习率，一定程度上可以提高准确率

### 3.残差的作用2：

解决网络层数加深和过拟合带来的网络退化问题

简单地增加[网络层数](https://zhida.zhihu.com/search?content_id=256990084&content_type=Article&match_order=1&q=网络层数&zhida_source=entity)并不总能带来性能提升，反而可能导致性能下降。这种现象被称为“退化问题”

目标函数几乎是恒等映射，最终的目标就是让残差fx=0，趋近于0即可，这样预测值和实际值就理论最接近了。

其实就是让模型学习的难度降低了引入残差机制，这里放一个对比图，有无残差的模型结构

![image-20261018000951228](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261018000951228.png)

## VI.代码练习⼆⾥，⽹络和1989年 Lecun 提出的 LeNet 有什么区别？

### 1.首先看写的model的代码



```python
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3,6,5)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(6,16,5)
    self.fc1 = nn.Linear(16*5*5,120)
    self.fc2 = nn.Linear(120,84)
    self.fc3 = nn.Linear(84,10)
  def forward(self,x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.001)
```

这个模型结构比较简单我就不划模型结构图了先看一下原始lenet结构

![image-20261018001537908](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261018001537908.png)C1 → S2 → C3 → S4 → C5 → F6 

### 2.不同

输入不同一个灰度图一个rgb

池化方式不同，原始是avgpooling，这里是maxpooling

激活函数lenet用的tanh，我们用的relu

linear层维度不一样

分类采用的函数不一样，我们是10 个 logits，配合交叉熵，他output是softmax

优化器用的也不一样，一个sgd一个adam

## VII.卷积以后feature map 尺⼨会变⼩，如何应⽤ Residual Learning?

1.首先理解feature map是什么，为什么feature map会发生变化以及变化成什么：

首先C ，H， W是feature map包括的主要内容。卷积的过程中可能导致H W变化缩放，也可能用不同的kernel让channel通道数发生变化。

2.这个map变化了那么残差上一层的就不能直接添加到后面，以为维度不一样

3.那怎么解决这个问题或者在这个变化的基础上怎么利用添加进去残差机制：

用1*1的kernel做卷积最简单直接，前文说到这个kernel不会改变HW参数，然后至于channel选取特定的kernel数量进行缩放即可。

但是这个是让feature map的变化最小，我们可以做填充把不同的维度补齐再用残差。

## VIII.有什么⽅法可以进⼀步提升准确率？

1.换优化器比如adamw

2.加入weight-decay的同时加入余弦退火

3.warmup

4.数据增强进行更复杂的transform

5.dropout（尤其是过拟合问题）

6.调整超参数---贝叶斯最优化啥的没用过不知道怎么样

7.Batch Normalization层

![image-20261018203603198](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261018203603198.png)

8.优化模型结构加入一些注意力机制等等

9.优化训练算法label smoothing以及llrd等等算法都有不错的效果

## IX.问题讨论

我感觉学的时候很逆天啊，比如我从自己学习开始到现在都对BN不是很了解太为什么这么大名鼎鼎，一直吹捧他的优异表现，但是一读发现讲解都是说他解决了什么Covariate shift问题协变量偏移，也看不懂。然后正则化的效果。还有就是测试集准确率一定比验证集低吗？