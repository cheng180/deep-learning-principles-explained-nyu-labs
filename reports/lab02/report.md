<center>姓名：程鹏炜  学号：24130061003</center>

| 姓名和学号？         | 程鹏炜，24130061003                                          |
| -------------------- | ------------------------------------------------------------ |
| 本实验属于哪门课程？ | 中国海洋大学25秋《软件工程原理与实践》                       |
| 实验名称？           | 实验2：深度学习基础                                          |
| 博客链接：           | https://blog.csdn.net/cpw123_/article/details/167387481?sharetype=blogdetail&sharerId=167387481&sharerefer=PC&sharesource=cpw123_&spm=1011.2480.3001.8118 |

## 一、实验内容

## 代码练习

1.colab完成基础配置：

![3228f04cc41215a135fd6eed797b4029](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-09\9e20f478899dc29eb19741386f9343c8\3228f04cc41215a135fd6eed797b4029.png)

挂在云盘方便创建文件夹管理文件。

2.基础pytorch训练----tensor语法：

![82eb4b1875b7330f3d8f50fee42e7fc9](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-09\9e20f478899dc29eb19741386f9343c8\82eb4b1875b7330f3d8f50fee42e7fc9.png)

处理基础数组

![7289359398f071c72343a05dafd0a381](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-09\9e20f478899dc29eb19741386f9343c8\7289359398f071c72343a05dafd0a381.png)

数据结构的

## ![9b2f4749e82f99689e09ab1df9322368](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\9b2f4749e82f99689e09ab1df9322368.png)

正态分布

## ![5dc7126d9fced4499758a0a2f4bf4ef9](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\5dc7126d9fced4499758a0a2f4bf4ef9.png)

优化正态分布和下载对应功能函数文件。

3.螺旋数据分类实践：

![0269fada5ec41ad31f002ba0fa1b427a](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\0269fada5ec41ad31f002ba0fa1b427a.png)

数据预处理简单处理维度通道channel

## ![e81af42ddbc267f80c710a559eed213a](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\e81af42ddbc267f80c710a559eed213a.png)



跑训练使用sgd优化器然后简单的线性linear层两层跑出来准确率很低，可视化结果：

![0599abfde1d44758bfe0e5a76265d722](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\0599abfde1d44758bfe0e5a76265d722.png)

然后改成relu的中间加入一层relu发现准确率几乎没这么上升，于是发现源代码里面是修改了优化器成为adam，而写的代码还在使用sgd所以没有进一步很明显的提升可以看到没有使用较好的优化器的时候根本训不起来，loss下降和relu的拟合都很差，可能在saddle/loacl point一直打转没法出山谷。原始实验链接nyu的可以参考这个https://oucaigroup.feishu.cn/wiki/Nbddwe54fiolLFkoslfcdYcAnZe

然后发现之前有照片位置不对可能也会降低一定的准确率：

于是进行修正

![e27950497aff92f16d5f0cdeb6e7f421](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\e27950497aff92f16d5f0cdeb6e7f421.png)

然后换成adam准确率到了九十多

![483376a1a816ba5d9b1b4fe816c44ab5](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\483376a1a816ba5d9b1b4fe816c44ab5.png)

![a32f41139bfa6e3c4442cba5ca114180](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\a32f41139bfa6e3c4442cba5ca114180.png)

可视化，然后再换成adamw试试，并且去掉relu只单独使用adam，对比发现效果也不是很好

![df71dba220977c61c269d9d2add00666](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\df71dba220977c61c269d9d2add00666.png)

## ![3ff80235720b5f8b80113812f89e0963](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\3ff80235720b5f8b80113812f89e0963.png)

单纯adam只有50，所以激活函数和优化器同样重要缺一不可

![image-20261008115242920](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008115242920.png)

4.思考总结：在发展中的深度学习中最让人头疼的事情就是训练训不起来，（目前已经比较成熟了有很多的万能优化器比较万精油的算法和function）关于relu的优点不用多说超越sigmoid的激活函数，让非线性引入，虽然看到他的函数图像只比正比例函数多一个平台但是这样两个relu拼接出来一个sigmoid然后无数个relu能拟合所有的复杂的函数，从而将这个非线性引入。

![image-20261008120406861](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008120406861.png)

![image-20261008120339848](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008120339848.png)

![image-20261008115952563](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008115952563.png)

然后就是简单的螺旋数据问题里面没有引入评估机制，对应任何学科来说评估机制都非常重要，而我们有了评估才能便于确定训练什么时候停止以及最优model保存哪一个参数权重，所以明显可以看到训练过程中有准确率比最后一轮要高或者loss更低。

## 二、问题总结与体会

### I.AlexNet有哪些特点？为什么可以比LeNet取得更好的性能？

#### 1.简单看一下LeNet的网络架构

![image-20261008171653238](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008171653238.png)

2.总共七层的cnn.其中pooling使用的是平均池化相当于在n*n的矩阵内选取平均值吧作为特征，而不是采用最大池化。参数较少，激活函数使用的是古老战神sigmoid和年轻一点点的战神 tanh。

![image-20261008172623500](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008172623500.png)

3.而这个alexnet也算是远古真神，第一次拉大参数量训练终于能train这么大的一个network了，首先max pooling采样保留的特征更加显著，然后激活函数使用了另一个真神--ReLU()这个效果激活函数确实好。首先根据BP算法，梯度反向传播的时候是要交叉相乘weight的其实就是激活函数的一阶导数，然而sigmoid也好tanh也好他们都是曲线并且梯度会存在很小的情况，那么反向传播如果模型非常的deep那么就出现了梯度消失的问题，而relu的导数恒为1（>0)的时候能规避这个问题，而且相对于指数函数和tan正切函数，线性函数倒数更好计算，也更容易加深deep network的层数让我们实现大参数量

4.其次relu具有稀疏激活的特性，大多数的神经元未激活就是0非常方便减少了大量计算，一定程度上能减缓overfitting

5.最后就是dropout了随机丢弃神经元，这个每次学习的相当于是个子网络，最大的好处就是增强鲁棒性，但是我不知道怎么证明他不会影响一写nueron之间的联系会不会被隐藏？

### II.激活函数有哪些作⽤？

1.激活函数首先把输入的feature或者说变量通过函式转换成特定的值，你首先肯定要通过矩阵乘法把参数和权重加权求和，但是求和的出来的一定能拟合实际的内容吗？我们需要一个桥梁把他转换成为可以拟合真实曲线真实关系的function，这就是激活函数，激活函数选取以前sigmoid现在relu，他们能拟合出无限的function任何的函数关系都能被拟合，而我们训练的目标就是找到一个function来产生输出尽可能的接近我们的实际值，类似于下图。

![image-20261008175213187](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008175213187.png)

2.我们分析一下，network的工作是什么：线性求和wi*xi+bi这样的公式，我就不写公式了太麻烦了放个计算图（本文图片大多截图自李宏毅的ML课程中，截的相关的内容做说明，直接自己画图不太现实）

![image-20261008175410102](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008175410102.png)

![image-20261008175508192](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008175508192.png)

那时候是neural network 后来cnn来了tf来了就叫deep network了，引入卷积技术，但是这样neural nw中的一列神经元还有激活函数封装成了一个layer，激活函数还是起到了非线性化的重要作用

2.不同场景下面我们要选择不同的激活函数，分类可能你就会用softmax，虽然很多时候relu是个万精油，但是dead neuron也会导致很多问题。

3.说点空洞的，激活函数能增强网络的表示能力和学习能力。

### III.梯度消失现象是什么？

1.这个在前文介绍relu的时候有提到过。梯度消失首先指的是反向传播过程中经过激活函数需要×激活函数的一阶导数导致如果导数小于1，梯度越来愈小，在比较深的network中最终消失趋近于0。

![image-20261008180423804](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008180423804.png)

2.造成这个问题的主要原因就是激活函数选取的较为饱和的函数（比如sigmoid这种），他们的导函数值域可以说是R导致会出现一阶导数很小的情况，然而使用relu之后梯度恒传播不会经过激活函数的时候衰减。

3.残差也可以有效缓解，我们保留之前的部分只对增量部分传播，这样子永远不会消失了梯度。

### IV.神经网络是更宽好还是更深好？

#### 1.首先肯定是deep比fat好。第一点就是deep的神经元k层可以制造出来2**k次方个linear有这么多的函数去拟合实际，他是指数增长而我们的fat network只能产生k个，线性增长，那么 显然同样的任务deep的需要的参数更少。原理证明就像逻辑门一样，并联的逻辑电路能产生2 k次方的真值表而不是串联。

![image-20261008181355552](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008181355552.png)

#### 2.网络变得更有效果了更加聪明

![image-20261008181504661](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008181504661.png)

兑成剪的准确率精准度一定更高，我们看下面这个图，从x映射到n layers的输出，看看输出是不是变得更加的多样，像信号一样它能够表征更多的信息

![image-20261008181640220](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008181640220.png)

### V.为什么要使⽤Softmax?

#### 1.关于这个首先我们要知道softmax是个什么，softmax是一个归一函数，总和为1，一般作为一个全连接层的分类器使用，他能把你输入的x最终转成类似于权重比例的数值，一眼就能看出来谁分数最高，这样模型就能给出置信度和最终输出的classification是哪一个？就是概率分布产生器，而有时候也被作为激活函数使用。

![image-20261008181801977](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008181801977.png)

![](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008182004647.png)

#### 2.最重要的就是梯度求导反向传播的过程，softmax函数让这个过程大大简化。如果选出来其中一类直接在数值上-1然后再bp就可以了，当然要配合crossentropy使用，基本上他俩连体的。

![image-20261008182526310](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261008182526310.png)

数学证明就不看了简单放一个完整bp过程，只需要直到最后推到出来反向传播的时候在对应class-1就得到了反向传播的导数，大大减少计算量而且本身也有非常出色的分类啊转化为非线性的概率分布的能力。

### VI.SGD 和 Adam 哪个更有效？

大多时候一定是Adam更好.首先介绍一下两者：

#### 1.两者都是属于我们的optimizer的范畴都是优化器，为了能够快速的更新模型的参数，这个参数就是使得我们定义出来的loss取到最小值的时候的参数集合

![801401c9941bc9b84265bcd2aab5f958](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\801401c9941bc9b84265bcd2aab5f958.png)

类似于这样我们需要不断对这个参数theta更新来找到函数的最小值使得模型的loss最小或者说最为准确，那么对应的问题就是函数的最小值极小值的判断和选择，数学上我们通过不断求导可以轻易的计算出来函数的最值或者是极小值极大值，对于三位函数我们可以做偏导，但是在train 一个model的时候如果我们要计算三阶导数二阶导数，那么带来的是非常庞大的计算量所我们需要找出一个合适的算法高效并且准确的更新参数。这时候gradient descent就问世了：

#### 2.Vanilla gradient descent或者sgd随机gradient descent

![98c0393bc837dbb94a4fc41757d8454d](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\98c0393bc837dbb94a4fc41757d8454d.png)

我们随机选取一个初始点然后进行梯度下降每次更新的步长由学习率×当前的梯度，这个梯度的计算之前也讨论过了我们用bp

![801401c9941bc9b84265bcd2aab5f958](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\801401c9941bc9b84265bcd2aab5f958.png)

sgd虽然是神但是在很多neuron network里面出现收敛速度极慢而且容易出现在saddle point地方来回打转，就像是下山的时候进入了山谷然后一直在山谷的两侧反复走没办法继续下山，这个在二维函数的时候对应local minima问题然而实际上是个假命题，因为二维世界没有下降的通路不代表三维的函数不能继续gd下降。

#### 3.Momentum：

在介绍adam之前我们先介绍一下动量这个物理领域很熟悉的东西。

![8d8dc4fe5ad88c16dce391cb9f28a19c](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\8d8dc4fe5ad88c16dce391cb9f28a19c.png)

想像这个球往下滚的过程，球在谷的位置能继续前进一段可能越过这个坡到达更低点，这个就是动量的表现，和物理世界的动量没有任何区别，那如果我们引入动量进入到下降算法里面是不是就能够解决一定的saddle point的问题呢？那难点就是怎么量化的加入动量到gd中去？

![8202df30a6383caeafd9b387e629d07e](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-09\9e20f478899dc29eb19741386f9343c8\8202df30a6383caeafd9b387e629d07e.png)

这个是传统梯度下降的向量表示图

![39c6c820883444bd57985ebf8a016f59](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-09\9e20f478899dc29eb19741386f9343c8\39c6c820883444bd57985ebf8a016f59.png)

这个是我们的加入动量之后的标识图，可以看到实际上就是做了一个向量的求和，movement变量就是上次的gradient的反方向的延长，梯度是本来没有m的运动方向的反方向。

![image-20261009141437448](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261009141437448.png)

可以看到收敛速度变快也更不容易在谷底打转

#### 4.Adam-AdamW：

我们现在遇到新的问题就是，学习率的变化现在作为一个超参数被我们定死了，但是实际上我需要他的变化动态调整，在平缓的地方学习步长变大，陡峭的地方学习步长变小。这时候我们那就得让学习率的变化变成一个动态调整的变量，而adagrad中是直接对伊塔学习率除grad即可，但是只关注单次的梯度缺少全局的视野，于是我们万能的hinton就发明了RMSProp，加入了之前的所有的gi都纳入计算并且给了个权重比例去调整，但是不知道为什么这样子效果就真的会变好，反正实际表现来说是非常好的效果。

#### ![8f5099f5d6e5f2d7c1411e20124214df](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\8f5099f5d6e5f2d7c1411e20124214df.png)

![image-20261009144324573](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261009144324573.png)

然后当我们把之前介绍的向量momentum纳入参数更新的公式就出现了Adam=RMSProp+Momentum。

![image-20261009144351077](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261009144351077.png)

那么sgd既没有adam的momentum又没有动态调整的学习率learning rate scheduling。那么大多时候都是adam表现更加出色。

![image-20261009144631154](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20261009144631154.png)

接下来登场的就是weight decay 在adam基础上引入权重衰减，就是直接加入到loss中去

loss = loss + weight decay parameter * L2 norm of the weights。

不细致的描述了有时候也会用这个。

### VII.总结

![58fa1a9cd74710797289fa9b98d0eddb](D:\微信预处理\xwechat_files\wxid_g5ilo63af7tn22_c93c\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\58fa1a9cd74710797289fa9b98d0eddb.png)

五步思考也是封装抽象模块化的思路。

学一点理论就不会对黑盒这样恐惧了。