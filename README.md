# Deep Learning Principles Explained — NYU Labs

> 深度学习原理解析——NYU 实验（Lab 1–4）

中国海洋大学《软件工程原理与实践》课程实验资料整理。目前完成实验 1—4：从深度学习基础、卷积神经网络，到 MobileNet、ShuffleNet、SENet、CBAM 和 HybridSN。

## 目录

```text
reports/
  lab01/                         实验一：课程热身与 AI 发展
  lab02/                         实验二：深度学习基础
  lab03/                         实验三：卷积神经网络
  lab04/                         实验四：MobileNet、ShuffleNet、HybridSN
notebooks/
  lab02/lab02_pytorch_spiral.ipynb
  lab03/lab03_cnn_mnist.ipynb
  lab04/lab04_lightweight_hybridsn.ipynb
papers/                           论文和阅读材料
data/                             数据集说明，不提交大体积数据集
```

## 如何在 Colab 中运行

1. 打开对应的 `.ipynb` 文件，上传到 Google Colab，或直接在 GitHub 页面中选择“Open in Colab”。
2. Lab2 不需要额外数据，直接运行即可。
3. Lab3 会自动下载 MNIST；如果 Colab 的数据下载较慢，可以重新运行数据加载单元。
4. Lab4 先提供 MobileNet/ShuffleNet 模块和 HybridSN 的随机输入 smoke test，确保网络结构能够直接运行；使用 Indian Pines 真数据时，按照 Notebook 中的数据路径说明上传 `.mat` 文件。

## 实验与课程链接

- [课程总览](https://oucai.club/classes/Software.html)
- [Lab1](https://oucai.club/classes/se/lab01)
- [Lab2](https://oucai.club/classes/se/lab02)
- [Lab3](https://oucai.club/classes/se/lab03)
- [Lab4](https://oucai.club/classes/se/lab04)

## CSDN 文章导读

这些文章是同一组实验的博客版记录，报告和 Notebook 适合查看完整材料，CSDN 文章适合按实验过程阅读和交流：

- [实验二：深度学习基础——PyTorch、优化器与螺旋数据分类](https://blog.csdn.net/cpw123_/article/details/167387481?spm=1001.2014.3001.5501)
- [实验三：卷积神经网络](https://blog.csdn.net/cpw123_/article/details/168344448)
- [实验四：MobileNet、ShuffleNet 与 HybridSN](https://blog.csdn.net/cpw123_/article/details/169268167)
- [作者 CSDN 主页：cpw123_](https://blog.csdn.net/cpw123_)

官网中的部分代码链接跳转到需要登录的飞书页面，因此本仓库中的 Notebook 是按照课程要求和报告中的实际实验过程重新整理的可运行版本，并没有把无法访问的页面内容冒充成官方源代码。

## 关键实验记录

实验四中 HybridSN 的第一层 `Conv3d` 曾误写成 `in_channels=1, out_channels=64`。由于下一层按照网络结构接收 8 个输入通道，前后张量无法衔接。修正为 `1 → 8 → 16 → 32` 后，3D 卷积部分的通道链路一致。报告中保留了错误现象、尺寸推导和修正解释。

## 大文件说明

原目录中的 `cifar-10-python.tar.gz` 体积较大，不放入 Git 仓库；数据集请按 Notebook 说明在线下载或手动上传。报告 PDF 和论文 PDF 保留在 `reports/` 与 `papers/` 中，方便 Typora、浏览器和 GitHub 直接查看。
