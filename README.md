# 驾驶员状态检测

[Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

![](driver.gif)

## 描述

使用深度学习方法检测驾驶员的状态。

* 输入：一张彩色图片
* 输出：十种状态的概率

状态列表：

* c0: 安全驾驶
* c1: 右手打字
* c2: 右手打电话
* c3: 左手打字
* c4: 左手打电话
* c5: 调收音机
* c6: 喝饮料
* c7: 拿后面的东西
* c8: 整理头发和化妆
* c9: 和其他乘客说话

## 数据

此数据集可以从 kaggle 上下载。[Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

如果你下载有困难，可以点这里:[百度云](http://pan.baidu.com/s/1dFzd0at)

## 使用库

* [OpenCV](https://github.com/opencv/opencv)
* [Matlibplot](https://matplotlib.org/)
* [Pytorch](https://pytorch.org/)
* [TensorboardX](https://github.com/lanpa/tensorboardX)

## 基础模型

* [VGGNet](https://arxiv.org/abs/1409.1556)
* [Xception](https://arxiv.org/abs/1610.02357)
* [pnasnet5large](https://arxiv.org/pdf/1712.00559.pdf)

## 源文件说明

* _ddd\_units/data\_mean.py_ 统计训练图片的均值与标准差
* _ddd\_units/splite\_valid.py_ 分离验证集与训练集
* _ddd\_units/visual\_classes.py_ 浏览每个驾驶状态
* _ddd\_units/visual\_samples.py_ 浏览随机的样本
* _ddd\_units/model\_plot.py_ 利用_tensorboardX_进行模型的绘制
* _ddd\_units/splite\_valid.py_ 分离验证集与训练集
* _ddd\_find\_tools_ 找到最好精度的模型
* _ddd\_image\_preprocessing.py_ 对图像进行预处理
* _ddd\_kfold\_split.py_ 对训练集进行_kfold_划分
* _ddd\_merge\_result.py_ 对结果集进行_bagging_融合
* _ddd\_xvgg.py_ 使用**xception**与**vgg16**联合用反卷得到的特征图做训练的模型
* _ddd\_masker.py_ 使用**vgg16**模型做的迁移学习的掩码图提取器
* _ddd\_resnet152.py_ 使用**resnet152**模型做的迁移学习的分类器
* _ddd\_pnasnet5large.py_ 使用**pnasnet5large**模型做的迁移学习的分类器


