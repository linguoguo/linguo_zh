---
layout:     post
title:      用mxnet的随机梯度下降训练房价预测时出现了NaN
subtitle:   哪里错了？
date:       2020-05-23
author:     少琳肆
header-img: post_img/2020-05/house.jpg
catalog: true
tags:
    - 机器学习
    - 博客
    - 深度学习
    - 学习资料
    - 菜鸟日记
    - 错题本
    - 学习笔记
---    
## 如题 ：

<img src="https://raw.githubusercontent.com/linguoguo/linguo_zh/master/post_img/2020-05/1.png" width="700" >

### 因为学习速率过大，梯度值过大，产生梯度爆炸

试了很多学习率之后看到损失函数大到上天了，但是为什么后面出现了NaN，我有两个猜想：

<img src="https://raw.githubusercontent.com/linguoguo/linguo_zh/master/post_img/2020-05/3.png" width="700" >

1，损失函数算出来的结果太大了，不显示inf无限大，而是NaN。

2，哪个小样本里面有NaN，所以就突然出现了。

每个步骤的数据都打印出来，因为这个数据集有331个特征，太多了，我抽了3个特征出来看，还是出现了NaN，不过我也在演算的过程中找到了原因：

w不小心就越来越小直到负无穷了，再和X相乘，就出现了NaN

<img src="https://raw.githubusercontent.com/linguoguo/linguo_zh/master/post_img/2020-05/2.png" width="300" >

### 数据没有正则化

同样的学习率，没有正则化更容易出现Nan。
<img src="https://raw.githubusercontent.com/linguoguo/linguo_zh/master/post_img/2020-05/4.png" width="700" >

### 最后一个很蠢的原因

因为w和b没有重置，沿用了前面的NaN，只改动学习率对结果没有影响。
