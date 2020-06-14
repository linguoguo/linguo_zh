---
layout:     post
title:      用mxnet的gluon线性回归训练只有两个特征的数据集
subtitle:   动手学深度学习课外作业之房价预测
date:       2020-06-14
author:     少琳肆
header-img: post_img/2020-05/house.jpg
catalog: true
tags:
    - 机器学习
    - 博客
    - 深度学习
    - 学习资料
    - 菜鸟日记
    - 学习笔记
    - 房价预测
    - 线性回归
    - mxnet
    - gluon
---   

# 前言

自从上次试着用最基础的线性回归训练一个有80个特征的数据集，梯度爆炸之后，今天拿一个简单到不能再简单的数据集试试能不能成功收敛。途中我们又会遇到什么问题？

## 数据集
来自吴恩达机器学习课程第二周的课后练习。原本是txt文件，我通过下面三行代码把数据集另存为了csv，可以在这里[下载](https://github.com/linguoguo/data_science/blob/master/house_pricing/data/house_2_features.csv)。


```python
import pandas as pd
df = pd.read_csv("ex1data2.txt",delimiter=',')
df.columns=['size','bedroom','price']
df.to_csv('house_simple.csv')
```

### 读取数据集

数据没有分训练集和测试集，房子的特征只有面积和房间数两个。
我们将通过`pandas`库读取并处理数据 

导入这里需要的包


```python
%matplotlib inline
import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import pandas as pd
```


```python
data = pd.read_csv('data/house/house_2_features.csv' ,index_col=0)
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>bedroom</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1600</td>
      <td>3</td>
      <td>329900</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2400</td>
      <td>3</td>
      <td>369000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1416</td>
      <td>2</td>
      <td>232000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3000</td>
      <td>4</td>
      <td>539900</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>4</td>
      <td>299900</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (46, 3)



### 预处理数据集

我们对连续数值的特征做`标准化（standardization)`：设该特征在整个数据集上的均值为$\mu$，标准差为$\sigma$。那么，我们可以将该特征的每个值先减去$\mu$再除以$\sigma$得到标准化后的每个特征值。对于缺失的特征值，我们将其替换成该特征的均值。


```python
data = data.apply(
    lambda x: (x - x.mean()) / (x.std()))

data.fillna(0);
```

标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值。


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>bedroom</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.495977</td>
      <td>-0.226166</td>
      <td>-0.073110</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.499874</td>
      <td>-0.226166</td>
      <td>0.236953</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.725023</td>
      <td>-1.526618</td>
      <td>-0.849457</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.246762</td>
      <td>1.074287</td>
      <td>1.592190</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.016724</td>
      <td>1.074287</td>
      <td>-0.311010</td>
    </tr>
  </tbody>
</table>
</div>



把数据集分成两部分，训练集和测试集，并通过`values`属性得到NumPy格式的数据，并转成`NDArray`方便后面的训练。


```python
n_train=36
train_features = nd.array(data[['size','bedroom']][:n_train].values)
test_features = nd.array(data[['size','bedroom']][n_train:].values)
train_labels = nd.array(data.price[:n_train].values).reshape((-1, 1))

```


```python
train_features.shape
```




    (36, 2)




```python
train_features[:3]
```




    
    [[-0.4959771  -0.22616564]
     [ 0.4998739  -0.22616564]
     [-0.72502285 -1.526618  ]]
    <NDArray 3x2 @cpu(0)>



### 定义模型

我们使用一个基本的线性回归模型和平方损失函数来训练模型。 关于更多gluon使用的步骤请参考[这里](https://zh.d2l.ai/chapter_deep-learning-basics/linear-regression-gluon.html)


```python
net = nn.Sequential()
net.add(nn.Dense(1))
```

### 初始化模型参数


```python
net.initialize(init.Normal(sigma=0.01))
```

### 定义损失函数


```python
loss = gloss.L2Loss()
```

### 定义优化算法

创建一个`Trainer`实例，并指定学习率为0.03的小批量随机梯度下降（`sgd`）为优化算法。该优化算法将用来迭代`net`实例所有通过`add`函数嵌套的层所包含的全部参数。这些参数可以通过`collect_params`函数获取。


```python
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

### 训练模型
随机读取包含batch_size个数据样本的小批量


```python
batch_size=4
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
```


```python
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    for X, y in train_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(train_features), train_labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
```

    epoch 1, loss: 0.349735
    epoch 2, loss: 0.255017
    epoch 3, loss: 0.207258
    epoch 4, loss: 0.180886
    epoch 5, loss: 0.166463
    epoch 6, loss: 0.156838
    epoch 7, loss: 0.150244
    epoch 8, loss: 0.145748
    epoch 9, loss: 0.142224
    epoch 10, loss: 0.139501


## 后记
暂时看训练是能收敛的，损失也比上次少很多很多。下次我们再看几个问题：
+ 怎么算测试集的房价
+ 有没有过拟
+ 损失函数的结果怎么看，是大还是小

新手村的小伙伴们，你们有什么看法呢？
<p align="center">
<img src="https://raw.githubusercontent.com/linguoguo/linguo_zh/master/img/end.png" width="200" >
</p>


此处围观我的[知乎博客](https://zhuanlan.zhihu.com/p/148303060)，这里[下载](https://github.com/linguoguo/data_science/blob/master/house_pricing/regression_house_2_features.ipynb)本文代码
