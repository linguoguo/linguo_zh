---
layout:     post
title:      用mxnet的gluon线性回归训练只有两个特征的数据集（3/3）
subtitle:   动手学深度学习课外作业之房价预测之第一次结果分析
date:       2020-06-22
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
    - 欠拟合
    - 损失函数
---  

# 跟吴恩达机器学习里面给的代码得出的结果比较
为了安心，我决定先跟该有的结果对比以下看看。
题目里面没有正则化，所以我们也试试看


```python
data_1 = pd.read_csv('data/house_2_features.csv' ,index_col=0)
n_train=36
train_features = nd.array(data_1[['size','bedroom']][:n_train].values)
test_features = nd.array(data_1[['size','bedroom']][n_train:].values)
train_labels = nd.array(data_1.price[:n_train].values).reshape((-1, 1))
test_labels = nd.array(data_1.price[n_train:].values).reshape((-1, 1))
```


```python
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
batch_size=2
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in train_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(train_features), train_labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
```

    epoch 1, loss: nan
    epoch 2, loss: nan
    epoch 3, loss: nan
    

NaN..... 我们重新算


```python
data_1 = pd.read_csv('data/house_2_features.csv' ,index_col=0)
print('mean ',data_1.mean())
print('std',data_1.std())
```

    mean  size         1998.434783
    bedroom         3.173913
    price      339119.456522
    dtype: float64
    std size          803.333019
    bedroom         0.768963
    price      126103.418369
    dtype: float64
    


```python

data_1 = data.apply(
    lambda x: (x - x.mean()) / (x.std()))

data.fillna(0);
n_train=36
train_features = nd.array(data_1[['size','bedroom']][:n_train].values)
test_features = nd.array(data_1[['size','bedroom']][n_train:].values)
train_labels = nd.array(data_1.price[:n_train].values).reshape((-1, 1))
test_labels = nd.array(data_1.price[n_train:].values).reshape((-1, 1))

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
batch_size=2
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    for X, y in train_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(train_features), train_labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
```

    epoch 1, loss: 0.410425
    epoch 2, loss: 0.318861
    epoch 3, loss: 0.260216
    epoch 4, loss: 0.222847
    epoch 5, loss: 0.198622
    epoch 6, loss: 0.182185
    epoch 7, loss: 0.171009
    epoch 8, loss: 0.162749
    epoch 9, loss: 0.156731
    epoch 10, loss: 0.152112
    epoch 11, loss: 0.148425
    epoch 12, loss: 0.145550
    epoch 13, loss: 0.143150
    epoch 14, loss: 0.141147
    epoch 15, loss: 0.139441
    epoch 16, loss: 0.138052
    epoch 17, loss: 0.136859
    epoch 18, loss: 0.135790
    epoch 19, loss: 0.134903
    epoch 20, loss: 0.134128
    epoch 21, loss: 0.133502
    epoch 22, loss: 0.132962
    epoch 23, loss: 0.132471
    epoch 24, loss: 0.132073
    epoch 25, loss: 0.131740
    epoch 26, loss: 0.131467
    epoch 27, loss: 0.131215
    epoch 28, loss: 0.130999
    epoch 29, loss: 0.130827
    epoch 30, loss: 0.130670
    epoch 31, loss: 0.130552
    epoch 32, loss: 0.130428
    epoch 33, loss: 0.130328
    epoch 34, loss: 0.130247
    epoch 35, loss: 0.130174
    epoch 36, loss: 0.130107
    epoch 37, loss: 0.130042
    epoch 38, loss: 0.129995
    epoch 39, loss: 0.129952
    epoch 40, loss: 0.129913
    epoch 41, loss: 0.129878
    epoch 42, loss: 0.129859
    epoch 43, loss: 0.129837
    epoch 44, loss: 0.129821
    epoch 45, loss: 0.129808
    epoch 46, loss: 0.129795
    epoch 47, loss: 0.129783
    epoch 48, loss: 0.129773
    epoch 49, loss: 0.129768
    epoch 50, loss: 0.129759
    


```python
res=net(nd.array([[(1650-1998.434783)/803.333019,(3-3.173913)/0.768963]]))
```


```python
res*126103.418369+339119.456522-293081
```




    
    [[-946.]]
    <NDArray 1x1 @cpu(0)>



对比一下发现结果差不远，如果按照练习的结果，其实我们这个计算已经算成功的了. 我们暂时可以安心了，不过我们的模型太简单，数据太少，有什么还可以做？ 还是拿一个复杂一点的数据集来练手？
这个问题再重新看看书，或许会有解答，第一次看的时候可能没有留意到的重点，再看一次会更有感觉。这里房价跟面积不是想象中的线性关系真是出乎意料的事情，我们似乎走进了死胡同，我们再找另外一个数据集试试看。再次之前我们再做最后一个尝试，把偏差太大的数据丢掉看看，从前面看，75%数据偏差不是很大，我们就只保留这部份。


```python
data_2 = pd.read_csv('data/house_2_features.csv' ,index_col=0)
```


```python
data_2=data_2[data_2['price']/data_2['size']<=200]
print('mean ',data_2.mean())
print('std',data_2.std())
print(data_2.describe())
data_2 = data_2.apply(
    lambda x: (x - x.mean()) / (x.std()))
```

    mean  size         2159.222222
    bedroom         3.305556
    price      343280.416667
    dtype: float64
    std size          820.069409
    bedroom         0.786291
    price      134856.214561
    dtype: float64
                  size    bedroom          price
    count    36.000000  36.000000      36.000000
    mean   2159.222222   3.305556  343280.416667
    std     820.069409   0.786291  134856.214561
    min    1000.000000   1.000000  169900.000000
    25%    1576.500000   3.000000  242800.000000
    50%    1973.500000   3.000000  299900.000000
    75%    2536.250000   4.000000  389225.000000
    max    4478.000000   5.000000  699900.000000
    


```python
data_2.describe()
```




<div>

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
      <th>count</th>
      <td>3.600000e+01</td>
      <td>3.600000e+01</td>
      <td>3.600000e+01</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.551115e-17</td>
      <td>1.541976e-16</td>
      <td>-1.511137e-16</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.413566e+00</td>
      <td>-2.932190e+00</td>
      <td>-1.285669e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-7.105767e-01</td>
      <td>-3.886035e-01</td>
      <td>-7.450930e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-2.264713e-01</td>
      <td>-3.886035e-01</td>
      <td>-3.216790e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.597511e-01</td>
      <td>8.831898e-01</td>
      <td>3.406931e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.827538e+00</td>
      <td>2.154983e+00</td>
      <td>2.644443e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
n_train=30
train_features = nd.array(data_2[['size','bedroom']][:n_train].values)
test_features = nd.array(data_2[['size','bedroom']][n_train:].values)
train_labels = nd.array(data_2.price[:n_train].values).reshape((-1, 1))
test_labels = nd.array(data_2.price[n_train:].values).reshape((-1, 1))

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
batch_size=2
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
num_epochs = 30
for epoch in range(1, num_epochs + 1):
    for X, y in train_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(train_features), train_labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
```

    epoch 1, loss: 0.232526
    epoch 2, loss: 0.144509
    epoch 3, loss: 0.109284
    epoch 4, loss: 0.093431
    epoch 5, loss: 0.084500
    epoch 6, loss: 0.078883
    epoch 7, loss: 0.075394
    epoch 8, loss: 0.073003
    epoch 9, loss: 0.071639
    epoch 10, loss: 0.070587
    epoch 11, loss: 0.069954
    epoch 12, loss: 0.069541
    epoch 13, loss: 0.069291
    epoch 14, loss: 0.069018
    epoch 15, loss: 0.068918
    epoch 16, loss: 0.068819
    epoch 17, loss: 0.068817
    epoch 18, loss: 0.068793
    epoch 19, loss: 0.068748
    epoch 20, loss: 0.068814
    epoch 21, loss: 0.068788
    epoch 22, loss: 0.068739
    epoch 23, loss: 0.068802
    epoch 24, loss: 0.068879
    epoch 25, loss: 0.068780
    epoch 26, loss: 0.068829
    epoch 27, loss: 0.068711
    epoch 28, loss: 0.068738
    epoch 29, loss: 0.068767
    epoch 30, loss: 0.068742
    


```python
y_predit=net(test_features)
l = loss(y_predit, test_labels)
print(l.mean().asnumpy())
```

    [0.18438375]
    


```python
res=net(nd.array([[(1650-2159.222222)/820.069409,(3-3.305556)/0.786291]]))
```


```python
res*134856.214561+343280.416667-293081
```




    
    [[-27316.25]]
    <NDArray 1x1 @cpu(0)>



妥妥的过拟合了，这个我还是可以看得出来的，测试集出来的结果跟训练集的相差太远了。


```python
n_train=32
train_features = nd.array(data_2[['size','bedroom']][:n_train].values)
test_features = nd.array(data_2[['size','bedroom']][n_train:].values)
train_labels = nd.array(data_2.price[:n_train].values).reshape((-1, 1))
test_labels = nd.array(data_2.price[n_train:].values).reshape((-1, 1))

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
batch_size=2
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
num_epochs = 30
for epoch in range(1, num_epochs + 1):
    for X, y in train_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(train_features), train_labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
```

    epoch 1, loss: 0.208121
    epoch 2, loss: 0.130775
    epoch 3, loss: 0.107976
    epoch 4, loss: 0.097792
    epoch 5, loss: 0.091314
    epoch 6, loss: 0.087608
    epoch 7, loss: 0.085346
    epoch 8, loss: 0.084056
    epoch 9, loss: 0.083329
    epoch 10, loss: 0.082728
    epoch 11, loss: 0.082381
    epoch 12, loss: 0.082370
    epoch 13, loss: 0.082308
    epoch 14, loss: 0.081970
    epoch 15, loss: 0.081898
    epoch 16, loss: 0.081902
    epoch 17, loss: 0.081832
    epoch 18, loss: 0.081776
    epoch 19, loss: 0.081799
    epoch 20, loss: 0.081761
    epoch 21, loss: 0.081896
    epoch 22, loss: 0.081773
    epoch 23, loss: 0.081761
    epoch 24, loss: 0.081749
    epoch 25, loss: 0.081772
    epoch 26, loss: 0.082074
    epoch 27, loss: 0.081745
    epoch 28, loss: 0.081766
    epoch 29, loss: 0.081863
    epoch 30, loss: 0.081754
    


```python
y_predit=net(test_features)
l = loss(y_predit, test_labels)
print(l.mean().asnumpy())
```

    [0.07616428]
    


```python
如果训练集有32个，就不过拟合了，嗯，数据还是太少，模型不是很稳定。
```
<p align="center">
<img src="https://raw.githubusercontent.com/linguoguo/linguo_zh/master/img/end.png" width="200" >
</p>


此处围观我的[知乎博客](https://zhuanlan.zhihu.com/p/150142720)，这里[下载](https://github.com/linguoguo/data_science/blob/master/house_pricing/regression_house_2_features.ipynb)本文代码
