---
layout:     post
title:      用mxnet的gluon线性回归训练只有两个特征的数据集(2/3)
subtitle:   动手学深度学习课外作业之房价预测之第一次结果分析
date:       2020-06-15
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

书接[上一回](https://linguoguo.github.io/linguo_zh/2020/06/14/houseprice_2/) 我们训练了一个线性回归模型，数据集为有两个特征，46个样本的房价预测。
# 预测结果
怎么算测试集的房价，我昨天脑子秀逗了，果然抄代码一时爽，一直抄代码一直爽，爽到后面的代码都没有看了！午夜梦回，突然想起，我当时是怎么算的损失函数？
我开心地去看看结果，好像有那么一丢丢大了点。


```python
y_predit=net(test_features)
l = loss(y_predit, test_labels)
print(l.mean().asnumpy())
```

    [0.1614004]


# 怎么看损失函数
我都不知道损失函数的取值是多少，知道那么多种损失函数有什么意义？兹 傲娇脸
上网找不到资料就自己看看吧，先看看数据集的取值


```python
data.describe()
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
      <th>count</th>
      <td>4.600000e+01</td>
      <td>4.600000e+01</td>
      <td>4.600000e+01</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-9.171408e-17</td>
      <td>1.339508e-16</td>
      <td>-4.344351e-17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.427098e+00</td>
      <td>-2.827070e+00</td>
      <td>-1.341910e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-7.082178e-01</td>
      <td>-2.261656e-01</td>
      <td>-7.075102e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-1.598774e-01</td>
      <td>-2.261656e-01</td>
      <td>-3.110103e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.560979e-01</td>
      <td>1.074287e+00</td>
      <td>2.359614e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.086597e+00</td>
      <td>2.374739e+00</td>
      <td>2.860989e+00</td>
    </tr>
  </tbody>
</table>
</div>



0.17的损失函数跟1的方差比好像不是很大。。。直到这里，我们的初步看法都是“好像”，“差不多”，“大概”。。。 作为一个未来的大神，怎么可以对自己要求这么低。

我们把结果打印出来看看


```python
for i in range(10):
    print(test_labels[i],y_predit[i])
```

    
    [0.0466327]
    <NDArray 1 @cpu(0)> 
    [0.20947975]
    <NDArray 1 @cpu(0)>
    
    [1.6643525]
    <NDArray 1 @cpu(0)> 
    [2.6858356]
    <NDArray 1 @cpu(0)>
    
    [-0.41330725]
    <NDArray 1 @cpu(0)> 
    [0.24514496]
    <NDArray 1 @cpu(0)>
    
    [0.23298769]
    <NDArray 1 @cpu(0)> 
    [-0.332606]
    <NDArray 1 @cpu(0)>
    
    [-0.07311028]
    <NDArray 1 @cpu(0)> 
    [0.34264287]
    <NDArray 1 @cpu(0)>
    
    [-0.19919726]
    <NDArray 1 @cpu(0)> 
    [0.7266256]
    <NDArray 1 @cpu(0)>
    
    [-0.31814724]
    <NDArray 1 @cpu(0)> 
    [-0.8913743]
    <NDArray 1 @cpu(0)>
    
    [-1.2626102]
    <NDArray 1 @cpu(0)> 
    [-1.297945]
    <NDArray 1 @cpu(0)>
    
    [-0.31101027]
    <NDArray 1 @cpu(0)> 
    [-0.12339576]
    <NDArray 1 @cpu(0)>
    
    [-0.7899822]
    <NDArray 1 @cpu(0)> 
    [-0.8878077]
    <NDArray 1 @cpu(0)>


看前三个对比也差太远了吧，按百分比再算一遍。


```python
for i in range(10):
    print( ((test_labels[i]-y_predit[i])*100/test_labels[i]).asnumpy(),'%')
```

    [-349.2121] %
    [-61.374203] %
    [159.31302] %
    [242.75688] %
    [568.6658] %
    [464.7769] %
    [-180.17665] %
    [-2.7985537] %
    [60.32421] %
    [-12.383257] %


我要用这个数据看一个房子值不值得买，会亏到没裤衩吧。最大差5倍，我自闭了，这个结果肯定是不行的！看看每平方均价


```python
data['price_size']=data['price']/data['size']
data['price_bedroom']=data['price']/data['bedroom']
data.describe()
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
      <th>price_size</th>
      <th>price_bedroom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.600000e+01</td>
      <td>4.600000e+01</td>
      <td>4.600000e+01</td>
      <td>46.000000</td>
      <td>46.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-9.171408e-17</td>
      <td>1.339508e-16</td>
      <td>-4.344351e-17</td>
      <td>2.387465</td>
      <td>0.450725</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>5.833545</td>
      <td>2.856655</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.427098e+00</td>
      <td>-2.827070e+00</td>
      <td>-1.341910e+00</td>
      <td>-3.711958</td>
      <td>-8.442439</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-7.082178e-01</td>
      <td>-2.261656e-01</td>
      <td>-7.075102e-01</td>
      <td>0.341222</td>
      <td>-0.286206</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-1.598774e-01</td>
      <td>-2.261656e-01</td>
      <td>-3.110103e-01</td>
      <td>0.969627</td>
      <td>0.455811</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.560979e-01</td>
      <td>1.074287e+00</td>
      <td>2.359614e-01</td>
      <td>1.594585</td>
      <td>1.826074</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.086597e+00</td>
      <td>2.374739e+00</td>
      <td>2.860989e+00</td>
      <td>32.073789</td>
      <td>4.913015</td>
    </tr>
  </tbody>
</table>
</div>



看每平方价钱，75%小于1.59，然而最大的数去到32，有点大得离谱了，可能这个数据集来自不同地方或者不同类型的房子，也可能有输入错误？我们现在怎么办？吴恩达第六周的课程好像可以给我们答案，我们且看下回分解。

新加入两个特征训练一下，第一次不小心就出现Nan了，果然对学习率很敏感啊，不小心就梯度爆炸了，此处可参考这篇博文。


```python
n_train=36
train_features = nd.array(data[['size','bedroom','price_size','price_bedroom']][:n_train].values)
test_features = nd.array(data[['size','bedroom','price_size','price_bedroom']][n_train:].values)
train_labels = nd.array(data.price[:n_train].values).reshape((-1, 1))
test_labels = nd.array(data.price[n_train:].values).reshape((-1, 1))

net = nn.Sequential()
net.add(nn.Dense(1))

net.initialize(init.Normal(sigma=0.01))

loss = gloss.L2Loss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

batch_size=2
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
num_epochs = 40
for epoch in range(1, num_epochs + 1):
    for X, y in train_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(train_features), train_labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
```

    epoch 1, loss: 2190.526367
    epoch 2, loss: 710725.562500
    epoch 3, loss: 284985280.000000
    epoch 4, loss: 227231154176.000000
    epoch 5, loss: 85445782274048.000000
    epoch 6, loss: 87628233465397248.000000
    epoch 7, loss: 37323010165786542080.000000
    epoch 8, loss: 4764028720072496250880.000000
    epoch 9, loss: 6082304267000688419012608.000000
    epoch 10, loss: 6613091637294114079303008256.000000
    epoch 11, loss: 2311386075767621512855384227840.000000
    epoch 12, loss: 909418451767419360324547522854912.000000
    epoch 13, loss: 287194146490916168776753067186978816.000000
    epoch 14, loss: inf
    epoch 15, loss: inf
    epoch 16, loss: inf
    epoch 17, loss: inf
    epoch 18, loss: inf
    epoch 19, loss: inf
    epoch 20, loss: inf
    epoch 21, loss: inf
    epoch 22, loss: inf
    epoch 23, loss: inf
    epoch 24, loss: inf
    epoch 25, loss: inf
    epoch 26, loss: inf
    epoch 27, loss: nan
    epoch 28, loss: nan
    epoch 29, loss: nan
    epoch 30, loss: nan
    epoch 31, loss: nan
    epoch 32, loss: nan
    epoch 33, loss: nan
    epoch 34, loss: nan
    epoch 35, loss: nan
    epoch 36, loss: nan
    epoch 37, loss: nan
    epoch 38, loss: nan
    epoch 39, loss: nan
    epoch 40, loss: nan



```python
n_train=36
train_features = nd.array(data[['size','bedroom','price_size','price_bedroom']][:n_train].values)
test_features = nd.array(data[['size','bedroom','price_size','price_bedroom']][n_train:].values)
train_labels = nd.array(data.price[:n_train].values).reshape((-1, 1))
test_labels = nd.array(data.price[n_train:].values).reshape((-1, 1))

net = nn.Sequential()
net.add(nn.Dense(1))

net.initialize(init.Normal(sigma=0.01))

loss = gloss.L2Loss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

batch_size=2
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
num_epochs = 40
for epoch in range(1, num_epochs + 1):
    for X, y in train_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(train_features), train_labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
```

    epoch 1, loss: 0.280175
    epoch 2, loss: 0.252596
    epoch 3, loss: 0.247836
    epoch 4, loss: 3.694590
    epoch 5, loss: 0.971740
    epoch 6, loss: 0.563291
    epoch 7, loss: 2.774067
    epoch 8, loss: 0.801260
    epoch 9, loss: 0.179848
    epoch 10, loss: 0.149339
    epoch 11, loss: 1.921999
    epoch 12, loss: 0.128801
    epoch 13, loss: 0.215510
    epoch 14, loss: 0.193499
    epoch 15, loss: 0.514674
    epoch 16, loss: 0.250536
    epoch 17, loss: 0.112822
    epoch 18, loss: 0.170746
    epoch 19, loss: 0.232429
    epoch 20, loss: 0.146247
    epoch 21, loss: 1.763635
    epoch 22, loss: 1.205917
    epoch 23, loss: 0.113654
    epoch 24, loss: 0.110538
    epoch 25, loss: 0.166103
    epoch 26, loss: 0.147857
    epoch 27, loss: 0.178377
    epoch 28, loss: 0.152319
    epoch 29, loss: 0.131106
    epoch 30, loss: 0.399766
    epoch 31, loss: 0.109490
    epoch 32, loss: 0.546786
    epoch 33, loss: 0.150535
    epoch 34, loss: 0.521338
    epoch 35, loss: 2.954722
    epoch 36, loss: 0.106273
    epoch 37, loss: 0.139209
    epoch 38, loss: 0.422830
    epoch 39, loss: 0.115425
    epoch 40, loss: 0.105446


这里看到loss有时会突然变大，可以看出我们已经在最优解左右徘徊，可以了，我们测试一下：


```python
y_predit=net(test_features)
l = loss(y_predit, test_labels)
print(l.mean().asnumpy())
```

    [0.1441034]


# 过拟合还是欠拟合？
测试集的0.144 和训练集的0.105差不是很多，但是结果不算好。直觉告诉我应该是欠拟合，因为这里是偏差比较大的。发现一个写得不错的文章：[神经网络:欠拟合和过拟合](https://www.jianshu.com/p/9b6b0d6d3bd0) 还有一篇关于过度训练的[过拟合详解：监督学习中不准确的「常识」](https://www.jiqizhixin.com/articles/2019-01-25-23)
最后关于欠拟合过拟合这件事，我还是不能只靠直觉，还是要用更专业的方法 [学习曲线——判断欠拟合还是过拟合](https://blog.csdn.net/geduo_feng/article/details/79547554)
### 预告
+ 明天我们再来看看怎么用mxnet写学习曲线。
+ 看其它更好的算法
+ 更多提取特征的方法

<p align="center">
<img src="https://raw.githubusercontent.com/linguoguo/linguo_zh/master/img/end.png" width="200" >
</p>


此处围观我的[知乎博客](https://zhuanlan.zhihu.com/p/148571981)，这里[下载](https://github.com/linguoguo/data_science/blob/master/house_pricing/regression_house_2_features.ipynb)本文代码
