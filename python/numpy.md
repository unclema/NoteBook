# 一、导入

## 1.1 导入

```python
import numpy as np
```

## 1.2 查看版本

```python
np.__version__
```

# 二、创建

## 2.1通过列表创建

**关键字：**`np.array`

### 2.1.1 创建一维数组

```python
np.array[(1, 2, 3)]
```

### 2.1.2 创建二维数组

```python
np.array[(1, 2, 3),(4, 5, 6)]
```

**设置数组类型**

使用`dtype`关键字

```python
x = np.array([1, 2, 3, 4, 5], dtype="float32")
```



## 2.2从头创建

### 2.2.1 固定元素

#### 全0 `np.zeros`

**创建长度为5的数组，值为0，类型为int**

```python
np.zeros(5, dtype=int)
```

**创建三行四列的二维数组**

```python
np.zeros((3,4))
```

#### 全1 `np.ones`

创建一个两行四列的浮点型数组，值都为1

```python
np.ones((2, 4), dtype=float)
```

#### 全某值`np.full`

**创建一个3行5列的数组，值都为8.8**

```python
np.full((3, 5), 8.8)
```

#### 单位矩阵`np.eye()`

**创建一个3\*3的单位矩阵**

```python
np.eye(3)
```

### 2.2.2 序列

#### 等差数列`np.arange`

**创建一个线性序列数组，从1开始，到15结束，步长为2**

```python
np.arange(1, 15, 2)
```

**创建二维等差数组：**

```python
np.arange(6).reshape(2, 3)
```

#### 均匀分布数列`np.linspace`

**0~1均匀分布为四份**

```python
np.linspace(0, 1, 4) # 分为四分
```

#### 等比数列`np.logspace`

**1到 $10^{9}$等比分为四份 **

```python
np.logspace(0, 9, 10)
```




### 2.2.3 随机

#### 0~1间所及构成数组`np.random.random()`

```python
np.random.random((3,3)) # 创建一个3*3的，在0~1之间均匀分布的随机数构成的数组
```

#### 创建小于某个值的随机数组`np.random.randint()`

```python
np.random.randint(3, 5, size=(2, 3)) # 创建二维随机整数数组（数值在3和5之间
```

#### 创建指定均值和标准差的正态分布数组`np.random.normal()`

```python
np.random.normal(0, 1, (3,3)) # 创建一个3*3的，均值为0，标准差为1的正态分布随机数构成的数组
```

#### 随机重排列`np.random.permutation`(不修改原始值)

```python
x = np.array([10, 20, 30, 40])
y = np.random.permutation(x)       # 生产新列表
print(x,y)
```

#### 随机重排列`np.random.shuffle`(修改原始值)

```python
print(x)
np.random.shuffle(x)          # 修改原列表
print(x)
```

#### 随机采样

* 按指定形状采样`np.random.choice)`

```python
x = np.arange(10, 25, dtype = float)
np.random.choice(x, size=(4, 3))
```

* 按概率随机采样

```python
np.random.choice(x, size=(4, 3), p=x/np.sum(x))
```



# 三、性质

## 3.1 属性

### 3.1.1数组的形状shape

```python
x.shape
```

### 3.1.2数组的维度ndim

```python
x.ndim
```

### 3.1.3 数组的大小size

```python
x.size
```

### 3.1.4数组的数据类型dtype

```python
x.dtype
```

## 3.2 索引

### 3.2.1一维索引

```python
x[index] # 类似数组
```

### 3.2.2 多维索引

```python
x[index1][index2] # 1行2列
```

**根据索引可以查看，也可以修改，但是一个numpy的数组的数据类型是固定的，向一个整型数组插入一个浮点值，浮点值会向下取整**

## 3.3 切片

### 3.3.1 一维切片

类似于列表

```python
x1[:3] 			# 从开始到第三个
x1[3:] 			# 从第三个到最后
x1[::-1] 		# 反向
```

### 3.3.2 多维切片--二维为例

```python
x2[:2, :3]            	# 前两行，前三列
x2[:2, 0:3:2]       	# 前两行 前三列（每隔一列）
x2[::-1, ::-1] 			# 斜着掉个
```

### 3.3.3 获取行或者列

```python
x3[1, :]   #第一行  从0开始计数
x3[1]    # 第一行简写
x3[:, 2]  # 第二列   从0开始计数
```

### 3.3.4 切片修改

切片切出来的是视图，不是副本，切片元素修改，原数组也会修改

**修改切片的安全方式：copy**

```python
x6 = x4[:2, :2].copy()
```

此时修改x6不会影响x4

## 3.4变形

### 3.4.1 `reshape` `resize`

```python
x6 = x5.reshape(3, 4) # 把x5变为三行四列的矩阵
```

**变形出来的是视图，不是副本，如果x6改变，则x5也改变。但是reshape不会改变原有的形状**

```python
a.resize(2, 3)  # resize 会改变原始数组
```



### 3.4.2 一维向量转行向量

```python
x7 = x5.reshape(1, x5.shape[0])
# 或
x8 = x5[np.newaxis, :]
```

### 3.4.3 一维向量转列向量

```python
x7 = x5.reshape(x5.sape[0] , 1)
# 或
x8 = x5[:, np.newaxis]
```

### 3.4.5 多维向量转一维向量

#### 返回副本`flatten`

```python
x9 = x6.flatten()
```

对返回值修改不影响原值

#### 返回视图`ravel`

```python
x10 = x6.ravel()
```

修改影响原值

#### 返回视图`reshape`

```python
x11 = x6.reshape(-1)
```

## 3.5拼接

### 3.5.1 水平拼接 `np.hstack` `np.c_`

```python
x3 = np.hstack([x1, x2])
x4 = np.c_[x1, x2]
```

**修改后的值不影响原值**

### 3.5.2 垂直拼接`np.vstack` `np.r_`

```pyth
x5 = np.vstack([x1, x2])
x6 = np.r_[x1, x2]
```

## 3.6分裂

### 3.6.1 `split`分元素

```python
x1, x2, x3 = np.split(x6, [2, 7]) # 从第二个元素和第七个元素开始分裂
```

### 3.6.2 `hsplit`分列

```python
left, middle, right = np.hsplit(x7, [2,4]) # 从第二列和第四列开始分割
```

### 3.6.3 `vsplit`分行

```python
upper, middle, lower = np.vsplit(x7, [2,4]) # 从第二行和第四行开始分割
```



# 四、四大运算
## 4.1向量化

### 4.1.1 与数字加减乘除

```python
x1 = np.arange(1,6)
print("x1+5", x1+5)
print("x1-5", x1-5)
print("x1*5", x1*5)
print("x1/5", x1/5)
print("-x1", -x1)
print("x1**2", x1**2)
print("x1//2", x1//2)
print("x1%2", x1%2)
```

```
array([1, 2, 3, 4, 5])
x1+5 [ 6  7  8  9 10]
x1-5 [-4 -3 -2 -1  0]
x1*5 [ 5 10 15 20 25]
x1/5 [0.2 0.4 0.6 0.8 1. ]
-x1 [-1 -2 -3 -4 -5]
x1**2 [ 1  4  9 16 25]
x1//2 [0 1 1 2 2]
x1%2 [1 0 1 0 1]
```

### 4.1.2 绝对值、三角函数、指数、对数

#### 绝对值`abs`

```python
x2 = np.array([1, -1, 2, -2, 0])
abs(x2)
# 或
np.abs(x2)
```

```
array([ 1, -1,  2, -2,  0])
array([1, 1, 2, 2, 0])
```

#### 三角函数

```python
theta = np.linspace(0, np.pi, 3)
print("sin(theta)", np.sin(theta))
print("con(theta)", np.cos(theta))
print("tan(theta)", np.tan(theta))
x = [1, 0 ,-1]
print("arcsin(x)", np.arcsin(x))
print("arccon(x)", np.arccos(x))
print("arctan(x)", np.arctan(x))
```

```
array([0.        , 1.57079633, 3.14159265])
sin(theta) [0.0000000e+00 1.0000000e+00 1.2246468e-16]
con(theta) [ 1.000000e+00  6.123234e-17 -1.000000e+00]
tan(theta) [ 0.00000000e+00  1.63312394e+16 -1.22464680e-16]
arcsin(x) [ 1.57079633  0.         -1.57079633]
arccon(x) [0.         1.57079633 3.14159265]
arctan(x) [ 0.78539816  0.         -0.78539816]
```

#### 指数运算

```python
x = np.arange(3)
np.exp(x) # 以自然对数函数为底数的指数函数
```

```
array([0, 1, 2])
array([1.        , 2.71828183, 7.3890561 ])
```

#### 对数运算

```python
x = np.array([1, 2, 4, 8 ,10])
print("ln(x)", np.log(x))
print("log2(x)", np.log2(x))
print("log10(x)", np.log10(x))
```

```
ln(x) [0.         0.69314718 1.38629436 2.07944154 2.30258509]
log2(x) [0.         1.         2.         3.         3.32192809]
log10(x) [0.         0.30103    0.60205999 0.90308999 1.        ]
```

#### 开方立方

```python
np.sqrt(a) # 开方运算
np.power(a, 3) # 立方运算
```



### 4.1.3 两个数组的运算

```python
x1 = np.arange(1,6)
x2 = np.arange(6,11)
print("x1+x2:", x1+x2)
print("x1-x2:", x1-x2)
print("x1*x2:", x1*x2)
print("x1/x2:", x1/x2)
```

## 4.2矩阵化

### 4.2.1 矩阵的转置`x.T`

```python
y = x.T
```

### 4.2.2 矩阵求逆`np.linalg.inv(A)`

```python
np.linalg.inv(A)
```



### 4.2.3 矩阵乘法`x.dot(y)`

```python
x.dot(y)
np.dot(x, y)
y.dot(x)
np.dot(y, x)
```

### 4.2.4 数组填充`np.pad()`

```python
np.pad(Z, pad_width=1, mode='constant', constant_values=0)
```

函数结构：

```python
pad(array, pad_width, mode, **kwargs)

# 返回值：数组
```

- array 需要填充的数组

- pad_width 每个轴边缘需要填充的数值数目

  参数输入方式为：（(before_1, after_1), … (before_N, after_N)）

  其中(before_1, after_1)表示第1轴两边缘分别填充before_1个和after_1个数值。

  取值为：{sequence, array_like, int}

- mode：表示填充的方式（取值：str字符串或用户提供的函数）,总共有11种填充模式；

  `constant`——表示连续填充相同的值，每个轴可以分别指定填充值，constant_values=（x, y）时前面用x填充，后面用y填充，缺省值填充0

  `edge`——表示用边缘值填充

  `linear_ramp`——表示用边缘递减的方式填充

  `maximum`——表示最大值填充

  `mean`——表示均值填充

  `median`——表示中位数填充

  `minimum`——表示最小值填充

  `reflect`——表示对称填充

  `symmetric`——表示对称填充

  `wrap`——表示用原数组后面的值填充前面，前面的值填充后面

**举例**

```python
Z = np.ones((5, 5))
```

```
array([[1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]])
```

```python
Z = np.pad(Z, pad_width=((1,0),(2,0)), mode='constant', constant_values=0)
'''
主要要理解pad_width这个参数的意义
首先括号内的括号意思是轴
在二维数组里 0横轴 1纵轴
（1，0）的意思是在z数组上方横轴加1行，下方横轴加0行
（2，0）的意思是在z数组纵轴左方加2列，右方纵轴加0列
'''
```

```
array([[0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 1., 1., 1., 1.],
       [0., 0., 1., 1., 1., 1., 1.],
       [0., 0., 1., 1., 1., 1., 1.],
       [0., 0., 1., 1., 1., 1., 1.],
       [0., 0., 1., 1., 1., 1., 1.]])
```

```python
Z = np.pad(Z,pad_width = ((2,1), (1,2)) , mode = 'constant', constant_values = 0)
```

```
array([[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 1., 1., 0., 0.],
       [0., 1., 1., 1., 1., 1., 0., 0.],
       [0., 1., 1., 1., 1., 1., 0., 0.],
       [0., 1., 1., 1., 1., 1., 0., 0.],
       [0., 1., 1., 1., 1., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]])
```

更多可参考这篇[博客](https://blog.csdn.net/zenghaitao0128/article/details/78713663)

### 4.2.5 特征值和特征向量 `np.linalg.eig(M)`

```python
M = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
w, v = np.linalg.eig(M)
# w 对应特征值，v 对应特征向量
w, v
```



## 4.3 广播

如果两个数组的形状在维度上不匹配

那么数组的形式会沿着维度为1的维度进行扩展以匹配另一个数组的形状。

## 4.4比较运算与掩码

### 4.4.1 比较运算

```python
x1 = np.random.randint(100, size=(10,10))
x1 > 50
```

```
array([[37, 44, 58, 79,  1, 24, 85, 90, 27, 56],
       [74, 68, 88, 27, 46, 34, 92,  1, 35, 45],
       [84, 80, 83, 72, 98, 15,  4, 77, 14, 98],
       [19, 85, 98, 32, 47, 50, 73,  3, 24,  2],
       [ 5, 28, 26, 31, 48, 43, 72, 73, 53, 64],
       [81, 87, 56, 59, 24, 42, 84, 34, 97, 65],
       [74,  9, 41, 54, 78, 62, 53, 49,  8, 70],
       [63, 44, 33, 35, 26, 83,  7, 14, 65, 84],
       [57, 10, 62,  8, 74, 47, 90, 25, 78, 48],
       [36, 31, 45, 39, 66, 82, 42, 25, 33, 84]])
 array([[False, False,  True,  True, False, False,  True,  True, False,
         True],
       [ True,  True,  True, False, False, False,  True, False, False,
        False],
       [ True,  True,  True,  True,  True, False, False,  True, False,
         True],
       [False,  True,  True, False, False, False,  True, False, False,
        False],
       [False, False, False, False, False, False,  True,  True,  True,
         True],
       [ True,  True,  True,  True, False, False,  True, False,  True,
         True],
       [ True, False, False,  True,  True,  True,  True, False, False,
         True],
       [ True, False, False, False, False,  True, False, False,  True,
         True],
       [ True, False,  True, False,  True, False,  True, False,  True,
        False],
       [False, False, False, False,  True,  True, False, False, False,
         True]])      
```

### 4.4.2 操作布尔数组

```python
x2 = np.random.randint(10, size=(3, 4))
print(x2 > 5) # 输出x大于5的布尔值
np.sum(x2 > 5) # 所有大于5的元素相加
np.all(x2 > 0) # 全大于0
np.any(x2 == 6) # 至少一个等于6
np.all(x2 < 9, axis=1)   # 按行进行判断
(x2 < 9) & (x2 >5) # 小于9且大于5的元素
np.sum((x2 < 9) & (x2 >5)) # 小于9且大于5的元素相加
x2[x2 > 5] # 挑选x中x大于5的元素
```



## 4.5花哨的索引

### 4.5.1 一维数组

```python
ind = [2, 6, 9]
x[ind]
ind = np.array([[1, 0],
               [2, 3]])
x[ind]
```

```
array([67, 55, 82])
array([[69, 43],
       [67,  9]])
```

**注意：结果的形状与索引数组ind一致**



### 4.5.2 多维数组

```python
x = np.arange(12).reshape(3, 4)

row = np.array([0, 1, 2])
col = np.array([1, 3, 0])
x[row, col]               # x(0, 1) x(1, 3) x(2, 0)

row[:, np.newaxis]       # 变为列向量 
x[row[:, np.newaxis], col]    # 广播机制
'''
0,1 0,3 0,0
1,1 1,3 1,0
2,1 2,3 2,0
'''
```

```
x 数组
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
第一种筛选
array([1, 7, 8])

第二种筛选
array([[ 1,  3,  0],
       [ 5,  7,  4],
       [ 9, 11,  8]])
```




# 五、通用函数
## 5.1排序

### 5.1.1 产生新的排序数组`np.sort(x)`

```python
np.sort(x)
```

### 5.1.2 替换原数组`x.sort()`

```python
x.sort()
```



### 5.1.3 获得排序索引`i = np.argsort(x)`

```python
i = np.argsort(x)
```

这个获取的数组中每个元素的排名

## 5.2最值

### 5.2.1 获取最值

```python
print("max:", np.max(x))
np.max(a, axis=0) # 获取每列最大值
print("min:", np.min(x))
```

### 5.2.2 获取最值索引

```python
print("max_index:", np.argmax(x))
print("min_index:", np.argmin(x))
```

## 5.3求和积

### 5.3.1 按行`np.sum(x1, axis=1)`

```python
np.sum(x1, axis=1)
```



### 5.3.2 按列`np.sum(x1, axis=0)`
```python
np.sum(x1, axis=0)
```


### 5.3.3 全体`np.sum(x1)`
```python
np.sum(x1)
```

求积只要改成`x.prod()`或`np.prod(x)`即可

## 5.4统计相关

### 5.4.1 中位数`np.median(x)`

```python
np.median(x)
```

### 5.4.2 均值`x.mean()` `np.mean(x)`

```python
x.mean()
np.mean(x)
```

### 5.4.3 方差`x.var()` `np.var(x)`
```python
np.median(x)
np.mean(x)
```


### 5.4.4 标准差`x.std()` `np.std(x)`
```python
x.std()
np.std(x)
```
