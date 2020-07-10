[toc]



# 一、特点

Numpy适合于向量化的数值计算

但是在处理更灵活、复杂的数据任务：如为数据添加标签、处理缺失值、分组和透视表等方面，numpy就有点不够用了。

**Pandas基于Numpy，提供了使得数据分析变得更快更简单的高级数据结构和操作工具**



# 二、对象创建

## 2.1 一维series

Series 是带标签数据的一维数组

通用结构: pd.Series(data, index=index, dtype=dtype)

data：数据，可以是列表，字典或Numpy数组

index：索引，为可选参数

dtype: 数据类型，为可选参数

### 2.1.1 用列表创建

#### 缺省index

默认为整数序列

```python
import pandas as pd

data = pd.Series([1.5, 3, 4.5, 6])
data
```

```
0    1.5
1    3.0
2    4.5
3    6.0
dtype: float64
```

#### 指定index

```python
data = pd.Series([1.5, 3, 4.5, 6], index=["a", "b", "c", "d"])
data
```

```
a    1.5
b    3.0
c    4.5
d    6.0
dtype: float64

```

#### 增加数据类型

如果缺省则自动判断

```python
data = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"], dtype="float")
data
```

```
a    1.0
b    2.0
c    3.0
d    4.0
dtype: float64
```

**data内的数据可以多种类型一起**

```python
data = pd.Series([1, 2, "3", 4], index=["a", "b", "c", "d"])
data
```

```
a    1
b    2
c    3
d    4
dtype: object
```

**数据类型可被强制改变**

```python
data = pd.Series([1, 2, "3", 4], index=["a", "b", "c", "d"], dtype=float)
data
```

### 2.1.2 用一维numpy数组创建

```python
import numpy as np

x = np.arange(5)
pd.Series(x)
```

### 2.1.3 用字典创建

```python
population_dict = {"BeiJing": 2154,
                   "ShangHai": 2424,
                   "ShenZhen": 1303,
                   "HangZhou": 981 }
population = pd.Series(population_dict)    
population
```

```
BeiJing     2154
ShangHai    2424
ShenZhen    1303
HangZhou     981
dtype: int64
```

#### 指定index

如果指定index的话会到字典的键中筛选，如果找不到，值设为NaN

```python
population = pd.Series(population_dict, index=["BeiJing", "HangZhou", "c", "d"])    
population
```

```
BeiJing     2154.0
HangZhou     981.0
c              NaN
d              NaN
dtype: float64
```

### 2.1.4 data为标量的情况

```python
pd.Series(5, index=[100, 200, 300])
```

```
100    5
200    5
300    5
dtype: int64
```

## 2.2 多维DataFrame

DataFrame 是带标签数据的多维数组

**DataFrame对象的创建**

通用结构: pd.DataFrame(data, index=index, columns=columns)

data：数据，可以是列表，字典或Numpy数组

index：索引，为可选参数

columns: 列标签，为可选参数

### 2.2.1 通过Series对象创建

```python
population_dict = {"BeiJing": 2154,
                   "ShangHai": 2424,
                   "ShenZhen": 1303,
                   "HangZhou": 981 }

population = pd.Series(population_dict)    
pd.DataFrame(population)
```

```
			0
BeiJing		2154
ShangHai	2424
ShenZhen	1303
HangZhou	981
```

```python
pd.DataFrame(population, columns=["population"])
```

```
			population
BeiJing		2154
ShangHai	2424
ShenZhen	1303
HangZhou	981
```



### 2.2.2 通过Series对象字典创建

```python
GDP_dict = {"BeiJing": 30320,
            "ShangHai": 32680,
            "ShenZhen": 24222,
            "HangZhou": 13468 }
population_dict = {"BeiJing": 2154,
                   "ShangHai": 2424,
                   "ShenZhen": 1303,
                   "HangZhou": 981 }
population = pd.Series(population_dict)  # population变为了Series对象
GDP = pd.Series(GDP_dict) # GDP变为了Series对象
pd.DataFrame({"population": population,
              "GDP": GDP}) # 给两个series对象分别赋予不同的列标签
```

```
			population	GDP
BeiJing		2154		30320
ShangHai	2424		32680
ShenZhen	1303		24222
HangZhou	981			13468
```

```python
## 如果数量不够会自动补齐
pd.DataFrame({"population": population,
              "GDP": GDP,
              "country": "China"})
```

```
			population		GDP			country
BeiJing		2154			30320		China
ShangHai	2424			32680		China
ShenZhen	1303			24222		China
HangZhou	981				13468		China
```



### 2.2.3 通过字典列表对象创建

字典索引作为index值，字典的键作为columns

```python
import numpy as np
import pandas as pd

data = [{"a": i, "b": 2*i} for i in range(3)]
data
```

```
[{'a': 0, 'b': 0}, {'a': 1, 'b': 2}, {'a': 2, 'b': 4}]
```

```python
data = pd.DataFrame(data)
```

```
	a	b
0	0	0
1	1	2
2	2	4
```

如果又不存在的键，则会被赋值为NaN

```python
data = [{"a": 1, "b":1},{"b": 3, "c":4}]
pd.DataFrame(data)
```

```
	a		b		c
0	1.0		1		NaN
1	NaN		3		4.0
```

### 2.2.4 通过Numpy二维数组创建

```python
data = np.random.randint(10, size=(3, 2))
pd.DataFrame(data, columns=["foo", "bar"], index=["a", "b", "c"])
```

```

	foo		bar
a	1		6
b	2		9
c	4		0
```



# 三、DataFrame性质

## 3.1 属性
```python
data = pd.DataFrame({"pop": population, "GDP": GDP})
data
```
```
			pop		GDP
BeiJing		2154	30320
ShangHai	2424	32680
ShenZhen	1303	24222
HangZhou	981		13468
```

### 3.1.1 返回numpy数组表示得数据`df.values`

```python
data.values
```

```
array([[ 2154, 30320],
       [ 2424, 32680],
       [ 1303, 24222],
       [  981, 13468]], dtype=int64)
```

### 3.1.2 返回行索引`df.index`
```python
data.index
```

```
Index(['BeiJing', 'ShangHai', 'ShenZhen', 'HangZhou'], dtype='object')
```

### 3.1.3 返回列索引`df.columns`

```python
data.columns
```

```
Index(['pop', 'GDP'], dtype='object')
```

### 3.1.4 查看形状`df.shape`

```python
data.shape
```

```
(4, 2)
```

### 3.1.5 查看大小`pd.size`

```python
data.size
```

```
8
```

### 3.1.6 返回每列数据类型`pd.dtypes`

```python
data.dtypes
```

```
pop    int64
GDP    int64
dtype: object
```


## 3.2 索引

### 3.2.1 获取列

两种方式可以获取列

#### 字典式获取

```python
data["pop"] # 取一列
data[["GDP","pop"]] # 取多列
```

#### 对象式获取

```python
data.GDP
```

### 3.2.2 获取行

有两种可以获取行元素的方法

#### 绝对索引`df.loc`

```python
data.loc["BeiJing"] # 单行
data.loc[["BeiJing", "HangZhou"]] # 多行
```

#### 相对索引`df.iloc`

```python
data.iloc[0]
data.iloc[[1, 3]]
```

### 3.2.3 获取标量

有四种方法可以获取

#### 绝对索引方法获取

```python
data.loc["BeiJing", "GDP"]
```

#### 相对索引方法获取

```python
data.iloc[0, 1]
```

#### 取值获取

```python
data.values[0][1]
```

#### 对象方法获取

```python
data.pop.beijing
```

### 3.2.4 series对象的索引

```python
GDP["BeiJing"]
```

## 3.3 切片

```python
# 创建一个data
dates = pd.date_range(start='2019-01-01', periods=6) # 创建一个日期索引
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=["A", "B", "C", "D"]) # 创建二维数组
```

```
				A			B			C			D
2019-01-01	0.414831	0.458100	-0.821264	-1.185715
2019-01-02	1.294424	0.599704	1.012147	0.789749
2019-01-03	-0.179173	-1.106951	-1.377899	0.213107
2019-01-04	-1.601717	-1.203744	-1.232206	-1.538080
2019-01-05	-1.165177	-1.138257	0.647488	-0.083231
2019-01-06	-0.580356	-0.975909	1.496404	-0.399852
```

### 3.3.1 行切片

共三种方式

#### 类似与列表

```python
df["2019-01-01": "2019-01-03"]
```

#### 相对索引切片`df.loc`

```python
df.loc["2019-01-01": "2019-01-03"]
```

#### 绝对索引切片`df.iloc`

```python
df.iloc[0: 3]
```

### 3.3.2 列切片

两种方式

#### 相对索引切片`df.loc`

```python
df.loc[:, "A": "C"]
```

#### 绝对索引切片`df.iloc`

```python
df.iloc[:, 0: 3]
```

### 3.3.3  行列同时切片

#### 相对索引切片`df.loc`

```python
df.loc["2019-01-02": "2019-01-03", "C":"D"]
```

#### 绝对索引切片`df.iloc`

```python
df.iloc[1: 3, 2:]
```



### 3.3.4 行切片列分散
#### 相对索引切片`df.loc`
```python
df.loc["2019-01-04": "2019-01-06", ["A", "C"]]
```
#### 绝对索引切片`df.iloc`
```python
df.iloc[3:, [0, 2]]
```
### 3.3.5 列切片行分散
#### 相对索引切片`df.loc`
```python
df.loc[["2019-01-02", "2019-01-06"], "C": "D"]
```
#### 绝对索引切片`df.iloc`
```python
df.iloc[[1, 5], 0: 3]
```

### 3.3.6 行列均分散
#### 相对索引切片`df.loc`
```python
df.loc[["2019-01-04", "2019-01-06"], ["A", "D"]]
```
#### 绝对索引切片`df.iloc`
```python
df.iloc[[1, 5], [0, 3]]
```

## 3.4 布尔索引

### 3.4.1 整表布尔值

```python
df > 0
```

```
			A		B		C		D
2019-01-01	False	False	True	False
2019-01-02	False	False	True	False
2019-01-03	False	True	True	True
2019-01-04	True	True	False	True
2019-01-05	True	True	True	False
2019-01-06	True	False	False	False
```



### 3.4.1 取整表符合条件的值

```python
df[df > 0]
```

```
			A			B			C			D
2019-01-01	NaN			NaN			0.925984	NaN
2019-01-02	NaN			NaN			1.080779	NaN
2019-01-03	NaN			0.058118	1.102248	1.207726
2019-01-04	0.305088	0.535920	NaN			0.177251
2019-01-05	0.313383	0.234041	0.163155	NaN
2019-01-06	0.250613	NaN			NaN			NaN
```



### 3.4.2 列布尔值

```python
df.A > 0
```

```
2019-01-01    False
2019-01-02    False
2019-01-03    False
2019-01-04     True
2019-01-05     True
2019-01-06     True
Freq: D, Name: A, dtype: bool
```

### 3.4.2 取某列满足条件的值

```python
df[df.A > 0]
```

```
			A			B			C			D
2019-01-04	0.305088	0.535920	-0.978434	0.177251
2019-01-05	0.313383	0.234041	0.163155	-0.296649
2019-01-06	0.250613	-0.904400	-0.858240	-1.573342
```



### 3.4.2 判断是否包含`isin`

```python
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
ind = df2["E"].isin(["two", "four"])
ind
```

```
2019-01-01    False
2019-01-02    False
2019-01-03     True
2019-01-04    False
2019-01-05     True
2019-01-06    False
Freq: D, Name: E, dtype: bool
```

```python
df2[ind]
```

```
			A			B			C			D			E
2019-01-03	-0.141572	0.058118	1.102248	1.207726	two
2019-01-05	0.313383	0.234041	0.163155	-0.296649	four
```

## 3.5 赋值

### 3.5.1 DataFrame新增列

```python
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20190101', periods=6))
df["E"] = s1  # 为df中新增了列标为E的列
df["F"] = np.nan # 新增列赋值为nan
```



### 3.5.2 修改赋值

```python
df.loc["2019-01-01", "A"] = 0
df.iloc[0, 1] = 0
df["D"] = np.array([5]*len(df))   # 可简化成df["D"] = 5 # 整列赋值
```

### 3.5.3 修改index和columns

```python
df.index = [i for i in range(len(df))] # 必须数量要一致
df.columns = [i for i in range(df.shape[1])]
```



# 四、数值运算及统计分析

```python
import pandas as pd
import numpy as np

dates = pd.date_range(start='2019-01-01', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=["A", "B", "C", "D"])
df
```

## 4.1 数据查看

### 4.1.1 查看前面的行`df.head()`

```python
df.head()    # 默认5行
df.head(2)
```

### 4.1.2 查看后面的行`df.tail()`

```python
df.tail()    # 默认5行
df.tail(3) 
```

### 4.1.3 查看总体信息`df.info`

```python
df.info()
```



## 4.2 通用函数

类似于numpy

```python
x = pd.DataFrame(np.arange(4).reshape(1, 4))
x
```

```
	0	1	2	3
0	0	1	2	3
```

### 4.2.1 向量化运算

```python
x+5 # 加
np.exp(x) # 求指
x*y # 和另一个数组相乘
```

### 4.2.2 矩阵化运算

```python
z = x.T # 转置
x.dot(y) # 矩阵乘法
np.dot(x, y) #矩阵乘法 数据量大的时候更快
```



### 4.2.3 广播运算

其实所谓的广播运算就是自动填充

```python
np.random.seed(42)
x = pd.DataFrame(np.random.randint(10, size=(3, 3)), columns=list("ABC"))
x
```

```
	A	B	C
0	6	3	7
1	4	6	9
2	2	6	7
```

```python
x.iloc[0]
```

```
A    6
B    3
C    7
Name: 0, dtype: int32
```

```python
x/x.iloc[0]
```

```
A	B			C
0	1.000000	1.0	1.000000
1	0.666667	2.0	1.285714
2	0.333333	2.0	1.000000
```

相当于

```
	A		B		C
0	6/6		3/3		7/7
1	4/6		6/3		9/7
2	2/6		6/3		7/7
```

```python
x.A
```

```
0    6
1    4
2    2
Name: A, dtype: int32
```

```python
x.div(x.A, axis=0) # 按列进行相除
```

```
	A		B		C
0	1.0		0.5		1.166667
1	1.0		1.5		2.250000
2	1.0		3.0		3.500000
```



## 4.3 统计相关

### 4.3.1 种类`Counter()`

统计函数需要引入一个新的包`collections`

```python
from collections import Counter
y = np.random.randint(3 , size = 20)
np.unique(y) # 去重
Counter(y) # 拥挤每个出现的次数
# 当然，也可以统计某列中元素出现的次数
y["A"].value_counts() 
```



### 4.3.2 排序

#### 根据值排序`sort_values`

```python
city_info.sort_values(by="per_GDP") # 根据per_GDP列递增排序
city_info.sort_values(by="per_GDP", ascending=False) # # 根据per_GDP列递减排序
```

#### 根据轴排序`sort_index()`

```python
data.sort_index(axis=1) # 根据行标进行排序
data.sort_index(axis=1, ascending=False) # 根据列标进行排序
```



### 4.3.3 非空、求和、最值

```python
df.count() # 非空个数
df.sum() # 求和
df.sum(axis=1) # 按行求和
df.min() # 列最小值
df.max(axis=1) # 行最大值 
df.idxmax() # 返回最大值的索引
```



### 4.3.4 均值、方差、标准差

```python
df.mean() # 均值
df.var() # 方差
df.std() # 标准差
```



### 4.3.5 中位数、众数、x%位数

```python
df.median() # 中位数
data.mode() # 众数
df.quantile(0.75) # 75%分位数
df.describe() # 显示所有统计
```



### 4.3.6 相关性系数、协方差

```python
df.corr() # 查看相关性系数
df.corrwith(df["A"]) # 与A列的相关性系数

```

### 4.3.8 apply()自定义输出

apply（method）的用法：使用method方法默认对每一列进行相应的操作

```python
df.apply(np.cumsum) # 逐行累加
df.apply(np.cumsum, axis=1) # 逐列累加
df.apply(sum) # 计算每一列的和
df.apply(lambda x: x.max()-x.min()) # 每一列的最大值减最小值
df.apply(my_describe) # 每一列都运行同一个自定函数
```

## 4.4 新用法

### 4.4.1 索引对齐

pandas可以根据列索引自动进行处理

```python
A = pd.DataFrame(np.random.randint(0, 20, size=(2, 2)), columns=list("AB"))
B = pd.DataFrame(np.random.randint(0, 10, size=(3, 3)), columns=list("ABC"))
print(A)
print(B)
```

```
	A	B
0	3	7
1	2	1

	A	B	C
0	7	5	1
1	4	0	9
2	5	8	0
```

```python
A+B
```

```
	A		B		C
0	10.0	12.0	NaN
1	6.0		1.0		NaN
2	NaN		NaN		NaN
```

**缺失值也可以用用`fill_value`填充**

```python
A.add(B, fill_value=0)
```

```
	A		B		C
0	10.0	12.0	1.0
1	6.0		1.0		9.0
2	5.0		8.0		0.0
```



# 五、处理缺失值

## 5.1 发现缺失值

**有None、字符串等，数据类型全部变为object，它比int和float更消耗资源**

np.nan为浮点型

`isnull`和`notnull`判断是否有缺失

```python
data.isnull()
data.notnull()
```

## 5.2 删除缺失值

```python
data.dropna() # 删除又缺失值的行
data.dropna(axis="columns") # 删除有缺失值的列
data.dropna(axis="columns", how="all") # 删除全部为缺失
data.dropna(axis="columns", how="any") # 只要有缺失就删除
data.dropna(how="all")
```

## 5.3 填充缺失值

```python
data.fillna(value=5) # 全填充为同一值
fill = data.mean() # 每列的均值
data.fillna(value=fill) # 用均值填充
fill = data.stack().mean() # 全部数据的平均值
data.fillna(value=fill) # 用均值填充
```



# 六、合并数据

## 6.1 行合并或列合并

```python
pd.concat([df_1, df_2]) # 垂直合并
pd.concat([df_3, df_4], axis=1) # 水平合并
```

## 6.2 索引重排

```python
pd.concat([df_5, df_6],ignore_index=True) # 垂直
pd.concat([df_7, df_8],axis=1, ignore_index=True) # 水平
```

## 6.3 对齐合并

```python
pd.merge(df_9, df_10) # 如果没有合适的列会不进行合并
pd.merge(population, GDP, how="outer") # 并集合并
```



# 七、分组和数据透视表

## 7.1 分组`groupby`

分组里面有个延迟计算的概念，也就是说已经分好组了，但是不返回内容
### 7.1.1 分组基础操作
```python
df.groupby("key") # 将df根据key列的内容进行分组，相同的为同一组。
df.groupby("key").sum() # 返回每组的和
df.groupby("key").mean() # 返回每组的平均值
df.groupby("key")["data2"].sum() # 返回每组，“data2”列的和
#### 分组可以按组进行迭代
for data, group in df.groupby("key"):
    print("{0:5} shape={1}".format(data, group.shape)) # data是组名，group为每组的内容
df.groupby("key")["data1"].describe() # 调用方法

```

分组还有更为复杂的操作

```python
df.groupby("key").aggregate(["min", "median", "max"]) # 返回每列的最大最小及中值
```

```

		data1					data2
		min		median	max		min		median		max
key						
A		0		2.5		5		2		5.0		8
B		1		2.5		4		2		3.0		4
C		2		2.5		3		3		5.5		8
```
### 7.1.2 过滤操作
```python
########过滤
def filter_func(x):
    return x["data2"].std() > 3 #过滤出标准差大于3的值
df.groupby("key").filter(filter_func) # 过滤出标准差大于3
```
### 7.1.3 转换操作
```python
df.groupby("key").transform(lambda x: x-x.mean())
```
```
	key		data1	data2
0	A		0		2
1	B		1		2
2	C		2		8
3	C		3		3
4	B		4		4
5	A		5		8

		data1	data2
0		-2.5	-3.0
1		-1.5	-1.0
2		-0.5	2.5
3		0.5		-2.5
4		1.5		1.0
5		2.5		3.0
```
### 7.1.4 apply（）方法
```python
def norm_by_data2(x):
    x["data1"] /= x["data2"].sum()
    return x
df.groupby("key").apply(norm_by_data2)
```

### 7.1.5 使用列表或者数组进行分组

```python
L = [0, 1, 0, 1, 2, 0]
df.groupby(L).sum() # 按照L进行分组，L为0的是一组，1的为一组，2的为一组
```

### 7.1.6 使用字典进行分组

```python
df2 = df.set_index("key") # 将key作为索引值
mapping = {"A": "first", "B": "constant", "C": "constant"} # 把A叫做first，B和C都叫做constant
df2.groupby(mapping).sum()
```

### 7.1.7 使用任意函数作为索引值

```python
df2.groupby(str.lower).mean()
```

### 7.1.8 可同时进行分组

```python
df2.groupby([str.lower, mapping]).mean()
```



