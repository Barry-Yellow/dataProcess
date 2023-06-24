# 处理LOB数据的脚本
import os
import pandas as pd
import numpy as np

"""
    1.读取对应目录的csv数据，分为 root/年/月/日/股票.csv 
    2.对某一列数据进行标准化
        2.1 创建一个scaler，根据不同的scaler标准化列。ScalerNew中的fit_transform(dataframe，axis)可以得到标准化后的dataframe
        2.2 将某几列数据合在一起进行标准化，然后再各自拆分。axis = None时，标准化的时候使用的是整个表的极端值进行标准化。axis=0时，是对每一列进行标准化。axis=1时是对每一行进行标准化
    3.对某一列数据进行rolling的相减，得到rate 有多种方式：
        3.1 现在的值与过去一段时间的均值的变化率 rollingBeforeK
        3.2 未来一段时间的均值与现在的值的变化率 rollingAfterK
        3.3 未来一段时间与过去一段时间均值变化率 rollingAfterBeforeK
    4.根据列rate，打上标签
        4.1 生成一个新列，根据某特定列的值，在对应行上生成新的值。默认是三分类
        4.2 可以输入条件(condition)，输入标签(choices)，然后直接使用select生成一个新列 可以实现多条件、多标签
    5.根据不同的阈值，得到三种标签的比率，用于平衡三种标签的数量

处理流程大致为：读取数据，筛选出所选数据，选择对应标准化方法标准化，使用rolling得到不同时间的变化率，使用变化率率分析合适的标签阈值，使用阈值对变化率打上标签。

"""


def midPrice(data1, data2):
    return (data2 + data1) / 2


def readTable(data_path):
    if not os.path.isfile(data_path):
        print(data_path)
        return
    data = pd.read_csv(data_path)
    return data


"""
    3.对某一列数据进行rolling的相减，得到rate 有多种方式：
        3.1 现在的值与过去一段时间的均值的变化率
        3.2 未来一段时间的均值与现在的值的变化率
        3.3 未来一段时间与过去一段时间均值变化率
"""


def rollingBeforeK(data, K):
    rate = (-data.rolling(window=K, min_periods=1).mean().shift(1) + data) / data
    return rate


def rollingAfterK(data, K):
    rate = (data.rolling(window=K, min_periods=1).mean().shift(-K) - data) / data
    return rate


def rollingAfterBeforeK(data, K):
    beforeM = data.rolling(window=K, min_periods=1).mean().shift(1)
    afterM = data.rolling(window=K, min_periods=1).mean().shift(-K)
    rate = (afterM - beforeM) / beforeM
    return rate


"""
    4 可以输入条件(condition)，输入标签(choices)，然后直接使用select生成一个新列 可以实现多条件、多标签
    conditions = [
        (df['特定列'] >= 1) & (df['特定列'] < 2),
        (df['特定列'] >= 2) & (df['特定列'] < 3),
        df['特定列'] >= 3,
        df['特定列'] < 1
    ]
    choices = [1, 2, 3, 4]  
        这个方法最后的记过就是满足某个condition的，按照顺序得到choices里面的标签
        df['特定列'] < 1 --> label = 4
"""


def labelBasedOnRate(conditions, choices):
    label = np.select(conditions, choices)
    return label


def calRate01(rate10, yuzhi):  # 计算标签比例
    rate_ = rate10[(rate10 >= -yuzhi) & (rate10 <= yuzhi)].shape[0] / rate10.shape[0]
    rateLarge = rate10[(rate10 <= -yuzhi)].shape[0] / rate10.shape[0]
    rateSmall = rate10[(rate10 >= yuzhi)].shape[0] / rate10.shape[0]
    print(rate_, rateSmall, rateLarge)
    return rate_, rateSmall, rateLarge


class ScalerNew:
    # 处理得到的数据，得到想要的预备值
    # axis = None时，标准化的时候使用的是整个表的极端值进行标准化。axis=0时，是对每一列进行标准化。axis=1时是对每一行进行标准化
    def fit(self, X, axis=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, axis=None):
        self.fit(X, axis)
        return self.transform(X)


class MinMaxScaler(ScalerNew):
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_val = None
        self.max_val = None

    def fit(self, X, axis=None):
        if axis is None:
            self.min_val = X.values.min()
            self.max_val = X.values.max()
        else:
            self.min_val = X.min(axis=axis)
            self.max_val = X.max(axis=axis)
        print(self.max_val, self.min_val)

    def transform(self, X):
        return (X - self.min_val) / (self.max_val - self.min_val) * (self.feature_range[1] - self.feature_range[0]) + \
            self.feature_range[0]


class StandardScaler(ScalerNew):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, axis=None):
        if axis is None:
            self.mean = X.values.mean()
            self.std = X.values.std()
        else:
            self.mean = X.mean(axis=axis)
            self.std = X.std(axis=axis)
        print(self.mean, self.std)

    def transform(self, X):
        return (X - self.mean) / self.std
