import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
# from sklearn.cross_validation import cross_val_score

from scipy import stats
import seaborn as sns
from copy import deepcopy

def data_check(train):
    # print('First 20 columns:', list(train.columns[:20]))
    # print(train.shape)
    # print(train.describe())

    # 查看空值情况 ,数据已经被清洗过，非常clean
    # print(pd.isnull(train).values.any())

    # 数据的整体状况
    # print(train.info())

    # 查看离散变量的个数
    cat_features = list(train.select_dtypes(include=['object']).columns)
    print("Categorical: {} features".format(len(cat_features)))
    # 查看连续变量的个数
    cont_features = [cont for cont in list(train.select_dtypes(
        include=['float64', 'int64']).columns) if cont not in ['loss', 'id']]
    print("Continuous: {} features".format(len(cont_features)))

    # 统计离散变量的类别数
    cat_uniques = []
    for cat in cat_features:
        cat_uniques.append(len(train[cat].unique()))

    uniq_values_in_categories = pd.DataFrame.from_items([('cat_name', cat_features), ('unique_values', cat_uniques)])
    # print(uniq_values_in_categories)

    # 正如我们所看到的，大部分的分类特征（72/116）是二值的，绝大多数特征（88/116）有四个值，其中有一个具有326个值的特征（一天的数量？）
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_size_inches(16, 5)
    # ax1.hist(uniq_values_in_categories.unique_values, bins=50)
    # ax1.set_title('Amount of categorical features with X distinct values')
    # ax1.set_xlabel('Distinct values in a feature')
    # ax1.set_ylabel('Features')
    # ax1.annotate('A feature with 326 vals', xy=(322, 2), xytext=(200, 38), arrowprops=dict(facecolor='black'))
    #
    # ax2.set_xlim(2, 30)
    # ax2.set_title('Zooming in the [0,30] part of left histogram')
    # ax2.set_xlabel('Distinct values in a feature')
    # ax2.set_ylabel('Features')
    # ax2.grid(True)
    # ax2.hist(uniq_values_in_categories[uniq_values_in_categories.unique_values <= 30].unique_values, bins=30)
    # ax2.annotate('Binary features', xy=(3, 71), xytext=(7, 71), arrowprops=dict(facecolor='black'))
    #
    plt.figure(figsize=(16, 8))
    # # 我们看一下损失值的分布 损失值中有几个显著的峰值表示严重事故。这样的数据分布，使得这个功能非常扭曲导致的回归表现不佳。
    plt.plot(train['id'], train['loss'])
    plt.title('Loss values per id')
    plt.xlabel('id')
    plt.ylabel('loss')
    plt.legend()
    # plt.show()
    # # 基本上，偏度度量了实值随机变量的均值分布的不对称性。让我们计算损失的偏度：
    # stats.mstats.skew(train['loss']).data
    #
    # # 数据确实是倾斜的  对数据进行对数变换通常可以改善倾斜，可以使用 np.log
    # stats.mstats.skew(np.log(train['loss'])).data
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_size_inches(16, 5)
    # ax1.hist(train['loss'], bins=50)
    # ax1.set_title('Train Loss target histogram')
    # ax1.grid(True)
    # ax2.hist(np.log(train['loss']), bins=50, color='g')
    # ax2.set_title('Train Log Loss target histogram')
    # ax2.grid(True)
    # plt.show()

    # 查看所有连续变量的分布
    # train[cont_features].hist(bins=50, figsize=(16, 12))

    # 特征之间的相关性
    plt.subplots(figsize=(16, 9))
    correlation_mat = train[cont_features].corr()
    sns.heatmap(correlation_mat, annot=True)
    plt.show()

if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    # test = pd.read_csv('test.csv')

    data_check(train)
