import pandas as pd
import re
import time
import datetime
import pickle
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegressionCV
# import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from numpy import log
from sklearn.metrics import roc_auc_score
import numpy as n


def data_check(train):
    # print('First 20 columns:', list(train.columns[:20]))
    print(train.shape)
    print(train.describe())

    # 查看空值情况 ,数据已经被清洗过，非常clean
    print(pd.isnull(train).values.any())

    # 数据的整体状况
    print(train.info())

    # 检查空值比例
    check_null = train.isnull().sum(axis=0).sort_values(ascending=False)
    print(check_null[check_null > 0])

    # 删除任何一行有空值的记录
    train.dropna(axis=0, how='any', inplace=True)

    print(pd.isnull(train).values.any())

    cont_features = [cont for cont in list(train.select_dtypes(
        include=['float64', 'int64']).columns) if cont not in ['idNum']]
    # print("Continuous: {} features".format(len(cont_features)))

    # 统计离散变量的类别数
    cat_uniques = []
    for cat in cont_features:
        cat_uniques.append(len(train[cat].unique()))

    uniq_values_in_categories = pd.DataFrame.from_items([('cat_name', cont_features), ('unique_values', cat_uniques)])
    # print(uniq_values_in_categories)

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

    # 特征之间的相关性
    plt.subplots(figsize=(16, 9))
    correlation_mat = train[cont_features].corr()
    sns.heatmap(correlation_mat, annot=True)
    plt.show()




if __name__ == '__main__':
    train = pd.read_excel('features.xls',sheetname='sheet1')

    data_check(train)
