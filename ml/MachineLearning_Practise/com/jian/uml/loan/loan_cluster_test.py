import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def kmeans(beer):
    # 2类
    km = KMeans(n_clusters=2).fit(beer)

    # 聚类结果
    beer['cluster'] = km.labels_
    # beer.sort_values('cluster')
    # print(beer['cluster'])

    # print(beer.head())

    new_df = pca_handle(pca_handle(beer[1:]))

    d = new_df[beer['cluster'] == 0]
    plt.plot(d[0], d[1], 'r.')
    d = new_df[beer['cluster'] == 1]
    plt.plot(d[0], d[1], 'go')
    #
    plt.show()

def dbscan(beer):
    db = DBSCAN(eps=10, min_samples=2).fit(beer)

    # 分类结果
    labels = db.labels_
    print(labels)
    beer['cluster'] = labels

    new_df = pca_handle(pca_handle(beer[1:300]))

    d = new_df[beer['cluster'] == 0]
    plt.plot(d[0], d[1], 'r.')
    d = new_df[beer['cluster'] == 1]
    plt.plot(d[0], d[1], 'go')
    #
    plt.show()

# PCA 降维
def pca_handle(new_df):
    pca = PCA(n_components=2)
    new_pca = pd.DataFrame(pca.fit_transform(new_df))

    return new_pca

# 将用户审核通过次数转换为是否审核通过
def map_label(x):
    if x == 0:
        return x
    else:
        return 1

# 是否通过
def loan_approve_test(train):
    # 删除任何一行有空值的记录
    train.dropna(axis=0, how='any', inplace=True)

    # 将逾期次数转化为0，1标签
    train['approve_sum_label'] = train['approve_sum_label'].map(map_label)

    # 处理标签：Fully Paid是正常用户；Charged Off是违约用户
    train['y'] = train['approve_sum_label']
    temp = train['y']
    print(len(train['y'].unique()))

    # 将不参与训练的特征数据删除
    train.drop(['apply_int_label', 'apply_pdl_label', 'apply_sum_label'
                   , 'approve_int_label', 'approve_pdl_label', 'approve_sum_label', 'overdue_pdl_label',
                'overdue_int_label', 'overdue_sum_label', 'maxOverdue_pdl_label',
                'maxOverdue_int_label', 'maxOverdue_sum_label', 'y', 'idNum'], axis=1, inplace=True)

    kmeans(train)
    # dbscan(train)

    train['y'] = temp

    auc = roc_auc_score(train['y'], train['cluster'])
    ks = sf.KS(train, 'cluster', 'y')
    print('准确度Area Under Curve auc', auc, '区分度 KS', ks)

# 是否逾期
from com.risk_score import scorecard_functions_V3 as sf
def loan_overdue_test(train):
    # 删除任何一行有空值的记录
    train.dropna(axis=0, how='any', inplace=True)

    # 将逾期次数转化为0，1标签
    train['overdue_sum_label'] = train['overdue_sum_label'].map(map_label)

    # 处理标签：Fully Paid是正常用户；Charged Off是违约用户
    train['y'] = train['overdue_sum_label']

    temp = train['y']
    print(len(train['y'].unique()))

    # 将不参与训练的特征数据删除
    train.drop(['apply_int_label', 'apply_pdl_label', 'apply_sum_label'
                   , 'approve_int_label', 'approve_pdl_label', 'approve_sum_label', 'overdue_pdl_label',
                'overdue_int_label', 'overdue_sum_label', 'maxOverdue_pdl_label',
                'maxOverdue_int_label', 'maxOverdue_sum_label', 'y','idNum'], axis=1, inplace=True)

    kmeans(train)

    train['y'] = temp

    auc = roc_auc_score(train['y'], train['cluster'])
    ks = sf.KS(train, 'cluster', 'y')
    print('准确度Area Under Curve auc',auc,'区分度 KS',ks)
    # dbscan(train)


if __name__ == '__main__':
    train = pd.read_excel('feature.xls', sheetname='sheet1')
    # train = pd.read_excel('approve_feature.xls', sheetname='sheet1')
    loan_approve_test(train)

    # loan_overdue_test(train)
