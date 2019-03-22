from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from pandas.plotting import scatter_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# 主要用于测试用户历史借贷记录数据 在无监督学习下的效果

def dbscan(X):
    db = DBSCAN(eps=10, min_samples=2).fit(X)

    # 分类结果
    labels = db.labels_
    # print(labels)
    beer['cluster_db'] = labels
    beer.sort_values('cluster_db')

    print(beer.sort_values('cluster_db'))
    #
    print(beer.groupby('cluster_db').mean())

    colors = np.array(['red', 'green', 'blue', 'yellow'])
    scatter_matrix(X, c=colors[beer.cluster_db], figsize=(10, 10), s=100)

    plt.show()

def kmeans(X):

    # 分别分2类3类聚类
    # km = KMeans(n_clusters=3).fit(X)
    km = KMeans(n_clusters=2).fit(X)

    # 聚类结果
    beer['cluster'] = km.labels_
    # beer['cluster2'] = km2.labels_
    beer.sort_values('cluster')

    # print(beer.sort_values('cluster'))

    cluster_centers = km.cluster_centers_
    # cluster_centers_2 = km2.cluster_centers_

    centers = beer.groupby("cluster").mean().reset_index()

    plt.rcParams['font.size'] = 14
    colors = np.array(['red', 'green', 'blue', 'yellow'])

    plt.scatter(beer["apply_sum_7"], beer["reject_sum_7"], c=colors[beer["cluster"]])

    plt.scatter(centers.apply_pdl_7, centers.reject_pdl_7, linewidths=3, marker='+', s=300, c='black')

    plt.xlabel("approve_sum_all")
    plt.ylabel("reject_sum_all")

    scatter_matrix(beer[["apply_sum_7", "reject_sum_7","apply_sum_all","reject_sum_all","approve_sum_all", "overdue_sum_all"]], s=100, alpha=1, c=colors[beer["cluster"]],
                   figsize=(10, 10))
    plt.suptitle("With 3 centroids initialized")

    plt.show()

def data_discover(train):
    # print(beer.describe())
    print(train.shape)

    # 查看连续变量的个数
    cont_features = [cont for cont in list(train.select_dtypes(
        include=['float64', 'int64']).columns) if cont not in ['idNum']]
    print("Continuous: {} features".format(cont_features))

    # 统计离散变量的类别数
    cat_uniques = []
    for cat in cont_features:
        cat_uniques.append(len(train[cat].unique()))

    uniq_values_in_categories = pd.DataFrame.from_items([('cat_name', cont_features), ('unique_values', cat_uniques)])
    print(uniq_values_in_categories)

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
    # plt.subplots(figsize=(16, 9))
    # correlation_mat = train[cont_features].corr()
    # sns.heatmap(correlation_mat, annot=True)
    # plt.show()

if __name__ == '__main__':
    beer = pd.read_csv('loan_history_data.txt', sep=',')

    # data_discover(beer)
    X = beer[['apply_pdl_7', 'apply_int_7', 'apply_sum_7', 'reject_pdl_7', 'reject_int_7', 'reject_sum_7', 'approve_pdl_7', 'approve_int_7', 'approve_sum_7', 'overdue_pdl_7', 'overdue_int_7', 'overdue_sum_7', 'loanamount_pdl_7', 'loanamount_int_7', 'loanamount_sum_7', 'maxOverdue_pdl_7', 'maxOverdue_int_7', 'maxOverdue_sum_7', 'apply_pdl_14', 'apply_int_14', 'apply_sum_14', 'reject_pdl_14', 'reject_int_14', 'reject_sum_14', 'approve_pdl_14', 'approve_int_14', 'approve_sum_14', 'overdue_pdl_14', 'overdue_int_14', 'overdue_sum_14', 'loanamount_pdl_14', 'loanamount_int_14', 'loanamount_sum_14', 'maxOverdue_pdl_14', 'maxOverdue_int_14', 'maxOverdue_sum_14', 'apply_pdl_30', 'apply_int_30', 'apply_sum_30', 'reject_pdl_30', 'reject_int_30', 'reject_sum_30', 'approve_pdl_30', 'approve_int_30', 'approve_sum_30', 'overdue_pdl_30', 'overdue_int_30', 'overdue_sum_30', 'loanamount_pdl_30', 'loanamount_int_30', 'loanamount_sum_30', 'maxOverdue_pdl_30', 'maxOverdue_int_30', 'maxOverdue_sum_30', 'apply_pdl_60', 'apply_int_60', 'apply_sum_60', 'reject_pdl_60', 'reject_int_60', 'reject_sum_60', 'approve_pdl_60', 'approve_int_60', 'approve_sum_60', 'overdue_pdl_60', 'overdue_int_60', 'overdue_sum_60', 'loanamount_pdl_60', 'loanamount_int_60', 'loanamount_sum_60', 'maxOverdue_pdl_60', 'maxOverdue_int_60', 'maxOverdue_sum_60', 'apply_pdl_90', 'apply_int_90', 'apply_sum_90', 'reject_pdl_90', 'reject_int_90', 'reject_sum_90', 'approve_pdl_90', 'approve_int_90', 'approve_sum_90', 'overdue_pdl_90', 'overdue_int_90', 'overdue_sum_90', 'loanamount_pdl_90', 'loanamount_int_90', 'loanamount_sum_90', 'maxOverdue_pdl_90', 'maxOverdue_int_90', 'maxOverdue_sum_90', 'apply_pdl_180', 'apply_int_180', 'apply_sum_180', 'reject_pdl_180', 'reject_int_180', 'reject_sum_180', 'approve_pdl_180', 'approve_int_180', 'approve_sum_180', 'overdue_pdl_180', 'overdue_int_180', 'overdue_sum_180', 'loanamount_pdl_180', 'loanamount_int_180', 'loanamount_sum_180', 'maxOverdue_pdl_180', 'maxOverdue_int_180', 'maxOverdue_sum_180', 'apply_pdl_all', 'apply_int_all', 'apply_sum_all', 'reject_pdl_all', 'reject_int_all', 'reject_sum_all', 'approve_pdl_all', 'approve_int_all', 'approve_sum_all', 'overdue_pdl_all', 'overdue_int_all', 'overdue_sum_all', 'loanamount_pdl_all', 'loanamount_int_all', 'loanamount_sum_all', 'maxOverdue_pdl_all', 'maxOverdue_int_all', 'maxOverdue_sum_all']]
    # kmeans(X)

    X = beer[
        ['apply_pdl_7', 'apply_int_7', 'apply_sum_7', 'reject_pdl_7', 'reject_int_7', 'reject_sum_7',
         'apply_pdl_14', 'apply_int_14', 'apply_sum_14', 'reject_pdl_14', 'reject_int_14', 'reject_sum_14',
         'approve_pdl_14', 'approve_int_14', 'approve_sum_14', 'overdue_pdl_14', 'overdue_int_14', 'overdue_sum_14',
         'apply_pdl_all', 'apply_int_all', 'apply_sum_all', 'reject_pdl_all', 'reject_int_all',
         'reject_sum_all', 'approve_pdl_all', 'approve_int_all', 'approve_sum_all', 'overdue_pdl_all',
         'overdue_int_all', 'overdue_sum_all', 'loanamount_pdl_all', 'loanamount_int_all', 'loanamount_sum_all',]]

    # kmeans(X)

    dbscan(X)
