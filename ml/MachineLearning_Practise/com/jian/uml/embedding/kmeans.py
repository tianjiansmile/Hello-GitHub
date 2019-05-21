import pandas as pd
from sklearn.cluster import KMeans
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def kmeans(X):
    # 分别分2类3类聚类
    km = KMeans(n_clusters=3).fit(X)
    km2 = KMeans(n_clusters=2).fit(X)

    # 聚类结果
    beer['cluster'] = km.labels_
    beer['cluster2'] = km2.labels_
    beer.sort_values('cluster')

    # print(beer.sort_values('cluster'))

    cluster_centers = km.cluster_centers_
    cluster_centers_2 = km2.cluster_centers_

    centers = beer.groupby("cluster").mean().reset_index()

    plt.rcParams['font.size'] = 14
    colors = np.array(['red', 'green', 'blue', 'yellow'])

    plt.scatter(beer["calories"], beer["alcohol"], c=colors[beer["cluster"]])

    plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')

    plt.xlabel("Calories")
    plt.ylabel("Alcohol")

    scatter_matrix(beer[["calories", "sodium", "alcohol", "cost"]], s=100, alpha=1, c=colors[beer["cluster"]],
                   figsize=(10, 10))
    plt.suptitle("With 3 centroids initialized")

    plt.show()

def scaled_kmeans(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=3).fit(X_scaled)

    beer["scaled_cluster"] = km.labels_

    colors = np.array(['red', 'green', 'blue', 'yellow'])

    scatter_matrix(X, c=colors[beer.scaled_cluster], alpha=1, figsize=(10, 10), s=100)
    plt.show()


# 聚类评估：轮廓系数（Silhouette Coefficient ）
# 计算样本i到同簇其他样本的平均距离ai。ai 越小，说明样本i越应该被聚类到该簇。将ai 称为样本i的簇内不相似度。
# 计算样本i到其他某簇Cj 的所有样本的平均距离bij，称为样本i与簇Cj 的不相似度。定义为样本i的簇间不相似度：bi =min{bi1, bi2, ..., bik}
# si接近1，则说明样本i聚类合理
# si接近-1，则说明样本i更应该分类到另外的簇
# 若si 近似为0，则说明样本i在两个簇的边界上。
def silhouette_coefficient(X):

    # score_scaled = metrics.silhouette_score(X, beer.scaled_cluster)
    # score = metrics.silhouette_score(X, beer.cluster)
    # print(score_scaled, score)

    scores = []
    for k in range(2, 20):
        labels = KMeans(n_clusters=k).fit(X).labels_
        score = metrics.silhouette_score(X, labels)
        scores.append(score)

    plt.plot(list(range(2, 20)), scores)
    plt.xlabel("Number of Clusters Initialized")
    plt.ylabel("Sihouette Score")

    plt.show()

if __name__ == '__main__':
    beer = pd.read_csv('data.txt', sep=' ')
    X = beer[["calories", "sodium", "alcohol", "cost"]]
    kmeans(X)
    # scaled_kmeans(X)

    # silhouette_coefficient(X)