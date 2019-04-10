import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def kmeans(X):
    # 2类
    km = KMeans(n_clusters=2).fit(X)

    # 聚类结果
    beer['cluster'] = km.labels_
    # beer.sort_values('cluster')
    # print(beer['cluster'])

    # print(beer.head())

    new_df = pca_handle(pca_handle(beer[1:300]))

    d = new_df[beer['cluster'] == 0]
    plt.plot(d[0], d[1], 'r.')
    d = new_df[beer['cluster'] == 1]
    plt.plot(d[0], d[1], 'go')
    #
    plt.show()

def dbscan(X):
    db = DBSCAN(eps=10, min_samples=2).fit(X)

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

def loop():
    head = 'id '
    for i in range(1,300):
        # print('id','v_'+str(i))
        head += 'v'+str(i)+' '

    print(head)

if __name__ == '__main__':
    beer = pd.read_csv('word_vec.txt', sep=' ')
    feature = ['v'+str(i) for i in range(1,301)]
    X = beer[feature]
    kmeans(X)
    # dbscan(X)

    # loop()