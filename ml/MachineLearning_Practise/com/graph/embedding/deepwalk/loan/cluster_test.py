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

def kmeans(X,w):

    # new_df = pca_handle(X)

    # nodes = [i for i in range(w)]
    #
    new_df = TSNE_handle(X)

    # 2类
    km = KMeans(n_clusters=2).fit(new_df)

    # 聚类结果
    beer['cluster'] = km.labels_
    # beer.sort_values('cluster')

    color_idx = {0:[],1:[]}
    color_idx.setdefault(0, [])
    count = 0
    for lab in beer['cluster']:

        if lab == 0:
            color_idx[0].append(count)
        else:
            color_idx[1].append(count)
        count += 1

    print(color_idx)

    for c, idx in color_idx.items():
        plt.scatter(new_df[idx, 0], new_df[idx, 1], label=c)
    plt.legend()
    plt.show()
    # d = new_df[beer['cluster'] == 0]
    # plt.plot(d[0], d[1], 'r.')
    # d = new_df[beer['cluster'] == 1]
    # plt.plot(d[0], d[1], 'go')
    #
    # plt.show()

def dbscan(X):

    new_df = TSNE_handle(X)

    db = DBSCAN(eps=10, min_samples=2).fit(new_df)

    # 分类结果
    labels = db.labels_
    print(labels)
    beer['cluster'] = labels

    color_idx = {0: [], 1: []}
    color_idx.setdefault(0, [])
    count = 0
    for lab in beer['cluster']:

        if lab == 0:
            color_idx[0].append(count)
        else:
            color_idx[1].append(count)
        count += 1

    print(color_idx)

    for c, idx in color_idx.items():
        plt.scatter(new_df[idx, 0], new_df[idx, 1], label=c)
    plt.legend()
    plt.show()

# PCA 降维
def pca_handle(new_df):
    pca = PCA(n_components=2)
    new_pca = pd.DataFrame(pca.fit_transform(new_df))

    return new_pca

# TSNE 降维
def TSNE_handle(embeddings):
    # 读取node id 和对应标签

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(embeddings)

    return node_pos

# 对子图deepwalk训练得到的word2vec特征进行UML
def word_vec_test(beer,w):
    feature = ['v' + str(i) for i in range(1, 301)]
    X = beer[feature]
    kmeans(X,w)
    # dbscan(X)


if __name__ == '__main__':
    beer = pd.read_csv('word_vec.txt', sep=' ')

    w,v = beer.shape
    print(w,v)


    word_vec_test(beer,w)