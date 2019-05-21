import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
from mpl_toolkits.mplot3d import Axes3D as p3d
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# 对社交网络中的一个社区的embedding向量进行无监督分类

def plot_2D(color_idx,new_df,centers,label_count):
    for c, idx in color_idx.items():
        plt.scatter(new_df[idx, 0], new_df[idx, 1], label=c)

    plt.scatter(centers.tn1, centers.tn2, linewidths=label_count, marker='+', s=300, c='black')

    plt.legend()
    plt.show()

def plot_3D(color_idx,new_df,centers,label_count):

    p3d = plt.figure().add_subplot(111, projection='3d')
    for c, idx in color_idx.items():
        p3d.scatter(new_df[idx, 0], new_df[idx, 1], new_df[idx, 2], zdir='z', label=c,s=30, c=None, depthshade=True)
    plt.legend()
    plt.show()

def kmeans(X,w):

    # 映射之后的维度
    col = 2
    node_pos = TSNE_handle(X,col)

    label_count = 3

    # 2类
    km = KMeans(n_clusters=label_count).fit(node_pos)

    # 聚类结果
    beer['cluster'] = km.labels_
    # beer.sort_values('cluster')
    beer['tn1'] = node_pos[:,0]
    beer['tn2'] = node_pos[:,1]
    # beer['tn3'] = node_pos[:,2]

    beer.drop(w, axis=1, inplace=True)
    centers = beer.groupby("cluster").mean().reset_index()

    color_idx = {}
    for i in range(label_count):
        color_idx.setdefault(i, [])

    count = 0
    for lab in beer['cluster']:
        for i in range(label_count):
            if lab == i:
                color_idx[i].append(count)
        count += 1

    # print(color_idx)

    plot_2D(color_idx, node_pos, centers, label_count)

    # plot_3D(color_idx, node_pos, centers, label_count)

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
def TSNE_handle(embeddings,col):
    # 读取node id 和对应标签

    model = TSNE(n_components=col)
    node_pos = model.fit_transform(embeddings)

    return node_pos

# 对子图deepwalk训练得到的word2vec特征进行UML
def word_vec_test(beer,w):
    feature = ['v' + str(i) for i in range(1, 301)]
    # print(feature)
    X = beer[feature]
    kmeans(X,feature)
    # dbscan(X)


if __name__ == '__main__':
    beer = pd.read_csv('little_word_vec.txt', sep=' ')

    w,v = beer.shape
    print(w,v)


    word_vec_test(beer,w)