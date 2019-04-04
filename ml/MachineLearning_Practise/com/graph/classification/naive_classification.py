#!-*- coding:utf8-*-
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# 朴素的节点分类方法，直接将图的邻接矩阵作为特征输入

# 邻接矩阵
def graph2matrix(G):
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    res = np.zeros([n,n])
    for edge in G.edges:
        res[int(int(edge[0]))][int(edge[1])] = 1
        res[int(int(edge[1]))][int(edge[0])] = 1

    return res



def train(edgeMat,label):
    X_train, X_test, y_train, y_test = train_test_split(edgeMat, label, test_size=0.3, random_state=1)

    # clf = SVC(kernel="linear")
    clf = KNeighborsClassifier(n_neighbors=3)

    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(predicted)

    score = clf.score(X_test, y_test)
    print(score)

if __name__ == '__main__':
    G = nx.karate_club_graph()
    # 标签
    label = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    edgeMat = graph2matrix(G)
    # 训练，预测
    train(edgeMat,label)