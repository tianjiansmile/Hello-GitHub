# -*- coding: utf-8 -*-
import scipy.sparse
import networkx as nx
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import layers as lg
# import utils as us
from sklearn.manifold import TSNE

def gcn():
    g = nx.read_edgelist('karate.edgelist', nodetype=int, create_using=nx.Graph())

    adj = nx.to_numpy_matrix(g)

    # Get important parameters of adjacency matrix
    n_nodes = adj.shape[0]

    # 得到具有自环的邻接矩阵 A_hat
    adj_tilde = adj + np.identity(n=n_nodes)
    print(adj_tilde)
    # 构造度矩阵 D_hat 用于聚合每一个节点的邻居以及自己的特征
    d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))

    d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1 / 2)
    # diag 获取对角线元素
    d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
    # dot 矩阵乘法
    adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
    print(d_tilde_diag)
    # adj_norm_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(adj_norm))


#    print(adj_norm_tuple)



if __name__ == '__main__':
    gcn()