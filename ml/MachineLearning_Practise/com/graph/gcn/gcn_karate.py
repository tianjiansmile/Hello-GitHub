import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

zkc = nx.karate_club_graph()
order = sorted(list(zkc.nodes()))
A = nx.to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())
# 带有自旋的邻接矩阵
A_hat = A + I
D_hat = np.array(np.sum(A_hat, axis=0))[0]
# 按节点度进行归一化处理
D_hat = np.matrix(np.diag(D_hat))

W_1 = np.random.normal(
    loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
W_2 = np.random.normal(
    loc=0, size=(W_1.shape[1], 2))

def relu(x):
    y = np.maximum(x, 0)
    return y

def gcn_layer(A_hat, D_hat, X, W):
    return relu(D_hat**-1 * A_hat * X * W)

def plot_embeddings(embeddings):

    for i,idx in embeddings.items():
        print(i,idx)
        plt.scatter(idx[0], idx[1], label=i)
    plt.legend()
    plt.show()

H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
output = H_2

feature_representations = {
    node: np.array(output)[node]
    for node in zkc.nodes()}

print(feature_representations)

# plot_embeddings(feature_representations)