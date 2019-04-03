
import numpy as np
import networkx as nw

def test():
    # 一个有向图的邻接矩阵
    A = np.matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 0]],
        dtype=float
    )

    X = np.matrix([
        [i, -i]
        for i in range(A.shape[0])
    ], dtype=float)

    temp = A * X

    print(temp)

def karate_gcn(zkc):
    order = sorted(list(zkc.nodes()))
    A = nw.to_numpy_matrix(zkc, nodelist=order)
    I = np.eye(zkc.number_of_nodes())
    A_hat = A + I
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = np.matrix(np.diag(D_hat))

    W_1 = np.random.normal(
        loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
    W_2 = np.random.normal(
        loc=0, size=(W_1.shape[1], 2))

    H_1 = gcn_layer(A_hat, D_hat, I, W_1)
    H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
    output = H_2

    feature_representations = {
        node: np.array(output)[node]
        for node in zkc.nodes()}


def gcn_layer(A_hat, D_hat, X, W):
    return np.relu(D_hat**-1 * A_hat * X * W)

if __name__ == '__main__':
    # test()
    zkc = nw.karate_club_graph()
    karate_gcn(zkc)