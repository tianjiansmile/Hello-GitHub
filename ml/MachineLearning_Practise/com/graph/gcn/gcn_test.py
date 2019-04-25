import numpy as np
import matplotlib.pyplot as plt

def plot_embeddings(embeddings):
    embeddings = embeddings.tolist()
    for idx in range(len(embeddings)):
        print(embeddings[idx])
        plt.scatter(embeddings[idx][0], embeddings[idx][1], label='0')
    plt.legend()
    plt.show()

def relu(x):
    y = np.maximum(x, 0)
    return y

if __name__ == '__main__':
    # 有向图邻接矩阵
    A = np.matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 0]],
        dtype=float
    )

    # 暂且虚构出每一个节点的特征，有两列
    X = np.matrix([
        [i, -i]
        for i in range(A.shape[0])
    ], dtype=float)

    # 对于0节点只有1是他的邻居。传播之后，0节点的特征是[1,-1],这是0节点邻居特征的和
    # 每个节点的表征（每一行）现在是其相邻节点特征的和！换句话说，图卷积层将每个节点表示为其相邻节点的聚合
    # 对角线都为0，无法传播当前节点自己的特征
    C = A * X

    # print(C)

    # 节点的聚合表征不包含它自己的特征！该表征是相邻节点的特征聚合
    # 为了解决第一个问题，我们可以直接为每个节点添加一个自环 [1, 2]。
    # 具体而言，这可以通过在应用传播规则之前将邻接矩阵 A 与单位矩阵 I 相加来实现。
    I = np.matrix(np.eye(A.shape[0]))

    A_hat = A + I
    C1 = A_hat * X

    # 现在，由于每个节点都是自己的邻居，每个节点在对相邻节点的特征求和过程中也会囊括自己的特征！
    # print(C1)

    # 计算度矩阵
    D = np.array(np.sum(A, axis=0))[0]
    D = np.matrix(np.diag(D))

    # 通过将邻接矩阵 A 与度矩阵 D 的逆相乘，对其进行变换，从而通过节点的度对特征表征进行归一化
    F = D ** -1 * A * X
    print(F)


    # 应用权重
    W = np.matrix([
        [1, -1],
        [-1, 1]
    ])

    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = np.matrix(np.diag(D_hat))

    D_hat ** -1 * A_hat * X * W

    out= relu(D_hat ** -1 * A_hat * X * W)

    print(out)

    plot_embeddings(out)

