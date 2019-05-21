import gensim
from com.graph.embedding.deepwalk import classify
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.discriminant_analysis import  QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D as p3d
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# 通过word2vec生成节点特征
def word_to_vec(nodes):
    with open('little_sentence.txt','r') as f:
        sentences = []
        for line in f:
            cols = line.strip().split('\t')
            sentences.append(cols)


    # 通过模型提取出300个特征
    model = gensim.models.Word2Vec(sentences, sg=1, size=300, alpha=0.025, window=3, min_count=1, max_vocab_size=None, sample=1e-3, seed=1, workers=45, min_alpha=0.0001, hs=0, negative=20, cbow_mean=1, hashfxn=hash, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=1e4)


    # outfile = './test'
    # fname = './testmodel-0103'
    # save
    # model.save(fname)
    # model.wv.save_word2vec_format(outfile + '.model.bin', binary=True)
    # 将特征保存
    # model.wv.save_word2vec_format('little_word_vec.txt', binary=False)
    # fname = './testmodel-0103'
    # model = gensim.models.Word2Vec.load(fname)
    # a = model.most_similar('0')
    # print(a)

    features = {}
    for i in nodes:
        embd = model.wv[str(i)]
        # print(i,embd)
        features[str(i)] = embd

    return features



# 训练并且预测
def train_predict(feature,label):

    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.7, random_state=1)

    # clf = SVC(kernel="linear")
    # K邻近算法效果最佳
    clf = KNeighborsClassifier(n_neighbors=3)
    # clf = QuadraticDiscriminantAnalysis()

    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(predicted)

    score = clf.score(X_test, y_test)
    print(score)

# 将embedding向量转化为二维空间数据
def plot_embeddings(embeddings, nodes):
    # 读取node id 和对应标签
    X = nodes

    emb_list = []
    for k in X:
        emb_list.append(embeddings[str(k)])
    emb_list = np.array(emb_list)

    # 降维
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault('0', [])
        color_idx['0'].append(i)

    # 利用list的有序性，将每一个节点的降维特征映射到画布上
    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()

# 将embedding向量转化为三维空间数据
def plot_label_embeddings_3D(embeddings, nodes):
    # 读取node id 和对应标签
    X = nodes

    emb_list = []
    for k in X:
        emb_list.append(embeddings[str(k)])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=3)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault('0', [])
        color_idx['0'].append(i)

    p3d = plt.figure().add_subplot(111, projection='3d')
    for c, idx in color_idx.items():
        p3d.scatter(node_pos[idx, 0], node_pos[idx, 1], node_pos[idx, 2], zdir='z', s = 30, c = None, depthshade = True )
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import time

    starttime = time.time()
    G = nx.read_edgelist('little_edgelist.txt',
                         create_using=nx.Graph(), nodetype=None,
                         data=[('type', str), ('call_len', float), ('times', int)])
    nodes = G.nodes


    # 成功提取word2vec特征
    features = word_to_vec(nodes)

    # 查看表示学习之后的空间分布
    plot_embeddings(features, nodes)

    # plot_label_embeddings_3D(features, nodes)

    endtime = time.time()
    print(' cost time: ', endtime - starttime)





