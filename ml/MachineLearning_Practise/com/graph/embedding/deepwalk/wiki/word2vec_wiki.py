import gensim
from com.graph.embedding.deepwalk import classify
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.discriminant_analysis import  QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# 通过word2vec生成节点特征
def word_to_vec(nodes):
    with open('wiki_sentence.txt','r') as f:
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
    # model.wv.save_word2vec_format(outfile + '.model.txt', binary=False)

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



def get_embedings():
    features = {}
    with open('test.model.txt','r') as f:
        count = 0
        for line in f:
            if count != 0:
                line = line.replace('\n','')
                line = line.replace("'", "")
                temp = line.split(' ')
                node = temp[0]
                embs = temp[1:]
                print(node, embs)
                features[node] = embs
            count+=1

    return features

def evaluate_embeddings(embeddings):
    X, Y = classify.read_node_label_pro('wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = classify.Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)

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

def plot_embeddings(embeddings,):
    X, Y = classify.read_node_label('wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    G = nx.read_edgelist('Wiki_edgelist.txt',
                         create_using=nx.Graph(), nodetype=None, data=[('weight', int)])
    nodes = G.nodes

    # 成功提取word2vec特征
    features = word_to_vec(nodes)
    # features = get_embedings()

    # 评估
    evaluate_embeddings(features)

    # 可视化
    plot_embeddings(features)




