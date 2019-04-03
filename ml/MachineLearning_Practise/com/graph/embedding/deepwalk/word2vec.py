import gensim
from com.graph.embedding.deepwalk import classify
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.discriminant_analysis import  QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import networkx as nx


# 通过word2vec生成节点特征
def word_to_vec(nodes):
    with open('sentence.txt','r') as f:
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



# 读取karate节点label
def get_label():
    with open('karate_comm.dat','r') as f:
        sentences = []
        for line in f:
            print(line)

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
    X, Y = classify.read_node_label_pro('karate_comm.dat')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = classify.Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)

# 训练并且预测
def train_predict(feature,label):

    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=1)

    # clf = SVC(kernel="linear")
    # K邻近算法效果最佳
    clf = KNeighborsClassifier(n_neighbors=3)
    # clf = QuadraticDiscriminantAnalysis()

    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(predicted)

    score = clf.score(X_test, y_test)
    print(score)


if __name__ == '__main__':
    G = nx.karate_club_graph()
    nodes = G.nodes

    # 标签
    label = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # 成功提取word2vec特征
    features = word_to_vec(nodes)
    # features = get_embedings()
    # evaluate_embeddings(features)

    f_data = []
    for re in features.values():
        f_data.append(re)

    train_predict(f_data, label)




