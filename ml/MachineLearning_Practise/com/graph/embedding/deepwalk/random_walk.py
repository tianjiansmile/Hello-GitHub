#!-*- coding:utf8-*-
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 随机游走的思路，就是按当前节点开始随机跳固定的跳数，为每一个节点生成一组随机游走的序列，用这个序列来代表当前这个节点的特征
# 之后通过word2vec进一步提取特征
def randomWalk(_g, _corpus_num, _deep_num, _current_word):
	_corpus = []
	for i in range(_corpus_num):
		sentence = [_current_word]
		current_word = _current_word
		count = 0
		while count<_deep_num:
			count+=1
			_node_list = []
			_weight_list = []
			for _nbr, _data in _g[current_word].items():
				_node_list.append(_nbr)
			# 	_weight_list.append(_data['weight'])
			# _ps = [float(_weight) / sum(_weight_list) for _weight in _weight_list]
			sel_node = roulette(_node_list,None)
			sentence.append(sel_node)
			current_word = sel_node
		_corpus.append(sentence)
	return _corpus

def roulette(_datas, _ps):
	return np.random.choice(_datas, p=_ps)

# 从图中提取节点特征
def node_vec(nodes):
    num = 3 # 重复10次
    deep_num = 30 # 跳数
    with open('sentence.txt','w') as f:
        k = 1
        for word in nodes:
            print(k)
            k += 1
            corpus = randomWalk(G, num, deep_num, word)
            print(corpus)
            for cols in corpus:
                col = [str(i) for i in cols]
                sentences = '\t'.join(col)
                f.write(sentences + '\n')

def draw_graph(G):
    # plt.subplots(1, 1, figsize=(15, 6))

    # 返回Zachary的空手道俱乐部图。
    G.clear()
    G = nx.karate_club_graph()
    plt.subplot(1, 2, 1)
    nx.draw(G, with_labels=True)
    plt.title('karate_club_graph')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == '__main__':
    G = nx.karate_club_graph()
    nodes = G.nodes

    # draw_graph(G)

    node_vec(nodes)
