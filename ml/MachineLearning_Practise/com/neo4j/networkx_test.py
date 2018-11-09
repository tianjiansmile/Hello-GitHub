import matplotlib.pyplot as plt
import networkx as nx
import community

def nex_matploy(G):
    G = nx.random_graphs.barabasi_albert_graph(100, 1)  # 生成一个BA无标度网络G
    print(G.number_of_nodes())
    nx.draw(G)
    plt.savefig("ba.png")  # 输出方式1: 将图像存为一个png格式的图片文件
    plt.show()
if __name__ == "__main__":
    G = nx.Graph()



