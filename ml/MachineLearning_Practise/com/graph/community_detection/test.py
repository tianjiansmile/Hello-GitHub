
from com.graph.community_detection import pylouvain as lou
import math

def test_karate_club(file):
    pyl = lou.PyLouvain.from_file(file)
    partition, q = pyl.apply_method()
    # Q就是模块度，模块度越大则表明社区划分效果越好。Q值的范围在[-0.5,1），论文表示当Q值在0.3~0.7之间时，说明聚类的效果很好
    print(partition,q)

if __name__ == '__main__':
    # file = 'data/loan_edgelist1.txt'
    file = 'data/little_edgelist.txt'
    test_karate_club(file)