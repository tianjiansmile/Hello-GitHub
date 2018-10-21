# -*- coding: UTF-8 -*-
# 在约会网站上使用 knn算法
#  1, 收集数据： 提供文本文件
#  2， 准备数据：使用python解析文件
#  3， 分析数据 利用Matplotlib画出二维扩散图
#  4， 训练算法  knn 无需训练
#  5   测试算法： 部分数据将成为测试样本
#  6  使用算法

from numpy import *
import matplotlib.pyplot as plt
import operator
from mpl_toolkits.mplot3d import Axes3D

def createDataSet():

    # 特征向量
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])

    # 目标向量
    labels = ['A','A','B','B']
    return group,labels

# 2 将文本转换为矩阵
def file2matrix(filename):
    fr = open('E:/personal/test/'+filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
    # 初始化一个对应的矩阵
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0

    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')

        # 特征数据矩阵
        returnMat[index,:] = listFromLine[0:3]

        # 获取列表倒数第一行，目标向量
        classLabelVector.append(int(listFromLine[-1]))
        index +=1

    return returnMat,classLabelVector

#  画出3D图像
def draw3DMap(matr,classVector):
    fig = plt.figure()
    ax = Axes3D(fig)
    #  分别设置三个坐标轴代表的数据，把数据的类别 1 2 3 映射到散点数据点的大小size，和 数据点的颜色
    # 我们把第三维的数据加上去以后，发现第三维的数据和数据类别标签之间并没有特别明显的关系，不如第一个特征和第二个特征明显
    ax.scatter(matr[:, 0], matr[:, 1], matr[:, 2], matr[:, 0], 15.0 * array(classVector), 15.0 * array(classVector), depthshade=True)
    plt.show()


# 3 创建散点图分析数据
#  数据说明
# 第一列是飞行里程数
# 第二列是玩游戏的时间占比情况
# 第三列是每周吃的冰淇量 以公升计
#  散布图中，一个点代表了一行数据的分布情况
def drowMap(matr,classVector):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 分别画出，这个矩阵的第二列和第三列的数据分布
    #  第一个参数是 X轴，第二个参数是Y轴
    # ax.scatter(matr[:, 0], matr[:, 1])

    # scatter(x,y,sz,c) c是设定散点的颜色，请将 c 指定为颜色名称或 RGB 三元数。
    # sz这个参数是设定散点的大小，
    # classVector数据解释：3，非常有魅力 2，魅力一般  1，不喜欢
    # 这里以这行数据的喜欢与否的数据做为设定圆圈大小的依据，
    # 这里将喜欢不喜欢这一向量数据映射到了点的大小上，也就是说，圆圈越大越喜欢
    # ax.scatter(matr[:,0],matr[:,1],15.0*array(classVector))

    # 这里将喜欢不喜欢这一向量数据映射到了点的颜色上，用来区分不同情况
    # 最大的圈圈是黄色来表示的，看的出来
    # 游戏时间占比再5%--10% 行程数在2000--5000范围内的用户对这个女士更有吸引力
    ax.scatter(matr[:, 0], matr[:, 1], 15.0 * array(classVector), 15.0 * array(classVector))
    # ax.scatter(matr[:, 1], matr[:, 2], 15.0 * array(classVector), 15.0 * array(classVector))

    # ax.scatter(matr[:, 0], matr[:, 2], 15.0 * array(classVector), 15.0 * array(classVector))
    plt.show()


# 分析一下，这里如果那这三个特征去计算数据间的相关性的话，那么旅行里程数将极大的影响最终的相关性
# 这里我们的本意是要这三个特征都必须享有同等的权重，这里构造一个归一化函数来处理这三个特征
# 将任意范围内的数值都转化为0--1之间的数
def autoNorm(dataset):
    # 找出dataset每一列的最小值，注意这个计算包括以下的计算基本都是矩阵计算，
    # 这里是找对每一列最小值，并生成一个矩阵
    minvals = dataset.min(0)
    print (minvals)

    maxvals = dataset.max(0)

    # 找到这一列最大值最小值之间的差距
    ranges = maxvals - minvals
    normalData = zeros(shape(dataset))

    # 列长度
    m = dataset.shape[0]
    # tile(minvals,(m,1)) 解释： 在列方向上重复m次。实则是构造了一个(m,3)维的矩阵，这里扩大了维数
    # 矩阵的减法  oldValue-minval
    normalData = dataset - tile(minvals,(m,1))

    # 新的数值应该是 (oldvalue-min)/(max-min)
    normalData = normalData/tile(ranges,(m,1))

    return normalData,ranges,minvals



# 获得向量之间的距离
def getdistance(v1,v2):
    d = 0.0

    for i in range(len(v1)):
        a = v1[i] -v2[i]
        d += pow(a,2)

    return math.sqrt(d)

# 正态分布
def gaussian(dist, sigma=10.0):
    return math.e ** (-dist ** 2 / sigma ** 2)

# test：测试数据，train训练数据，label目标特征
def predict(test,train,lables,k=3):
    dataSetSize = train.shape[0]

    # 利用矩阵计算向量之间的欧几里得距离
    # 扩充test数据维度和 train一样 ，然后相减
    diffMat = tile(test,(dataSetSize,1)) - train

    # 对每一行数据做平方
    sqDiffMat = diffMat**2
    # 对每一行数据做和，axis=1 按行相加
    sqDistance = sqDiffMat.sum(axis=1)
    # 开方
    distance = sqDistance**0.5

    # argsort() 获得从小到大排列的元素的位置，，，吊炸天的函数库
    sorted = distance.argsort()

    classCount = {}
    for i in range(k):
        # 获得与当前数据相似度最大的数据的目标特征 1 2 3
        votellabel = lables[sorted[i]]

        # dissgaus = gaussian(distance[sorted[i]])

        # 记录目标特征出现的次数，字典真的非常好用
        classCount[votellabel] = classCount.get(votellabel,0) +1

    # 找到字典最大值的key
    # 缺陷也很明显， 如果 三种类型各出现一次，那么这一次判断很可能出现失误
    minkey = max(classCount,key=classCount.get)

    if len(classCount) == 3:
        print (classCount, minkey)


    return minkey


# 自创： 将该分类下的权重和距离做累加，最后两个分类的得分做差
def classify(prd,k):
    wieght = [0,0,0]
    # 取前K个数据进行评估
    for i in range(k):
        if prd[i][2] == 1:
            # wieght[0] += prd[i][0]
            # wieght[0] += prd[i][1]
            wieght[0] +=1
        elif prd[i][2] == 2:
            # wieght[1] += prd[i][0]
            # wieght[1] += prd[i][1]
            wieght[1] += 1
        else:
            # wieght[2] += prd[i][0]
            # wieght[2] += prd[i][1]
            wieght[2] += 1

    # 找到列表最大值的位置
    return wieght.index(max(wieght))+1


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLables = file2matrix('datingTestSet2.txt')
    normalMatr, ranges, minvlas = autoNorm(datingDataMat)

    m = normalMatr.shape[0]
    # 测试数据的个数 总数据的十分之一, 这里取100条数据用于测试
    numTestVecs = int(m * hoRatio)

    # print normalMatr,datingLables

    errorCount = 0.0
    for i in range(numTestVecs):
        # 注意对于矩阵的操作，normalMatr[i,j]  i:代表行， j:代表列
        # normalMatr[i,:] 取第i行数据，列全取
        # normalMatr[numTestVecs:m,:] 取numTestVecs到m行的数据
        classifierResult = predict(normalMatr[i,:],normalMatr[numTestVecs:m,:], datingLables[numTestVecs:m])

        print ("Test is: %d, Real is：%d" % (classifierResult,datingLables[i]))
        if classifierResult != datingLables[i]:
            errorCount += 1.0


    print  (errorCount/numTestVecs)




if __name__ == '__main__':
    # group,lables = createDataSet()
    # print group,lables
    #
    # prd = [1.3,1.1]
    # diss = predict(prd,group,lables)
    #
    # print diss
    #
    # print classify(diss)

    # 生成特征数据矩阵
    matr,labels = file2matrix('datingTestSet2.txt')
    # print matr,labels

    # 画出散点图
    drowMap(matr,labels)
    draw3DMap(matr, labels)
    # 对数据进行归一化
    normalMatro,ranges,minvals = autoNorm(matr)
    # print normalMatro,ranges,minvals

    # 测试算法
    # datingClassTest()


