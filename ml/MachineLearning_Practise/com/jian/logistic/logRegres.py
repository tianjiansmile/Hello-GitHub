# -*- coding: UTF-8 -*-

from numpy import *
import matplotlib.pyplot as plt
import os

# 需要提前知道的一些东西
# 原本的问题是，我们只需要把数据特征给输入进去，那么这个模型就要给出分类结果
# 先来看一下Sigmoid函数，这个函数可以对任意有理数进行分类
# 这里把 Sigmoid函数的输入记为Z
# Z = w0X0 + w1X1 + ......... + wnXn
# 矩阵表示就是 Z = WT * X
# X是变量，既是我们的特征数据
# W矩阵便是回归系数，分类模型的好坏取决于回归系数
# 那么模型的优化问题变成了最佳回归系数的计算问题

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open(os.getcwd()+'/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))

    return dataMat,labelMat

# 用来对计算结果进行分类
def sigmoid(inx):
    return 1.0/(1+exp(-inx))

# 梯度上升算法
# so 回归系数是什么呢，这个系数需要对我们的每行数据做一次回归，这行数据与回归系数相乘，然后利用sigmoid分类
def gradAscent(dataMatin,classLabels):
    # 转化为矩阵 100X3
    dataMatrix = mat(dataMatin)
    # print dataMatrix
    # 把目标特征转化为矩阵并转置此矩阵
    labelMat = mat(classLabels).transpose()
    # print labelMat

    # 得到矩阵的维度
    m,n = shape(dataMatrix)
    # print m,n
    # 梯度上升的度量，
    alpha = 0.001
    # 迭代次数 梯度上升迭代500次
    maxCycles = 500

    # ones函数 生成一个 nx1 (3X1)的矩阵--回归系数矩阵
    # 这个矩阵就是这个模型中我们需要优化的
    weights = ones((n,1))
    # print weights

    # print weights
    for k in range(maxCycles):

        # 100x3 X 3x1
        # 为了实现logistic回归分类器，我们在每一个特征上面都乘以一个回归系数并相加
        # 上述的运算可以简单的用矩阵的乘法实现
        # 然后将这个总和带入sigmoid函数进而得到一个0-1之间的结果，即可得出这行数据的分类结果
        h = sigmoid(dataMatrix*weights)
        # 刚开始回归系数初始化为1
        # 得出这次迭代之后的误差 100X1
        error = (labelMat - h)

        # 这里不是严格的梯度上升算法，毕竟要算偏导数的，沿着切面不断的上升 找到极大值
        # 而是以真实值和训练值之间的误差作为指导方向，使得回归系数不断精确
        weights = weights + alpha * dataMatrix.transpose()*error

    # print weights
    return weights

# 画出决策边界
# 如何画出决策边界其实就是说，
# 找出满足 Z = 0 = WT * X  的解x 便是一条直线
def plotBestFit(wei):
    # 是将一个numpy矩阵转换为数组
    weights = wei.getA()
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]

    # 两种分类
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30, c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30, c='green')
    # 设置x轴的区间 y函数的定义域 （-3,3）
    x = arange(-3.0,3.0,0.1)
    # # 设置 Y轴的区间  h画出决策边界线
    # ok 为了明白这里是什么回事，我画了一下 过程是这样的
    # 1 现在我们知道 回归系数了 WT = [4,0.4,-0.6]
    # Z = WT * X  X = [x1,x2,x3] 我们知道X1是1  [1,x2,x3]
    #  Z = 4 + 0.4X2 - 0.6X3
    # 令 Z = 0 -----> 0.6X3 = 4 + 0.4X2 .......>  X3 = (4 + 0.4X2)/0.6
    #  ok  > Y = (4 + 0.4X)/0.6 -----> Y= (w0 + w1*x)/w2 是不是很清晰了
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x,y)
    # 轴的名称
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

# 随机梯度上升算法
# 缺陷，对于一些非线性的特征需要经过多次的迭代才能得到收敛性比较好的回归系数
def stocGradAscent(dataMat,classLabel):
    m,n = shape(dataMat)
    alpha = 0.01
    weights = ones(n)
    print (weights)
    # m 行数 100行
    for i in range(m):
        # 1X3*3X1
        h = sigmoid(sum(dataMat[i]*weights))
        error = classLabel[i] - h
        weights = weights + alpha*error*dataMat[i]

    return weights

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones((n,0))   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# 改进的随机梯度算法
def stocGradAscentImprove(dataMat,classLabel,numIter = 150):
    m,n = shape(dataMat)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)

        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            # 随机选一行，样本动态选择使得回归系数收敛的更快，那么迭代次数就越小
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMat[randIndex]*weights))
            error = classLabel[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]
            del(dataIndex[randIndex])

        return weights


if __name__ == '__main__':
    dataMat,labelMat = loadDataSet()

    weights = gradAscent(dataMat,labelMat)
    plotBestFit(weights)
    print (weights)

    # weights1 = stocGradAscent0(dataMat,labelMat)
    # plotBestFit(weights1)

    # weights1 = stocGradAscentImprove(array(dataMat),labelMat)
    # plotBestFit(weights1)