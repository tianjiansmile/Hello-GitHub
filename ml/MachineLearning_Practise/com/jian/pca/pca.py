# -*- coding: UTF-8 -*-
from numpy import *
import matplotlib.pyplot as plt
import operator
from mpl_toolkits.mplot3d import Axes3D
import os

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]

    return mat(datArr)

# 主成分分析
def pca(dataMat,topNfeat=9999999):
    # axis 不设置值，对 m*n 个数求均值，返回一个实数
    # axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
    # axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
    # 每一列做一个向量meanVals算出了每个向量的平均值，就是这个向量的期望 EX
    meanVals = mean(dataMat, axis=0)
    # print(meanVals)
    meanRemoved = dataMat - meanVals
    # print('Ex mart:',meanRemoved)
    # cov() 协方差表示Xy之间相互关系的数字特征 cov(x,y) = (x-Ex)(y-Ey)
    # 当Cov(X, Y) > 0时，X与Y正相关；
    # 当Cov(X, Y) < 0时，X与Y负相关；
    # 当Cov(X, Y) = 0时，X与Y不相关，独立；
    # 协方差矩阵的意义：
    # 方差是这个向量减均值的期望，DX  方差用来度量随机变量和其数学期望（即均值）之间的偏离程度 离散程度，描述的一个变量的特性
    # 方差是变量减均值的期望，两个变量的协方差是变量一减均值，乘以，变量二减均值，的期望。
    # 协方差矩阵，就是多个变量两两间协方差值，按顺序排成的矩阵。这个矩阵是二维矩阵，故其协方差矩阵是二维的方阵
    covMat = cov(meanRemoved,rowvar=0)
    # print('cov mart',covMat)
    # 对于机器学习领域的PCA来说，如果遇到的矩阵不是方阵，需要计算他的协方差矩阵来进行下一步计算，
    # 因为协方差矩阵一定是方阵，而特征值分解针对的必须是方阵，svd针对的可以是非方阵情况。
    # 协方差矩阵其实保留了原矩阵的主要数据特性，每一个向量本身的性质DX，以及各个向量之间的性质cov（x, y）
    # 也就是说，每一个向量的数据之间的离散程度我们保留了，而且各个向量之间的相关性我们也保留下来了，接下来可以去进行特征值分解了
    # 获得协方差矩阵的特征值和特征向量，
    eigVal,eigVects = linalg.eig(mat(covMat))
    print(eigVal,eigVects)
    eigValInd = argsort(eigVal) # 排序
    print(eigValInd,eigVal)
    eigValInd = eigValInd[:-(topNfeat+1):-1] # 取前99999999列向量
    print(eigValInd,eigVal)
    # 这里其实是将两个特征向量按其对应的特征值大小调换了顺序，这里得到的由特征向量组成的矩阵就是P  A = P-1特征值对角矩阵P
    redEigVects = eigVects[:,eigValInd]
    print(redEigVects)
    # # P就是我们想要的新空间的坐标系 然后 A*P 的运算就将原数据转换到了新空间的坐标系中表示了
    # lowDDataMat是新空间的矩阵表示
    lowDDataMat = meanRemoved * redEigVects
    # 这里做了一个简单的验证 AP = lowDDataMat ==> A = lowDDataMat*P-1 (p-1 = pT)
    reconMat = (lowDDataMat*redEigVects.T) + meanVals
    print(lowDDataMat,reconMat)
    return lowDDataMat,reconMat

# 散点图
def drowMap(matr):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 分别画出，这个矩阵的第二列和第三列的数据分布
    #  第一个参数是 X轴，第二个参数是Y轴
    # ax.scatter(matr[:, 0], matr[:, 1])
    ax.scatter(matr[:, 0].tolist(), matr[:, 1].tolist())
    plt.show()

def drawHistogram(matr):
    plt.style.use('fivethirtyeight')
    plt.hist(matr[0], bins=100, edgecolor='k');
    plt.xlabel('Score');
    plt.ylabel('Number of Buildings');
    plt.title('Energy Star Score Distribution');
    plt.show()


def replaceNanWithMean(dataMat):
    # dataMat = loadDataSet(os.getcwd()+'/secom.data.txt')
    numFeat = shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0],i])
        dataMat[nonzero(isnan(dataMat[:,i].A))[0],i] = meanVal


    return dataMat
if __name__ == '__main__':
    dataMat = loadDataSet(os.getcwd()+'/testSet_pca.txt')
    # print(shape(dataMat))
    # 从图像上可以看出 两组数据是正相关的，也应证了两个向量的协方差是大于零的
    # drowMap(dataMat)
    # drawHistogram(dataMat)
    # 降维数据到一维
    # lowDate,recon = pca(dataMat,2)
    # drowMap(lowDate)

    dataMat = loadDataSet(os.getcwd()+'/secom')
    newData = replaceNanWithMean(dataMat)
    lowDate, recon = pca(newData, 20)
    drowMap(lowDate)

