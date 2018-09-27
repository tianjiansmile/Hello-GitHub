# -*- coding: UTF-8 -*-

from random import random, randint
import math


def wineprice(rating, age):
    peak_age = rating - 50

    # 根据等级来计算价格
    price = rating / 2
    if age > peak_age:
        # 经过峰值年之后，后继5年内品质将会变差
        price = price * (5 - (age - peak_age))
    else:
        # 价格在接近峰值年时会增加到原值的5倍
        price = price * (5 * ((age + 1) / peak_age))

    if price < 0: price = 0
    return price


def wineset1():
    rows = []
    for i in range(300):
        # 随机生成年代和等级
        rating = random() * 50 + 50
        age = random() * 50

        # 得到一个参考价格
        price = wineprice(rating, age)

        # 添加噪声
        price *= (random() * 0.4 + 0.8)
        # 加入数据集
        rows.append({'input': (rating, age), 'result': price})
    return rows


# 定义两个向量的相似度为欧氏距离
def euclidean(v1, v2):
    d = 0.0
    for i in range(len(v1)):
        a = v1[i]
        b = v2[i]
        d += pow(a - b, 2)
    return math.sqrt(d)


# 获取要预测的向量vec1与数据集data中所有元素的距离
def getdistances(data, vec1):
    distancelist = []
    for i in range(len(data)):
        vec2 = data[i]['input']
        distancelist.append((euclidean(vec1, vec2), i))
    distancelist.sort()
    return distancelist


# knn函数，对列表的前k项结果求了平均值
def knnestimate(data, vec1, k=5):
    # 得到经过排序的距离值
    dlist = getdistances(data, vec1)
    avg = 0.0

    # 对前K项结果求平均值
    for i in range(k):
        idx = dlist[i][1]
        avg += data[idx]['result']
    avg = avg / k
    return avg


# 反函数 将距离转换为权重
def inverseweight(dist, num=1.0, const=0.1):
    return num / (dist + const)


# 减法函数
def subtractweight(dist, const=1.0):
    if dist > const:
        return 0
    else:
        return const - dist

    # 高斯函数


def gaussian(dist, sigma=10.0):
    return math.e ** (-dist ** 2 / sigma ** 2)


# 加权KNN算法，根据距离对K个近邻加权，权值乘以对应的价格作累加最后除以权值之和
# 参数weightf是函数，指示使用哪一种权值衰减方式
# 试验得出，k=3时 误差最小
def weightedKnn(data, vec1, k=3, weightf=gaussian):
    dlist = getdistances(data, vec1)
    result = 0.0
    weight = 0.0

    for i in range(k):
        price = data[dlist[i][1]]['result']  # 价格
        result += price * weightf(dlist[i][0])  # 距离加权，累加价格和
        weight += weightf(dlist[i][0])  # 统计权值和

    return result / weight


# 交叉验证
# 1 随机划分数据集，test指定了测试集所占的比例
# 典型的情况下，测试集只会包含一小部分数据，大概是所有数据的5%，剩下的95%都是训练集
def dividedata(data, test=0.05):
    trainset = []
    testset = []
    for row in data:
        if random() < test:
            testset.append(row)
        else:
            trainset.append(row)

    return trainset, testset


# 2 对测试集进行预测算出误差，针对测试集中的每一项内容调用算法，返回误差
#  其中参数algf是一个函数，可以是 knnestimate,weightedknn或者其他计算价格的函数
def testalgorithm(algf, trainset, testset):
    error = 0.0
    for row in testset:
        # 对测试集的每一项数据都进行预测
        guess = algf(trainset, row['input'])
        # 对预测结果与正确结果进行做差，得出误差
        error += (row['result'] - guess) ** 2

    return error / len(testset)


# 3 交叉验证 多次调用dividedata函数对数据进行随机划分，并计算误差，取所有随机划分的均值
def crossvalidate(algf, data, trials=100, test=0.05):
    error = 0.0
    # trials 代表随机划分的次数
    for i in range(trials):
        trainset, testset = dividedata(data, test)
        # 100多次的交叉验证之后，对累计的误差求平均值
        error += testalgorithm(algf, trainset, testset)

    return error / trials


# 重新生成数据集，加入干扰变量
def wineset2():
    rows = []
    for i in range(300):
        rating = random() * 50 + 50
        age = random() * 50

        aisle = float(randint(1, 20))
        bottleszie = [375.0, 750.0, 1500.0, 3000.0][randint(0, 3)]

        price = wineprice(rating, age)
        price *= (bottleszie / 750)
        price *= (random() * 0.9 + 0.2)
        rows.append({'input': (rating, age, aisle, bottleszie), 'result': price})
    return rows


# 缩放，参数scale的长度与训练数据特征的长度相同. 每个参数乘以训练数据中的特征以达到缩放特征的目的
def rescale(data, scale):
    scaledata = []
    for row in data:
        scaled = [scale[i] * row['input'][i] for i in range(len(scale))]
        scaledata.append({'input': scaled, 'result': row['result']})
    return scaledata


# 构造 优化搜索算法 的代价函数
def createcostfunction(algf, data):
    def costf(scale):
        sdata = rescale(data, scale)
        return crossvalidate(algf, data)

    return costf


if __name__ == '__main__':
    # 构造数据
    data = wineset1()
    print data
    print getdistances(data, (99.0, 5.0))

    # 价格预测
    print '----------------------------------k-最近邻算法---------------------------------------'
    print knnestimate(data, (99.0, 5.0))

    print '----------------------------------加权 k-最近邻算法---------------------------------------'
    # 优化knnestimate函数，对不同距离的近邻进行距离加权
    print weightedKnn(data, (99.0, 5.0))

    # 交叉验证
    print '----------------------------------交叉验证，误差均值---------------------------------------'
    print crossvalidate(weightedKnn, data)