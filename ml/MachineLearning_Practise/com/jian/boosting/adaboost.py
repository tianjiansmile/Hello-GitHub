# -*- coding: UTF-8 -*-

from numpy import *
import matplotlib.pyplot as plt

# 分类的核心过程无非是说，当前列向量的每一个值都只有两种分类结果 1 or -1
# 用 threshVal 来当做分类界限，他的判断方式就是，在坐标轴上切一刀，左边的是一类，右边的是另一类
# 这个分类界限 从最小值过渡到最大值，就是不断的切，然后比对一下分类结果的准确率。记录下错误率最低的

def loadSimpData():
    # 测试样例特征
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    # 测试样例正确结果
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

# 分类器
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    # shape 获得矩阵行列数
    # ones 生成一个 矩阵 用来给出这次分类的判断结果
    retArray = ones((shape(dataMatrix)[0],1))

    if threshIneq == 'lt':
        # 矩阵的比对方式，拿第i行向量与分类界限做比较
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0

    return retArray

# 利用单层决策树分类 找到最佳单层决策树
def buildStump(dataArr,classLabels,D):
    #  mat(classLabels).T 获得转置矩阵
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n =shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasMat = mat(classLabels).T
    # 初始化为无穷大
    minError = inf

    # 对每一个列向量进行分类测评
    for i in range(n):
        # 取第i列向量最小值，最大值
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();

        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            # 分类的核心过程无非是说，当前列向量的每一个值都只有两种分类结果 1 or -1
            # 用 threshVal 来当做分类界限，他的判断方式就是，在坐标轴上切一刀，左边的是一类，右边的是另一类
            # 这个分类界限 从最小值过渡到最大值，就是不断的切，然后比对一下分类结果的准确率。记录下错误率最低的

            # ['lt','gt'] 这个东西无非是，在做判断的时候给个准信 ，lt 规定认为分类界限左边的是数据是-1类的
            # 而 gt规定认为分类界限右边的是数据是-1类的 恰恰相反，就是在给定分类界限的时候情况都判断到了
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j)*stepSize)
                # 用来给出这次分类的判断结果
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m, 1)))
                # errArr 用来记录这次分类的结果 ，把这次判断正确的结果标0
                errArr[predictedVals == labelMat] = 0

                # 其实就是错误率，如果全是0 那么错误率将为0
                weightedError = D.T * errArr

                # 记录错误率最小的一组数据
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr, classLabels,numIt=40):
    # 用来记录每一组弱分类器
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    # 记录投票表决结果
    aggClassEst = mat(zeros((m,1)))

    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print ("D",D.T)

        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha

        weakClassArr.append(bestStump)
        print ("classEst: ", classEst.T)

        # multiply的作用和*是不一样的，在矩阵运算中*是矩阵的乘法运算，multiply是对应位置的数相乘，所以我们看到传入的两个矩阵的维数是相同的
        # 把真确结果和预测结果传入，那么预测失败的对应位置就可以显现出来，这里明显大于零的位置时预测错误的位置
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)


        # exp是指数函数，指数函数时单调递增函数，这里可以凸显出D中出问题的位置，
        # 因为第一个位置判断错误，所以 D的第一个元素会比之前大，而其他元素都会变小，D这么做设置的原因是
        # 把这一次分类分错的位置上的权值升高，分错的降低，这样在下一次的迭代中，分类器只有把这个位置分对
        # 才有可能拿到最小的错误率，想想看 weightedError = D.T * errArr，为了使weightedError最小的话
        # 那么权值高的位置必须分对，这样的话在下一次投票的时候，这个原来出错的位置上基本不会出错，有利用这个位置上的投票结果趋向正确
        # 第一次分类的的时候第一个位置分错了，但是第二次迭代却分对了，这样我们看到第二次迭代投票的时候，第一个位置的结果是正确的
        # 但是第二次分类是最后一个位置又分错了，那第三次迭代的时候最后一个位置上的权值就变得高了，促使这一次把最后一个位置分对，这样
        # 第三次累计的时候最后一个位置上的情况是，第二次迭代错了一次其他两次迭代对了两次，再辅以权值做累加，这个位置就又分类正确了
        # 如果这一次分类其他位置出了问题那就继续迭代下去，直到全部分类正确
        D = multiply(D,exp(expon))
        # 重新设置D的值，保持总量为1
        D = D/D.sum()   # D.sum() 是算出元素之和

        # 每一次迭代都会进行一次投票表决，表决的形式就是，每个弱分类器的分类结果乘以各自的权重a，每一个分类结果做累加
        aggClassEst += alpha*classEst
        print ("aggClassEst: ", aggClassEst.T)

        # 程序最终通过aggClassEst的正负是否与真实分类一致来判定，分类迭代是否终止，这时候理解aggClassEst的意义尤为关键
        # aggClassEst是靠累加类记录的  aggClassEst += alpha*classEst 每一个弱分类器的权重不同， error越低它的a就越高
        # 这样这个弱分类器对最终结果的影响就越大
        # 一直迭代直到最终的分类结果与正确结果一直，然后停止迭代
        aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))

        # 错误率
        errorRate = aggErrors.sum()/m
        print ("total error: ",errorRate,"\n")

        if errorRate == 0.0: break
    return weakClassArr





if __name__ == '__main__':

    dataMatrix,labels = loadSimpData();

    # stumpClassify(dataMatrix,labels,None,None)

    D = mat(ones((5,1))/5)

    # bestStump, minError, bestClasEst = buildStump(dataMatrix,labels,D)
    #
    # print bestStump,minError,bestClasEst

    error = adaBoostTrainDS(dataMatrix,labels,9)

    print (error)