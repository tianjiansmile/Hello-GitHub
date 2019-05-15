
# https://blog.csdn.net/qq_37608890/article/details/79146555

# 关联分析
# if-then形式：若出现A，可以推导出有很大可能性出现B，若出现{A,B}，那么很有可能出现C。
# 我们的目标就是寻找这样的组合对，问题的关键就是如何统计衡量这个可能性

#支持度 support：当前组合出现的概率：当前集合的频率 / 订单数  F_{A,B} / F_orders

# 置信度 confidence：类似一个条件概率，A出现的情况下，B出现的概率，有方向
# confidence{A->B} = support{A,B}/support{A}
# confidence{B->A} = support{A,B}/support{B}

# 提升度 lift：lift{A,B} = support{A,B}/support{A}*support{B}
# 分子是A B同时出现的概率；分母是若A B完全独立，同时出现的概率，或者是说A B完全随机分布时同时出现的概率。
# 考察A B的关联程度：
# >1 同时出现的频率高于随机分布，正相关；
# <1 同时出现的频率低于随机分布，负相关；
# =1 相互独立，无关。

# 加载数据集
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

'''创建集合C1即对dataSet去重排序'''
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # frozenset表示冻结的set 集合，元素无改变把它当字典的 key 来使用
    return C1
    # return map(frozenset, C1)


''' 计算候选数据集CK在数据集D中的支持度，返回大于最小支持度的数据'''
def scanD(D,Ck,minSupport):
    # ssCnt 临时存放所有候选项集和频率.
    ssCnt = {}
    for tid in D:
        # print('1:',tid)
        for can in map(frozenset,Ck):      #每个候选项集can
            # print('2:',can.issubset(tid),can,tid)
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] +=1

    numItems = float(len(D)) # 所有项集数目
    # 满足最小支持度的频繁项集
    retList  = []
    # 满足最小支持度的频繁项集和频率
    supportData = {}

    for key in ssCnt:
        support = ssCnt[key]/numItems   #除以总的记录条数，即为其支持度
        if support >= minSupport:
            retList.insert(0,key)       #超过最小支持度的项集，将其记录下来。
        supportData[key] = support
    return retList, supportData

#构建多个商品对应的项集
def aprioriGen(Lk,k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i]|Lk[j])
    return retList

# 查找所有支持度大于0.5的频繁集项
def apriori(dataSet,minSupport = 0.5):
    c1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1,supportData = scanD(D,c1,minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData

#使用关联规则生成函数
def generateRules(L,supportData,minConf = 0.7):
    bigRuleList = []
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

#集合右边一个元素
def calcConf(freqSet,H,supportData,brl,minConf = 0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

#生成更多的关联规则
def rulesFromConseq(freqSet,H,supportData,br1,minConf = 0.7):
    m = len(H[0])
    if (len(freqSet)>(m + 1)):
        Hmp1 = aprioriGen(H,m+1)
        Hmp1 = calcConf(freqSet,Hmp1,supportData,br1,minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet,Hmp1,supportData,br1,minConf)

if __name__ == '__main__':
    dataSet = loadDataSet()
    # 最小子集列表
    # c1 = createC1(dataSet)
    # print(c1)
    # # 去重映射到字典并放到list
    # D = list(map(set, dataSet))
    # print(D)
    # L1, suppData0 = scanD(D, c1, 0.5)
    #
    # print(L1)

    # 1 寻找频繁集项
    minSupport = 0.5
    L, suppData = apriori(dataSet, minSupport)
    print(L)
    print(suppData)

    # 2 从频繁项集中挖掘关联规则

    rules = generateRules(L, suppData, minConf=0.5)
    print(rules)

    # 毒蘑菇数据集
    mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
    L, supportData = apriori(mushDatSet, minSupport=0.3)
    for item in L[1]:
        if item.intersection('2'):
            print(item)
