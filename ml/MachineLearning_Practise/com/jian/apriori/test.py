
# https://blog.csdn.net/qq_37608890/article/details/79146555

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

if __name__ == '__main__':
    dataSet = loadDataSet()
    c1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, suppData0 = scanD(D, c1, 0.5)

    print(L1)