# -*- coding: UTF-8 -*-

import re
import math
from bokeh.charts.attributes import cat
from nltk.corpus.reader.reviews import FEATURES


def getwords(doc):
    splitter = re.compile('\\W*')
    # 将句子分成单词，然后生成一个列表、
    words = [s.lower() for s in splitter.split(doc) if len(s) > 2 and len(s) < 20]

    # 只返回一组不重复的单词,该字典以w为key，以1为value，此时，就算重复的单词，也不会多保留，因为key只有一个
    return dict([(w, 1) for w in words])


class classfilter:
    def __init__(self, getfeatures, filename=None):
        # 统计特征(例子中的单词)和分类（是否是垃圾邮件）的组合数量
        self.fc = {}
        # 一个纪录各分类被使用次数的字典,也这个分类中有多少多少文档
        self.cc = {}
        # 从内容出提取特征，在例子中就是getwords函数
        self.getfeatures = getfeatures

    # 对特征和分类变量（self.fc）的改变
    # f代表feature，本例中的单词，cat是类型  good 还是bad
    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1

    # 增加对某一个分类的计数器
    def incc(self, cat):
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1

    # 某一单词出现于某一分类中的次数
    def fcount(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])
        return 0

    # 属于某一个分类的内容项的数量
    def catcount(self, cat):
        if cat in self.cc:
            return float(self.cc[cat])
        return 0

    # 所有内容项的数量
    def totalcount(self):
        return sum(self.cc.values())

    # 所有分类的列表
    def categories(self):
        return self.cc.keys()

    def train(self, item, cat):
        # 单词字典
        features = self.getfeatures(item)
        #         print features
        # 我们就把每一个特征（单词）在响应的特征里面增加一次
        for f in features:
            self.incf(f, cat)
        # 因为传入了一份文档和分类，然后我们就把这个类型加一次就好。
        self.incc(cat)

    # 某一个单词出现在某一分类中的概率
    def fprob(self, f, cat):
        if self.catcount(cat) == 0: return 0

        return self.fcount(f, cat) / self.catcount(cat)

    def wightedprob(self, f, cat, prf, weight=1.0, ap=0.5):
        # 计算当前的概率值，就是计算没有初始假设概率的影响下，只有训练文档的数据产生出来的概率
        basicprob = prf(f, cat)
        #         print(basicprob)
        totals = sum([self.fcount(f, c) for c in self.categories()])
        #         print(totals)
        bp = ((weight * ap) + (totals * basicprob)) / (weight + totals)
        return bp


# 写一个训练的代码，训练一些样本数据
def sampletrain(cl):
    cl.train('Nobody owns the water.', 'good')
    cl.train('the quick rabbit jumps fences', 'good')
    cl.train('buy pharmaceuticals now', 'bad')
    cl.train('make quick money at the online casino', 'bad')
    cl.train('the quick brown fox junmps', 'good')


cl = classfilter(getwords)
sampletrain(cl)
print cl.fc
print cl.cc
print cl.fcount('quick', 'good'), cl.fcount('the', 'bad')

# quick  在good 分类语句中 出现了2次，good 分类语句一共有3句， 故概率是 fcount/catcount
print cl.fprob('quick', 'good')

print cl.fprob('money', 'good')  # money 没有出现在good分类中，结果是0 太过于偏激


# print '训练样本执行一次的Pr（money|good）:%f'%(cl.wightedprob('money','good',cl.fprob))
# sampletrain(cl)
# print '训练样本执行二次的Pr（money|good）:%f'%(cl.wightedprob('money','good',cl.fprob))

# 朴素贝叶斯分类器,之所以称为其为朴素，是因为：它假设将要被组合的各个单词的概率是彼此独立的
# 为了学好朴素贝叶斯分类器，我们首先要计算一个概率：新文档属于给定分类的概率。就是说，来了个新文档，我想知道其属于bad的概率
# 简单理解朴素贝叶斯原理  P(AB) = P(B)*P(A|B) >= P(A|B) = P(AB)/P(B)
# 又 P(B|A) = P(AB)/P(A) >= P(A|B) = (P(B|A)*P(A))/P(B)
# 即  P(good|money) = (P(money|good)*P(good))/P(money)
# 求出现money 这一词的邮件是正常邮件的概率， 可以分为两个事件，1，选择出现money这一词的邮件。 2 选择正常邮件‘good’这一分类

class naivebayes(classfilter):
    # P(B|A)
    def docprob(self, item, cat):
        features = self.getfeatures(item)

        p = 1
        for f in features:
            p *= self.wightedprob(f, cat, self.fprob)
        return p

    def prob(self, item, cat):
        # p(good)
        catprob = self.catcount(cat) / self.totalcount()
        # p(money|good)
        docprob = self.docprob(item, cat)
        # p(money|good)*P(good)/P(money)
        return docprob * catprob