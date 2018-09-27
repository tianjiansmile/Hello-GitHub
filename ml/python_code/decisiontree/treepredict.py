# -*- coding: UTF-8 -*-
from moivescore import my_data
from dask.array.core import log2
from math import log

from PIL import Image, ImageDraw


# 决策树的生成过程就是 使用满足划分准则的特征不断的将数据集划分为纯度更高，
# 不确定性更小的子集的过程。对于当前数据集D的每一次的划分，
# 都希望根据某特征划分之后的各个子集的纯度更高，不确定性更小。

# 决策树算法3要素：
#  1.特征选择
#    特征选择方法，比如：熵，信息增益，信息增益率，基尼指数
#    特征选择准则：用某特征对数据集划分之后，各数据子集的纯度要比划分前的数据集D的纯度高：：
#    常用算法：ID3(信息增益)， CART（基尼不纯度）
#  2.决策树生成
#  3.决策树剪枝

# 决策树上的节点
class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col  # 待检验的判断条件所对应的列索引值
        self.value = value  # 为了使结果为true，当前列必须匹配的值
        self.results = results  # 针对当前分支的结果，它是一个字典。除了叶子节点以外，在其他节点上该值都为None
        self.tb = tb  # 子节点 True or False 分支
        self.fb = fb


# 根据value 将data 分类
def divideset(rows, column, value):
    split_function = None
    # 如果 value是数字  split_function 按数字判断  >= 为true  <=为false
    # 如果 value是字符串  split_function 按按字符判断  yes 为true  not yes为false
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value

    # 分离
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    # 从数据分离结果上来看并不是特别好，数据的混杂程度还是比较高
    #         print len(set1),set1
    #         print len(set2),set2
    return (set1, set2)


# 按是否阅读过FAQ来分类

# divideset(my_data, 2, 'yes')

# 统计各个服务类型的个数
def uniquecounts(rows, cat):
    results = {}
    for row in rows:
        r = row[cat]
        if r not in results: results[r] = 0
        results[r] += 1
    #     print results
    return results


# 基尼不纯度：将来自集合中某种结果随机应用于集合中某一数据项的预期误差率，用来评估对于数据的拆封结果的好坏
# 基尼不纯度的大概意思是 一个随机事件变成它的对立事件的概率,如果集合中的每一个数据项
# 都属于同一个分类，那么预测结果与总是正确的，此时预测误差率为0，如果有均分的四种可能，可以预测有75%
# 的概率是不正确的,这里服务类型有三种 基尼不纯度0.63
# 所以基尼不纯度也可以作为 衡量系统混乱程度的 标准,概率越高说明对数据的拆分越不理想
def giniimpurity(rows, cat):
    total = len(rows)
    counts = uniquecounts(rows, cat)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2: continue
            p2 = float(counts[k2]) / total
            #             print k1,k2
            imp += p1 * p2
    print imp
    return imp


print '------------------基尼不纯度-------------------'


# giniimpurity(my_data,1)
# giniimpurity(my_data,2)
# giniimpurity(my_data,3)
# giniimpurity(my_data,4)

# 熵 代表的是集合的无序程度--混乱程度
# 一个离散型随机变量 X 的熵 H(X) 定义为：
# H(X) = -sum( p(x)*log(px) )
def entropy(row, cat):
    log2 = lambda x: log(x) / log(2)

    results = uniquecounts(row, cat)

    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(row)
        # px*logpx
        ent = ent - p * log2(p)

    #     print ent
    return ent


#     print -sum( (float(results[val])/len(row))*log2(float(results[val])/len(row)) for val in results)

print '------------------熵-------------------'


# entropy(my_data,1)
# entropy(my_data,2)
# entropy(my_data,3)
# entropy(my_data,4)

# 为了弄明白一个属性的混乱程度，我们先用选用熵来进行评估，然后选取比较适合属性进行拆分，
# 并对新的两个群组继续计算熵，然后再次拆分， 显然以‘是否阅读过FAQ’来进行第一次拆分是比较合适的
# 拆分之后，继续对子树进行评估拆分，这就是决策树构造的过程
def buildtree(rows, scoref=entropy):
    if len(rows) == 0: return decisionnode()

    # 刚开始默认以服务类型作为特征来评估熵
    current_score = scoref(rows, 4)

    # 定义一些变量记录最佳拆分条件
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1
    for col in range(0, column_count):
        # 在当前列中生成一个由不同值构成的序列
        column_value = {}
        # 先对这一行不重复的元素进行统计
        for row in rows:
            column_value[row[col]] = 1
        # 接下来尝试对这一列数据集进行拆分
        for value in column_value.keys():
            # 对每一行都当作主键都进行一次拆分,拆分之后会产生两个新的集合
            (set1, set2) = divideset(rows, col, value)

            # 计算 set1在总集合中的占比
            p = float(len(set1)) / len(rows)
            # (1-p) set2

            # 信息增益  信息增益 =  entroy(前)(current_score) -  entroy(后)
            # entroy(前)： 以整个集合为数据，计算了集合的熵
            # entroy(后)： 分别以set1，set2分拆之后的子树计算 子集熵值
            # 然后就可以得到信息增益值，这里对信息增益的计算辅以加权平均
            # 这里也可以看到，把集合里的每一个存在的特征都分拆了一遍以求找到最大增益
            # 增益最大说明这次对应的拆分是目前计算情况下熵最小的一次拆分
            # 记录拆分的位置，即它的行和列数，也记录那一次拆分的两个子集，，
            #  注意这一第一次尝试拆分，得到了一个深度为2的二叉树
            gain = current_score - p * scoref(set1, 4) - (1 - p) * scoref(set2, 4)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    print best_gain, best_criteria  # ,best_sets
    # 创建子树
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return decisionnode(col=best_criteria[0], value=best_criteria[1]
                            , tb=trueBranch, fb=falseBranch)
    else:  # gain==0.0
        return decisionnode(results=uniquecounts(rows, 4))

    # buildtree(my_data)


def printtree(tree, indent=''):
    # 是否是一个叶结点
    if tree.results != None:
        print str(tree.results)
    else:
        print str(tree.col) + ':' + str(tree.value) + '?'

        # 打印分支
        print indent + 'T->',
        printtree(tree.tb, indent + '  ')
        print indent + 'F->'
        printtree(tree.fb, indent + '  ')


tree = buildtree(my_data)


# printtree(tree)

def getwidth(tree):
    if tree.tb == None and tree.fb == None: return 1
    return getwidth(tree.tb) + getwidth(tree.fb)


def getdepth(tree):
    if tree.tb == None and tree.fb == None: return 0
    return max(getdepth(tree.tb), getdepth(tree.fb)) + 1


def drawtree(tree, jpeg='tree.jpg'):
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100 + 120

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    drawnode(draw, tree, w / 2, 20)
    img.save(jpeg, 'JPEG')


def drawnode(draw, tree, x, y):
    if tree.results == None:
        # Get the width of each branch
        w1 = getwidth(tree.fb) * 100
        w2 = getwidth(tree.tb) * 100

        # Determine the total space required by this node
        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2

        # Draw the condition string
        draw.text((x - 20, y - 10), str(tree.col) + ':' + str(tree.value), (0, 0, 0))

        # Draw links to the branches
        draw.line((x, y, left + w1 / 2, y + 100), fill=(255, 0, 0))
        draw.line((x, y, right - w2 / 2, y + 100), fill=(255, 0, 0))

        # Draw the branch nodes
        drawnode(draw, tree.fb, left + w1 / 2, y + 100)
        drawnode(draw, tree.tb, right - w2 / 2, y + 100)
    else:
        txt = ' \n'.join(['%s:%d' % v for v in tree.results.items()])
        draw.text((x - 20, y), txt, (0, 0, 0))


# drawtree(tree, jpeg='tree.jpg')

# 对新数据进行分类，新数据缺少一个特征， 我们的目标就是把这个特征作一个判断
def classify(observation, tree):
    if tree.results != None:  # 该节点是叶结点
        return tree.results
    else:  # 非叶子节点
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return classify(observation, branch)


print uniquecounts(my_data, 4)

# 判断新数据属于哪个服务类型
print classify(['(direct)', 'USA', 'yes', 5], tree)


# 决策树的剪枝， 上述算法是直到无法再进一步降低熵的时候才会停止分支的创建，
# 所以一种比较好的解决方案就是， 只要熵减少的数量小于某一个最小值的时候，就停止分支的创建
# 我们会遇到这样的数据，某一次分支的创建并不会使熵下降多少，但是随后的分支却会使熵大大降低，
# 剪枝的过程就是消除多余的节点，
def prune(tree, mingain):
    # 如果不是叶结点，则对其进行剪枝操作
    if tree.tb.results == None:
        prune(tree.tb, mingain)
    if tree.fb.results == None:
        prune(tree.fb, mingain)

    # 如果两个分支都是叶子节点，判断他们是否需要合并，
    if tree.tb.results != None and tree.fb.results != None:
        tb, fb = [], []
        for v, c in tree.tb.results.items():
            tb += [[v]] * c
        for v, c in tree.fb.results.items():
            fb += [[v]] * c

        # 检查熵的减少情况,先将两个叶结点的result合并到一起计算熵值
        delta = entropy(tb + fb, 4) - (entropy(tb, 4) + entropy(fb, 4) / 2)

        if delta < mingain:
            # 合并分支
            tree.tb, tree.fb = None, None
            tree.results = uniquecounts(tb + fb, 4)


prune(tree, 0.1)