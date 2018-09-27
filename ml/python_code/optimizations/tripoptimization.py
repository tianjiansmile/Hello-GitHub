# -*- coding: UTF-8 -*-
# 对组团旅游计划进行优化
# 对来自不同地方的人去往同一地点的人们安排一次合理的旅游计划

import time
import random
import math

people = [('Seymour', 'BOS'), ('Franny', 'DAL'), ('Zooey', 'CAK'), ('Wait', 'MIA'), ('Buddy', 'ORD'), ('Les', 'OMA')]

# Glass 一家6口人从五湖四海赶来，都要在同一天到达目的地 LGA机场，并且在同一天离开，他们想搭乘相同的交通工具往返机场
destination = 'LGA'  # Newyork LGA 机场

flights = {}

r = open('C:/Users/jt90787/Desktop/AIdatafile/ariplane.txt', 'r')
for line in r.readlines():
    origin, dest, depart, arrive, price = line.strip().split(',')

    flights.setdefault((origin, dest), [])

    flights[(origin, dest)].append((depart, arrive, int(price)))


def getminutes(t):
    x = time.strptime(t, '%H:%M')
    return x[3] * 60 + x[4]


# 最优解，不仅仅是要让总票价将下来，还要考虑到飞行时间和候机时间

def printschedule(r):
    for d in range(len(r) / 2):
        name = people[d][0]
        origin = people[d][1]
        out = flights[(origin, destination)][r[2 * d]]
        ret = flights[(destination, origin)][r[2 * d + 1]]
        print '%10s%10s %5s-%5s $%3s %5s-%5s $%3s' % (name, origin, out[0], out[1], out[2],
                                                      ret[0], ret[1], ret[2])


# 确定成本函数，在这个案例中需要考虑的变量， 价格，旅行时间，机场等待时间，出发时间，汽车租用时间。
# 通过飞行策略来计算旅行总cost
def schedulecost(sol):
    totalprice = 0
    latestarrival = 0
    earliestdep = 24 * 60

    for d in range(len(sol) / 2):
        origin = people[d][1]
        # 往返航班
        outbound = flights[origin, destination][int(sol[2 * d])]
        returnf = flights[destination, origin][int(sol[2 * d + 1])]

        # 往返航班的总消费
        totalprice += outbound[2]
        totalprice += returnf[2]

        # 记录最晚到达时间和最早离开时间
        if latestarrival < getminutes(outbound[1]):
            latestarrival = getminutes(outbound[1])
        if earliestdep > getminutes(returnf[0]):
            earliestdep = getminutes(returnf[0])

    # 每一个人都必须在机场等到最后一个人到达为止，
    # 他们也必须在相同的时间到达机场，并返回他们的居住地
    totalwait = 0
    for d in range(len(sol) / 2):
        origin = people[d][1]
        outbound = flights[origin, destination][int(sol[2 * d])]
        returnf = flights[destination, origin][int(sol[2 * d + 1])]

        # 所有人等待时间总和，等待时间换算是一分钟一美元
        totalwait += latestarrival - getminutes(outbound[1])
        totalwait += getminutes(returnf[0]) - earliestdep

    # 租车要用50美元
    if latestarrival < earliestdep: totalprice += 50

    return totalprice + totalwait


# 随机搜索，用来评估其他算法优劣的基线，也可以让我们更清楚地理解所有算法的真正意图
def randomoptimize(domain, costf):
    best = 999999999
    bestr = None
    for j in range(10000):
        # 创建一个随机解
        r = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]

        # 得到成本
        cost = costf(r)
        # 与到目前为止的最优解进行比较
        if cost < best:
            best = cost
            bestr = r

    #     print best
    return bestr


# 随即算法是蛮干，毫无依据的进行尝试，其实我们可以通过已经的得出的优解进行调整测试
# 爬山法，在其邻近解中寻求更优解，对于最初的一个随即安排我们可以对某个人使用其相邻航班去计算成本
# 缺点。得出的是局部最优解，它只是一个初始随机解的相邻解中的最优解，就像极小值一样，不是全局最优解，不是最小值
def hillclimb(domain, costf, sol):
    # 随机一个解
    if sol == None:
        sol = [random.randint(item[0], item[1]) for item in domain]

    best_cost = 99999
    while True:
        # 记录周围解
        neigbors = []
        # 对于所有人的解，都进行修改
        for i in range(len(domain)):
            if sol[i] > domain[i][0]:
                neigbors.append(sol[0:i] + [sol[i] - 1] + sol[i + 1:])
            if sol[i] < domain[i][1]:
                neigbors.append(sol[0:i] + [sol[i] + 1] + sol[i + 1:])

        for item in neigbors:
            cost = costf(item)
            if cost < best_cost:
                best_cost = cost
                sol = item
        # 最优解是上次的解，说明已经到了最低点
        if sol not in neigbors:
            return best_cost, sol


# 退火法
# T代表温度，cool代表退火率，step代表改变范围
def backfire(domain, fn_cost=schedulecost, T=10000, cool=0.95, step=3):
    # 随机解
    sol = [random.randint(item[0], item[1]) for item in domain]

    while T > 0.1:
        # 随机出来一个要改变的位置
        i = random.randint(0, len(domain) - 1)
        # 随机一个要改变的数
        change = random.randint(-step, step)

        sol_copy = sol[:]
        sol_copy[i] += change
        # 保证数据没有越界
        if sol_copy[i] < domain[i][0]:
            sol_copy[i] = domain[i][0]

        if sol_copy[i] > domain[i][1]:
            sol_copy[i] = domain[i][1]
        c = fn_cost(sol)
        cc = fn_cost(sol_copy)
        # 如果是更优解或者现在的随机概率在范围内
        # 随着计算的进行,此范围会越来越小,越来越倾向于接收更优解
        if cc < c or random.random() < pow(math.e, -(cc - c) / T):
            sol = sol_copy[:]

        T *= cool

    return fn_cost(sol), sol


# 遗传算法的运行过程是先随机生成一组解所谓初始族群，族群中每次迭代都会选取更优秀的一部分，
# 然后这部分优秀种进行变异或者交叉来补充族群到最大，并重复这一过程
# 变异示例：
# ​ [7,5,3,2,5,3,0,1,1,5,3,6] ->[7,5,3,2,5,3,5,1,1,5,3,6]
# ​ 交叉示例：
# ​ [7,5,3,2,5,3,**交叉位置**0,1,1,5,3,6]
# ​ [3,8,2,6,5,4,**交叉位置**1,2,3,5,4,6]
# ​ ->[7,5,3,2,5,3,1,2,3,5,4,6]
# popsize:数据量  variation:变异概率, elite:遗传率,  maxiter:遗传多少代
def genetic(domain, fn_cost=schedulecost, popsize=50, variation=0.2, elite=0.3, maxiter=100):
    # 变异
    def mutate(r):
        i = random.randint(0, len(domain) - 1)
        # 两个方向的变异概率都是50%，random.random() 随机数范围小于一
        if random.random() < 0.5:
            res = r[0:i] + [r[i] - 1] + r[i + 1:]
        else:
            res = r[0:i] + [r[i] + 1] + r[i + 1:]

        if res[i] < domain[i][0]:
            res[i] = domain[i][0]

        if res[i] > domain[i][1]:
            res[i] = domain[i][1]
        return res

    # 交叉
    def crossover(r1, r2, pops):
        if r1 is None or r2 is None:
            for line in pops:
                print(str(line))
        # 交叉必然发生
        i = random.randint(1, len(domain) - 2)
        return r1[0:i] + r2[i:]

    # 计算能存活下来的数量。每一代可以存活下来的个体数量
    elite_count = int(popsize * elite)

    # 随机出第一代,一共50个个体
    pops = [[random.randint(item[0], item[1]) for item in domain] for i in range(popsize)]

    # 一共遗传100代
    for i in range(maxiter):
        # 计算消耗，每一个个体的消耗
        pops_cost = [(fn_cost(item), item) for item in pops]
        pops_cost.sort()

        # 只取最优解遗传，保留下比较优秀的前elite_count 个
        pops = [item[1] for item in pops_cost[0:elite_count]]

        # 补充族群至最大
        while len(pops) < popsize:
            # 随机进行变异和交叉
            # 变异 ，变异率 0.2
            if random.random() < variation:
                # 对族群中的随机一个个体进行变异
                pops.append(mutate(pops[random.randint(0, len(pops) - 1)]))
            else:
                # 随机选出两个解进行交叉，类似于繁衍，两个个体诞生一个新个体，父本母本各取一些基因
                r1 = pops[random.randint(0, len(pops) - 1)]
                r2 = pops[random.randint(0, len(pops) - 1)]
                pops.append(crossover(r1, r2, pops))

    pops_cost = [(fn_cost(item), item) for item in pops]
    pops_cost.sort()
    return pops_cost[0]


if __name__ == '__main__':
    # 分别代表每一个家庭成员的飞行策略，最终就是要确定这样一个飞行策略
    # 1，4 代表Seymour 从BOS到LGA做这一天的第2次航班， 从LGA回到BOS座第5趟航班
    s = [1, 3, 3, 2, 7, 3, 6, 3, 2, 4, 5, 3]

    print flights
    print schedulecost(s)

    # 各位置取值范围,初始化飞行策略，每个人的航班每天都有10趟
    domain = [(0, 9)] * (len(people) * 2)

    # 随机搜索
    r = randomoptimize(domain, schedulecost)
    print '----------------------------------随机搜索-----------------------------------------------'
    print schedulecost(r), r
    printschedule(r)

    # 爬山法
    b = hillclimb(domain, schedulecost, None)
    print '--------------------------------爬山法-----------------------------------------------'
    print b
    printschedule(b[1])

    # 爬山法
    b = hillclimb(domain, schedulecost, r)
    print '--------------------------------利用随机搜索之后的最优解再使用--爬山法-----------------------------------------------'
    print b
    printschedule(b[1])

    # 退火算法
    print '--------------------------------退火算法-----------------------------------------------'
    fire = backfire(domain, schedulecost)
    print fire
    printschedule(fire[1])

    # 遗传算法
    print '--------------------------------遗传算法-----------------------------------------------'
    gene = genetic(domain, schedulecost)
    print gene
    printschedule(gene[1])