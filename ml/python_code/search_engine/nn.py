# -*- coding: UTF-8 -*-

import urllib2
import mysql.connector
from math import tanh

def dtanh(y):
    return 1.0-y*y

class searchnet:
    def __init__(self,dbname):

        config = {'host': '127.0.0.1',  # default localhost
                  'user': 'root',
                  'password': 'root',
                  'port': 3306,  # 默认即为3306
                  'database':dbname, #无默认数据库
                  'charset': 'utf8',  # 默认即为utf8
                  'buffered': True,
                  }
        self.con = mysql.connector.connect(**config) #连接数据库

    def __del__(self):
        self.con.close()

    def commit(self):
        self.con.commit()

    def cursor(self):
        return self.con.cursor()

    # 判断当前连接的强度
    # wordhidden 记录 word 到hidden层的强度值，
    # hiddenurl  记录 hidden到url 层的强度值
    # hiddennode 记录隐藏层，不用记录强度，只有一个字段
    def getstrength(self,fromid,toid,layer):
        if layer==0: table = 'wordhidden'
        else: table = 'hiddenurl'

        cur = self.con.cursor()
        cur.execute('select strength from %s where fromid=%d and toid=%d' % (table,fromid,toid))
        res = cur.fetchone()

        if res==None:
            if layer==0: return -0.2
            if layer==1: return 0
        return res[0]

    # 用以判断连接是否已经存在，并利用新的强度值更新连接或者创建连接
    def setstrength(self,fromid,toid,layer,strength):
        if layer==0: table='wordhidden'
        else: table = 'hiddenurl'

        print strength

        cur = self.cursor()
        cur.execute('select id from %s where fromid=%d and toid=%d' % (table,fromid,toid))
        res = cur.fetchone()
        if res  == None:
            cur.execute('insert into %s (fromid,toid,strength) values (%d,%d,%f)' % (table,fromid,toid,strength))
        else:
            id = res[0]
            cur.execute('update %s set strength=%f where id=%d' % (table,strength,id))


    # 每传入一组从未见过的单词组合，该函数就会在隐藏层中建立一个新的节点。
    # 随后会在各层之间建立默认权重的连接
    def generatehiddennode(self,wordids,urls):
        if len(wordids)>3: return None

        sorted_words = [str(id) for id in wordids]
        sorted_words.sort()

        # 检查是否已经为这组单词建好了一个节点
        createkey = '_'.join(sorted_words)
        print createkey

        cur = self.cursor()
        cur.execute( "select id from hiddennode where create_key='%s'" % createkey)
        res = cur.fetchone()

        if res == None:
            cur.execute("insert into hiddennode (create_key) values ('%s')" % createkey)
            hiddenid = cur.lastrowid

            # 为wordhidden 到 hiddennode层 建立关系 默认strength是 1/len(wordids)
            for wordid in wordids:
                self.setstrength(wordid,hiddenid,0,1.0/len(wordids))
            # 为hiddennode层 到 hiddenurl层 建立关系
            for urlid in urls:
                self.setstrength(hiddenid,urlid,1,0.1)

            self.con.commit()

    def test_update(self):
        cur = self.cursor()
        cur.execute('insert into %s (fromid,toid,strength) values (%d,%d,%f)' % ('wordhidden', 11, 22, 0.55))

        self.commit()

    # 前馈法，
    # 接收一组单词作为输入，激活网络中的连接，并针对URL给出一组输出结果
    #  返回一个有关联关系的隐藏层节点 字典
    def getallhiddenids(self,wordids,urlids):
        l1 = {}
        cur = self.cursor()

        # 看看这个单词有没有被激活过，
        for wordid in wordids:
            cur.execute(
                'select toid from wordhidden where fromid=%d' % wordid
            )
            # 找出这个单词到隐藏层的关系网有几条
            for row in cur: l1[row[0]] = 1

        # 找出 隐藏层到输出层的关系
        for urlid in urlids:
            cur.execute(
                'select fromid from hiddenurl where toid=%d' % urlid
            )

            for row in cur: l1[row[0]] =1

        return l1.keys()

    # 建立神经网络
    def setupnetwork(self,wordids,urlids):

        # 值列表
        self.wordids = wordids
        self.hiddenids = self.getallhiddenids(wordids,urlids)
        self.urlids = urlids

        # 节点输出 ai 输入节点
        #  ah 隐藏节点
        #  ao 输出节点
        self.ai = [1.0] * len(self.wordids)
        self.ah = [1.0] * len(self.hiddenids)
        self.ao = [1.0] * len(self.urlids)

        # 建立权重矩阵 输入层到隐藏层
        # self.wi = []
        # for wordid in self.wordids:
        #     for hiddenid in self.hiddenids:
        #         self.wi.append(self.getstrength(wordid,hiddenid,0))

        self.wi = [[self.getstrength(wordid,hiddenid,0) for hiddenid in self.hiddenids] for wordid in self.wordids]


        #            隐藏层到输出层
        # self.wo = []
        # for hiddenid in self.hiddenids:
        #     for urlid in self.urlids:
        #         self.wo.append(self.getstrength(hiddenid,urlid,1))

        self.wo = [[self.getstrength(hiddenid,urlid,1) for urlid in self.urlids] for hiddenid in self.hiddenids]

    # 前馈法 ，得到这个节点相关的上一层权值总和
    def feedforward(self):

        for i in range(len(self.wordids)):
            self.ai[i] = 1.0

        # 隐藏层节点的活跃程度
        # 这里就是计算了当前隐藏节点的全部输入权值总和，并做用反函数
        #  得出这个隐藏节点的输入权值总和
        for j in range(len(self.hiddenids)):
            sum = 0.0
            for i in range(len(self.wordids)):
                a = self.ai[i]
                w = self.wi[i][j]
                sum = sum + a * w

            # 这里得出了这个隐藏节点的权值总和，然后作用反正切函数
            # 这里是对权值做修正，越接近于0，值变化越快
            self.ah[j] = tanh(sum)

        # 输出层节点的活跃程度
        # 这里其实是先拿到了，当前隐藏节点的活跃值，然后在计算
        # 每一个输出节点的活跃值，就是当前输出节点的全部输入隐藏权值和
        # 并作用反正切函数
        # 这么以来就知道了神经网络这么计算输出节点的活跃程度，
        # 就是 和这个节点所有相关联的权值总和
        for k in range(len(self.urlids)):
            sum = 0.0
            for j in range(len(self.hiddenids)):

                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = tanh(sum)

        return self.ao[:]

    def getresult(self,wordids,urlids):
        self.setupnetwork(wordids,urlids)

        return self.feedforward()

    # 反向传播算法
    def backPropagate(self,targets, N=0.5):

        # 计算输出层的误差
        output_deltas = [0.0] * len(self.urlids)
        for k in range(len(self.urlids)):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dtanh(self.ao[k]) * error

        # 计算隐藏层误差
        hidden_deltas = [0.0] * len(self.hiddenids)
        for j in range((len(self.hiddenids))):
            error = 0.0
            for k in range(len(self.urlids)):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dtanh(self.ah[j]) * error

        # 更新输出权重
        for j in range(len(self.hiddenids)):
            for k in range(len(self.urlids)):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change

        #  更新输入权重
        for i in range(len(self.wordids)):
            for j in range(len(self.hiddenids)):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change

    def trainquery(self,wordids,urlids,selectedurl):
        self.generatehiddennode(wordids,urlids)

        self.setupnetwork(wordids,urlids)
        # 得到各个输出节点的活跃度
        self.feedforward()

        target = [0.0] * len(urlids)
        target[urlids.index(selectedurl)] = 1.0

        self.backPropagate(target)
        # self.updatedatabase()

    def updatedatabase(self):

        for i in range(len(self.wordids)):
            for j in range(len(self.hiddenids)):
                self.setstrength(self.wordids[i],self.hiddenids[j],0,self.wi[i][j])

        for j in range(len(self.hiddenids)):
            for k in range(len(self.urlids)):
                self.setstrength(self.hiddenids[j],self.urlids[k],1,self.wo[j][k])

        self.commit()






if __name__ == '__main__':
    # 构造测试数据，神经网络的训练是在，用户输入关键字之后，又点击了他们想要的页面，
    # 这样一个完整的操作来 生成神经网络的。,w我用两组数据构造一个神经网络，就相当于用户完成了两次操作。
    wordids = [101,102,103]
    urls = [201,202,203]

    wordids1 = [101,102]
    urls1 = [202,203]

    nn = searchnet('test')
    print '----------激活神经网络----------'
    # nn.generatehiddennode(wordids1,urls1)

    # # 这么以来就知道了神经网络这么计算输出节点的活跃程度，
     # 就是 和这个节点所有相关联的权值总和
    print '----------得到当前查询完成之后的神经网络变化情况----------'
    print nn.getresult(wordids,urls)

    # ok 现在我可以做测试了
    # 用反向传播算法先训练一次
    #  这次用户输入同样的关键词，最后点击了202这个页面，这给神经网络一个反馈
    # 正常情况下，用户输入[101,102,103]关键词，在神经网络的预期之内应该是点击[201,202,203]
    #  最后他点击了 202，那么202这个输出节点以及这个节点关联的其他层的权值需要做修正
    # 这里可以，比如用户点击了202这个页面，那么可以把202这个节点的拓扑序列单独抽出来做修改，这样好理解
    nn.trainquery(wordids,urls,202)

    # 然后再看看结果的变化
    print '----------反向传播算法训练之后202这个网页的活跃程度的变化----------'
    print nn.getresult(wordids, urls)
    # nn.test_update()
