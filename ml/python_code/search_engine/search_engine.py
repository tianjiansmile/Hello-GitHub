# -*- coding: UTF-8 -*-
import urllib2
import mysql.connector
import bs4
from bs4 import BeautifulSoup
from urlparse import urljoin
import re

# 需要被忽略的单词列表
ignorewords={'the':1,'of':1,'to':1,'and':1,'a':1,'in':1,'is':1,'it':1}

class crawler:

    #初始化crawler类并传入数据库名称
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

    def dbcommit(self):
        self.con.commit()

    # 获取条目的ID 如果条目不存在将其加入数据库
    def getentryid(self,table,field,value,createnew=True):
        cur = self.con.cursor()
        cur.execute(
            "select id from %s where %s='%s'" % (table,field,value))
        res = cur.fetchone()
        if res == None:
            cur = self.con.cursor()
            cur.execute(
                "insert into %s (%s) values ('%s')" % (table, field, value))
            return cur.lastrowid
        else:
            return res[0]

    # 为每一个网页建立索引
    def addtoindex(self,url,soup):
        if self.isindexed(url): return
        print 'Indexing  '+url

        # 获取每一个单词
        text = self.gettextonly(soup)
        words = self.separatewords(text)

        # 得到url的id
        urlid = self.getentryid('urllist','url',url)

        #
        for i in range(len(words)):
            word = words[i]
            if word in ignorewords: continue
            wordid = self.getentryid('wordlist','word',word)
            self.con.cursor().execute("insert into wordlocation(urlid,wordid,location) values (%d,%d,%d)" % (urlid,wordid,i))

    # 从一个HTML网页中提取文字
    def gettextonly(self,soup):
        v = soup.string
        if v==None:
            c = soup.contents
            resulttext = ''
            for t in c:
                subtext = self.gettextonly(t)
                resulttext += subtext+'\n'
            return resulttext
        else:
            return v.strip()

    # 根据任何非空白字符进行分词处理
    def separatewords(self,text):
        splitter = re.compile('\\W*')
        return [s.lower() for s in splitter.split(text) if s!='']

    # 检查url是否已经建立索引
    def isindexed(self,url):
        u = self.con.cursor()
        u.execute("select id from urllist where url='%s'" % url)
        res = u.fetchone()
        if res != None:
            v = self.con.cursor()
            v.execute('select * from wordlocation where urlid=%d' % res[0])
            ces = v.fetchone()
            if ces != None: return True
        return False

    # 添加一个关联两个网页的链接
    def addlinkref(self, urlFrom, urlTo, linkText):
        words = self.separatewords(linkText)
        fromid = self.getentryid('urllist', 'url', urlFrom)
        toid = self.getentryid('urllist', 'url', urlTo)
        if fromid == toid: return
        cur = self.con.cursor()
        cur.execute("insert into link(fromid,toid) values (%d,%d)" % (fromid, toid))
        linkid = cur.lastrowid
        for word in words:
            if word in ignorewords: continue
            wordid = self.getentryid('wordlist', 'word', word)
            cur = self.con.cursor()
            cur.execute("insert into linkwords(linkid,wordid) values (%d,%d)" % (linkid, wordid))

    # 从一组网页开始进行广度优先搜索，直至某一深度
    # 期间为网页建立索引
    def crawler(self,pages,depth=2):
        for i in range(depth):
            newpages = {}
            for page in pages:
                try:
                    c = urllib2.urlopen(page)
                except:
                    print "Could not open %s" % page
                    continue
                try:
                    soup = BeautifulSoup(c.read(),'html5lib')
                    self.addtoindex(page, soup)

                    links = soup('a')
                    for link in links:
                        if ('href' in dict(link.attrs)):
                            url = urljoin(page, link['href'])
                            if url.find("'") != -1: continue
                            url = url.split('#')[0]  # remove location portion
                            if url[0:4] == 'http' and not self.isindexed(url):
                                newpages[url] = 1
                            linkText = self.gettextonly(link)
                            self.addlinkref(page, url, linkText)

                    self.dbcommit()
                except Exception, e:
                    print 'e.message:\t', e.message
                    print "Could not parse page %s" % page
            pages = newpages


    # 创建数据库表
    def createindextables(self):
        cursor = self.con.cursor()
        # cursor.execute('CREATE INDEX wordidx ON wordlist (word)')
        cursor.execute('CREATE INDEX urlidx ON urllist (url)')
        cursor.execute('CREATE INDEX wordurlidx ON wordlocation (wordid)')
        cursor.execute('CREATE INDEX urltoidx ON link (toid)')
        cursor.execute('CREATE INDEX urlfromidx ON link(fromid)')
        # self.con.execute('create index wordidx on wordlist(word)')
        # self.con.execute('create index urlidx on urllist(url)')
        # self.con.execute('create index wordurlidx on wordlocation(wordid)')
        # self.con.execute('create index urltoidx on link(toid)')
        # self.con.execute('create index urlfromidx on link(fromid)')
        self.dbcommit()



class searcher:

    def __init__(self,dbname):
        configs = {'host': '127.0.0.1',  # default localhost
                   'user': 'root',
                   'password': 'root',
                   'port': 3306,  # 默认即为3306
                   'database': dbname,  # 无默认数据库
                   'charset': 'utf8',  # 默认即为utf8
                   'buffered': True,
                   }
        self.con = mysql.connector.connect(**configs)  # 连接数据库

    def __del__(self):
        self.con.close()

    def dbcommit(self):
        self.con.commit()

    # 联合查询，两个及多个单词的查询
    def getmatchrows(self,q):

        fieldlist = 'w0.urlid'
        tablelist = ''
        clauselist = ''
        wordids = []

        words = q.split(' ')
        tablenumber = 0

        for word in words:
            cur = self.con.cursor()
            cur.execute("select id from wordlist where word='%s'" % word)
            wordrow = cur.fetchone()

            if wordrow != None:
                wordid = wordrow[0]
                # 查到所有单词的 id
                wordids.append(wordid)
                if tablenumber > 0:
                    tablelist +=','
                    clauselist += ' and '
                    clauselist += 'w%d.urlid=w%d.urlid and ' % (tablenumber - 1, tablenumber)
                fieldlist += ',w%d.location' % tablenumber
                tablelist += 'wordlocation w%d' % tablenumber
                clauselist += 'w%d.wordid=%d' % (tablenumber, wordid)
                tablenumber += 1

        fullquery = 'select %s from %s where %s' % (fieldlist, tablelist, clauselist)
        print fullquery

        cur.execute(fullquery)
        # 单词位置，以及单词所属的url
        rows = [row for row in cur]
        print('---------urlid, location1, location2----------')
        print rows
        print '--------------wordid1, wordid2 -----------------------------'
        print wordids
        return rows,wordids


    # 基于内容的排名
    # 单词频度（单词在文档中出现的次数）
    # 文档位置（文档的主题有可能出现在靠近文档的开始处）
    # 单词距离（如果查询条件中有多个单词，则它们在文档中应该很靠近）

    def getscoredlist(self,rows,wordids):

        # 给出每一个url的 评估分数，初始为零
        totalscores = dict([(row[0],0) for row in rows])

        #这里放置的是不同的排名方法函数

        weights = [
            (1.0, self.frequencyscore(rows)),
            (1.0,self.inboundlinkscore(rows))
        ]

        # weights = [
        #     # (1.0, self.locationscore(rows))
        #     (1.0,self.distancescore(rows))
        #     # (1.0, self.frequencyscore(rows))
        # ]

        print weights

        for (weight,scores) in weights:
            for url in totalscores:

                # 为每一个url打分， 分别把不通的排名函数做加权平均
                totalscores[url] += weight*scores[url]

        print '----------网页最终排名-------------'
        print totalscores
        return totalscores

    def geturlname(self,id):
        cur = self.con.cursor()
        cur.execute("select url from urllist where id=%d" % id)

        return cur.fetchone()[0]

    def query(self,q):
        # rows 存放单词查询位置和所属url，wordids 存放每一个word的id
        rows,wordids = self.getmatchrows(q)
        # 评分字典，每一个urlid 为k  分数为 value
        scores = self.getscoredlist(rows,wordids)
        # 调换位置
        rankedscores = [(score,url) for (url,score) in scores.items()]
        # 排序
        rankedscores.sort()
        rankedscores.reverse()

        # 拿出排名前10 的网页
        for (score, urlid) in rankedscores[0:10]:
            print '%f\t%s' % (score, self.geturlname(urlid))
        return wordids, [r[1] for r in rankedscores[0:10]]

    # 归一化函数
    #  有的评分函数分值越大越好，有的越小越好，需要对不同的评分函数做归一化处理，
    # 最后只返回一个介于0--1之间的数字
    def normalizescores(self,scores,smallIsBetter=0):
        vsmall = 0.00001  # Avoid division by zero errors

        if smallIsBetter:
            minscore = min(scores.values())
            return dict([(u, float(minscore) / max(vsmall, l)) for (u, l) in scores.items()])
        else:
            maxscore = max(scores.values())
            if maxscore == 0: maxscore = vsmall
            return dict([(u, float(c) / maxscore) for (u, c) in scores.items()])

    # 单词频度评价方法  rows----(401, u'11', u'1058'), (401, u'14', u'1058')
    def frequencyscore(self,rows):
        #网页计数字典  401:0， 402:0 .......
        counts = dict([(row[0], 0) for row in rows])
        # 就是累算每一个urlid在列表里出现的次数
        for row in rows:
            counts[row[0]] += 1

        print '-----该多个单词在此网页中出现的次数--这里计算是任何一个单词出现一次就加一---'
        print counts
        return self.normalizescores(counts,smallIsBetter=1)


    # 文档位置评价方法
    # 表wordlocation记录了单词在文档中出现的位置，单词出现的位置越靠前，得分越高
    def locationscore(self, rows):
        locations = dict([(row[0], 1000000) for row in rows])
        for row in rows:
            loc = sum(row[1:])
            if loc < locations[row[0]]: locations[row[0]] = loc

        return self.normalizescores(locations, smallIsBetter=1)

    # 单词距离
    #  查询中包含多个单词的时候，单词之间的距离是一个很重要的评估要素
    #
    def distancescore(self, rows):
        # 如果只搜索有一个单词， 则得分都一样
        if len(rows[0]) <= 2: return dict([(row[0], 1.0) for row in rows])

        # Initialize the dictionary with large values
        mindistance = dict([(row[0], 1000000) for row in rows])

        # 便利单词位置时，计算出前后单词之间的位置差距，。最后找出距离最小值
        for row in rows:
            dist = sum([abs(row[i] - row[i - 1]) for i in range(2, len(row))])
            if dist < mindistance[row[0]]: mindistance[row[0]] = dist
        return self.normalizescores(mindistance, smallIsBetter=1)

    # 外部回指链接排名和PageRank
    # 本节采用外界就该网页提供的信息—尤其是谁链向了该网页，以及他们对该网页的评价，来对于网页进行排名。对于每个遇到的链接，links表中记录了与其源和目的相对应的URL
    # ID，而且linkwords表还记录了单词与链接的关联。本节主要介绍以下三个方面内容：
    #
    # 简单计数
    # PageRank
    # 利用链接文本

    #简单计数
    # 统计每个网页上链接的数目，并将链接总数作为针对网页的度量。
    # 科研论文的评价经常采用这种方式。下面的代码中通过对查询link表所得到的行集中的每个唯一的URL
    # ID进行计数，建立起了一个字典。随后，函数返回一个经过归一化处理的评价结果。
    def inboundlinkscore(self,rows):
        uniqueurls = dict([(row[0],1) for row in rows])

        for u in uniqueurls:
            cur = self.con.cursor()
            cur.execute('select count(*) from link where toid=%d' % u)
            uniqueurls[u] = cur.fetchone()[0]

        print '------------------简单计数，统计其他网页链向这个网页的个数--------------------------'
        print uniqueurls
        return self.normalizescores(uniqueurls)

    # PgeRank 算法
    # 阻尼因子：从当前网页点击其他任何一个链接网页的概率都是 0.85
    # 从有向图的角度来看，通过所有指向当前节点的其他节点的PageRank值
    # 来计算当前节点的PageRank值，比如 A B C 三个网页指向 D网页
    # A 还有4 指向其他网页的链接，B 有 2 ， C只有一个指向D的链接
    # PR(D) = 0.15 + 0.85(p(A)/4 + p(B)/2 + p(C)/1)  才对结果的影响最大，这也是情理之中的
    # 目前网页都还没有PageRank值，初始值设置为1，面对目前的规模然后迭代20就差不多得到真实的网页值了
    def calculatepagerank(self,iteration=20):

        cur = self.con.cursor()

        # 初始化数据
        # cur.execute('insert into pagerank select id, 1.0 from urllist')
        # self.dbcommit()

        for i in range(iteration):
            print 'Iteration %d' % (i)
            cur.execute('select id from urllist')
            ids = cur.fetchall()
            for (urlid,) in ids:
                pr = 0.15
                # 找到指向当前网页的所有网页
                cur.execute('select distinct fromid from link where toid=%d' % urlid)
                fromids = cur.fetchall()
                for (linker,) in fromids:
                    # 获得他们的score
                    cur.execute('select score from pagerank where urlid=%d' % linker)
                    linkingpr = cur.fetchone()[0]

                    # 关联网页数
                    cur.execute('select count(*) from link where fromid =%d' % linker)
                    linkingcount = cur.fetchone()[0]

                    print linkingcount, linkingpr

                    # 计算当前网页pagerank值
                    pr += 0.85*(linkingpr/linkingcount)
                    print pr
                print pr
                cur.execute('update pagerank set score=%f where urlid=%d' % (pr, urlid))

        self.dbcommit()







if __name__ == '__main__':
    # 读取网页数据
    # c = urllib2.urlopen('https://blog.csdn.net/u013007900/article/details/54706336')
    # content = c.read()
    # print content[0:50000]

    # read = open('C:/Users/ChenYan/Desktop/feedlist.txt', 'r')
    # alllines = read.readlines()
    # pagelists = [line for line in alllines]
    # print pagelists
    #  爬虫，爬取网页数据到数据库
    # pagelist = ['https://en.wikipedia.org/wiki/Japen_rat']
    # cler = crawler('test')
    # cler.crawler(pagelist)
    # cler.createindextables()

    sear = searcher('test')
    print '---------------------查询单词位置--------------------------'
    # sear.getmatchrows('current date')

    # sear.query('better usage')

    sear.calculatepagerank()


