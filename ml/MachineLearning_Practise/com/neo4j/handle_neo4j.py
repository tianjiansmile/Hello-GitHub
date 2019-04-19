# -*- coding: UTF-8 -*-
from numpy import *
import matplotlib.pyplot as plt
from neo4j.v1 import GraphDatabase
from com.untils import public_function
import json
import difflib

#  用GraphDatabase访问neo4j
class Neo4jHandler:
    # 对neo4j 进行读写
    def __init__(self,driver):
        self.driver = driver

    # 查出所有数据，并已列表返回
    def listreader(self, cypher, keys='tom'):

        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                data = []
                person = keys[0];movie = keys[1]
                result = tx.run(cypher)
                for record in result:
                    # 查看一个未知对象的类型  查询到的是对象和对象的嵌套， <Record: <Node: person>, <Node:movie>>
                    # type(record)
                    # 查看一个对象的所有方法
                    p_id = record[person].id
                    m_id = record[movie].id

                    p = dict(record[person])
                    p['p_id'] = p_id
                    m = dict(record[movie])
                    m['m_id'] = m_id
                    # 合并两个字典
                    p_m = dict(p,**m)
                    data.append(p_m)

                return data

        session.close()


    #     执行cypher语句
    def cypherexecuter(self, cypher):
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                result = tx.run(cypher)
                return result
        session.close()

    # 对电影名称进行相似度检查， data里面每三元组是一个字典
    def moviename_check(self,data,field):
        # title 提纯
        movie_name = [di.get(field) for di in data]
        movie_name = list(set(movie_name))
        # print(movie_name)

        # 字符相似度的阈值
        threadhold = 0.8
        # 过滤出相似度高于阈值的name
        similar = [(name,n,difflib.SequenceMatcher(None, name, n).quick_ratio()) for name in movie_name
                   for n in movie_name if name!=n and difflib.SequenceMatcher(None, name, n).quick_ratio() > threadhold]
        print(len(similar),similar)
        print(data)

        # 找到需要merger 的movie节点id 进行merge 也就是说 找到错误movie节点的id
        # 手动筛查 Top Guns 和 The Matrixs 需要进行merge
        merge_ids = {'Top Guns':0,'Top Gun':0,'The Matrixs':0,'The Matrix':0}
        fault_title = [('Top Guns','Top Gun'),('The Matrixs','The Matrix')]
        for fau in fault_title:
            for record in data:
                if record.get('title') == fau[0]:
                    fault_id = record.get('m_id')
                    merge_ids[fau[0]] = int(fault_id)
                if record.get('title') == fau[1]:
                    fault_id = record.get('m_id')
                    merge_ids[fau[1]] = int(fault_id)


        print(merge_ids)



if __name__ == "__main__":
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "123456"))

    my_neo4j = Neo4jHandler(driver)
    # print(my_neo4j)
    cypher_read = 'match (p:Person)-[rel:ACTED_IN]->(movie) return p,movie'
    data = my_neo4j.listreader(cypher_read,['p','movie'])
    my_neo4j.moviename_check(data,'title')

