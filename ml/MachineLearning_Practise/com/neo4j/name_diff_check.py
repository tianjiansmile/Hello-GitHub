# -*- coding: UTF-8 -*-
from numpy import *
import matplotlib.pyplot as plt
from neo4j.v1 import GraphDatabase
from com.untils import public_function
import json
from py2neo import Graph
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
                    # print(record)
                    # 查看一个未知对象的类型  查询到的是对象和对象的嵌套， <Record: <Node: person>, <Node:movie>>
                    # type(record)
                    # 查看一个对象的所有方法
                    p_id = record[person].id
                    m_id = record[movie].id

                    # p = dict(record[person])
                    m = dict(record[movie])
                    m['m_id'] = m_id
                    m['p_id'] = p_id
                    # 合并两个字典
                    # p_m = dict(p,**m)
                    data.append(m)

                return data

        session.close()


    #     执行cypher语句
    def cypherexecuter(self, cypher):
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                result = tx.run(cypher)
                for item in result:
                    type = public_function.type_check(item)
                    # json.load(item)
                    print(type,item)
        session.close()

    # 对公司名称进行相似度检查， data里面每三元组是一个字典
    def moviename_check(self,data,field):
        # title 提纯
        movie_name = [di.get(field) for di in data]
        print('before',len(data), len(movie_name))
        movie_name = list(set(movie_name))
        print('after',len(movie_name))

        # 字符相似度的阈值
        threadhold = 0.8
        # 1 过滤出相似度高于阈值的name
        similar = [(name,n,difflib.SequenceMatcher(None, name, n).quick_ratio()) for name in movie_name
                   for n in movie_name if name!=n and difflib.SequenceMatcher(None, name, n).quick_ratio() > threadhold]
        print(len(similar),similar)
        print(data)


        # 2 手动筛查,标记出两个相似节点中，正确节点和错误节点

        # 对相似节点的数据整理，记录下每一个问题节点的id和对应person的id
        # 手动筛查
        final = {}
        for sim in similar:
            name = sim[0]
            temp = []
            for re in data:
                if re.get('nodeID') == name:
                    temp.append(re.get('p_id'))
                    temp.append(re.get('m_id'))
                    final[name] = temp
            print(name,final[name])




        # print(merge_ids)



if __name__ == "__main__":
    uri = "bolt://pro3:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "123456"))

    my_neo4j = Neo4jHandler(driver)
    # print(my_neo4j)
    company_cypher_read = 'match (p:PERSON)-[rel:HAS_COMPANYNAME]->(com:COMPANYNAME) return p,com limit 1000'
    address_cypher_read = 'match (p:PERSON)-[rel]->(com:ADDRESS) return p,com limit 100'
    data = my_neo4j.listreader(company_cypher_read,['p','com'])
    my_neo4j.moviename_check(data,'nodeID')

    # data = my_neo4j.listreader(address_cypher_read, ['p', 'com'])
    # my_neo4j.moviename_check(data, 'nodeID')

