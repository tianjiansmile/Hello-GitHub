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


    #     执行cypher语句
    def cypherexecuter(self, cypher):
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                result = tx.run(cypher)
                return result
        session.close()



def write_csv_neo4j(my_neo4j,id):

    # node 节点写入
    cypher_node = "LOAD CSV WITH HEADERS  FROM 'file:///"+id+"node.csv' AS line " \
    "MERGE (p:phone{id:line.id,is_loaner:line.is_loaner}) "
    # "MERGE (p:phone{id:line.id,is_loaner:line.is_loaner}) ON CREATE SET p.is_loaner=line.is_loaner ON MATCH SET p.is_loaner=1 return p"

    # rel写入
    cypher_rel = "LOAD CSV WITH HEADERS  FROM 'file:///"+id+"rel.csv' AS line " \
                  "match (from:phone{id:line.from_id}),(to:phone{id:line.to_id})" \
                  "merge (from)-[r:called{call:line.property1,times:line.property2}]-(to)"

    my_neo4j.cypherexecuter(cypher_node)
    my_neo4j.cypherexecuter(cypher_rel)


if __name__ == "__main__":
    uri = "bolt://localhost:11008"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "123456"))

    my_neo4j = Neo4jHandler(driver)
    write_csv_neo4j(my_neo4j)
    # print(my_neo4j)


