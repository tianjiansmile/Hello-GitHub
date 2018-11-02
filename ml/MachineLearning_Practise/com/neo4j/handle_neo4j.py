# -*- coding: UTF-8 -*-
from numpy import *
import matplotlib.pyplot as plt
from neo4j.v1 import GraphDatabase

class Neo4jHandler:
    # 对neo4j 进行读写
    def __init__(self,driver):
        self.driver = driver

    def listreader(self, cypher, keys):

        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                data = []
                result = tx.run
                for record in result:
                    rows = []
                    for key in keys:
                        rows.append(record[key])
                    data.append(rows)
                return data

        session.close()

    #     执行cypher语句
    def cypherexecuter(self, cypher):
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                tx.run(cypher)
        session.close()


if __name__ == "__main__":
    uri = "bolt://*:7687"
    driver = GraphDatabase.driver(uri, auth=("***", "***"))

    my_neo4j = Neo4jHandler(driver)
    print(my_neo4j)
    cypher_read = ''
