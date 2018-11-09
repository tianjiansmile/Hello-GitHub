# -*- coding: UTF-8 -*-
from numpy import *
import matplotlib.pyplot as plt
from neo4j.v1 import GraphDatabase
from com.untils import public_function
import json
import time
from py2neo import Graph,Node,Relationship
import cypher8
import networkx as nx
import community
from com.neo4j import handle_neo4j as neo4j
import difflib

def graphDatabase_con(uri):
    driver = GraphDatabase.driver(uri, auth=("neo4j", "123456"))
    my_neo4j = neo4j.Neo4jHandler(driver)
    return my_neo4j

def py2neo_con(url):
    test_graph = Graph(
        url,
        username="neo4j",
        password="123456"
    )
    return test_graph

def cypher_runner(py2neo,query):
    result = py2neo.run(query)
    g = nx.from_dict_of_dicts(result)
    for re in result:
        print(re)

if __name__ == "__main__":
    uri = "bolt://localhost:7687"
    my_neo4j = graphDatabase_con(uri)

    query = """
                MATCH p = ()-[]-()
                RETURN p
                """

    # records = my_neo4j.cypherexecuter(query)

    py2neo = py2neo_con(uri)
    cypher_runner(py2neo,query)
    start_time = time.time()

    # degree_centrality = nx.degree_centrality(records)
    # print("Degree Centrality spent %s seconds" % (time.time() - start_time))

    # for re in records:
    #     print(re)