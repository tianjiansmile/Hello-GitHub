# -*- coding: UTF-8 -*-
from neo4j.v1 import GraphDatabase
from com.neo4j.neo4j_phone import write_data

def write_to_neo4j(id_list):
    uri = "bolt://localhost:11004"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "123456"))
    my_neo4j = write_data.Neo4jHandler(driver)

    # node 节点写入
    cypher_gephi = "match path = (p:phone)-[rel]->(t:phone),(t)<-[r]-(other)" \
			" WITH path with collect(path) as paths" \
			" call apoc.gephi.add(null,'test', paths) yield nodes, relationships, time" \
			" return nodes, relationships, time"

    my_neo4j.cypherexecuter(cypher_gephi)