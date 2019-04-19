from neo4j.v1 import GraphDatabase
import requests

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

def write_csv_neo4j(my_neo4j):

    # node 节点写入
    cypher_node = "match (p:person) where p.community=1477374 return p.nid as id"
    result = my_neo4j.cypherexecuter(cypher_node)
    for item in result:
        idNum = item[0]
        url = 'http://127.0.0.1:5000/getEncyUserFeaturesTest?identityNo=%s&currDate=20190418'
        res = requests.get(url % (idNum))
        if res.status_code == 200:
            all_list = []
            res = res.json()
            result = res.get('result')
            features = result.get('features')
            apply_sum_all = features.get('apply_sum_all')
            approve_sum_all = features.get('approve_sum_all')
            overdue_sum_all = features.get('overdue_sum_all')
            maxOverdue_sum_all = features.get('maxOverdue_sum_all')
            if maxOverdue_sum_all == -99999:
                maxOverdue_sum_all = 0

            update_cypher = "match (n:person) where n.nid='"+str(idNum)+"' set n.apply="+str(apply_sum_all)+" set n.approve="+str(approve_sum_all)\
                            +" set n.overdue="+str(overdue_sum_all)+" set n.maxoverdue="+str(maxOverdue_sum_all)+" return n"
            print(update_cypher)
            if apply_sum_all:
                result = my_neo4j.cypherexecuter(update_cypher)
                print(result)

if __name__ == "__main__":
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "123456"))

    my_neo4j = Neo4jHandler(driver)
    write_csv_neo4j(my_neo4j)
    # print(my_neo4j)
