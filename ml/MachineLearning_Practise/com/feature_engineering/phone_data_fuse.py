import json
import requests
import re
import pymysql
from neo4j.v1 import GraphDatabase
import csv

from com.neo4j.neo4j_phone import write_data

# 主要实现对用户的通话数据做融合处理---- 数据融合---- 用户纯电话同构网络
class Phone:
    def __init__(self,phone,call_record):
        self.phone_d = phone[0]
        self.call_record_d = call_record

    def phone(self):
        phone_d = self.phone_d
        if phone_d != None:
            print(len(phone_d),phone_d[0])


    # 通过记录电话提取
    def call_extact(self):
        call_history = {}
        try:
            for callrec in self.call_record_d:
                if callrec != None:
                        if callrec != None and callrec.strip() != '':
                            ca = json.loads(callrec)
                            for c in ca:
                                p = c.get('phone')
                                # 通话时长
                                use_time = c.get('use_time')
                                # 呼叫类型
                                type = c.get('type')
                                if call_history.get(p) == None:
                                    call_history[p] =1
                                else:
                                    call_history[p] +=1
            print('len:',len(call_history.keys()),'call_history: ',call_history)
            print(len(call_history.keys()))

        except Exception as e:
            # raise TypeError('Exception') from e
            print('error................',callrec)


        self.call_history = call_history

    #  过滤官方电话
    def call_filter(self):
        filter_call_his = {}
        # 100各个运营商，955各种保险，400运营商营销
        office_phone = ['100','955','400']

        useless_phone = {'02195511':"",'075595511':"广东深圳平安保险"}
        for call in self.call_history.keys():
            if (len(call)) == 11 or (len(call)) == 10:
                if not call.startswith(office_phone[0]) and not call.startswith(office_phone[1] ) \
                        and not call.startswith(office_phone[2]) and not call.startswith('0'):
                        filter_call_his[call] = self.call_history.get(call)

        print('len:', len(filter_call_his), 'call_history: ', filter_call_his)

        self.filter_call_his = filter_call_his

    # 写入csv文件
    def write_to_csv(self,id):
        path = 'D:/Neo4j_data/neo4jDatabases/database-e5e35f12-db93-4e51-820a-000b3b6c0d68/installation-3.4.9/import/'

        phone = self.phone_d[0]

        with open(path+id+'node.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            # 标记是否为借款人
            writer.writerow(['id','is_loaner'])
            writer.writerow([phone,1])
            for row in self.filter_call_his.keys():
                writer.writerow([row,-1])

        with open(path+id+'rel.csv', 'w', newline='') as f:
            phone = self.phone_d[0]
            writer = csv.writer(f)
            writer.writerow(['from_id','property1','property2','to_id'])
            for row in self.filter_call_his.keys():
                writer.writerow([phone,'called',self.filter_call_his[row],row])



# 尽可能收集用户通话数据
def collect_phone_data(p_list, cc,id):
    phone = Phone(p_list, cc)
    phone.phone()
    # 提取
    phone.call_extact()
    # 过滤
    phone.call_filter()
    # 写入文件
    phone.write_to_csv(id)

def user_prepare():
    db = pymysql.connect(host="*", user="root",
                         password="*", db="user_db", port=8066)

    cur = db.cursor()
    user_sql  = 'select ub.identity_no from user_call_record uc left join user_basics_info ub on ub.user_id = uc.user_id' \
     ' where uc.user_data is not null limit 0,5000'

    cur.execute(user_sql)  # 像sql语句传递参数
    # 用户手机号
    u_list = cur.fetchall()
    list = []
    for u in u_list:
        list.append(u[0])
        print(u[0])

    db.close()

    return list

# 查询电话该用户数据
def mysql_connect(id_list):
    db = pymysql.connect(host="*", user="root",
                         password="*", db="user_db", port=8066)

    cur = db.cursor()
    phone_sql = "select distinct phone_num from user_phone_info up LEFT JOIN " \
          "user_basics_info ub on ub.user_id=up.user_id where ub.identity_no = %s"

    call_sql = "select up.user_data,up.sys_date from user_call_record up LEFT JOIN user_basics_info ub on ub.user_id=up.user_id " \
               " where ub.identity_no = %s order by up.sys_date desc"

    for id in id_list:
        # id = '510216198209130411'
        try:
            cur.execute(phone_sql,id)  # 像sql语句传递参数
            # 用户手机号
            p_list = cur.fetchall()

            cur.execute(call_sql,id)
            c_list = cur.fetchall()
            # 用户通话记录
            # print(len(c_list))
            cc = []
            for i in range(len(c_list)):
                if c_list[i][0] !=None and c_list[i][0].strip() !='':
                    cc.append(c_list[i][0])
                    break

        # print(cc)
            collect_phone_data(p_list,cc,id)
        except:
            print('error')
        continue

    db.close()


def write_to_neo4j(id_list):
    uri = "bolt://localhost:11004"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "123456"))

    my_neo4j = write_data.Neo4jHandler(driver)

    count = 0
    for id in id_list:
        try:
            write_data.write_csv_neo4j(my_neo4j,id)
            count+=1
            print(count)
        except:
            print('error')
        continue


if __name__ == '__main__':
    id_list = user_prepare()
    mysql_connect(id_list)
    write_to_neo4j(id_list)